use clap::{Parser, Subcommand};
use csv::Writer;
use indicatif::{ProgressBar, ProgressStyle};
use mzdata::{self, io::{MZReaderType, MassSpectrometryFormat, infer_format}, prelude::*, spectrum::{ArrayType, BinaryDataArrayType}};
use mzpeak_prototyping::{writer::sample_array_types, *};
use mzpeaks::{CentroidPeak, DeconvolutedPeak};
use std::{
    collections::{HashMap, VecDeque},
    fs, io,
    path::{Path, PathBuf},
    panic::{self, AssertUnwindSafe},
    sync::{mpsc::sync_channel, Arc, Mutex},
    thread,
    time::Instant
};
use tempfile::TempDir;
use walkdir::WalkDir;

#[derive(Parser)]
#[command(name = "mzpeak_prototyping")]
#[command(about = "A tool for converting and benchmarking mass spectrometry data")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Convert a single mass spectrometry file to mzpeak format
    Convert(ConvertArgs),
    /// Benchmark conversion of all supported files in a directory
    Benchmark(BenchmarkArgs),
}

#[derive(Parser)]
struct ConvertArgs {
    /// Input file path
    filename: PathBuf,

    #[arg(short = 'm', long = "mz-f32", help="Encode the m/z values using float32 instead of float64")]
    mz_f32: bool,

    #[arg(short = 'd', long = "ion-mobility-f32", help="Encode the ion mobility values using float32 instead of float64")]
    ion_mobility_f32: bool,

    #[arg(short = 'y', long = "intensity-f32", help="Encode the intensity values using float32")]
    intensity_f32: bool,

    #[arg(short = 'i',
          long = "intensity-i32",
          help="Encode the intensity values as int32 instead of floats which may improve compression at the cost of the decimal component")]
    intensity_i32: bool,

    #[arg(short = 'z', long = "shuffle-mz", help="Shuffle the m/z array, which may improve the compression of profile spectra")]
    shuffle_mz: bool,

    #[arg(short='u', long, help="Null mask out sparse zero intensity peaks")]
    null_zeros: bool,

    #[arg(short = 'o', help="Output file path")]
    outpath: Option<PathBuf>,
}

#[derive(Parser)]
struct BenchmarkArgs {
    /// Directory to scan for mass spectrometry files
    directory: PathBuf,

    #[arg(short = 'o', long = "output-csv", help="Path to save benchmark results CSV")]
    output_csv: Option<PathBuf>,

    #[arg(short = 't', long = "threads", help="Number of threads to use (default: number of CPU cores)")]
    threads: Option<usize>,

    #[arg(long = "temp-dir", help="Temporary directory for converted files")]
    temp_dir: Option<PathBuf>,

    #[arg(long = "no-progress", help="Disable progress bar")]
    no_progress: bool,

    // Include all conversion options
    #[arg(short = 'm', long = "mz-f32", help="Encode the m/z values using float32 instead of float64")]
    mz_f32: bool,

    #[arg(short = 'd', long = "ion-mobility-f32", help="Encode the ion mobility values using float32 instead of float64")]
    ion_mobility_f32: bool,

    #[arg(short = 'y', long = "intensity-f32", help="Encode the intensity values using float32")]
    intensity_f32: bool,

    #[arg(short = 'i',
          long = "intensity-i32",
          help="Encode the intensity values as int32 instead of floats which may improve compression at the cost of the decimal component")]
    intensity_i32: bool,

    #[arg(short = 'z', long = "shuffle-mz", help="Shuffle the m/z array, which may improve the compression of profile spectra")]
    shuffle_mz: bool,

    #[arg(short='u', long, help="Null mask out sparse zero intensity peaks")]
    null_zeros: bool,
}

#[derive(Debug, Clone)]
struct ConvertOptions {
    mz_f32: bool,
    ion_mobility_f32: bool,
    intensity_f32: bool,
    intensity_i32: bool,
    shuffle_mz: bool,
    null_zeros: bool,
}

#[derive(Debug)]
struct BenchmarkResult {
    filename: String,
    original_size: u64,
    final_size: u64,
    time_taken: f64, // seconds
    status: String,
}

fn main() -> io::Result<()> {
    env_logger::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Convert(args) => run_convert(args),
        Commands::Benchmark(args) => run_benchmark(args),
    }
}

fn run_convert(args: ConvertArgs) -> io::Result<()> {
    let start = Instant::now();
    
    let options = ConvertOptions {
        mz_f32: args.mz_f32,
        ion_mobility_f32: args.ion_mobility_f32,
        intensity_f32: args.intensity_f32,
        intensity_i32: args.intensity_i32,
        shuffle_mz: args.shuffle_mz,
        null_zeros: args.null_zeros,
    };
    
    let outpath = args.outpath.unwrap_or_else(|| {
        args.filename.with_extension("mzpeak")
    });
    
    convert_file(&args.filename, &outpath, &options)?;
    
    let end = Instant::now();
    eprintln!("{:0.2} seconds elapsed", (end - start).as_secs_f64());
    
    let stat = fs::metadata(&outpath)?;
    let size = stat.len() as f64 / 1e9;
    eprintln!("{} was {size:0.3}GB", outpath.display());
    
    Ok(())
}

fn run_benchmark(args: BenchmarkArgs) -> io::Result<()> {
    let start = Instant::now();
    
    // Setup
    let threads = args.threads.unwrap_or_else(num_cpus::get);
    let temp_dir = if let Some(temp_dir) = args.temp_dir {
        TempDir::new_in(temp_dir)?
    } else {
        TempDir::new()?
    };
    
    let options = ConvertOptions {
        mz_f32: args.mz_f32,
        ion_mobility_f32: args.ion_mobility_f32,
        intensity_f32: args.intensity_f32,
        intensity_i32: args.intensity_i32,
        shuffle_mz: args.shuffle_mz,
        null_zeros: args.null_zeros,
    };
    
    // Discover files
    eprintln!("Scanning directory for supported mass spectrometry files...");
    let files = discover_supported_files(&args.directory)?;
    
    if files.is_empty() {
        eprintln!("No supported mass spectrometry files found in {}", args.directory.display());
        return Ok(());
    }
    
    eprintln!("Found {} supported files", files.len());
    
    // Setup progress bar
    let progress = if args.no_progress {
        None
    } else {
        let pb = ProgressBar::new(files.len() as u64);
        pb.set_style(ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, {eta})")
            .unwrap()
            .progress_chars("#>-"));
        Some(pb)
    };
    
    // Process files in parallel
    let results = process_files_parallel(files, temp_dir.path(), &options, threads, progress.as_ref())?;
    
    // Write CSV output
    let output_path = args.output_csv.unwrap_or_else(|| PathBuf::from("benchmark_results.csv"));
    write_csv_results(&results, &output_path)?;
    
    let end = Instant::now();
    let total_time = (end - start).as_secs_f64();
    
    eprintln!("\nBenchmark completed in {:.2} seconds", total_time);
    eprintln!("Results written to {}", output_path.display());
    eprintln!("Processed {} files with {} threads", results.len(), threads);
    
    // Summary stats
    let successful = results.iter().filter(|r| r.status == "success").count();
    let failed = results.len() - successful;
    eprintln!("Success: {}, Failed: {}", successful, failed);
    
    Ok(())
}

fn discover_supported_files(directory: &Path) -> io::Result<Vec<PathBuf>> {
    let mut supported_files = Vec::new();
    
    for entry in WalkDir::new(directory) {
        let entry = entry?;
        let path = entry.path();
        
        // Try to infer format using mzdata
        if is_supported_format(path) {
            supported_files.push(path.to_path_buf());
        }
    }
    
    Ok(supported_files)
}

fn is_supported_format(path: &Path) -> bool {
    // Use mzdata's format inference
    match infer_format(path) {
        Ok((format, _)) => {
            // Debug: Print file name and detected format
            log::debug!("File: {:?}, Format: {:?}", path.file_name().unwrap_or_default(), format);
            format != MassSpectrometryFormat::Unknown
        },
        Err(_) => false, // Can't read file or determine format
    }
}

fn process_files_parallel(
    files: Vec<PathBuf>,
    temp_dir: &Path,
    options: &ConvertOptions,
    max_threads: usize,
    progress: Option<&ProgressBar>,
) -> io::Result<Vec<BenchmarkResult>> {
    let results = Arc::new(Mutex::new(Vec::new()));
    let work_queue = Arc::new(Mutex::new(files.into_iter().collect::<VecDeque<_>>()));
    
    // Spawn worker threads
    let mut handles = Vec::new();
    for _ in 0..max_threads {
        let work_queue = Arc::clone(&work_queue);
        let results = Arc::clone(&results);
        let options = options.clone();
        let temp_dir = temp_dir.to_path_buf();
        let progress = progress.map(|p| p.clone());
        
        let handle = thread::spawn(move || {
            loop {
                let file_path = {
                    let mut queue = work_queue.lock().unwrap();
                    queue.pop_front()
                };
                
                match file_path {
                    Some(path) => {
                        let result = panic::catch_unwind(AssertUnwindSafe(|| {
                            process_single_file(path.clone(), &temp_dir, &options)
                        }));
                        
                        let benchmark_result = match result {
                            Ok(result) => result,
                            Err(_) => {
                                // Thread panicked, create error result
                                let filename = path.file_name()
                                    .unwrap_or_default()
                                    .to_string_lossy()
                                    .to_string();
                                eprintln!("PANIC: Conversion failed for file: {}", path.display());
                                BenchmarkResult {
                                    filename,
                                    original_size: fs::metadata(&path).map(|m| m.len()).unwrap_or(0),
                                    final_size: 0,
                                    time_taken: 0.0,
                                    status: "error: conversion panicked".to_string(),
                                }
                            }
                        };
                        
                        results.lock().unwrap().push(benchmark_result);
                        if let Some(ref pb) = progress {
                            pb.inc(1);
                        }
                    }
                    None => break, // No more work
                }
            }
        });
        handles.push(handle);
    }
    
    // Wait for all workers to complete
    for handle in handles {
        if let Err(e) = handle.join() {
            eprintln!("Worker thread failed: {:?}", e);
        }
    }
    
    if let Some(pb) = progress {
        pb.finish_with_message("Done!");
    }
    
    let results = Arc::try_unwrap(results).unwrap().into_inner().unwrap();
    Ok(results)
}

fn process_single_file(
    file_path: PathBuf,
    temp_dir: &Path,
    options: &ConvertOptions,
) -> BenchmarkResult {
    let filename = file_path.file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    
    // Get original file size
    let original_size = match fs::metadata(&file_path) {
        Ok(metadata) => metadata.len(),
        Err(_) => {
            return BenchmarkResult {
                filename,
                original_size: 0,
                final_size: 0,
                time_taken: 0.0,
                status: "error: could not read file metadata".to_string(),
            };
        }
    };
    
    // Create output path in temp directory
    let output_path = temp_dir.join(format!("{}.mzpeak", filename));
    
    // Time the conversion
    let start = Instant::now();
    let conversion_result = convert_file(&file_path, &output_path, options);
    let end = Instant::now();
    let time_taken = (end - start).as_secs_f64();
    
    match conversion_result {
        Ok(()) => {
            // Get final file size
            let final_size = match fs::metadata(&output_path) {
                Ok(metadata) => metadata.len(),
                Err(_) => 0,
            };
            
            // Clean up the converted file
            let _ = fs::remove_file(&output_path);
            
            BenchmarkResult {
                filename,
                original_size,
                final_size,
                time_taken,
                status: "success".to_string(),
            }
        }
        Err(e) => {
            // Clean up any partial file
            let _ = fs::remove_file(&output_path);
            
            BenchmarkResult {
                filename,
                original_size,
                final_size: 0,
                time_taken,
                status: format!("error: {}", e),
            }
        }
    }
}

fn convert_file(input_path: &Path, output_path: &Path, options: &ConvertOptions) -> io::Result<()> {
    let mut reader = MZReaderType::<_, CentroidPeak, DeconvolutedPeak>::open_path(input_path)
        .inspect_err(|e| eprintln!("Failed to open data file: {e}"))?;

    let overrides = create_type_overrides(options);

    let handle = fs::File::create(output_path)?;
    let mut writer = MzPeakWriterType::<fs::File>::builder()
        .add_spectrum_peak_type::<CentroidPeak>()
        .add_spectrum_peak_type::<DeconvolutedPeak>()
        .add_default_chromatogram_fields()
        .buffer_size(5000)
        .shuffle_mz(options.shuffle_mz);

    if options.null_zeros {
        writer = writer.null_zeros(true);
    }

    writer = sample_array_types::<CentroidPeak, DeconvolutedPeak>(&mut reader, &overrides)
        .into_iter()
        .fold(writer, |writer, f| writer.add_spectrum_field(f));

    for (from, to) in overrides.iter() {
        writer = writer.add_spectrum_override(from.clone(), to.clone());
    }

    let mut writer = writer.build(handle, true);
    writer.copy_metadata_from(&reader);
    writer.add_file_description(reader.file_description());

    let (send, recv) = sync_channel(1);

    let read_handle = thread::spawn(move || {
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            for entry in reader.into_iter() {
                if send.send(entry).is_err() {
                    break; // Receiver dropped
                }
            }
        }));
        if result.is_err() {
            eprintln!("Reader thread panicked");
        }
    });

    let write_handle = thread::spawn(move || {
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            for (i, mut batch) in recv.into_iter().enumerate() {
                if i % 5000 == 0 {
                    log::info!("Writing batch {i}");
                }
                batch.peaks = None;
                writer.write_spectrum(&batch)?;
            }
            writer.finish()
        }));
        match result {
            Ok(Ok(())) => {},
            Ok(Err(e)) => eprintln!("Writer thread error: {}", e),
            Err(_) => eprintln!("Writer thread panicked"),
        }
    });

    if let Err(e) = read_handle.join() {
        eprintln!("Failed to join reader thread: {:?}", e);
        return Err(io::Error::new(io::ErrorKind::Other, "Reader thread failed"));
    }
    
    if let Err(e) = write_handle.join() {
        eprintln!("Failed to join writer thread: {:?}", e);
        return Err(io::Error::new(io::ErrorKind::Other, "Writer thread failed"));
    }

    Ok(())
}

fn create_type_overrides(options: &ConvertOptions) -> HashMap<BufferName, BufferName> {
    let mut overrides = HashMap::new();
    
    if options.mz_f32 {
        overrides.insert(
            BufferName::new(
                BufferContext::Spectrum,
                ArrayType::MZArray,
                BinaryDataArrayType::Float64,
            ),
            BufferName::new(
                BufferContext::Spectrum,
                ArrayType::MZArray,
                BinaryDataArrayType::Float32,
            ),
        );
    }
    
    if options.intensity_f32 {
        overrides.insert(
            BufferName::new(
                BufferContext::Spectrum,
                ArrayType::IntensityArray,
                BinaryDataArrayType::Float64,
            ),
            BufferName::new(
                BufferContext::Spectrum,
                ArrayType::IntensityArray,
                BinaryDataArrayType::Float32,
            ),
        );
    }
    
    if options.intensity_i32 {
        overrides.insert(
            BufferName::new(
                BufferContext::Spectrum,
                ArrayType::IntensityArray,
                BinaryDataArrayType::Float32,
            ),
            BufferName::new(
                BufferContext::Spectrum,
                ArrayType::IntensityArray,
                BinaryDataArrayType::Int32,
            ),
        );
        overrides.insert(
            BufferName::new(
                BufferContext::Spectrum,
                ArrayType::IntensityArray,
                BinaryDataArrayType::Float64,
            ),
            BufferName::new(
                BufferContext::Spectrum,
                ArrayType::IntensityArray,
                BinaryDataArrayType::Int32,
            ),
        );
    }
    
    if options.ion_mobility_f32 {
        for t in [
            ArrayType::MeanInverseReducedIonMobilityArray,
            ArrayType::RawDriftTimeArray,
            ArrayType::RawIonMobilityArray,
        ] {
            overrides.insert(
                BufferName::new(
                    BufferContext::Spectrum,
                    t.clone(),
                    BinaryDataArrayType::Float64,
                ),
                BufferName::new(
                    BufferContext::Spectrum,
                    t.clone(),
                    BinaryDataArrayType::Float32,
                ),
            );
        }
    }
    
    overrides
}

fn write_csv_results(results: &[BenchmarkResult], output_path: &Path) -> io::Result<()> {
    let mut writer = Writer::from_path(output_path)?;
    
    // Write header
    writer.write_record(&["filename", "originalsize", "finalsize", "timetaken", "status"])?;
    
    // Write results
    for result in results {
        writer.write_record(&[
            &result.filename,
            &result.original_size.to_string(),
            &result.final_size.to_string(),
            &format!("{:.3}", result.time_taken),
            &result.status,
        ])?;
    }
    
    writer.flush()?;
    Ok(())
}
