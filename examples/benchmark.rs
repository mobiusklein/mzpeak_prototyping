use clap::Parser;
use csv::Writer;
use indicatif::{ProgressBar, ProgressStyle};
use mzdata::{self, io::{MassSpectrometryFormat, infer_format}};
use std::{
    collections::VecDeque,
    fs, io,
    path::{Path, PathBuf},
    panic::{self, AssertUnwindSafe},
    sync::{Arc, Mutex},
    thread,
    time::Instant
};
use tempfile::TempDir;
use walkdir::WalkDir;

// Import convert functionality - include it directly for standalone usage
mod convert {
    include!("convert.rs");
}
use convert::{ConvertArgs, convert_file};

// ============================================================================
// Public Library Interface
// ============================================================================

#[derive(Parser, Debug, Clone)]
pub struct BenchmarkArgs {
    /// Directory to scan for mass spectrometry files
    pub directory: PathBuf,

    #[arg(short = 'o', long = "output-csv", help="Path to save benchmark results CSV")]
    pub output_csv: Option<PathBuf>,

    #[arg(short = 't', long = "threads", help="Number of threads to use (default: number of CPU cores)")]
    pub threads: Option<usize>,

    #[arg(long = "temp-dir", help="Temporary directory for converted files")]
    pub temp_dir: Option<PathBuf>,

    #[arg(long = "no-progress", help="Disable progress bar")]
    pub no_progress: bool,

    // Include all conversion options
    #[arg(short = 'm', long = "mz-f32", help="Encode the m/z values using float32 instead of float64")]
    pub mz_f32: bool,

    #[arg(short = 'd', long = "ion-mobility-f32", help="Encode the ion mobility values using float32 instead of float64")]
    pub ion_mobility_f32: bool,

    #[arg(short = 'y', long = "intensity-f32", help="Encode the intensity values using float32")]
    pub intensity_f32: bool,

    #[arg(short = 'i',
          long = "intensity-i32",
          help="Encode the intensity values as int32 instead of floats which may improve compression at the cost of the decimal component")]
    pub intensity_i32: bool,

    #[arg(short = 'z', long = "shuffle-mz", help="Shuffle the m/z array, which may improve the compression of profile spectra")]
    pub shuffle_mz: bool,

    #[arg(short='u', long, help="Null mask out sparse zero intensity peaks")]
    pub null_zeros: bool,
}

#[derive(Debug)]
pub struct BenchmarkResult {
    pub filename: String,
    pub original_size: u64,
    pub final_size: u64,
    pub time_taken: f64, // seconds
    pub status: String,
}

pub fn run_benchmark(args: BenchmarkArgs) -> io::Result<()> {
    let start = Instant::now();
    
    // Setup
    let threads = args.threads.unwrap_or_else(num_cpus::get);
    let temp_dir = if let Some(temp_dir) = &args.temp_dir {
        TempDir::new_in(temp_dir)?
    } else {
        TempDir::new()?
    };
    
    let convert_args = ConvertArgs {
        filename: PathBuf::new(), // Will be set per file
        mz_f32: args.mz_f32,
        ion_mobility_f32: args.ion_mobility_f32,
        intensity_f32: args.intensity_f32,
        intensity_i32: args.intensity_i32,
        shuffle_mz: args.shuffle_mz,
        null_zeros: args.null_zeros,
        outpath: None, // Will be set per file
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
    let results = process_files_parallel(files, temp_dir.path(), &convert_args, threads, progress.as_ref())?;
    
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

pub fn discover_supported_files(directory: &Path) -> io::Result<Vec<PathBuf>> {
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

pub fn is_supported_format(path: &Path) -> bool {
    // Use mzdata's format inference
    match infer_format(path) {
        Ok((format, _)) => format != MassSpectrometryFormat::Unknown,
        Err(_) => false, // Can't read file or determine format
    }
}

pub fn process_files_parallel(
    files: Vec<PathBuf>,
    temp_dir: &Path,
    convert_args: &ConvertArgs,
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
        let convert_args = convert_args.clone();
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
                            process_single_file(path.clone(), &temp_dir, &convert_args)
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

pub fn process_single_file(
    file_path: PathBuf,
    temp_dir: &Path,
    convert_args: &ConvertArgs,
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
    let file_convert_args = ConvertArgs {
        filename: file_path.clone(),
        mz_f32: convert_args.mz_f32,
        ion_mobility_f32: convert_args.ion_mobility_f32,
        intensity_f32: convert_args.intensity_f32,
        intensity_i32: convert_args.intensity_i32,
        shuffle_mz: convert_args.shuffle_mz,
        null_zeros: convert_args.null_zeros,
        outpath: Some(output_path.clone()),
    };
    let conversion_result = convert_file(&file_path, &output_path, &file_convert_args);
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

pub fn write_csv_results(results: &[BenchmarkResult], output_path: &Path) -> io::Result<()> {
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

// ============================================================================
// CLI Interface for standalone usage
// ============================================================================

#[allow(dead_code)]
fn main() -> io::Result<()> {
    env_logger::init();
    let args = BenchmarkArgs::parse();
    run_benchmark(args)
}