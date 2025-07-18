use clap::Parser;
use mzdata::{
    self,
    io::MZReaderType,
    prelude::*,
    spectrum::{ArrayType, BinaryDataArrayType},
};
use mzpeak_prototyping::{writer::sample_array_types, *};
use mzpeaks::{CentroidPeak, DeconvolutedPeak};
use std::{
    collections::HashMap,
    fs,
    io,
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::mpsc::sync_channel,
    thread,
    time::Instant,
};


// ============================================================================
// CLI Interface for standalone usage
// ============================================================================

#[allow(dead_code)]
fn main() -> io::Result<()> {
    env_logger::init();
    let args = ConvertArgs::parse();
    run_convert(args)
}


// ============================================================================
// Public Library Interface
// ============================================================================

#[derive(Parser, Debug, Clone)]
pub struct ConvertArgs {
    /// Input file path
    pub filename: PathBuf,

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

    #[arg(short = 'o', help="Output file path")]
    pub outpath: Option<PathBuf>,
}

pub fn run_convert(args: ConvertArgs) -> io::Result<()> {
    let start = Instant::now();
    
    let outpath = args.outpath.as_ref().map(|p| p.clone()).unwrap_or_else(|| {
        args.filename.with_extension("mzpeak")
    });
    
    convert_file(&args.filename, &outpath, &args)?;
    
    let end = Instant::now();
    eprintln!("{:0.2} seconds elapsed", (end - start).as_secs_f64());
    
    let stat = fs::metadata(&outpath)?;
    let size = stat.len() as f64 / 1e9;
    eprintln!("{} was {size:0.3}GB", outpath.display());
    
    Ok(())
}

pub fn convert_file(input_path: &Path, output_path: &Path, args: &ConvertArgs) -> io::Result<()> {
    let mut reader = MZReaderType::<_, CentroidPeak, DeconvolutedPeak>::open_path(input_path)
        .inspect_err(|e| eprintln!("Failed to open data file: {e}"))?;

    let overrides = create_type_overrides(args);

    let handle = fs::File::create(output_path)?;
    let mut writer = MzPeakWriterType::<fs::File>::builder()
        .add_spectrum_peak_type::<CentroidPeak>()
        .add_spectrum_peak_type::<DeconvolutedPeak>()
        .add_default_chromatogram_fields()
        .buffer_size(5000)
        .shuffle_mz(args.shuffle_mz);

    if args.null_zeros {
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

pub fn create_type_overrides(args: &ConvertArgs) -> HashMap<BufferName, BufferName> {
    let mut overrides = HashMap::new();
    
    if args.mz_f32 {
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
    
    if args.intensity_f32 {
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
    
    if args.intensity_i32 {
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
    
    if args.ion_mobility_f32 {
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
