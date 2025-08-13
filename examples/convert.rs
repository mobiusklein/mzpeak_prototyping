use clap::Parser;
use mzdata::{
    self,
    io::MZReaderType,
    prelude::*,
    spectrum::{ArrayType, BinaryDataArrayType, SignalContinuity},
};
use mzpeak_prototyping::{
    chunk_series::ChunkingStrategy,
    peak_series::{BufferContext, BufferName},
    writer::{sample_array_types_from_file_reader, ArrayBuffersBuilder, MzPeakWriterType},
};
use mzpeaks::{CentroidPeak, DeconvolutedPeak};
use parquet::basic::{Compression, ZstdLevel};
use std::{
    collections::HashMap,
    fs, io,
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::mpsc::sync_channel,
    thread,
    time::Instant,
};

// ============================================================================
// CLI Interface for standalone usage
// ============================================================================

#[derive(Parser, Debug, Clone)]
pub struct ConvertCli {
    /// Input file path
    pub filename: PathBuf,

    #[command(flatten)]
    pub convert_args: ConvertArgs,
}

#[allow(dead_code)]
fn main() -> io::Result<()> {
    env_logger::init();
    let cli = ConvertCli::parse();
    run_convert(&cli.filename, cli.convert_args)
}

// ============================================================================
// Public Library Interface
// ============================================================================

fn chunk_encoding_parser(method_str: &str) -> Result<ChunkingStrategy, String> {
    if let Some((method, chunk_size)) = method_str
        .split_once(":")
        .or_else(|| Some((method_str, "50")))
    {
        let chunk_size = chunk_size.parse::<f64>().unwrap_or(50.0);
        let v = match method.to_ascii_lowercase().as_str() {
            "delta" => ChunkingStrategy::Delta { chunk_size },
            "basic" | "plain" => ChunkingStrategy::Basic { chunk_size },
            "numpress" => ChunkingStrategy::NumpressLinear { chunk_size },
            _ => {
                log::warn!("Failed to parse {method}, defaulting to delta encoding");
                ChunkingStrategy::Delta { chunk_size }
            }
        };
        Ok(v)
    } else {
        if method_str == "" {
            Ok(ChunkingStrategy::Delta { chunk_size: 50.0 })
        } else {
            Err(format!("Failed to parse {method_str}"))
        }
    }
}

#[derive(Parser, Debug, Clone)]
pub struct ConvertArgs {
    #[arg(
        short = 'm',
        long = "mz-f32",
        help = "Encode the m/z values using float32 instead of float64"
    )]
    pub mz_f32: bool,

    #[arg(
        short = 'd',
        long = "ion-mobility-f32",
        help = "Encode the ion mobility values using float32 instead of float64"
    )]
    pub ion_mobility_f32: bool,

    #[arg(
        short = 'y',
        long = "intensity-f32",
        help = "Encode the intensity values using float32"
    )]
    pub intensity_f32: bool,

    #[arg(
        short = 'i',
        long = "intensity-i32",
        help = "Encode the intensity values as int32 instead of floats which may improve compression at the cost of the decimal component"
    )]
    pub intensity_i32: bool,

    #[arg(
        short = 'z',
        long = "shuffle-mz",
        help = "Shuffle the m/z array, which may improve the compression of profile spectra"
    )]
    pub shuffle_mz: bool,

    #[arg(short = 'u', long, help = "Null mask out sparse zero intensity peaks")]
    pub null_zeros: bool,

    #[arg(short = 'o', long, help = "Output file path")]
    pub outpath: Option<PathBuf>,

    #[arg(
        short,
        long,
        default_value_t = 5000,
        help = "The number of spectra to buffer between writes"
    )]
    buffer_size: usize,

    #[arg(
        short,
        long,
        help = "Use the chunked encoding instead of the flat peak array layout",
        value_parser=chunk_encoding_parser,
        default_missing_value="delta:50",
        num_args=0..=1,
    )]
    chunked_encoding: Option<ChunkingStrategy>,

    #[arg(
        short = 'k',
        long,
        default_value_t = 3,
        help = "The Zstd compression level to use. Defaults to 3, but ranges from 1-22"
    )]
    compression_level: i32,

    #[arg(
        short = 'p',
        long,
        help = "Whether or not to write both profile and peak picked data in the same file."
    )]
    write_peaks_and_profiles: bool,
}

pub fn run_convert(filename: &Path, args: ConvertArgs) -> io::Result<()> {
    let start = Instant::now();

    let outpath = args
        .outpath
        .as_ref()
        .map(|p| p.clone())
        .unwrap_or_else(|| filename.with_extension("mzpeak"));

    convert_file(filename, &outpath, &args)?;

    eprintln!("{:0.2} seconds elapsed", start.elapsed().as_secs_f64());

    let stat = fs::metadata(&outpath)?;
    let size = stat.len() as f64 / 1e9;
    eprintln!("{} was {size:0.3}GB", outpath.display());

    Ok(())
}

pub fn convert_file(input_path: &Path, output_path: &Path, args: &ConvertArgs) -> io::Result<()> {
    let mut reader = MZReaderType::<_, CentroidPeak, DeconvolutedPeak>::open_path(&input_path)
        .inspect_err(|e| eprintln!("Failed to open data file: {e}"))?;

    let overrides = create_type_overrides(args);

    if let Some(c) = args.chunked_encoding.as_ref() {
        log::debug!("Using chunking method {c:?}");
    }

    let handle = fs::File::create(output_path)?;
    let mut writer = MzPeakWriterType::<fs::File>::builder()
        .add_default_chromatogram_fields()
        .buffer_size(args.buffer_size)
        .shuffle_mz(args.shuffle_mz)
        .chunked_encoding(args.chunked_encoding)
        .null_zeros(args.null_zeros)
        .compression(Compression::ZSTD(
            ZstdLevel::try_new(args.compression_level).unwrap(),
        ));

    if args.write_peaks_and_profiles {
        let mut point_builder = ArrayBuffersBuilder::default().prefix("point");
        for f in sample_array_types_from_file_reader::<CentroidPeak, DeconvolutedPeak>(
            &mut reader,
            &overrides,
            None,
        ) {
            point_builder = point_builder.add_field(f);
        }
        writer = writer.store_peaks_and_profiles_apart(Some(point_builder));
    }

    writer = sample_array_types_from_file_reader::<CentroidPeak, DeconvolutedPeak>(
        &mut reader,
        &overrides,
        args.chunked_encoding,
    )
    .into_iter()
    .fold(writer, |writer, f| writer.add_spectrum_field(f));

    for (from, to) in overrides.iter() {
        writer = writer.add_spectrum_override(from.clone(), to.clone());
    }

    let mut writer = writer.build(handle, true);
    writer.copy_metadata_from(&reader);
    writer.add_file_description(reader.file_description());

    let (send, recv) = sync_channel(1);

    let write_peaks_and_profiles = args.write_peaks_and_profiles;
    let read_handle = thread::spawn(move || {
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            for mut entry in reader.into_iter() {
                if entry.has_ion_mobility_dimension() {
                    if let Some(mut arrays) = entry.arrays {
                        arrays.sort_by_array(&ArrayType::MZArray).unwrap();
                        entry.arrays = Some(arrays);
                    }
                } else if write_peaks_and_profiles && entry.peaks.is_none() {
                    entry.pick_peaks(3.0).unwrap();
                }
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
                if i % 10 == 0 {
                    log::debug!("Writing batch {i}");
                }
                if batch.signal_continuity() != SignalContinuity::Profile {
                    batch.peaks = None;
                }
                writer.write_spectrum(&batch)?;
            }
            writer.finish()
        }));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                eprintln!("Writer thread error: {}", e);
            }
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

#[cfg(test)]
mod test {}
