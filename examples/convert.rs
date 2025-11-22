use clap::Parser;
use mzdata::{
    self,
    io::MZReaderType,
    params::Unit,
    prelude::*,
    spectrum::{ArrayType, BinaryDataArrayType, SignalContinuity},
};
use mzpeak_prototyping::{
    buffer_descriptors::{BufferOverrideTable, BufferTransform},
    chunk_series::ChunkingStrategy,
    peak_series::{BufferContext, BufferName},
    writer::{
        AbstractMzPeakWriter, ArrayBuffersBuilder, MzPeakWriterType,
        sample_array_types_from_file_reader,
    },
};
use mzpeaks::{CentroidPeak, DeconvolutedPeak};
use parquet::basic::{Compression, ZstdLevel};
use std::{
    collections::HashMap,
    fmt::Debug,
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

/// Convert a single mass spectrometry file to mzPeak format
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

#[derive(Clone, Copy)]
pub struct SuffixedSize(usize);

impl Debug for SuffixedSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<usize> for SuffixedSize {
    fn from(value: usize) -> Self {
        Self(value)
    }
}

impl From<SuffixedSize> for usize {
    fn from(value: SuffixedSize) -> Self {
        value.0
    }
}

impl std::str::FromStr for SuffixedSize {
    type Err = std::num::ParseIntError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if let Some(c) = s.chars().last() {
            let val = match c {
                'K' => 2usize.pow(10u32),
                'M' => 2usize.pow(20u32),
                'G' => 2usize.pow(30u32),
                _ => {
                    panic!(
                        "Unrecognized suffix {c}, accepts K (2**10), M(2**20), or G(2**30) or no suffix"
                    )
                }
            } * s[..s.len().saturating_sub(1)].parse::<usize>()?;
            Ok(Self(val))
        } else {
            s.parse().map(Self)
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
        long = "intensity-numpress-slof",
        help = "Encode the intensity values using the Numpress Short Logged Float transform. This requires the chunked encoding."
    )]
    pub intensity_slof: bool,

    #[arg(
        short = 'i',
        long = "intensity-i32",
        help = "Encode the intensity values as int32 instead of floats which may improve compression at the cost of the decimal component"
    )]
    pub intensity_i32: bool,

    #[arg(
        short = 'z',
        long = "shuffle-mz",
        help = "Shuffle the m/z array, which can improve the compression of profile spectra or densely packed centroids."
    )]
    pub shuffle_mz: bool,

    #[arg(
        short = 'u',
        long,
        help = "Null mask out sparse zero intensity peaks. This is appropriate for *sparse* data with many gaps."
    )]
    pub null_zeros: bool,

    #[arg(short = 'o', long, help = "Output file path")]
    pub outpath: Option<PathBuf>,

    #[arg(
        short,
        long,
        default_value_t = 5000,
        help = "The number of spectra to buffer between writes"
    )]
    pub buffer_size: usize,

    #[arg(
        long,
        help = "The number of rows to write in a batch between deciding to open a new page or row group segment. Defaults to 1K. Supports SI suffixes K, M, G."
    )]
    pub write_batch_size: Option<SuffixedSize>,

    #[arg(
        long,
        help = "The approximate number of *bytes* per data page. Defaults to 1M. Supports SI suffixes K, M, G."
    )]
    pub data_page_size: Option<SuffixedSize>,

    #[arg(
        long,
        help = "The approximate number of rows per row group. Defaults to 1M. Supports SI suffixes K, M, G."
    )]
    pub row_group_size: Option<SuffixedSize>,

    #[arg(
        long,
        help = "The approximate number of *bytes* per dictionary page. Defaults to 1M. Supports SI suffixes K, M, G."
    )]
    pub dictionary_page_size: Option<SuffixedSize>,

    #[arg(
        short,
        long,
        help = "Use the chunked encoding instead of the flat peak array layout, valid options are 'delta', 'basic', 'numpress'. You can also specify a chunk size like 'delta:50'. Defaults to 'delta:50'",
        value_parser=chunk_encoding_parser,
        default_missing_value="delta:50",
        num_args=0..=1,
    )]
    pub chunked_encoding: Option<ChunkingStrategy>,

    #[arg(
        short = 'k',
        long,
        default_value_t = 3,
        help = "The Zstd compression level to use. Defaults to 3, but ranges from 1-22"
    )]
    pub compression_level: i32,

    #[arg(
        short = 'p',
        long,
        help = "Whether or not to write both profile and peak picked data in the same file."
    )]
    pub write_peaks_and_profiles: bool,

    #[arg(
        short = 't',
        long,
        help = "Include an extra 'spectrum_time' array alongside the 'spectrum_index' array."
    )]
    pub include_time_with_spectrum_data: bool,
}

impl ConvertArgs {
    pub fn create_type_overrides(&self) -> BufferOverrideTable {
        let mut overrides = HashMap::new();

        if self.mz_f32 {
            for unit in [Unit::MZ, Unit::Unknown] {
                overrides.insert(
                    BufferName::new(
                        BufferContext::Spectrum,
                        ArrayType::MZArray,
                        BinaryDataArrayType::Float64,
                    )
                    .with_unit(unit),
                    BufferName::new(
                        BufferContext::Spectrum,
                        ArrayType::MZArray,
                        BinaryDataArrayType::Float32,
                    )
                    .with_unit(unit),
                );
            }
        }

        let intensity_transform = self.intensity_slof.then(|| BufferTransform::NumpressSLOF);
        if self.chunked_encoding.is_none() && intensity_transform.is_some() {
            log::warn!(
                "Signal transforms require chunked encoding, without chunked encoding this will have no effect"
            );
        }
        for ctx in [BufferContext::Chromatogram, BufferContext::Spectrum] {
            for unit in [
                Unit::Unknown,
                Unit::DetectorCounts,
                Unit::PercentBasePeak,
                Unit::PercentBasePeakTimes100,
                Unit::AbsorbanceUnit,
                Unit::CountsPerSecond,
                Unit::Pascal,
                Unit::Percent,
                Unit::Psi,
                Unit::Kelvin,
                Unit::MicrolitersPerMinute,
            ] {
                if self.intensity_f32 {
                    overrides.insert(
                        BufferName::new(
                            ctx,
                            ArrayType::IntensityArray,
                            BinaryDataArrayType::Float64,
                        )
                        .with_unit(unit),
                        BufferName::new(
                            ctx,
                            ArrayType::IntensityArray,
                            BinaryDataArrayType::Float32,
                        )
                        .with_unit(unit)
                        .with_transform(intensity_transform),
                    );
                }
                if intensity_transform.is_some() {
                    overrides.insert(
                        BufferName::new(
                            ctx,
                            ArrayType::IntensityArray,
                            BinaryDataArrayType::Float32,
                        )
                        .with_unit(unit),
                        BufferName::new(
                            ctx,
                            ArrayType::IntensityArray,
                            BinaryDataArrayType::Float32,
                        )
                        .with_unit(unit)
                        .with_transform(intensity_transform),
                    );
                }
                if self.intensity_i32 {
                    overrides.insert(
                        BufferName::new(
                            ctx,
                            ArrayType::IntensityArray,
                            BinaryDataArrayType::Float32,
                        )
                        .with_unit(unit),
                        BufferName::new(ctx, ArrayType::IntensityArray, BinaryDataArrayType::Int32)
                            .with_unit(unit)
                            .with_transform(intensity_transform),
                    );
                    overrides.insert(
                        BufferName::new(
                            ctx,
                            ArrayType::IntensityArray,
                            BinaryDataArrayType::Float64,
                        )
                        .with_unit(unit),
                        BufferName::new(ctx, ArrayType::IntensityArray, BinaryDataArrayType::Int32)
                            .with_unit(unit)
                            .with_transform(intensity_transform),
                    );
                }
            }
        }
        if self.ion_mobility_f32 {
            for unit in [
                Unit::Unknown,
                Unit::Millisecond,
                Unit::VoltSecondPerSquareCentimeter,
                Unit::Volt,
            ] {
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
                        )
                        .with_unit(unit),
                        BufferName::new(
                            BufferContext::Spectrum,
                            t.clone(),
                            BinaryDataArrayType::Float32,
                        )
                        .with_unit(unit),
                    );
                }
            }
        }
        overrides.into()
    }
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

pub fn configure_writer_builder(
    args: &ConvertArgs,
) -> mzpeak_prototyping::writer::MzPeakWriterBuilder {
    MzPeakWriterType::<fs::File>::builder()
        .add_default_chromatogram_fields()
        .buffer_size(args.buffer_size)
        .include_time_with_spectrum_data(args.include_time_with_spectrum_data)
        .shuffle_mz(args.shuffle_mz)
        .chunked_encoding(args.chunked_encoding)
        .null_zeros(args.null_zeros)
        .write_batch_size(args.write_batch_size.map(usize::from))
        .page_size(
            args.data_page_size
                .inspect(|e| log::debug!("data page size: {e:?}"))
                .map(usize::from),
        )
        .dictionary_page_size(
            args.dictionary_page_size
                .inspect(|e| log::debug!("dictionary page size: {e:?}"))
                .map(usize::from),
        )
        .row_group_size(
            args.row_group_size
                .inspect(|e| log::debug!("row group size: {e:?}"))
                .map(usize::from),
        )
        .compression(Compression::ZSTD(
            ZstdLevel::try_new(args.compression_level).unwrap(),
        ))
}

pub fn add_processing_metadata(writer: &mut MzPeakWriterType<fs::File>) {
    writer.softwares_mut().push(mzdata::meta::Software::new(
        "mzpeak_prototyping_convert1".into(),
        "0.1.0".into(),
        vec![mzdata::meta::custom_software_name(
            "mzpeak_prototyping_convert",
        )],
    ));
    writer
        .data_processings_mut()
        .push(mzdata::meta::DataProcessing {
            id: "mzpeak_conversion1".to_string(),
            methods: vec![mzdata::meta::ProcessingMethod {
                order: 1,
                software_reference: "mzpeak_prototyping_convert1".to_string(),
                params: vec![mzdata::Param::new_key_value(
                    "conversion options",
                    std::env::args().skip(1).collect::<Vec<String>>().join(" "),
                )],
            }],
        });
}

pub fn convert_file(input_path: &Path, output_path: &Path, args: &ConvertArgs) -> io::Result<()> {
    let mut reader = MZReaderType::<_, CentroidPeak, DeconvolutedPeak>::open_path(&input_path)
        .inspect_err(|e| eprintln!("Failed to open data file: {e}"))?;

    let n = reader.len();
    let overrides = args.create_type_overrides();

    if let Some(c) = args.chunked_encoding.as_ref() {
        log::debug!("Using chunking method {c:?}");
    }

    let handle = fs::File::create(output_path)?;
    let mut writer = configure_writer_builder(&args);

    if args.write_peaks_and_profiles {
        let mut point_builder = ArrayBuffersBuilder::default()
            .prefix("point")
            .with_context(BufferContext::Spectrum);
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
        writer = writer.add_chromatogram_override(from.clone(), to.clone());
    }

    let mut writer = writer.build(handle, true);
    writer.copy_metadata_from(&reader);
    add_processing_metadata(&mut writer);

    let (send, recv) = sync_channel(1);

    let write_peaks_and_profiles = args.write_peaks_and_profiles;
    let read_handle = thread::spawn(move || {
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            match &mut reader {
                MZReaderType::BrukerTDF(tdfspectrum_reader_type) => {
                    tdfspectrum_reader_type.set_consolidate_peaks(false);
                }
                _ => {}
            }
            for mut entry in reader.iter() {
                if write_peaks_and_profiles && entry.peaks.is_none() {
                    entry.pick_peaks(3.0).unwrap();
                }
                if send.send((Some(entry), None)).is_err() {
                    break; // Receiver dropped
                }
            }

            for entry in reader.iter_chromatograms() {
                if send.send((None, Some(entry))).is_err() {
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
            for (i, (spectrum, chromatogram)) in recv.into_iter().enumerate() {
                if i % 5000 == 0 {
                    log::info!("Writing batch {i} ({:0.2}%)", i as f64 / n as f64 * 100.0);
                }
                if i % 10 == 0 {
                    log::debug!("Writing batch {i} ({}%)", i as f64 / n as f64 * 100.0);
                }
                if let Some(mut spectrum) = spectrum {
                    if spectrum.signal_continuity() != SignalContinuity::Profile {
                        spectrum.peaks = None;
                    }
                    writer.write_spectrum(&spectrum)?;
                } else if let Some(chromatogram) = chromatogram {
                    writer.write_chromatogram(&chromatogram)?;
                }
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

#[cfg(test)]
mod test {}
