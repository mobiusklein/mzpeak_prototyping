use clap::Parser;
use mzdata::{
    self,
    io::MZReaderType,
    prelude::*,
    spectrum::{ArrayType, BinaryDataArrayType},
};
use mzpeak_prototyping::{writer::sample_array_types_from_file_reader, *};
use mzpeaks::{CentroidPeak, DeconvolutedPeak};
use std::{collections::HashMap, fs, io, path::PathBuf, sync::mpsc::sync_channel, thread};

#[derive(Debug, Parser)]
pub struct App {
    #[arg()]
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

    #[arg(short = 'o')]
    outpath: Option<PathBuf>,

    #[arg(short, long, default_value_t=5000, help="The number of spectra to buffer between writes")]
    buffer_size: usize,

    #[arg(short, long, help="Use the chunked encoding instead of the flat peak array layout")]
    chunked_encoding: bool
}

fn main() -> io::Result<()> {
    env_logger::init();
    let args = App::parse();
    let filename = args.filename;

    let start = std::time::Instant::now();

    if filename.to_string_lossy() == "-" {

    }



    let mut reader = MZReaderType::<_, CentroidPeak, DeconvolutedPeak>::open_path(&filename)
        .inspect_err(|e| eprintln!("Failed to open data file: {e}"))?;

    let outname = args
        .outpath
        .unwrap_or_else(|| filename.with_extension("mzpeak").file_name().unwrap().into());
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

    let handle = fs::File::create(&outname.file_name().unwrap())?;
    let mut writer = MzPeakWriterType::<fs::File>::builder()
        // If we had peak data here, we would want to add these types' arrays
        // .add_spectrum_peak_type::<CentroidPeak>()
        // .add_spectrum_peak_type::<DeconvolutedPeak>()
        .add_default_chromatogram_fields()
        .buffer_size(args.buffer_size)
        .shuffle_mz(args.shuffle_mz);

    if args.null_zeros {
        writer = writer.null_zeros(true)
    }

    writer = sample_array_types_from_file_reader::<CentroidPeak, DeconvolutedPeak>(&mut reader, &overrides, args.chunked_encoding)
        .into_iter()
        .fold(writer, |writer, f| writer.add_spectrum_field(f));

    for (from, to) in overrides.iter() {
        writer = writer.add_spectrum_override(from.clone(), to.clone());
    }
    writer = writer.chunked_encoding(args.chunked_encoding);

    let mut writer = writer.build(handle, true);
    writer.copy_metadata_from(&reader);
    writer.add_file_description(reader.file_description());

    let (send, recv) = sync_channel(1);

    let read_handle = thread::spawn(move || {
        for entry in reader.into_iter() {
            send.send(entry).unwrap();
        }
    });

    let write_handle = thread::spawn(move || {
        for (i, mut batch) in recv.into_iter().enumerate() {
            if i % 5000 == 0 {
                log::info!("Writing batch {i}");
            }
            if i % 10 == 0 {
                log::debug!("Writing batch {i}");
            }
            batch.peaks = None;
            writer.write_spectrum(&batch).unwrap();
        }
        writer.finish().unwrap();
    });

    read_handle.join().unwrap();
    write_handle.join().unwrap();

    let end = std::time::Instant::now();
    eprintln!("{:0.2} seconds elapsed", (end - start).as_secs_f64());

    let stat = fs::metadata(&outname)?;
    let size = stat.len() as f64 / 1e9;
    eprintln!("{} was {size:0.3}GB", outname.display());
    Ok(())
}
