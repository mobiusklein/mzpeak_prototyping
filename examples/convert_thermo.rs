use arrow::array::{ArrayRef, Float32Array, Float64Array, Int32Array, UInt64Array};
use clap::Parser;
use mzdata::{
    self,
    io::MZReaderType,
    params::Unit,
    prelude::*,
    spectrum::{
        ArrayType, BinaryArrayMap, BinaryDataArrayType, DataArray, SignalContinuity,
        bindata::BinaryCompressionType,
    },
};
use mzpeak_prototyping::{
    ToMzPeakDataSeries,
    buffer_descriptors::BufferOverrideTable,
    peak_series::{BufferContext, BufferName},
    writer::{AbstractMzPeakWriter, ArrayBuffersBuilder, sample_array_types_from_file_reader},
};
use mzpeaks::{CentroidPeak, DeconvolutedPeak};
use std::{
    fmt::Debug,
    fs, io,
    panic::{self, AssertUnwindSafe},
    path::{Path, PathBuf},
    sync::{Arc, mpsc::sync_channel},
    thread,
    time::Instant,
};

mod convert;

/// Convert a single Thermo RAW file to mzPeak format
#[derive(Parser, Debug, Clone)]
pub struct ConvertCli {
    /// Input file path
    pub filename: PathBuf,

    #[command(flatten)]
    pub convert_args: convert::ConvertArgs,
}

fn main() -> io::Result<()> {
    env_logger::init();
    let cli = ConvertCli::parse();
    let start = Instant::now();

    let outpath = cli
        .convert_args
        .outpath
        .as_ref()
        .map(|p| p.clone())
        .unwrap_or_else(|| cli.filename.with_extension("mzpeak"));

    convert_file(&cli.filename, &outpath, &cli.convert_args)?;

    eprintln!("{:0.2} seconds elapsed", start.elapsed().as_secs_f64());

    let stat = fs::metadata(&outpath)?;
    let size = stat.len() as f64 / 1e9;
    eprintln!("{} was {size:0.3}GB", outpath.display());
    Ok(())
}

#[derive(Debug, Default, Clone, PartialEq, PartialOrd)]
pub struct ThermoPeak {
    peak: mzpeaks::CentroidPeak,
    pub baseline: Option<f32>,
    pub noise: Option<f32>,
}

impl mzpeaks::CoordinateLike<mzpeaks::MZ> for ThermoPeak {
    fn coordinate(&self) -> f64 {
        self.peak.mz()
    }
}

impl mzpeaks::IntensityMeasurement for ThermoPeak {
    fn intensity(&self) -> f32 {
        self.peak.intensity()
    }
}

impl mzpeaks::IndexedCoordinate<mzpeaks::MZ> for ThermoPeak {
    fn get_index(&self) -> mzpeaks::IndexType {
        self.peak.get_index()
    }

    fn set_index(&mut self, index: mzpeaks::IndexType) {
        self.peak.set_index(index);
    }
}

impl From<CentroidPeak> for ThermoPeak {
    fn from(value: CentroidPeak) -> Self {
        Self {
            peak: value,
            ..Default::default()
        }
    }
}

impl mzdata::prelude::BuildArrayMapFrom for ThermoPeak {
    fn as_arrays(source: &[Self]) -> mzdata::spectrum::BinaryArrayMap {
        let mut arrays = BinaryArrayMap::new();

        let mut mz_array = DataArray::from_name_type_size(
            &ArrayType::MZArray,
            BinaryDataArrayType::Float64,
            source.len() * BinaryDataArrayType::Float64.size_of(),
        );
        mz_array.unit = Unit::MZ;

        let mut intensity_array = DataArray::from_name_type_size(
            &ArrayType::IntensityArray,
            BinaryDataArrayType::Float32,
            source.len() * BinaryDataArrayType::Float32.size_of(),
        );
        intensity_array.unit = Unit::DetectorCounts;

        let mut baseline_array = DataArray::from_name_type_size(
            &ArrayType::BaselineArray,
            BinaryDataArrayType::Float32,
            source.len() * BinaryDataArrayType::Float32.size_of(),
        );
        baseline_array.unit = Unit::Dimensionless;

        let mut snr_array = DataArray::from_name_type_size(
            &ArrayType::SignalToNoiseArray,
            BinaryDataArrayType::Float32,
            source.len() * BinaryDataArrayType::Float32.size_of(),
        );
        snr_array.unit = Unit::Dimensionless;

        mz_array.compression = BinaryCompressionType::Decoded;
        intensity_array.compression = BinaryCompressionType::Decoded;
        baseline_array.compression = BinaryCompressionType::Decoded;
        snr_array.compression = BinaryCompressionType::Decoded;

        for p in source.iter() {
            let mz: f64 = p.coordinate();
            let inten: f32 = p.intensity();

            mz_array.data.extend_from_slice(&mz.to_le_bytes());
            intensity_array.data.extend_from_slice(&inten.to_le_bytes());

            baseline_array
                .data
                .extend_from_slice(&p.baseline.unwrap_or_default().to_le_bytes());
            snr_array
                .data
                .extend_from_slice(&p.noise.unwrap_or_default().to_le_bytes());
        }

        arrays.add(mz_array);
        arrays.add(intensity_array);
        arrays.add(baseline_array);
        arrays.add(snr_array);
        arrays
    }
}

impl mzdata::prelude::BuildFromArrayMap for ThermoPeak {
    fn try_from_arrays(
        arrays: &BinaryArrayMap,
    ) -> Result<Vec<Self>, mzdata::spectrum::bindata::ArrayRetrievalError> {
        let mzs = arrays.mzs()?;
        let ints = arrays.intensities()?;
        let snrs = arrays
            .get(&ArrayType::SignalToNoiseArray)
            .and_then(|v| v.to_f32().ok());
        let baseline = arrays
            .get(&ArrayType::BaselineArray)
            .and_then(|v| v.to_f32().ok());

        let mut peaks = Vec::with_capacity(mzs.len());

        for (i, (mz, int)) in mzs.iter().copied().zip(ints.iter().copied()).enumerate() {
            let peak = CentroidPeak::new(mz, int, i as mzpeaks::IndexType);
            peaks.push(ThermoPeak {
                peak,
                baseline: baseline.as_ref().and_then(|v| v.get(i).copied()),
                noise: snrs.as_ref().and_then(|v| v.get(i).copied()),
            });
        }

        Ok(peaks)
    }
}

impl ToMzPeakDataSeries for ThermoPeak {
    fn to_fields() -> arrow::datatypes::Fields {
        vec![
            BufferContext::Spectrum.index_field(),
            mzpeak_prototyping::peak_series::MZ_ARRAY.to_field(),
            mzpeak_prototyping::peak_series::INTENSITY_ARRAY.to_field(),
            BufferName::new(
                BufferContext::Spectrum,
                ArrayType::BaselineArray,
                BinaryDataArrayType::Float32,
            )
            .to_field(),
            BufferName::new(
                BufferContext::Spectrum,
                ArrayType::SignalToNoiseArray,
                BinaryDataArrayType::Float32,
            )
            .to_field(),
        ]
        .into()
    }

    fn to_arrays(
        spectrum_index: u64,
        spectrum_time: Option<f32>,
        peaks: &[Self],
        overrides: &BufferOverrideTable,
    ) -> (arrow::datatypes::Fields, Vec<arrow::array::ArrayRef>) {
        let mut fields = Vec::new();
        let mut arrays = Vec::new();
        let n = peaks.len();

        fields.push(BufferContext::Spectrum.index_field());
        let index_array = Arc::new(UInt64Array::from_value(spectrum_index, n));
        arrays.push(index_array as ArrayRef);
        if let Some(spectrum_time) = spectrum_time {
            fields.push(BufferContext::Spectrum.time_field());
            arrays.push(Arc::new(Float32Array::from_value(
                spectrum_time,
                peaks.len(),
            )));
        }

        let mut mz_array = Vec::with_capacity(n);
        let mut intensity_array = Vec::with_capacity(n);
        let mut baseline_array = Vec::with_capacity(n);
        let mut snr_array = Vec::with_capacity(n);
        for p in peaks {
            mz_array.push(p.mz());
            intensity_array.push(p.intensity());
            baseline_array.push(p.baseline);
            snr_array.push(p.noise);
        }

        let buffer_name = mzpeak_prototyping::peak_series::MZ_ARRAY;
        let buffer_name = overrides.map(&buffer_name);
        match buffer_name.dtype {
            BinaryDataArrayType::Float64 => {
                arrays.push(Arc::new(Float64Array::from(mz_array)));
            }
            BinaryDataArrayType::Float32 => {
                arrays.push(Arc::new(Float32Array::from_iter_values(
                    mz_array.into_iter().map(|v| v as f32),
                )));
            }
            _ => {
                panic!("Unsupported {buffer_name:?}")
            }
        }
        fields.push(buffer_name.to_field());

        let buffer_name = mzpeak_prototyping::peak_series::INTENSITY_ARRAY;
        let buffer_name = overrides.map(&buffer_name);
        match buffer_name.dtype {
            BinaryDataArrayType::Float64 => {
                arrays.push(Arc::new(Float64Array::from_iter_values(
                    intensity_array.into_iter().map(|v| v as f64),
                )));
            }
            BinaryDataArrayType::Float32 => {
                arrays.push(Arc::new(Float32Array::from(intensity_array)));
            }
            BinaryDataArrayType::Int32 => {
                arrays.push(Arc::new(Int32Array::from_iter_values(
                    intensity_array.into_iter().map(|v| v as i32),
                )));
            }
            _ => {
                panic!("Unsupported {buffer_name:?}")
            }
        }
        fields.push(buffer_name.to_field());

        let buffer_name = BufferName::new(
            BufferContext::Spectrum,
            ArrayType::BaselineArray,
            BinaryDataArrayType::Float32,
        );
        let buffer_name = overrides.map(&buffer_name);
        match buffer_name.dtype {
            BinaryDataArrayType::Float64 => {
                arrays.push(Arc::new(Float64Array::from_iter(
                    baseline_array.into_iter().map(|v| v.map(|v| v as f64)),
                )));
            }
            BinaryDataArrayType::Float32 => {
                arrays.push(Arc::new(Float32Array::from(baseline_array)));
            }
            BinaryDataArrayType::Int32 => {
                arrays.push(Arc::new(Int32Array::from_iter(
                    baseline_array.into_iter().map(|v| v.map(|v| v as i32)),
                )));
            }
            _ => {
                panic!("Unsupported {buffer_name:?}")
            }
        }
        fields.push(buffer_name.to_field());

        let buffer_name = BufferName::new(
            BufferContext::Spectrum,
            ArrayType::SignalToNoiseArray,
            BinaryDataArrayType::Float32,
        );
        let buffer_name = overrides.map(&buffer_name);
        match buffer_name.dtype {
            BinaryDataArrayType::Float64 => {
                arrays.push(Arc::new(Float64Array::from_iter(
                    snr_array.into_iter().map(|v| v.map(|v| v as f64)),
                )));
            }
            BinaryDataArrayType::Float32 => {
                arrays.push(Arc::new(Float32Array::from(snr_array)));
            }
            BinaryDataArrayType::Int32 => {
                arrays.push(Arc::new(Int32Array::from_iter(
                    snr_array.into_iter().map(|v| v.map(|v| v as i32)),
                )));
            }
            _ => {
                panic!("Unsupported {buffer_name:?}")
            }
        }
        fields.push(buffer_name.to_field());

        (fields.into(), arrays)
    }
}

fn convert_file(
    input_path: &Path,
    output_path: &Path,
    args: &convert::ConvertArgs,
) -> io::Result<()> {
    let mut reader = MZReaderType::<_, ThermoPeak, DeconvolutedPeak>::open_path(&input_path)
        .inspect_err(|e| eprintln!("Failed to open data file: {e}"))?;

    let n = reader.len();
    let overrides = args.create_type_overrides();

    if let Some(c) = args.chunked_encoding.as_ref() {
        log::debug!("Using chunking method {c:?}");
    }

    let handle = fs::File::create(output_path)?;
    let mut writer = convert::configure_writer_builder(&args);

    writer = writer.add_spectrum_peak_type::<ThermoPeak>();

    if args.write_peaks_and_profiles {
        let mut point_builder = ArrayBuffersBuilder::default()
            .prefix("point")
            .with_context(BufferContext::Spectrum)
            .add_peak_type::<ThermoPeak>();
        for f in sample_array_types_from_file_reader::<ThermoPeak, DeconvolutedPeak>(
            &mut reader,
            &overrides,
            None,
        ) {
            point_builder = point_builder.add_field(f);
        }
        writer = writer.store_peaks_and_profiles_apart(Some(point_builder));
    }

    writer = sample_array_types_from_file_reader::<ThermoPeak, DeconvolutedPeak>(
        &mut reader,
        &overrides,
        args.chunked_encoding,
    )
    .into_iter()
    .fold(writer, |writer, f| writer.add_spectrum_field(f));

    for (from, to) in overrides.iter() {
        writer = writer.add_spectrum_array_override(from.clone(), to.clone());
        writer = writer.add_chromatogram_array_override(from.clone(), to.clone());
    }

    let mut writer = writer.build(handle, true);
    writer.copy_metadata_from(&reader);
    convert::add_processing_metadata(&mut writer);

    let (send, recv) = sync_channel(1);

    let mut reader = match reader {
        MZReaderType::ThermoRaw(reader) => reader,
        _ => {
            return Err(io::Error::other(format!(
                "Invalid format for convert_thermo: {}",
                reader.as_format()
            )));
        }
    };

    reader.set_load_extended_spectrum_data(true);

    let write_peaks_and_profiles = args.write_peaks_and_profiles;
    let read_handle = thread::spawn(move || {
        let result = panic::catch_unwind(AssertUnwindSafe(|| {
            for i in 0..reader.len() {
                let mut entry = reader.get_spectrum_by_index(i).unwrap();
                if entry.signal_continuity() == SignalContinuity::Profile {
                    if let Some(arrays) = entry.arrays {
                        let mut tmp = BinaryArrayMap::new();
                        for (k, arr) in arrays.into_iter() {
                            if k == ArrayType::MZArray || k == ArrayType::IntensityArray {
                                tmp.add(arr)
                            }
                        }
                        entry.arrays = Some(tmp);
                    }
                }
                if write_peaks_and_profiles {
                    if let Some(peak_data) = reader.get_data_arrays_for(entry.index(), true, true) {
                        entry.peaks = Some(ThermoPeak::from_arrays(&peak_data).into())
                    }
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
            let traces: Vec<_> = reader.status_log_names().collect();
            for trace in traces
                .into_iter()
                .flat_map(|v| reader.get_status_log_trace_by_name(&v))
            {
                if send.send((None, Some(trace))).is_err() {
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
                if let Some(spectrum) = spectrum {
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
