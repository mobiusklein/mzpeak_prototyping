use mzdata::{self, io::MZReaderType, prelude::*};
use mzpeak_prototyping::{peak_series::ToMzPeakDataSeries, *};
use mzpeaks::{CentroidPeak, DeconvolutedPeak};
use std::{collections::{HashMap, HashSet}, env, fs, io, path::PathBuf, sync::mpsc::sync_channel, thread};

fn sample_array_types<
    C: CentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<CentroidPeak>,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<DeconvolutedPeak>,
>(
    reader: &mut MZReaderType<fs::File, C, D>,
    overrides: &HashMap<BufferName, BufferName>,
) -> HashSet<std::sync::Arc<arrow::datatypes::Field>> {
    let n = reader.len();
    let mut arrays = HashSet::new();

    arrays.extend(C::to_fields().iter().cloned());
    arrays.extend(D::to_fields().iter().cloned());

    if n == 0 {
        return arrays;
    }

    let field_it = [0, 100.min(n - 1), n / 2]
        .into_iter()
        .flat_map(|i| {
            reader.get_spectrum_by_index(i).and_then(|s| {
                s.raw_arrays().and_then(|map| {
                    peak_series::array_map_to_schema_arrays(
                        BufferContext::Spectrum,
                        map,
                        map.mzs().map(|a| a.len()).unwrap_or_default(),
                        0,
                        "spectrum_index",
                        overrides
                    )
                    .ok()
                })
            })
        })
        .map(|(fields, _arrs)| {
            let fields: Vec<_> = fields.iter().cloned().collect();
            fields
        })
        .flatten();

    arrays.extend(field_it);
    arrays
}

fn main() -> io::Result<()> {
    env_logger::init();
    let filename = PathBuf::from(env::args().skip(1).next().unwrap());

    let start = std::time::Instant::now();

    let mut reader = MZReaderType::<_, CentroidPeak, DeconvolutedPeak>::open_path(&filename)
        .inspect_err(|e| eprintln!("Failed to open data file: {e}"))?;

    let outname = filename.with_extension("mzpeak");

    let overrides = HashMap::new();

    let handle = fs::File::create(outname.file_name().unwrap())?;
    // let data_handle = fs::File::create(outname.with_extension("data.mzpeak"))?;

    let mut writer = MzPeakWriterType::<fs::File>::builder()
        .add_spectrum_peak_type::<CentroidPeak>()
        .add_spectrum_peak_type::<DeconvolutedPeak>()
        .add_default_chromatogram_fields()
        .buffer_size(5000);

    writer = sample_array_types::<CentroidPeak, DeconvolutedPeak>(&mut reader, &overrides)
        .into_iter()
        .fold(writer, |writer, f| writer.add_spectrum_field(f));

    let mut writer = writer.build(handle, false);
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
            batch.peaks = None;
            writer.write_spectrum(&batch).unwrap();
        }
        writer.finish().unwrap();
    });

    read_handle.join().unwrap();
    write_handle.join().unwrap();

    let end = std::time::Instant::now();
    eprintln!("{:0.2} seconds elapsed", (end - start).as_secs_f64());
    Ok(())
}
