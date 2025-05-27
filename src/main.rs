use arrow::datatypes::FieldRef;
use itertools::Itertools;
use mzdata::{self, prelude::*};
use mzpeak_prototyping::*;
use parquet::{
    basic::{Encoding, ZstdLevel},
    file::properties::{EnabledStatistics, WriterProperties, WriterVersion},
};
use serde_arrow::schema::SchemaLike;
use std::{env, fs, io, path::PathBuf, sync::{Arc, mpsc::sync_channel}, thread};

fn main() -> io::Result<()> {
    env_logger::init();
    let filename = PathBuf::from(env::args().skip(1).next().unwrap());

    let start = std::time::Instant::now();

    let mut reader = mzdata::MZReader::open_path(&filename).inspect_err(|e| {
        eprintln!("Failed to open data file: {e}")
    })?;

    let fields =
        Vec::<FieldRef>::from_type::<MzPeaksEntry<MzPeaksMZIMPoint>>(Default::default()).unwrap();
    let schema = Arc::new(arrow::datatypes::Schema::new(fields.clone()));

    let (send, recv) = sync_channel(1);

    let read_handle = thread::spawn(move || {
        let mut precursor_index: u64 = 0;
        let entries_iter = reader
            .iter()
            .map(|s| {
                MzPeaksEntry::<MzPeaksMZIMPoint>::from_spectrum_with_precursors(
                    &s,
                    &mut precursor_index,
                )
            })
            .chunks(1000);

        for entry in entries_iter.into_iter() {
            let entry = entry.flatten().collect_vec();
            let batch = serde_arrow::to_record_batch(&fields, &entry).unwrap();
            send.send(batch).unwrap();
        }
    });

    let outname = filename.with_extension("mzpeak");
    let handle = fs::File::create(outname.file_name().unwrap())?;
    let props = WriterProperties::builder()
        .set_compression(parquet::basic::Compression::ZSTD(
            ZstdLevel::try_new(3).unwrap(),
        ))
        .set_column_encoding(
            "point.mz".into(),
            Encoding::BYTE_STREAM_SPLIT,
        )
        .set_column_encoding(
            "point.intensity".into(),
            Encoding::BYTE_STREAM_SPLIT,
        )
        .set_column_encoding(
            "point.im".into(),
            Encoding::BYTE_STREAM_SPLIT,
        )
        .set_column_encoding("point.spectrum_index".into(), Encoding::RLE)
        .set_column_bloom_filter_enabled("spectrum.id".into(), true)
        .set_writer_version(WriterVersion::PARQUET_2_0)
        .set_statistics_enabled(EnabledStatistics::Page)
        .build();
    let mut writer = parquet::arrow::ArrowWriter::try_new(handle, schema, Some(props))?;

    let write_handle = thread::spawn(move || {
        for (i, batch) in recv.into_iter().enumerate() {
            log::info!("Writing batch {i}");
            writer.write(&batch).unwrap();
        }
        writer.finish().unwrap();
    });

    read_handle.join().unwrap();
    write_handle.join().unwrap();

    let end = std::time::Instant::now();
    eprintln!("{:0.2} seconds elapsed", (end - start).as_secs_f64());
    Ok(())
}
