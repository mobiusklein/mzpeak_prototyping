
use std::{fs, io, sync::Arc};
use arrow::datatypes::FieldRef;
use itertools::Itertools;
use parquet::{basic::ZstdLevel, file::properties::{EnabledStatistics, WriterProperties}};
use mzdata::{self, prelude::*};
use mzpeak_prototyping::*;
use serde_arrow::schema::SchemaLike;


fn main() -> io::Result<()> {
    let mut reader = mzdata::MZReader::open_path("../mzdata/converted/E_20221108_EvoSepOne_3rdGenAurora15cm_CC_40SPD_whisper_scLF1108_M10_S1-A1_1_3103.zlib.mzML")?;
    let entries_iter = reader
        .iter()
        .map(|s| MzPeaksEntry::<MzPeaksMZIMPoint>::from_spectrum(&s)).chunks(1000);

    let fields = Vec::<FieldRef>::from_type::<MzPeaksEntry<MzPeaksMZIMPoint>>(Default::default()).unwrap();
    let schema = Arc::new(arrow::datatypes::Schema::new(fields.clone()));

    let handle = fs::File::create("E_20221108_EvoSepOne_3rdGenAurora15cm_CC_40SPD_whisper_scLF1108_M10_S1-A1_1_3103.mzpeak")?;
    let props = WriterProperties::builder()
        .set_compression(parquet::basic::Compression::ZSTD(ZstdLevel::try_new(3).unwrap()))
        .set_column_encoding("point.mz".into(), parquet::basic::Encoding::BYTE_STREAM_SPLIT)
        .set_column_encoding("point.intensity".into(), parquet::basic::Encoding::BYTE_STREAM_SPLIT)
        .set_column_encoding("point.im".into(), parquet::basic::Encoding::BYTE_STREAM_SPLIT)
        .set_column_encoding("point.spectrum_index".into(), parquet::basic::Encoding::RLE)
        .set_writer_version(parquet::file::properties::WriterVersion::PARQUET_2_0)
        .set_statistics_enabled(EnabledStatistics::Page)
        .build();
    let mut writer = parquet::arrow::ArrowWriter::try_new(handle, schema, Some(props))?;

    for (i, entries) in entries_iter.into_iter().enumerate() {
        eprintln!("Writing batch {i}");
        let chunk = entries.flatten().collect_vec();
        let batch = serde_arrow::to_record_batch(&fields, &chunk).unwrap();
        writer.write(&batch)?;
    }

    writer.finish()?;
    Ok(())
}