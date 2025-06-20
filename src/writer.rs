use std::{
    collections::{HashMap, HashSet},
    io::{self, prelude::*},
    sync::Arc,
};

use arrow::{
    array::{ArrayRef, RecordBatch, StructArray, new_null_array},
    datatypes::{DataType, Field, FieldRef, Fields, Schema, SchemaRef},
};
use parquet::{
    arrow::{ArrowSchemaConverter, ArrowWriter, arrow_writer::ArrowWriterOptions},
    basic::{Encoding, ZstdLevel},
    file::properties::{EnabledStatistics, WriterProperties, WriterVersion},
    format::{KeyValue, SortingColumn},
};

use mzdata::{
    params::Unit,
    prelude::*,
    spectrum::{ArrayType, BinaryDataArrayType},
};
use serde_arrow::schema::{SchemaLike, TracingOptions};

use crate::{
    archive::ZipArchiveWriter,
    entry::Entry,
    param::{DataProcessing, FileDescription, InstrumentConfiguration, Software},
    peak_series::{
        ArrayIndex, ArrayIndexEntry, BufferContext, BufferName, ToMzPeakDataSeries,
        array_map_to_schema_arrays,
    },
};

#[derive(Debug)]
pub struct ArrayBuffers {
    pub peak_array_fields: Fields,
    pub schema: SchemaRef,
    pub prefix: String,
    pub array_chunks: HashMap<String, Vec<ArrayRef>>,
}

impl ArrayBuffers {
    pub fn dtype(&self) -> DataType {
        DataType::Struct(self.peak_array_fields.clone())
    }

    pub fn len(&self) -> usize {
        self.array_chunks.values().map(|v| v.len()).sum()
    }

    pub fn num_chunks(&self) -> usize {
        self.array_chunks
            .values()
            .map(|v| v.len())
            .next()
            .unwrap_or_default()
    }

    pub fn promote_batch(&self, batch: RecordBatch, schema: SchemaRef) -> RecordBatch {
        let num_rows = batch.num_rows();
        let mut batch = Some(batch);
        let mut arrays = Vec::with_capacity(schema.fields().len());
        for f in schema.fields().iter() {
            if f.name() == self.prefix.as_str() {
                if let Some(batch) = batch.take() {
                    let x = Arc::new(StructArray::from(batch));
                    arrays.push(x as ArrayRef);
                }
            } else {
                arrays.push(new_null_array(f.data_type(), num_rows));
            }
        }
        RecordBatch::try_new(schema, arrays).unwrap()
    }

    pub fn add<T: ToMzPeakDataSeries>(&mut self, spectrum_index: u64, peaks: &[T]) {
        let (fields, chunks) = T::to_arrays(spectrum_index, peaks);
        let mut visited = HashSet::new();
        for (f, arr) in fields.iter().zip(chunks.into_iter()) {
            self.array_chunks
                .get_mut(f.name())
                .unwrap_or_else(|| panic!("Unexpected field {f:?}"))
                .push(arr);
            visited.insert(f.name());
        }
        for (f, chunk) in self.array_chunks.iter_mut() {
            if !visited.contains(&f) {
                if let Some(t) = chunk.first().map(|a| a.data_type()).or_else(|| {
                    self.peak_array_fields
                        .iter()
                        .find(|a| a.name() == f)
                        .map(|a| a.data_type())
                }) {
                    chunk.push(new_null_array(t, peaks.len()));
                }
            }
        }
    }

    pub fn add_arrays(&mut self, fields: Fields, arrays: Vec<ArrayRef>, size: usize) {
        let mut visited = HashSet::new();
        for (f, arr) in fields.iter().zip(arrays) {
            self.array_chunks
                .get_mut(f.name())
                .unwrap_or_else(|| panic!("Unexpected field {f:?}"))
                .push(arr);
            visited.insert(f.name());
        }
        for (f, chunk) in self.array_chunks.iter_mut() {
            if !visited.contains(&f) {
                if let Some(t) = chunk.first().map(|a| a.data_type()).or_else(|| {
                    self.peak_array_fields
                        .iter()
                        .find(|a| a.name() == f)
                        .map(|a| a.data_type())
                }) {
                    chunk.push(new_null_array(t, size));
                }
            }
        }
    }

    pub fn drain(&mut self) -> impl Iterator<Item = RecordBatch> {
        let n_chunks = self.num_chunks();
        let mut chunks: Vec<Vec<ArrayRef>> = Vec::with_capacity(n_chunks);
        chunks.resize(n_chunks, Vec::new());
        for f in self.peak_array_fields.iter() {
            let series = self.array_chunks.get_mut(f.name()).unwrap();
            for (chunk, container) in series.drain(..).zip(chunks.iter_mut()) {
                container.push(chunk);
            }
        }

        let schema = SchemaRef::new(Schema::new(self.peak_array_fields.clone()));
        chunks
            .into_iter()
            .map(move |arrs| {
                RecordBatch::try_new(schema.clone(), arrs.clone()).unwrap_or_else(|e| {
                    let fields: Vec<_> = arrs.iter().map(|f| f.data_type()).collect();
                    panic!("Failed to convert peak buffers to record batch: {e}\n{fields:#?}\n{schema:#?}")
                })
            })
            .map(|batch| self.promote_batch(batch, self.schema.clone()))
    }
}

#[derive(Debug)]
pub struct ArrayBuffersBuilder {
    prefix: String,
    peak_array_fields: Vec<FieldRef>,
}

impl Default for ArrayBuffersBuilder {
    fn default() -> Self {
        Self {
            prefix: "point".to_string(),
            peak_array_fields: Default::default(),
        }
    }
}

impl ArrayBuffersBuilder {
    pub fn prefix(mut self, value: impl ToString) -> Self {
        self.prefix = value.to_string();
        self
    }

    pub fn dtype(&self) -> DataType {
        DataType::Struct(self.peak_array_fields.clone().into())
    }

    pub fn add_field(mut self, field: FieldRef) -> Self {
        if self
            .peak_array_fields
            .iter()
            .find(|f| f.name() == field.name())
            .is_none()
        {
            self.peak_array_fields.push(field);
        }
        self
    }

    pub fn add_peak_type<T: ToMzPeakDataSeries>(mut self) -> Self {
        for f in T::to_fields().iter().cloned() {
            self = self.add_field(f);
        }
        self
    }

    pub fn build(&self, schema: SchemaRef) -> ArrayBuffers {
        let mut fields: Vec<FieldRef> = schema.fields().iter().cloned().collect();
        fields.push(Field::new(self.prefix.clone(), self.dtype(), true).into());
        let buffers = self
            .peak_array_fields
            .iter()
            .map(|f| (f.name().clone(), Vec::new()))
            .collect();
        ArrayBuffers {
            peak_array_fields: self.peak_array_fields.clone().into(),
            schema: Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone())),
            prefix: self.prefix.clone(),
            array_chunks: buffers,
        }
    }
}

#[derive(Debug)]
pub struct MzPeakWriterBuilder {
    spectrum_arrays: ArrayBuffersBuilder,
    chromatogram_arrays: ArrayBuffersBuilder,
    buffer_size: usize,
}

impl Default for MzPeakWriterBuilder {
    fn default() -> Self {
        Self {
            spectrum_arrays: ArrayBuffersBuilder::default().prefix("point"),
            chromatogram_arrays: ArrayBuffersBuilder::default().prefix("chromatogram_point"),
            buffer_size: 5_000,
        }
    }
}

impl MzPeakWriterBuilder {
    pub fn add_spectrum_field(mut self, f: FieldRef) -> Self {
        self.spectrum_arrays = self.spectrum_arrays.add_field(f);
        self
    }

    pub fn add_spectrum_peak_type<T: ToMzPeakDataSeries>(mut self) -> Self {
        self.spectrum_arrays = self.spectrum_arrays.add_peak_type::<T>();
        self
    }

    pub fn spectrum_data_prefix(mut self, value: impl ToString) -> Self {
        self.spectrum_arrays = self.spectrum_arrays.prefix(value);
        self
    }

    pub fn add_chromatogram_field(mut self, f: FieldRef) -> Self {
        self.chromatogram_arrays = self.chromatogram_arrays.add_field(f);
        self
    }

    pub fn chromatogram_data_prefix(mut self, value: impl ToString) -> Self {
        self.chromatogram_arrays = self.chromatogram_arrays.prefix(value);
        self
    }

    pub fn buffer_size(mut self, value: usize) -> Self {
        self.buffer_size = value;
        self
    }

    pub fn build_split<W: Write + Send>(
        self,
        data_writer: W,
        metadata_writer: W,
    ) -> MzPeakSplitWriter<W> {
        MzPeakSplitWriter::new(
            data_writer,
            metadata_writer,
            self.spectrum_arrays,
            self.chromatogram_arrays,
            self.buffer_size,
        )
    }

    pub fn build<W: Write + Send + Seek>(self, writer: W) -> MzPeakWriter<W> {
        MzPeakWriter::new(
            writer,
            self.spectrum_arrays,
            self.chromatogram_arrays,
            self.buffer_size,
        )
    }

    pub fn add_default_chromatogram_fields(mut self) -> Self {
        let time = BufferName::new(
            BufferContext::Chromatogram,
            ArrayType::TimeArray,
            BinaryDataArrayType::Float64,
        )
        .with_unit(Unit::Minute)
        .to_field();
        let intensity = BufferName::new(
            BufferContext::Chromatogram,
            ArrayType::IntensityArray,
            BinaryDataArrayType::Float32,
        )
        .with_unit(Unit::DetectorCounts)
        .to_field();

        self = self
            .add_chromatogram_field(time)
            .add_chromatogram_field(intensity)
            .add_chromatogram_field(Arc::new(Field::new(
                "chromatogram_index",
                DataType::UInt64,
                false,
            )));
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct MzMetadata {
    pub file_description: FileDescription,
    pub data_processing_list: Vec<DataProcessing>,
    pub software_list: Vec<Software>,
    pub instrument_configurations: Vec<InstrumentConfiguration>,
}

impl MzMetadata {
    pub fn new(
        file_description: FileDescription,
        data_processing_list: Vec<DataProcessing>,
        software_list: Vec<Software>,
        instrument_configurations: Vec<InstrumentConfiguration>,
    ) -> Self {
        Self {
            file_description,
            data_processing_list,
            software_list,
            instrument_configurations,
        }
    }

    pub fn add_file_description(&mut self, value: &mzdata::meta::FileDescription) {
        self.file_description = value.into();
    }

    pub fn add_instrument_configuration(&mut self, value: &mzdata::meta::InstrumentConfiguration) {
        self.instrument_configurations.push(value.into());
    }

    pub fn add_software(&mut self, value: &mzdata::meta::Software) {
        self.software_list.push(value.into());
    }

    pub fn add_data_processing(&mut self, value: &mzdata::meta::DataProcessing) {
        self.data_processing_list.push(value.into());
    }
}

macro_rules! implement_mz_metadata {
    () => {
        pub fn add_file_description(&mut self, value: &mzdata::meta::FileDescription) {
            self.mz_metadata.add_file_description(value);
        }

        pub fn add_instrument_configuration(
            &mut self,
            value: &mzdata::meta::InstrumentConfiguration,
        ) {
            self.mz_metadata.add_instrument_configuration(value);
        }

        pub fn add_software(&mut self, value: &mzdata::meta::Software) {
            self.mz_metadata.add_software(value);
        }

        pub fn add_data_processing(&mut self, value: &mzdata::meta::DataProcessing) {
            self.mz_metadata.add_data_processing(value);
        }

        pub(crate) fn append_metadata(&mut self) {
            self.append_key_value_metadata(
                "file_description",
                Some(serde_json::to_string_pretty(&self.mz_metadata.file_description).unwrap()),
            );
            self.append_key_value_metadata(
                "instrument_configuration_list",
                Some(
                    serde_json::to_string_pretty(&self.mz_metadata.instrument_configurations)
                        .unwrap(),
                ),
            );
            self.append_key_value_metadata(
                "data_processing_method_list",
                Some(serde_json::to_string_pretty(&self.mz_metadata.data_processing_list).unwrap()),
            );
            self.append_key_value_metadata(
                "software_list",
                Some(serde_json::to_string_pretty(&self.mz_metadata.software_list).unwrap()),
            );
        }
    };
}

pub trait AbstractMzPeakWriter {
    fn append_key_value_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<Option<String>>,
    );

    fn spectrum_entry_buffer_mut(&mut self) -> &mut Vec<Entry>;
    fn spectrum_data_buffer_mut(&mut self) -> &mut ArrayBuffers;

    fn check_data_buffer(&mut self) -> io::Result<()>;

    fn spectrum_counter(&self) -> u64;
    fn spectrum_counter_mut(&mut self) -> &mut u64;

    fn spectrum_precursor_counter(&self) -> u64;
    fn spectrum_precursor_counter_mut(&mut self) -> &mut u64;

    fn spectrum_to_entries<
        C: ToMzPeakDataSeries + CentroidLike,
        D: ToMzPeakDataSeries + DeconvolutedCentroidLike,
    >(
        &mut self,
        spectrum: &impl SpectrumLike<C, D>,
    ) -> Vec<Entry> {
        Entry::from_spectrum(
            spectrum,
            Some(self.spectrum_counter()),
            Some(self.spectrum_precursor_counter_mut()),
        )
    }

    fn write_spectrum<
        C: ToMzPeakDataSeries + CentroidLike,
        D: ToMzPeakDataSeries + DeconvolutedCentroidLike,
    >(
        &mut self,
        spectrum: &impl SpectrumLike<C, D>,
    ) -> io::Result<()> {
        let entries = self.spectrum_to_entries(spectrum);
        self.spectrum_entry_buffer_mut().extend(entries);
        self.write_spectrum_peaks(spectrum)?;
        *self.spectrum_counter_mut() += 1;
        self.check_data_buffer()?;
        Ok(())
    }

    fn write_spectrum_peaks<
        C: ToMzPeakDataSeries + CentroidLike,
        D: ToMzPeakDataSeries + DeconvolutedCentroidLike,
    >(
        &mut self,
        spectrum: &impl SpectrumLike<C, D>,
    ) -> io::Result<()> {
        let spectrum_count = self.spectrum_counter();
        match spectrum.peaks() {
            mzdata::spectrum::RefPeakDataLevel::Missing => {}
            mzdata::spectrum::RefPeakDataLevel::RawData(binary_array_map) => {
                let n_points = spectrum.peaks().len();
                let (fields, data) = array_map_to_schema_arrays(
                    crate::BufferContext::Spectrum,
                    binary_array_map,
                    n_points,
                    spectrum_count,
                    "spectrum_index",
                )?;
                self.spectrum_data_buffer_mut()
                    .add_arrays(fields, data, n_points);
            }
            mzdata::spectrum::RefPeakDataLevel::Centroid(peaks) => {
                self.spectrum_data_buffer_mut()
                    .add(spectrum_count, peaks.as_slice());
            }
            mzdata::spectrum::RefPeakDataLevel::Deconvoluted(peaks) => {
                self.spectrum_data_buffer_mut()
                    .add(spectrum_count, peaks.as_slice());
            }
        }

        Ok(())
    }
}

pub struct MzPeakWriter<W: Write + Send + Seek> {
    archive_writer: Option<ArrowWriter<ZipArchiveWriter<W>>>,
    spectrum_buffers: ArrayBuffers,
    #[allow(unused)]
    chromatogram_buffers: ArrayBuffers,
    metadata_buffer: Vec<Entry>,

    metadata_fields: SchemaRef,

    spectrum_counter: u64,
    spectrum_precursor_counter: u64,
    spectrum_data_point_counter: u64,
    buffer_size: usize,

    mz_metadata: MzMetadata,
}

impl<W: Write + Send + Seek> AbstractMzPeakWriter for MzPeakWriter<W> {
    fn append_key_value_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<Option<String>>,
    ) {
        self.archive_writer
            .as_mut()
            .unwrap()
            .append_key_value_metadata(KeyValue::new(key.into(), value));
    }

    fn spectrum_entry_buffer_mut(&mut self) -> &mut Vec<Entry> {
        &mut self.metadata_buffer
    }

    fn spectrum_data_buffer_mut(&mut self) -> &mut ArrayBuffers {
        &mut self.spectrum_buffers
    }

    fn check_data_buffer(&mut self) -> io::Result<()> {
        if self.spectrum_counter % (self.buffer_size as u64) == 0 {
            self.flush_data_arrays()?;
        }
        Ok(())
    }

    fn spectrum_counter(&self) -> u64 {
        self.spectrum_counter
    }

    fn spectrum_counter_mut(&mut self) -> &mut u64 {
        &mut self.spectrum_counter
    }

    fn spectrum_precursor_counter(&self) -> u64 {
        self.spectrum_precursor_counter
    }

    fn spectrum_precursor_counter_mut(&mut self) -> &mut u64 {
        &mut self.spectrum_precursor_counter
    }
}

impl<W: Write + Send + Seek> MzPeakWriter<W> {
    pub fn builder() -> MzPeakWriterBuilder {
        MzPeakWriterBuilder::default()
    }

    fn spectrum_data_writer_props(
        data_buffer: &ArrayBuffers,
        index_path: String,
    ) -> WriterProperties {
        let parquet_schema = Arc::new(
            ArrowSchemaConverter::new()
                .convert(&data_buffer.schema)
                .unwrap(),
        );

        let mut sorted = Vec::new();
        for (i, c) in parquet_schema.columns().iter().enumerate() {
            match c.path().string().as_ref() {
                x if x == index_path => {
                    sorted.push(SortingColumn::new(i as i32, false, false));
                }
                _ => {}
            }
        }

        let mut data_props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(
                ZstdLevel::try_new(3).unwrap(),
            ))
            .set_dictionary_enabled(true)
            .set_sorting_columns(Some(sorted))
            .set_column_encoding(index_path.into(), Encoding::RLE)
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_statistics_enabled(EnabledStatistics::Page);

        for f in parquet_schema.columns().iter().skip(1) {
            if f.name().contains("_mz_") {
                data_props =
                    data_props.set_column_encoding(f.path().clone(), Encoding::BYTE_STREAM_SPLIT);
            }
        }

        let data_props = data_props.build();
        data_props
    }

    fn metadata_writer_props(metadata_fields: &SchemaRef) -> WriterProperties {
        let parquet_schema = Arc::new(
            ArrowSchemaConverter::new()
                .convert(&metadata_fields)
                .unwrap(),
        );

        let mut sorted = Vec::new();
        for (i, c) in parquet_schema.columns().iter().enumerate() {
            match c.path().string().as_ref() {
                "spectrum.index" => {
                    sorted.push(SortingColumn::new(i as i32, false, false));
                }
                _ => {}
            }
        }
        let metadata_props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(
                ZstdLevel::try_new(3).unwrap(),
            ))
            .set_dictionary_enabled(true)
            .set_sorting_columns(Some(sorted))
            .set_column_bloom_filter_enabled("spectrum.id".into(), true)
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_statistics_enabled(EnabledStatistics::Page)
            .build();

        metadata_props
    }

    pub fn new(
        writer: W,
        spectrum_buffers: ArrayBuffersBuilder,
        chromatogram_buffers: ArrayBuffersBuilder,
        buffer_size: usize,
    ) -> Self {
        let fields: Vec<FieldRef> = SchemaLike::from_type::<Entry>(TracingOptions::new()).unwrap();
        let metadata_fields: SchemaRef = Arc::new(Schema::new(fields));
        let spectrum_buffers = spectrum_buffers.build(Arc::new(Schema::empty()));
        let chromatogram_buffers = chromatogram_buffers.build(Arc::new(Schema::empty()));

        let mut writer = ZipArchiveWriter::new(writer);
        writer.start_spectrum_data().unwrap();

        let data_props = Self::spectrum_data_writer_props(
            &spectrum_buffers,
            format!("{}.spectrum_index", spectrum_buffers.prefix),
        );

        let mut this = Self {
            archive_writer: Some(
                ArrowWriter::try_new_with_options(
                    writer,
                    spectrum_buffers.schema.clone(),
                    ArrowWriterOptions::new().with_properties(data_props),
                )
                .unwrap(),
            ),
            metadata_fields,
            metadata_buffer: Vec::new(),
            spectrum_buffers,
            chromatogram_buffers,
            spectrum_counter: 0,
            spectrum_precursor_counter: 0,
            spectrum_data_point_counter: 0,
            buffer_size: buffer_size,
            mz_metadata: Default::default(),
        };
        this.add_array_metadata();
        this
    }

    implement_mz_metadata!();

    fn add_array_metadata(&mut self) {
        let mut spectrum_array_index =
            ArrayIndex::new(self.spectrum_buffers.prefix.clone(), HashMap::new());
        if let Ok(sub) = self
            .spectrum_buffers
            .schema
            .field_with_name(&self.spectrum_buffers.prefix)
            .cloned()
        {
            if let DataType::Struct(fields) = sub.data_type() {
                for f in fields.iter() {
                    if f.name() == "spectrum_index" {
                        continue;
                    }
                    let buffer_name =
                        BufferName::from_field(BufferContext::Spectrum, f.clone()).unwrap();
                    let aie = ArrayIndexEntry::from_buffer_name(
                        self.spectrum_buffers.prefix.clone(),
                        buffer_name,
                    );
                    spectrum_array_index.insert(aie.array_type.clone(), aie);
                }
            }
        }
        self.append_key_value_metadata(
            "spectrum_array_index".to_string(),
            spectrum_array_index.to_json(),
        );

        // let mut chromatogram_array_index =
        //     ArrayIndex::new(self.chromatogram_buffers.prefix.clone(), HashMap::new());
        // if let Ok(sub) = self
        //     .chromatogram_buffers
        //     .schema
        //     .field_with_name(&self.chromatogram_buffers.prefix)
        //     .cloned()
        // {
        //     if let DataType::Struct(fields) = sub.data_type() {
        //         for f in fields.iter() {
        //             if f.name() == "chromatogram_index" {
        //                 continue;
        //             }
        //             let buffer_name =
        //                 BufferName::from_field(BufferContext::Chromatogram, f.clone()).unwrap();
        //             let aie = ArrayIndexEntry::from_buffer_name(
        //                 self.chromatogram_buffers.prefix.clone(),
        //                 buffer_name,
        //             );
        //             chromatogram_array_index.insert(aie.array_type.clone(), aie);
        //         }
        //     }
        // }
        // self.append_key_value_metadata(
        //     "chromatogram_array_index".to_string(),
        //     chromatogram_array_index.to_json(),
        // );
    }

    pub fn write_spectrum<
        C: ToMzPeakDataSeries + CentroidLike,
        D: ToMzPeakDataSeries + DeconvolutedCentroidLike,
    >(
        &mut self,
        spectrum: &impl SpectrumLike<C, D>,
    ) -> io::Result<()> {
        AbstractMzPeakWriter::write_spectrum(self, spectrum)
    }

    fn flush_data_arrays(&mut self) -> io::Result<()> {
        for batch in self.spectrum_buffers.drain() {
            self.spectrum_data_point_counter += batch.num_rows() as u64;
            self.archive_writer.as_mut().unwrap().write(&batch)?;
        }
        Ok(())
    }

    fn flush_metadata_records(&mut self) -> io::Result<()> {
        let batch =
            serde_arrow::to_record_batch(&self.metadata_fields.fields(), &self.metadata_buffer)
                .unwrap();
        self.archive_writer.as_mut().unwrap().write(&batch)?;
        self.metadata_buffer.clear();
        Ok(())
    }

    pub fn finish(&mut self) -> Result<(), parquet::errors::ParquetError> {
        if self.archive_writer.is_some() {
            self.flush_data_arrays()?;
            self.append_key_value_metadata(
                "spectrum_count",
                Some(self.spectrum_counter.to_string()),
            );
            self.append_key_value_metadata(
                "spectrum_data_point_count",
                Some(self.spectrum_data_point_counter.to_string()),
            );

            let mut writer = self.archive_writer.take().unwrap().into_inner()?;

            writer.start_spectrum_metadata().unwrap();
            self.archive_writer = Some(ArrowWriter::try_new_with_options(
                writer,
                self.metadata_fields.clone(),
                ArrowWriterOptions::new()
                    .with_properties(Self::metadata_writer_props(&self.metadata_fields)),
            )?);
            self.flush_metadata_records()?;
            self.append_metadata();
            self.append_key_value_metadata(
                "spectrum_count",
                Some(self.spectrum_counter.to_string()),
            );
            self.append_key_value_metadata(
                "spectrum_data_point_count",
                Some(self.spectrum_data_point_counter.to_string()),
            );

            let writer = self.archive_writer.take().unwrap().into_inner()?;
            writer.finish().unwrap();
            Ok(())
        } else {
            Err(parquet::errors::ParquetError::EOF(
                "Already closed file".into(),
            ))
        }
    }
}

impl<W: Write + Send + Seek> Drop for MzPeakWriter<W> {
    fn drop(&mut self) {
        if let Err(_) = self.finish() {}
    }
}

pub struct MzPeakSplitWriter<W: Write + Send> {
    data_writer: ArrowWriter<W>,
    metadata_writer: ArrowWriter<W>,

    spectrum_buffers: ArrayBuffers,
    #[allow(unused)]
    chromatogram_buffers: ArrayBuffers,
    metadata_buffer: Vec<Entry>,

    metadata_fields: SchemaRef,

    spectrum_counter: u64,
    spectrum_precursor_counter: u64,
    spectrum_data_point_counter: u64,
    buffer_size: usize,

    mz_metadata: MzMetadata,
}

impl<W: Write + Send> Drop for MzPeakSplitWriter<W> {
    fn drop(&mut self) {
        if let Err(_) = self.finish() {}
    }
}

impl<W: Write + Send> AbstractMzPeakWriter for MzPeakSplitWriter<W> {
    fn append_key_value_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<Option<String>>,
    ) {
        self.append_key_value_metadata(key, value);
    }

    fn spectrum_counter(&self) -> u64 {
        self.spectrum_counter
    }

    fn spectrum_counter_mut(&mut self) -> &mut u64 {
        &mut self.spectrum_counter
    }

    fn spectrum_data_buffer_mut(&mut self) -> &mut ArrayBuffers {
        &mut self.spectrum_buffers
    }

    fn spectrum_entry_buffer_mut(&mut self) -> &mut Vec<Entry> {
        &mut self.metadata_buffer
    }

    fn check_data_buffer(&mut self) -> io::Result<()> {
        if self.spectrum_counter % (self.buffer_size as u64) == 0 {
            self.flush_data_arrays()?;
        }
        Ok(())
    }

    fn spectrum_precursor_counter(&self) -> u64 {
        self.spectrum_precursor_counter
    }

    fn spectrum_precursor_counter_mut(&mut self) -> &mut u64 {
        &mut self.spectrum_precursor_counter
    }
}

impl<W: Write + Send> MzPeakSplitWriter<W> {
    fn data_writer_props(data_buffer: &ArrayBuffers) -> WriterProperties {
        let spectrum_point_prefix = format!("{}.spectrum_index", data_buffer.prefix);
        let parquet_schema = Arc::new(
            ArrowSchemaConverter::new()
                .convert(&data_buffer.schema)
                .unwrap(),
        );
        let mut sorted = Vec::new();
        for (i, c) in parquet_schema.columns().iter().enumerate() {
            match c.path().string().as_ref() {
                x if x == spectrum_point_prefix => {
                    sorted.push(SortingColumn::new(i as i32, false, false));
                }
                _ => {}
            }
        }

        let data_props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(
                ZstdLevel::try_new(3).unwrap(),
            ))
            .set_dictionary_enabled(true)
            .set_sorting_columns(Some(sorted))
            .set_column_encoding(spectrum_point_prefix.into(), Encoding::RLE)
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_statistics_enabled(EnabledStatistics::Page)
            .build();
        data_props
    }

    fn metadata_writer_props(metadata_fields: &SchemaRef) -> WriterProperties {
        let parquet_schema = Arc::new(
            ArrowSchemaConverter::new()
                .convert(&metadata_fields)
                .unwrap(),
        );

        let mut sorted = Vec::new();
        for (i, c) in parquet_schema.columns().iter().enumerate() {
            match c.path().string().as_ref() {
                "spectrum.index" => {
                    sorted.push(SortingColumn::new(i as i32, false, false));
                }
                _ => {}
            }
        }
        let metadata_props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(
                ZstdLevel::try_new(3).unwrap(),
            ))
            .set_dictionary_enabled(true)
            .set_sorting_columns(Some(sorted))
            .set_column_bloom_filter_enabled("spectrum.id".into(), true)
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_statistics_enabled(EnabledStatistics::Page)
            .build();

        metadata_props
    }

    pub fn new(
        data_writer: W,
        metadata_writer: W,
        spectrum_buffers: ArrayBuffersBuilder,
        chromatogram_buffers: ArrayBuffersBuilder,
        buffer_size: usize,
    ) -> Self {
        let fields: Vec<FieldRef> = SchemaLike::from_type::<Entry>(TracingOptions::new()).unwrap();
        let metadata_fields: SchemaRef = Arc::new(Schema::new(fields));
        let spectrum_buffers = spectrum_buffers.build(Arc::new(Schema::empty()));
        let chromatogram_buffers = chromatogram_buffers.build(Arc::new(Schema::empty()));

        let data_props = Self::data_writer_props(&spectrum_buffers);
        let metadata_props = Self::metadata_writer_props(&metadata_fields);

        let mut this = Self {
            metadata_fields: metadata_fields.clone(),
            data_writer: ArrowWriter::try_new_with_options(
                data_writer,
                spectrum_buffers.schema.clone(),
                ArrowWriterOptions::new().with_properties(data_props),
            )
            .unwrap(),
            metadata_writer: ArrowWriter::try_new_with_options(
                metadata_writer,
                metadata_fields.clone(),
                ArrowWriterOptions::new().with_properties(metadata_props),
            )
            .unwrap(),
            metadata_buffer: Vec::new(),
            spectrum_buffers,
            chromatogram_buffers,
            spectrum_counter: 0,
            spectrum_precursor_counter: 0,
            spectrum_data_point_counter: 0,
            buffer_size: buffer_size,
            mz_metadata: Default::default(),
        };
        this.add_array_metadata();
        this
    }

    implement_mz_metadata!();

    fn add_array_metadata(&mut self) {
        let mut spectrum_array_index =
            ArrayIndex::new(self.spectrum_buffers.prefix.clone(), HashMap::new());
        if let Ok(sub) = self
            .spectrum_buffers
            .schema
            .field_with_name(&self.spectrum_buffers.prefix)
            .cloned()
        {
            if let DataType::Struct(fields) = sub.data_type() {
                for f in fields.iter() {
                    if f.name() == "spectrum_index" {
                        continue;
                    }
                    let buffer_name =
                        BufferName::from_field(BufferContext::Spectrum, f.clone()).unwrap();
                    let aie = ArrayIndexEntry::from_buffer_name(
                        self.spectrum_buffers.prefix.clone(),
                        buffer_name,
                    );
                    spectrum_array_index.insert(aie.array_type.clone(), aie);
                }
            }
        }
        self.data_writer.append_key_value_metadata(KeyValue::new(
            "spectrum_array_index".to_string(),
            spectrum_array_index.to_json(),
        ));

        // let mut chromatogram_array_index =
        //     ArrayIndex::new(self.chromatogram_buffers.prefix.clone(), HashMap::new());
        // if let Ok(sub) = self
        //     .chromatogram_buffers
        //     .schema
        //     .field_with_name(&self.chromatogram_buffers.prefix)
        //     .cloned()
        // {
        //     if let DataType::Struct(fields) = sub.data_type() {
        //         for f in fields.iter() {
        //             if f.name() == "chromatogram_index" {
        //                 continue;
        //             }
        //             let buffer_name =
        //                 BufferName::from_field(BufferContext::Chromatogram, f.clone()).unwrap();
        //             let aie = ArrayIndexEntry::from_buffer_name(
        //                 self.chromatogram_buffers.prefix.clone(),
        //                 buffer_name,
        //             );
        //             chromatogram_array_index.insert(aie.array_type.clone(), aie);
        //         }
        //     }
        // }
        // self.append_key_value_metadata(
        //     "chromatogram_array_index".to_string(),
        //     chromatogram_array_index.to_json(),
        // );
    }

    pub fn append_key_value_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<Option<String>>,
    ) {
        self.metadata_writer
            .append_key_value_metadata(KeyValue::new(key.into(), value));
    }

    pub fn write_spectrum<
        C: ToMzPeakDataSeries + CentroidLike,
        D: ToMzPeakDataSeries + DeconvolutedCentroidLike,
    >(
        &mut self,
        spectrum: &impl SpectrumLike<C, D>,
    ) -> io::Result<()> {
        AbstractMzPeakWriter::write_spectrum(self, spectrum)
    }

    fn flush_data_arrays(&mut self) -> io::Result<()> {
        for batch in self.spectrum_buffers.drain() {
            self.spectrum_data_point_counter += batch.num_rows() as u64;
            self.data_writer.write(&batch)?;
        }

        // for batch in self.chromatogram_buffers.drain() {
        //     self.writer.write(&batch)?;
        // }
        Ok(())
    }

    fn flush_metadata_records(&mut self) -> io::Result<()> {
        let batch =
            serde_arrow::to_record_batch(&self.metadata_fields.fields(), &self.metadata_buffer)
                .unwrap();
        self.metadata_writer.write(&batch)?;
        self.metadata_buffer.clear();
        Ok(())
    }

    pub fn finish(
        &mut self,
    ) -> Result<parquet::format::FileMetaData, parquet::errors::ParquetError> {
        self.flush_data_arrays()?;
        self.flush_metadata_records()?;
        self.append_metadata();
        self.append_key_value_metadata("spectrum_count", Some(self.spectrum_counter.to_string()));
        self.append_key_value_metadata(
            "spectrum_data_point_count",
            Some(self.spectrum_data_point_counter.to_string()),
        );
        self.data_writer.finish()?;
        self.metadata_writer.finish()
    }
}
