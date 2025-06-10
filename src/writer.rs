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
    arrow::{
        ArrowSchemaConverter, ArrowWriter,
        arrow_writer::{
            ArrowColumnChunk, ArrowColumnWriter, ArrowLeafColumn, ArrowWriterOptions,
            compute_leaves, get_column_writers,
        },
    },
    basic::{Encoding, ZstdLevel},
    file::{
        properties::{EnabledStatistics, WriterProperties, WriterVersion},
        writer::SerializedFileWriter,
    },
    format::KeyValue,
    schema::types::SchemaDescriptor,
};

use mzdata::{
    params::Unit,
    prelude::*,
    spectrum::{ArrayType, BinaryDataArrayType},
};
use serde_arrow::schema::{SchemaLike, TracingOptions};

use crate::{
    entry::MzPeaksEntry,
    param::{
        MzPeaksDataProcessing, MzPeaksFileDescription, MzPeaksInstrumentConfiguration,
        MzPeaksSoftware,
    },
    peak_series::{
        ArrayIndex, ArrayIndexEntry, BufferContext, BufferName, ToMzPeaksDataSeries,
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

    pub fn add<T: ToMzPeaksDataSeries>(&mut self, spectrum_index: u64, peaks: &[T]) {
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

    pub fn add_peak_type<T: ToMzPeaksDataSeries>(mut self) -> Self {
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
pub struct MzPeaksWriterBuilder {
    spectrum_arrays: ArrayBuffersBuilder,
    chromatogram_arrays: ArrayBuffersBuilder,
    buffer_size: usize,
}

impl Default for MzPeaksWriterBuilder {
    fn default() -> Self {
        Self {
            spectrum_arrays: ArrayBuffersBuilder::default().prefix("point"),
            chromatogram_arrays: ArrayBuffersBuilder::default().prefix("chromatogram_point"),
            buffer_size: 5_000,
        }
    }
}

impl MzPeaksWriterBuilder {
    pub fn add_spectrum_field(mut self, f: FieldRef) -> Self {
        self.spectrum_arrays = self.spectrum_arrays.add_field(f);
        self
    }

    pub fn add_spectrum_peak_type<T: ToMzPeaksDataSeries>(mut self) -> Self {
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

    pub fn build<W: Write + Send>(self, writer: W) -> MzPeaksWriter<W> {
        MzPeaksWriter::new(
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

pub struct MzPeaksWriter<W: Write + Send> {
    writer: ArrowWriter<W>,
    fields: SchemaRef,
    #[allow(unused)]
    parquet_schema: Arc<SchemaDescriptor>,
    record_buffer: Vec<MzPeaksEntry>,
    spectrum_buffers: ArrayBuffers,
    chromatogram_buffers: ArrayBuffers,
    spectrum_counter: u64,
    buffer_size: usize,

    file_description: MzPeaksFileDescription,
    data_processing_list: Vec<MzPeaksDataProcessing>,
    software_list: Vec<MzPeaksSoftware>,
    instrument_configurations: Vec<MzPeaksInstrumentConfiguration>,
}

impl<W: Write + Send> Drop for MzPeaksWriter<W> {
    fn drop(&mut self) {
        if !self.record_buffer.is_empty() {
            self.flush_buffer().unwrap();
        }
        if let Err(_) = self.writer.finish() {}
    }
}

impl<W: Write + Send> MzPeaksWriter<W> {
    pub fn builder() -> MzPeaksWriterBuilder {
        MzPeaksWriterBuilder::default()
    }

    pub fn new(
        writer: W,
        spectrum_buffers: ArrayBuffersBuilder,
        chromatogram_buffers: ArrayBuffersBuilder,
        buffer_size: usize,
    ) -> Self {
        let fields: Vec<FieldRef> =
            SchemaLike::from_type::<MzPeaksEntry>(TracingOptions::new().allow_null_fields(true))
                .unwrap();
        let fields: SchemaRef = Arc::new(Schema::new(fields));
        let mut spectrum_buffers = spectrum_buffers.build(fields);
        let chromatogram_buffers = chromatogram_buffers.build(spectrum_buffers.schema);
        spectrum_buffers.schema = chromatogram_buffers.schema.clone();
        let fields = chromatogram_buffers.schema.clone();

        let props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(
                ZstdLevel::try_new(3).unwrap(),
            ))
            .set_dictionary_enabled(true)
            .set_column_encoding(
                format!("{}.spectrum_index", spectrum_buffers.prefix).into(),
                Encoding::RLE,
            )
            .set_column_bloom_filter_enabled("spectrum.id".into(), true)
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_statistics_enabled(EnabledStatistics::Page)
            .build();

        let parquet_schema = Arc::new(ArrowSchemaConverter::new().convert(&fields).unwrap());

        // let mut writer = SerializedFileWriter::new(
        //     writer,
        //     parquet_schema.root_schema_ptr(),
        //     Arc::new(props.clone()),
        // )
        // .unwrap();

        let mut this = Self {
            fields: fields.clone(),
            parquet_schema,
            writer: ArrowWriter::try_new_with_options(
                writer,
                fields,
                ArrowWriterOptions::new().with_properties(props),
            )
            .unwrap(),
            record_buffer: Vec::new(),
            spectrum_buffers,
            chromatogram_buffers,
            spectrum_counter: 0,
            buffer_size: buffer_size,
            file_description: Default::default(),
            data_processing_list: Vec::new(),
            instrument_configurations: Vec::new(),
            software_list: Vec::new(),
        };
        this.add_array_metadata();
        this
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

        let mut chromatogram_array_index =
            ArrayIndex::new(self.chromatogram_buffers.prefix.clone(), HashMap::new());
        if let Ok(sub) = self
            .chromatogram_buffers
            .schema
            .field_with_name(&self.chromatogram_buffers.prefix)
            .cloned()
        {
            if let DataType::Struct(fields) = sub.data_type() {
                for f in fields.iter() {
                    if f.name() == "chromatogram_index" {
                        continue;
                    }
                    let buffer_name =
                        BufferName::from_field(BufferContext::Chromatogram, f.clone()).unwrap();
                    let aie = ArrayIndexEntry::from_buffer_name(
                        self.chromatogram_buffers.prefix.clone(),
                        buffer_name,
                    );
                    chromatogram_array_index.insert(aie.array_type.clone(), aie);
                }
            }
        }
        self.append_key_value_metadata(
            "chromatogram_array_index".to_string(),
            chromatogram_array_index.to_json(),
        );
    }

    pub fn write_spectrum<
        C: ToMzPeaksDataSeries + CentroidLike,
        D: ToMzPeaksDataSeries + DeconvolutedCentroidLike,
    >(
        &mut self,
        spectrum: &impl SpectrumLike<C, D>,
    ) -> io::Result<()> {
        let entries = MzPeaksEntry::from_spectrum(spectrum, Some(self.spectrum_counter));

        self.record_buffer.extend(entries);
        match spectrum.peaks() {
            mzdata::spectrum::RefPeakDataLevel::Missing => {}
            mzdata::spectrum::RefPeakDataLevel::RawData(binary_array_map) => {
                let n_points = spectrum.peaks().len();
                let (fields, data) = array_map_to_schema_arrays(
                    crate::BufferContext::Spectrum,
                    binary_array_map,
                    n_points,
                    self.spectrum_counter,
                    "spectrum_index",
                )?;
                self.spectrum_buffers.add_arrays(fields, data, n_points);
            }
            mzdata::spectrum::RefPeakDataLevel::Centroid(peaks) => {
                self.spectrum_buffers
                    .add(self.spectrum_counter, peaks.as_slice());
            }
            mzdata::spectrum::RefPeakDataLevel::Deconvoluted(peaks) => {
                self.spectrum_buffers
                    .add(self.spectrum_counter, peaks.as_slice());
            }
        }

        self.spectrum_counter += 1;

        if self.record_buffer.len() > self.buffer_size {
            self.flush_buffer()?;
        }
        Ok(())
    }

    pub fn flush_buffer(&mut self) -> io::Result<()> {
        log::debug!("Flushing buffers");
        let batch =
            serde_arrow::to_record_batch(&self.fields.fields(), &self.record_buffer).unwrap();
        self.writer.write(&batch)?;
        self.record_buffer.clear();

        for batch in self.spectrum_buffers.drain() {
            self.writer.write(&batch)?;
        }

        for batch in self.chromatogram_buffers.drain() {
            self.writer.write(&batch)?;
        }
        Ok(())
    }

    pub fn append_key_value_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<Option<String>>,
    ) {
        self.writer
            .append_key_value_metadata(KeyValue::new(key.into(), value));
    }

    pub fn finish(
        &mut self,
    ) -> Result<parquet::format::FileMetaData, parquet::errors::ParquetError> {
        if !self.record_buffer.is_empty() {
            self.flush_buffer()?;
        }
        self.append_key_value_metadata(
            "file_description",
            Some(serde_json::to_string_pretty(&self.file_description).unwrap()),
        );
        self.append_key_value_metadata(
            "instrument_configuration_list",
            Some(serde_json::to_string_pretty(&self.instrument_configurations).unwrap()),
        );
        self.append_key_value_metadata(
            "data_processing_method_list",
            Some(serde_json::to_string_pretty(&self.data_processing_list).unwrap()),
        );
        self.append_key_value_metadata(
            "software_list",
            Some(serde_json::to_string_pretty(&self.software_list).unwrap()),
        );
        self.writer.finish()
    }
}
