use std::{
    collections::{HashMap, HashSet},
    io::{self, prelude::*},
    marker::PhantomData,
    sync::Arc,
};

use arrow::{
    array::{ArrayRef, RecordBatch, StructArray, new_null_array},
    datatypes::{DataType, Field, FieldRef, Fields, Schema, SchemaRef},
};
use mzpeaks::{CentroidPeak, DeconvolutedPeak};
use parquet::{
    arrow::{ArrowSchemaConverter, ArrowWriter, arrow_writer::ArrowWriterOptions},
    basic::{Encoding, ZstdLevel},
    file::properties::{EnabledStatistics, WriterProperties, WriterVersion},
    format::{KeyValue, SortingColumn},
};

use mzdata::{
    io::MZReaderType, meta::{
        DataProcessing, FileDescription, FileMetadataConfig, InstrumentConfiguration,
        MSDataFileMetadata, MassSpectrometryRun, Sample, Software,
    }, params::Unit, prelude::*, spectrum::{ArrayType, BinaryDataArrayType}
};
use serde_arrow::schema::{SchemaLike, TracingOptions};

use crate::{
    archive::ZipArchiveWriter,
    entry::Entry,
    param::{
        DataProcessing as MzDataProcessing, FileDescription as MzFileDescription,
        InstrumentConfiguration as MzInstrumentConfiguration, Sample as MzSample,
        Software as MzSoftware,
    },
    peak_series::{
        ArrayIndex, ArrayIndexEntry, BufferContext, BufferName, ToMzPeakDataSeries,
        array_map_to_schema_arrays,
    },
};

pub fn sample_array_types<
    C: CentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<CentroidPeak>,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<DeconvolutedPeak>,
>(
    reader: &mut MZReaderType<std::fs::File, C, D>,
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
                    array_map_to_schema_arrays(
                        BufferContext::Spectrum,
                        map,
                        map.mzs().map(|a| a.len()).unwrap_or_default(),
                        0,
                        "spectrum_index",
                        overrides,
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

#[derive(Debug)]
pub struct ArrayBuffers {
    pub peak_array_fields: Fields,
    pub schema: SchemaRef,
    pub prefix: String,
    pub array_chunks: HashMap<String, Vec<ArrayRef>>,
    pub overrides: HashMap<BufferName, BufferName>,
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
        let (fields, chunks) = T::to_arrays(spectrum_index, peaks, &self.overrides);
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
    overrides: HashMap<BufferName, BufferName>
}

impl Default for ArrayBuffersBuilder {
    fn default() -> Self {
        Self {
            prefix: "point".to_string(),
            peak_array_fields: Default::default(),
            overrides: HashMap::new(),
        }
    }
}

impl ArrayBuffersBuilder {
    pub fn prefix(mut self, value: impl ToString) -> Self {
        self.prefix = value.to_string();
        self
    }

    fn deduplicate_fields(&mut self) {
        let mut acc = Vec::new();
        for f in self.peak_array_fields.iter() {
            if !acc.iter().find(|(a, _)| *a == f.name()).is_some() {
                acc.push((f.name(), f.clone()));
            }
        }
        self.peak_array_fields = acc.into_iter().map(|v| v.1).collect();
    }

    fn apply_overrides(&mut self) {
        self.deduplicate_fields();
        for (k, v) in self.overrides.iter() {
            let f = k.to_field();
            if let Some(i) = self.peak_array_fields.iter().position(|p| p.name() == f.name()) {
                self.peak_array_fields[i] = v.to_field();
            }
        }
        self.deduplicate_fields();
    }

    pub fn add_override(mut self, from: impl Into<BufferName>, to: impl Into<BufferName>) -> Self {
        self.overrides.insert(from.into(), to.into());
        self.apply_overrides();
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
        self.apply_overrides();
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
            overrides: self.overrides.clone(),
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

    pub fn add_spectrum_override(mut self, from: impl Into<BufferName>, to: impl Into<BufferName>) -> Self {
        self.spectrum_arrays = self.spectrum_arrays.add_override(from, to);
        self
    }

    pub fn add_chromatogram_override(mut self, from: impl Into<BufferName>, to: impl Into<BufferName>) -> Self {
        self.chromatogram_arrays = self.chromatogram_arrays.add_override(from, to);
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

    pub fn build<W: Write + Send + Seek>(self, writer: W) -> MzPeakWriterType<W> {
        MzPeakWriterType::new(
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
    pub sample_list: Vec<Sample>,
    pub mass_spectrometry_run: MassSpectrometryRun,
}

impl MzMetadata {
    pub fn new(
        file_description: FileDescription,
        data_processing_list: Vec<DataProcessing>,
        software_list: Vec<Software>,
        instrument_configurations: Vec<InstrumentConfiguration>,
        sample_list: Vec<Sample>,
        mass_spectrometry_run: MassSpectrometryRun,
    ) -> Self {
        Self {
            file_description,
            data_processing_list,
            software_list,
            instrument_configurations,
            sample_list,
            mass_spectrometry_run,
        }
    }
}

macro_rules! implement_mz_metadata {
    () => {
        pub fn add_file_description(&mut self, value: &mzdata::meta::FileDescription) {
            *self.mz_metadata.file_description_mut() = value.clone();
        }

        pub fn add_instrument_configuration(
            &mut self,
            value: &mzdata::meta::InstrumentConfiguration,
        ) {
            self.mz_metadata
                .instrument_configurations_mut()
                .insert(value.id, value.clone());
        }

        pub fn add_software(&mut self, value: &mzdata::meta::Software) {
            self.mz_metadata.softwares_mut().push(value.clone());
        }

        pub fn add_data_processing(&mut self, value: &mzdata::meta::DataProcessing) {
            self.mz_metadata.data_processings_mut().push(value.clone());
        }

        pub fn add_sample(&mut self, value: &mzdata::meta::Sample) {
            self.mz_metadata.samples_mut().push(value.clone())
        }

        pub(crate) fn append_metadata(&mut self) {
            self.append_key_value_metadata(
                "file_description",
                Some(
                    serde_json::to_string_pretty(&MzFileDescription::from(
                        self.mz_metadata.file_description(),
                    ))
                    .unwrap(),
                ),
            );
            let tmp: Vec<_> = self
                .mz_metadata
                .instrument_configurations()
                .values()
                .map(|v| MzInstrumentConfiguration::from(v))
                .collect();
            self.append_key_value_metadata(
                "instrument_configuration_list",
                Some(serde_json::to_string_pretty(&tmp).unwrap()),
            );

            let tmp: Vec<_> = self
                .mz_metadata
                .data_processings()
                .iter()
                .map(|v| MzDataProcessing::from(v))
                .collect();
            self.append_key_value_metadata(
                "data_processing_method_list",
                Some(serde_json::to_string_pretty(&tmp).unwrap()),
            );

            let tmp: Vec<_> = self
                .mz_metadata
                .softwares()
                .iter()
                .map(|v| MzSoftware::from(v))
                .collect();
            self.append_key_value_metadata(
                "software_list",
                Some(serde_json::to_string_pretty(&tmp).unwrap()),
            );

            let tmp: Vec<_> = self
                .mz_metadata
                .samples()
                .iter()
                .map(|v| MzSample::from(v))
                .collect();
            self.append_key_value_metadata(
                "sample_list",
                Some(serde_json::to_string_pretty(&tmp).unwrap()),
            );

            self.append_key_value_metadata(
                "run",
                Some(
                    serde_json::to_string_pretty(self.mz_metadata.run_description().unwrap())
                        .unwrap(),
                ),
            )
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
                    &self.spectrum_data_buffer_mut().overrides
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

pub struct MzPeakWriterType<
    W: Write + Send + Seek,
    C: CentroidLike + ToMzPeakDataSeries = CentroidPeak,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries = DeconvolutedPeak,
> {
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

    mz_metadata: FileMetadataConfig,
    _t: PhantomData<(C, D)>,
}

impl<
    W: Write + Send + Seek,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> AbstractMzPeakWriter for MzPeakWriterType<W, C, D>
{
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

impl<
    W: Write + Send + Seek,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> MSDataFileMetadata for MzPeakWriterType<W, C, D>
{
    mzdata::delegate_impl_metadata_trait!(mz_metadata);
}

impl<
    W: Write + Send + Seek,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> MzPeakWriterType<W, C, D>
{
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

        for (i, c) in parquet_schema.columns().iter().enumerate() {
            if c.name().contains("_mz_") {
                log::info!("Shuffling column {i} {}", c.path());
                data_props = data_props.set_column_encoding(c.path().clone(), Encoding::BYTE_STREAM_SPLIT);
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
            _t: PhantomData,
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
        A: ToMzPeakDataSeries + CentroidLike,
        B: ToMzPeakDataSeries + DeconvolutedCentroidLike,
    >(
        &mut self,
        spectrum: &impl SpectrumLike<A, B>,
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

impl<
    W: Write + Send + Seek,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> Drop for MzPeakWriterType<W, C, D>
{
    fn drop(&mut self) {
        if let Err(_) = self.finish() {}
    }
}


/// Writer for the MzPeak format that writes the different data types to separate files
/// in an unarchived format.
pub struct MzPeakSplitWriter<
    W: Write + Send,
    C: CentroidLike + ToMzPeakDataSeries = CentroidPeak,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries = DeconvolutedPeak,
> {
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

    mz_metadata: FileMetadataConfig,
    _t: PhantomData<(C, D)>,
}

impl<
    W: Write + Send,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> Drop for MzPeakSplitWriter<W, C, D>
{
    fn drop(&mut self) {
        if let Err(_) = self.finish() {}
    }
}

impl<
    W: Write + Send,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> MSDataFileMetadata for MzPeakSplitWriter<W, C, D>
{
    mzdata::delegate_impl_metadata_trait!(mz_metadata);
}

impl<
    W: Write + Send,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> AbstractMzPeakWriter for MzPeakSplitWriter<W, C, D>
{
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

impl<
    W: Write + Send,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> MzPeakSplitWriter<W, C, D>
{
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
            _t: PhantomData,
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
        A: ToMzPeakDataSeries + CentroidLike,
        B: ToMzPeakDataSeries + DeconvolutedCentroidLike,
    >(
        &mut self,
        spectrum: &impl SpectrumLike<A, B>,
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

impl<
    W: Write + Send + Seek,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> SpectrumWriter<C, D> for MzPeakWriterType<W, C, D>
{
    fn write<S: SpectrumLike<C, D> + 'static>(&mut self, spectrum: &S) -> io::Result<usize> {
        self.write_spectrum(spectrum).map(|_| 1)
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(w) = self.archive_writer.as_mut() {
            w.flush()?;
        }
        Ok(())
    }

    fn close(&mut self) -> io::Result<()> {
        self.finish()?;
        Ok(())
    }
}

impl<
    W: Write + Send + Seek,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> SpectrumWriter<C, D> for MzPeakSplitWriter<W, C, D>
{
    fn write<S: SpectrumLike<C, D> + 'static>(&mut self, spectrum: &S) -> io::Result<usize> {
        self.write_spectrum(spectrum).map(|_| 1)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.data_writer.flush()?;
        self.metadata_writer.flush()?;
        Ok(())
    }

    fn close(&mut self) -> io::Result<()> {
        self.finish()?;
        Ok(())
    }
}
