use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    io::{self, prelude::*},
    marker::PhantomData,
    sync::Arc,
};

use arrow::{
    array::{Array, ArrayRef, RecordBatch, StructArray, new_null_array},
    datatypes::{DataType, Field, FieldRef, Fields, Schema, SchemaRef},
};
use mzpeaks::{CentroidPeak, DeconvolutedPeak};
use parquet::{
    arrow::{arrow_writer::ArrowWriterOptions, ArrowSchemaConverter, ArrowWriter},
    basic::{Compression, Encoding, ZstdLevel},
    file::properties::{EnabledStatistics, WriterProperties, WriterVersion},
    format::{KeyValue, SortingColumn},
};

use mzdata::{
    io::MZReaderType,
    meta::{
        DataProcessing, FileDescription, FileMetadataConfig, InstrumentConfiguration,
        MSDataFileMetadata, MassSpectrometryRun, Sample, Software,
    },
    params::Unit,
    prelude::*,
    spectrum::{ArrayType, BinaryDataArrayType, MultiLayerSpectrum, SignalContinuity},
};
use serde_arrow::schema::{SchemaLike, TracingOptions};

#[allow(unused)]
use crate::{
    archive::ZipArchiveWriter,
    entry::Entry,
    filter::{drop_where_column_is_zero, nullify_at_zero},
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
use crate::{
    chunk_series::ArrayChunk,
    filter::select_delta_model,
    peak_series::{MZ_ARRAY, array_map_to_schema_arrays_and_excess},
    spectrum::AuxiliaryArray,
};

fn _eval_spectra_from_iter_for_fields<
    C: CentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<CentroidPeak>,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<DeconvolutedPeak>,
>(
    iter: impl Iterator<Item = MultiLayerSpectrum<C, D>>,
    overrides: &HashMap<BufferName, BufferName>,
    use_chunked_encoding: bool,
) -> Vec<Arc<Field>> {
    let mut arrays: Vec<Arc<Field>> = Vec::new();
    let mut is_profile = 0;

    let field_it = iter
        .flat_map(|s| {
            log::trace!("Sampling arrays from {}", s.id());
            if s.signal_continuity() == SignalContinuity::Profile {
                is_profile += 1;
            }
            s.raw_arrays().and_then(|map| {
                if use_chunked_encoding {
                    ArrayChunk::from_arrays_delta(0, MZ_ARRAY, map, 50.0, None)
                        .ok()
                        .map(|s| {
                            (
                                s[0].to_schema(
                                    "spectrum_index",
                                    BufferContext::Spectrum,
                                    overrides,
                                )
                                .fields,
                                Vec::new(),
                            )
                        })
                } else {
                    array_map_to_schema_arrays(
                        BufferContext::Spectrum,
                        map,
                        map.mzs().map(|a| a.len()).unwrap_or_default(),
                        0,
                        "spectrum_index",
                        overrides,
                    )
                    .ok()
                }
            })
        })
        .map(|(fields, _arrs)| {
            let fields: Vec<_> = fields.iter().cloned().collect();
            fields
        })
        .flatten();

    for field in field_it {
        if arrays.iter().find(|f| f.name() == field.name()).is_none() {
            arrays.push(field);
        }
    }
    if is_profile > 0 {
        log::info!("Detected profile spectra");
    }
    arrays
}

pub fn sample_array_types_from_file_reader<
    C: CentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<CentroidPeak>,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<DeconvolutedPeak>,
>(
    reader: &mut MZReaderType<std::fs::File, C, D>,
    overrides: &HashMap<BufferName, BufferName>,
    use_chunked_encoding: bool,
) -> Vec<std::sync::Arc<arrow::datatypes::Field>> {
    let n = reader.len();
    if n == 0 {
        return Vec::new();
    }

    let it = [0, 100.min(n - 1), n / 2]
        .into_iter()
        .flat_map(|i| reader.get_spectrum_by_index(i));
    return _eval_spectra_from_iter_for_fields(it, overrides, use_chunked_encoding);
}

pub trait ArrayBufferWriter {
    fn buffer_context(&self) -> BufferContext;
    fn schema(&self) -> &SchemaRef;
    fn fields(&self) -> &Fields;
    fn prefix(&self) -> &str;
    fn add_arrays(&mut self, fields: Fields, arrays: Vec<ArrayRef>, size: usize, is_profile: bool);

    fn add<T: ToMzPeakDataSeries>(&mut self, spectrum_index: u64, peaks: &[T]);

    fn num_chunks(&self) -> usize;
    fn drain(&mut self) -> impl Iterator<Item = RecordBatch>;

    fn promote_record_batch_to_struct(
        prefix: &str,
        batch: RecordBatch,
        schema: SchemaRef,
    ) -> RecordBatch {
        let num_rows = batch.num_rows();
        let mut batch = Some(batch);
        let mut arrays = Vec::with_capacity(schema.fields().len());
        for f in schema.fields().iter() {
            if f.name() == prefix {
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

    fn overrides(&self) -> &HashMap<BufferName, BufferName>;

    fn as_array_index(&self) -> ArrayIndex {
        let mut spectrum_array_index: ArrayIndex =
            ArrayIndex::new(self.prefix().to_string(), HashMap::new());
        if let Ok(sub) = self
            .schema()
            .field_with_name(&self.prefix())
            .cloned()
        {
            if let DataType::Struct(fields) = sub.data_type() {
                for f in fields.iter() {
                    if f.name() == "spectrum_index" {
                        continue;
                    }
                    if let Some(buffer_name) =
                        BufferName::from_field(BufferContext::Spectrum, f.clone())
                    {
                        let aie = ArrayIndexEntry::from_buffer_name(
                            self.prefix().to_string(),
                            buffer_name,
                        );
                        spectrum_array_index.insert(aie.array_type.clone(), aie);
                    } else {
                        if f.name().ends_with("_chunk_end") || f.name().ends_with("_chunk_start") || f.name() != "chunk_encoding" {
                            continue;
                        }
                        log::warn!("Failed to construct metadata index for {f:#?}");
                    }
                }
            }
        }
        spectrum_array_index
    }
}

#[derive(Debug)]
pub struct ArrayBuffers {
    pub peak_array_fields: Fields,
    pub buffer_context: BufferContext,
    pub schema: SchemaRef,
    pub prefix: String,
    pub array_chunks: HashMap<String, Vec<ArrayRef>>,
    pub overrides: HashMap<BufferName, BufferName>,
    pub drop_zero_column: Option<Vec<String>>,
    pub null_zeros: bool,
    pub is_profile_buffer: Vec<bool>,
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
        self.is_profile_buffer.push(false);
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

    pub fn add_arrays(
        &mut self,
        fields: Fields,
        arrays: Vec<ArrayRef>,
        size: usize,
        is_profile: bool,
    ) {
        let mut visited = HashSet::new();
        for (f, arr) in fields.iter().zip(arrays) {
            self.array_chunks
                .get_mut(f.name())
                .unwrap_or_else(|| {
                    panic!("Unexpected field {f:?} for {:?}", self.peak_array_fields)
                })
                .push(arr);
            visited.insert(f.name());
        }
        self.is_profile_buffer.push(is_profile);
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

        let null_zeros = self.null_zeros;
        let is_profile = core::mem::take(&mut self.is_profile_buffer);
        let schema = SchemaRef::new(Schema::new(self.peak_array_fields.clone()));

        let mut null_targets = Vec::new();
        if null_zeros {
            for (i, f) in schema.fields().iter().enumerate() {
                let is_nullable =
                    if let Some(bname) = BufferName::from_field(self.buffer_context, f.clone()) {
                        matches!(
                            bname.array_type,
                            ArrayType::MZArray | ArrayType::IntensityArray
                        )
                    } else {
                        false
                    };
                if is_nullable {
                    null_targets.push(i);
                }
            }
        }

        let drop_zero_columns = self.drop_zero_column.clone();
        chunks
            .into_iter()
            .zip(is_profile)
            .map(move |(arrs, is_profile)| {
                let mut batch = RecordBatch::try_new(schema.clone(), arrs.clone()).unwrap_or_else(|e| {
                    let fields: Vec<_> = arrs.iter().map(|f| f.data_type()).collect();
                    panic!("Failed to convert peak buffers to record batch: {e}\n{fields:#?}\n{schema:#?}")
                });
                if is_profile {
                    if let Some(cols) = drop_zero_columns.as_ref() {
                        for (i, _f) in schema.fields().iter().enumerate().filter(|(_, f)| cols.contains(f.name())) {
                            match drop_where_column_is_zero(&batch,i) {
                                Ok(b) => {
                                    batch = b;
                                },
                                Err(e) => {
                                    log::error!("Failed to subset batch: {e}");
                                }
                            }
                            if null_zeros {
                                match nullify_at_zero(&batch, i, &null_targets) {
                                    Ok(b) => {
                                        batch = b;
                                    },
                                    Err(e) => {
                                        log::error!("Failed to nullify batch: {e}");
                                    }
                                }
                            }
                        }
                    }
                }
                batch
            })
            .map(|batch| self.promote_batch(batch, self.schema.clone()))
    }
}

impl ArrayBufferWriter for ArrayBuffers {
    fn buffer_context(&self) -> BufferContext {
        self.buffer_context
    }

    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn fields(&self) -> &Fields {
        &self.peak_array_fields
    }

    fn add_arrays(&mut self, fields: Fields, arrays: Vec<ArrayRef>, size: usize, is_profile: bool) {
        self.add_arrays(fields, arrays, size, is_profile);
    }

    fn add<T: ToMzPeakDataSeries>(&mut self, spectrum_index: u64, peaks: &[T]) {
        self.add(spectrum_index, peaks);
    }

    fn num_chunks(&self) -> usize {
        self.num_chunks()
    }

    fn drain(&mut self) -> impl Iterator<Item = RecordBatch> {
        self.drain()
    }

    fn prefix(&self) -> &str {
        &self.prefix
    }

    fn overrides(&self) -> &HashMap<BufferName, BufferName> {
        &self.overrides
    }
}

#[derive(Debug)]
pub struct ChunkBuffers {
    pub chunk_array_fields: Fields,
    pub buffer_context: BufferContext,
    pub schema: SchemaRef,
    pub prefix: String,
    pub chunks: Vec<StructArray>,
    pub overrides: HashMap<BufferName, BufferName>,
    pub drop_zero_column: Option<Vec<String>>,
    pub null_zeros: bool,
    pub is_profile_buffer: Vec<bool>,
}

impl ChunkBuffers {
    pub fn new(
        chunk_array_fields: Fields,
        buffer_context: BufferContext,
        schema: SchemaRef,
        prefix: String,
        chunks: Vec<StructArray>,
        overrides: HashMap<BufferName, BufferName>,
        drop_zero_column: Option<Vec<String>>,
        null_zeros: bool,
        is_profile_buffer: Vec<bool>,
    ) -> Self {
        Self {
            chunk_array_fields,
            buffer_context,
            schema,
            prefix,
            chunks,
            overrides,
            drop_zero_column,
            null_zeros,
            is_profile_buffer,
        }
    }
}

impl ArrayBufferWriter for ChunkBuffers {
    fn buffer_context(&self) -> BufferContext {
        self.buffer_context
    }

    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn fields(&self) -> &Fields {
        &self.chunk_array_fields
    }

    fn add_arrays(
        &mut self,
        fields: Fields,
        arrays: Vec<ArrayRef>,
        _size: usize,
        is_profile: bool,
    ) {
        self.chunks.push(StructArray::new(fields, arrays, None));
        self.is_profile_buffer.push(is_profile);
    }

    fn num_chunks(&self) -> usize {
        self.chunks.len()
    }

    fn add<T: ToMzPeakDataSeries>(&mut self, _spectrum_index: u64, _peaks: &[T]) {
        todo!("not ready yet")
    }

    fn drain(&mut self) -> impl Iterator<Item = RecordBatch> {
        let prefix = self.prefix().to_string();
        let schema = self.schema.clone();
        let is_profile = core::mem::take(&mut self.is_profile_buffer);
        let drop_zero_columns = self.drop_zero_column.clone();
        self.chunks.drain(..).zip(is_profile).map(move |(batch, is_profile)| {
            let mut batch = RecordBatch::from(batch);
            if is_profile {
                if let Some(cols) = drop_zero_columns.as_ref() {
                    for (i, _f) in schema.fields().iter().enumerate().filter(|(_, f)| cols.contains(f.name())) {
                        match drop_where_column_is_zero(&batch,i) {
                            Ok(b) => {
                                batch = b;
                            },
                            Err(e) => {
                                log::error!("Failed to subset batch: {e}");
                            }
                        }
                    }
                }
            }
            Self::promote_record_batch_to_struct(&prefix, batch, schema.clone())
        })
    }

    fn prefix(&self) -> &str {
        &self.prefix
    }

    fn overrides(&self) -> &HashMap<BufferName, BufferName> {
        &self.overrides
    }
}

#[derive(Debug)]
pub enum ArrayBufferWriterVariants {
    ChunkBuffers(ChunkBuffers),
    ArrayBuffers(ArrayBuffers),
}

impl From<ChunkBuffers> for ArrayBufferWriterVariants {
    fn from(value: ChunkBuffers) -> Self {
        Self::ChunkBuffers(value)
    }
}

impl From<ArrayBuffers> for ArrayBufferWriterVariants {
    fn from(value: ArrayBuffers) -> Self {
        Self::ArrayBuffers(value)
    }
}

impl ArrayBufferWriter for ArrayBufferWriterVariants {
    fn buffer_context(&self) -> BufferContext {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => {
                chunk_buffers.buffer_context()
            }
            ArrayBufferWriterVariants::ArrayBuffers(array_buffers) => {
                array_buffers.buffer_context()
            }
        }
    }

    fn schema(&self) -> &SchemaRef {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => chunk_buffers.schema(),
            ArrayBufferWriterVariants::ArrayBuffers(array_buffers) => array_buffers.schema(),
        }
    }

    fn fields(&self) -> &Fields {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => chunk_buffers.fields(),
            ArrayBufferWriterVariants::ArrayBuffers(array_buffers) => array_buffers.fields(),
        }
    }

    fn prefix(&self) -> &str {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => chunk_buffers.prefix(),
            ArrayBufferWriterVariants::ArrayBuffers(array_buffers) => array_buffers.prefix(),
        }
    }

    fn add<T: ToMzPeakDataSeries>(&mut self, spectrum_index: u64, peaks: &[T]) {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => {
                chunk_buffers.add(spectrum_index, peaks)
            }
            ArrayBufferWriterVariants::ArrayBuffers(array_buffers) => {
                array_buffers.add(spectrum_index, peaks)
            }
        }
    }

    fn add_arrays(&mut self, fields: Fields, arrays: Vec<ArrayRef>, size: usize, is_profile: bool) {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => {
                chunk_buffers.add_arrays(fields, arrays, size, is_profile)
            }
            ArrayBufferWriterVariants::ArrayBuffers(array_buffers) => {
                array_buffers.add_arrays(fields, arrays, size, is_profile)
            }
        }
    }

    fn num_chunks(&self) -> usize {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => chunk_buffers.num_chunks(),
            ArrayBufferWriterVariants::ArrayBuffers(array_buffers) => array_buffers.num_chunks(),
        }
    }

    fn drain(&mut self) -> impl Iterator<Item = RecordBatch> {
        let chunks: Vec<_> = match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => {
                chunk_buffers.drain().collect()
            }
            ArrayBufferWriterVariants::ArrayBuffers(array_buffers) => {
                array_buffers.drain().collect()
            }
        };
        log::debug!("Draining {} chunks", chunks.len());
        chunks.into_iter()
    }

    fn overrides(&self) -> &HashMap<BufferName, BufferName> {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => chunk_buffers.overrides(),
            ArrayBufferWriterVariants::ArrayBuffers(array_buffers) => array_buffers.overrides(),
        }
    }
}

#[derive(Debug)]
pub struct ArrayBuffersBuilder {
    prefix: String,
    array_fields: Vec<FieldRef>,
    overrides: HashMap<BufferName, BufferName>,
    null_zeros: bool,
}

impl Default for ArrayBuffersBuilder {
    fn default() -> Self {
        Self {
            prefix: "point".to_string(),
            array_fields: Default::default(),
            overrides: HashMap::new(),
            null_zeros: false,
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
        for f in self.array_fields.iter() {
            if !acc.iter().find(|(a, _)| *a == f.name()).is_some() {
                acc.push((f.name(), f.clone()));
            }
        }
        self.array_fields = acc.into_iter().map(|v| v.1).collect();
    }

    fn apply_overrides(&mut self) {
        self.deduplicate_fields();
        for (k, v) in self.overrides.iter() {
            let f = k.to_field();
            if let Some(i) = self.array_fields.iter().position(|p| p.name() == f.name()) {
                self.array_fields[i] = v.to_field();
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
        DataType::Struct(self.array_fields.clone().into())
    }

    pub fn add_field(mut self, field: FieldRef) -> Self {
        if self
            .array_fields
            .iter()
            .find(|f| f.name() == field.name())
            .is_none()
        {
            self.array_fields.push(field);
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

    pub fn canonicalize_field_order(&mut self) {
        self.array_fields.sort_by(|a, b| {
            if a.name() == "spectrum_index" || a.name() == "chromatogram_index" {
                return Ordering::Less;
            }
            if b.name() == "spectrum_index" || b.name() == "chromatogram_index" {
                return Ordering::Greater;
            }
            let a_name = BufferName::from_field(BufferContext::Spectrum, a.clone());
            let b_name = BufferName::from_field(BufferContext::Spectrum, b.clone());
            match (a_name, b_name) {
                (Some(a), Some(b)) => a.partial_cmp(&b).unwrap(),
                (Some(_), _) => Ordering::Less,
                (_, Some(_)) => Ordering::Greater,
                (_, _) => a.name().cmp(b.name()),
            }
        });
    }

    pub fn null_zeros(mut self, null_zeros: bool) -> Self {
        self.null_zeros = null_zeros;
        self
    }

    pub fn build_chunked(
        mut self,
        schema: SchemaRef,
        buffer_context: BufferContext,
        mask_zero_intensity_runs: bool,
    ) -> ChunkBuffers {
        let mut fields: Vec<FieldRef> = schema.fields().iter().cloned().collect();
        self.prefix = "chunk".to_string();
        fields.push(Field::new(self.prefix.clone(), self.dtype(), true).into());
        let schema = Arc::new(Schema::new_with_metadata(
            fields.clone(),
            schema.metadata().clone(),
        ));
        let drop_zero_column = if mask_zero_intensity_runs {
            Some(
                fields
                    .iter()
                    .filter(|c| c.name().starts_with("spectrum_intensity_"))
                    .map(|s| s.to_string())
                    .collect(),
            )
        } else {
            None
        };
        ChunkBuffers::new(
            fields.into(),
            buffer_context,
            schema,
            self.prefix.clone(),
            Vec::new(),
            self.overrides,
            drop_zero_column,
            self.null_zeros,
            Vec::new(),
        )
    }

    pub fn build(
        mut self,
        schema: SchemaRef,
        buffer_context: BufferContext,
        mask_zero_intensity_runs: bool,
    ) -> ArrayBuffers {
        self.canonicalize_field_order();
        let mut fields: Vec<FieldRef> = schema.fields().iter().cloned().collect();
        fields.push(Field::new(self.prefix.clone(), self.dtype(), true).into());

        let buffers: HashMap<String, _> = self
            .array_fields
            .iter()
            .map(|f| (f.name().clone(), Vec::new()))
            .collect();
        let drop_zero_column = if mask_zero_intensity_runs {
            Some(
                buffers
                    .keys()
                    .filter(|c| c.starts_with("spectrum_intensity_"))
                    .map(|s| s.to_string())
                    .collect(),
            )
        } else {
            None
        };
        ArrayBuffers {
            buffer_context,
            peak_array_fields: self.array_fields.clone().into(),
            schema: Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone())),
            prefix: self.prefix.clone(),
            array_chunks: buffers,
            overrides: self.overrides.clone(),
            drop_zero_column,
            is_profile_buffer: Vec::new(),
            null_zeros: self.null_zeros,
        }
    }
}

#[derive(Debug)]
pub struct MzPeakWriterBuilder {
    spectrum_arrays: ArrayBuffersBuilder,
    chromatogram_arrays: ArrayBuffersBuilder,
    buffer_size: usize,
    shuffle_mz: bool,
    chunked_encoding: bool,
    compression: Compression
}

impl Default for MzPeakWriterBuilder {
    fn default() -> Self {
        Self {
            spectrum_arrays: ArrayBuffersBuilder::default().prefix("point"),
            chromatogram_arrays: ArrayBuffersBuilder::default().prefix("chromatogram_point"),
            buffer_size: 5_000,
            shuffle_mz: false,
            chunked_encoding: false,
            compression: Compression::ZSTD(ZstdLevel::default())
        }
    }
}

impl MzPeakWriterBuilder {
    pub fn compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    pub fn add_spectrum_field(mut self, f: FieldRef) -> Self {
        self.spectrum_arrays = self.spectrum_arrays.add_field(f);
        self
    }

    pub fn chunked_encoding(mut self, value: bool) -> Self {
        self.chunked_encoding = value;
        self
    }

    pub fn add_spectrum_override(
        mut self,
        from: impl Into<BufferName>,
        to: impl Into<BufferName>,
    ) -> Self {
        self.spectrum_arrays = self.spectrum_arrays.add_override(from, to);
        self
    }

    pub fn shuffle_mz(mut self, shuffle_mz: bool) -> Self {
        self.shuffle_mz = shuffle_mz;
        self
    }

    pub fn null_zeros(mut self, null_zeros: bool) -> Self {
        self.spectrum_arrays = self.spectrum_arrays.null_zeros(null_zeros);
        self
    }

    pub fn add_chromatogram_override(
        mut self,
        from: impl Into<BufferName>,
        to: impl Into<BufferName>,
    ) -> Self {
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
        mask_zero_intensity_runs: bool,
    ) -> MzPeakSplitWriter<W> {
        MzPeakSplitWriter::new(
            data_writer,
            metadata_writer,
            self.spectrum_arrays,
            self.chromatogram_arrays,
            self.buffer_size,
            mask_zero_intensity_runs,
            self.shuffle_mz,
        )
    }

    pub fn build<W: Write + Send + Seek>(
        self,
        writer: W,
        mask_zero_intensity_runs: bool,
    ) -> MzPeakWriterType<W> {
        MzPeakWriterType::new(
            writer,
            self.spectrum_arrays,
            self.chromatogram_arrays,
            self.buffer_size,
            mask_zero_intensity_runs,
            self.shuffle_mz,
            self.chunked_encoding,
            self.compression,
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

    fn use_chunked_encoding(&self) -> bool {
        false
    }

    fn spectrum_entry_buffer_mut(&mut self) -> &mut Vec<Entry>;
    fn spectrum_data_buffer_mut(&mut self) -> &mut ArrayBufferWriterVariants;

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
        log::trace!("Writing spectrum {}", spectrum.id());
        let (median_delta, aux_arrays) = self.write_spectrum_peaks(spectrum)?;
        let mut entries = self.spectrum_to_entries(spectrum);
        if let Some(entry) = entries.first_mut() {
            if let Some(spec_ent) = entry.spectrum.as_mut() {
                spec_ent.median_delta = median_delta;
                spec_ent
                    .auxiliary_arrays
                    .extend(aux_arrays.unwrap_or_default());
            }
        }
        self.spectrum_entry_buffer_mut().extend(entries);
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
    ) -> io::Result<(Option<Vec<f64>>, Option<Vec<AuxiliaryArray>>)> {
        let spectrum_count = self.spectrum_counter();
        let (median_delta, aux_arrays) = match spectrum.peaks() {
            mzdata::spectrum::RefPeakDataLevel::Missing => (None, None),
            mzdata::spectrum::RefPeakDataLevel::RawData(binary_array_map) => {
                let n_points = spectrum.peaks().len();
                let is_profile = spectrum.signal_continuity() == SignalContinuity::Profile;

                if self.use_chunked_encoding() {
                    let chunks = ArrayChunk::from_arrays_delta(
                        spectrum_count,
                        MZ_ARRAY,
                        binary_array_map,
                        50.0,
                        None,
                    )?;
                    if !chunks.is_empty() {
                        let buffer = self.spectrum_data_buffer_mut();
                        let chunks = ArrayChunk::to_arrays(
                            &chunks,
                            "spectrum_index",
                            BufferContext::Spectrum,
                            buffer.schema(),
                            buffer.overrides(),
                        )?;
                        let size = chunks.len();
                        let (fields, arrays, _nulls) = chunks.into_parts();
                        buffer.add_arrays(fields, arrays, size, is_profile);
                    }

                    (None, None)
                } else {
                    let median_delta = if is_profile {
                        if let Ok(mzs) = binary_array_map.mzs() {
                            let delta_model = if let Ok(ints) = binary_array_map.intensities() {
                                let weights: Vec<f64> =
                                    ints.iter().map(|i| (*i + 1.0).ln().sqrt() as f64).collect();
                                select_delta_model(&mzs, Some(&weights))
                            } else {
                                select_delta_model(&mzs, None)
                            };
                            Some(delta_model)
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    let buffer = self.spectrum_data_buffer_mut();

                    let (fields, data, extra_arrays) = array_map_to_schema_arrays_and_excess(
                        BufferContext::Spectrum,
                        binary_array_map,
                        n_points,
                        spectrum_count,
                        "spectrum_index",
                        &buffer.fields(),
                        &buffer.overrides(),
                    )?;

                    buffer.add_arrays(fields, data, n_points, is_profile);
                    (median_delta, Some(extra_arrays))
                }
            }
            mzdata::spectrum::RefPeakDataLevel::Centroid(peaks) => {
                if self.use_chunked_encoding() {
                    todo!("Doesn't support centroid data and chunked encoding")
                }
                self.spectrum_data_buffer_mut()
                    .add(spectrum_count, peaks.as_slice());
                (None, None)
            }
            mzdata::spectrum::RefPeakDataLevel::Deconvoluted(peaks) => {
                if self.use_chunked_encoding() {
                    todo!("Doesn't support centroid data and chunked encoding")
                }
                self.spectrum_data_buffer_mut()
                    .add(spectrum_count, peaks.as_slice());
                (None, None)
            }
        };

        Ok((median_delta, aux_arrays))
    }
}

pub struct MzPeakWriterType<
    W: Write + Send + Seek,
    C: CentroidLike + ToMzPeakDataSeries = CentroidPeak,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries = DeconvolutedPeak,
> {
    archive_writer: Option<ArrowWriter<ZipArchiveWriter<W>>>,
    spectrum_buffers: ArrayBufferWriterVariants,
    #[allow(unused)]
    chromatogram_buffers: ArrayBuffers,
    metadata_buffer: Vec<Entry>,
    use_chunked_encoding: bool,

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

    fn use_chunked_encoding(&self) -> bool {
        self.use_chunked_encoding
    }

    fn spectrum_entry_buffer_mut(&mut self) -> &mut Vec<Entry> {
        &mut self.metadata_buffer
    }

    fn spectrum_data_buffer_mut(&mut self) -> &mut ArrayBufferWriterVariants {
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
        data_buffer: &ArrayBufferWriterVariants,
        index_path: String,
        shuffle_mz: bool,
        use_chunked_encoding: bool,
        compression: Compression,
    ) -> WriterProperties {
        let parquet_schema = Arc::new(
            ArrowSchemaConverter::new()
                .convert(&data_buffer.schema())
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
            .set_compression(compression)
            .set_dictionary_enabled(true)
            .set_sorting_columns(Some(sorted))
            .set_column_encoding(index_path.into(), Encoding::RLE)
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_statistics_enabled(EnabledStatistics::Page);

        if use_chunked_encoding {
            data_props = data_props.set_max_row_group_size(1024 * 256)
        }

        for (i, c) in parquet_schema.columns().iter().enumerate() {
            if c.path().to_string().contains("_mz_") && shuffle_mz {
                log::info!("Shuffling column {i} {}", c.path());
                data_props =
                    data_props.set_column_encoding(c.path().clone(), Encoding::BYTE_STREAM_SPLIT);
            }
            if c.name().ends_with("_index") {
                data_props =
                    data_props.set_column_encoding(c.path().clone(), Encoding::DELTA_BINARY_PACKED);
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
        mask_zero_intensity_runs: bool,
        shuffle_mz: bool,
        use_chunked_encoding: bool,
        compression: Compression,
    ) -> Self {
        let fields: Vec<FieldRef> = SchemaLike::from_type::<Entry>(TracingOptions::new()).unwrap();
        let metadata_fields: SchemaRef = Arc::new(Schema::new(fields));

        let spectrum_buffers: ArrayBufferWriterVariants = if use_chunked_encoding {
            spectrum_buffers
                .build_chunked(
                    Arc::new(Schema::empty()),
                    BufferContext::Spectrum,
                    mask_zero_intensity_runs,
                )
                .into()
        } else {
            let spectrum_buffers = spectrum_buffers.build(
                Arc::new(Schema::empty()),
                BufferContext::Spectrum,
                mask_zero_intensity_runs,
            );
            spectrum_buffers.into()
        };

        let chromatogram_buffers = chromatogram_buffers.build(
            Arc::new(Schema::empty()),
            BufferContext::Chromatogram,
            false,
        );

        let mut writer = ZipArchiveWriter::new(writer);
        writer.start_spectrum_data().unwrap();

        let data_props = Self::spectrum_data_writer_props(
            &spectrum_buffers,
            format!("{}.spectrum_index", spectrum_buffers.prefix()),
            shuffle_mz,
            use_chunked_encoding,
            compression,
        );

        let mut this = Self {
            archive_writer: Some(
                ArrowWriter::try_new_with_options(
                    writer,
                    spectrum_buffers.schema().clone(),
                    ArrowWriterOptions::new().with_properties(data_props),
                )
                .unwrap(),
            ),
            use_chunked_encoding,
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
        let spectrum_array_index: ArrayIndex = self.spectrum_buffers.as_array_index();
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

    spectrum_buffers: ArrayBufferWriterVariants,
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

    fn spectrum_data_buffer_mut(&mut self) -> &mut ArrayBufferWriterVariants {
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
    fn data_writer_props(data_buffer: &ArrayBuffers, shuffle_mz: bool) -> WriterProperties {
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

        let mut data_props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(
                ZstdLevel::try_new(3).unwrap(),
            ))
            .set_dictionary_enabled(true)
            .set_sorting_columns(Some(sorted))
            .set_column_encoding(spectrum_point_prefix.into(), Encoding::RLE)
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_statistics_enabled(EnabledStatistics::Page);

        for col in parquet_schema.columns() {
            if col.name().contains("_mz_") && shuffle_mz {
                data_props =
                    data_props.set_column_encoding(col.path().clone(), Encoding::BYTE_STREAM_SPLIT);
            }
        }

        data_props.build()
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
        mask_zero_intensity_runs: bool,
        shuffle_mz: bool,
    ) -> Self {
        let fields: Vec<FieldRef> = SchemaLike::from_type::<Entry>(TracingOptions::new()).unwrap();
        let metadata_fields: SchemaRef = Arc::new(Schema::new(fields));
        let spectrum_buffers = spectrum_buffers.build(
            Arc::new(Schema::empty()),
            BufferContext::Spectrum,
            mask_zero_intensity_runs,
        );
        let chromatogram_buffers = chromatogram_buffers.build(
            Arc::new(Schema::empty()),
            BufferContext::Chromatogram,
            false,
        );

        let data_props = Self::data_writer_props(&spectrum_buffers, shuffle_mz);
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
            spectrum_buffers: spectrum_buffers.into(),
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
            ArrayIndex::new(self.spectrum_buffers.prefix().to_string(), HashMap::new());
        if let Ok(sub) = self
            .spectrum_buffers
            .schema()
            .field_with_name(&self.spectrum_buffers.prefix().to_string())
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
                        self.spectrum_buffers.prefix().to_string(),
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
