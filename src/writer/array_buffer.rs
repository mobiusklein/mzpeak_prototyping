use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    sync::Arc,
};

use arrow::{
    array::{Array, ArrayRef, RecordBatch, StructArray, new_null_array},
    datatypes::{DataType, Field, FieldRef, Fields, Schema, SchemaRef},
};
use mzdata::spectrum::ArrayType;

use crate::{
    BufferContext, BufferName, ToMzPeakDataSeries,
    buffer_descriptors::{BufferOverrideTable, BufferPriority, BufferTransform},
    filter::{drop_where_column_is_zero, nullify_at_zero},
    peak_series::{ArrayIndex, ArrayIndexEntry},
};

pub trait ArrayBufferWriter {
    /// Whether the buffer describes a spectrum or chromatogram
    fn buffer_context(&self) -> BufferContext;
    /// The Arrow schema this buffer is embedded in
    fn schema(&self) -> &SchemaRef;
    /// The individual fields in this buffer's schema
    fn fields(&self) -> &Fields;
    /// The name of the prefix in the schema for these fields
    fn prefix(&self) -> &str;

    /// Whether or not to write a separate time series for each entry
    fn include_time(&self) -> bool;

    /// The path in the schema to reach the spectrum index column
    fn index_path(&self) -> String {
        format!("{}.spectrum_index", self.prefix())
    }

    /// Add the provided `arrays` belonging to `fields` to the buffer
    fn add_arrays(&mut self, fields: Fields, arrays: Vec<ArrayRef>, size: usize, is_profile: bool);

    /// Whether or not to use a gapped sparse encoding, filling zero-intensity points with nulls left
    /// after zero intensity runs were dropped ([`ArrayBufferWriter::drop_zero_intensity`]).
    fn nullify_zero_intensity(&self) -> bool;

    /// Whether or not to drop runs of zero-intensity points from profile data, leaving only one zero-intensity
    /// point flanking the gaps.
    fn drop_zero_intensity(&self) -> bool;

    /// Add a peak list to the buffer.
    ///
    /// This might call [`ArrayBufferWriter::add_arrays`].
    fn add<T: ToMzPeakDataSeries>(
        &mut self,
        spectrum_index: u64,
        spectrum_time: Option<f32>,
        peaks: &[T],
    );

    /// The number of distinct blocks of data points buffered
    fn num_chunks(&self) -> usize;

    /// Drain the internal buffers into a sequence of [`RecordBatch`]
    fn drain(&mut self) -> impl Iterator<Item = RecordBatch>;

    /// Convert a flat [`RecordBatch`] to a nested [`RecordBatch`] under `prefix`
    /// and fill any missing top-level arrays in `schema` with null arrays.
    fn promote_record_batch_to_struct(
        prefix: &str,
        batch: RecordBatch,
        schema: SchemaRef,
    ) -> RecordBatch {
        let num_rows = batch.num_rows();
        let batch_schema = batch.schema();
        let mut batch = Some(batch);
        let mut arrays = Vec::with_capacity(schema.fields().len());

        /*
        Search the host schema for this buffer's prefix, and fill in those arrays with
        the incoming batch of data. Otherwise fill those rows with nulls. In the simplest
        case, those branches do not exist.

        NB: When things are really messed up, this can lead to misleading error messages
        implying that the prefix is missing from the batch's schema. This is where the
        prefix is *added* to the incoming data! In that case, it's usually that the arrays
        are in a different order.
        */
        for f in schema.fields().iter() {
            if f.name() == prefix {
                if let Some(mut batch) = batch.take() {
                    let mut columns = Vec::with_capacity(batch.num_columns());
                    if let DataType::Struct(fields_of) = f.data_type() {
                        for col in fields_of.iter() {
                            if let Some(col) = batch.column_by_name(col.name()).cloned() {
                                columns.push(col);
                            } else {
                                log::trace!(
                                    "{col:?} was not found in the schema, populating with {num_rows} nulls"
                                );
                                columns
                                    .push(arrow::array::new_null_array(col.data_type(), num_rows));
                            }
                        }
                        batch =
                            RecordBatch::try_new(Arc::new(Schema::new(fields_of.clone())), columns)
                                .unwrap();
                    }
                    let x = Arc::new(StructArray::from(batch));
                    arrays.push(x as ArrayRef);
                }
            } else {
                arrays.push(new_null_array(f.data_type(), num_rows));
            }
        }
        RecordBatch::try_new(schema.clone(), arrays).unwrap_or_else(|e| {

            // Try to explain why the schema failed to be projected onto the arrays:
            match schema.field_with_name(prefix) {
                // We found the prefix data type, so we can show
                Ok(subset) => {
                    if let DataType::Struct(fields) = subset.data_type() {
                        let expected = Schema::new(fields.clone());
                        log::error!("Expected: {expected:#?}");
                        log::error!("Received: {:#?}", batch_schema);
                    } else {
                        log::error!("Expected data type is malformed: {prefix} => {subset:?}, expected struct/group")
                    }
                }
                Err(e2) => {
                    log::error!("Expected data type is malformed: {prefix} not found: {e2}")
                }
            }
            panic!("Failed to convert arrays to record batch: {e:#?}");
        })
    }

    /// Get the override table to control how array data types should be mapped to [`BufferName`]
    fn overrides(&self) -> &BufferOverrideTable;

    /// Convert the registered set of [`BufferName`] embedded in the Arrow schema.
    ///
    /// If the column doesn't have any Arrow metadata, it will not be tracked.
    fn as_array_index(&self) -> ArrayIndex {
        let mut array_index: ArrayIndex =
            ArrayIndex::new(self.prefix().to_string(), HashMap::new());
        if let Ok(sub) = self.schema().field_with_name(self.prefix()).cloned() {
            if let DataType::Struct(fields) = sub.data_type() {
                for f in fields.iter() {
                    if f.name() == BufferContext::Spectrum.index_field().name()
                        || f.name() == BufferContext::Chromatogram.index_field().name()
                    {
                        continue;
                    }
                    if let Some(buffer_name) =
                        BufferName::from_field(self.buffer_context(), f.clone())
                    {
                        let aie = ArrayIndexEntry::from_buffer_name(
                            self.prefix().to_string(),
                            buffer_name,
                            Some(f),
                        );
                        array_index.push(aie);
                    }
                }
            }
        }
        log::trace!(
            "{} array indices: {}",
            self.buffer_context(),
            array_index.to_json()
        );
        array_index
    }
}

/// A data buffer for the `point layout`
#[derive(Debug)]
pub struct PointBuffers {
    /// The materialized columns
    peak_array_fields: Fields,
    buffer_context: BufferContext,
    /// The complete schema for this table
    schema: SchemaRef,
    /// The name of the top-level node for this group
    prefix: String,
    /// Field name to chunks of array data for each column
    array_chunks: HashMap<String, Vec<ArrayRef>>,
    overrides: BufferOverrideTable,
    drop_zero_column: Option<Vec<String>>,
    null_zeros: bool,
    is_profile_buffer: Vec<bool>,
    include_time: bool,
}

impl PointBuffers {
    pub fn dtype(&self) -> DataType {
        DataType::Struct(self.peak_array_fields.clone())
    }

    pub fn len(&self) -> usize {
        self.array_chunks
            .values()
            .map(|v| v.iter().map(|s| s.len()).sum::<usize>())
            .max()
            .unwrap_or_default()
    }

    pub fn is_empty(&self) -> bool {
        self.array_chunks.values().all(|v| v.is_empty())
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

    pub fn add<T: ToMzPeakDataSeries>(
        &mut self,
        spectrum_index: u64,
        spectrum_time: Option<f32>,
        peaks: &[T],
    ) {
        let (fields, chunks) = T::to_arrays(spectrum_index, spectrum_time, peaks, &self.overrides);
        let mut visited = HashSet::new();
        for (f, arr) in fields.iter().zip(chunks.into_iter()) {
            let name = BufferName::from_field(self.buffer_context, f.clone())
                .map(|b| b.to_string())
                .unwrap_or(f.name().to_string());
            self.array_chunks
                .get_mut(&name)
                .unwrap_or_else(|| panic!("Unexpected field {f:?}"))
                .push(arr);
            visited.insert(name);
        }
        self.is_profile_buffer.push(false);
        for (f, chunk) in self.array_chunks.iter_mut() {
            if !visited.contains(f) {
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
                    panic!("Unexpected field {f:?}\nfor\n{:#?}", self.peak_array_fields)
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

impl ArrayBufferWriter for PointBuffers {
    fn buffer_context(&self) -> BufferContext {
        self.buffer_context
    }

    fn include_time(&self) -> bool {
        self.include_time
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

    fn add<T: ToMzPeakDataSeries>(
        &mut self,
        spectrum_index: u64,
        spectrum_time: Option<f32>,
        peaks: &[T],
    ) {
        self.add(spectrum_index, spectrum_time, peaks);
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

    fn overrides(&self) -> &BufferOverrideTable {
        &self.overrides
    }

    fn nullify_zero_intensity(&self) -> bool {
        self.null_zeros
    }

    fn drop_zero_intensity(&self) -> bool {
        self.drop_zero_column.is_some()
    }
}

/// A data buffer for the `chunked layout`
#[derive(Debug)]
pub struct ChunkBuffers {
    chunk_array_fields: Fields,
    buffer_context: BufferContext,
    schema: SchemaRef,
    prefix: String,
    chunk_buffer: Vec<StructArray>,
    overrides: BufferOverrideTable,
    /// Currently not used. This requires supporting changes in [`ArrowArrayChunk`](crate::chunk_series::ArrowArrayChunk).
    drop_zero_column: Option<Vec<String>>,
    null_zeros: bool,
    is_profile_buffer: Vec<bool>,
    include_time: bool,
}

impl ChunkBuffers {
    pub fn new(
        chunk_array_fields: Fields,
        buffer_context: BufferContext,
        schema: SchemaRef,
        prefix: String,
        chunks: Vec<StructArray>,
        overrides: BufferOverrideTable,
        drop_zero_column: Option<Vec<String>>,
        null_zeros: bool,
        is_profile_buffer: Vec<bool>,
        include_time: bool,
    ) -> Self {
        Self {
            chunk_array_fields,
            buffer_context,
            schema,
            prefix,
            chunk_buffer: chunks,
            overrides,
            drop_zero_column,
            null_zeros,
            is_profile_buffer,
            include_time,
        }
    }

    pub fn len(&self) -> usize {
        self.chunk_buffer.iter().map(|c| c.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.chunk_buffer.is_empty()
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
        self.chunk_buffer
            .push(StructArray::new(fields, arrays, None));
        self.is_profile_buffer.push(is_profile);
    }

    fn num_chunks(&self) -> usize {
        self.chunk_buffer.len()
    }

    fn add<T: ToMzPeakDataSeries>(
        &mut self,
        _spectrum_index: u64,
        _spectrum_time: Option<f32>,
        _peaks: &[T],
    ) {
        todo!("not ready yet")
    }

    fn drain(&mut self) -> impl Iterator<Item = RecordBatch> {
        let prefix = self.prefix().to_string();
        let schema = self.schema.clone();
        let is_profile = core::mem::take(&mut self.is_profile_buffer);
        self.chunk_buffer
            .drain(..)
            .zip(is_profile)
            .map(move |(batch, _is_profile)| {
                let batch = RecordBatch::from(batch);
                Self::promote_record_batch_to_struct(&prefix, batch, schema.clone())
            })
    }

    fn prefix(&self) -> &str {
        &self.prefix
    }

    fn overrides(&self) -> &BufferOverrideTable {
        &self.overrides
    }

    fn nullify_zero_intensity(&self) -> bool {
        self.null_zeros
    }

    fn drop_zero_intensity(&self) -> bool {
        self.drop_zero_column.is_some()
    }

    fn include_time(&self) -> bool {
        self.include_time
    }
}

/// An abstraction over [`ArrayBufferWriter`] types
#[derive(Debug)]
pub enum ArrayBufferWriterVariants {
    /// This writer uses the `chunked` layout
    ChunkBuffers(ChunkBuffers),
    /// This writer uses the `point` layout
    PointBuffers(PointBuffers),
}

impl ArrayBufferWriterVariants {
    pub fn len(&self) -> usize {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => chunk_buffers.len(),
            ArrayBufferWriterVariants::PointBuffers(point_buffers) => point_buffers.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => chunk_buffers.is_empty(),
            ArrayBufferWriterVariants::PointBuffers(point_buffers) => point_buffers.is_empty(),
        }
    }
}

impl From<ChunkBuffers> for ArrayBufferWriterVariants {
    fn from(value: ChunkBuffers) -> Self {
        Self::ChunkBuffers(value)
    }
}

impl From<PointBuffers> for ArrayBufferWriterVariants {
    fn from(value: PointBuffers) -> Self {
        Self::PointBuffers(value)
    }
}

impl ArrayBufferWriter for ArrayBufferWriterVariants {
    fn buffer_context(&self) -> BufferContext {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => {
                chunk_buffers.buffer_context()
            }
            ArrayBufferWriterVariants::PointBuffers(array_buffers) => {
                array_buffers.buffer_context()
            }
        }
    }

    fn schema(&self) -> &SchemaRef {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => chunk_buffers.schema(),
            ArrayBufferWriterVariants::PointBuffers(array_buffers) => array_buffers.schema(),
        }
    }

    fn fields(&self) -> &Fields {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => chunk_buffers.fields(),
            ArrayBufferWriterVariants::PointBuffers(array_buffers) => array_buffers.fields(),
        }
    }

    fn prefix(&self) -> &str {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => chunk_buffers.prefix(),
            ArrayBufferWriterVariants::PointBuffers(array_buffers) => array_buffers.prefix(),
        }
    }

    fn add<T: ToMzPeakDataSeries>(
        &mut self,
        spectrum_index: u64,
        spectrum_time: Option<f32>,
        peaks: &[T],
    ) {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => {
                chunk_buffers.add(spectrum_index, spectrum_time, peaks)
            }
            ArrayBufferWriterVariants::PointBuffers(array_buffers) => {
                array_buffers.add(spectrum_index, spectrum_time, peaks)
            }
        }
    }

    fn add_arrays(&mut self, fields: Fields, arrays: Vec<ArrayRef>, size: usize, is_profile: bool) {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => {
                chunk_buffers.add_arrays(fields, arrays, size, is_profile)
            }
            ArrayBufferWriterVariants::PointBuffers(array_buffers) => {
                array_buffers.add_arrays(fields, arrays, size, is_profile)
            }
        }
    }

    fn num_chunks(&self) -> usize {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => chunk_buffers.num_chunks(),
            ArrayBufferWriterVariants::PointBuffers(array_buffers) => array_buffers.num_chunks(),
        }
    }

    fn drain(&mut self) -> impl Iterator<Item = RecordBatch> {
        let chunks: Vec<_> = match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => {
                chunk_buffers.drain().collect()
            }
            ArrayBufferWriterVariants::PointBuffers(array_buffers) => {
                array_buffers.drain().collect()
            }
        };
        log::trace!("Draining {} chunks", chunks.len());
        chunks.into_iter()
    }

    fn overrides(&self) -> &BufferOverrideTable {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => chunk_buffers.overrides(),
            ArrayBufferWriterVariants::PointBuffers(array_buffers) => array_buffers.overrides(),
        }
    }

    fn nullify_zero_intensity(&self) -> bool {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => {
                chunk_buffers.nullify_zero_intensity()
            }
            ArrayBufferWriterVariants::PointBuffers(point_buffers) => {
                point_buffers.nullify_zero_intensity()
            }
        }
    }

    fn drop_zero_intensity(&self) -> bool {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => {
                chunk_buffers.drop_zero_intensity()
            }
            ArrayBufferWriterVariants::PointBuffers(point_buffers) => {
                point_buffers.drop_zero_intensity()
            }
        }
    }

    fn include_time(&self) -> bool {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => chunk_buffers.include_time(),
            ArrayBufferWriterVariants::PointBuffers(point_buffers) => point_buffers.include_time(),
        }
    }
}

/// A builder for [`ArrayBufferWriter`] types.
#[derive(Debug)]
pub struct ArrayBuffersBuilder {
    prefix: String,
    array_fields: Vec<FieldRef>,
    overrides: BufferOverrideTable,
    null_zeros: bool,
    include_time: bool,
    buffer_context: BufferContext,
}

/// The builder will default to the `point` layout
impl Default for ArrayBuffersBuilder {
    fn default() -> Self {
        Self {
            prefix: "point".to_string(),
            array_fields: Default::default(),
            overrides: BufferOverrideTable::default(),
            null_zeros: false,
            include_time: false,
            buffer_context: BufferContext::Spectrum,
        }
    }
}

impl ArrayBuffersBuilder {
    /// Set the prefix for the data structure, all other arrays will be nested under a group/struct
    /// with this name.
    pub fn prefix(mut self, value: impl ToString) -> Self {
        self.prefix = value.to_string();
        self
    }

    /// Set the [`BufferContext`] that this will build arrays for
    pub fn with_context(mut self, value: BufferContext) -> Self {
        self.buffer_context = value;
        self
    }

    fn deduplicate_fields(&mut self) {
        let mut acc = Vec::new();
        for f in self.array_fields.iter() {
            if !acc.iter().any(|(a, _)| *a == f.name()) {
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

    /// Register an new rule mapping from one [`BufferName`]-like to another [`BufferName`]-like
    /// when later writing arrays
    pub fn add_override(mut self, from: impl Into<BufferName>, to: impl Into<BufferName>) -> Self {
        self.overrides.insert(from.into(), to.into());
        self.apply_overrides();
        self
    }

    /// Get a copy of the set of registered override rules
    pub fn overrides(&self) -> BufferOverrideTable {
        self.overrides.clone()
    }

    /// Get the current set of [`arrow::datatypes::Field`] for the set of arrays to be written
    pub fn dtype(&self) -> DataType {
        DataType::Struct(self.array_fields.clone().into())
    }

    /// Register a new [`arrow::datatypes::FieldRef`] with the current schema
    pub fn add_field(mut self, field: FieldRef) -> Self {
        if !self.array_fields.iter().any(|f| f.name() == field.name()) {
            self.array_fields.push(field);
        }
        self.apply_overrides();
        self
    }

    /// Register all the fields of `T` as given by [`ToMzPeakDataSeries::to_fields`]
    /// with the current schema.
    ///
    /// See [`Self::add_field`]
    pub fn add_peak_type<T: ToMzPeakDataSeries>(mut self) -> Self {
        for f in T::to_fields().iter().cloned() {
            self = self.add_field(f);
        }
        self
    }

    /// Canonicalize the order of array fields
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

    fn mark_primary_arrays(&mut self) -> Vec<BufferName> {
        let mut seen = HashSet::new();
        let mut has_priority = Vec::new();
        for f in self.array_fields.iter_mut() {
            if let Some(mut buff) = BufferName::from_field(self.buffer_context, f.clone()) {
                if !seen.contains(&(buff.array_type.clone(), buff.buffer_format)) {
                    seen.insert((buff.array_type.clone(), buff.buffer_format));
                    log::debug!("Setting {buff} {buff:?} to be primary");
                    buff = buff.with_priority(Some(BufferPriority::Primary));
                    has_priority.push(buff.clone());
                    *f = Arc::new(
                        f.as_ref()
                            .clone()
                            .with_metadata(buff.as_field_metadata())
                            .with_name(buff.to_string()),
                    );
                }
            }
        }
        has_priority
    }

    /// Store zero intensity points as nulls in the intensity and coordinate domain
    pub fn null_zeros(mut self, null_zeros: bool) -> Self {
        self.null_zeros = null_zeros;
        self
    }

    /// Add a column to the data file holding the entity's time in addition to the index
    pub fn include_time(mut self, include_time: bool) -> Self {
        self.include_time = include_time;
        self
    }

    /// Inject zero null marking transform post-array schema inference because it complicates hash matching buffer names.
    ///
    /// This special case will be handled by [`ArrowArrayChunk`](crate::chunk_series::ArrowArrayChunk) which is sensitive
    /// to the schema itself. The point layout managed by [`PointBuffers`] doesn't check [`BufferName`] mapping as strictly
    /// and needs no special treatment.
    fn apply_null_zero_transform_modifier(&mut self) {
        if self.null_zeros {
            for f in self.array_fields.iter_mut() {
                if let Some(mut name) = BufferName::from_field(self.buffer_context, f.clone()) {
                    match self.buffer_context {
                        BufferContext::Spectrum => match name.array_type {
                            ArrayType::MZArray => {
                                if name.transform.is_none() {
                                    let new_name = name
                                        .clone()
                                        .with_transform(Some(BufferTransform::NullInterpolate));
                                    self.overrides.insert(name.clone(), new_name.clone());
                                    let to_replace: Vec<_> = self
                                        .overrides
                                        .iter()
                                        .filter_map(|(k, v)| {
                                            (k.array_type == ArrayType::MZArray)
                                                .then(|| (k.clone(), v.clone()))
                                        })
                                        .collect();
                                    for (k, v) in to_replace {
                                        self.overrides.insert(
                                            k.clone(),
                                            v.clone().with_transform(Some(
                                                BufferTransform::NullInterpolate,
                                            )),
                                        );
                                    }
                                    name = new_name;
                                }
                            }
                            ArrayType::IntensityArray => {
                                if name.transform.is_none() {
                                    let new_name = name
                                        .clone()
                                        .with_transform(Some(BufferTransform::NullZero));
                                    self.overrides.insert(name.clone(), new_name.clone());
                                    let to_replace: Vec<_> = self
                                        .overrides
                                        .iter()
                                        .filter_map(|(k, v)| {
                                            (k.array_type == ArrayType::IntensityArray)
                                                .then(|| (k.clone(), v.clone()))
                                        })
                                        .collect();
                                    for (k, v) in to_replace {
                                        self.overrides.insert(
                                            k.clone(),
                                            v.clone()
                                                .with_transform(Some(BufferTransform::NullZero)),
                                        );
                                    }
                                    name = new_name;
                                }
                            }
                            _ => {}
                        },
                        BufferContext::Chromatogram => match name.array_type {
                            _ => {}
                        },
                    }
                    *f = name.update_field(f.clone());
                }
            }
        }
    }

    /// Build a [`ChunkBuffer`], configuring the underlying schema
    pub fn build_chunked(
        mut self,
        schema: SchemaRef,
        buffer_context: BufferContext,
        mask_zero_intensity_runs: bool,
    ) -> ChunkBuffers {
        if self.include_time {
            self = self.add_time_field(buffer_context);
        }
        self.canonicalize_field_order();
        let primaries = self.mark_primary_arrays();
        self.overrides.propagate_priorities(&primaries);
        let mut fields: Vec<FieldRef> = schema.fields().iter().cloned().collect();
        self.prefix = "chunk".to_string();
        if self.null_zeros {
            self.apply_null_zero_transform_modifier()
        };
        fields.push(Field::new(self.prefix.clone(), self.dtype(), true).into());
        let schema = Arc::new(Schema::new_with_metadata(
            fields.clone(),
            schema.metadata().clone(),
        ));
        let drop_zero_column = if mask_zero_intensity_runs {
            Some(
                fields
                    .iter()
                    .filter(|c| c.name().starts_with("_intensity"))
                    .map(|s| s.to_string())
                    .collect(),
            )
        } else {
            None
        };
        ChunkBuffers::new(
            self.array_fields.clone().into(),
            buffer_context,
            schema,
            self.prefix.clone(),
            Vec::new(),
            self.overrides.clone(),
            drop_zero_column,
            self.null_zeros,
            Vec::new(),
            self.include_time,
        )
    }

    fn add_time_field(mut self, buffer_context: BufferContext) -> Self {
        self = self.add_field(buffer_context.time_field());
        self
    }

    /// Build a [`PointBuffers`], configuring the underlying schema
    pub fn build(
        mut self,
        schema: SchemaRef,
        buffer_context: BufferContext,
        mask_zero_intensity_runs: bool,
    ) -> PointBuffers {
        if self.include_time {
            self = self.add_time_field(buffer_context);
        }
        self.canonicalize_field_order();
        let primaries = self.mark_primary_arrays();
        self.overrides.propagate_priorities(&primaries);
        let mut fields: Vec<FieldRef> = schema.fields().iter().cloned().collect();
        if self.null_zeros {
            self.apply_null_zero_transform_modifier()
        };
        fields.push(Field::new(self.prefix.clone(), self.dtype(), true).into());
        let mut buffers: HashMap<String, Vec<ArrayRef>> =
            HashMap::with_capacity(self.array_fields.len());
        let mut drop_zero_column = Vec::new();
        for f in self.array_fields.iter() {
            let name = f.name();
            if mask_zero_intensity_runs {
                if let Some(bufname) = BufferName::from_field(self.buffer_context, f.clone()) {
                    if matches!(bufname.array_type, ArrayType::IntensityArray) {
                        drop_zero_column.push(name.to_string());
                    }
                }
            }
            buffers.insert(name.to_string(), Vec::new());
        }
        let drop_zero_column = mask_zero_intensity_runs.then_some(drop_zero_column);
        PointBuffers {
            buffer_context,
            peak_array_fields: self.array_fields.clone().into(),
            schema: Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone())),
            prefix: self.prefix.clone(),
            array_chunks: buffers,
            overrides: self.overrides.clone(),
            drop_zero_column,
            is_profile_buffer: Vec::new(),
            null_zeros: self.null_zeros,
            include_time: self.include_time,
        }
    }
}
