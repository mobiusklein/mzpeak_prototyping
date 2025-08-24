use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    sync::Arc,
};

use arrow::{
    array::{ArrayRef, RecordBatch, StructArray, new_null_array},
    datatypes::{DataType, Field, FieldRef, Fields, Schema, SchemaRef},
};
use mzdata::spectrum::ArrayType;

use crate::{
    BufferContext, BufferName, ToMzPeakDataSeries,
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
    fn add<T: ToMzPeakDataSeries>(&mut self, spectrum_index: u64, peaks: &[T]);

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
        RecordBatch::try_new(schema, arrays).unwrap_or_else(|e| {
            panic!("Failed to convert arrays to record batch: {e:#?}");
        })
    }

    fn overrides(&self) -> &HashMap<BufferName, BufferName>;

    fn as_array_index(&self) -> ArrayIndex {
        let mut array_index: ArrayIndex =
            ArrayIndex::new(self.prefix().to_string(), HashMap::new());
        if let Ok(sub) = self.schema().field_with_name(&self.prefix()).cloned() {
            if let DataType::Struct(fields) = sub.data_type() {
                for f in fields.iter() {
                    if f.name() == BufferContext::Spectrum.index_field().name() || f.name() == BufferContext::Chromatogram.index_field().name() {
                        continue;
                    }
                    if let Some(buffer_name) =
                        BufferName::from_field(self.buffer_context(), f.clone())
                    {
                        let aie = ArrayIndexEntry::from_buffer_name(
                            self.prefix().to_string(),
                            buffer_name,
                            Some(&f)
                        );
                        array_index.insert(aie.array_type.clone(), aie);
                    } else {
                        if f.name().ends_with("_chunk_end")
                            || f.name().ends_with("_chunk_start")
                            || f.name() == "chunk_encoding"
                        {
                            continue;
                        }
                        log::warn!("Failed to construct metadata index for {f:#?}");
                    }
                }
            }
        }
        log::trace!("{} array indices: {}", self.buffer_context(), array_index.to_json());
        array_index
    }
}

#[derive(Debug)]
pub struct PointBuffers {
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

impl PointBuffers {
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

impl ArrayBufferWriter for PointBuffers {
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

    fn nullify_zero_intensity(&self) -> bool {
        self.null_zeros
    }

    fn drop_zero_intensity(&self) -> bool {
        self.drop_zero_column.is_some()
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
        self.chunks
            .drain(..)
            .zip(is_profile)
            .map(move |(batch, is_profile)| {
                let mut batch = RecordBatch::from(batch);
                if is_profile {
                    if let Some(cols) = drop_zero_columns.as_ref() {
                        for (i, _f) in schema
                            .fields()
                            .iter()
                            .enumerate()
                            .filter(|(_, f)| cols.contains(f.name()))
                        {
                            match drop_where_column_is_zero(&batch, i) {
                                Ok(b) => {
                                    batch = b;
                                }
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

    fn nullify_zero_intensity(&self) -> bool {
        self.null_zeros
    }

    fn drop_zero_intensity(&self) -> bool {
        self.drop_zero_column.is_some()
    }
}

#[derive(Debug)]
pub enum ArrayBufferWriterVariants {
    ChunkBuffers(ChunkBuffers),
    PointBuffers(PointBuffers),
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

    fn add<T: ToMzPeakDataSeries>(&mut self, spectrum_index: u64, peaks: &[T]) {
        match self {
            ArrayBufferWriterVariants::ChunkBuffers(chunk_buffers) => {
                chunk_buffers.add(spectrum_index, peaks)
            }
            ArrayBufferWriterVariants::PointBuffers(array_buffers) => {
                array_buffers.add(spectrum_index, peaks)
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

    fn overrides(&self) -> &HashMap<BufferName, BufferName> {
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
            self.array_fields.clone().into(),
            buffer_context,
            schema,
            self.prefix.clone(),
            Vec::new(),
            self.overrides.clone(),
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
    ) -> PointBuffers {
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
        }
    }
}
