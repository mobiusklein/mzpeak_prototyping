use std::{collections::HashMap, io, sync::Arc};

use arrow::{
    array::{
        Array, ArrayRef, AsArray, Float32Array, Float64Array, Int32Array, Int64Array,
        PrimitiveArray, RecordBatch, StructArray, UInt8Array, UInt64Array,
    },
    datatypes::{
        DataType, Field, Fields, Float32Type, Float64Type, Int8Type, Int32Type, Int64Type, Schema,
        UInt8Type, UInt32Type, UInt64Type,
    },
    error::ArrowError,
};
use parquet::{
    arrow::{
        ProjectionMask,
        arrow_reader::{
            ArrowPredicate, ArrowPredicateFn, ParquetRecordBatchReaderBuilder, RowFilter,
            RowSelection,
        },
    },
    file::{metadata::ParquetMetaData, reader::ChunkReader},
    schema::types::SchemaDescriptor,
};

use mzdata::{
    prelude::*,
    spectrum::{ArrayType, BinaryArrayMap, DataArray, bindata::ArrayRetrievalError},
};
use mzpeaks::coordinate::SimpleInterval;

use crate::{
    BufferContext, BufferName,
    chunk_series::{
        BufferTransformDecoder, ChunkingStrategy, DELTA_ENCODE, NO_COMPRESSION, NUMPRESS_LINEAR,
    },
    filter::RegressionDeltaModel,
    peak_series::{ArrayIndex, ArrayIndexEntry, BufferFormat, data_array_to_arrow_array},
    reader::{
        ReaderMetadata,
        index::{BasicChunkQueryIndex, PageQuery, RangeIndex, SpanDynNumeric},
        point::binary_search_arrow_index,
        utils::MaskSet,
        visitor::AnyCURIEArray,
    },
};

use super::utils::OneCache;

pub(crate) struct DataChunkCache {
    pub(crate) row_group: RecordBatch,
    pub(crate) spectrum_index_range: SimpleInterval<u64>,
    pub(crate) spectrum_array_indices: Arc<ArrayIndex>,
    pub(crate) last_query_index: Option<u64>,
    pub(crate) last_query_span: Option<(usize, usize)>,
}

impl DataChunkCache {
    pub(crate) fn new(
        row_group: RecordBatch,
        spectrum_index_range: SimpleInterval<u64>,
        spectrum_array_indices: Arc<ArrayIndex>,
        last_query_index: Option<u64>,
        last_query_span: Option<(usize, usize)>,
    ) -> Self {
        Self {
            row_group,
            spectrum_index_range,
            spectrum_array_indices,
            last_query_index,
            last_query_span,
        }
    }

    pub(crate) fn find_span_for_query(&self, index: u64) -> (Option<usize>, Option<usize>) {
        let mut begin_hint = None;
        let mut end_hint = None;
        if let Some(last_query_index) = self.last_query_index {
            if last_query_index < index {
                begin_hint = Some(self.last_query_span.unwrap().1);
            } else if last_query_index > index {
                end_hint = Some(self.last_query_span.unwrap().1)
            } else if last_query_index == index {
                let (a, b) = self.last_query_span.unwrap();
                begin_hint = Some(a);
                end_hint = Some(b);
            }
        }

        let chunks = self.row_group.column(0).as_struct();
        let indices: &UInt64Array = chunks
            .column_by_name("spectrum_index")
            .unwrap()
            .as_any()
            .downcast_ref()
            .unwrap();
        let bounds = binary_search_arrow_index(indices, index, begin_hint, end_hint);

        let mut start = None;
        let mut end = None;

        if let Some((bstart, bend)) = bounds {
            let at = indices.value(bstart);
            assert_eq!(at, index);
            let at = indices.value(bend - 1);
            assert_eq!(at, index);
            start = Some(bstart);
            end = Some(bend);
        }
        (start, end)
    }

    pub(crate) fn slice_to_arrays_of(
        &mut self,
        index: u64,
        mz_delta_model: Option<&RegressionDeltaModel<f64>>,
    ) -> io::Result<BinaryArrayMap> {
        let (start, end) = self.find_span_for_query(index);
        if !(start.is_some() && end.is_some()) {
            panic!("Could not find start and end in binary search");
        }

        let chunks = self.row_group.column(0).as_struct();

        let chunks = match (start, end) {
            (Some(start), Some(end)) => {
                let len = end - start;
                self.last_query_span = Some((start, end));
                self.last_query_index = Some(index);
                chunks.slice(start, len)
            }
            (Some(start), None) => {
                self.last_query_span = Some((start, start + 1));
                self.last_query_index = Some(index);
                chunks.slice(start, 1)
            }
            _ => {
                let mut bin_map = HashMap::new();
                for v in self.spectrum_array_indices.iter() {
                    bin_map.insert(&v.name, v.as_buffer_name().as_data_array(0));
                }
                let mut out = BinaryArrayMap::new();
                for v in bin_map.into_values() {
                    out.add(v);
                }
                return Ok(out);
            }
        };

        let subschema = Arc::new(Schema::new(vec![Arc::new(Field::new(
            "chunk",
            DataType::Struct(chunks.fields().clone()),
            false,
        ))]));

        let batch = RecordBatch::from(StructArray::new(
            subschema.fields().clone(),
            vec![Arc::new(chunks)],
            None,
        ));

        let out = ChunkDataReader::<std::fs::File>::decode_chunks(
            [batch].into_iter(),
            &self.spectrum_array_indices,
            mz_delta_model,
        )?;

        Ok(out)
    }
}

trait ChunkQuerySource {
    fn buffer_context(&self) -> BufferContext;

    fn metadata(&self) -> &ParquetMetaData;

    fn parquet_schema(&self) -> Arc<SchemaDescriptor> {
        self.metadata().file_metadata().schema_descr_ptr()
    }

    fn array_prefix<'a>() -> &'a str {
        BufferFormat::Chunk.prefix()
    }

    fn prepare_predicate_for_index(&self, index_range: MaskSet) -> Box<dyn ArrowPredicate> {
        let sidx = format!(
            "{}.{}",
            Self::array_prefix(),
            self.buffer_context().index_name()
        );
        let proj = ProjectionMask::columns(&self.parquet_schema(), [sidx.as_str()]);

        let predicate = ArrowPredicateFn::new(proj, move |batch| {
            let root = batch.column(0).as_struct();
            let spectrum_index = root.column(0);
            Ok(index_range.contains_dy(spectrum_index))
        });
        Box::new(predicate)
    }

    fn find_chunk_value_arrays_by_buffer_format<'a>(
        &self,
        array_indices: &'a ArrayIndex,
    ) -> Option<(
        &'a ArrayIndexEntry,
        &'a ArrayIndexEntry,
        &'a ArrayIndexEntry,
        Vec<&'a ArrayIndexEntry>,
    )> {
        let mut start_entry = None;
        let mut end_entry = None;
        let mut values_entry = None;
        let mut additional_entries = Vec::new();
        let cache = BufferNameCache::new(array_indices);

        for entry in array_indices.iter() {
            match entry.buffer_format {
                BufferFormat::Point => {}
                BufferFormat::ChunkBoundsStart => start_entry = Some(entry),
                BufferFormat::ChunkBoundsEnd => end_entry = Some(entry),
                BufferFormat::Chunk => values_entry = Some(entry),
                BufferFormat::ChunkEncoding => {}
                BufferFormat::ChunkSecondary => {}
                BufferFormat::ChunkTransform => {
                    if cache.is_chunk_transform_for_main_chunk(entry.field_name(), &entry.as_buffer_name()) {
                        additional_entries.push(entry);
                    }
                }
            }
        }

        match (start_entry, end_entry, values_entry) {
            (Some(start), Some(end), Some(values)) => Some((start, end, values, additional_entries)),
            (_, _, _) => None,
        }
    }

    fn find_chunk_values_by_name<'a>(
        &self,
        array_indices: &'a ArrayIndex,
    ) -> Option<&'a ArrayIndexEntry> {
        array_indices
            .iter()
            .find(|e| e.path.ends_with("_chunk_values"))
    }

    fn prepare_predicate_for_chunk_range(
        &self,
        query_range: SimpleInterval<f64>,
        array_indices: &ArrayIndex,
    ) -> Option<Box<dyn ArrowPredicate>> {
        let mut fields = None;
        if let Some((start, end, _values, _)) =
            self.find_chunk_value_arrays_by_buffer_format(array_indices)
        {
            let start_name = start.path.clone();
            let end_name = end.path.clone();
            fields = Some([start_name, end_name])
        } else if let Some(e) = self.find_chunk_values_by_name(array_indices) {
            let prefix = e.path.as_str();
            let prefix = if prefix.ends_with("_chunk_values") {
                prefix.replace("_chunk_values", "")
            } else {
                prefix.to_string()
            };
            fields = Some([
                format!("{prefix}_chunk_start"),
                format!("{prefix}_chunk_end"),
            ]);
        }

        if let Some(fields) = fields {
            let proj =
                ProjectionMask::columns(&self.parquet_schema(), fields.iter().map(|s| s.as_str()));
            let predicate = ArrowPredicateFn::new(proj, move |batch| {
                let root = batch.column(0).as_struct();
                let start_array = root.column(0);
                let end_array = root.column(1);
                let it2 = query_range.overlaps_dy(start_array, end_array);
                Ok(it2)
            });
            Some(Box::new(predicate))
        } else {
            None
        }
    }

    fn prepare_scan(
        &self,
        index_range: MaskSet,
        query_range: Option<SimpleInterval<f64>>,
        array_indices: &ArrayIndex,
        query_indices: &impl BasicChunkQueryIndex,
    ) -> (RowSelection, Vec<usize>, RowFilter) {
        let mut rows = query_indices
            .primary_data_index()
            .row_selection_overlaps(&index_range);

        let PageQuery {
            row_group_indices,
            pages: _,
        } = query_indices.query_pages_overlaps(&index_range);

        let mut up_to_first_row = 0;
        if !row_group_indices.is_empty() {
            let meta = self.metadata();
            for i in 0..row_group_indices[0] {
                let rg = meta.row_group(i);
                up_to_first_row += rg.num_rows();
            }
        }

        if let Some(query_range) = query_range.as_ref() {
            let chunk_range_idx = RangeIndex::new(
                &query_indices.chunk_start_index(),
                &query_indices.chunk_end_index(),
            );
            rows = rows.intersection(&chunk_range_idx.row_selection_overlaps(query_range));
        }

        rows.split_off(up_to_first_row as usize);

        let sidx = format!(
            "{}.{}",
            Self::array_prefix(),
            self.buffer_context().index_name()
        );

        let mut fields: Vec<String> = Vec::new();

        fields.push(sidx);

        if let Some((start, end, values, additional)) =
            self.find_chunk_value_arrays_by_buffer_format(array_indices)
        {
            let values_name = values.path.clone();
            let start_name = start.path.clone();
            let end_name = end.path.clone();
            fields.extend([values_name, start_name, end_name]);
            for f in additional {
                fields.push(f.path.clone());
            }
        } else if let Some(e) = self.find_chunk_values_by_name(array_indices) {
            let prefix = e.path.as_str();
            let prefix = if prefix.ends_with("_chunk_values") {
                prefix.replace("_chunk_values", "")
            } else {
                prefix.to_string()
            };
            fields.extend([
                e.path.clone(),
                format!("{prefix}_chunk_start"),
                format!("{prefix}_chunk_end"),
            ]);
        }

        for e in array_indices.get_all(&ArrayType::IntensityArray) {
            fields.push(e.path.to_string());
        }

        for v in array_indices.iter() {
            if v.is_ion_mobility() {
                fields.push(v.path.to_string());
                break;
            }
        }

        log::trace!("Executing query on {fields:?}");

        let mut predicates = vec![self.prepare_predicate_for_index(index_range)];

        if let Some(query_range) = query_range {
            predicates.extend(self.prepare_predicate_for_chunk_range(query_range, array_indices));
        }

        let predicate = RowFilter::new(predicates);
        (rows, row_group_indices, predicate)
    }

    fn prepare_chunks_of(
        &self,
        query: u64,
        query_indices: &impl BasicChunkQueryIndex,
        page_query: Option<PageQuery>,
    ) -> Option<(
        RowSelection,
        Vec<usize>,
        ProjectionMask,
        ArrowPredicateFn<
            impl FnMut(RecordBatch) -> Result<arrow::array::BooleanArray, ArrowError> + 'static,
        >,
    )> {
        let PageQuery {
            row_group_indices: rg_idx_acc,
            pages,
        } = page_query.unwrap_or_else(|| query_indices.query_pages(query));

        // Otherwise we must construct a more intricate read plan, first pruning rows and row groups
        // based upon the pages matched
        let first_row = if !pages.is_empty() {
            let mut rg_row_skip = 0;
            for i in 0..rg_idx_acc[0] {
                let rg = self.metadata().row_group(i);
                rg_row_skip += rg.num_rows();
            }
            rg_row_skip
        } else {
            return None;
        };

        let rows = query_indices
            .primary_data_index()
            .pages_to_row_selection(&pages, first_row);

        let sidx = format!(
            "{}.{}",
            Self::array_prefix(),
            self.buffer_context().index_name()
        );

        let predicate_mask = ProjectionMask::columns(&self.parquet_schema(), [sidx.as_str()]);

        let predicate = ArrowPredicateFn::new(predicate_mask, move |batch| {
            let entity_index: &UInt64Array = batch
                .column(0)
                .as_struct()
                .column(0)
                .as_any()
                .downcast_ref()
                .unwrap();

            let it = entity_index
                .iter()
                .map(|val| val.is_some_and(|val| val == query));

            Ok(it.map(Some).collect())
        });

        let proj = ProjectionMask::columns(&self.parquet_schema(), [Self::array_prefix()]);
        Some((rows, rg_idx_acc, proj, predicate))
    }

    fn prepare_cache_block_query(
        &self,
        index_range: SimpleInterval<u64>,
        query_indices: &impl BasicChunkQueryIndex,
    ) -> (
        RowSelection,
        ArrowPredicateFn<
            impl FnMut(RecordBatch) -> Result<arrow::array::BooleanArray, ArrowError> + 'static,
        >,
    ) {
        let rows = query_indices
            .primary_data_index()
            .row_selection_overlaps(&index_range);

        let sidx = format!(
            "{}.{}",
            Self::array_prefix(),
            self.buffer_context().index_name()
        );

        let proj = ProjectionMask::columns(&self.parquet_schema(), vec![sidx.as_str()]);
        let predicate_mask = proj;

        let predicate = ArrowPredicateFn::new(predicate_mask, move |batch| {
            let root = batch.column(0).as_struct();
            let spectrum_index: &UInt64Array = root.column(0).as_any().downcast_ref().unwrap();

            let it = spectrum_index
                .iter()
                .map(|v| v.map(|v| index_range.contains(&v)));

            Ok(it.collect())
        });
        (rows, predicate)
    }
}

fn coerce_bounds_array(arr: &ArrayRef) -> Vec<f64> {
    if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
        arr.values().to_vec()
    } else if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
        arr.values().into_iter().map(|v| *v as f64).collect()
    } else {
        unimplemented!("Bounds array of type {:?} not implemented", arr.data_type());
    }
}


struct BufferNameCache<'a>  {
    buffer_name_cache: HashMap<String, Option<Arc<BufferName>>>,
    array_indices: &'a ArrayIndex,
}

impl<'a> BufferNameCache<'a> {
    fn new(array_indices: &'a ArrayIndex) -> Self {
        Self { buffer_name_cache: Default::default(), array_indices }
    }

    fn get(&mut self, field_name: &str) -> Option<Arc<BufferName>> {
        if let Some(v) = self.buffer_name_cache.get(field_name) {
            return v.clone()
        }
        let name = self
                .array_indices
                .iter()
                .find(|col| col.path.split(".").last().unwrap() == field_name)
                .map(|v| v.as_buffer_name()).map(Arc::new);
        self.buffer_name_cache.insert(field_name.to_string(), name.clone());
        name
    }

    fn is_chunk_transform_for_main_chunk(&self, field_name: &str, buffer_name: &BufferName) -> bool {
        self.get_transform(field_name, buffer_name).is_some_and(|f| matches!(f.buffer_format, BufferFormat::Chunk))
    }

    fn get_transform(&self, field_name: &str, buffer_name: &BufferName) -> Option<Arc<BufferName>> {
        for array_index_entry in self.array_indices.iter() {
            if matches!(array_index_entry.buffer_format, BufferFormat::Chunk | BufferFormat::ChunkSecondary) {
                let qname = array_index_entry.as_buffer_name();
                if qname.array_type == buffer_name.array_type
                    && qname.dtype == buffer_name.dtype
                    && qname.unit == buffer_name.unit
                    && qname.data_processing_id == buffer_name.data_processing_id {
                        let qname = Arc::new(qname);
                        log::trace!("Mapping {field_name} to parent {qname}");
                        return Some(qname)
                }
            }
        }
        return None
    }
}


struct ChunkDecoder<'a> {
    buffers: HashMap<Arc<BufferName>, Vec<ArrayRef>>,
    main_axis_buffers: Vec<(Arc<BufferName>, ArrayRef)>,
    main_axis: Option<DataArray>,
    bin_map: BinaryArrayMap,
    array_indices: &'a ArrayIndex,
    delta_model: Option<&'a RegressionDeltaModel<f64>>,
    buffer_name_cache: BufferNameCache<'a>
}

impl<'a> ChunkDecoder<'a> {
    fn new(
        array_indices: &'a ArrayIndex,
        delta_model: Option<&'a RegressionDeltaModel<f64>>,
    ) -> Self {
        Self {
            buffers: Default::default(),
            main_axis_buffers: Default::default(),
            main_axis: None,
            bin_map: Default::default(),
            array_indices,
            delta_model,
            buffer_name_cache: BufferNameCache::new(array_indices),
        }
    }

    fn compile_buffers(mut self) -> Result<BinaryArrayMap, ArrayRetrievalError> {
        // If we never populated the main axis, exit early and return empty arrays.
        if self.main_axis.is_none() {
            for k in self.array_indices.iter() {
                self.bin_map.add(k.as_buffer_name().as_data_array(0));
            }
            return Ok(self.bin_map);
        }

        let main_axis = self.main_axis.unwrap();
        let n = main_axis.data_len()?;
        self.bin_map.add(main_axis);

        for (name, chunks) in self.buffers {
            let mut store = DataArray::from_name_type_size(
                &name.array_type,
                name.dtype,
                name.dtype.size_of() * n,
            );
            let decoder: Option<BufferTransformDecoder> = name.transform.try_into().ok();

            macro_rules! extend_array {
                ($buf:ident, $tp:ty) => {
                    if $buf.null_count() > 0 {
                        let buf: &$tp = $buf.as_primitive();
                        for val in buf.iter() {
                            store.push(val.unwrap_or_default()).unwrap();
                        }
                    } else {
                        let buf: &$tp = $buf.as_primitive();
                        store.extend(buf.values()).unwrap();
                    }
                };
            }

            for arr in chunks {
                if let Some(arr) = arr.as_list_opt::<i64>() {
                    if arr.is_empty() {
                        continue;
                    }

                    // Decode the list if a decoding transform is required, lazily
                    let mut arr_iter = arr
                        .iter()
                        .flatten()
                        .map(|arr| {
                            decoder
                                .as_ref()
                                .map(|decoder| decoder.decode(&name, &arr))
                                .unwrap_or(arr)
                        })
                        .peekable();

                    // Use the first array post-decode here to infer the "real" data type
                    if let Some(first) = arr_iter.peek() {
                        match first.data_type() {
                            DataType::Float32 => {
                                for arr in arr_iter {
                                    extend_array!(arr, Float32Array);
                                }
                            }
                            DataType::Float64 => {
                                for arr in arr_iter {
                                    extend_array!(arr, Float64Array);
                                }
                            }
                            DataType::Int32 => {
                                for arr in arr_iter {
                                    extend_array!(arr, Int32Array);
                                }
                            }
                            DataType::Int64 => {
                                for arr in arr_iter {
                                    extend_array!(arr, Int64Array);
                                }
                            }
                            DataType::UInt8 => {
                                for arr in arr_iter {
                                    extend_array!(arr, UInt8Array);
                                }
                            }
                            DataType::LargeUtf8 => {
                                todo!("String arrays not supported yet")
                            }
                            DataType::Utf8 => {}
                            dt => {
                                panic!("Unsupported array type: {dt:?}");
                            }
                        }
                    }
                }
            }
            if store.raw_len() == 0 {
                continue;
            }
            self.bin_map.add(store);
        }
        Ok(self.bin_map)
    }

    fn make_axis_sequence(&mut self) -> Vec<Vec<(Arc<BufferName>, Option<Arc<dyn Array>>)>> {
        let mut rows: Vec<Vec<(Arc<BufferName>, Option<Arc<dyn Array>>)>> = Vec::new();
        let n_rows = if let Some((_, block)) = self.main_axis_buffers.first() {
            block.len()
        } else {
            return rows;
        };
        rows.resize(n_rows, Vec::new());
        for (name, view) in self.main_axis_buffers.drain(..) {
            let view_rows = view.as_list::<i64>();
            for (i, row) in view_rows.iter().enumerate() {
                rows[i].push((name.clone(), row));
            }
        }
        return rows;
    }

    fn decode_batch(&mut self, batch: RecordBatch) {
        let root = batch.column(0).as_struct();
        let mut chunk_encodings: Vec<_> = Vec::new();
        let mut chunk_starts = Vec::new();
        let mut chunk_ends = Vec::new();
        for (f, arr) in root.fields().iter().zip(root.columns()).skip(1) {
            let name = self.buffer_name_cache.get(f.name());
            match f.name().as_str() {
                "chunk_encoding" => {
                    chunk_encodings = AnyCURIEArray::try_from(arr).unwrap().to_vec();
                }
                s if s.ends_with("chunk_start") => {
                    chunk_starts = coerce_bounds_array(arr);
                }
                s if s.ends_with("chunk_end") => {
                    chunk_ends = coerce_bounds_array(arr);
                }
                _ => {
                    if let Some(name) = name {
                        match name.buffer_format {
                            BufferFormat::Chunk => {
                                log::trace!(
                                    "Storing {name} with {:?} and {} entries",
                                    arr.data_type(),
                                    arr.len()
                                );
                                self.main_axis_buffers.push((name, arr.clone()));
                            }
                            BufferFormat::ChunkSecondary | BufferFormat::Point => {
                                log::trace!(
                                    "Storing (secondary) {name} with {:?} and {} entries",
                                    arr.data_type(),
                                    arr.len()
                                );
                                self.buffers.entry(name).or_default().push(arr.clone());
                            }
                            BufferFormat::ChunkBoundsStart => {
                                chunk_starts = coerce_bounds_array(arr);
                            }
                            BufferFormat::ChunkBoundsEnd => {
                                chunk_ends = coerce_bounds_array(arr);
                            },
                            BufferFormat::ChunkEncoding => {
                                chunk_encodings = AnyCURIEArray::try_from(arr).unwrap().to_vec()
                            }
                            BufferFormat::ChunkTransform => {
                                if self.buffer_name_cache.get_transform(f.name(), &name).is_some_and(|qname| matches!(qname.buffer_format, BufferFormat::Chunk)) {
                                    self.main_axis_buffers.push((name, arr.clone()));
                                } else {
                                    self.buffers.entry(name).or_default().push(arr.clone());
                                }
                            },
                        }
                    } else {
                        log::warn!("{f:?} failed to map to a chunk buffer");
                    }
                }
            }
        }

        let chunk_iter = self.make_axis_sequence().into_iter().zip(
            chunk_encodings
                .iter()
                .copied()
                .zip(chunk_starts)
                .zip(chunk_ends)
        );

        // For each chunk row
        for (row, ((encoding, start), end)) in chunk_iter {
            // For each possible main axis array (e.g. BufferFormat::Chunked)
            for (name, chunk_vals) in row {
                let mut did_decode = false;
                if let Some(chunk_vals) = chunk_vals {
                    if self.main_axis.is_none() {
                        self.main_axis =
                            Some(DataArray::from_name_and_type(&name.array_type, name.dtype))
                    }
                    if !chunk_vals.is_empty() {
                        did_decode = true;
                        match encoding {
                            NO_COMPRESSION => {
                                (ChunkingStrategy::Basic { chunk_size: 50.0 })
                                    .decode_arrow(
                                        &chunk_vals,
                                        start as f64,
                                        end as f64,
                                        self.main_axis.as_mut().unwrap(),
                                        self.delta_model,
                                    );
                            }
                            DELTA_ENCODE => {
                                (ChunkingStrategy::Delta { chunk_size: 50.0 })
                                    .decode_arrow(
                                        &chunk_vals,
                                        start as f64,
                                        end as f64,
                                        self.main_axis.as_mut().unwrap(),
                                        self.delta_model,
                                    );
                            }
                            NUMPRESS_LINEAR => {
                                (ChunkingStrategy::NumpressLinear { chunk_size: 50.0 })
                                        .decode_arrow(
                                            &chunk_vals,
                                            start as f64,
                                            end as f64,
                                            self.main_axis.as_mut().unwrap(),
                                            self.delta_model,
                                        );
                            }
                            _ => {
                                unimplemented!("{encoding}")
                            }
                        }
                    }
                }
                if !did_decode {
                    match encoding {
                        NO_COMPRESSION => {
                            (ChunkingStrategy::Basic { chunk_size: 50.0 })
                                .decode_arrow(
                                    &arrow::array::new_empty_array(&DataType::Float64),
                                    start as f64,
                                    end as f64,
                                    self.main_axis.as_mut().unwrap(),
                                    self.delta_model,
                                );
                        }
                        DELTA_ENCODE => {
                            (ChunkingStrategy::Delta { chunk_size: 50.0 })
                                .decode_arrow(
                                    &arrow::array::new_empty_array(&DataType::Float64),
                                    start as f64,
                                    end as f64,
                                    self.main_axis.as_mut().unwrap(),
                                    self.delta_model,
                                );
                        }
                        NUMPRESS_LINEAR => {
                            // This chunk is never empty if it is valid
                        }
                        _ => {
                            unimplemented!("{encoding}")
                        }
                    }
                }
            }
        }
    }
}

struct ChunkScanDecoder<'a> {
    buffer_context: BufferContext,
    buffers: HashMap<Arc<BufferName>, Vec<ArrayRef>>,
    main_axis_buffers: Vec<(Arc<BufferName>, ArrayRef)>,
    main_axis: Option<DataArray>,
    metadata: &'a ReaderMetadata,
    query_range: Option<SimpleInterval<f64>>,
    buffer_name_cache: BufferNameCache<'a>
}

impl<'a> ChunkScanDecoder<'a> {
    fn new(
        buffer_context: BufferContext,
        metadata: &'a ReaderMetadata,
        query_range: Option<SimpleInterval<f64>>,
        array_indices: &'a ArrayIndex,
    ) -> Self {
        Self {
            buffer_context,
            buffers: Default::default(),
            main_axis_buffers: Default::default(),
            main_axis: None,
            metadata,
            query_range,
            buffer_name_cache: BufferNameCache::new(array_indices),
        }
    }

    fn clear(&mut self) {
        self.buffers.clear();
        self.main_axis_buffers.clear();
        self.main_axis = None;
    }

    fn make_axis_sequence(&mut self) -> Vec<Vec<(Arc<BufferName>, Option<Arc<dyn Array>>)>> {
        let mut rows: Vec<Vec<(Arc<BufferName>, Option<Arc<dyn Array>>)>> = Vec::new();
        let n_rows = if let Some((_, block)) = self.main_axis_buffers.first() {
            block.len()
        } else {
            return rows;
        };
        rows.resize(n_rows, Vec::new());
        for (name, view) in self.main_axis_buffers.drain(..) {
            let view_rows = view.as_list::<i64>();
            for (i, row) in view_rows.iter().enumerate() {
                rows[i].push((name.clone(), row));
            }
        }
        return rows;
    }

    fn decode_batch(&mut self, batch: RecordBatch) -> Result<RecordBatch, ArrowError> {
        let root = batch.column(0).as_struct();
        let mut chunk_encodings: Vec<_> = Vec::new();
        let mut chunk_starts = None;
        let mut chunk_ends = None;
        let mut delta_model_cache = OneCache::default();

        let entity_index = root
            .column(0)
            .as_primitive::<UInt64Type>()
            .values()
            .to_vec();

        for (f, arr) in root.fields().iter().zip(root.columns()).skip(1) {
            let name = self.buffer_name_cache.get(f.name());

            match f.name().as_str() {
                "chunk_encoding" => {
                    chunk_encodings = AnyCURIEArray::try_from(arr).unwrap().to_vec();
                }
                s if s.ends_with("chunk_start") => {
                    chunk_starts = Some(coerce_bounds_array(arr));
                }
                s if s.ends_with("chunk_end") => {
                    chunk_ends = Some(coerce_bounds_array(arr));
                }
                _ => {
                    if let Some(name) = name {
                        match name.buffer_format {
                            BufferFormat::Chunk => {
                                log::trace!(
                                    "Storing {name} with {:?} and {} entries",
                                    arr.data_type(),
                                    arr.len(),
                                );
                                self.main_axis_buffers.push((name, arr.clone()));
                            }
                            BufferFormat::ChunkSecondary | BufferFormat::Point => {
                                self.buffers.entry(name).or_default().push(arr.clone());
                            }
                            BufferFormat::ChunkBoundsStart => {
                                chunk_starts = Some(coerce_bounds_array(arr));
                            }
                            BufferFormat::ChunkBoundsEnd => {
                                chunk_ends = Some(coerce_bounds_array(arr));
                            }
                            BufferFormat::ChunkEncoding => {
                                chunk_encodings = AnyCURIEArray::try_from(arr).unwrap().to_vec()
                            }
                            BufferFormat::ChunkTransform => {
                                if self.buffer_name_cache.is_chunk_transform_for_main_chunk(f.name(), &name) {
                                    self.main_axis_buffers.push((name, arr.clone()));
                                } else {
                                    self.buffers.entry(name).or_default().push(arr.clone());
                                }
                            },
                        }
                    } else {
                        log::warn!("{f:?} failed to map to a chunk buffer");
                    }
                }
            }
        }

        // Accumulate the per-point spectrum association
        let mut entity_idx_acc: Vec<u64> = Vec::with_capacity(entity_index.len());

        let chunk_iter = self.make_axis_sequence().into_iter().zip(
            chunk_encodings
                .iter()
                .copied()
                .zip(chunk_starts.unwrap())
                .zip(chunk_ends.unwrap())
                .zip(entity_index),
        );

        // For each chunk row
        for (rows, (((encoding, start), end), entity_index)) in chunk_iter {
            // For each possible main axis array (e.g. BufferFormat::Chunked)
            let mut did_decode = false;
            for (name, chunk_vals) in rows {
                if self.main_axis.is_none() {
                    self.main_axis =
                        Some(DataArray::from_name_and_type(&name.array_type, name.dtype))
                }
                if let Some(chunk_vals) = chunk_vals {
                    if !chunk_vals.is_empty() {
                        did_decode = true;
                        match encoding {
                            NO_COMPRESSION => {
                                let delta_model = delta_model_cache.get(entity_index, || {
                                    self.metadata.model_deltas_for(entity_index as usize)
                                });

                                let n_points_added = (ChunkingStrategy::Basic { chunk_size: 50.0 })
                                    .decode_arrow(
                                        &chunk_vals,
                                        start as f64,
                                        end as f64,
                                        self.main_axis.as_mut().unwrap(),
                                        delta_model.as_ref(),
                                    );
                                entity_idx_acc
                                    .extend(std::iter::repeat_n(entity_index, n_points_added));
                            }
                            DELTA_ENCODE => {
                                let delta_model = delta_model_cache.get(entity_index, || {
                                    self.metadata.model_deltas_for(entity_index as usize)
                                });

                                let n_points_added = (ChunkingStrategy::Delta { chunk_size: 50.0 })
                                    .decode_arrow(
                                        &chunk_vals,
                                        start as f64,
                                        end as f64,
                                        self.main_axis.as_mut().unwrap(),
                                        delta_model.as_ref(),
                                    );
                                entity_idx_acc
                                    .extend(std::iter::repeat_n(entity_index, n_points_added));
                            }
                            NUMPRESS_LINEAR => {
                                let delta_model = delta_model_cache.get(entity_index, || {
                                    self.metadata.model_deltas_for(entity_index as usize)
                                });
                                let n_points_added =
                                    (ChunkingStrategy::NumpressLinear { chunk_size: 50.0 })
                                        .decode_arrow(
                                            &chunk_vals,
                                            start as f64,
                                            end as f64,
                                            self.main_axis.as_mut().unwrap(),
                                            delta_model.as_ref(),
                                        );
                                entity_idx_acc
                                    .extend(std::iter::repeat_n(entity_index, n_points_added));
                            }
                            _ => {
                                unimplemented!("{encoding}")
                            }
                        }
                    }
                }
            }
            if !did_decode {
                    match encoding {
                        NO_COMPRESSION => {
                            (ChunkingStrategy::Basic { chunk_size: 50.0 })
                                .decode_arrow(
                                    &arrow::array::new_empty_array(&DataType::Float64),
                                    start as f64,
                                    end as f64,
                                    self.main_axis.as_mut().unwrap(),
                                    None,
                                );
                        }
                        DELTA_ENCODE => {
                            (ChunkingStrategy::Delta { chunk_size: 50.0 })
                                .decode_arrow(
                                    &arrow::array::new_empty_array(&DataType::Float64),
                                    start as f64,
                                    end as f64,
                                    self.main_axis.as_mut().unwrap(),
                                    None,
                                );
                        }
                        NUMPRESS_LINEAR => {
                            // This chunk is never empty if it is valid
                        }
                        _ => {
                            unimplemented!("{encoding}")
                        }
                    }
                }
        }

        // Reuse the same API that builds into a [`DataArray`] incrementally
        let axis = self.main_axis.take().unwrap();
        let buffer_name = BufferName::from_data_array(self.buffer_context, &axis);
        let axis = data_array_to_arrow_array(&buffer_name, &axis).unwrap();

        let mut fields = Vec::with_capacity(self.buffers.len() + 1);
        fields.push(buffer_name.context.index_field());
        fields.push(buffer_name.to_field());

        let mut arrays = Vec::with_capacity(self.buffers.len() + 1);
        arrays.push(Arc::new(UInt64Array::from(entity_idx_acc)) as ArrayRef);
        arrays.push(axis);

        for (name, chunks) in self.buffers.drain() {
            let decoder: Option<BufferTransformDecoder> = name.transform.try_into().ok();
            let chunks = match decoder {
                Some(decoder) => {
                    let chunks: Vec<ArrayRef> = chunks
                        .iter()
                        .flat_map(|a| {
                            a.as_list::<i64>()
                                .iter()
                                .map(|b| decoder.decode(&name, b.as_ref().unwrap()))
                        })
                        .collect();
                    let chunks: Vec<&dyn Array> = chunks.iter().map(|a| a as &dyn Array).collect();
                    let chunks = arrow::compute::concat(&chunks)?;
                    chunks
                }
                None => {
                    let chunks: Vec<&ArrayRef> = chunks
                        .iter()
                        .map(|a| a.as_ref().as_list::<i64>().values())
                        .collect();

                    macro_rules! fill_null {
                        ($arr:ident, $p:ty, $out:expr) => {
                            if let Some(arr) = $arr.as_primitive_opt::<$p>() {
                                let vals = arr.values().clone();
                                $out.push(
                                    Arc::new(PrimitiveArray::<$p>::new(vals, None)) as ArrayRef
                                );
                                true
                            } else {
                                false
                            }
                        };
                    }

                    let had_nulls = chunks.iter().any(|c| c.null_count() > 0);
                    log::trace!("Found nulls in {name:?}");
                    if had_nulls {
                        let mut chunks_out: Vec<Arc<dyn Array>> = Vec::with_capacity(chunks.len());
                        for chunk in chunks.iter() {
                            if chunk.null_count() == 0 {
                                chunks_out.push((*chunk).clone());
                                continue;
                            }
                            if fill_null!(chunk, Int64Type, &mut chunks_out) {
                            } else if fill_null!(chunk, UInt64Type, &mut chunks_out) {
                            } else if fill_null!(chunk, Float64Type, &mut chunks_out) {
                            } else if fill_null!(chunk, Int32Type, &mut chunks_out) {
                            } else if fill_null!(chunk, UInt32Type, &mut chunks_out) {
                            } else if fill_null!(chunk, Float32Type, &mut chunks_out) {
                            } else if fill_null!(chunk, Int8Type, &mut chunks_out) {
                            } else if fill_null!(chunk, UInt8Type, &mut chunks_out) {
                            } else {
                                chunks_out.push((*chunk).clone());
                            }
                        }
                        let chunks: Vec<_> = chunks_out.iter().map(|v| v.as_ref()).collect();
                        arrow::compute::concat(&chunks)?
                    } else {
                        let chunks: Vec<_> = chunks.iter().map(|v| v.as_ref()).collect();
                        arrow::compute::concat(&chunks)?
                    }
                }
            };
            arrays.push(chunks);
            fields.push(name.to_field());
        }

        let fields: Fields = fields.into();

        let mut batch: ArrayRef = Arc::new(StructArray::new(fields.clone(), arrays, None));

        if let Some(query_range) = self.query_range.as_ref() {
            let v = batch.as_struct().column(1);
            let mask = query_range.contains_dy(v);
            batch = arrow::compute::filter(&batch, &mask)?;
        }

        let dt = DataType::Struct(fields.clone());
        let batch = StructArray::new(
            vec![Arc::new(Field::new("chunk", dt, false))].into(),
            vec![batch],
            None,
        );
        self.clear();
        let batch = RecordBatch::from(batch);
        Ok(batch)
    }
}

#[derive(Debug)]
pub struct ChunkDataReader<T: ChunkReader + 'static> {
    builder: ParquetRecordBatchReaderBuilder<T>,
    buffer_context: BufferContext,
}

impl<T: ChunkReader + 'static> ChunkQuerySource for ChunkDataReader<T> {
    fn metadata(&self) -> &ParquetMetaData {
        self.builder.metadata()
    }

    fn buffer_context(&self) -> BufferContext {
        self.buffer_context
    }
}

impl<T: ChunkReader + 'static> ChunkDataReader<T> {
    pub fn new(builder: ParquetRecordBatchReaderBuilder<T>, buffer_context: BufferContext) -> Self {
        Self {
            builder,
            buffer_context,
        }
    }

    pub(crate) fn load_cache_block(
        self,
        index_range: SimpleInterval<u64>,
        metadata: &ReaderMetadata,
        query_indices: &impl BasicChunkQueryIndex,
    ) -> io::Result<DataChunkCache> {
        let (rows, predicate) = self.prepare_cache_block_query(index_range, query_indices);

        let schema = self.builder.schema().clone();
        let reader = self
            .builder
            .with_row_selection(rows)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .build()?;

        let mut batches = Vec::new();
        for bat in reader {
            batches.push(bat.map_err(io::Error::other)?);
        }

        let batch =
            arrow::compute::concat_batches(&schema, batches.iter()).map_err(io::Error::other)?;

        Ok(DataChunkCache::new(
            batch,
            index_range,
            metadata.spectrum_array_indices.clone(),
            None,
            None,
        ))
    }

    pub fn scan_chunks_for<'a>(
        self,
        index_range: MaskSet,
        query_range: Option<SimpleInterval<f64>>,
        metadata: &'a ReaderMetadata,
        array_indices: &'a ArrayIndex,
        query_indices: &impl BasicChunkQueryIndex,
    ) -> io::Result<impl Iterator<Item = Result<RecordBatch, ArrowError>>> {
        let buffer_context = self.buffer_context();
        let (rows, row_group_indices, predicate) =
            self.prepare_scan(index_range, query_range, array_indices, query_indices);

        let reader = self
            .builder
            .with_row_groups(row_group_indices)
            .with_row_selection(rows)
            .with_row_filter(predicate)
            .with_batch_size(4096)
            .build()?;

        let mut decoder =
            ChunkScanDecoder::new(buffer_context, metadata, query_range, array_indices);

        let it = reader.map(move |batch| -> Result<RecordBatch, ArrowError> {
            batch.and_then(|batch| decoder.decode_batch(batch))
        });

        Ok(Box::new(it))
    }

    pub fn decode_chunks<I: Iterator<Item = RecordBatch>>(
        reader: I,
        array_indices: &ArrayIndex,
        delta_model: Option<&RegressionDeltaModel<f64>>,
    ) -> io::Result<BinaryArrayMap> {
        let mut decoder = ChunkDecoder::new(array_indices, delta_model);
        for batch in reader {
            decoder.decode_batch(batch);
        }
        let bin_map = decoder.compile_buffers()?;
        Ok(bin_map)
    }

    pub fn read_chunks_for(
        self,
        query: u64,
        query_indices: &impl BasicChunkQueryIndex,
        array_indices: &ArrayIndex,
        delta_model: Option<&RegressionDeltaModel<f64>>,
        page_query: Option<PageQuery>,
    ) -> io::Result<BinaryArrayMap> {
        if let Some((rows, rg_idx_acc, proj, predicate)) =
            self.prepare_chunks_of(query, query_indices, page_query)
        {
            log::trace!("{query} @ chunk spread across row groups {rg_idx_acc:?}");
            let reader = self
                .builder
                .with_row_groups(rg_idx_acc)
                .with_row_selection(rows)
                .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
                .with_projection(proj)
                .build()?;

            Self::decode_chunks(reader.flatten(), array_indices, delta_model)
        } else {
            let mut bin_map = BinaryArrayMap::new();
            for k in array_indices.iter() {
                bin_map.add(k.as_buffer_name().as_data_array(0));
            }
            Ok(bin_map)
        }
    }
}

#[cfg(feature = "async")]
mod async_impl {
    use super::*;
    use futures::{StreamExt, stream::BoxStream};
    use parquet::arrow::{ParquetRecordBatchStreamBuilder, async_reader::AsyncFileReader};

    use crate::reader::chunk::ChunkQuerySource;

    pub struct AsyncSpectrumChunkReader<T: AsyncFileReader + 'static + Unpin + Send> {
        builder: ParquetRecordBatchStreamBuilder<T>,
    }

    impl<T: AsyncFileReader + 'static + Unpin + Send> AsyncSpectrumChunkReader<T> {
        pub fn new(builder: ParquetRecordBatchStreamBuilder<T>) -> Self {
            Self { builder }
        }

        pub fn scan_chunks_for<'a>(
            self,
            index_range: MaskSet,
            query_range: Option<SimpleInterval<f64>>,
            metadata: &'a ReaderMetadata,
            array_indices: &'a ArrayIndex,
            query_indices: &'a impl BasicChunkQueryIndex,
        ) -> io::Result<BoxStream<'a, Result<RecordBatch, ArrowError>>> {
            let buffer_context = self.buffer_context();
            let (rows, row_group_indices, predicate) =
                self.prepare_scan(index_range, query_range, array_indices, query_indices);

            let reader = self
                .builder
                .with_row_groups(row_group_indices)
                .with_row_selection(rows)
                .with_row_filter(predicate)
                .with_batch_size(4096)
                .build()?;

            let mut decoder =
                ChunkScanDecoder::new(buffer_context, metadata, query_range, array_indices);

            let it = reader.map(move |batch| -> Result<RecordBatch, ArrowError> {
                decoder.decode_batch(batch?)
            });

            Ok(it.boxed())
        }

        pub async fn read_chunks_for_entity(
            self,
            query: u64,
            query_indices: &impl BasicChunkQueryIndex,
            array_indices: &ArrayIndex,
            delta_model: Option<&RegressionDeltaModel<f64>>,
            page_query: Option<PageQuery>,
        ) -> io::Result<BinaryArrayMap> {
            if let Some((rows, rg_idx_acc, proj, predicate)) =
                self.prepare_chunks_of(query, query_indices, page_query)
            {
                log::trace!("{query} @ chunk spread across row groups {rg_idx_acc:?}");
                let mut reader = self
                    .builder
                    .with_row_groups(rg_idx_acc)
                    .with_row_selection(rows)
                    .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
                    .with_projection(proj)
                    .build()?;

                let mut decoder = ChunkDecoder::new(array_indices, delta_model);

                while let Some(bat) = reader.next().await.transpose()? {
                    decoder.decode_batch(bat);
                }

                Ok(decoder.compile_buffers()?)
                // Self::decode_chunks(reader.flatten(), spectrum_array_indices, delta_model)
            } else {
                let mut bin_map = BinaryArrayMap::new();
                for k in array_indices.iter() {
                    bin_map.add(k.as_buffer_name().as_data_array(0));
                }
                Ok(bin_map)
            }
        }

        pub(crate) async fn load_cache_block(
            self,
            index_range: SimpleInterval<u64>,
            metadata: &ReaderMetadata,
            query_indices: &impl BasicChunkQueryIndex,
        ) -> io::Result<DataChunkCache> {
            let (rows, predicate) = self.prepare_cache_block_query(index_range, query_indices);

            let schema = self.builder.schema().clone();
            let mut reader = self
                .builder
                .with_row_selection(rows)
                .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
                .build()?;

            let mut batches = Vec::new();

            while let Some(bat) = reader.next().await.transpose()? {
                batches.push(bat);
            }

            let batch = arrow::compute::concat_batches(&schema, batches.iter())
                .map_err(io::Error::other)?;

            Ok(DataChunkCache::new(
                batch,
                index_range,
                metadata.spectrum_array_indices.clone(),
                None,
                None,
            ))
        }
    }

    impl<T: AsyncFileReader + 'static + Unpin + Send> ChunkQuerySource for AsyncSpectrumChunkReader<T> {
        fn metadata(&self) -> &parquet::file::metadata::ParquetMetaData {
            self.builder.metadata()
        }

        fn buffer_context(&self) -> BufferContext {
            BufferContext::Spectrum
        }
    }
}

#[cfg(feature = "async")]
pub use async_impl::AsyncSpectrumChunkReader;

pub(crate) fn make_ion_mobility_filter<'a>(
    it: Box<dyn Iterator<Item = Result<RecordBatch, ArrowError>> + 'a>,
    ion_mobility_range: SimpleInterval<f64>,
    im_name: &'a ArrayIndexEntry,
) -> Box<dyn Iterator<Item = Result<RecordBatch, ArrowError>> + 'a> {
    let it = it.map(move |bat| -> Result<RecordBatch, ArrowError> {
        let bat = bat?;
        let arr = bat
            .column(0)
            .as_struct()
            .column_by_name(&im_name.name)
            .unwrap();
        let mask = ion_mobility_range.contains_dy(&arr);
        arrow::compute::filter_record_batch(&bat, &mask)
    });
    Box::new(it)
}
