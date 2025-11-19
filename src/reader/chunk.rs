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
        index::{PageQuery, QueryIndex, RangeIndex, SpanDynNumeric},
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

        let out = SpectrumChunkReader::<std::fs::File>::decode_chunks(
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
        "chunk"
    }

    fn prepare_predicate_for_index(
        &self,
        index_range: MaskSet,
    ) -> Box<dyn ArrowPredicate> {
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

    fn prepare_predicate_for_mz(
        &self,
        query_range: SimpleInterval<f64>,
        metadata: &ReaderMetadata,
    ) -> Option<Box<dyn ArrowPredicate>> {
        if let Some(e) = metadata.spectrum_array_indices.get(&ArrayType::MZArray) {
            // Maybe rewrite this in terms of `BufferFormat` values instead of raw names, but these
            // are unlikely to change.
            let prefix = e.path.as_str();
            let prefix = if prefix.ends_with("_chunk_values") {
                prefix.replace("_chunk_values", "")
            } else {
                prefix.to_string()
            };
            let fields = [
                format!("{prefix}_chunk_start"),
                format!("{prefix}_chunk_end"),
            ];

            let proj =
                ProjectionMask::columns(&self.parquet_schema(), fields.iter().map(|s| s.as_str()));
            let predicate = ArrowPredicateFn::new(proj, move |batch| {
                let root = batch.column(0).as_struct();
                let mz_start_array = root.column(0);
                let mz_end_array = root.column(1);
                let it2 = query_range.overlaps_dy(mz_start_array, mz_end_array);
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
        metadata: &ReaderMetadata,
        query_indices: &QueryIndex,
    ) -> (RowSelection, Vec<usize>, RowFilter) {
        let mut rows = query_indices
            .spectrum_chunk_index
            .spectrum_index
            .row_selection_overlaps(&index_range);

        let PageQuery {
            row_group_indices,
            pages: _,
        } = query_indices
            .spectrum_chunk_index
            .query_pages_overlaps(&index_range);

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
                &query_indices.spectrum_chunk_index.start_mz_index,
                &query_indices.spectrum_chunk_index.end_mz_index,
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

        if let Some(e) = metadata.spectrum_array_indices.get(&ArrayType::MZArray) {
            let prefix = e.path.as_str();
            let prefix = if prefix.ends_with("_chunk_values") {
                prefix.replace("_chunk_values", "")
            } else {
                prefix.to_string()
            };
            fields.push(format!("{prefix}_chunk_values"));
            fields.push(format!("{prefix}_chunk_start"));
            fields.push(format!("{prefix}_chunk_end"));
        }

        if let Some(e) = metadata
            .spectrum_array_indices
            .get(&ArrayType::IntensityArray)
        {
            fields.push(e.path.to_string());
        }

        for v in metadata.spectrum_array_indices.iter() {
            if v.is_ion_mobility() {
                fields.push(v.path.to_string());
                break;
            }
        }

        log::trace!("Executing query on {fields:?}");

        let mut predicates = vec![self.prepare_predicate_for_index(index_range)];

        if let Some(query_range) = query_range {
            predicates.extend(self.prepare_predicate_for_mz(query_range, metadata));
        }

        let predicate = RowFilter::new(predicates);
        (rows, row_group_indices, predicate)
    }

    fn prepare_chunks_of(
        &self,
        query: u64,
        query_indices: &QueryIndex,
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
        } = page_query.unwrap_or_else(|| query_indices.spectrum_chunk_index.query_pages(query));

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
            .spectrum_point_index
            .spectrum_index
            .pages_to_row_selection(&pages, first_row);

        let sidx = format!(
            "{}.{}",
            Self::array_prefix(),
            self.buffer_context().index_name()
        );

        let predicate_mask =
            ProjectionMask::columns(&self.parquet_schema(), [sidx.as_str()]);

        let predicate = ArrowPredicateFn::new(predicate_mask, move |batch| {
            let spectrum_index: &UInt64Array = batch
                .column(0)
                .as_struct()
                .column(0)
                .as_any()
                .downcast_ref()
                .unwrap();

            let it = spectrum_index
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
        query_indices: &QueryIndex,
    ) -> (
        RowSelection,
        ArrowPredicateFn<
            impl FnMut(RecordBatch) -> Result<arrow::array::BooleanArray, ArrowError> + 'static,
        >,
    ) {
        let rows = query_indices
            .spectrum_chunk_index
            .spectrum_index
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

struct ChunkDecoder<'a> {
    buffers: HashMap<BufferName, Vec<ArrayRef>>,
    main_axis_buffers: Vec<(BufferName, ArrayRef)>,
    main_axis_starts: Vec<ArrayRef>,
    main_axis_ends: Vec<ArrayRef>,
    main_axis: Option<DataArray>,
    bin_map: BinaryArrayMap,
    array_indices: &'a ArrayIndex,
    delta_model: Option<&'a RegressionDeltaModel<f64>>,
}

impl<'a> ChunkDecoder<'a> {
    fn new(
        spectrum_array_indices: &'a ArrayIndex,
        delta_model: Option<&'a RegressionDeltaModel<f64>>,
    ) -> Self {
        Self {
            buffers: Default::default(),
            main_axis_buffers: Default::default(),
            main_axis_starts: Default::default(),
            main_axis_ends: Default::default(),
            main_axis: None,
            bin_map: Default::default(),
            array_indices: spectrum_array_indices,
            delta_model,
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
            let decoder = name.transform.map(BufferTransformDecoder);

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
                    let first = arr_iter.peek().unwrap();
                    {
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
            // log::debug!(
            //     "Unpacked {} values, main axis had {n}",
            //     store.data_len().unwrap()
            // );
            self.bin_map.add(store);
        }
        Ok(self.bin_map)
    }

    fn decode_batch(&mut self, batch: RecordBatch) {
        let root = batch.column(0).as_struct();
        let mut chunk_encodings: Vec<_> = Vec::new();
        for (f, arr) in root.fields().iter().zip(root.columns()) {
            match f.name().as_str() {
                "spectrum_index" => {
                    continue;
                }
                "chunk_encoding" => {
                    chunk_encodings = AnyCURIEArray::try_from(arr).unwrap().to_vec();
                }
                s if s.ends_with("chunk_start") => {
                    self.main_axis_starts.push(arr.clone());
                }
                s if s.ends_with("chunk_end") => {
                    self.main_axis_ends.push(arr.clone());
                }
                _ => {
                    if let Some(name) =
                        BufferName::from_field(crate::BufferContext::Spectrum, f.clone())
                    {
                        match name.buffer_format {
                            BufferFormat::Chunked => {
                                log::trace!(
                                    "Storing {name} with {:?} and {} entries",
                                    arr.data_type(),
                                    arr.len()
                                );
                                self.main_axis_buffers.push((name, arr.clone()));
                            }
                            BufferFormat::ChunkedSecondary | BufferFormat::Point => {
                                self.buffers.entry(name).or_default().push(arr.clone());
                            }
                            BufferFormat::ChunkBoundsStart => {
                                self.main_axis_starts.push(arr.clone())
                            }
                            BufferFormat::ChunkBoundsEnd => self.main_axis_ends.push(arr.clone()),
                            BufferFormat::ChunkEncoding => {
                                chunk_encodings = AnyCURIEArray::try_from(arr).unwrap().to_vec()
                            }
                        }
                    } else {
                        log::warn!("{f:?} failed to map to a chunk buffer");
                    }
                }
            }
        }

        let total_size: usize = self
            .main_axis_buffers
            .iter()
            .map(|(_, v)| v.len() + 1)
            .sum();

        for (name, chunk_values) in self.main_axis_buffers.drain(..) {
            for (encoding, (chunk_starts, chunk_ends)) in chunk_encodings
                .iter()
                .copied()
                .zip(self.main_axis_starts.iter().zip(self.main_axis_ends.iter()))
            {
                // This may over-allocate, but not by more than a few bytes per chunk
                if self.main_axis.is_none() {
                    self.main_axis = Some(DataArray::from_name_type_size(
                        &name.array_type,
                        name.dtype,
                        total_size * name.dtype.size_of(),
                    ))
                }
                match encoding {
                    NO_COMPRESSION => {
                        macro_rules! decode_no_compression {
                            ($chunk_values:ident, $starts:ident, $ends:ident, $accumulator:expr) => {
                                for (chunk_vals, (start, end)) in
                                    $chunk_values.iter().zip($starts.iter().zip($ends.iter()))
                                {
                                    let chunk_vals = chunk_vals.unwrap();
                                    let start = start.unwrap();
                                    let end = end.unwrap();
                                    (ChunkingStrategy::Basic { chunk_size: 50.0 }).decode_arrow(
                                        &chunk_vals,
                                        start as f64,
                                        end as f64,
                                        $accumulator.as_mut().unwrap(),
                                        None,
                                    );
                                }
                            };
                        }
                        let chunk_values = chunk_values.as_list::<i64>();
                        if let Some(starts) = chunk_starts.as_primitive_opt::<Float64Type>() {
                            let ends = chunk_ends.as_primitive::<Float64Type>();
                            decode_no_compression!(chunk_values, starts, ends, self.main_axis);
                        } else if let Some(starts) = chunk_starts.as_primitive_opt::<Float32Type>()
                        {
                            let ends = chunk_ends.as_primitive::<Float32Type>();
                            decode_no_compression!(chunk_values, starts, ends, self.main_axis);
                        } else {
                            unimplemented!("Starts were {:?}", chunk_starts.data_type());
                        };
                    }
                    DELTA_ENCODE => {
                        macro_rules! decode_deltas {
                            ($chunk_values:ident, $starts:ident, $ends:ident, $accumulator:expr) => {
                                for (chunk_vals, (start, end)) in
                                    $chunk_values.iter().zip($starts.iter().zip($ends.iter()))
                                {
                                    if let Some(chunk_vals) = chunk_vals {
                                        let start = start.unwrap();
                                        let end = end.unwrap();
                                        (ChunkingStrategy::Delta { chunk_size: 50.0 })
                                            .decode_arrow(
                                                &chunk_vals,
                                                start as f64,
                                                end as f64,
                                                $accumulator.as_mut().unwrap(),
                                                self.delta_model,
                                            );
                                    }
                                }
                            };
                        }
                        let chunk_values = chunk_values.as_list::<i64>();
                        if let Some(starts) = chunk_starts.as_primitive_opt::<Float64Type>() {
                            let ends = chunk_ends.as_primitive::<Float64Type>();
                            decode_deltas!(chunk_values, starts, ends, self.main_axis);
                        } else if let Some(starts) = chunk_starts.as_primitive_opt::<Float32Type>()
                        {
                            let ends = chunk_ends.as_primitive::<Float32Type>();
                            decode_deltas!(chunk_values, starts, ends, self.main_axis);
                        } else {
                            unimplemented!("Starts were {:?}", chunk_starts.data_type());
                        };
                    }
                    NUMPRESS_LINEAR => {
                        let chunk_values = chunk_values.as_list::<i64>();
                        macro_rules! decode_numpress {
                            ($chunk_values:ident, $starts:ident, $ends:ident, $accumulator:expr) => {
                                for (chunk_vals, (start, end)) in
                                    $chunk_values.iter().zip($starts.iter().zip($ends.iter()))
                                {
                                    if let Some(chunk_vals) = chunk_vals {
                                        let start = start.unwrap();
                                        let end = end.unwrap();
                                        (ChunkingStrategy::NumpressLinear { chunk_size: 50.0 })
                                            .decode_arrow(
                                                &chunk_vals,
                                                start as f64,
                                                end as f64,
                                                $accumulator.as_mut().unwrap(),
                                                self.delta_model,
                                            );
                                    }
                                }
                            };
                        }
                        if let Some(starts) = chunk_starts.as_primitive_opt::<Float64Type>() {
                            let ends = chunk_ends.as_primitive::<Float64Type>();
                            decode_numpress!(chunk_values, starts, ends, self.main_axis);
                        } else {
                            unimplemented!("Starts were {:?}", chunk_starts.data_type());
                        }
                    }
                    _ => {
                        panic!("Unknown or unsupported chunk encoding: {encoding}")
                    }
                }
            }
        }
    }
}

struct ChunkScanDecoder<'a> {
    buffers: HashMap<BufferName, Vec<ArrayRef>>,
    main_axis_buffers: Vec<(BufferName, ArrayRef)>,
    main_axis_starts: Vec<ArrayRef>,
    main_axis_ends: Vec<ArrayRef>,
    spectrum_index: Vec<ArrayRef>,
    main_axis: Option<DataArray>,
    metadata: &'a ReaderMetadata,
    query_range: Option<SimpleInterval<f64>>,
}

impl<'a> ChunkScanDecoder<'a> {
    fn new(metadata: &'a ReaderMetadata, query_range: Option<SimpleInterval<f64>>) -> Self {
        Self {
            buffers: Default::default(),
            main_axis_buffers: Default::default(),
            main_axis_starts: Default::default(),
            main_axis_ends: Default::default(),
            main_axis: None,
            spectrum_index: Default::default(),
            metadata,
            query_range,
        }
    }

    fn clear(&mut self) {
        self.buffers.clear();
        self.main_axis_buffers.clear();
        self.main_axis_starts.clear();
        self.main_axis_ends.clear();
        self.main_axis = None;
        self.spectrum_index.clear();
    }

    fn decode_batch(&mut self, batch: RecordBatch) -> Result<RecordBatch, ArrowError> {
        let root = batch.column(0).as_struct();
        let mut chunk_encodings: Vec<_> = Vec::new();

        let mut delta_model_cache = OneCache::default();

        for (f, arr) in root.fields().iter().zip(root.columns()) {
            match f.name().as_str() {
                "spectrum_index" => {
                    self.spectrum_index.push(arr.clone());
                }
                "chunk_encoding" => {
                    chunk_encodings = AnyCURIEArray::try_from(arr).unwrap().to_vec();
                }
                s if s.ends_with("chunk_start") => {
                    self.main_axis_starts.push(arr.clone());
                }
                s if s.ends_with("chunk_end") => {
                    self.main_axis_ends.push(arr.clone());
                }
                _ => {
                    if let Some(name) =
                        BufferName::from_field(crate::BufferContext::Spectrum, f.clone())
                    {
                        match name.buffer_format {
                            BufferFormat::Chunked => {
                                log::trace!(
                                    "Storing {name} with {:?} and {} entries",
                                    arr.data_type(),
                                    arr.len(),
                                );
                                self.main_axis_buffers.push((name, arr.clone()));
                            }
                            BufferFormat::ChunkedSecondary | BufferFormat::Point => {
                                self.buffers.entry(name).or_default().push(arr.clone());
                            }
                            BufferFormat::ChunkBoundsStart => {
                                self.main_axis_starts.push(arr.clone())
                            }
                            BufferFormat::ChunkBoundsEnd => self.main_axis_ends.push(arr.clone()),
                            BufferFormat::ChunkEncoding => {
                                chunk_encodings = AnyCURIEArray::try_from(arr).unwrap().to_vec()
                            }
                        }
                    } else {
                        log::warn!("{f:?} failed to map to a chunk buffer");
                    }
                }
            }
        }

        // Accumulate the per-point spectrum association
        let mut spectrum_idx_acc: Vec<u64> =
            Vec::with_capacity(self.spectrum_index.iter().map(|v| v.len()).sum());

        for (name, chunk_values) in self.main_axis_buffers.drain(..) {
            for ((encoding, (chunk_starts, chunk_ends)), spectrum_idxs) in chunk_encodings
                .iter()
                .copied()
                .zip(self.main_axis_starts.iter().zip(self.main_axis_ends.iter()))
                .zip(
                    self.spectrum_index
                        .iter()
                        .map(|v| v.as_primitive::<UInt64Type>()),
                )
            {
                if self.main_axis.is_none() {
                    self.main_axis =
                        Some(DataArray::from_name_and_type(&name.array_type, name.dtype))
                }
                match encoding {
                    NO_COMPRESSION => {
                        macro_rules! decode_no_compression {
                            ($chunk_values:ident, $starts:ident, $ends:ident, $accumulator:expr) => {
                                for (chunk_vals, ((start, end), spectrum_idx)) in
                                    $chunk_values.iter().zip(
                                        $starts
                                            .iter()
                                            .zip($ends.iter())
                                            .zip(spectrum_idxs.iter().flatten()),
                                    )
                                {
                                    let chunk_vals = chunk_vals.unwrap();
                                    let start = start.unwrap();
                                    let end = end.unwrap();
                                    let main_axis_size_before = self
                                        .main_axis
                                        .as_ref()
                                        .map(|a| a.data_len().unwrap_or_default())
                                        .unwrap_or_default();
                                    (ChunkingStrategy::Basic { chunk_size: 50.0 }).decode_arrow(
                                        &chunk_vals,
                                        start as f64,
                                        end as f64,
                                        $accumulator.as_mut().unwrap(),
                                        None,
                                    );
                                    let main_axis_size_after = self
                                        .main_axis
                                        .as_ref()
                                        .map(|a| a.data_len().unwrap_or_default())
                                        .unwrap_or_default();
                                    let n_points_added =
                                        main_axis_size_after - main_axis_size_before;
                                    spectrum_idx_acc.extend(std::iter::repeat_n(
                                        spectrum_idx,
                                        n_points_added as usize,
                                    ));
                                }
                            };
                        }
                        let chunk_values = chunk_values.as_list::<i64>();
                        if let Some(starts) = chunk_starts.as_primitive_opt::<Float64Type>() {
                            let ends = chunk_ends.as_primitive::<Float64Type>();
                            decode_no_compression!(chunk_values, starts, ends, self.main_axis);
                        } else if let Some(starts) = chunk_starts.as_primitive_opt::<Float32Type>()
                        {
                            let ends = chunk_ends.as_primitive::<Float32Type>();
                            decode_no_compression!(chunk_values, starts, ends, self.main_axis);
                        } else {
                            unimplemented!("Starts were {:?}", chunk_starts.data_type());
                        };
                    }
                    DELTA_ENCODE => {
                        macro_rules! decode_deltas {
                            ($chunk_values:ident, $starts:ident, $ends:ident, $accumulator:expr) => {
                                for (chunk_vals, ((start, end), spectrum_idx)) in
                                    $chunk_values.iter().zip(
                                        $starts
                                            .iter()
                                            .zip($ends.iter())
                                            .zip(spectrum_idxs.iter().flatten()),
                                    )
                                {
                                    if let Some(chunk_vals) = chunk_vals {
                                        let start = start.unwrap();
                                        let end = end.unwrap();
                                        let delta_model =
                                            delta_model_cache.get(spectrum_idx, || {
                                                self.metadata
                                                    .model_deltas_for(spectrum_idx as usize)
                                            });
                                        let main_axis_size_before = self
                                            .main_axis
                                            .as_ref()
                                            .map(|a| a.data_len().unwrap_or_default())
                                            .unwrap_or_default();
                                        (ChunkingStrategy::Delta { chunk_size: 50.0 })
                                            .decode_arrow(
                                                &chunk_vals,
                                                start as f64,
                                                end as f64,
                                                $accumulator.as_mut().unwrap(),
                                                delta_model.as_ref(),
                                            );
                                        let main_axis_size_after = self
                                            .main_axis
                                            .as_ref()
                                            .map(|a| a.data_len().unwrap_or_default())
                                            .unwrap_or_default();
                                        let n_points_added =
                                            main_axis_size_after - main_axis_size_before;
                                        spectrum_idx_acc.extend(std::iter::repeat_n(
                                            spectrum_idx,
                                            n_points_added as usize,
                                        ));
                                    }
                                }
                            };
                        }
                        let chunk_values = chunk_values.as_list::<i64>();
                        if let Some(starts) = chunk_starts.as_primitive_opt::<Float64Type>() {
                            let ends = chunk_ends.as_primitive::<Float64Type>();
                            decode_deltas!(chunk_values, starts, ends, self.main_axis);
                        } else if let Some(starts) = chunk_starts.as_primitive_opt::<Float32Type>()
                        {
                            let ends = chunk_ends.as_primitive::<Float32Type>();
                            decode_deltas!(chunk_values, starts, ends, self.main_axis);
                        } else {
                            unimplemented!("Starts were {:?}", chunk_starts.data_type());
                        };
                    }
                    NUMPRESS_LINEAR => {
                        let chunk_values = chunk_values.as_list::<i64>();
                        macro_rules! decode_numpress {
                            ($chunk_values:ident, $starts:ident, $ends:ident, $accumulator:expr) => {
                                for (chunk_vals, ((start, end), spectrum_idx)) in
                                    $chunk_values.iter().zip(
                                        $starts
                                            .iter()
                                            .zip($ends.iter())
                                            .zip(spectrum_idxs.iter().flatten()),
                                    )
                                {
                                    if let Some(chunk_vals) = chunk_vals {
                                        let start = start.unwrap();
                                        let end = end.unwrap();
                                        let delta_model =
                                            delta_model_cache.get(spectrum_idx, || {
                                                self.metadata
                                                    .model_deltas_for(spectrum_idx as usize)
                                            });
                                        let main_axis_size_before = self
                                            .main_axis
                                            .as_ref()
                                            .map(|a| a.data_len().unwrap_or_default())
                                            .unwrap_or_default();
                                        (ChunkingStrategy::NumpressLinear { chunk_size: 50.0 })
                                            .decode_arrow(
                                                &chunk_vals,
                                                start as f64,
                                                end as f64,
                                                $accumulator.as_mut().unwrap(),
                                                delta_model.as_ref(),
                                            );
                                        let main_axis_size_after = self
                                            .main_axis
                                            .as_ref()
                                            .map(|a| a.data_len().unwrap_or_default())
                                            .unwrap_or_default();
                                        let n_points_added =
                                            main_axis_size_after - main_axis_size_before;
                                        spectrum_idx_acc.extend(std::iter::repeat_n(
                                            spectrum_idx,
                                            n_points_added as usize,
                                        ));
                                    }
                                }
                            };
                        }
                        if let Some(starts) = chunk_starts.as_primitive_opt::<Float64Type>() {
                            let ends = chunk_ends.as_primitive::<Float64Type>();
                            decode_numpress!(chunk_values, starts, ends, self.main_axis);
                        } else {
                            unimplemented!("Starts were {:?}", chunk_starts.data_type());
                        }
                    }
                    _ => {
                        unimplemented!("{encoding}")
                    }
                }
            }
        }

        let axis = self.main_axis.take().unwrap();
        let buffer_name = BufferName::from_data_array(crate::BufferContext::Spectrum, &axis);
        let axis = data_array_to_arrow_array(&buffer_name, &axis).unwrap();

        let mut fields = Vec::with_capacity(self.buffers.len() + 1);
        fields.push(buffer_name.context.index_field());
        fields.push(buffer_name.to_field());

        let mut arrays = Vec::with_capacity(self.buffers.len() + 1);
        arrays.push(Arc::new(UInt64Array::from(spectrum_idx_acc)) as ArrayRef);
        arrays.push(axis);

        for (name, chunks) in self.buffers.drain() {
            let decoder = name.transform.map(BufferTransformDecoder);
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
pub struct SpectrumChunkReader<T: ChunkReader + 'static> {
    builder: ParquetRecordBatchReaderBuilder<T>,
}

impl<T: ChunkReader + 'static> ChunkQuerySource for SpectrumChunkReader<T> {
    fn metadata(&self) -> &ParquetMetaData {
        self.builder.metadata()
    }

    fn buffer_context(&self) -> BufferContext {
        BufferContext::Spectrum
    }
}

impl<T: ChunkReader + 'static> SpectrumChunkReader<T> {
    pub fn new(builder: ParquetRecordBatchReaderBuilder<T>) -> Self {
        Self { builder }
    }

    pub(crate) fn load_cache_block(
        self,
        index_range: SimpleInterval<u64>,
        metadata: &ReaderMetadata,
        query_indices: &QueryIndex,
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

    pub fn scan_chunks_for(
        self,
        index_range: MaskSet,
        query_range: Option<SimpleInterval<f64>>,
        metadata: &ReaderMetadata,
        query_indices: &QueryIndex,
    ) -> io::Result<impl Iterator<Item = Result<RecordBatch, ArrowError>>> {
        let (rows, row_group_indices, predicate) =
            self.prepare_scan(index_range, query_range, metadata, query_indices);

        let reader = self
            .builder
            .with_row_groups(row_group_indices)
            .with_row_selection(rows)
            .with_row_filter(predicate)
            .with_batch_size(4096)
            .build()?;

        let mut decoder = ChunkScanDecoder::new(metadata, query_range);

        let it = reader.map(move |batch| -> Result<RecordBatch, ArrowError> {
            batch.and_then(|batch| decoder.decode_batch(batch))
        });

        Ok(Box::new(it))
    }

    pub fn decode_chunks<I: Iterator<Item = RecordBatch>>(
        reader: I,
        spectrum_array_indices: &ArrayIndex,
        delta_model: Option<&RegressionDeltaModel<f64>>,
    ) -> io::Result<BinaryArrayMap> {
        let mut decoder = ChunkDecoder::new(spectrum_array_indices, delta_model);
        for batch in reader {
            decoder.decode_batch(batch);
        }
        let bin_map = decoder.compile_buffers()?;
        Ok(bin_map)
    }

    pub fn read_chunks_for_spectrum(
        self,
        query: u64,
        query_indices: &QueryIndex,
        spectrum_array_indices: &ArrayIndex,
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

            Self::decode_chunks(reader.flatten(), spectrum_array_indices, delta_model)
        } else {
            let mut bin_map = BinaryArrayMap::new();
            for k in spectrum_array_indices.iter() {
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
            query_indices: &'a QueryIndex,
        ) -> io::Result<BoxStream<'a, Result<RecordBatch, ArrowError>>> {
            let (rows, row_group_indices, predicate) =
                self.prepare_scan(index_range, query_range, metadata, query_indices);

            let reader = self
                .builder
                .with_row_groups(row_group_indices)
                .with_row_selection(rows)
                .with_row_filter(predicate)
                .with_batch_size(4096)
                .build()?;

            let mut decoder = ChunkScanDecoder::new(metadata, query_range);

            let it = reader.map(move |batch| -> Result<RecordBatch, ArrowError> {
                decoder.decode_batch(batch?)
            });

            Ok(it.boxed())
        }

        pub async fn read_chunks_for_spectrum(
            self,
            query: u64,
            query_indices: &QueryIndex,
            spectrum_array_indices: &ArrayIndex,
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

                let mut decoder = ChunkDecoder::new(spectrum_array_indices, delta_model);

                while let Some(bat) = reader.next().await.transpose()? {
                    decoder.decode_batch(bat);
                }

                Ok(decoder.compile_buffers()?)
                // Self::decode_chunks(reader.flatten(), spectrum_array_indices, delta_model)
            } else {
                let mut bin_map = BinaryArrayMap::new();
                for k in spectrum_array_indices.iter() {
                    bin_map.add(k.as_buffer_name().as_data_array(0));
                }
                Ok(bin_map)
            }
        }

        pub(crate) async fn load_cache_block(
            self,
            index_range: SimpleInterval<u64>,
            metadata: &ReaderMetadata,
            query_indices: &QueryIndex,
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
