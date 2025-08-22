use std::{collections::HashMap, io, sync::Arc};

use arrow::{
    array::{
        Array, ArrayRef, AsArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
        StructArray, UInt8Array, UInt64Array,
    },
    datatypes::{DataType, Field, Float32Type, Float64Type, Schema, UInt64Type},
    error::ArrowError,
};
use parquet::arrow::{
    ProjectionMask,
    arrow_reader::{ArrowPredicateFn, ParquetRecordBatchReaderBuilder, RowFilter},
};

use mzdata::{
    prelude::*,
    spectrum::{ArrayType, BinaryArrayMap, DataArray},
};
use mzpeaks::coordinate::SimpleInterval;

use crate::{
    BufferName, CURIE,
    buffer_descriptors::arrow_to_array_type,
    chunk_series::{ChunkingStrategy, DELTA_ENCODE, NO_COMPRESSION, NUMPRESS_LINEAR},
    filter::RegressionDeltaModel,
    peak_series::{ArrayIndex, BufferFormat, data_array_to_arrow_array},
    reader::{
        MzPeakReaderMetadata,
        index::{PageQuery, QueryIndex, RangeIndex, SpanDynNumeric},
        point::binary_search_arrow_index,
    },
};

#[allow(unused)]
pub(crate) struct SpectrumDataChunkCache {
    pub(crate) row_group: RecordBatch,
    pub(crate) spectrum_index_range: SimpleInterval<u64>,
    pub(crate) spectrum_array_indices: Arc<ArrayIndex>,
    pub(crate) last_query_index: Option<u64>,
    pub(crate) last_query_span: Option<(usize, usize)>,
}

#[allow(unused)]
impl SpectrumDataChunkCache {
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
                for (k, v) in self.spectrum_array_indices.iter() {
                    let dtype = arrow_to_array_type(&v.data_type).unwrap();
                    bin_map.insert(&v.name, DataArray::from_name_and_type(k, dtype));
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

#[derive(Debug)]
pub struct SpectrumChunkReader<T: parquet::file::reader::ChunkReader + 'static> {
    builder: ParquetRecordBatchReaderBuilder<T>,
}

impl<T: parquet::file::reader::ChunkReader + 'static> SpectrumChunkReader<T> {
    pub fn new(builder: ParquetRecordBatchReaderBuilder<T>) -> Self {
        Self { builder }
    }

    #[allow(unused)]
    pub(crate) fn load_cache_block(
        self,
        index_range: SimpleInterval<u64>,
        metadata: &MzPeakReaderMetadata,
        query_indices: &QueryIndex,
    ) -> io::Result<SpectrumDataChunkCache> {
        let rows = query_indices
            .spectrum_chunk_index
            .spectrum_index
            .row_selection_overlaps(&index_range);

        let proj =
            ProjectionMask::columns(self.builder.parquet_schema(), vec!["chunk.spectrum_index"]);
        let predicate_mask = proj.clone();

        let schema = self.builder.schema().clone();

        let predicate = ArrowPredicateFn::new(predicate_mask, move |batch| {
            let root = batch.column(0).as_struct();
            let spectrum_index: &UInt64Array = root.column(0).as_any().downcast_ref().unwrap();

            let it = spectrum_index
                .iter()
                .map(|v| v.map(|v| index_range.contains(&v)));

            Ok(it.collect())
        });

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

        Ok(SpectrumDataChunkCache::new(
            batch,
            index_range,
            metadata.spectrum_array_indices.clone(),
            None,
            None,
        ))
    }

    pub fn scan_chunks_for(
        self,
        index_range: SimpleInterval<u64>,
        query_range: Option<SimpleInterval<f64>>,
        metadata: &MzPeakReaderMetadata,
        query_indices: &QueryIndex,
    ) -> io::Result<impl Iterator<Item = Result<RecordBatch, ArrowError>>> {
        let mut rows = query_indices
            .spectrum_chunk_index
            .spectrum_index
            .row_selection_overlaps(&index_range);

        if let Some(query_range) = query_range.as_ref() {
            let chunk_range_idx = RangeIndex::new(
                &query_indices.spectrum_chunk_index.start_mz_index,
                &query_indices.spectrum_chunk_index.end_mz_index,
            );
            rows = rows.intersection(&chunk_range_idx.row_selection_overlaps(query_range));
        }

        let sidx = format!("{}.spectrum_index", metadata.spectrum_array_indices.prefix);

        let mut fields: Vec<String> = Vec::new();

        fields.push(sidx);

        if let Some(e) = metadata.spectrum_array_indices.get(&ArrayType::MZArray) {
            let prefix = e.path.as_str();
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

        for (k, v) in metadata.spectrum_array_indices.iter() {
            if k.is_ion_mobility() {
                fields.push(v.path.to_string());
                break;
            }
        }

        let proj = ProjectionMask::columns(
            self.builder.parquet_schema(),
            fields.iter().map(|s| s.as_str()),
        );
        let predicate_mask = proj.clone();

        let predicate = ArrowPredicateFn::new(predicate_mask, move |batch| {
            let root = batch.column(0).as_struct();
            let spectrum_index: &UInt64Array = root.column(0).as_any().downcast_ref().unwrap();

            let it = spectrum_index
                .iter()
                .map(|v| v.map(|v| index_range.contains(&v)));

            match query_range {
                None => Ok(it.collect()),
                Some(mz_range) => {
                    let mz_start_array = root.column(1);
                    let mz_end_array = root.column(2);

                    let it2 = mz_range.overlaps_dy(mz_start_array, mz_end_array);
                    let it2 = it2.iter();
                    let it = it
                        .zip(it2)
                        .map(|(a, b)| Some(a.is_some_and(|a| a && b.unwrap_or_default())));
                    Ok(it.collect())
                }
            }
        });

        let reader = self
            .builder
            .with_row_selection(rows)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .build()?;

        let it = reader.map(move |batch| -> Result<RecordBatch, ArrowError> {
            let batch = batch?;
            let root = batch.column(0).as_struct();
            let mut buffers: HashMap<BufferName, Vec<ArrayRef>> = HashMap::new();
            let mut main_axis_buffers = Vec::new();
            let mut main_axis_starts = Vec::new();
            let mut chunk_encodings: Vec<CURIE> = Vec::new();
            let mut spectrum_index = None;
            let mut main_axis = None;
            for (f, arr) in root.fields().iter().zip(root.columns()) {
                match f.name().as_str() {
                    "spectrum_index" => {
                        spectrum_index = Some(arr.clone());
                    }
                    "chunk_encoding" => {
                        let arr = arr.as_struct();
                        chunk_encodings =
                            serde_arrow::from_arrow(arr.fields(), arr.columns()).unwrap();
                    }
                    s if s.ends_with("chunk_start") => {
                        main_axis_starts.push(arr.clone());
                    }
                    s if s.ends_with("chunk_end") => {
                        continue;
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
                                    main_axis_buffers.push((name, arr.clone()));
                                }
                                BufferFormat::ChunkedSecondary | BufferFormat::Point => {
                                    buffers.entry(name).or_default().push(arr.clone());
                                }
                            }
                        } else {
                            log::warn!("{f:?} failed to map to a chunk buffer");
                        }
                    }
                }
            }

            for (name, chunk_values) in main_axis_buffers.drain(..) {
                for ((encoding, chunk_starts), spectrum_idx) in chunk_encodings
                    .iter()
                    .copied()
                    .zip(main_axis_starts.iter())
                    .zip(
                        spectrum_index
                            .as_ref()
                            .unwrap()
                            .as_primitive::<UInt64Type>()
                            .iter()
                            .flatten(),
                    )
                {
                    if main_axis.is_none() {
                        main_axis =
                            Some(DataArray::from_name_and_type(&name.array_type, name.dtype))
                    }

                    match encoding {
                        NO_COMPRESSION => {
                            macro_rules! decode_no_compression {
                                ($chunk_values:ident, $starts:ident, $accumulator:ident) => {
                                    for (chunk_vals, start) in
                                        $chunk_values.iter().zip($starts.iter())
                                    {
                                        let chunk_vals = chunk_vals.unwrap();
                                        let start = start.unwrap();
                                        (ChunkingStrategy::Basic { chunk_size: 50.0 })
                                            .decode_arrow(
                                                &chunk_vals,
                                                start as f64,
                                                $accumulator.as_mut().unwrap(),
                                                None,
                                            );
                                    }
                                };
                            }
                            let chunk_values = chunk_values.as_list::<i64>();
                            if let Some(starts) = chunk_starts.as_primitive_opt::<Float64Type>() {
                                decode_no_compression!(chunk_values, starts, main_axis);
                            } else if let Some(starts) =
                                chunk_starts.as_primitive_opt::<Float32Type>()
                            {
                                decode_no_compression!(chunk_values, starts, main_axis);
                            } else {
                                unimplemented!("Starts were {:?}", chunk_starts.data_type());
                            };
                        }
                        DELTA_ENCODE => {
                            let delta_model = metadata.model_deltas_for_conv(spectrum_idx as usize);
                            macro_rules! decode_deltas {
                                ($chunk_values:ident, $starts:ident, $accumulator:ident) => {
                                    for (chunk_vals, start) in
                                        $chunk_values.iter().zip($starts.iter())
                                    {
                                        if let Some(chunk_vals) = chunk_vals {
                                            let start = start.unwrap();
                                            (ChunkingStrategy::Delta { chunk_size: 50.0 })
                                                .decode_arrow(
                                                    &chunk_vals,
                                                    start as f64,
                                                    $accumulator.as_mut().unwrap(),
                                                    delta_model.as_ref(),
                                                );
                                        }
                                    }
                                };
                            }
                            let chunk_values = chunk_values.as_list::<i64>();
                            if let Some(starts) = chunk_starts.as_primitive_opt::<Float64Type>() {
                                decode_deltas!(chunk_values, starts, main_axis);
                            } else if let Some(starts) =
                                chunk_starts.as_primitive_opt::<Float32Type>()
                            {
                                decode_deltas!(chunk_values, starts, main_axis);
                            } else {
                                unimplemented!("Starts were {:?}", chunk_starts.data_type());
                            };
                        }
                        NUMPRESS_LINEAR => {
                            let delta_model = metadata.model_deltas_for_conv(spectrum_idx as usize);
                            let chunk_values = chunk_values.as_list::<i64>();
                            macro_rules! decode_numpress {
                                ($chunk_values:ident, $starts:ident, $accumulator:ident) => {
                                    for (chunk_vals, start) in
                                        $chunk_values.iter().zip($starts.iter())
                                    {
                                        if let Some(chunk_vals) = chunk_vals {
                                            let start = start.unwrap();
                                            (ChunkingStrategy::NumpressLinear { chunk_size: 50.0 })
                                                .decode_arrow(
                                                    &chunk_vals,
                                                    start as f64,
                                                    $accumulator.as_mut().unwrap(),
                                                    delta_model.as_ref(),
                                                );
                                        }
                                    }
                                };
                            }
                            if let Some(starts) = chunk_starts.as_primitive_opt::<Float64Type>() {
                                decode_numpress!(chunk_values, starts, main_axis);
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

            let axis = main_axis.unwrap();
            let buffer_name = BufferName::from_data_array(crate::BufferContext::Spectrum, &axis);
            let axis = data_array_to_arrow_array(&buffer_name, &axis).unwrap();

            let mut fields = Vec::with_capacity(buffers.len() + 1);
            fields.push(buffer_name.to_field());

            let mut arrays = Vec::with_capacity(buffers.len() + 1);
            arrays.push(axis);

            for (name, chunks) in buffers {
                let chunks: Vec<&dyn Array> = chunks
                    .iter()
                    .map(|a| a.as_ref().as_list::<i64>().values().as_ref())
                    .collect();
                let chunks = arrow::compute::concat(&chunks)?;
                arrays.push(chunks);
                fields.push(name.to_field());
            }

            let mut batch: ArrayRef = Arc::new(StructArray::new(fields.into(), arrays, None));

            if let Some(query_range) = query_range.as_ref() {
                let v = batch.as_struct().column(0);
                let mask = query_range.contains_dy(v);
                batch = arrow::compute::filter(&batch, &mask)?;
            }
            let batch = RecordBatch::from(batch.as_struct());
            Ok(batch)
        });

        Ok(it)
    }

    pub fn decode_chunks<I: Iterator<Item = RecordBatch>>(
        reader: I,
        spectrum_array_indices: &ArrayIndex,
        delta_model: Option<&RegressionDeltaModel<f64>>,
    ) -> io::Result<BinaryArrayMap> {
        let mut buffers: HashMap<BufferName, Vec<ArrayRef>> = HashMap::new();
        let mut main_axis_buffers = Vec::new();
        let mut main_axis_starts = Vec::new();
        let mut main_axis = None;
        let mut bin_map = BinaryArrayMap::new();
        for batch in reader {
            let root = batch.column(0).as_struct();
            let mut chunk_encodings: Vec<CURIE> = Vec::new();
            for (f, arr) in root.fields().iter().zip(root.columns()) {
                match f.name().as_str() {
                    "spectrum_index" => {
                        continue;
                    }
                    "chunk_encoding" => {
                        let arr = arr.as_struct();
                        chunk_encodings =
                            serde_arrow::from_arrow(arr.fields(), arr.columns()).unwrap();
                    }
                    s if s.ends_with("chunk_start") => {
                        main_axis_starts.push(arr.clone());
                    }
                    s if s.ends_with("chunk_end") => {
                        continue;
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
                                    main_axis_buffers.push((name, arr.clone()));
                                }
                                BufferFormat::ChunkedSecondary | BufferFormat::Point => {
                                    buffers.entry(name).or_default().push(arr.clone());
                                }
                            }
                        } else {
                            log::warn!("{f:?} failed to map to a chunk buffer");
                        }
                    }
                }
            }
            let total_size: usize = main_axis_buffers.iter().map(|(_, v)| v.len() + 1).sum();

            // TODO: Rewrite the main_axis_buffers loop below to use the chunked form
            // let mut main_axis_buffers_chunked: Vec<(usize, &BufferName, Arc<dyn Array>)> = main_axis_buffers.iter().map(|(name, chunk_values)| {
            //     let chunk_values = chunk_values.as_list::<i64>();
            //     let dim: Vec<_> = chunk_values.iter().enumerate().filter_map(|(i, vi)| {
            //         if vi.is_some() {
            //             Some((i, name, vi.unwrap()))
            //         } else {
            //             None
            //         }
            //     }).collect();
            //     dim
            // }).flatten().collect();
            // main_axis_buffers_chunked.sort_by(|a, b|a.0.cmp(&b.0));

            // This loop is not suited to decoding multiple kinds of encodings which require *different*
            // arrays like mixing numpress
            for (name, chunk_values) in main_axis_buffers.drain(..) {
                for (encoding, chunk_starts) in
                    chunk_encodings.iter().copied().zip(main_axis_starts.iter())
                {
                    // This may over-allocate, but not by more than a few bytes per chunk
                    if main_axis.is_none() {
                        main_axis = Some(DataArray::from_name_type_size(
                            &name.array_type,
                            name.dtype,
                            total_size * name.dtype.size_of(),
                        ))
                    }
                    match encoding {
                        NO_COMPRESSION => {
                            macro_rules! decode_no_compression {
                                ($chunk_values:ident, $starts:ident, $accumulator:ident) => {
                                    for (chunk_vals, start) in
                                        $chunk_values.iter().zip($starts.iter())
                                    {
                                        let chunk_vals = chunk_vals.unwrap();
                                        let start = start.unwrap();
                                        (ChunkingStrategy::Basic { chunk_size: 50.0 })
                                            .decode_arrow(
                                                &chunk_vals,
                                                start as f64,
                                                $accumulator.as_mut().unwrap(),
                                                None,
                                            );
                                    }
                                };
                            }
                            let chunk_values = chunk_values.as_list::<i64>();
                            if let Some(starts) = chunk_starts.as_primitive_opt::<Float64Type>() {
                                decode_no_compression!(chunk_values, starts, main_axis);
                            } else if let Some(starts) =
                                chunk_starts.as_primitive_opt::<Float32Type>()
                            {
                                decode_no_compression!(chunk_values, starts, main_axis);
                            } else {
                                unimplemented!("Starts were {:?}", chunk_starts.data_type());
                            };
                        }
                        DELTA_ENCODE => {
                            macro_rules! decode_deltas {
                                ($chunk_values:ident, $starts:ident, $accumulator:ident) => {
                                    for (chunk_vals, start) in
                                        $chunk_values.iter().zip($starts.iter())
                                    {
                                        if let Some(chunk_vals) = chunk_vals {
                                            let start = start.unwrap();
                                            (ChunkingStrategy::Delta { chunk_size: 50.0 })
                                                .decode_arrow(
                                                    &chunk_vals,
                                                    start as f64,
                                                    $accumulator.as_mut().unwrap(),
                                                    delta_model,
                                                );
                                        }
                                    }
                                };
                            }
                            let chunk_values = chunk_values.as_list::<i64>();
                            if let Some(starts) = chunk_starts.as_primitive_opt::<Float64Type>() {
                                decode_deltas!(chunk_values, starts, main_axis);
                            } else if let Some(starts) =
                                chunk_starts.as_primitive_opt::<Float32Type>()
                            {
                                decode_deltas!(chunk_values, starts, main_axis);
                            } else {
                                unimplemented!("Starts were {:?}", chunk_starts.data_type());
                            };
                        }
                        NUMPRESS_LINEAR => {
                            let chunk_values = chunk_values.as_list::<i64>();
                            macro_rules! decode_numpress {
                                ($chunk_values:ident, $starts:ident, $accumulator:ident) => {
                                    for (chunk_vals, start) in
                                        $chunk_values.iter().zip($starts.iter())
                                    {
                                        if let Some(chunk_vals) = chunk_vals {
                                            let start = start.unwrap();
                                            (ChunkingStrategy::NumpressLinear { chunk_size: 50.0 })
                                                .decode_arrow(
                                                    &chunk_vals,
                                                    start as f64,
                                                    $accumulator.as_mut().unwrap(),
                                                    delta_model,
                                                );
                                        }
                                    }
                                };
                            }
                            if let Some(starts) = chunk_starts.as_primitive_opt::<Float64Type>() {
                                decode_numpress!(chunk_values, starts, main_axis);
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

        // If we never populated the main axis, exit early and return empty arrays.
        if main_axis.is_none() {
            for k in spectrum_array_indices.entries.values() {
                bin_map.add(k.as_buffer_name().as_data_array(0));
            }
            return Ok(bin_map);
        }

        let main_axis = main_axis.unwrap();
        let n = main_axis.data_len()?;
        bin_map.add(main_axis);

        for (name, chunks) in buffers {
            let mut store = DataArray::from_name_type_size(
                &name.array_type,
                name.dtype,
                name.dtype.size_of() * n,
            );
            // log::debug!("Unpacking {name} from {} chunks", chunks.len());
            macro_rules! extend_array {
                ($buf:ident) => {
                    if $buf.null_count() > 0 {
                        for val in $buf.iter() {
                            store.push(val.unwrap_or_default()).unwrap();
                        }
                    } else {
                        store.extend($buf.values()).unwrap();
                    }
                };
            }
            for arr in chunks {
                if let Some(arr) = arr.as_list_opt::<i64>() {
                    match arr.value_type() {
                        DataType::Float32 => {
                            for arr in arr.iter().flatten() {
                                let buf: &Float32Array = arr.as_primitive();
                                extend_array!(buf);
                            }
                        }
                        DataType::Float64 => {
                            for arr in arr.iter().flatten() {
                                let buf: &Float64Array = arr.as_primitive();
                                extend_array!(buf);
                            }
                        }
                        DataType::Int32 => {
                            for arr in arr.iter().flatten() {
                                let buf: &Int32Array = arr.as_primitive();
                                extend_array!(buf);
                            }
                        }
                        DataType::Int64 => {
                            for arr in arr.iter().flatten() {
                                let buf: &Int64Array = arr.as_primitive();
                                extend_array!(buf);
                            }
                        }
                        DataType::UInt8 => {
                            for arr in arr.iter().flatten() {
                                let buf: &UInt8Array = arr.as_primitive();
                                extend_array!(buf);
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
            // log::debug!(
            //     "Unpacked {} values, main axis had {n}",
            //     store.data_len().unwrap()
            // );
            bin_map.add(store);
        }
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
        let PageQuery {
            row_group_indices: rg_idx_acc,
            pages,
        } = page_query.unwrap_or_else(|| query_indices.spectrum_chunk_index.query_pages(query));

        // Otherwise we must construct a more intricate read plan, first pruning rows and row groups
        // based upon the pages matched
        let first_row = if !pages.is_empty() {
            let mut rg_row_skip = 0;
            for i in 0..rg_idx_acc[0] {
                let rg = self.builder.metadata().row_group(i);
                rg_row_skip += rg.num_rows();
            }
            rg_row_skip
        } else {
            let mut bin_map = BinaryArrayMap::new();
            for k in spectrum_array_indices.entries.values() {
                bin_map.add(k.as_buffer_name().as_data_array(0));
            }
            return Ok(bin_map);
        };

        let rows = query_indices
            .spectrum_point_index
            .spectrum_index
            .pages_to_row_selection(pages.iter(), first_row);

        let predicate_mask =
            ProjectionMask::columns(self.builder.parquet_schema(), ["chunk.spectrum_index"]);

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

        let proj = ProjectionMask::columns(&self.builder.parquet_schema(), ["chunk"]);

        log::trace!("{query} @ chunk spread across row groups {rg_idx_acc:?}");

        let reader = self
            .builder
            .with_row_groups(rg_idx_acc)
            .with_row_selection(rows)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .with_projection(proj)
            .build()?;

        Self::decode_chunks(reader.flatten(), spectrum_array_indices, delta_model)
    }
}
