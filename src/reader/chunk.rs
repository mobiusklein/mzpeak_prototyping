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
    BufferName,
    buffer_descriptors::arrow_to_array_type,
    chunk_series::{
        BufferTransformDecoder, ChunkingStrategy, DELTA_ENCODE, NO_COMPRESSION, NUMPRESS_LINEAR,
    },
    filter::RegressionDeltaModel,
    peak_series::{ArrayIndex, BufferFormat, data_array_to_arrow_array},
    reader::{
        ReaderMetadata,
        index::{PageQuery, QueryIndex, RangeIndex, SpanDynNumeric},
        point::binary_search_arrow_index,
        visitor::AnyCURIEArray,
    },
};

#[allow(unused)]
pub(crate) struct DataChunkCache {
    pub(crate) row_group: RecordBatch,
    pub(crate) spectrum_index_range: SimpleInterval<u64>,
    pub(crate) spectrum_array_indices: Arc<ArrayIndex>,
    pub(crate) last_query_index: Option<u64>,
    pub(crate) last_query_span: Option<(usize, usize)>,
}

#[allow(unused)]
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
                    let dtype = arrow_to_array_type(&v.data_type).unwrap();
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

#[derive(Debug)]
pub struct SpectrumChunkReader<T: parquet::file::reader::ChunkReader + 'static> {
    builder: ParquetRecordBatchReaderBuilder<T>,
}

#[derive(Default, Debug)]
struct OneCache<T: PartialEq + Eq, U> {
    last_key: Option<T>,
    last_value: Option<U>,
}

impl<T: PartialEq + Eq, U> OneCache<T, U> {
    fn get<F: FnOnce() -> U>(&mut self, key: T, callback: F) -> &U {
        let key = Some(key);
        if self.last_key == key {
            return self.last_value.as_ref().unwrap();
        } else {
            self.last_key = key;
            self.last_value = Some(callback());
            return self.last_value.as_ref().unwrap();
        }
    }
}

impl<T: parquet::file::reader::ChunkReader + 'static> SpectrumChunkReader<T> {
    pub fn new(builder: ParquetRecordBatchReaderBuilder<T>) -> Self {
        Self { builder }
    }

    pub(crate) fn load_cache_block(
        self,
        index_range: SimpleInterval<u64>,
        metadata: &ReaderMetadata,
        query_indices: &QueryIndex,
    ) -> io::Result<DataChunkCache> {
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
        index_range: SimpleInterval<u64>,
        query_range: Option<SimpleInterval<f64>>,
        metadata: &ReaderMetadata,
        query_indices: &QueryIndex,
    ) -> io::Result<impl Iterator<Item = Result<RecordBatch, ArrowError>>> {
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
            let meta = self.builder.metadata();
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

        let sidx = format!("{}.spectrum_index", metadata.spectrum_array_indices.prefix);

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
            .with_row_groups(row_group_indices)
            .with_row_selection(rows)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .with_batch_size(4096)
            .build()?;

        let mut delta_model_cache: OneCache<u64, Option<RegressionDeltaModel<f64>>> =
            OneCache::default();

        let it = reader.map(move |batch| -> Result<RecordBatch, ArrowError> {
            let batch = batch?;
            let root = batch.column(0).as_struct();
            let mut buffers: HashMap<BufferName, Vec<ArrayRef>> = HashMap::new();
            let mut main_axis_buffers = Vec::new();
            let mut main_axis_starts = Vec::new();
            let mut chunk_encodings: Vec<_> = Vec::new();
            let mut spectrum_index = Vec::new();
            let mut main_axis = None;

            for (f, arr) in root.fields().iter().zip(root.columns()) {
                match f.name().as_str() {
                    "spectrum_index" => {
                        spectrum_index.push(arr.clone());
                    }
                    "chunk_encoding" => {
                        chunk_encodings = AnyCURIEArray::try_from(arr).unwrap().to_vec();
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
                                        arr.len(),
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

            // Accumulate the per-point spectrum association
            let mut spectrum_idx_acc: Vec<u64> =
                Vec::with_capacity(spectrum_index.iter().map(|v| v.len()).sum());

            for (name, chunk_values) in main_axis_buffers.drain(..) {
                for ((encoding, chunk_starts), spectrum_idxs) in chunk_encodings
                    .iter()
                    .copied()
                    .zip(main_axis_starts.iter())
                    .zip(
                        spectrum_index
                            .iter()
                            .map(|v| v.as_primitive::<UInt64Type>()),
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
                                    for (chunk_vals, (start, spectrum_idx)) in $chunk_values
                                        .iter()
                                        .zip($starts.iter().zip(spectrum_idxs.iter().flatten()))
                                    {
                                        let chunk_vals = chunk_vals.unwrap();
                                        let start = start.unwrap();
                                        let main_axis_size_before = main_axis
                                            .as_ref()
                                            .map(|a| a.data_len().unwrap_or_default())
                                            .unwrap_or_default();
                                        (ChunkingStrategy::Basic { chunk_size: 50.0 })
                                            .decode_arrow(
                                                &chunk_vals,
                                                start as f64,
                                                $accumulator.as_mut().unwrap(),
                                                None,
                                            );
                                        let main_axis_size_after = main_axis
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
                                    for (chunk_vals, (start, spectrum_idx)) in $chunk_values
                                        .iter()
                                        .zip($starts.iter().zip(spectrum_idxs.iter().flatten()))
                                    {
                                        if let Some(chunk_vals) = chunk_vals {
                                            let start = start.unwrap();
                                            let delta_model = delta_model_cache
                                                .get(spectrum_idx, || {
                                                    metadata.model_deltas_for(spectrum_idx as usize)
                                                });
                                            let main_axis_size_before = main_axis
                                                .as_ref()
                                                .map(|a| a.data_len().unwrap_or_default())
                                                .unwrap_or_default();
                                            (ChunkingStrategy::Delta { chunk_size: 50.0 })
                                                .decode_arrow(
                                                    &chunk_vals,
                                                    start as f64,
                                                    $accumulator.as_mut().unwrap(),
                                                    delta_model.as_ref(),
                                                );
                                            let main_axis_size_after = main_axis
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
                                    for (chunk_vals, (start, spectrum_idx)) in $chunk_values
                                        .iter()
                                        .zip($starts.iter().zip(spectrum_idxs.iter().flatten()))
                                    {
                                        if let Some(chunk_vals) = chunk_vals {
                                            let start = start.unwrap();
                                            let delta_model = delta_model_cache
                                                .get(spectrum_idx, || {
                                                    metadata.model_deltas_for(spectrum_idx as usize)
                                                });
                                            let main_axis_size_before = main_axis
                                                .as_ref()
                                                .map(|a| a.data_len().unwrap_or_default())
                                                .unwrap_or_default();
                                            (ChunkingStrategy::NumpressLinear { chunk_size: 50.0 })
                                                .decode_arrow(
                                                    &chunk_vals,
                                                    start as f64,
                                                    $accumulator.as_mut().unwrap(),
                                                    delta_model.as_ref(),
                                                );
                                            let main_axis_size_after = main_axis
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

            // let uniq_idxs = spectrum_index
            //     .iter()
            //     .map(|v| HashSet::from_iter(v.as_primitive::<UInt64Type>().iter()))
            //     .reduce(|prev, next| {
            //         let combo: HashSet<_> = prev.union(&next).copied().collect();
            //         combo
            //     })
            //     .unwrap_or_default();
            // let mut uniq_idxs: Vec<_> = uniq_idxs.into_iter().flatten().collect();
            // uniq_idxs.sort();
            // log::debug!("{uniq_idxs:?}");
            // log::debug!("{spectrum_idx_acc:?}");

            let axis = main_axis.unwrap();
            let buffer_name = BufferName::from_data_array(crate::BufferContext::Spectrum, &axis);
            let axis = data_array_to_arrow_array(&buffer_name, &axis).unwrap();

            let mut fields = Vec::with_capacity(buffers.len() + 1);
            fields.push(buffer_name.context.index_field());
            fields.push(buffer_name.to_field());

            let mut arrays = Vec::with_capacity(buffers.len() + 1);
            arrays.push(Arc::new(UInt64Array::from(spectrum_idx_acc)) as ArrayRef);
            arrays.push(axis);

            for (name, chunks) in buffers {
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
                        let chunks: Vec<&dyn Array> =
                            chunks.iter().map(|a| a as &dyn Array).collect();
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
                            let mut chunks_out: Vec<Arc<dyn Array>> =
                                Vec::with_capacity(chunks.len());
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

            if let Some(query_range) = query_range.as_ref() {
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

            let batch = RecordBatch::from(batch);
            Ok(batch)
        });

        Ok(Box::new(it))
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
            let mut chunk_encodings: Vec<_> = Vec::new();
            for (f, arr) in root.fields().iter().zip(root.columns()) {
                match f.name().as_str() {
                    "spectrum_index" => {
                        continue;
                    }
                    "chunk_encoding" => {
                        chunk_encodings = AnyCURIEArray::try_from(arr).unwrap().to_vec();
                        // let arr = arr.as_struct();
                        // chunk_encodings =
                        //     serde_arrow::from_arrow(arr.fields(), arr.columns()).unwrap();
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
            for k in spectrum_array_indices.iter() {
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
            for k in spectrum_array_indices.iter() {
                bin_map.add(k.as_buffer_name().as_data_array(0));
            }
            return Ok(bin_map);
        };

        let rows = query_indices
            .spectrum_point_index
            .spectrum_index
            .pages_to_row_selection(&pages, first_row);

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
