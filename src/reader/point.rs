use std::{collections::HashMap, io, sync::Arc};

use arrow::{
    array::{
        Array, AsArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
        StructArray, UInt8Array, UInt64Array,
    },
    datatypes::DataType,
};
use mzdata::spectrum::{ArrayType, BinaryArrayMap, DataArray};
use parquet::{
    arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder},
    file::reader::ChunkReader,
};

use crate::{
    filter::{RegressionDeltaModel, fill_nulls_for},
    peak_series::ArrayIndex,
};

pub(crate) fn binary_search_arrow_index(
    array: &UInt64Array,
    query: u64,
    begin: Option<usize>,
    end: Option<usize>,
) -> Option<(usize, usize)> {
    let mut lo = begin.unwrap_or(0);
    let n = array.len() as usize;
    let mut hi = end.unwrap_or(n);

    while hi != lo {
        let mid = (hi + lo) / 2;
        let found = array.value(mid);
        if found == query {
            let mut i = mid;
            while i > 0 && array.value(i) == query {
                i -= 1;
            }
            if array.value(i) != query {
                i += 1;
            }
            let begin = i;

            i = mid;
            while i < n && array.value(i) == query {
                i += 1;
            }
            let end = i;

            return Some((begin, end));
        } else if hi - lo == 1 {
            return None;
        } else if found > query {
            hi = mid;
        } else {
            lo = mid;
        }
    }

    None
}

pub(crate) trait SpectrumDataArrayReader {
    fn populate_arrays_from_struct_array(
        &self,
        points: &StructArray,
        bin_map: &mut HashMap<&String, DataArray>,
        mz_delta_model: Option<&RegressionDeltaModel<f64>>,
    ) {
        for (f, arr) in points.fields().iter().zip(points.columns()) {
            if f.name() == "spectrum_index" || f.name() == "spectrum_time" {
                continue;
            }
            let store = bin_map.get_mut(f.name()).unwrap();

            let has_nulls = arr.null_count() > 0;
            let is_mz_array = matches!(store.name, ArrayType::MZArray);

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

            match f.data_type() {
                DataType::Float32 => {
                    let buf: &Float32Array = arr.as_primitive();
                    if has_nulls {
                        if is_mz_array {
                            if let Some(mz_delta_model) = mz_delta_model {
                                let interpolated = fill_nulls_for(buf, mz_delta_model);
                                store.extend(&interpolated).unwrap();
                                continue;
                            }
                        }
                    }
                    extend_array!(buf);
                }
                DataType::Float64 => {
                    let buf: &Float64Array = arr.as_primitive();
                    if has_nulls {
                        if is_mz_array {
                            if let Some(mz_delta_model) = mz_delta_model {
                                let interpolated = fill_nulls_for(buf, mz_delta_model);
                                store.extend(&interpolated).unwrap();
                                continue;
                            }
                        }
                    }
                    extend_array!(buf);
                }
                DataType::Int32 => {
                    let buf: &Int32Array = arr.as_primitive();
                    extend_array!(buf);
                }
                DataType::Int64 => {
                    let buf: &Int64Array = arr.as_primitive();
                    extend_array!(buf);
                }
                DataType::UInt8 => {
                    let buf: &UInt8Array = arr.as_primitive();
                    extend_array!(buf);
                }
                DataType::LargeUtf8 => {}
                DataType::Utf8 => {}
                _ => {}
            }
        }
    }

    /// Read a specific Parquet row group into memory as a single [`RecordBatch`]
    ///
    /// This reads from the spectrum data file
    fn load_spectrum_data_row_group<T: ChunkReader + 'static>(
        &self,
        builder: ParquetRecordBatchReaderBuilder<T>,
        row_group: usize,
    ) -> io::Result<RecordBatch> {
        log::trace!("Loading row group {row_group}");
        let schema = builder.parquet_schema();
        let leaves=  schema
                    .columns()
                    .iter()
                    .enumerate()
                    .filter_map(|(i, f)| if f.path().string() != "point.spectrum_time" { Some(i)} else { None });
        let mask = ProjectionMask::leaves(
            schema,
            leaves,
        );

        let batch = builder
            .with_row_groups(vec![row_group])
            .with_projection(mask)
            .with_batch_size(usize::MAX)
            .build()?
            .flatten()
            .next();
        if let Some(batch) = batch {
            Ok(batch)
        } else {
            Err(parquet::errors::ParquetError::General(format!(
                "Couldn't read row group {row_group}"
            ))
            .into())
        }
    }
}

pub(crate) struct SpectrumDataPointCache {
    pub(crate) row_group: RecordBatch,
    pub(crate) row_group_index: usize,
    pub(crate) spectrum_array_indices: Arc<ArrayIndex>,
    pub(crate) last_query_index: Option<u64>,
    pub(crate) last_query_span: Option<(usize, usize)>,
}

impl SpectrumDataArrayReader for SpectrumDataPointCache {}

impl SpectrumDataPointCache {
    pub(crate) fn new(
        row_group: RecordBatch,
        spectrum_array_indices: Arc<ArrayIndex>,
        row_group_index: usize,
        last_query_index: Option<u64>,
        last_query_span: Option<(usize, usize)>,
    ) -> Self {
        Self {
            row_group,
            spectrum_array_indices,
            row_group_index,
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

        let points = self.row_group.column(0).as_struct();
        let indices: &UInt64Array = points
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
        let mut bin_map = HashMap::new();
        for v in self.spectrum_array_indices.iter() {
            bin_map.insert(&v.name, v.as_buffer_name().as_data_array(0));
        }

        let (start, end) = self.find_span_for_query(index);

        if !(start.is_some() && end.is_some()) {
            panic!("Could not find start and end in binary search");
            // for (i, idx) in indices.iter().enumerate() {
            //     if idx.unwrap() == index {
            //         if start.is_some() {
            //             end = Some(i + 1)
            //         } else {
            //             start = Some(i)
            //         }
            //     }
            // }
        }

        let points = self.row_group.column(0).as_struct();

        let points = match (start, end) {
            (Some(start), Some(end)) => {
                let len = end - start;
                self.last_query_span = Some((start, end));
                self.last_query_index = Some(index);
                points.slice(start, len)
            }
            (Some(start), None) => {
                self.last_query_span = Some((start, start + 1));
                self.last_query_index = Some(index);
                points.slice(start, 1)
            }
            _ => {
                let mut out = BinaryArrayMap::new();
                for v in bin_map.into_values() {
                    out.add(v);
                }
                return Ok(out);
            }
        };

        self.populate_arrays_from_struct_array(&points, &mut bin_map, mz_delta_model);

        let mut out = BinaryArrayMap::new();
        for v in bin_map.into_values() {
            out.add(v);
        }
        Ok(out)
    }
}
