use std::{collections::HashMap, fmt::Debug, io, sync::Arc};

use arrow::{
    array::{
        Array, AsArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch, StructArray, UInt64Array, UInt8Array
    },
    datatypes::DataType, error::ArrowError,
};
use mzdata::{
    prelude::BuildFromArrayMap,
    spectrum::{ArrayType, BinaryArrayMap, DataArray, PeakDataLevel},
};
use mzpeaks::{coordinate::SimpleInterval, CentroidLike, DeconvolutedCentroidLike};
use parquet::{
    arrow::{
        arrow_reader::{ArrowPredicateFn, ParquetRecordBatchReader, ParquetRecordBatchReaderBuilder, RowFilter, RowSelection}, ProjectionMask
    },
    file::reader::ChunkReader,
};

use crate::{
    filter::{fill_nulls_for, RegressionDeltaModel},
    peak_series::ArrayIndex,
    reader::{index::{PageQuery, SpanDynNumeric, SpectrumQueryIndex}, metadata::PeakMetadata}, BufferContext,
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

/// An internal shared behavior set for reading point-layout data
pub(crate) trait PointDataArrayReader {

    /// Read a [`StructArray`] of parallel array values into a map of [`DataArray`] instances.
    ///
    /// If `incremental` is not true, assume we have all the information available and skip work
    /// on completely null arrays.
    fn populate_arrays_from_struct_array(
        points: &StructArray,
        bin_map: &mut HashMap<&String, DataArray>,
        mz_delta_model: Option<&RegressionDeltaModel<f64>>,
        incremental: bool
    ) {
        for (f, arr) in points.fields().iter().zip(points.columns()) {
            if f.name() == "spectrum_index" || f.name() == "spectrum_time" || f.name() == "chromatogram_index" {
                continue;
            }

            if arr.null_count() == arr.len() && !incremental {
                bin_map.remove(f.name());
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
    /// This may potentially use a lot of memory if row groups are large.
    fn load_cache_block<T: ChunkReader + 'static>(
        &self,
        builder: ParquetRecordBatchReaderBuilder<T>,
        row_group: usize,
    ) -> io::Result<RecordBatch> {
        log::trace!("Loading row group {row_group}");
        let schema = builder.parquet_schema();
        let leaves = schema.columns().iter().enumerate().filter_map(|(i, f)| {
            if f.path().string() != "point.spectrum_time" {
                log::trace!("Adding {f:?} to the point cache");
                Some(i)
            } else {
                None
            }
        });
        let mask = ProjectionMask::leaves(schema, leaves);

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

/// An internal data structure for caching a [`RecordBatch`] corresponding to a complete
/// row group in memory, and reading out slices of the batch. This helps avoid repeated re-parsing
/// of the Parquet file.
pub(crate) struct DataPointCache {
    pub(crate) row_group: RecordBatch,
    pub(crate) row_group_index: usize,
    pub(crate) spectrum_array_indices: Arc<ArrayIndex>,
    pub(crate) last_query_index: Option<u64>,
    pub(crate) last_query_span: Option<(usize, usize)>,
    pub(crate) buffer_context: BufferContext,
}

impl PointDataArrayReader for DataPointCache {}

impl DataPointCache {
    pub(crate) fn new(
        row_group: RecordBatch,
        spectrum_array_indices: Arc<ArrayIndex>,
        row_group_index: usize,
        last_query_index: Option<u64>,
        last_query_span: Option<(usize, usize)>,
        buffer_context: BufferContext,
    ) -> Self {
        Self {
            row_group,
            spectrum_array_indices,
            row_group_index,
            last_query_index,
            last_query_span,
            buffer_context,
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
            .column_by_name(self.buffer_context.index_name())
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

        Self::populate_arrays_from_struct_array(&points, &mut bin_map, mz_delta_model, false);

        let mut out = BinaryArrayMap::new();
        for v in bin_map.into_values() {
            out.add(v);
        }
        Ok(out)
    }
}


/// A facet that wraps the behavior for reading point-layout data.
pub(crate) struct PointDataReader<T: ChunkReader + 'static>(pub(crate) ParquetRecordBatchReaderBuilder<T>, pub(crate) BufferContext);

impl<T: ChunkReader + 'static> PointDataArrayReader for PointDataReader<T> {}

impl<T: ChunkReader + 'static> PointDataReader<T> {

    pub(crate) fn find_row_groups_query<'a, I: SpectrumQueryIndex + 'a>(&self, index: u64, query_index: &'a I) -> (RowSelection, Vec<usize>) {
        let PageQuery {
            pages,
            row_group_indices,
        } = query_index.query_pages(index);

        // Find which row groups we need to touch and the first possible row to read from relative to the start of the table
        // because all `RowSelection` offsets are w.r.t. the row groups read, not the total possible rows in the table.
        let first_row = if !pages.is_empty() {
            let mut rg_row_skip = 0;
            let meta = self.0.metadata();
            for i in 0..row_group_indices[0] {
                let rg = meta.row_group(i);
                rg_row_skip += rg.num_rows();
            }
            rg_row_skip
        } else {
            0
        };

        let rows = query_index
            .spectrum_data_index()
            .pages_to_row_selection(&pages, first_row);

        (rows, row_group_indices)
    }

    /// Read the arrays associated with the points of `index`
    pub(crate) fn read_points_of<'a, I: SpectrumQueryIndex + Debug + 'a>(self, index: u64, query_index: &'a I, array_indices: &'a ArrayIndex) -> io::Result<Option<BinaryArrayMap>> {
        let (rows, row_group_indices) = self.find_row_groups_query(index, query_index);
        let predicate_mask = ProjectionMask::columns(
            self.0.parquet_schema(),
            [
                match self.1 {
                    BufferContext::Spectrum => format!("{}.{}", array_indices.prefix, self.1.index_name()),
                    BufferContext::Chromatogram => format!("{}.{}", array_indices.prefix, self.1.index_name())
                }.as_str()
            ],
        );

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
                .map(|val| val.is_some_and(|val| val == index));

            Ok(it.map(Some).collect())
        });

        let proj = ProjectionMask::columns(
            &self.0.parquet_schema(),
            [array_indices.prefix.as_str()],
        );

        log::trace!("{index} spread across row groups {row_group_indices:?}");

        let reader = self
            .0
            .with_row_groups(row_group_indices)
            .with_row_selection(rows)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .with_projection(proj)
            .build()?;

        let mut bin_map = HashMap::new();
        for v in array_indices.iter() {
            bin_map.insert(&v.name, v.as_buffer_name().as_data_array(1024));
        }

        let batches: Vec<_> = reader.flatten().collect();
        if !batches.is_empty() {
            let batch = arrow::compute::concat_batches(batches[0].schema_ref(), &batches).unwrap();
            let points = batch.column(0).as_struct();
            Self::populate_arrays_from_struct_array(points, &mut bin_map, None, false);
        }

        let mut out = BinaryArrayMap::new();
        for v in bin_map.into_values() {
            out.add(v);
        }
        Ok(Some(out))
    }

    pub(crate) fn get_peak_list_for<
        C: CentroidLike + BuildFromArrayMap,
        D: DeconvolutedCentroidLike + BuildFromArrayMap,
    >(
        self,
        index: u64,
        meta_index: &PeakMetadata,
    ) -> io::Result<Option<PeakDataLevel<C, D>>> {
        let out = self.read_points_of(index, &meta_index.query_index, &meta_index.array_indices)?;
        match out {
            Some(out) => {
                match PeakDataLevel::try_from(&out) {
                    Ok(val) => return Ok(Some(val)),
                    Err(e) => return Err(e.into()),
                }
            },
            None => Ok(None)
        }
    }

    pub(crate) fn query_points<'a, I: SpectrumQueryIndex + 'a>(
        self,
        index_range: SimpleInterval<u64>,
        mz_range: Option<SimpleInterval<f64>>,
        ion_mobility_range: Option<SimpleInterval<f64>>,
        query_index: &'a I,
        array_indices: &'a ArrayIndex,
    ) -> io::Result<Box<dyn Iterator<Item = Result<RecordBatch, ArrowError>> + 'a>> {
        let mut rows = query_index
            .index_overlaps(&index_range);

        let PageQuery { row_group_indices, pages } = query_index.query_pages_overlaps(&index_range);

        if pages.is_empty() {
            return Ok(Box::new(std::iter::empty()));
        }

        let mut up_to_first_row = 0;
        let meta = self.0.metadata();
        for i in 0..row_group_indices[0] {
            let rg = meta.row_group(i);
            up_to_first_row += rg.num_rows();
        }

        if let Some(mz_range) = mz_range.as_ref() {
            rows = rows.intersection(
                &query_index.mz_overlaps(mz_range),
            );
        }

        if let Some(ion_mobility_range) = ion_mobility_range.as_ref() {
            rows = rows.union(
                &query_index
                    .im_overlaps(&ion_mobility_range),
            );
        }

        rows.split_off(up_to_first_row as usize);

        let sidx = format!(
            "{}.spectrum_index",
            array_indices.prefix
        );

        let mut fields: Vec<&str> = Vec::new();

        fields.push(&sidx);

        if let Some(e) = array_indices
            .get(&ArrayType::MZArray)
        {
            fields.push(e.path.as_str());
        }

        if let Some(e) = array_indices
            .get(&ArrayType::IntensityArray)
        {
            fields.push(e.path.as_str());
        }

        for v in array_indices.iter() {
            if v.is_ion_mobility() {
                fields.push(v.path.as_str());
                break;
            }
        }

        let proj = ProjectionMask::columns(self.0.parquet_schema(), fields.iter().copied());
        let predicate_mask = proj.clone();

        let predicate = ArrowPredicateFn::new(predicate_mask, move |batch| {
            let root = batch.column(0).as_struct();
            let it = index_range.contains_dy(root.column(0));

            match (mz_range, ion_mobility_range) {
                (None, None) => Ok(it),
                (None, Some(ion_mobility_range)) => {
                    let im_array = root.column(1);
                    let it2 = ion_mobility_range.contains_dy(im_array);
                    arrow::compute::and(&it, &it2)
                }
                (Some(mz_range), None) => {
                    let mz_array = root.column(1);
                    let it2 = mz_range.contains_dy(mz_array);
                    arrow::compute::and(&it, &it2)
                }
                (Some(mz_range), Some(ion_mobility_range)) => {
                    let mz_array = root.column(1);
                    let im_array = root.column(2);
                    let it2 = mz_range.contains_dy(mz_array);
                    let it3 = ion_mobility_range.contains_dy(im_array);
                    arrow::compute::and(&arrow::compute::and(&it2, &it3)?, &it)
                }
            }
        });

        let reader: ParquetRecordBatchReader = self.0
            .with_row_groups(row_group_indices)
            .with_row_selection(rows)
            .with_projection(proj)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .with_batch_size(10_000)
            .build()?;

        Ok(Box::new(reader))
    }
}
