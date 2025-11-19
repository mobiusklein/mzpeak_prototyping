use std::{
    collections::{HashMap, VecDeque},
    fmt::Debug,
    io,
    sync::Arc,
};

use arrow::{
    array::{
        Array, ArrayRef, AsArray, Float32Array, Float64Array, Int32Array, Int64Array, RecordBatch,
        RecordBatchReader, StructArray, UInt8Array, UInt64Array,
    },
    datatypes::{DataType, Float32Type, Float64Type, SchemaRef},
    error::ArrowError,
};
use mzdata::{
    prelude::BuildFromArrayMap,
    spectrum::{ArrayType, BinaryArrayMap, DataArray, PeakDataLevel},
};
use mzpeaks::{CentroidLike, DeconvolutedCentroidLike, coordinate::SimpleInterval};
use parquet::{
    arrow::{
        ProjectionMask,
        arrow_reader::{
            ArrowPredicate, ArrowPredicateFn, ArrowReaderBuilder, ParquetRecordBatchReader,
            ParquetRecordBatchReaderBuilder, RowFilter, RowSelection,
        },
    },
    file::{metadata::ParquetMetaData, reader::ChunkReader},
    schema::types::SchemaDescriptor,
};

#[cfg(feature = "async")]
use parquet::arrow::async_reader::{AsyncFileReader, ParquetRecordBatchStreamBuilder};

use crate::{
    BufferContext,
    filter::{RegressionDeltaModel, fill_nulls_for},
    peak_series::ArrayIndex,
    reader::{
        ReaderMetadata,
        index::{PageQuery, SpanDynNumeric, SpectrumQueryIndex},
        metadata::PeakMetadata,
    },
};

use super::utils::MaskSet;

#[cfg(feature = "async")]
use futures::StreamExt;

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
        incremental: bool,
    ) {
        for (f, arr) in points.fields().iter().zip(points.columns()) {
            if f.name() == BufferContext::Spectrum.index_name()
                || f.name() == BufferContext::Spectrum.time_name()
                || f.name() == BufferContext::Chromatogram.index_name()
            {
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

    fn configure_cache_block_reader<T>(
        &self,
        builder: ArrowReaderBuilder<T>,
        row_group: usize,
    ) -> ArrowReaderBuilder<T> {
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
            .with_batch_size(usize::MAX);

        batch
    }

    #[cfg(feature = "async")]
    fn load_cache_block_async<T: AsyncFileReader + Unpin + Send + 'static>(
        &self,
        builder: ParquetRecordBatchStreamBuilder<T>,
        row_group: usize,
    ) -> impl Future<Output = io::Result<RecordBatch>> {
        let builder = self.configure_cache_block_reader(builder, row_group);
        async move {
            let mut stream = match builder.build() {
                Ok(stream) => stream,
                Err(e) => return Err(e.into()),
            };
            let batch = stream.next().await.transpose()?;
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

    /// Read a specific Parquet row group into memory as a single [`RecordBatch`]
    ///
    /// This may potentially use a lot of memory if row groups are large.
    fn load_cache_block<T: ChunkReader + 'static>(
        &self,
        builder: ParquetRecordBatchReaderBuilder<T>,
        row_group: usize,
    ) -> io::Result<RecordBatch> {
        let builder = self.configure_cache_block_reader(builder, row_group);

        let batch = builder.build()?.flatten().next();
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

trait PointQuerySource {
    fn metadata(&self) -> &ParquetMetaData;

    fn parquet_schema(&self) -> Arc<SchemaDescriptor>;

    fn find_row_groups_query<'a, I: SpectrumQueryIndex + 'a>(
        &self,
        index: u64,
        query_index: &'a I,
    ) -> (RowSelection, Vec<usize>) {
        let PageQuery {
            pages,
            row_group_indices,
        } = query_index.query_pages(index);

        // Find which row groups we need to touch and the first possible row to read from relative to the start of the table
        // because all `RowSelection` offsets are w.r.t. the row groups read, not the total possible rows in the table.
        let first_row = if !pages.is_empty() {
            let mut rg_row_skip = 0;
            let meta = self.metadata();
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

    fn prepare_points_of<'a>(
        schema: Arc<SchemaDescriptor>,
        index: u64,
        array_indices: &'a ArrayIndex,
        context: BufferContext,
    ) -> (
        ArrowPredicateFn<
            impl FnMut(RecordBatch) -> Result<arrow::array::BooleanArray, ArrowError> + 'static,
        >,
        ProjectionMask,
    ) {
        let predicate_mask = ProjectionMask::columns(
            &schema,
            [match context {
                BufferContext::Spectrum => {
                    format!("{}.{}", array_indices.prefix, context.index_name())
                }
                BufferContext::Chromatogram => {
                    format!("{}.{}", array_indices.prefix, context.index_name())
                }
            }
            .as_str()],
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

        let proj = ProjectionMask::columns(&schema, [array_indices.prefix.as_str()]);
        (predicate, proj)
    }

    fn buffer_context(&self) -> BufferContext;

    fn prepare_predicate_for_index<'a>(
        &self,
        index_range: MaskSet,
        array_indices: &'a ArrayIndex,
    ) -> Box<dyn ArrowPredicate> {
        let sidx = format!(
            "{}.{}",
            array_indices.prefix,
            self.buffer_context().index_name()
        );
        let predicate_mask = ProjectionMask::columns(&self.parquet_schema(), [sidx.as_str()]);

        let predicate = ArrowPredicateFn::new(predicate_mask, move |batch| {
            let root = batch.column(0).as_struct();
            let it = index_range.contains_dy(root.column(0));
            Ok(it)
        });
        Box::new(predicate)
    }

    fn prepare_predicate_for_mz<'a>(
        &self,
        mz_range: SimpleInterval<f64>,
        array_indices: &'a ArrayIndex,
    ) -> Option<Box<dyn ArrowPredicate>> {
        if let Some(e) = array_indices.get(&ArrayType::MZArray) {
            let predicate_mask = ProjectionMask::columns(&self.parquet_schema(), [e.path.as_str()]);
            Some(Box::new(ArrowPredicateFn::new(
                predicate_mask,
                move |batch| {
                    let root = batch.column(0).as_struct();
                    let it = mz_range.contains_dy(root.column(0));
                    Ok(it)
                },
            )))
        } else {
            None
        }
    }

    fn prepare_predicate_for_ion_mobility<'a>(
        &self,
        ion_mobility_range: SimpleInterval<f64>,
        array_indices: &'a ArrayIndex,
    ) -> Option<Box<dyn ArrowPredicate>> {
        if let Some(e) = array_indices.iter().find(|e| e.is_ion_mobility()) {
            let predicate_mask = ProjectionMask::columns(&self.parquet_schema(), [e.path.as_str()]);
            Some(Box::new(ArrowPredicateFn::new(
                predicate_mask,
                move |batch| {
                    let root = batch.column(0).as_struct();
                    let it = ion_mobility_range.contains_dy(root.column(0));
                    Ok(it)
                },
            )))
        } else {
            None
        }
    }

    fn prepare_query<'a, I: SpectrumQueryIndex + 'a>(
        &self,
        index_range: MaskSet,
        mz_range: Option<SimpleInterval<f64>>,
        ion_mobility_range: Option<SimpleInterval<f64>>,
        query_index: &'a I,
        array_indices: &'a ArrayIndex,
        query: Option<PageQuery>,
    ) -> Option<(RowSelection, Vec<usize>, ProjectionMask, RowFilter)> {
        let mut rows = query_index.index_overlaps(&index_range.range);

        let query = query.unwrap_or_else(|| query_index.query_pages_overlaps(&index_range));

        if query.is_empty() {
            return None;
        }

        let up_to_first_row = query.get_num_rows_to_skip_for_row_groups(self.metadata());

        let PageQuery {
            row_group_indices,
            pages: _,
        } = query;

        if let Some(mz_range) = mz_range.as_ref() {
            rows = rows.intersection(&query_index.mz_overlaps(mz_range));
        }

        // if let Some(ion_mobility_range) = ion_mobility_range.as_ref() {
        //     let im_rows = query_index.ion_mobility_overlaps(&ion_mobility_range);
        //     rows = rows.union(&im_rows);
        // }

        rows.split_off(up_to_first_row as usize);

        let sidx = format!(
            "{}.{}",
            array_indices.prefix,
            self.buffer_context().index_name()
        );

        let mut fields = Vec::new();

        fields.push(sidx);

        if let Some(e) = array_indices.get(&ArrayType::MZArray) {
            fields.push(e.path.to_string());
        }

        if let Some(e) = array_indices.get(&ArrayType::IntensityArray) {
            fields.push(e.path.to_string());
        }

        for v in array_indices.iter() {
            if v.is_ion_mobility() {
                fields.push(v.path.to_string());
                break;
            }
        }

        let proj =
            ProjectionMask::columns(&self.parquet_schema(), fields.iter().map(|s| s.as_str()));

        let mut predicates: Vec<Box<dyn ArrowPredicate>> =
            vec![self.prepare_predicate_for_index(index_range, array_indices)];
        if let Some(mz_range) = mz_range {
            predicates.extend(self.prepare_predicate_for_mz(mz_range, array_indices));
        }
        if let Some(ion_mobility_range) = ion_mobility_range {
            predicates
                .extend(self.prepare_predicate_for_ion_mobility(ion_mobility_range, array_indices));
        }

        let row_filter = RowFilter::new(predicates);

        Some((rows, row_group_indices, proj, row_filter))
    }
}

#[cfg(feature = "async")]
mod async_impl {
    use super::*;

    use arrow::array::RecordBatchIterator;
    use futures::stream::BoxStream;

    pub(crate) struct AsyncPointDataReader<T: AsyncFileReader + Unpin + Send + 'static>(
        pub(crate) ParquetRecordBatchStreamBuilder<T>,
        pub(crate) BufferContext,
    );

    impl<T: AsyncFileReader + Unpin + Send + 'static> PointQuerySource for AsyncPointDataReader<T> {
        fn metadata(&self) -> &ParquetMetaData {
            self.0.metadata()
        }

        fn parquet_schema(&self) -> Arc<SchemaDescriptor> {
            self.0.metadata().file_metadata().schema_descr_ptr()
        }

        fn buffer_context(&self) -> BufferContext {
            self.1
        }
    }

    impl<T: AsyncFileReader + Unpin + Send + 'static> PointDataArrayReader for AsyncPointDataReader<T> {}

    impl<T: AsyncFileReader + Unpin + Send + 'static> AsyncPointDataReader<T> {
        /// Read the arrays associated with the points of `index`
        pub(crate) async fn read_points_of<'a, I: SpectrumQueryIndex + Debug + 'a>(
            self,
            index: u64,
            query_index: &'a I,
            array_indices: &'a ArrayIndex,
            mz_delta_model: Option<&RegressionDeltaModel<f64>>,
        ) -> io::Result<Option<BinaryArrayMap>> {
            let (rows, row_group_indices) = self.find_row_groups_query(index, query_index);
            let schem = self.parquet_schema();
            let (predicate, proj) = Self::prepare_points_of(schem, index, array_indices, self.1);

            log::trace!("{index} spread across row groups {row_group_indices:?}");

            let mut reader = self
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

            let mut batches = Vec::new();
            while let Some(batch) = reader.next().await.transpose()? {
                batches.push(batch);
            }

            if !batches.is_empty() {
                let batch =
                    arrow::compute::concat_batches(batches[0].schema_ref(), &batches).unwrap();
                let points = batch.column(0).as_struct();
                Self::populate_arrays_from_struct_array(
                    points,
                    &mut bin_map,
                    mz_delta_model,
                    false,
                );
            }

            let mut out = BinaryArrayMap::new();
            for v in bin_map.into_values() {
                out.add(v);
            }
            Ok(Some(out))
        }

        pub(crate) async fn get_peak_list_for<
            C: CentroidLike + BuildFromArrayMap,
            D: DeconvolutedCentroidLike + BuildFromArrayMap,
        >(
            self,
            index: u64,
            meta_index: &PeakMetadata,
        ) -> io::Result<Option<PeakDataLevel<C, D>>> {
            let out = self
                .read_points_of(
                    index,
                    &meta_index.query_index,
                    &meta_index.array_indices,
                    None,
                )
                .await?;
            match out {
                Some(out) => match PeakDataLevel::try_from(&out) {
                    Ok(val) => return Ok(Some(val)),
                    Err(e) => return Err(e.into()),
                },
                None => Ok(None),
            }
        }

        pub(crate) async fn query_points<'a, I: SpectrumQueryIndex + 'a>(
            self,
            index_range: MaskSet,
            mz_range: Option<SimpleInterval<f64>>,
            ion_mobility_range: Option<SimpleInterval<f64>>,
            query_index: &'a I,
            array_indices: &'a ArrayIndex,
            metadata: &'a ReaderMetadata,
        ) -> io::Result<BoxStream<'a, Result<RecordBatch, ArrowError>>> {
            if let Some((rows, row_group_indices, proj, predicate)) = self.prepare_query(
                index_range,
                mz_range,
                ion_mobility_range,
                query_index,
                array_indices,
                None,
            ) {
                let schema = self.0.schema().clone();
                let (_, subset) = schema.column_with_name(&array_indices.prefix).unwrap();
                let subset = match subset.data_type() {
                    DataType::Struct(subset) => subset,
                    _ => panic!("Invalid point type"),
                };

                let context = self.1;

                let mut index_column_idx = None;
                let mut mz_column_idx = None;

                if matches!(context, BufferContext::Spectrum) {
                    let subset = arrow::datatypes::Schema::new(subset.clone());
                    index_column_idx = subset
                        .column_with_name(BufferContext::Spectrum.index_name())
                        .map(|(i, _)| i);
                    if let Some(mz_array_idx) = array_indices.get(&ArrayType::MZArray) {
                        mz_column_idx = subset
                            .column_with_name(&mz_array_idx.path.split(".").last().unwrap())
                            .map(|(i, _)| i);
                    }
                }

                let mut reader = self
                    .0
                    .with_row_groups(row_group_indices)
                    .with_row_selection(rows)
                    .with_projection(proj)
                    .with_row_filter(predicate)
                    .with_batch_size(10_000)
                    .build()?;

                let (send, recv) = tokio::sync::mpsc::unbounded_channel();

                let mut row_groups = futures::stream::FuturesOrdered::new();

                while let Some(batch_reader) = reader.next_row_group().await? {
                    row_groups.push_back(tokio::task::spawn_blocking(|| {
                        let batches: Vec<_> = batch_reader.collect();
                        batches
                    }));
                }

                while let Some(bats) = row_groups.next().await.transpose()? {
                    if !matches!(context, BufferContext::Spectrum)
                        || index_column_idx.is_none()
                        || mz_column_idx.is_none()
                    {
                        for bat in bats {
                            send.send(bat).unwrap();
                        }
                    } else {
                        let it = InterpolateIter::new(
                            RecordBatchIterator::new(bats.into_iter(), schema.clone()),
                            metadata,
                            index_column_idx.unwrap(),
                            mz_column_idx.unwrap(),
                        );
                        for bat in it {
                            send.send(bat).unwrap();
                        }
                    }
                }

                let reader = tokio_stream::wrappers::UnboundedReceiverStream::new(recv);
                Ok(reader.boxed())
            } else {
                Ok(futures::stream::empty().boxed())
            }
        }
    }
}

#[cfg(feature = "async")]
pub(crate) use async_impl::*;

#[derive(Debug)]
pub(crate) struct IndexSplittingIter {
    source: VecDeque<RecordBatch>,
    schema: SchemaRef,
}

impl Iterator for IndexSplittingIter {
    type Item = Result<RecordBatch, arrow::error::ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next()
    }
}

impl IndexSplittingIter {
    #[allow(unused)]
    pub fn new(
        batch: RecordBatch,
        spectrum_index_array_idx: usize,
    ) -> Result<Self, arrow::error::ArrowError> {
        let root = batch.column(0);
        let root = root.as_struct();
        let indices = root.column(spectrum_index_array_idx);
        let parts = arrow::compute::partition(std::slice::from_ref(indices))?;
        let slices = parts.ranges();
        let source = slices
            .into_iter()
            .map(|batch_idx| batch.slice(batch_idx.start, batch_idx.end - batch_idx.start))
            .collect();
        Ok(Self {
            source,
            schema: batch.schema(),
        })
    }

    fn next(&mut self) -> Option<Result<RecordBatch, arrow::error::ArrowError>> {
        self.source.pop_front().map(Ok)
    }

    pub fn empty(schema: SchemaRef) -> Self {
        Self {
            source: Default::default(),
            schema,
        }
    }

    pub fn add_and_split(
        &mut self,
        batch: RecordBatch,
        spectrum_index_array_idx: usize,
    ) -> Result<(), ArrowError> {
        let root = batch.column(0);
        let root = root.as_struct();
        let indices = root.column(spectrum_index_array_idx);
        let parts = arrow::compute::partition(std::slice::from_ref(indices))?;
        let slices = parts.ranges();
        self.source.extend(
            slices
                .into_iter()
                .map(|batch_idx| batch.slice(batch_idx.start, batch_idx.end - batch_idx.start)),
        );
        Ok(())
    }
}

impl RecordBatchReader for IndexSplittingIter {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

pub(crate) struct BatchIterpolater<'a> {
    metadata: &'a ReaderMetadata,
    spectrum_index_idx: usize,
    mz_array_idx: usize,
}

impl<'a> BatchIterpolater<'a> {
    pub fn new(
        metadata: &'a ReaderMetadata,
        spectrum_index_idx: usize,
        mz_array_idx: usize,
    ) -> Self {
        Self {
            metadata,
            spectrum_index_idx,
            mz_array_idx,
        }
    }

    fn check_batch_has_nulls(&self, batch: &RecordBatch) -> bool {
        let root = batch.column(0);
        let root_as = root.as_struct();
        let mz_arr = root_as.column(self.mz_array_idx);
        mz_arr.null_count() > 0
    }

    fn process_batch(&mut self, batch: RecordBatch) -> Result<RecordBatch, ArrowError> {
        let root = batch.column(0);
        let root_as = root.as_struct();
        let index_arr: &UInt64Array = root_as.column(self.spectrum_index_idx).as_primitive();
        let mz_arr = root_as.column(self.mz_array_idx);

        if index_arr.is_empty() {
            return Ok(batch);
        }

        // Assume that the batch is a single index wide
        let spec_index = index_arr.value(0);
        let model = match self.metadata.model_deltas_for(spec_index as usize) {
            Some(model) => model,
            None => return Ok(batch),
        };

        let mz_arr = if let Some(mz_arr) = mz_arr.as_primitive_opt::<Float32Type>() {
            let mz_arr: Float32Array = fill_nulls_for(mz_arr, &model).into();
            Arc::new(mz_arr) as ArrayRef
        } else if let Some(mz_arr) = mz_arr.as_primitive_opt::<Float64Type>() {
            let mz_arr: Float64Array = fill_nulls_for(mz_arr, &model).into();
            Arc::new(mz_arr) as ArrayRef
        } else {
            todo!()
        };

        let mut cols: Vec<_> = root_as.columns().iter().cloned().collect();
        cols[self.mz_array_idx] = mz_arr;
        let new_root: ArrayRef = Arc::new(StructArray::new(
            root_as.fields().clone(),
            cols,
            root_as.nulls().cloned(),
        ));

        let (schema, mut batch_parts, _n_rows) = batch.into_parts();
        batch_parts[0] = new_root;
        let batch = RecordBatch::try_new(schema, batch_parts).unwrap();
        return Ok(batch);
    }
}

pub struct InterpolateIter<'a, I: Iterator<Item = Result<RecordBatch, ArrowError>>> {
    source: I,
    spectrum_index_idx: usize,
    interpolator: BatchIterpolater<'a>,
    splitter: IndexSplittingIter,
}

impl<'a, I: Iterator<Item = Result<RecordBatch, ArrowError>> + RecordBatchReader> Iterator
    for InterpolateIter<'a, I>
{
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch()
    }
}

impl<'a, I: Iterator<Item = Result<RecordBatch, ArrowError>> + RecordBatchReader>
    InterpolateIter<'a, I>
{
    pub fn new(
        source: I,
        metadata: &'a ReaderMetadata,
        spectrum_index_idx: usize,
        mz_array_idx: usize,
    ) -> Self {
        let interpolator = BatchIterpolater::new(metadata, spectrum_index_idx, mz_array_idx);
        let schema = source.schema();
        Self {
            source,
            spectrum_index_idx,
            interpolator,
            splitter: IndexSplittingIter::empty(schema),
        }
    }

    pub fn next_batch(&mut self) -> Option<Result<RecordBatch, ArrowError>> {
        let batch = match self.splitter.next() {
            Some(batch) => batch,
            None => {
                let batch = self.source.next()?;
                let batch = match batch {
                    Ok(batch) => batch,
                    Err(e) => return Some(Err(e)),
                };
                if !self.interpolator.check_batch_has_nulls(&batch) {
                    return Some(Ok(batch));
                }
                if let Err(e) = self.splitter.add_and_split(batch, self.spectrum_index_idx) {
                    return Some(Err(e));
                }
                self.splitter.next()?
            }
        };

        let batch = match batch {
            Err(_) => return Some(batch),
            Ok(batch) => batch,
        };

        Some(self.interpolator.process_batch(batch))
    }
}

mod sync_impl {
    use super::*;

    /// A facet that wraps the behavior for reading point-layout data.
    pub(crate) struct PointDataReader<T: ChunkReader + 'static>(
        pub(crate) ParquetRecordBatchReaderBuilder<T>,
        pub(crate) BufferContext,
    );

    impl<T: ChunkReader + 'static> PointQuerySource for PointDataReader<T> {
        fn metadata(&self) -> &ParquetMetaData {
            self.0.metadata()
        }

        fn parquet_schema(&self) -> Arc<SchemaDescriptor> {
            self.metadata().file_metadata().schema_descr_ptr()
        }

        fn buffer_context(&self) -> BufferContext {
            self.1
        }
    }

    impl<T: ChunkReader + 'static> PointDataArrayReader for PointDataReader<T> {}

    impl<T: ChunkReader + 'static> PointDataReader<T> {
        /// Read the arrays associated with the points of `index`
        pub(crate) fn read_points_of<'a, I: SpectrumQueryIndex + Debug + 'a>(
            self,
            index: u64,
            query_index: &'a I,
            array_indices: &'a ArrayIndex,
            mz_delta_model: Option<&RegressionDeltaModel<f64>>,
        ) -> io::Result<Option<BinaryArrayMap>> {
            let (rows, row_group_indices) = self.find_row_groups_query(index, query_index);
            let schem = self.parquet_schema();
            let (predicate, proj) = Self::prepare_points_of(schem, index, array_indices, self.1);

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
                let batch =
                    arrow::compute::concat_batches(batches[0].schema_ref(), &batches).unwrap();
                let points = batch.column(0).as_struct();
                Self::populate_arrays_from_struct_array(
                    points,
                    &mut bin_map,
                    mz_delta_model,
                    false,
                );
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
            let out = self.read_points_of(
                index,
                &meta_index.query_index,
                &meta_index.array_indices,
                None,
            )?;
            match out {
                Some(out) => match PeakDataLevel::try_from(&out) {
                    Ok(val) => return Ok(Some(val)),
                    Err(e) => return Err(e.into()),
                },
                None => Ok(None),
            }
        }

        pub(crate) fn query_points<'a, I: SpectrumQueryIndex + 'a>(
            self,
            index_range: MaskSet,
            mz_range: Option<SimpleInterval<f64>>,
            ion_mobility_range: Option<SimpleInterval<f64>>,
            query_index: &'a I,
            array_indices: &'a ArrayIndex,
            metadata: &'a ReaderMetadata,
            query: Option<PageQuery>,
        ) -> io::Result<Box<dyn Iterator<Item = Result<RecordBatch, ArrowError>> + 'a + Send>>
        {
            if let Some((rows, row_group_indices, proj, predicate)) = self.prepare_query(
                index_range.into(),
                mz_range,
                ion_mobility_range,
                query_index,
                array_indices,
                query,
            ) {
                let schema = self.0.schema();
                let (_, subset) = schema.column_with_name(&array_indices.prefix).unwrap();
                let subset = match subset.data_type() {
                    DataType::Struct(subset) => subset,
                    _ => panic!("Invalid point type"),
                };

                let context = self.1;

                let mut index_column_idx = None;
                let mut mz_column_idx = None;

                if matches!(context, BufferContext::Spectrum) {
                    let subset = arrow::datatypes::Schema::new(subset.clone());
                    index_column_idx = subset
                        .column_with_name(BufferContext::Spectrum.index_name())
                        .map(|(i, _)| i);
                    if let Some(mz_array_idx) = array_indices.get(&ArrayType::MZArray) {
                        mz_column_idx = subset
                            .column_with_name(&mz_array_idx.path.split(".").last().unwrap())
                            .map(|(i, _)| i);
                    }
                }

                let it: ParquetRecordBatchReader = self
                    .0
                    .with_row_groups(row_group_indices)
                    .with_row_selection(rows)
                    .with_projection(proj)
                    .with_row_filter(predicate)
                    .with_batch_size(10_000)
                    .build()?;

                // We don't have spectra in this reader, or we do but they don't have an m/z axis
                if !matches!(context, BufferContext::Spectrum)
                    || index_column_idx.is_none()
                    || mz_column_idx.is_none()
                {
                    return Ok(Box::new(it));
                }

                Ok(Box::new(InterpolateIter::new(
                    it,
                    metadata,
                    index_column_idx.unwrap(),
                    mz_column_idx.unwrap(),
                )))
            } else {
                Ok(Box::new(std::iter::empty()))
            }
        }
    }
}

pub(crate) use sync_impl::*;
