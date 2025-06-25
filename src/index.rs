use std::{fs, sync::Arc};

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
    Int64Array, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use mzpeaks::coordinate::SimpleInterval;
use mzpeaks::{coordinate::IntervalTree, prelude::HasProximity};
use parquet::file::metadata::ParquetMetaData;

use parquet::{
    self,
    arrow::arrow_reader::{ParquetRecordBatchReaderBuilder, RowSelection, RowSelector},
    file::page_index::index::Index as ParquetTypedIndex,
    schema::types::SchemaDescriptor,
};

use mzdata::mzpeaks::coordinate::Span1D;
use serde::{Deserialize, Serialize};

pub fn parquet_column(schema: &SchemaDescriptor, column: &str) -> Option<usize> {
    let mut column_ix: Option<usize> = None;
    for (i, col) in schema.columns().iter().enumerate() {
        if col.path().string() == column {
            column_ix = Some(i);
            break;
        }
    }
    column_ix
}

#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize)]
pub struct PageIndexEntry<T> {
    pub row_group_i: usize,
    pub page_i: usize,
    pub min: T,
    pub max: T,
    pub start_row: i64,
    pub end_row: i64,
}

impl<T> PageIndexEntry<T> {
    pub fn row_len(&self) -> i64 {
        self.end_row - self.start_row
    }
}

impl<T: HasProximity> Span1D for PageIndexEntry<T> {
    type DimType = T;

    fn start(&self) -> Self::DimType {
        self.min
    }

    fn end(&self) -> Self::DimType {
        self.max
    }
}

impl<T: HasProximity> PageIndexType<T> for PageIndexEntry<T> {
    fn start_row(&self) -> i64 {
        self.start_row
    }

    fn end_row(&self) -> i64 {
        self.end_row
    }
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct PageIndex<T: HasProximity>(Vec<PageIndexEntry<T>>)
where
    PageIndexEntry<T>: PageIndexType<T>;

impl<T: HasProximity> IntoIterator for PageIndex<T>
where
    PageIndexEntry<T>: PageIndexType<T>,
{
    type Item = PageIndexEntry<T>;

    type IntoIter = std::vec::IntoIter<PageIndexEntry<T>>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<T: HasProximity> PageIndex<T>
where
    PageIndexEntry<T>: PageIndexType<T>,
{
    pub fn as_interval_tree(&self) -> IntervalTree<T, PageIndexEntry<T>>
    where
        T: num_traits::real::Real + core::iter::Sum,
    {
        IntervalTree::new(self.0.clone())
    }

    pub fn get(&self, index: usize) -> Option<&PageIndexEntry<T>> {
        self.0.get(index)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, PageIndexEntry<T>> {
        self.0.iter()
    }

    pub fn first(&self) -> Option<&PageIndexEntry<T>> {
        self.0.first()
    }

    pub fn last(&self) -> Option<&PageIndexEntry<T>> {
        self.0.last()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn sort(&mut self) {
        self.0.sort_by(|a, b| {
            a.min
                .partial_cmp(&b.min)
                .unwrap()
                .then_with(|| a.max.partial_cmp(&b.max).unwrap())
        });
    }

    pub fn row_selection_is_not_null(&self) -> RowSelection {
        let mut selectors = Vec::new();
        let mut last_row = 0;

        for page in self.iter() {
            if page.start_row() != last_row {
                selectors.push(RowSelector::skip((page.start_row() - last_row) as usize));
            }
            selectors.push(RowSelector::select(page.row_len() as usize));
            last_row = page.end_row();
        }
        selectors.into()
    }

    pub fn pages_not_null(&self) -> std::slice::Iter<'_, PageIndexEntry<T>> {
        self.iter()
    }

    pub fn row_selection_contains(&self, query: T) -> RowSelection {
        let mut selectors = Vec::new();
        let mut last_row = 0;
        for page in self.iter() {
            if page.start_row() != last_row {
                selectors.push(RowSelector::skip((page.start_row() - last_row) as usize));
            }
            if page.contains(&query) {
                selectors.push(RowSelector::select(page.row_len() as usize));
            } else {
                selectors.push(RowSelector::skip(page.row_len() as usize))
            }
            last_row = page.end_row();
        }

        selectors.into()
    }

    pub fn pages_contains(&self, query: T) -> impl Iterator<Item=&PageIndexEntry<T>> {
        self.iter().filter(move |p| p.contains(&query))
    }

    pub fn row_selection_overlaps(&self, query: &impl Span1D<DimType = T>) -> RowSelection {
        let mut selectors = Vec::new();
        let mut last_row = 0;
        for page in self.iter() {
            if page.start_row() != last_row {
                selectors.push(RowSelector::skip((page.start_row() - last_row) as usize));
            }
            if page.overlaps(&query) {
                selectors.push(RowSelector::select(page.row_len() as usize));
            } else {
                selectors.push(RowSelector::skip(page.row_len() as usize))
            }
            last_row = page.end_row();
        }
        selectors.into()
    }

    pub fn pages_overlaps(&self, query: &impl Span1D<DimType = T>) -> impl Iterator<Item=&PageIndexEntry<T>> {
        self.iter().filter(move |p| p.overlaps(query))
    }

    pub fn pages_to_row_selection<'a>(&'a self, it:  impl IntoIterator<Item=&'a PageIndexEntry<T>>, mut last_row: i64) -> RowSelection {
        let mut selectors = Vec::new();
        for page in it {
            if page.start_row() != last_row {
                selectors.push(RowSelector::skip((page.start_row() - last_row) as usize));
            }
            selectors.push(RowSelector::select(page.row_len() as usize));
            last_row = page.end_row();
        }
        selectors.into()
    }
}

pub trait PageIndexType<T>: Span1D<DimType = T> {
    fn start_row(&self) -> i64;
    fn end_row(&self) -> i64;

    fn page_span(&self) -> SimpleInterval<i64> {
        SimpleInterval::new(self.start_row(), self.end_row())
    }

    fn row_len(&self) -> i64 {
        self.end_row() - self.start_row()
    }
}

pub type TimeIndexPage = PageIndexEntry<f32>;
pub type PointMZIndexPage = PageIndexEntry<f64>;
pub type PointIonMobilityIndexPage = PageIndexEntry<f64>;
pub type PointSpectrumIndexPage = PageIndexEntry<u64>;

macro_rules! read_pages {
    ($rg:ident, $i:ident, $native_index:expr, $vtype:ty, $pages:ident, $total_rows:ident, $offset_list:ident) => {
        for (page_i, (q, offset)) in $native_index
            .indexes
            .iter()
            .zip($offset_list.page_locations().iter())
            .enumerate()
        {
            if q.min().is_none() {
                continue;
            }
            let min = *q.min().unwrap() as $vtype;
            let max = *q.max().unwrap() as $vtype;
            let start_row = offset.first_row_index + $total_rows;
            let end_row =
                if let Some(next_loc) = $offset_list.page_locations().get(page_i + 1) {
                    next_loc.first_row_index + $total_rows
                } else {
                    $rg.num_rows() + $total_rows
                };
            $pages.push(PageIndexEntry::<$vtype> {
                row_group_i: $i,
                page_i: page_i,
                min,
                max,
                start_row,
                end_row,
            })
        }
    };
}

macro_rules! read_numeric_page_index {
    ($metadata:expr, $pq_schema:expr, $column_path:expr, $type:ty) => {{
        let column_ix = parquet_column($pq_schema, $column_path)?;

        let rg_meta = $metadata.row_groups();
        let column_offset_index = $metadata.offset_index()?;
        let column_index = $metadata.column_index()?;

        let mut total_rows = 0;
        let mut pages = Vec::new();
        for (i, (rg, (offset_list, idx_list))) in rg_meta
            .iter()
            .zip(column_offset_index.iter().zip(column_index.iter()))
            .enumerate()
        {
            let idx_list = &idx_list[column_ix];
            let offset_list = &offset_list[column_ix];

            match idx_list {
                $crate::index::ParquetTypedIndex::FLOAT(native_index) => {
                    read_pages!(rg, i, native_index, $type, pages, total_rows, offset_list);
                }
                $crate::index::ParquetTypedIndex::DOUBLE(native_index) => {
                    read_pages!(rg, i, native_index, $type, pages, total_rows, offset_list);
                }
                $crate::index::ParquetTypedIndex::INT32(native_index) => {
                    read_pages!(rg, i, native_index, $type, pages, total_rows, offset_list);
                }
                $crate::index::ParquetTypedIndex::INT64(native_index) => {
                    read_pages!(rg, i, native_index, $type, pages, total_rows, offset_list);
                }
                tp => {
                    panic!("Wrong type of index! {tp:?}");
                }
            }
            total_rows += rg.num_rows();
        }
        Some(PageIndex(pages))
    }};
    ($reader:expr, $column_path:expr, $type:ty) => {{
        let metadata = $reader.metadata();
        let pq_schema = $reader.parquet_schema();

        read_numeric_page_index!(metadata, pq_schema, $column_path, $type)
    }};
}

pub fn read_f32_page_index_from(
    metadata: &Arc<ParquetMetaData>,
    pq_schema: &SchemaDescriptor,
    column_path: &str,
) -> Option<PageIndex<f32>> {
    read_numeric_page_index!(metadata, pq_schema, column_path, f32)
}

pub fn read_f64_page_index_from(
    metadata: &Arc<ParquetMetaData>,
    pq_schema: &SchemaDescriptor,
    column_path: &str,
) -> Option<PageIndex<f64>> {
    read_numeric_page_index!(metadata, pq_schema, column_path, f64)
}

pub fn read_i32_page_index_from(
    metadata: &Arc<ParquetMetaData>,
    pq_schema: &SchemaDescriptor,
    column_path: &str,
) -> Option<PageIndex<i32>> {
    read_numeric_page_index!(metadata, pq_schema, column_path, i32)
}

pub fn read_i64_page_index_from(
    metadata: &Arc<ParquetMetaData>,
    pq_schema: &SchemaDescriptor,
    column_path: &str,
) -> Option<PageIndex<i64>> {
    read_numeric_page_index!(metadata, pq_schema, column_path, i64)
}

pub fn read_u32_page_index_from(
    metadata: &Arc<ParquetMetaData>,
    pq_schema: &SchemaDescriptor,
    column_path: &str,
) -> Option<PageIndex<u32>> {
    read_numeric_page_index!(metadata, pq_schema, column_path, u32)
}

pub fn read_u64_page_index_from(
    metadata: &Arc<ParquetMetaData>,
    pq_schema: &SchemaDescriptor,
    column_path: &str,
) -> Option<PageIndex<u64>> {
    read_numeric_page_index!(metadata, pq_schema, column_path, u64)
}

pub fn read_u8_page_index_from(
    metadata: &Arc<ParquetMetaData>,
    pq_schema: &SchemaDescriptor,
    column_path: &str,
) -> Option<PageIndex<u8>> {
    read_numeric_page_index!(metadata, pq_schema, column_path, u8)
}

pub fn read_i8_page_index_from(
    metadata: &Arc<ParquetMetaData>,
    pq_schema: &SchemaDescriptor,
    column_path: &str,
) -> Option<PageIndex<i8>> {
    read_numeric_page_index!(metadata, pq_schema, column_path, i8)
}

pub fn read_f32_page_index(
    reader: &ParquetRecordBatchReaderBuilder<fs::File>,
    column_path: &str,
) -> Option<PageIndex<f32>> {
    read_numeric_page_index!(reader, column_path, f32)
}

pub fn read_f64_page_index(
    reader: &ParquetRecordBatchReaderBuilder<fs::File>,
    column_path: &str,
) -> Option<PageIndex<f64>> {
    read_numeric_page_index!(reader, column_path, f64)
}

pub fn read_i32_page_index(
    reader: &ParquetRecordBatchReaderBuilder<fs::File>,
    column_path: &str,
) -> Option<PageIndex<i32>> {
    read_numeric_page_index!(reader, column_path, i32)
}

pub fn read_u32_page_index(
    reader: &ParquetRecordBatchReaderBuilder<fs::File>,
    column_path: &str,
) -> Option<PageIndex<u32>> {
    read_numeric_page_index!(reader, column_path, u32)
}

pub fn read_i64_page_index(
    reader: &ParquetRecordBatchReaderBuilder<fs::File>,
    column_path: &str,
) -> Option<PageIndex<i64>> {
    read_numeric_page_index!(reader, column_path, i64)
}

pub fn read_u64_page_index(
    reader: &ParquetRecordBatchReaderBuilder<fs::File>,
    column_path: &str,
) -> Option<PageIndex<u64>> {
    read_numeric_page_index!(reader, column_path, u64)
}

pub fn read_spectrum_time_index(
    reader: &ParquetRecordBatchReaderBuilder<fs::File>,
) -> Option<PageIndex<f32>> {
    read_f32_page_index(reader, "spectrum.time")
}

pub fn read_point_spectrum_index(
    reader: &ParquetRecordBatchReaderBuilder<fs::File>,
) -> Option<PageIndex<u64>> {
    read_u64_page_index(reader, "point.spectrum_index")
}

pub fn read_point_mz_index(
    reader: &ParquetRecordBatchReaderBuilder<fs::File>,
) -> Option<PageIndex<f64>> {
    read_f64_page_index(reader, "point.mz")
}

pub fn read_point_im_index(
    reader: &ParquetRecordBatchReaderBuilder<fs::File>,
) -> Option<PageIndex<f64>> {
    read_f64_page_index(reader, "point.im")
}

pub trait SpanDynNumeric: Span1D
where
    Self::DimType: num_traits::NumCast,
{
    fn contains_dy_iter<'a>(
        &'a self,
        array: &'a ArrayRef,
    ) -> impl Iterator<Item = Option<bool>> + 'a {
        let n = array.len();
        let it = 0..n;

        macro_rules! span_dyn_impl {
            ($raw_ty:ty, $arr_ty:ty) => {{
                let start = <$raw_ty as num_traits::NumCast>::from(self.start()).unwrap();
                let end = <$raw_ty as num_traits::NumCast>::from(self.end()).unwrap();
                let span = SimpleInterval::new(start, end);
                let array: &$arr_ty = array.as_any().downcast_ref().unwrap();
                let closure: Box<dyn Fn(usize) -> Option<bool>> = Box::new(move |i| {
                    if array.is_valid(i) {
                        Some(span.contains(&array.value(i)))
                    } else {
                        None
                    }
                });
                it.map(closure)
            }};
        }

        match array.data_type() {
            arrow::datatypes::DataType::Int8 => {
                span_dyn_impl!(i8, Int8Array)
            }
            arrow::datatypes::DataType::Int16 => {
                span_dyn_impl!(i16, Int16Array)
            }
            arrow::datatypes::DataType::Int32 => span_dyn_impl!(i32, Int32Array),
            arrow::datatypes::DataType::Int64 => span_dyn_impl!(i64, Int64Array),
            arrow::datatypes::DataType::UInt8 => span_dyn_impl!(u8, UInt8Array),
            arrow::datatypes::DataType::UInt16 => span_dyn_impl!(u16, UInt16Array),
            arrow::datatypes::DataType::UInt32 => span_dyn_impl!(u32, UInt32Array),
            arrow::datatypes::DataType::UInt64 => span_dyn_impl!(u64, UInt64Array),
            arrow::datatypes::DataType::Float32 => span_dyn_impl!(f32, Float32Array),
            arrow::datatypes::DataType::Float64 => span_dyn_impl!(f64, Float64Array),
            _ => {
                let f: Box<dyn Fn(usize) -> Option<bool>> = Box::new(|_| None);
                it.map(f)
            }
        }
    }

    fn contains_dy(&self, array: &ArrayRef) -> BooleanArray {
        macro_rules! span_dyn_impl {
            ($raw_ty:ty, $arr_ty:ty) => {{
                let start = <$raw_ty as num_traits::NumCast>::from(self.start()).unwrap();
                let end = <$raw_ty as num_traits::NumCast>::from(self.end()).unwrap();
                let span = SimpleInterval::new(start, end);
                let array: &$arr_ty = array.as_any().downcast_ref().unwrap();
                array.iter().map(|v| v.map(|v| span.contains(&v))).collect()
            }};
        }
        match array.data_type() {
            arrow::datatypes::DataType::Int8 => {
                span_dyn_impl!(i8, Int8Array)
            }
            arrow::datatypes::DataType::Int16 => {
                span_dyn_impl!(i16, Int16Array)
            }
            arrow::datatypes::DataType::Int32 => span_dyn_impl!(i32, Int32Array),
            arrow::datatypes::DataType::Int64 => span_dyn_impl!(i64, Int64Array),
            arrow::datatypes::DataType::UInt8 => span_dyn_impl!(u8, UInt8Array),
            arrow::datatypes::DataType::UInt16 => span_dyn_impl!(u16, UInt16Array),
            arrow::datatypes::DataType::UInt32 => span_dyn_impl!(u32, UInt32Array),
            arrow::datatypes::DataType::UInt64 => span_dyn_impl!(u64, UInt64Array),
            arrow::datatypes::DataType::Float32 => span_dyn_impl!(f32, Float32Array),
            arrow::datatypes::DataType::Float64 => span_dyn_impl!(f64, Float64Array),
            _ => BooleanArray::new_null(array.len()),
        }
    }
}

impl<T: Span1D> SpanDynNumeric for T where T::DimType: num_traits::NumCast {}
