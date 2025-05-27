use std::{fs, io};

use mzpeaks::prelude::HasProximity;
#[allow(unused)]
use parquet::{
    self,
    arrow::{
        arrow_reader::{
            statistics::StatisticsConverter, ArrowPredicateFn, ArrowReaderOptions,
            ParquetRecordBatchReaderBuilder, RowFilter, RowSelection, RowSelector,
        },
        ProjectionMask,
    },
    basic::Type as PhysicalType,
    file::{
        metadata::{ParquetColumnIndex, ParquetOffsetIndex, RowGroupMetaData},
        page_index::index::Index as ParquetTypedIndex,
    },
    schema::types::{SchemaDescPtr, SchemaDescriptor},
};

#[allow(unused)]
use arrow::{
    self,
    array::{Array, AsArray, BooleanArray, Float64Array, PrimitiveArray, RecordBatch},
    datatypes::{DataType, Float64Type, UInt64Type, Field, FieldRef, Float32Type},
};

use mzdata::mzpeaks::coordinate::{SimpleInterval, Span1D};
use serde::{Deserialize, Serialize};

#[allow(unused)]
use serde_arrow::schema::TracingOptions;

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



pub trait PageIndexType<T> : Span1D<DimType = T> + Sized {
    fn start_row(&self) -> i64;
    fn end_row(&self) -> i64;

    fn row_len(&self) -> i64 {
        self.end_row() - self.start_row()
    }

    fn build_row_selection_contains(index: &[Self], query: T) -> RowSelection {
        let mut selectors = Vec::new();
        let mut last_row = 0;
        for page in index.iter() {
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

    fn build_row_selection_overlaps(index: &[Self], query: &impl Span1D<DimType = T>) -> RowSelection {
        let mut selectors = Vec::new();
        let mut last_row = 0;
        for page in index.iter() {
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
}


pub type TimeIndexPage = PageIndexEntry<f32>;
pub type PointMZIndexPage = PageIndexEntry<f64>;
pub type PointIonMobilityIndexPage = PageIndexEntry<f64>;
pub type PointSpectrumIndexPage = PageIndexEntry<u64>;


pub fn read_spectrum_time_index(
    reader: &ParquetRecordBatchReaderBuilder<fs::File>,
) -> io::Result<Vec<TimeIndexPage>> {
    let metadata = reader.metadata();

    let pq_schema = reader.parquet_schema();
    let column_ix = parquet_column(pq_schema, "spectrum.time").unwrap();

    let rg_meta = metadata.row_groups();
    let column_offset_index = metadata.offset_index().unwrap();
    let column_index = metadata.column_index().unwrap();

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
            ParquetTypedIndex::FLOAT(native_index) => {
                for (page_i, (q, offset)) in native_index
                    .indexes
                    .iter()
                    .zip(offset_list.page_locations().iter())
                    .enumerate()
                {
                    if q.min().is_none() {
                        continue;
                    }
                    let min = *q.min().unwrap() as f32;
                    let max = *q.max().unwrap() as f32;
                    let start_row = offset.first_row_index + total_rows;
                    let end_row =
                        if let Some(next_loc) = offset_list.page_locations().get(page_i + 1) {
                            next_loc.first_row_index + total_rows
                        } else {
                            rg.num_rows() + total_rows
                        };
                    pages.push(TimeIndexPage {
                        row_group_i: i,
                        page_i: page_i,
                        min,
                        max,
                        start_row,
                        end_row,
                    })
                }
            }
            ParquetTypedIndex::DOUBLE(native_index) => {
                for (page_i, (q, offset)) in native_index
                    .indexes
                    .iter()
                    .zip(offset_list.page_locations().iter())
                    .enumerate()
                {
                    if q.min().is_none() {
                        continue;
                    }
                    let min = *q.min().unwrap() as f32;
                    let max = *q.max().unwrap() as f32;
                    let start_row = offset.first_row_index + total_rows;
                    let end_row =
                        if let Some(next_loc) = offset_list.page_locations().get(page_i + 1) {
                            next_loc.first_row_index + total_rows
                        } else {
                            rg.num_rows() + total_rows
                        };
                    pages.push(TimeIndexPage {
                        row_group_i: i,
                        page_i: page_i,
                        min,
                        max,
                        start_row,
                        end_row,
                    })
                }
            }
            _ => {
                panic!("Wrong type of index!");
            }
        }
        total_rows += rg.num_rows();
    }

    Ok(pages)
}


pub fn read_point_spectrum_index(
    reader: &ParquetRecordBatchReaderBuilder<fs::File>,
) -> io::Result<Vec<PointSpectrumIndexPage>> {
    let metadata = reader.metadata();

    let pq_schema = reader.parquet_schema();
    let column_ix = parquet_column(pq_schema, "point.spectrum_index").unwrap();

    let rg_meta = metadata.row_groups();
    let column_offset_index = metadata.offset_index().unwrap();
    let column_index = metadata.column_index().unwrap();

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
            ParquetTypedIndex::INT32(native_index) => {
                for (page_i, (q, offset)) in native_index
                    .indexes
                    .iter()
                    .zip(offset_list.page_locations().iter())
                    .enumerate()
                {
                    if q.min().is_none() {
                        continue;
                    }
                    let min = *q.min().unwrap() as u64;
                    let max = *q.max().unwrap() as u64;
                    let start_row = offset.first_row_index + total_rows;
                    let end_row =
                        if let Some(next_loc) = offset_list.page_locations().get(page_i + 1) {
                            next_loc.first_row_index + total_rows
                        } else {
                            rg.num_rows() + total_rows
                        };
                    pages.push(PointSpectrumIndexPage {
                        row_group_i: i,
                        page_i: page_i,
                        min,
                        max,
                        start_row,
                        end_row,
                    })
                }
            }
            ParquetTypedIndex::INT64(native_index) => {
                for (page_i, (q, offset)) in native_index
                    .indexes
                    .iter()
                    .zip(offset_list.page_locations().iter())
                    .enumerate()
                {
                    if q.min().is_none() {
                        continue;
                    }
                    let min = *q.min().unwrap() as u64;
                    let max = *q.max().unwrap() as u64;
                    let start_row = offset.first_row_index + total_rows;
                    let end_row =
                        if let Some(next_loc) = offset_list.page_locations().get(page_i + 1) {
                            next_loc.first_row_index + total_rows
                        } else {
                            rg.num_rows() + total_rows
                        };
                    pages.push(PointSpectrumIndexPage {
                        row_group_i: i,
                        page_i: page_i,
                        min,
                        max,
                        start_row,
                        end_row,
                    })
                }
            }
            x => {
                panic!("Wrong type of index: {x:?}!");
            }
        }
        total_rows += rg.num_rows();
    }

    Ok(pages)
}


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


pub(crate) fn read_f64_page_index(
    reader: &ParquetRecordBatchReaderBuilder<fs::File>,
    column_path: &str
) -> io::Result<Vec<PageIndexEntry<f64>>> {
    let metadata = reader.metadata();

    let pq_schema = reader.parquet_schema();
    let column_ix = parquet_column(pq_schema, column_path).unwrap();

    let rg_meta = metadata.row_groups();
    let column_offset_index = metadata.offset_index().unwrap();
    let column_index = metadata.column_index().unwrap();

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
            ParquetTypedIndex::FLOAT(native_index) => {
                read_pages!(rg, i, native_index, f64, pages, total_rows, offset_list);
            }
            ParquetTypedIndex::DOUBLE(native_index) => {
                read_pages!(rg, i, native_index, f64, pages, total_rows, offset_list);
            }
            _ => {
                panic!("Wrong type of index!");
            }
        }
        total_rows += rg.num_rows();
    }

    Ok(pages)
}


pub fn read_point_mz_index(
    reader: &ParquetRecordBatchReaderBuilder<fs::File>,
) -> io::Result<Vec<PointMZIndexPage>> {
    read_f64_page_index(reader, "point.mz")
}


pub fn read_point_im_index(
    reader: &ParquetRecordBatchReaderBuilder<fs::File>,
) -> io::Result<Vec<PointIonMobilityIndexPage>> {
    read_f64_page_index(reader, "point.im")
}


pub fn spectrum_index_range_for_time_range(handle: &mut fs::File, query: SimpleInterval<f32>) -> io::Result<SimpleInterval<u64>> {
    let reader = ParquetRecordBatchReaderBuilder::try_new_with_options(
        handle.try_clone()?,
        ArrowReaderOptions::new().with_page_index(true),
    )?;

    let index = read_spectrum_time_index(&reader)?;

    let selectors = TimeIndexPage::build_row_selection_overlaps(&index, &query);
    let selection = RowSelection::from(selectors);

    let projection = ProjectionMask::columns(reader.parquet_schema(), ["spectrum.index", "spectrum.time"]);
    let filter_projection = ProjectionMask::columns(reader.parquet_schema(), ["spectrum.time"]);

    let predicate = move |batch: RecordBatch| {
        let spec = batch.column(0).as_struct();
        let col: &PrimitiveArray<Float32Type> = spec.column(0).as_primitive::<Float32Type>();
        let mask: Vec<bool> = col
            .iter()
            .map(|val| {
                val.map(|val| query.contains(&val))
                    .unwrap_or_default()
            })
            .collect();
        Ok(BooleanArray::from(mask))
    };

    let start = std::time::Instant::now();
    let reader = reader
        .with_row_selection(selection)
        .with_projection(projection)
        .with_row_filter(RowFilter::new(vec![Box::new(ArrowPredicateFn::new(
            filter_projection,
            predicate,
        ))]))
        .build()?;

    let mut indices = SimpleInterval::new(u64::MAX, u64::MIN);
    'a: for batch in reader.flatten() {
        let spec = batch.column(0).as_struct();
        let idx_arr: &PrimitiveArray<UInt64Type> = spec.column_by_name("index").unwrap().as_primitive();
        let time_arr: &PrimitiveArray<Float32Type> = spec.column_by_name("time").unwrap().as_primitive();
        for (val, ti) in idx_arr.iter().zip(time_arr.iter()) {
            if let (Some(val), Some(ti)) = (val, ti) {
                if query.contains(&ti) {
                    if indices.start > val {
                        indices.start = val;
                    }
                    if indices.end < val {
                        indices.end = val
                    }
                    if ti > query.end {
                        break 'a;
                    }
                }
            }
        }
    }

    let end = std::time::Instant::now();
    eprintln!("{} seconds elapsed (time index)", (end - start).as_secs_f64());
    Ok(indices)
}
