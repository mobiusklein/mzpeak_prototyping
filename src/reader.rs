use std::{
    collections::HashMap,
    fs::File,
    io::{self, prelude::*},
    path::{Path, PathBuf},
    sync::Arc,
};

use arrow::array::{Array, Int32Array, Int64Array, LargeListArray, UInt32Array};
#[allow(unused)]
use arrow::{
    array::{
        ArrayRef, AsArray, BooleanArray, Float32Array, Float64Array, Int8Array, LargeStringArray,
        RecordBatch, StructArray, UInt8Array, UInt64Array,
    },
    datatypes::{DataType, Field, FieldRef, Fields, Schema, SchemaRef},
};

use mzdata::{
    params::{ControlledVocabulary, ParamDescribed, Unit},
    prelude::PrecursorSelection,
    spectrum::{
        ArrayType, BinaryArrayMap, DataArray, ScanEvent, ScanPolarity, SpectrumDescription,
    },
};

#[allow(unused)]
use parquet::{
    arrow::{
        ProjectionMask,
        arrow_reader::{
            ArrowPredicate, ArrowPredicateFn, ArrowReaderOptions, ParquetRecordBatchReader,
            ParquetRecordBatchReaderBuilder, RowFilter,
        },
    },
    file::{metadata::ParquetMetaData, reader::ChunkReader},
    schema::types::{ColumnPath, SchemaDescriptor},
};
use serde_arrow::schema::SchemaLike;

use crate::{CURIE, MZPeaksSelectedIonEntry, MzPeaksPrecursorEntry, curie};
#[allow(unused)]
use crate::{
    index::{
        PageIndex, PageIndexEntry, PageIndexType, parquet_column, read_f32_page_index_from,
        read_f64_page_index_from, read_i32_page_index_from, read_i64_page_index_from,
        read_u8_page_index_from, read_u32_page_index_from, read_u64_page_index_from,
    },
    param::{
        MzPeaksDataProcessing, MzPeaksFileDescription, MzPeaksInstrumentConfiguration,
        MzPeaksSoftware,
    },
    peak_series::{ArrayIndex, SerializedArrayIndex},
};

pub struct MzPeaksReaderMetadata {
    pub metadata: Arc<ParquetMetaData>,
    pub schema: SchemaRef,
    pub parquet_schema: Arc<SchemaDescriptor>,
    pub file_description: MzPeaksFileDescription,
    pub instrument_configurations: Vec<MzPeaksInstrumentConfiguration>,
    pub software_list: Vec<MzPeaksSoftware>,
    pub data_processing_list: Vec<MzPeaksDataProcessing>,
    pub spectrum_array_indices: ArrayIndex,
    pub chromatogram_array_indices: ArrayIndex,
}

#[derive(Debug, Default, Clone)]
pub struct QueryIndex {
    pub spectrum_time_index: PageIndex<f32>,
    pub spectrum_index_index: PageIndex<u64>,
    pub spectrum_ms_level_index: PageIndex<u8>,
    pub scan_index: PageIndex<u64>,
    pub precursor_index: PageIndex<u64>,
    pub selected_ion_index: PageIndex<u64>,

    pub spectrum_point_spectrum_index: PageIndex<u64>,
    pub spectrum_mz_index: PageIndex<f64>,
    pub spectrum_im_index: PageIndex<f64>,
}

pub struct MzPeaksReader {
    #[allow(unused)]
    path: PathBuf,
    handle: File,
    pub metadata: MzPeaksReaderMetadata,
    pub query_indices: QueryIndex,
}

impl MzPeaksReader {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().into();
        let mut handle = File::open(&path)?;

        let (metadata, query_indices) = Self::load_indices_from(handle.try_clone()?, &path)?;

        handle.seek(io::SeekFrom::Start(0))?;

        let this = Self {
            path,
            handle,
            metadata,
            query_indices,
        };
        Ok(this)
    }

    pub fn num_rows(&self) -> usize {
        self.metadata
            .metadata
            .row_groups()
            .iter()
            .map(|rg| rg.num_rows() as usize)
            .sum()
    }

    fn load_indices_from(
        mut handle: File,
        path: &PathBuf,
    ) -> io::Result<(MzPeaksReaderMetadata, QueryIndex)> {
        handle.seek(io::SeekFrom::Start(0))?;

        let mut file_description: MzPeaksFileDescription = Default::default();
        let mut instrument_configurations: Vec<MzPeaksInstrumentConfiguration> = Default::default();
        let mut software_list: Vec<MzPeaksSoftware> = Default::default();
        let mut data_processing_list: Vec<MzPeaksDataProcessing> = Default::default();
        let mut spectrum_array_indices: ArrayIndex = Default::default();
        let mut chromatogram_array_indices: ArrayIndex = Default::default();

        let handle = ParquetRecordBatchReaderBuilder::try_new_with_options(
            handle,
            ArrowReaderOptions::new().with_page_index(true),
        )?;
        let metadata = handle.metadata().clone();
        let schema = handle.schema().clone();

        for kv in metadata
            .file_metadata()
            .key_value_metadata()
            .into_iter()
            .flatten()
        {
            match kv.key.as_str() {
                "spectrum_array_index" => {
                    if let Some(val) = kv.value.as_ref() {
                        let array_index: SerializedArrayIndex = serde_json::from_str(&val)?;
                        spectrum_array_indices = array_index.into();
                    } else {
                        log::warn!("spectrum array index was empty for {}", path.display());
                    }
                }
                "chromatogram_array_index" => {
                    if let Some(val) = kv.value.as_ref() {
                        let array_index: SerializedArrayIndex = serde_json::from_str(&val)?;
                        chromatogram_array_indices = array_index.into();
                    } else {
                        log::warn!("chromatogram array index was empty for {}", path.display());
                    }
                }
                "file_description" => {
                    if let Some(val) = kv.value.as_ref() {
                        file_description = serde_json::from_str(&val)?;
                    } else {
                        log::warn!("file description was empty for {}", path.display());
                    }
                }
                "instrument_configuration_list" => {
                    if let Some(val) = kv.value.as_ref() {
                        instrument_configurations = serde_json::from_str(&val)?;
                    } else {
                        log::warn!(
                            "instrument configurations list was empty for {}",
                            path.display()
                        );
                    }
                }
                "data_processing_method_list" => {
                    if let Some(val) = kv.value.as_ref() {
                        data_processing_list = serde_json::from_str(&val)?;
                    } else {
                        log::warn!(
                            "data processing method list was empty for {}",
                            path.display()
                        );
                    }
                }
                "software_list" => {
                    if let Some(val) = kv.value.as_ref() {
                        software_list = serde_json::from_str(&val)?;
                    } else {
                        log::warn!("software list was empty for {}", path.display());
                    }
                }
                _ => {}
            }
        }

        let pq_schema = handle.parquet_schema();
        let pq_schema = SchemaDescriptor::new(pq_schema.root_schema_ptr().clone());

        let mut query_index = QueryIndex::default();
        query_index.spectrum_index_index =
            read_u64_page_index_from(&metadata, &pq_schema, "spectrum.index").unwrap_or_default();
        query_index.spectrum_time_index =
            read_f32_page_index_from(&metadata, &pq_schema, "spectrum.time").unwrap_or_default();
        query_index.spectrum_ms_level_index =
            read_u8_page_index_from(&metadata, &pq_schema, "spectrum.ms_level").unwrap_or_default();
        query_index.scan_index =
            read_u64_page_index_from(&metadata, &pq_schema, "scan.spectrum_index")
                .unwrap_or_default();
        query_index.precursor_index =
            read_u64_page_index_from(&metadata, &pq_schema, "precursor.spectrum_index")
                .unwrap_or_default();
        query_index.selected_ion_index =
            read_u64_page_index_from(&metadata, &pq_schema, "selected_ion.spectrum_index")
                .unwrap_or_default();

        query_index.spectrum_point_spectrum_index = read_u64_page_index_from(
            &metadata,
            &pq_schema,
            &format!("{}.spectrum_index", spectrum_array_indices.prefix),
        )
        .unwrap_or_default();

        for (arr, entry) in spectrum_array_indices.iter() {
            if matches!(arr, ArrayType::MZArray) {
                query_index.spectrum_mz_index =
                    read_f64_page_index_from(&metadata, &pq_schema, &entry.path)
                        .unwrap_or_default();
            } else if arr.is_ion_mobility() {
                query_index.spectrum_im_index =
                    read_f64_page_index_from(&metadata, &pq_schema, &entry.path)
                        .unwrap_or_default();
            }
        }

        let bundle = MzPeaksReaderMetadata {
            metadata,
            schema,
            parquet_schema: Arc::new(pq_schema),
            file_description,
            instrument_configurations,
            software_list,
            data_processing_list,
            spectrum_array_indices,
            chromatogram_array_indices,
        };

        Ok((bundle, query_index))
    }

    pub fn get_spectrum_arrays(&mut self, index: u64) -> io::Result<BinaryArrayMap> {
        let handle = self.handle.try_clone()?;

        let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(
            handle,
            ArrowReaderOptions::new().with_page_index(true),
        )?;

        let rows = self
            .query_indices
            .spectrum_point_spectrum_index
            .row_selection_contains(index);

        let predicate_mask = ProjectionMask::columns(
            &self.metadata.parquet_schema,
            [format!(
                "{}.spectrum_index",
                self.metadata.spectrum_array_indices.prefix
            )
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

        let reader = builder
            .with_row_selection(rows)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .with_projection(ProjectionMask::columns(
                &self.metadata.parquet_schema,
                [self.metadata.spectrum_array_indices.prefix.as_str()],
            ))
            .build()?;

        let mut bin_map = HashMap::new();
        for (k, v) in self.metadata.spectrum_array_indices.iter() {
            let dtype = crate::peak_series::arrow_to_array_type(&v.data_type).unwrap();
            bin_map.insert(&v.name, DataArray::from_name_and_type(k, dtype));
        }

        for batch in reader.flatten() {
            let points = batch.column(0).as_struct();
            for (f, arr) in points.fields().iter().zip(batch.columns()) {
                if f.name() == "spectrum_index" {
                    continue;
                }
                let store = bin_map.get_mut(f.name()).unwrap();

                macro_rules! extend_array {
                    ($buf:ident) => {
                        if $buf.null_count() > 0 {
                            for val in $buf.iter() {
                                if let Some(val) = val {
                                    store.push(val).unwrap();
                                }
                            }
                        } else {
                            store.extend($buf.values()).unwrap();
                        }
                    };
                }

                match f.data_type() {
                    DataType::Float32 => {
                        let buf: &Float32Array = arr.as_primitive();
                        extend_array!(buf);
                    },
                    DataType::Float64 => {
                        let buf: &Float64Array = arr.as_primitive();
                        extend_array!(buf);
                    },
                    DataType::Int32 => {
                        let buf: &Int32Array = arr.as_primitive();
                        extend_array!(buf);
                    },
                    DataType::Int64 => {
                        let buf: &Int64Array = arr.as_primitive();
                        extend_array!(buf);
                    },
                    DataType::UInt8 => {
                        let buf: &UInt8Array = arr.as_primitive();
                        extend_array!(buf);
                    },
                    DataType::LargeUtf8 => {},
                    DataType::Utf8 => {}
                    _ => {}
                }
            }
        }

        let mut out = BinaryArrayMap::new();
        for v in bin_map.into_values() {
            out.add(v);
        }
        Ok(out)
    }

    pub fn get_spectrum_metadata(&mut self, index: u64) -> io::Result<SpectrumDescription> {
        let handle = self.handle.try_clone()?;

        let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(
            handle,
            ArrowReaderOptions::new().with_page_index(true),
        )?;

        let mut rows = self
            .query_indices
            .spectrum_index_index
            .row_selection_contains(index);

        rows = rows.union(&self.query_indices.scan_index.row_selection_contains(index));
        rows = rows.union(
            &self
                .query_indices
                .precursor_index
                .row_selection_contains(index),
        );
        rows = rows.union(
            &self
                .query_indices
                .selected_ion_index
                .row_selection_contains(index),
        );

        let predicate_mask = ProjectionMask::columns(
            &self.metadata.parquet_schema,
            [
                "spectrum.index",
                "scan.spectrum_index",
                "precursor.spectrum_index",
                "selected_ion.spectrum_index",
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
            let scan_spectrum_index: &UInt64Array = batch
                .column(1)
                .as_struct()
                .column(0)
                .as_any()
                .downcast_ref()
                .unwrap();
            let precursor_spectrum_index: &UInt64Array = batch
                .column(2)
                .as_struct()
                .column(0)
                .as_any()
                .downcast_ref()
                .unwrap();
            let selected_ion_spectrum_index: &UInt64Array = batch
                .column(3)
                .as_struct()
                .column(0)
                .as_any()
                .downcast_ref()
                .unwrap();

            let it = spectrum_index
                .iter()
                .map(|val| val.is_some_and(|val| val == index));
            let it = scan_spectrum_index
                .iter()
                .map(|val| val.is_some_and(|val| val == index))
                .zip(it)
                .map(|(a, b)| a || b);

            let it = precursor_spectrum_index
                .iter()
                .map(|val| val.is_some_and(|val| val == index))
                .zip(it)
                .map(|(a, b)| a || b);

            let it = selected_ion_spectrum_index
                .iter()
                .map(|val| val.is_some_and(|val| val == index))
                .zip(it)
                .map(|(a, b)| a || b);

            Ok(it.map(Some).collect())
        });

        let reader = builder
            .with_row_selection(rows)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .with_projection(ProjectionMask::columns(
                &self.metadata.parquet_schema,
                ["spectrum", "scan", "precursor", "selected_ion"],
            ))
            .build()?;

        // let curie_fields: Vec<FieldRef> =
        //     SchemaLike::from_type::<CURIE>(TracingOptions::new()).unwrap();

        let param_fields: Vec<FieldRef> =
            SchemaLike::from_type::<crate::Param>(Default::default()).unwrap();

        let precursor_fields: Vec<FieldRef> =
            SchemaLike::from_type::<MzPeaksPrecursorEntry>(Default::default()).unwrap();

        let selected_ion_fields: Vec<FieldRef> =
            SchemaLike::from_type::<MZPeaksSelectedIonEntry>(Default::default()).unwrap();

        let mut descr = SpectrumDescription::default();

        let mut precursors: Vec<MzPeaksPrecursorEntry> = Vec::new();
        let mut selected_ions: Vec<MZPeaksSelectedIonEntry> = Vec::new();

        for batch in reader.flatten() {
            let spec_arr = batch.column_by_name("spectrum").unwrap().as_struct();
            let index_arr: &UInt64Array = spec_arr
                .column_by_name("index")
                .unwrap()
                .as_any()
                .downcast_ref()
                .unwrap();
            if let Some(pos) = index_arr.iter().position(|i| i.is_some_and(|i| i == index)) {
                let spec_arr = spec_arr.slice(pos, 1);
                {}
                let idx_arr: &UInt64Array = spec_arr.column(0).as_any().downcast_ref().unwrap();
                let idx_val = idx_arr.value(0);
                descr.index = idx_val as usize;

                let id_arr: &LargeStringArray = spec_arr.column(1).as_string();
                let id_val = id_arr.value(0);
                descr.id = id_val.to_string();

                let ms_level_arr: &UInt8Array = spec_arr.column(2).as_any().downcast_ref().unwrap();
                let ms_level_val = ms_level_arr.value(0);
                descr.ms_level = ms_level_val;

                let polarity_arr: &Int8Array = spec_arr.column(4).as_any().downcast_ref().unwrap();
                let polarity_val = polarity_arr.value(0);
                match polarity_val {
                    1 => descr.polarity = ScanPolarity::Positive,
                    -1 => descr.polarity = ScanPolarity::Negative,
                    _ => {
                        todo!("Don't know how to deal with polarity {polarity_val}")
                    }
                }

                let continuity_array = spec_arr.column(5).as_struct();
                let cv_id_arr: &UInt8Array =
                    continuity_array.column(0).as_any().downcast_ref().unwrap();
                let accession_arr: &UInt32Array =
                    continuity_array.column(1).as_any().downcast_ref().unwrap();
                let continuity_curie = CURIE::new(cv_id_arr.value(0), accession_arr.value(0));

                descr.signal_continuity = match continuity_curie {
                    curie!(MS:1000525) => mzdata::spectrum::SignalContinuity::Unknown,
                    curie!(MS:1000127) => mzdata::spectrum::SignalContinuity::Centroid,
                    curie!(MS:1000128) => mzdata::spectrum::SignalContinuity::Profile,
                    _ => todo!("Don't know how to deal with {continuity_curie}"),
                };

                if let Some(mz_arr) = spec_arr
                    .column_by_name("lowest_observed_mz")
                    .and_then(|arr| arr.as_any().downcast_ref::<Float64Array>())
                {
                    let p = mzdata::Param::builder()
                        .curie(mzdata::curie!(MS:1000528))
                        .name("lowest observed m/z")
                        .unit(Unit::MZ)
                        .value(mz_arr.value(0))
                        .build();
                    descr.add_param(p);
                }

                if let Some(mz_arr) = spec_arr
                    .column_by_name("highest_observed_mz")
                    .and_then(|arr| arr.as_any().downcast_ref::<Float64Array>())
                {
                    let p = mzdata::Param::builder()
                        .curie(mzdata::curie!(MS:1000527))
                        .name("highest observed m/z")
                        .unit(Unit::MZ)
                        .value(mz_arr.value(0))
                        .build();
                    descr.add_param(p);
                }

                if let Some(mz_arr) = spec_arr
                    .column_by_name("base_peak_mz")
                    .and_then(|arr| arr.as_any().downcast_ref::<Float64Array>())
                {
                    let p = mzdata::Param::builder()
                        .curie(mzdata::curie!(MS:1000504))
                        .name("base peak m/z")
                        .unit(Unit::MZ)
                        .value(mz_arr.value(0))
                        .build();
                    descr.add_param(p);
                }

                if let Some(arr) = spec_arr
                    .column_by_name("base_peak_intensty")
                    .and_then(|arr| arr.as_any().downcast_ref::<Float32Array>())
                {
                    let p = mzdata::Param::builder()
                        .curie(mzdata::curie!(MS:1000505))
                        .name("base peak intensity")
                        .unit(Unit::DetectorCounts)
                        .value(arr.value(0))
                        .build();
                    descr.add_param(p);
                }

                if let Some(arr) = spec_arr
                    .column_by_name("total_ion_current")
                    .and_then(|arr| arr.as_any().downcast_ref::<Float32Array>())
                {
                    let p = mzdata::Param::builder()
                        .curie(mzdata::curie!(MS:1000285))
                        .name("total ion current")
                        .unit(Unit::DetectorCounts)
                        .value(arr.value(0))
                        .build();
                    descr.add_param(p);
                }

                if let Some(arr) = spec_arr
                    .column_by_name("lowest_observed_mz")
                    .and_then(|arr| arr.as_any().downcast_ref::<Float64Array>())
                {
                    let p = mzdata::Param::builder()
                        .curie(mzdata::curie!(MS:1000528))
                        .name("lowest observed m/z")
                        .unit(Unit::MZ)
                        .value(arr.value(0))
                        .build();
                    descr.add_param(p);
                }

                if let Some(arr) = spec_arr
                    .column_by_name("highest_observed_mz")
                    .and_then(|arr| arr.as_any().downcast_ref::<Float64Array>())
                {
                    let p = mzdata::Param::builder()
                        .curie(mzdata::curie!(MS:1000527))
                        .name("highest observed m/z")
                        .unit(Unit::MZ)
                        .value(arr.value(0))
                        .build();
                    descr.add_param(p);
                }

                let params_array: &LargeListArray =
                    spec_arr.column_by_name("parameters").unwrap().as_list();
                let params = params_array.value(0);
                let params = params.as_struct();

                const SKIP_PARAMS: [CURIE; 6] = [
                    crate::curie!(MS:1000505),
                    crate::curie!(MS:1000504),
                    crate::curie!(MS:1000257),
                    crate::curie!(MS:1000285),
                    crate::curie!(MS:1000527),
                    crate::curie!(MS:1000528),
                ];
                let params: Vec<crate::Param> =
                    serde_arrow::from_arrow(&param_fields, params.columns()).unwrap();

                for p in params {
                    if let Some(acc) = p.accession {
                        if !SKIP_PARAMS.contains(&acc) {
                            descr.add_param(p.into());
                        }
                    } else {
                        descr.add_param(p.into());
                    }
                }
            }

            let scan_arr = batch.column_by_name("scan").unwrap().as_struct();
            {
                let index_arr: &UInt64Array = scan_arr
                    .column_by_name("spectrum_index")
                    .unwrap()
                    .as_any()
                    .downcast_ref()
                    .unwrap();

                let time_arr: &Float32Array = scan_arr.column(1).as_any().downcast_ref().unwrap();
                let _config_arr: &UInt32Array = scan_arr.column(2).as_any().downcast_ref().unwrap();
                let _filter_string_arr: &LargeStringArray = scan_arr.column(3).as_string();
                let inject_arr: &Float32Array = scan_arr.column(4).as_any().downcast_ref().unwrap();
                let _ion_mobility_arr: &Float64Array =
                    scan_arr.column(5).as_any().downcast_ref().unwrap();
                let _ion_mobility_tp_arr = scan_arr.column(6).as_struct();
                let instrument_configuration_ref_arr: &UInt32Array =
                    scan_arr.column(7).as_any().downcast_ref().unwrap();
                let params_array: &LargeListArray = scan_arr.column(8).as_list();
                for (pos, _) in index_arr
                    .iter()
                    .enumerate()
                    .filter(|i| i.1.is_some_and(|i| i == index))
                {
                    let mut event = ScanEvent::default();
                    event.start_time = time_arr.value(pos) as f64;
                    event.injection_time = if inject_arr.is_valid(pos) {
                        inject_arr.value(pos)
                    } else {
                        0.0
                    };
                    event.instrument_configuration_id =
                        if instrument_configuration_ref_arr.is_valid(pos) {
                            instrument_configuration_ref_arr.value(pos)
                        } else {
                            0
                        };

                    let params = params_array.value(pos);
                    let params = params.as_struct();
                    let params: Vec<crate::Param> =
                        serde_arrow::from_arrow(&param_fields, params.columns()).unwrap();
                    event
                        .params_mut()
                        .extend(params.into_iter().map(|p| p.into()));

                    descr.acquisition.scans.push(event);
                }
            }

            let precursor_arr = batch.column_by_name("precursor").unwrap().as_struct();
            {
                let index_arr: &UInt64Array = precursor_arr
                    .column_by_name("spectrum_index")
                    .unwrap()
                    .as_any()
                    .downcast_ref()
                    .unwrap();

                for (pos, _) in index_arr
                    .iter()
                    .enumerate()
                    .filter(|i| i.1.is_some_and(|i| i == index))
                {
                    let row = precursor_arr.slice(pos, 1);
                    precursors.extend(
                        serde_arrow::from_arrow::<Vec<MzPeaksPrecursorEntry>, _>(
                            &precursor_fields,
                            row.columns(),
                        )
                        .unwrap(),
                    );
                }
            }

            let selected_ion_arr = batch.column_by_name("selected_ion").unwrap().as_struct();
            {
                let index_arr: &UInt64Array = selected_ion_arr
                    .column_by_name("spectrum_index")
                    .unwrap()
                    .as_any()
                    .downcast_ref()
                    .unwrap();

                for (pos, _) in index_arr
                    .iter()
                    .enumerate()
                    .filter(|i| i.1.is_some_and(|i| i == index))
                {
                    let row = selected_ion_arr.slice(pos, 1);
                    selected_ions.extend(
                        serde_arrow::from_arrow::<Vec<MZPeaksSelectedIonEntry>, _>(
                            &selected_ion_fields,
                            row.columns(),
                        )
                        .unwrap(),
                    );
                }
            }
        }

        for precursor in precursors {
            let mut prec = mzdata::spectrum::Precursor::default();
            prec.isolation_window = precursor.isolation_window.into();
            prec.activation = precursor.activation.into();
            prec.precursor_id = precursor.precursor_id;
            for selected_ion in selected_ions.iter_mut() {
                if selected_ion.precursor_index != precursor.precursor_index {
                    continue;
                }

                let mut si = mzdata::spectrum::SelectedIon::default();
                si.charge = selected_ion.charge_state;
                si.intensity = selected_ion.intensity.unwrap_or_default();
                si.mz = selected_ion.selected_ion_mz.unwrap_or_default();

                for p in selected_ion.parameters.drain(..) {
                    si.add_param(p.into());
                }

                if let Some(im) = selected_ion.ion_mobility {
                    let im_type: mzdata::params::CURIE =
                        selected_ion.ion_mobility_type.unwrap().into();
                    let im_param = mzdata::params::Param::builder();
                    let im_param = match im_type {
                        mzdata::params::CURIE {
                            controlled_vocabulary: ControlledVocabulary::MS,
                            accession: 1002476,
                        } => im_param
                            .curie(im_type)
                            .name("ion mobility drift time")
                            .value(im)
                            .unit(Unit::Millisecond),
                        mzdata::params::CURIE {
                            controlled_vocabulary: ControlledVocabulary::MS,
                            accession: 1002815,
                        } => im_param
                            .curie(im_type)
                            .name("inverse reduced ion mobility drift time")
                            .value(im)
                            .unit(Unit::VoltSecondPerSquareCentimeter),
                        mzdata::params::CURIE {
                            controlled_vocabulary: ControlledVocabulary::MS,
                            accession: 1001581,
                        } => im_param
                            .curie(im_type)
                            .name("FAIMS compensation voltage")
                            .value(im)
                            .unit(Unit::Volt),
                        mzdata::params::CURIE {
                            controlled_vocabulary: ControlledVocabulary::MS,
                            accession: 1003371,
                        } => im_param
                            .curie(im_type)
                            .name("SELEXION compensation voltage")
                            .value(im)
                            .unit(Unit::Volt),
                        _ => todo!("Don't know how to deal with {im_type}"),
                    }
                    .build();
                    si.add_param(im_param);
                }

                for p in selected_ion.parameters.iter().map(|p| p.into()) {
                    si.add_param(p);
                }
                prec.add_ion(si);
            }
            descr.precursor = Some(prec);
            // Does not yet support MSn for n > 2
        }
        Ok(descr)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_small() -> io::Result<()> {
        let mut reader = MzPeaksReader::new("small.mzpeak")?;
        let descr= reader.get_spectrum_metadata(0)?;
        eprintln!("{descr:?}");
        Ok(())
    }

    #[test]
    fn test_small2() -> io::Result<()> {
        let mut reader = MzPeaksReader::new("small.mzpeak")?;
        let arrays = reader.get_spectrum_arrays(0)?;
        eprintln!("Read {} points", arrays.mzs()?.len());
        Ok(())
    }
}
