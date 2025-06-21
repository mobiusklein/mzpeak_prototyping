use std::{
    collections::HashMap,
    fs::File,
    io,
    path::{Path, PathBuf},
};

#[allow(unused)]
use arrow::{
    array::{
        Array, ArrayRef, AsArray, BooleanArray, Float32Array, Float64Array, Int8Array, Int32Array,
        Int64Array, LargeListArray, LargeStringArray, RecordBatch, StructArray, UInt8Array,
        UInt32Array, UInt64Array,
    },
    datatypes::{DataType, FieldRef},
};

use mzdata::{
    params::{ParamDescribed, Unit},
    prelude::PrecursorSelection,
    spectrum::{
        ArrayType, BinaryArrayMap, DataArray, ScanEvent, ScanPolarity, SpectrumDescription,
    },
};
use mzpeaks::{coordinate::SimpleInterval, prelude::Span1D};

use crate::archive::ZipArchiveReader;

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

use crate::{CURIE, PrecursorEntry, SelectedIonEntry, curie};

#[allow(unused)]
use crate::{
    index::{
        PageIndex, PageIndexEntry, PageIndexType, read_f32_page_index_from,
        read_f64_page_index_from, read_i32_page_index_from, read_i64_page_index_from,
        read_u8_page_index_from, read_u32_page_index_from, read_u64_page_index_from,
    },
    param::{DataProcessing, FileDescription, InstrumentConfiguration, Software},
    peak_series::{ArrayIndex, SerializedArrayIndex},
};

pub struct MzPeakReaderMetadata {
    pub file_description: FileDescription,
    pub instrument_configurations: Vec<InstrumentConfiguration>,
    pub software_list: Vec<Software>,
    pub data_processing_list: Vec<DataProcessing>,
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

pub struct MzPeakReader {
    #[allow(unused)]
    path: PathBuf,
    handle: ZipArchiveReader,
    pub metadata: MzPeakReaderMetadata,
    pub query_indices: QueryIndex,
}

impl MzPeakReader {
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().into();
        let handle = File::open(&path)?;
        let mut handle = ZipArchiveReader::new(handle)?;

        let (metadata, query_indices) = Self::load_indices_from(&mut handle)?;

        let this = Self {
            path,
            handle,
            metadata,
            query_indices,
        };
        Ok(this)
    }

    fn load_indices_from(
        handle: &mut ZipArchiveReader,
    ) -> io::Result<(MzPeakReaderMetadata, QueryIndex)> {
        let spectrum_metadata_reader = handle.spectrum_metadata()?;
        let spectrum_data_reader = handle.spectra_data()?;

        let mut file_description: FileDescription = Default::default();
        let mut instrument_configurations: Vec<InstrumentConfiguration> = Default::default();
        let mut software_list: Vec<Software> = Default::default();
        let mut data_processing_list: Vec<DataProcessing> = Default::default();

        let mut spectrum_array_indices: ArrayIndex = Default::default();
        let mut chromatogram_array_indices: ArrayIndex = Default::default();

        for kv in spectrum_metadata_reader
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .into_iter()
            .flatten()
            .chain(
                spectrum_data_reader
                    .metadata()
                    .file_metadata()
                    .key_value_metadata()
                    .into_iter()
                    .flatten(),
            )
        {
            match kv.key.as_str() {
                "spectrum_array_index" => {
                    if let Some(val) = kv.value.as_ref() {
                        let array_index: SerializedArrayIndex = serde_json::from_str(&val)?;
                        spectrum_array_indices = array_index.into();
                    } else {
                        log::warn!("spectrum array index was empty");
                    }
                }
                "chromatogram_array_index" => {
                    if let Some(val) = kv.value.as_ref() {
                        let array_index: SerializedArrayIndex = serde_json::from_str(&val)?;
                        chromatogram_array_indices = array_index.into();
                    } else {
                        log::warn!("chromatogram array index was empty");
                    }
                }
                "file_description" => {
                    if let Some(val) = kv.value.as_ref() {
                        file_description = serde_json::from_str(&val)?;
                    } else {
                        log::warn!("file description was empty");
                    }
                }
                "instrument_configuration_list" => {
                    if let Some(val) = kv.value.as_ref() {
                        instrument_configurations = serde_json::from_str(&val)?;
                    } else {
                        log::warn!("instrument configurations list was empty for",);
                    }
                }
                "data_processing_method_list" => {
                    if let Some(val) = kv.value.as_ref() {
                        data_processing_list = serde_json::from_str(&val)?;
                    } else {
                        log::warn!("data processing method list was empty");
                    }
                }
                "software_list" => {
                    if let Some(val) = kv.value.as_ref() {
                        software_list = serde_json::from_str(&val)?;
                    } else {
                        log::warn!("software list was empty");
                    }
                }
                _ => {}
            }
        }

        let pq_schema = spectrum_metadata_reader.parquet_schema();

        let mut query_index = QueryIndex::default();
        query_index.spectrum_index_index = read_u64_page_index_from(
            &spectrum_metadata_reader.metadata(),
            &pq_schema,
            "spectrum.index",
        )
        .unwrap_or_default();
        query_index.spectrum_time_index = read_f32_page_index_from(
            &spectrum_metadata_reader.metadata(),
            &pq_schema,
            "spectrum.time",
        )
        .unwrap_or_default();
        query_index.spectrum_ms_level_index = read_u8_page_index_from(
            &spectrum_metadata_reader.metadata(),
            &pq_schema,
            "spectrum.ms_level",
        )
        .unwrap_or_default();
        query_index.scan_index = read_u64_page_index_from(
            &spectrum_metadata_reader.metadata(),
            &pq_schema,
            "scan.spectrum_index",
        )
        .unwrap_or_default();
        query_index.precursor_index = read_u64_page_index_from(
            &spectrum_metadata_reader.metadata(),
            &pq_schema,
            "precursor.spectrum_index",
        )
        .unwrap_or_default();
        query_index.selected_ion_index = read_u64_page_index_from(
            &spectrum_metadata_reader.metadata(),
            &pq_schema,
            "selected_ion.spectrum_index",
        )
        .unwrap_or_default();

        let peak_pq_schema = spectrum_data_reader.parquet_schema();

        query_index.spectrum_point_spectrum_index = read_u64_page_index_from(
            &spectrum_data_reader.metadata(),
            &peak_pq_schema,
            &format!("{}.spectrum_index", spectrum_array_indices.prefix),
        )
        .unwrap_or_default();

        for (arr, entry) in spectrum_array_indices.iter() {
            if matches!(arr, ArrayType::MZArray) {
                query_index.spectrum_mz_index = read_f64_page_index_from(
                    &spectrum_data_reader.metadata(),
                    &peak_pq_schema,
                    &entry.path,
                )
                .unwrap_or_default();
            } else if arr.is_ion_mobility() {
                query_index.spectrum_im_index = read_f64_page_index_from(
                    &spectrum_data_reader.metadata(),
                    &peak_pq_schema,
                    &entry.path,
                )
                .unwrap_or_default();
            }
        }

        let bundle = MzPeakReaderMetadata {
            file_description,
            instrument_configurations,
            software_list,
            data_processing_list,
            spectrum_array_indices,
            chromatogram_array_indices,
        };

        Ok((bundle, query_index))
    }

    /// Read the complete data arrays for the spectrum at `index`
    pub fn get_spectrum_arrays(&mut self, index: u64) -> io::Result<BinaryArrayMap> {
        let builder = self.handle.spectra_data()?;

        let pq_schema = builder.parquet_schema();

        let rows = self
            .query_indices
            .spectrum_point_spectrum_index
            .row_selection_contains(index);

        let predicate_mask = ProjectionMask::columns(
            builder.parquet_schema(),
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

        let proj = ProjectionMask::columns(
            &pq_schema,
            [self.metadata.spectrum_array_indices.prefix.as_str()],
        );

        let reader = builder
            .with_row_selection(rows)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .with_projection(proj)
            .build()?;

        let mut bin_map = HashMap::new();
        for (k, v) in self.metadata.spectrum_array_indices.iter() {
            let dtype = crate::peak_series::arrow_to_array_type(&v.data_type).unwrap();
            bin_map.insert(&v.name, DataArray::from_name_and_type(k, dtype));
        }

        for batch in reader.flatten() {
            let points = batch.column(0).as_struct();
            for (f, arr) in points.fields().iter().zip(points.columns()) {
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
                    }
                    DataType::Float64 => {
                        let buf: &Float64Array = arr.as_primitive();
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

        let mut out = BinaryArrayMap::new();
        for v in bin_map.into_values() {
            out.add(v);
        }
        Ok(out)
    }

    fn load_scan_events_from(
        &self,
        scan_arr: &StructArray,
        param_fields: &[FieldRef],
        scan_accumulator: &mut Vec<(u64, ScanEvent)>,
    ) {
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
            for (pos, index) in index_arr.iter().enumerate().filter(|(_, v)| v.is_some()) {
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

                scan_accumulator.push((index.unwrap(), event));
            }
        }
    }

    fn load_metadata_from_slice_into_description(
        &self,
        spec_arr: &StructArray,
        descr: &mut SpectrumDescription,
        offset: usize,
        param_fields: &[FieldRef],
    ) {
        let idx_arr: &UInt64Array = spec_arr.column(0).as_any().downcast_ref().unwrap();
        let idx_val = idx_arr.value(offset);
        descr.index = idx_val as usize;

        let id_arr: &LargeStringArray = spec_arr.column(1).as_string();
        let id_val = id_arr.value(offset);
        descr.id = id_val.to_string();

        let ms_level_arr: &UInt8Array = spec_arr.column(2).as_any().downcast_ref().unwrap();
        let ms_level_val = ms_level_arr.value(offset);
        descr.ms_level = ms_level_val;

        let polarity_arr: &Int8Array = spec_arr.column(4).as_any().downcast_ref().unwrap();
        let polarity_val = polarity_arr.value(offset);
        match polarity_val {
            1 => descr.polarity = ScanPolarity::Positive,
            -1 => descr.polarity = ScanPolarity::Negative,
            _ => {
                todo!("Don't know how to deal with polarity {polarity_val}")
            }
        }

        let continuity_array = spec_arr.column(5).as_struct();
        let cv_id_arr: &UInt8Array = continuity_array.column(0).as_any().downcast_ref().unwrap();
        let accession_arr: &UInt32Array =
            continuity_array.column(1).as_any().downcast_ref().unwrap();
        let continuity_curie = CURIE::new(cv_id_arr.value(offset), accession_arr.value(offset));

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
                .value(mz_arr.value(offset))
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
                .value(mz_arr.value(offset))
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
                .value(mz_arr.value(offset))
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
                .value(arr.value(offset))
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
                .value(arr.value(offset))
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
                .value(arr.value(offset))
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
                .value(arr.value(offset))
                .build();
            descr.add_param(p);
        }

        let params_array: &LargeListArray =
            spec_arr.column_by_name("parameters").unwrap().as_list();
        let params = params_array.value(offset);
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

    pub fn get_spectrum_index_range_for_time_range(
        &mut self,
        time_range: SimpleInterval<f32>,
    ) -> io::Result<(HashMap<u64, f32>, SimpleInterval<u64>)> {
        let rows = self
            .query_indices
            .spectrum_time_index
            .row_selection_overlaps(&time_range);
        let builder = self.handle.spectrum_metadata()?;

        let predicate_mask = ProjectionMask::columns(builder.parquet_schema(), ["spectrum.time"]);

        let predicate = ArrowPredicateFn::new(predicate_mask, move |batch| {
            let times: &Float32Array = batch
                .column(0)
                .as_struct()
                .column(0)
                .as_any()
                .downcast_ref()
                .unwrap();

            Ok(times
                .iter()
                .map(|v| v.map(|v| time_range.contains(&v)))
                .collect())
        });

        let proj = ProjectionMask::columns(builder.parquet_schema(), ["spectrum.index", "spectrum.time"]);

        let reader = builder
            .with_row_selection(rows)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .with_projection(proj)
            .build()?;

        let mut min = u64::MAX;
        let mut max = 0;

        let mut times: HashMap<u64, f32> = HashMap::new();

        for batch in reader.flatten() {
            let root = batch
                .column(0)
                .as_struct();
            let arr: &UInt64Array =  root
                .column(0)
                .as_any()
                .downcast_ref()
                .unwrap();
            let time_arr: &Float32Array = root
                .column(1)
                .as_any()
                .downcast_ref()
                .unwrap();
            for (val, time) in arr.iter().flatten().zip(time_arr.iter().flatten()) {
                min = min.min(val);
                max = max.max(val);
                times.insert(val, time);
            }
        }
        Ok((times, SimpleInterval::new(min, max)))
    }

    pub fn extract_peaks(&mut self, time_range: SimpleInterval<f32>, mz_range: Option<SimpleInterval<f64>>, ion_mobility_range: Option<SimpleInterval<f64>>) -> io::Result<(ParquetRecordBatchReader, HashMap<u64, f32>)> {
        let (time_index, index_range) = self.get_spectrum_index_range_for_time_range(time_range)?;

        let mut rows = self.query_indices.spectrum_point_spectrum_index.row_selection_overlaps(&index_range);
        if let Some(mz_range) = mz_range.as_ref() {
            rows = rows.intersection(&self.query_indices.spectrum_mz_index.row_selection_overlaps(mz_range));
        }
        if let Some(ion_mobility_range) = ion_mobility_range.as_ref() {
            rows = rows.union(&self.query_indices.spectrum_im_index.row_selection_overlaps(&ion_mobility_range));
        }

        let sidx = format!("{}.spectrum_index", self.metadata.spectrum_array_indices.prefix);

        let mut fields: Vec<&str> = Vec::new();

        fields.push(&sidx);

        if let Some(e) = self.metadata.spectrum_array_indices.get(&ArrayType::MZArray) {
            fields.push(e.path.as_str())
        }

        if let Some(e) = self.metadata.spectrum_array_indices.get(&ArrayType::IntensityArray) {
            fields.push(e.path.as_str())
        }

        for (k, v) in self.metadata.spectrum_array_indices.iter() {
            if k.is_ion_mobility() {
                fields.push(v.path.as_str());
                break;
            }
        }

        let builder = self.handle.spectra_data()?;

        let proj = ProjectionMask::columns(builder.parquet_schema(), fields.iter().copied());
        let predicate_mask = proj.clone();

        let predicate = ArrowPredicateFn::new(predicate_mask, move |batch| {
            let root = batch.column(0).as_struct();
            let spectrum_index: &UInt64Array = root
                .column(0)
                .as_any()
                .downcast_ref()
                .unwrap();

            let it = spectrum_index.iter().map(|v| {
                v.map(|v| index_range.contains(&v))
            });

            match (mz_range, ion_mobility_range) {
                (None, None) => Ok(it.collect()),
                (None, Some(ion_mobility_range)) => {
                    let im_array: &Float64Array = root.column(1).as_any().downcast_ref().unwrap();
                    let it2 = im_array.iter().map(|v| v.map(|v| ion_mobility_range.contains(&v)));
                    let it = it.zip(it2).map(|(a, b)| Some(a.is_some_and(|a| a && b.unwrap_or_default())));
                    Ok(it.collect())
                },
                (Some(mz_range), None) => {
                    let mz_array: &Float64Array = root.column(1).as_any().downcast_ref().unwrap();
                    let it2 = mz_array.iter().map(|v| v.map(|v| mz_range.contains(&v)));
                    let it = it.zip(it2).map(|(a, b)| Some(a.is_some_and(|a| a && b.unwrap_or_default())));
                    Ok(it.collect())
                },
                (Some(mz_range), Some(ion_mobility_range)) => {
                    let mz_array: &Float64Array = root.column(1).as_any().downcast_ref().unwrap();
                    let im_array: &Float64Array = root.column(2).as_any().downcast_ref().unwrap();
                    let it2 = mz_array.iter().map(|v| v.map(|v| mz_range.contains(&v)));
                    let it3 = im_array.iter().map(|v| v.map(|v| ion_mobility_range.contains(&v)));
                    let it = it.zip(it2).map(|(a, b)| Some(a.is_some_and(|a| a && b.unwrap_or_default())));
                    let it = it.zip(it3).map(|(a, b)| Some(a.is_some_and(|a| a && b.unwrap_or_default())));
                    Ok(it.collect())
                },
            }
        });

        let reader: ParquetRecordBatchReader = builder
            .with_row_selection(rows)
            .with_projection(proj)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .build()?;

        Ok((reader, time_index))
    }

    /// Read load descriptive metadata for the spectrum at `index`
    pub fn get_spectrum_metadata(&mut self, index: u64) -> io::Result<SpectrumDescription> {
        let builder = self.handle.spectrum_metadata()?;

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
            builder.parquet_schema(),
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
            .build()?;

        // let curie_fields: Vec<FieldRef> =
        //     SchemaLike::from_type::<CURIE>(TracingOptions::new()).unwrap();

        let param_fields: Vec<FieldRef> =
            SchemaLike::from_type::<crate::Param>(Default::default()).unwrap();

        let precursor_fields: Vec<FieldRef> =
            SchemaLike::from_type::<PrecursorEntry>(Default::default()).unwrap();

        let selected_ion_fields: Vec<FieldRef> =
            SchemaLike::from_type::<SelectedIonEntry>(Default::default()).unwrap();

        let mut descr = SpectrumDescription::default();

        let mut precursors: Vec<PrecursorEntry> = Vec::new();
        let mut selected_ions: Vec<SelectedIonEntry> = Vec::new();

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
                self.load_metadata_from_slice_into_description(
                    &spec_arr,
                    &mut descr,
                    0,
                    &param_fields,
                );
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
                        serde_arrow::from_arrow::<Vec<PrecursorEntry>, _>(
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
                        serde_arrow::from_arrow::<Vec<SelectedIonEntry>, _>(
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

                let si = selected_ion.to_mzdata();
                prec.add_ion(si);
            }
            descr.precursor = Some(prec);
            // Does not yet support MSn for n > 2
        }
        Ok(descr)
    }

    /// Load the descriptive metadata for all spectra
    pub fn load_all_spectrum_metadata(&mut self) -> io::Result<Vec<SpectrumDescription>> {
        let builder = self.handle.spectrum_metadata()?;

        let mut rows = self
            .query_indices
            .spectrum_index_index
            .row_selection_is_not_null();

        rows = rows.union(&self.query_indices.scan_index.row_selection_is_not_null());
        rows = rows.union(
            &self
                .query_indices
                .precursor_index
                .row_selection_is_not_null(),
        );
        rows = rows.union(
            &self
                .query_indices
                .selected_ion_index
                .row_selection_is_not_null(),
        );

        let predicate_mask = ProjectionMask::columns(
            builder.parquet_schema(),
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

            let it = spectrum_index.iter().map(|val| val.is_some());
            let it = scan_spectrum_index
                .iter()
                .map(|val| val.is_some())
                .zip(it)
                .map(|(a, b)| a || b);

            let it = precursor_spectrum_index
                .iter()
                .map(|val| val.is_some())
                .zip(it)
                .map(|(a, b)| a || b);

            let it = selected_ion_spectrum_index
                .iter()
                .map(|val| val.is_some())
                .zip(it)
                .map(|(a, b)| a || b);

            Ok(it.map(Some).collect())
        });

        let reader = builder
            .with_row_selection(rows)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .build()?;

        let mut descriptions: Vec<SpectrumDescription> = Default::default();
        let mut precursors: Vec<PrecursorEntry> = Vec::new();
        let mut selected_ions: Vec<SelectedIonEntry> = Vec::new();
        let mut scan_events: Vec<(u64, ScanEvent)> = Vec::new();

        let param_fields: Vec<FieldRef> =
            SchemaLike::from_type::<crate::Param>(Default::default()).unwrap();
        let precursor_fields: Vec<FieldRef> =
            SchemaLike::from_type::<PrecursorEntry>(Default::default()).unwrap();
        let selected_ion_fields: Vec<FieldRef> =
            SchemaLike::from_type::<SelectedIonEntry>(Default::default()).unwrap();

        for batch in reader.flatten() {
            let spec_arr = batch.column_by_name("spectrum").unwrap().as_struct();
            let index_arr: &UInt64Array = spec_arr
                .column_by_name("index")
                .unwrap()
                .as_any()
                .downcast_ref()
                .unwrap();
            for (i, val) in index_arr.iter().enumerate() {
                if val.is_some() {
                    let mut descr = SpectrumDescription::default();
                    self.load_metadata_from_slice_into_description(
                        spec_arr,
                        &mut descr,
                        i,
                        &param_fields,
                    );
                    descriptions.push(descr);
                }
            }

            if let Some(scan_arr) = batch.column_by_name("scan").map(|arr| arr.as_struct()) {
                self.load_scan_events_from(scan_arr, &param_fields, &mut scan_events);
            }

            let precursor_arr = batch.column_by_name("precursor").unwrap().as_struct();
            {
                let index_arr: &UInt64Array = precursor_arr
                    .column_by_name("spectrum_index")
                    .unwrap()
                    .as_any()
                    .downcast_ref()
                    .unwrap();

                for (pos, _) in index_arr.iter().enumerate().filter(|i| i.1.is_some()) {
                    let row = precursor_arr.slice(pos, 1);
                    precursors.extend(
                        serde_arrow::from_arrow::<Vec<PrecursorEntry>, _>(
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

                for (pos, _) in index_arr.iter().enumerate().filter(|i| i.1.is_some()) {
                    let row = selected_ion_arr.slice(pos, 1);
                    selected_ions.extend(
                        serde_arrow::from_arrow::<Vec<SelectedIonEntry>, _>(
                            &selected_ion_fields,
                            row.columns(),
                        )
                        .unwrap(),
                    );
                }
            }
        }

        precursors.sort_unstable_by(|a, b| {
            a.spectrum_index
                .cmp(&b.spectrum_index)
                .then_with(|| a.precursor_index.cmp(&b.precursor_index))
        });

        let mut precursors: Vec<(Option<u64>, mzdata::spectrum::Precursor)> = precursors
            .into_iter()
            .map(|p| (p.spectrum_index, p.to_mzdata()))
            .collect();

        for sie in selected_ions.iter() {
            let si = sie.to_mzdata();
            let prec = &mut precursors[sie.precursor_index.unwrap() as usize].1;
            prec.add_ion(si);
        }

        for (idx, scan) in scan_events {
            descriptions[idx as usize].acquisition.scans.push(scan);
        }

        for (idx, precursor) in precursors {
            descriptions[idx.unwrap() as usize].precursor = Some(precursor);
        }

        Ok(descriptions)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_small() -> io::Result<()> {
        let mut reader = MzPeakReader::new("small.mzpeak")?;
        let descr = reader.get_spectrum_metadata(0)?;
        assert_eq!(descr.index, 0);
        eprintln!("{descr:?}");
        Ok(())
    }

    #[test]
    fn test_small3() -> io::Result<()> {
        let mut reader = MzPeakReader::new("small.mzpeak")?;
        let out = reader.load_all_spectrum_metadata()?;
        assert_eq!(out.len(), 48);
        eprintln!("{}", out.len());
        Ok(())
    }

    #[test]
    fn test_small2() -> io::Result<()> {
        let mut reader = MzPeakReader::new("small.mzpeak")?;
        let arrays = reader.get_spectrum_arrays(0)?;
        eprintln!("Read {} points", arrays.mzs()?.len());
        Ok(())
    }

    #[test]
    fn test_eic() -> io::Result<()> {
        let mut reader = MzPeakReader::new("small.mzpeak")?;

        let (it, _time_index) = reader.extract_peaks(
            (0.3..0.5).into(),
            Some((800.0..820.0).into()),
            None
        )?;

        for batch in it.flatten() {
            eprintln!("{:?}", batch);
        }
        Ok(())
    }
}
