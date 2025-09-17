use std::{io, sync::Arc};

use arrow::{
    array::{
        Array, AsArray, BooleanArray, Float32Array, Float64Array, Int8Array, Int32Array,
        Int64Array, LargeListArray, LargeStringArray, StructArray, UInt8Array, UInt32Array,
        UInt64Array,
    },
    datatypes::{
        DataType, FieldRef, Float32Type, Float64Type, Int32Type, Int64Type, UInt8Type, UInt32Type,
        UInt64Type,
    },
};
use mzdata::{
    io::OffsetIndex,
    meta::{self, SpectrumType},
    params::Unit,
    prelude::*,
    spectrum::{ScanEvent, ScanPolarity, SelectedIon, SpectrumDescription},
};
use parquet::{
    arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder},
    file::reader::ChunkReader,
    schema::types::SchemaDescriptor,
};
use serde_arrow::schema::SchemaLike;

use crate::{
    CURIE, MS_CV_ID,
    archive::ZipArchiveReader,
    buffer_descriptors::{ArrayIndex, SerializedArrayIndex},
    filter::RegressionDeltaModel,
    param::MetadataColumn,
    reader::index::{QueryIndex, SpectrumPointIndex},
    spectrum::ScanWindowEntry,
};

pub struct ReaderMetadata {
    pub(crate) mz_metadata: mzdata::meta::FileMetadataConfig,
    pub spectrum_array_indices: Arc<ArrayIndex>,
    pub chromatogram_array_indices: Arc<ArrayIndex>,
    pub spectrum_id_index: OffsetIndex,
    pub(crate) model_deltas: Vec<Option<Vec<f64>>>,
    pub(crate) auxliary_array_counts: Vec<u32>,
    pub(crate) spectrum_metadata_map: Option<Vec<MetadataColumn>>,
    #[allow(unused)]
    pub(crate) scan_metadata_map: Option<Vec<MetadataColumn>>,
    #[allow(unused)]
    pub(crate) selected_ion_metadata_map: Option<Vec<MetadataColumn>>,
    #[allow(unused)]
    pub(crate) chromatogram_metadata_map: Option<Vec<MetadataColumn>>,
    pub peak_indices: Option<PeakMetadata>,
}

impl ReaderMetadata {
    pub fn new(
        mz_metadata: mzdata::meta::FileMetadataConfig,
        spectrum_array_indices: Arc<ArrayIndex>,
        chromatogram_array_indices: Arc<ArrayIndex>,
        spectrum_id_index: OffsetIndex,
        model_deltas: Vec<Option<Vec<f64>>>,
        auxliary_array_counts: Vec<u32>,
        spectrum_metadata_map: Option<Vec<MetadataColumn>>,
        scan_metadata_map: Option<Vec<MetadataColumn>>,
        selected_ion_metadata_map: Option<Vec<MetadataColumn>>,
        chromatogram_metadata_map: Option<Vec<MetadataColumn>>,
        peak_indices: Option<PeakMetadata>,
    ) -> Self {
        Self {
            mz_metadata,
            spectrum_array_indices,
            chromatogram_array_indices,
            spectrum_id_index,
            model_deltas,
            auxliary_array_counts,
            spectrum_metadata_map,
            scan_metadata_map,
            selected_ion_metadata_map,
            chromatogram_metadata_map,
            peak_indices,
        }
    }

    pub fn model_deltas_for(&self, index: usize) -> Option<Vec<f64>> {
        self.model_deltas.get(index).cloned().unwrap_or_default()
    }

    pub fn model_deltas_for_conv(&self, index: usize) -> Option<RegressionDeltaModel<f64>> {
        self.model_deltas_for(index)
            .map(|v| RegressionDeltaModel::from(v))
    }

    pub fn auxliary_array_counts(&self) -> &[u32] {
        &self.auxliary_array_counts
    }

    pub fn file_metadata(&self) -> &mzdata::meta::FileMetadataConfig {
        &self.mz_metadata
    }
}

impl MSDataFileMetadata for ReaderMetadata {
    mzdata::delegate_impl_metadata_trait!(mz_metadata);
}

pub(crate) fn build_spectrum_index(
    handle: &ZipArchiveReader,
    pq_schema: &SchemaDescriptor,
) -> io::Result<OffsetIndex> {
    let mut spectrum_id_index = OffsetIndex::new("spectrum".into());
    for batch in handle
        .spectrum_metadata()?
        .with_projection(ProjectionMask::columns(
            pq_schema,
            ["spectrum.id", "spectrum.index"],
        ))
        .build()?
        .flatten()
    {
        let root = batch.column(0).as_struct();
        let ids = root.column_by_name("id").unwrap().as_string::<i64>();
        let indices: &UInt64Array = root
            .column_by_name("index")
            .unwrap()
            .as_any()
            .downcast_ref()
            .unwrap();
        for (id, idx) in ids.iter().zip(indices.iter()) {
            if let Some(id) = id {
                spectrum_id_index.insert(id, idx.unwrap());
            }
        }
    }
    spectrum_id_index.init = true;
    Ok(spectrum_id_index)
}

#[derive(Debug, Default, Clone)]
pub struct PeakMetadata {
    pub array_indices: ArrayIndex,
    pub query_index: SpectrumPointIndex,
}

impl PeakMetadata {
    pub fn new(array_indices: ArrayIndex, query_index: SpectrumPointIndex) -> Self {
        Self {
            array_indices,
            query_index,
        }
    }

    pub fn from_metadata<T: ChunkReader>(
        reader: &ParquetRecordBatchReaderBuilder<T>,
    ) -> Option<Self> {
        let metadata = reader.metadata();
        let mut this = Self::default();
        let mut has_arrays = false;
        if let Some(kvs) = metadata.file_metadata().key_value_metadata() {
            for kv in kvs {
                match kv.key.as_str() {
                    "spectrum_array_index" => {
                        if let Some(data) = kv.value.as_deref() {
                            this.array_indices = ArrayIndex::from_json(data);
                            has_arrays = true;
                        }
                    }
                    _ => {}
                }
            }
        }
        if has_arrays {
            let index = SpectrumPointIndex::from_reader(reader, &this.array_indices);
            this.query_index = index;
            Some(this)
        } else {
            None
        }
    }
}

/// Load the various metadata, indices and reference data
pub(crate) fn load_indices_from(
    handle: &mut ZipArchiveReader,
) -> io::Result<(ReaderMetadata, QueryIndex)> {
    let spectrum_metadata_reader = handle.spectrum_metadata()?;
    let spectrum_data_reader = handle.spectra_data()?;

    let mut mz_metadata: meta::FileMetadataConfig = Default::default();
    let mut spectrum_array_indices: ArrayIndex = Default::default();
    let mut chromatogram_array_indices: ArrayIndex = Default::default();
    let mut spectrum_metadata_mapping = None;
    let mut scan_metadata_mapping = None;
    let mut selected_ion_metadata_mapping = None;
    let mut chromatogram_metadata_mapping = None;

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
                    let file_description: crate::param::FileDescription =
                        serde_json::from_str(&val)?;
                    *mz_metadata.file_description_mut() = file_description.into();
                } else {
                    log::warn!("file description was empty");
                }
            }
            "instrument_configuration_list" => {
                if let Some(val) = kv.value.as_ref() {
                    let instrument_configurations: Vec<crate::param::InstrumentConfiguration> =
                        serde_json::from_str(&val)?;
                    for ic in instrument_configurations {
                        mz_metadata
                            .instrument_configurations_mut()
                            .insert(ic.id, ic.into());
                    }
                } else {
                    log::warn!("instrument configurations list was empty for",);
                }
            }
            "data_processing_method_list" => {
                if let Some(val) = kv.value.as_ref() {
                    let data_processing_list: Vec<crate::param::DataProcessing> =
                        serde_json::from_str(&val)?;
                    for dp in data_processing_list {
                        mz_metadata.data_processings_mut().push(dp.into());
                    }
                } else {
                    log::warn!("data processing method list was empty");
                }
            }
            "sample_list" => {
                if let Some(val) = kv.value.as_ref() {
                    let meta_list: Vec<crate::param::Sample> = serde_json::from_str(&val)?;
                    for sw in meta_list {
                        mz_metadata.samples_mut().push(sw.into());
                    }
                } else {
                    log::warn!("sample list was empty");
                }
            }
            "software_list" => {
                if let Some(val) = kv.value.as_ref() {
                    let software_list: Vec<crate::param::Software> = serde_json::from_str(&val)?;
                    for sw in software_list {
                        mz_metadata.softwares_mut().push(sw.into());
                    }
                } else {
                    log::warn!("software list was empty");
                }
            }
            "run" => {
                if let Some(val) = kv.value.as_ref() {
                    let run: meta::MassSpectrometryRun = serde_json::from_str(&val)?;
                    *mz_metadata.run_description_mut().unwrap() = run;
                } else {
                    log::warn!("run was empty")
                }
            }
            "spectrum_column_metadata_mapping" => {
                if let Some(val) = kv.value.as_ref() {
                    let metacols: Vec<MetadataColumn> = serde_json::from_str(&val)?;
                    spectrum_metadata_mapping = Some(metacols);
                }
            }
            "scan_column_metadata_mapping" => {
                if let Some(val) = kv.value.as_ref() {
                    let metacols: Vec<MetadataColumn> = serde_json::from_str(&val)?;
                    scan_metadata_mapping = Some(metacols);
                }
            },
            "selected_ion_column_metadata_mapping" => {
                if let Some(val) = kv.value.as_ref() {
                    let metacols: Vec<MetadataColumn> = serde_json::from_str(&val)?;
                    selected_ion_metadata_mapping = Some(metacols);
                }
            },
            "chromatogram_column_metadata_mapping" => {
                if let Some(val) = kv.value.as_ref() {
                    let metacols: Vec<MetadataColumn> = serde_json::from_str(&val)?;
                    chromatogram_metadata_mapping = Some(metacols);
                }
            },
            _ => {}
        }
    }

    let pq_schema = spectrum_metadata_reader.parquet_schema();

    let spectrum_id_index = build_spectrum_index(&handle, pq_schema)?;

    let mut query_index = QueryIndex::default();
    query_index.populate_spectrum_metadata_indices(&spectrum_metadata_reader);
    query_index.populate_spectrum_data_indices(&spectrum_data_reader, &spectrum_array_indices);

    let peak_metadata = handle
        .spectrum_peaks()
        .ok()
        .and_then(|r| PeakMetadata::from_metadata(&r));

    let bundle = ReaderMetadata::new(
        mz_metadata,
        Arc::new(spectrum_array_indices),
        Arc::new(chromatogram_array_indices),
        spectrum_id_index,
        Vec::new(),
        Vec::new(),
        spectrum_metadata_mapping,
        scan_metadata_mapping,
        selected_ion_metadata_mapping,
        chromatogram_metadata_mapping,
        peak_metadata,
    );

    Ok((bundle, query_index))
}

pub(crate) struct MzSpectrumBuilder<'a> {
    pub(crate) descriptions: &'a mut [SpectrumDescription],
    pub(crate) metadata_map: &'a [MetadataColumn],
    pub(crate) base_offset: usize,
    pub(crate) offsets: Vec<usize>,
}

impl<'a> MzSpectrumBuilder<'a> {
    pub(crate) fn new(
        descriptions: &'a mut [SpectrumDescription],
        metadata_map: &'a [MetadataColumn],
        base_offset: usize,
    ) -> Self {
        Self {
            descriptions,
            metadata_map,
            base_offset,
            offsets: Vec::new(),
        }
    }

    fn visit_index(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index).as_primitive::<UInt64Type>();
        let mut offsets = Vec::with_capacity(self.descriptions.len());
        let mut j = 0;
        for (i, descr) in self.descriptions.iter_mut().enumerate() {
            while arr.is_null(self.base_offset + i + j) {
                j += 1;
            }
            let val = arr.value(self.base_offset + i + j);
            offsets.push(self.base_offset + i + j);
            descr.index = val as usize;
        }
        self.offsets = offsets
    }

    fn visit_id(&mut self, spec_arr: &StructArray, index: usize) {
        let arr: &LargeStringArray = spec_arr.column(index).as_string();
        for (i, descr) in self
            .offsets
            .iter()
            .copied()
            .zip(self.descriptions.iter_mut())
        {
            let val = arr.value(i);
            descr.id = val.to_string();
        }
    }

    fn visit_ms_level(&mut self, spec_arr: &StructArray, index: usize, _metacol: &MetadataColumn) {
        let arr = spec_arr.column(index).as_primitive::<UInt8Type>();
        for (i, descr) in self
            .offsets
            .iter()
            .copied()
            .zip(self.descriptions.iter_mut())
        {
            if arr.is_null(i) {
                continue;
            };
            let ms_level_val = arr.value(i);
            descr.ms_level = ms_level_val;
        }
    }

    fn visit_polarity(&mut self, spec_arr: &StructArray, index: usize, _metacol: &MetadataColumn) {
        let polarity_arr: &Int8Array = spec_arr.column(index).as_any().downcast_ref().unwrap();
        for (i, descr) in self
            .offsets
            .iter()
            .copied()
            .zip(self.descriptions.iter_mut())
        {
            let polarity_val = polarity_arr.value(i);
            match polarity_val {
                1 => descr.polarity = ScanPolarity::Positive,
                -1 => descr.polarity = ScanPolarity::Negative,
                _ => {
                    todo!("Don't know how to deal with polarity {polarity_val}")
                }
            }
        }
    }

    fn visit_mz_signal_continuity(
        &mut self,
        spec_arr: &StructArray,
        index: usize,
        _metacol: &MetadataColumn,
    ) {
        let continuity_array = spec_arr.column(index).as_struct();
        let cv_id_arr: &UInt8Array = continuity_array.column(0).as_any().downcast_ref().unwrap();
        let accession_arr: &UInt32Array =
            continuity_array.column(1).as_any().downcast_ref().unwrap();
        for (i, descr) in self
            .offsets
            .iter()
            .copied()
            .zip(self.descriptions.iter_mut())
        {
            if accession_arr.is_null(i) {
                continue;
            };
            let continuity_curie = CURIE::new(cv_id_arr.value(i), accession_arr.value(i));
            descr.signal_continuity = match continuity_curie {
                crate::curie!(MS:1000525) => mzdata::spectrum::SignalContinuity::Unknown,
                crate::curie!(MS:1000127) => mzdata::spectrum::SignalContinuity::Centroid,
                crate::curie!(MS:1000128) => mzdata::spectrum::SignalContinuity::Profile,
                _ => todo!("Don't know how to deal with {continuity_curie}"),
            };
        }
    }

    fn visit_spectrum_type(
        &mut self,
        spec_arr: &StructArray,
        index: usize,
        _metacol: &MetadataColumn,
    ) {
        let spec_type_array = spec_arr.column(index).as_struct();
        let cv_id_arr: &UInt8Array = spec_type_array.column(0).as_any().downcast_ref().unwrap();
        let accession_arr: &UInt32Array =
            spec_type_array.column(1).as_any().downcast_ref().unwrap();

        for (i, descr) in self
            .offsets
            .iter()
            .copied()
            .zip(self.descriptions.iter_mut())
        {
            let spec_type_curie = CURIE::new(cv_id_arr.value(i), accession_arr.value(i));
            let spec_type = SpectrumType::from_accession(spec_type_curie.accession);
            if let Some(spec_type) = spec_type {
                descr.set_spectrum_type(spec_type);
            }
        }
    }

    fn visit_lowest_mz(&mut self, spec_arr: &StructArray, index: usize, _metacol: &MetadataColumn) {
        let arr: &Float64Array = spec_arr.column(index).as_primitive();
        for (i, descr) in self
            .offsets
            .iter()
            .copied()
            .zip(self.descriptions.iter_mut())
        {
            if arr.is_null(i) {
                continue;
            };
            let p = mzdata::Param::builder()
                .curie(mzdata::curie!(MS:1000528))
                .name("lowest observed m/z")
                .unit(Unit::MZ)
                .value(arr.value(i))
                .build();
            descr.add_param(p);
        }
    }

    fn visit_highest_mz(
        &mut self,
        spec_arr: &StructArray,
        index: usize,
        _metacol: &MetadataColumn,
    ) {
        let arr: &Float64Array = spec_arr.column(index).as_primitive();
        for (i, descr) in self
            .offsets
            .iter()
            .copied()
            .zip(self.descriptions.iter_mut())
        {
            if arr.is_null(i) {
                continue;
            };
            let p = mzdata::Param::builder()
                .curie(mzdata::curie!(MS:1000527))
                .name("highest observed m/z")
                .unit(Unit::MZ)
                .value(arr.value(i))
                .build();
            descr.add_param(p);
        }
    }

    fn visit_base_peak_mz(
        &mut self,
        spec_arr: &StructArray,
        index: usize,
        _metacol: &MetadataColumn,
    ) {
        let arr: &Float64Array = spec_arr.column(index).as_primitive();
        for (i, descr) in self
            .offsets
            .iter()
            .copied()
            .zip(self.descriptions.iter_mut())
        {
            if arr.is_null(i) {
                continue;
            };
            let p = mzdata::Param::builder()
                .curie(mzdata::curie!(MS:1000504))
                .name("base peak m/z")
                .unit(Unit::MZ)
                .value(arr.value(i))
                .build();
            descr.add_param(p);
        }
    }

    fn visit_base_peak_intensity(
        &mut self,
        spec_arr: &StructArray,
        index: usize,
        metacol: &MetadataColumn,
    ) {
        let unit = match &metacol.unit {
            crate::param::PathOrCURIE::Path(_items) => todo!(),
            crate::param::PathOrCURIE::CURIE(curie) => Unit::from_curie(&(*curie).into()),
            crate::param::PathOrCURIE::None => Unit::DetectorCounts,
        };

        macro_rules! extract {
            ($arr:expr) => {
                for (i, descr) in self.offsets.iter().copied().zip(self.descriptions.iter_mut()) {
                    if $arr.is_null(i) { continue };
                    let p = mzdata::Param::builder()
                        .curie(mzdata::curie!(MS:1000505))
                        .name("base peak intensity")
                        .unit(unit)
                        .value($arr.value(i))
                        .build();
                    descr.add_param(p);
                }
            };
        }

        let arr = spec_arr.column(index);

        if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            extract!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            extract!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Int32Type>() {
            extract!(arr);
        } else {
            unimplemented!("{:?} not supported for {metacol:?}", arr.data_type())
        }
    }

    fn visit_total_ion_current(
        &mut self,
        spec_arr: &StructArray,
        index: usize,
        metacol: &MetadataColumn,
    ) {
        let unit = match &metacol.unit {
            crate::param::PathOrCURIE::Path(_items) => todo!(),
            crate::param::PathOrCURIE::CURIE(curie) => Unit::from_curie(&(*curie).into()),
            crate::param::PathOrCURIE::None => Unit::DetectorCounts,
        };

        macro_rules! extract {
            ($arr:expr) => {
                for (i, descr) in self.offsets.iter().copied().zip(self.descriptions.iter_mut()) {
                    if $arr.is_null(i) { continue };
                    let p = mzdata::Param::builder()
                        .curie(mzdata::curie!(MS:1000285))
                        .name("total ion current")
                        .unit(unit)
                        .value($arr.value(i))
                        .build();
                    descr.add_param(p);
                }
            };
        }

        let arr = spec_arr.column(index);

        if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            extract!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            extract!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Int32Type>() {
            extract!(arr);
        } else {
            unimplemented!("{:?} not supported for {metacol:?}", arr.data_type())
        }
    }

    fn visit_parameters(&mut self, spec_arr: &StructArray) {
        let params_array: &LargeListArray =
            spec_arr.column_by_name("parameters").unwrap().as_list();
        let param_fields: Vec<FieldRef> =
            SchemaLike::from_type::<crate::param::Param>(Default::default()).unwrap();
        for (i, descr) in self
            .offsets
            .iter()
            .copied()
            .zip(self.descriptions.iter_mut())
        {
            let params = params_array.value(i);
            let params = params.as_struct();

            const SKIP_PARAMS: [CURIE; 6] = [
                crate::curie!(MS:1000505),
                crate::curie!(MS:1000504),
                crate::curie!(MS:1000257),
                crate::curie!(MS:1000285),
                crate::curie!(MS:1000527),
                crate::curie!(MS:1000528),
            ];

            let params: Vec<crate::param::Param> =
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
    }

    fn visit_as_param(&mut self, spec_arr: &StructArray, index: usize, metacol: &MetadataColumn) {
        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }

        let unit = match &metacol.unit {
            crate::param::PathOrCURIE::Path(_items) => todo!(),
            crate::param::PathOrCURIE::CURIE(curie) => Unit::from_curie(&(*curie).into()),
            crate::param::PathOrCURIE::None => Unit::Unknown,
        };

        let accession: Option<mzdata::params::CURIE> = metacol.accession.map(|v| v.into());

        macro_rules! convert {
            ($arr:ident) => {
                for (i, descr) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    if $arr.is_null(i) {
                        continue;
                    };
                    let mut p = mzdata::Param::builder();
                    p = p.name(&metacol.name).value($arr.value(i)).unit(unit);
                    if let Some(acc) = accession {
                        p = p.curie(acc)
                    }
                    descr.add_param(p.build());
                }
            };
        }

        match arr.data_type() {
            DataType::Int32 => {
                let arr: &Int32Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Int64 => {
                let arr: &Int64Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Float32 => {
                let arr: &Float32Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Float64 => {
                let arr: &Float64Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Boolean => {
                let arr: &BooleanArray = spec_arr.column(index).as_boolean();
                convert!(arr);
            }
            DataType::Utf8 => {
                let arr: &LargeStringArray = spec_arr.column(index).as_string();
                convert!(arr);
            }
            DataType::UInt32 => {
                let arr: &UInt32Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::UInt64 => {
                let arr: &UInt64Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            _ => {
                todo!("Unsupported data type {:?}", arr.data_type())
            }
        }
    }

    pub(crate) fn visit(&mut self, spec_arr: &StructArray) {
        // Must visit the index first, to infer null spacing
        self.visit_index(spec_arr, 0);
        self.visit_id(spec_arr, 1);
        for col in self.metadata_map.iter() {
            if let Some(accession) = col.accession {
                if crate::curie!(MS:1000511) == accession {
                    self.visit_ms_level(spec_arr, col.index, col);
                } else if crate::curie!(MS:1000465) == accession {
                    self.visit_polarity(spec_arr, col.index, col);
                } else if crate::curie!(MS:1000525) == accession {
                    self.visit_mz_signal_continuity(spec_arr, col.index, col);
                } else if crate::curie!(MS:1000559) == accession {
                    self.visit_spectrum_type(spec_arr, col.index, col);
                } else if crate::curie!(MS:1000528) == accession {
                    self.visit_lowest_mz(spec_arr, col.index, col);
                } else if crate::curie!(MS:1000527) == accession {
                    self.visit_highest_mz(spec_arr, col.index, col);
                } else if crate::curie!(MS:1000504) == accession {
                    self.visit_base_peak_mz(spec_arr, col.index, col);
                } else if crate::curie!(MS:1000505) == accession {
                    self.visit_base_peak_intensity(spec_arr, col.index, col);
                } else if crate::curie!(MS:1000285) == accession {
                    self.visit_total_ion_current(spec_arr, col.index, col);
                } else {
                    self.visit_as_param(spec_arr, col.index, col);
                }
            }
        }
        self.visit_parameters(spec_arr);
    }
}

pub(crate) struct MzScanBuilder<'a> {
    pub(crate) descriptions: &'a mut [(u64, ScanEvent)],
    pub(crate) metadata_map: &'a [MetadataColumn],
    pub(crate) base_offset: usize,
    pub(crate) offsets: Vec<usize>,
}

impl<'a> MzScanBuilder<'a> {
    pub(crate) fn new(
        descriptions: &'a mut [(u64, ScanEvent)],
        metadata_map: &'a [MetadataColumn],
        base_offset: usize,
        offsets: Vec<usize>,
    ) -> Self {
        Self {
            descriptions,
            metadata_map,
            base_offset,
            offsets,
        }
    }

    fn visit_index(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index).as_primitive::<UInt64Type>();
        let mut offsets = Vec::with_capacity(self.descriptions.len());
        let mut j = 0;
        for (i, (index, _descr)) in self.descriptions.iter_mut().enumerate() {
            while arr.is_null(self.base_offset + i + j) {
                j += 1;
            }
            let val = arr.value(self.base_offset + i + j);
            offsets.push(self.base_offset + i + j);
            *index = val;
        }
        self.offsets = offsets
    }

    fn visit_instrument_configuration_ref(&mut self, spec_arr: &StructArray, index: usize) {
        macro_rules! pack {
            ($arr:ident) => {
                for (i, (_, descr)) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    if $arr.is_null(i) {
                        continue;
                    };
                    descr.instrument_configuration_id = $arr.value(i) as u32;
                }
            };
        }

        let arr = spec_arr.column(index);

        if let Some(arr) = arr.as_primitive_opt::<UInt32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<UInt64Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Int32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Int64Type>() {
            pack!(arr);
        } else {
            unimplemented!("{:?}", arr.data_type());
        }
    }

    fn visit_preset_scan_configuration(&mut self, spec_arr: &StructArray, index: usize) {
        macro_rules! pack {
            ($arr:ident) => {
                for (i, (_, descr)) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    if $arr.is_null(i) {
                        continue;
                    };
                    let p = mzdata::Param::builder()
                        .name("preset scan configuration")
                        .curie(mzdata::curie!(MS:1000616))
                        .value($arr.value(i))
                        .build();
                    descr.add_param(p);
                }
            };
        }
        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }
        if let Some(arr) = arr.as_primitive_opt::<UInt32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<UInt64Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Int32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Int64Type>() {
            pack!(arr);
        } else {
            unimplemented!("{:?}", arr.data_type());
        }
    }

    fn visit_filter_string(&mut self, spec_arr: &StructArray, index: usize) {
        macro_rules! pack {
            ($arr:ident) => {
                for (i, (_, descr)) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    if $arr.is_null(i) {
                        continue;
                    };
                    let p = mzdata::Param::builder()
                        .name("filter string")
                        .curie(mzdata::curie!(MS:1000512))
                        .value($arr.value(i))
                        .build();
                    descr.add_param(p);
                }
            };
        }

        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }
        if let Some(arr) = arr.as_string_opt::<i32>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_string_opt::<i64>() {
            pack!(arr);
        } else {
            unimplemented!("{:?}", arr.data_type());
        }
    }

    fn visit_scan_start_time(&mut self, spec_arr: &StructArray, index: usize, unit: Option<Unit>) {
        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }
        let unit = unit.unwrap_or(Unit::Minute);
        let scalar = match unit {
            Unit::Minute => 1.0,
            Unit::Second => 1.0 / 60.0,
            Unit::Millisecond => 1.0 / (60.0 * 1000.0),
            _ => {
                log::error!("A unit {unit} other than a time unit provided, defaulting to minutes");
                1.0
            }
        };
        macro_rules! pack {
            ($arr:ident) => {
                for (i, (_, descr)) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    if $arr.is_null(i) {
                        continue;
                    };
                    descr.start_time = $arr.value(i) as f64 * scalar;
                }
            };
        }
        if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            pack!(arr);
        } else {
            todo!("{:?}", arr.data_type());
        }
    }

    fn visit_injection_time(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }
        macro_rules! pack {
            ($arr:ident) => {
                for (i, (_, descr)) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    if $arr.is_null(i) {
                        continue;
                    };
                    descr.injection_time = $arr.value(i) as f32;
                }
            };
        }
        if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            pack!(arr);
        } else {
            todo!("{:?}", arr.data_type());
        }
    }

    fn visit_parameters(&mut self, spec_arr: &StructArray) {
        let params_array: &LargeListArray =
            spec_arr.column_by_name("parameters").unwrap().as_list();
        let param_fields: Vec<FieldRef> =
            SchemaLike::from_type::<crate::param::Param>(Default::default()).unwrap();
        for (i, (_, descr)) in self
            .offsets
            .iter()
            .copied()
            .zip(self.descriptions.iter_mut())
        {
            let params = params_array.value(i);
            let params = params.as_struct();
            let params: Vec<crate::param::Param> =
                serde_arrow::from_arrow(&param_fields, params.columns()).unwrap();

            for p in params {
                descr.add_param(p.into());
            }
        }
    }

    fn visit_as_param(
        &mut self,
        spec_arr: &StructArray,
        index: usize,
        metacol: Option<&MetadataColumn>,
        name: Option<&str>,
    ) {
        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }

        let (name, unit, accession) = if let Some(metacol) = metacol {
            let unit = match &metacol.unit {
                crate::param::PathOrCURIE::Path(_items) => todo!(),
                crate::param::PathOrCURIE::CURIE(curie) => Unit::from_curie(&(*curie).into()),
                crate::param::PathOrCURIE::None => Unit::Unknown,
            };
            let accession: Option<mzdata::params::CURIE> = metacol.accession.map(|v| v.into());
            (metacol.name.as_str(), unit, accession)
        } else if let Some(name) = name {
            (name, Unit::Unknown, None)
        } else {
            panic!("One of `metacol` or `name` must be defined")
        };

        macro_rules! convert {
            ($arr:ident) => {
                for (i, (_, descr)) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    if $arr.is_null(i) {
                        continue;
                    };
                    let mut p = mzdata::Param::builder();
                    p = p.name(&name).value($arr.value(i)).unit(unit);
                    if let Some(acc) = accession {
                        p = p.curie(acc)
                    }
                    descr.add_param(p.build());
                }
            };
        }

        match arr.data_type() {
            DataType::Int32 => {
                let arr: &Int32Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Int64 => {
                let arr: &Int64Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Float32 => {
                let arr: &Float32Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Float64 => {
                let arr: &Float64Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Boolean => {
                let arr: &BooleanArray = spec_arr.column(index).as_boolean();
                convert!(arr);
            }
            DataType::Utf8 => {
                let arr: &LargeStringArray = spec_arr.column(index).as_string();
                convert!(arr);
            }
            DataType::UInt32 => {
                let arr: &UInt32Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::UInt64 => {
                let arr: &UInt64Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            _ => {
                todo!("Unsupported data type {:?}", arr.data_type())
            }
        }
    }

    fn visit_scan_windows(&mut self, spec_arr: &StructArray, index: usize) {
        let fields: Vec<FieldRef> =
            SchemaLike::from_type::<ScanWindowEntry>(Default::default()).unwrap();

        let arr = spec_arr.column(index);

        macro_rules! pack {
            ($arr:ident) => {
                for (i, (_, descr)) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    let windows = $arr.value(i);
                    let windows = windows.as_struct();
                    let windows: Vec<ScanWindowEntry> = serde_arrow::from_arrow(&fields, windows.columns()).unwrap();
                    for w in windows {
                        descr.scan_windows.push((&w).into());
                    }
                }

            };
        }

        if let Some(arr) =  arr.as_list_opt::<i64>() {
            pack!(arr);
        }
        else if let Some(arr) =  arr.as_list_opt::<i32>() {
            pack!(arr);
        }
        else {
            unimplemented!("{:?}", arr.data_type())
        }

    }

    fn visit_ion_mobility(
        &mut self,
        spec_arr: &StructArray,
        ion_mobility_value_index: usize,
        ion_mobility_type_index: usize,
    ) {
        let arr = spec_arr.column(ion_mobility_value_index);
        if arr.null_count() == arr.len() {
            return;
        }
        let curie_fields: Vec<FieldRef> =
            SchemaLike::from_type::<Option<CURIE>>(Default::default()).unwrap();
        let ion_mobility_types = spec_arr.column(ion_mobility_type_index).as_struct();
        let ion_mobility_types: Vec<Option<CURIE>> =
            serde_arrow::from_arrow(&curie_fields, ion_mobility_types.columns()).unwrap();

        macro_rules! pack {
            ($arr:ident) => {
                        for (i, (_, descr)) in self
            .offsets
            .iter()
            .copied()
            .zip(self.descriptions.iter_mut())
        {
            if $arr.is_null(i) {
                continue;
            };
            let im_val = $arr.value(i);
            let im_tp = ion_mobility_types[i].unwrap();
            match im_tp {
                // ion mobility drift time
                CURIE {
                    cv_id: MS_CV_ID,
                    accession: 1002476,
                } => descr.add_param(
                    mzdata::Param::builder()
                        .name("ion mobility drift time")
                        .curie(mzdata::curie!(MS:1002476))
                        .value(im_val)
                        .unit(Unit::Millisecond)
                        .build(),
                ),
                // inverse reduced ion mobility drift time
                CURIE {
                    cv_id: MS_CV_ID,
                    accession: 1002815,
                } => {
                    descr.add_param(
                        mzdata::Param::builder()
                            .name("inverse reduced ion mobility drift time")
                            .curie(mzdata::curie!(MS:1002815))
                            .value(im_val)
                            .unit(Unit::VoltSecondPerSquareCentimeter)
                            .build()
                    )
                }
                // FAIMS compensation voltage
                CURIE {
                    cv_id: MS_CV_ID,
                    accession: 1001581,
                } => {
                    descr.add_param(
                        mzdata::Param::builder()
                            .name("FAIMS compensation voltage")
                            .curie(mzdata::curie!(MS:1001581))
                            .value(im_val)
                            .unit(Unit::Volt)
                            .build()
                    )
                }
                // SELEXION compensation voltage
                CURIE {
                    cv_id: MS_CV_ID,
                    accession: 1003371,
                } => {
                    descr.add_param(
                        mzdata::Param::builder()
                            .name("SELEXION compensation voltage")
                            .curie(mzdata::curie!(MS:1003371))
                            .value(im_val)
                            .unit(Unit::Volt)
                            .build()
                    )
                }
                _ => todo!("{im_tp} not supported yet"),
            }
        }
            };
        }

        if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            pack!(arr);
        } else {
            todo!("{:?} not supported for ion mobility", arr.data_type());
        }
    }

    pub(crate) fn visit(&mut self, spec_arr: &StructArray) {
        // Must visit the index first, to infer null spacing
        self.visit_index(spec_arr, 0);

        let names = spec_arr.column_names();

        let mut visited = vec![false; names.len()];
        visited[0] = true;

        for col in self.metadata_map.iter() {
            if let Some(accession) = col.accession {
                if crate::curie!(MS:1000512) == accession {
                    self.visit_filter_string(spec_arr, col.index);
                    visited[col.index] = true;
                } else if crate::curie!(MS:1000616) == accession {
                    self.visit_preset_scan_configuration(spec_arr, col.index);
                    visited[col.index] = true;
                } else if crate::curie!(MS:1000616) == accession {
                    self.visit_preset_scan_configuration(spec_arr, col.index);
                    visited[col.index] = true;
                } else if crate::curie!(MS:1000016) == accession {
                    let unit = match &col.unit {
                        crate::param::PathOrCURIE::Path(_items) => todo!(),
                        crate::param::PathOrCURIE::CURIE(curie) => {
                            Some(Unit::from_curie(&((*curie).into())))
                        }
                        crate::param::PathOrCURIE::None => None,
                    };
                    self.visit_scan_start_time(spec_arr, col.index, unit);
                    visited[col.index] = true;
                } else if crate::curie!(MS:1000927) == accession {
                    self.visit_injection_time(spec_arr, col.index);
                    visited[col.index] = true;
                }
            }
        }

        let mut ion_mobility_value_index = None;
        let mut ion_mobility_type_index = None;

        for (_, (index, colname)) in visited
            .into_iter()
            .zip(names.into_iter().enumerate())
            .filter(|(seen, _)| !seen)
        {
            match colname {
                "parameters" => {
                    self.visit_parameters(spec_arr);
                }
                "instrument_configuration_ref" => {
                    self.visit_instrument_configuration_ref(spec_arr, index);
                }
                "scan_windows" => {
                    self.visit_scan_windows(spec_arr, index);
                }
                "ion_mobility_value" => {
                    ion_mobility_value_index = Some(index);
                }
                "ion_mobility_type" => {
                    ion_mobility_type_index = Some(index);
                }
                _ => {
                    self.visit_as_param(spec_arr, index, None, Some(colname));
                }
            }
        }

        match (ion_mobility_value_index, ion_mobility_type_index) {
            (Some(ion_mobility_value_index), Some(ion_mobility_type_index)) => {
                self.visit_ion_mobility(
                    spec_arr,
                    ion_mobility_value_index,
                    ion_mobility_type_index,
                );
            }
            (_, _) => {}
        }
    }
}

pub(crate) struct MzSelectedIonBuilder<'a> {
    pub(crate) descriptions: &'a mut [(u64, u64, SelectedIon)],
    pub(crate) metadata_map: &'a [MetadataColumn],
    pub(crate) base_offset: usize,
    pub(crate) offsets: Vec<usize>,
}

#[allow(unused)]
impl<'a> MzSelectedIonBuilder<'a> {
    pub(crate) fn new(
        descriptions: &'a mut [(u64, u64, SelectedIon)],
        metadata_map: &'a [MetadataColumn],
        base_offset: usize,
        offsets: Vec<usize>,
    ) -> Self {
        Self {
            descriptions,
            metadata_map,
            base_offset,
            offsets,
        }
    }

    fn visit_spectrum_index(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index).as_primitive::<UInt64Type>();
        let mut offsets = Vec::with_capacity(self.descriptions.len());
        let mut j = 0;
        for (i, (spec_index, prec_index, _descr)) in self.descriptions.iter_mut().enumerate() {
            while arr.is_null(self.base_offset + i + j) {
                j += 1;
            }
            let val = arr.value(self.base_offset + i + j);
            offsets.push(self.base_offset + i + j);
            *spec_index = val;
        }
        self.offsets = offsets
    }

    fn visit_precursor_index(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index).as_primitive::<UInt64Type>();
        for (i, (spec_index, prec_index, descr)) in self
            .offsets
            .iter()
            .copied()
            .zip(self.descriptions.iter_mut())
        {
            if arr.is_null(i) {
                continue;
            };
            *prec_index = arr.value(i);
        }
    }

    fn visit_selected_ion_mz(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index);
        macro_rules! pack {
            ($arr:ident) => {
                for (i, (_, _, descr)) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    if $arr.is_null(i) {
                        continue;
                    };
                    descr.mz = $arr.value(i) as f64;
                }
            };
        }
        if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            pack!(arr);
        } else {
            todo!("{:?}", arr.data_type());
        }
    }

    fn visit_peak_intensity(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }
        macro_rules! pack {
            ($arr:ident) => {
                for (i, (_, _, descr)) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    if $arr.is_null(i) {
                        continue;
                    };
                    descr.intensity = $arr.value(i) as f32;
                }
            };
        }
        if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            pack!(arr);
        } else {
            todo!("{:?}", arr.data_type());
        }
    }

    fn visit_charge(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }
        macro_rules! pack {
            ($arr:ident) => {
                for (i, (_, _, descr)) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    if $arr.is_null(i) {
                        descr.charge = None;
                    } else {
                        descr.charge = Some($arr.value(i) as i32);
                    }
                }
            };
        }
        if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Int32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Int64Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Int32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<UInt32Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<UInt64Type>() {
            pack!(arr);
        } else {
            todo!("{:?}", arr.data_type());
        }
    }

    fn visit_ion_mobility(
        &mut self,
        spec_arr: &StructArray,
        ion_mobility_value_index: usize,
        ion_mobility_type_index: usize,
    ) {
        let arr = spec_arr.column(ion_mobility_value_index);
        if arr.null_count() == arr.len() {
            return;
        }
        let curie_fields: Vec<FieldRef> =
            SchemaLike::from_type::<Option<CURIE>>(Default::default()).unwrap();
        let ion_mobility_types = spec_arr.column(ion_mobility_type_index).as_struct();
        let ion_mobility_types: Vec<Option<CURIE>> =
            serde_arrow::from_arrow(&curie_fields, ion_mobility_types.columns()).unwrap();

        macro_rules! pack {
            ($arr:ident) => {
                        for (i, (_, _, descr)) in self
            .offsets
            .iter()
            .copied()
            .zip(self.descriptions.iter_mut())
        {
            if $arr.is_null(i) {
                continue;
            };
            let im_val = $arr.value(i);
            let im_tp = ion_mobility_types[i].unwrap();
            match im_tp {
                // ion mobility drift time
                CURIE {
                    cv_id: MS_CV_ID,
                    accession: 1002476,
                } => descr.add_param(
                    mzdata::Param::builder()
                        .name("ion mobility drift time")
                        .curie(mzdata::curie!(MS:1002476))
                        .value(im_val)
                        .unit(Unit::Millisecond)
                        .build(),
                ),
                // inverse reduced ion mobility drift time
                CURIE {
                    cv_id: MS_CV_ID,
                    accession: 1002815,
                } => {
                    descr.add_param(
                        mzdata::Param::builder()
                            .name("inverse reduced ion mobility drift time")
                            .curie(mzdata::curie!(MS:1002815))
                            .value(im_val)
                            .unit(Unit::VoltSecondPerSquareCentimeter)
                            .build()
                    )
                }
                // FAIMS compensation voltage
                CURIE {
                    cv_id: MS_CV_ID,
                    accession: 1001581,
                } => {
                    descr.add_param(
                        mzdata::Param::builder()
                            .name("FAIMS compensation voltage")
                            .curie(mzdata::curie!(MS:1001581))
                            .value(im_val)
                            .unit(Unit::Volt)
                            .build()
                    )
                }
                // SELEXION compensation voltage
                CURIE {
                    cv_id: MS_CV_ID,
                    accession: 1003371,
                } => {
                    descr.add_param(
                        mzdata::Param::builder()
                            .name("SELEXION compensation voltage")
                            .curie(mzdata::curie!(MS:1003371))
                            .value(im_val)
                            .unit(Unit::Volt)
                            .build()
                    )
                }
                _ => todo!("{im_tp} not supported yet"),
            }
        }
            };
        }

        if let Some(arr) = arr.as_primitive_opt::<Float64Type>() {
            pack!(arr);
        } else if let Some(arr) = arr.as_primitive_opt::<Float32Type>() {
            pack!(arr);
        } else {
            todo!("{:?} not supported for ion mobility", arr.data_type());
        }
    }

    fn visit_parameters(&mut self, spec_arr: &StructArray) {
        let params_array: &LargeListArray =
            spec_arr.column_by_name("parameters").unwrap().as_list();
        let param_fields: Vec<FieldRef> =
            SchemaLike::from_type::<crate::param::Param>(Default::default()).unwrap();
        for (i, (_, _, descr)) in self
            .offsets
            .iter()
            .copied()
            .zip(self.descriptions.iter_mut())
        {
            let params = params_array.value(i);
            let params = params.as_struct();
            let params: Vec<crate::param::Param> =
                serde_arrow::from_arrow(&param_fields, params.columns()).unwrap();

            for p in params {
                descr.add_param(p.into());
            }
        }
    }

    fn visit_as_param(
        &mut self,
        spec_arr: &StructArray,
        index: usize,
        metacol: Option<&MetadataColumn>,
        name: Option<&str>,
    ) {
        let arr = spec_arr.column(index);
        if arr.null_count() == arr.len() {
            return;
        }

        let (name, unit, accession) = if let Some(metacol) = metacol {
            let unit = match &metacol.unit {
                crate::param::PathOrCURIE::Path(_items) => todo!(),
                crate::param::PathOrCURIE::CURIE(curie) => Unit::from_curie(&(*curie).into()),
                crate::param::PathOrCURIE::None => Unit::Unknown,
            };
            let accession: Option<mzdata::params::CURIE> = metacol.accession.map(|v| v.into());
            (metacol.name.as_str(), unit, accession)
        } else if let Some(name) = name {
            (name, Unit::Unknown, None)
        } else {
            panic!("One of `metacol` or `name` must be defined")
        };

        macro_rules! convert {
            ($arr:ident) => {
                for (i, (_, _, descr)) in self
                    .offsets
                    .iter()
                    .copied()
                    .zip(self.descriptions.iter_mut())
                {
                    if $arr.is_null(i) {
                        continue;
                    };
                    let mut p = mzdata::Param::builder();
                    p = p.name(&name).value($arr.value(i)).unit(unit);
                    if let Some(acc) = accession {
                        p = p.curie(acc)
                    }
                    descr.add_param(p.build());
                }
            };
        }

        match arr.data_type() {
            DataType::Int32 => {
                let arr: &Int32Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Int64 => {
                let arr: &Int64Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Float32 => {
                let arr: &Float32Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Float64 => {
                let arr: &Float64Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::Boolean => {
                let arr: &BooleanArray = spec_arr.column(index).as_boolean();
                convert!(arr);
            }
            DataType::Utf8 => {
                let arr: &LargeStringArray = spec_arr.column(index).as_string();
                convert!(arr);
            }
            DataType::UInt32 => {
                let arr: &UInt32Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            DataType::UInt64 => {
                let arr: &UInt64Array = spec_arr.column(index).as_primitive();
                convert!(arr);
            }
            _ => {
                todo!("Unsupported data type {:?}", arr.data_type())
            }
        }
    }

    pub(crate) fn visit(&mut self, spec_arr: &StructArray) {
        // Must visit the index first, to infer null spacing
        self.visit_spectrum_index(spec_arr, 0);
        self.visit_precursor_index(spec_arr, 1);

        let names = spec_arr.column_names();

        let mut visited = vec![false; names.len()];
        visited[0] = true;
        visited[1] = true;

        for col in self.metadata_map.iter() {
            if let Some(accession) = col.accession {
                if accession == crate::curie!(MS:1000744) {
                    self.visit_selected_ion_mz(spec_arr, col.index);
                    visited[col.index];
                } else if accession == crate::curie!(MS:1000041) {
                    self.visit_charge(spec_arr, col.index);
                    visited[col.index];
                } else if accession == crate::curie!(MS:1000042) {
                    self.visit_peak_intensity(spec_arr, col.index);
                    visited[col.index];
                }
            }
        }

        let mut ion_mobility_value_index = None;
        let mut ion_mobility_type_index = None;

        for (_, (index, colname)) in visited
            .into_iter()
            .zip(names.into_iter().enumerate())
            .filter(|(seen, _)| !seen)
        {
            match colname {
                "parameters" => {
                    self.visit_parameters(spec_arr);
                }
                "ion_mobility_value" => {
                    ion_mobility_value_index = Some(index);
                }
                "ion_mobility_type" => {
                    ion_mobility_type_index = Some(index);
                }
                _ => {
                    self.visit_as_param(spec_arr, index, None, Some(colname));
                }
            }
        }

        match (ion_mobility_value_index, ion_mobility_type_index) {
            (Some(ion_mobility_value_index), Some(ion_mobility_type_index)) => {
                self.visit_ion_mobility(
                    spec_arr,
                    ion_mobility_value_index,
                    ion_mobility_type_index,
                );
            }
            (_, _) => {}
        }
    }
}
