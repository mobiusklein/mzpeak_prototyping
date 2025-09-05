use std::{io, sync::Arc};

use arrow::{array::{Array, AsArray, BooleanArray, Float32Array, Float64Array, Int32Array, Int64Array, Int8Array, LargeListArray, LargeStringArray, StructArray, UInt32Array, UInt64Array, UInt8Array}, datatypes::{DataType, FieldRef, UInt64Type, UInt8Type}};
use mzdata::{io::OffsetIndex, meta::{self, SpectrumType}, params::Unit, prelude::*, spectrum::{ScanPolarity, SpectrumDescription}};
use parquet::{arrow::ProjectionMask, schema::types::SchemaDescriptor};
use serde_arrow::schema::SchemaLike;

use crate::{
    archive::ZipArchiveReader,
    buffer_descriptors::{ArrayIndex, SerializedArrayIndex},
    filter::RegressionDeltaModel,
    param::MetadataColumn,
    reader::index::QueryIndex, CURIE,
};

pub struct MzPeakReaderMetadata {
    pub(crate) mz_metadata: mzdata::meta::FileMetadataConfig,
    pub spectrum_array_indices: Arc<ArrayIndex>,
    pub chromatogram_array_indices: Arc<ArrayIndex>,
    pub spectrum_id_index: OffsetIndex,
    pub(crate) model_deltas: Vec<Option<Vec<f64>>>,
    pub(crate) auxliary_array_counts: Vec<u32>,
    pub(crate) spectrum_metadata_map: Option<Vec<MetadataColumn>>,
}

impl MzPeakReaderMetadata {
    pub fn new(
        mz_metadata: mzdata::meta::FileMetadataConfig,
        spectrum_array_indices: Arc<ArrayIndex>,
        chromatogram_array_indices: Arc<ArrayIndex>,
        spectrum_id_index: OffsetIndex,
        model_deltas: Vec<Option<Vec<f64>>>,
        auxliary_array_counts: Vec<u32>,
        spectrum_metadata_map: Option<Vec<MetadataColumn>>
    ) -> Self {
        Self {
            mz_metadata,
            spectrum_array_indices,
            chromatogram_array_indices,
            spectrum_id_index,
            model_deltas,
            auxliary_array_counts,
            spectrum_metadata_map,
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

impl MSDataFileMetadata for MzPeakReaderMetadata {
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

/// Load the various metadata, indices and reference data
pub(crate) fn load_indices_from(
    handle: &mut ZipArchiveReader,
) -> io::Result<(MzPeakReaderMetadata, QueryIndex)> {
    let spectrum_metadata_reader = handle.spectrum_metadata()?;
    let spectrum_data_reader = handle.spectra_data()?;

    let mut mz_metadata: meta::FileMetadataConfig = Default::default();
    let mut spectrum_array_indices: ArrayIndex = Default::default();
    let mut chromatogram_array_indices: ArrayIndex = Default::default();
    let mut spectrum_metadata_mapping = None;

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
            _ => {}
        }
    }

    let pq_schema = spectrum_metadata_reader.parquet_schema();

    let spectrum_id_index = build_spectrum_index(&handle, pq_schema)?;

    let mut query_index = QueryIndex::default();
    query_index.populate_spectrum_metadata_indices(&spectrum_metadata_reader);
    query_index.populate_spectrum_data_indices(&spectrum_data_reader, &spectrum_array_indices);

    let bundle = MzPeakReaderMetadata::new(
        mz_metadata,
        Arc::new(spectrum_array_indices),
        Arc::new(chromatogram_array_indices),
        spectrum_id_index,
        Vec::new(),
        Vec::new(),
        spectrum_metadata_mapping,
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
            offsets: Vec::new()
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
        for (i, descr) in self.offsets.iter().copied().zip(self.descriptions.iter_mut()) {
            let val = arr.value(i);
            descr.id = val.to_string();
        }
    }

    fn visit_ms_level(&mut self, spec_arr: &StructArray, index: usize) {
        let arr = spec_arr.column(index).as_primitive::<UInt8Type>();
        for (i, descr) in self.offsets.iter().copied().zip(self.descriptions.iter_mut()) {
            if arr.is_null(i) { continue };
            let ms_level_val = arr.value(i);
            descr.ms_level = ms_level_val;
        }
    }

    fn visit_polarity(&mut self, spec_arr: &StructArray, index: usize) {
        let polarity_arr: &Int8Array = spec_arr.column(index).as_any().downcast_ref().unwrap();
        for (i, descr) in self.offsets.iter().copied().zip(self.descriptions.iter_mut()) {
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

    fn visit_mz_signal_continuity(&mut self, spec_arr: &StructArray, index: usize) {
        let continuity_array = spec_arr.column(index).as_struct();
        let cv_id_arr: &UInt8Array = continuity_array.column(0).as_any().downcast_ref().unwrap();
        let accession_arr: &UInt32Array =
            continuity_array.column(1).as_any().downcast_ref().unwrap();
        for (i, descr) in self.offsets.iter().copied().zip(self.descriptions.iter_mut()) {
            if accession_arr.is_null(i) { continue };
            let continuity_curie = CURIE::new(cv_id_arr.value(i), accession_arr.value(i));
            descr.signal_continuity = match continuity_curie {
                crate::curie!(MS:1000525) => mzdata::spectrum::SignalContinuity::Unknown,
                crate::curie!(MS:1000127) => mzdata::spectrum::SignalContinuity::Centroid,
                crate::curie!(MS:1000128) => mzdata::spectrum::SignalContinuity::Profile,
                _ => todo!("Don't know how to deal with {continuity_curie}"),
            };
        }
    }

    fn visit_spectrum_type(&mut self, spec_arr: &StructArray, index: usize) {
        let spec_type_array = spec_arr.column(index).as_struct();
        let cv_id_arr: &UInt8Array = spec_type_array.column(0).as_any().downcast_ref().unwrap();
        let accession_arr: &UInt32Array =
            spec_type_array.column(1).as_any().downcast_ref().unwrap();

        for (i, descr) in self.offsets.iter().copied().zip(self.descriptions.iter_mut()) {
            let spec_type_curie = CURIE::new(cv_id_arr.value(i), accession_arr.value(i));
            let spec_type = SpectrumType::from_accession(spec_type_curie.accession);
            if let Some(spec_type) = spec_type {
                descr.set_spectrum_type(spec_type);
            }
        }
    }

    fn visit_lowest_mz(&mut self, spec_arr: &StructArray, index: usize) {
        let arr: &Float64Array = spec_arr.column(index).as_primitive();
        for (i, descr) in self.offsets.iter().copied().zip(self.descriptions.iter_mut()) {
            if arr.is_null(i) { continue };
            let p = mzdata::Param::builder()
                    .curie(mzdata::curie!(MS:1000528))
                    .name("lowest observed m/z")
                    .unit(Unit::MZ)
                    .value(arr.value(i))
                    .build();
            descr.add_param(p);
        }

    }

    fn visit_highest_mz(&mut self, spec_arr: &StructArray, index: usize) {
        let arr: &Float64Array = spec_arr.column(index).as_primitive();
        for (i, descr) in self.offsets.iter().copied().zip(self.descriptions.iter_mut()) {
            if arr.is_null(i) { continue };
            let p = mzdata::Param::builder()
                    .curie(mzdata::curie!(MS:1000527))
                    .name("highest observed m/z")
                    .unit(Unit::MZ)
                    .value(arr.value(i))
                    .build();
            descr.add_param(p);
        }

    }

    fn visit_base_peak_mz(&mut self, spec_arr: &StructArray, index: usize) {
        let arr: &Float64Array = spec_arr.column(index).as_primitive();
        for (i, descr) in self.offsets.iter().copied().zip(self.descriptions.iter_mut()) {
            if arr.is_null(i) { continue };
            let p = mzdata::Param::builder()
                    .curie(mzdata::curie!(MS:1000504))
                    .name("base peak m/z")
                    .unit(Unit::MZ)
                    .value(arr.value(i))
                    .build();
                descr.add_param(p);
        }

    }

    fn visit_base_peak_intensity(&mut self, spec_arr: &StructArray, index: usize) {
        let arr: &Float32Array = spec_arr.column(index).as_primitive();
        for (i, descr) in self.offsets.iter().copied().zip(self.descriptions.iter_mut()) {
            if arr.is_null(i) { continue };
            let p = mzdata::Param::builder()
                .curie(mzdata::curie!(MS:1000505))
                .name("base peak intensity")
                .unit(Unit::DetectorCounts)
                .value(arr.value(i))
                .build();
            descr.add_param(p);
        }
    }

    fn visit_total_ion_current(&mut self, spec_arr: &StructArray, index: usize) {
        let arr: &Float32Array = spec_arr.column(index).as_primitive();
        for (i, descr) in self.offsets.iter().copied().zip(self.descriptions.iter_mut()) {
            if arr.is_null(i) { continue };
            let p = mzdata::Param::builder()
                .curie(mzdata::curie!(MS:1000285))
                .name("total ion current")
                .unit(Unit::DetectorCounts)
                .value(arr.value(i))
                .build();
            descr.add_param(p);
        }
    }

    fn visit_parameters(&mut self, spec_arr: &StructArray) {
        let params_array: &LargeListArray =
            spec_arr.column_by_name("parameters").unwrap().as_list();
        let param_fields: Vec<FieldRef> =
            SchemaLike::from_type::<crate::param::Param>(Default::default()).unwrap();
        for (i, descr) in self.offsets.iter().copied().zip(self.descriptions.iter_mut()) {
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
            return
        }
        let accession: Option<mzdata::params::CURIE> = metacol.accession.map(|v| v.into());
        macro_rules! convert {
            ($arr:ident) => {
                for (i, descr) in self.offsets.iter().copied().zip(self.descriptions.iter_mut()) {
                    if $arr.is_null(i) { continue };
                    let mut p = mzdata::Param::builder();
                    p = p.name(&metacol.name)
                         .value($arr.value(i));
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
                    self.visit_ms_level(spec_arr, col.index);
                }
                else if crate::curie!(MS:1000465) == accession {
                    self.visit_polarity(spec_arr, col.index);
                }
                else if crate::curie!(MS:1000525) == accession {
                    self.visit_mz_signal_continuity(spec_arr, col.index);
                }
                else if crate::curie!(MS:1000559) == accession {
                    self.visit_spectrum_type(spec_arr, col.index);
                }
                else if crate::curie!(MS:1000528) == accession {
                    self.visit_lowest_mz(spec_arr, col.index);
                }
                else if crate::curie!(MS:1000527) == accession {
                    self.visit_highest_mz(spec_arr, col.index);
                }
                else if crate::curie!(MS:1000504) == accession {
                    self.visit_base_peak_mz(spec_arr, col.index);
                }
                else if crate::curie!(MS:1000505) == accession {
                    self.visit_base_peak_intensity(spec_arr, col.index);
                }
                else if crate::curie!(MS:1000285) == accession {
                    self.visit_total_ion_current(spec_arr, col.index);
                }
                else {
                    self.visit_as_param(spec_arr, col.index, col);
                }
            }
        }
        self.visit_parameters(spec_arr);
    }
}
