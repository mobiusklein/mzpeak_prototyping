use std::{io, sync::Arc};

use crate::{
    archive::{ArchiveReader, ArchiveSource},
    buffer_descriptors::{ArrayIndex, SerializedArrayIndex},
    filter::RegressionDeltaModel,
    param::MetadataColumn,
    reader::{index::{QueryIndex, SpectrumPointIndex}, visitor::{metadata_columns_to_definition_map, schema_to_metadata_cols}},
};
use arrow::{array::{Array, AsArray, UInt64Array}, datatypes::DataType};
use mzdata::{io::OffsetIndex, meta, prelude::*};
use parquet::{
    arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder},
    file::reader::ChunkReader,
    schema::types::SchemaDescriptor,
};

#[derive(Debug)]
pub struct ReaderMetadata {
    pub(crate) mz_metadata: mzdata::meta::FileMetadataConfig,
    pub(crate) spectrum_array_indices: Arc<ArrayIndex>,
    pub(crate) chromatogram_array_indices: Arc<ArrayIndex>,
    pub(crate) spectrum_id_index: OffsetIndex,
    pub(crate) mz_model_deltas: Vec<Option<Vec<f64>>>,
    pub(crate) spectrum_auxiliary_array_counts: Vec<u32>,
    pub(crate) chromatogram_auxiliary_array_counts: Vec<u32>,
    pub(crate) spectrum_metadata_map: Option<Vec<MetadataColumn>>,
    pub(crate) scan_metadata_map: Option<Vec<MetadataColumn>>,
    pub(crate) selected_ion_metadata_map: Option<Vec<MetadataColumn>>,
    pub(crate) chromatogram_metadata_map: Option<Vec<MetadataColumn>>,
    pub(crate) peak_indices: Option<PeakMetadata>,
}

impl ReaderMetadata {
    pub fn new(
        mz_metadata: mzdata::meta::FileMetadataConfig,
        spectrum_array_indices: Arc<ArrayIndex>,
        chromatogram_array_indices: Arc<ArrayIndex>,
        spectrum_id_index: OffsetIndex,
        model_deltas: Vec<Option<Vec<f64>>>,
        spectrum_auxiliary_array_counts: Vec<u32>,
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
            mz_model_deltas: model_deltas,
            spectrum_auxiliary_array_counts,
            spectrum_metadata_map,
            scan_metadata_map,
            selected_ion_metadata_map,
            chromatogram_metadata_map,
            peak_indices,
            chromatogram_auxiliary_array_counts: Vec::new(),
        }
    }

    pub fn model_deltas_for(&self, index: usize) -> Option<RegressionDeltaModel<f64>> {
        self.mz_model_deltas
            .get(index)
            .cloned()
            .unwrap_or_default()
            .map(|v| RegressionDeltaModel::from(v))
    }

    pub fn spectrum_auxiliary_array_counts(&self) -> &[u32] {
        &self.spectrum_auxiliary_array_counts
    }

    pub fn chromatogram_auxiliary_array_counts(&self) -> &[u32] {
        &self.chromatogram_auxiliary_array_counts
    }

    pub fn peak_array_indices(&self) -> Option<&ArrayIndex> {
        self.peak_indices.as_ref().map(|v| &v.array_indices)
    }

    pub fn spectrum_array_indices(&self) -> &ArrayIndex {
        &self.spectrum_array_indices
    }

    pub fn chromatogram_array_indices(&self) -> &ArrayIndex {
        &self.chromatogram_array_indices
    }

    pub fn file_metadata(&self) -> &mzdata::meta::FileMetadataConfig {
        &self.mz_metadata
    }
}

impl MSDataFileMetadata for ReaderMetadata {
    mzdata::delegate_impl_metadata_trait!(mz_metadata);
}

pub(crate) fn build_spectrum_index<T: ArchiveSource>(
    handle: &ArchiveReader<T>,
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
pub(crate) fn load_indices_from<T: ArchiveSource>(
    handle: &mut ArchiveReader<T>,
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

    let arrow_schema = spectrum_metadata_reader.schema();
    if let Ok(root) = arrow_schema.field_with_name("spectrum") {
        if let DataType::Struct(fields) = root.data_type() {
            let defaults = crate::spectrum::SpectrumEntry::metadata_columns();
            let defined_columns = metadata_columns_to_definition_map(defaults);
            spectrum_metadata_mapping = Some(schema_to_metadata_cols(fields, "spectrum".into(), Some(&defined_columns)));
        }
    }
    if let Ok(root) = arrow_schema.field_with_name("scan") {
        if let DataType::Struct(fields) = root.data_type() {
            let defaults = crate::spectrum::ScanEntry::metadata_columns();
            let defined_columns = metadata_columns_to_definition_map(defaults);
            scan_metadata_mapping = Some(schema_to_metadata_cols(fields, "scan".into(), Some(&defined_columns)));
        }
    }
    if let Ok(root) = arrow_schema.field_with_name("selected_ion") {
        if let DataType::Struct(fields) = root.data_type() {
            let defaults = crate::spectrum::SelectedIonEntry::metadata_columns();
            let defined_columns = metadata_columns_to_definition_map(defaults);
            selected_ion_metadata_mapping = Some(schema_to_metadata_cols(fields, "selected_ion".into(), Some(&defined_columns)));
        }
    }

    for kv in spectrum_metadata_reader
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .into_iter()
        .flatten()
    {
        match kv.key.as_str() {
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
            "chromatogram_column_metadata_mapping" => {
                if let Some(val) = kv.value.as_ref() {
                    let metacols: Vec<MetadataColumn> = serde_json::from_str(&val)?;
                    chromatogram_metadata_mapping = Some(metacols);
                }
            }
            _ => {}
        }
    }

    for kv in spectrum_data_reader
        .metadata()
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
                    log::warn!("spectrum array index was empty");
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

    if let Ok(chromatogram_metadata_reader) = handle.chromatograms_metadata() {

        let arrow_schema = chromatogram_metadata_reader.schema();
        if let Ok(root) = arrow_schema.field_with_name("chromatogram") {
            if let DataType::Struct(fields) = root.data_type() {
                let defaults = crate::spectrum::ChromatogramEntry::metadata_columns();
                let defined_columns = metadata_columns_to_definition_map(defaults);
                chromatogram_metadata_mapping = Some(schema_to_metadata_cols(fields, "chromatogram".into(), Some(&defined_columns)));
            }
        }

        for kv in chromatogram_metadata_reader
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .into_iter()
            .flatten()
        {
            match kv.key.as_str() {
                _ => {}
            }
        }
        query_index.populate_chromatogram_metadata_indices(&chromatogram_metadata_reader);
    }
    if let Ok(chromatogram_data_reader) = handle.chromatograms_data() {
        for kv in chromatogram_data_reader
            .metadata()
            .file_metadata()
            .key_value_metadata()
            .into_iter()
            .flatten()
        {
            match kv.key.as_str() {
                "chromatogram_array_index" => {
                    if let Some(val) = kv.value.as_ref() {
                        let array_index: SerializedArrayIndex = serde_json::from_str(&val)?;
                        chromatogram_array_indices = array_index.into();
                    } else {
                        log::warn!("chromatogram array index was empty");
                    }
                }
                _ => {}
            }
        }
        query_index.populate_chromatogram_data_indices(
            &chromatogram_data_reader,
            &chromatogram_array_indices,
        );
    }

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
