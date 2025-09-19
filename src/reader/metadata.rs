use std::{io, sync::Arc};

use crate::{
    archive::ZipArchiveReader,
    buffer_descriptors::{ArrayIndex, SerializedArrayIndex},
    filter::RegressionDeltaModel,
    param::MetadataColumn,
    reader::index::{QueryIndex, SpectrumPointIndex},
};
use arrow::array::{Array, AsArray, UInt64Array};
use mzdata::{io::OffsetIndex, meta, prelude::*};
use parquet::{
    arrow::{ProjectionMask, arrow_reader::ParquetRecordBatchReaderBuilder},
    file::reader::ChunkReader,
    schema::types::SchemaDescriptor,
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
            }
            "selected_ion_column_metadata_mapping" => {
                if let Some(val) = kv.value.as_ref() {
                    let metacols: Vec<MetadataColumn> = serde_json::from_str(&val)?;
                    selected_ion_metadata_mapping = Some(metacols);
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
