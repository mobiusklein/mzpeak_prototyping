use std::{collections::HashMap, io, sync::Arc};

use crate::{
    archive::{ArchiveReader, ArchiveSource},
    buffer_descriptors::{ArrayIndex, SerializedArrayIndex},
    filter::RegressionDeltaModel,
    param::MetadataColumn,
    reader::{
        index::{QueryIndex, SpectrumPointIndex},
        visitor::{
            MzChromatogramBuilder, MzPrecursorVisitor, MzScanVisitor, MzSelectedIonVisitor,
            MzSpectrumVisitor, metadata_columns_to_definition_map, schema_to_metadata_cols,
        },
    },
};
use arrow::{
    array::{Array, AsArray, RecordBatch, StructArray, UInt64Array},
    datatypes::{DataType, Float32Type},
};
use identity_hash::BuildIdentityHasher;
use mzdata::{
    io::OffsetIndex,
    meta,
    prelude::*,
    spectrum::{ChromatogramDescription, Precursor, ScanEvent, SelectedIon, SpectrumDescription},
};
use mzpeaks::coordinate::SimpleInterval;
use parquet::{
    arrow::{
        ProjectionMask,
        arrow_reader::{
            ArrowPredicateFn, ArrowReaderBuilder, ParquetRecordBatchReaderBuilder, RowSelection,
        },
    },
    file::{metadata::ParquetMetaData, reader::ChunkReader},
    schema::types::{SchemaDescPtr, SchemaDescriptor},
};

#[derive(Debug, Clone)]
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

    pub fn from_metadata<T>(reader: &ArrowReaderBuilder<T>) -> Option<Self> {
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

#[derive(Debug, Default)]
pub(crate) struct ParquetIndexExtractor {
    pub mz_metadata: meta::FileMetadataConfig,

    pub spectrum_array_indices: ArrayIndex,
    pub chromatogram_array_indices: ArrayIndex,

    pub spectrum_metadata_mapping: Option<Vec<MetadataColumn>>,
    pub scan_metadata_mapping: Option<Vec<MetadataColumn>>,
    pub selected_ion_metadata_mapping: Option<Vec<MetadataColumn>>,
    pub chromatogram_metadata_mapping: Option<Vec<MetadataColumn>>,

    pub query_index: QueryIndex,

    pub peak_metadata: Option<PeakMetadata>,
}

impl ParquetIndexExtractor {
    pub(crate) fn visit_spectrum_metadata_reader<T>(
        &mut self,
        spectrum_metadata_reader: ArrowReaderBuilder<T>,
    ) -> io::Result<()> {
        let arrow_schema = spectrum_metadata_reader.schema();
        if let Ok(root) = arrow_schema.field_with_name("spectrum") {
            if let DataType::Struct(fields) = root.data_type() {
                let defaults = crate::spectrum::SpectrumEntry::metadata_columns();
                let defined_columns = metadata_columns_to_definition_map(defaults);
                self.spectrum_metadata_mapping = Some(schema_to_metadata_cols(
                    fields,
                    "spectrum".into(),
                    Some(&defined_columns),
                ));
            }
        }
        if let Ok(root) = arrow_schema.field_with_name("scan") {
            if let DataType::Struct(fields) = root.data_type() {
                let defaults = crate::spectrum::ScanEntry::metadata_columns();
                let defined_columns = metadata_columns_to_definition_map(defaults);
                self.scan_metadata_mapping = Some(schema_to_metadata_cols(
                    fields,
                    "scan".into(),
                    Some(&defined_columns),
                ));
            }
        }
        if let Ok(root) = arrow_schema.field_with_name("selected_ion") {
            if let DataType::Struct(fields) = root.data_type() {
                let defaults = crate::spectrum::SelectedIonEntry::metadata_columns();
                let defined_columns = metadata_columns_to_definition_map(defaults);
                self.selected_ion_metadata_mapping = Some(schema_to_metadata_cols(
                    fields,
                    "selected_ion".into(),
                    Some(&defined_columns),
                ));
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
                        *self.mz_metadata.file_description_mut() = file_description.into();
                    } else {
                        log::warn!("file description was empty");
                    }
                }
                "instrument_configuration_list" => {
                    if let Some(val) = kv.value.as_ref() {
                        let instrument_configurations: Vec<crate::param::InstrumentConfiguration> =
                            serde_json::from_str(&val)?;
                        for ic in instrument_configurations {
                            self.mz_metadata
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
                            self.mz_metadata.data_processings_mut().push(dp.into());
                        }
                    } else {
                        log::warn!("data processing method list was empty");
                    }
                }
                "sample_list" => {
                    if let Some(val) = kv.value.as_ref() {
                        let meta_list: Vec<crate::param::Sample> = serde_json::from_str(&val)?;
                        for sw in meta_list {
                            self.mz_metadata.samples_mut().push(sw.into());
                        }
                    } else {
                        log::warn!("sample list was empty");
                    }
                }
                "software_list" => {
                    if let Some(val) = kv.value.as_ref() {
                        let software_list: Vec<crate::param::Software> =
                            serde_json::from_str(&val)?;
                        for sw in software_list {
                            self.mz_metadata.softwares_mut().push(sw.into());
                        }
                    } else {
                        log::warn!("software list was empty");
                    }
                }
                "run" => {
                    if let Some(val) = kv.value.as_ref() {
                        let run: meta::MassSpectrometryRun = serde_json::from_str(&val)?;
                        *self.mz_metadata.run_description_mut().unwrap() = run;
                    } else {
                        log::warn!("run was empty")
                    }
                }
                _ => {}
            }
        }

        self.query_index
            .populate_spectrum_metadata_indices(&spectrum_metadata_reader);

        Ok(())
    }

    pub(crate) fn visit_spectrum_data_reader<T>(
        &mut self,
        spectrum_data_reader: ArrowReaderBuilder<T>,
    ) -> io::Result<()> {
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
                        self.spectrum_array_indices = array_index.into();
                    } else {
                        log::warn!("spectrum array index was empty");
                    }
                }
                _ => {}
            }
        }
        self.query_index
            .populate_spectrum_data_indices(&spectrum_data_reader, &self.spectrum_array_indices);
        Ok(())
    }

    pub(crate) fn visit_chromatogram_metadata_reader<T>(
        &mut self,
        chromatogram_metadata_reader: ArrowReaderBuilder<T>,
    ) -> io::Result<()> {
        let arrow_schema = chromatogram_metadata_reader.schema();
        if let Ok(root) = arrow_schema.field_with_name("chromatogram") {
            if let DataType::Struct(fields) = root.data_type() {
                let defaults = crate::spectrum::ChromatogramEntry::metadata_columns();
                let defined_columns = metadata_columns_to_definition_map(defaults);
                self.chromatogram_metadata_mapping = Some(schema_to_metadata_cols(
                    fields,
                    "chromatogram".into(),
                    Some(&defined_columns),
                ));
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
        self.query_index
            .populate_chromatogram_metadata_indices(&chromatogram_metadata_reader);

        Ok(())
    }

    pub(crate) fn visit_chromatogram_data_reader<T>(
        &mut self,
        chromatogram_data_reader: ArrowReaderBuilder<T>,
    ) -> io::Result<()> {
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
                        self.chromatogram_array_indices = array_index.into();
                    } else {
                        log::warn!("chromatogram array index was empty");
                    }
                }
                _ => {}
            }
        }
        self.query_index.populate_chromatogram_data_indices(
            &chromatogram_data_reader,
            &self.chromatogram_array_indices,
        );
        Ok(())
    }

    pub(crate) fn visit_spectrum_peaks<T>(
        &mut self,
        spectrum_peaks_data_reader: ArrowReaderBuilder<T>,
    ) -> io::Result<()> {
        self.peak_metadata = PeakMetadata::from_metadata(&spectrum_peaks_data_reader);
        Ok(())
    }
}

/// Load the various metadata, indices and reference data
pub(crate) fn load_indices_from<T: ArchiveSource>(
    handle: &mut ArchiveReader<T>,
) -> io::Result<(ReaderMetadata, QueryIndex)> {
    let spectrum_metadata_reader = handle.spectrum_metadata()?;
    let spectrum_data_reader = handle.spectra_data()?;

    let pq_schema = spectrum_metadata_reader.parquet_schema();
    let spectrum_id_index = build_spectrum_index(&handle, pq_schema)?;

    let mut this = ParquetIndexExtractor::default();
    this.visit_spectrum_metadata_reader(spectrum_metadata_reader)?;
    this.visit_spectrum_data_reader(spectrum_data_reader)?;

    if let Ok(chromatogram_metadata_reader) = handle.chromatograms_metadata() {
        this.visit_chromatogram_metadata_reader(chromatogram_metadata_reader)?;
    }
    if let Ok(chromatogram_data_reader) = handle.chromatograms_data() {
        this.visit_chromatogram_data_reader(chromatogram_data_reader)?;
    }

    handle
        .spectrum_peaks()
        .ok()
        .and_then(|r| this.visit_spectrum_peaks(r).ok());

    let bundle = ReaderMetadata::new(
        this.mz_metadata,
        Arc::new(this.spectrum_array_indices),
        Arc::new(this.chromatogram_array_indices),
        spectrum_id_index,
        Vec::new(),
        Vec::new(),
        this.spectrum_metadata_mapping,
        this.scan_metadata_mapping,
        this.selected_ion_metadata_mapping,
        this.chromatogram_metadata_mapping,
        this.peak_metadata,
    );

    Ok((bundle, this.query_index))
}

pub(crate) trait BaseMetadataQuerySource {
    fn metadata(&self) -> &ParquetMetaData;

    fn parquet_schema(&self) -> SchemaDescPtr {
        self.metadata().file_metadata().schema_descr_ptr()
    }
}

pub(crate) trait SpectrumMetadataQuerySource: BaseMetadataQuerySource {
    fn prepare_predicate_for_all(
        &self,
    ) -> ArrowPredicateFn<
        impl FnMut(
            arrow::array::RecordBatch,
        ) -> Result<arrow::array::BooleanArray, arrow::error::ArrowError>
        + 'static,
    > {
        let predicate_mask = ProjectionMask::columns(
            &self.parquet_schema(),
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
        predicate
    }

    fn prepare_rows_for_all(&self, query_indices: &QueryIndex) -> RowSelection {
        let mut rows = query_indices
            .spectrum_index_index
            .row_selection_is_not_null();

        rows = rows.union(
            &query_indices
                .spectrum_scan_index
                .row_selection_is_not_null(),
        );
        rows = rows.union(
            &query_indices
                .spectrum_precursor_index
                .row_selection_is_not_null(),
        );
        rows = rows.union(
            &query_indices
                .spectrum_selected_ion_index
                .row_selection_is_not_null(),
        );

        rows
    }

    fn prepare_rows_for(&self, index: u64, query_indices: &QueryIndex) -> RowSelection {
        let mut rows = query_indices
            .spectrum_index_index
            .row_selection_contains(index);

        rows = rows.union(
            &query_indices
                .spectrum_scan_index
                .row_selection_contains(index),
        );
        rows = rows.union(
            &query_indices
                .spectrum_precursor_index
                .row_selection_contains(index),
        );
        rows = rows.union(
            &query_indices
                .spectrum_selected_ion_index
                .row_selection_contains(index),
        );
        rows
    }

    fn prepare_predicate_for(
        &self,
        index: u64,
    ) -> ArrowPredicateFn<
        impl FnMut(
            arrow::array::RecordBatch,
        ) -> Result<arrow::array::BooleanArray, arrow::error::ArrowError>
        + 'static,
    > {
        let predicate_mask = ProjectionMask::columns(
            &self.parquet_schema(),
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
        predicate
    }
}

const EMPTY_FIELDS: [MetadataColumn; 0] = [];

#[derive(Debug)]
pub struct SpectrumMetadataDecoder<'a> {
    pub descriptions: Vec<SpectrumDescription>,
    pub precursors: Vec<(u64, u64, Precursor)>,
    pub selected_ions: Vec<(u64, u64, SelectedIon)>,
    pub scan_events: Vec<(u64, ScanEvent)>,
    metadata: &'a ReaderMetadata,
}

fn segment_by_index_array(
    group: &StructArray,
    index_array: &UInt64Array,
    target: u64,
) -> Result<Vec<StructArray>, arrow::error::ArrowError> {
    let mask = arrow::compute::kernels::cmp::eq(index_array, &UInt64Array::new_scalar(target))?;
    let it = arrow::compute::SlicesIterator::new(&mask);

    Ok(it
        .map(|(start, end)| group.slice(start, end - start))
        .collect())
}

impl<'a> SpectrumMetadataDecoder<'a> {
    pub fn new(metadata: &'a ReaderMetadata) -> Self {
        Self {
            descriptions: Vec::new(),
            precursors: Vec::new(),
            selected_ions: Vec::new(),
            scan_events: Vec::new(),
            metadata,
        }
    }

    fn load_precursors_from(
        &self,
        precursor_arr: &StructArray,
        acc: &mut Vec<(u64, u64, Precursor)>,
    ) {
        let n = precursor_arr
            .column_by_name("spectrum_index")
            .map(|a| a.len() - a.null_count())
            .unwrap_or_default();
        if acc.is_empty() && n > 0 {
            acc.resize(n, Default::default());
        }
        if n > 0 {
            MzPrecursorVisitor::new(acc, &[], 0, Vec::new()).visit(&precursor_arr);
        }
    }

    fn load_selected_ions_from(
        &self,
        si_arr: &StructArray,
        acc: &mut Vec<(u64, u64, SelectedIon)>,
    ) {
        let metacols = self
            .metadata
            .selected_ion_metadata_map
            .as_deref()
            .unwrap_or(&EMPTY_FIELDS);
        let n = si_arr
            .column_by_name("spectrum_index")
            .map(|a| a.len() - a.null_count())
            .unwrap_or_default();
        if acc.is_empty() && n > 0 {
            acc.resize(n, Default::default());
        }
        if n > 0 {
            MzSelectedIonVisitor::new(acc, &metacols, 0, Vec::new()).visit(&si_arr);
        }
    }

    fn load_scan_events_from(
        &self,
        scan_arr: &StructArray,
        scan_accumulator: &mut Vec<(u64, ScanEvent)>,
    ) {
        let metacols = self
            .metadata
            .scan_metadata_map
            .as_deref()
            .unwrap_or(&EMPTY_FIELDS);
        let n = scan_arr
            .column_by_name("spectrum_index")
            .map(|a| a.len() - a.null_count())
            .unwrap_or_default();
        if scan_accumulator.is_empty() && n > 0 {
            scan_accumulator.resize(n, Default::default());
        }
        let mut builder = MzScanVisitor::new(scan_accumulator, &metacols, 0, Vec::new());
        builder.visit(scan_arr);
    }

    // This function is almost right, but something is missing during the decoding process
    #[allow(unused)]
    pub fn decode_batch_for(&mut self, batch: RecordBatch, spectrum_index: u64) {
        let spec_arr = batch.column_by_name("spectrum").unwrap().as_struct();
        let index_arr: &UInt64Array = spec_arr
            .column_by_name("index")
            .unwrap()
            .as_any()
            .downcast_ref()
            .unwrap();
        let spec_arrays = segment_by_index_array(spec_arr, index_arr, spectrum_index).unwrap();

        for spec_arr in spec_arrays {
            let n_spec = index_arr.len() - index_arr.null_count();
            if n_spec > 0 {
                let mut local_descr = vec![SpectrumDescription::default(); n_spec];
                let mut builder = MzSpectrumVisitor::new(
                    &mut local_descr,
                    &self
                        .metadata
                        .spectrum_metadata_map
                        .as_deref()
                        .unwrap_or(&EMPTY_FIELDS),
                    0,
                );
                builder.visit(&spec_arr);
                if self.descriptions.is_empty() {
                    self.descriptions = local_descr;
                } else {
                    self.descriptions.extend(local_descr);
                }
            }
        }

        if let Some(scan_arr) = batch.column_by_name("scan").map(|arr| arr.as_struct()) {
            let index_arr: &UInt64Array = scan_arr
                .column_by_name("spectrum_index")
                .unwrap()
                .as_primitive();
            for scan_arr in segment_by_index_array(scan_arr, index_arr, spectrum_index).unwrap() {
                let mut acc = Vec::new();
                self.load_scan_events_from(&scan_arr, &mut acc);
                if self.scan_events.is_empty() {
                    self.scan_events = acc;
                } else {
                    self.scan_events.extend(acc);
                }
            }
        }

        if let Some(precursor_arr) = batch.column_by_name("precursor").map(|v| v.as_struct()) {
            let index_arr: &UInt64Array = precursor_arr
                .column_by_name("spectrum_index")
                .unwrap()
                .as_primitive();
            for precursor_arr in
                segment_by_index_array(precursor_arr, index_arr, spectrum_index).unwrap()
            {
                let mut precursor_acc = Vec::new();
                self.load_precursors_from(&precursor_arr, &mut precursor_acc);
                if self.precursors.is_empty() {
                    self.precursors = precursor_acc
                } else {
                    self.precursors.extend(precursor_acc);
                }
            }
        }

        if let Some(selected_ion_arr) = batch.column_by_name("selected_ion").map(|v| v.as_struct())
        {
            let index_arr: &UInt64Array = selected_ion_arr
                .column_by_name("spectrum_index")
                .unwrap()
                .as_primitive();
            for selected_ion_arr in
                segment_by_index_array(selected_ion_arr, &index_arr, spectrum_index).unwrap()
            {
                let mut acc = Vec::new();
                self.load_selected_ions_from(&selected_ion_arr, &mut acc);
                if self.selected_ions.is_empty() {
                    self.selected_ions = acc;
                } else {
                    self.selected_ions.extend(acc);
                }
            }
        }
    }

    pub fn decode_batch(&mut self, batch: RecordBatch) {
        let spec_arr = batch.column_by_name("spectrum").unwrap().as_struct();
        let index_arr: &UInt64Array = spec_arr
            .column_by_name("index")
            .unwrap()
            .as_any()
            .downcast_ref()
            .unwrap();
        let n_spec = index_arr.len() - index_arr.null_count();
        if n_spec > 0 {
            let mut local_descr = vec![SpectrumDescription::default(); n_spec];
            let mut builder = MzSpectrumVisitor::new(
                &mut local_descr,
                &self
                    .metadata
                    .spectrum_metadata_map
                    .as_deref()
                    .unwrap_or(&EMPTY_FIELDS),
                0,
            );
            builder.visit(spec_arr);
            if self.descriptions.is_empty() {
                self.descriptions = local_descr;
            } else {
                self.descriptions.extend(local_descr);
            }
        }

        if let Some(scan_arr) = batch.column_by_name("scan").map(|arr| arr.as_struct()) {
            let mut acc = Vec::new();
            self.load_scan_events_from(scan_arr, &mut acc);
            if self.scan_events.is_empty() {
                self.scan_events = acc;
            } else {
                self.scan_events.extend(acc);
            }
        }

        let precursor_arr = batch.column_by_name("precursor").unwrap().as_struct();
        {
            let mut precursor_acc = Vec::new();
            self.load_precursors_from(precursor_arr, &mut precursor_acc);
            if self.precursors.is_empty() {
                self.precursors = precursor_acc
            } else {
                self.precursors.extend(precursor_acc);
            }
        }

        let selected_ion_arr = batch.column_by_name("selected_ion").unwrap().as_struct();
        {
            let mut acc = Vec::new();
            self.load_selected_ions_from(&selected_ion_arr, &mut acc);
            if self.selected_ions.is_empty() {
                self.selected_ions = acc;
            } else {
                self.selected_ions.extend(acc);
            }
        }
    }

    pub fn finish(mut self) -> Vec<SpectrumDescription> {
        self.precursors.sort_unstable_by(|a, b| a.1.cmp(&b.1));

        let spec_offset = match self.descriptions.first().map(|v| v.index) {
            Some(i) => i,
            None => return self.descriptions,
        };

        let prec_offset = match self.precursors.first().map(|(_a, b, _v)| *b) {
            Some(i) => i as usize,
            None => {
                self.selected_ions.clear();
                0
            }
        };

        for (_, prec_idx, si) in self.selected_ions {
            if let Some(prec) = self.precursors.get_mut(prec_idx as usize - prec_offset) {
                prec.2.add_ion(si);
            }
        }

        for (idx, scan) in self.scan_events {
            if let Some(spec) = self.descriptions.get_mut(idx as usize - spec_offset) {
                spec.acquisition.scans.push(scan);
            }
        }

        for (idx, _, precursor) in self.precursors {
            if let Some(spec) = self.descriptions.get_mut(idx as usize - spec_offset) {
                spec.precursor.push(precursor);
            }
        }
        self.descriptions
    }
}

pub(crate) struct SpectrumMetadataReader<T: ChunkReader + 'static>(
    pub(crate) ParquetRecordBatchReaderBuilder<T>,
);

impl<T: ChunkReader + 'static> BaseMetadataQuerySource for SpectrumMetadataReader<T> {
    fn metadata(&self) -> &ParquetMetaData {
        self.0.metadata()
    }
}

impl<T: ChunkReader + 'static> SpectrumMetadataQuerySource for SpectrumMetadataReader<T> {}

pub(crate) trait ChromatogramMetadataQuerySource: BaseMetadataQuerySource {
    fn prepare_predicate_for_all(
        &self,
    ) -> ArrowPredicateFn<
        impl FnMut(RecordBatch) -> Result<arrow::array::BooleanArray, arrow::error::ArrowError>
        + 'static,
    > {
        let predicate_mask = ProjectionMask::columns(
            &self.parquet_schema(),
            [
                "chromatogram.index",
                "precursor.spectrum_index",
                "selected_ion.spectrum_index",
            ],
        );

        let predicate = ArrowPredicateFn::new(predicate_mask, move |batch| {
            let chromatogram_index: &UInt64Array = batch
                .column(0)
                .as_struct()
                .column(0)
                .as_any()
                .downcast_ref()
                .unwrap();
            let precursor_spectrum_index: &UInt64Array = batch
                .column(1)
                .as_struct()
                .column(0)
                .as_any()
                .downcast_ref()
                .unwrap();
            let selected_ion_spectrum_index: &UInt64Array = batch
                .column(2)
                .as_struct()
                .column(0)
                .as_any()
                .downcast_ref()
                .unwrap();

            let it = chromatogram_index.iter().map(|val| val.is_some());

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
        predicate
    }
}

pub struct ChromatogramMetadataDecoder<'a> {
    pub descriptions: Vec<ChromatogramDescription>,
    pub precursors: Vec<(u64, u64, Precursor)>,
    pub selected_ions: Vec<(u64, u64, SelectedIon)>,
    metadata: &'a ReaderMetadata,
}

impl<'a> ChromatogramMetadataDecoder<'a> {
    pub fn new(metadata: &'a ReaderMetadata) -> Self {
        Self {
            descriptions: Vec::new(),
            precursors: Vec::new(),
            selected_ions: Vec::new(),
            metadata,
        }
    }

    fn load_precursors_from(
        &self,
        precursor_arr: &StructArray,
        acc: &mut Vec<(u64, u64, Precursor)>,
    ) {
        let n = precursor_arr
            .column_by_name("spectrum_index")
            .map(|a| a.len() - a.null_count())
            .unwrap_or_default();
        if acc.is_empty() && n > 0 {
            acc.resize(n, Default::default());
        }
        if n > 0 {
            MzPrecursorVisitor::new(acc, &[], 0, Vec::new()).visit(&precursor_arr);
        }
    }

    fn load_selected_ions_from(
        &self,
        si_arr: &StructArray,
        acc: &mut Vec<(u64, u64, SelectedIon)>,
    ) {
        let metacols = self
            .metadata
            .selected_ion_metadata_map
            .as_deref()
            .unwrap_or(&EMPTY_FIELDS);
        let n = si_arr
            .column_by_name("spectrum_index")
            .map(|a| a.len() - a.null_count())
            .unwrap_or_default();
        if acc.is_empty() && n > 0 {
            acc.resize(n, Default::default());
        }
        if n > 0 {
            MzSelectedIonVisitor::new(acc, &metacols, 0, Vec::new()).visit(&si_arr);
        }
    }

    pub fn decode_batch(&mut self, batch: RecordBatch) {
        let chrom_arr = batch.column_by_name("chromatogram").unwrap().as_struct();
        let index_arr: &UInt64Array = chrom_arr
            .column_by_name("index")
            .unwrap()
            .as_any()
            .downcast_ref()
            .unwrap();
        let n_spec = index_arr.len() - index_arr.null_count();
        let mut local_descr = vec![ChromatogramDescription::default(); n_spec];
        let mut builder = MzChromatogramBuilder::new(
            &mut local_descr,
            &self
                .metadata
                .chromatogram_metadata_map
                .as_ref()
                .map(|v| v.as_slice())
                .unwrap_or(&EMPTY_FIELDS),
            0,
        );
        builder.visit(chrom_arr);
        self.descriptions.extend(local_descr);

        let precursor_arr = batch.column_by_name("precursor").unwrap().as_struct();
        {
            let mut acc = Vec::new();
            self.load_precursors_from(precursor_arr, &mut acc);
            self.precursors.extend(acc);
        }

        let selected_ion_arr = batch.column_by_name("selected_ion").unwrap().as_struct();
        {
            let mut acc = Vec::new();
            self.load_selected_ions_from(selected_ion_arr, &mut acc);
            self.selected_ions.extend(acc);
        }
    }

    // This function is almost right, but something is missing during the decoding process
    #[allow(unused)]
    pub fn decode_batch_for(&mut self, batch: RecordBatch, target: u64) {
        let chrom_arr = batch.column_by_name("chromatogram").unwrap().as_struct();
        let index_arr: &UInt64Array = chrom_arr
            .column_by_name("index")
            .unwrap()
            .as_any()
            .downcast_ref()
            .unwrap();

        let n_spec = index_arr.len() - index_arr.null_count();
        let mut local_descr = vec![ChromatogramDescription::default(); n_spec];
        let mut builder = MzChromatogramBuilder::new(
            &mut local_descr,
            &self
                .metadata
                .chromatogram_metadata_map
                .as_ref()
                .map(|v| v.as_slice())
                .unwrap_or(&EMPTY_FIELDS),
            0,
        );
        builder.visit(chrom_arr);
        self.descriptions.extend(local_descr);

        let precursor_arr = batch.column_by_name("precursor").unwrap().as_struct();
        {
            let mut acc = Vec::new();
            self.load_precursors_from(precursor_arr, &mut acc);
            self.precursors.extend(acc);
        }

        let selected_ion_arr = batch.column_by_name("selected_ion").unwrap().as_struct();
        {
            let mut acc = Vec::new();
            self.load_selected_ions_from(selected_ion_arr, &mut acc);
            self.selected_ions.extend(acc);
        }
    }

    pub fn finish(mut self) -> Vec<ChromatogramDescription> {
        self.precursors.sort_unstable_by(|a, b| a.1.cmp(&b.1));

        for (_spec_idx, prec_idx, si) in self.selected_ions {
            let prec = &mut self.precursors[prec_idx as usize].2;
            prec.add_ion(si);
        }

        for (idx, _prec_idx, precursor) in self.precursors {
            self.descriptions[idx as usize].precursor.push(precursor);
        }
        self.descriptions
    }
}

pub(crate) struct ChromatogramMetadataReader<T: ChunkReader + 'static>(
    pub(crate) ParquetRecordBatchReaderBuilder<T>,
);

impl<T: ChunkReader + 'static> ChromatogramMetadataQuerySource for ChromatogramMetadataReader<T> {}

impl<T: ChunkReader + 'static> BaseMetadataQuerySource for ChromatogramMetadataReader<T> {
    fn metadata(&self) -> &ParquetMetaData {
        self.0.metadata()
    }
}

pub struct TimeIndexDecoder {
    times: HashMap<u64, f32, BuildIdentityHasher<u64>>,
    time_range: SimpleInterval<f32>,
    min: u64,
    max: u64,
}

impl TimeIndexDecoder {
    pub fn new(time_range: SimpleInterval<f32>) -> Self {
        Self {
            time_range,
            min: u64::MAX,
            max: 0,
            times: Default::default(),
        }
    }

    pub fn from_descriptions(&mut self, descriptions: &[SpectrumDescription]) {
        let n = descriptions.len();

        let offset_start = match descriptions.binary_search_by(|descr| {
            self.time_range
                .start()
                .total_cmp(&(descr.acquisition.start_time() as f32))
                .reverse()
        }) {
            Ok(i) => i,
            Err(i) => i.min(n),
        }
        .saturating_sub(1);
        let offset = offset_start;

        // TODO: Rewrite the output data structure and maybe this will be faster
        // let mut i = offset_start;
        // while i > 0 {
        //     if time_range.start >= cache[i].acquisition.start_time()  as f32 {
        //         i -= 1;
        //     }
        //     break;
        // }
        // while !time_range.contains(&(cache[i].acquisition.start_time() as f32)) {
        //     i += 1;
        // }
        // offset_start = i;

        // let mut offset_end = match cache.binary_search_by(|descr| {
        //     time_range.end().total_cmp(&(descr.acquisition.start_time() as f32)).reverse()
        // }) {
        //     Ok(i) => i,
        //     Err(i) => i.min(n),
        // };

        // i = offset_end;
        // while i < n {
        //     if time_range.end <= cache[i].acquisition.start_time()  as f32 {
        //         i += 1;
        //     }
        //     break;
        // }
        // while !time_range.contains(&(cache[i].acquisition.start_time() as f32)) {
        //     i = i.saturating_sub(1);
        // }
        // offset_end = i;

        for (i, descr) in descriptions.iter().enumerate().skip(offset) {
            let i = i as u64;
            let t = descr.acquisition.start_time() as f32;
            if self.time_range.contains(&t) {
                self.min = self.min.min(i);
                self.max = self.max.max(i);
                self.times.insert(i, t);
            } else if !self.times.is_empty() {
                break;
            }
        }
    }

    pub fn decode_batch(
        &mut self,
        batch: RecordBatch,
    ) -> Result<(), parquet::errors::ParquetError> {
        let root = batch.column(0).as_struct();
        let arr: &UInt64Array = root.column(0).as_primitive();
        let time_arr = root.column(1);
        if let Some(time_arr) = time_arr.as_primitive_opt::<Float32Type>() {
            for (val, time) in arr.iter().flatten().zip(time_arr.iter().flatten()) {
                if self.time_range.contains(&time) {
                    self.min = self.min.min(val);
                    self.max = self.max.max(val);
                    self.times.insert(val, time);
                }
            }
        } else if let Some(time_arr) = time_arr.as_primitive_opt::<Float32Type>() {
            for (val, time) in arr
                .iter()
                .flatten()
                .zip(time_arr.iter().flatten().map(|v| v as f32))
            {
                if self.time_range.contains(&time) {
                    self.min = self.min.min(val);
                    self.max = self.max.max(val);
                    self.times.insert(val, time);
                }
            }
        } else {
            return Err(parquet::errors::ParquetError::ArrowError(format!(
                "Invalid time array data type: {:?}",
                time_arr.data_type()
            ))
            .into());
        }
        Ok(())
    }

    pub fn finish(
        self,
    ) -> (
        HashMap<u64, f32, BuildIdentityHasher<u64>>,
        SimpleInterval<u64>,
    ) {
        (self.times, SimpleInterval::new(self.min, self.max))
    }
}
