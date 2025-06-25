use std::{
    collections::HashMap,
    fs::File,
    io,
    marker::PhantomData,
    path::{Path, PathBuf},
};

use arrow::{
    array::{
        Array, AsArray, Float32Array, Float64Array, Int32Array, Int64Array, Int8Array, LargeListArray, LargeStringArray, RecordBatch, StructArray, UInt32Array, UInt64Array, UInt8Array
    },
    datatypes::{DataType, FieldRef},
};

use mzdata::{
    io::{DetailLevel, OffsetIndex},
    meta::{self, MSDataFileMetadata},
    params::{ParamDescribed, Unit},
    prelude::*,
    spectrum::{
        ArrayType, BinaryArrayMap, Chromatogram, DataArray, MultiLayerSpectrum, ScanEvent,
        ScanPolarity, SpectrumDescription, bindata::BuildFromArrayMap,
    },
};
use mzpeaks::{
    CentroidPeak, DeconvolutedCentroidLike, DeconvolutedPeak, coordinate::SimpleInterval,
    prelude::Span1D,
};

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
        PageIndex, PageIndexEntry, PageIndexType, SpanDynNumeric, read_f32_page_index_from,
        read_f64_page_index_from, read_i32_page_index_from, read_i64_page_index_from,
        read_u8_page_index_from, read_u32_page_index_from, read_u64_page_index_from,
    },
    param::{DataProcessing, FileDescription, InstrumentConfiguration, Software},
    peak_series::{ArrayIndex, SerializedArrayIndex},
};

pub struct MzPeakReaderMetadata {
    mz_metadata: meta::FileMetadataConfig,
    pub spectrum_array_indices: ArrayIndex,
    pub chromatogram_array_indices: ArrayIndex,
    pub spectrum_id_index: OffsetIndex,
}

impl MSDataFileMetadata for MzPeakReaderMetadata {
    mzdata::delegate_impl_metadata_trait!(mz_metadata);
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

pub struct MzPeakReaderType<
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap = CentroidPeak,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap = DeconvolutedPeak,
> {
    path: PathBuf,
    handle: ZipArchiveReader,
    index: usize,
    detail_level: DetailLevel,
    pub metadata: MzPeakReaderMetadata,
    pub query_indices: QueryIndex,
    spectrum_metadata_cache: Option<Vec<SpectrumDescription>>,
    spectrum_row_group_cache: Option<(usize, RecordBatch)>,
    _t: PhantomData<(C, D)>,
}

impl<
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
> ChromatogramSource for MzPeakReaderType<C, D>
{
    fn get_chromatogram_by_id(&mut self, id: &str) -> Option<Chromatogram> {
        match id {
            "TIC" => self.tic().ok(),
            "BPC" => self.bpc().ok(),
            _ => None,
        }
    }

    fn get_chromatogram_by_index(&mut self, index: usize) -> Option<Chromatogram> {
        match index {
            0 => self.tic().ok(),
            1 => self.bpc().ok(),
            _ => None,
        }
    }
}

impl<
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
> ExactSizeIterator for MzPeakReaderType<C, D>
{
    fn len(&self) -> usize {
        self.len()
    }
}

impl<
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
> Iterator for MzPeakReaderType<C, D>
{
    type Item = MultiLayerSpectrum<C, D>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.spectrum_metadata_cache.is_none() {
            self.populate_metadata_cache().ok()?;
        }
        let x = self.get_spectrum(self.index).ok();
        self.index += 1;
        x
    }
}

impl<
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
> SpectrumSource<C, D> for MzPeakReaderType<C, D>
{
    fn reset(&mut self) {
        self.index = 0;
    }

    fn detail_level(&self) -> &mzdata::io::DetailLevel {
        &self.detail_level
    }

    fn set_detail_level(&mut self, detail_level: mzdata::io::DetailLevel) {
        self.detail_level = detail_level
    }

    fn get_spectrum_by_id(&mut self, id: &str) -> Option<MultiLayerSpectrum<C, D>> {
        let description = self.get_spectrum_metadata_by_id(id).ok()?;
        let arrays = if self.detail_level == DetailLevel::Full {
            self.get_spectrum_arrays(description.index as u64).ok()?
        } else {
            BinaryArrayMap::new()
        };
        Some(MultiLayerSpectrum::from_arrays_and_description(
            arrays,
            description,
        ))
    }

    fn get_spectrum_by_index(&mut self, index: usize) -> Option<MultiLayerSpectrum<C, D>> {
        self.get_spectrum(index).ok()
    }

    fn get_index(&self) -> &OffsetIndex {
        &self.metadata.spectrum_id_index
    }

    fn set_index(&mut self, index: OffsetIndex) {
        self.metadata.spectrum_id_index = index;
    }

    fn iter(&mut self) -> mzdata::io::SpectrumIterator<C, D, MultiLayerSpectrum<C, D>, Self>
        where
            Self: Sized, {
        if let Err(_) = self.populate_metadata_cache() {}
        mzdata::io::SpectrumIterator::new(self)
    }
}

impl<
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
> RandomAccessSpectrumIterator<C, D> for MzPeakReaderType<C, D>
{
    fn start_from_id(&mut self, id: &str) -> Result<&mut Self, SpectrumAccessError> {
        let s = self
            .get_spectrum_metadata_by_id(id)
            .map_err(|e| SpectrumAccessError::IOError(Some(e)))?;
        self.index = s.index;
        Ok(self)
    }

    fn start_from_index(&mut self, index: usize) -> Result<&mut Self, SpectrumAccessError> {
        self.index = index;
        Ok(self)
    }

    fn start_from_time(&mut self, time: f64) -> Result<&mut Self, SpectrumAccessError> {
        let dl = *self.detail_level();
        self.set_detail_level(DetailLevel::MetadataOnly);
        if let Some(spec) = self.get_spectrum_by_time(time) {
            self.index = spec.index();
            self.set_detail_level(dl);
            Ok(self)
        } else {
            self.set_detail_level(dl);
            Err(SpectrumAccessError::SpectrumNotFound)
        }
    }
}

impl<
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
> MSDataFileMetadata for MzPeakReaderType<C, D>
{
    mzdata::delegate_impl_metadata_trait!(metadata);
}

impl<
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
> MzPeakReaderType<C, D>
{
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().into();
        let handle = File::open(&path)?;
        let mut handle = ZipArchiveReader::new(handle)?;

        let (metadata, query_indices) = Self::load_indices_from(&mut handle)?;

        let this = Self {
            path,
            index: 0,
            detail_level: DetailLevel::Full,
            handle,
            metadata,
            query_indices,
            spectrum_metadata_cache: None,
            spectrum_row_group_cache: None,
            _t: Default::default(),
        };
        Ok(this)
    }

    pub fn populate_metadata_cache(&mut self) -> io::Result<()> {
        if self.spectrum_metadata_cache.is_none() {
            self.spectrum_metadata_cache = Some(self.load_all_spectrum_metadata().inspect_err(|e| log::error!("Failed to load spectrum metadata cache: {e}"))?);
        }
        Ok(())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn load_indices_from(
        handle: &mut ZipArchiveReader,
    ) -> io::Result<(MzPeakReaderMetadata, QueryIndex)> {
        let spectrum_metadata_reader = handle.spectrum_metadata()?;
        let spectrum_data_reader = handle.spectra_data()?;

        let mut mz_metadata = meta::FileMetadataConfig::default();
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
                        let software_list: Vec<crate::param::Software> =
                            serde_json::from_str(&val)?;
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
            mz_metadata,
            spectrum_array_indices,
            chromatogram_array_indices,
            spectrum_id_index,
        };

        Ok((bundle, query_index))
    }

    fn load_spectrum_data_row_group(&self, row_group: usize) -> io::Result<RecordBatch> {
        let builder = self.handle.spectra_data()?;
        let batch = builder.with_row_groups(vec![row_group]).with_batch_size(usize::MAX).build()?.flatten().next();
        if let Some(batch) = batch {
            Ok(batch)
        } else {
            Err(parquet::errors::ParquetError::General(format!("Couldn't read row group {row_group}")).into())
        }
    }

    fn read_spectrum_data_cache(&mut self, row_group: usize) -> io::Result<RecordBatch> {
        let cache_hit = if let Some((i, _)) = self.spectrum_row_group_cache.as_ref() {
            *i == row_group
        } else {
            false
        };
        if cache_hit {
            Ok(self.spectrum_row_group_cache.as_ref().map(|(_, rg)| rg.clone()).unwrap())
        } else {
            let rg = self.load_spectrum_data_row_group(row_group)?;
            self.spectrum_row_group_cache = Some((row_group, rg));
            Ok(self.spectrum_row_group_cache.as_ref().map(|(_, rg)| rg.clone()).unwrap())
        }
    }

    fn populate_arrays_from_struct_array(&self, points: &StructArray, bin_map: &mut HashMap<&String, DataArray>) {
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

    fn slice_spectrum_data_record_batch_to_arrays_of(&self, batch: &RecordBatch, index: u64) -> io::Result<BinaryArrayMap> {
        let mut bin_map = HashMap::new();
        for (k, v) in self.metadata.spectrum_array_indices.iter() {
            let dtype = crate::peak_series::arrow_to_array_type(&v.data_type).unwrap();
            bin_map.insert(&v.name, DataArray::from_name_and_type(k, dtype));
        }

        let points = batch.column(0).as_struct();
        let indices: &UInt64Array = points.column_by_name("spectrum_index").unwrap().as_any().downcast_ref().unwrap();
        let mut start = None;
        let mut end = None;

        for (i, idx) in indices.iter().enumerate() {
            if idx.unwrap() == index {
                if start.is_some() {
                    end = Some(i)
                } else {
                    start = Some(i)
                }
            }
        }

        let points = match (start, end) {
            (Some(start), Some(end)) => {
                let len = end - start + 1;
                points.slice(start, len)
            },
            (Some(start), None) => {
                points.slice(start, 1)
            },
            _ => {
                let mut out = BinaryArrayMap::new();
                for v in bin_map.into_values() {
                    out.add(v);
                }
                return Ok(out)
            }
        };

        self.populate_arrays_from_struct_array(&points, &mut bin_map);


        let mut out = BinaryArrayMap::new();
        for v in bin_map.into_values() {
            out.add(v);
        }
        Ok(out)
    }

    /// Read the complete data arrays for the spectrum at `index`
    pub fn get_spectrum_arrays(&mut self, index: u64) -> io::Result<BinaryArrayMap> {
        let builder = self.handle.spectra_data()?;

        let pq_schema = builder.parquet_schema();

        let mut rg_idx_acc = Vec::new();
        let mut pages: Vec<PageIndexEntry<u64>> = Vec::new();

        for page in self.query_indices.spectrum_point_spectrum_index.pages_contains(index) {
            if !rg_idx_acc.contains(&page.row_group_i) {
                rg_idx_acc.push(page.row_group_i);
            }
            pages.push(*page);
        }
        log::debug!("Reading pages: {pages:?} for index {index}");

        if rg_idx_acc.len() == 1 {
            let rg = self.read_spectrum_data_cache(rg_idx_acc[0])?;
            return self.slice_spectrum_data_record_batch_to_arrays_of(&rg, index)
        }

        log::info!("Reading index {index} from row groups {:?}", rg_idx_acc);
        let first_row = if !pages.is_empty() {
            let mut rg_row_skip = 0;

            for i in 0..rg_idx_acc[0] {
                log::debug!("Skipping row group {i}");
                let rg = builder.metadata().row_group(i);
                rg_row_skip += rg.num_rows();
            }
            let rg = builder.metadata().row_group(rg_idx_acc[0]);
            log::info!("{}-{}", rg.file_offset().unwrap(), rg.file_offset().unwrap() + rg.compressed_size());
            rg_row_skip
        } else {
            return Ok(BinaryArrayMap::new())
        };

        let rows = self.query_indices.spectrum_point_spectrum_index.pages_to_row_selection(pages.iter(), first_row);

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
            .with_row_groups(rg_idx_acc)
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
            self.populate_arrays_from_struct_array(points, &mut bin_map);
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
            let config_arr: &UInt32Array = scan_arr.column(2).as_any().downcast_ref().unwrap();
            let filter_string_arr: &LargeStringArray = scan_arr.column(3).as_string();
            let inject_arr: &Float32Array = scan_arr.column(4).as_any().downcast_ref().unwrap();
            let ion_mobility_arr: &Float64Array =
                scan_arr.column(5).as_any().downcast_ref().unwrap();
            let ion_mobility_tp_arr = scan_arr.column(6).as_struct();
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
                let params: Vec<crate::param::Param> =
                    serde_arrow::from_arrow(&param_fields, params.columns()).unwrap();
                event
                    .params_mut()
                    .extend(params.into_iter().map(|p| p.into()));

                if filter_string_arr.is_valid(pos) {
                    let filter = filter_string_arr.value(pos);
                    event.add_param(
                        mzdata::Param::builder()
                            .name("filter string")
                            .curie(mzdata::curie!(MS:1000512))
                            .value(filter)
                            .build(),
                    );
                }

                if config_arr.is_valid(pos) {
                    let conf = config_arr.value(pos);
                    event.add_param(
                        mzdata::Param::builder()
                            .name("preset scan configuration")
                            .curie(mzdata::curie!(MS:1000616))
                            .value(conf)
                            .build(),
                    );
                }

                if ion_mobility_arr.is_valid(pos) {
                    let val = ion_mobility_arr.value(pos);
                    if ion_mobility_tp_arr.is_valid(pos) {
                        let tp = ion_mobility_tp_arr
                            .column(1)
                            .as_any()
                            .downcast_ref::<UInt32Array>()
                            .unwrap();
                        let tp = mzdata::params::CURIE::new(
                            mzdata::params::ControlledVocabulary::MS,
                            tp.value(pos),
                        );
                        let param_builder = mzdata::Param::builder().curie(tp).value(val);
                        if tp == crate::param::ION_MOBILITY_SCAN_TERMS[0] {
                            event.add_param(
                                param_builder
                                    .name("ion mobility drift time")
                                    .unit(Unit::Millisecond)
                                    .build(),
                            );
                        } else if tp == crate::param::ION_MOBILITY_SCAN_TERMS[1] {
                            event.add_param(
                                param_builder
                                    .name("inverse reduced ion mobility drift time")
                                    .unit(Unit::VoltSecondPerSquareCentimeter)
                                    .build(),
                            );
                        } else if tp == crate::param::ION_MOBILITY_SCAN_TERMS[2] {
                            event.add_param(
                                param_builder
                                    .name("FAIMS compensation voltage")
                                    .unit(Unit::Volt)
                                    .build(),
                            );
                        } else if tp == crate::param::ION_MOBILITY_SCAN_TERMS[3] {
                            event.add_param(
                                param_builder
                                    .name("SELEXION compensation voltage")
                                    .unit(Unit::Volt)
                                    .build(),
                            );
                        }
                    }
                }

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
            let times = batch.column(0).as_struct().column(0);
            Ok(time_range.contains_dy(times))
        });

        let proj = ProjectionMask::columns(
            builder.parquet_schema(),
            ["spectrum.index", "spectrum.time"],
        );

        let reader = builder
            .with_row_selection(rows)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .with_projection(proj)
            .build()?;

        let mut min = u64::MAX;
        let mut max = 0;

        let mut times: HashMap<u64, f32> = HashMap::new();

        for batch in reader.flatten() {
            let root = batch.column(0).as_struct();
            let arr: &UInt64Array = root.column(0).as_any().downcast_ref().unwrap();
            let time_arr = root.column(1);
            if let Some(time_arr) = time_arr.as_any().downcast_ref::<Float32Array>() {
                for (val, time) in arr.iter().flatten().zip(time_arr.iter().flatten()) {
                    min = min.min(val);
                    max = max.max(val);
                    times.insert(val, time);
                }
            } else if let Some(time_arr) = time_arr.as_any().downcast_ref::<Float64Array>() {
                for (val, time) in arr
                    .iter()
                    .flatten()
                    .zip(time_arr.iter().flatten().map(|v| v as f32))
                {
                    min = min.min(val);
                    max = max.max(val);
                    times.insert(val, time);
                }
            } else {
                return Err(parquet::errors::ParquetError::ArrowError(format!(
                    "Invalid time array data type: {:?}",
                    time_arr.data_type()
                ))
                .into());
            }
        }
        Ok((times, SimpleInterval::new(min, max)))
    }

    pub fn extract_peaks(
        &mut self,
        time_range: SimpleInterval<f32>,
        mz_range: Option<SimpleInterval<f64>>,
        ion_mobility_range: Option<SimpleInterval<f64>>,
    ) -> io::Result<(ParquetRecordBatchReader, HashMap<u64, f32>)> {
        let (time_index, index_range) = self.get_spectrum_index_range_for_time_range(time_range)?;

        let mut rows = self
            .query_indices
            .spectrum_point_spectrum_index
            .row_selection_overlaps(&index_range);
        if let Some(mz_range) = mz_range.as_ref() {
            rows = rows.intersection(
                &self
                    .query_indices
                    .spectrum_mz_index
                    .row_selection_overlaps(mz_range),
            );
        }
        if let Some(ion_mobility_range) = ion_mobility_range.as_ref() {
            rows = rows.union(
                &self
                    .query_indices
                    .spectrum_im_index
                    .row_selection_overlaps(&ion_mobility_range),
            );
        }

        let sidx = format!(
            "{}.spectrum_index",
            self.metadata.spectrum_array_indices.prefix
        );

        let mut fields: Vec<&str> = Vec::new();

        fields.push(&sidx);

        if let Some(e) = self
            .metadata
            .spectrum_array_indices
            .get(&ArrayType::MZArray)
        {
            fields.push(e.path.as_str())
        }

        if let Some(e) = self
            .metadata
            .spectrum_array_indices
            .get(&ArrayType::IntensityArray)
        {
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
            let spectrum_index: &UInt64Array = root.column(0).as_any().downcast_ref().unwrap();

            let it = spectrum_index
                .iter()
                .map(|v| v.map(|v| index_range.contains(&v)));

            match (mz_range, ion_mobility_range) {
                (None, None) => Ok(it.collect()),
                (None, Some(ion_mobility_range)) => {
                    let im_array = root.column(1);
                    let it2 = ion_mobility_range.contains_dy(im_array);
                    let it2 = it2.iter();
                    // let it2 = im_array
                    //     .iter()
                    //     .map(|v| v.map(|v| ion_mobility_range.contains(&v)));
                    let it = it
                        .zip(it2)
                        .map(|(a, b)| Some(a.is_some_and(|a| a && b.unwrap_or_default())));
                    Ok(it.collect())
                }
                (Some(mz_range), None) => {
                    let mz_array = root.column(1);
                    let it2 = mz_range.contains_dy(mz_array);
                    let it2 = it2.iter();
                    let it = it
                        .zip(it2)
                        .map(|(a, b)| Some(a.is_some_and(|a| a && b.unwrap_or_default())));
                    Ok(it.collect())
                }
                (Some(mz_range), Some(ion_mobility_range)) => {
                    let mz_array = root.column(1);
                    let im_array = root.column(2);
                    let it2 = mz_range.contains_dy_iter(mz_array);
                    let it3 = ion_mobility_range.contains_dy_iter(im_array);
                    let it = it
                        .zip(it2)
                        .map(|(a, b)| Some(a.is_some_and(|a| a && b.unwrap_or_default())));
                    let it = it
                        .zip(it3)
                        .map(|(a, b)| Some(a.is_some_and(|a| a && b.unwrap_or_default())));
                    Ok(it.collect())
                }
            }
        });

        let reader: ParquetRecordBatchReader = builder
            .with_row_selection(rows)
            .with_projection(proj)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .build()?;

        Ok((reader, time_index))
    }

    pub fn len(&self) -> usize {
        self.metadata.spectrum_id_index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.metadata.spectrum_id_index.is_empty()
    }

    /// Read load descriptive metadata for the spectrum at `index`
    pub fn get_spectrum_metadata(&mut self, index: u64) -> io::Result<SpectrumDescription> {
        if let Some(cache) = self.spectrum_metadata_cache.as_ref() {
            if let Some(descr) =  cache.get(index as usize) {
                return Ok(descr.clone());
            }
        }

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
            SchemaLike::from_type::<crate::param::Param>(Default::default()).unwrap();

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
                    let params: Vec<crate::param::Param> =
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
            SchemaLike::from_type::<crate::param::Param>(Default::default()).unwrap();
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

    pub fn get_spectrum_metadata_by_id(&mut self, id: &str) -> io::Result<SpectrumDescription> {
        if let Some(idx) = self.metadata.spectrum_id_index.get(id) {
            return self.get_spectrum_metadata(idx);
        }
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Spectrum id \"{id}\" not found"),
        ))
    }

    pub fn get_spectrum(&mut self, index: usize) -> io::Result<MultiLayerSpectrum<C, D>> {
        let description = self.get_spectrum_metadata(index as u64)?;
        let arrays = if self.detail_level == DetailLevel::Full {
            self.get_spectrum_arrays(index as u64)?
        } else {
            BinaryArrayMap::new()
        };

        Ok(MultiLayerSpectrum::from_arrays_and_description(
            arrays,
            description,
        ))
    }

    pub fn tic(&mut self) -> io::Result<Chromatogram> {
        let builder = self.handle.spectrum_metadata()?;
        let rows = self
            .query_indices
            .spectrum_index_index
            .row_selection_is_not_null();

        let proj = ProjectionMask::columns(
            builder.parquet_schema(),
            ["spectrum.time", "spectrum.total_ion_current"],
        );

        let reader = builder
            .with_projection(proj)
            .with_row_selection(rows)
            .build()?;
        let mut time_array: Vec<u8> = Vec::new();
        let mut intensity_array: Vec<u8> = Vec::new();
        for batch in reader.flatten() {
            let root = batch.column(0).as_struct();
            let times: &Float32Array = root
                .column_by_name("time")
                .unwrap()
                .as_any()
                .downcast_ref()
                .unwrap();
            let ints: &Float32Array = root
                .column_by_name("total_ion_current")
                .unwrap()
                .as_any()
                .downcast_ref()
                .unwrap();
            for time in times {
                let time = time.unwrap() as f64;
                time_array.extend_from_slice(&time.to_le_bytes());
            }
            for int in ints {
                let int = int.unwrap();
                intensity_array.extend_from_slice(&int.to_le_bytes());
            }
        }

        let mut descr = mzdata::spectrum::ChromatogramDescription::default();
        descr.id = "TIC".to_string();
        descr.index = 0;
        descr.ms_level = None;
        descr.chromatogram_type = mzdata::spectrum::ChromatogramType::TotalIonCurrentChromatogram;

        let mut arrays = BinaryArrayMap::new();
        let mut time_array = DataArray::wrap(
            &ArrayType::TimeArray,
            mzdata::spectrum::BinaryDataArrayType::Float64,
            time_array,
        );
        time_array.unit = Unit::Minute;
        arrays.add(time_array);

        let mut intensity_array = DataArray::wrap(
            &ArrayType::IntensityArray,
            mzdata::spectrum::BinaryDataArrayType::Float32,
            intensity_array,
        );
        intensity_array.unit = Unit::DetectorCounts;
        arrays.add(intensity_array);

        let chrom = mzdata::spectrum::Chromatogram::new(descr, arrays);
        Ok(chrom)
    }

    pub fn bpc(&mut self) -> io::Result<Chromatogram> {
        let builder = self.handle.spectrum_metadata()?;
        let rows = self
            .query_indices
            .spectrum_index_index
            .row_selection_is_not_null();

        let proj = ProjectionMask::columns(
            builder.parquet_schema(),
            ["spectrum.time", "spectrum.base_peak_intensity"],
        );

        let reader = builder
            .with_projection(proj)
            .with_row_selection(rows)
            .build()?;
        let mut time_array: Vec<u8> = Vec::new();
        let mut intensity_array: Vec<u8> = Vec::new();
        for batch in reader.flatten() {
            let root = batch.column(0).as_struct();
            let times: &Float32Array = root
                .column_by_name("time")
                .unwrap()
                .as_any()
                .downcast_ref()
                .unwrap();
            let ints: &Float32Array = root
                .column_by_name("base_peak_intensity")
                .unwrap()
                .as_any()
                .downcast_ref()
                .unwrap();
            for time in times {
                let time = time.unwrap() as f64;
                time_array.extend_from_slice(&time.to_le_bytes());
            }
            for int in ints {
                let int = int.unwrap();
                intensity_array.extend_from_slice(&int.to_le_bytes());
            }
        }

        let mut descr = mzdata::spectrum::ChromatogramDescription::default();
        descr.id = "BPC".to_string();
        descr.index = 1;
        descr.ms_level = None;
        descr.chromatogram_type = mzdata::spectrum::ChromatogramType::TotalIonCurrentChromatogram;

        let mut arrays = BinaryArrayMap::new();
        let mut time_array = DataArray::wrap(
            &ArrayType::TimeArray,
            mzdata::spectrum::BinaryDataArrayType::Float64,
            time_array,
        );
        time_array.unit = Unit::Minute;
        arrays.add(time_array);

        let mut intensity_array = DataArray::wrap(
            &ArrayType::IntensityArray,
            mzdata::spectrum::BinaryDataArrayType::Float32,
            intensity_array,
        );
        intensity_array.unit = Unit::DetectorCounts;
        arrays.add(intensity_array);

        let chrom = mzdata::spectrum::Chromatogram::new(descr, arrays);
        Ok(chrom)
    }
}

pub type MzPeakReader = MzPeakReaderType<CentroidPeak, DeconvolutedPeak>;



#[cfg(test)]
mod test {
    use super::*;
    use mzdata::spectrum::ChromatogramLike;

    #[test]
    fn test_read_spectrum() -> io::Result<()> {
        let mut reader = MzPeakReader::new("small.mzpeak")?;
        let descr = reader.get_spectrum(0)?;
        assert_eq!(descr.index(), 0);
        assert_eq!(descr.peaks().len(), 19913);
        let descr = reader.get_spectrum(5)?;
        assert_eq!(descr.index(), 5);
        assert_eq!(descr.peaks().len(), 650);
        let descr = reader.get_spectrum(25)?;
        assert_eq!(descr.index(), 25);
        assert_eq!(descr.peaks().len(), 789);
        Ok(())
    }

    #[test]
    fn test_tic() -> io::Result<()> {
        let mut reader = MzPeakReader::new("small.mzpeak")?;
        let tic = reader.tic()?;
        assert_eq!(tic.index(), 0);
        assert_eq!(tic.time()?.len(), 48);
        Ok(())
    }

    #[test]
    fn test_load_all_metadata() -> io::Result<()> {
        let mut reader = MzPeakReader::new("small.mzpeak")?;
        let out = reader.load_all_spectrum_metadata()?;
        assert_eq!(out.len(), 48);
        Ok(())
    }

    #[test]
    fn test_eic() -> io::Result<()> {
        let mut reader = MzPeakReader::new("small.mzpeak")?;

        let (it, _time_index) =
            reader.extract_peaks((0.3..0.4).into(), Some((800.0..820.0).into()), None)?;

        for batch in it.flatten() {
            eprintln!("{:?}", batch);
        }
        Ok(())
    }
}
