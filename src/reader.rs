use std::{
    collections::HashMap,
    fs::File,
    io,
    marker::PhantomData,
    path::{Path, PathBuf},
};

use arrow::{
    array::{
        Array, AsArray, Float32Array, Float64Array, Int8Array, LargeListArray, LargeStringArray,
        RecordBatch, StructArray, UInt8Array, UInt32Array, UInt64Array,
    },
    datatypes::{DataType, FieldRef, Float32Type, Float64Type, UInt32Type, UInt64Type},
    error::ArrowError,
};

use identity_hash::BuildIdentityHasher;
use mzdata::{
    io::{DetailLevel, OffsetIndex},
    meta::{MSDataFileMetadata, SpectrumType},
    params::{ParamDescribed, Unit},
    prelude::*,
    spectrum::{
        ArrayType, BinaryArrayMap, Chromatogram, ChromatogramDescription, ChromatogramType,
        DataArray, MultiLayerSpectrum, ScanEvent, ScanPolarity, SpectrumDescription,
        bindata::BuildFromArrayMap,
    },
};
use mzpeaks::{
    CentroidPeak, DeconvolutedCentroidLike, DeconvolutedPeak, coordinate::SimpleInterval,
    prelude::Span1D,
};

use parquet::arrow::{
    ProjectionMask,
    arrow_reader::{ArrowPredicateFn, ParquetRecordBatchReader, RowFilter},
};
use serde_arrow::schema::SchemaLike;

use crate::{
    CURIE, PrecursorEntry, SelectedIonEntry,
    archive::ZipArchiveReader,
    filter::RegressionDeltaModel,
    reader::{
        chunk::SpectrumDataChunkCache,
        index::{PageQuery, QueryIndex, SpanDynNumeric},
        metadata::MzSpectrumBuilder,
    },
    spectrum::AuxiliaryArray,
};

mod chunk;
pub mod index;
mod metadata;
mod point;

pub use chunk::SpectrumChunkReader;
pub use metadata::MzPeakReaderMetadata;
use point::{SpectrumDataArrayReader, SpectrumDataPointCache};

const DRIFT_TIME: mzdata::params::CURIE = crate::param::ION_MOBILITY_SCAN_TERMS[0];
const INVERSE_K_DRIFT_TIME: mzdata::params::CURIE = crate::param::ION_MOBILITY_SCAN_TERMS[1];
const FAIMS: mzdata::params::CURIE = crate::param::ION_MOBILITY_SCAN_TERMS[2];
const SELEXION: mzdata::params::CURIE = crate::param::ION_MOBILITY_SCAN_TERMS[3];

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
    spectrum_row_group_cache: Option<DataCache>,
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
            self.load_all_spectrum_metadata().ok()?;
        }
        if self.index >= self.len() {
            return None;
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
        Self: Sized,
    {
        if let Err(_) = self.load_all_spectrum_metadata() {}
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
> SpectrumDataArrayReader for MzPeakReaderType<C, D>
{
}

#[allow(unused)]
#[derive(Debug, Default, Clone)]
pub struct TimeQueryResult {
    index_range: SimpleInterval<u64>,
    time_index: Vec<f32>,
}

impl TimeQueryResult {
    pub fn new(index_range: SimpleInterval<u64>, time_index: Vec<f32>) -> Self {
        Self {
            index_range,
            time_index,
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (u64, f32)> {
        self.time_index
            .iter()
            .enumerate()
            .map(|(i, v)| (i as u64 + self.start(), *v))
    }

    pub fn get(&self, index: u64) -> Option<f32> {
        if !self.index_range.contains(&index) {
            None
        } else {
            self.time_index
                .get((index - self.start()) as usize)
                .copied()
        }
    }
}

impl Span1D for TimeQueryResult {
    type DimType = u64;

    fn start(&self) -> Self::DimType {
        self.index_range.start
    }

    fn end(&self) -> Self::DimType {
        self.index_range.end
    }
}

pub(crate) enum DataCache {
    Point(SpectrumDataPointCache),
    Chunk(SpectrumDataChunkCache),
}

impl DataCache {
    pub fn slice_to_arrays_of(
        &mut self,
        row_group_index: usize,
        spectrum_index: u64,
        mz_delta_model: Option<&RegressionDeltaModel<f64>>,
    ) -> io::Result<BinaryArrayMap> {
        if self.contains(row_group_index, spectrum_index) {
            match self {
                DataCache::Point(spectrum_data_point_cache) => {
                    spectrum_data_point_cache.slice_to_arrays_of(spectrum_index, mz_delta_model)
                }
                DataCache::Chunk(spectrum_data_chunk_cache) => {
                    spectrum_data_chunk_cache.slice_to_arrays_of(spectrum_index, mz_delta_model)
                }
            }
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("Entries not found for {row_group_index}:{spectrum_index}"),
            ))
        }
    }

    pub fn contains(&self, row_group_index: usize, spectrum_index: u64) -> bool {
        match self {
            DataCache::Point(spectrum_data_point_cache) => {
                spectrum_data_point_cache.row_group_index == row_group_index
            }
            DataCache::Chunk(spectrum_data_chunk_cache) => spectrum_data_chunk_cache
                .spectrum_index_range
                .contains(&spectrum_index),
        }
    }

    pub fn load_data_for<
        C: CentroidLike + BuildFromArrayMap + BuildArrayMapFrom,
        D: DeconvolutedCentroidLike + BuildFromArrayMap + BuildArrayMapFrom,
    >(
        reader: &MzPeakReaderType<C, D>,
        row_group_index: usize,
        spectrum_index: u64,
    ) -> io::Result<Option<Self>> {
        if reader.query_indices.spectrum_point_index.is_populated() {
            let rg = reader
                .load_spectrum_data_row_group(reader.handle.spectra_data()?, row_group_index)?;
            let cache = SpectrumDataPointCache::new(
                rg,
                reader.metadata.spectrum_array_indices.clone(),
                row_group_index,
                None,
                None,
            );

            Ok(Some(Self::Point(cache)))
        } else if reader.query_indices.spectrum_chunk_index.is_populated() {
            let builder = reader.handle.spectra_data()?;
            let builder = SpectrumChunkReader::new(builder);
            let cache = builder.load_cache_block(
                SimpleInterval::new(spectrum_index, spectrum_index + 100),
                &reader.metadata,
                &reader.query_indices,
            )?;
            Ok(Some(Self::Chunk(cache)))
        } else {
            Ok(None)
        }
    }
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

        let mut this = Self {
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

        this.metadata.model_deltas = this.load_delta_models()?;
        this.metadata.auxliary_array_counts = this.load_auxiliary_array_count()?;

        Ok(this)
    }

    /// Load the descriptive metadata for all spectra
    ///
    /// This method caches the data after its first use.
    pub fn load_all_spectrum_metadata(&mut self) -> io::Result<Option<&[SpectrumDescription]>> {
        if self.spectrum_metadata_cache.is_none() {
            self.spectrum_metadata_cache = Some(
                self.load_all_spectrum_metadata_impl()
                    .inspect_err(|e| log::error!("Failed to load spectrum metadata cache: {e}"))?,
            );
        }
        Ok(self.spectrum_metadata_cache.as_deref())
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Load the various metadata, indices and reference data
    fn load_indices_from(
        handle: &mut ZipArchiveReader,
    ) -> io::Result<(MzPeakReaderMetadata, QueryIndex)> {
        metadata::load_indices_from(handle)
    }

    /// Load the [`SpectrumDataCache`] row group or retrieve the current cache if it matches the request
    fn read_spectrum_data_cache(
        &mut self,
        row_group_index: usize,
        spectrum_index: u64,
    ) -> io::Result<&mut DataCache> {
        let cache_hit = if let Some(cache) = self.spectrum_row_group_cache.as_ref() {
            cache.contains(row_group_index, spectrum_index)
        } else {
            false
        };

        if cache_hit {
            log::trace!("Spectrum data cache hit {row_group_index:?}:{spectrum_index}");
            Ok(self.spectrum_row_group_cache.as_mut().unwrap())
        } else {
            log::trace!("Spectrum data cache miss {row_group_index:?}:{spectrum_index}");
            if let Some(cache) = DataCache::load_data_for(self, row_group_index, spectrum_index)? {
                self.spectrum_row_group_cache = Some(cache);
                Ok(self.spectrum_row_group_cache.as_mut().unwrap())
            } else {
                Err(io::Error::other(format!(
                    "Failed to load data cache for {row_group_index:?} {spectrum_index}"
                )))
            }
        }
    }

    /// Read the complete data arrays for the spectrum at `index`
    pub fn get_spectrum_arrays(&mut self, index: u64) -> io::Result<BinaryArrayMap> {
        let delta_model = self.metadata.model_deltas_for_conv(index as usize);
        let builder = self.handle.spectra_data()?;
        let pq_schema = builder.parquet_schema();

        let PageQuery {
            pages,
            row_group_indices,
        } = self.query_indices.query_pages(index);

        // If there is only one row group in the scan, take the fast path through the cache
        if row_group_indices.len() == 1 {
            let row_group_index = row_group_indices[0];
            let rg = self.read_spectrum_data_cache(row_group_index, index)?;
            let mut arrays = rg.slice_to_arrays_of(row_group_index, index, delta_model.as_ref())?;
            for v in self.load_auxiliary_arrays_for(index)? {
                arrays.add(v);
            }
            return Ok(arrays);
        }

        if self.query_indices.spectrum_chunk_index.is_populated() {
            log::trace!("Using chunk strategy for reading spectrum {index}");
            return SpectrumChunkReader::new(builder).read_chunks_for_spectrum(
                index,
                &self.query_indices,
                &self.metadata.spectrum_array_indices,
                delta_model.as_ref(),
                Some(PageQuery::new(row_group_indices, pages)),
            );
        }

        // Otherwise we must construct a more intricate read plan, first pruning rows and row groups
        // based upon the pages matched
        let first_row = if !pages.is_empty() {
            let mut rg_row_skip = 0;
            let meta = builder.metadata();
            for i in 0..row_group_indices[0] {
                let rg = meta.row_group(i);
                rg_row_skip += rg.num_rows();
            }
            rg_row_skip
        } else {
            let mut out = BinaryArrayMap::new();
            for v in self.load_auxiliary_arrays_for(index)? {
                out.add(v);
            }
            return Ok(out);
        };

        let rows = self
            .query_indices
            .spectrum_point_index
            .spectrum_index
            .pages_to_row_selection(&pages, first_row);

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

        log::trace!("{index} spread across row groups {row_group_indices:?}");

        let reader = builder
            .with_row_groups(row_group_indices)
            .with_row_selection(rows)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .with_projection(proj)
            .build()?;

        let mut bin_map = HashMap::new();
        for v in self.metadata.spectrum_array_indices.iter() {
            bin_map.insert(&v.name, v.as_buffer_name().as_data_array(1024));
        }

        let batches: Vec<_> = reader.flatten().collect();
        if !batches.is_empty() {
            let batch = arrow::compute::concat_batches(batches[0].schema_ref(), &batches).unwrap();
            let points = batch.column(0).as_struct();
            self.populate_arrays_from_struct_array(points, &mut bin_map, delta_model.as_ref());
        }

        let mut out = BinaryArrayMap::new();
        for v in bin_map.into_values() {
            out.add(v);
        }

        for v in self.load_auxiliary_arrays_for(index)? {
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

                        match tp {
                            DRIFT_TIME => {
                                event.add_param(
                                    param_builder
                                        .name("ion mobility drift time")
                                        .unit(Unit::Millisecond)
                                        .build(),
                                );
                            }
                            INVERSE_K_DRIFT_TIME => {
                                event.add_param(
                                    param_builder
                                        .name("inverse reduced ion mobility drift time")
                                        .unit(Unit::VoltSecondPerSquareCentimeter)
                                        .build(),
                                );
                            }
                            FAIMS => {
                                event.add_param(
                                    param_builder
                                        .name("inverse reduced ion mobility drift time")
                                        .unit(Unit::VoltSecondPerSquareCentimeter)
                                        .build(),
                                );
                            }
                            SELEXION => {
                                event.add_param(
                                    param_builder
                                        .name("SELEXION compensation voltage")
                                        .unit(Unit::Volt)
                                        .build(),
                                );
                            }
                            _ => {}
                        }
                    }
                }

                scan_accumulator.push((index.unwrap(), event));
            }
        }
    }

    fn load_spectrum_metadata_from_slice_into_description(
        &self,
        spec_arr: &StructArray,
        descr: &mut SpectrumDescription,
        offset: usize,
        param_fields: &[FieldRef],
    ) {
        if let Some(metacols) = self.metadata.spectrum_metadata_map.as_ref() {
            log::warn!("Loading single description at {offset}");
            let mut builder =
                MzSpectrumBuilder::new(std::slice::from_mut(descr), &metacols, offset);
            builder.visit(spec_arr);
            return;
        }

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
            crate::curie!(MS:1000525) => mzdata::spectrum::SignalContinuity::Unknown,
            crate::curie!(MS:1000127) => mzdata::spectrum::SignalContinuity::Centroid,
            crate::curie!(MS:1000128) => mzdata::spectrum::SignalContinuity::Profile,
            _ => todo!("Don't know how to deal with {continuity_curie}"),
        };

        let spec_type_array = spec_arr.column(6).as_struct();
        let cv_id_arr: &UInt8Array = spec_type_array.column(0).as_any().downcast_ref().unwrap();
        let accession_arr: &UInt32Array =
            spec_type_array.column(1).as_any().downcast_ref().unwrap();
        let spec_type_curie = CURIE::new(cv_id_arr.value(offset), accession_arr.value(offset));
        let spec_type = SpectrumType::from_accession(spec_type_curie.accession);
        if let Some(spec_type) = spec_type {
            descr.set_spectrum_type(spec_type);
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

    fn load_chromatogram_metadata_from_slice_into_description(
        &self,
        spec_arr: &StructArray,
        descr: &mut ChromatogramDescription,
        offset: usize,
        param_fields: &[FieldRef],
    ) {
        let idx_arr: &UInt64Array = spec_arr.column(0).as_any().downcast_ref().unwrap();
        let idx_val = idx_arr.value(offset);
        descr.index = idx_val as usize;

        let id_arr: &LargeStringArray = spec_arr.column(1).as_string();
        let id_val = id_arr.value(offset);
        descr.id = id_val.to_string();

        let polarity_arr: &Int8Array = spec_arr.column(2).as_any().downcast_ref().unwrap();
        let polarity_val = polarity_arr.value(offset);
        match polarity_val {
            1 => descr.polarity = ScanPolarity::Positive,
            -1 => descr.polarity = ScanPolarity::Negative,
            _ => {
                todo!("Don't know how to deal with polarity {polarity_val}")
            }
        }

        let chromatogram_type_array = spec_arr.column(3).as_struct();
        let cv_id_arr: &UInt8Array = chromatogram_type_array
            .column(0)
            .as_any()
            .downcast_ref()
            .unwrap();
        let accession_arr: &UInt32Array = chromatogram_type_array
            .column(1)
            .as_any()
            .downcast_ref()
            .unwrap();
        let chromatogram_type = CURIE::new(cv_id_arr.value(offset), accession_arr.value(offset));
        if let Some(tp) = ChromatogramType::from_accession(chromatogram_type.accession) {
            descr.chromatogram_type = tp;
        } else {
            descr.chromatogram_type = ChromatogramType::Unknown;
        }

        let params_array: &LargeListArray =
            spec_arr.column_by_name("parameters").unwrap().as_list();
        let params = params_array.value(offset);
        let params = params.as_struct();

        const SKIP_PARAMS: &[CURIE] = &[];
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
        &self,
        time_range: SimpleInterval<f32>,
    ) -> io::Result<(
        HashMap<u64, f32, BuildIdentityHasher<u64>>,
        SimpleInterval<u64>,
    )> {
        // let rows = self
        //     .query_indices
        //     .spectrum_time_index
        //     .row_selection_overlaps(&time_range);

        if let Some(cache) = self.spectrum_metadata_cache.as_ref() {
            let mut times: HashMap<u64, f32, _> = HashMap::default();
            let mut min = u64::MAX;
            let mut max = 0;

            let n = cache.len();

            let offset_start = match cache.binary_search_by(|descr| {
                time_range
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

            for (i, descr) in cache.iter().enumerate().skip(offset) {
                let i = i as u64;
                let t = descr.acquisition.start_time() as f32;
                if time_range.contains(&t) {
                    min = min.min(i);
                    max = max.max(i);
                    times.insert(i, t);
                } else if !times.is_empty() {
                    break;
                }
            }
            return Ok((times, SimpleInterval::new(min, max)));
        }

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
            // .with_row_selection(rows)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .with_projection(proj)
            .build()?;

        let mut min = u64::MAX;
        let mut max = 0;

        let mut times: HashMap<u64, f32, _> = HashMap::default();

        for batch in reader.flatten() {
            let root = batch.column(0).as_struct();
            let arr: &UInt64Array = root.column(0).as_primitive();
            let time_arr = root.column(1);
            if let Some(time_arr) = time_arr.as_primitive_opt::<Float32Type>() {
                for (val, time) in arr.iter().flatten().zip(time_arr.iter().flatten()) {
                    min = min.min(val);
                    max = max.max(val);
                    times.insert(val, time);
                }
            } else if let Some(time_arr) = time_arr.as_primitive_opt::<Float64Type>() {
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

    /// Read all signal data within the specified `time_range`, optionally constrained to `mz_range` m/z values and/or
    /// `ion_mobility_range` IM values.
    pub fn extract_peaks(
        &mut self,
        time_range: SimpleInterval<f32>,
        mz_range: Option<SimpleInterval<f64>>,
        ion_mobility_range: Option<SimpleInterval<f64>>,
    ) -> io::Result<(
        Box<dyn Iterator<Item = Result<RecordBatch, ArrowError>> + '_>,
        HashMap<u64, f32, BuildIdentityHasher<u64>>,
    )> {
        let (time_index, index_range) = self.get_spectrum_index_range_for_time_range(time_range)?;
        let builder = self.handle.spectra_data()?;

        if self.query_indices.spectrum_chunk_index.is_populated() {
            if ion_mobility_range.is_some() {
                todo!("Ion mobility filter is not implemented for the chunked encoding");
            }
            return Ok((
                Box::new(SpectrumChunkReader::new(builder).scan_chunks_for(
                    index_range,
                    mz_range,
                    &self.metadata,
                    &self.query_indices,
                )?),
                time_index,
            ));
        }

        let mut rows = self
            .query_indices
            .spectrum_point_index
            .spectrum_index
            .row_selection_overlaps(&index_range);

        let PageQuery { row_group_indices, pages } = self.query_indices.spectrum_point_index.query_pages_overlaps(&index_range);

        if pages.is_empty() {
            return Ok((Box::new(std::iter::empty()), HashMap::default()));
        }

        let mut up_to_first_row = 0;
        let meta = builder.metadata();
        for i in 0..row_group_indices[0] {
            let rg = meta.row_group(i);
            up_to_first_row += rg.num_rows();
        }

        if let Some(mz_range) = mz_range.as_ref() {
            rows = rows.intersection(
                &self
                    .query_indices
                    .spectrum_point_index
                    .mz_index
                    .row_selection_overlaps(mz_range),
            );
        }

        if let Some(ion_mobility_range) = ion_mobility_range.as_ref() {
            rows = rows.union(
                &self
                    .query_indices
                    .spectrum_point_index
                    .im_index
                    .row_selection_overlaps(&ion_mobility_range),
            );
        }

        rows.split_off(up_to_first_row as usize);

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
            fields.push(e.path.as_str());
        }

        if let Some(e) = self
            .metadata
            .spectrum_array_indices
            .get(&ArrayType::IntensityArray)
        {
            fields.push(e.path.as_str());
        }

        for v in self.metadata.spectrum_array_indices.iter() {
            if v.is_ion_mobility() {
                fields.push(v.path.as_str());
                break;
            }
        }

        let proj = ProjectionMask::columns(builder.parquet_schema(), fields.iter().copied());
        let predicate_mask = proj.clone();

        let predicate = ArrowPredicateFn::new(predicate_mask, move |batch| {
            let root = batch.column(0).as_struct();
            let it = index_range.contains_dy(root.column(0));

            match (mz_range, ion_mobility_range) {
                (None, None) => Ok(it),
                (None, Some(ion_mobility_range)) => {
                    let im_array = root.column(1);
                    let it2 = ion_mobility_range.contains_dy(im_array);
                    arrow::compute::and(&it, &it2)
                }
                (Some(mz_range), None) => {
                    let mz_array = root.column(1);
                    let it2 = mz_range.contains_dy(mz_array);
                    arrow::compute::and(&it, &it2)
                }
                (Some(mz_range), Some(ion_mobility_range)) => {
                    let mz_array = root.column(1);
                    let im_array = root.column(2);
                    let it2 = mz_range.contains_dy(mz_array);
                    let it3 = ion_mobility_range.contains_dy(im_array);
                    arrow::compute::and(&arrow::compute::and(&it2, &it3)?, &it)
                }
            }
        });

        let reader: ParquetRecordBatchReader = builder
            .with_row_groups(row_group_indices)
            .with_row_selection(rows)
            .with_projection(proj)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .with_batch_size(10_000)
            .build()?;

        Ok((Box::new(reader), time_index))
    }

    /// Get the number of spectra in the file
    pub fn len(&self) -> usize {
        self.metadata.spectrum_id_index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.metadata.spectrum_id_index.is_empty()
    }

    /// Read load descriptive metadata for the spectrum at `index`
    pub fn get_spectrum_metadata(&mut self, index: u64) -> io::Result<SpectrumDescription> {
        if let Some(cache) = self.spectrum_metadata_cache.as_ref() {
            if let Some(descr) = cache.get(index as usize) {
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
                self.load_spectrum_metadata_from_slice_into_description(
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

    pub(crate) fn load_auxiliary_array_count(&self) -> io::Result<Vec<u32>> {
        let builder = self.handle.spectrum_metadata()?;

        let schema = builder.parquet_schema();
        let mut index_i = None;
        let mut auxiliary_count_i = None;
        for (i, c) in schema.columns().iter().enumerate() {
            let parts = c.path().parts();
            if parts == ["spectrum", "index"] {
                index_i = Some(i);
            }
            if parts
                .iter()
                .zip(["spectrum", "number_of_auxiliary_arrays"])
                .all(|(a, b)| a == b)
            {
                auxiliary_count_i = Some(i);
            }
        }

        let proj = match (index_i, auxiliary_count_i) {
            (Some(i), Some(j)) => ProjectionMask::leaves(schema, [i, j]),
            _ => return Ok(Vec::new()),
        };

        let reader = builder.with_projection(proj).build()?;
        let n = self.len();
        let mut number_of_auxiliary_arrays = Vec::new();
        number_of_auxiliary_arrays.resize(n, 0);
        for batch in reader.flatten() {
            let root = batch.column(0).as_struct();
            let index_array: &UInt64Array = root.column(0).as_primitive();
            let values_array = root.column(1);

            if let Some(values) = values_array.as_primitive_opt::<UInt32Type>() {
                for (i, c) in index_array.iter().zip(values.iter()) {
                    number_of_auxiliary_arrays[i.unwrap() as usize] = c.unwrap();
                }
            } else if let Some(values) = values_array.as_primitive_opt::<UInt64Type>() {
                for (i, c) in index_array.iter().zip(values.iter()) {
                    number_of_auxiliary_arrays[i.unwrap() as usize] = c.unwrap() as u32;
                }
            } else {
                unimplemented!(
                    "auxiliary array count stored as {:?}",
                    values_array.data_type()
                )
            }
        }
        Ok(number_of_auxiliary_arrays)
    }

    pub(crate) fn load_auxiliary_arrays_for(&self, index: u64) -> io::Result<Vec<DataArray>> {
        if self
            .metadata
            .auxliary_array_counts
            .get(index as usize)
            .copied()
            .unwrap_or_default()
            == 0
        {
            return Ok(Vec::new());
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
            ["spectrum.index", "spectrum.auxiliary_arrays"],
        );

        let proj = predicate_mask.clone();

        let predicate = ArrowPredicateFn::new(predicate_mask, move |batch| {
            let spectrum_index: &UInt64Array = batch
                .column(0)
                .as_struct()
                .column(0)
                .as_any()
                .downcast_ref()
                .unwrap();
            Ok(spectrum_index
                .iter()
                .map(|v| v.map(|i| i == index))
                .collect())
        });

        let filter = RowFilter::new(vec![Box::new(predicate)]);

        let reader = builder
            .with_projection(proj)
            .with_row_filter(filter)
            .with_row_selection(rows)
            .build()?;

        let mut results = Vec::new();
        let fields: Vec<FieldRef> =
            SchemaLike::from_type::<AuxiliaryArray>(Default::default()).unwrap();
        for bat in reader.flatten() {
            let root = bat.column(0);
            let root = root.as_struct();
            let data = root.column(1).as_list::<i64>();
            let data = data.values().as_struct();
            let arrays: Vec<AuxiliaryArray> =
                serde_arrow::from_arrow(&fields, data.columns()).unwrap();
            for array in arrays {
                results.push(array.into());
            }
        }

        Ok(results)
    }

    /// Load median delta coefficient column if it is present.
    pub(crate) fn load_delta_models(&mut self) -> io::Result<Vec<Option<Vec<f64>>>> {
        let builder = self.handle.spectrum_metadata()?;

        let schema = builder.parquet_schema();
        let mut index_i = None;
        let mut median_i = None;
        for (i, c) in schema.columns().iter().enumerate() {
            let parts = c.path().parts();
            if parts == ["spectrum", "index"] {
                index_i = Some(i);
            }
            if parts
                .iter()
                .zip(["spectrum", "median_delta"])
                .all(|(a, b)| a == b)
                || parts
                    .iter()
                    .zip(["spectrum", "mz_delta_model"])
                    .all(|(a, b)| a == b)
            {
                median_i = Some(i);
            }
        }

        let proj = match (index_i, median_i) {
            (Some(i), Some(j)) => ProjectionMask::leaves(schema, [i, j]),
            _ => return Ok(Vec::new()),
        };

        let reader = builder.with_projection(proj).build()?;
        let n = self.len();
        let mut medians = Vec::new();
        medians.resize(n, None);
        for batch in reader.flatten() {
            let root = batch.column(0).as_struct();
            let index_array: &UInt64Array = root.column(0).as_primitive();

            if let Some(val_array) = root.column(1).as_list_opt::<i64>() {
                match val_array.value_type() {
                    DataType::Float32 => {
                        for (i, val) in index_array.iter().zip(val_array.iter()) {
                            if let Some(i) = i {
                                medians[i as usize] = val.map(|v| -> Vec<f64> {
                                    v.as_primitive::<Float32Type>()
                                        .iter()
                                        .map(|i| i.unwrap() as f64)
                                        .collect()
                                });
                            }
                        }
                    }
                    DataType::Float64 => {
                        for (i, val) in index_array.iter().zip(val_array.iter()) {
                            if let Some(i) = i {
                                medians[i as usize] = val.map(|v| -> Vec<f64> {
                                    let val = v.as_primitive::<Float64Type>();
                                    val.values().to_vec()
                                });
                            }
                        }
                    }
                    _ => {}
                }
            } else if let Some(val_array) = root.column(1).as_primitive_opt::<Float32Type>() {
                for (i, val) in index_array.iter().zip(val_array) {
                    if let Some(i) = i {
                        medians[i as usize] = val.map(|v| vec![v as f64]);
                    }
                }
            } else if let Some(val_array) = root.column(1).as_primitive_opt::<Float64Type>() {
                for (i, val) in index_array.iter().zip(val_array) {
                    if let Some(i) = i {
                        medians[i as usize] = val.map(|v| vec![v]);
                    }
                }
            }
        }

        Ok(medians)
    }

    pub(crate) fn load_all_spectrum_metadata_impl(
        &mut self,
    ) -> io::Result<Vec<SpectrumDescription>> {
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
            .with_batch_size(10_000)
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
            let n_spec = index_arr.len() - index_arr.null_count();
            if let Some(metacols) = self.metadata.spectrum_metadata_map.as_ref() {
                let mut local_descr = vec![SpectrumDescription::default(); n_spec];
                let mut builder =
                    MzSpectrumBuilder::new(&mut local_descr, &metacols, 0);
                builder.visit(spec_arr);
                descriptions.extend(local_descr);
            } else {
                for (i, val) in index_arr.iter().enumerate() {
                    if val.is_some() {
                        let mut descr = SpectrumDescription::default();
                        self.load_spectrum_metadata_from_slice_into_description(
                            spec_arr,
                            &mut descr,
                            i,
                            &param_fields,
                        );
                        descriptions.push(descr);
                    }
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

    #[allow(unused)]
    pub(crate) fn load_all_chromatgram_metadata_impl(
        &self,
    ) -> io::Result<Vec<ChromatogramDescription>> {
        let builder = self.handle.chromatograms_metadata()?;
        let predicate_mask = ProjectionMask::columns(
            builder.parquet_schema(),
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

        let reader = builder
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .build()?;

        let mut descriptions: Vec<ChromatogramDescription> = Default::default();
        let mut precursors: Vec<PrecursorEntry> = Vec::new();
        let mut selected_ions: Vec<SelectedIonEntry> = Vec::new();

        let param_fields: Vec<FieldRef> =
            SchemaLike::from_type::<crate::param::Param>(Default::default()).unwrap();
        let precursor_fields: Vec<FieldRef> =
            SchemaLike::from_type::<PrecursorEntry>(Default::default()).unwrap();
        let selected_ion_fields: Vec<FieldRef> =
            SchemaLike::from_type::<SelectedIonEntry>(Default::default()).unwrap();

        for batch in reader.flatten() {
            let spec_arr = batch.column_by_name("chromatogram").unwrap().as_struct();
            let index_arr: &UInt64Array = spec_arr
                .column_by_name("index")
                .unwrap()
                .as_any()
                .downcast_ref()
                .unwrap();

            for (i, val) in index_arr.iter().enumerate() {
                if val.is_some() {
                    let mut descr = ChromatogramDescription::default();
                    self.load_chromatogram_metadata_from_slice_into_description(
                        spec_arr,
                        &mut descr,
                        i,
                        &param_fields,
                    );
                    descriptions.push(descr);
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

        for (idx, precursor) in precursors {
            descriptions[idx.unwrap() as usize].precursor = Some(precursor);
        }

        Ok(descriptions)
    }

    /// Retrieve the metadata for a spectrum by its `nativeId`
    pub fn get_spectrum_metadata_by_id(&mut self, id: &str) -> io::Result<SpectrumDescription> {
        if let Some(idx) = self.metadata.spectrum_id_index.get(id) {
            return self.get_spectrum_metadata(idx);
        }
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Spectrum id \"{id}\" not found"),
        ))
    }

    /// Retrieve a complete spectrum by its index
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

    /// Read the total ion chromatogram from the surrogate metadata in the spectrum table
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

    /// Read the base peak chromatogram from the surrogate metadata in the spectrum table
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
    use mzdata::spectrum::{ChromatogramLike, SignalContinuity};

    #[test]
    fn test_read_spectrum() -> io::Result<()> {
        let mut reader = MzPeakReader::new("small.mzpeak")?;
        let descr = reader.get_spectrum(0)?;
        assert_eq!(descr.index(), 0);
        assert_eq!(descr.signal_continuity(), SignalContinuity::Profile);
        assert_eq!(descr.peaks().len(), 13589);
        let descr = reader.get_spectrum(5)?;
        assert_eq!(descr.index(), 5);
        assert_eq!(descr.peaks().len(), 650);
        let descr = reader.get_spectrum(25)?;
        assert_eq!(descr.index(), 25);
        assert_eq!(descr.peaks().len(), 789);
        Ok(())
    }

    #[test]
    fn test_read_spectrum_chunked() -> io::Result<()> {
        let mut reader = MzPeakReader::new("small.chunked.mzpeak")?;
        let descr = reader.get_spectrum(0)?;
        assert_eq!(descr.index(), 0);
        assert_eq!(descr.signal_continuity(), SignalContinuity::Profile);
        assert_eq!(descr.peaks().len(), 13589);
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
        let out = reader.load_all_spectrum_metadata_impl()?;
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

    #[test]
    fn test_eic_chunked() -> io::Result<()> {
        let mut reader = MzPeakReader::new("small.chunked.mzpeak")?;

        let (it, _time_index) =
            reader.extract_peaks((0.3..0.4).into(), Some((800.0..820.0).into()), None)?;

        for batch in it.flatten() {
            eprintln!("{:?}", batch);
        }
        Ok(())
    }
}
