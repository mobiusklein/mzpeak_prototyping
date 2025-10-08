use std::{
    collections::HashMap,
    io,
    marker::PhantomData,
    path::{Path, PathBuf},
};

use arrow::{
    array::{Array, AsArray, Float32Array, RecordBatch, StructArray, UInt64Array},
    datatypes::{DataType, Float32Type, Float64Type, Int32Type, Int64Type, UInt32Type, UInt64Type},
    error::ArrowError,
};

use identity_hash::BuildIdentityHasher;
use mzdata::{
    io::{DetailLevel, OffsetIndex},
    meta::MSDataFileMetadata,
    params::Unit,
    prelude::*,
    spectrum::{
        ArrayType, BinaryArrayMap, Chromatogram, ChromatogramDescription, DataArray,
        MultiLayerSpectrum, PeakDataLevel, Precursor, ScanEvent, SelectedIon, SpectrumDescription,
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

use crate::{
    BufferContext,
    archive::{ArchiveReader, ArchiveSource, DirectorySource, ZipArchiveSource},
    filter::RegressionDeltaModel,
    param::MetadataColumn,
    reader::{
        chunk::DataChunkCache,
        index::{PageQuery, QueryIndex, SpanDynNumeric},
        point::PointDataReader,
        visitor::{
            AuxiliaryArrayVisitor, MzChromatogramBuilder, MzPrecursorVisitor, MzScanVisitor,
            MzSelectedIonVisitor, MzSpectrumVisitor,
        },
    },
};

mod chunk;
pub mod index;
mod metadata;
mod point;
mod visitor;

pub use visitor::{CURIEArray, UnitArray};

pub use chunk::SpectrumChunkReader;
pub use metadata::ReaderMetadata;
use point::{DataPointCache, PointDataArrayReader};

/// A reader for mzPeak files, abstract over the source type.
pub struct MzPeakReaderTypeOfSource<
    T: ArchiveSource = ZipArchiveSource,
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap = CentroidPeak,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap = DeconvolutedPeak,
> {
    path: PathBuf,
    handle: ArchiveReader<T>,
    index: usize,
    detail_level: DetailLevel,
    pub metadata: ReaderMetadata,
    pub query_indices: QueryIndex,
    spectrum_metadata_cache: Option<Vec<SpectrumDescription>>,
    spectrum_row_group_cache: Option<SpectrumDataCache>,
    _t: PhantomData<(C, D)>,
}

impl<
    T: ArchiveSource,
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
> ChromatogramSource for MzPeakReaderTypeOfSource<T, C, D>
{
    fn get_chromatogram_by_id(&mut self, id: &str) -> Option<Chromatogram> {
        if let Some(chrom) = self.get_chromatogram_by_id(id) {
            return Some(chrom);
        }
        match id {
            "TIC" => self.encoded_tic().ok(),
            "BPC" => self.encoded_bpc().ok(),
            _ => None,
        }
    }

    fn get_chromatogram_by_index(&mut self, index: usize) -> Option<Chromatogram> {
        if let Some(chrom) = self.get_chromatogram(index) {
            return Some(chrom);
        }
        match index {
            0 => self.encoded_tic().ok(),
            1 => self.encoded_bpc().ok(),
            _ => None,
        }
    }
}

impl<
    T: ArchiveSource,
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
> ExactSizeIterator for MzPeakReaderTypeOfSource<T, C, D>
{
    fn len(&self) -> usize {
        self.len()
    }
}

/// [`MzPeakReaderType`] implements the [`Iterator`] trait, but the first time `next` is called
/// will call [`MzPeakReaderType::load_all_spectrum_metadata`], which may produce a brief delay
/// before the first spectrum is produced.
impl<
    T: ArchiveSource,
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
> Iterator for MzPeakReaderTypeOfSource<T, C, D>
{
    type Item = MultiLayerSpectrum<C, D>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.spectrum_metadata_cache.is_none() {
            self.load_all_spectrum_metadata().ok()?;
        }
        if self.index >= self.len() {
            return None;
        }
        let x = self.get_spectrum(self.index);
        self.index += 1;
        x
    }
}

impl<
    T: ArchiveSource,
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
> SpectrumSource<C, D> for MzPeakReaderTypeOfSource<T, C, D>
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
        let description = self.get_spectrum_metadata_by_id(id).ok()??;
        let arrays = if self.detail_level == DetailLevel::Full {
            self.get_spectrum_arrays(description.index as u64).ok()??
        } else {
            BinaryArrayMap::new()
        };
        Some(MultiLayerSpectrum::from_arrays_and_description(
            arrays,
            description,
        ))
    }

    fn get_spectrum_by_index(&mut self, index: usize) -> Option<MultiLayerSpectrum<C, D>> {
        self.get_spectrum(index)
    }

    fn get_index(&self) -> &OffsetIndex {
        &self.metadata.spectrum_id_index
    }

    fn set_index(&mut self, index: OffsetIndex) {
        self.metadata.spectrum_id_index = index;
    }

    fn iter(&mut self) -> mzdata::io::SpectrumIterator<'_, C, D, MultiLayerSpectrum<C, D>, Self>
    where
        Self: Sized,
    {
        if let Err(_) = self.load_all_spectrum_metadata() {}
        mzdata::io::SpectrumIterator::new(self)
    }
}

impl<
    T: ArchiveSource,
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
> RandomAccessSpectrumIterator<C, D> for MzPeakReaderTypeOfSource<T, C, D>
{
    fn start_from_id(&mut self, id: &str) -> Result<&mut Self, SpectrumAccessError> {
        let s = self
            .get_spectrum_metadata_by_id(id)
            .map_err(|e| SpectrumAccessError::IOError(Some(e)))?
            .unwrap();
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
    T: ArchiveSource,
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
> MSDataFileMetadata for MzPeakReaderTypeOfSource<T, C, D>
{
    mzdata::delegate_impl_metadata_trait!(metadata);
}

impl<
    T: ArchiveSource,
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
> PointDataArrayReader for MzPeakReaderTypeOfSource<T, C, D>
{
}

// This value can be made larger for a modest (<10%) improvement in linear reading performance
// but the trade-off in memory load makes this impractical, especially if spectra are very,
// very dense.
const CHUNK_CACHE_BLOCK_SIZE: u64 = 100;

const EMPTY_FIELDS: [MetadataColumn; 0] = [];

pub(crate) enum SpectrumDataCache {
    Point(DataPointCache),
    Chunk(DataChunkCache),
}

impl SpectrumDataCache {
    pub fn slice_to_arrays_of(
        &mut self,
        row_group_index: usize,
        spectrum_index: u64,
        mz_delta_model: Option<&RegressionDeltaModel<f64>>,
    ) -> io::Result<BinaryArrayMap> {
        if self.contains(row_group_index, spectrum_index) {
            match self {
                SpectrumDataCache::Point(spectrum_data_point_cache) => {
                    spectrum_data_point_cache.slice_to_arrays_of(spectrum_index, mz_delta_model)
                }
                SpectrumDataCache::Chunk(spectrum_data_chunk_cache) => {
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
            SpectrumDataCache::Point(spectrum_data_point_cache) => {
                spectrum_data_point_cache.row_group_index == row_group_index
            }
            SpectrumDataCache::Chunk(spectrum_data_chunk_cache) => spectrum_data_chunk_cache
                .spectrum_index_range
                .contains(&spectrum_index),
        }
    }

    pub fn load_data_for<
        T: ArchiveSource,
        C: CentroidLike + BuildFromArrayMap + BuildArrayMapFrom,
        D: DeconvolutedCentroidLike + BuildFromArrayMap + BuildArrayMapFrom,
    >(
        reader: &MzPeakReaderTypeOfSource<T, C, D>,
        row_group_index: usize,
        spectrum_index: u64,
    ) -> io::Result<Option<Self>> {
        if reader.query_indices.spectrum_point_index.is_populated() {
            let rg = reader.load_cache_block(reader.handle.spectra_data()?, row_group_index)?;
            let cache = DataPointCache::new(
                rg,
                reader.metadata.spectrum_array_indices.clone(),
                row_group_index,
                None,
                None,
                BufferContext::Spectrum,
            );

            Ok(Some(Self::Point(cache)))
        } else if reader.query_indices.spectrum_chunk_index.is_populated() {
            let builder = reader.handle.spectra_data()?;
            let builder = SpectrumChunkReader::new(builder);
            let cache = builder.load_cache_block(
                SimpleInterval::new(spectrum_index, spectrum_index + CHUNK_CACHE_BLOCK_SIZE),
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
    T: ArchiveSource,
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap,
> MzPeakReaderTypeOfSource<T, C, D>
{
    /// Open an mzPeak archive found at a specified path
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path: PathBuf = path.as_ref().into();
        let mut handle = ArchiveReader::<T>::from_path(path.clone())?;
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

        this.metadata.mz_model_deltas = this.load_delta_models()?;
        this.metadata.spectrum_auxiliary_array_counts =
            this.load_spectrum_auxiliary_array_count()?;
        this.metadata.chromatogram_auxiliary_array_counts =
            this.load_chromatogram_auxiliary_array_count()?;

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

    /// The location of the archive.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Load the various metadata, indices and reference data
    fn load_indices_from(
        handle: &mut ArchiveReader<T>,
    ) -> io::Result<(ReaderMetadata, QueryIndex)> {
        metadata::load_indices_from(handle)
    }

    /// Load the [`SpectrumDataCache`] row group or retrieve the current cache if it matches the request
    fn read_spectrum_data_cache(
        &mut self,
        row_group_index: usize,
        spectrum_index: u64,
    ) -> io::Result<&mut SpectrumDataCache> {
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
            if let Some(cache) =
                SpectrumDataCache::load_data_for(self, row_group_index, spectrum_index)?
            {
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
    pub fn get_spectrum_arrays(&mut self, index: u64) -> io::Result<Option<BinaryArrayMap>> {
        let delta_model = self.metadata.model_deltas_for(index as usize);
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
            for v in self.load_auxiliary_arrays_for_spectrum(index)? {
                arrays.add(v);
            }
            return Ok(Some(arrays));
        }

        if self.query_indices.spectrum_chunk_index.is_populated() {
            log::trace!("Using chunk strategy for reading spectrum {index}");
            return SpectrumChunkReader::new(builder)
                .read_chunks_for_spectrum(
                    index,
                    &self.query_indices,
                    &self.metadata.spectrum_array_indices,
                    delta_model.as_ref(),
                    Some(PageQuery::new(row_group_indices, pages)),
                )
                .map(Some);
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
            for v in self.load_auxiliary_arrays_for_spectrum(index)? {
                out.add(v);
            }
            return Ok(Some(out));
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
            Self::populate_arrays_from_struct_array(
                points,
                &mut bin_map,
                delta_model.as_ref(),
                false,
            );
        }

        let mut out = BinaryArrayMap::new();
        for v in bin_map.into_values() {
            out.add(v);
        }

        for v in self.load_auxiliary_arrays_for_spectrum(index)? {
            out.add(v);
        }
        Ok(Some(out))
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

    fn load_spectrum_metadata_from_slice_into_description(
        &self,
        spec_arr: &StructArray,
        descr: &mut SpectrumDescription,
        offset: usize,
    ) -> usize {
        let metacols = self
            .metadata
            .spectrum_metadata_map
            .as_deref()
            .unwrap_or(&EMPTY_FIELDS);
        {
            let mut builder =
                MzSpectrumVisitor::new(std::slice::from_mut(descr), &metacols, offset);
            builder.visit(spec_arr)
        }
    }

    #[allow(unused)]
    fn load_chromatogram_metadata_from_slice_into_description(
        &self,
        chrom_arr: &StructArray,
        descr: &mut ChromatogramDescription,
        offset: usize,
    ) -> usize {
        MzChromatogramBuilder::new(
            std::slice::from_mut(descr),
            self.metadata
                .chromatogram_metadata_map
                .as_ref()
                .map(|v| v.as_slice())
                .unwrap_or_else(|| &EMPTY_FIELDS),
            offset,
        )
        .visit(chrom_arr)
    }

    pub fn get_spectrum_index_range_for_time_range(
        &self,
        time_range: SimpleInterval<f32>,
    ) -> io::Result<(
        HashMap<u64, f32, BuildIdentityHasher<u64>>,
        SimpleInterval<u64>,
    )> {
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
    ///
    /// # Arguments
    /// - `time_range`: A time interval to select spectra from.
    /// - `mz_range`: An optional m/z range to filter within.
    /// - `ion_mobility_range`: An optional ion mobility range to filter within.
    ///
    /// # Returns
    /// - An iterator over record batches covering the spectrum data: `Box<dyn Iterator<Item = Result<RecordBatch, ArrowError>> + '_>`.
    /// - A mapping from spectrum index to scan start time.
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
            let it = Box::new(SpectrumChunkReader::new(builder).scan_chunks_for(
                index_range,
                mz_range,
                &self.metadata,
                &self.query_indices,
            )?);
            let it: Box<dyn Iterator<Item = Result<RecordBatch, ArrowError>> + '_> =
                if ion_mobility_range.is_some() {
                    // If there is an ion mobility array constraint, the chunked encoding doesn't support filtering on this
                    // dimension directly.
                    if let Some(im_name) = self
                        .metadata
                        .spectrum_array_indices
                        .iter()
                        .find(|v| v.is_ion_mobility())
                    {
                        let it = it.map(move |bat| -> Result<RecordBatch, ArrowError> {
                            let bat = bat?;
                            let arr = bat
                                .column(0)
                                .as_struct()
                                .column_by_name(&im_name.name)
                                .unwrap();
                            let mask = ion_mobility_range.unwrap().contains_dy(&arr);
                            arrow::compute::filter_record_batch(&bat, &mask)
                        });
                        Box::new(it)
                    } else {
                        it
                    }
                } else {
                    it
                };
            return Ok((it, time_index));
        }

        let reader = PointDataReader(builder, BufferContext::Spectrum).query_points(
            index_range,
            mz_range,
            ion_mobility_range,
            &self.query_indices,
            &self.metadata.spectrum_array_indices,
        )?;
        Ok((reader, time_index))
    }

    /// Get the number of spectra in the archive
    pub fn len(&self) -> usize {
        self.metadata.spectrum_id_index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.metadata.spectrum_id_index.is_empty()
    }

    /// Read peak data for a spectrum.
    ///
    /// # Returns
    /// - If this mzPeak archive does not have a peak data file, this method will return an Err([`io::Error`])
    /// - If this mzPeak archive does have a peak data file, but does not have an entry for the requested
    ///   spectrum index, this method will return `Ok(None)`. There may still be peak data available in the main
    ///   spectrum data file.
    pub fn get_spectrum_peaks_for(
        &mut self,
        index: u64,
    ) -> io::Result<Option<PeakDataLevel<C, D>>> {
        let builder = self.handle.spectrum_peaks()?;
        let meta_index = self.metadata.peak_indices.as_ref().ok_or(io::Error::new(
            io::ErrorKind::NotFound,
            "peak data index was not found",
        ))?;

        return PointDataReader(builder, BufferContext::Spectrum)
            .get_peak_list_for(index, meta_index);
    }

    /// Perform slicing random access over the peak data for spectra in this file.
    ///
    /// If there are no stored peaks for a given spectrum, there will be gaps.
    ///
    /// # Arguments
    /// - `time_range`: A time interval to select spectra from.
    /// - `mz_range`: An optional m/z range to filter within.
    /// - `ion_mobility_range`: An optional ion mobility range to filter within.
    ///
    /// # Returns
    /// - If this mzPeak archive does not have a peak data file, this method will return an Err([`io::Error`])
    /// - An iterator over record batches covering the spectrum data: `Box<dyn Iterator<Item = Result<RecordBatch, ArrowError>> + '_>`.
    /// - A mapping from spectrum index to scan start time.
    pub fn query_peaks(
        &mut self,
        time_range: SimpleInterval<f32>,
        mz_range: Option<SimpleInterval<f64>>,
        ion_mobility_range: Option<SimpleInterval<f64>>,
    ) -> io::Result<(
        Box<dyn Iterator<Item = Result<RecordBatch, ArrowError>> + '_>,
        HashMap<u64, f32, BuildIdentityHasher<u64>>,
    )> {
        let builder = self.handle.spectrum_peaks()?;
        let meta_index = self.metadata.peak_indices.as_ref().ok_or(io::Error::new(
            io::ErrorKind::NotFound,
            "peak metadata was not found",
        ))?;

        let (time_index, index_range) = self.get_spectrum_index_range_for_time_range(time_range)?;

        let iter = PointDataReader(builder, BufferContext::Spectrum).query_points(
            index_range,
            mz_range,
            ion_mobility_range,
            &meta_index.query_index,
            &meta_index.array_indices,
        )?;
        Ok((iter, time_index))
    }

    /// Read load descriptive metadata for the spectrum at `index`
    pub fn get_spectrum_metadata(&mut self, index: u64) -> io::Result<Option<SpectrumDescription>> {
        if let Some(cache) = self.spectrum_metadata_cache.as_ref() {
            return Ok(cache.get(index as usize).cloned());
        }

        let builder = self.handle.spectrum_metadata()?;

        let mut rows = self
            .query_indices
            .spectrum_index_index
            .row_selection_contains(index);

        rows = rows.union(
            &self
                .query_indices
                .spectrum_scan_index
                .row_selection_contains(index),
        );
        rows = rows.union(
            &self
                .query_indices
                .spectrum_precursor_index
                .row_selection_contains(index),
        );
        rows = rows.union(
            &self
                .query_indices
                .spectrum_selected_ion_index
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

        let mut descr = SpectrumDescription::default();

        let mut precursors = Vec::new();
        let mut selected_ions = Vec::new();

        let mut k = 0;

        for batch in reader {
            let batch = match batch {
                Ok(batch) => batch,
                Err(e) => return Err(io::Error::other(e)),
            };
            let spec_arr = batch.column_by_name("spectrum").unwrap().as_struct();
            let index_arr: &UInt64Array = spec_arr
                .column_by_name("index")
                .unwrap()
                .as_any()
                .downcast_ref()
                .unwrap();

            if let Some(pos) = index_arr.iter().position(|i| i.is_some_and(|i| i == index)) {
                let spec_arr = spec_arr.slice(pos, 1);
                k += self
                    .load_spectrum_metadata_from_slice_into_description(&spec_arr, &mut descr, 0);
            }

            let scan_arr = batch.column_by_name("scan").unwrap().as_struct();
            {
                let index_arr: &UInt64Array = scan_arr
                    .column_by_name("spectrum_index")
                    .unwrap()
                    .as_any()
                    .downcast_ref()
                    .unwrap();

                let n_valid = index_arr.len() - index_arr.null_count();

                if n_valid > 0 {
                    let mut acc = Vec::new();
                    self.load_scan_events_from(scan_arr, &mut acc);
                    for (_, event) in acc {
                        descr.acquisition.scans.push(event);
                    }
                }
            }

            let precursor_arr = batch.column_by_name("precursor").unwrap().as_struct();
            {
                let mut precursors_acc = Vec::new();
                self.load_precursors_from(precursor_arr, &mut precursors_acc);
                precursors.extend(precursors_acc);
            }

            let selected_ion_arr = batch.column_by_name("selected_ion").unwrap().as_struct();
            {
                let mut acc = Vec::new();
                self.load_selected_ions_from(selected_ion_arr, &mut acc);
                selected_ions.extend(acc.into_iter().map(|(a, b, c)| (a, b, Some(c))));
            }
        }

        for (_spec_index, precursor_index, mut prec) in precursors {
            for selected_ion in selected_ions.iter_mut() {
                if selected_ion.1 != precursor_index {
                    continue;
                }

                let si = selected_ion.2.take().unwrap();
                prec.add_ion(si);
            }
            descr.precursor.push(prec);
        }

        if k > 0 { Ok(Some(descr)) } else { Ok(None) }
    }

    pub fn get_chromatogram_metadata(
        &mut self,
        index: u64,
    ) -> io::Result<Option<ChromatogramDescription>> {
        self.load_all_chromatgram_metadata_impl()
            .map(|v| v.into_iter().nth(index as usize))
    }

    pub fn get_chromatogram_arrays(&mut self, index: u64) -> io::Result<Option<BinaryArrayMap>> {
        let builder = self.handle.chromatograms_data()?;
        let reader = PointDataReader(builder, BufferContext::Chromatogram);
        let out = reader.read_points_of(
            index,
            &self.query_indices.chromatogram_point_index,
            &self.metadata.chromatogram_array_indices,
        )?;

        if let Some(mut out) = out {
            for v in self.load_auxiliary_arrays_for_chromatogram(index)? {
                out.add(v);
            }
            Ok(Some(out))
        } else {
            Ok(None)
        }
    }

    pub(crate) fn load_spectrum_auxiliary_array_count(&self) -> io::Result<Vec<u32>> {
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

        macro_rules! unpack {
            ($index_array:ident, $values_array:ident, $dtype:ty) => {
                if let Some(values) = $values_array.as_primitive_opt::<$dtype>() {
                    for (i, c) in $index_array.iter().zip(values.iter()) {
                        number_of_auxiliary_arrays[i.unwrap() as usize] =
                            c.unwrap_or_default() as u32;
                    }
                    true
                } else {
                    false
                }
            };
        }

        for batch in reader.flatten() {
            let root = batch.column(0).as_struct();
            let index_array: &UInt64Array = root.column(0).as_primitive();
            let values_array = root.column(1);

            if unpack!(index_array, values_array, UInt32Type) {
            } else if unpack!(index_array, values_array, UInt64Type) {
            } else if unpack!(index_array, values_array, Int32Type) {
            } else if unpack!(index_array, values_array, Int64Type) {
            } else {
                unimplemented!(
                    "auxiliary array count stored as {:?}",
                    values_array.data_type()
                )
            }
        }
        Ok(number_of_auxiliary_arrays)
    }

    pub(crate) fn load_chromatogram_auxiliary_array_count(&self) -> io::Result<Vec<u32>> {
        let builder = match self.handle.chromatograms_metadata() {
            Ok(builder) => builder,
            Err(e) => {
                log::trace!("{e}");
                return Ok(Vec::new());
            }
        };

        let schema = builder.parquet_schema();
        let mut index_i = None;
        let mut auxiliary_count_i = None;
        for (i, c) in schema.columns().iter().enumerate() {
            let parts = c.path().parts();
            if parts == ["chromatogram", "index"] {
                index_i = Some(i);
            }
            if parts
                .iter()
                .zip(["chromatogram", "number_of_auxiliary_arrays"])
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

        macro_rules! unpack {
            ($index_array:ident, $values_array:ident, $dtype:ty) => {
                if let Some(values) = $values_array.as_primitive_opt::<$dtype>() {
                    for (i, c) in $index_array.iter().zip(values.iter()) {
                        number_of_auxiliary_arrays[i.unwrap() as usize] =
                            c.unwrap_or_default() as u32;
                    }
                    true
                } else {
                    false
                }
            };
        }

        for batch in reader.flatten() {
            let root = batch.column(0).as_struct();
            let index_array: &UInt64Array = root.column(0).as_primitive();
            let values_array = root.column(1);

            if unpack!(index_array, values_array, UInt32Type) {
            } else if unpack!(index_array, values_array, UInt64Type) {
            } else if unpack!(index_array, values_array, Int32Type) {
            } else if unpack!(index_array, values_array, Int64Type) {
            } else {
                unimplemented!(
                    "auxiliary array count stored as {:?}",
                    values_array.data_type()
                )
            }
        }
        Ok(number_of_auxiliary_arrays)
    }

    fn load_auxiliary_arrays_from(&self, reader: ParquetRecordBatchReader) -> Vec<DataArray> {
        let mut results = Vec::new();
        for bat in reader.flatten() {
            let root = bat.column(0);
            let root = root.as_struct();
            let data = root.column(1).as_list::<i64>();
            let data = data.values().as_struct();
            let arrays = AuxiliaryArrayVisitor::default().visit(data);
            results.extend(arrays);
        }

        results
    }

    pub(crate) fn load_auxiliary_arrays_for_chromatogram(
        &self,
        index: u64,
    ) -> io::Result<Vec<DataArray>> {
        if self
            .metadata
            .chromatogram_auxiliary_array_counts()
            .get(index as usize)
            .copied()
            .unwrap_or_default()
            == 0
        {
            return Ok(Vec::new());
        }

        let builder = self.handle.chromatograms_metadata()?;
        let predicate_mask = ProjectionMask::columns(
            builder.parquet_schema(),
            ["chromatogram.index", "chromatogram.auxiliary_arrays"],
        );

        let proj = predicate_mask.clone();

        let predicate = ArrowPredicateFn::new(predicate_mask, move |batch| {
            let index_array: &UInt64Array = batch.column(0).as_struct().column(0).as_primitive();
            Ok(index_array.iter().map(|v| v.map(|i| i == index)).collect())
        });

        let filter = RowFilter::new(vec![Box::new(predicate)]);

        let reader = builder
            .with_projection(proj)
            .with_row_filter(filter)
            .build()?;

        let results = self.load_auxiliary_arrays_from(reader);
        Ok(results)
    }

    pub(crate) fn load_auxiliary_arrays_for_spectrum(
        &self,
        index: u64,
    ) -> io::Result<Vec<DataArray>> {
        if self
            .metadata
            .spectrum_auxiliary_array_counts()
            .get(index as usize)
            .copied()
            .unwrap_or_default()
            == 0
        {
            return Ok(Vec::new());
        }

        let builder = self.handle.spectrum_metadata()?;

        let rows = self
            .query_indices
            .spectrum_index_index
            .row_selection_contains(index);

        let predicate_mask = ProjectionMask::columns(
            builder.parquet_schema(),
            ["spectrum.index", "spectrum.auxiliary_arrays"],
        );

        let proj = predicate_mask.clone();

        let predicate = ArrowPredicateFn::new(predicate_mask, move |batch| {
            let spectrum_index: &UInt64Array = batch.column(0).as_struct().column(0).as_primitive();
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

        let results = self.load_auxiliary_arrays_from(reader);
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

        let reader = builder
            .with_projection(proj)
            .with_batch_size(10_000)
            .build()?;
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

    pub(crate) fn load_all_spectrum_metadata_impl(&self) -> io::Result<Vec<SpectrumDescription>> {
        log::trace!("Loading all spectrum metadata");
        let builder = self.handle.spectrum_metadata()?;

        let mut rows = self
            .query_indices
            .spectrum_index_index
            .row_selection_is_not_null();

        rows = rows.union(
            &self
                .query_indices
                .spectrum_scan_index
                .row_selection_is_not_null(),
        );
        rows = rows.union(
            &self
                .query_indices
                .spectrum_precursor_index
                .row_selection_is_not_null(),
        );
        rows = rows.union(
            &self
                .query_indices
                .spectrum_selected_ion_index
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

        let mut descriptions: Vec<SpectrumDescription> = Vec::new();
        let mut precursors: Vec<(u64, u64, Precursor)> = Vec::new();
        let mut selected_ions: Vec<(u64, u64, SelectedIon)> = Vec::new();
        let mut scan_events: Vec<(u64, ScanEvent)> = Vec::new();

        for batch in reader.flatten() {
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
                if descriptions.is_empty() {
                    descriptions = local_descr;
                } else {
                    descriptions.extend(local_descr);
                }
            }

            if let Some(scan_arr) = batch.column_by_name("scan").map(|arr| arr.as_struct()) {
                let mut acc = Vec::new();
                self.load_scan_events_from(scan_arr, &mut acc);
                if scan_events.is_empty() {
                    scan_events = acc;
                } else {
                    scan_events.extend(acc);
                }
            }

            let precursor_arr = batch.column_by_name("precursor").unwrap().as_struct();
            {
                let mut precursor_acc = Vec::new();
                self.load_precursors_from(precursor_arr, &mut precursor_acc);
                if precursors.is_empty() {
                    precursors = precursor_acc
                } else {
                    precursors.extend(precursor_acc);
                }
            }

            let selected_ion_arr = batch.column_by_name("selected_ion").unwrap().as_struct();
            {
                let mut acc = Vec::new();
                self.load_selected_ions_from(&selected_ion_arr, &mut acc);
                if selected_ions.is_empty() {
                    selected_ions = acc;
                } else {
                    selected_ions.extend(acc);
                }
            }
        }

        precursors.sort_unstable_by(|a, b| a.1.cmp(&b.1));

        for (_, prec_idx, si) in selected_ions {
            let prec = &mut precursors[prec_idx as usize].2;
            prec.add_ion(si);
        }

        for (idx, scan) in scan_events {
            descriptions[idx as usize].acquisition.scans.push(scan);
        }

        for (idx, _, precursor) in precursors {
            descriptions[idx as usize].precursor.push(precursor);
        }
        log::trace!("Finished loading all spectrum metadata");
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
        let mut precursors = Vec::new();
        let mut selected_ions = Vec::new();

        for batch in reader.flatten() {
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
            descriptions.extend(local_descr);

            let precursor_arr = batch.column_by_name("precursor").unwrap().as_struct();
            {
                let mut acc = Vec::new();
                self.load_precursors_from(precursor_arr, &mut acc);
                precursors.extend(acc);
            }

            let selected_ion_arr = batch.column_by_name("selected_ion").unwrap().as_struct();
            {
                let mut acc = Vec::new();
                self.load_selected_ions_from(selected_ion_arr, &mut acc);
                selected_ions.extend(acc);
            }
        }

        precursors.sort_unstable_by(|a, b| a.1.cmp(&b.1));

        for (spec_idx, prec_idx, si) in selected_ions {
            let prec = &mut precursors[prec_idx as usize].2;
            prec.add_ion(si);
        }

        for (idx, _prec_idx, precursor) in precursors {
            descriptions[idx as usize].precursor.push(precursor);
        }

        Ok(descriptions)
    }

    /// Retrieve the metadata for a spectrum by its `nativeId`
    pub fn get_spectrum_metadata_by_id(
        &mut self,
        id: &str,
    ) -> io::Result<Option<SpectrumDescription>> {
        if let Some(idx) = self.metadata.spectrum_id_index.get(id) {
            return self.get_spectrum_metadata(idx);
        }
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Spectrum id \"{id}\" not found"),
        ))
    }

    /// Retrieve a complete spectrum by its index
    pub fn get_spectrum(&mut self, index: usize) -> Option<MultiLayerSpectrum<C, D>> {
        let description = self
            .get_spectrum_metadata(index as u64)
            .inspect_err(|e| log::error!("Failed to read spectrum metadata for {index}: {e}"))
            .ok()??;
        let arrays = if self.detail_level == DetailLevel::Full {
            self.get_spectrum_arrays(index as u64)
                .inspect_err(|e| log::error!("Failed to read spectrum data for {index}: {e}"))
                .ok()??
        } else {
            BinaryArrayMap::new()
        };

        Some(MultiLayerSpectrum::from_arrays_and_description(
            arrays,
            description,
        ))
    }

    /// Retrieve a complete chromatogram by its index
    pub fn get_chromatogram(&mut self, index: usize) -> Option<Chromatogram> {
        let description = self
            .get_chromatogram_metadata(index as u64)
            .inspect_err(|e| log::error!("Failed to read chromatogram metadata for {index}: {e}"))
            .ok()??;
        let arrays = if self.detail_level == DetailLevel::Full {
            self.get_chromatogram_arrays(index as u64)
                .inspect_err(|e| log::error!("Failed to read chromatogram data for {index}: {e}"))
                .ok()??
        } else {
            BinaryArrayMap::new()
        };

        Some(Chromatogram::new(description, arrays))
    }

    pub fn get_chromatogram_by_id(&mut self, id: &str) -> Option<Chromatogram> {
        if let Some(description) = self
            .load_all_chromatgram_metadata_impl()
            .ok()?
            .into_iter()
            .find(|v| v.id == id)
        {
            let arrays = if self.detail_level == DetailLevel::Full {
                self.get_chromatogram_arrays(description.index as u64)
                    .inspect_err(|e| log::error!("Failed to read chromatogram data for {id}: {e}"))
                    .ok()??
            } else {
                BinaryArrayMap::new()
            };

            Some(Chromatogram::new(description, arrays))
        } else {
            None
        }
    }

    /// Read the total ion chromatogram from the surrogate metadata in the spectrum table
    pub fn encoded_tic(&mut self) -> io::Result<Chromatogram> {
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
    pub fn encoded_bpc(&mut self) -> io::Result<Chromatogram> {
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

pub type MzPeakReaderType<C, D> = MzPeakReaderTypeOfSource<ZipArchiveSource, C, D>;
pub type UnpackedMzPeakReaderType<C, D> = MzPeakReaderTypeOfSource<DirectorySource, C, D>;

pub type MzPeakReader = MzPeakReaderTypeOfSource<ZipArchiveSource, CentroidPeak, DeconvolutedPeak>;
pub type UnpackedMzPeakReader =
    MzPeakReaderTypeOfSource<DirectorySource, CentroidPeak, DeconvolutedPeak>;

#[cfg(test)]
mod test {
    use super::*;
    use mzdata::spectrum::{ChromatogramLike, SignalContinuity};

    #[test_log::test]
    fn test_read_spectrum() -> io::Result<()> {
        let mut reader = MzPeakReader::new("small.mzpeak")?;
        let descr = reader.get_spectrum(0).unwrap();
        assert_eq!(descr.index(), 0);
        assert_eq!(descr.signal_continuity(), SignalContinuity::Profile);
        assert_eq!(descr.peaks().len(), 13589);
        let descr = reader.get_spectrum(5).unwrap();
        assert_eq!(descr.index(), 5);
        assert_eq!(descr.peaks().len(), 650);
        let descr = reader.get_spectrum(25).unwrap();
        assert_eq!(descr.index(), 25);
        assert_eq!(descr.peaks().len(), 789);
        Ok(())
    }

    #[test_log::test]
    fn test_read_spectrum_chunked() -> io::Result<()> {
        let mut reader = MzPeakReader::new("small.chunked.mzpeak")?;
        let descr = reader.get_spectrum(0).unwrap();
        assert_eq!(descr.index(), 0);
        assert_eq!(descr.signal_continuity(), SignalContinuity::Profile);
        assert_eq!(descr.peaks().len(), 13589);
        let descr = reader.get_spectrum(5).unwrap();
        assert_eq!(descr.index(), 5);
        assert_eq!(descr.peaks().len(), 650);
        eprintln!("{:#?}", descr.description());
        let descr = reader.get_spectrum(25).unwrap();
        assert_eq!(descr.index(), 25);
        assert_eq!(descr.peaks().len(), 789);
        Ok(())
    }

    #[test_log::test]
    fn test_tic() -> io::Result<()> {
        let mut reader = MzPeakReader::new("small.mzpeak")?;
        let tic = reader.encoded_tic()?;
        assert_eq!(tic.index(), 0);
        assert_eq!(tic.time()?.len(), 48);
        Ok(())
    }

    #[test_log::test]
    fn test_read_chromatogram() -> io::Result<()> {
        let mut reader = MzPeakReader::new("small.mzpeak").unwrap();
        let tic = reader.get_chromatogram(0).unwrap();
        eprintln!("{tic:?}");
        assert_eq!(tic.index(), 0);
        assert_eq!(tic.time()?.len(), 48);
        Ok(())
    }

    #[test_log::test]
    fn test_load_all_metadata() -> io::Result<()> {
        let reader = MzPeakReader::new("small.mzpeak")?;
        let out = reader.load_all_spectrum_metadata_impl()?;
        assert_eq!(out.len(), 48);
        Ok(())
    }

    #[test_log::test]
    fn test_load_all_chromatogram_metadata() -> io::Result<()> {
        let reader = MzPeakReader::new("small.mzpeak")?;
        let out = reader.load_all_chromatgram_metadata_impl()?;
        eprintln!("{out:?}");
        Ok(())
    }

    #[test_log::test]
    fn test_eic() -> io::Result<()> {
        let mut reader = MzPeakReader::new("small.mzpeak")?;

        let (it, _time_index) =
            reader.extract_peaks((0.3..0.4).into(), Some((800.0..820.0).into()), None)?;

        for batch in it.flatten() {
            eprintln!("{:?}", batch);
        }
        Ok(())
    }

    #[test_log::test]
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
