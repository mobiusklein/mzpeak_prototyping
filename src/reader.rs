use std::{
    collections::HashMap,
    io,
    marker::PhantomData,
    path::{Path, PathBuf},
};

use arrow::{
    array::{Array, AsArray, Float32Array, RecordBatch, UInt64Array},
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
        MultiLayerSpectrum, PeakDataLevel, SpectrumDescription, bindata::BuildFromArrayMap,
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

use crate::{archive::DispatchArchiveSource, reader::utils::MaskSet};
#[allow(unused_imports)]
use crate::{
    archive::{ArchiveReader, ArchiveSource, DirectorySource, ZipArchiveSource, SplittingZipArchiveSource}, filter::RegressionDeltaModel, reader::{
        chunk::{DataChunkCache, SpectrumChunkReader},
        index::{PageQuery, QueryIndex, SpanDynNumeric, SpectrumQueryIndex},
        metadata::{
            ChromatogramMetadataDecoder, ChromatogramMetadataQuerySource,
            ChromatogramMetadataReader, SpectrumMetadataDecoder, SpectrumMetadataQuerySource,
            SpectrumMetadataReader, TimeIndexDecoder,
        },
        point::PointDataReader,
        visitor::AuxiliaryArrayVisitor,
    }, BufferContext
};

mod chunk;
mod metadata;
mod point;
pub(crate) mod utils;

pub mod index;
pub mod visitor;

#[cfg(feature = "async")]
mod object_store_async;

pub use metadata::ReaderMetadata;
use point::{DataPointCache, PointDataArrayReader};

/// A reader for mzPeak files, abstract over the source type.
pub struct MzPeakReaderTypeOfSource<
    T: ArchiveSource = SplittingZipArchiveSource,
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
    ///
    pub fn get_spectrum_arrays(&mut self, index: u64) -> io::Result<Option<BinaryArrayMap>> {
        let delta_model = self.metadata.model_deltas_for(index as usize);
        let builder = self.handle.spectra_data()?;

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

        let reader = PointDataReader(builder, BufferContext::Spectrum);
        if let Some(mut out) = reader.read_points_of(
            index,
            &self.query_indices.spectrum_point_index,
            self.metadata.spectrum_array_indices(),
            delta_model.as_ref(),
        )? {
            for v in self.load_auxiliary_arrays_for_spectrum(index)? {
                out.add(v);
            }
            return Ok(Some(out));
        } else {
            if let Ok(arrays) = self.load_auxiliary_arrays_for_spectrum(index) {
                let mut out = BinaryArrayMap::new();
                for arr in arrays {
                    out.add(arr);
                }
                return Ok(Some(out));
            } else {
                return Ok(None);
            }
        }
    }

    pub fn get_spectrum_index_range_for_time_range(
        &self,
        time_range: SimpleInterval<f32>,
        ms_level_range: Option<SimpleInterval<u8>>,
    ) -> io::Result<(
        HashMap<u64, f32, BuildIdentityHasher<u64>>,
        MaskSet,
    )> {
        let mut time_indexer = TimeIndexDecoder::new(time_range, ms_level_range);
        if let Some(cache) = self.spectrum_metadata_cache.as_ref() {
            time_indexer.from_descriptions(cache.as_slice());
            return Ok(time_indexer.finish());
        }

        let rows = self
            .query_indices
            .spectrum_time_index
            .row_selection_overlaps(&time_range);

        let builder = self.handle.spectrum_metadata()?;

        let has_ms_level_range = ms_level_range.is_some();
        let ms_level_range = ms_level_range.unwrap_or_default();
        let columns_for_predicate: &[&str] = if has_ms_level_range {
            &["spectrum.time", "spectrum.ms_level"]
        } else {
            &["spectrum.time"]
        };

        let predicate_mask = ProjectionMask::columns(builder.parquet_schema(), columns_for_predicate.into_iter().copied());

        let predicate = ArrowPredicateFn::new(predicate_mask, move |batch| {
            let times = batch.column(0).as_struct().column_by_name("time").unwrap();
            if has_ms_level_range {
                let ms_levels = batch.column(0).as_struct().column_by_name("ms_level").unwrap();
                arrow::compute::and(
                    &time_range.contains_dy(times),
                    &ms_level_range.contains_dy(ms_levels)
                )
            } else {
                Ok(time_range.contains_dy(times))
            }
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

        for batch in reader.flatten() {
            time_indexer.decode_batch(batch)?;
        }

        Ok(time_indexer.finish())
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
        ms_level_range: Option<SimpleInterval<u8>>,
    ) -> io::Result<(
        Box<dyn Iterator<Item = Result<RecordBatch, ArrowError>> + '_>,
        HashMap<u64, f32, BuildIdentityHasher<u64>>,
    )> {
        let (time_index, index_range) = self.get_spectrum_index_range_for_time_range(time_range, ms_level_range)?;
        let builder = self.handle.spectra_data()?;

        let ion_mobility_range = if !self.metadata.spectrum_array_indices().has_ion_mobility() {
            None
        } else {
            ion_mobility_range
        };

        if self.query_indices.spectrum_chunk_index.is_populated() {
            let reader = SpectrumChunkReader::new(builder);

            let query = self.query_indices
                .spectrum_chunk_index
                .query_pages_overlaps(&index_range);

            let it: Box<dyn Iterator<Item = Result<RecordBatch, ArrowError>> + '_> = if query.can_split() && self.handle.can_split() {
                let mut index_range1 = index_range.clone();
                if let Some(index_range2) = index_range1.split() {
                    log::trace!("Splitting chunk query");
                    let builder2 = self.handle.spectra_data()?;
                    let reader2 = SpectrumChunkReader::new(builder2);
                    std::thread::scope(|ctx| -> io::Result<_> {
                        let handle = ctx.spawn(|| {
                            reader.scan_chunks_for(
                                index_range1,
                                mz_range,
                                &self.metadata,
                                &self.query_indices,
                            )
                        });
                        let handle2 = ctx.spawn(|| {
                            reader2.scan_chunks_for(
                                index_range2,
                                mz_range,
                                &self.metadata,
                                &self.query_indices,
                            )
                        });
                        let reader = handle.join().unwrap()?;
                        let reader2 = handle2.join().unwrap()?;
                        Ok(Box::new(reader.chain(reader2)))
                    })?
                } else {
                    Box::new(reader.scan_chunks_for(
                        index_range,
                        mz_range,
                        &self.metadata,
                        &self.query_indices,
                    )?)
                }
            } else {
                Box::new(reader.scan_chunks_for(
                    index_range,
                    mz_range,
                    &self.metadata,
                    &self.query_indices,
                )?)
            };

            let it: Box<dyn Iterator<Item = Result<RecordBatch, ArrowError>> + '_> =
                if let Some(ion_mobility_range) = ion_mobility_range {
                    // If there is an ion mobility array constraint, the chunked encoding doesn't support filtering on this
                    // dimension directly.
                    if let Some(im_name) = self
                        .metadata
                        .spectrum_array_indices
                        .iter()
                        .find(|v| v.is_ion_mobility())
                    {
                        chunk::make_ion_mobility_filter(it, ion_mobility_range, im_name)
                    } else {
                        it
                    }
                } else {
                    it
                };
            return Ok((it, time_index));
        }

        let reader = PointDataReader(builder, BufferContext::Spectrum);

        let query = self.query_indices.query_pages_overlaps(&index_range);

        if query.can_split() && self.handle.can_split() {
            let mut index_range1 = index_range.clone();
            if let Some(index_range2) = index_range1.split() {
                log::trace!("Splitting point query");
                {
                    let builder2 = self.handle.spectra_data()?;
                    let reader2 = PointDataReader(builder2, BufferContext::Spectrum);

                    let reader = std::thread::scope(|ctx| -> io::Result<_> {
                        let handle = ctx.spawn(|| {
                            reader.query_points(
                                index_range1,
                                mz_range,
                                ion_mobility_range,
                                &self.query_indices,
                                &self.metadata.spectrum_array_indices,
                                &self.metadata,
                                None,
                            )
                        });
                        let handle2 = ctx.spawn(|| {
                            reader2.query_points(
                                index_range2,
                                mz_range,
                                ion_mobility_range,
                                &self.query_indices,
                                &self.metadata.spectrum_array_indices,
                                &self.metadata,
                                None,
                            )
                        });
                        let reader = handle.join().unwrap()?;
                        let reader2 = handle2.join().unwrap()?;
                        Ok(Box::new(reader.chain(reader2)))
                    });

                    return Ok((reader?, time_index))
                }

            }
        }
        let reader = reader.query_points(
            index_range,
            mz_range,
            ion_mobility_range,
            &self.query_indices,
            &self.metadata.spectrum_array_indices,
            &self.metadata,
            Some(query),
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
        ms_level_range: Option<SimpleInterval<u8>>,
    ) -> io::Result<(
        Box<dyn Iterator<Item = Result<RecordBatch, ArrowError>> + '_>,
        HashMap<u64, f32, BuildIdentityHasher<u64>>,
    )> {
        let builder = self.handle.spectrum_peaks()?;
        let meta_index = self.metadata.peak_indices.as_ref().ok_or(io::Error::new(
            io::ErrorKind::NotFound,
            "peak metadata was not found",
        ))?;

        let ion_mobility_range = if !meta_index.array_indices.has_ion_mobility() {
            None
        } else {
            ion_mobility_range
        };

        let (time_index, index_range) = self.get_spectrum_index_range_for_time_range(time_range, ms_level_range)?;

        let iter = PointDataReader(builder, BufferContext::Spectrum).query_points(
            index_range,
            mz_range,
            ion_mobility_range,
            &meta_index.query_index,
            &meta_index.array_indices,
            &self.metadata,
            None,
        )?;
        Ok((iter, time_index))
    }

    /// Read load descriptive metadata for the spectrum at `index`
    pub fn get_spectrum_metadata(&mut self, index: u64) -> io::Result<Option<SpectrumDescription>> {
        if let Some(cache) = self.spectrum_metadata_cache.as_ref() {
            return Ok(cache.get(index as usize).cloned());
        }

        let builder = SpectrumMetadataReader(self.handle.spectrum_metadata()?);

        let rows = builder.prepare_rows_for(index, &self.query_indices);
        let predicate = builder.prepare_predicate_for(index);

        let reader = builder
            .0
            .with_row_selection(rows)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .build()?;

        let mut decoder = SpectrumMetadataDecoder::new(&self.metadata);

        for batch in reader {
            let batch = match batch {
                Ok(batch) => batch,
                Err(e) => return Err(io::Error::other(e)),
            };
            decoder.decode_batch(batch);
        }

        let descriptions = decoder.finish();
        Ok(descriptions.into_iter().find(|v| v.index as u64 == index))
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
            None,
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

        let builder = SpectrumMetadataReader(builder);

        let rows = builder.prepare_rows_for_all(&self.query_indices);
        let predicate = builder.prepare_predicate_for_all();

        let reader = builder
            .0
            .with_row_selection(rows)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .with_batch_size(10_000)
            .build()?;

        let mut decoder = SpectrumMetadataDecoder::new(&self.metadata);

        for batch in reader.flatten() {
            decoder.decode_batch(batch);
        }

        let descriptions = decoder.finish();
        log::trace!("Finished loading all spectrum metadata");
        Ok(descriptions)
    }

    pub(crate) fn load_all_chromatgram_metadata_impl(
        &self,
    ) -> io::Result<Vec<ChromatogramDescription>> {
        let builder = ChromatogramMetadataReader(self.handle.chromatograms_metadata()?);

        let predicate = builder.prepare_predicate_for_all();

        let reader = builder
            .0
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .build()?;

        let mut decoder = ChromatogramMetadataDecoder::new(&self.metadata);

        for batch in reader.flatten() {
            decoder.decode_batch(batch);
        }

        Ok(decoder.finish())
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

pub type MzPeakReaderType<C, D> = MzPeakReaderTypeOfSource<DispatchArchiveSource, C, D>;
pub type UnpackedMzPeakReaderType<C, D> = MzPeakReaderTypeOfSource<DirectorySource, C, D>;

pub type MzPeakReader = MzPeakReaderTypeOfSource<DispatchArchiveSource, CentroidPeak, DeconvolutedPeak>;
pub type UnpackedMzPeakReader =
    MzPeakReaderTypeOfSource<DirectorySource, CentroidPeak, DeconvolutedPeak>;

#[cfg(feature = "async")]
pub use object_store_async::{AsyncMzPeakReader, AsyncMzPeakReaderType};

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
            reader.extract_peaks((0.3..0.4).into(), Some((800.0..820.0).into()), None, None)?;

        for batch in it.flatten() {
            eprintln!("{:?}", batch);
        }
        Ok(())
    }

    #[test_log::test]
    fn test_eic_chunked() -> io::Result<()> {
        let mut reader = MzPeakReader::new("small.chunked.mzpeak")?;

        let (it, _time_index) =
            reader.extract_peaks((0.3..0.4).into(), Some((800.0..820.0).into()), None, None)?;

        for batch in it.flatten() {
            eprintln!("{:?}", batch);
        }
        Ok(())
    }
}
