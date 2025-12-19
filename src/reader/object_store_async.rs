use std::{collections::HashMap, io, marker::PhantomData, sync::Arc};

use arrow::{
    array::{Array, AsArray, RecordBatch, UInt64Array},
    error::ArrowError,
};
use futures::{StreamExt, stream::BoxStream};
use identity_hash::BuildIdentityHasher;
use object_store::{ObjectStore, path::Path as ObjectPath};

use mzdata::{
    io::{AsyncRandomAccessSpectrumIterator, AsyncSpectrumSource, DetailLevel, OffsetIndex},
    meta::MSDataFileMetadata,
    prelude::*,
    spectrum::{
        BinaryArrayMap, ChromatogramDescription, DataArray, MultiLayerSpectrum, PeakDataLevel,
        SpectrumDescription, bindata::BuildFromArrayMap,
    },
};

use mzpeaks::{
    CentroidPeak, DeconvolutedCentroidLike, DeconvolutedPeak, coordinate::SimpleInterval,
    prelude::Span1D,
};

use parquet::{
    arrow::{
        ParquetRecordBatchStreamBuilder, ProjectionMask,
        arrow_reader::{ArrowPredicateFn, RowFilter},
        async_reader::{AsyncFileReader, ParquetRecordBatchStream},
    },
    file::metadata::ParquetMetaData,
    schema::types::SchemaDescriptor,
};
use url::Url;

use crate::{
    BufferContext,
    archive::{AsyncArchiveReader, AsyncArchiveSource, AsyncZipArchiveSource},
    filter::RegressionDeltaModel,
    reader::{
        CHUNK_CACHE_BLOCK_SIZE, ReaderMetadata,
        chunk::{AsyncSpectrumChunkReader, DataChunkCache},
        index::{PageQuery, QueryIndex, SpanDynNumeric},
        metadata::{
            AuxiliaryArrayCountDecoder, BaseMetadataQuerySource, ChromatogramMetadataDecoder,
            ChromatogramMetadataQuerySource, DeltaModelDecoder, ParquetIndexExtractor,
            SpectrumMetadataDecoder, SpectrumMetadataQuerySource, TimeIndexDecoder,
        },
        point::{AsyncPointDataReader, DataPointCache, PointDataArrayReader},
        utils::MaskSet,
        visitor::AuxiliaryArrayVisitor,
    },
};

pub(crate) struct SpectrumMetadataReader<T: AsyncFileReader + 'static + Unpin + Send>(
    pub(crate) ParquetRecordBatchStreamBuilder<T>,
);

impl<T: AsyncFileReader + 'static + Unpin + Send> BaseMetadataQuerySource
    for SpectrumMetadataReader<T>
{
    fn metadata(&self) -> &ParquetMetaData {
        self.0.metadata()
    }
}

impl<T: AsyncFileReader + 'static + Unpin + Send> SpectrumMetadataQuerySource
    for SpectrumMetadataReader<T>
{
}

pub(crate) struct ChromatogramMetadataReader<T: AsyncFileReader + 'static + Unpin + Send>(
    pub(crate) ParquetRecordBatchStreamBuilder<T>,
);

impl<T: AsyncFileReader + 'static + Unpin + Send> ChromatogramMetadataQuerySource
    for ChromatogramMetadataReader<T>
{
}

impl<T: AsyncFileReader + 'static + Unpin + Send> BaseMetadataQuerySource
    for ChromatogramMetadataReader<T>
{
    fn metadata(&self) -> &ParquetMetaData {
        self.0.metadata()
    }
}

pub(crate) async fn build_spectrum_index<T: AsyncArchiveSource>(
    handle: &AsyncArchiveReader<T>,
    pq_schema: &SchemaDescriptor,
) -> io::Result<OffsetIndex> {
    let mut spectrum_id_index = OffsetIndex::new("spectrum".into());

    let mut stream = handle
        .spectrum_metadata()
        .await?
        .with_projection(ProjectionMask::columns(
            pq_schema,
            ["spectrum.id", "spectrum.index"],
        ))
        .build()?;

    while let Some(batch) = stream.next().await.transpose()? {
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
pub(crate) async fn load_indices_from<T: AsyncArchiveSource>(
    handle: &AsyncArchiveReader<T>,
) -> io::Result<(ReaderMetadata, QueryIndex)> {
    let spectrum_metadata_reader = handle.spectrum_metadata().await?;
    let spectrum_data_reader = handle.spectra_data().await?;

    let pq_schema = spectrum_metadata_reader.parquet_schema();
    let spectrum_id_index = build_spectrum_index(handle, pq_schema).await?;

    let mut this = ParquetIndexExtractor::default();
    this.visit_spectrum_metadata_reader(spectrum_metadata_reader)?;
    this.visit_spectrum_data_reader(spectrum_data_reader)?;

    if let Ok(chromatogram_metadata_reader) = handle.chromatograms_metadata().await {
        this.visit_chromatogram_metadata_reader(chromatogram_metadata_reader)?;
    }
    if let Ok(chromatogram_data_reader) = handle.chromatograms_data().await {
        this.visit_chromatogram_data_reader(chromatogram_data_reader)?;
    }

    handle
        .spectrum_peaks()
        .await
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

pub(crate) enum AsyncSpectrumDataCache {
    Point(DataPointCache),
    Chunk(DataChunkCache),
}

impl AsyncSpectrumDataCache {
    pub fn slice_to_arrays_of(
        &mut self,
        row_group_index: usize,
        spectrum_index: u64,
        mz_delta_model: Option<&RegressionDeltaModel<f64>>,
    ) -> io::Result<BinaryArrayMap> {
        if self.contains(row_group_index, spectrum_index) {
            match self {
                Self::Point(spectrum_data_point_cache) => {
                    spectrum_data_point_cache.slice_to_arrays_of(spectrum_index, mz_delta_model)
                }
                Self::Chunk(spectrum_data_chunk_cache) => {
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
            Self::Point(spectrum_data_point_cache) => {
                spectrum_data_point_cache.row_group_index == row_group_index
            }
            Self::Chunk(spectrum_data_chunk_cache) => spectrum_data_chunk_cache
                .spectrum_index_range
                .contains(&spectrum_index),
        }
    }

    pub async fn load_data_for<
        T: AsyncArchiveSource + Sync + Send,
        C: CentroidLike + BuildFromArrayMap + BuildArrayMapFrom + Sync + Send,
        D: DeconvolutedCentroidLike + BuildFromArrayMap + BuildArrayMapFrom + Sync + Send,
    >(
        reader: &AsyncMzPeakReaderType<T, C, D>,
        row_group_index: usize,
        spectrum_index: u64,
    ) -> io::Result<Option<Self>> {
        if reader.query_indices.spectrum_point_index.is_populated() {
            let rg = reader
                .load_cache_block_async(reader.handle.spectra_data().await?, row_group_index)
                .await?;
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
            let builder = reader.handle.spectra_data().await?;
            let builder = AsyncSpectrumChunkReader::new(builder);
            let cache = builder
                .load_cache_block(
                    SimpleInterval::new(spectrum_index, spectrum_index + CHUNK_CACHE_BLOCK_SIZE),
                    &reader.metadata,
                    &reader.query_indices.spectrum_chunk_index,
                )
                .await?;
            Ok(Some(Self::Chunk(cache)))
        } else {
            Ok(None)
        }
    }
}

/// A reader for mzPeak files, abstract over the source type.
pub struct AsyncMzPeakReaderType<
    T: AsyncArchiveSource + Send + Sync = AsyncZipArchiveSource,
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap + Send + Sync = CentroidPeak,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap + Send + Sync = DeconvolutedPeak,
> {
    url: Option<url::Url>,
    handle: AsyncArchiveReader<T>,
    index: usize,
    detail_level: DetailLevel,
    pub metadata: Arc<ReaderMetadata>,
    pub query_indices: Arc<QueryIndex>,
    spectrum_metadata_cache: Option<Arc<Vec<SpectrumDescription>>>,
    spectrum_row_group_cache: Option<AsyncSpectrumDataCache>,
    _t: PhantomData<(C, D)>,
}

impl<
    T: AsyncArchiveSource + Send + Sync,
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap + Send + Sync,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap + Send + Sync,
> PointDataArrayReader for AsyncMzPeakReaderType<T, C, D>
{
}

impl<
    T: AsyncArchiveSource + Send + Sync,
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap + Send + Sync,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap + Send + Sync,
> MSDataFileMetadata for AsyncMzPeakReaderType<T, C, D>
{
    mzdata::delegate_impl_metadata_trait!(metadata);
}

impl<
    T: AsyncArchiveSource + Send + Sync,
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap + Send + Sync,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap + Send + Sync,
> AsyncRandomAccessSpectrumIterator<C, D, MultiLayerSpectrum<C, D>>
    for AsyncMzPeakReaderType<T, C, D>
{
    async fn start_from_id(
        &mut self,
        id: &str,
    ) -> Result<&mut Self, SpectrumAccessError> {
        if let Some(idx) = self.metadata.spectrum_id_index.get(id) {
            self.index = idx as usize;
            Ok(self)
        } else {
            Err(SpectrumAccessError::SpectrumIdNotFound(id.to_string()))
        }
    }

    async fn start_from_index(
        &mut self,
        index: usize,
    ) -> Result<&mut Self, SpectrumAccessError> {
        if index < self.len() {
            self.index = index;
            Ok(self)
        } else {
            Err(SpectrumAccessError::SpectrumIndexNotFound(index))
        }
    }

    async fn start_from_time(
        &mut self,
        time: f64,
    ) -> Result<&mut Self, SpectrumAccessError> {
        match self.get_spectrum_by_time(time).await {
            Some(s) => {
                self.index = s.index();
                Ok(self)
            }
            None => Err(SpectrumAccessError::SpectrumNotFound),
        }
    }
}

impl<
    T: AsyncArchiveSource + Send + Sync,
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap + Send + Sync,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap + Send + Sync,
> AsyncSpectrumSource<C, D, MultiLayerSpectrum<C, D>> for AsyncMzPeakReaderType<T, C, D>
{
    async fn reset(&mut self) {
        self.index = 0;
    }

    fn detail_level(&self) -> &DetailLevel {
        &self.detail_level
    }

    fn set_detail_level(&mut self, detail_level: DetailLevel) {
        self.detail_level = detail_level
    }

    async fn get_spectrum_by_id(
        &mut self,
        id: &str,
    ) -> Option<MultiLayerSpectrum<C, D>> {
        let index = self.metadata.spectrum_id_index.get(id)?;
        self.get_spectrum(index as usize).await
    }

    async fn get_spectrum_by_index(
        &mut self,
        index: usize,
    ) -> Option<MultiLayerSpectrum<C, D>> {
        self.get_spectrum(index).await
    }

    fn get_index(&self) -> &OffsetIndex {
        &self.metadata.spectrum_id_index
    }

    fn set_index(&mut self, index: OffsetIndex) {
        let mut meta = (*self.metadata).clone();
        meta.spectrum_id_index = index;
        self.metadata = Arc::new(meta);
    }

    async fn read_next(&mut self) -> Option<MultiLayerSpectrum<C, D>> {
        if self.spectrum_metadata_cache.is_none() {
            self.load_all_spectrum_metadata().await.ok();
        }
        let spec = self.get_spectrum(self.index).await;
        self.index += 1;
        spec
    }
}

impl<
    T: AsyncArchiveSource + Send + Sync,
    C: CentroidLike + BuildArrayMapFrom + BuildFromArrayMap + Send + Sync,
    D: DeconvolutedCentroidLike + BuildArrayMapFrom + BuildFromArrayMap + Send + Sync,
> AsyncMzPeakReaderType<T, C, D>
{
    async fn init_from_store(handle: AsyncArchiveReader<T>, url: Option<Url>) -> io::Result<Self> {
        let (metadata, query_indices) = load_indices_from(&handle).await?;
        let mut this = Self {
            url,
            index: 0,
            detail_level: DetailLevel::Full,
            handle,
            metadata: Arc::new(metadata),
            query_indices: Arc::new(query_indices),
            spectrum_metadata_cache: None,
            spectrum_row_group_cache: None,
            _t: PhantomData,
        };

        let mz_model_deltas = this.load_delta_models().await?;
        let spectrum_auxiliary_array_counts = this.load_spectrum_auxiliary_array_count().await?;
        let chromatogram_auxiliary_array_counts =
            this.load_chromatogram_auxiliary_array_count().await?;

        let meta = Arc::get_mut(&mut this.metadata).unwrap();
        meta.mz_model_deltas = mz_model_deltas;
        meta.spectrum_auxiliary_array_counts = spectrum_auxiliary_array_counts;
        meta.chromatogram_auxiliary_array_counts = chromatogram_auxiliary_array_counts;

        Ok(this)
    }

    pub async fn from_store_path(
        handle: Arc<dyn ObjectStore>,
        path: ObjectPath,
    ) -> io::Result<Self> {
        let handle = AsyncArchiveReader::from_store_path(handle, path).await?;
        Self::init_from_store(handle, None).await
    }

    pub async fn from_url(url: Url) -> io::Result<Self> {
        let handle = AsyncArchiveReader::<T>::from_url(&url).await?;
        Self::init_from_store(handle, Some(url)).await
    }

    /// Get the number of spectra in the archive
    pub fn len(&self) -> usize {
        self.metadata.spectrum_id_index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.metadata.spectrum_id_index.is_empty()
    }

    pub fn url(&self) -> Option<&Url> {
        self.url.as_ref()
    }

    /// Load the descriptive metadata for all spectra
    ///
    /// This method caches the data after its first use.
    pub async fn load_all_spectrum_metadata(
        &mut self,
    ) -> io::Result<Option<Arc<Vec<SpectrumDescription>>>> {
        if self.spectrum_metadata_cache.is_none() {
            self.spectrum_metadata_cache = Some(Arc::new(
                self.load_all_spectrum_metadata_impl()
                    .await
                    .inspect_err(|e| log::error!("Failed to load spectrum metadata cache: {e}"))?,
            ));
        }
        Ok(self.spectrum_metadata_cache.clone())
    }

    /// Load the [`AsyncSpectrumDataCache`] row group or retrieve the current cache if it matches the request
    async fn read_spectrum_data_cache(
        &mut self,
        row_group_index: usize,
        spectrum_index: u64,
    ) -> io::Result<&mut AsyncSpectrumDataCache> {
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
                AsyncSpectrumDataCache::load_data_for(self, row_group_index, spectrum_index).await?
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

    /// Read load descriptive metadata for the spectrum at `index`
    pub async fn get_spectrum_metadata(
        &self,
        index: u64,
    ) -> io::Result<Option<SpectrumDescription>> {
        if let Some(cache) = self.spectrum_metadata_cache.as_ref() {
            return Ok(cache.get(index as usize).cloned());
        }

        let builder = SpectrumMetadataReader(self.handle.spectrum_metadata().await?);

        let rows = builder.prepare_rows_for(index, &self.query_indices);
        let predicate = builder.prepare_predicate_for(index);

        let mut reader = builder
            .0
            .with_row_selection(rows)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .build()?;

        let mut decoder = SpectrumMetadataDecoder::new(&self.metadata);

        while let Some(batch) = reader.next().await.transpose()? {
            decoder.decode_batch(batch);
        }

        let descriptions = decoder.finish();
        Ok(descriptions.into_iter().find(|v| v.index as u64 == index))
    }

    /// Retrieve the metadata for a spectrum by its `nativeId`
    pub async fn get_spectrum_metadata_by_id(
        &self,
        id: &str,
    ) -> io::Result<Option<SpectrumDescription>> {
        if let Some(idx) = self.metadata.spectrum_id_index.get(id) {
            return self.get_spectrum_metadata(idx).await;
        }
        Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("Spectrum id \"{id}\" not found"),
        ))
    }

    /// Retrieve a complete spectrum by its index
    pub async fn get_spectrum(&mut self, index: usize) -> Option<MultiLayerSpectrum<C, D>> {
        let description = self
            .get_spectrum_metadata(index as u64)
            .await
            .inspect_err(|e| log::error!("Failed to read spectrum metadata for {index}: {e}"))
            .ok()??;
        let arrays = if self.detail_level == DetailLevel::Full {
            self.get_spectrum_arrays(index as u64)
                .await
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

    /// Read peak data for a spectrum.
    ///
    /// # Returns
    /// - If this mzPeak archive does not have a peak data file, this method will return an Err([`io::Error`])
    /// - If this mzPeak archive does have a peak data file, but does not have an entry for the requested
    ///   spectrum index, this method will return `Ok(None)`. There may still be peak data available in the main
    ///   spectrum data file.
    pub async fn get_spectrum_peaks_for(
        &mut self,
        index: u64,
    ) -> io::Result<Option<PeakDataLevel<C, D>>> {
        let builder = self.handle.spectrum_peaks().await?;
        let meta_index = self.metadata.peak_indices.as_ref().ok_or(io::Error::new(
            io::ErrorKind::NotFound,
            "peak data index was not found",
        ))?;

        return AsyncPointDataReader(builder, BufferContext::Spectrum)
            .get_peak_list_for(index, meta_index)
            .await;
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
    pub async fn extract_peaks(
        &mut self,
        time_range: SimpleInterval<f32>,
        mz_range: Option<SimpleInterval<f64>>,
        ion_mobility_range: Option<SimpleInterval<f64>>,
        ms_level_range: Option<SimpleInterval<u8>>,
    ) -> io::Result<(
        BoxStream<'_, Result<RecordBatch, ArrowError>>,
        HashMap<u64, f32, BuildIdentityHasher<u64>>,
    )> {
        let (time_index, index_range) = self
            .get_spectrum_index_range_for_time_range(time_range, ms_level_range)
            .await?;
        let builder = self.handle.spectra_data().await?;

        let ion_mobility_range = if !self.metadata.spectrum_array_indices().has_ion_mobility() {
            None
        } else {
            ion_mobility_range
        };

        if self.query_indices.spectrum_chunk_index.is_populated() {
            let it = AsyncSpectrumChunkReader::new(builder).scan_chunks_for(
                index_range,
                mz_range,
                &self.metadata,
                self.metadata.spectrum_array_indices(),
                &self.query_indices.spectrum_chunk_index,
            )?;
            let it: BoxStream<'_, Result<RecordBatch, ArrowError>> = if ion_mobility_range.is_some()
            {
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
                        let mask = ion_mobility_range.unwrap().contains_dy(arr);
                        arrow::compute::filter_record_batch(&bat, &mask)
                    });
                    it.boxed()
                } else {
                    log::warn!(
                        "An ion mobility range was requested, but no ion mobility array was found"
                    );
                    it.boxed()
                }
            } else {
                it.boxed()
            };
            return Ok((it, time_index));
        }

        let reader = AsyncPointDataReader(builder, BufferContext::Spectrum)
            .query_points(
                index_range,
                mz_range,
                ion_mobility_range,
                &self.query_indices.spectrum_point_index,
                &self.metadata.spectrum_array_indices,
                &self.metadata,
            )
            .await?
            .boxed();
        Ok((reader, time_index))
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
    pub async fn query_peaks(
        &mut self,
        time_range: SimpleInterval<f32>,
        mz_range: Option<SimpleInterval<f64>>,
        ion_mobility_range: Option<SimpleInterval<f64>>,
        ms_level_range: Option<SimpleInterval<u8>>,
    ) -> io::Result<(
        BoxStream<'_, Result<RecordBatch, ArrowError>>,
        HashMap<u64, f32, BuildIdentityHasher<u64>>,
    )> {
        let builder = self.handle.spectrum_peaks().await?;
        let meta_index = self.metadata.peak_indices.as_ref().ok_or(io::Error::new(
            io::ErrorKind::NotFound,
            "peak metadata was not found",
        ))?;

        let ion_mobility_range = if !meta_index.array_indices.has_ion_mobility() {
            None
        } else {
            ion_mobility_range
        };

        let (time_index, index_range) = self
            .get_spectrum_index_range_for_time_range(time_range, ms_level_range)
            .await?;

        let iter = AsyncPointDataReader(builder, BufferContext::Spectrum)
            .query_points(
                index_range,
                mz_range,
                ion_mobility_range,
                &meta_index.query_index,
                &meta_index.array_indices,
                &self.metadata,
            )
            .await?;
        Ok((iter, time_index))
    }

    pub async fn get_spectrum_index_range_for_time_range(
        &self,
        time_range: SimpleInterval<f32>,
        ms_level_range: Option<SimpleInterval<u8>>,
    ) -> io::Result<(HashMap<u64, f32, BuildIdentityHasher<u64>>, MaskSet)> {
        let mut time_indexer = TimeIndexDecoder::new(time_range, ms_level_range);
        if let Some(cache) = self.spectrum_metadata_cache.as_ref() {
            time_indexer.from_descriptions(cache.as_slice());
            return Ok(time_indexer.finish());
        }

        let rows = self
            .query_indices
            .spectrum_time_index
            .row_selection_overlaps(&time_range);

        let builder = self.handle.spectrum_metadata().await?;

        let has_ms_level_range = ms_level_range.is_some();
        let ms_level_range = ms_level_range.unwrap_or_default();
        let columns_for_predicate: &[&str] = if has_ms_level_range {
            &[
                "spectrum.time",
                "spectrum.ms_level",
                "spectrum.MS_1000511_ms_level",
            ]
        } else {
            &["spectrum.time"]
        };

        let predicate_mask = ProjectionMask::columns(
            builder.parquet_schema(),
            columns_for_predicate.iter().copied(),
        );

        let predicate = ArrowPredicateFn::new(predicate_mask, move |batch| {
            let root = batch.column(0).as_struct();
            let times = root.column_by_name("time").unwrap();
            if has_ms_level_range {
                let ms_levels = root
                    .column_by_name("ms_level")
                    .or_else(|| root.column_by_name("MS_1000511_ms_level"))
                    .unwrap();
                arrow::compute::and(
                    &time_range.contains_dy(times),
                    &ms_level_range.contains_dy(ms_levels),
                )
            } else {
                Ok(time_range.contains_dy(times))
            }
        });

        let proj = ProjectionMask::columns(
            builder.parquet_schema(),
            ["spectrum.index", "spectrum.time"],
        );

        let mut reader = builder
            .with_row_selection(rows)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .with_projection(proj)
            .build()?;

        while let Some(batch) = reader.next().await.transpose()? {
            time_indexer.decode_batch(batch)?;
        }

        Ok(time_indexer.finish())
    }

    pub(crate) async fn load_all_spectrum_metadata_impl(
        &self,
    ) -> io::Result<Vec<SpectrumDescription>> {
        log::trace!("Loading all spectrum metadata");
        let builder = SpectrumMetadataReader(self.handle.spectrum_metadata().await?);

        let rows = builder.prepare_rows_for_all(&self.query_indices);
        let predicate = builder.prepare_predicate_for_all();

        let mut reader = builder
            .0
            .with_row_selection(rows)
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .with_batch_size(10_000)
            .build()?;

        let mut decoder = SpectrumMetadataDecoder::new(&self.metadata);

        while let Some(batch) = reader.next().await.transpose()? {
            decoder.decode_batch(batch);
        }

        let descriptions = decoder.finish();

        log::trace!("Finished loading all spectrum metadata");
        Ok(descriptions)
    }

    pub(crate) async fn load_all_chromatgram_metadata_impl(
        &self,
    ) -> io::Result<Vec<ChromatogramDescription>> {
        let builder = ChromatogramMetadataReader(self.handle.chromatograms_metadata().await?);

        let predicate = builder.prepare_predicate_for_all();

        let mut reader = builder
            .0
            .with_row_filter(RowFilter::new(vec![Box::new(predicate)]))
            .build()?;

        let mut decoder = ChromatogramMetadataDecoder::new(&self.metadata);

        while let Some(batch) = reader.next().await.transpose()? {
            decoder.decode_batch(batch);
        }

        Ok(decoder.finish())
    }

    pub(crate) async fn load_spectrum_auxiliary_array_count(&self) -> io::Result<Vec<u32>> {
        let builder = self.handle.spectrum_metadata().await?;

        let mut decoder = AuxiliaryArrayCountDecoder::new(BufferContext::Spectrum);

        let proj = match decoder.build_projection(&builder) {
            Some(proj) => proj,
            None => return Ok(Vec::new()),
        };

        let mut reader = builder.with_projection(proj).build()?;
        let n = self.len();
        decoder.resize(n);

        while let Some(batch) = reader.next().await.transpose()? {
            decoder.decode_batch(&batch);
        }
        Ok(decoder.finish())
    }

    pub(crate) async fn load_chromatogram_auxiliary_array_count(&self) -> io::Result<Vec<u32>> {
        let builder = match self.handle.chromatograms_metadata().await {
            Ok(builder) => builder,
            Err(e) => {
                log::trace!("{e}");
                return Ok(Vec::new());
            }
        };

        let mut decoder = AuxiliaryArrayCountDecoder::new(BufferContext::Chromatogram);

        let proj = match decoder.build_projection(&builder) {
            Some(proj) => proj,
            None => return Ok(Vec::new()),
        };

        let mut reader = builder.with_projection(proj).build()?;
        let n = self.len();
        decoder.resize(n);

        while let Some(batch) = reader.next().await.transpose()? {
            decoder.decode_batch(&batch);
        }
        Ok(decoder.finish())
    }

    async fn load_auxiliary_arrays_from(
        &self,
        mut reader: ParquetRecordBatchStream<T::File>,
    ) -> Vec<DataArray> {
        let mut results = Vec::new();

        while let Some(bat) = reader.next().await.transpose().unwrap() {
            let root = bat.column(0);
            let root = root.as_struct();
            let data = root.column(1).as_list::<i64>();
            let data = data.values().as_struct();
            let arrays = AuxiliaryArrayVisitor::default().visit(data);
            results.extend(arrays);
        }

        results
    }

    pub(crate) async fn load_auxiliary_arrays_for_chromatogram(
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

        let builder = self.handle.chromatograms_metadata().await?;
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

        let results = self.load_auxiliary_arrays_from(reader).await;
        Ok(results)
    }

    pub(crate) async fn load_auxiliary_arrays_for_spectrum(
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

        let builder = self.handle.spectrum_metadata().await?;

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

        let results = self.load_auxiliary_arrays_from(reader).await;
        Ok(results)
    }

    /// Load median delta coefficient column if it is present.
    pub(crate) async fn load_delta_models(&self) -> io::Result<Vec<Option<Vec<f64>>>> {
        let builder = self.handle.spectrum_metadata().await?;

        let mut decoder = DeltaModelDecoder::default();

        let proj = match decoder.build_projection(&builder) {
            Some(proj) => proj,
            None => return Ok(Vec::new()),
        };

        let mut reader = builder
            .with_projection(proj)
            .with_batch_size(10_000)
            .build()?;

        let n = self.metadata.spectrum_id_index.len();
        decoder.resize(n);

        while let Some(batch) = reader.next().await.transpose()? {
            decoder.decode_batch(&batch);
        }

        Ok(decoder.finish())
    }

    /// Read the complete data arrays for the spectrum at `index`
    pub async fn get_spectrum_arrays(&mut self, index: u64) -> io::Result<Option<BinaryArrayMap>> {
        let delta_model = self.metadata.model_deltas_for(index as usize);
        let builder = self.handle.spectra_data().await?;

        let PageQuery {
            pages,
            row_group_indices,
        } = self.query_indices.query_pages(index);

        // If there is only one row group in the scan, take the fast path through the cache
        if row_group_indices.len() == 1 {
            let row_group_index = row_group_indices[0];
            let rg = self
                .read_spectrum_data_cache(row_group_index, index)
                .await?;
            let mut arrays = rg.slice_to_arrays_of(row_group_index, index, delta_model.as_ref())?;
            for v in self.load_auxiliary_arrays_for_spectrum(index).await? {
                arrays.add(v);
            }
            return Ok(Some(arrays));
        }

        if self.query_indices.spectrum_chunk_index.is_populated() {
            log::trace!("Using chunk strategy for reading spectrum {index}");
            let mut out = AsyncSpectrumChunkReader::new(builder)
                .read_chunks_for_entity(
                    index,
                    &self.query_indices.spectrum_chunk_index,
                    &self.metadata.spectrum_array_indices,
                    delta_model.as_ref(),
                    Some(PageQuery::new(row_group_indices, pages)),
                )
                .await?;
            for v in self.load_auxiliary_arrays_for_spectrum(index).await? {
                out.add(v);
            }
            return Ok(Some(out));
        }

        if pages.is_empty() {
            let mut out = BinaryArrayMap::new();
            for v in self.load_auxiliary_arrays_for_spectrum(index).await? {
                out.add(v);
            }
            return Ok(Some(out));
        };

        let reader = AsyncPointDataReader(builder, crate::BufferContext::Spectrum);

        if let Some(mut out) = reader
            .read_points_of(
                index,
                &self.query_indices.spectrum_point_index,
                self.metadata.spectrum_array_indices(),
                delta_model.as_ref(),
            )
            .await?
        {
            for v in self.load_auxiliary_arrays_for_spectrum(index).await? {
                out.add(v);
            }
            Ok(Some(out))
        } else {
            Ok(None)
        }
    }

    pub async fn get_chromatogram_metadata(
        &mut self,
        index: u64,
    ) -> io::Result<Option<ChromatogramDescription>> {
        self.load_all_chromatgram_metadata_impl()
            .await
            .map(|v| v.into_iter().nth(index as usize))
    }

    pub async fn get_chromatogram_arrays(
        &mut self,
        index: u64,
    ) -> io::Result<Option<BinaryArrayMap>> {
        let builder = self.handle.chromatograms_data().await?;
        let reader = AsyncPointDataReader(builder, BufferContext::Chromatogram);
        let out = reader
            .read_points_of(
                index,
                &self.query_indices.chromatogram_point_index,
                &self.metadata.chromatogram_array_indices,
                None,
            )
            .await?;

        if let Some(mut out) = out {
            for v in self.load_auxiliary_arrays_for_chromatogram(index).await? {
                out.add(v);
            }
            Ok(Some(out))
        } else {
            Ok(None)
        }
    }

    pub fn file_index(&self) -> &crate::archive::FileIndex {
        self.handle.file_index()
    }

    pub fn list_files(&self) -> &[String] {
        self.handle.list_files()
    }

    pub fn open_stream(
        &self,
        name: &str,
    ) -> impl Future<Output = Result<<T as AsyncArchiveSource>::File, io::Error>> {
        self.handle.open_stream(name)
    }
}

pub type AsyncMzPeakReader =
    AsyncMzPeakReaderType<AsyncZipArchiveSource, CentroidPeak, DeconvolutedPeak>;

#[cfg(test)]
mod test {
    use object_store::local::LocalFileSystem;

    use super::*;

    #[tokio::test]
    async fn test_url() -> io::Result<()> {
        let store = LocalFileSystem::new_with_prefix(".")?;
        let mut handle =
            AsyncMzPeakReader::from_store_path(Arc::new(store), ObjectPath::from("small.mzpeak"))
                .await?;
        let _spec = handle.get_spectrum(0).await.unwrap();
        Ok(())
    }
}
