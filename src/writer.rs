use std::{
    fs,
    io::{self, prelude::*},
    marker::PhantomData,
    sync::Arc,
};

use arrow::{
    array::{ArrayBuilder, AsArray, RecordBatch},
    datatypes::{Field, Schema},
};
use mzpeaks::{CentroidPeak, DeconvolutedPeak};
use parquet::{
    arrow::{ArrowWriter, arrow_writer::ArrowWriterOptions},
    basic::Compression,
    file::metadata::KeyValue,
};

use mzdata::{
    io::{RandomAccessSpectrumSource, StreamingSpectrumIterator},
    meta::{FileMetadataConfig, MSDataFileMetadata},
    prelude::*,
    spectrum::{ArrayType, Chromatogram, MultiLayerSpectrum, SignalContinuity},
};

use crate::{
    BufferName,
    archive::{MzPeakArchiveType, ZipArchiveWriter},
    buffer_descriptors::{BufferOverrideTable, BufferPriority},
    peak_series::{
        ArrayIndex, BufferContext, TIME_ARRAY, ToMzPeakDataSeries, array_map_to_schema_arrays,
    },
    writer::builder::SpectrumFieldVisitors,
};
use crate::{
    chunk_series::{ArrowArrayChunk, ChunkingStrategy},
    peak_series::MZ_ARRAY,
};

mod array_buffer;
mod base;
mod builder;
mod mini_peak;
mod split;
mod visitor;

pub use array_buffer::{
    ArrayBufferWriter, ArrayBufferWriterVariants, ArrayBuffersBuilder, ChunkBuffers, PointBuffers,
};
pub use base::AbstractMzPeakWriter;
pub use builder::{MzPeakWriterBuilder, WriteBatchConfig, ArrayConversionHelper};
pub use split::UnpackedMzPeakWriterType;

pub use visitor::{
    ActivationBuilder, AuxiliaryArrayBuilder, CURIEBuilder, ChromatogramBuilder,
    ChromatogramDetailsBuilder, CustomBuilderFromParameter, IsolationWindowBuilder, ParamBuilder,
    ParamListBuilder, ParamValueBuilder, PrecursorBuilder, ScanBuilder, ScanWindowBuilder,
    SelectedIonBuilder, SpectrumBuilder, SpectrumDetailsBuilder, SpectrumVisitor, StructVisitor,
    StructVisitorBuilder, VisitorBase, inflect_cv_term_to_column_name,
};

pub(crate) use base::implement_mz_metadata;
pub(crate) use mini_peak::MiniPeakWriterType;

/*
Internal helper function that, given an iterator over spectra, will
perform the requested overrides and encodings to the data buffers and
construct a Parquet schema.
*/
fn _eval_spectra_from_iter_for_fields<
    C: CentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<CentroidPeak>,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<DeconvolutedPeak>,
>(
    iter: impl Iterator<Item = MultiLayerSpectrum<C, D>>,
    overrides: &BufferOverrideTable,
    use_chunked_encoding: Option<ChunkingStrategy>,
) -> Vec<Arc<Field>> {
    let mut arrays: Vec<Arc<Field>> = Vec::new();
    let mut is_profile = 0;

    let field_it = iter
        .flat_map(|s| {
            log::trace!("Sampling arrays from {}", s.id());
            if s.signal_continuity() == SignalContinuity::Profile {
                is_profile += 1;
            }
            // Using the raw data arrays (not peak lists), to generate a
            // dataset schema.
            s.raw_arrays().and_then(|map| {
                // generate a schema for this chunked
                if let Some(use_chunked_encoding) = use_chunked_encoding {
                    ArrowArrayChunk::from_arrays(
                        0,
                        None,
                        MZ_ARRAY
                            .clone()
                            .with_priority(Some(BufferPriority::Primary))
                            .with_sorting_rank(Some(1)),
                        map,
                        use_chunked_encoding,
                        overrides,
                        false,
                        false,
                        None,
                    )
                    .ok()
                    .map(|s| {
                        (
                            s.0[0]
                                .to_schema(
                                    BufferContext::Spectrum,
                                    &[
                                        use_chunked_encoding,
                                        ChunkingStrategy::Basic { chunk_size: 50.0 },
                                    ],
                                    false,
                                )
                                .fields,
                            Vec::new(),
                        )
                    })
                } else {
                    array_map_to_schema_arrays(
                        BufferContext::Spectrum,
                        map,
                        map.mzs().map(|a| a.len()).unwrap_or_default(),
                        0,
                        None,
                        overrides,
                    )
                    .ok()
                }
            })
        })
        .flat_map(|(fields, _arrs)| {
            let fields: Vec<_> = fields.iter().cloned().collect();
            fields
        });

    for field in field_it {
        if !arrays.iter().any(|f| f.name() == field.name()) {
            if let Some(buffer) = BufferName::from_field(BufferContext::Spectrum, field.clone()) {
                log::trace!("Adding {buffer:?} to schema")
            }
            arrays.push(field);
        }
    }
    if is_profile > 0 {
        log::debug!("Detected profile spectra");
    }
    arrays
}

/// Collect arrays fields from an iterator of chromatograms to prepare the data file schema.
///
/// This consumes the entire iterator.
///
/// # Arguments
/// `reader`: The stream of chromatograms to read from
/// `overrides`: The array mapping rules override array data types.
/// `use_chunked_encoding`: The chunk encoding format to use, if any
pub fn sample_array_types_from_chromatograms<I: Iterator<Item = Chromatogram>>(
    iter: I,
    overrides: &BufferOverrideTable,
    use_chunked_encoding: Option<ChunkingStrategy>,
) -> Vec<Arc<Field>> {
    let field_it = iter
        .flat_map(|s| {
            log::trace!("Sampling arrays from {}", s.id());
            let map = &s.arrays;
            if let Some(use_chunked_encoding) = use_chunked_encoding {
                ArrowArrayChunk::from_arrays(
                    0,
                    None,
                    TIME_ARRAY
                        .clone()
                        .with_priority(Some(BufferPriority::Primary))
                        .with_sorting_rank(Some(1)),
                    map,
                    use_chunked_encoding,
                    overrides,
                    false,
                    false,
                    None,
                )
                .ok()
                .map(|s| {
                    (
                        s.0[0]
                            .to_schema(
                                BufferContext::Chromatogram,
                                &[
                                    use_chunked_encoding,
                                    ChunkingStrategy::Basic { chunk_size: 50.0 },
                                ],
                                false,
                            )
                            .fields,
                        Vec::new(),
                    )
                })
            } else {
                array_map_to_schema_arrays(
                    BufferContext::Chromatogram,
                    map,
                    map.get(&ArrayType::TimeArray)
                        .and_then(|a| a.data_len().ok())
                        .unwrap_or_default(),
                    0,
                    None,
                    overrides,
                )
                .ok()
            }
        })
        .flat_map(|(fields, _arrs)| {
            let fields: Vec<_> = fields.iter().cloned().collect();
            fields
        });
    let mut arrays: Vec<Arc<Field>> = Vec::new();
    for field in field_it {
        if !arrays.iter().any(|f| f.name() == field.name()) {
            arrays.push(field);
        }
    }
    arrays
}

/// Collect arrays fields from spectra in a [`StreamingSpectrumIterator`] to prepare
/// the data file schema.
///
/// This consumes only the next 10 spectra.
///
/// # Arguments
/// `reader`: The stream of spectra to read from
/// `overrides`: The array mapping rules override array data types.
/// `use_chunked_encoding`: The chunk encoding format to use, if any
pub fn sample_array_types_from_spectrum_stream<
    C: CentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<CentroidPeak>,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<DeconvolutedPeak>,
    I: Iterator<Item = MultiLayerSpectrum<C, D>>,
>(
    reader: &mut StreamingSpectrumIterator<C, D, MultiLayerSpectrum<C, D>, I>,
    overrides: &BufferOverrideTable,
    use_chunked_encoding: Option<ChunkingStrategy>,
) -> Vec<Arc<Field>>
where
    MultiLayerSpectrum<C, D>: Clone,
{
    reader.populate_buffer(10);
    _eval_spectra_from_iter_for_fields(
        reader.iter_buffer().cloned(),
        overrides,
        use_chunked_encoding,
    )
}

/// Collect arrays fields from spectra in a [`RandomAccessSpectrumSource`] to prepare
/// the data file schema.
///
/// This examines the first, 100th, and middle spectrum from `reader`.
///
/// # Arguments
/// `reader`: The stream of spectra to read from
/// `overrides`: The array mapping rules override array data types.
/// `use_chunked_encoding`: The chunk encoding format to use, if any
pub fn sample_array_types_from_spectrum_source<
    C: CentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<CentroidPeak>,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<DeconvolutedPeak>,
    R: RandomAccessSpectrumSource<C, D, MultiLayerSpectrum<C, D>>,
>(
    reader: &mut R,
    overrides: &BufferOverrideTable,
    use_chunked_encoding: Option<ChunkingStrategy>,
) -> Vec<Arc<Field>> {
    let n = reader.len();
    if n == 0 {
        return Vec::new();
    }

    let it = [0, 100.min(n - 1), n / 2]
        .into_iter()
        .flat_map(|i| reader.get_spectrum_by_index(i));
    _eval_spectra_from_iter_for_fields(it, overrides, use_chunked_encoding)
}

/// Array type inference from inputs
impl MzPeakWriterBuilder {
    /// Collect arrays fields from spectra in a [`RandomAccessSpectrumSource`] to prepare
    /// the data file schema.
    ///
    /// This examines the first, 100th, and middle spectrum from `reader`.
    ///
    /// # Arguments
    /// `reader`: The stream of spectra to read from
    pub fn sample_array_types_from_spectrum_source<
        C: CentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<CentroidPeak>,
        D: DeconvolutedCentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<DeconvolutedPeak>,
        R: RandomAccessSpectrumSource<C, D, MultiLayerSpectrum<C, D>>,
    >(
        mut self,
        reader: &mut R,
    ) -> Self {
        let fields = sample_array_types_from_spectrum_source(
            reader,
            &self.spectrum_overrides(),
            self.chunked_encoding,
        );

        for f in fields {
            self = self.add_spectrum_field(f);
        }

        self
    }

    fn take_or_initialize_peak_builder(&mut self) -> ArrayBuffersBuilder {
        let point_builder = self
            .store_peaks_and_profiles_apart
            .take()
            .unwrap_or_else(|| {
                ArrayBuffersBuilder::default()
                    .prefix("point")
                    .with_context(BufferContext::Spectrum)
            });
        point_builder
    }

    pub fn register_spectrum_peak_type<T: ToMzPeakDataSeries>(mut self) -> Self {
        let point_builder = self.take_or_initialize_peak_builder();
        self.store_peaks_and_profiles_apart(Some(point_builder.add_peak_type::<T>()))
    }

    pub fn sample_array_types_for_peaks_from_spectrum_source<
        C: CentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<CentroidPeak>,
        D: DeconvolutedCentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<DeconvolutedPeak>,
        R: RandomAccessSpectrumSource<C, D, MultiLayerSpectrum<C, D>>,
    >(
        mut self,
        reader: &mut R,
    ) -> Self {
        let mut point_builder = self.take_or_initialize_peak_builder();
        for f in sample_array_types_from_spectrum_source(reader, &self.spectrum_overrides(), None) {
            point_builder = point_builder.add_field(f);
        }
        self.store_peaks_and_profiles_apart(Some(point_builder))
    }

    /// Collect arrays fields from spectra in a [`StreamingSpectrumIterator`] to prepare
    /// the data file schema.
    ///
    /// This consumes only the next 10 spectra.
    ///
    /// # Arguments
    /// `reader`: The stream of spectra to read from
    pub fn sample_array_types_from_spectrum_stream<
        C: CentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<CentroidPeak>,
        D: DeconvolutedCentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<DeconvolutedPeak>,
        I: Iterator<Item = MultiLayerSpectrum<C, D>>,
    >(
        mut self,
        reader: &mut StreamingSpectrumIterator<C, D, MultiLayerSpectrum<C, D>, I>,
    ) -> Self
    where
        MultiLayerSpectrum<C, D>: Clone,
    {
        let fields = sample_array_types_from_spectrum_stream(
            reader,
            &self.spectrum_overrides(),
            self.chunked_encoding,
        );

        for f in fields {
            self = self.add_spectrum_field(f);
        }

        self
    }

    /// Collect arrays fields from an iterator of chromatograms to prepare the data file schema.
    ///
    /// This consumes the entire iterator.
    ///
    /// # Arguments
    /// `reader`: The stream of chromatograms to read from
    pub fn sample_array_types_from_chromatograms<I: Iterator<Item = Chromatogram>>(
        mut self,
        iter: I,
    ) -> Self {
        let fields = sample_array_types_from_chromatograms(
            iter,
            &self.chromatogram_overrides(),
            self.chromatogram_chunked_encoding,
        );
        for f in fields {
            self = self.add_chromatogram_field(f);
        }
        self
    }
}

/// Write an mzPeak archive to an uncompressed ZIP archive
pub struct MzPeakWriterType<
    W: Write + Send + Seek,
    C: CentroidLike + ToMzPeakDataSeries = CentroidPeak,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries = DeconvolutedPeak,
> {
    archive_writer: Option<ArrowWriter<ZipArchiveWriter<W>>>,
    spectrum_buffers: ArrayBufferWriterVariants,
    separate_peak_writer: Option<MiniPeakWriterType<fs::File>>,

    chromatogram_buffers: ArrayBufferWriterVariants,

    spectrum_metadata_buffer: SpectrumBuilder,
    chromatogram_metadata_buffer: ChromatogramBuilder,

    use_chunked_encoding: Option<ChunkingStrategy>,
    use_chromatogram_chunked_encoding: Option<ChunkingStrategy>,

    spectrum_data_point_counter: u64,
    chromatogram_data_point_counter: u64,

    buffer_size: usize,
    compression: Compression,

    #[allow(unused)]
    write_batch_config: WriteBatchConfig,
    mz_metadata: FileMetadataConfig,
    _t: PhantomData<(C, D)>,
}

impl<
    W: Write + Send + Seek,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> AbstractMzPeakWriter for MzPeakWriterType<W, C, D>
{
    fn append_key_value_metadata(&mut self, key: String, value: Option<String>) {
        self.archive_writer
            .as_mut()
            .unwrap()
            .append_key_value_metadata(KeyValue::new(key, value));
    }

    fn use_chunked_encoding(&self) -> Option<&ChunkingStrategy> {
        self.use_chunked_encoding.as_ref()
    }

    fn use_chromatogram_chunked_encoding(&self) -> Option<&ChunkingStrategy> {
        self.use_chromatogram_chunked_encoding.as_ref()
    }

    fn spectrum_entry_buffer_mut(&mut self) -> &mut SpectrumBuilder {
        &mut self.spectrum_metadata_buffer
    }

    fn spectrum_data_buffer_mut(&mut self) -> &mut ArrayBufferWriterVariants {
        &mut self.spectrum_buffers
    }

    fn check_data_buffer(&mut self) -> io::Result<()> {
        if self.spectrum_counter() % (self.buffer_size as u64) == 0 {
            log::debug!(
                "Flushing data buffer. {} spectra written so far. {} rows in the buffer",
                self.spectrum_counter(),
                self.buffered_spectrum_data()
            );
            self.flush_data_arrays()?;
        }
        Ok(())
    }

    fn spectrum_counter(&self) -> u64 {
        self.spectrum_metadata_buffer.index_counter()
    }

    fn spectrum_precursor_counter(&self) -> u64 {
        self.spectrum_metadata_buffer.precursor_index_counter()
    }

    fn separate_peak_writer(&mut self) -> Option<&mut MiniPeakWriterType<fs::File>> {
        self.separate_peak_writer.as_mut()
    }

    fn chromatogram_counter(&self) -> u64 {
        self.chromatogram_metadata_buffer.index_counter()
    }

    fn chromatogram_entry_buffer_mut(&mut self) -> &mut ChromatogramBuilder {
        &mut self.chromatogram_metadata_buffer
    }

    fn chromatogram_data_buffer_mut(&mut self) -> &mut ArrayBufferWriterVariants {
        &mut self.chromatogram_buffers
    }
}

impl<
    W: Write + Send + Seek,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> MSDataFileMetadata for MzPeakWriterType<W, C, D>
{
    mzdata::delegate_impl_metadata_trait!(mz_metadata);
}

impl<
    W: Write + Send + Seek,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> MzPeakWriterType<W, C, D>
{
    pub fn builder() -> MzPeakWriterBuilder {
        MzPeakWriterBuilder::default()
    }

    pub fn new(
        writer: W,
        spectrum_buffers_builder: ArrayBuffersBuilder,
        chromatogram_buffers_builder: ArrayBuffersBuilder,
        buffer_size: usize,
        mask_zero_intensity_runs: bool,
        shuffle_mz: bool,
        use_chunked_encoding: Option<ChunkingStrategy>,
        use_chromatogram_chunked_encoding: Option<ChunkingStrategy>,
        compression: Compression,
        store_peaks_and_profiles_apart: Option<ArrayBuffersBuilder>,
        write_batch_config: WriteBatchConfig,
        spectrum_fields: SpectrumFieldVisitors,
    ) -> Self {
        let mut spectrum_metadata_buffer = SpectrumBuilder::default();
        spectrum_metadata_buffer.add_visitors_from(spectrum_fields);

        let spectrum_buffers: ArrayBufferWriterVariants = if use_chunked_encoding.is_some() {
            spectrum_buffers_builder
                .build_chunked(
                    Arc::new(Schema::empty()),
                    BufferContext::Spectrum,
                    mask_zero_intensity_runs,
                )
                .into()
        } else {
            let spectrum_buffers = spectrum_buffers_builder.build(
                Arc::new(Schema::empty()),
                BufferContext::Spectrum,
                mask_zero_intensity_runs,
            );
            spectrum_buffers.into()
        };

        let chromatogram_buffers: ArrayBufferWriterVariants = if use_chunked_encoding.is_some() {
            chromatogram_buffers_builder
                .build_chunked(
                    Arc::new(Schema::empty()),
                    BufferContext::Chromatogram,
                    false,
                )
                .into()
        } else {
            chromatogram_buffers_builder
                .build(
                    Arc::new(Schema::empty()),
                    BufferContext::Chromatogram,
                    false,
                )
                .into()
        };

        let mut writer = ZipArchiveWriter::new(writer);
        writer.start_spectrum_data().unwrap();

        let data_props = Self::spectrum_data_writer_props(
            &spectrum_buffers,
            spectrum_buffers.index_path(),
            shuffle_mz,
            &use_chunked_encoding,
            compression,
            write_batch_config,
        );

        let separate_peak_writer = if let Some(peak_buffer_builder) = store_peaks_and_profiles_apart
        {
            let peak_buffer_file =
                tempfile::tempfile().expect("Failed to create temporary file to write peaks to");
            let peak_buffer = peak_buffer_builder
                .include_time(spectrum_buffers.include_time())
                .build(Arc::new(Schema::empty()), BufferContext::Spectrum, false);

            let peak_data_props = Self::spectrum_data_writer_props(
                &peak_buffer,
                peak_buffer.index_path(),
                shuffle_mz,
                &None,
                compression,
                write_batch_config,
            );

            let peak_writer = ArrowWriter::try_new_with_options(
                peak_buffer_file,
                peak_buffer.schema().clone(),
                ArrowWriterOptions::new().with_properties(peak_data_props),
            )
            .unwrap();

            Some(MiniPeakWriterType::new(
                peak_writer,
                peak_buffer,
                buffer_size,
            ))
        } else {
            None
        };

        let mut this = Self {
            archive_writer: Some(
                ArrowWriter::try_new_with_options(
                    writer,
                    spectrum_buffers.schema().clone(),
                    ArrowWriterOptions::new().with_properties(data_props),
                )
                .unwrap(),
            ),
            separate_peak_writer,
            use_chunked_encoding,
            use_chromatogram_chunked_encoding,
            spectrum_metadata_buffer,
            spectrum_buffers,
            chromatogram_buffers,
            spectrum_data_point_counter: 0,
            chromatogram_data_point_counter: 0,
            chromatogram_metadata_buffer: Default::default(),
            buffer_size: buffer_size,
            mz_metadata: Default::default(),
            compression,
            write_batch_config,
            _t: PhantomData,
        };
        this.add_spectrum_array_metadata();
        this
    }

    implement_mz_metadata!();

    fn add_spectrum_array_metadata(&mut self) {
        let spectrum_array_index: ArrayIndex = self.spectrum_buffers.as_array_index();
        self.append_key_value_metadata(
            "spectrum_array_index".to_string(),
            spectrum_array_index.to_json().into(),
        );
    }

    fn add_chromatogram_array_metadata(&mut self) {
        let chromatogram_array_index: ArrayIndex = self.chromatogram_buffers.as_array_index();
        self.append_key_value_metadata(
            "chromatogram_array_index".to_string(),
            Some(chromatogram_array_index.to_json()),
        );
    }

    pub fn write_spectrum<
        A: ToMzPeakDataSeries + CentroidLike,
        B: ToMzPeakDataSeries + DeconvolutedCentroidLike,
        S: SpectrumLike<A, B> + 'static,
    >(
        &mut self,
        spectrum: &S,
    ) -> io::Result<()> {
        AbstractMzPeakWriter::write_spectrum(self, spectrum)
    }

    fn flush_data_arrays(&mut self) -> io::Result<()> {
        let use_chunks = self.use_chunked_encoding().is_some();
        for batch in self.spectrum_buffers.drain() {
            self.spectrum_data_point_counter += batch.num_rows() as u64;
            if let Some(writer) = self.archive_writer.as_mut() {
                writer.write(&batch)?;
                if writer.in_progress_size() > 512_000_000 && use_chunks {
                    log::debug!(
                        "Flushing row group buffer with approximately {} bytes",
                        writer.in_progress_size()
                    );
                    writer.flush()?;
                }
            } else {
                panic!("Attempted to write spectrum data but writer does not exist");
            }
        }
        Ok(())
    }

    fn flush_spectrum_metadata_records(&mut self) -> io::Result<()> {
        let arrays = self.spectrum_metadata_buffer.finish();
        let batch = RecordBatch::from(arrays.as_struct());
        self.archive_writer.as_mut().unwrap().write(&batch)?;
        Ok(())
    }

    pub fn buffered_spectrum_data(&self) -> usize {
        self.spectrum_buffers.len()
    }

    fn flush_chromatogram_metadata_records(&mut self) -> io::Result<()> {
        let arrays = self.chromatogram_metadata_buffer.finish();
        let batch = RecordBatch::from(arrays.as_struct());
        self.archive_writer.as_mut().unwrap().write(&batch)?;
        Ok(())
    }

    fn flush_chromatogram_data_records(&mut self) -> io::Result<()> {
        for batch in self.chromatogram_buffers.drain() {
            self.chromatogram_data_point_counter += batch.num_rows() as u64;
            if let Some(writer) = self.archive_writer.as_mut() {
                writer.write(&batch)?;
                // if writer.in_progress_size() > 512_000_000 && use_chunks {
                //     log::debug!(
                //         "Flushing row group buffer with approximately {} bytes",
                //         writer.in_progress_size()
                //     );
                //     writer.flush()?;
                // }
            } else {
                panic!("Attempted to write spectrum data but writer does not exist");
            }
        }
        Ok(())
    }

    fn finish_parquet_inner(&mut self) -> Result<ZipArchiveWriter<W>, parquet::errors::ParquetError> {
        if self.archive_writer.is_some() {
            self.flush_data_arrays()?;
            self.append_key_value_metadata(
                "spectrum_count".into(),
                Some(self.spectrum_counter().to_string()),
            );
            self.append_key_value_metadata(
                "spectrum_data_point_count".into(),
                Some(self.spectrum_data_point_counter.to_string()),
            );

            let mut writer = self.archive_writer.take().unwrap().into_inner()?;

            if let Some(peak_file_writer) = self.separate_peak_writer.take() {
                let mut peak_file = peak_file_writer.finish()?;
                log::trace!("Copying peaks file into zip archive");
                peak_file.rewind()?;
                writer.add_file_from_read(
                    &mut peak_file,
                    Some(&MzPeakArchiveType::SpectrumPeakDataArrays.tag_file_suffix()),
                    Some(MzPeakArchiveType::SpectrumPeakDataArrays.into()),
                )?;
            }

            writer.start_spectrum_metadata().unwrap();
            let metadata_fields = self.spectrum_metadata_buffer.schema();
            self.archive_writer = Some(ArrowWriter::try_new_with_options(
                writer,
                metadata_fields.clone(),
                ArrowWriterOptions::new()
                    .with_properties(Self::spectrum_metadata_writer_props(&metadata_fields)),
            )?);
            self.flush_spectrum_metadata_records()?;
            self.append_metadata();
            self.append_key_value_metadata(
                "spectrum_count".into(),
                Some(self.spectrum_counter().to_string()),
            );
            self.append_key_value_metadata(
                "spectrum_data_point_count".into(),
                Some(self.spectrum_data_point_counter.to_string()),
            );

            writer = self.archive_writer.take().unwrap().into_inner()?;

            if !self.chromatogram_metadata_buffer.is_empty() {
                writer.start_chromatogram_metadata().unwrap();
                let metadata_fields = self.chromatogram_metadata_buffer.schema();
                self.archive_writer = Some(ArrowWriter::try_new_with_options(
                    writer,
                    metadata_fields.clone(),
                    ArrowWriterOptions::new()
                        .with_properties(Self::spectrum_metadata_writer_props(&metadata_fields)),
                )?);
                self.flush_chromatogram_metadata_records()?;
                self.append_key_value_metadata(
                    "chromatogram_count".into(),
                    Some(self.chromatogram_counter().to_string()),
                );
                self.append_key_value_metadata(
                    "chromatogram_data_point_count".into(),
                    Some(self.chromatogram_data_point_counter.to_string()),
                );
                writer = self.archive_writer.take().unwrap().into_inner()?;

                writer.start_chromatogram_data().unwrap();
                self.archive_writer = Some(ArrowWriter::try_new_with_options(
                    writer,
                    self.chromatogram_buffers.schema().clone(),
                    ArrowWriterOptions::new().with_properties(
                        Self::chromatogram_data_writer_props(
                            &self.chromatogram_buffers,
                            BufferContext::Chromatogram.index_field().name().to_string(),
                            &None,
                            self.compression,
                        ),
                    ),
                )?);
                self.flush_chromatogram_data_records()?;
                self.add_chromatogram_array_metadata();
                self.append_key_value_metadata(
                    "chromatogram_data_point_count".into(),
                    Some(self.chromatogram_data_point_counter.to_string()),
                );
                self.append_metadata();
                writer = self.archive_writer.take().unwrap().into_inner()?;
            }
            Ok(writer)
        } else {
            Err(parquet::errors::ParquetError::EOF(
                "Already closed file".into(),
            ))
        }
    }

    pub fn finish_parquet(mut self) -> Result<ZipArchiveWriter<W>, parquet::errors::ParquetError> {
        self.finish_parquet_inner()
    }

    pub fn finish(&mut self) -> Result<(), parquet::errors::ParquetError> {
        if self.archive_writer.is_some() {
            let writer = self.finish_parquet_inner()?;
            writer.finish().unwrap();
            Ok(())
        } else {
            Err(parquet::errors::ParquetError::EOF(
                "Already closed file".into(),
            ))
        }
    }
}

impl<
    W: Write + Send + Seek,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> Drop for MzPeakWriterType<W, C, D>
{
    fn drop(&mut self) {
        if let Err(e) = self.finish() {
            log::trace!("While dropping MzPeakWriterType: {e}")
        }
    }
}

impl<
    W: Write + Send + Seek,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> SpectrumWriter<C, D> for MzPeakWriterType<W, C, D>
{
    fn write<S: SpectrumLike<C, D> + 'static>(&mut self, spectrum: &S) -> io::Result<usize> {
        self.write_spectrum(spectrum).map(|_| 1)
    }

    fn flush(&mut self) -> io::Result<()> {
        if let Some(w) = self.archive_writer.as_mut() {
            w.flush()?;
        }
        Ok(())
    }

    fn close(&mut self) -> io::Result<()> {
        self.finish()?;
        Ok(())
    }
}

pub type MzPeakWriter<W> = MzPeakWriterType<W, CentroidPeak, DeconvolutedPeak>;

#[cfg(test)]
mod test {
    use arrow::datatypes::DataType;
    use mzdata::{params::Unit, spectrum::BinaryDataArrayType};

    use crate::{BufferName, MzPeakReader, archive::FileEntry, peak_series::BufferFormat};

    use super::*;
    use std::io;

    #[test_log::test]
    fn test_array_type_sampling() -> io::Result<()> {
        let mut reader = mzdata::MZReader::open_path("small.mzML")?;
        let overrides1 = BufferOverrideTable::default();
        let array_types = sample_array_types_from_spectrum_source(&mut reader, &overrides1, None);

        assert_eq!(array_types.len(), 3);
        let mz_buffer = BufferName::new(
            BufferContext::Spectrum,
            ArrayType::MZArray,
            BinaryDataArrayType::Float64,
        )
        .with_unit(Unit::MZ);
        let intensity_buffer = BufferName::new(
            BufferContext::Spectrum,
            ArrayType::IntensityArray,
            BinaryDataArrayType::Float32,
        )
        .with_unit(Unit::DetectorCounts);
        for f in array_types {
            if let Some(name) = BufferName::from_field(BufferContext::Spectrum, f.clone()) {
                if mz_buffer == name {
                } else if intensity_buffer == name {
                } else {
                    panic!("Unexpected {name:?}");
                }
            }
        }

        let array_types = sample_array_types_from_spectrum_source(
            &mut reader,
            &overrides1,
            Some(ChunkingStrategy::Delta { chunk_size: 50.0 }),
        );

        assert_eq!(array_types.len(), 6);
        for f in array_types {
            if let Some(name) = BufferName::from_field(BufferContext::Spectrum, f.clone()) {
                if mz_buffer
                    .clone()
                    .with_format(BufferFormat::ChunkBoundsStart)
                    == name
                {
                } else if mz_buffer.clone().with_format(BufferFormat::ChunkBoundsEnd) == name {
                } else if mz_buffer.clone().with_format(BufferFormat::Chunk) == name {
                } else if mz_buffer.clone().with_format(BufferFormat::ChunkEncoding) == name {
                } else if intensity_buffer
                    .clone()
                    .with_format(BufferFormat::ChunkSecondary)
                    == name
                {
                } else {
                    panic!("Unexpected {name:?}");
                }
            }
        }

        let mut it = StreamingSpectrumIterator::new(reader.iter());

        let array_types = sample_array_types_from_spectrum_stream(&mut it, &overrides1, None);
        assert_eq!(array_types.len(), 3);
        for f in array_types {
            if let Some(name) = BufferName::from_field(BufferContext::Spectrum, f.clone()) {
                if mz_buffer == name {
                } else if intensity_buffer == name {
                } else {
                    panic!("Unexpected {name:?}");
                }
            }
        }

        let mut builder = MzPeakWriter::<io::Cursor<Vec<u8>>>::builder();
        builder = builder.sample_array_types_from_spectrum_stream(&mut it);
        if let DataType::Struct(fields) = builder.spectrum_arrays.dtype() {
            assert_eq!(fields.len(), 3);
        }
        Ok(())
    }

    #[test_log::test]
    fn test_array_building() -> io::Result<()> {
        let mut reader = mzdata::MZReader::open_path("small.mzML")?;
        let mut builder = MzPeakWriter::<fs::File>::builder();
        builder = builder
            .register_spectrum_peak_type::<CentroidPeak>()
            .sample_array_types_from_spectrum_source(&mut reader)
            .sample_array_types_from_chromatograms(reader.iter_chromatograms())
            .include_time_with_spectrum_data(true)
            .null_zeros(true)
            .shuffle_mz(true)
            .write_batch_size(Some(5))
            .dictionary_page_size(Some(2usize.pow(16)))
            .row_group_size(Some(2usize.pow(16)))
            .page_size(Some(2usize.pow(16)))
            .add_spectrum_activation_field(CustomBuilderFromParameter::from_spec(
                mzdata::curie!(MS:1000045),
                "collision energy",
                DataType::Float64,
            ));

        let arc = tempfile::NamedTempFile::with_suffix("_test.mzpeak")?;
        let source = arc.as_file().try_clone()?;
        let mut writer = builder.build(source, true);
        writer.copy_metadata_from(&reader);
        let overrides = writer.spectrum_buffers.overrides();
        assert!(overrides.iter().any(|(_, v)| {
            v.buffer_priority
                .is_some_and(|v| matches!(v, BufferPriority::Primary))
        }));
        writer.write_all_owned(reader.iter())?;
        for chrom in reader.iter_chromatograms() {
            writer.write_chromatogram(&chrom)?;
        }

        let mut zip_writer = writer.finish_parquet()?;

        zip_writer.start_other(&"example.config")?;
        zip_writer.write_all(b"<config><foo>some XML gobbledygook</foo></config>")?;

        let job_entry = FileEntry::new("job.sig".into(), crate::archive::EntityType::Other("other".into()), crate::archive::DataKind::Proprietary);
        zip_writer.add_file_from_read(&mut b"some binary sludge".as_slice(), None::<&String>, Some(job_entry))?;

        zip_writer.finish()?;

        let mut new_reader = MzPeakReader::new(arc.path())?;
        assert_eq!(reader.len(), new_reader.len());
        assert!(new_reader.metadata.spectrum_auxiliary_array_counts.iter().all(|z| *z == 0));
        assert_eq!(new_reader.list_all_files_in_archive().len(), 8);
        let mut buf = Vec::new();
        new_reader.open_stream("example.config")?.read_to_end(&mut buf)?;
        assert_eq!(buf, b"<config><foo>some XML gobbledygook</foo></config>");
        reader.reset();
        new_reader.reset();

        for (a, b)  in reader.iter().zip(new_reader.into_iter()) {
            assert_eq!(a.id(), b.id());
        }
        Ok(())
    }

    #[test_log::test]
    fn test_array_building_chunked() -> io::Result<()> {
        let mut reader = mzdata::MZReader::open_path("small.mzML")?;
        let mut builder = MzPeakWriter::<fs::File>::builder();
        builder = builder
            .chunked_encoding(Some(ChunkingStrategy::Delta { chunk_size: 50.0 }))
            .chromatogram_chunked_encoding(Some(ChunkingStrategy::Delta { chunk_size: 50.0 }))
            .sample_array_types_from_spectrum_source(&mut reader)
            .sample_array_types_from_chromatograms(reader.iter_chromatograms())
            .null_zeros(true)
            .shuffle_mz(true);

        let arc = tempfile::NamedTempFile::with_suffix("_test_chunk.mzpeak")?;
        let source = arc.as_file().try_clone()?;
        let mut writer = builder.build(source, true);
        writer.copy_metadata_from(&reader);
        let overrides = writer.spectrum_buffers.overrides();
        assert!(overrides.iter().any(|(_, v)| {
            v.buffer_priority
                .is_some_and(|v| matches!(v, BufferPriority::Primary))
        }));
        writer.write_all_owned(reader.iter())?;
        for chrom in reader.iter_chromatograms() {
            writer.write_chromatogram(&chrom)?;
        }
        drop(writer);

        let mut new_reader = MzPeakReader::new(arc.path())?;
        assert_eq!(reader.len(), new_reader.len());
        reader.reset();
        new_reader.reset();
        for (a, b)  in reader.iter().zip(new_reader.iter()) {
            assert_eq!(a.id(), b.id());
        }
        Ok(())
    }

    #[test_log::test]
    #[test_log(default_log_filter = "debug")]
    fn test_array_building_numpress() -> io::Result<()> {
        let mut reader = mzdata::MZReader::open_path("small.mzML")?;
        let mut builder = MzPeakWriter::<fs::File>::builder();

        let spectrum_overrides = ArrayConversionHelper::new(false, true, false, true, true).create_type_overrides(
            Some(ChunkingStrategy::NumpressLinear { chunk_size: 50.0 })
        );
        for (k, v) in spectrum_overrides.iter() {
            builder = builder.add_spectrum_array_override(k.clone(), v.clone())
        }
        builder = builder
            .shuffle_mz(true)
            .chunked_encoding(Some(ChunkingStrategy::NumpressLinear { chunk_size: 50.0 }))
            .chromatogram_chunked_encoding(Some(ChunkingStrategy::Delta { chunk_size: 50.0 }))
            .sample_array_types_from_spectrum_source(&mut reader)
            .sample_array_types_from_chromatograms(reader.iter_chromatograms());

        let arc = tempfile::NamedTempFile::with_suffix("_test_chunk_numpress.mzpeak")?;
        let source = arc.as_file().try_clone()?;
        let mut writer = builder.build(source, true);
        writer.copy_metadata_from(&reader);
        let overrides = writer.spectrum_buffers.overrides();
        assert!(overrides.iter().any(|(_, v)| {
            v.buffer_priority
                .is_some_and(|v| matches!(v, BufferPriority::Primary))
        }));

        assert!(writer.spectrum_buffers.fields().iter().any(|f| f.name() == "mz_chunk_values"));
        assert!(writer.spectrum_buffers.fields().iter().any(|f| f.name() == "mz_numpress_linear_bytes"));
        assert!(writer.spectrum_buffers.fields().iter().any(|f| f.name() == "intensity_numpress_slof_bytes"));

        writer.write_all_owned(reader.iter())?;
        for chrom in reader.iter_chromatograms() {
            writer.write_chromatogram(&chrom)?;
        }
        drop(writer);

        let mut new_reader = MzPeakReader::new(arc.path())?;
        let array_indices = new_reader.metadata.spectrum_array_indices();
        for arr in array_indices.iter() {
            if arr.path.ends_with("mz_numpress_linear_bytes") {
                assert!(matches!(arr.buffer_format, BufferFormat::ChunkTransform));
            }
            else if arr.path.ends_with("intensity_numpress_slof_bytes") {
                assert!(matches!(arr.buffer_format, BufferFormat::ChunkTransform));
            }
            else if arr.path.ends_with("mz_chunk_values") {
                assert!(matches!(arr.buffer_format, BufferFormat::Chunk));
            }
        }
        assert_eq!(reader.len(), new_reader.len());
        reader.reset();
        new_reader.reset();
        for (a, b)  in reader.iter().zip(new_reader.iter()) {
            assert_eq!(a.id(), b.id());
        }
        Ok(())
    }
}
