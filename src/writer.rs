use std::{
    collections::HashMap,
    fs,
    io::{self, prelude::*},
    marker::PhantomData,
    sync::Arc,
};

use arrow::{array::{ArrayBuilder, AsArray, RecordBatch}, datatypes::{Field, Schema}};
use mzpeaks::{CentroidPeak, DeconvolutedPeak};
use parquet::{
    arrow::{ArrowWriter, arrow_writer::ArrowWriterOptions},
    basic::Compression,
    format::KeyValue,
};

use mzdata::{
    io::{MZReaderType, StreamingSpectrumIterator},
    meta::{FileMetadataConfig, MSDataFileMetadata},
    prelude::*,
    spectrum::{ArrayType, Chromatogram, MultiLayerSpectrum, SignalContinuity},
};

use crate::{
    archive::{MzPeakArchiveType, ZipArchiveWriter},
    peak_series::{
        ArrayIndex, BufferContext, BufferName, ToMzPeakDataSeries, array_map_to_schema_arrays,
    },
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
pub use builder::MzPeakWriterBuilder;
pub use split::MzPeakSplitWriter;

pub use visitor::{
    ScanBuilder,
    ScanWindowBuilder,
    SelectedIonBuilder,
    ActivationBuilder,
    IsolationWindowBuilder,
    PrecursorBuilder,
    CURIEBuilder,
    ParamValueBuilder,
    ParamBuilder,
    CustomBuilderFromParameter,
    ParamListBuilder,
    AuxiliaryArrayBuilder,
    SpectrumDetailsBuilder,
    SpectrumBuilder,
    StructVisitor,
    VisitorBase,
    StructVisitorBuilder,
    SpectrumVisitor,
    inflect_cv_term_to_column_name,
    ChromatogramDetailsBuilder,
    ChromatogramBuilder,
};

pub(crate) use base::implement_mz_metadata;
pub(crate) use mini_peak::MiniPeakWriterType;

fn _eval_spectra_from_iter_for_fields<
    C: CentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<CentroidPeak>,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<DeconvolutedPeak>,
>(
    iter: impl Iterator<Item = MultiLayerSpectrum<C, D>>,
    overrides: &HashMap<BufferName, BufferName>,
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
            s.raw_arrays().and_then(|map| {
                if let Some(use_chunked_encoding) = use_chunked_encoding {
                    ArrowArrayChunk::from_arrays(
                        0,
                        None,
                        MZ_ARRAY,
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
        .map(|(fields, _arrs)| {
            let fields: Vec<_> = fields.iter().cloned().collect();
            fields
        })
        .flatten();

    for field in field_it {
        if arrays.iter().find(|f| f.name() == field.name()).is_none() {
            arrays.push(field);
        }
    }
    if is_profile > 0 {
        log::info!("Detected profile spectra");
    }
    arrays
}

pub fn sample_array_types_from_chromatograms<I: Iterator<Item = Chromatogram>>(
    iter: I,
    overrides: &HashMap<BufferName, BufferName>,
) -> Vec<Arc<Field>> {
    let field_it = iter
        .flat_map(|s| {
            log::trace!("Sampling arrays from {}", s.id());
            Some(&s.arrays).and_then(|map| {
                {
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
        })
        .map(|(fields, _arrs)| {
            let fields: Vec<_> = fields.iter().cloned().collect();
            fields
        })
        .flatten();
    let mut arrays: Vec<Arc<Field>> = Vec::new();
    for field in field_it {
        if arrays.iter().find(|f| f.name() == field.name()).is_none() {
            arrays.push(field);
        }
    }
    arrays
}

pub fn sample_array_types_from_stream<
    C: CentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<CentroidPeak>,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<DeconvolutedPeak>,
    I: Iterator<Item = MultiLayerSpectrum<C, D>>,
>(
    reader: &mut StreamingSpectrumIterator<C, D, MultiLayerSpectrum<C, D>, I>,
    overrides: &HashMap<BufferName, BufferName>,
    use_chunked_encoding: Option<ChunkingStrategy>,
) -> Vec<std::sync::Arc<arrow::datatypes::Field>>
where
    MultiLayerSpectrum<C, D>: Clone,
{
    reader.populate_buffer(10);
    let fields = _eval_spectra_from_iter_for_fields(
        reader.iter_buffer().cloned(),
        overrides,
        use_chunked_encoding,
    );
    fields
}

pub fn sample_array_types_from_file_reader<
    C: CentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<CentroidPeak>,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries + BuildFromArrayMap + From<DeconvolutedPeak>,
>(
    reader: &mut MZReaderType<std::fs::File, C, D>,
    overrides: &HashMap<BufferName, BufferName>,
    use_chunked_encoding: Option<ChunkingStrategy>,
) -> Vec<Arc<arrow::datatypes::Field>> {
    let n = reader.len();
    if n == 0 {
        return Vec::new();
    }

    let it = [0, 100.min(n - 1), n / 2]
        .into_iter()
        .flat_map(|i| reader.get_spectrum_by_index(i));
    return _eval_spectra_from_iter_for_fields(it, overrides, use_chunked_encoding);
}

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

    spectrum_data_point_counter: u64,
    chromatogram_data_point_counter: u64,

    buffer_size: usize,
    compression: Compression,
    mz_metadata: FileMetadataConfig,
    _t: PhantomData<(C, D)>,
}

impl<
    W: Write + Send + Seek,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> AbstractMzPeakWriter for MzPeakWriterType<W, C, D>
{
    fn append_key_value_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<Option<String>>,
    ) {
        self.archive_writer
            .as_mut()
            .unwrap()
            .append_key_value_metadata(KeyValue::new(key.into(), value));
    }

    fn use_chunked_encoding(&self) -> Option<&ChunkingStrategy> {
        self.use_chunked_encoding.as_ref()
    }

    fn spectrum_entry_buffer_mut(&mut self) -> &mut SpectrumBuilder {
        &mut self.spectrum_metadata_buffer
    }

    fn spectrum_data_buffer_mut(&mut self) -> &mut ArrayBufferWriterVariants {
        &mut self.spectrum_buffers
    }

    fn check_data_buffer(&mut self) -> io::Result<()> {
        if self.spectrum_counter() % (self.buffer_size as u64) == 0 {
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
        compression: Compression,
        store_peaks_and_profiles_apart: Option<ArrayBuffersBuilder>,
    ) -> Self {

        let spectrum_metadata_buffer = SpectrumBuilder::default();

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

        let chromatogram_buffers =
            ArrayBufferWriterVariants::PointBuffers(chromatogram_buffers_builder.build(
                Arc::new(Schema::empty()),
                BufferContext::Chromatogram,
                false,
            ));

        let mut writer = ZipArchiveWriter::new(writer);
        writer.start_spectrum_data().unwrap();

        let data_props = Self::spectrum_data_writer_props(
            &spectrum_buffers,
            spectrum_buffers.index_path(),
            shuffle_mz,
            &use_chunked_encoding,
            compression,
        );

        let separate_peak_writer = if let Some(peak_buffer_builder) = store_peaks_and_profiles_apart
        {
            let peak_buffer_file =
                tempfile::tempfile().expect("Failed to create temporary file to write peaks to");
            let peak_buffer = peak_buffer_builder.include_time(spectrum_buffers.include_time()).build(
                Arc::new(Schema::empty()),
                BufferContext::Spectrum,
                false,
            );

            let data_props = Self::spectrum_data_writer_props(
                &peak_buffer,
                peak_buffer.index_path(),
                shuffle_mz,
                &None,
                compression,
            );

            let peak_writer = ArrowWriter::try_new_with_options(
                peak_buffer_file,
                peak_buffer.schema().clone(),
                ArrowWriterOptions::new().with_properties(data_props),
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
            spectrum_metadata_buffer,
            spectrum_buffers,
            chromatogram_buffers,
            spectrum_data_point_counter: 0,
            chromatogram_data_point_counter: 0,
            chromatogram_metadata_buffer: Default::default(),
            buffer_size: buffer_size,
            mz_metadata: Default::default(),
            compression,
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
            spectrum_array_index.to_json(),
        );
    }

    fn add_chromatogram_array_metadata(&mut self) {
        let chromatogram_array_index: ArrayIndex = self.chromatogram_buffers.as_array_index();
        self.append_key_value_metadata(
            "chromatogram_array_index".to_string(),
            chromatogram_array_index.to_json(),
        );
    }

    pub fn write_spectrum<
        A: ToMzPeakDataSeries + CentroidLike,
        B: ToMzPeakDataSeries + DeconvolutedCentroidLike,
        S: SpectrumLike<A, B> + 'static
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
        // let batch = serde_arrow::to_record_batch(
        //     &self.metadata_fields.fields(),
        //     &self.spectrum_metadata_buffer,
        // )
        // .unwrap();
        let arrays = self.spectrum_metadata_buffer.finish();
        let batch = RecordBatch::from(arrays.as_struct());
        self.archive_writer.as_mut().unwrap().write(&batch)?;
        // self.spectrum_metadata_buffer.clear();
        Ok(())
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

    pub fn finish(&mut self) -> Result<(), parquet::errors::ParquetError> {
        if self.archive_writer.is_some() {
            self.flush_data_arrays()?;
            self.append_key_value_metadata(
                "spectrum_count",
                Some(self.spectrum_counter().to_string()),
            );
            self.append_key_value_metadata(
                "spectrum_data_point_count",
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
                "spectrum_count",
                Some(self.spectrum_counter().to_string()),
            );
            self.append_key_value_metadata(
                "spectrum_data_point_count",
                Some(self.spectrum_data_point_counter.to_string()),
            );

            writer = self.archive_writer.take().unwrap().into_inner()?;

            if !self.chromatogram_metadata_buffer.is_empty() {
                writer.start_chromatogram_metadata().unwrap();
                let metadata_fields = self.chromatogram_metadata_buffer.schema();
                self.archive_writer = Some(ArrowWriter::try_new_with_options(
                    writer,
                    metadata_fields.clone(),
                    ArrowWriterOptions::new().with_properties(
                        Self::spectrum_metadata_writer_props(&metadata_fields),
                    ),
                )?);
                self.flush_chromatogram_metadata_records()?;
                self.append_key_value_metadata(
                    "chromatogram_count",
                    Some(self.chromatogram_counter().to_string()),
                );
                self.append_key_value_metadata(
                    "chromatogram_data_point_count",
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
                    "chromatogram_data_point_count",
                    Some(self.chromatogram_data_point_counter.to_string()),
                );
                self.append_metadata();
                writer = self.archive_writer.take().unwrap().into_inner()?;
            }

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
        if let Err(_) = self.finish() {}
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
