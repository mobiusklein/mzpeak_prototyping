use std::{
    collections::HashMap,
    fs,
    io::{self, prelude::*},
    marker::PhantomData,
    sync::Arc,
};

use arrow::{
    datatypes::{Field, FieldRef, Schema, SchemaRef},
};
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
    spectrum::{
        MultiLayerSpectrum,
        SignalContinuity,
    },
};
use serde_arrow::schema::{SchemaLike, TracingOptions};

use crate::{
    archive::{MzPeakArchiveType, ZipArchiveWriter},
    entry::Entry,
    peak_series::{
        array_map_to_schema_arrays, ArrayIndex, BufferContext, BufferName, ToMzPeakDataSeries
    },
};
use crate::{
    chunk_series::{ArrowArrayChunk, ChunkingStrategy},
    peak_series::MZ_ARRAY,
};

mod array_buffer;
mod split;
mod builder;
mod base;
mod mini_peak;

pub use array_buffer::{
    ArrayBufferWriter, ArrayBufferWriterVariants, ArrayBuffersBuilder, ChunkBuffers, PointBuffers,
};
pub use split::MzPeakSplitWriter;
pub use builder::MzPeakWriterBuilder;
pub use base::AbstractMzPeakWriter;

pub(crate) use mini_peak::MiniPeakWriterType;
pub(crate) use base::implement_mz_metadata;


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
                                    "spectrum_index",
                                    &[
                                        use_chunked_encoding,
                                        ChunkingStrategy::Basic { chunk_size: 50.0 },
                                    ],
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
                        "spectrum_index",
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

    #[allow(unused)]
    chromatogram_buffers: PointBuffers,
    metadata_buffer: Vec<Entry>,
    use_chunked_encoding: Option<ChunkingStrategy>,
    metadata_fields: SchemaRef,

    spectrum_counter: u64,
    spectrum_precursor_counter: u64,
    spectrum_data_point_counter: u64,
    buffer_size: usize,

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

    fn spectrum_entry_buffer_mut(&mut self) -> &mut Vec<Entry> {
        &mut self.metadata_buffer
    }

    fn spectrum_data_buffer_mut(&mut self) -> &mut ArrayBufferWriterVariants {
        &mut self.spectrum_buffers
    }

    fn check_data_buffer(&mut self) -> io::Result<()> {
        if self.spectrum_counter % (self.buffer_size as u64) == 0 {
            self.flush_data_arrays()?;
        }
        Ok(())
    }

    fn spectrum_counter(&self) -> u64 {
        self.spectrum_counter
    }

    fn spectrum_counter_mut(&mut self) -> &mut u64 {
        &mut self.spectrum_counter
    }

    fn spectrum_precursor_counter(&self) -> u64 {
        self.spectrum_precursor_counter
    }

    fn spectrum_precursor_counter_mut(&mut self) -> &mut u64 {
        &mut self.spectrum_precursor_counter
    }

    fn separate_peak_writer(&mut self) -> Option<&mut MiniPeakWriterType<fs::File>> {
        self.separate_peak_writer.as_mut()
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
        let fields: Vec<FieldRef> = SchemaLike::from_type::<Entry>(TracingOptions::new()).unwrap();
        let metadata_fields: SchemaRef = Arc::new(Schema::new(fields));

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

        let chromatogram_buffers = chromatogram_buffers_builder.build(
            Arc::new(Schema::empty()),
            BufferContext::Chromatogram,
            false,
        );

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
            let peak_buffer = peak_buffer_builder.build(
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
            metadata_fields,
            metadata_buffer: Vec::new(),
            spectrum_buffers,
            chromatogram_buffers,
            spectrum_counter: 0,
            spectrum_precursor_counter: 0,
            spectrum_data_point_counter: 0,
            buffer_size: buffer_size,
            mz_metadata: Default::default(),
            _t: PhantomData,
        };
        this.add_array_metadata();
        this
    }

    implement_mz_metadata!();

    fn add_array_metadata(&mut self) {
        let spectrum_array_index: ArrayIndex = self.spectrum_buffers.as_array_index();
        self.append_key_value_metadata(
            "spectrum_array_index".to_string(),
            spectrum_array_index.to_json(),
        );
    }

    pub fn write_spectrum<
        A: ToMzPeakDataSeries + CentroidLike,
        B: ToMzPeakDataSeries + DeconvolutedCentroidLike,
    >(
        &mut self,
        spectrum: &impl SpectrumLike<A, B>,
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

    fn flush_metadata_records(&mut self) -> io::Result<()> {
        let batch =
            serde_arrow::to_record_batch(&self.metadata_fields.fields(), &self.metadata_buffer)
                .unwrap();
        self.archive_writer.as_mut().unwrap().write(&batch)?;
        self.metadata_buffer.clear();
        Ok(())
    }

    pub fn finish(&mut self) -> Result<(), parquet::errors::ParquetError> {
        if self.archive_writer.is_some() {
            self.flush_data_arrays()?;
            self.append_key_value_metadata(
                "spectrum_count",
                Some(self.spectrum_counter.to_string()),
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
                    Some(&MzPeakArchiveType::SpectrumPeakDataArrays.tag_file_suffix())
                )?;
            }

            writer.start_spectrum_metadata().unwrap();
            self.archive_writer = Some(ArrowWriter::try_new_with_options(
                writer,
                self.metadata_fields.clone(),
                ArrowWriterOptions::new()
                    .with_properties(Self::spectrum_metadata_writer_props(&self.metadata_fields)),
            )?);
            self.flush_metadata_records()?;
            self.append_metadata();
            self.append_key_value_metadata(
                "spectrum_count",
                Some(self.spectrum_counter.to_string()),
            );
            self.append_key_value_metadata(
                "spectrum_data_point_count",
                Some(self.spectrum_data_point_counter.to_string()),
            );

            let writer = self.archive_writer.take().unwrap().into_inner()?;
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
