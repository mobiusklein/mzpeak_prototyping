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
    arrow::{ArrowSchemaConverter, ArrowWriter, arrow_writer::ArrowWriterOptions},
    basic::{Compression, Encoding, ZstdLevel},
    file::properties::{
        DEFAULT_DICTIONARY_PAGE_SIZE_LIMIT, EnabledStatistics, WriterProperties, WriterVersion,
    },
    format::{KeyValue, SortingColumn},
};

use mzdata::{
    io::{MZReaderType, StreamingSpectrumIterator},
    meta::{FileMetadataConfig, MSDataFileMetadata},
    prelude::*,
    spectrum::{
        MultiLayerSpectrum, RefPeakDataLevel,
        SignalContinuity,
    },
};
use serde_arrow::schema::{SchemaLike, TracingOptions};

use crate::{
    archive::ZipArchiveWriter,
    entry::Entry,
    peak_series::{
        ArrayIndex, BufferContext, BufferName, ToMzPeakDataSeries, array_map_to_schema_arrays,
    },
};
use crate::{
    chunk_series::{ArrowArrayChunk, ChunkingStrategy},
    peak_series::MZ_ARRAY,
    spectrum::AuxiliaryArray,
};

mod array_buffer;
mod split;
mod builder;
mod base;

pub use array_buffer::{
    ArrayBufferWriter, ArrayBufferWriterVariants, ArrayBuffersBuilder, ChunkBuffers, PointBuffers,
};
pub use split::MzPeakSplitWriter;
pub use builder::MzPeakWriterBuilder;
pub use base::AbstractMzPeakWriter;

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

macro_rules! implement_mz_metadata {
    () => {
        pub(crate) fn append_metadata(&mut self) {
            self.append_key_value_metadata(
                "file_description",
                Some(
                    serde_json::to_string_pretty(&$crate::param::FileDescription::from(
                        self.mz_metadata.file_description(),
                    ))
                    .unwrap(),
                ),
            );
            let tmp: Vec<_> = self
                .mz_metadata
                .instrument_configurations()
                .values()
                .map(|v| $crate::param::InstrumentConfiguration::from(v))
                .collect();
            self.append_key_value_metadata(
                "instrument_configuration_list",
                Some(serde_json::to_string_pretty(&tmp).unwrap()),
            );

            let tmp: Vec<_> = self
                .mz_metadata
                .data_processings()
                .iter()
                .map(|v| $crate::param::DataProcessing::from(v))
                .collect();
            self.append_key_value_metadata(
                "data_processing_method_list",
                Some(serde_json::to_string_pretty(&tmp).unwrap()),
            );

            let tmp: Vec<_> = self
                .mz_metadata
                .softwares()
                .iter()
                .map(|v| $crate::param::Software::from(v))
                .collect();
            self.append_key_value_metadata(
                "software_list",
                Some(serde_json::to_string_pretty(&tmp).unwrap()),
            );

            let tmp: Vec<_> = self
                .mz_metadata
                .samples()
                .iter()
                .map(|v| $crate::param::Sample::from(v))
                .collect();
            self.append_key_value_metadata(
                "sample_list",
                Some(serde_json::to_string_pretty(&tmp).unwrap()),
            );

            self.append_key_value_metadata(
                "run",
                Some(
                    serde_json::to_string_pretty(self.mz_metadata.run_description().unwrap())
                        .unwrap(),
                ),
            )
        }
    };
}

pub(crate) use implement_mz_metadata;

//// A small helper for writing peak list data to another stream with very narrow options.
pub(crate) struct MiniPeakWriterType<W: Write + Send + Seek> {
    writer: ArrowWriter<W>,
    spectrum_buffers: PointBuffers,
    buffer_size: usize,
}

impl<W: Write + Send + Seek> MiniPeakWriterType<W> {
    pub(crate) fn new(
        writer: ArrowWriter<W>,
        spectrum_buffers: PointBuffers,
        buffer_size: usize,
    ) -> Self {
        let mut this = Self {
            writer,
            spectrum_buffers,
            buffer_size,
        };
        let spectrum_array_index: ArrayIndex = this.spectrum_buffers.as_array_index();
        this.append_key_value_metadata(
            "spectrum_array_index".to_string(),
            Some(spectrum_array_index.to_json()),
        );
        this
    }

    pub(crate) fn append_key_value_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<Option<String>>,
    ) {
        self.writer
            .append_key_value_metadata(KeyValue::new(key.into(), value));
    }

    pub(crate) fn add_peaks<
        C: CentroidLike + ToMzPeakDataSeries,
        D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
    >(
        &mut self,
        spectrum_count: u64,
        peaks: RefPeakDataLevel<C, D>,
    ) -> io::Result<()> {
        match peaks {
            RefPeakDataLevel::Centroid(peaks) => {
                self.spectrum_buffers.add(spectrum_count, peaks.as_slice());
            }
            RefPeakDataLevel::Deconvoluted(peaks) => {
                self.spectrum_buffers.add(spectrum_count, peaks.as_slice());
            }
            RefPeakDataLevel::Missing => unimplemented!(),
            RefPeakDataLevel::RawData(_) => unimplemented!(),
        }

        if self.spectrum_buffers.len() >= self.buffer_size {
            self.flush()?;
        }
        Ok(())
    }

    pub(crate) fn flush(&mut self) -> io::Result<()> {
        for batch in self.spectrum_buffers.drain() {
            self.writer.write(&batch)?;
        }
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<W, parquet::errors::ParquetError> {
        self.flush()?;
        self.writer.into_inner()
    }
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

    fn write_spectrum_data<
        CI: ToMzPeakDataSeries + CentroidLike,
        DI: ToMzPeakDataSeries + DeconvolutedCentroidLike,
    >(
        &mut self,
        spectrum: &impl SpectrumLike<CI, DI>,
    ) -> io::Result<(Option<Vec<f64>>, Option<Vec<AuxiliaryArray>>)> {
        let spectrum_count = self.spectrum_counter();

        let peaks = spectrum.peaks();

        let (delta_params, aux_arrays) = if self.separate_peak_writer.is_some()
            && matches!(
                spectrum.peaks(),
                RefPeakDataLevel::Centroid(_) | RefPeakDataLevel::Deconvoluted(_)
            )
            && spectrum.raw_arrays().is_some()
            && spectrum.signal_continuity() == SignalContinuity::Profile
        {
            log::trace!("Writing both profile signal and peaks for {spectrum_count}");
            let raw_arrays = spectrum.raw_arrays().unwrap();
            let (delta_params, aux_arrays) =
                self.write_binary_array_map(spectrum, spectrum_count, raw_arrays)?;
            self.separate_peak_writer
                .as_mut()
                .unwrap()
                .add_peaks(spectrum_count, peaks)?;
            (delta_params, aux_arrays)
        } else {
            let (delta_params, aux_arrays) = match peaks {
                mzdata::spectrum::RefPeakDataLevel::Missing => (None, None),
                mzdata::spectrum::RefPeakDataLevel::RawData(binary_array_map) => {
                    self.write_binary_array_map(spectrum, spectrum_count, binary_array_map)?
                }
                mzdata::spectrum::RefPeakDataLevel::Centroid(peaks) => {
                    self.write_peaks(spectrum_count, peaks.as_slice())?
                }
                mzdata::spectrum::RefPeakDataLevel::Deconvoluted(peaks) => {
                    self.write_peaks(spectrum_count, peaks.as_slice())?
                }
            };
            (delta_params, aux_arrays)
        };

        Ok((delta_params, aux_arrays))
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

    fn spectrum_data_writer_props(
        data_buffer: &impl ArrayBufferWriter,
        index_path: String,
        shuffle_mz: bool,
        use_chunked_encoding: &Option<ChunkingStrategy>,
        compression: Compression,
    ) -> WriterProperties {
        let parquet_schema = Arc::new(
            ArrowSchemaConverter::new()
                .convert(&data_buffer.schema())
                .unwrap(),
        );

        let mut sorted = Vec::new();
        for (i, c) in parquet_schema.columns().iter().enumerate() {
            match c.path().string().as_ref() {
                x if x == index_path => {
                    sorted.push(SortingColumn::new(i as i32, false, false));
                }
                _ => {}
            }
        }

        let mut data_props = WriterProperties::builder()
            .set_compression(compression)
            .set_dictionary_enabled(true)
            .set_sorting_columns(Some(sorted))
            .set_column_encoding(index_path.into(), Encoding::RLE)
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_statistics_enabled(EnabledStatistics::Page);

        if use_chunked_encoding.is_some() {
            data_props = data_props.set_max_row_group_size(1024 * 100)
        }

        for c in parquet_schema.columns().iter() {
            let colpath = c.path().to_string();
            if colpath.contains("_mz_")
                && shuffle_mz
                && matches!(
                    c.physical_type(),
                    parquet::basic::Type::DOUBLE | parquet::basic::Type::FLOAT
                )
            {
                log::debug!("{}: shuffling", c.path());
                data_props =
                    data_props.set_column_encoding(c.path().clone(), Encoding::BYTE_STREAM_SPLIT);
            }
            if colpath.contains("_ion_mobility") {
                log::debug!(
                    "{}: ion mobility detected, increasing dictionary size",
                    c.path()
                );
                data_props = data_props
                    .set_dictionary_page_size_limit(DEFAULT_DICTIONARY_PAGE_SIZE_LIMIT * 2);
            }
            if c.name().ends_with("_index") {
                log::debug!("{}: delta binary packing", c.path());
                data_props =
                    data_props.set_column_encoding(c.path().clone(), Encoding::DELTA_BINARY_PACKED);
            }
        }

        let data_props = data_props.build();
        data_props
    }

    fn metadata_writer_props(metadata_fields: &SchemaRef) -> WriterProperties {
        let parquet_schema = Arc::new(
            ArrowSchemaConverter::new()
                .convert(&metadata_fields)
                .unwrap(),
        );

        let mut sorted = Vec::new();
        for (i, c) in parquet_schema.columns().iter().enumerate() {
            match c.path().string().as_ref() {
                "spectrum.index" => {
                    sorted.push(SortingColumn::new(i as i32, false, false));
                }
                _ => {}
            }
        }
        let metadata_props = WriterProperties::builder()
            .set_compression(parquet::basic::Compression::ZSTD(
                ZstdLevel::try_new(3).unwrap(),
            ))
            .set_dictionary_enabled(true)
            .set_sorting_columns(Some(sorted))
            .set_column_bloom_filter_enabled("spectrum.id".into(), true)
            .set_writer_version(WriterVersion::PARQUET_2_0)
            .set_statistics_enabled(EnabledStatistics::Page)
            .build();

        metadata_props
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
            format!("{}.spectrum_index", spectrum_buffers.prefix()),
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
                format!("{}.spectrum_index", peak_buffer.prefix()),
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
                writer.add_file_from_read(&mut peak_file, Some(&"peaks.mzpeak"))?;
            }

            writer.start_spectrum_metadata().unwrap();
            self.archive_writer = Some(ArrowWriter::try_new_with_options(
                writer,
                self.metadata_fields.clone(),
                ArrowWriterOptions::new()
                    .with_properties(Self::metadata_writer_props(&self.metadata_fields)),
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
