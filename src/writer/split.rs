use std::{fs, io, marker::PhantomData, path::PathBuf, sync::Arc};

use arrow::{
    array::{ArrayBuilder, AsArray, RecordBatch},
    datatypes::{FieldRef, Schema, SchemaRef},
};
use mzpeaks::{CentroidPeak, DeconvolutedPeak};
use parquet::{
    arrow::{ArrowWriter, arrow_writer::ArrowWriterOptions},
    basic::Compression,
    format::KeyValue,
};

use mzdata::{meta::FileMetadataConfig, prelude::*};

use crate::{
    BufferContext, ToMzPeakDataSeries,
    archive::MzPeakArchiveType,
    chunk_series::ChunkingStrategy,
    peak_series::ArrayIndex,
    writer::{
        AbstractMzPeakWriter, ArrayBufferWriter, ArrayBufferWriterVariants, ArrayBuffersBuilder,
        ChromatogramBuilder, MiniPeakWriterType, SpectrumBuilder, VisitorBase,
        implement_mz_metadata,
    },
};

/// Writer for the MzPeak format that writes the different data types to separate files
/// in an unarchived format.
pub struct UnpackedMzPeakWriterType<
    C: CentroidLike + ToMzPeakDataSeries = CentroidPeak,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries = DeconvolutedPeak,
> {
    path: PathBuf,

    spectrum_data_writer: ArrowWriter<fs::File>,
    spectrum_metadata_writer: ArrowWriter<fs::File>,

    spectrum_buffers: ArrayBufferWriterVariants,
    chromatogram_buffers: ArrayBufferWriterVariants,
    separate_peak_writer: Option<MiniPeakWriterType<fs::File>>,

    spectrum_metadata_buffer: SpectrumBuilder,
    chromatogram_metadata_buffer: ChromatogramBuilder,

    spectrum_data_point_counter: u64,

    #[allow(unused)]
    chromatogram_data_point_counter: u64,

    buffer_size: usize,
    compression: Compression,
    mz_metadata: FileMetadataConfig,
    _t: PhantomData<(C, D)>,
}

impl<C: CentroidLike + ToMzPeakDataSeries, D: DeconvolutedCentroidLike + ToMzPeakDataSeries> Drop
    for UnpackedMzPeakWriterType<C, D>
{
    fn drop(&mut self) {
        if let Err(e) = self.finish() {
            log::trace!("While dropping UnpackedMzPeakWriterType: {e}")
        }
    }
}

impl<C: CentroidLike + ToMzPeakDataSeries, D: DeconvolutedCentroidLike + ToMzPeakDataSeries>
    MSDataFileMetadata for UnpackedMzPeakWriterType<C, D>
{
    mzdata::delegate_impl_metadata_trait!(mz_metadata);
}

impl<C: CentroidLike + ToMzPeakDataSeries, D: DeconvolutedCentroidLike + ToMzPeakDataSeries>
    AbstractMzPeakWriter for UnpackedMzPeakWriterType<C, D>
{
    fn append_key_value_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<Option<String>>,
    ) {
        self.append_key_value_metadata(key, value);
    }

    fn spectrum_counter(&self) -> u64 {
        self.spectrum_metadata_buffer.index_counter()
    }

    fn spectrum_data_buffer_mut(&mut self) -> &mut ArrayBufferWriterVariants {
        &mut self.spectrum_buffers
    }

    fn spectrum_entry_buffer_mut(&mut self) -> &mut SpectrumBuilder {
        &mut self.spectrum_metadata_buffer
    }

    fn check_data_buffer(&mut self) -> io::Result<()> {
        if self.spectrum_counter() % (self.buffer_size as u64) == 0 {
            self.flush_spectrum_data_arrays()?;
        }
        Ok(())
    }

    fn separate_peak_writer(&mut self) -> Option<&mut MiniPeakWriterType<fs::File>> {
        self.separate_peak_writer.as_mut()
    }

    fn spectrum_precursor_counter(&self) -> u64 {
        self.spectrum_metadata_buffer.precursor_index_counter()
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

impl<C: CentroidLike + ToMzPeakDataSeries, D: DeconvolutedCentroidLike + ToMzPeakDataSeries>
    UnpackedMzPeakWriterType<C, D>
{
    pub fn new(
        path: PathBuf,
        spectrum_buffers_builder: ArrayBuffersBuilder,
        chromatogram_buffers_builder: ArrayBuffersBuilder,
        buffer_size: usize,
        mask_zero_intensity_runs: bool,
        shuffle_mz: bool,
        use_chunked_encoding: Option<ChunkingStrategy>,
        compression: Compression,
        store_peaks_and_profiles_apart: Option<ArrayBuffersBuilder>,
    ) -> Self {
        let data_writer_path = path.join(MzPeakArchiveType::SpectrumDataArrays.tag_file_suffix());
        let metadata_writer_path = path.join(MzPeakArchiveType::SpectrumMetadata.tag_file_suffix());

        let spectrum_data_writer = fs::File::create(data_writer_path).unwrap();
        let spectrum_metadata_writer = fs::File::create(metadata_writer_path).unwrap();

        let spectrum_metadata_buffer = SpectrumBuilder::default();

        let fields: Vec<FieldRef> = spectrum_metadata_buffer.fields();
        let metadata_fields: SchemaRef = Arc::new(Schema::new(fields));
        let spectrum_buffers = spectrum_buffers_builder.build(
            Arc::new(Schema::empty()),
            BufferContext::Spectrum,
            mask_zero_intensity_runs,
        );

        let chromatogram_buffers =
            ArrayBufferWriterVariants::PointBuffers(chromatogram_buffers_builder.build(
                Arc::new(Schema::empty()),
                BufferContext::Chromatogram,
                false,
            ));

        let data_props = Self::spectrum_data_writer_props(
            &spectrum_buffers,
            spectrum_buffers.index_path(),
            shuffle_mz,
            &use_chunked_encoding,
            compression,
        );

        let separate_peak_writer = if let Some(peak_buffer_builder) = store_peaks_and_profiles_apart
        {
            let peak_buffer_file = fs::File::create(
                path.join(MzPeakArchiveType::SpectrumPeakDataArrays.tag_file_suffix()),
            )
            .unwrap();
            let peak_buffer = peak_buffer_builder
                .include_time(spectrum_buffers.include_time())
                .build(Arc::new(Schema::empty()), BufferContext::Spectrum, false);

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

        let metadata_props = Self::spectrum_metadata_writer_props(&metadata_fields);

        let mut this = Self {
            path,
            spectrum_data_writer: ArrowWriter::try_new_with_options(
                spectrum_data_writer,
                spectrum_buffers.schema.clone(),
                ArrowWriterOptions::new().with_properties(data_props),
            )
            .unwrap(),
            spectrum_metadata_writer: ArrowWriter::try_new_with_options(
                spectrum_metadata_writer,
                metadata_fields.clone(),
                ArrowWriterOptions::new().with_properties(metadata_props),
            )
            .unwrap(),
            separate_peak_writer,
            spectrum_metadata_buffer,
            spectrum_buffers: spectrum_buffers.into(),
            chromatogram_buffers,
            chromatogram_metadata_buffer: Default::default(),
            spectrum_data_point_counter: 0,

            chromatogram_data_point_counter: 0,
            compression,
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
        self.spectrum_data_writer
            .append_key_value_metadata(KeyValue::new(
                "spectrum_array_index".to_string(),
                spectrum_array_index.to_json(),
            ));

        // let mut chromatogram_array_index =
        //     ArrayIndex::new(self.chromatogram_buffers.prefix.clone(), HashMap::new());
        // if let Ok(sub) = self
        //     .chromatogram_buffers
        //     .schema
        //     .field_with_name(&self.chromatogram_buffers.prefix)
        //     .cloned()
        // {
        //     if let DataType::Struct(fields) = sub.data_type() {
        //         for f in fields.iter() {
        //             if f.name() == "chromatogram_index" {
        //                 continue;
        //             }
        //             let buffer_name =
        //                 BufferName::from_field(BufferContext::Chromatogram, f.clone()).unwrap();
        //             let aie = ArrayIndexEntry::from_buffer_name(
        //                 self.chromatogram_buffers.prefix.clone(),
        //                 buffer_name,
        //             );
        //             chromatogram_array_index.insert(aie.array_type.clone(), aie);
        //         }
        //     }
        // }
        // self.append_key_value_metadata(
        //     "chromatogram_array_index".to_string(),
        //     chromatogram_array_index.to_json(),
        // );
    }

    pub fn append_key_value_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<Option<String>>,
    ) {
        self.spectrum_metadata_writer
            .append_key_value_metadata(KeyValue::new(key.into(), value));
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

    fn flush_spectrum_data_arrays(&mut self) -> io::Result<()> {
        for batch in self.spectrum_buffers.drain() {
            self.spectrum_data_point_counter += batch.num_rows() as u64;
            self.spectrum_data_writer.write(&batch)?;
        }
        Ok(())
    }

    fn flush_chromatogram_data_records<W: Write + Send>(
        &mut self,
        writer: &mut ArrowWriter<W>,
    ) -> io::Result<()> {
        for batch in self.chromatogram_buffers.drain() {
            self.chromatogram_data_point_counter += batch.num_rows() as u64;
            writer.write(&batch)?;
        }
        Ok(())
    }

    fn flush_spectrum_metadata_records(&mut self) -> io::Result<()> {
        let batch = RecordBatch::from(self.spectrum_metadata_buffer.finish().as_struct());
        self.spectrum_metadata_writer.write(&batch)?;
        Ok(())
    }

    fn flush_chromatogram_metadata_records<W: Write + Send>(
        &mut self,
        writer: &mut ArrowWriter<W>,
    ) -> io::Result<()> {
        let batch = RecordBatch::from(self.chromatogram_metadata_buffer.finish().as_struct());
        writer.write(&batch)?;
        Ok(())
    }

    pub fn finish(
        &mut self,
    ) -> Result<parquet::format::FileMetaData, parquet::errors::ParquetError> {
        self.flush_spectrum_data_arrays()?;
        self.flush_spectrum_metadata_records()?;
        self.append_metadata();
        self.append_key_value_metadata("spectrum_count", Some(self.spectrum_counter().to_string()));
        self.append_key_value_metadata(
            "spectrum_data_point_count",
            Some(self.spectrum_data_point_counter.to_string()),
        );
        self.spectrum_data_writer.finish()?;

        if let Some(peak_file_writer) = self.separate_peak_writer.take() {
            let peak_file = peak_file_writer.finish()?;
            drop(peak_file);
        }

        if !self.chromatogram_metadata_buffer.is_empty() {
            let metadata_fields = self.chromatogram_metadata_buffer.schema();
            let mut writer = ArrowWriter::try_new_with_options(
                fs::File::create(
                    self.path
                        .join(MzPeakArchiveType::ChromatogramMetadata.tag_file_suffix()),
                )?,
                metadata_fields.clone(),
                ArrowWriterOptions::new()
                    .with_properties(Self::spectrum_metadata_writer_props(&metadata_fields)),
            )?;
            self.flush_chromatogram_metadata_records(&mut writer)?;
            self.append_key_value_metadata(
                "chromatogram_count",
                Some(self.chromatogram_counter().to_string()),
            );
            self.append_key_value_metadata(
                "chromatogram_data_point_count",
                Some(self.chromatogram_data_point_counter.to_string()),
            );
            writer.finish()?;

            let mut writer = ArrowWriter::try_new_with_options(
                fs::File::create(
                    self.path
                        .join(MzPeakArchiveType::ChromatogramDataArrays.tag_file_suffix()),
                )?,
                self.chromatogram_buffers.schema().clone(),
                ArrowWriterOptions::new().with_properties(Self::chromatogram_data_writer_props(
                    &self.chromatogram_buffers,
                    BufferContext::Chromatogram.index_field().name().to_string(),
                    &None,
                    self.compression,
                )),
            )?;

            self.flush_chromatogram_data_records(&mut writer)?;
            let chromatogram_array_index: ArrayIndex = self.chromatogram_buffers.as_array_index();
            writer.append_key_value_metadata(KeyValue::new(
                "chromatogram_array_index".to_string(),
                Some(chromatogram_array_index.to_json()),
            ));
            writer.append_key_value_metadata(KeyValue::new(
                "chromatogram_data_point_count".into(),
                Some(self.chromatogram_data_point_counter.to_string()),
            ));
        }
        self.spectrum_metadata_writer.finish()
    }
}

impl<C: CentroidLike + ToMzPeakDataSeries, D: DeconvolutedCentroidLike + ToMzPeakDataSeries>
    SpectrumWriter<C, D> for UnpackedMzPeakWriterType<C, D>
{
    fn write<S: SpectrumLike<C, D> + 'static>(&mut self, spectrum: &S) -> io::Result<usize> {
        self.write_spectrum(spectrum).map(|_| 1)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.spectrum_data_writer.flush()?;
        self.spectrum_metadata_writer.flush()?;
        Ok(())
    }

    fn close(&mut self) -> io::Result<()> {
        self.finish()?;
        Ok(())
    }
}
