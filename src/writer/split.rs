use std::{io, marker::PhantomData, sync::Arc};

use arrow::datatypes::{FieldRef, Schema, SchemaRef};
use mzpeaks::{CentroidPeak, DeconvolutedPeak};
use parquet::{
    arrow::{ArrowWriter, arrow_writer::ArrowWriterOptions},
    basic::Compression,
    format::KeyValue,
};

use mzdata::{meta::FileMetadataConfig, prelude::*};
use serde_arrow::schema::{SchemaLike, TracingOptions};

use crate::{
    chunk_series::ChunkingStrategy, entry::{ChromatogramMetadataEntry, Entry}, peak_series::ArrayIndex, writer::{
        implement_mz_metadata, AbstractMzPeakWriter, ArrayBufferWriter, ArrayBufferWriterVariants, ArrayBuffersBuilder
    }, BufferContext, ToMzPeakDataSeries
};

/// Writer for the MzPeak format that writes the different data types to separate files
/// in an unarchived format.
pub struct MzPeakSplitWriter<
    W: Write + Send,
    C: CentroidLike + ToMzPeakDataSeries = CentroidPeak,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries = DeconvolutedPeak,
> {
    spectrum_data_writer: ArrowWriter<W>,
    spectrum_metadata_writer: ArrowWriter<W>,

    spectrum_buffers: ArrayBufferWriterVariants,
    #[allow(unused)]
    chromatogram_buffers: ArrayBufferWriterVariants,

    spectrum_metadata_buffer: Vec<Entry>,
    chromatogram_metadata_buffer: Vec<ChromatogramMetadataEntry>,

    metadata_fields: SchemaRef,

    spectrum_counter: u64,
    spectrum_precursor_counter: u64,
    spectrum_data_point_counter: u64,

    chromatogram_counter: u64,
    chromatogram_precursor_counter: u64,
    #[allow(unused)]
    chromatogram_data_point_counter: u64,

    buffer_size: usize,

    mz_metadata: FileMetadataConfig,
    _t: PhantomData<(C, D)>,
}

impl<
    W: Write + Send,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> Drop for MzPeakSplitWriter<W, C, D>
{
    fn drop(&mut self) {
        if let Err(_) = self.finish() {}
    }
}

impl<
    W: Write + Send,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> MSDataFileMetadata for MzPeakSplitWriter<W, C, D>
{
    mzdata::delegate_impl_metadata_trait!(mz_metadata);
}

impl<
    W: Write + Send,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> AbstractMzPeakWriter for MzPeakSplitWriter<W, C, D>
{
    fn append_key_value_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<Option<String>>,
    ) {
        self.append_key_value_metadata(key, value);
    }

    fn spectrum_counter(&self) -> u64 {
        self.spectrum_counter
    }

    fn spectrum_counter_mut(&mut self) -> &mut u64 {
        &mut self.spectrum_counter
    }

    fn spectrum_data_buffer_mut(&mut self) -> &mut ArrayBufferWriterVariants {
        &mut self.spectrum_buffers
    }

    fn spectrum_entry_buffer_mut(&mut self) -> &mut Vec<Entry> {
        &mut self.spectrum_metadata_buffer
    }

    fn check_data_buffer(&mut self) -> io::Result<()> {
        if self.spectrum_counter % (self.buffer_size as u64) == 0 {
            self.flush_data_arrays()?;
        }
        Ok(())
    }

    fn spectrum_precursor_counter(&self) -> u64 {
        self.spectrum_precursor_counter
    }

    fn spectrum_precursor_counter_mut(&mut self) -> &mut u64 {
        &mut self.spectrum_precursor_counter
    }

    fn chromatogram_counter(&self) -> u64 {
        self.chromatogram_counter
    }

    fn chromatogram_counter_mut(&mut self) -> &mut u64 {
        &mut self.chromatogram_counter
    }

    fn chromatogram_precursor_counter_mut(&mut self) -> &mut u64 {
        &mut self.chromatogram_precursor_counter
    }

    fn chromatogram_entry_buffer_mut(&mut self) -> &mut Vec<ChromatogramMetadataEntry> {
        &mut self.chromatogram_metadata_buffer
    }

    fn chromatogram_data_buffer_mut(&mut self) -> &mut ArrayBufferWriterVariants {
        &mut self.chromatogram_buffers
    }
}

impl<
    W: Write + Send,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> MzPeakSplitWriter<W, C, D>
{
    pub fn new(
        data_writer: W,
        metadata_writer: W,
        spectrum_buffers_builder: ArrayBuffersBuilder,
        chromatogram_buffers_builder: ArrayBuffersBuilder,
        buffer_size: usize,
        mask_zero_intensity_runs: bool,
        shuffle_mz: bool,
        use_chunked_encoding: Option<ChunkingStrategy>,
        compression: Compression,
    ) -> Self {
        let fields: Vec<FieldRef> = SchemaLike::from_type::<Entry>(TracingOptions::new()).unwrap();
        let metadata_fields: SchemaRef = Arc::new(Schema::new(fields));
        let spectrum_buffers = spectrum_buffers_builder.build(
            Arc::new(Schema::empty()),
            BufferContext::Spectrum,
            mask_zero_intensity_runs,
        );
        let chromatogram_buffers = ArrayBufferWriterVariants::PointBuffers(chromatogram_buffers_builder.build(
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

        let metadata_props = Self::spectrum_metadata_writer_props(&metadata_fields);

        let mut this = Self {
            metadata_fields: metadata_fields.clone(),
            spectrum_data_writer: ArrowWriter::try_new_with_options(
                data_writer,
                spectrum_buffers.schema.clone(),
                ArrowWriterOptions::new().with_properties(data_props),
            )
            .unwrap(),
            spectrum_metadata_writer: ArrowWriter::try_new_with_options(
                metadata_writer,
                metadata_fields.clone(),
                ArrowWriterOptions::new().with_properties(metadata_props),
            )
            .unwrap(),
            spectrum_metadata_buffer: Vec::new(),
            spectrum_buffers: spectrum_buffers.into(),
            chromatogram_buffers,
            chromatogram_metadata_buffer: Vec::new(),
            spectrum_counter: 0,
            spectrum_precursor_counter: 0,
            spectrum_data_point_counter: 0,

            chromatogram_counter: 0,
            chromatogram_precursor_counter: 0,
            chromatogram_data_point_counter: 0,

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
        self.spectrum_data_writer.append_key_value_metadata(KeyValue::new(
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
    >(
        &mut self,
        spectrum: &impl SpectrumLike<A, B>,
    ) -> io::Result<()> {
        AbstractMzPeakWriter::write_spectrum(self, spectrum)
    }

    fn flush_data_arrays(&mut self) -> io::Result<()> {
        for batch in self.spectrum_buffers.drain() {
            self.spectrum_data_point_counter += batch.num_rows() as u64;
            self.spectrum_data_writer.write(&batch)?;
        }

        // for batch in self.chromatogram_buffers.drain() {
        //     self.writer.write(&batch)?;
        // }
        Ok(())
    }

    fn flush_metadata_records(&mut self) -> io::Result<()> {
        let batch =
            serde_arrow::to_record_batch(&self.metadata_fields.fields(), &self.spectrum_metadata_buffer)
                .unwrap();
        self.spectrum_metadata_writer.write(&batch)?;
        self.spectrum_metadata_buffer.clear();
        Ok(())
    }

    pub fn finish(
        &mut self,
    ) -> Result<parquet::format::FileMetaData, parquet::errors::ParquetError> {
        self.flush_data_arrays()?;
        self.flush_metadata_records()?;
        self.append_metadata();
        self.append_key_value_metadata("spectrum_count", Some(self.spectrum_counter.to_string()));
        self.append_key_value_metadata(
            "spectrum_data_point_count",
            Some(self.spectrum_data_point_counter.to_string()),
        );
        self.spectrum_data_writer.finish()?;
        self.spectrum_metadata_writer.finish()
    }
}

impl<
    W: Write + Send + Seek,
    C: CentroidLike + ToMzPeakDataSeries,
    D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
> SpectrumWriter<C, D> for MzPeakSplitWriter<W, C, D>
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
