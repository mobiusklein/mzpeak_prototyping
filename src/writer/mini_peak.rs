use std::io::{self, prelude::*};

use mzdata::spectrum::RefPeakDataLevel;
use mzpeaks::{CentroidLike, DeconvolutedCentroidLike};
use parquet::{arrow::ArrowWriter, file::metadata::KeyValue};

use crate::{
    ToMzPeakDataSeries,
    peak_series::ArrayIndex,
    writer::{ArrayBufferWriter, PointBuffers},
};

/// A small helper for writing peak list data to another stream with very narrow options.
pub struct MiniPeakWriterType<W: Write + Send + Seek> {
    writer: ArrowWriter<W>,
    spectrum_buffers: PointBuffers,
    buffer_size: usize,
    n_points: u64,
    n_spectra: u64,
}

impl<W: Write + Send + Seek> MiniPeakWriterType<W> {
    pub fn new(
        writer: ArrowWriter<W>,
        spectrum_buffers: PointBuffers,
        buffer_size: usize,
    ) -> Self {
        let mut this = Self {
            writer,
            spectrum_buffers,
            buffer_size,
            n_points: 0,
            n_spectra: 0,
        };
        let spectrum_array_index: ArrayIndex = this.spectrum_buffers.as_array_index();
        this.append_key_value_metadata(
            "spectrum_array_index".to_string(),
            Some(spectrum_array_index.to_json()),
        );
        this
    }

    pub fn append_key_value_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<Option<String>>,
    ) {
        self.writer
            .append_key_value_metadata(KeyValue::new(key.into(), value));
    }

    pub fn add_peaks<
        C: CentroidLike + ToMzPeakDataSeries,
        D: DeconvolutedCentroidLike + ToMzPeakDataSeries,
    >(
        &mut self,
        spectrum_count: u64,
        spectrum_time: Option<f32>,
        peaks: RefPeakDataLevel<C, D>,
    ) -> io::Result<()> {
        match peaks {
            RefPeakDataLevel::Centroid(peaks) => {
                self.spectrum_buffers.add(spectrum_count, spectrum_time, peaks.as_slice());
            }
            RefPeakDataLevel::Deconvoluted(peaks) => {
                self.spectrum_buffers.add(spectrum_count, spectrum_time, peaks.as_slice());
            }
            RefPeakDataLevel::Missing => unimplemented!(),
            RefPeakDataLevel::RawData(_) => unimplemented!(),
        }

        self.n_points += peaks.len() as u64;
        self.n_spectra += 1;

        if self.spectrum_buffers.len() >= self.buffer_size {
            self.flush()?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        for batch in self.spectrum_buffers.drain() {
            self.writer.write(&batch)?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<W, parquet::errors::ParquetError> {
        self.append_key_value_metadata("spectrum_count", Some(self.n_spectra.to_string()));
        self.append_key_value_metadata("spectrum_data_point_count", Some(self.n_points.to_string()));
        self.flush()?;
        self.writer.into_inner()
    }
}
