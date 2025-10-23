use arrow::datatypes::{DataType, Field, FieldRef};
use mzdata::{
    params::Unit,
    spectrum::{ArrayType, BinaryDataArrayType},
};

use parquet::basic::{Compression, ZstdLevel};
use std::{fmt::Debug, path::PathBuf};
use std::{io::prelude::*, sync::Arc};

use crate::{
    BufferContext, BufferName, ToMzPeakDataSeries,
    chunk_series::ChunkingStrategy,
    writer::{ArrayBuffersBuilder, MzPeakWriterType, UnpackedMzPeakWriterType},
};

/// A builder for mzPeak writers
///
/// This allows the caller to configure array content types, compression settings,
/// and data layout.
#[derive(Debug)]
pub struct MzPeakWriterBuilder {
    spectrum_arrays: ArrayBuffersBuilder,
    chromatogram_arrays: ArrayBuffersBuilder,
    buffer_size: usize,
    shuffle_mz: bool,
    chunked_encoding: Option<ChunkingStrategy>,
    compression: Compression,
    // The schema to store peaks under, separate from the profile data (if any)
    store_peaks_and_profiles_apart: Option<ArrayBuffersBuilder>,
}

impl Default for MzPeakWriterBuilder {
    fn default() -> Self {
        Self {
            spectrum_arrays: ArrayBuffersBuilder::default().prefix("point"),
            chromatogram_arrays: ArrayBuffersBuilder::default().prefix("point"),
            buffer_size: 5_000,
            shuffle_mz: false,
            chunked_encoding: None,
            compression: Compression::ZSTD(ZstdLevel::default()),
            store_peaks_and_profiles_apart: None,
        }
    }
}

impl MzPeakWriterBuilder {
    /// Set the compression codec and level for all files to be written
    pub fn compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Add a column to the spectrum data file holding the spectrum's time in addition to the index.
    ///
    /// This is a convenience feature for building queries along the time spectrum peak/signal data,
    /// as building a covering index from the metadata table is just as efficient.
    pub fn include_time_with_spectrum_data(mut self, include_time: bool) -> Self {
        self.spectrum_arrays = self.spectrum_arrays.include_time(include_time);
        self
    }

    /// Add a column to the spectrum data file's schema
    pub fn add_spectrum_field(mut self, f: FieldRef) -> Self {
        self.spectrum_arrays = self.spectrum_arrays.add_field(f);
        self
    }

    /// Use the chunked representation for spectrum data using the provided chunking strategy
    /// if `Some`, otherwise use the point list representation.
    pub fn chunked_encoding(mut self, value: Option<ChunkingStrategy>) -> Self {
        self.chunked_encoding = value;
        self
    }

    /// Add a rule to store the `from` buffer as the type given by the `to` buffer name for the
    /// spectrum data.
    pub fn add_spectrum_override(
        mut self,
        from: impl Into<BufferName>,
        to: impl Into<BufferName>,
    ) -> Self {
        self.spectrum_arrays = self.spectrum_arrays.add_override(from, to);
        self
    }

    /// Shuffle m/z arrays using [`Encoding::BYTE_STREAM_SPLIT`] encoding (or not)
    pub fn shuffle_mz(mut self, shuffle_mz: bool) -> Self {
        self.shuffle_mz = shuffle_mz;
        self
    }

    /// In addition to trimming runs of zero intensity, replace points with zero intensity with null
    /// values which are stored more efficiently in Parquet.
    pub fn null_zeros(mut self, null_zeros: bool) -> Self {
        self.spectrum_arrays = self.spectrum_arrays.null_zeros(null_zeros);
        self
    }

    /// Set a separate array buffer schema for storing peak data in addition to profile data in the
    /// main sequence of spectrum data.
    ///
    /// If set to a non-`None` value, a separate file will be used.
    pub fn store_peaks_and_profiles_apart(mut self, value: Option<ArrayBuffersBuilder>) -> Self {
        self.store_peaks_and_profiles_apart = value;
        self
    }

    /// Add a rule to store the `from` buffer as the type given by the `to` buffer name for the
    /// chromatogram data.
    pub fn add_chromatogram_override(
        mut self,
        from: impl Into<BufferName>,
        to: impl Into<BufferName>,
    ) -> Self {
        self.chromatogram_arrays = self.chromatogram_arrays.add_override(from, to);
        self
    }

    /// Add columns to the spectrum data file's schema to support serializing `T`
    pub fn add_spectrum_peak_type<T: ToMzPeakDataSeries>(mut self) -> Self {
        self.spectrum_arrays = self.spectrum_arrays.add_peak_type::<T>();
        self
    }

    /// Specify the schema prefix for spectrum data
    pub fn spectrum_data_prefix(mut self, value: impl ToString) -> Self {
        self.spectrum_arrays = self.spectrum_arrays.prefix(value);
        self
    }

    /// Add a column to the chromatogram data file's schema
    pub fn add_chromatogram_field(mut self, f: FieldRef) -> Self {
        self.chromatogram_arrays = self.chromatogram_arrays.add_field(f);
        self
    }

    /// Specify the schema prefix for chromatogram data
    pub fn chromatogram_data_prefix(mut self, value: impl ToString) -> Self {
        self.chromatogram_arrays = self.chromatogram_arrays.prefix(value);
        self
    }

    /// Set the number of rows to buffer in memory before dumping to file
    pub fn buffer_size(mut self, value: usize) -> Self {
        self.buffer_size = value;
        self
    }

    /// Build an unpacked writer, a directory on disk where all files can be written to at once,
    /// but may be more work to move about.
    pub fn build_unpacked(
        self,
        path: PathBuf,
        mask_zero_intensity_runs: bool,
    ) -> UnpackedMzPeakWriterType {
        UnpackedMzPeakWriterType::new(
            path,
            self.spectrum_arrays,
            self.chromatogram_arrays,
            self.buffer_size,
            mask_zero_intensity_runs,
            self.shuffle_mz,
            self.chunked_encoding,
            self.compression,
            self.store_peaks_and_profiles_apart,
        )
    }

    /// Build a zip archive-packed writer, where the spectrum data facet is written to disk
    /// and all other facets are buffered in memory until the spectrum data facet is complete.
    pub fn build<W: Write + Send + Seek>(
        self,
        writer: W,
        mask_zero_intensity_runs: bool,
    ) -> MzPeakWriterType<W> {
        MzPeakWriterType::new(
            writer,
            self.spectrum_arrays,
            self.chromatogram_arrays,
            self.buffer_size,
            mask_zero_intensity_runs,
            self.shuffle_mz,
            self.chunked_encoding,
            self.compression,
            self.store_peaks_and_profiles_apart,
        )
    }

    /// Add the default time (f64) and intensity (f32) arrays for chromatograms
    pub fn add_default_chromatogram_fields(mut self) -> Self {
        let time = BufferName::new(
            BufferContext::Chromatogram,
            ArrayType::TimeArray,
            BinaryDataArrayType::Float64,
        )
        .with_unit(Unit::Minute)
        .to_field();
        let intensity = BufferName::new(
            BufferContext::Chromatogram,
            ArrayType::IntensityArray,
            BinaryDataArrayType::Float32,
        )
        .with_unit(Unit::DetectorCounts)
        .to_field();

        self = self
            .add_chromatogram_field(time)
            .add_chromatogram_field(intensity)
            .add_chromatogram_field(Arc::new(Field::new(
                "chromatogram_index",
                DataType::UInt64,
                false,
            )));
        self
    }
}
