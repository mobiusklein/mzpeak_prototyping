use arrow::datatypes::FieldRef;
use mzdata::params::Unit;
use mzdata::spectrum::{
    Activation, ArrayType, BinaryDataArrayType, ScanEvent, SelectedIon, SpectrumDescription,
};

use parquet::basic::{Compression, ZstdLevel};
use std::collections::HashMap;
use std::io::prelude::*;
use std::{fmt::Debug, path::PathBuf};

use crate::buffer_descriptors::BufferTransform;
use crate::peak_series::{INTENSITY_UNITS, ION_MOBILITY_ARRAY_TYPES, ION_MOBILITY_UNITS};
use crate::{
    BufferContext, BufferName, ToMzPeakDataSeries,
    buffer_descriptors::BufferOverrideTable,
    chunk_series::ChunkingStrategy,
    writer::{
        ArrayBuffersBuilder, MzPeakWriterType, SpectrumVisitor, StructVisitorBuilder,
        UnpackedMzPeakWriterType,
    },
};

#[derive(Clone, Copy, Debug, Default)]
pub struct WriteBatchConfig {
    pub write_batch_size: Option<usize>,
    pub page_size: Option<usize>,
    pub row_group_size: Option<usize>,
    pub dictionary_page_size: Option<usize>,
}

pub struct SpectrumFieldVisitors {
    pub(crate) spectrum_fields: Vec<SpectrumVisitor>,
    pub(crate) spectrum_selected_ion_fields: Vec<Box<dyn StructVisitorBuilder<SelectedIon>>>,
    pub(crate) spectrum_scan_fields: Vec<Box<dyn StructVisitorBuilder<ScanEvent>>>,
    pub(crate) spectrum_activation_fields: Vec<Box<dyn StructVisitorBuilder<Activation>>>,
}

/// A builder for mzPeak writers
///
/// This allows the caller to configure array content types, compression settings,
/// and data layout.
#[derive(Debug)]
pub struct MzPeakWriterBuilder {
    pub(crate) spectrum_arrays: ArrayBuffersBuilder,
    pub(crate) chromatogram_arrays: ArrayBuffersBuilder,
    pub(crate) buffer_size: usize,
    pub(crate) shuffle_mz: bool,
    pub(crate) chunked_encoding: Option<ChunkingStrategy>,
    pub(crate) chromatogram_chunked_encoding: Option<ChunkingStrategy>,
    pub(crate) compression: Compression,
    // The schema to store peaks under, separate from the profile data (if any)
    pub(crate) store_peaks_and_profiles_apart: Option<ArrayBuffersBuilder>,
    pub(crate) write_batch_config: WriteBatchConfig,
    pub(crate) spectrum_fields: Vec<SpectrumVisitor>,
    pub(crate) spectrum_selected_ion_fields: Vec<Box<dyn StructVisitorBuilder<SelectedIon>>>,
    pub(crate) spectrum_scan_fields: Vec<Box<dyn StructVisitorBuilder<ScanEvent>>>,
    pub(crate) spectrum_activation_fields: Vec<Box<dyn StructVisitorBuilder<Activation>>>,
}

impl Default for MzPeakWriterBuilder {
    fn default() -> Self {
        Self {
            spectrum_arrays: ArrayBuffersBuilder::default()
                .prefix("point")
                .with_context(BufferContext::Spectrum),
            chromatogram_arrays: ArrayBuffersBuilder::default()
                .prefix("point")
                .with_context(BufferContext::Chromatogram),
            buffer_size: 5_000,
            shuffle_mz: false,
            chunked_encoding: None,
            chromatogram_chunked_encoding: None,
            compression: Compression::ZSTD(ZstdLevel::default()),
            store_peaks_and_profiles_apart: None,
            write_batch_config: Default::default(),
            spectrum_fields: Vec::new(),
            spectrum_selected_ion_fields: Vec::new(),
            spectrum_scan_fields: Vec::new(),
            spectrum_activation_fields: Vec::new(),
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
        self.spectrum_arrays = self.spectrum_arrays.chunking_strategy(value);
        self
    }

    /// Use the chunked representation for chromatogram data using the provided chunking strategy
    /// if `Some`, otherwise use the point list representation.
    pub fn chromatogram_chunked_encoding(mut self, value: Option<ChunkingStrategy>) -> Self {
        self.chromatogram_chunked_encoding = value;
        self.chromatogram_arrays = self.chromatogram_arrays.chunking_strategy(value);
        self
    }

    /// Add a rule to store the `from` buffer as the type given by the `to` buffer name for the
    /// spectrum data.
    pub fn add_spectrum_array_override(
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
    pub fn add_chromatogram_array_override(
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

    /// Add a column to the chromatogram data file's schema
    pub fn add_chromatogram_field(mut self, f: FieldRef) -> Self {
        self.chromatogram_arrays = self.chromatogram_arrays.add_field(f);
        self
    }

    /// Set the number of rows to buffer in memory before dumping to file
    pub fn buffer_size(mut self, value: usize) -> Self {
        self.buffer_size = value;
        self
    }

    pub fn write_batch_size(mut self, value: Option<usize>) -> Self {
        self.write_batch_config.write_batch_size = value;
        self
    }

    /// Set to control the approximate number of bytes for individual data pages.
    pub fn page_size(mut self, value: Option<usize>) -> Self {
        self.write_batch_config.page_size = value;
        self
    }

    pub fn row_group_size(mut self, value: Option<usize>) -> Self {
        self.write_batch_config.row_group_size = value;
        self
    }

    pub fn dictionary_page_size(mut self, value: Option<usize>) -> Self {
        self.write_batch_config.dictionary_page_size = value;
        self
    }

    pub fn add_spectrum_param_field<T: StructVisitorBuilder<SpectrumDescription>>(
        mut self,
        visitor: T,
    ) -> MzPeakWriterBuilder {
        self.spectrum_fields
            .push(SpectrumVisitor::Description(Box::new(visitor)));
        self
    }

    pub fn add_spectrum_selected_ion_param_field<T: StructVisitorBuilder<SelectedIon>>(
        mut self,
        visitor: T,
    ) -> MzPeakWriterBuilder {
        self.spectrum_selected_ion_fields.push(Box::new(visitor));
        self
    }

    pub fn add_spectrum_scan_field<T: StructVisitorBuilder<ScanEvent>>(
        mut self,
        visitor: T,
    ) -> MzPeakWriterBuilder {
        self.spectrum_scan_fields.push(Box::new(visitor));
        self
    }

    pub fn add_spectrum_activation_field<T: StructVisitorBuilder<Activation>>(
        mut self,
        visitor: T,
    ) -> MzPeakWriterBuilder {
        self.spectrum_activation_fields.push(Box::new(visitor));
        self
    }

    /// Build an unpacked writer, a directory on disk where all files can be written to at once,
    /// but may be more work to move about.
    pub fn build_unpacked(
        self,
        path: PathBuf,
        mask_zero_intensity_runs: bool,
    ) -> UnpackedMzPeakWriterType {
        let spectrum_fields = SpectrumFieldVisitors {
            spectrum_activation_fields: self.spectrum_activation_fields,
            spectrum_fields: self.spectrum_fields,
            spectrum_scan_fields: self.spectrum_scan_fields,
            spectrum_selected_ion_fields: self.spectrum_selected_ion_fields,
        };
        UnpackedMzPeakWriterType::new(
            path,
            self.spectrum_arrays,
            self.chromatogram_arrays,
            self.buffer_size,
            mask_zero_intensity_runs,
            self.shuffle_mz,
            self.chunked_encoding,
            self.chromatogram_chunked_encoding,
            self.compression,
            self.store_peaks_and_profiles_apart,
            self.write_batch_config,
            spectrum_fields,
        )
    }

    /// Build a zip archive-packed writer, where the spectrum data facet is written to disk
    /// and all other facets are buffered in memory until the spectrum data facet is complete.
    pub fn build<W: Write + Send + Seek>(
        self,
        writer: W,
        mask_zero_intensity_runs: bool,
    ) -> MzPeakWriterType<W> {
        let spectrum_fields = SpectrumFieldVisitors {
            spectrum_activation_fields: self.spectrum_activation_fields,
            spectrum_fields: self.spectrum_fields,
            spectrum_scan_fields: self.spectrum_scan_fields,
            spectrum_selected_ion_fields: self.spectrum_selected_ion_fields,
        };
        MzPeakWriterType::new(
            writer,
            self.spectrum_arrays,
            self.chromatogram_arrays,
            self.buffer_size,
            mask_zero_intensity_runs,
            self.shuffle_mz,
            self.chunked_encoding,
            self.chromatogram_chunked_encoding,
            self.compression,
            self.store_peaks_and_profiles_apart,
            self.write_batch_config,
            spectrum_fields,
        )
    }

    pub fn spectrum_overrides(&self) -> BufferOverrideTable {
        self.spectrum_arrays.overrides()
    }

    pub fn chromatogram_overrides(&self) -> BufferOverrideTable {
        self.chromatogram_arrays.overrides()
    }
}

#[derive(Debug, Default, Clone)]
pub struct ArrayConversionHelper {
    mz_f32: bool,
    intensity_f32: bool,
    intensity_i32: bool,
    ion_mobility_f32: bool,
    intensity_slof: bool,
}

impl ArrayConversionHelper {
    pub fn new(
        mz_f32: bool,
        intensity_f32: bool,
        intensity_i32: bool,
        ion_mobility_f32: bool,
        intensity_slof: bool,
    ) -> Self {
        Self {
            mz_f32,
            intensity_f32,
            intensity_i32,
            ion_mobility_f32,
            intensity_slof,
        }
    }

    pub fn create_type_overrides(&self) -> BufferOverrideTable {
        let mut overrides = HashMap::new();

        if self.mz_f32 {
            for unit in [Unit::MZ, Unit::Unknown] {
                overrides.insert(
                    BufferName::new(
                        BufferContext::Spectrum,
                        ArrayType::MZArray,
                        BinaryDataArrayType::Float64,
                    )
                    .with_unit(unit),
                    BufferName::new(
                        BufferContext::Spectrum,
                        ArrayType::MZArray,
                        BinaryDataArrayType::Float32,
                    )
                    .with_unit(unit),
                );
            }
        }

        let intensity_transform = self.intensity_slof.then_some(BufferTransform::NumpressSLOF);
        for ctx in [BufferContext::Chromatogram, BufferContext::Spectrum] {
            for unit in INTENSITY_UNITS {
                if self.intensity_f32 {
                    overrides.insert(
                        BufferName::new(
                            ctx,
                            ArrayType::IntensityArray,
                            BinaryDataArrayType::Float64,
                        )
                        .with_unit(unit),
                        BufferName::new(
                            ctx,
                            ArrayType::IntensityArray,
                            BinaryDataArrayType::Float32,
                        )
                        .with_unit(unit)
                        .with_transform(intensity_transform),
                    );
                }
                if intensity_transform.is_some() {
                    overrides.insert(
                        BufferName::new(
                            ctx,
                            ArrayType::IntensityArray,
                            BinaryDataArrayType::Float32,
                        )
                        .with_unit(unit),
                        BufferName::new(
                            ctx,
                            ArrayType::IntensityArray,
                            BinaryDataArrayType::Float32,
                        )
                        .with_unit(unit)
                        .with_transform(intensity_transform),
                    );
                }
                if self.intensity_i32 {
                    overrides.insert(
                        BufferName::new(
                            ctx,
                            ArrayType::IntensityArray,
                            BinaryDataArrayType::Float32,
                        )
                        .with_unit(unit),
                        BufferName::new(ctx, ArrayType::IntensityArray, BinaryDataArrayType::Int32)
                            .with_unit(unit)
                            .with_transform(intensity_transform),
                    );
                    overrides.insert(
                        BufferName::new(
                            ctx,
                            ArrayType::IntensityArray,
                            BinaryDataArrayType::Float64,
                        )
                        .with_unit(unit),
                        BufferName::new(ctx, ArrayType::IntensityArray, BinaryDataArrayType::Int32)
                            .with_unit(unit)
                            .with_transform(intensity_transform),
                    );
                }
            }
        }
        if self.ion_mobility_f32 {
            for unit in ION_MOBILITY_UNITS {
                for t in ION_MOBILITY_ARRAY_TYPES {
                    overrides.insert(
                        BufferName::new(
                            BufferContext::Spectrum,
                            t.clone(),
                            BinaryDataArrayType::Float64,
                        )
                        .with_unit(unit),
                        BufferName::new(
                            BufferContext::Spectrum,
                            t.clone(),
                            BinaryDataArrayType::Float32,
                        )
                        .with_unit(unit),
                    );
                }
            }
        }
        overrides.into()
    }
}
