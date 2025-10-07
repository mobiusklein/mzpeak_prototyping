use std::{fs, io, sync::Arc};

use arrow::{array::Array, datatypes::SchemaRef};
use mzdata::{
    prelude::*,
    spectrum::{
        ArrayType, BinaryArrayMap, Chromatogram, RefPeakDataLevel, SignalContinuity,
        bindata::ArrayRetrievalError,
    },
};
use parquet::{
    arrow::ArrowSchemaConverter,
    basic::{Compression, Encoding, ZstdLevel},
    file::properties::{
        DEFAULT_DICTIONARY_PAGE_SIZE_LIMIT, EnabledStatistics, WriterProperties, WriterVersion,
    },
    format::SortingColumn,
};

use crate::{
    BufferContext, BufferName, ToMzPeakDataSeries,
    chunk_series::{ArrowArrayChunk, ChunkingStrategy},
    entry::ChromatogramMetadataEntry,
    filter::select_delta_model,
    peak_series::{MZ_ARRAY, array_map_to_schema_arrays_and_excess},
    spectrum::{AuxiliaryArray, ChromatogramEntry},
    writer::{
        ArrayBufferWriter, ArrayBufferWriterVariants, ChromatogramBuilder, MiniPeakWriterType,
        SpectrumBuilder,
    },
};

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
            );
        }
    };
}

pub(crate) use implement_mz_metadata;

pub trait AbstractMzPeakWriter {
    /// Append an arbitrary key bytestring with an optional value to the (current) Parquet file
    fn append_key_value_metadata(
        &mut self,
        key: impl Into<String>,
        value: impl Into<Option<String>>,
    );

    /// Whether or not a chunking strategy is being used
    fn use_chunked_encoding(&self) -> Option<&ChunkingStrategy> {
        None
    }

    /// Get a mutable reference to the buffer of spectrum metadata values,
    /// for appending only
    fn spectrum_entry_buffer_mut(&mut self) -> &mut SpectrumBuilder;

    /// Get a mutable reference to the buffer of chromatogram metadata values,
    /// for appending only
    fn chromatogram_entry_buffer_mut(&mut self) -> &mut ChromatogramBuilder;

    /// Get a mutable reference to the buffer of spectrum signal data values,
    /// for appending only
    fn spectrum_data_buffer_mut(&mut self) -> &mut ArrayBufferWriterVariants;

    fn chromatogram_data_buffer_mut(&mut self) -> &mut ArrayBufferWriterVariants;

    /// Check if the data buffers are full, and flush them if so
    fn check_data_buffer(&mut self) -> io::Result<()>;

    /// The current number of spectra having been written to the MzPeak file
    fn spectrum_counter(&self) -> u64;

    /// The current number of distinct precursors having been written to the MzPeak file
    fn spectrum_precursor_counter(&self) -> u64;

    fn chromatogram_counter(&self) -> u64;

    fn chromatogram_to_entries(
        &mut self,
        chromatogram: &impl ChromatogramLike,
    ) -> Vec<ChromatogramMetadataEntry> {
        let mut ent = ChromatogramEntry::from_chromatogram(chromatogram);
        ent.index = Some(self.chromatogram_counter());
        vec![ent.into()]
    }

    fn write_chromatogram_arrays(
        &mut self,
        _chromatogram: &impl ChromatogramLike,
        binary_array_map: &BinaryArrayMap,
    ) -> io::Result<Option<Vec<AuxiliaryArray>>> {
        let time = binary_array_map.get(&ArrayType::TimeArray).unwrap();
        let n_points = time.data_len()?;
        let time_axis = BufferName::from_data_array(BufferContext::Chromatogram, time);
        let chromatogram_index = self.chromatogram_counter();
        let extra_arrays = if let Some(chunking) = None
        // self.use_chunked_encoding().copied()
        {
            let buffer_ref = self.chromatogram_data_buffer_mut();
            let (chunks, auxiliary_arrays) = ArrowArrayChunk::from_arrays(
                chromatogram_index,
                None,
                time_axis,
                binary_array_map,
                chunking,
                buffer_ref.overrides(),
                false,
                false,
                Some(buffer_ref.fields()),
            )?;
            if !chunks.is_empty() {
                let chunks = ArrowArrayChunk::to_struct_array(
                    &chunks,
                    BufferContext::Chromatogram,
                    buffer_ref.schema().fields(),
                    &[chunking, ChunkingStrategy::Basic { chunk_size: 50.0 }],
                    false,
                );
                let size = chunks.len();
                let (fields, arrays, _nulls) = chunks.into_parts();
                buffer_ref.add_arrays(fields, arrays, size, true);
            }

            Some(auxiliary_arrays)
        } else {
            let buffer = self.chromatogram_data_buffer_mut();

            let (fields, data, extra_arrays) = array_map_to_schema_arrays_and_excess(
                BufferContext::Chromatogram,
                binary_array_map,
                n_points,
                chromatogram_index,
                None,
                Some(&buffer.fields()),
                &buffer.overrides(),
            )?;

            buffer.add_arrays(fields, data, n_points, true);
            Some(extra_arrays)
        };

        Ok(extra_arrays)
    }

    fn write_chromatogram(&mut self, chromatogram: &Chromatogram) -> io::Result<()> {
        log::trace!("Writing chromatogram {}", chromatogram.id());
        let aux_arrays = self.write_chromatogram_arrays(chromatogram, &chromatogram.arrays)?;
        self.chromatogram_entry_buffer_mut()
            .append_value(chromatogram, aux_arrays);
        self.check_data_buffer()?;
        Ok(())
    }

    /// Write a `spectrum` to the MzPeak file
    fn write_spectrum<
        C: ToMzPeakDataSeries + CentroidLike,
        D: ToMzPeakDataSeries + DeconvolutedCentroidLike,
        S: SpectrumLike<C, D> + 'static,
    >(
        &mut self,
        spectrum: &S,
    ) -> io::Result<()> {
        log::trace!("Writing spectrum {}", spectrum.id());
        let (mz_delta_model, aux_arrays) = self.write_spectrum_data(spectrum)?;
        self.spectrum_entry_buffer_mut()
            .append_value(spectrum, mz_delta_model, aux_arrays);
        self.check_data_buffer()?;
        Ok(())
    }

    /// Fit an [`MZDeltaModel`] instance on the provided (sparse) spectrum signal, and return the parameter
    /// buffer.
    ///
    /// If an intensity array is available, it will be used to weight the parameter estimation procedure.
    ///
    /// If no m/z array is available, `None` is returned
    fn build_delta_model(&self, binary_array_map: &BinaryArrayMap) -> Option<Vec<f64>> {
        if let Ok(mzs) = binary_array_map.mzs() {
            let delta_model = if let Ok(ints) = binary_array_map.intensities() {
                let weights: Vec<f64> =
                    ints.iter().map(|i| (*i + 1.0).ln().sqrt() as f64).collect();
                select_delta_model(&mzs, Some(&weights))
            } else {
                select_delta_model(&mzs, None)
            };
            Some(delta_model)
        } else {
            None
        }
    }

    /// Write a [`BinaryArrayMap`] to the data buffer.
    ///
    /// If sparse data encoding is enabled ([`ArrayBufferWriter::nullify_zero_intensity`]), and the
    /// `spectrum` is in profile mode, this will fit a delta model with [`AbstractMzPeakWriter::build_delta_model`].
    ///
    /// If chunked encoding is enabled, the [`ChunkingStrategy`] will be applied, regardless of whether or not the
    /// spectrum is in profile mode. This might change in the future.
    ///
    /// This is a helper method for [`AbstractMzPeakWriter::write_spectrum_data`].
    fn write_spectrum_binary_array_map<
        C: ToMzPeakDataSeries + CentroidLike,
        D: ToMzPeakDataSeries + DeconvolutedCentroidLike,
    >(
        &mut self,
        spectrum: &impl SpectrumLike<C, D>,
        spectrum_count: u64,
        binary_array_map: &BinaryArrayMap,
    ) -> Result<(Option<Vec<f64>>, Option<Vec<AuxiliaryArray>>), ArrayRetrievalError> {
        let mzs = binary_array_map.mzs();
        let (_had_mzs, n_points) = if let Ok(mzs) = mzs {
            (true, mzs.len())
        } else {
            (false, 0)
        };

        let is_profile = spectrum.signal_continuity() == SignalContinuity::Profile;
        let include_time = self.spectrum_data_buffer_mut().include_time();
        let spectrum_time = if include_time {
            Some(spectrum.start_time() as f32)
        } else {
            None
        };

        let (delta_params, extra_arrays) = if let Some(chunking) =
            self.use_chunked_encoding().copied()
        {
            // If we use the chunked encoding, we pre-encode everything
            let nullify_zero_intensity = self.spectrum_data_buffer_mut().nullify_zero_intensity();
            let delta_model = if is_profile && nullify_zero_intensity {
                self.build_delta_model(binary_array_map)
            } else {
                None
            };
            let buffer_ref = self.spectrum_data_buffer_mut();
            let (chunks, auxiliary_arrays) = ArrowArrayChunk::from_arrays(
                spectrum_count,
                spectrum_time,
                MZ_ARRAY,
                binary_array_map,
                chunking,
                buffer_ref.overrides(),
                is_profile,
                nullify_zero_intensity,
                Some(buffer_ref.fields()),
            )?;
            if !chunks.is_empty() {
                let chunks = ArrowArrayChunk::to_struct_array(
                    &chunks,
                    BufferContext::Spectrum,
                    buffer_ref.schema().fields(),
                    &[chunking, ChunkingStrategy::Basic { chunk_size: 50.0 }],
                    include_time,
                );
                let size = chunks.len();
                let (fields, arrays, _nulls) = chunks.into_parts();
                buffer_ref.add_arrays(fields, arrays, size, is_profile);
            }

            (delta_model, Some(auxiliary_arrays))
        } else {
            let nullify_zero_intensity = self.spectrum_data_buffer_mut().nullify_zero_intensity();
            let delta_model = if is_profile && nullify_zero_intensity {
                self.build_delta_model(binary_array_map)
            } else {
                None
            };

            let buffer = self.spectrum_data_buffer_mut();

            let (fields, data, extra_arrays) = array_map_to_schema_arrays_and_excess(
                BufferContext::Spectrum,
                binary_array_map,
                n_points,
                spectrum_count,
                spectrum_time,
                Some(&buffer.fields()),
                &buffer.overrides(),
            )?;

            buffer.add_arrays(fields, data, n_points, is_profile);
            (delta_model, Some(extra_arrays))
        };

        Ok((delta_params, extra_arrays))
    }

    /// Write a peak list to the data buffer.
    ///
    /// If chunked encoding is enabled, [`ChunkingStrategy::Basic`] will be used.
    fn write_peaks<C: ToMzPeakDataSeries>(
        &mut self,
        spectrum_count: u64,
        mut spectrum_time: Option<f32>,
        peaks: &[C],
    ) -> Result<(Option<Vec<f64>>, Option<Vec<AuxiliaryArray>>), ArrayRetrievalError> {
        let include_time = self.spectrum_data_buffer_mut().include_time();
        if !include_time {
            spectrum_time = None;
        }
        if let Some(encoding) = self.use_chunked_encoding().copied() {
            let arrays = C::as_arrays(peaks);
            let buffer_ref = self.spectrum_data_buffer_mut();
            let (chunks, auxiliary_arrays) = ArrowArrayChunk::from_arrays(
                spectrum_count,
                spectrum_time,
                MZ_ARRAY,
                &arrays,
                ChunkingStrategy::Basic {
                    chunk_size: encoding.chunk_size(),
                },
                buffer_ref.overrides(),
                false,
                false,
                Some(buffer_ref.fields()),
            )?;
            if !chunks.is_empty() {
                let chunks = ArrowArrayChunk::to_struct_array(
                    &chunks,
                    BufferContext::Spectrum,
                    buffer_ref.schema().fields(),
                    &[
                        encoding,
                        ChunkingStrategy::Basic {
                            chunk_size: encoding.chunk_size(),
                        },
                    ],
                    include_time,
                );
                let size = chunks.len();
                let (fields, arrays, _nulls) = chunks.into_parts();
                buffer_ref.add_arrays(fields, arrays, size, false);
            }
            Ok((None, Some(auxiliary_arrays)))
        } else {
            self.spectrum_data_buffer_mut()
                .add(spectrum_count, spectrum_time, peaks);
            Ok((None, None))
        }
    }

    /// Write the spectrum data of any dimensions to the data buffer.
    ///
    /// Uses [`SpectrumLike::peaks`] to decide which kind of data to write.
    fn write_spectrum_data<
        CI: ToMzPeakDataSeries + CentroidLike,
        DI: ToMzPeakDataSeries + DeconvolutedCentroidLike,
    >(
        &mut self,
        spectrum: &impl SpectrumLike<CI, DI>,
    ) -> io::Result<(Option<Vec<f64>>, Option<Vec<AuxiliaryArray>>)> {
        let spectrum_count = self.spectrum_counter();

        let peaks = spectrum.peaks();

        let spectrum_time = if self.spectrum_data_buffer_mut().include_time() {
            Some(spectrum.start_time() as f32)
        } else {
            None
        };

        let (delta_params, aux_arrays) = if self.separate_peak_writer().is_some()
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
                self.write_spectrum_binary_array_map(spectrum, spectrum_count, raw_arrays)?;
            self.separate_peak_writer()
                .unwrap()
                .add_peaks(spectrum_count, spectrum_time, peaks)?;
            (delta_params, aux_arrays)
        } else {
            let (delta_params, aux_arrays) = match peaks {
                mzdata::spectrum::RefPeakDataLevel::Missing => (None, None),
                mzdata::spectrum::RefPeakDataLevel::RawData(binary_array_map) => self
                    .write_spectrum_binary_array_map(spectrum, spectrum_count, binary_array_map)?,
                mzdata::spectrum::RefPeakDataLevel::Centroid(peaks) => {
                    self.write_peaks(spectrum_count, spectrum_time, peaks.as_slice())?
                }
                mzdata::spectrum::RefPeakDataLevel::Deconvoluted(peaks) => {
                    self.write_peaks(spectrum_count, spectrum_time, peaks.as_slice())?
                }
            };
            (delta_params, aux_arrays)
        };

        Ok((delta_params, aux_arrays))
    }

    fn separate_peak_writer(&mut self) -> Option<&mut MiniPeakWriterType<fs::File>> {
        None
    }

    fn spectrum_metadata_writer_props(metadata_fields: &SchemaRef) -> WriterProperties {
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

    fn chromatogram_data_writer_props(
        data_buffer: &impl ArrayBufferWriter,
        index_path: String,
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
            if colpath.contains("_time")
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
}
