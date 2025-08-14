use std::io;

use arrow::array::Array;
use mzdata::{prelude::*, spectrum::{bindata::ArrayRetrievalError, BinaryArrayMap, SignalContinuity}};

use crate::{chunk_series::{ArrowArrayChunk, ChunkingStrategy}, entry::Entry, filter::select_delta_model, peak_series::{array_map_to_schema_arrays_and_excess, MZ_ARRAY}, spectrum::AuxiliaryArray, writer::{ArrayBufferWriter, ArrayBufferWriterVariants}, BufferContext, ToMzPeakDataSeries};


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
    fn spectrum_entry_buffer_mut(&mut self) -> &mut Vec<Entry>;

    /// Get a mutable reference to the buffer of spectrum signal data values,
    /// for appending only
    fn spectrum_data_buffer_mut(&mut self) -> &mut ArrayBufferWriterVariants;

    /// Check if the data buffers are full, and flush them if so
    fn check_data_buffer(&mut self) -> io::Result<()>;

    /// The current number of spectra having been written to the MzPeak file
    fn spectrum_counter(&self) -> u64;

    /// A mutable reference to the number of spectra having been written to the MzPeak file,
    /// for incrementing
    fn spectrum_counter_mut(&mut self) -> &mut u64;

    /// The current number of distinct precursors having been written to the MzPeak file
    fn spectrum_precursor_counter(&self) -> u64;
    /// A mutable reference to the number of distinct precursors having been written to the MzPeak file,
    /// for incrementing
    fn spectrum_precursor_counter_mut(&mut self) -> &mut u64;

    /// Convert a `spectrum` into one or more [`Entry`] records describing the spectrum and its subsidiary
    /// structures.
    ///
    /// This method will update internal precursor counters if needed.
    fn spectrum_to_entries<
        C: ToMzPeakDataSeries + CentroidLike,
        D: ToMzPeakDataSeries + DeconvolutedCentroidLike,
    >(
        &mut self,
        spectrum: &impl SpectrumLike<C, D>,
    ) -> Vec<Entry> {
        Entry::from_spectrum(
            spectrum,
            Some(self.spectrum_counter()),
            Some(self.spectrum_precursor_counter_mut()),
        )
    }

    /// Write a `spectrum` to the MzPeak file
    fn write_spectrum<
        C: ToMzPeakDataSeries + CentroidLike,
        D: ToMzPeakDataSeries + DeconvolutedCentroidLike,
    >(
        &mut self,
        spectrum: &impl SpectrumLike<C, D>,
    ) -> io::Result<()> {
        log::trace!("Writing spectrum {}", spectrum.id());
        let (median_delta, aux_arrays) = self.write_spectrum_data(spectrum)?;
        let mut entries = self.spectrum_to_entries(spectrum);
        if let Some(entry) = entries.first_mut() {
            if let Some(spec_ent) = entry.spectrum.as_mut() {
                spec_ent.median_delta = median_delta;
                spec_ent
                    .auxiliary_arrays
                    .extend(aux_arrays.unwrap_or_default());
            }
        }
        self.spectrum_entry_buffer_mut().extend(entries);
        *self.spectrum_counter_mut() += 1;
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
    fn write_binary_array_map<
        C: ToMzPeakDataSeries + CentroidLike,
        D: ToMzPeakDataSeries + DeconvolutedCentroidLike,
    >(
        &mut self,
        spectrum: &impl SpectrumLike<C, D>,
        spectrum_count: u64,
        binary_array_map: &BinaryArrayMap,
    ) -> Result<(Option<Vec<f64>>, Option<Vec<AuxiliaryArray>>), ArrayRetrievalError> {
        let n_points = binary_array_map.mzs()?.len();
        let is_profile = spectrum.signal_continuity() == SignalContinuity::Profile;

        let (delta_params, extra_arrays) = if let Some(chunking) =
            self.use_chunked_encoding().copied()
        {
            let nullify_zero_intensity = self.spectrum_data_buffer_mut().nullify_zero_intensity();
            let median_delta = if is_profile {
                self.build_delta_model(binary_array_map)
            } else {
                None
            };
            let buffer_ref = self.spectrum_data_buffer_mut();
            let (chunks, auxiliary_arrays) = ArrowArrayChunk::from_arrays(
                spectrum_count,
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
                    "spectrum_index",
                    buffer_ref.schema().fields(),
                    &[chunking, ChunkingStrategy::Basic { chunk_size: 50.0 }],
                );
                let size = chunks.len();
                let (fields, arrays, _nulls) = chunks.into_parts();
                buffer_ref.add_arrays(fields, arrays, size, is_profile);
            }

            (median_delta, Some(auxiliary_arrays))
        } else {
            let median_delta = if is_profile {
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
                "spectrum_index",
                &buffer.fields(),
                &buffer.overrides(),
            )?;

            buffer.add_arrays(fields, data, n_points, is_profile);
            (median_delta, Some(extra_arrays))
        };

        Ok((delta_params, extra_arrays))
    }

    /// Write a peak list to the data buffer.
    ///
    /// If chunked encoding is enabled, [`ChunkingStrategy::Basic`] will be used.
    fn write_peaks<C: ToMzPeakDataSeries>(
        &mut self,
        spectrum_count: u64,
        peaks: &[C],
    ) -> Result<(Option<Vec<f64>>, Option<Vec<AuxiliaryArray>>), ArrayRetrievalError> {
        if let Some(encoding) = self.use_chunked_encoding().copied() {
            let arrays = C::as_arrays(peaks);
            let buffer_ref = self.spectrum_data_buffer_mut();
            let (chunks, auxiliary_arrays) = ArrowArrayChunk::from_arrays(
                spectrum_count,
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
                    "spectrum_index",
                    buffer_ref.schema().fields(),
                    &[
                        encoding,
                        ChunkingStrategy::Basic {
                            chunk_size: encoding.chunk_size(),
                        },
                    ],
                );
                let size = chunks.len();
                let (fields, arrays, _nulls) = chunks.into_parts();
                buffer_ref.add_arrays(fields, arrays, size, false);
            }
            Ok((None, Some(auxiliary_arrays)))
        } else {
            self.spectrum_data_buffer_mut().add(spectrum_count, peaks);
            Ok((None, None))
        }
    }

    /// Write the spectrum data of any dimensions to the data buffer.
    ///
    /// Uses [`SpectrumLike::peaks`] to decide which kind of data to write.
    fn write_spectrum_data<
        C: ToMzPeakDataSeries + CentroidLike,
        D: ToMzPeakDataSeries + DeconvolutedCentroidLike,
    >(
        &mut self,
        spectrum: &impl SpectrumLike<C, D>,
    ) -> io::Result<(Option<Vec<f64>>, Option<Vec<AuxiliaryArray>>)> {
        let spectrum_count = self.spectrum_counter();
        let (delta_params, aux_arrays) = match spectrum.peaks() {
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
        Ok((delta_params, aux_arrays))
    }
}
