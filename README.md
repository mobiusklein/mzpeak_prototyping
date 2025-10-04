# mzPeak file format prototyping

This repository contains prototype implementations of the mzPeak format initially described in https://pubs.acs.org/doi/10.1021/acs.jproteome.5c00435.

**NOTE**: This is a **work in progress**, no stability is guaranteed at this point.

The primary work shown here is written in Rust at the repository root, including a library for reading and writing mzPeak files,
as well as command line tools for converting existing formats into mzPeak. There is a separate Python implementation in `python/`
which is a complete re-implementation for _reading_ mzPeak files using [`pyarrow`](https://arrow.apache.org/docs/python/index.html),
and the PyData stack. The Python codebase does not support writing at this time although this is subject to change in the future.

Other languages are planned in the future in rough order of priority:
- C++
- C#
- Java
- R
- JavaScript/WebAssembly


## High level overview

mzPeak is a archive of multiple [Parquet](https://parquet.apache.org/) files, stored directly in an _uncompressed_ [ZIP](https://en.wikipedia.org/wiki/ZIP_(file_format))
archive. Each Parquet file describes a different facet of the stored mass spectrometry run. While the the data model draws on prior
art like mzML (https://peptideatlas.org/tmp/mzML1.1.0.html), it is not a direct re-implementation in a Parquet table. It does attempt
to re-use concepts like controlled vocabularies where feasible as well as arbitrary additional user metadata.

Components of an mzPeak archive:
  - `spectra_metadata.mzpeak`: Spectrum level metadata and file-level metadata. Includes spectrum descriptions, scans, precursors, and selected ions using packed parallel tables.
  - `spectra_data.mzpeak`: Spectrum signal data in either profile or centroid mode. May be in point layout or chunked layout which have different size and random access characteristics.
  - `spectra_peaks.mzpeak` (optional): Spectrum centroids stored explicitly separately from whatever signal is in `spectra_data.mzpeak`, such as from instrument vendors who store both profile and centroid versions of the same spectra. This file may not always be present.
  - `chromatograms_metadata.mzpeak` (optional): Chromatogram-level metadata and file-level metadata. Includes chromatogram descriptions, as well as precursors and selected ions using packed parallel tables.
  - `chromatograms_data.mzpeak` (optional): Chromatogram signal data. May be in point layout or chunked layout which have different size and random access characteristics. Intensity measures with different units may be stored in parallel.

### File level metadata

mzPeak file-level metadata, including descriptions of the file's contents, the instrumentation, software, and data transformation pipeline are stored in the Parquet metadata segment as JSON documents.
