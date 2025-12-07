# mzPeak File Format

- [mzPeak File Format](#mzpeak-file-format)
- [Introduction](#introduction)
  - [Overview](#overview)
    - [What _is_ mzPeak?](#what-is-mzpeak)
  - [Anatomy of a Parquet file](#anatomy-of-a-parquet-file)
    - [The schema](#the-schema)
    - [The metadata key-value pairs](#the-metadata-key-value-pairs)
    - [The columnar data](#the-columnar-data)
- [Container](#container)
    - [ZIP archives](#zip-archives)
      - [Why not TAR?](#why-not-tar)
    - [Unpacked archives](#unpacked-archives)
- [Data Layouts](#data-layouts)
  - [Packed Parallel Metadata Tables](#packed-parallel-metadata-tables)
  - [Data Transformations And Null Semantics](#data-transformations-and-null-semantics)
  - [Point Layout](#point-layout)
  - [Chunked Layout](#chunked-layout)
- [Index File - `mzpeak_index.json`](#index-file---mzpeak_indexjson)
- [Spectrum Signal Data File - `spectra_data.mzpeak`](#spectrum-signal-data-file---spectra_datamzpeak)
- [Spectrum Metadata File - `spectra_metadata.mzpeak`](#spectrum-metadata-file---spectra_metadatamzpeak)
- [Spectrum Peak Data - `spectra_peaks.mzpeak`](#spectrum-peak-data---spectra_peaksmzpeak)
- [Chromatogram Signal Data - `chromatograms_data.mzpeak`](#chromatogram-signal-data---chromatograms_datamzpeak)
- [Chromatogram Metadata - `chromatograms_metadata.mzpeak`](#chromatogram-metadata---chromatograms_metadatamzpeak)

# Introduction

## Overview

### What _is_ mzPeak?

mzPeak is an archive of multiple [Parquet](https://parquet.apache.org/) files, stored directly in an _uncompressed_ [ZIP](<https://en.wikipedia.org/wiki/ZIP_(file_format)>)
archive or unpacked directory/prefix. Each Parquet file describes a different facet of the stored mass spectrometry run. While the the data model draws on prior
art like mzML (https://peptideatlas.org/tmp/mzML1.1.0.html), it is not a direct re-implementation in a Parquet table. It does attempt
to re-use concepts like controlled vocabularies where feasible as well as arbitrary additional user metadata.

Components of an mzPeak archive:

- `spectra_metadata.mzpeak`: Spectrum level metadata and file-level metadata. Includes spectrum descriptions, scans, precursors, and selected ions using packed parallel tables.
- `spectra_data.mzpeak`: Spectrum signal data in either profile or centroid mode. May be in point layout or chunked layout which have different size and random access characteristics.
- `spectra_peaks.mzpeak` (optional): Spectrum centroids stored explicitly separately from whatever signal is in `spectra_data.mzpeak`, such as from instrument vendors who store both profile and centroid versions of the same spectra. This file may not always be present.
- `chromatograms_metadata.mzpeak`: Chromatogram-level metadata and file-level metadata. Includes chromatogram descriptions, as well as precursors and selected ions using packed parallel tables.
- `chromatograms_data.mzpeak`: Chromatogram signal data. May be in point layout or chunked layout which have different size and random access characteristics. Intensity measures with different units may be stored in parallel.

## Anatomy of a Parquet file

This is a minimal overview of Parquet. For more details, please see <https://parquet.apache.org/> for further explanation.

### The schema

Parquet files contain a physical data schema defining how their data columns are encoded in bytes on disk. This schema supports arbitrary levels of nullability, nesting (groups) or repetition (lists). These physical data types may also be mapped to one or more "logical types".

There is a broader many-to-many mapping between Parquet schemas and [Apache Arrow](https://arrow.apache.org/) schemas. Arrow supports many types that Parquet does not, but they share a common abstraction of columnar data storage with a notion of per-value nullability, and while they store these concepts differently, it is straight-forwards to convert from one to the other.

### The metadata key-value pairs

![Parquet schematic](https://parquet.apache.org/images/FileLayout.gif)

At the end of a Parquet file is a footer containing user-defined metadata along with the file's schema, offsets and search indices. This user-defined metadata is stored in key-value pairs, which makes it amenable to serializing light-weight, immediately interesting metadata there that do not make sense to force into the data columns.

### The columnar data

Parquet is a strongly typed binary columnar data format with layered blocked compression that permits random access.

TODO: write more here

# Container

### ZIP archives

In order to pack multiple Parquet tables together under a single file name on disk, we need a container file format. To that end we use the [ZIP](https://www.iana.org/assignments/media-types/application/zip) archive to bundle multiple files together. ZIP files start with a header containing the magic bytes followed by a sequence of blocks of (header, file) pairs, terminating with a central directory listing how to find each file in the archive. Files saved in a ZIP may be stored compressed or uncompressed. When mzPeak is stored in a ZIP it **MUST** store its member files uncompressed.

#### Why not TAR?

TAR archives are designed for a linear traversal. In order to know all of the files in the archive, you must jump from header entry to header entry until you reach the end of the archive. Compared to ZIP's central directory index, this is less efficient and more expensive for object stores. TAR does not support per file encryption either, making protecting parts of the archive that are _not_ in Parquet more difficult.

### Unpacked archives

If an mzPeak archive is stored in an unpacked directory, the directory name is treated as the name of the name of the run file.

# Data Layouts

## Packed Parallel Metadata Tables

The `spectra_metadata.mzpeak` and `chromatograms_metadata.mzpeak` store multiple schemas in parallel. In these Parquet files, the root schema is made up of several branched "group" or "struct" (Parquet vs. Arrow nomenclature) that may be null at any level.

Here is a stripped down example where two rows of related MS1 and MS2 spectra. Treat `scan.source_index`, `precursor.source_index`, `precursor.precursor_index`, `selected_ion.source_index`, and `selected_ion.precursor_index` as a foreign key with respect to `spectrum.index`. `precursor.source_index` refers to the `spectrum` which this `precursor` record belongs to and `precursor.precursor_index` refers to the `spectrum` that is that *is* the precursor of the `spectrum` referenced by `precursor.source_index`. Any of these columns may be `null` which means that such a record does not exist in the table. This also applies to the `selected_ion` facet.

<style>
  .packed-table thead tr th {
    font-size: 0.7em;
    padding: 0.6em;
    text-align: center;
  }
  .packed-table tbody tr td {
    font-size: 0.7em;
    padding: 0.2em
  }
</style>

<table class="packed-table" style="font-size=0.2em">
  <thead>
    <tr>
      <th colspan=4>spectrum</th>
      <th colspan=2>scan</th>
      <th colspan=3>precursor</th>
      <th colspan=3>selected_ion</th>
    </tr>
    <tr>
      <th>
      index
      </th>
      <th>
      id
      </th>
      <th>
      time
      </th>
      <th>
      MS_1000511_<br/>ms_level
      </th>
      <th>
      source_<br/>index
      </th>
      <th>
      MS_1000616_preset_<br/>scan_configuration
      </th>
      <th>
      source_<br/>index
      </th>
      <th>
      precursor_<br/>index
      </th>
      <th>
      isolation_<br/>window
      </th>
      <th>
      source_<br/>index
      </th>
      <th>
      precursor_<br/>index
      </th>
      <th>
      MS_1000744_selected_<br/>ion_mz
      </th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td colspan=12 style="text-align:center;">...</td>
    </tr>
    <tr>
      <td>502</td>
      <td>scan=502</td>
      <td>20.51</td>
      <td>1</td>
      <td>502</td>
      <td>3</td>
      <td>503</td>
      <td>502</td>
      <td>{...}</td>
      <td>503</td>
      <td>502</td>
      <td>233.5</td>
    </tr>
    <tr>
      <td>503</td>
      <td>scan=503</td>
      <td>20.531</td>
      <td>2</td>
      <td>503</td>
      <td>2</td>
      <td>504</td>
      <td>502</td>
      <td>{...}</td>
      <td>504</td>
      <td>502</td>
      <td>562.3</td>
    </tr>
    <tr>
      <td colspan=12 style="text-align:center;">...</td>
    </tr>
  </tbody>
</table>

A writer implementation is SHOULD to minimize the number of interspersed rows that are `null`, but this is not strictly required. Minimizing the interspersed nulls improves compressability. See the images below, the "Packed Tables" has all of the rows of each parallel table contiguous, while the "Sparse Tables" diagram shows rows of nulls intermixed

<img src="static/img/packed_tables.png" width="40%" style="background-color: white; padding: 1em;"/> <img src="static/img/sparse_tables.png" width="40%" style="background-color: white; padding: 1em;"/>

## Data Transformations And Null Semantics

## Point Layout

<img src="static/img/point_layout.png" height="600pt" style="background-color: white; padding: 1em;"/>


## Chunked Layout

<img src="static/img/chunked_layout.png" height="600pt" style="background-color: white; padding: 1em;"/>


# Index File - `mzpeak_index.json`

```yaml
mzpeak_index.json:
  files:
    - name: spectra_data.mzpeak
      entity_type: spectrum
      data_kind: data arrays
    - name: spectra_metadata.mzpeak
      entity_type: spectrum
      data_kind: metadata
    - name: chromatograms_data.mzpeak
      entity_type: chromatogram
      data_kind: data arrays
    - name: chromatograms_metadata.mzpeak
      entity_type: chromatogram
      data_kind: metadata
  metadata: {}
```

Governed by JSONSchema `schema/mzpeak_index.json`

# Spectrum Signal Data File - `spectra_data.mzpeak`

**File index entry:**

```json
{
  "name": "spectra_data.mzpeak",
  "entry_kind": "spectrum",
  "data_kind": "data arrays"
}
```

# Spectrum Metadata File - `spectra_metadata.mzpeak`

**File index entry:**

```json
{
  "name": "spectra_metadata.mzpeak",
  "entry_kind": "spectrum",
  "data_kind": "metadata"
}
```

# Spectrum Peak Data - `spectra_peaks.mzpeak`

**File index entry:**

```json
{
  "name": "spectra_peaks.mzpeak",
  "entry_kind": "spectrum",
  "data_kind": "peaks"
}
```

# Chromatogram Signal Data - `chromatograms_data.mzpeak`

**File index entry:**

```json
{
  "name": "chromatograms_data.mzpeak",
  "entry_kind": "chromatogram",
  "data_kind": "data arrays"
}
```

# Chromatogram Metadata - `chromatograms_metadata.mzpeak`

**File index entry:**

```json
{
  "name": "chromatograms_metadata.mzpeak",
  "entry_kind": "chromatogram",
  "data_kind": "metadata"
}
```
