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
    - [Null semantics](#null-semantics)
    - [File-Level Metadata](#file-level-metadata)
  - [Signal Data Layouts](#signal-data-layouts)
    - [Arrays and Columns](#arrays-and-columns)
      - [The Array Index](#the-array-index)
    - [Data Arrays, Encoding, Transformations and Parquet](#data-arrays-encoding-transformations-and-parquet)
      - [Zero Run Stripping](#zero-run-stripping)
        - [Null Marking](#null-marking)
    - [Point Layout](#point-layout)
    - [Chunked Layout](#chunked-layout)
  - [Why all these root nodes?](#why-all-these-root-nodes)
- [Index File - `mzpeak_index.json`](#index-file---mzpeak_indexjson)
  - [Data Kind](#data-kind)
  - [Entity Type](#entity-type)
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

The `spectra_metadata.mzpeak` and `chromatograms_metadata.mzpeak` store multiple schemas in parallel. In these Parquet files, the root schema is made up of several branched "group" or "struct" (Parquet vs. Arrow nomenclature) that may be null at any level. We use relational database language, specifically, "primary key" and "foreign key" to describe the interconnections between the different tables that are packed together here.

Here is a stripped down example where two rows of related MS1 and MS2 spectra. Treat `scan.source_index`, `precursor.source_index`, `precursor.precursor_index`, `selected_ion.source_index`, and `selected_ion.precursor_index` as a foreign key with respect to `spectrum.index`, a primary key. `precursor.source_index` refers to the `spectrum` which this `precursor` record belongs to and `precursor.precursor_index` refers to the `spectrum` that is that _is_ the precursor of the `spectrum` referenced by `precursor.source_index`, and (`precursor.source_index`, `precursor.precursor_index`) forms a compound primary key. Any of these columns may be `null` which means that such a record does not exist in the table. This also applies to the `selected_ion` facet.

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

### Null semantics

A row value that is `null` should be treated as being absent, having no value. If a foreign key column is `null`, assume the entry does not exist in the table, as in the case where an MS2 spectrum is stored without MS1 spectra as in MGF files, or a slice of an MS run. If it is the primary key of the table, the reader _SHOULD_ skip of the columns in that row for that table.

A writer implementation is _SHOULD_ to minimize the number of interspersed rows that are `null`, but this is not strictly required. Minimizing the interspersed nulls improves compressability. See the images below, the "Packed Tables" has all of the rows of each parallel table contiguous, while the "Sparse Tables" diagram shows rows of nulls intermixed

<img src="static/img/packed_tables.png" width="40%" style="background-color: white; padding: 1em;"/> <img src="static/img/sparse_tables.png" width="40%" style="background-color: white; padding: 1em;"/>

### File-Level Metadata

Some metadata is not arranged

## Signal Data Layouts

### Arrays and Columns

It is common in mass spectrometry to talk about a spectrum *having* an m/z array as synonymous with having been measured in the m/z dimension, and those m/z values are represented using some kind of physical data type in memory, likewise having an intensity array corresponding to the abundance of the signal parallel to the m/z array. In mzML, it is possible to use different physical data types for these two dimensions on different spectra in the same file, and there may well be legitimate use-cases for that. mzPeak can store array data in two ways. One way to store the arrays as columns in a signal data layout, burning the column into the schema and added to the `array index`. Another way is to store it as an `auxiliary array` which will be stored in the associated metadata table's `*.auxiliary_arrays` value for that entity's row. Auxiliary data arrays can be individually configured by the writer, have custom compression or data type decoding or cvParams, but it cannot be searched or sliced (read a segment of) without decoding the entire array, just as in mzML. By contrast, any array that is written as a column is encoded directly in Parquet, is part of the schema, and subject to its adaptive encoding process and compression. Currently, we assume that the first sorted array is the axis around which all other values are arranged, and any arrays that are shorter or longer _SHOULD_ instead be stored in `auxiliary_arrays` as well.

#### The Array Index

In order to properly annotate what kind of array a column *is*, we include a JSON-serialized `array index`, a table of data structures that describe each array in controlled vocabulary. A column is part of the Parquet file's schema and must always exist and have a homogenous type of value or a be marked `null` for each row. Columns can be sliced without needing to read the entire array.

### Data Arrays, Encoding, Transformations and Parquet

Parquet can write [page indices](https://parquet.apache.org/docs/file-format/pageindex/) on any column that is a *leaf* node in the schema based upon the value being stored prior to applying [encoding](https://parquet.apache.org/docs/file-format/data-pages/encodings/) and [compression](https://parquet.apache.org/docs/file-format/data-pages/compression/). To that effect, we must take care when trying to store data cleverly. The following section may refer to spectra, but these are applicable more broadly.

#### Zero Run Stripping

When storing spectrum data, some vendors will produce arrays with lots of "empty" regions filled with zero intensity values along a semi-regularly spaced m/z axis. These regions hold little information, so all but the first and last zero intensity points are removed. This is only meaningful for profile data. Readers SHOULD assume that zero runs have been stripped.

##### Null Marking

For spectra with many small gaps, even zero run stripping leaves too much unhelpful information in the data. We can instead replace the flanking zero intensity points with `null` m/z and intensity values and Parquet will skip storing the expensive 32- and/or 64-bit values, retaining only the validity buffer bit flag. We can separately fit a simple m/z spacing model using weighted least squares of the form:

$$
    δ mz \sim β_0 + β_1 mz + β_2 mz^2 + ϵ
$$

or using the following Python code:
<details>

<summary>Python code for fitting the weighted least squares model</summary>

```python
class DeltaCurveRegressionModel:
    beta: np.ndarray

    def __init__(self, beta: np.ndarray):
        self.beta = beta

    @classmethod
    def fit(
        cls,
        mz_array,
        delta_array,
        weights: np.ndarray | None = None,
        threshold: float | None = None,
        rank: int = 2,
    ):
        if weights is None:
            weights = np.ones(len(mz_array))
        else:
            weights = weights

        if threshold is None:
            threshold = 1.0

        # Drop all entries where the gap between m/z values > threshold
        raw = mz_array[1:][delta_array <= threshold]
        w = weights[1:][delta_array <= threshold]
        y = delta_array[delta_array <= threshold]

        # Build the design matrix
        data = [data.append(np.ones_like(raw))]
        for i in range(1, rank + 1):
            data.append(raw**i)
        data = np.stack(data, axis=-1)

        # Use the QR decomposition to solve the weighted least squares problem
        # to estimate weights predicting δ m/z.
        # https://stats.stackexchange.com/a/490782/59613
        chol_w = np.sqrt(w)
        qr = np.linalg.qr(chol_w[:, None] * data)
        v = qr.Q.T.dot(chol_w * y)
        beta = solve_triangular(qr.R, v)

        # Numerically equivalent to and more stable than the direct inversion
        # beta = np.linalg.inv((data.T * w).dot(data)).dot(data.T * w).dot(y)
        return cls(beta)

    def predict(self, mz: float) -> float:
        acc = self.beta[0]
        for i in range(1, len(self.beta)):
            acc += self.beta[i] * mz ** i
        return acc
```

</details>

Then when reading the the null-marked data, use either the local median $δ mz$ or the learned model for that spectrum to compute the m/z spacing for singleton points to achieve an very accurate reconstruction. Because the non-zero m/z points remain unchanged, the reconstructed signal's peak apex or centroid should be unaffected. If the peak is composed of only three points including the two zero intensity spots, no meaningful peak model can be fit in any case so the minute angle change this would induce are still effectively lossless.

![Thermo dataset with null marking](../static/thermo_null_marking_err.png)
![Sciex dataset with delta encoding and null marking](../static/sciex_null_marking_delta_encoding_error.png)

Keep in mind that all Numpress compression methods are still available and still provide superior size reduction, but carry this slightly larger loss of accuracy. Using a Numpress compression is a transformation that requires the [Chunked Layout](#chunked-layout).

### Point Layout

When storing data arrays, the point layout stores the data as-is in parallel arrays alongside a repeated index column. The top-level node is named `point` and it is a group with an arbitrary number of columns. The entity index column _MUST_ be the first column under `point`.

<img src="static/img/point_layout.png" height="600pt" style="background-color: white; padding: 1em;"/>

<style>
  .point-table {
    text-align: center;
  }

  .point-table thead th {
    text-align: center;
  }
</style>
<table class="point-table">
  <thead>
    <tr>
      <th colspan=3>point</th>
    </tr>
    <tr>
      <th>spectrum_index</th>
      <th>mz</th>
      <th>intensity</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1</td> <td>213.2</td><td>1002</td></tr>
    <tr><td>1 </td> <td>506.9</td><td>500</td> </tr>
    <tr><td>1 </td>  <td>758</td><td>405</td> </tr>
    <tr><td>...</td> <td>...</td><td>...</td> </tr>
    <tr><td>2 </td><td>329.1</td><td>50</td> </tr>
    <tr><td>2 </td><td>516.5</td><td>5002</td> </tr>
    <tr><td>2 </td><td>783.8</td><td>302</td> </tr>
  </tbody>
</table>

This layout is simple, but carries several advantages. Scalar columns are easily filtered along the page-level range index. This makes multi-dimensional queries easier to write and optimize. The arrays are transparently encoded and compressed by Parquet, so the data may still be stored compactly. The data must be stored as-is in order to use the page index so no additional obscuring transformations can be used.

<!-- The [zero run stripping](#zero-run-stripping) and [null marking](#null-marking) methods may still be employed as they only remove non-meaningful points from the array. -->

### Chunked Layout

When storing data arrays, the chunked layout treats one array, which must be sorted, as the "primary" axis, cutting the array into chunks of a fixed size along that coordinate space (e.g. steps of 50 m/z) and taking the same segments from parallel arrays. The primary axis chunks' start, end, and a repeated index are recorded as columns, and then each array may be encoded as-is or with an opaque transform (e.g. δ-encoding, Numpress). The start and end interval permits granular random access along the primary axis as well as the source index. The top-level node is named `chunk` and it has a layout as shown below. The entity index column _MUST_ be the first column under `chunk`.

<img src="static/img/chunked_layout.png" height="600pt" style="background-color: white; padding: 1em;"/>

<style>
  .chunk-table {
    text-align: center;
  }

  .chunk-table thead th {
    text-align: center;
  }
</style>
<table class="chunk-table">
<thead>
  <tr>
    <th colspan=6>chunk</th>
  </tr>
  <tr><th>spectrum_index</td><th>mz_start</td><th>mz_end</td><th>mz_chunk values</td><th>chunk_encoding</td><th>intensity</td></tr>
</thead>
<tbody>
<tr><td>1</td><td>200</td><td>250</td><td>[0.0013, ..., 0.0013]</td><td>MS:1003089</td><td>[...]</td></tr>
<tr><td>1</td><td>250</td><td>300</td><td>[0.0014, ..., 0.0014]</td><td>MS:1003089</td><td>[...]</td></tr>
<tr><td>1</td><td>500</td><td>550</td><td>[0.0014, ..., 0.0015]</td><td>MS:1003089</td><td>[...]</td></tr>
<tr><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td><td>...</td></tr>
<tr><td>2</td><td>200</td><td>250</td><td>[0.0013, ..., 0.0013]</td><td>MS:1003089</td><td>[...]</td></tr>
<tr><td>2</td><td>350</td><td>400</td><td>[0.0014, ..., 0.0014]</td><td>MS:1003089</td><td>[...]</td></tr>
<tr><td>2</td><td>400</td><td>450</td><td>[0.0013, ..., 0.0014]</td><td>MS:1003089</td><td>[...]</td></tr>
</tbody>
</table>

This example uses a δ-encoding for the m/z array chunks' values, which can be efficiently reconstructed with very high precision for 64-bit floats. The m/z values within the `mz_chunk_values` list aren't accessible to the page index, but the `_start` and `_end` columns are. The chunk values are still subject to Parquet encodings so they can be byte shuffled as well which further improves compression.

## Why all these root nodes?

**Couldn't we just unwrap the top-level struct and move on with things?**

Perhaps, but the top-level structure leaves the door open for two use-cases:

1. Unaligned proprietary data. A specialized writer or reader might wish to embed other information that is not directly connected to the primary schema's addressible unit (e.g. a spectrum, a data point), and this leaves open a door for that to be introduced. It is assumed that this is unlikely at this time, but it is a quantum physics universe.
2. More table packing. Early in mzPeak's design, we tried to pack tables together as much as possible as in the [packed parallel table](#packed-parallel-metadata-tables) layout, but this proved to be very inefficient to _write_ despite being no slower to _read_. This might have been an implementation detail, and not Parquet itself. We don't want to throw out the opportunity to return to that in the future, requiring a schema-breaking change rather than just how we get to the tables that break.

# Index File - `mzpeak_index.json`

An mzPeak archive is made up of multiple named files. To leave room for future files and avoid having to do complicated file name resolution, we use an index file that identifies the contents of each file. This broadly defines the kinds of schemas those files might have.

```json
{
  "files": [
    {
      "name": "spectra_data.mzpeak",
      "entity_type": "spectrum",
      "data_kind": "data arrays"
    },
    {
      "name": "spectra_metadata.mzpeak",
      "entity_type": "spectrum",
      "data_kind": "metadata"
    },
    {
      "name": "chromatograms_data.mzpeak",
      "entity_type": "chromatogram",
      "data_kind": "data arrays"
    },
    {
      "name": "chromatograms_metadata.mzpeak",
      "entity_type": "chromatogram",
      "data_kind": "metadata"
    }
  ],
  "metadata": {}
}
```

Governed by JSONSchema `schema/mzpeak_index.json`

The `data_kind` and `entity_kind` fields are loose enumerations. They are expected to grow over time.

## Data Kind

The `data_kind` field tells the reader the semantics of the data stored in this file, and approximately what kind of schema to expect.

There are currently 5 controlled values for `data_kind`:

- `data arrays`: Expected to use one of the [point](#point-layout) or [chunked](#chunked-layout) layout. These files contain the signal data, raw or processed for the `entity_type` being described.
- `peaks`: This, like `data arrays`, is expected to use the [point](#point-layout) or [chunked](#chunked-layout) layout as well. Where `data arrays` might store any kind of signal data, `peaks` implies that the data are processed and that there exists an entry in `data arrays` that is less refined. This is useful when storing both profile and centroid signal for a spectrum, for example.
- `metadata`: Expected to use the [packed parallel table](#packed-parallel-metadata-tables) layout. This describes the entity's metadata, everything but the homogenous signal arrays stored in the `data arrays` file. This file may still be large.
- `proprietary`: The layout and schema of this file is entirely the purview of the writer which may be an instrument vendor. These files should be ignored unless the reader _is_ for that instrument vendor. It may not be a Parquet file. Instrument vendors are encouraged to use this classification on binary files or other difficult to digest contents. Text or XML configuratin files may still be of interest to the broader community in an evolving metadata landscape.
- `other`: The file is none of the other listed types. This may not be a Parquet file.

Any value outside of these is assumed to be treated as `other`. Files labeled as `other`. Any files treated as `other` data kinds are implementation defined, as are `proprietary` files, but `other` files may be still be of interest to non-vendor readers.

## Entity Type

The `entity_type` tells the reader what is _being_ described in this file, in concert with the `data_kind`. This makes helps the reader connect the right file to the right API.

There are currently 3 controlled values for `entity_type`

- `spectrum`: The file describes spectra (mass or otherwise), entities defined as occuring at a singular point in time, or as semantically close to this as possible in the face of framed or cycled acquisition.
- `chromatogram`: The file describes chromatograms or other measurements _over time_ like diagnostic traces.
- `other`: The file is none of the other listed types. This may describe something not yet covered by the living specification.

Any value outside of these is assumed to be treated as `other`.

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
