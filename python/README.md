# mzpeak

A Python of the mzPeak file format.

**NOTE**: This is a **work in progress**, no stability is guaranteed at this point.

## Usage

The `MzPeakFile` class handles the all the internal details of reading the archive.

```python
from mzpeak import MzPeakFile

reader = MzPeakFile("small.mzpeak")

spec = reader[2]
```

The `MzPeakFile` has a `Sequence`-like interface for spectra. It supports random access and sequential iteration.
For chromatograms, use the `get_chromatogram` method.

## To be implemented

- Filtering by m/z while filtering by index range
- Slicing by time range
- Slicing chromatograms
- More efficient block caching