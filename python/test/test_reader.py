from pathlib import Path

import pytest
import numpy as np
import pyarrow as pa

from mzpeak import MzPeakFile
from mzpeak.mz_reader import BufferFormat
from mzpeak.filters import find_zero_runs, is_zero_pair_mask, null_chunk_every

point_path = Path("small.mzpeak")
chunk_path = Path("small.chunked.mzpeak")
unpacked_path = Path("small.unpacked.mzpeak")


def common_checks(reader: MzPeakFile, subtests: pytest.Subtests):
    with subtests.test("file level metadata"):
        assert reader.file_index
        assert reader.file_metadata
        assert reader.spectrum_data.array_index

        assert len(reader) == 48
        assert len(reader.spectra) == 48
        assert len(reader.scans) == 48
        assert len(reader.selected_ions) == 34
        assert len(reader.precursors) == 34

        assert reader.file_metadata.keys() == {
            "data_processing_method_list",
            "file_description",
            "instrument_configuration_list",
            "run",
            "sample_list",
            "software_list",
            "spectrum_count",
            "spectrum_data_point_count",
        }

    with subtests.test("chromatogram"):
        assert len(reader.chromatograms) == 1
        assert len(reader.bpc()[0]) == 48
        assert len(reader.read_chromatogram(0)['time array']) == 48

    with subtests.test("spectrum 0"):
        spec = reader[0]
        assert spec['index'] == 0
        assert len(spec['m/z array']) == 13589
        assert len(spec["intensity array"]) == 13589

        if reader.has_secondary_peaks_data:
            peaks = reader.read_peaks_for(0)
            assert len(peaks['m/z array']) == 1612

    with subtests.test("spectrum 4"):
        spec = reader[4]
        assert len(spec["m/z array"]) == 837
        assert len(spec["intensity array"]) == 837
        assert spec["spectrum representation"] == "MS:1000127"
        assert spec['index'] == 4

def test_load_base_point(subtests: pytest.Subtests):
    reader = MzPeakFile(point_path)
    assert reader.spectrum_data.buffer_format() == BufferFormat.Point
    common_checks(reader, subtests)


def test_load_base_chunk(subtests: pytest.Subtests):
    reader = MzPeakFile(chunk_path)
    assert reader.spectrum_data.buffer_format() == BufferFormat.Chunk
    common_checks(reader, subtests)
    assert reader.has_secondary_peaks_data


def test_load_unpacked(subtests: pytest.Subtests):
    reader = MzPeakFile(unpacked_path)
    assert reader.spectrum_data.buffer_format() == BufferFormat.Point
    common_checks(reader, subtests)


def test_chunking():
    mzs = []
    intensities = []
    with Path("test/data/sparse_large_gaps.txt").open('rt') as fh:
        for line in fh:
            i, j = map(float, line.strip().split("\t"))
            mzs.append(i)
            intensities.append(j)

    mzs = np.array(mzs)
    intensities = np.array(intensities)

    assert len(intensities) == 9317
    assert np.count_nonzero(intensities) == 4243

    nonzero_run_mask = find_zero_runs(intensities)
    mzs = mzs[nonzero_run_mask]
    intensities = intensities[nonzero_run_mask]

    assert len(intensities) == 5546
    assert np.count_nonzero(intensities) == 4243

    zero_pair_mask = is_zero_pair_mask(intensities)
    assert zero_pair_mask.sum() == 1286

    mzs = pa.array(mzs, mask=zero_pair_mask)
    intensities = pa.array(intensities, mask=zero_pair_mask)
    assert mzs.null_count == 1286

    chunk_indices = len(null_chunk_every(mzs, 50))
    assert chunk_indices == 33