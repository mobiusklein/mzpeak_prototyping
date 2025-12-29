from pathlib import Path

import pytest

from mzpeak import MzPeakFile
from mzpeak.mz_reader import BufferFormat

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

    with subtests.test("read slice"):
        idx_slc = reader.time.resolve(slice(0.3, 0.4))
        values = reader.spectra_signal_for_indices(idx_slc)
        assert len(values) > 0


def check_iterator(reader: MzPeakFile, n: int=10):
    it = iter(reader)

    for i in range(n):
        spec = next(it)
        assert spec['index'] == i


def test_load_base_point(subtests: pytest.Subtests):
    reader = MzPeakFile(point_path)
    assert reader.spectrum_data.buffer_format() == BufferFormat.Point
    common_checks(reader, subtests)
    with subtests.test("iterator"):
        check_iterator(reader)


def test_load_base_chunk(subtests: pytest.Subtests):
    reader = MzPeakFile(chunk_path)
    assert reader.spectrum_data.buffer_format() == BufferFormat.Chunk
    common_checks(reader, subtests)
    assert reader.has_secondary_peaks_data
    with subtests.test("iterator"):
        check_iterator(reader, len(reader))

def test_load_unpacked(subtests: pytest.Subtests):
    reader = MzPeakFile(unpacked_path)
    assert reader.spectrum_data.buffer_format() == BufferFormat.Point
    common_checks(reader, subtests)
    with subtests.test("iterator"):
        check_iterator(reader)


