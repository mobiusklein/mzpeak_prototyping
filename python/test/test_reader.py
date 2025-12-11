from pathlib import Path

from mzpeak import MzPeakFile
from mzpeak.mz_reader import BufferFormat

point_path = Path("small.mzpeak")
chunk_path = Path("small.chunked.mzpeak")
unpacked_path = Path("small.unpacked.mzpeak")


def common_checks(reader: MzPeakFile):
    assert reader.file_index
    assert reader.file_metadata
    assert reader.spectrum_data.array_index
    assert len(reader) == 48
    assert len(reader.chromatograms) == 1
    assert len(reader.bpc()[0]) == 48


def test_load_base_point():
    reader = MzPeakFile(point_path)
    assert reader.spectrum_data.buffer_format() == BufferFormat.Point
    common_checks(reader)


def test_load_base_chunk():
    reader = MzPeakFile(chunk_path)
    assert reader.spectrum_data.buffer_format() == BufferFormat.Chunk
    common_checks(reader)


def test_load_unpacked():
    reader = MzPeakFile(unpacked_path)
    assert reader.spectrum_data.buffer_format() == BufferFormat.Point
    common_checks(reader)