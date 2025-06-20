import json
from pathlib import Path
import zipfile

from numbers import Number
from typing import Generic, TypeVar
from typing import IO

import numpy as np
import pandas as pd

import pyarrow as pa

from pyarrow import compute as pc
from pyarrow import parquet as pq


Q = TypeVar("Q", bound=Number)


class Span(Generic[Q]):
    start: Q
    end: Q

    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __contains__(self, val: Q) -> bool:
        return self.start <= val and val <= self.end

    def overlaps(self, other: "Span[Q]") -> bool:
        return self.end >= other.start and other.end >= self.start

    def size(self):
        return self.end - self.start

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.start}, {self.end})"


def value_normalize(val: dict):
    for v in val.values():
        if v is not None:
            return v
    return None


def normalize_value_of(param: dict):
    param = param.copy()
    param["value"] = value_normalize(param["value"])
    return param


class MzPeakSpectrumMetadataReader:
    handle: pq.ParquetFile
    meta: pq.FileMetaData
    num_spectra: int
    num_spectrum_points: int

    spectrum_index_i: int
    scan_index_i: int
    precursor_index_i: int
    selected_ion_i: int

    spectra: pd.DataFrame
    scans: pd.DataFrame
    precursors: pd.DataFrame
    selected_ions: pd.DataFrame

    def __init__(self, handle: pq.ParquetFile):
        if not isinstance(handle, pq.ParquetFile):
            handle = pq.ParquetFile(handle)
        self.handle = handle
        self.meta = handle.metadata
        self.num_spectra = int(handle.metadata.metadata[b"spectrum_count"])
        self.num_spectrum_points = int(
            handle.metadata.metadata[b"spectrum_data_point_count"]
        )
        self._infer_schema_idx()
        self._read_spectra()
        self._read_scans()
        self._read_precursors()
        self._read_selected_ions()

    def tic(self):
        return np.array(self.spectra["time"]), np.array(
            self.spectra["total_ion_current"]
        )

    def bpc(self):
        return np.array(self.spectra["time"]), np.array(
            self.spectra["base_peak_intensity"]
        )

    def _infer_schema_idx(self):
        rg = self.meta.row_group(0)
        for i in range(rg.num_columns):
            col = rg.column(i)
            if col.path_in_schema == "spectrum.index":
                self.spectrum_index_i = i
            elif col.path_in_schema == "scan.spectrum_index":
                self.scan_index_i = i
            elif col.path_in_schema == "precursor.spectrum_index":
                self.precursor_index_i = i
            elif col.path_in_schema == "selected_ion.spectrum_index":
                self.selected_ion_i = i

    def _read_spectra(self):
        spectra = []
        for i in range(self.meta.num_row_groups):
            rg = self.meta.row_group(i)
            col_idx = rg.column(self.spectrum_index_i)
            if col_idx.statistics.has_min_max:
                bat = self.handle.read_row_group(i, columns=["spectrum"])
                spectra.append(bat.filter(bat[0].is_valid()))
        self.spectra = (
            pa.record_batch(pa.concat_tables(spectra)["spectrum"].combine_chunks())
            .to_pandas()
            .set_index("index")
            .dropna(axis=1)
        )

    def _read_scans(self):
        blocks = []
        for i in range(self.meta.num_row_groups):
            rg = self.meta.row_group(i)
            col_idx = rg.column(self.scan_index_i)
            if col_idx.statistics.has_min_max:
                bat = self.handle.read_row_group(i, columns=["scan"])
                blocks.append(bat.filter(bat[0].is_valid()))
        self.scans = (
            pa.record_batch(pa.concat_tables(blocks)["scan"].combine_chunks())
            .to_pandas()
            .set_index("spectrum_index")
            .dropna(axis=1)
        )

    def _read_precursors(self):
        blocks = []
        for i in range(self.meta.num_row_groups):
            rg = self.meta.row_group(i)
            col_idx = rg.column(self.precursor_index_i)
            if col_idx.statistics.has_min_max:
                bat = self.handle.read_row_group(i, columns=["precursor"])
                blocks.append(bat.filter(bat[0].is_valid()))
        self.precursors = (
            pa.record_batch(pa.concat_tables(blocks)["precursor"].combine_chunks())
            .to_pandas()
            .set_index("spectrum_index")
            .dropna(axis=1)
        )

    def _read_selected_ions(self):
        blocks = []
        for i in range(self.meta.num_row_groups):
            rg = self.meta.row_group(i)
            col_idx = rg.column(self.selected_ion_i)
            if col_idx.statistics.has_min_max:
                bat = self.handle.read_row_group(i, columns=["selected_ion"])
                blocks.append(bat.filter(bat[0].is_valid()))
        self.selected_ions = (
            pa.record_batch(pa.concat_tables(blocks)["selected_ion"].combine_chunks())
            .to_pandas()
            .set_index("spectrum_index")
            .dropna(axis=1)
        )

    def __getitem__(self, i):
        spec = self.spectra.loc[i].to_dict()
        spec['parameters'] = [normalize_value_of(v) for v in spec['parameters']]
        spec["scans"] = self.scans.loc[i].to_dict()
        try:
            precursors_of = self.precursors.loc[[i]]
            precursors_of['activation'] = precursors_of["activation"].apply(lambda x: [normalize_value_of(v) for v in x['parameters']])
            try:
                ions = self.selected_ions.loc[[i]]
                ions["parameters"] = [normalize_value_of(v) for v in ions["parameters"]]
                precursors_of = precursors_of.merge(ions, on="precursor_index")
            except KeyError:
                pass
            spec["precursors"] = precursors_of.to_dict()
        except KeyError:
            pass
        return spec

    def __len__(self):
        return self.spectra.index.size

    def __repr__(self):
        return f"{self.__class__.__name__}({self.handle})"


class MzPeakSpectrumDataReader:
    handle: pq.ParquetFile
    meta: pq.FileMetaData
    point_index_i: int
    array_index: dict[str, dict]
    n_spectra: int

    def __init__(self, handle: pq.ParquetFile):
        if not isinstance(handle, pq.ParquetFile):
            handle = pq.ParquetFile(handle)
        self.handle = handle
        self.meta = self.handle.metadata
        self._infer_schema_idx()

    def _infer_schema_idx(self):
        rg = self.meta.row_group(0)
        for i in range(rg.num_columns):
            col = rg.column(i)
            if col.path_in_schema == "point.spectrum_index":
                self.point_index_i = i

        index = json.loads(self.meta.metadata[b"spectrum_array_index"])
        self.array_index = {
            entry["path"].rsplit(".")[-1]: entry for entry in index["entries"]
        }
        self.n_spectra = int(self.meta.metadata[b"spectrum_count"])

    def _clean_batch(self, data: pa.RecordBatch):
        nulls = []
        for i, col in enumerate(data.columns):
            if col.null_count == len(col):
                nulls.append(i)
        if len(nulls) > 1:
            nulls.sort()
        for i, j in enumerate(nulls):
            data = data.drop_columns(data.schema[j - i].name)
        data = data.rename_columns(
            {
                k: v["array_name"]
                for k, v in self.array_index.items()
                if k in data.column_names
            }
        )
        return {k: np.asarray(v) for k, v in zip(data.column_names, data.columns)}

    def read_data_for_spectrum_range(self, spectrum_index_range: slice | list):
        is_slice = False
        if isinstance(spectrum_index_range, slice):
            start = spectrum_index_range.start or 0
            end = spectrum_index_range.stop or self.n_spectra
            is_slice = True
        else:
            start = min(spectrum_index_range)
            end = max(spectrum_index_range)

        span = Span(start, end)
        rgs = []
        for i in range(self.meta.num_row_groups):
            rg = self.meta.row_group(i)
            col_idx = rg.column(self.point_index_i)
            if col_idx.statistics.has_min_max:
                other = Span(col_idx.statistics.min, col_idx.statistics.max)
                if span.overlaps(other):
                    rgs.append(i)
                if other.start > span.end:
                    break
            else:
                break

        block = self.handle.read_row_groups(rgs, columns=["point"])["point"]
        idx_col = pc.struct_field(block, "spectrum_index")

        if is_slice:
            mask = pc.and_(
                pc.less_equal(idx_col, end), pc.greater_equal(idx_col, start)
            )
        else:
            mask = pc.is_in(idx_col, pa.array(spectrum_index_range))
        block = pc.filter(block, mask)
        data: pa.RecordBatch = pa.record_batch(block.combine_chunks()).drop_columns(
            "spectrum_index"
        )
        return self._clean_batch(data)

    def _read_data_for(self, spectrum_index: int):
        rgs = []
        for i in range(self.meta.num_row_groups):
            rg = self.meta.row_group(i)
            col_idx = rg.column(self.point_index_i)
            if col_idx.statistics.has_min_max:
                if (
                    col_idx.statistics.min <= spectrum_index
                    and spectrum_index <= col_idx.statistics.max
                ):
                    rgs.append(i)
                if col_idx.statistics.min > spectrum_index:
                    break
            else:
                break
        block = self.handle.read_row_groups(rgs, columns=["point"])["point"]
        block = pc.filter(
            block, pc.equal(pc.struct_field(block, "spectrum_index"), spectrum_index)
        )

        data: pa.RecordBatch = pa.record_batch(block.combine_chunks()).drop_columns(
            "spectrum_index"
        )
        return self._clean_batch(data)

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return self.read_data_for_spectrum_range(index)
        return self._read_data_for(index)


class MzPeakFile:
    archive: zipfile.ZipFile
    spectrum_data: pq.ParquetFile
    spectrum_metadata: MzPeakSpectrumMetadataReader

    def __init__(self, path: str | Path | zipfile.ZipFile | IO[bytes]):
        if isinstance(path, zipfile.ZipFile):
            self.archive = path
        else:
            self.archive = zipfile.ZipFile(path)
        for f in self.archive.filelist:
            if f.filename == "spectra_data.mzpeak":
                self.spectrum_data = MzPeakSpectrumDataReader(
                    pa.PythonFile(self.archive.open(f))
                )
            elif f.filename == "spectra_metadata.mzpeak":
                self.spectrum_metadata = MzPeakSpectrumMetadataReader(
                    pa.PythonFile(self.archive.open(f))
                )

    def __getitem__(self, index: int):
        meta = self.spectrum_metadata[index]
        data = self.spectrum_data[index]
        return meta, data

    def __len__(self):
        return len(self.spectrum_metadata)

    def tic(self):
        return self.spectrum_metadata.tic()

    def bpc(self):
        return self.spectrum_metadata.bpc()