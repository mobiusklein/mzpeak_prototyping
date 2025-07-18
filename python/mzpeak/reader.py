import json
from pathlib import Path
import zipfile

from numbers import Number
from collections.abc import Iterable
from typing import Any, Generic, TypeVar
from typing import IO

import numpy as np
import pandas as pd

import pyarrow as pa

from pyarrow import compute as pc
from pyarrow import parquet as pq

from .filters import fill_nulls


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


def _clean_frame(df: pd.DataFrame):
    columns = df.columns[~df.isna().all(axis=0)]
    return df[columns]


class MzPeakSpectrumMetadataReader:
    handle: pq.ParquetFile
    meta: pq.FileMetaData
    num_spectra: int
    num_spectrum_points: int

    spectrum_index_i: int
    scan_index_i: int
    precursor_index_i: int
    selected_ion_i: int

    id_index: pd.Series
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
        self.spectra = _clean_frame(
            pa.record_batch(pa.concat_tables(spectra)["spectrum"].combine_chunks())
            .to_pandas()
            .set_index("index")
        )
        self.id_index = self.spectra[["id"]].reset_index().set_index("id")["index"]

    def _read_scans(self):
        blocks = []
        for i in range(self.meta.num_row_groups):
            rg = self.meta.row_group(i)
            col_idx = rg.column(self.scan_index_i)
            if col_idx.statistics.has_min_max:
                bat = self.handle.read_row_group(i, columns=["scan"])
                blocks.append(bat.filter(bat[0].is_valid()))
        self.scans = _clean_frame(
            pa.record_batch(pa.concat_tables(blocks)["scan"].combine_chunks())
            .to_pandas()
            .set_index("spectrum_index")
        )

    def _read_precursors(self):
        blocks = []
        for i in range(self.meta.num_row_groups):
            rg = self.meta.row_group(i)
            col_idx = rg.column(self.precursor_index_i)
            if col_idx.statistics.has_min_max:
                bat = self.handle.read_row_group(i, columns=["precursor"])
                blocks.append(bat.filter(bat[0].is_valid()))
        self.precursors = _clean_frame(
            pa.record_batch(pa.concat_tables(blocks)["precursor"].combine_chunks())
            .to_pandas()
            .set_index("spectrum_index")
        )

    def _read_selected_ions(self):
        blocks = []
        for i in range(self.meta.num_row_groups):
            rg = self.meta.row_group(i)
            col_idx = rg.column(self.selected_ion_i)
            if col_idx.statistics.has_min_max:
                bat = self.handle.read_row_group(i, columns=["selected_ion"])
                blocks.append(bat.filter(bat[0].is_valid()))
        self.selected_ions = _clean_frame(
            pa.record_batch(pa.concat_tables(blocks)["selected_ion"].combine_chunks())
            .to_pandas()
            .set_index("spectrum_index")
        )

    def __getitem__(self, i: int | str):
        if isinstance(i, str):
            i = self.id_index[i]
        spec = self.spectra.loc[i].to_dict()
        spec["parameters"] = [normalize_value_of(v) for v in spec["parameters"]]
        spec["scans"] = self.scans.loc[i].to_dict()
        try:
            precursors_of = self.precursors.loc[[i]]
            precursors_of["activation"] = precursors_of["activation"].apply(
                lambda x: [normalize_value_of(v) for v in x["parameters"]]
            )
            try:
                ions = self.selected_ions.loc[[i]]
                ions["parameters"] = ions["parameters"].apply(
                    lambda x: [normalize_value_of(v) for v in x]
                )
                precursors_of = precursors_of.merge(ions, on="precursor_index")
            except KeyError:
                pass
            spec["precursors"] = precursors_of.to_dict()
        except KeyError:
            pass
        spec["index"] = i
        return spec

    def __len__(self):
        return self.spectra.index.size

    def __repr__(self):
        return f"{self.__class__.__name__}({self.handle})"

    def _get_median_deltas(self):
        if "median_delta" in self.spectra:
            return self.spectra["median_delta"].to_numpy()
        return None


class _SpectrumDataIndex:
    meta: pq.FileMetaData
    prefix: str
    index_i: int
    init: bool

    def __init__(self, meta: pq.FileMetaData, prefix: str):
        self.meta = meta
        self.prefix = prefix
        self.index_i = 0
        self.init = False
        self._infer_schema_idx()

    def _infer_schema_idx(self):
        rg = self.meta.row_group(0)
        q = f"{self.prefix}.spectrum_index"
        for i in range(rg.num_columns):
            col = rg.column(i)
            if col.path_in_schema == q:
                self.index_i = i
                self.init = True

        index = json.loads(self.meta.metadata[b"spectrum_array_index"])
        self.array_index = {
            entry["path"].rsplit(".")[-1]: entry for entry in index["entries"]
        }
        self.n_spectra = int(self.meta.metadata[b"spectrum_count"])

    def row_groups_for_index(self, spectrum_index: int):
        rgs = []
        if not self.init:
            return rgs
        for i in range(self.meta.num_row_groups):
            rg = self.meta.row_group(i)
            col_idx = rg.column(self.index_i)
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
        return rgs

    def row_groups_for_spectrum_range(
        self, spectrum_index_range: slice | list
    ) -> list[int]:
        if isinstance(spectrum_index_range, slice):
            start = spectrum_index_range.start or 0
            end = spectrum_index_range.stop or self.n_spectra
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
        return rgs


class MzPeakSpectrumDataReader:
    handle: pq.ParquetFile
    meta: pq.FileMetaData
    _point_index: _SpectrumDataIndex
    _chunk_index: _SpectrumDataIndex
    array_index: dict[str, dict]
    n_spectra: int
    _median_delta_series: np.ndarray | None
    _do_null_filling: bool = True

    def __init__(self, handle: pq.ParquetFile):
        if not isinstance(handle, pq.ParquetFile):
            handle = pq.ParquetFile(handle)
        self.handle = handle
        self.meta = self.handle.metadata
        self._point_index = _SpectrumDataIndex(self.meta, "point")
        self._chunk_index = _SpectrumDataIndex(self.meta, "chunk")
        self._infer_schema_idx()
        self._median_delta_series = None

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

    def _clean_point_batch(
        self, data: pa.RecordBatch, median_delta: float | None = None
    ):
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
        result = {}
        for k, v in zip(data.column_names, data.columns):
            if v.null_count and self._do_null_filling:
                if (
                    k == "m/z array"
                    and median_delta is not None
                    and not np.any(np.isnan(median_delta))
                ):
                    v = fill_nulls(v, median_delta)
                elif (
                    k == "intensity array"
                    and median_delta is not None
                    and not np.any(np.isnan(median_delta))
                ):
                    v = np.asarray(v)
                    v[np.isnan(v)] = 0.0
            result[k] = np.asarray(v)
        return result

    def read_data_for_spectrum_range(self, spectrum_index_range: slice | list):
        prefix = "point"
        rgs = self._point_index.row_groups_for_spectrum_range(spectrum_index_range)
        if not rgs:
            rgs = self._chunk_index.row_groups_for_spectrum_range(spectrum_index_range)
            if rgs:
                prefix = "chunk"

        is_slice = False
        if isinstance(spectrum_index_range, slice):
            start = spectrum_index_range.start or 0
            end = spectrum_index_range.stop or self.n_spectra
            is_slice = True
        else:
            start = min(spectrum_index_range)
            end = max(spectrum_index_range)

        if prefix == "point":
            return self._read_point_range(
                start, end, spectrum_index_range, is_slice, rgs
            )
        elif prefix == "chunk":
            return self._read_chunk_range(
                start, end, spectrum_index_range, is_slice, rgs
            )
        else:
            raise NotImplementedError(prefix)

    def _expand_chunks(
        self, chunks: list[dict[str, Any]], axis_prefix: str = "spectrum_mz_f64"
    ) -> dict[str, np.ndarray]:
        n = 0
        for chunk in chunks:
            # The +1 is to account for the starting point
            n += len(chunk[f"{axis_prefix}_chunk_values"]) + 1
        if n == 0:
            return {axis_prefix: np.array([])}

        arrays_of = {}
        for k, v in chunks[0].items():
            if k in ("spectrum_index", "chunk_encoding") or k.startswith(axis_prefix):
                continue
            arrays_of[k] = np.zeros(n, dtype=np.asarray(v.values).dtype)

        main_axis_array = np.zeros(n)
        offset = 0
        for chunk in chunks:
            start = chunk[f"{axis_prefix}_chunk_start"].as_py()
            steps = chunk[f"{axis_prefix}_chunk_values"]
            chunk_size = len(steps) + 1
            main_axis_array[offset : offset + chunk_size] = start
            main_axis_array[offset + 1 : offset + chunk_size] += np.cumsum(steps.values)

            for k, v in chunk.items():
                if k in ("spectrum_index", "chunk_encoding") or k.startswith(
                    axis_prefix
                ):
                    continue
                arrays_of[k][offset : offset + chunk_size] = np.asarray(v.values)

            offset += chunk_size
        arrays_of[axis_prefix] = main_axis_array

        rename_map = {
            k: v["array_name"] for k, v in self.array_index.items() if k in arrays_of
        }
        for k, v in rename_map.items():
            arrays_of[v] = arrays_of.pop(k)
        return arrays_of

    def _read_point(
        self,
        spectrum_index: int,
        rgs: list[int],
        median_delta: float | list[float] | None,
    ):
        block = self.handle.read_row_groups(rgs, columns=["point"])["point"]
        block = pc.filter(
            block, pc.equal(pc.struct_field(block, "spectrum_index"), spectrum_index)
        )

        data: pa.RecordBatch = pa.record_batch(block.combine_chunks()).drop_columns(
            "spectrum_index"
        )
        return self._clean_point_batch(data, median_delta)

    def _read_point_range(
        self,
        start: int,
        end: int,
        spectrum_index_range: list[int] | slice,
        is_slice: bool,
        rgs: list[int],
    ):
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
        return self._clean_point_batch(data)

    def _read_chunk(self, spectrum_index: int, rgs: list[int]):
        chunks = []
        for batch in self.handle.iter_batches(128, row_groups=rgs, columns=["chunk"]):
            batch = batch["chunk"]
            batch = pc.filter(
                batch,
                pc.equal(pc.struct_field(batch, "spectrum_index"), spectrum_index),
            )
            if len(batch) == 0:
                if chunks:
                    break
                else:
                    continue
            if isinstance(batch, pa.ChunkedArray):
                chunks.extend(batch.chunks)
            else:
                chunks.append(batch)
        chunks = pa.chunked_array(chunks)
        return self._expand_chunks(chunks)

    def _read_chunk_range(
        self,
        start: int,
        end: int,
        spectrum_index_range: list[int] | slice,
        is_slice: bool,
        rgs: list[int],
    ):
        chunks = []
        for batch in self.handle.iter_batches(128, row_groups=rgs, columns=["chunk"]):
            batch = batch["chunk"]
            idx_col = pc.struct_field(batch, "spectrum_index")
            batch_end = pc.max(idx_col)
            if batch_end < start:
                continue
            if is_slice:
                mask = pc.and_(
                    pc.less_equal(idx_col, end), pc.greater_equal(idx_col, start)
                )
            else:
                mask = pc.is_in(idx_col, pa.array(spectrum_index_range))
            batch = pc.filter(
                batch,
                mask,
            )
            if len(batch) == 0:
                if batch_end > end:
                    break
                else:
                    continue
            if isinstance(batch, pa.ChunkedArray):
                chunks.extend(batch.chunks)
            else:
                chunks.append(batch)
        chunks = pa.chunked_array(chunks)
        return self._expand_chunks(chunks)

    def _read_data_for(self, spectrum_index: int):
        if self._median_delta_series is not None:
            median_delta = self._median_delta_series[spectrum_index]
        else:
            median_delta = None

        prefix = "point"
        rgs = self._point_index.row_groups_for_index(spectrum_index)
        if not rgs:
            rgs = self._chunk_index.row_groups_for_index(spectrum_index)
            if rgs:
                prefix = "chunk"
        if prefix == "point":
            return self._read_point(spectrum_index, rgs, median_delta)
        elif prefix == "chunk":
            return self._read_chunk(
                spectrum_index,
                rgs,
            )
        else:
            raise NotImplementedError(prefix)

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return self.read_data_for_spectrum_range(index)
        return self._read_data_for(index)


class MzPeakFile:
    archive: zipfile.ZipFile
    spectrum_data: MzPeakSpectrumDataReader
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
        metadata = {}
        if self.spectrum_metadata:
            for k, v in self.spectrum_metadata.meta.metadata.items():
                k = k.decode("utf8")
                if k == "ARROW:schema":
                    continue
                v = json.loads(v)
                metadata[k] = v
        self.file_metadata = metadata

        if self.spectrum_data and self.spectrum_metadata:
            self.spectrum_data._median_delta_series = (
                self.spectrum_metadata._get_median_deltas()
            )

    def __getitem__(self, index):
        if isinstance(index, (int, str)):
            spec = self.spectrum_metadata[index]
            index = spec["index"]
            data = self.spectrum_data[index]
            spec.update(data)
        elif isinstance(index, Iterable):
            if not index:
                return []
            spec = [self[i] for i in index]
        elif isinstance(index, slice):
            start = index.start or 0
            end = index.stop or len(self)
            step = index.step or 1
            spec = self[range(start, end, step)]
        return spec

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.spectrum_metadata)

    def tic(self):
        return self.spectrum_metadata.tic()

    def bpc(self):
        return self.spectrum_metadata.bpc()
