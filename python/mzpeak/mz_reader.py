import logging
import json

from typing import Any, Iterator
from enum import Enum

import numpy as np

import pynumpress
import pyarrow as pa

from pyarrow import compute as pc
from pyarrow import parquet as pq

from .util import Span
from .filters import null_delta_decode, fill_nulls

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class _SpectrumDataIndex:
    meta: pq.FileMetaData
    prefix: str
    index_i: int
    init: bool
    row_group_index_ranges: list[Span[int] | None]

    def __init__(self, meta: pq.FileMetaData, prefix: str):
        self.meta = meta
        self.prefix = prefix
        self.index_i = 0
        self.init = False
        self.row_group_index_ranges = []
        self._infer_schema_idx()

    def _infer_schema_idx(self):
        rg = self.meta.row_group(0)
        q = f"{self.prefix}.spectrum_index"
        self.row_group_index_ranges = []
        for i in range(rg.num_columns):
            col = rg.column(i)
            if col.path_in_schema == q:
                self.index_i = i
                self.init = True

        if self.index_i is not None:
            for i in range(self.meta.num_row_groups):
                rg = self.meta.row_group(i)
                col_idx = rg.column(self.index_i)
                if col_idx.statistics.has_min_max:
                    self.row_group_index_ranges.append(
                        Span(col_idx.statistics.min, col_idx.statistics.max)
                    )
                else:
                    self.row_group_index_ranges.append(None)

        index = json.loads(self.meta.metadata[b"spectrum_array_index"])
        self.array_index = {
            entry["path"].rsplit(".")[-1]: entry for entry in index["entries"]
        }
        self.n_spectra = int(self.meta.metadata[b"spectrum_count"])

    def row_groups_for_index(self, spectrum_index: int) -> list[int]:
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
        if not self.init:
            return []
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
            col_idx = rg.column(self.index_i)
            if col_idx.statistics.has_min_max:
                other = Span(col_idx.statistics.min, col_idx.statistics.max)
                if span.overlaps(other):
                    rgs.append(i)
                if other.start > span.end:
                    break
            else:
                break
        return rgs


class BufferFormat(Enum):
    Point = 1
    Chunk = 2


class _ChunkIterator:
    it: Iterator[pa.RecordBatch]
    batch: pa.StructArray
    current_index: int
    index_column: str

    def __init__(
        self,
        it: Iterator[pa.StructArray],
        current_index: int = None,
        index_column: str = "spectrum_index",
    ):
        self.it = it
        self.buffer = []
        self.batch = None
        self.current_index = current_index
        self.index_column = index_column
        self._read_next_chunk()

    def _infer_starting_index(self):
        return pc.min(pc.struct_field(self.batch, self.index_column)).as_py()

    def __next__(self):
        batch = self._extract_next()
        i = self.current_index
        self.current_index += 1
        return i, batch

    def __iter__(self):
        return self

    def _read_next_chunk(self):
        logger.debug(f"Reading next batch looking for {self.current_index}")
        batch = next(self.it).column(0)
        lowest_index = pc.min(pc.struct_field(batch, self.index_column)).as_py()
        if (
            self.current_index is not None and lowest_index > self.current_index
        ) or self.current_index is None:
            self.current_index = lowest_index
        logger.debug(f"New batch starts with {lowest_index}")
        self.batch = batch

    def _extract_next(self):
        mask = pc.equal(
            pc.struct_field(self.batch, self.index_column), self.current_index
        )
        indices = np.where(mask)[0]
        if len(indices) == 0:
            n = len(self.batch)
            chunk = self.batch.slice(0, 0)
        else:
            start = indices[0]
            n = len(indices)

            chunk = self.batch.slice(start, n)

        if n == len(self.batch):
            self._read_next_chunk()
            rest = self._extract_next()
            chunk = pa.concat_arrays([chunk, rest])
        else:
            self.batch = self.batch.slice(n)
        return chunk

    def seek(self, index: int):
        if index < self.current_index:
            raise ValueError("Cannot rewind iterator")
        if index == self.current_index:
            return self
        else:
            while self.current_index != index:
                next(self)
            return self


DELTA_ENCODING = {"cv_id": 1, "accession": 1003089}
NO_COMPRESSION = {"cv_id": 1, "accession": 1000576}
NUMPRESS_LINEAR = {"cv_id": 1, "accession": 1002312}


class MzPeakSpectrumDataReader:
    handle: pq.ParquetFile
    meta: pq.FileMetaData
    _point_index: _SpectrumDataIndex
    _chunk_index: _SpectrumDataIndex
    array_index: dict[str, dict]
    n_spectra: int
    _delta_model_series: np.ndarray | None
    _do_null_filling: bool = True

    def __init__(self, handle: pq.ParquetFile):
        if not isinstance(handle, pq.ParquetFile):
            handle = pq.ParquetFile(handle)
        self.handle = handle
        self.meta = self.handle.metadata
        self._point_index = _SpectrumDataIndex(self.meta, "point")
        self._chunk_index = _SpectrumDataIndex(self.meta, "chunk")
        self._infer_schema_idx()
        self._delta_model_series = None

    def _infer_schema_idx(self):
        index = json.loads(self.meta.metadata[b"spectrum_array_index"])
        self.array_index = {
            entry["path"].rsplit(".")[-1]: entry for entry in index["entries"]
        }
        self.n_spectra = int(self.meta.metadata[b"spectrum_count"])

    def _clean_point_batch(
        self,
        data: pa.RecordBatch | pa.StructArray,
        delta_model: float | np.ndarray | None = None,
    ):
        if isinstance(data, pa.StructArray):
            nulls = []
            for i, name in enumerate(data.type):
                col = data.field(name)
                if col.null_count == len(col):
                    nulls.append(i)
            if len(nulls) > 1:
                nulls.sort()

            fields = [f.name for f in data.type]
            data = {f: v for f, v in zip(fields, data.flatten())}

            for i in nulls:
                data.pop(fields[i])

            for k, v in {
                k: v["array_name"]
                for k, v in self.array_index.items()
                if k in data.keys()
            }.items():
                data[v] = data.pop(k)
            it = data.items()
        else:
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
            it = zip(data.column_names, data.columns)

        result = {}
        for k, v in it:
            if v.null_count and self._do_null_filling:
                if (
                    k == "m/z array"
                    and delta_model is not None
                    and not np.any(np.isnan(delta_model))
                ):
                    v = fill_nulls(v, delta_model)
                elif (
                    k == "intensity array"
                    and delta_model is not None
                    and not np.any(np.isnan(delta_model))
                ):
                    v = np.asarray(v)
                    v[np.isnan(v)] = 0.0
            result[k] = np.asarray(v)
        return result

    def read_data_for_spectrum_range(self, spectrum_index_range: slice | list):
        prefix = BufferFormat.Point
        rgs = self._point_index.row_groups_for_spectrum_range(spectrum_index_range)
        if not rgs:
            rgs = self._chunk_index.row_groups_for_spectrum_range(spectrum_index_range)
            if rgs:
                prefix = BufferFormat.Chunk

        is_slice = False
        if isinstance(spectrum_index_range, slice):
            start = spectrum_index_range.start or 0
            end = spectrum_index_range.stop or self.n_spectra
            is_slice = True
        else:
            start = min(spectrum_index_range)
            end = max(spectrum_index_range)

        if prefix == BufferFormat.Point:
            return self._read_point_range(
                start, end, spectrum_index_range, is_slice, rgs
            )
        elif prefix == BufferFormat.Chunk:
            return self._read_chunk_range(
                start, end, spectrum_index_range, is_slice, rgs
            )
        else:
            raise NotImplementedError(prefix)

    def _expand_chunks(
        self,
        chunks: list[dict[str, Any]],
        axis_prefix: str = "spectrum_mz_f64",
        delta_model: float | np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        n = 0
        numpress_chunks = []
        for chunk in chunks:
            # The +1 is to account for the starting point
            encoding = chunk["chunk_encoding"].as_py()
            if encoding == DELTA_ENCODING or encoding == NO_COMPRESSION:
                n += len(chunk[f"{axis_prefix}_chunk_values"]) + 1
            elif encoding == NUMPRESS_LINEAR:
                raw = chunk[f"{axis_prefix}_numpress_bytes"].as_py()
                part = pynumpress.decode_linear(raw)
                n += len(part)
                numpress_chunks.append(part)
            else:
                raise ValueError(f"Unsupported chunk encoding {encoding}")

        if n == 0:
            return {axis_prefix: np.array([])}

        arrays_of = {}
        for k, v in chunks[0].items():
            if (
                k == "spectrum_index"
                or k == "chunk_encoding"
                or k.startswith(axis_prefix)
            ):
                continue
            else:
                arrays_of[k] = np.zeros(n, dtype=np.asarray(v.values).dtype)

        main_axis_array = np.zeros(n)
        offset = 0
        had_nulls = False
        numpress_chunks_it = iter(numpress_chunks)
        for i, chunk in enumerate(chunks):
            start = chunk[f"{axis_prefix}_chunk_start"].as_py()
            steps = chunk[f"{axis_prefix}_chunk_values"]
            encoding = chunk["chunk_encoding"].as_py()

            # Delta encoding
            if encoding == DELTA_ENCODING:
                if steps.values.null_count > 0:
                    had_nulls = True
                    # The presence null values leads to sometimes restoring one fewer values because the chunk start is
                    # included in the data itself.
                    steps = null_delta_decode(
                        steps.values, pa.scalar(start, type=steps.values.type)
                    )
                    chunk_size = len(steps)
                    if delta_model is not None:
                        steps = fill_nulls(steps, delta_model)
                    else:
                        steps = np.asarray(steps)
                    main_axis_array[offset : offset + len(steps)] = steps
                else:
                    chunk_size = len(steps) + 1
                    main_axis_array[offset : offset + chunk_size] = start
                    main_axis_array[offset + 1 : offset + chunk_size] += np.cumsum(
                        steps.values
                    )
            # Direct encoding
            elif encoding == NO_COMPRESSION:
                chunk_size = len(steps)
                main_axis_array[offset : offset + chunk_size] = np.asarray(steps.values)
            elif encoding == NUMPRESS_LINEAR:
                part: np.ndarray = next(numpress_chunks_it)
                chunk_size = len(part)
                zeros = part == 0
                if zeros.sum() > 0:
                    had_nulls = True
                    if delta_model is not None:
                        part = pa.array(part, mask=zeros)
                        part = fill_nulls(part, delta_model)
                        part = np.asarray(part)
                        chunk_size = len(part)
                main_axis_array[offset : offset + chunk_size] = part
            else:
                raise ValueError(f"Unsupported chunk encoding {encoding}")

            for k, v in chunk.items():
                if k in ("spectrum_index", "chunk_encoding") or k.startswith(
                    axis_prefix
                ):
                    continue
                else:
                    arrays_of[k][offset : offset + chunk_size] = np.asarray(v.values)

            offset += chunk_size
        arrays_of[axis_prefix] = main_axis_array

        rename_map = {
            k: v["array_name"] for k, v in self.array_index.items() if k in arrays_of
        }
        for k, v in rename_map.items():
            arrays_of[v] = arrays_of.pop(k)
            if v == "intensity array" and had_nulls:
                arrays_of[v][np.isnan(arrays_of[v])] = 0

        # truncate the arrays to just the size used in case we over-allocated
        truncated = {}
        for k, v in arrays_of.items():
            truncated[k] = v[:offset]

        return truncated

    def _read_point(
        self,
        spectrum_index: int,
        rgs: list[int],
        delta_model: float | list[float] | None,
    ):
        block = self.handle.read_row_groups(rgs, columns=["point"])["point"]
        block = pc.filter(
            block, pc.equal(pc.struct_field(block, "spectrum_index"), spectrum_index)
        )

        data: pa.RecordBatch = pa.record_batch(block.combine_chunks()).drop_columns(
            "spectrum_index"
        )
        return self._clean_point_batch(data, delta_model)

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

    def _read_chunk(self, spectrum_index: int, rgs: list[int], debug: bool = False):
        chunks = []
        it = _ChunkIterator(
            self.handle.iter_batches(128, row_groups=rgs, columns=["chunk"])
        )
        it.seek(spectrum_index)
        batch: pa.RecordBatch
        for idx, batch in it:
            # batch = pc.filter(
            #     batch,
            #     pc.equal(pc.struct_field(batch, "spectrum_index"), spectrum_index),
            # )
            if idx > spectrum_index:
                break
            if len(batch) == 0:
                if chunks or idx > spectrum_index:
                    break
                else:
                    continue
            if isinstance(batch, pa.ChunkedArray):
                chunks.extend(batch.chunks)
            else:
                chunks.append(batch)
        chunks = pa.chunked_array(chunks)
        if debug:
            return chunks

        delta_model = None
        if self._do_null_filling and self._delta_model_series is not None:
            delta_model = self._delta_model_series[spectrum_index]

        return self._expand_chunks(chunks, delta_model=delta_model)

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
            batch_end = pc.max(idx_col).as_py()
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
        if self._delta_model_series is not None:
            median_delta = self._delta_model_series[spectrum_index]
        else:
            median_delta = None

        prefix = BufferFormat.Point
        rgs = self._point_index.row_groups_for_index(spectrum_index)
        if not rgs:
            rgs = self._chunk_index.row_groups_for_index(spectrum_index)
            if rgs:
                prefix = BufferFormat.Chunk
        if prefix == BufferFormat.Point:
            return self._read_point(spectrum_index, rgs, median_delta)
        elif prefix == BufferFormat.Chunk:
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

    def buffer_format(self):
        if self._point_index.init:
            return BufferFormat.Point
        elif self._chunk_index.init:
            return BufferFormat.Chunk
        else:
            raise ValueError("Could not infer buffer format")
