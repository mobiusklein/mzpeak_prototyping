from dataclasses import dataclass
import logging
import json

from typing import Any, Iterator, Sequence
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

@dataclass(frozen=True)
class BufferName:
    key: str
    array_name: str
    buffer_format: "BufferFormat"
    context: str
    prefix: str
    path: str
    data_type: str
    array_type: str
    unit: str | None = None
    transform: str | None = None

    @classmethod
    def from_index(cls, key, fields):
        fields = fields.copy()
        fmt = fields.pop('buffer_format', None)
        if fmt:
            fields['buffer_format'] = BufferFormat.from_str(fmt.title())
        return cls(key=key, **fields)


class _DataIndex:
    meta: pq.FileMetaData
    prefix: str
    index_i: int
    init: bool
    namespace: str
    row_group_index_ranges: list[Span[int] | None]

    def __init__(self, meta: pq.FileMetaData, prefix: str, namespace: str="spectrum"):
        self.meta = meta
        self.prefix = prefix
        self.index_i = 0
        self.init = False
        self.row_group_index_ranges = []
        self.namespace = namespace
        self._infer_schema_idx()

    def _infer_schema_idx(self):
        rg = self.meta.row_group(0)
        q = f"{self.prefix}.{self.namespace}_index"
        self.row_group_index_ranges = []
        max_index = 0
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
                    max_index = max(max_index, col_idx.statistics.max)
                else:
                    self.row_group_index_ranges.append(None)

        index = json.loads(self.meta.metadata[f"{self.namespace}_array_index".encode('utf8')])
        self.array_index = {
            entry["path"].rsplit(".")[-1]: entry for entry in index["entries"]
        }
        try:
            self.n_entries = int(self.meta.metadata[f"{self.namespace}_count".encode('utf8')])
        except KeyError:
            self.n_entries = max_index

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
            end = spectrum_index_range.stop or self.n_entries
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

    def __len__(self):
        try:
            return self.meta.row_group(
                self.meta.num_row_groups - 1
            ).column(self.index_i).statistics.max
        except IndexError:
            return 0


class BufferFormat(Enum):
    """
    The different orientations data arrays may be written in.

    - ``Point`` - Each data point is stored as a row in the table as-is. Easy to do filtering random access queries over.
    - ``Chunk`` - Segments of data in a specific start-stop range are stored in an encoded block. More compact but harder to do queries over.
    - ``Secondary_Chunk`` - Paired with ``Chunk``, these points are stored in separate blocks parallel to the paired ``Chunk``.
    """
    Point = 1
    Chunk = 2
    Secondary_Chunk = 3

    @classmethod
    def from_str(cls, name: str):
        try:
            return cls[name]
        except KeyError:
            if name.lower() == 'chunk_values':
                return cls.Chunk
            else:
                raise


class _BatchIterator:
    """
    Incrementally partition a stream of :class:`pyarrow.RecordBatch` instances into
    blocks of single spectrum or chromatogram data.

    Attributes
    ----------
    it : :class:`Iterator` of :class:`pyarrow.RecordBatch`
        The stream of Arrow batches to unpack from.
    batch : :class:`pyarrow.StructArray`
        The current batch being segmented.
    current_index : int
        The entry index that was last processed.
    index_column : str
        The name of entry index column.
    """
    it: Iterator[pa.RecordBatch]
    batch: pa.StructArray
    current_index: int
    index_column: str

    def __repr__(self):
        return f"{self.__class__.__name__}({self.it}, {self.current_index}, {self.index_column}" \
               f", buffered={len(self.batch) if self.batch is not None else None})"

    def __init__(
        self,
        it: Iterator[pa.StructArray],
        current_index: int = None,
        index_column: str = "spectrum_index",
    ):
        self.it = it
        self.batch = None
        self.current_index = current_index
        self.index_column = index_column
        self._read_next_chunk(update_index=True)

    def _infer_starting_index(self):
        return pc.min(pc.struct_field(self.batch, self.index_column)).as_py()

    def __next__(self):
        batch = self._extract_for_index()
        i = self.current_index
        self.current_index += 1
        return i, batch

    def __iter__(self):
        return self

    def _read_next_chunk(self, update_index: bool):
        logger.debug("Reading next batch looking for %r", self.current_index)
        batch = next(self.it).column(0)
        lowest_index = pc.min(pc.struct_field(batch, self.index_column)).as_py()
        if update_index and (
            (self.current_index is not None and lowest_index > self.current_index)
            or self.current_index is None
        ):
            self.current_index = lowest_index
        logger.debug("New batch starts with %r", self.current_index)
        self.batch = batch

    def _batch_has_index(self) -> bool:
        mask = pc.equal(
            pc.struct_field(self.batch, self.index_column), self.current_index
        )
        return np.any(mask)

    def _extract_for_index(self):
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
            # This batch is only composed of the current index, so we should also read the next batch in and
            # take whatever rows correspond to this index too before advancing.
            self._read_next_chunk(update_index=False)
            if self._batch_has_index():
                rest = self._extract_for_index()
                chunk = pa.concat_arrays([chunk, rest])
        else:
            self.batch = self.batch.slice(n)
        return chunk

    def seek(self, index: int):
        """
        Advance the iterator until it reaches the requested index.

        .. note:: The iterator cannot go backwards.

        Arguments
        ---------
        index : int
            The index to seek to
        """
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

DELTA_ENCODING_CURIE = "MS:1003089"
NO_COMPRESSION_CURIE = "MS:1000576"
NUMPRESS_LINEAR_CURIE = "MS:1002312"

NUMPRESS_SLOF_CURIE = "MS:1002314"
NUMPRESS_PIC_CURIE = "MS:1002313"

psims_dtypes = {
    "MS:1000521": np.float32,
    "MS:1000523": np.float64,
    "MS:1000519": np.int32,
    "MS:1000522": np.int64,
    "MS:1001479": np.uint8,
}


_SpectrumArrays = dict[str, np.ndarray]


class MzPeakArrayDataReader(Sequence[_SpectrumArrays]):
    """
    A generic reader for mzPeak data array reading.

    Abstracts reading either point or chunk formats, and
    provides index-based slicing over spectra.

    Attributes
    ----------
    handle : :class:`pyarrow.parquet.ParquetFile`
        The underlying Parquet file reader
    meta : :class:`pyarrow.parquet.FileMetaData`
        The metadata segment of the underlying Parquet file
    array_index : dict[str, dict]
        Descriptions of the different arrays stored in the data file
    """
    handle: pq.ParquetFile
    meta: pq.FileMetaData
    _point_index: _DataIndex
    _chunk_index: _DataIndex
    array_index: dict[str, dict]
    n_entries: int
    _delta_model_series: np.ndarray | None
    _do_null_filling: bool = True
    _namespace: str

    def __init__(self, handle: pq.ParquetFile, namespace: str | None=None):
        if not isinstance(handle, pq.ParquetFile):
            handle = pq.ParquetFile(handle)
        self.handle = handle
        self.meta = self.handle.metadata
        self._namespace = namespace
        if namespace is None:
            self._infer_namespace()
        self._point_index = _DataIndex(self.meta, "point", namespace=self._namespace)
        self._chunk_index = _DataIndex(self.meta, "chunk", namespace=self._namespace)
        self._infer_schema_idx()
        self._delta_model_series = None

    def _infer_namespace(self):
        if b"chromatogram_array_index" in self.meta.metadata:
            self._namespace = "chromatogram"
        elif b"spectrum_array_index" in self.meta.metadata:
            self._namespace = "spectrum"
        else:
            logger.warning("Defaulting namespace for %r to 'spectrum'", self)
            self._namespace = "spectrum"

    def _infer_schema_idx(self):
        index = json.loads(self.meta.metadata[f"{self._namespace}_array_index".encode('utf8')])
        self.array_index = {
            entry["path"].rsplit(".")[-1]: entry for entry in index["entries"]
        }
        self.n_entries = max(self._point_index.n_entries, self._chunk_index.n_entries)

    def _clean_point_batch(
        self,
        data: pa.RecordBatch | pa.StructArray,
        delta_model: float | np.ndarray | None = None,
    ) -> _SpectrumArrays:
        if isinstance(data, pa.StructArray):
            nulls = []
            for i, name in enumerate(data.type):
                col = data.field(i)
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

    def read_data_for_range(self, index_range: slice | list[int]) -> _SpectrumArrays:
        prefix = BufferFormat.Point
        rgs = self._point_index.row_groups_for_spectrum_range(index_range)
        if not rgs:
            rgs = self._chunk_index.row_groups_for_spectrum_range(index_range)
            if rgs:
                prefix = BufferFormat.Chunk

        is_slice = False
        if isinstance(index_range, slice):
            start = index_range.start or 0
            end = index_range.stop or self.n_entries
            is_slice = True
        else:
            start = min(index_range)
            end = max(index_range)

        if prefix == BufferFormat.Point:
            return self._read_point_range(
                start, end, index_range, is_slice, rgs
            )
        elif prefix == BufferFormat.Chunk:
            return self._read_chunk_range(
                start, end, index_range, is_slice, rgs
            )
        else:
            raise NotImplementedError(prefix)

    def _expand_chunks(
        self,
        chunks: list[dict[str, Any]],
        axis_prefix: str | None = None,
        delta_model: float | np.ndarray | None = None,
    ) -> _SpectrumArrays:
        if axis_prefix is None:
            if self._namespace == "spectrum":
                axis_prefix = f"{self._namespace}_mz"
            elif self._namespace == "chromatogram":
                axis_prefix = f"{self._namespace}_time"
            else:
                raise ValueError(self._namespace)

        time_label = f"{self._namespace}_time"


        has_transforms = {k: v.get('transform') for k, v in self.array_index.items() if v.get('transform')}
        for k, v in self.array_index.items():
            if k.startswith(axis_prefix):
                axis_prefix = k.removesuffix("_chunk_values")

        n = 0
        numpress_chunks = []
        for chunk in chunks:
            # The +1 is to account for the starting point
            encoding = chunk["chunk_encoding"].as_py()
            if encoding in (DELTA_ENCODING, NO_COMPRESSION, NO_COMPRESSION_CURIE, DELTA_ENCODING_CURIE):
                n += len(chunk[f"{axis_prefix}_chunk_values"]) + 1
            elif encoding in (NUMPRESS_LINEAR, NUMPRESS_LINEAR_CURIE):
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
                k == f"{self._namespace}_index"
                or k == "chunk_encoding"
                or k == time_label
                or k.startswith(axis_prefix)
            ):
                continue
            else:
                arrays_of[k] = np.zeros(
                    n, dtype=psims_dtypes[self.array_index[k]["data_type"]]
                )

        main_axis_array = np.zeros(n)
        offset = 0
        had_nulls = False
        numpress_chunks_it = iter(numpress_chunks)
        for i, chunk in enumerate(chunks):
            start = chunk[f"{axis_prefix}_chunk_start"].as_py()
            end = chunk[f"{axis_prefix}_chunk_end"].as_py()

            steps = chunk[f"{axis_prefix}_chunk_values"]
            encoding = chunk["chunk_encoding"].as_py()

            # Delta encoding
            if encoding in (DELTA_ENCODING, DELTA_ENCODING_CURIE):
                # This indicates an empty chunk containing no information beyond possibly a null value.
                # Skip it and keep going.
                if (start == end) and start == 0.0:
                    continue

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
            elif encoding in (NO_COMPRESSION, NO_COMPRESSION_CURIE):
                chunk_size = len(steps)
                main_axis_array[offset : offset + chunk_size] = np.asarray(steps.values)
            elif encoding in (NUMPRESS_LINEAR, NUMPRESS_LINEAR_CURIE):
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
                if k in (f"{self._namespace}_index", "chunk_encoding", time_label) or k.startswith(
                    axis_prefix
                ):
                    continue
                else:
                    values = np.asarray(v.values)
                    if k in has_transforms:
                        if has_transforms[k] == NUMPRESS_SLOF_CURIE:
                            values = pynumpress.decode_slof(values)
                        elif has_transforms[k] == NUMPRESS_PIC_CURIE:
                            values = pynumpress.decode_pic(values)
                        else:
                            raise NotImplementedError(has_transforms[k])
                    arrays_of[k][offset : offset + chunk_size] = values

            offset += chunk_size
        arrays_of[axis_prefix] = main_axis_array

        rename_map = {
            k: v["array_name"] for k, v in self.array_index.items() if k in arrays_of
        }

        if f"{axis_prefix}_chunk_values" in self.array_index:
            rename_map[axis_prefix] = self.array_index[
                f"{axis_prefix}_chunk_values"
            ]['array_name']
        elif axis_prefix in self.array_index:
            rename_map[axis_prefix] = self.array_index[axis_prefix][
                "array_name"
            ]

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
    ) -> _SpectrumArrays:
        block = self.handle.read_row_groups(rgs, columns=["point"])["point"]
        block = pc.filter(
            block,
            pc.equal(
                pc.struct_field(block, f"{self._namespace}_index"), spectrum_index
            ),
        )

        data: pa.RecordBatch = pa.record_batch(block.combine_chunks()).drop_columns(
            f"{self._namespace}_index"
        )
        time_label = f"{self._namespace}_time"
        if time_label in data.column_names:
            data = data.drop_columns(time_label)
        return self._clean_point_batch(data, delta_model)

    def _read_point_range(
        self,
        start: int,
        end: int,
        spectrum_index_range: list[int] | slice,
        is_slice: bool,
        rgs: list[int],
    ) -> _SpectrumArrays:
        block = self.handle.read_row_groups(rgs, columns=["point"])["point"]
        idx_col = pc.struct_field(block, f"{self._namespace}_index")

        if is_slice:
            mask = pc.and_(
                pc.less_equal(idx_col, end), pc.greater_equal(idx_col, start)
            )
        else:
            mask = pc.is_in(idx_col, pa.array(spectrum_index_range))
        block = pc.filter(block, mask)
        data: pa.RecordBatch = pa.record_batch(block.combine_chunks()).drop_columns(
            f"{self._namespace}_index"
        )
        return self._clean_point_batch(data)

    def _read_chunk(
        self, index: int, rgs: list[int], debug: bool = False, buffer_size: int = 512
    ) -> _SpectrumArrays:
        chunks = []
        it = _BatchIterator(
            self.handle.iter_batches(buffer_size, row_groups=rgs, columns=["chunk"])
        )
        it.seek(index)
        batch: pa.RecordBatch
        for idx, batch in it:
            if idx > index:
                break
            if len(batch) == 0:
                if chunks or idx > index:
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
            delta_model = self._delta_model_series[index]

        return self._expand_chunks(chunks, delta_model=delta_model)

    def _read_chunk_range(
        self,
        start: int,
        end: int,
        index_range: list[int] | slice,
        is_slice: bool,
        rgs: list[int],
    ) -> _SpectrumArrays:
        chunks = []
        if not is_slice:
            index_range = pa.array(index_range)
        for batch in self.handle.iter_batches(128, row_groups=rgs, columns=["chunk"]):
            batch = batch["chunk"]
            idx_col = pc.struct_field(batch, f"{self._namespace}_index")
            batch_end = pc.max(idx_col).as_py()
            if batch_end < start:
                continue
            if is_slice:
                mask = pc.and_(
                    pc.less_equal(idx_col, end), pc.greater_equal(idx_col, start)
                )
            else:
                mask = pc.is_in(idx_col, index_range)
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

    def _read_data_for(self, index: int) -> _SpectrumArrays:
        if self._delta_model_series is not None:
            median_delta = self._delta_model_series[index]
        else:
            median_delta = None

        prefix = BufferFormat.Point
        rgs = self._point_index.row_groups_for_index(index)
        if not rgs:
            rgs = self._chunk_index.row_groups_for_index(index)
            if rgs:
                prefix = BufferFormat.Chunk
        if prefix == BufferFormat.Point:
            return self._read_point(index, rgs, median_delta)
        elif prefix == BufferFormat.Chunk:
            return self._read_chunk(
                index,
                rgs,
            )
        else:
            raise NotImplementedError(prefix)

    def __getitem__(self, index: int | slice) -> _SpectrumArrays:
        if isinstance(index, slice):
            return self.read_data_for_range(index)
        return self._read_data_for(index)

    def __len__(self):
        if self._point_index.init:
            return len(self._point_index)
        elif self._chunk_index.init:
            return len(self._chunk_index)
        else:
            return 0

    def buffer_format(self) -> BufferFormat:
        """
        The kind of data layout that the underlying file uses, either
        :attr:`BufferFormat.Point` or :attr:`BufferFormat.Chunk`
        """
        if self._point_index.init:
            return BufferFormat.Point
        elif self._chunk_index.init:
            return BufferFormat.Chunk
        else:
            raise ValueError("Could not infer buffer format")


MzPeakSpectrumDataReader = MzPeakArrayDataReader