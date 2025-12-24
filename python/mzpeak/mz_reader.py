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

from .util import Span, _slice_to_range
from .filters import null_delta_decode, fill_nulls

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class BufferFormat(Enum):
    """
    The different orientations data arrays may be written in.

    - ``Point`` - Each data point is stored as a row in the table as-is. Easy to do filtering random access queries over.
    - ``Chunk`` - Segments of data in a specific start-stop range are stored in an encoded block. More compact but harder to do queries over.
    - ``Secondary_Chunk`` - Paired with ``Chunk``, these points are stored in separate blocks parallel to the paired ``Chunk``.
    - ``ChunkStart`` - The start of an encoded chunk's value range.
    - ``ChunkEnd`` - The end of an encoded chunk's value range.
    - ``ChunkEncoding`` - The method used to encode the chunk's values.
    """

    Point = 1
    ChunkStart = 2
    ChunkEnd = 3
    ChunkValues = 4
    ChunkEncoding = 5
    SecondaryChunk = 6

    Chunk = ChunkValues

    @classmethod
    def from_str(cls, name: str):
        try:
            return cls[name.title().replace("_", "")]
        except KeyError:
            if name.lower() == "chunk_values":
                return cls.ChunkValues
            else:
                raise


class BufferPriority(Enum):
    Primary = 1
    Secondary = 2

    @classmethod
    def from_str(cls, name: str):
        return cls[name.title()]


@dataclass(frozen=True)
class ArrayIndexEntry:
    key: str
    array_name: str
    buffer_format: BufferFormat
    context: str
    prefix: str
    path: str
    data_type: str
    array_type: str
    data_processing_id: str | None = None
    unit: str | None = None
    transform: str | None = None
    buffer_priority: BufferPriority | None = None
    sorting_rank: int | None = None

    @classmethod
    def from_index(cls, key, fields):
        fields = fields.copy()

        fmt = fields.pop("buffer_format", None)
        if fmt:
            fields["buffer_format"] = BufferFormat.from_str(fmt)

        priority = fields.pop("buffer_priority", None)
        if priority:
            fields["buffer_priority"] = BufferPriority.from_str(priority)
        return cls(key=key, **fields)


class _DataIndex:
    meta: pq.FileMetaData
    prefix: str
    index_i: int
    init: bool
    namespace: str
    row_group_index_ranges: list[Span[int] | None]

    def __init__(self, meta: pq.FileMetaData, prefix: str, namespace: str = "spectrum"):
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

        index = json.loads(
            self.meta.metadata[f"{self.namespace}_array_index".encode("utf8")]
        )
        self.array_index = {
            entry["path"].rsplit(".")[-1]: entry for entry in index["entries"]
        }
        try:
            self.n_entries = int(
                self.meta.metadata[f"{self.namespace}_count".encode("utf8")]
            )
        except KeyError:
            self.n_entries = max_index

    def row_groups_for_index(self, query_index: int) -> list[int]:
        """
        Find the row group indices that span ``query_index``

        Parameters
        ----------
        query_index : int
            The index to select

        Returns
        -------
        list[int]:
            The row group indices that span the query index
        """
        rgs = []
        if not self.init:
            return rgs
        for i in range(self.meta.num_row_groups):
            rg = self.meta.row_group(i)
            col_idx = rg.column(self.index_i)
            if col_idx.statistics.has_min_max:
                if (
                    col_idx.statistics.min <= query_index
                    and query_index <= col_idx.statistics.max
                ):
                    rgs.append(i)
                if col_idx.statistics.min > query_index:
                    break
            else:
                break
        return rgs

    def row_groups_for_spectrum_range(
        self, query_index_range: slice | list
    ) -> list[int]:
        """
        Find the row group indices that span ``query_index_range``

        Parameters
        ----------
        query_index_range : list[int] or slice
            The indices to select

        Returns
        -------
        list[int]:
            The row group indices that span the query index range
        """
        if not self.init:
            return []
        if isinstance(query_index_range, slice):
            start = query_index_range.start or 0
            end = query_index_range.stop or self.n_entries
        else:
            start = min(query_index_range)
            end = max(query_index_range)

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
            return (
                self.meta.row_group(self.meta.num_row_groups - 1)
                .column(self.index_i)
                .statistics.max
            )
        except IndexError:
            return 0


class _BatchIterator:
    """
    Incrementally partition a stream of :class:`pyarrow.RecordBatch` instances into
    blocks of single spectrum or chromatogram data.

    This class serves two purposes:
        1. Providing a more efficient sequential scan of a data file that pre-fetches larger
           blocks of data.
        2. Granular reading for single entities if row group filters are applied, even if the
           entity spans multiple row groups.

    .. note::

        The iterator will *start* from the first index in the first row group that
        :attr:`it` spans, so care must be taken to ensure that the desired entity is
        fully contained in :attr:`it`'s spanned row groups **and** :meth:`seek` must
        be used to advance the iterator to the desired entity's index if a sequential
        scan is not desired.


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
        return (
            f"{self.__class__.__name__}({self.it}, {self.current_index}, {self.index_column}"
            f", buffered={len(self.batch) if self.batch is not None else None})"
        )

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
        if current_index is not None:
            self.seek(current_index)

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
        is_initialized = self.current_index is not None
        if is_initialized:
            logger.debug("Reading next batch looking for %r", self.current_index)
        try:
            batch = next(self.it).column(0)
        except StopIteration:
            logger.debug("Reached the end of the batch stream")
            self.batch = None
            return
        lowest_index = pc.min(pc.struct_field(batch, self.index_column)).as_py()
        if update_index and (
            (self.current_index is not None and lowest_index > self.current_index)
            or self.current_index is None
        ):
            self.current_index = lowest_index
        if is_initialized:
            logger.debug("New batch starts with %r", self.current_index)
        else:
            logger.debug("Batch iterator stream starts from %r", self.current_index)
        self.batch = batch

    def _batch_has_index(self) -> bool:
        if self.batch is None:
            return False
        mask = pc.equal(
            pc.struct_field(self.batch, self.index_column), self.current_index
        )
        return np.any(mask)

    def _batch_before_current_index(self) -> bool:
        if self.batch is None:
            return False
        index_col = pc.struct_field(self.batch, self.index_column)
        mask = pc.less(index_col, self.current_index)
        return np.all(mask)

    def _extract_for_index(self):
        if self.batch is None:
            raise StopIteration()
        mask = pc.equal(
            pc.struct_field(self.batch, self.index_column), self.current_index
        )
        indices = np.where(mask)[0]
        last_possible_row_index = len(self.batch) - 1
        if len(indices) == 0:
            logger.debug("No rows match the requested index %r from _BatchIterator", self.current_index)
            n = len(self.batch)
            chunk = self.batch.slice(0, 0)
        else:
            start = indices[0]
            n = len(indices)

            chunk = self.batch.slice(start, n)

        if n == len(self.batch) or last_possible_row_index in indices:
            # This batch is only composed of the current index or the
            # current index starts somewhere in the middle, so we should
            # also read the next batch in and take whatever rows correspond
            # to this index too before advancing.
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


# These parsed CURIEs will be removed closer to formalization
DELTA_ENCODING = {"cv_id": 1, "accession": 1003089}
NO_COMPRESSION = {"cv_id": 1, "accession": 1000576}
NUMPRESS_LINEAR = {"cv_id": 1, "accession": 1002312}

DELTA_ENCODING_CURIE = "MS:1003089"
NO_COMPRESSION_CURIE = "MS:1000576"
NUMPRESS_LINEAR_CURIE = "MS:1002312"

NUMPRESS_SLOF_CURIE = "MS:1002314"
NUMPRESS_PIC_CURIE = "MS:1002313"

NULL_INTERPOLATE = "MS:1003901"
NULL_ZERO = "MS:1003902"

psims_dtypes = {
    "MS:1000521": np.float32,
    "MS:1000523": np.float64,
    "MS:1000519": np.int32,
    "MS:1000522": np.int64,
    "MS:1001479": np.uint8,
}


_SpectrumArrays = dict[str, np.ndarray]


class _PointBatchCleaner:
    """
    Help clean :term:`Point Layout` data, formatting
    for more ease of use outside of Arrow.
    """

    namespace: str
    array_index: dict[str, dict]
    drop_index: bool
    delta_model: float | np.ndarray | dict[int, np.ndarray] | None = None
    has_mulitple_indices: bool
    has_mulitple_delta_models: bool
    index_name: str

    def __init__(
        self,
        namespace: str,
        array_index: dict[str, dict],
        drop_index: bool = True,
        delta_model=None,
        has_multiple_indices: bool = False,
    ):
        self.namespace = namespace
        self.array_index = array_index
        self.drop_index = drop_index
        self.delta_model = delta_model
        self.has_mulitple_indices = has_multiple_indices
        self.has_multiple_delta_models = isinstance(self.delta_model, dict)
        self.index_name = f"{self.namespace}_index"
        if self.has_multiple_delta_models and not self.has_mulitple_indices:
            logger.warning(
                "Cleaning a point data batch with multiple delta models %r but not multiple indices",
                self.delta_model,
            )

    def get_index_runs(self, data: pa.StructArray):
        index_values = data.field(self.index_name)
        index_runs_pa = pc.run_end_encode(index_values, run_end_type=pa.int64())
        index_runs = np.asarray(index_runs_pa.run_ends).view(np.uint64)
        index_values_unique = np.asarray(index_runs_pa.values)
        return index_runs, index_values_unique

    def null_column_indices(self, data: pa.StructArray) -> list[int]:
        nulls = []
        for i, _name in enumerate(data.type):
            col = data.field(i)
            if col.null_count == len(col):
                nulls.append(i)

        if len(nulls) > 1:
            nulls.sort()

        return nulls

    def convert_struct_array_to_dict(self, data: pa.StructArray) -> dict[str, pa.Array]:
        nulls = self.null_column_indices(data)

        fields = [f.name for f in data.type]
        data = {f: v for f, v in zip(fields, data.flatten())}

        for i in nulls:
            data.pop(fields[i])

        for k, v in {
            k: v["array_name"] for k, v in self.array_index.items() if k in data.keys()
        }.items():
            data[v] = data.pop(k)

        return data

    def fill_nulls(
        self, v: pa.Array, index_runs: np.ndarray, index_values_unique: np.ndarray
    ) -> np.ndarray:
        """
        Fill null values into a data array using the delta model.

        If :attr:`has_multiple_delta_models` is set this will require
        ``index_runs`` and ``index_values_unique`` is not :const:`None`.
        """
        if self.has_multiple_delta_models and self.has_mulitple_indices:
            if index_runs is None or index_values_unique is None:
                raise ValueError(
                    "Cannot decode nulls across multiple indices and models if index runs are None"
                )
            v_np = np.asarray(v)
            start = 0
            for run_i, run_end in enumerate(index_runs):
                index = index_values_unique[run_i]
                delta_model_for = self.delta_model[index]
                v_for = v.slice(start, run_end - start)
                v_np[start:run_end] = fill_nulls(v_for, delta_model_for)
                start = run_end
            v = v_np

        elif not np.any(np.isnan(self.delta_model)):
            v = fill_nulls(v, self.delta_model)
        return v

    def expand(self, data: pa.RecordBatch | pa.StructArray) -> _SpectrumArrays:
        """
        This internal helper takes a :class:`pyarrow.RecordBatch` or
        :class:`pyarrow.StructArray` in the :term:`Point Layout`,
        cleans it up and converts it into a :class:`dict` mapping
        :class:`str` keys to :class:`numpy.ndarray` instances.

        Parameters
        ----------
        data : :class:`pyarrow.RecordBatch` or :class:`pyarrow.StructArray`
            The raw Arrow data read from the Parquet file to be transformed.

        Returns
        -------
        dict[str, np.ndarray]:
            The arrays of the entity in ``data``
        """
        if isinstance(data, pa.RecordBatch):
            data = data.to_struct_array()

        index_runs = index_values_unique = None

        if self.has_mulitple_indices and self.has_multiple_delta_models:
            (index_runs, index_values_unique) = self.get_index_runs(data)

        data = self.convert_struct_array_to_dict(data)
        it = data.items()

        result = {}
        for k, v in it:
            if v.null_count and self.delta_model is not None:
                if k == "m/z array":
                    v = self.fill_nulls(
                        v,
                        index_runs=index_runs,
                        index_values_unique=index_values_unique,
                    )

                elif k == "intensity array":
                    v = np.asarray(v)
                    v[np.isnan(v)] = 0.0

            result[k] = np.asarray(v)

        if self.drop_index and self.index_name in result:
            result.pop(self.index_name)
        return result


class _ChunkBatchCleaner:
    namespace: str
    array_index: dict[str, dict]
    drop_index: bool
    delta_model: float | np.ndarray | dict[int, np.ndarray] | None = None
    has_mulitple_indices: bool
    has_mulitple_delta_models: bool
    index_name: str
    time_label: str
    axis_prefix: str

    def __init__(
        self,
        namespace: str,
        array_index: dict[str, dict],
        drop_index: bool = True,
        delta_model=None,
    ):
        self.namespace = namespace
        self.array_index = array_index
        self.drop_index = drop_index
        self.delta_model = delta_model
        self.has_multiple_delta_models = isinstance(self.delta_model, dict)
        self.index_name = f"{self.namespace}_index"
        self.time_label = f"{self.namespace}_time"
        self.axis_prefix = self.find_axis_prefix()
        self.has_transforms = self.arrays_has_transforms()
        if self.has_multiple_delta_models:
            logger.warning(
                "Cleaning a point data batch with multiple delta models %r but not multiple indices",
                self.delta_model,
            )

    def arrays_has_transforms(self):
        has_transforms = {
            k: v.get("transform")
            for k, v in self.array_index.items()
            if v.get("transform")
        }
        return has_transforms

    def find_axis_prefix(self):
        axis_prefix = None
        for k, v in self.array_index.items():
            name = ArrayIndexEntry.from_index(k, v)
            if (
                name.buffer_format == BufferFormat.ChunkValues
                or name.buffer_format == BufferFormat.Chunk
            ):
                axis_prefix = k.removesuffix("_chunk_values")
        if axis_prefix is None:
            for k, v in self.array_index.items():
                if k.endswith("_chunk_values"):
                    axis_prefix = k.removesuffix("_chunk_values")
                    break
        if axis_prefix is None:
            raise ValueError(f"Could not infer axis prefix from {self.array_index}")
        return axis_prefix

    def prescan_chunks(self, chunks: list[dict[str, Any]]):
        n = 0
        numpress_chunks = []
        for chunk in chunks:
            # The +1 is to account for the starting point
            encoding = chunk["chunk_encoding"].as_py()
            if encoding in (
                DELTA_ENCODING,
                NO_COMPRESSION,
                NO_COMPRESSION_CURIE,
                DELTA_ENCODING_CURIE,
            ):
                n += len(chunk[f"{self.axis_prefix}_chunk_values"]) + 1
            elif encoding in (NUMPRESS_LINEAR, NUMPRESS_LINEAR_CURIE):
                raw = chunk[f"{self.axis_prefix}_numpress_bytes"].as_py()
                part = pynumpress.decode_linear(raw)
                n += len(part)
                numpress_chunks.append(part)
            else:
                raise ValueError(f"Unsupported chunk encoding {encoding}")
        return n, numpress_chunks

    def initialize_arrays(
        self, n: int, chunks: list[dict[str, Any]]
    ) -> dict[str, np.ndarray]:
        arrays_of = {}
        for k, v in chunks[0].items():
            if k == self.index_name:
                if not self.drop_index:
                    arrays_of[k] = np.zeros(n, dtype=np.uint64)
            elif (
                k == "chunk_encoding"
                or k == self.time_label
                or k.startswith(self.axis_prefix)
            ):
                continue
            else:
                arrays_of[k] = np.zeros(
                    n, dtype=psims_dtypes[self.array_index[k]["data_type"]]
                )
        return arrays_of

    def process_chunks(
        self,
        n: int,
        numpress_chunks: list,
        chunks: list[dict[str, pa.Array]],
        arrays_of: dict[str, np.ndarray],
    ) -> tuple[int, bool]:
        main_axis_array = np.zeros(n)
        offset = 0
        had_nulls = False
        numpress_chunks_it = iter(numpress_chunks)
        skip = set()
        for _i, chunk in enumerate(chunks):
            start = chunk[f"{self.axis_prefix}_chunk_start"].as_py()
            end = chunk[f"{self.axis_prefix}_chunk_end"].as_py()

            steps = chunk[f"{self.axis_prefix}_chunk_values"]
            encoding = chunk["chunk_encoding"].as_py()
            index_val = chunk[self.index_name].as_py()

            delta_model_ = self.delta_model
            if self.has_multiple_delta_models:
                delta_model_ = self.delta_model[index_val]

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
                    if delta_model_ is not None:
                        steps = fill_nulls(steps, delta_model_)
                    else:
                        logger.warning(
                            "Null values detected in chunk %0.3f-%0.3f for %s:%r",
                            start,
                            end,
                            self.namespace,
                            index_val,
                        )
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
                steps = pa.chunked_array([pa.scalar(start, steps.type), steps.values])
                chunk_size = len(steps)
                if steps.null_count > 0:
                    if delta_model_ is not None:
                        steps = fill_nulls(steps, delta_model_)
                    else:
                        logger.warning(
                            "Null values detected in chunk %0.3f-%0.3f for %s:%r",
                            start,
                            end,
                            self.namespace,
                            index_val,
                        )
                    main_axis_array[offset : offset + chunk_size] = np.asarray(
                        steps.values
                    )
                else:
                    main_axis_array[offset : offset + chunk_size] = np.asarray(steps.values)
            elif encoding in (NUMPRESS_LINEAR, NUMPRESS_LINEAR_CURIE):
                part: np.ndarray = next(numpress_chunks_it)
                chunk_size = len(part)
                zeros = part == 0
                if zeros.sum() > 0:
                    had_nulls = True
                    if delta_model_ is not None:
                        part = pa.array(part, mask=zeros)
                        part = fill_nulls(part, delta_model_)
                        part = np.asarray(part)
                        chunk_size = len(part)
                main_axis_array[offset : offset + chunk_size] = part
            else:
                raise ValueError(f"Unsupported chunk encoding {encoding}")

            for k, v in chunk.items():
                if k == self.index_name:
                    if not self.drop_index:
                        arrays_of[k][offset : offset + chunk_size] = index_val
                elif k in ("chunk_encoding", self.time_label) or k.startswith(
                    self.axis_prefix
                ) or k in skip:
                    continue
                else:
                    if v.values is not None:
                        values = np.asarray(v.values)
                        if k in self.has_transforms:
                            if self.has_transforms[k] == NUMPRESS_SLOF_CURIE:
                                values = pynumpress.decode_slof(values)
                            elif self.has_transforms[k] == NUMPRESS_PIC_CURIE:
                                values = pynumpress.decode_pic(values)
                            elif self.has_transforms[k] in (NULL_INTERPOLATE, NULL_ZERO):
                                # These transforms do not require any special handling
                                pass
                            else:
                                raise NotImplementedError(self.has_transforms[k])
                        arrays_of[k][offset : offset + chunk_size] = values
                    else:
                        arrays_of.pop(k)
                        skip.add(k)

            offset += chunk_size
        arrays_of[self.axis_prefix] = main_axis_array
        return offset, had_nulls

    def expand(self, chunks: list[dict[str, Any]]):
        axis_prefix = self.axis_prefix

        n, numpress_chunks = self.prescan_chunks(chunks)
        if n == 0:
            return {axis_prefix: np.array([])}

        arrays_of = self.initialize_arrays(n, chunks)
        (offset, had_nulls) = self.process_chunks(
            n, numpress_chunks=numpress_chunks, chunks=chunks, arrays_of=arrays_of
        )

        rename_map = {
            k: v["array_name"] for k, v in self.array_index.items() if k in arrays_of
        }

        if f"{axis_prefix}_chunk_values" in self.array_index:
            rename_map[rename_map.pop(axis_prefix, axis_prefix)] = self.array_index[
                f"{axis_prefix}_chunk_values"
            ]["array_name"]

        elif axis_prefix in self.array_index:
            rename_map[axis_prefix] = self.array_index[axis_prefix]["array_name"]

        for k, v in rename_map.items():
            arrays_of[v] = arrays_of.pop(k)
            if v == "intensity array" and had_nulls:
                arrays_of[v][np.isnan(arrays_of[v])] = 0

        # truncate the arrays to just the size used in case we over-allocated
        truncated = {}
        for k, v in arrays_of.items():
            truncated[k] = v[:offset]

        return truncated


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

    def __init__(self, handle: pq.ParquetFile, namespace: str | None = None):
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
        index = json.loads(
            self.meta.metadata[f"{self._namespace}_array_index".encode("utf8")]
        )
        self.array_index = {
            entry["path"].rsplit(".")[-1]: entry for entry in index["entries"]
        }
        self.n_entries = max(self._point_index.n_entries, self._chunk_index.n_entries)

    def _clean_point_batch(
        self,
        data: pa.RecordBatch | pa.StructArray,
        delta_model: float | np.ndarray | dict[int, np.ndarray] | None = None,
        drop_index: bool = True,
        has_mulitple_indices: bool = False,
    ) -> _SpectrumArrays:
        """
        This internal helper takes a :class:`pyarrow.RecordBatch` or
        :class:`pyarrow.StructArray` in the :term:`Point Layout`,
        cleans it up and converts it into a :class:`dict` mapping
        :class:`str` keys to :class:`numpy.ndarray` instances.

        Parameters
        ----------
        data : :class:`pyarrow.RecordBatch` or :class:`pyarrow.StructArray`
            The raw Arrow data read from the Parquet file to be transformed.
        delta_model : :class:`float` or :class:`numpy.ndarray`, optional
            The parameters of a null-filling model if available to be used
            to fill in any null value gaps in the m/z axis.
        drop_index : :class:`bool`
            Whether or not to remove the index column from the output. Defaults to :const:`True`.
        has_multiple_indices : :class:`bool`
            Whether or not to run the more complex algorithm to process the data in
            batches in order to fill null values correctly across multiple entities
            in the same batch. Defaults to :const:`False`

        Returns
        -------
        dict[str, np.ndarray]:
            The arrays of the entity in ``data``
        """

        cleaner = _PointBatchCleaner(
            self._namespace,
            self.array_index,
            drop_index,
            delta_model=delta_model if self._do_null_filling else None,
            has_multiple_indices=has_mulitple_indices,
        )
        return cleaner.expand(data)

    def read_data_for_range(self, index_range: slice | list[int]) -> _SpectrumArrays:
        """
        Perform a read that slices along the primary index of the series,
        loading data for multiple entities.

        Parameters
        ----------
        index_range : slice[int] or list[int]
            The indices or index range to load data for. If this is a slice,
            the ``step`` value is ignored.

        Returns
        -------
        dict[str, np.ndarray]:
            The arrays of the entities selected, with an index array denoting which entity
            all points at that index came from.
        """
        prefix = BufferFormat.Point
        rgs = self._point_index.row_groups_for_spectrum_range(index_range)
        if not rgs:
            rgs = self._chunk_index.row_groups_for_spectrum_range(index_range)
            if rgs:
                prefix = BufferFormat.ChunkValues

        is_slice = False
        if isinstance(index_range, slice):
            start = index_range.start or 0
            end = index_range.stop or self.n_entries
            is_slice = True
        else:
            start = min(index_range)
            end = max(index_range)

        if prefix == BufferFormat.Point:
            return self._read_point_range(start, end, index_range, is_slice, rgs)
        elif prefix == BufferFormat.ChunkValues:
            return self._read_chunk_range(start, end, index_range, is_slice, rgs)
        else:
            raise NotImplementedError(prefix)

    def _expand_chunks(
        self,
        chunks: list[dict[str, Any]],
        axis_prefix: str | None = None,
        delta_model: float | np.ndarray | dict[int, np.ndarray] | None = None,
        preserve_index: bool = False,
    ) -> _SpectrumArrays:
        cleaner = _ChunkBatchCleaner(
            self._namespace,
            array_index=self.array_index,
            drop_index=not preserve_index,
            delta_model=delta_model,
        )
        return cleaner.expand(chunks)

    def _read_point(
        self,
        spectrum_index: int,
        rgs: list[int],
        delta_model: float | list[float] | None,
    ) -> _SpectrumArrays:
        block = self.handle.read_row_groups(rgs, columns=["point"])["point"]
        index_col = f"{self._namespace}_index"
        block = pc.filter(
            block,
            pc.equal(pc.struct_field(block, index_col), spectrum_index),
        )

        data: pa.RecordBatch = pa.record_batch(block.combine_chunks()).drop_columns(
            index_col
        )
        time_label = f"{self._namespace}_time"
        if time_label in data.column_names:
            data = data.drop_columns(time_label)
        return self._clean_point_batch(
            data, delta_model, drop_index=True, has_mulitple_indices=False
        )

    def _read_point_range(
        self,
        start: int,
        end: int,
        index_range: list[int] | slice,
        is_slice: bool,
        rgs: list[int],
    ) -> _SpectrumArrays:
        index_name = f"{self._namespace}_index"
        block = self.handle.read_row_groups(rgs, columns=["point"])["point"]
        idx_col = pc.struct_field(block, index_name)
        delta_models = None
        if not is_slice:
            if self._delta_model_series is not None and self._do_null_filling:
                delta_models = {i: self._delta_model_series[i] for i in index_range}
            index_range = pa.array(index_range)
        else:
            if self._delta_model_series is not None and self._do_null_filling:
                delta_models = {
                    i: self._delta_model_series[i]
                    for i in _slice_to_range(index_range, len(self))
                }
        if is_slice:
            mask = pc.and_(
                pc.less_equal(idx_col, end), pc.greater_equal(idx_col, start)
            )
        else:
            mask = pc.is_in(idx_col, pa.array(index_range))
        block = pc.filter(block, mask)
        data: pa.RecordBatch = pa.record_batch(block.combine_chunks())
        data = self._clean_point_batch(
            data, delta_model=delta_models, has_mulitple_indices=True, drop_index=False
        )
        return data

    def _read_chunk(
        self, index: int, rgs: list[int], debug: bool = False, batch_size: int = 512
    ) -> _SpectrumArrays:
        """
        Implementation to read a single chunk layout entry by index.

        Parameters
        ----------
        index : int
            The entity index to be read
        rgs : list[int]
            The row group indices which contain ``index``
        debug : bool, optional
            This internal option will skip unpacking and expanding the
            :class:`pyarrow.ChunkedArray` into NumPy arrays.
        batch_size : int, optional
            The number of rows to buffer in memory at any given time. This
            trades performance for memory consumption when scanning over many
            rows. For single index reads the default value is safe.

        Returns
        -------
        :class:`_SpectrumArrays`
            The unpacked mapping of :class:`np.ndarray`
        """
        chunks = []
        it = _BatchIterator(
            self.handle.iter_batches(batch_size, row_groups=rgs, columns=["chunk"]),
            current_index=index,
            index_column=f"{self._namespace}_index"
        )
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
        batch_size: int = 128,
    ) -> _SpectrumArrays:
        chunks = []
        delta_models = None
        if not is_slice:
            if self._delta_model_series is not None:
                delta_models = {i: self._delta_model_series[i] for i in index_range}
            index_range = pa.array(index_range)
        else:
            if self._delta_model_series is not None:
                delta_models = {
                    i: self._delta_model_series[i]
                    for i in _slice_to_range(index_range, len(self))
                }

        for batch in self.handle.iter_batches(
            batch_size, row_groups=rgs, columns=["chunk"]
        ):
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
        return self._expand_chunks(
            chunks, preserve_index=True, delta_model=delta_models
        )

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
                prefix = BufferFormat.ChunkValues
        if prefix == BufferFormat.Point:
            return self._read_point(index, rgs, median_delta)
        elif prefix == BufferFormat.ChunkValues:
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
        :attr:`BufferFormat.Point` or :attr:`BufferFormat.ChunkValues`
        """
        if self._point_index.init:
            return BufferFormat.Point
        elif self._chunk_index.init:
            return BufferFormat.ChunkValues
        else:
            raise ValueError("Could not infer buffer format")


MzPeakSpectrumDataReader = MzPeakArrayDataReader
