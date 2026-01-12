import re
import logging

from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Generic, Mapping, TypeVar

import pandas as pd
import pyarrow as pa

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

TRACE = logging.DEBUG - 5
logging.addLevelName(TRACE, "TRACE")

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


def _slice_to_range(slice_val: slice, n: int) -> range:
    start = slice_val.start or 0
    end = slice_val.stop or n
    return range(start, end)


NOT_ALLOWED_IN_COLNAME_PATTERN = re.compile("[^a-zA-Z0-9_\\\\-]+")

PERMITTED_CV_NAMES = ('MS', 'UO', )


def inflect_cv_name(accession: str, name: str) -> str:
    parts = [accession.replace(":", "_")]
    parts.append(NOT_ALLOWED_IN_COLNAME_PATTERN.sub("_", name))
    return "_".join(parts)


@dataclass(frozen=True)
class ColumnName:
    accession: str | None
    name: str
    is_unit: bool | str = field(default=False)

    def has_unit_curie(self) -> bool:
        return isinstance(self.is_unit, str)

    def is_unit_column(self) -> bool:
        return self.is_unit and isinstance(self.is_unit, bool)

    def __iter__(self):
        yield self.accession
        yield self.name
        yield self.is_unit


def parse_inflected_cv_name(name: str) -> ColumnName:
    tokens = iter(name.split("_"))
    try:
        prefix = next(tokens)
        accession = next(tokens)
        rest = '_'.join(tokens)
    except StopIteration:
        return ColumnName(None, name, False)

    if not rest or prefix not in PERMITTED_CV_NAMES:
        return ColumnName(None, name, False)

    curie = f"{prefix}:{accession}"

    if rest.endswith("_unit"):
        return ColumnName(curie, rest, True)

    if "_unit_" in rest:
        try:
            (rest, unit) = rest.rsplit("_unit_", 1)
            (unit_cv, unit_accession) = unit.split("_", 1)
            unit_curie = f"{unit_cv}:{unit_accession}"
            return ColumnName(curie, rest, unit_curie)
        except ValueError:
            pass

    return ColumnName(curie, rest, False)


class MappingProxy(object):
    """An object that proxies :meth:`__getitem__` to another object which is loaded lazily through a callable :attr:`loader`."""

    def __init__(self, loader):
        assert callable(loader)
        self.loader = loader
        self.mapping = None

    @property
    def metadata(self):
        """The metadata forwarded from the wrapped object."""
        self._ensure_mapping()
        return self.mapping.metadata

    def _ensure_mapping(self):
        if self.mapping is None:
            self.mapping = self.loader()

    def __getitem__(self, key):
        self._ensure_mapping()
        return self.mapping[key]

    def get(self, key, default=None):
        self._ensure_mapping()
        if self.mapping is None:
            raise ImportError(
                "Failed to load controlled vocabulary. "
                "Please ensure 'psims' is installed: pip install psims"
            )
        return self.mapping.get(key, default)


def _lazy_load_psims():
    try:
        from psims.controlled_vocabulary.controlled_vocabulary import load_psims
        logger.debug("Loading PSI-MS controlled vocabulary")
        cv = load_psims()
    except Exception:  # pragma: no cover
        cv = None
    return cv


def _lazy_load_uo():
    try:
        from psims.controlled_vocabulary.controlled_vocabulary import load_uo

        logger.debug("Loading UO controlled vocabulary")
        cv = load_uo()
    except Exception:  # pragma: no cover
        cv = None
    return cv


CV_PSIMS = MappingProxy(_lazy_load_psims)
CV_UO = MappingProxy(_lazy_load_uo)


class OntologyMapper:
    cv_psims: Mapping[str, Any]
    cv_uo: Mapping[str, Any]
    overrides: dict[str, str]

    def __init__(self, cv_psims=CV_PSIMS, cv_uo=CV_UO, overrides: dict[str, str]=None):
        self.cv_psims = cv_psims
        self.cv_uo = cv_uo
        self.overrides = overrides or {}

    def __getitem__(self, value: str):
        colname = parse_inflected_cv_name(value)
        accession, name, _unit = colname
        suffix = ' unit' if colname.is_unit_column() else ''
        if accession is None:
            alt_name = name.replace("_", " ").replace("mz", 'm/z')
            alt_term = self.cv_psims.get(alt_name)
            if alt_term and alt_name not in ('id', 'index', 'name', 'activation', 'precursor'):
                logger.log(TRACE, 'Mapped %r to %s|%s', name, alt_term['id'], alt_term['name'])
                return alt_term['name']
            return self.overrides.get(name, name) + suffix
        cv_id = accession.split(":")[0]
        if cv_id == "MS":
            term = self.cv_psims[accession]
            logger.log(TRACE, "Mapped %r to %s|%s", name, term["id"], term["name"])
            return term["name"] + suffix
        elif cv_id == "UO":
            term = self.cv_uo[accession]
            logger.log(TRACE, "Mapped %r to %s|%s", name, term["id"], term["name"])
            return term["name"] + suffix
        else:
            logger.warning("Unknown prefix %r from %r", cv_id, value)
            return self.overrides.get(name, name) + suffix

    def __call__(self, value: str):
        return self[value]

    def clean_column_names(self, df: pd.DataFrame):
        df.columns = df.columns.map(self)
        return df

    def clean_schema(self, table: pa.Table) -> pa.Table:
        blocks = []
        fields = []
        for f, block in zip(table.schema, table):
            chunks = []
            clean_f = None
            for chunk in block.chunks:
                node = _NameCleaningNode.from_array(f, chunk, self)
                clean_f, clean_chunk = node.clean()
                chunks.append(clean_chunk)
            fields.append(clean_f)
            blocks.append(chunks)

        chunks = []
        for block in zip(*blocks):
            chunks.append(pa.StructArray.from_arrays(block, fields=fields))
        return pa.Table.from_struct_array(
            pa.chunked_array(chunks)
        )


@dataclass
class _NameCleaningNode:
    '''
    A helper type for doing recursive Arrow schema column renaming.

    Attributes
    ----------
    field : pa.Field or None
        The field object with the name and type for this column.
    array : pa.Array
        The actual data stored in this column. This may be a pa.StructArray which will itself
        have multiple arrays under it.
    mapper : OntologyMapper
        The renaming mapping table to use to update names
    children : list of _NameCleaningNode
        The sub-arrays of this array, nested columns used to handle the recursive case
    '''
    field: pa.Field
    array: pa.Array
    mapper: OntologyMapper
    children: list["_NameCleaningNode"] = field(default_factory=list)

    def __post_init__(self):
        if self.field is not None:
            self.field = self.field.with_name(self.mapper(self.field.name))
            if self.children:
                if self.is_struct():
                    new_fields = [f.field for f in self.children]
                    self.field = self.field.with_type(pa.struct(new_fields))

    def is_struct(self) -> bool:
        if not self.field:
            return False
        return isinstance(self.field.type, pa.StructType)

    @classmethod
    def from_array(cls, field: pa.Field, array: pa.Array, mapper: OntologyMapper):
        '''The main entry point'''
        if isinstance(array.type, pa.StructType):
            return cls.from_struct_array(field, array, mapper)
        # elif isinstance(array.type, (pa.ListType, pa.LargeListType)):
        #     pass
        return cls(field, array, mapper)

    @classmethod
    def from_struct_array(
        cls, field: pa.Field, arrays: pa.StructArray, mapper: OntologyMapper
    ):
        nodes = []
        for f, a in zip(arrays.type.fields, arrays.flatten()):
            nodes.append(cls.from_array(f, a, mapper))
        return cls(field, arrays, mapper, nodes)

    def clean(self):
        if self.is_struct() or self.field is None:
            fields = []
            arrays = []
            for node in self.children:
                f, a = node.clean()
                fields.append(f)
                arrays.append(a)
            return (self.field, pa.StructArray.from_arrays(arrays, fields=fields))
        else:
            return (self.field, self.array)


__all__ = [
    "Span",
    "inflect_cv_name",
    "_slice_to_range",
    "parse_inflected_cv_name",
    "OntologyMapper",
]