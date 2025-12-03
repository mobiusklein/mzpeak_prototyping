import re
import logging

from dataclasses import dataclass, field
from numbers import Number
from typing import Any, Generic, Mapping, TypeVar

import pandas as pd

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

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
            if alt_term and alt_name not in ('id', 'index', 'name'):
                return alt_term['name']
            return self.overrides.get(name, name) + suffix
        cv_id = accession.split(":")[0]
        if cv_id == "MS":
            term = self.cv_psims[accession]
            return term["name"] + suffix
        elif cv_id == "UO":
            term = self.cv_uo[accession]
            return term["name"] + suffix
        else:
            logger.warning("Unknown prefix %r from %r", cv_id, value)
            return self.overrides.get(name, name) + suffix

    def __call__(self, value: str):
        return self[value]

    def clean_column_names(self, df: pd.DataFrame):
        df.columns = df.columns.map(self)
        return df


__all__ = [
    "Span",
    "inflect_cv_name",
    "_slice_to_range",
    "parse_inflected_cv_name",
    "OntologyMapper",
]