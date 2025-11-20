from typing import ClassVar
from dataclasses import dataclass, field
from enum import StrEnum
from collections.abc import MutableSequence

OTHER = "other"

SPECTRUM = "spectrum"
CHROMATOGRAM = "chromatogram"

DATA_ARRAYS = "data arrays"
METADATA = "metadata"
PEAKS = "peaks"
PROPRIETARY = "proprietary"


class EntityType(StrEnum):
    Spectrum = SPECTRUM
    Chromatogram = CHROMATOGRAM
    Other = OTHER

    @classmethod
    def get(cls, value: str):
        try:
            return cls(value)
        except ValueError:
            return cls.Other


class DataKind(StrEnum):
    DataArrays = DATA_ARRAYS
    Peaks = PEAKS
    Metadata = METADATA
    Other = OTHER
    Proprietary = PROPRIETARY

    @classmethod
    def get(cls, value: str):
        try:
            return cls(value)
        except ValueError:
            return cls.Other


@dataclass
class FileEntry:
    name: str
    entity_type: str
    data_kind: str

    def as_data_kind(self) -> DataKind:
        return DataKind.get(self.data_kind)

    def as_entity_type(self) -> EntityType:
        return EntityType.get(self.entity_type)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "data_kind": self.data_kind,
        }

    @classmethod
    def from_json(cls, data: dict) -> 'FileEntry':
        return cls(
            data['name'],
            data['entity_type'],
            data['data_kind']
        )

    def entry_type(self) -> tuple[EntityType, DataKind]:
        return (self.as_entity_type(), self.as_data_kind())


@dataclass
class FileIndex(MutableSequence[FileEntry]):
    FILE_NAME: ClassVar[str] = "mzpeak_index.json"

    files: list[FileEntry] = field(default_factory=list)
    metadata: dict[str, int | float | list | dict] = field(default_factory=dict)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i: int):
        return self.files[i]

    def __setitem__(self, i: int, value: FileEntry):
        self.files[i] = value

    def __delitem__(self, i: int):
        del self.files[i]

    def __iter__(self):
        return iter(self.files)

    def append(self, value: FileEntry):
        self.files.append(value)

    def remove(self, value: FileEntry):
        self.files.remove(value)

    def insert(self, i: int, value: FileEntry):
        self.files.insert(i, value)

    def to_json(self) -> dict:
        return {
            "files": [v.to_json() for v in self.files],
            "metadata": self.metadata
        }

    @classmethod
    def from_json(cls, data: dict) -> 'FileIndex':
        files = [FileEntry.from_json(f) for f in data['files']]
        return cls(files, data['metadata'])