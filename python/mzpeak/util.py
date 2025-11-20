from numbers import Number
from typing import Generic, TypeVar


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