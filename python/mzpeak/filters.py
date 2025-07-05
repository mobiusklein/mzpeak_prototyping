from typing import Sequence
from numbers import Number
from enum import Enum, auto

import numpy as np
import pyarrow as pa


def estimate_median_delta(data: Sequence[Number]):
    deltas = np.diff(data)
    median = np.median(deltas)
    deltas_below = deltas[deltas <= median]
    median = np.median(deltas_below)
    return median, deltas_below


def fill_nulls(data: pa.Array, common_delta: float) -> "np.typing.NDArray":
    if isinstance(data, pa.Array):
        if not data.null_count:
            return np.asarray(data)

        tokenizer = NullTokenizer(data.is_null())
    else:
        data = pa.array(data)
        tokenizer = NullTokenizer(data.is_nan())
    buffer = []
    n = len(data)
    for token in tokenizer:
        if token[-1] == NullFillState.NullStart:
            (start, _, _) = token
            length = (n - start) - 1
            real_values = np.asarray(data.slice(start + 1, length))
            if length == 1:
                val = real_values[0]
                buffer.append((val - common_delta, val))
            else:
                local_delta, _ = estimate_median_delta(real_values)
                val0 = real_values[0]
                buffer.append((val0 - local_delta,))
                buffer.append(real_values)
        elif token[-1] == NullFillState.NullEnd:
            start = 0
            (_, end, _) = token
            length = end
            real_values = np.asarray(data.slice(start, length))
            if length == 1:
                val = real_values[0]
                buffer.append((val, val + common_delta))
            else:
                local_delta, _ = estimate_median_delta(real_values)
                buffer.append(real_values)
                buffer.append((real_values[-1] + local_delta,))
        elif token[-1] == NullFillState.NullBounded:
            (start, end, _) = token
            length = (end - start) - 1
            real_values = np.asarray(data.slice(start + 1, length))
            if length == 1:
                val = real_values[0]
                buffer.append((val - common_delta, val, val + common_delta))
            else:
                local_delta, _ = estimate_median_delta(real_values)
                val0 = real_values[0]
                val1 = real_values[-1]
                buffer.append((val0 - local_delta,))
                buffer.append(real_values)
                buffer.append((val1 + local_delta,))
        else:
            raise NotImplementedError(token)
    return np.concat(buffer)


class NullFillState(Enum):
    Unset = auto()
    NullStart = auto()
    NullEnd = auto()
    NullBounded = auto()


class NullTokenizer:
    array: Sequence[bool]
    index: int
    state: tuple[int, int | None, NullFillState] | None
    next_state: tuple[int, int | None, NullFillState] | None

    def __init__(self, array):
        self.array = np.asarray(array)
        self.index = 0
        self.state = None
        self.next_state = None
        self._initialize_state()

    def is_null(self) -> bool:
        return self.array[self.index]

    def _advance(self) -> bool:
        if self.index < len(self.array) - 1:
            self.index += 1
            return True
        return False

    def find_next_null(self):
        self._advance()
        while self.index < len(self.array) and not self.is_null():
            if not self._advance():
                break

    def update_next_state(self):
        prev = self.index
        self.find_next_null()
        diff = self.index - prev
        if diff == 0:
            self.next_state = None
        elif diff == 1:
            start = self.index
            self.find_next_null()
            end = self.index
            if self.is_null():
                self.next_state = (start, end, NullFillState.NullBounded)
            else:
                self.next_state = (start, None, NullFillState.NullStart)
        else:
            raise Exception(f"Trip run in null sequence at {prev}-{self.index}")

    def _initialize_state(self):
        self.index = 0
        start_null = self.is_null()
        self.find_next_null()
        if not start_null:
            self.state = (None, self.index, NullFillState.NullEnd)
            self.update_next_state()
        else:
            if self.is_null():
                self.state = (0, self.index, NullFillState.NullBounded)
                self.update_next_state()

    def emit(self) -> tuple[int | None, int | None, NullFillState] | None:
        state = self.state
        self.state = self.next_state
        self.update_next_state()
        return state

    def __iter__(self):
        while val := self.emit():
            yield val
