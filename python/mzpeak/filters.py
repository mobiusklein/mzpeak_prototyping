from typing import Iterator, Sequence
from numbers import Number
from enum import Enum, auto

import numpy as np
import pyarrow as pa

from scipy.linalg import solve_triangular


def fit_qr(x: np.ndarray, y: np.ndarray):
    quadx = np.stack([np.ones_like(x), x, x**2], axis=-1)
    qr = np.linalg.qr(quadx)
    v = qr.Q.T.dot(y)
    return solve_triangular(qr.R, v)


def fit_qr_weighted(x: np.ndarray, y: np.ndarray, weights: np.ndarray):
    quadx = np.stack([np.ones_like(x), x, x**2], axis=-1)
    chol_w = np.sqrt(weights)
    qr = np.linalg.qr(chol_w[:, None] * quadx)
    v = qr.Q.T.dot(chol_w * y)
    return solve_triangular(qr.R, v)


class DeltaModelBase:
    @classmethod
    def fit(
        cls,
        mz_array,
        delta_array,
        weights: np.ndarray | None = None,
        threshold: float | None = None,
    ):
        raise NotImplementedError()

    def predict(self, mz: float) -> float:
        raise NotImplementedError()

    def mse(self, mz_array: np.ndarray, delta_array: np.ndarray):
        err = self(mz_array) - delta_array
        return err.dot(err)

    def __call__(self, mz: float) -> float:
        return self.predict(mz)


class ConstantDeltaModel(DeltaModelBase):
    delta: float

    def __init__(self, delta: float):
        self.delta = delta

    @classmethod
    def fit(
        cls,
        mz_array,
        delta_array,
        weights: np.ndarray | None = None,
        threshold: float | None = None,
    ):
        val = estimate_median_delta(mz_array)[0]
        return cls(val)

    def predict(self, mz: float) -> float:
        return self.delta

    def __call__(self, mz: float) -> float:
        return self.predict(mz)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.delta})"


class DeltaCurveRegressionModel(DeltaModelBase):
    beta: np.ndarray

    def __init__(self, beta: np.ndarray):
        self.beta = beta

    @classmethod
    def fit(
        cls,
        mz_array,
        delta_array,
        weights: np.ndarray | None = None,
        threshold: float | None = None,
        rank: int = 2,
    ):
        if weights is None:
            weights = np.ones(len(mz_array))
        else:
            weights = weights

        if threshold is None:
            threshold = 1.0
        data = []
        raw = mz_array[1:][delta_array <= threshold]
        w = weights[1:][delta_array <= threshold]
        for i in range(rank + 1):
            if i == 0:
                data.append(np.ones_like(raw))
            elif i == 1:
                data.append(raw)
            else:
                data.append(raw**i)
        data = np.stack(data, axis=-1)
        y = delta_array[delta_array <= threshold]

        beta = np.linalg.inv((data.T * w).dot(data)).dot(data.T * w).dot(y)
        return cls(beta)

    def predict(self, mz: float) -> float:
        acc = self.beta[0]
        for i in range(1, len(self.beta)):
            acc += self.beta[i] * mz ** i
        return acc

    def __call__(self, mz: float) -> float:
        return self.predict(mz)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.beta})"


def estimate_median_delta(data: Sequence[Number]):
    deltas = np.diff(data)
    median = np.median(deltas)
    deltas_below = deltas[deltas <= median]
    median = np.median(deltas_below)
    return median, deltas_below


def fill_nulls(data: pa.Array, common_delta: DeltaModelBase | Number) -> "np.typing.NDArray":
    if not isinstance(common_delta, DeltaModelBase):
        if isinstance(common_delta, Number):
            common_delta = ConstantDeltaModel(common_delta)
        else:
            common_delta = DeltaCurveRegressionModel(np.asarray(common_delta))

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
                buffer.append((val - common_delta(val), val))
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
                buffer.append((val, val + common_delta(val)))
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
                buffer.append((val - common_delta(val), val, val + common_delta(val)))
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
    _map: Sequence[int]
    _map_iter: Iterator[int]
    index: int
    state: tuple[int, int | None, NullFillState] | None
    next_state: tuple[int, int | None, NullFillState] | None

    def __init__(self, array):
        self.array = np.asarray(array)
        self._map = np.where(self.array)[0]
        self._map_iter = iter(np.where(self.array)[0])
        self.index = 0
        self.state = None
        self.next_state = None
        self._initialize_state()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.index}, {self.state}, {self.next_state})"
        )

    def is_null(self) -> bool:
        return self.array[self.index]

    def _advance(self) -> bool:
        if self.index < len(self.array) - 1:
            self.index += 1
            return True
        return False

    def find_next_null(self):
        try:
            self.index = next(self._map_iter)
            return
        except StopIteration:
            pass
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


def find_zero_runs(arr: list) -> list[int]:
    n = len(arr)
    n1 = n - 1
    was_zero = False
    acc = []
    i = 0
    while i < n:
        v = arr[i]
        if v is not None:
            if v == 0:
                if (was_zero or (len(acc) == 0)) and (
                    (i < n1 and arr[i + 1] == 0) or i == n1
                ):
                    pass
                else:
                    acc.append(i)
                was_zero = True
            else:
                acc.append(i)
                was_zero = False
        else:
            acc.append(i)
            was_zero = False
        i += 1
    return np.array(acc)
