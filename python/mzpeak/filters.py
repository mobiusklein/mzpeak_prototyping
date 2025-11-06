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
    """
    A :class:`DeltaModelBase` that uses a constant value like the median spacing
    across the spectrum.

    Effectively equivalent to :class:`DeltaCurveRegressionModel` with only the intercept
    coefficient at inference time.

    Attributes
    ----------
    delta : float
        The spacing between m/z values
    """
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

    def __repr__(self):
        return f"{self.__class__.__name__}({self.delta})"


class DeltaCurveRegressionModel(DeltaModelBase):
    r"""
    A :class:`DeltaModelBase` that uses a weighted least squares regression model
    for m/z spacing as a function of m/z value.

    .. math::
        δ mz \sim β_0 + β_1 mz + β_2 mz^2 + ... + ϵ

    Attributes
    ----------
    beta : :class:`np.ndarray`
        The estimated linear model coefficients
    """
    beta: np.ndarray

    def __init__(self, beta: np.ndarray):
        self.beta = beta

    @classmethod
    def fit(
        cls,
        mz_array: np.ndarray,
        delta_array: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        threshold: float | None = None,
        rank: int = 2,
    ):
        """
        Fit a weighted least squares model on the provided m/z array.

        Arguments
        ---------
        mz_array : np.ndarray
            The m/z array to fit the model on.
        delta_array : np.ndarray, optional
            The difference between successive values in the m/z array. Should be
            one shorter than ``mz_array``. If not provided, will be computed from
            ``np.diff(mz_array)`` which has the expected alignment.
        weights : np.ndarray, optional
            The weight to put on each point in the m/z array. Is expected to be **sqrt**
            transformed. The intensity at the m/z value is an appropriate quantity.
        threshold : float, optional
            The maximum m/z delta to consider, discarding any point whose delta value is larger
            than this number. Defaults to ``1.0``.
        rank : int, optional
            The rank of the feature polynomial to construct, e.g. ``2 = β_0 + β_1 mz + β_2 mz^2``.
            Defaults to ``2``.
        """
        if delta_array is None:
            delta_array = np.diff(mz_array)

        if weights is None:
            weights = np.ones(len(mz_array))
        else:
            weights = weights

        if threshold is None:
            threshold = 1.0
        data = []
        raw = mz_array[1:][delta_array <= threshold]
        w = weights[1:][delta_array <= threshold]
        y = delta_array[delta_array <= threshold]
        for i in range(rank + 1):
            if i == 0:
                data.append(np.ones_like(raw))
            elif i == 1:
                data.append(raw)
            else:
                data.append(raw**i)
        data = np.stack(data, axis=-1)

        # Use the QR decomposition to solve the weighted least squares problem
        # to estimate weights predicting δ m/z.
        # https://stats.stackexchange.com/a/490782/59613
        chol_w = np.sqrt(w)
        qr = np.linalg.qr(chol_w[:, None] * data)
        v = qr.Q.T.dot(chol_w * y)
        beta = solve_triangular(qr.R, v)

        # Numerically equivalent to and more stable than the direct inversion
        # beta = np.linalg.inv((data.T * w).dot(data)).dot(data.T * w).dot(y)
        return cls(beta)

    def predict(self, mz: float) -> float:
        acc = self.beta[0]
        for i in range(1, len(self.beta)):
            acc += self.beta[i] * mz ** i
        return acc

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

        tokenizer = _NullTokenizer(data.is_null())
    else:
        data = pa.array(data)
        tokenizer = _NullTokenizer(data.is_nan())
    buffer = []
    n = len(data)
    for token in tokenizer:
        if token[-1] == _NullFillState.NullStart:
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
        elif token[-1] == _NullFillState.NullEnd:
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
        elif token[-1] == _NullFillState.NullBounded:
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


class _NullFillState(Enum):
    Unset = auto()
    NullStart = auto()
    NullEnd = auto()
    NullBounded = auto()


class _NullTokenizer:
    array: Sequence[bool]
    _map: Sequence[int]
    _map_iter: Iterator[int]
    index: int
    state: tuple[int, int | None, _NullFillState] | None
    next_state: tuple[int, int | None, _NullFillState] | None

    def __init__(self, array):
        self.array = np.asarray(array)
        self._map = np.where(self.array)[0]
        if len(self._map) == 1:
            pass
        self._map_iter = iter(self._map)
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
            prev = self.index
            self.index = next(self._map_iter)
            # If the 0th position is null, then this will double-count index 0
            # so we check for it and try again if the index is the same.
            if self.index == prev:
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
                self.next_state = (start, end, _NullFillState.NullBounded)
            else:
                self.next_state = (start, None, _NullFillState.NullStart)
        else:
            raise Exception(f"Trip run in null sequence at {prev}-{self.index}")

    def _initialize_state(self):
        self.index = 0
        start_null = self.is_null()
        self.find_next_null()
        if not start_null:
            self.state = (None, self.index, _NullFillState.NullEnd)
            self.update_next_state()
        else:
            if self.is_null():
                self.state = (0, self.index, _NullFillState.NullBounded)
                self.update_next_state()
            else:
                self.state = (0, None, _NullFillState.NullStart)

        if len(self.array) < 3 and self.state is None:
            if start_null and not self.is_null():
                self.state = (0, None, _NullFillState.NullStart)
            elif not start_null and self.is_null():
                self.state = (0, None, _NullFillState.NullStart)

    def emit(self) -> tuple[int | None, int | None, _NullFillState] | None:
        state = self.state
        self.state = self.next_state
        self.update_next_state()
        return state

    def __iter__(self):
        while val := self.emit():
            yield val


def find_zero_runs(arr: Sequence[Number]) -> Sequence[int]:
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


def is_zero_pair_mask(data: Sequence[Number]):
    n = len(data)
    n1 = n - 1
    was_zero = False
    acc = []
    for i, v in enumerate(data):
        if v == 0:
            if was_zero or (i < n1 and data[i + 1] == 0):
                acc.append(True)
            else:
                acc.append(False)
            was_zero = True
        else:
            acc.append(False)
            was_zero = False
    return np.array(acc)


def null_delta_encode(data: pa.Array) -> pa.Array:
    acc = []
    it = iter(data)
    last = next(it)
    if not last.is_valid:
        acc.append(last)

    for item in it:
        if item.is_valid:
            val = item.as_py()
            if last.is_valid:
                acc.append(pa.scalar(val - last.as_py()))
            else:
                acc.append(item)
            last = item
        else:
            acc.append(item)
            last = item
    return pa.array(acc)


def null_delta_decode(data: pa.Array, start: pa.Scalar) -> pa.Array:
    acc = []
    if not data[0].is_valid:
        if not data[1].is_valid:
            acc.append(start)
        start = pa.scalar(None, data.type)
    else:
        acc.append(start.as_py())
    last = start
    for item in data:
        if item.is_valid:
            val = item.as_py()
            if last.is_valid:
                last = pa.scalar(val + last.as_py())
                acc.append(last)
            else:
                acc.append(item)
                last = item
        else:
            acc.append(item)
            last = item
    return pa.array(acc)


def null_chunk_every(data: pa.Array, k: float) -> list[tuple[int, int]]:
    start = None
    n = len(data)
    i = 0
    # Find the first non-null position
    while i < n:
        v = data[i]
        if v.is_valid:
            start = v.as_py()
            break
        else:
            i += 1

    # If we never found a non-null position, just return a single chunk
    if start is None:
        return [(0, n)]

    chunks = []
    offset = 0
    t = start + k
    i = 0
    while i < n:
        v = data[i]
        if v.is_valid:
            v = v.as_py()
            if v > t:
                if ((i + 1) < n) and (not data[i + 1].is_valid):
                    i += 2
                # We don't want to create a chunk of length 1, especially not if it is a null
                # point.
                if i - offset > 1:
                    chunks.append((offset, i))
                    offset = i
                while t < v:
                    t += k
        elif ((i + 1) < n) and (data[i + 1].is_valid):
            i += 1
            v = data[i].as_py()
            if v > t:
                i -= 1
                chunks.append((offset, i))
                offset = i
                while t < v:
                    t += k
        i += 1
    if offset != n:
        chunks.append((offset, n))
    return chunks
