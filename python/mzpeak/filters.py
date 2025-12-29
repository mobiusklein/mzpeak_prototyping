import logging

from typing import Any, Generator, Iterator, Sequence, NamedTuple
from numbers import Number
from enum import Enum, auto

import numpy as np
import pyarrow as pa

try:
    from scipy.linalg import solve_triangular
except ImportError:
    solve_triangular = None

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class DeltaModelBase:
    """A base class for coordinate spacing inference models"""
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
        return err.dot(err) / len(delta_array)

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
            The weight to put on each point in the m/z array. The intensity at the m/z value
            is an appropriate quantity.
        threshold : float, optional
            The maximum m/z delta to consider, discarding any point whose delta value is larger
            than this number. Defaults to ``1.0``.
        rank : int, optional
            The rank of the feature polynomial to construct, e.g. ``2 = β_0 + β_1 mz + β_2 mz^2``.
            Defaults to ``2``.
        """
        if solve_triangular is None:
            raise ImportError("To fit a linear model, install scipy")
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


def estimate_median_delta(data: Sequence[Number]) -> tuple[Number, np.typing.NDArray]:
    """
    Find the 2nd median of ``np.diff(data)``.

    This is a relatively crude spacing estimate for continuous profile data.

    Returns
    -------
    :class:`Number`
        The 2nd median of ``np.diff(data)``
    :class:`np.ndarray`
        The values from which the previous return values were estimated
    """
    deltas = np.diff(data)
    median = np.median(deltas)
    deltas_below = deltas[deltas <= median]
    median = np.median(deltas_below)
    return median, deltas_below


def fill_nulls(data: pa.Array, common_delta: DeltaModelBase | Number) -> "np.typing.NDArray":
    """
    Fill ``null`` values in ``data`` using the ``common_delta`` model or the locally estimated
    median delta if sufficient data are available.

    .. note::
        If ``data`` is a :class:`pyarrow.Array`, this will use :meth:`pyarrow.Array.is_null` to
        identify ``null``. Otherwise, it will assume :const:`np.nan` is the ``null`` value.

    Parameters
    ----------
    data : :class:`pyarrow.Array`
        The data array to fill nulls in with ``common_delta``
    common_delta : :class:`DeltaModelBase` or :class:`Number`
        The common spacing model, either specified as a model instance or as a single constant spacing term.

    Returns
    -------
    np.ndarray
    """
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
            elif not length:
                # The empty slice needs nothing, but this should not happen per
                logger.warn("An empty slice was found: %r", token)
                continue
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
    '''
    Describes a span that is bounded by nulls on one or more sides.

    Pairs with :class:`NullTokenizer`.

    .. note::
        This is an implementation detail of :func:`fill_nulls`. Do not use this directly.
    '''

    # Undefined state
    Unset = auto()
    # The associated range is null at the start of the interval, as
    # when we have a null pair that runs to the end of the array.
    NullStart = auto()
    # The associated range is null at the end of the interval, as
    # when we have an array that ends with null but started with a
    # value.
    NullEnd = auto()
    # The associated range has a null on both ends
    NullBounded = auto()


class NullSpanStep(NamedTuple):
    start: int | None
    end: int | None
    tag: NullFillState


class NullTokenizer:
    """
    An :class:`Iterator` that finds (start, end, :class:`NullFillState`) triples.

    .. note::
        This is an implementation detail of :func:`fill_nulls`. Do not use this directly.
    """
    array: Sequence[bool]
    _map: Sequence[int]
    _map_iter: Iterator[int]
    index: int
    state: NullSpanStep
    next_state: NullSpanStep

    def __init__(self, array: Sequence[bool]):
        self.array = np.asarray(array)
        self._map = np.where(self.array)[0]
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
        '''The current position is ``null``'''
        return self.array[self.index]

    def advance(self) -> bool:
        '''
        Advance the index one step, if possible.

        Returns
        -------
        :class:`bool`
            If the index was advanced or not.
        '''
        if self.index < len(self.array) - 1:
            self.index += 1
            return True
        return False

    def next_is_null(self) -> bool:
        '''If next position is ``null``'''
        return (self.index + 1 < len(self.array)) and self.array[self.index + 1]

    def previous_is_null(self) -> bool:
        '''If the previous position is ``null``'''
        i = self.index - 1
        return i > 0 and self.array[i]

    def consume_null_run(self):
        '''While :meth:`next_is_null` call :meth:`advance`'''
        while self.next_is_null():
            self.advance()

    def has_null_run_at(self) -> bool:
        '''Test if :meth:`is_null`, :meth:`previous_is_null`, and :meth:`next_is_null` are all :const:`True`'''
        return self.is_null() and self.previous_is_null() and self.next_is_null()

    def find_next_null(self):
        '''Find the next index where this array was null.'''
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
        # If we've exhausted the fast path iterator, fall back to this
        self.advance()
        while self.index < len(self.array) and not self.is_null():
            if not self.advance():
                break

    def update_next_state(self):
        '''
        Fidly bit, figure out what the next state is, given the current state and the
        next.
        '''
        prev = self.index
        # Move the iterator to the next null index
        self.find_next_null()
        # How big a step did we take?
        diff = self.index - prev
        if diff == 0:
            # We went nowhere, we must be at the end of the array and so our next state is undefined.
            self.next_state = None
        # We advanced one position, that means the next value was null too. This must be a null pair.
        elif diff == 1:
            # If this is amidst a null run, we'll behave differently.
            has_null_run = self.has_null_run_at()
            start = self.index
            # Hop ahead again to prepare the next state.
            self.find_next_null()
            # If we're in a null run, we must work our way out it.
            if has_null_run:
                # Traverse the null run and try again.
                self.consume_null_run()
                logger.error(f"Had null run, {prev}-{self.index}")
                return self.update_next_state()
            end = self.index
            if self.is_null():
                self.next_state = NullSpanStep(start, end, NullFillState.NullBounded)
            else:
                self.next_state = NullSpanStep(start, None, NullFillState.NullStart)
        # Or else we hit an unpaired null, a single null followed by a run of values that is
        # not a terminal.
        else:
            raise Exception(f"Malformed unpaired null sequence at {prev}-{self.index}")

    def _initialize_state(self):
        self.index = 0
        start_null = self.is_null()
        self.find_next_null()
        if not start_null:
            self.state = NullSpanStep(None, self.index, NullFillState.NullEnd)
            self.update_next_state()
        else:
            if self.is_null():
                self.state = NullSpanStep(0, self.index, NullFillState.NullBounded)
                self.update_next_state()
            else:
                self.state = NullSpanStep(0, None, NullFillState.NullStart)

        if len(self.array) < 3 and self.state is None:
            if start_null and not self.is_null():
                self.state = NullSpanStep(0, None, NullFillState.NullStart)
            elif not start_null and self.is_null():
                self.state = NullSpanStep(0, len(self.array), NullFillState.NullEnd)

    def emit(self) -> NullSpanStep | None:
        state = self.state
        self.state = self.next_state
        self.update_next_state()
        return state

    def __iter__(
        self,
    ) -> Generator[NullSpanStep, Any, None]:
        while val := self.emit():
            yield val

    @classmethod
    def from_pyarrow(cls, array: pa.Array) -> "NullTokenizer":
        return cls(array.is_null())


def find_where_not_zero_run(data: Sequence[Number]) -> Sequence[int]:
    """
    Construct a list of positions that are not part of a zero run.

    A zero run is any position *i* such that:
        1. ``x[i] == 0``
        2. ``(i == 0) or (x[i - 1] == 0)``
        3. ``(i == (len(x) - 1)) or (x[i + 1] == 0)``

    Parameters
    ----------
    data : :class:`Sequence` of :class:`Number`
        The numerical data to traverse

    Returns
    -------
    :class:`np.ndarray` of :class:`np.uintp`
    """
    n = len(data)
    n1 = n - 1

    # Whether we are currently in a zero run
    was_zero = False

    acc = []
    i = 0
    while i < n:
        v = data[i]
        if v is not None:
            if v == 0:
                if (was_zero or (len(acc) == 0)) and (
                    (i < n1 and data[i + 1] == 0) or i == n1
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
    return np.array(acc, dtype=np.uintp)


def is_zero_pair_mask(data: Sequence[Number]) -> "np.typing.NDArray[np.bool_]":
    '''
    Create a boolean mask for positions that are composed of two zeroes in a row.

    Parameters
    ----------
    data : :class:`Sequence` of :class:`Number`
        The numerical data to traverse

    Returns
    -------
    :class:`np.ndarray` of :class:`bool`
    '''
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
    """
    Delta-encode an Arrow array containing nulls. Nulls are encoded as null values, and treated as 0.0
    for the purposes of computing the next delta.

    Parameters
    ----------
    data : pa.Array
        The data to delta encode

    Returns
    -------
    pa.Array
    """
    acc = []
    it = iter(data)
    # Get the first entry in the array. It will be the first point of reference but not part
    # of the delta sequence unless it is `null`
    last = next(it)
    if not last.is_valid:
        acc.append(last)

    for item in it:
        # If the value isn't `null`,
        if item.is_valid:
            val = item.as_py()
            # Compute a delta relative to the last item if it was not `null`
            if last.is_valid:
                acc.append(pa.scalar(val - last.as_py()))
            # otherwise treat the last value as 0.0, the additive identity
            else:
                acc.append(item)
            # Update last item
            last = item
        else:
            # Append the `null` unmodified and update the last item.
            acc.append(item)
            last = item
    return pa.array(acc)


def null_delta_decode(data: pa.Array, start: pa.Scalar) -> pa.Array:
    """
    Decode an Arrow array that was delta-encoded *with* nulls.

    This is necessarily a copying operation.

    Parameters
    ----------
    data : pa.Array
        The data to be decoded.
    start : pa.Scalar
        The starting value, an offset

    Returns
    -------
    pa.Array
    """
    acc = []
    # If the first value is `null`,
    if not data[0].is_valid:
        # and the second value is `null`,
        if not data[1].is_valid:
            # then append the `start` value, we started at a non-null value immediately followed by a null pair.
            acc.append(start)
        start = pa.scalar(None, data.type)
    else:
        # otherwise use the starting point
        acc.append(start.as_py())
    last = start
    for item in data:
        # if the current point is valid
        if item.is_valid:
            val = item.as_py()
            # and the last is valid
            if last.is_valid:
                # reconstitute the delta encoded value at this position
                last = pa.scalar(val + last.as_py())
                acc.append(last)
            else:
                # otherwise the last value is assumed to be zero so it does
                # not need to be adjusted
                acc.append(item)
                last = item
        else:
            # otherwise this position is null and we carry it forward as such
            acc.append(item)
            last = item
    return pa.array(acc)


def null_chunk_every(data: pa.Array, width: float) -> list[tuple[int, int]]:
    """
    Partition a sorted numerical array into segments spanning `width` units.

    This operation is null-aware, so sparse arrays can be partitioned.

    Parameter
    ---------
    data : pa.Array
        The data to be partitioned
    width : float
        The spacing (in units along the data dimension) between chunks

    Returns
    -------
    list[tuple[int, int]]
        The start and end index of each chunk
    """
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
    threshold = start + width
    i = 0
    while i < n:
        v = data[i]
        if v.is_valid:
            v = v.as_py()
            if v > threshold:
                if ((i + 1) < n) and (not data[i + 1].is_valid):
                    while ((i + 1) < n) and (not data[i + 1].is_valid):
                        i += 1
                # We don't want to create a chunk of length 1, especially not if it is a null
                # point. If not, we have to relax the width requirement. Also, the way this cut
                # is made ensures that if we do have a null pair, this will split one end evenly into
                # one chunk and the other end into the next chunk. In the event of a null run, the final
                # null will be part of the next chunk but all other nulls will go in the first chunk.
                if i - offset > 1:
                    chunks.append((offset, i))
                    offset = i
                # Update the threshold. We might need to update multiple times if the next value
                # is far away.
                while threshold < v:
                    threshold += width
        # Look ahead and see if the next value is not null since this one is.
        elif ((i + 1) < n) and (data[i + 1].is_valid):
            i += 1
            v = data[i].as_py()
            if v > threshold:
                i -= 1
                chunks.append((offset, i))
                offset = i
                # Update the threshold. We might need to update multiple times if the next value
                # is far away.
                while threshold < v:
                    threshold += width
        i += 1
    if offset != n:
        chunks.append((offset, n))
    return chunks