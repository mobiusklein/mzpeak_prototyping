from pathlib import Path

import numpy as np
import pyarrow as pa

from mzpeak.filters import (
    find_where_not_zero_run,
    is_zero_pair_mask,
    null_chunk_every,
    DeltaCurveRegressionModel,
    null_delta_encode,
    NullTokenizer
)


def test_null_singleton():
    it = NullTokenizer([False, True, False])
    list(it)


def test_delta_encode():
    dat = np.arange(0, 1000.0, 0.01)
    arrow_dat = pa.array(dat)
    deltas = null_delta_encode(arrow_dat)
    deltas2 = np.diff(dat)
    assert (deltas == deltas2).all()


def test_chunking():
    mzs = []
    intensities = []
    with Path("test/data/sparse_large_gaps.txt").open("rt") as fh:
        for line in fh:
            i, j = map(float, line.strip().split("\t"))
            mzs.append(i)
            intensities.append(j)

    mzs = np.array(mzs)
    intensities = np.array(intensities)

    assert len(intensities) == 9317
    assert np.count_nonzero(intensities) == 4243

    # Test that the model learned from the full data is the same as the same as
    # the model learned from the data without excess zero-intensity points.
    model1 = DeltaCurveRegressionModel.fit(mzs, np.diff(mzs), np.sqrt(intensities))

    nonzero_run_mask = find_where_not_zero_run(intensities)
    mzs = mzs[nonzero_run_mask]
    intensities = intensities[nonzero_run_mask]

    model2 = DeltaCurveRegressionModel.fit(mzs, np.diff(mzs), np.sqrt(intensities))

    assert len(model1.beta) == 3
    assert len(model2.beta) == len(model1.beta)

    for i in range(len(model1.beta)):
        assert np.isclose(model1.beta[i], model2.beta[i]), (
            f"Parameter {i} of model {model1.beta[i]} !~ {model2.beta[i]}"
        )
    err = model2.mse(mzs[1:], np.diff(mzs))
    assert abs(err - 43.48575002014233) < 1e-3

    # Test null masking does not fail
    assert len(intensities) == 5546
    assert np.count_nonzero(intensities) == 4243

    # This quantity is smaller than the total number of zeros because not all zeros are paired
    zero_pair_mask = is_zero_pair_mask(intensities)
    assert zero_pair_mask.sum() == 1286

    mzs = pa.array(mzs, mask=zero_pair_mask)
    intensities = pa.array(intensities, mask=zero_pair_mask)
    assert mzs.null_count == 1286

    chunk_indices = len(null_chunk_every(mzs, 50))
    assert chunk_indices == 33