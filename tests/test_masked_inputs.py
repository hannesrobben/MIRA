"""Masked arrays keep their mask in the analysis functions (MI-77).

`x.data if hasattr(x, "data") else x` unmasks a numpy MaskedArray, and so
does `np.asarray`. Each test feeds a masked array whose masked entries
would visibly change the result if they were used.
"""

import numpy as np
import numpy.ma as ma
import pytest
from scipy import stats

from mira.analysis.comparison import fit_linear_regression
from mira.analysis.topology import analyze_error_vs_topology


def test_regression_plain_input_matches_scipy():
    rng = np.random.default_rng(3)
    x = rng.normal(size=40)
    y = 2.5 * x - 1.0 + rng.normal(0, 0.1, size=40)

    result = fit_linear_regression(x, y)
    reference = stats.linregress(x, y)

    assert result.slope == reference.slope
    assert result.intercept == reference.intercept
    assert result.n_points == 40


def test_regression_ignores_masked_pairs():
    x = np.arange(10, dtype=float)
    y = 3.0 * x + 1.0
    y[[2, 7]] = 1000.0  # outliers, masked below
    y_masked = ma.array(y, mask=np.isin(np.arange(10), [2, 7]))

    result = fit_linear_regression(x, y_masked)

    assert result.n_points == 8
    assert result.slope == pytest.approx(3.0, abs=1e-12)
    assert result.intercept == pytest.approx(1.0, abs=1e-12)
    assert 1000.0 not in result.y_values


def test_regression_masked_too_few_points():
    x = ma.array([1.0, 2.0, 3.0], mask=[False, True, True])
    with pytest.raises(Exception, match="At least 2"):
        fit_linear_regression(x, np.array([1.0, 2.0, 3.0]))


def test_topology_respects_mask():
    yy, xx = np.mgrid[0:50, 0:50]
    reference = (xx**2 + yy**2).astype(float) * 1e-9  # metres
    error = np.random.default_rng(4).normal(size=reference.shape)
    mask = np.zeros(reference.shape, dtype=bool)
    mask[:10, :10] = True

    result = analyze_error_vs_topology(
        ma.array(error, mask=mask), reference, pixel_size=(1e-6, 1e-6)
    )

    assert result.n_pixels == 2400
