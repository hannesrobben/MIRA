"""Bland-Altman split into compute and plot (MI-72, MI-77).

The expected numbers were produced on 2026-11-17 with the former combined
function `mira.visualization.bland_altman_snippet.plot_bland_altman`
(commit a7696ae) on the same synthetic data, so these tests show that the
split reproduces it exactly (rel=1e-12 only absorbs float formatting).
"""

import numpy as np
import numpy.ma as ma
import pytest
from matplotlib.figure import Figure

import mira.visualization as visualization
from mira.analysis.comparison import BlandAltmanResult, bland_altman_compute
from mira.utils.exceptions import ValidationError, VisualizationError
from mira.visualization.bland_altman import plot_bland_altman as plot_combined
from mira.visualization.plotter_stats import plot_bland_altman


@pytest.fixture
def pair():
    rng = np.random.default_rng(42)
    a = rng.normal(5.0, 1.0, (50, 50))
    b = a + 0.2 + rng.normal(0, 0.1, a.shape)
    mask = np.zeros(a.shape, dtype=bool)
    mask[:10, :10] = True
    return a, b, mask


def _stats(result):
    return (
        result.bias,
        result.std_diff,
        result.loa_lower,
        result.loa_upper,
        result.n_points,
        result.mean_of_means,
    )


EXPECTED = {
    "plain": (
        -0.2004574145047272,
        0.09927646038729573,
        -0.39503927686382684,
        -0.005875552145627572,
        2500,
        5.055900483220694,
    ),
    "float32": (
        -0.20045741276741028,
        0.09927646263211701,
        -0.39503927952635964,
        -0.005875546008460936,
        2500,
        5.055900483608246,
    ),
    "relative": (
        -4.122692433265314,
        2.2453152045338336,
        -8.523510234151628,
        0.2781253676209996,
        2500,
        5.055900483220694,
    ),
    "masked": (
        -0.2006038028801408,
        0.09911836315935492,
        -0.39487579467247647,
        -0.006331811087805145,
        2400,
        5.055392506397505,
    ),
    "masked_relative": (
        -4.1266411126446005,
        2.2436846988472587,
        -8.524263122385229,
        0.2709808970960266,
        2400,
        5.055392506397505,
    ),
}


def _inputs(case, a, b, mask):
    if case == "float32":
        return a.astype(np.float32), b.astype(np.float32), {}
    if case.startswith("masked"):
        a, b = ma.array(a, mask=mask), ma.array(b, mask=mask)
    return a, b, {"relative": case.endswith("relative")}


@pytest.mark.parametrize("case", sorted(EXPECTED))
def test_compute_reproduces_former_function(pair, case):
    data_1, data_2, kwargs = _inputs(case, *pair)
    result = bland_altman_compute(data_1, data_2, **kwargs)
    assert _stats(result) == pytest.approx(EXPECTED[case], rel=1e-12)
    assert result.means.shape == result.differences.shape == (result.n_points,)


def test_masked_entries_are_left_out(pair):
    a, b, mask = pair
    b_bad = b.copy()
    b_bad[mask] = 1e6  # would dominate the bias if used
    result = bland_altman_compute(ma.array(a, mask=mask), ma.array(b_bad, mask=mask))
    assert result.n_points == 2400
    assert abs(result.bias) < 1.0


def test_wrapper_object_is_unwrapped(pair):
    a, b, _ = pair

    class Wrapper:
        def __init__(self, data):
            self.data = data

    assert _stats(bland_altman_compute(Wrapper(a), Wrapper(b))) == pytest.approx(
        EXPECTED["plain"], rel=1e-12
    )


def test_compute_validates_input():
    with pytest.raises(ValidationError, match="Shape mismatch"):
        bland_altman_compute(np.zeros(3), np.zeros(4))
    with pytest.raises(ValidationError, match="No finite"):
        bland_altman_compute(np.full(3, np.nan), np.zeros(3))


def test_plot_takes_a_result(pair, tmp_path):
    a, b, _ = pair
    fig = plot_bland_altman(
        bland_altman_compute(a, b), dpi=50, save_path=str(tmp_path / "ba.png")
    )
    assert isinstance(fig, Figure)
    assert (tmp_path / "ba.png").exists()


def test_plot_rejects_raw_data(pair):
    a, b, _ = pair
    with pytest.raises(VisualizationError, match="bland_altman_compute"):
        plot_bland_altman(a, b)


def test_combined_wrapper_keeps_former_call(pair):
    a, b, mask = pair
    fig, result = plot_combined(ma.array(a, mask=mask), ma.array(b, mask=mask), dpi=50)
    assert isinstance(fig, Figure)
    assert _stats(result) == pytest.approx(EXPECTED["masked"], rel=1e-12)
    with pytest.raises(VisualizationError, match="Shape mismatch"):
        plot_combined(np.zeros(3), np.zeros(4))


def test_package_exports_the_split_functions():
    assert visualization.plot_bland_altman is plot_bland_altman
    assert visualization.BlandAltmanResult is BlandAltmanResult
