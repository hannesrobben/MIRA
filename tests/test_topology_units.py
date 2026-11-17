"""The topology analysis expects the pixel size in metres (MI-78)."""

import numpy as np
import pytest

from mira.analysis.topology import (
    analyze_error_vs_topology,
    compute_curvature,
    compute_gradient_magnitude,
)
from mira.utils.exceptions import ValidationError


@pytest.fixture
def surface():
    yy, xx = np.mgrid[0:40, 0:40] * 1e-6  # metres
    reference = 2.0 * xx  # slope 2 m/m along x
    error = np.random.default_rng(5).normal(size=reference.shape)
    return reference, error


def test_gradient_is_per_metre(surface):
    reference, _ = surface
    gradient = compute_gradient_magnitude(reference, pixel_size=(1e-6, 1e-6))
    np.testing.assert_allclose(gradient, 2.0)


@pytest.mark.parametrize(
    "func", [compute_gradient_magnitude, compute_curvature, "analyze"]
)
@pytest.mark.parametrize(
    "pixel_size, message",
    [
        ((0.5, 0.5), "µm"),
        ((1e-6, 2.0), "µm"),
        ((0.0, 1e-6), "positive"),
        ((-1e-6, 1e-6), "positive"),
        ((np.nan, 1e-6), "positive"),
        ((1e-6,), "pair"),
        ("1e-6", "pair"),
    ],
)
def test_implausible_pixel_size_is_rejected(surface, func, pixel_size, message):
    reference, error = surface
    with pytest.raises(ValidationError, match=message):
        if func == "analyze":
            analyze_error_vs_topology(error, reference, pixel_size=pixel_size)
        else:
            func(reference, pixel_size=pixel_size)


def test_metre_pixel_size_is_accepted(surface):
    reference, error = surface
    result = analyze_error_vs_topology(error, reference, pixel_size=(1e-6, 1e-6))
    assert result.n_pixels == reference.size
