"""StructureProcessor.fit_segment_regression (MI-73).

Synthetic pair: a 60 x 80 px rectangle split into 2 x 2 quadrants with
the levels 11, 12, 13, 14 on a flat background of 0 (reference; all
levels well above the background, so that the threshold detection finds
one rectangle), and the same layout with every level scaled by 3 plus an
offset of 0.5 (measured).
With background-relative means the fit must give slope 3 exactly and,
because the background is a point at (0, 0), intercept 0.
"""

import logging

import numpy as np
import pytest

from mira import StructureProcessor
from mira.analysis.comparison import RegressionResult, fit_segment_regression
from mira.utils.exceptions import ValidationError

LEVELS = 10.0 + np.array([[1.0, 2.0], [3.0, 4.0]])


def _field(scale: float, offset: float) -> np.ndarray:
    field = np.zeros((120, 140))
    for i in range(2):
        for j in range(2):
            rows = slice(30 + 30 * i, 60 + 30 * i)
            cols = slice(30 + 40 * j, 70 + 40 * j)
            field[rows, cols] = scale * LEVELS[i, j]
    return field + offset


def _processor(field: np.ndarray, segment: bool = True) -> StructureProcessor:
    proc = StructureProcessor(
        phase_data=field, pixel_size=(1e-6, 1e-6), log_level=logging.ERROR
    ).detect_structure()
    if segment:
        proc.segment_structure(num_segments=(2, 2))
    return proc


def test_method_fits_and_stores_the_regression():
    reference = _processor(_field(1.0, 0.0))
    measured = _processor(_field(3.0, 0.5))

    result = measured.fit_segment_regression(reference, offset_pixels=5)

    assert isinstance(result, RegressionResult)
    assert result.slope == pytest.approx(3.0, abs=1e-9)
    assert result.intercept == pytest.approx(0.0, abs=1e-9)
    assert result.n_points == 5  # four segments plus the background
    assert measured.structure.statistics["segment_regression"] is result


def test_method_matches_the_analysis_function():
    reference = _processor(_field(1.0, 0.0))
    measured = _processor(_field(3.0, 0.5))

    result = measured.fit_segment_regression(
        reference,
        offset_pixels=5,
        include_background=False,
        relative=False,
        statistic_name="vs_reference",
    )
    direct = fit_segment_regression(
        reference.structure,
        measured.structure,
        offset_pixels=5,
        include_background=False,
        relative=False,
    )

    assert (result.slope, result.intercept) == (direct.slope, direct.intercept)
    assert "vs_reference" in measured.structure.statistics


def test_method_requires_segmented_processors():
    segmented = _processor(_field(1.0, 0.0))
    unsegmented = _processor(_field(1.0, 0.0), segment=False)

    with pytest.raises(ValidationError, match="self must be segmented"):
        unsegmented.fit_segment_regression(segmented)
    with pytest.raises(ValidationError, match="reference must be segmented"):
        segmented.fit_segment_regression(unsegmented)
