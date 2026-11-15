"""
Shared segment-statistics infrastructure.

Extracts per-segment (and optional background) mean/std statistics from a
segmented `Structure`, with a fixed-pixel edge margin excluded to reduce
the influence of imprecise segmentation boundaries, and pairs those
statistics between two identically segmented structures for downstream
fitting.

This is the single place where "how do we turn a segmented Structure into
a list of representative values" is decided - every comparison that
operates segment-wise (delta_n fitting in `processing.calibration`,
regression fitting in `analysis.comparison`, and future methods such as
error-vs-curvature or multi-print repeatability analysis) should build on
`paired_segment_values` here rather than re-implementing segment
extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy import ma

from ..core.structure import Background, Structure
from ..utils.exceptions import ValidationError

# Sentinel id used for the background "segment" in results below, chosen to
# never collide with a real segment id (segment ids start at 0).
BACKGROUND_SEGMENT_ID = -1


@dataclass
class SegmentStatistic:
    """
    Mean/std of one segment (or the background), computed on the eroded
    (inner) region only.

    Attributes:
        id:              Segment id. `BACKGROUND_SEGMENT_ID` (-1) marks the
                         background entry.
        mean:            Mean value on the eroded region.
        std:             Standard deviation on the eroded region.
        n_pixels:        Number of pixels the statistic was computed from.
        mean_relative:   `mean` minus the background mean of the same
                         channel. Left at 0.0 until `make_relative()` is
                         applied; equals exactly 0.0 for the background
                         entry itself.
    """

    id: int
    mean: float
    std: float
    n_pixels: int
    mean_relative: float = field(default=0.0)


def extract_segment_statistics(
    structure: Structure, offset_pixels: int, include_background: bool
) -> list[SegmentStatistic]:
    """
    Compute eroded-region mean/std for every segment of `structure` (and
    optionally its background), in ascending segment-id order.

    Args:
        structure:           Segmented Structure to extract statistics from.
        offset_pixels:       Pixels excluded from every segment/background
                             edge before computing statistics.
        include_background:  If True, the background is included as the
                             first entry (id = BACKGROUND_SEGMENT_ID).

    Returns:
        List of SegmentStatistic, background first (if included), then
        segments in ascending id order.

    Raises:
        ValidationError: If `structure` has no segments, or a segment/the
            background has no pixels left after erosion.
    """
    if not structure.segments:
        raise ValidationError(
            "Structure has no segments - call segment_structure() before "
            "extracting segment statistics."
        )

    stats: list[SegmentStatistic] = []

    if include_background:
        bg_mask = structure.data.mask  # True = background (outside structure)
        bg_data = ma.masked_array(structure.data.data, mask=~bg_mask)
        background = Background(data=bg_data, pixel_size=structure.pixel_size)
        eroded = background.erode_by_pixels(offset_pixels)
        n_pixels = int(np.sum(~eroded.mask))
        if n_pixels == 0:
            raise ValidationError(
                f"Background has no pixels left after eroding by "
                f"{offset_pixels} px - reduce offset_pixels."
            )
        stats.append(
            SegmentStatistic(
                id=BACKGROUND_SEGMENT_ID,
                mean=float(eroded.mean()),
                std=float(eroded.std()),
                n_pixels=n_pixels,
            )
        )

    for segment in sorted(structure.segments, key=lambda s: s.id):
        eroded = segment.erode_by_pixels(offset_pixels)
        n_pixels = int(np.sum(~eroded.mask))
        if n_pixels == 0:
            raise ValidationError(
                f"Segment {segment.id} has no pixels left after eroding by "
                f"{offset_pixels} px - reduce offset_pixels or check the "
                f"segmentation."
            )
        stats.append(
            SegmentStatistic(
                id=segment.id,
                mean=float(eroded.mean()),
                std=float(eroded.std()),
                n_pixels=n_pixels,
            )
        )

    return stats


def make_relative(stats: list[SegmentStatistic]) -> None:
    """
    In-place: set `mean_relative` on every entry to `mean` minus the
    background mean (0.0 if no background entry is present in `stats`).
    """
    background = next((s for s in stats if s.id == BACKGROUND_SEGMENT_ID), None)
    background_mean = background.mean if background is not None else 0.0
    for stat in stats:
        stat.mean_relative = stat.mean - background_mean


def paired_segment_values(
    structure_a: Structure,
    structure_b: Structure,
    offset_pixels: int = 10,
    include_background: bool = True,
    relative: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[SegmentStatistic], list[SegmentStatistic]]:
    """
    Extract segment statistics from two identically segmented structures
    and return them as aligned value arrays, ready for a fit.

    Args:
        structure_a:         First segmented Structure (e.g. phase, or a
                             "reference"/x-axis quantity).
        structure_b:         Second segmented Structure, segmented
                             identically to `structure_a` (same number and
                             ordering of segments).
        offset_pixels:       Pixels excluded from every segment/background
                             edge before computing statistics.
        include_background:  If True, include the background as an
                             additional data point.
        relative:            If True, return background-relative values
                             (`mean_relative`); if False, return absolute
                             means (`mean`).

    Returns:
        (values_a, values_b, stats_a, stats_b) - the two value arrays in
        matching segment order, plus the full SegmentStatistic lists for
        callers that need std/n_pixels/id as well.

    Raises:
        ValidationError: If either structure is unsegmented, or their
            segment counts (incl. background, if requested) don't match.
    """
    stats_a = extract_segment_statistics(structure_a, offset_pixels, include_background)
    stats_b = extract_segment_statistics(structure_b, offset_pixels, include_background)

    if len(stats_a) != len(stats_b):
        raise ValidationError(
            f"Segment count mismatch: first structure has {len(stats_a)} "
            f"data points (incl. background), second has {len(stats_b)}. "
            f"Both structures must be segmented identically."
        )

    if relative:
        make_relative(stats_a)
        make_relative(stats_b)
        values_a = np.array([s.mean_relative for s in stats_a])
        values_b = np.array([s.mean_relative for s in stats_b])
    else:
        values_a = np.array([s.mean for s in stats_a])
        values_b = np.array([s.mean for s in stats_b])

    return values_a, values_b, stats_a, stats_b
