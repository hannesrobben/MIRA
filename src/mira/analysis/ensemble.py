"""
Ensemble comparison across multiple structures (e.g. repeated prints of the
same design, or several DOEs).

Where every other module in `analysis/` compares exactly two structures
(delta_n fitting, regression), this module answers a different question:
"is a quantity consistent across N structures?" - e.g. is delta_n stable
across DOE3, DOE4, DOE5, or does segment 7 always read high regardless of
which print it came from?

The approach is to flatten each structure's segment statistics into rows
of a tidy ("long-format") table - one row per segment per structure -
using the identifying metadata each structure carries in `structure.info`
(see `Structure.set_info`). Every new comparison across structures then
becomes a `groupby` on this table, instead of a new bespoke function per
comparison, which was the core problem with the legacy nested-dict-based
scripts (`only_calculation_new.py`, `calibration_paper.py`): every new
question required a new near-duplicate function tied to one specific dict
shape.
"""

from __future__ import annotations

from ..core.structure import Structure
from ..utils.exceptions import ProcessingError
from .segment_data import extract_segment_statistics, make_relative


def structure_to_records(
    structure: Structure,
    offset_pixels: int = 10,
    include_background: bool = True,
    id_keys: tuple[str, ...] = ("doe_id", "print_id", "channel"),
) -> list[dict]:
    """
    Flatten one structure's segment statistics into a list of dict records
    ("long format"), one row per segment (+ background if included).

    Identifying columns are pulled from `structure.info` (populate it via
    `Structure.set_info(doe_id=..., print_id=..., channel=...)` before
    calling this). Every record gets every key in `id_keys`, filled with
    `None` if that structure's `info` doesn't have it - so records from
    different structures always share the same columns and concatenate
    cleanly.

    Args:
        structure:            Segmented Structure to flatten.
        offset_pixels:        Pixels excluded from every segment/background
                              edge before computing statistics.
        include_background:   If True, include the background as an
                              additional row.
        id_keys:              Which `structure.info` keys to carry over as
                              identifying columns.

    Returns:
        List of dict records, one per segment (+ background).

    Raises:
        ValidationError: If `structure` is unsegmented (propagated from
            `extract_segment_statistics`).
    """
    stats = extract_segment_statistics(structure, offset_pixels, include_background)
    make_relative(stats)

    records = []
    for stat in stats:
        record = {key: structure.info.get(key) for key in id_keys}
        record.update(
            {
                "segment_id": stat.id,
                "is_background": stat.id < 0,
                "mean": stat.mean,
                "mean_relative": stat.mean_relative,
                "std": stat.std,
                "n_pixels": stat.n_pixels,
            }
        )
        records.append(record)
    return records


def build_ensemble_dataframe(
    structures: list[Structure],
    offset_pixels: int = 10,
    include_background: bool = True,
    id_keys: tuple[str, ...] = ("doe_id", "print_id", "channel"),
):
    """
    Build a tidy, long-format DataFrame (one row per segment measurement)
    across many structures - e.g. the same design printed several times,
    several DOEs, or several measurement channels of the same structure.

    Args:
        structures:           List of segmented Structures to combine. Set
                              identifying metadata on each beforehand via
                              `Structure.set_info(...)`.
        offset_pixels:        Pixels excluded from every segment/background
                              edge before computing statistics.
        include_background:   If True, include the background as an
                              additional row per structure.
        id_keys:              `structure.info` keys to carry over as
                              identifying columns (same set for every
                              structure, missing ones become `None`).

    Returns:
        pandas.DataFrame with columns `id_keys` + ["segment_id",
        "is_background", "mean", "mean_relative", "std", "n_pixels"].

    Raises:
        ProcessingError: If pandas is not installed.
        ValidationError: If any structure is unsegmented (propagated from
            `structure_to_records`).
    """
    try:
        import pandas as pd
    except ImportError as e:
        raise ProcessingError(
            "build_ensemble_dataframe requires pandas (pip install pandas)."
        ) from e

    all_records: list[dict] = []
    for structure in structures:
        all_records.extend(
            structure_to_records(structure, offset_pixels, include_background, id_keys)
        )
    return pd.DataFrame.from_records(all_records)


def summarize_by_group(df, group_keys: tuple[str, ...], value_col: str = "mean_relative"):
    """
    Group an ensemble DataFrame and compute mean / std / coefficient of
    variation of `value_col` within each group.

    Typical use: `summarize_by_group(df, group_keys=("segment_id",))` on a
    DataFrame built from multiple prints of the same design answers "how
    reproducible is each segment's value across prints?" - a high CV
    flags either a genuinely inconsistent print or a segment sitting near
    a systematic artifact (e.g. always the edge-most segment).

    Args:
        df:          DataFrame as returned by `build_ensemble_dataframe`.
        group_keys:  Column(s) to group by, e.g. ("segment_id",) for
                     per-segment repeatability across prints, or
                     ("doe_id", "segment_id") for per-segment repeatability
                     within each DOE separately.
        value_col:   Column to summarize. Defaults to "mean_relative"
                     (background-relative segment mean).

    Returns:
        DataFrame indexed by `group_keys` with columns "mean", "std", "cv"
        (coefficient of variation, std/|mean|, NaN where mean is 0) and
        "n" (number of rows the group was computed from).
    """
    import numpy as np

    grouped = df.groupby(list(group_keys))[value_col]
    summary = grouped.agg(mean="mean", std="std", n="count")
    summary["cv"] = np.where(
        summary["mean"] != 0, summary["std"] / summary["mean"].abs(), np.nan
    )
    return summary.reset_index()
