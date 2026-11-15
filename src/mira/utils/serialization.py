"""
Shared serialization helpers.

`to_json_safe` is the single place that knows how to turn a dataclass /
numpy-heavy result object (RegressionResult, DeltaNResult, BlandAltmanResult,
...) into plain Python types that `json.dump` can write directly - so
individual analysis modules don't each need their own ad-hoc dict-building
and numpy-to-native conversion before dumping results to disk.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np


def to_json_safe(value: Any) -> Any:
    """
    Recursively convert a value into JSON-serializable plain Python types.

    Handles (recursively, in any combination):
        - dataclass instances (via `dataclasses.asdict`)
        - dict, list, tuple
        - numpy arrays (-> list) and numpy scalar types (-> native Python)

    Any other value (str, float, int, bool, None, already-plain dict/list
    contents) is returned unchanged.

    Args:
        value: Value to convert. Typically a result dataclass (e.g.
            `DeltaNResult`) or a dict/list containing such dataclasses.

    Returns:
        A JSON-serializable equivalent of `value`.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)

    if isinstance(value, dict):
        return {key: to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):  # covers np.floating, np.integer, np.bool_, ...
        return value.item()

    return value
