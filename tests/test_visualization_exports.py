"""Every name in `mira.visualization.__all__` must exist."""

import mira.visualization as visualization


def test_all_names_exist():
    missing = [
        name for name in visualization.__all__ if not hasattr(visualization, name)
    ]
    assert missing == []
