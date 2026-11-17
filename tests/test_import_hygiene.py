"""`import mira` must not load matplotlib, OpenCV or pandas (MI-05)."""

import os
import subprocess
import sys

import pytest

HEAVY_MODULES = ("matplotlib", "cv2", "pandas")


def _modules_after(statement: str) -> set[str]:
    """Run `statement` in a fresh interpreter and return the loaded top-level modules."""
    code = (
        f"{statement}\n"
        "import sys\n"
        "print(','.join(sorted({name.split('.')[0] for name in sys.modules})))\n"
    )
    # The child gets the parent's sys.path, so that it finds `mira` through
    # pytest's `pythonpath = ["src"]` also without an editable install.
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    assert out.returncode == 0, out.stderr
    return set(out.stdout.strip().split(","))


@pytest.mark.parametrize(
    "statement",
    [
        "import mira",
        "import mira.analysis.comparison, mira.analysis.reliability",
        "import mira.analysis.topology, mira.analysis.sensitivity",
        "import mira.processing.registration, mira.processing.calibration",
    ],
)
def test_import_does_not_load_heavy_modules(statement):
    loaded = _modules_after(statement)
    assert "mira" in loaded
    assert not loaded.intersection(HEAVY_MODULES), loaded.intersection(HEAVY_MODULES)


def test_inter_linear_constant_matches_opencv():
    cv2 = pytest.importorskip("cv2")
    from mira.processor import _CV2_INTER_LINEAR

    assert _CV2_INTER_LINEAR == cv2.INTER_LINEAR


def test_ecc_motion_names_exist_in_opencv():
    cv2 = pytest.importorskip("cv2")
    from mira.processing.registration import ECCRefinement

    for name in ECCRefinement._MOTION.values():
        assert isinstance(getattr(cv2, name), int)
