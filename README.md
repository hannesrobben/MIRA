# MIRA — Microscopy Image Registration & Analysis

`mira` relates measured phase and height images of two-photon-printed micro-structures to place and to
their design. Today it is the analysis library behind the DOE publication work: structure detection,
segmentation, alignment, phase ↔ height conversion, refractive-index (Δn) calibration and statistics
(regression, Bland-Altman, ensembles over prints, ICC, sensitivity, error vs. topology), with plotting kept
separate from computation.

In the real-time feedback 2PP system it takes over four more tasks (planned, see `TODO.md`):

1. **Registration** of DHM products by stage position, refined in the image.
2. **Speckle fusion** of captures of one structure at shifted positions.
3. **Stitching** of a raster of products into a mosaic larger than the field of view.
4. **Target comparison** of a product with the target structure, returned as a `DeviationReport`.

Hologram reconstruction, hardware, network and stage control are out of scope.

## Place in the system

MIRA is one of four cooperating repositories (plan: `docs/system/SOFTWARE_PLAN_OVERVIEW.md`):

| Package | Owns |
|---|---|
| `offaxisholo` | Hologram reconstruction → phase, intensity, complex field. |
| `dhmconnect` | DHM protocol and service; calls MIRA's fusion / stitching plug-in on the DHM-PC. |
| `nanofactorysystem` | Orchestration and layer targets; calls MIRA's comparison and stores the report. |
| `mira` | Registration, fusion, stitching, target comparison, statistical analysis. |

`mira` never imports `dhmconnect` or `nanofactorysystem`, and `offaxisholo` only optionally; its test suite
runs without the other three packages.

## Installation

Python 3.10 or newer (the project baseline is open decision M9; development uses 3.12 and NumPy 2).

```bash
python -m pip install -e .               # core: computation without plotting and pandas
python -m pip install -e ".[viz]"        # + matplotlib (mira.visualization, plot_* methods)
python -m pip install -e ".[analysis]"   # + pandas (ensemble tables, ICC from a DataFrame, CSV files)
python -m pip install -e ".[dev]"        # everything above + pytest, black, numpydoc, pingouin
```

`import mira` loads neither matplotlib, OpenCV nor pandas; they are imported when a function needs them.

## Package layout

```
src/mira/
├── processor.py            StructureProcessor: facade with method chaining
├── core/                   ImageData, Structure (info / statistics dicts), Segment, Background
├── processing/
│   ├── detection.py        structure detection (threshold, intensity, adaptive)
│   ├── segmentation.py     grid, k-means and watershed segmentation
│   ├── alignment.py        alignment of two processors (rotation, scale, translation)
│   ├── registration.py     content-based refiners: NoRefinement, ECCRefinement, PhaseCorrelationRefinement
│   ├── phase_retrieval.py  phase ↔ height, unwrapping, background adjustment
│   └── calibration.py      Δn fit (calculate_delta_n → DeltaNResult)
├── analysis/               computation only, results as dataclasses
│   ├── segment_data.py     shared segment statistics (edge erosion, background-relative values)
│   ├── comparison.py       linear / segment regression, Bland-Altman
│   ├── ensemble.py         records and DataFrames over prints, summary per group
│   ├── reliability.py      ICC(1,1), ICC(2,1), ICC(3,1) with CI and F-test
│   ├── sensitivity.py      Δn uncertainty propagation and sweep
│   └── topology.py         error vs. gradient and curvature (pixel size in metres)
├── visualization/          plotting only: plotter_2d, plotter_3d, plotter_stats
└── utils/                  exceptions, helpers, JSON serialization
```

Files named `*_old*.py` and the Seafile conflict copy are legacy and not used; they move to `attic/` after
review (`docs/ATTIC_REVIEW.md`).

## Usage

### StructureProcessor

Every geometric change of a processor goes through `apply_affine_transform`, which transforms the raw
channels, the structure mask and all segments together.

```python
import numpy as np
from mira import StructureProcessor

phase = np.load("insitu_phase.npy")       # rad
height = np.load("sensofar_height.npy")   # m

dhm = (
    StructureProcessor(phase_data=phase, pixel_size=(0.25e-6, 0.25e-6))
    .detect_structure()
    .segment_structure(method="grid", num_segments=(3, 3))
)
reference = (
    StructureProcessor(height_data=height, pixel_size=(0.25e-6, 0.25e-6))
    .detect_structure()
    .segment_structure(method="grid", num_segments=(3, 3))
)

delta_n = dhm.calculate_delta_n(reference, offset_pixels=10)       # stored in dhm.structure.statistics
regression = dhm.fit_segment_regression(reference, offset_pixels=10)
print(delta_n.delta_n, regression.r_squared)

dhm.structure.set_info(doe_id="DOE3", print_id="P01")              # ids for ensembles and ICC
dhm.save_snapshot("dhm_P01.npz")                                    # npz with JSON metadata
restored = StructureProcessor.load_snapshot("dhm_P01.npz")
```

After `load_snapshot`, `structure.statistics` holds plain JSON dicts; rebuild a result explicitly, e.g.
`DeltaNResult(**restored.structure.statistics["delta_n_segment_mean"])`.

### Analyses: compute, then plot

Each analysis is a compute function in `mira.analysis` that returns a dataclass, and a plot function in
`mira.visualization.plotter_stats` that takes that dataclass.

```python
from mira.analysis.comparison import bland_altman_compute
from mira.analysis.reliability import compute_icc, interpret_icc
from mira.visualization.plotter_stats import plot_bland_altman

result = bland_altman_compute(predicted_height, measured_height)   # masked arrays keep their mask
fig = plot_bland_altman(result, label_1="DHM", label_2="Sensofar", units="µm")

icc = compute_icc(table, icc_type="ICC(2,1)")                       # table: segments x prints, complete
print(icc.value, icc.ci_lower, icc.ci_upper, interpret_icc(icc.value))
```

`compute_icc` is checked against `pingouin.intraclass_corr` (see `tests/test_reliability_reference.py`).

## Conventions

- Units in names where new code is written (`_um`, `_rad`); the existing API uses metres for heights,
  wavelengths and pixel sizes.
- Image axis 0 is y, axis 1 is x.
- Computation (`mira.analysis`, `mira.processing`) and plotting (`mira.visualization`) stay separate.
- Errors are reported with the exceptions in `mira.utils.exceptions` (`ValidationError`,
  `ProcessingError`, `SegmentationError`, `AlignmentError`, `VisualizationError`, …) rather than silent
  fallbacks.

## Development

```bash
python -m pytest        # tests in tests/, headless (matplotlib backend Agg)
black src tests         # formatting, line length 88
```

Project files: `CLAUDE.md` (architecture and rules), `TODO.md` (task list, phase plan, open decisions),
`WORKLOG.md` (work log), `docs/` (system plan, notes, reviews).

## License

See `LICENSE`.
