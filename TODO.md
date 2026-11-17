# TODO

<!--
Format rules (for humans and Claude):
- ID: MI-<number>, never reused. Follow-ups reference their origin: "(follow-up of MI-20)" or
  "(found during MI-20)". Number blocks: 0x baseline, 1x products and geometry, 2x registration,
  3x fusion, 4x stitching, 5x comparison, 6x integration, 7x analysis backlog.
- Phase: every open todo carries "[phase: <n>]" (see "Phase plan").
- Priority: high | medium | low
- Size: S (hours) | M (about a day) | L (several days)
- Depends on: IDs or "–". IDs of the other repositories are allowed: K…/L…/P… (OffAxisHolo),
  DL-… (DHMConnect), T…/F… (NanoFactorySystem_automated); G… are the global decisions in
  docs/system/SOFTWARE_PLAN_OVERVIEW.md, M… the decisions of this repository (table below).
- Done when: one line or sub-bullets; every criterion must be verifiable.
- "Decision:" marks a question to the maintainer (working rule 3).
- Sort "Open" by phase, then by blockers/dependencies, then priority.
- "Done" is a staging area: the status review routine moves entries to ARCHIVE.md.
-->

## Phase plan

Target state: MIRA turns DHM products into registered, fused and stitched products and compares them with
the target structure (`DeviationReport` for nanofactorysystem), while the existing publication analysis
keeps its results and becomes tested.

| Phase | Topic | Todos | Gate before the dependent todos start |
|---|---|---|---|
| 0 | Baseline: clean working state, packaging, tests, import hygiene | MI-01, MI-02, MI-04, MI-05, MI-06, MI-07, MI-08 | MI-04 regression exists before non-trivial refactoring |
| 1 | Products and geometry | MI-10, MI-11 | G2 for the final product schema |
| 2 | Registration | MI-22, MI-20, MI-21 | – |
| 3 | Speckle fusion | MI-30, MI-31 | M3, M4 for the default method |
| 4 | Stitching | MI-40, MI-41, MI-42 | – |
| 5 | Target comparison | MI-50, MI-51, MI-52, MI-53 | M5, M6 (G4, G7) |
| 6 | Integration | MI-60, MI-62, MI-61 | G5; MI-61 only after profiling |
| A | Analysis backlog of the DOE session (publication work) | MI-70 … MI-80 | M11 for MI-79 |

Suggested order: MI-01 → MI-02, MI-04, MI-05, MI-06 → MI-10 → MI-22 → MI-50, MI-51, MI-52, MI-53 (needed
first, for the operating modes `monitor` and `shadow` of nanofactorysystem) → MI-11, MI-20, MI-21 → MI-30,
MI-31 → MI-40, MI-41, MI-42 → MI-60, MI-62 → MI-61 if needed. Phase A runs whenever the maintainer needs
it for a paper; MI-76 (ICC check) before any ICC number is published.

### Working rules for Claude Code (in addition to CLAUDE.md)
1. Follow the suggested order; take the first open todo whose dependencies are done. Phase A todos only
   when the maintainer names one, or when nothing else is available.
2. Design questions end with a short document in `docs/` and the line `Status: approved <date>` written
   by the maintainer before dependent work starts.
3. If a todo leaves a decision open ("Decision:" or a reference to the table below), propose an answer
   and ask the maintainer before implementing the part that depends on it.
4. The results of the publication analysis must not change (rule 7 of CLAUDE.md).
5. At the end of each phase run `python -m pytest` and append a WORKLOG.md entry
   "[Phase <n>] Phase summary" with the finished todos, open follow-ups and what the maintainer has to
   check with real data.

## Open decisions (maintainer)

| ID | Decision | Recommendation | Blocks |
|---|---|---|---|
| M1 | What happens to the `*_old*` files and the Seafile conflict copy (the working state is committed since `254b25e`). | Move them to `attic/` after review of the list (MI-01). | MI-01 |
| M2 | Data model for products: extend `StructureProcessor`, or add a light `Product` type with an adapter. | New `Product` type; `StructureProcessor.from_product()` as bridge. | MI-10 |
| M3 | Averaging domain for speckle fusion: complex field, intensity, or phase (open in F5). | Complex field after equalising offset and tilt; intensity separately. Confirm with data. | MI-30 |
| M4 | Scope of "speckle noise optimisation": fusion of shifted captures only, or also weighting, selection and the choice of the shift pattern. | Fusion first; the pattern is planned by nanofactorysystem (T85). | MI-30 |
| M5 | Comparison domain: expected phase computed from the target, or measured height from the phase. | Phase domain; height as derived output. Depends on G4 and G7. | MI-51 |
| M6 | Who rasterises the target: nanofactorysystem delivers a raster, or MIRA rasterises a height map or outline onto the product grid. | nanofactorysystem delivers a height map in µm; MIRA resamples it. | MI-50 |
| M7 | Acceptance tolerances for registration and comparison. | Set after the noise floor is measured; until then tests use synthetic data with stated tolerances. | MI-22, MI-52 |
| M8 | GPU use on the DHM-PC (G5). | CPU first; port single kernels only after profiling. | MI-61 |
| M9 | Python and NumPy baseline (`requires-python >= 3.10` today). | Python 3.12, NumPy 2, as the other packages (G9). | MI-02 |
| M10 | ICC type for the paper statement: ICC(2,1) (prints of this design are reproducible in general) or ICC(3,1) (statement about exactly these N prints). | – | the ICC number in the paper (the check of all three forms, MI-76, is done) |
| M11 | `ECCRefinement` returns the inverse warp (MI-79). Was `align_with(..., refine="ecc")` or `refine=True` used for published results? | Fix it (invert the matrix); if ECC was used for the paper, re-run those numbers and compare first. | MI-79 |

## Open

### Phase 0 — Baseline

- [ ] MI-07: Python and NumPy baseline (follow-up of MI-02) [phase: 0]
  - Priority: medium
  - Size: S
  - Depends on: M9
  - Done when: `requires-python`, the classifiers and black's `target-version` follow M9 (recommended:
    Python 3.12, NumPy 2); a fresh install of `.[dev]` passes `python -m pytest`.

- [ ] MI-08: Move the approved legacy files to the attic (follow-up of MI-01) [phase: 0]
  - Priority: high
  - Size: S
  - Depends on: M1 (the maintainer approves the list in `docs/ATTIC_REVIEW.md`, completely or in part)
  - Done when: the approved files are moved with `git mv` to `attic/` keeping their relative path (one
    commit), nothing is deleted, `python -m pytest` passes, CLAUDE.md names the attic.

### Phase 1 — Products and geometry

- [ ] MI-10: Product data model [phase: 1]
  - Priority: high
  - Size: M
  - Depends on: M2, G2 (final schema)
  - Done when:
    - `Product` holds complex field or phase plus intensity, pixel size (y, x) in µm, stage position, z,
      wavelength, medium, quality, ids and material snapshot;
    - constructors from a `ProductSet` message, from an OffAxisHolo `ReconstructionResult` (optional
      import) and from `experiment.h5` (`h5py` only); a bridge to `StructureProcessor`;
    - a round trip against the shared example messages (vendored under `tests/contracts/`) passes.

- [ ] MI-11: Stage-to-image geometry [phase: 1]
  - Priority: medium
  - Size: M
  - Depends on: MI-10
  - Done when:
    - a model maps stage micrometres to product pixels (rotation, scale, axis flips, offset) per
      objective; it is estimated from captures with known shifts and saved / loaded;
    - `tools/Transform` of nanofactorysystem (camera case) is reviewed for reuse;
    - partial result allowed: model and estimation on synthetic data; the real calibration needs captures.

### Phase 2 — Registration

- [ ] MI-22: Synthetic test fields [phase: 2]
  - Priority: high
  - Size: S
  - Depends on: MI-10
  - Done when:
    - a generator produces fields with known shift, piston, tilt and speckle-like noise; used by MI-20,
      MI-21, MI-30, MI-40; final tolerances follow M7.

- [ ] MI-20: Registration by position and image [phase: 2]
  - Priority: high
  - Size: L
  - Depends on: MI-11, MI-22
  - Done when:
    - coarse placement from positions and MI-11, then sub-pixel refinement restricted to the overlap
      (reusing the refiners in `processing/registration.py`);
    - returns transform and a confidence value; low confidence falls back to the stage position and is
      flagged; tested on MI-22 fields.

- [ ] MI-21: Phase offset and tilt between products [phase: 2]
  - Priority: high
  - Size: M
  - Depends on: MI-22
  - Done when: a robust estimate of piston and linear phase terms from the overlap of two products is
    tested on MI-22 fields.

### Phase 3 — Speckle fusion

- [ ] MI-30: Fusion of a speckle set [phase: 3]
  - Priority: high
  - Size: L
  - Depends on: MI-20, MI-21; M3, M4
  - Done when:
    - `fuse(products, method=...)` registers, equalises and averages; returns the fused product, the
      transform of every member and the noise in the reference region before and after;
    - partial result allowed: complex and intensity averaging implemented, default left open.

- [ ] MI-31: Noise metrics [phase: 3]
  - Priority: medium
  - Size: S
  - Depends on: –
  - Done when: shared functions for the noise in a reference region and its spectrum (reusing the
    ensemble statistics and PSD tools, see MI-75) exist, so that fusion, the N-study (OffAxisHolo K13)
    and the comparison report the same number.

### Phase 4 — Stitching

- [ ] MI-40: Stitching [phase: 4]
  - Priority: high
  - Size: L
  - Depends on: MI-20, MI-21
  - Done when:
    - pairwise registration in the overlaps, a global least-squares adjustment of positions, phase
      offsets and tilts over the whole raster, and blending with weights that fall off towards the tile
      edge produce mosaic, tiles and transforms;
    - a tile may itself be a fused speckle set; tested on a synthetic raster.

- [ ] MI-41: Stage-to-camera calibration from stitching [phase: 4]
  - Priority: low
  - Size: S
  - Depends on: MI-40, MI-11
  - Done when: the residual shifts of MI-40 update the geometry model of MI-11 (rotation, scale).

- [ ] MI-42: Size limits [phase: 4]
  - Priority: low
  - Size: S
  - Depends on: MI-40
  - Done when: memory use for a raster of k × k tiles is documented and tested; float32 and complex64
    throughout.

### Phase 5 — Target comparison

- [ ] MI-50: Target on the product grid [phase: 5]
  - Priority: high
  - Size: M
  - Depends on: MI-10; M6
  - Done when: a `LayerTarget` (height map and mask in µm, position) is resampled onto the pixel grid of
    a product using MI-11 (or the stage position until MI-11 exists) and the detected pose of the
    structure (existing detection and alignment).

- [ ] MI-51: Forward model [phase: 5]
  - Priority: high
  - Size: M
  - Depends on: MI-50; M5, G4, G7
  - Done when: the expected phase is computed from target height, wavelength and the index difference
    of the material state, with the inverse for the height output; `calculate_delta_n` is reused for
    calibration. Partial result allowed with the index difference as explicit input.

- [ ] MI-52: Comparison [phase: 5]
  - Priority: high
  - Size: M
  - Depends on: MI-51; M7
  - Done when: `compare(product, target, *, sigma, k) -> DeviationReport` returns the deviation map,
    RMSE, MAE, statistics per segment, the share of the area above k·sigma, the classification too low /
    too high, and validity flags (registration confidence, coverage, product quality); tested on
    synthetic cases.

- [ ] MI-53: Stable entry point and report format [phase: 5]
  - Priority: high
  - Size: M
  - Depends on: MI-52
  - Done when: `mira.api.evaluate_layer(product, target, context, settings) -> DeviationReport` exists;
    the report is a header dictionary plus arrays with `schema_version` and the computing time; a test
    validates it against the shared example. This is what nanofactorysystem T90 calls.

### Phase 6 — Integration

- [ ] MI-60: Plug-in for the DHM service [phase: 6]
  - Priority: medium
  - Size: M
  - Depends on: MI-30, MI-40; G5
  - Done when: `mira.plugins.dhmconnect` offers `fuse_group(members, params)` and
    `stitch_group(members, params)` returning a `GroupProduct`; pure functions, no network; called by
    DHMConnect DL-B9.

- [ ] MI-62: Offline use on experiment files [phase: 6]
  - Priority: medium
  - Size: M
  - Depends on: MI-10, MI-30, MI-40, MI-53
  - Done when: fusion, stitching and comparison run on `experiment.h5` and the raw archive, and write
    results under `/analysis` through the access layer of nanofactorysystem (`ExperimentFile`, optional
    dependency).

- [ ] MI-61: GPU kernels [phase: 6]
  - Priority: low
  - Size: M
  - Depends on: a profile that shows the need; M8
  - Done when: only the kernels the profile names are ported, with parity tests against the CPU path.

### Phase A — Analysis backlog of the DOE session (from `docs/notes/*.txt`)

- [ ] MI-70: Integration test of the DOE analysis chain with real data [phase: A]
  - Priority: medium
  - Size: M
  - Depends on: MI-04
  - Done when:
    - one test (or documented script run) goes through load → segment → `calculate_delta_n` →
      `fit_segment_regression` → `analyze_error_vs_topology` → `sweep_delta_n_sensitivity` →
      `save_snapshot` → `load_snapshot` with real in-situ / Sensofar data and checks that everything is
      restored;
    - R² of Δn fit and regression is plausible (> 0.9 expected; otherwise segmentation or
      `edge_offset_px` is reported);
    - an existing old `.npz` snapshot loads without exception (backward compatibility);
    - the usage rules are in the docstrings: `structure.set_info(doe_id=..., print_id=...)` on every
      structure before `build_ensemble_dataframe` / `compute_icc_from_dataframe`; statistics come back
      as plain dicts after loading.

- [ ] MI-71: Merge the duplicate segmentation implementations [phase: A]
  - Priority: medium
  - Size: M
  - Depends on: MI-04
  - Done when: `Structure.segment()` and `processing.segmentation.segment_structure()` share one
    implementation (like `analysis/segment_data.py`), so that the grid shape is written into the `info`
    dict on both paths; regression unchanged.

- [ ] MI-74: Zernike decomposition of error maps [phase: A]
  - Priority: low
  - Size: M
  - Depends on: OffAxisHolo P15 (final function signature)
  - Done when: `mira/analysis/wavefront.py` is a thin wrapper around the OffAxisHolo function: takes a
    `Structure` / `ImageData` and an aperture mask, maps the structure onto the unit disc, returns a
    `ZernikeResult` (named coefficients such as `defocus`, `astigmatism_0`, `coma_x`, share of variance
    per mode, residual RMS after N modes), optionally stores it in `structure.statistics`; a bar-chart
    plot with mode names is in `plotter_stats.py`; tests on synthetic data. The Zernike math itself is
    not duplicated here.

- [ ] MI-75: Radial PSD of error and height maps [phase: A]
  - Priority: low
  - Size: S
  - Depends on: OffAxisHolo P15
  - Done when: `mira/analysis/spectral.py` wraps the OffAxisHolo function and returns a `PSDResult`
    (radial frequency axis in 1/µm, mean power per bin, optional half-power frequency); a log-log plot
    with optional reference slopes is in `plotter_stats.py`; tests on synthetic data; MI-31 uses it.

- [ ] MI-79: Invert the warp of `ECCRefinement` (found during MI-05) [phase: A]
  - Priority: high
  - Size: S
  - Depends on: M11
  - Done when: `ECCRefinement.estimate` returns the matrix that maps `moving` onto `fixed` for
    `cv2.warpAffine(moving, M)`, as `RegistrationRefiner` documents (today it returns the inverse, the
    `cv2.findTransformECC` convention, so a residual shift doubles instead of vanishing); the strict
    xfail `test_ecc_refinement_maps_moving_onto_fixed` in `tests/test_processor_smoke.py` passes without
    the marker; published numbers that used ECC are re-run and compared (M11). MI-20 reuses the refiners.

- [ ] MI-80: Detection and corners of multi-level structures (found during MI-73) [phase: A]
  - Priority: low
  - Size: S
  - Depends on: –
  - Done when: it is checked whether `detect_structure()` (default threshold) splits a structure whose
    segments have very different heights (synthetic case: 2 x 2 quadrants with levels 1, 2, 3, 4 on a
    background of 0 gave six "corner points" and an almost empty first segment); the result is either
    a fix with a test or a documented limit with a validation error instead of empty segments.

## In Progress
<!-- Claude Code moves a todo here when starting work. -->
- [ ] MI-01: Clean up the legacy copies [phase: 0]
  - Priority: high
  - Size: S
  - Depends on: M1
  - Done when:
    - the maintainer has seen a diff of the conflict copy `visualization/plotter_2d (SFConflict …).py`
      against `plotter_2d.py` and the list of legacy files (`core/image_old.py`, `core/structure_old.py`,
      `processing/alignment_old*.py`, `processing/detection_old.py`, `processing/phase_retrieval_old*.py`,
      `processor_old*.py`, `processor_old^1.py`, `visualization/plotter_2d_old*.py`,
      `visualization/bland_altman_snippet.py` if superseded by `bland_altman.py`);
    - approved files are moved with `git mv` to `attic/` (one commit), nothing is deleted;
    - the two German session notes in `src/mira/` move to `docs/notes/` (they are not package data).
  - Progress 2026-11-17: the session notes are in `docs/notes/` and `docs/ATTIC_REVIEW.md` holds the
    conflict-copy diff (one docstring, no code) and the legacy list (`ebfc2b0`). Remaining: the move
    itself after M1, see MI-08.
  - Note: the working state itself was committed by the maintainer on 2026-11-15 (`254b25e`), which
    covers the first part of the original MI-01.

- [ ] MI-02: Packaging [phase: 0]
  - Priority: high
  - Size: S
  - Depends on: M9
  - Done when:
    - the conflict between static and dynamic dependencies is removed (no `requirements.txt` reference);
      author and description are set; extras `viz` (matplotlib), `analysis` (pandas), `dev`;
    - pandas is declared (needed by `analysis/ensemble.py` and `analysis/reliability.py`);
    - an editable install works in a fresh environment and `python -m pytest` runs there.
  - Progress 2026-11-17: everything except the Python / NumPy baseline (`4280aec`; pingouin in `dev`
    with `91c2cd5`); checked in fresh environments with `.[dev]` and with the core only. Remaining:
    MI-07 (needs M9).


## Blocked
<!-- Format: todo as above, plus the line "Blocked by: <reason or ID>". -->
- [ ] MI-04: Test baseline [phase: 0]
  - Priority: high
  - Size: M
  - Depends on: the maintainer names where the test suite of the DOE3 work lives
  - Done when:
    - that suite is in `tests/` and runs; a small regression on stored snapshots covers the analysis
      results of the publication;
    - `python -m pytest` runs and its result is recorded as baseline in `WORKLOG.md`.
  - Blocked by: the maintainer has to name the location of the test suite of the DOE3 work (it is not
    in this folder).

## Done
<!-- Claude Code adds: - [x] MI-<n>: title — YYYY-MM-DD — 1–2 sentences on what changed — commits: `<sha>`, … -->
- [x] DOE analysis extensions (before this list existed) — 2026-11-15 committed — Bland-Altman, Δn calibration, shared segment statistics, regression, `info` / `statistics` dicts with snapshot persistence, ensembles, error vs. topology, Δn sensitivity, ICC (details in `docs/notes/zusammenfassung_aenderungen_und_tests.txt`). — commits: `254b25e`
- [x] MI-03: Project files — 2026-11-15 — `CLAUDE.md`, `TODO.md`, `WORKLOG.md` and `docs/system/SOFTWARE_PLAN_OVERVIEW.md` created; the README update is split off as MI-06. Committed 2026-11-17 on branch `docs/project-files`. — commits: `b819b2c`, `bf079be`, `fe99316`
- [x] MI-05: Import hygiene — 2026-11-17 — OpenCV, matplotlib and pandas are imported where used; a subprocess test checks `sys.modules` after `import mira`, and smoke tests run the lazy paths (they found MI-79). — commits: `e3adbbd`
- [x] MI-77: Audit the masked-array unmasking pattern — 2026-11-17 — all places checked (list in commit `a7696ae`); `fit_linear_regression` leaves masked pairs out, `bland_altman.py` is replaced by MI-72, `compute_icc` rejects masked cells (MI-76); tests in `tests/test_masked_inputs.py`. — commits: `a7696ae`, `4d88116`, `91c2cd5`
- [x] MI-72: Split Bland-Altman into compute and plot — 2026-11-17 — `bland_altman_compute()` → `BlandAltmanResult` and `plot_bland_altman(result)`; results equal the former exported function (recorded values, rel 1e-12); the former call stays in `visualization/bland_altman.py` (BREAKING CHANGE of `mira.visualization.plot_bland_altman`); a missing comma in `visualization.__all__` was fixed on the way. — commits: `a37b2d3`, `4d88116`
- [x] MI-78: Pixel-size units of the topology analysis — 2026-11-17 — the pixel size is validated as metres (finite, positive, at most 1e-3 m), so a size in µm raises instead of scaling gradient and curvature silently. — commits: `23948fc`
- [x] MI-73: Processor method for the segment regression — 2026-11-17 — `StructureProcessor.fit_segment_regression(reference, ...)` fits and stores the result with `add_statistic()` (key `segment_regression`). — commits: `374c734`
- [x] MI-76: Check the ICC implementation against a reference — 2026-11-17 — all three forms agree with `pingouin.intraclass_corr` (rel < 3e-14, incl. the ICC(2,1) CI) and reproduce the Shrout & Fleiss example; p-value via `f.sf`; incomplete tables raise; references in the docstring. The form used in the paper is still M10. — commits: `91c2cd5`
- [x] MI-06: README for the real package — 2026-11-17 — the README describes `mira`, its layout, installation with extras, StructureProcessor, the compute-then-plot analyses and the role in the four-repository system. — commits: `f212d59`
- [x] MI-81: Translate the remaining German documentation in active code — 2026-11-17 — `ImageData.save_compressed` (docstring now NumPy style) and the comments in `StructureProcessor.phase_retrieval` are in English; legacy copies and `docs/notes/*.txt` stay German on purpose (found during the cross-repository review of 2026-11-17). — commits: `076c6af`
