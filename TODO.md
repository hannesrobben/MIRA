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

Target state (Release V1.0, decision session 2026-11-17/18, master plan v0.4): MIRA runs on **both PCs**.
On the DHM-PC, as a plug-in in worker processes of the DHMConnect service, it registers, unwraps per tile,
levels, fuses and stitches the tiles of one structure and layer into a `LayerProduct`; on the Work-PC it
compares the `LayerProduct` with the target (`evaluate_layer` → `DeviationReport` for nanofactorysystem).
The existing publication analysis keeps its results and becomes tested (MI-04).

| Phase | Topic | Todos | Gate before the dependent todos start |
|---|---|---|---|
| 0 | Baseline: clean working state, packaging, tests, import hygiene | MI-01, MI-08 (done), MI-02, MI-04, MI-07 | MI-04 regression exists before non-trivial refactoring |
| 1 | Products and geometry | MI-10, MI-11 | DHMConnect DL-A3 for the final message schema |
| 2 | Registration | MI-22, MI-20, MI-21 | – |
| 3 | Speckle fusion | MI-30, MI-31 | – (M3, M4 decided 2026-11-18) |
| 4 | Stitching | MI-40, MI-41, MI-42 | – (G10 SA6 decided 2026-11-18) |
| 5 | Target comparison | MI-50, MI-51, MI-52, MI-53 | – (M5, M6 decided); thresholds wait for M7 |
| 6 | Integration | MI-60, MI-62, MI-61 | MI-61 only after profiling (after V1.0) |
| A | Analysis backlog of the DOE session (publication work) | MI-70 … MI-82 | MI-04 for MI-79 (M11) and MI-82 |

Suggested order: MI-07 → MI-04 → MI-10 → MI-22 → MI-50, MI-51, MI-52, MI-53 (needed first, for the
operating modes `monitor` and `shadow` of nanofactorysystem) → MI-11, MI-20, MI-21 → MI-30, MI-31 →
MI-40, MI-41, MI-42 → MI-60, MI-62 → MI-61 after V1.0. Phase A runs whenever the maintainer needs it for a
paper; MI-76 (ICC check, done) before any ICC number is published.

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

| ID | Decision | When / by what | Blocks |
|---|---|---|---|
| M7 | Acceptance tolerances for registration and comparison. | **Deferred 2026-11-18:** after a sufficient data base (noise floor, effect of the speckle reduction; study M4 with OffAxisHolo K13 and MI-31). Until then synthetic tests with tolerances in the test only; monitor thresholds stay empty. | final tolerances of MI-22, MI-52 |
| M10 | ICC type for the paper statement: ICC(2,1) or ICC(3,1). | **Deferred 2026-11-18** to MI-04, with the paper. | the ICC number in the paper |
| M11 | Was `align_with(..., refine="ecc")` or `refine=True` used for published results? | **Deferred 2026-11-18** to MI-04, with the paper. Recommendation unchanged: fix it; if ECC was used, re-run and compare first. | MI-79 |

## Decided

| ID | Date | Result |
|---|---|---|
| M1 | 2026-11-18 | `docs/ATTIC_REVIEW.md` approved; executed with the attic procedure X-B (local attic, `docs/ATTIC_INDEX.md`); `bland_altman.py` stays; `MIRA_composition.py` stays until MI-10. |
| M2 | 2026-11-18 | New immutable, backend-neutral (NumPy / CuPy) types `TileProduct` (one capture: complex64 field, grid, actual position, quality, provenance) and `LayerProduct` (mosaic: unwrapped phase + `intensity_rel` float32, optional field, validity mask, tile list with transforms and anchors); units in names, `(y, x)` in µm (X-G); `from_message` / `to_message`; complex field as a regular input; the bridge `StructureProcessor.from_product()` converts µm → m and `(y, x)` → `(x, y)` in exactly one place; `StructureProcessor` unchanged. |
| M3 | 2026-11-18 | Fusion in the complex field; registration with stage prior + sub-pixel phase correlation; piston / tilt with the same complex overlap fit as stitching (one implementation, MI-21); implicit amplitude weighting, optional quality weight per capture; count / validity per pixel (`min_count`). |
| M4 | 2026-11-18 | One plan generator in nanofactorysystem (T85), **one algorithm here**: fusion (MI-30) and stitching (MI-40) are two configurations of one pipeline. Rule: at least 9 positions per structure; jitter for large structures. V1.0 scope: fusion + weighting + planned pattern; later selection and adaptive pattern. Study with K13 / MI-31: σ² ≈ σ²_t / (N_frames · N_pos) + σ²_coh / N_pos. |
| M5 | 2026-11-18 | = OM4 of nanofactorysystem: flexible comparison domain (phase preferred, else height; can be fixed); the report states its `domain`. |
| M6 | 2026-11-18 | nanofactorysystem rasterises at `pixel_um = min(half hatch distance, product pitch)`; MIRA only shifts / transforms and downsamples (area-weighted), never upsamples; exact masks from the footprint polygons; later source `design_heightmap`. |
| M8 | 2026-11-18 | CPU first on the DHM-PC (32 threads, worker processes); GPU after profiling, after V1.0. |
| M9 (G9) | 2026-11-18 | `requires-python >= 3.12`, NumPy 2; tests on 3.12 and 3.14 (MI-07). |
| G4 / G7 | 2026-11-18 | The phase ↔ height formulas are owned by OffAxisHolo `fieldops` (K20); MIRA imports them optionally or mirrors them with a parity test. No silent defaults for indices; publication scripts set n = 1.52 explicitly (MI-82). |
| G5-new / N3 | 2026-11-18 | MIRA runs as a plug-in on the DHM-PC (registration, fusion, stitching, unwrapping per tile, detection, levelling; SA5) and on the Work-PC (`evaluate_layer`, G6; offline analysis). |
| G10 SA6 | 2026-11-18 | Stitching algorithm of the maintainer (MI-40). |
| MI-04 | 2026-11-18 | No DOE3 test suite exists → a **new baseline** from the publication scripts on stored data (MI-04). |

## Open

### Phase 0 — Baseline

- [ ] MI-07: Python and NumPy baseline (follow-up of MI-02) [phase: 0]
  - Priority: medium
  - Size: S
  - Depends on: – (M9 decided 2026-11-18)
  - Done when: `requires-python >= 3.12`, classifiers 3.12 and 3.14, black's `target-version` and NumPy 2
    follow M9; a fresh install of `.[dev]` passes `python -m pytest` on Python 3.12 and 3.14 (versions
    and date in `WORKLOG.md`).

- [ ] MI-04: Test baseline from the publication scripts (new baseline) [phase: 0]
  - Priority: high
  - Size: L
  - Depends on: the maintainer's answer on the data (asked when the work starts)
  - Done when:
    - **first** the maintainer is asked which stored data to use, together with a proposal for the
      implementation and the snapshot data format; the old experiment directory does not follow today's
      storage layout, so a mapping is part of the proposal; the maintainer provides the paper or
      information about it;
    - the publication scripts run with the current `main` on those data; key numbers (Δn, slope / R²,
      Bland-Altman, ICC, sensitivity) and one or two snapshots are frozen as a regression test with
      stated tolerances; the numbers are compared once by hand with the paper;
    - M10 (ICC form) and M11 (ECC used?) are answered with the paper and recorded in "Decided";
    - `python -m pytest` runs and its result is recorded as baseline in `WORKLOG.md`.
  - Note: reviewing the data probably needs several follow-up questions to the maintainer.

### Phase 1 — Products and geometry

- [ ] MI-10: Product data model `TileProduct` / `LayerProduct` (M2) [phase: 1]
  - Priority: high
  - Size: L
  - Depends on: DHMConnect DL-A3 (final schema)
  - Done when:
    - immutable, backend-neutral (NumPy / CuPy) types `TileProduct` (complex64 field, grid with
      `pixel_um` `(y, x)` and `origin_um`, actual stage position, z with `propagation_index`, wavelength,
      quality, ids, material snapshot, provenance) and `LayerProduct` (unwrapped phase + `intensity_rel`
      float32, optional complex field, validity mask, grid, tile list with transforms, piston / tilt,
      anchor, residual, discarded border band, quality per tile);
    - the complex field is a regular input (today MIRA has no complex-field input);
    - `from_message` / `to_message` against the shared example messages (vendored under
      `tests/contracts/`), constructors from an OffAxisHolo `ReconstructionResult` (optional import) and
      from `experiment.h5` (`h5py` only);
    - the bridge `StructureProcessor.from_product()` converts µm → m and `(y, x)` → `(x, y)` in exactly one
      place; `StructureProcessor` is unchanged;
    - `MIRA_composition.py` is reviewed while designing and then moved to the attic (X-B).
  - Contracts available 2026-11-18: DHMConnect `fc85f5c` (DL-A3): `ProductSet`, `LayerProduct`,
    `LayerTarget`, `DeviationReport` with examples (`DHMConnect/docs/CONTRACTS.md`); vendor with
    `python -m dhmconnect.schemas.export tests/contracts --commit fc85f5c`.

- [ ] MI-11: Stage-to-image geometry [phase: 1]
  - Priority: medium
  - Size: M
  - Depends on: MI-10
  - Done when:
    - a model maps stage micrometres to product pixels (rotation, scale, shear, axis flips, offset) per
      objective and per camera / DHM; it loads the versioned calibration file of nanofactorysystem T104
      (id, date, residuals) and falls back to rotation 0 and the nominal pixel size with a generous
      tolerance when none exists;
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
    - coarse placement from the stage position (prior) and MI-11, then sub-pixel phase correlation
      restricted to the overlap and to the pose tolerance (`pose_tolerance_um`), reusing the refiners in
      `processing/registration.py`;
    - returns transform and a confidence value; low confidence falls back to the stage position and is
      flagged; tested on MI-22 fields.

- [ ] MI-21: Phase offset and tilt between products [phase: 2]
  - Priority: high
  - Size: M
  - Depends on: MI-22
  - Done when: a robust estimate of piston (angle of Σ U_new · conj(U_ref) over the complex overlap) and
    linear phase terms from the overlap of two products is tested on MI-22 fields; it is the **one**
    implementation used by fusion (MI-30) and stitching (MI-40).

### Phase 3 — Speckle fusion

- [ ] MI-30: Fusion of a speckle set (M3, M4) [phase: 3]
  - Priority: high
  - Size: L
  - Depends on: MI-20, MI-21
  - Done when:
    - the pipeline shared with MI-40 registers, equalises piston / tilt (MI-21) and averages in the
      **complex field** with implicit amplitude weighting and an optional quality weight per capture;
      count and validity per pixel (`min_count`); returns the fused `LayerProduct`, the transform of every
      member and the noise in the reference region before and after;
    - a configuration for small structures (≥ 9 positions around the centre, `single_shot` allowed) and
      a comparison of the phase median (noise in the substrate, edge sharpness) on synthetic data exist;
    - all parameters are configurable with the start values of nanofactorysystem T85.

- [ ] MI-31: Noise metrics [phase: 3]
  - Priority: medium
  - Size: S
  - Depends on: –
  - Done when: shared functions for the noise in a reference region and its spectrum (reusing the
    ensemble statistics and PSD tools, see MI-75) exist, so that fusion, the N-study (OffAxisHolo K13)
    and the comparison report the same number;
    - the speckle / position study (M4, with K13) can be run from them: noise vs. frames (fixed
      positions), noise vs. positions (fixed frames), minimum number of positions, correlation length of
      the coherent noise pattern; model σ² ≈ σ²_t / (N_frames · N_pos) + σ²_coh / N_pos.

### Phase 4 — Stitching

- [ ] MI-40: Stitching (G10 SA6) [phase: 4]
  - Priority: high
  - Size: L
  - Depends on: MI-20, MI-21
  - Done when the maintainer's algorithm runs on a synthetic raster:
    - order: first a ring of tiles with ≈ 50 % substrate (each anchored to the substrate), then inwards;
      the tile measured before is ground truth;
    - a border band of every tile is discarded after propagation (width a parameter; estimate
      |z| · tan θ_max with sin θ_max = NA / n);
    - **unwrapping per tile before stitching** (OffAxisHolo K11 helper, `fast2d` default): piston from the
      complex overlap, integer k · 2π from the unwrapped difference, consistency check (constant
      difference, otherwise flag the tile / guided unwrapping);
    - a final global least-squares adjustment of all overlaps with the ring tiles as absolute anchors;
      tiles without substrate are anchored through the overlaps and the anchor is recorded per tile
      (`reference_source` / `reference_fraction` per tile, SA4);
    - output `LayerProduct` with mosaic, tiles, transforms, residuals; works incrementally (tile by
      tile, MI-60); a tile may itself be a fused speckle set;
    - note for K11-T (OffAxisHolo `docs/realtime/unwrapping_comparison.md`): store spatial and, once
      available, temporal results per layer during the study.

- [ ] MI-41: Detection and fit of the calibration marks (G10 SA2) [phase: 4]
  - Priority: high
  - Size: M
  - Depends on: MI-11
  - Done when:
    - the printed cross + square of nanofactorysystem T104 are detected in camera and DHM images; the
      cross gives the offset of the centres, the square rotation / scale / shear; the cross imaged at known
      stage offsets (e.g. 3 × 3) gives an affine least-squares fit independent of the galvo scale, with
      residuals;
    - the result feeds the calibration file of T104 and the model of MI-11; residual shifts of MI-40 can
      refine it later; tested on synthetic marks.

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
  - Depends on: MI-10
  - Done when: a `LayerTarget` (height map and exact mask in µm, position) is shifted / transformed onto the
    pixel grid of a `LayerProduct` using MI-11 (or the stage position until MI-11 exists) and the detected
    pose of the structure; it is only **downsampled** (area-weighted), never upsampled (M6).

- [ ] MI-51: Forward model [phase: 5]
  - Priority: high
  - Size: M
  - Depends on: MI-50, OffAxisHolo K20 (formulas)
  - Done when: the expected phase is computed from target height, wavelength and `n_structure` −
    `n_surround` of the material snapshot (in-situ: cured in resin vs. liquid resin), with the inverse
    for the height output; the comparison domain is chosen automatically (phase preferred, else height)
    or fixed (M5 = OM4); the formulas come from OffAxisHolo `fieldops` (optional import) or are mirrored
    with a parity test; no default index; `calculate_delta_n` is reused for calibration.

- [ ] MI-52: Comparison [phase: 5]
  - Priority: high
  - Size: M
  - Depends on: MI-51 (final tolerances: M7, deferred)
  - Done when: `compare(product, target, *, sigma, k) -> DeviationReport` returns the deviation map,
    RMSE, MAE, statistics per segment, the share of the area above k·sigma, the classification too low /
    too high, and validity flags (registration confidence, coverage, product quality); tested on
    synthetic cases.

- [ ] MI-53: Stable entry point and report format [phase: 5]
  - Priority: high
  - Size: M
  - Depends on: MI-52
  - Done when: `mira.api.evaluate_layer(product, target, context, settings) -> DeviationReport` exists;
    the report is a header dictionary plus arrays with `schema_version`, `domain` and the computing
    time; a test validates it against the shared example. This is what nanofactorysystem T90 calls on the
    Work-PC (G6).

### Phase 6 — Integration

- [ ] MI-60: Plug-in for the DHM service (G5-new) [phase: 6]
  - Priority: high
  - Size: M
  - Depends on: MI-30, MI-40
  - Done when: `mira.plugins.dhmconnect` offers an incremental group API (open with the `GroupSpec`,
    `add_tile(TileProduct)` while the stage moves to the next position, `close()` → `LayerProduct`) for the
    fusion and the stitching configuration; pure functions, no network, picklable-free inputs (arrays +
    JSON); runs in DHMConnect worker processes on the CPU (M8); called by DL-B9; timings per tile.

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
  - Depends on: a profile that shows the need (after V1.0, M8)
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

- [ ] MI-79: Invert the warp of `ECCRefinement` (found during MI-05; M11 answered in MI-04) [phase: A]
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

- [ ] MI-82: No silent default refractive index (G7, found during the decision session 2026-11-18) [phase: A]
  - Priority: medium
  - Size: S
  - Depends on: MI-04
  - Done when: the default `n_resin = 1.52` no longer applies silently in new code paths (missing index →
    error or warning naming the source); the publication scripts pass n = 1.52 explicitly; the MI-04
    regression shows unchanged numbers.

## In Progress
<!-- Claude Code moves a todo here when starting work. -->
- [ ] MI-02: Packaging [phase: 0]
  - Priority: high
  - Size: S
  - Depends on: MI-07
  - Done when:
    - the conflict between static and dynamic dependencies is removed (no `requirements.txt` reference);
      author and description are set; extras `viz` (matplotlib), `analysis` (pandas), `dev`;
    - pandas is declared (needed by `analysis/ensemble.py` and `analysis/reliability.py`);
    - an editable install works in a fresh environment and `python -m pytest` runs there.
  - Progress 2026-11-17: everything except the Python / NumPy baseline (`4280aec`; pingouin in `dev`
    with `91c2cd5`); checked in fresh environments with `.[dev]` and with the core only. Remaining:
    MI-07 (M9 decided 2026-11-18).


## Blocked
<!-- Format: todo as above, plus the line "Blocked by: <reason or ID>". -->

## Done
<!-- Claude Code adds: - [x] MI-<n>: title — YYYY-MM-DD — 1–2 sentences on what changed — commits: `<sha>`, … -->
- [x] MI-01: Clean up the legacy copies — 2026-11-18 — Longer `plot_dual_profile` docstring taken over, then the approved list (M1) moved to the local attic (X-B) — commits: `dc176cd`, `2a4e995`.
- [x] MI-08: Move the approved legacy files to the attic — 2026-11-18 — Conflict copy, 19 `*_old*` modules, `bland_altman_snippet.py`, the empty `tests/_test/test_dhm.py` and `todo_claude/` moved to the local, Git-ignored `attic/`; `docs/ATTIC_INDEX.md` lists original path and last commit; 119 tests pass — commits: `2a4e995`.
- [x] MI-83: Refresh the system plan copy — 2026-11-18 — `docs/system/SOFTWARE_PLAN_OVERVIEW.md` is the unchanged copy of master v0.4 (X-D) — commits: `b611c4e`.
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
