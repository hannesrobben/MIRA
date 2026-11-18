# Work Log

Append-only log of completed and partial tasks. Newest entry at the bottom.
Format and rules: see "Work log (mandatory)" in CLAUDE.md. Earlier work is summarised in
`src/mira/zusammenfassung_aenderungen_und_tests.txt` and in the git history.

### 2026-11-15 23:15 CET — [MI-03] Project files: CLAUDE.md, TODO.md, system plan
- **Status:** done (README update split off as MI-06)
- **Context:** Maintainer request: create a `CLAUDE.md` in the style of NanoFactorySystem_automated and
  sort the todos. Inputs: `todo_claude/04_MIRA_PLAN.md`, `00_SOFTWARE_PLAN_OVERVIEW.md`,
  `src/mira/offene_punkte_und_zernike_psd_wrapper.txt`, `src/mira/zusammenfassung_aenderungen_und_tests.txt`,
  the current code (commit `254b25e`).
- **Decisions taken with the maintainer:** the communication package is `dhmconnect` (G1); the system plan
  is copied into every repository, master copy in NanoFactorySystem_automated; all four repositories use
  the TODO.md format of nanofactorysystem.
- **Changes:**
  - `CLAUDE.md` (new): purpose, commands, architecture of `mira`, known pitfalls, system context,
    repository rules, maintainer instructions.
  - `TODO.md` (new): phases 0–6 from the plan (MI-01 reduced, because the working state is committed in
    `254b25e`; MI-04 blocked by a question; MI-06 new) and phase A with the open points of the DOE session
    (MI-70–MI-78); decisions M1–M10.
  - `docs/system/SOFTWARE_PLAN_OVERVIEW.md` (new): copy of the system plan v0.2.
  - `WORKLOG.md` (new): this file.
- **Tests:** none (documentation only).
- **Commits:** uncommitted — proposed messages (one per topic, on a branch `docs/project-files` from `main`):
  1. `docs(system): add the plan for the four cooperating repositories [MI-03]`
     Files: `docs/system/SOFTWARE_PLAN_OVERVIEW.md` (new)
  2. `docs: add CLAUDE.md with architecture, rules and commit conventions [MI-03]`
     Files: `CLAUDE.md` (new)
  3. `docs(todo): plan registration, fusion, stitching and comparison work [MI-03]`
     Files: `TODO.md` (new), `WORKLOG.md` (new)
- **Follow-ups:** MI-04 needs the location of the DOE3 test suite; MI-01 needs the approval of the attic
  list; the untracked folder `todo_claude/` can be removed once the maintainer has reviewed these files.

### 2026-11-17 13:02 CET — [MI-03] Commit the project files
- **Status:** done
- **Context:** Maintainer request of 2026-11-17: read the whole repository (incl. `todo_claude/` and all
  `.md` files), summarise the todos and work off every todo that needs no decision. The project files of
  MI-03 were still untracked; the todo bookkeeping of this session needs them in git.
- **Changes:** none beyond committing the files of the previous entry unchanged, with the proposed
  messages, on branch `docs/project-files` (from `main`). All further work is on
  `feat/baseline-and-analysis-backlog`, branched from `docs/project-files`.
- **Tests:** none (documentation only).
- **Commits:** `b819b2c` docs(system): add the plan for the four cooperating repositories [MI-03],
  `bf079be` docs: add CLAUDE.md with architecture, rules and commit conventions [MI-03],
  `fe99316` docs(todo): plan registration, fusion, stitching and comparison work [MI-03]
- **Follow-ups:** `todo_claude/` stays untracked; it only holds the v0.1 drafts (`04_MIRA_PLAN.md`,
  `00_SOFTWARE_PLAN_OVERVIEW.md`) and the CLAUDE.md of nanofactorysystem used as template. Everything in it
  is absorbed into `CLAUDE.md`, `TODO.md` and `docs/system/`; the maintainer can delete the folder.

### 2026-11-17 13:02 CET — [MI-01] Session notes to docs/notes, attic proposal
- **Status:** partial (the move of legacy files waits for M1; follow-up MI-08)
- **Changes:**
  - `src/mira/offene_punkte_und_zernike_psd_wrapper.txt` (renamed → `docs/notes/offene_punkte_und_zernike_psd_wrapper.txt`)
  - `src/mira/zusammenfassung_aenderungen_und_tests.txt` (renamed → `docs/notes/zusammenfassung_aenderungen_und_tests.txt`)
  - `docs/ATTIC_REVIEW.md` (new): diff of the Seafile conflict copy against `plotter_2d.py` (one docstring
    of `plot_dual_profile`, no code), the 19 legacy files with lines and last commit, the Bland-Altman
    modules, two further candidates (`MIRA_composition.py`, empty `tests/_test/test_dhm.py`).
  - `CLAUDE.md`: new path of the notes.
  - `TODO.md`: new path of the notes (in the bookkeeping commit).
- **Tests:** none (documentation and file moves; no module imports the notes).
- **Commits:** `ebfc2b0` docs: move the DOE session notes to docs/notes and list attic candidates [MI-01]
- **Follow-ups:** MI-08 (move the approved files, needs M1). Note for M1: the MI-01 text says
  "`bland_altman_snippet.py` if superseded by `bland_altman.py`"; it was the other way round, see the review.

### 2026-11-17 13:02 CET — [MI-05] Import hygiene
- **Status:** done
- **Changes:**
  - `src/mira/processor.py`: cv2, matplotlib and the debug plot imported in the methods that use them;
    `_CV2_INTER_LINEAR` as default of `apply_affine_transform`.
  - `src/mira/core/structure.py`, `src/mira/processing/alignment.py`: lazy cv2 import.
  - `src/mira/processing/registration.py`: lazy cv2 import; `ECCRefinement._MOTION` holds the names of
    the cv2 constants.
  - `src/mira/core/image.py`: lazy pandas import (CSV read / write only).
  - `tests/conftest.py` (new): Agg backend.
  - `tests/test_import_hygiene.py` (new): `sys.modules` after `import mira` (and analysis/processing
    modules) in a subprocess; constant and motion names checked against cv2.
  - `tests/test_processor_smoke.py` (new): detection, perimeter, `apply_affine_transform` with segments,
    both refiners on synthetic fields.
- **Tests:** `python -m pytest`: 9 passed, 1 xfailed (ECC, see MI-79).
- **Commits:** `e3adbbd` refactor(processor): import OpenCV, matplotlib and pandas on use [MI-05]
- **Follow-ups:** MI-79 (new): `ECCRefinement` returns the inverse warp; verified on a textured field
  shifted by (2, -3) px: after `cv2.warpAffine(moving, M)` the RMS error is 0.317 with ECC and 0.0 with
  phase correlation. Test is `xfail(strict=True)`; decision M11 (new) because the fix may change published
  numbers if `refine="ecc"`/`True` was used.

### 2026-11-17 13:02 CET — [MI-02] Packaging
- **Status:** partial (Python / NumPy baseline waits for M9; follow-up MI-07)
- **Changes:**
  - `pyproject.toml`: removed the dynamic dependency table (missing `requirements.txt`); core
    dependencies numpy, scipy, opencv-python>=4.10.0.84, scikit-image, scikit-learn; extras `viz`
    (matplotlib), `analysis` (pandas), `dev` (both + pytest, black, yapf, numpydoc, pingouin); author and
    description set; pillow dropped (no module imports it).
- **Tests:** two fresh virtual environments in the scratchpad (Python 3.12.3, NumPy 2.5.5, OpenCV
  5.0.0.93, pandas 3.1.1): `pip install -e ".[dev]"` + `python -m pytest` → 9 passed, 1 xfailed;
  `pip install -e .` → `import mira` loads neither matplotlib nor pandas, detect + segment work,
  `import mira.visualization` fails with "No module named 'matplotlib'" as expected.
- **Commits:** `4280aec` build(packaging): declare dependencies once and add viz/analysis extras [MI-02]
  (pingouin added to `dev` in `91c2cd5`)
- **Follow-ups:** MI-07 (new, needs M9). The local `.venv` was empty (no numpy); the dependencies were
  installed into it. The system Python 3.12 has an OpenCV build that fails under NumPy 2.

### 2026-11-17 13:02 CET — [MI-77] Audit of the masked-array unmasking pattern
- **Status:** done
- **Changes:**
  - `src/mira/analysis/comparison.py`: `fit_linear_regression` leaves pairs masked in x or y out (it used
    `np.asarray`, which drops the mask); plain arrays take the old path.
  - `tests/test_masked_inputs.py` (new): regression parity with scipy, masked outliers, too few points,
    topology with a masked corner (2400 of 2500 pixels).
  - Audit result (also in the commit message): `analysis/topology.py` and `bland_altman_snippet.py` were
    already correct; `visualization/bland_altman.py` unmasked (replaced in MI-72); `compute_icc` unmasked
    (fixed in MI-76); all other places use `isinstance(x, ImageData)`.
- **Tests:** `python -m pytest`: 14 passed, 1 xfailed; the two new regression tests fail on the old code.
- **Commits:** `a7696ae` fix(comparison): leave masked pairs out of fit_linear_regression [MI-77]
- **Follow-ups:** none.

### 2026-11-17 13:02 CET — [MI-72] Split Bland-Altman into compute and plot
- **Status:** done
- **Changes:**
  - `src/mira/visualization/__init__.py`: missing comma joined "BlandAltmanResult" and "plot_surface" in
    `__all__` (separate fix commit); now exports the split functions.
  - `tests/test_visualization_exports.py` (new): every name in `__all__` exists.
  - `src/mira/analysis/comparison.py`: `BlandAltmanResult` (first six fields as before, plus
    `ci_multiplier`, `relative`, `means`, `differences`) and `bland_altman_compute()`.
  - `src/mira/visualization/plotter_stats.py`: `plot_bland_altman(result, ...) -> Figure`; arrays as input
    raise a `VisualizationError` that names both functions.
  - `src/mira/visualization/bland_altman.py`: former combined call `plot_bland_altman(data_1, data_2,
    ...) -> (fig, result)` as a wrapper (still raises `VisualizationError`).
  - `src/mira/visualization/bland_altman_snippet.py`: re-export only.
  - `tests/test_bland_altman.py` (new): values recorded from the former exported snippet (plain, float32,
    relative, masked, masked + relative) reproduced to rel 1e-12; masks; ImageData-like wrapper; input
    validation; plot to `tmp_path`; former call; package exports.
- **Tests:** `python -m pytest`: 26 passed, 1 xfailed.
- **Commits:** `a37b2d3` fix(visualization): separate two names in __all__ that a missing comma joined
  [MI-72], `4d88116` refactor(analysis): split Bland-Altman into compute and plot functions [MI-72]
- **Follow-ups:** BREAKING CHANGE: `mira.visualization.plot_bland_altman` now takes a result; old scripts
  should import `mira.visualization.bland_altman.plot_bland_altman`. The two former modules disagreed:
  the older `bland_altman.py` unmasked masked arrays and computed float32 input in float32 (bias
  -0.2004574239 instead of -0.2004574128 on the test data). The split follows the exported snippet.

### 2026-11-17 13:02 CET — [MI-78] Pixel-size units of the topology analysis
- **Status:** done
- **Changes:**
  - `src/mira/analysis/topology.py`: `_validate_pixel_size_m` (two finite, positive values, each at most
    `MAX_PIXEL_SIZE_M = 1e-3`) in `analyze_error_vs_topology`, `compute_gradient_magnitude` and
    `compute_curvature`; docstrings name metres. Parameter name unchanged.
  - `tests/test_topology_units.py` (new): gradient per metre, seven invalid inputs for all three
    functions, valid metres accepted.
- **Tests:** `python -m pytest`: 52 passed, 1 xfailed (after MI-73 tests were added in the same run).
- **Commits:** `23948fc` fix(topology): reject pixel sizes that are not plausible metres [MI-78]
- **Follow-ups:** none. Correlations were never affected by the unit (scale invariant); bin centres were.

### 2026-11-17 13:02 CET — [MI-73] Processor method for the segment regression
- **Status:** done
- **Changes:**
  - `src/mira/processor.py`: `StructureProcessor.fit_segment_regression(reference, offset_pixels=10,
    include_background=True, relative=True, statistic_name="segment_regression")`; reference = x, self =
    y; stored with `add_statistic()`; no background adjustment (documented).
  - `tests/test_processor_regression.py` (new): slope 3 / intercept 0 on a synthetic pair, parity with the
    analysis function, validation of unsegmented processors.
- **Tests:** `python -m pytest`: 52 passed, 1 xfailed.
- **Commits:** `374c734` feat(processor): add fit_segment_regression that stores its result [MI-73]
- **Follow-ups:** MI-80 (new): with quadrant levels 1–4 on a background of 0 the default threshold
  detection returned six corner points and an almost empty first segment; the test uses levels 11–14.

### 2026-11-17 13:02 CET — [MI-76] Check the ICC implementation against a reference
- **Status:** done (the form for the paper stays decision M10)
- **Changes:**
  - `src/mira/analysis/reliability.py`: p-value via `f.sf` instead of `1 - f.cdf`; NaN / inf cells and
    masked entries raise a `ValidationError` ("complete table"); module docstring: check result, naming
    in pingouin, full references (Shrout & Fleiss 1979, McGraw & Wong 1996, Koo & Li 2016).
  - `tests/test_reliability_reference.py` (new): 20 random balanced tables × 3 forms against
    `pingouin.intraclass_corr` with rounding off (ICC, CI, F, df, p); Shrout & Fleiss example (.17 / .29 /
    .71); pingouin's wine data set; incomplete tables (NaN, inf, masked, incomplete DataFrame).
  - `pyproject.toml`: pingouin in `dev`.
- **Tests:** `python -m pytest`: 119 passed, 1 xfailed. Largest relative difference to pingouin 0.7.0:
  3e-14 (lower CI bound of ICC(3,1)); pingouin implements the same Satterthwaite formula for ICC(2,1).
- **Commits:** `91c2cd5` test(reliability): check compute_icc against pingouin and reject gaps [MI-76]
- **Follow-ups:** the p-value change is at most 1e-15 absolute (only visible for p near 1e-15).

### 2026-11-17 13:02 CET — [MI-06] README for the real package
- **Status:** done
- **Changes:**
  - `README.md`: rewritten for `mira` (purpose, system role, installation with extras, real layout,
    StructureProcessor example, compute-then-plot analyses, conventions, development).
  - `CLAUDE.md`: architecture notes updated for MI-02, MI-05, MI-72, MI-77 and the ECC bug (separate
    commit; status sections untouched).
- **Tests:** the StructureProcessor example ran on synthetic data (Δn = 0.012 recovered, R² = 1.0,
  snapshot round trip keeps `info` and `statistics`, `DeltaNResult(**dict)` works).
- **Commits:** `f212d59` docs(readme): describe the mira package instead of image_processing [MI-06],
  `fa0a066` docs: update CLAUDE.md for lazy imports, extras, tests and Bland-Altman
- **Follow-ups:** none.

### 2026-11-17 13:02 CET — Session summary (no phase completed)
- **Status:** partial — phase 0 is not complete (MI-01 / MI-08 need M1, MI-02 / MI-07 need M9, MI-04 is
  blocked); phase A done except MI-70, MI-71 (need MI-04), MI-74, MI-75 (need OffAxisHolo P15) and the
  new MI-79 (needs M11), MI-80.
- **Not started because they need a decision or a blocked todo:** MI-10 (M2, G2), MI-11 / MI-22 / MI-20 /
  MI-21 (MI-10), MI-30 (M3, M4), MI-31 (definition of the shared noise number with OffAxisHolo K13, and
  MI-75 for the spectrum), MI-40–MI-42, MI-50–MI-53 (M5, M6, M7, G4, G7), MI-60–MI-62.
- **Tests:** `python -m pytest` on the branch: 119 passed, 1 xfailed (strict, MI-79).
- **To check with real data:** (1) run the publication scripts against this branch — they must give the
  same numbers; scripts that import `plot_bland_altman` from `mira.visualization` need
  `mira.visualization.bland_altman.plot_bland_altman` instead; (2) calls of `analyze_error_vs_topology`
  with a pixel size in µm now raise (intended); (3) whether ECC refinement was used for any published
  result (M11).
- **Commits:** see the entries above; bookkeeping in the following `docs(todo)` commit.
- **Follow-ups:** PR from `feat/baseline-and-analysis-backlog` (contains `docs/project-files`) into `main`.

### 2026-11-17 15:56 CET — [MI-81] Translate the remaining German documentation in active code
- **Status:** done
- **Changes:**
  - `src/mira/core/image.py`: `ImageData.save_compressed` docstring translated and converted to NumPy
    style; inline comments translated.
  - `src/mira/processor.py`: comments in `phase_retrieval` translated.
  - `TODO.md`: MI-81 added directly to Done.
- **Tests:** `python -m pytest`: 119 passed, 1 xfailed (strict, MI-79), unchanged.
- **Commits:** `076c6af` docs(core): translate German docstring and comments to English [MI-81];
  bookkeeping in the following `docs(todo)` commit.
- **Follow-ups:** German that stays on purpose: `docs/notes/*.txt` (session notes, translated when turned
  into documentation), the legacy copies, the cited German title in `docs/system/SOFTWARE_PLAN_OVERVIEW.md`.
  The cross-repository overview of this session lives outside the repository in
  `../_system/` (ERRORS.md, DUPLICATE_CODE.md, DECISIONS.md, …).

### 2026-11-18 17:42 CET — [MI-01, MI-08, X-C] Attic move of the approved legacy files
- **Status:** done
- **Changes:**
  - `src/mira/visualization/plotter_2d.py`: longer `plot_dual_profile` docstring from the conflict copy (M1)
  - `.gitignore`: ignore `attic/` (X-B)
  - `docs/ATTIC_INDEX.md` (new): original path, last commit, date and reason of every moved file
  - `docs/ATTIC_REVIEW.md`: status approved and executed
  - `src/mira/core/image_old.py`, `src/mira/core/structure_old.py`, `src/mira/processing/alignment_old.py`, `alignment_old1.py`, `alignment_old2.py`, `alignment_old3.py`, `detection_old.py`, `phase_retrieval_old.py`, `phase_retrieval_old1.py`, `phase_retrieval_old2.py`, `src/mira/processor_old.py`, `processor_old2.py`, `processor_old3.py`, `processor_old4.py`, `processor_old5.py`, `processor_old^1.py`, `src/mira/visualization/plotter_2d_old.py`, `plotter_2d_old1.py`, `plotter_2d_old2.py`, `src/mira/visualization/plotter_2d (SFConflict … 2026-10-06-15-55-59).py`, `src/mira/visualization/bland_altman_snippet.py`, `tests/_test/test_dhm.py`, `todo_claude/00_SOFTWARE_PLAN_OVERVIEW.md`, `todo_claude/04_MIRA_PLAN.md`, `todo_claude/CLAUDE.md` (removed from Git with `git rm --cached`, moved to the local `attic/` at the same relative path)
- **Tests:** `python -m pytest` before and after: 119 passed, 1 xfailed.
- **Commits:** `dc176cd` docs(visualization): keep the longer plot_dual_profile docstring [MI-01], `2a4e995` chore(attic): move approved legacy files to the local attic [MI-08], merged with `b733af2`; before that `6eb6798` merged `docs/translate-german` (X-J).
- **Follow-ups:** none.

### 2026-11-18 17:42 CET — [X-D, MI-04, MI-10] Revision after the decision session 2026-11-18
- **Status:** done
- **Changes:**
  - `docs/system/SOFTWARE_PLAN_OVERVIEW.md`: unchanged copy of master v0.4
  - `TODO.md`: phase plan, "Open decisions" (M7, M10, M11 deferred), new table "Decided", rewritten MI-04, MI-07, MI-10, MI-11, MI-20, MI-21, MI-30, MI-31, MI-40, MI-41, MI-50–MI-53, MI-60, MI-61, MI-79; new MI-82; MI-01, MI-08, MI-83 done
  - `CLAUDE.md`: legacy copies in the local attic, role on both PCs after G5-new
  - `WORKLOG.md`: these entries
- **Tests:** none (documentation only); the full suite ran after the attic move (119 passed, 1 xfailed).
- **Commits:** `b611c4e` docs(system): copy master plan v0.4 unchanged [X-D], `8439a2d` docs(todo): derive the todos from the decisions of 2026-11-18, plus the `docs(todo)` commit of this entry.
- **Follow-ups:** MI-04 starts by asking the maintainer which data to use (with a proposal).

### 2026-11-18 21:55 CET — [MI-07] Python and NumPy baseline
- **Status:** done
- **Changes:**
  - `pyproject.toml`: `requires-python >= 3.12` (was 3.10), classifiers 3.12 / 3.14, `numpy>=2.0`, black `target-version = ["py312"]`
- **Tests:** fresh venvs on the Office PC, 2026-11-18, `pip install -e ".[dev]"` then `python -m pytest`: CPython 3.12.3 (`%LOCALAPPDATA%\Temp\mira312`, NumPy 2.5.5, pandas 3.1.1, matplotlib 3.11.2) 119 passed, 1 xfailed; CPython 3.14.10 (`%LOCALAPPDATA%\Temp\mira314`) 119 passed, 1 xfailed (the install builds one dependency from source and took several minutes).
- **Commits:** `7e6ecd7` build(packaging): require Python 3.12 and NumPy 2 (M9) [MI-07]
- **Follow-ups:** `dev` still installs both black and yapf (ERRORS MI-E12, not part of M9); the self-referencing `mira[viz,analysis]` extra resolved fine with pip 24.0 (compare DHMConnect DL-A9).

### 2026-11-18 22:30 CET — [MI-02] Close the packaging todo
- **Status:** done
- **Changes:**
  - `TODO.md`: MI-02 moved from "In Progress" to "Done"; its only remaining criterion (Python / NumPy baseline) was met by MI-07
  - `WORKLOG.md`: this entry
- **Tests:** `python -m pytest` (`.venv`, CPython 3.12.3): 119 passed, 1 xfailed.
- **Commits:** see the `docs(todo)` commit of this entry.
- **Follow-ups:** none.

### 2026-11-18 22:36 CET — [MI-10] Product data model TileProduct / LayerProduct
- **Status:** done
- **Changes:**
  - `src/mira/products.py` (new): `Grid`, `TileRecord`, `TileProduct` (ProductSet 1.0), `LayerProduct` (LayerProduct 1.0), `to_structure_processor` (the one (y, x) µm → (x, y) m conversion)
  - `src/mira/processor.py`: classmethod `StructureProcessor.from_product` (delegates to the bridge; nothing else changed)
  - `pyproject.toml`: extra `io` (h5py), part of `dev`
  - `tests/test_products.py` (new): 14 tests against the vendored examples
  - `tests/contracts/` (new): `ProductSet`, `LayerProduct`, `ReconstructionResult`, `CaptureContext` examples + `MANIFEST.json` (DHMConnect `fc85f5c`)
  - `src/mira/MIRA_composition.py` (moved to the local attic, X-B), `docs/ATTIC_INDEX.md`: index entry
  - `TODO.md`: MI-10 to Done; `WORKLOG.md`: this entry
- **Tests:** `python -m pytest` (`.venv`, CPython 3.12.3, h5py 3.16.0 installed for the HDF5 test): 133 passed, 1 xfailed (was 119 + 1).
- **Commits:** `5771568` feat(products): add TileProduct / LayerProduct per contracts 1.0 [MI-10], `79d7091` test(products): check the product types against vendored contracts [MI-10], `0e84ba8` chore(attic): move MIRA_composition.py to the local attic [MI-10], plus the `docs(todo)` commit of this entry.
- **Follow-ups:** the place and layout of a `LayerProduct` in `experiment.h5` is proposed here (`header_json` attribute + one dataset per buffer) and noted for nanofactorysystem T83; the tile grid is axis-aligned and centred on the actual stage position until MI-11.
