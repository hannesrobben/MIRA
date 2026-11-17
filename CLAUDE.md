# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

MIRA ("Microscopy Image Registration & Analysis", package `mira`, src layout, Python 3.12) relates measured phase / height images of printed micro-structures to place and to the design. Today it is the analysis library behind the maintainer's DOE publication work: structure detection, segmentation, alignment, phase ↔ height conversion, refractive-index (Δn) calibration and statistics (regression, ensembles over prints, ICC, sensitivity, error vs. topology) with separate plotting.

In the real-time feedback system it takes over four new tasks (see `TODO.md`):
1. **Registration:** place products by their stage position and refine the placement in the image.
2. **Speckle fusion:** combine captures of the same structure at shifted positions into one product with less speckle noise.
3. **Stitching:** combine a raster of products into a mosaic of a structure larger than the DHM field of view.
4. **Target comparison:** compare a product with the target structure and return a `DeviationReport` to nanofactorysystem.

Out of scope: hologram reconstruction (`offaxisholo`), hardware, network, stage control, decisions about corrections.

Repository: `github.com/hannesrobben/MIRA`, branch `main`.

Project files:
- `TODO.md`: the task list (sections Open / In Progress / Blocked / Done; format rules, phase plan and open decisions at the top). It includes the open points of the DOE analysis session (phase A).
- `WORKLOG.md`: append-only work log (see "Work log").
- `docs/system/SOFTWARE_PLAN_OVERVIEW.md`: the plan for the four cooperating repositories (copy; master in NanoFactorySystem_automated).
- `docs/notes/zusammenfassung_aenderungen_und_tests.txt` and `docs/notes/offene_punkte_und_zernike_psd_wrapper.txt` (German): summary and open points of the DOE analysis session (Bland-Altman, Δn calibration, regression, info / statistics dicts, ensembles, snapshots, topology, sensitivity, ICC, planned Zernike / PSD wrappers). Their open items are in `TODO.md`; they moved from `src/mira/` to `docs/notes/` with MI-01.
- `MIRA_tree.txt`: an old tree snapshot (gitignored).

## Commands

```bash
python -m pip install -e ".[dev]"   # editable install (pytest, black, numpydoc)
python -m pytest                    # tests in tests/ (pythonpath = src); the suite is being rebuilt (MI-04)
black src tests                     # formatting (line length 88)
```

- pytest configuration is in `pyproject.toml` (`testpaths = ["tests"]`, `pythonpath = ["src"]`). Today `tests/_test/test_dhm.py` is empty; the test suite of the DOE work lives elsewhere (MI-04).
- `pyproject.toml` declares the dependencies statically **and** dynamically from a `requirements.txt` that does not exist, and has a placeholder author (MI-02).
- A local `.venv` (Python 3.12.3) exists inside the folder (gitignored).

## Architecture

- `processor.py`: `StructureProcessor`, the facade with method chaining. Loading (`load_phase_data`, `load_height_data`, `load_intensity_data`, `from_file`), `detect_structure`, `realign_structure`, `segment_structure`, `align_with`, `convert_to_height` / `convert_to_phase`, `unwrap_phase`, `phase_retrieval`, `adjust_background`, `calculate_delta_n`, plots, `save_data`, `save_snapshot` / `load_snapshot` (npz with JSON metadata). **`apply_affine_transform` is the single gateway for geometry**: raw channels, masks and segments are transformed together.
- `core/`: `ImageData` (data, pixel size, profiles, resize, padding), `Structure` (mask, bounds, corners, `info` dict with ids via `set_info`, `statistics` dict via `add_statistic`), `Segment`, `Background` (incl. `erode_by_pixels`).
- `processing/`: `detection.py`, `segmentation.py` (`segment_structure`, the implementation the processor uses; `Structure.segment()` is a duplicate, MI-71), `alignment.py`, `registration.py` (`RegistrationRefiner`, `NoRefinement`, `ECCRefinement`, `PhaseCorrelationRefinement`, `get_refiner`), `phase_retrieval.py` (phase ↔ height, unwrapping, background adjustment), `calibration.py` (`calculate_delta_n` → `DeltaNResult`, method registry, currently `"segment_mean"`).
- `analysis/` (computation only): `segment_data.py` (shared segment statistics with edge erosion and background-relative values), `comparison.py` (`fit_linear_regression`, `fit_segment_regression` → `RegressionResult`), `ensemble.py` (records and pandas DataFrames over prints, `summarize_by_group`), `reliability.py` (`compute_icc`, ICC(1,1), ICC(2,1), ICC(3,1) with CI and F-test; `interpret_icc`), `sensitivity.py` (Δn uncertainty propagation and sweep), `topology.py` (error vs. gradient and curvature; expects pixel size in metres).
- `visualization/` (plotting only): `plotter_2d.py`, `plotter_3d.py`, `plotter_stats.py`, `bland_altman.py` and `bland_altman_snippet.py` (compute and plot still combined, MI-72), `visualization_debug.py`.
- `utils/`: `exceptions.py` (`DataLoadError`, `ValidationError`, `ProcessingError`, `StructureError`, `SegmentationError`, `AlignmentError`, …), `helpers.py` (logger, conversions, geometry helpers), `serialization.py` (`to_json_safe`).
- `MIRA_composition.py`: an unused sketch of a future data model (`ImageData` / `Structure` / `Measurement` / `Processor`).
- Legacy copies, not the active path: `*_old.py`, `*_old<n>.py`, `processor_old^1.py`, and the Seafile conflict copy `visualization/plotter_2d (SFConflict … 2026-10-06-15-55-59).py`. They are committed (`254b25e`) and move to `attic/` only after the maintainer approves the list (MI-01).
- Import weight: `import mira` imports `processor.py`, which imports OpenCV and matplotlib at module level (MI-05 changes this).
- Known pitfall (fixed in two places, MI-77 audits the rest): `x.data if hasattr(x, "data") else x` unmasks a `numpy.ma.MaskedArray`; check for ndarray / MaskedArray first and use `.data` only for wrapper objects such as `ImageData`.
- Known limitation: after `load_snapshot`, `structure.statistics` holds plain JSON dicts, not the original dataclasses (rebuild explicitly, e.g. `DeltaNResult(**d)`).

## System context

This repository is one of four that together build a 2PP system with real-time feedback from the DHM (details: `docs/system/SOFTWARE_PLAN_OVERVIEW.md`).

| Package | Repository | Owns |
|---|---|---|
| `mira` (this one) | MIRA | Registration by stage position, speckle fusion, stitching, comparison with the target structure, deviation report; the existing statistical analysis. |
| `offaxisholo` | OffAxisHolo (branch `Holo_HR`) | Hologram reconstruction → `ReconstructionResult` (phase, intensity, optional complex field, pixel size, metadata). Zernike / PSD functions are planned there (P15). |
| `dhmconnect` | DHMConnect | Protocol, DHM service, shared contracts (`dhmconnect/schemas/`). The service calls MIRA's group plug-in (MI-60) on the DHM-PC. |
| `nanofactorysystem` | NanoFactorySystem_automated | Orchestration, `LayerTarget`, operating modes; calls `mira.api.evaluate_layer` (MI-53) and stores the `DeviationReport`. |

Rules that follow from this:
- No hardware, network or hologram reconstruction code in this repository.
- `mira` may import `offaxisholo` only optionally; it never imports `dhmconnect` or `nanofactorysystem` at import time (the plug-in for the service and the offline access to `experiment.h5` use optional imports). The test suite passes without the other three packages.
- Shared contracts (`ProductSet`, `GroupProduct`, `LayerTarget`, `DeviationReport`, …) are defined in DHMConnect; this repository vendors the example messages under `tests/contracts/`. Conventions: units in names (`_um`, `_rad`), float32 / complex64, axis 0 = y, axis 1 = x, `schema_version` in every message, never `pickle` across a process or network boundary.
- Todo IDs of the other repositories may be referenced: `K…`/`L…`/`P…` (OffAxisHolo), `DL-…` (DHMConnect), `T…`/`F…` (NanoFactorySystem_automated).

## Repository rules

1. Nothing is deleted. Old copies move to `attic/` only after the maintainer approves the list (MI-01).
2. Fix problems at their source, not downstream. Prefer a validation error over a silent fallback.
3. Every geometric transform of a processor goes through `apply_affine_transform`; raw channels, masks and segments are transformed together.
4. Computation and plotting stay separate (`mira.analysis`, `mira.processing` against `mira.visualization`). `import mira` must not import matplotlib (MI-05). New analyses follow the pattern compute function → result dataclass (in `analysis/`) + plot function (in `visualization/plotter_stats.py`).
5. New functions for products are pure: arrays and metadata in, arrays and metadata out. No file access, no global state, backend-agnostic signatures (NumPy first).
6. Units in names (`_um`, `_rad`); heights in metres only where the existing API requires it, converted at the boundary.
7. Results of the publication analysis must not change. Run the regression test of MI-04 after every non-trivial change; a change of a published number needs the maintainer's approval.
8. Statistical formulas used in a paper (ICC, CI, regression) are checked against a reference implementation before use (MI-76). Do not cite formulas from memory without such a check.
9. One deliverable per session, with a short integration note for the maintainer at the end (what to check with real data).

## Maintainer Instructions (Hannes)

### Language
Everything is written in English: variable, function, class and module names, docstrings,
comments, README and other documentation, design documents, todo entries, commit messages,
pull request texts and log entries. If you encounter German identifiers or comments within
the scope of your current task, translate them. Outside that scope, add a todo instead of
renaming them unasked. The German session notes in `docs/notes/*.txt` are translated when their
content is turned into documentation.

### Session start
1. Read `## Current Status` and `## Status Update Anchor` at the end of this file. As long as
   they hold no status yet, go by the phase plan in `TODO.md`.
2. Read `TODO.md` and pick a task (working rule 1 at the top of `TODO.md`).
3. Read the most recent entries in `WORKLOG.md`.
4. For products, registration, fusion, stitching or the comparison, read
   `docs/system/SOFTWARE_PLAN_OVERVIEW.md` first.

### Work log (mandatory)
After every completed task, and before ending a session even if the task is unfinished,
append an entry to `WORKLOG.md`.

- Get the timestamp from the system: `TZ=Europe/Berlin date "+%Y-%m-%d %H:%M %Z"`. Never guess it.
  (In Git Bash on Windows without tzdata, use PowerShell `Get-Date -Format "yyyy-MM-dd HH:mm"`.)
- The log is append-only. Put the newest entry at the bottom. Never edit past entries.
- Write one entry per task. List **every** file that was created, modified, deleted or renamed.

Format:
```
### <YYYY-MM-DD HH:MM TZ> — [<Todo ID>] <Short task title>
- **Status:** done | partial | blocked
- **Changes:**
  - `path/to/file.py`: what changed and why
  - `path/to/new_file.py` (new): purpose
  - `path/to/old_file.py` (deleted | renamed → `new/path.py`)
- **Tests:** what was run and the result, or "none" with the reason
- **Commits:** `<short SHA>` <commit subject line>, … (or "uncommitted — proposed message:" followed by the full message)
- **Follow-ups:** open points or new todos created
```

### Definition of Done
A task counts as done only if all of the following hold:
- All acceptance criteria stated in the todo are met.
- The code runs, and existing tests pass, including the regression of the publication analysis
  (once MI-04 exists). New logic has tests where that is feasible (synthetic data with stated
  tolerances).
- Public functions and classes have English docstrings (NumPy style).
- No new `TODO`/`FIXME` related to the task remains in the code.
- The change is committed and logged in `WORKLOG.md`.

If a task does not meet this, mark it `partial` in the log and describe the remaining gap.

### Todos
- Follow the format rules at the top of `TODO.md`.
- When you start a todo, move it to "In Progress". If it cannot proceed, move it to "Blocked"
  and state the reason.
- When a todo meets the Definition of Done, move it to "Done" with the date, 1–2 sentences
  and the commit SHAs. Never move entries to `ARCHIVE.md`; the status review routine does that.
- If a todo is only partially done, keep it in "In Progress" and add a follow-up todo that
  describes the remaining gap and the decision or measurement it needs.
- A todo that needs an open decision (table "Open decisions" in `TODO.md`) is never guessed:
  implement the independent part and ask.
- If you discover new problems, add them to "Open" and tag them with `(found during <ID>)`.

### Commits (mandatory)
- Work on a feature branch created from `main` (`feat/…`, `fix/…`, `docs/…`); the maintainer
  merges. No more "saving current status" commits: every commit has one topic.
- Commit after every completed task and after every coherent sub-step of larger tasks. A task
  that ends without a commit is not done.
- Split commits by topic: one logical change per commit. Do not mix refactoring, formatting,
  feature work, tests of unrelated code and documentation of other tasks. Todo bookkeeping
  (`TODO.md`, `WORKLOG.md`) goes into its own `docs(todo): …` commit at the end of a task.
- Write every message for a human who reads `git log` months later without the session: the
  subject says what the change does (not "update files", "fixes", "WIP"), the body says why it
  was needed and what a reader must know (decisions, side effects, limits, changed numbers).
  Example of a good subject: `fix(topology): keep the mask of masked arrays when extracting data [MI-77]`.
- Always write a complete commit message in this format (Conventional Commits):
```
  <type>(<scope>): <imperative summary, max. 72 chars> [<Todo ID>]

  <Body: what changed and why, wrapped at 72 chars. Mention design
  decisions and side effects. No restatement of the diff.>

  Files:
  - path/to/file.py: short description of the change
  - path/to/new_file.py (new): purpose

  Refs: <Todo ID>
```
  - `type` is one of: `feat`, `fix`, `refactor`, `test`, `docs`, `perf`, `build`, `chore`.
  - `scope` is the affected module or package, e.g. `processor`, `registration`, `fusion`,
    `stitching`, `comparison`, `analysis`, `visualization`, `packaging`.
  - Add `BREAKING CHANGE: <description>` as a footer if a public API changes.
- If you cannot or should not commit, do not commit. This applies to failing tests, unrelated
  uncommitted changes in the working tree, and cases where the maintainer said not to commit.
  Still generate the full commit message: print it at the end of your response and record it
  in WORKLOG.md.
- When working on a feature branch, also generate a pull request title and description:
  Summary, Changes (with files), How it was tested, Linked todos, Open points.
- Do not push, amend, rebase or force-push unless explicitly instructed.
- Do not edit `## Status Update Anchor` or `## Current Status`. These sections are maintained
  exclusively by the status review routine.

### Testing conventions
- `python -m pytest` must pass without `offaxisholo`, `dhmconnect` or `nanofactorysystem`
  installed and without a display (matplotlib backend `Agg` in tests).
- Registration, fusion, stitching and comparison are tested on synthetic fields with known
  shift, piston, tilt and noise (MI-22); tolerances are stated in the test.
- Never make a test pass by weakening or deleting assertions without justification.
  If a test reveals a real bug, fix it in a separate `fix:` commit if the fix is small and clear.
  Otherwise, mark the test `xfail(reason=..., strict=True)` and add a todo.
- Tests write only to `tmp_path`.
- Never modify the legacy copies (`*_old*.py`, the conflict copy).

## Current Status
<!-- Maintained exclusively by the status review routine. Claude Code reads it, never edits it. -->
No status written yet (file created 2026-11-15).

## Status Update Anchor
<!-- Maintained exclusively by the status review routine. Claude Code reads it, never edits it. -->
No anchor set yet.
