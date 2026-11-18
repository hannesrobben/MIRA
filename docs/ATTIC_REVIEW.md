# Attic review (MI-01)

Status: approved 2026-11-18 (decision M1, decision session 2026-11-17/18). Executed 2026-11-18.

All files of §1–§3 and `tests/_test/test_dhm.py` (§4) moved to the local `attic/` folder, keeping their
relative path (for example `src/mira/processor_old.py` → `attic/src/mira/processor_old.py`). Per decision
X-B the attic is not on GitHub: the files were removed from Git with `git rm --cached`, and
`docs/ATTIC_INDEX.md` lists every original path with its last commit for restoring. Nothing is deleted.
The longer docstring of `plot_dual_profile` (§1) was taken into `plotter_2d.py` first.
`src/mira/MIRA_composition.py` stays until the product model MI-10 is designed (decision M2).

## 1. Seafile conflict copy

`src/mira/visualization/plotter_2d (SFConflict … 2026-10-06-15-55-59).py` and `plotter_2d.py` are both
2392 lines long. Apart from line endings they differ in one docstring only (`plot_dual_profile`):

```diff
-        save_path_heatmap:    Optional path to save the heatmap to. Has to be with heatmap name. (.png)
-        save_path_profile:    Optional path to save the profile plot to. Has to be with profile name. (.png)
+        save_path_heatmap:    Optional path to save the heatmap to.
+        save_path_profile:    Optional path to save the profile plot to.
```

(`-` = conflict copy, `+` = `plotter_2d.py`.) No code differs. Proposal: move the conflict copy to the
attic; if the longer docstring is wanted, it can be taken into `plotter_2d.py` by hand.

## 2. Legacy copies

No active module imports any of these files (checked with `grep` on 2026-11-17). Lines and the last
commit that touched each file:

| File | Lines | Last commit |
|---|---|---|
| `src/mira/core/image_old.py` | 397 | `254b25e` |
| `src/mira/core/structure_old.py` | 836 | `254b25e` |
| `src/mira/processing/alignment_old.py` | 1747 | `38d2bd8` |
| `src/mira/processing/alignment_old1.py` | 1233 | `38d2bd8` |
| `src/mira/processing/alignment_old2.py` | 1252 | `38d2bd8` |
| `src/mira/processing/alignment_old3.py` | 1327 | `254b25e` |
| `src/mira/processing/detection_old.py` | 520 | `254b25e` |
| `src/mira/processing/phase_retrieval_old.py` | 701 | `254b25e` |
| `src/mira/processing/phase_retrieval_old1.py` | 568 | `254b25e` |
| `src/mira/processing/phase_retrieval_old2.py` | 602 | `254b25e` |
| `src/mira/processor_old.py` | 1641 | `38d2bd8` |
| `src/mira/processor_old2.py` | 1361 | `38d2bd8` |
| `src/mira/processor_old3.py` | 1378 | `254b25e` |
| `src/mira/processor_old4.py` | 1579 | `254b25e` |
| `src/mira/processor_old5.py` | 1821 | `254b25e` |
| `src/mira/processor_old^1.py` | 1487 | `38d2bd8` |
| `src/mira/visualization/plotter_2d_old.py` | 883 | `38d2bd8` |
| `src/mira/visualization/plotter_2d_old1.py` | 1244 | `38d2bd8` |
| `src/mira/visualization/plotter_2d_old2.py` | 1351 | `38d2bd8` |

## 3. Bland-Altman modules

The list in MI-01 says "`bland_altman_snippet.py` if superseded by `bland_altman.py`". It is the other way
round: `bland_altman_snippet.py` is the newer version (it keeps the mask of masked arrays),
`bland_altman.py` the older one. Since MI-72 both are thin compatibility wrappers around
`mira.analysis.comparison.bland_altman_compute` and `mira.visualization.plotter_stats.plot_bland_altman`.
Proposal: keep `bland_altman.py` as the compatibility module for old scripts and move
`bland_altman_snippet.py` to the attic.

## 4. Other candidates (not in the MI-01 list, for the maintainer to decide)

- `src/mira/MIRA_composition.py`: an unused sketch of a future data model. It may be useful for MI-10
  (decision M2); keep until M2 is decided.
- `tests/_test/test_dhm.py`: empty file.
