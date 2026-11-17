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
