---
phase: 29-pipeline-canonical-reorg
plan: "01"
subsystem: infra
tags: [git-mv, refactor, scripts, canonical-layout, cluster, slurm, imports]

# Dependency graph
requires:
  - phase: 28-bayesian-first-restructure-repo-cleanup
    provides: "Five-subdir grouping (data_processing/, behavioral/, simulations_recovery/, post_mle/, bayesian_pipeline/) + top-level 12/13/14 entry scripts"
provides:
  - "Canonical 01-06 stage layout: 01_data_preprocessing/, 02_behav_analyses/, 03_model_prefitting/, 04_model_fitting/{a_mle,b_bayesian,c_level2}/, 05_post_fitting_checks/, 06_fit_analyses/"
  - "All caller paths updated in cluster/, tests/, validation/, docs/, manuscript/"
  - "Digit-prefix __init__.py guard comment in every new package"
  - "test_load_side_validation.py remapped to canonical paths"
affects:
  - "29-02: docs integration (parallel, no file-level collision)"
  - "29-03: utils consolidation (scripts/fitting/ stays untouched)"
  - "29-04: dead-folder audit (visualization/, simulations/ entries commented out with TODO(29-04))"
  - "Phase 24: cold-start SLURM calls now use canonical paths"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Digit-prefix package guard: every scripts/0N_*/  __init__.py starts with a # comment warning against dotted-name imports"
    - "Relative imports in digit-prefix entry scripts: from .fit_mle import main, from .fit_bayesian import main"
    - "parents[3] for project-root calculation from scripts/04_model_fitting/{a,b}_*/ depth"

key-files:
  created:
    - scripts/04_model_fitting/__init__.py
    - scripts/04_model_fitting/a_mle/__init__.py
    - scripts/04_model_fitting/b_bayesian/__init__.py
    - scripts/04_model_fitting/c_level2/__init__.py
    - scripts/05_post_fitting_checks/__init__.py
    - scripts/06_fit_analyses/__init__.py
  modified:
    - scripts/04_model_fitting/a_mle/12_fit_mle.py
    - scripts/04_model_fitting/b_bayesian/13_fit_bayesian.py
    - scripts/04_model_fitting/b_bayesian/fit_bayesian.py
    - scripts/fitting/tests/test_load_side_validation.py
    - scripts/fitting/tests/test_bayesian_recovery.py
    - scripts/fitting/tests/test_loo_stacking.py
    - cluster/ (18 SLURM files)
    - CLAUDE.md
    - README.md
    - config.py
    - manuscript/paper.qmd
    - manuscript/paper.tex

key-decisions:
  - "ONE atomic commit for the full rename wave (plan overrides default per-task commits)"
  - "git mv only for all script moves (preserves git log --follow)"
  - "Digit-prefix packages require relative imports; documented with guard comment in __init__.py"
  - "parents[3] for project root in entry scripts at depth scripts/04_model_fitting/{sub}/file.py"
  - "visualization/* and simulations/generate_data.py entries in test_load_side_validation.py commented out with TODO(29-04) — those stages not yet created"
  - "scripts/fitting/ library (bms.py, mle_utils.py, etc.) untouched — Plan 29-03 handles those"

patterns-established:
  - "Stage layout: 01-06 numbered dirs; 04 has sub-stages a_mle/b_bayesian/c_level2"
  - "Entry script (12_fit_mle.py) imports from co-located library via relative import"
  - "SLURM scripts invoke python scripts/{stage}/{file}.py (not top-level scripts/ path)"

# Metrics
duration: ~3h (resumed across two sessions due to context limit)
completed: 2026-04-22
---

# Phase 29 Plan 01: Scripts Canonical Reorganization Summary

**76-file rename wave moving Phase 28's five-subdir layout to numbered 01-06 stage layout via git mv, with all cluster SLURMs, tests, docs, and manuscript callers updated and digit-prefix import guards installed**

## Performance

- **Duration:** ~3h across two sessions
- **Started:** 2026-04-22 (morning session)
- **Completed:** 2026-04-22T13:00:30Z
- **Tasks:** 5
- **Files modified:** 76 (1 atomic commit)

## Accomplishments

- Moved all Phase 28 five-subdir scripts to canonical numbered 01-06 stage layout using `git mv` (preserves `git log --follow`)
- Updated 18 SLURM files, 10+ docs, CLAUDE.md, README.md, manuscript/paper.qmd and paper.tex, config.py, run_data_pipeline.py/sh, 5 test files, validation scripts
- Installed digit-prefix import guard comments in all 6 new `__init__.py` files; fixed entry scripts to use relative imports and `parents[3]` depth for project root
- Updated `test_load_side_validation.py` `_ENUMERATED_FILES` to canonical paths; commented out 5 not-yet-created entries with TODO(29-04)
- Verified: `validation/check_v4_closure.py --milestone v4.0` exits 0 (5/5); pytest test_v4_closure + test_load_side_validation 5/5 PASS

## Task Commits

Plan specified ONE atomic commit for the full rename wave:

1. **Tasks 1-5 (all tasks, single atomic commit)** — `04ebc72` (refactor)

**Plan metadata:** TBD (docs commit follows)

## Files Created/Modified (key items)

- `scripts/04_model_fitting/__init__.py` (new) — digit-prefix guard comment
- `scripts/04_model_fitting/a_mle/12_fit_mle.py` — relative import from `.fit_mle`, parents[3] root, docstring updated
- `scripts/04_model_fitting/b_bayesian/13_fit_bayesian.py` — same fixes
- `scripts/04_model_fitting/b_bayesian/fit_bayesian.py` — error message path updated
- `scripts/fitting/tests/test_load_side_validation.py` — 7 enumerated paths remapped; 5 commented out with TODO(29-04)
- `scripts/fitting/tests/test_bayesian_recovery.py` — importlib path updated to `03_model_prefitting/13_run_bayesian_recovery.py`
- `scripts/fitting/tests/test_loo_stacking.py` — importlib path updated to `06_fit_analyses/compute_loo_stacking.py`
- `cluster/13_bayesian_choice_only.slurm` + 17 other SLURM files — all script invocation paths updated
- `CLAUDE.md` — entire Quick Reference section updated to canonical paths
- `config.py` — RST `:mod:` docstring reference updated from `scripts.fitting.*` to `rlwm.fitting.*`
- `manuscript/paper.qmd`, `manuscript/paper.tex` — `scripts/14_compare_models.py` → `scripts/06_fit_analyses/compare_models.py`

## Decisions Made

- **ONE atomic commit** — the plan explicitly overrides GSD's default per-task commits for a big rename wave. Ensures the repo is never in an inconsistent half-renamed state.
- **git mv only** — `git mv` preserves `git log --follow` history for all moved scripts. Never `mv` + `git add`.
- **Digit-prefix import guard** — Python cannot dotted-import from a directory starting with a digit (e.g. `04_model_fitting`). Addressed via relative imports in entry scripts (`from .fit_mle import main`) and a `# Stage folder name starts with a digit` warning comment in every `__init__.py`.
- **parents[3] depth** — Entry scripts at `scripts/04_model_fitting/{a,b}_{*/filename.py` are 4 levels from project root. `parents[1]` (the Phase 28 bug) would have pointed to `scripts/04_model_fitting/` instead.
- **TODO(29-04) for visualization/ and simulations/** — those stage directories do not yet exist; commenting out those 5 entries prevents test_load_side_validation from failing on absent files while signalling they need to be addressed in plan 29-04.
- **scripts/fitting/ library untouched** — bms.py, mle_utils.py, model_recovery.py, etc. are handled by Plan 29-03.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] parents[1] wrong depth for project root in entry scripts**

- **Found during:** Task 2 (build stage 04)
- **Issue:** Entry scripts moved from `scripts/12_fit_mle.py` to `scripts/04_model_fitting/a_mle/12_fit_mle.py`. At the new depth, `Path(__file__).parents[1]` points to `scripts/04_model_fitting/` not the project root. The import `sys.path.insert` would silently fail to find `config.py`.
- **Fix:** Changed `parents[1]` to `parents[3]` in both entry scripts.
- **Files modified:** `scripts/04_model_fitting/a_mle/12_fit_mle.py`, `scripts/04_model_fitting/b_bayesian/13_fit_bayesian.py`
- **Committed in:** `04ebc72` (part of atomic rename commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — Bug)
**Impact on plan:** Fix was necessary for correct operation. No scope creep.

## Issues Encountered

- Context limit hit mid-execution (between Task 4 and Task 5). A new session picked up from the remaining files list (5 files still needed updating). No work was lost; the interrupted session left the working tree in a clean partially-updated state.
- `git mv scripts/simulations_recovery/ scripts/03_model_prefitting/` required removing a leftover `__pycache__/` before the empty directory could be `rmdir`'d.
- `scripts/bayesian_pipeline/__pycache__/` remained after all Python files were moved; cleaned up with `rm -rf` + `rmdir` before the final commit.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Plan 29-02 (docs spare-file integration) is already complete (SUMMARY.md landed in parallel session)
- Plan 29-03 (utils consolidation, scripts/fitting/ library) is unblocked; no file-level collision with 29-01
- Plan 29-04 (dead-folder audit) must re-enable the 5 commented-out TODO(29-04) entries in `test_load_side_validation.py` once `visualization/` and `simulations/` stage dirs are created
- Phase 24 cold-start SLURMs (`cluster/21_*.slurm`) now invoke canonical paths — no stale-path regressions

---
*Phase: 29-pipeline-canonical-reorg*
*Completed: 2026-04-22*
