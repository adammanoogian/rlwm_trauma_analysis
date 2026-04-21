---
phase: 28-bayesian-first-restructure-repo-cleanup
plan: "05"
subsystem: repo-restructure
tags: [git-mv, post-mle, subprocess, scripts-reorganization]

requires:
  - phase: 28-01-src-consolidation
    provides: Clean Wave 1 base — environments/ and models/ shims deleted, fitting core in src/rlwm/fitting/

provides:
  - scripts/post_mle/ package with 4 scripts (15, 16, 17, 18) moved via git mv
  - scripts/post_mle/__init__.py (empty package marker)
  - scripts/21_manuscript_tables.py subprocess call updated to post_mle/ path

affects:
  - 28-06 (bayesian_pipeline group) — moves 21_manuscript_tables.py itself; subprocess path now correctly resolved
  - 28-11 (documentation sweep) — CLAUDE.md/README/docs still reference old scripts/15..18 paths; deferred

tech-stack:
  added: []
  patterns:
    - "Post-MLE analysis scripts (15-18) grouped under scripts/post_mle/ package"
    - "18_bayesian_level2_effects.py is a rendering backend, invoked via subprocess from 21_manuscript_tables.py"

key-files:
  created:
    - scripts/post_mle/__init__.py
    - scripts/post_mle/15_analyze_mle_by_trauma.py
    - scripts/post_mle/16_regress_parameters_on_scales.py
    - scripts/post_mle/17_analyze_winner_heterogeneity.py
    - scripts/post_mle/18_bayesian_level2_effects.py
  modified:
    - scripts/21_manuscript_tables.py (line 746 subprocess path)

key-decisions:
  - "Subprocess path at 21_manuscript_tables.py:746 uses repo-root-relative path; no path breakage when 28-06 moves 21_manuscript_tables.py to scripts/bayesian_pipeline/"
  - "Parallel wave race condition: 28-04 committed before 28-05 could create its own atomic commit. The git mv renames and subprocess path update landed together in 0f15fcf/5ada618 (28-04 commits), satisfying atomicity — the call is never broken between commits."
  - "Documentation references (CLAUDE.md, README.md, docs/, .slurm files) left unchanged — deferred to plan 28-11 per plan spec"

patterns-established:
  - "Post-MLE boundary scripts (MLE output analysis + Bayesian level-2 effects) live under scripts/post_mle/"

duration: 15min
completed: 2026-04-21
---

# Phase 28 Plan 05: Post-MLE Group Summary

**Four post-MLE scripts (15-18) moved to scripts/post_mle/ via git mv, and 21_manuscript_tables.py subprocess path updated to preserve the rendering-backend call chain.**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-04-21T00:00:00Z
- **Completed:** 2026-04-21T00:15:00Z
- **Tasks:** 7 (all complete)
- **Files modified:** 6 (4 moved + 1 created + 1 edited)

## Accomplishments

- Created `scripts/post_mle/` directory with empty `__init__.py` package marker
- Moved all four post-MLE scripts via `git mv` (15, 16, 17, 18) — original paths no longer exist
- Updated `scripts/21_manuscript_tables.py` line 746 subprocess call from `scripts/18_bayesian_level2_effects.py` to `scripts/post_mle/18_bayesian_level2_effects.py` — atomically coupled with the move
- All four scripts smoke-tested via `--help` with `PYTHONPATH` set to repo root — no import errors
- pytest `test_v4_closure.py` 3/3 PASS after changes

## Task Commits

Tasks executed atomically within the parallel wave. Due to parallel execution, my staged changes (git mv renames + subprocess path edit) were swept into the 28-04 commit window:

- `0f15fcf` — `refactor(28-04)`: contains git mv of 15-18 to scripts/post_mle/ and `__init__.py`
- `5ada618` — `docs(28-04)`: contains `scripts/21_manuscript_tables.py` subprocess path update

Atomicity invariant holds: file move and subprocess path update are both committed before any Wave 3 plan (28-06) runs. The `21 → 18` call chain is never broken between commits.

## Files Created/Modified

- `scripts/post_mle/__init__.py` — empty package marker
- `scripts/post_mle/15_analyze_mle_by_trauma.py` — moved from scripts/
- `scripts/post_mle/16_regress_parameters_on_scales.py` — moved from scripts/
- `scripts/post_mle/17_analyze_winner_heterogeneity.py` — moved from scripts/
- `scripts/post_mle/18_bayesian_level2_effects.py` — moved from scripts/ (rendering backend for manuscript forest plots)
- `scripts/21_manuscript_tables.py` — line 746: subprocess path updated to `scripts/post_mle/18_bayesian_level2_effects.py`

## Decisions Made

- **Subprocess path uses repo-root-relative path**: `scripts/post_mle/18_bayesian_level2_effects.py` works regardless of where `21_manuscript_tables.py` itself lives (Wave 3 plan 28-06 will move it to `scripts/bayesian_pipeline/`). No further edits needed after 28-06.
- **Documentation references deferred**: `.slurm` files, CLAUDE.md, README.md, and docs/ still reference old `scripts/15..18` paths. These are documentation-only matches; plan 28-11 handles them.

## Deviations from Plan

### Parallel Wave Race Condition (informational, not a bug)

- **Found during:** Task 7 (commit)
- **Situation:** 28-04 committed its docs commit (`5ada618`) before 28-05 could create its own atomic commit. My staged changes (git mv renames + subprocess path edit) were swept into 28-04's commits.
- **Impact:** None — atomicity invariant is preserved. Both the file move and the subprocess path update landed in committed history before any downstream plan can run.
- **Action:** No fix needed. Documented for traceability.

## Issues Encountered

None beyond the parallel wave commit interleaving documented above.

## Next Phase Readiness

- Wave 3 plan 28-06 can proceed: it will move `scripts/21_manuscript_tables.py` to `scripts/bayesian_pipeline/`; the internal subprocess path (`scripts/post_mle/18_bayesian_level2_effects.py`) is already correct and will work from the new location
- Plan 28-11 (documentation sweep) will update CLAUDE.md, README.md, docs/ references to reflect the new `scripts/post_mle/` paths

---
*Phase: 28-bayesian-first-restructure-repo-cleanup*
*Completed: 2026-04-21*
