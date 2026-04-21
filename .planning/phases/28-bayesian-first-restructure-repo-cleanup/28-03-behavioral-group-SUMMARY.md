---
phase: 28-bayesian-first-restructure-repo-cleanup
plan: "03"
subsystem: scripts-organization
tags: [git-mv, behavioral-analysis, refactor, pipeline]

# Dependency graph
requires:
  - phase: 28-01
    provides: src/rlwm/fitting/ consolidation; shims deleted; call sites rewritten
provides:
  - scripts/behavioral/ package with 05-08 scripts moved via git mv
  - scripts/behavioral/__init__.py marker
  - Updated cluster/13_full_pipeline.slurm and run_data_pipeline.py to reference new paths
affects: [28-08, 28-10, 28-11]

# Tech tracking
tech-stack:
  added: []
  patterns: [behavioral scripts grouped under scripts/behavioral/ subdirectory]

key-files:
  created:
    - scripts/behavioral/__init__.py
    - scripts/behavioral/05_summarize_behavioral_data.py (moved)
    - scripts/behavioral/06_visualize_task_performance.py (moved)
    - scripts/behavioral/07_analyze_trauma_groups.py (moved)
    - scripts/behavioral/08_run_statistical_analyses.py (moved)
  modified:
    - cluster/13_full_pipeline.slurm
    - run_data_pipeline.py

key-decisions:
  - "git mv preserves history; no content changes to scripts"
  - "Code-level path references updated (cluster slurm + run_data_pipeline.py); doc-level refs deferred to plan 28-11"
  - "Behavioral script changes landed in concurrent 28-04 refactor commit (0f15fcf) due to wave execution order"

patterns-established:
  - "Behavioral pipeline scripts (05-08) live under scripts/behavioral/"

# Metrics
duration: 8min
completed: 2026-04-21
---

# Phase 28 Plan 03: Behavioral Group Summary

**scripts/behavioral/ package created; 05-08 scripts moved via git mv, path references updated in cluster slurm and run_data_pipeline.py**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-04-21T20:00:00Z
- **Completed:** 2026-04-21T20:08:00Z
- **Tasks:** 7
- **Files modified:** 7 (4 renames + 1 created + 2 modified)

## Accomplishments
- Created `scripts/behavioral/` as a Python package with `__init__.py`
- Moved all four behavioral pipeline scripts (05-08) via `git mv`, preserving full history
- Updated `cluster/13_full_pipeline.slurm` behavioral invocations to new paths
- Updated `run_data_pipeline.py` behavioral invocations to new paths
- All four scripts smoke-test clean from new location (--help passes)
- pytest test_v4_closure.py 3/3 PASS

## Task Commits

Plan 28-03 ran concurrently with plan 28-04. Due to wave execution ordering, the behavioral script renames landed inside commit `0f15fcf` (refactor(28-04)). No separate `refactor(28-03)` commit needed — work is present and verified in HEAD.

**Plan metadata:** docs(28-03) commit (this SUMMARY)

## Files Created/Modified
- `scripts/behavioral/__init__.py` - package marker, docstring only
- `scripts/behavioral/05_summarize_behavioral_data.py` - moved from scripts/
- `scripts/behavioral/06_visualize_task_performance.py` - moved from scripts/
- `scripts/behavioral/07_analyze_trauma_groups.py` - moved from scripts/
- `scripts/behavioral/08_run_statistical_analyses.py` - moved from scripts/
- `cluster/13_full_pipeline.slurm` - updated invocations for behavioral scripts
- `run_data_pipeline.py` - updated invocations for behavioral scripts

## Decisions Made
- Code-level path references (slurm, pipeline runner) updated immediately; doc-level in-script docstring examples deferred to plan 28-11 per plan spec.
- Behavioral 28-03 changes absorbed into 28-04 wave execution commit; no conflict or duplication.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Updated code-level path references in cluster slurm and pipeline runner**
- **Found during:** Task 1 (pre-flight grep)
- **Issue:** `cluster/13_full_pipeline.slurm` and `run_data_pipeline.py` had hardcoded `scripts/0{5,6,7,8}_*.py` invocations that would break after the move
- **Fix:** Updated both files to use `scripts/behavioral/0{5,6,7,8}_*.py` paths
- **Files modified:** cluster/13_full_pipeline.slurm, run_data_pipeline.py
- **Verification:** Confirmed grep finds no remaining code-level references; smoke tests pass
- **Committed in:** 0f15fcf (absorbed into 28-04 concurrent commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical — in-code path reference update)
**Impact on plan:** Necessary for pipeline correctness; matches plan scope (task 5 explicitly lists this).

## Issues Encountered
- Wave 2 execution order: plan 28-04 committed first and absorbed the behavioral git mv renames. All changes verified present in HEAD at commit `0f15fcf`. No functional impact; all verification criteria met.

## Next Phase Readiness
- `scripts/behavioral/` package ready for Wave 4 cluster slurm updates (plan 28-08)
- Doc-level path refs in script docstrings still use old paths; plan 28-11 handles those
- No blockers for downstream plans

---
*Phase: 28-bayesian-first-restructure-repo-cleanup*
*Completed: 2026-04-21*
