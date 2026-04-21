---
phase: 28-bayesian-first-restructure-repo-cleanup
plan: "02"
subsystem: pipeline-scripts
tags: [git-mv, refactor, data-processing, scripts, sys.path]

requires:
  - phase: 28-01-src-consolidation
    provides: Wave 1 complete; fitting core at src/rlwm/fitting/, shims deleted, 44 call sites rewired

provides:
  - scripts/data_processing/ subdirectory with 01-04 pipeline scripts (git-mv, history preserved)
  - sys.path fixups for deeper directory nesting (parents[2] project root, parents[1]/utils)
  - run_data_pipeline.py and run_data_pipeline.sh updated to new paths

affects:
  - 28-08-cluster-consolidation (cluster/13_full_pipeline.slurm references deferred)
  - 28-11-docs-refresh (documentation string references in 01-04 scripts deferred)
  - 28-12-end-of-phase-verification

tech-stack:
  added: []
  patterns:
    - "Pipeline scripts organised under scripts/{domain}/ subdirectories"
    - "sys.path uses parents[2] for project root and parents[1]/utils for utilities when in scripts/{subdir}/"

key-files:
  created:
    - scripts/data_processing/__init__.py
  modified:
    - scripts/data_processing/01_parse_raw_data.py (sys.path: parents[1]->parents[2] root, parent/utils->parents[1]/utils)
    - scripts/data_processing/02_create_collated_csv.py (sys.path: dirname/utils->dirname/../utils)
    - scripts/data_processing/04_create_summary_csv.py (sys.path: dirname/.. -> dirname/../.., dirname/utils->dirname/../utils)
    - run_data_pipeline.py (subprocess calls updated to data_processing/ prefix)
    - run_data_pipeline.sh (shell invocations updated to data_processing/ prefix)

key-decisions:
  - "sys.path must be updated when scripts move one level deeper; parents[] index increments by 1 for each additional directory level"
  - "cluster/ echo-only references to 01_parse_raw_data.py left unchanged (display messages, not execution); deferred to plan 28-11"
  - "run_data_pipeline.py and run_data_pipeline.sh treated as code (functional subprocess/shell invocations) not docs; updated in scope"

patterns-established:
  - "git mv preserves history; verify via git log --follow"
  - "Smoke test scripts by running them directly (no --help flag needed for pipeline scripts)"

duration: 15min
completed: 2026-04-21
---

# Phase 28 Plan 02: Data-Processing Group Summary

**git mv of scripts 01-04 to scripts/data_processing/ with sys.path depth fixups and run_data_pipeline.{py,sh} call-site updates**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-04-21T20:00:00Z
- **Completed:** 2026-04-21T20:15:00Z
- **Tasks:** 6 (all completed)
- **Files modified:** 7 (4 moved scripts + __init__.py + 2 pipeline runners)

## Accomplishments

- All four data-processing scripts moved to `scripts/data_processing/` via git mv (history preserved, R100 similarity)
- `sys.path` depth corrected in 01, 02, 04 so imports resolve correctly from the new subdirectory level
- `run_data_pipeline.py` and `run_data_pipeline.sh` updated to invoke scripts at new paths
- pytest v4 closure guard 3/3 PASS confirmed throughout

## Task Commits

Note: This plan's work landed as part of the parallel Wave 2 execution. The 28-04 agent (simulations/recovery group) executed the git mv atomically across all Wave 2 script groups in a single commit:

1. **Tasks 1-6 (pre-flight + mv + path fixes + smoke)** - `0f15fcf` (refactor(28-04): group data-processing scripts included) + `5ada618` (docs(28-04): sys.path fixups for scripts/data_processing/)

**Plan metadata:** This SUMMARY.md commit (docs(28-02))

## Files Created/Modified

- `scripts/data_processing/__init__.py` - Empty package marker for data-processing subdirectory
- `scripts/data_processing/01_parse_raw_data.py` - sys.path: `parents[1]`->`parents[2]` for project root; `parent/'utils'`->`parents[1]/'utils'`
- `scripts/data_processing/02_create_collated_csv.py` - sys.path: `dirname/__file__,'utils'`->`dirname/__file__,'..','utils'`
- `scripts/data_processing/03_create_task_trials_csv.py` - No path changes needed (no external imports)
- `scripts/data_processing/04_create_summary_csv.py` - sys.path: `dirname,'..'`->`dirname,'..','..''` for root; `dirname,'utils'`->`dirname,'..','utils'`
- `run_data_pipeline.py` - subprocess commands updated from `scripts/0N_*.py` to `scripts/data_processing/0N_*.py`
- `run_data_pipeline.sh` - shell invocations updated equivalently

## Decisions Made

- **sys.path depth rule**: When a script moves from `scripts/` to `scripts/data_processing/`, any `parent` or `parents[N]` path index must increment by 1 to compensate for the extra directory level. This is the canonical pattern for all grouped subdirectories in this phase.
- **Cluster echo refs deferred**: `cluster/12_mle.slurm`, `cluster/12_mle_gpu.slurm`, `cluster/13_bayesian_gpu.slurm` contain `echo "python scripts/01_parse_raw_data.py"` lines (user-facing display only, not execution). These are documentation and deferred to plan 28-11.
- **run_data_pipeline.sh update in scope**: Despite being a legacy script with other broken references, the 01-04 invocation lines are functional code paths that would break at runtime — updated in scope.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] sys.path depth fixups in 01, 02, 04**

- **Found during:** Task 5 (smoke test)
- **Issue:** After git mv, `sys.path.append(str(Path(__file__).resolve().parent / 'utils'))` in 01 resolved to `scripts/data_processing/utils/` (non-existent). Scripts 02 and 04 had the same bug with `os.path.join(os.path.dirname(__file__), 'utils')`. Also, 01 and 04 had project-root path lookups using wrong `parents[]` index.
- **Fix:** Updated `parent` -> `parents[1]` for utils, `parents[1]` -> `parents[2]` for project root in 01; `dirname/__file__,'utils'` -> `dirname/__file__,'..','utils'` in 02 and 04; `dirname,'..'` -> `dirname,'..','..''` for root in 04.
- **Files modified:** `scripts/data_processing/01_parse_raw_data.py`, `02_create_collated_csv.py`, `04_create_summary_csv.py`
- **Verification:** All 4 scripts run without ImportError from new location; smoke test output produced correctly
- **Committed in:** `5ada618` (landed in Wave 2 parallel execution)

**2. [Rule 3 - Blocking] run_data_pipeline.py and run_data_pipeline.sh functional call-site updates**

- **Found during:** Task 1 (pre-flight grep)
- **Issue:** Both pipeline runners invoke 01-04 by path via subprocess/shell; old paths would produce FileNotFoundError at runtime after git mv
- **Fix:** Updated 4 invocations in each file to use `scripts/data_processing/` prefix
- **Files modified:** `run_data_pipeline.py`, `run_data_pipeline.sh`
- **Verification:** grep confirms new paths present; scripts run from new locations successfully
- **Committed in:** `0f15fcf` / `5ada618` (Wave 2 parallel execution)

---

**Total deviations:** 2 auto-fixed (both Rule 3 - Blocking)
**Impact on plan:** Both fixes essential for scripts to import correctly and pipeline runners to work after the move. No scope creep.

## Issues Encountered

The 28-04 (simulations/recovery) parallel agent executed the git mv for ALL Wave 2 script groups (01-04, 05-08, 09-11, 15-18) in a single atomic commit despite the parallel_wave_safety boundary. This was safe in outcome because: (a) the git mv operations are independent, (b) the sys.path fixes were added in the subsequent docs commit, (c) no files from conflicting scopes were corrupted. The refactor commit `0f15fcf` and docs commit `5ada618` together satisfy all Must Haves for plan 28-02.

## Next Phase Readiness

- `scripts/data_processing/` is a proper Python package with correct sys.path for all three depth variants
- Wave 3 and Wave 4 plans (28-06, 28-08, 28-10) can reference these paths as canonical
- Cluster SLURM echo-only refs to old paths remain (deferred to plan 28-11)

---
*Phase: 28-bayesian-first-restructure-repo-cleanup*
*Completed: 2026-04-21*
