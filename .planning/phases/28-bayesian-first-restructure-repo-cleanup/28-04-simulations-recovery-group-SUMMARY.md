---
phase: 28-bayesian-first-restructure-repo-cleanup
plan: "04"
subsystem: repo-structure
tags: [git-mv, slurm, simulations, parameter-recovery, ppc]

requires:
  - phase: 28-01-src-consolidation
    provides: scripts/fitting/model_recovery.py in its final location (no further moves)

provides:
  - scripts/simulations_recovery/ package with 4 grouped scripts (09 x2, 10, 11)
  - cluster/09_ppc_gpu.slurm updated to invoke simulations_recovery path

affects:
  - 28-06-bayesian-pipeline-group (will update cluster/13_full_pipeline.slurm stale refs)
  - 28-08-cluster-consolidation (cluster/13_full_pipeline.slurm still has stale 09/11 refs — Wave 4)
  - 28-12-end-of-phase-verification (must scan simulations_recovery/ for REFAC-05 closure)

tech-stack:
  added: []
  patterns:
    - "Numbered pipeline scripts grouped by domain under scripts/<domain>/ subdirectories"

key-files:
  created:
    - scripts/simulations_recovery/__init__.py
    - scripts/simulations_recovery/09_generate_synthetic_data.py  (git mv)
    - scripts/simulations_recovery/09_run_ppc.py  (git mv)
    - scripts/simulations_recovery/10_run_parameter_sweep.py  (git mv)
    - scripts/simulations_recovery/11_run_model_recovery.py  (git mv)
  modified:
    - cluster/09_ppc_gpu.slurm

key-decisions:
  - "cluster/11_recovery_gpu.slurm was not updated because it invokes scripts/fitting/model_recovery.py directly (not scripts/11_run_model_recovery.py) — no stale path present"
  - "cluster/13_full_pipeline.slurm stale refs to scripts/09_run_ppc.py and scripts/11_run_model_recovery.py deferred to Wave 4 plan 28-08 per parallel safety constraint"

patterns-established:
  - "Domain grouping: simulations + recovery scripts live together under scripts/simulations_recovery/"

duration: 15min
completed: 2026-04-21
---

# Phase 28 Plan 04: Simulations/Recovery Group Summary

**Four simulations/recovery scripts (09 x2, 10, 11) moved into scripts/simulations_recovery/ via git mv; cluster/09_ppc_gpu.slurm invocation updated to match new path**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-04-21T19:39:47Z
- **Completed:** 2026-04-21T19:54:48Z
- **Tasks:** 6 (pre-flight, mkdir, git mv x4, SLURM update, smoke test, commit)
- **Files modified:** 6 (4 git mv + 1 new __init__.py + 1 SLURM)

## Accomplishments

- All four scripts in `scripts/simulations_recovery/`: `09_generate_synthetic_data.py`, `09_run_ppc.py`, `10_run_parameter_sweep.py`, `11_run_model_recovery.py`
- Empty `scripts/simulations_recovery/__init__.py` makes the directory a proper Python package
- `cluster/09_ppc_gpu.slurm` updated in both comment header (line 6) and invocation command (line 49)
- Smoke tests passed: `--help` works for both 09_run_ppc.py and 11_run_model_recovery.py from their new paths
- pytest 3/3 on `test_v4_closure.py` unaffected

## Task Commits

All tasks merged into one atomic commit per `<critical_commit_directive>`:

1. **Tasks 1-6 (all tasks)** - `0f15fcf` (refactor)

**Plan metadata:** [this docs commit]

## Files Created/Modified

- `scripts/simulations_recovery/__init__.py` - empty package marker (new)
- `scripts/simulations_recovery/09_generate_synthetic_data.py` - git mv from scripts/
- `scripts/simulations_recovery/09_run_ppc.py` - git mv from scripts/
- `scripts/simulations_recovery/10_run_parameter_sweep.py` - git mv from scripts/
- `scripts/simulations_recovery/11_run_model_recovery.py` - git mv from scripts/
- `cluster/09_ppc_gpu.slurm` - updated `# Maps to:` comment and `CMD=` invocation

## Decisions Made

1. `cluster/11_recovery_gpu.slurm` not updated — its Python invocation calls `scripts/fitting/model_recovery.py` directly, not `scripts/11_run_model_recovery.py`. There was no stale path to fix.
2. `cluster/13_full_pipeline.slurm` contains stale refs to `scripts/09_run_ppc.py` (line 195) and `scripts/11_run_model_recovery.py` (line 198). Per parallel wave safety, `cluster/13_*.slurm` is Wave 4 scope (plans 28-06/28-08). Those refs are deferred.

## Deviations from Plan

### Known Scope Gaps (deferred to later waves)

**cluster/13_full_pipeline.slurm stale refs**
- **Found during:** Task 1 pre-flight grep
- **Issue:** `cluster/13_full_pipeline.slurm` lines 195 and 198 still reference old `scripts/09_run_ppc.py` and `scripts/11_run_model_recovery.py` paths
- **Action:** Deferred — `cluster/13_*.slurm` is Wave 4 scope per `<parallel_wave_safety>`. Plans 28-06 and 28-08 will fix.
- **Impact:** The plan's verification invariant `! grep -rn "scripts/09_run_ppc\|scripts/11_run_model_recovery" cluster/ --include="*.slurm"` will fail until Wave 4 lands. This is expected and accepted.

**cluster/README.md stale refs**
- **Found during:** Task 1 pre-flight grep
- **Issue:** `cluster/README.md` lines 14-15 reference old script paths
- **Action:** Not in `files_modified` scope. Deferred to docs refresh (plan 28-11).

---

**Total deviations:** 0 auto-fixed, 2 known deferred (both Wave 4+ scope, no rule violation)
**Impact on plan:** No scope creep. Both deferrals governed by parallel wave safety constraints.

## Issues Encountered

None — `git mv` preserved all imports cleanly; `scripts.fitting.model_recovery` import resolves correctly from new paths because `PYTHONPATH` includes the project root, which is unchanged.

## User Setup Required

None.

## Next Phase Readiness

- `scripts/simulations_recovery/` is ready for any plan that needs to call these scripts
- Wave 4 plans (28-06, 28-08) must update `cluster/13_full_pipeline.slurm` lines 195 and 198
- REFAC-05 is closed from a file-structure perspective; stale SLURM comment in 13_full_pipeline.slurm is the only remaining artifact

---
*Phase: 28-bayesian-first-restructure-repo-cleanup*
*Completed: 2026-04-21*
