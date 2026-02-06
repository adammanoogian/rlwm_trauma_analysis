---
phase: 05-parameter-recovery
plan: 05
subsystem: validation
tags: [ppc, model-recovery, script-09, orchestrator, pipeline, confusion-matrix, gpu-cluster]

# Dependency graph
requires:
  - phase: 05-04
    provides: PPC mode implementation (run_posterior_predictive_check)
  - phase: 05-02
    provides: MLE fitting pipeline (Script 12)
provides:
  - run_model_recovery_check() fits all models to synthetic data and reports winner
  - Script 09 (09_run_ppc.py) orchestrates full PPC pipeline
  - GPU SLURM script (cluster/run_ppc_gpu.slurm) for cluster execution
  - Complete PPC validation workflow with behavioral comparison + model recovery
affects: [model-validation, ppc-workflow, cluster-execution, pipeline-scripts]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Script 09 orchestrates multi-step workflow (generate -> analyze -> compare -> model recovery)"
    - "Model recovery via subprocess calls to Script 12 (fit all models) and comparison"
    - "Exit code 0 if all pass, 1 if any fail for automation"

key-files:
  created:
    - scripts/09_run_ppc.py
    - cluster/run_ppc_gpu.slurm
  modified:
    - scripts/fitting/model_recovery.py

key-decisions:
  - "run_model_recovery_check() uses subprocess to call Script 12 for each model"
  - "Model recovery determines winner by summing AIC across participants (lowest wins)"
  - "Script 09 supports --model all for batch validation"
  - "Fixed load_fitted_params to handle both sona_id and participant_id column names"
  - "PPC outputs confirmed: synthetic_trials.csv, behavioral_comparison.csv, comparison plots"

patterns-established:
  - "Pipeline scripts orchestrate via subprocess calls to other numbered scripts"
  - "Exit codes indicate validation success (0) or failure (1) for automation"
  - "Auto-detect ID column name and normalize to sona_id internally"

# Metrics
duration: 9min
completed: 2026-02-06
---

# Phase 05 Plan 05: Model Recovery Evaluation and PPC Orchestrator Summary

**Script 09 orchestrates full PPC pipeline with model recovery check verifying generative model wins via AIC comparison**

## Performance

- **Duration:** 9 minutes
- **Started:** 2026-02-06T18:42:05Z
- **Completed:** 2026-02-06T18:51:06Z
- **Tasks:** 4
- **Files modified:** 3

## Accomplishments
- run_model_recovery_check() fits all models to synthetic data and reports if generative model wins
- Script 09 created as user-facing PPC orchestrator calling run_posterior_predictive_check() and run_model_recovery_check()
- GPU SLURM script enables cluster PPC execution with 8-hour time limit and GPU acceleration
- End-to-end pipeline verified with qlearning model (behavioral comparison + plots generated)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add run_model_recovery_check() function** - `e861a77` (feat)
2. **Task 2: Create Script 09 as PPC orchestrator** - `cef2033` (feat)
3. **Task 3: Create GPU SLURM script for PPC** - `161a73b` (feat)
4. **Task 4: End-to-end verification** - `9fe1e5c` (fix) + `b1ac94e` (test)

**Deviation fixes:** Auto-fixed bugs found during Task 4 verification (Rule 1 - bugs)

## Files Created/Modified
- `scripts/fitting/model_recovery.py` - Added run_model_recovery_check(), fixed load_fitted_params column access bugs
- `scripts/09_run_ppc.py` - Full PPC orchestrator script (202 lines)
- `cluster/run_ppc_gpu.slurm` - GPU SLURM script for cluster execution

## Decisions Made

1. **Model recovery via subprocess:** run_model_recovery_check() uses subprocess.run() to call Script 12 for each model (qlearning, wmrl, wmrl_m3) rather than importing fitting functions directly - maintains separation between pipeline scripts
2. **Winner by summed AIC:** Determine winning model by summing AIC across all participants (lowest total wins) - follows Senta et al. (2025) methodology
3. **Script 09 batch support:** Added --model all flag to run PPC for all models sequentially with final summary
4. **ID column auto-detection:** load_fitted_params() checks for both 'sona_id' and 'participant_id' columns and normalizes to sona_id internally for compatibility with both recovery and MLE results formats
5. **Exit code semantics:** Script 09 returns 0 if all model recoveries pass, 1 if any fail (enables automated validation in CI/cluster)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed row.columns AttributeError**
- **Found during:** Task 4 (End-to-end verification)
- **Issue:** load_fitted_params() accessed row.columns instead of df.columns, causing AttributeError: 'Series' object has no attribute 'columns'
- **Fix:** Changed `if p in row.columns` to `if p in df.columns`
- **Files modified:** scripts/fitting/model_recovery.py
- **Verification:** PPC test ran successfully without AttributeError
- **Committed in:** 9fe1e5c (fix commit)

**2. [Rule 1 - Bug] Fixed KeyError for sona_id column**
- **Found during:** Task 4 (End-to-end verification)
- **Issue:** MLE results use 'participant_id' column but load_fitted_params() expected 'sona_id', causing KeyError
- **Fix:** Added auto-detection of ID column name (checks for 'sona_id' first, falls back to 'participant_id') and normalizes to 'sona_id' key internally
- **Files modified:** scripts/fitting/model_recovery.py
- **Verification:** PPC test loaded MLE results successfully and generated all outputs
- **Committed in:** 9fe1e5c (fix commit)

---

**Total deviations:** 2 auto-fixed (2 bugs)
**Impact on plan:** Both bugs discovered during end-to-end testing and fixed to enable PPC pipeline to work with existing MLE results format. No scope creep.

## Issues Encountered

None - bugs caught and fixed during Task 4 verification before completion.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Complete PPC pipeline operational (behavioral comparison + model recovery)
- User can run `python scripts/09_run_ppc.py --model wmrl_m3` for full validation
- GPU cluster execution via `sbatch cluster/run_ppc_gpu.slurm`
- Phase 5 complete: parameter recovery (05-01 to 05-03) + PPC validation (05-04 to 05-05)
- Ready for Phase 6: Cluster Monitoring (monitor long-running jobs, results aggregation)

---
*Phase: 05-parameter-recovery*
*Completed: 2026-02-06*
