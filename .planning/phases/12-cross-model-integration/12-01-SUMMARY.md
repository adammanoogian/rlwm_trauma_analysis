---
phase: 12-cross-model-integration
plan: 01
subsystem: validation
tags: [model-recovery, cross-model, AIC, confusion-matrix, identifiability]

# Dependency graph
requires:
  - phase: 08-m5-rl-forgetting
    provides: wmrl_m5 model implementation and pipeline integration
  - phase: 09-m6a-stimulus-specific-perseveration
    provides: wmrl_m6a model implementation and pipeline integration
  - phase: 10-m6b-dual-perseveration
    provides: wmrl_m6b model implementation and pipeline integration
provides:
  - run_model_recovery_check() with configurable comparison_models (default 6 choice-only)
  - run_cross_model_recovery() producing confusion matrix across all generating models
  - Script 11 --mode cross-model flag for end-to-end cross-model recovery
  - CHOICE_ONLY_MODELS constant excluding M4
affects: [12-cross-model-integration]

# Tech tracking
tech-stack:
  added: []
  patterns: [cross-model confusion matrix validation, temp directory subprocess fitting]

key-files:
  modified:
    - scripts/fitting/model_recovery.py
    - scripts/11_run_model_recovery.py

key-decisions:
  - "M4 excluded from cross-model AIC comparison (joint likelihood incommensurable with choice-only)"
  - "Confusion matrix uses plurality criterion (generator must win >50% of datasets)"
  - "Temp directories used for per-dataset synthetic data to avoid disk clutter"

patterns-established:
  - "Cross-model recovery pattern: generate-fit-compare loop with confusion matrix aggregation"
  - "CHOICE_ONLY_MODELS constant as single source of truth for model sets excluding M4"

# Metrics
duration: 19min
completed: 2026-04-03
---

# Phase 12 Plan 01: Cross-Model Recovery Summary

**Configurable cross-model AIC recovery with 6-model confusion matrix and script 11 --mode cross-model flag**

## Performance

- **Duration:** 19 min
- **Started:** 2026-04-03T13:19:22Z
- **Completed:** 2026-04-03T13:38:10Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Extended run_model_recovery_check() with configurable comparison_models parameter defaulting to all 6 choice-only models
- Added run_cross_model_recovery() that generates from each model, fits all competitors, and produces confusion matrix
- Script 11 --mode cross-model flag works end-to-end with confusion matrix CSV output
- Smoke test confirmed M5 wins by AIC against all 6 choice-only models (dAIC ~40 over next best)

## Task Commits

Each task was committed atomically:

1. **Task 1: Extend run_model_recovery_check() with configurable comparison models** - `a57f7fc` (feat)
2. **Task 2: Add --mode cross-model to script 11 and run smoke test** - `c54ca10` (feat)

## Files Created/Modified
- `scripts/fitting/model_recovery.py` - Added comparison_models param, run_cross_model_recovery(), CHOICE_ONLY_MODELS constant
- `scripts/11_run_model_recovery.py` - Added --mode argparse, cross-model branch in main(), updated docstring

## Decisions Made
- M4 excluded from cross-model AIC comparison because joint choice+RT likelihood is incommensurable with choice-only models
- Confusion matrix uses plurality criterion: generator must win more than 50% of its datasets to PASS
- Temporary directories used for synthetic data per dataset to avoid cluttering the output directory
- n_starts parameter exposed through to subprocess call for optimization control in smoke tests

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed --output argument to 12_fit_mle.py subprocess**
- **Found during:** Task 2 (smoke test)
- **Issue:** run_model_recovery_check() passed a file path to --output (e.g., `mle_results/qlearning_individual_fits.csv`), but fit_mle.py expects a directory. This caused fit_mle to create a directory named `qlearning_individual_fits.csv` and write the actual CSV inside it, then the read failed with PermissionError on Windows
- **Fix:** Changed --output to pass the directory path (`mle_results_dir`) instead of a file path. fit_mle.py auto-generates the filename as `{model}_individual_fits.csv` inside the directory
- **Files modified:** scripts/fitting/model_recovery.py
- **Verification:** Smoke test completes without error, all models fit and results read correctly
- **Committed in:** c54ca10 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Bug fix necessary for correct subprocess invocation. No scope creep.

## Issues Encountered
None beyond the --output bug fixed above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Cross-model recovery infrastructure complete for all 6 choice-only models
- Full validation (N=50, n_datasets=10) should be run on cluster: `python scripts/11_run_model_recovery.py --mode cross-model --model all --n-subjects 50 --n-datasets 10 --n-jobs 8`
- Plan 12-02 can proceed with unified comparison table and manuscript integration

---
*Phase: 12-cross-model-integration*
*Completed: 2026-04-03*
