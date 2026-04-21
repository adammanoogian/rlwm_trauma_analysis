---
phase: 05-parameter-recovery
plan: 04
subsystem: validation
tags: [posterior-predictive-check, ppc, behavioral-comparison, model-validation, synthetic-data]

# Dependency graph
requires:
  - phase: 05-01
    provides: Core parameter recovery infrastructure (sample_parameters, generate_synthetic_participant)
  - phase: 05-02
    provides: MLE fitting results in output/mle_results/{model}_individual_fits.csv
provides:
  - PPC mode in model_recovery.py for generating synthetic data from fitted params
  - Behavioral comparison metrics (accuracy by set-size, learning curves, post-reversal)
  - Overlay plots comparing real vs synthetic behavioral patterns
affects: [05-05-ppc-script, model-validation, behavioral-analysis]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "PPC mode as CLI flag in existing recovery script (--mode ppc)"
    - "Behavioral comparison using pandas groupby and aggregation"
    - "KDE overlay plots for distribution comparison"

key-files:
  created: []
  modified:
    - scripts/fitting/model_recovery.py
    - scripts/utils/plotting_utils.py

key-decisions:
  - "PPC mode outputs to output/ppc/ and figures/ppc/ instead of output/recovery/"
  - "Auto-detect fitted params path from model name if not specified"
  - "Use participant sona_id as seed for reproducible synthetic data generation"
  - "Behavioral metrics: overall accuracy, set-size accuracy, learning curve, post-reversal"

patterns-established:
  - "plot_behavioral_comparison() creates 3 standard overlay plots (set-size, learning curve, distribution)"
  - "compare_behavior() returns DataFrame with real, synthetic, and difference rows"

# Metrics
duration: 15min
completed: 2026-02-06
---

# Phase 05 Plan 04: PPC Mode Implementation Summary

**PPC mode in model_recovery.py generates synthetic data from MLE-fitted parameters and compares behavioral patterns via overlay plots**

## Performance

- **Duration:** 15 minutes
- **Started:** 2026-02-06 19:29:31
- **Completed:** 2026-02-06 19:45:00
- **Tasks:** 5
- **Files modified:** 2

## Accomplishments
- PPC mode added to model_recovery.py with --mode {recovery,ppc} CLI argument
- Behavioral comparison metrics compute accuracy by set-size, learning curves, and post-reversal patterns
- Overlay plots (set-size bars, learning curve lines, KDE distributions) visualize real vs synthetic agreement

## Task Commits

Each task was committed atomically:

1. **Task 1: Add PPC CLI arguments to model_recovery.py** - `006fb17` (feat)
2. **Task 2: Add load_fitted_params() function** - `4ca9a0e` (feat)
3. **Task 3: Add run_posterior_predictive_check() function** - `a2bb77b` (feat)
4. **Task 4: Add compare_behavior() function** - `b3603ea` (feat)
5. **Task 5: Add plot_behavioral_comparison() to plotting_utils.py** - `3a3a4ed` (feat)

## Files Created/Modified
- `scripts/fitting/model_recovery.py` - Added PPC mode, load_fitted_params(), run_posterior_predictive_check(), compare_behavior()
- `scripts/utils/plotting_utils.py` - Added plot_behavioral_comparison() for overlay plots

## Decisions Made

1. **PPC output directories:** PPC mode outputs to `output/ppc/{model}/` and `figures/ppc/{model}/` instead of `output/recovery/` to separate validation artifacts from parameter recovery results
2. **Auto-detection of fitted params:** If `--fitted-params` not specified in PPC mode, auto-detects from `output/mle_results/{model}_individual_fits.csv`
3. **Reproducible synthetic data:** Use participant sona_id as seed for generate_synthetic_participant() to ensure reproducible synthetic data
4. **Behavioral metrics selection:** Chose overall accuracy, accuracy by set-size (2,3,5,6), learning curve (early vs late blocks), and post-reversal accuracy (first 5 trials per block) following Wilson & Collins (2019)
5. **Difference row in comparison:** compare_behavior() returns 3-row DataFrame with real, synthetic, and difference for easy quantitative assessment

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- PPC mode fully functional and ready for testing with MLE-fitted parameters
- Can run `python scripts/fitting/model_recovery.py --mode ppc --model wmrl_m3` to validate model
- Next plan (05-05) can create thin CLI wrapper script (11_run_ppc.py) calling these functions
- Behavioral comparison metrics ready for quantitative model evaluation

---
*Phase: 05-parameter-recovery*
*Completed: 2026-02-06*
