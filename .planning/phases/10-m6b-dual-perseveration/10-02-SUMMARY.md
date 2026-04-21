---
phase: 10-m6b-dual-perseveration
plan: 02
subsystem: modelling
tags: [mle, parameter-recovery, model-comparison, downstream-pipeline, dual-kernel, stick-breaking]

# Dependency graph
requires:
  - phase: 10-m6b-dual-perseveration
    provides: WMRL_M6B_BOUNDS, WMRL_M6B_PARAMS, wmrl_m6b MLE fitting pipeline (10-01)
  - phase: 09-m6a-stimulus-specific-perseveration
    provides: M6a downstream extension pattern (elif wmrl_m6a everywhere)
provides:
  - M6b parameter recovery with dual-kernel synthetic generation (global + per-stimulus)
  - Stick-breaking decode in synthetic generation (kappa = kappa_total * kappa_share)
  - M6b in AIC/BIC comparison table (script 14 auto-detection via wmrl_m6b_individual_fits.csv)
  - Script 15 kappa_total/kappa_share display names and MODEL_CONFIG entry
  - Script 16 kappa_total_mean/kappa_share_mean param column dispatch
  - wmrl_m6b in 'all' expansion for scripts 11, 15, 16
affects:
  - 11-m4-lba (model comparison context — M6b is now fully integrated before M4 phase)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Dual-kernel recovery: M6b generate_synthetic_participant maintains BOTH global last_action (scalar) and per-stimulus last_actions (dict), both reset at block boundaries"
    - "Stick-breaking decode in generation mirrors decode in objectives: kappa = kappa_total * kappa_share; kappa_s = kappa_total * (1 - kappa_share)"
    - "Downstream extension pattern: add wmrl_m6b elif everywhere wmrl_m6a appears, additive only"
    - "Script 15 load_data() return tuple extended: now returns 8-tuple (qlearning, wmrl, wmrl_m3, surveys, groups, wmrl_m5, wmrl_m6a, wmrl_m6b)"

key-files:
  created: []
  modified:
    - scripts/fitting/model_recovery.py
    - scripts/11_run_model_recovery.py
    - scripts/14_compare_models.py
    - scripts/15_analyze_mle_by_trauma.py
    - scripts/16_regress_parameters_on_scales.py

key-decisions:
  - "Both global and per-stimulus kernels applied independently in synthetic generation: global kernel applied if last_action is not None AND kappa > 0.0; stim kernel applied if last_actions.get(stimulus) is not None AND kappa_s > 0.0; normalization done once after both"
  - "Dual kernel sharing the hybrid_probs copy: first kernel copies if needed, second kernel reuses the copy — avoids double copy overhead"
  - "Recovery CSV stores true_kappa_total/recovered_kappa_total and true_kappa_share/recovered_kappa_share (the 8 free params), NOT the decoded kappa/kappa_s"
  - "Script 15 load_data() return tuple extended to 8 values — all callers updated (only main() calls load_data())"

patterns-established:
  - "M6b extension pattern is now complete: model_recovery.py follows M6a elif chain pattern for all 5 dispatch functions"
  - "Script 14 patterns dict + argparse --mXX + load block = complete model registration"
  - "Script 15 defensive load (mle/ then output/) + conditional MODEL_CONFIG entry = robust to missing files"

# Metrics
duration: 21min
completed: 2026-04-03
---

# Phase 10 Plan 02: M6b Downstream Pipeline Integration Summary

**M6b dual-kernel parameter recovery (stick-breaking decode, BOTH global+per-stimulus tracking), and full downstream integration into scripts 14/15/16/11 — M6b in AIC/BIC table, kappa_total/kappa_share in trauma regressions**

## Performance

- **Duration:** 21 min
- **Started:** 2026-04-03T09:09:00Z
- **Completed:** 2026-04-03T09:30:51Z
- **Tasks:** 2/2
- **Files modified:** 5

## Accomplishments

- M6b parameter recovery generates synthetic data with BOTH global last_action and per-stimulus last_actions dict; stick-breaking decode (kappa = kappa_total * kappa_share) matches the likelihood exactly
- Recovery CSV has all 8 columns: true/recovered for kappa_total, kappa_share, plus 6 shared params
- Script 14 auto-detects wmrl_m6b_individual_fits.csv; M6b appears in AIC/BIC comparison table with correct 46/46 converged count; --m6b argparse arg for explicit path override
- Script 15 displays kappa_total (LaTeX: kappa_{total}) and kappa_share (LaTeX: kappa_{share}) with proper PARAM_NAMES display; wmrl_m6b MODEL_CONFIG entry active when fits file exists
- Script 16 renames kappa_total -> kappa_total_mean and kappa_share -> kappa_share_mean; param_cols dispatch covers all 8 M6b parameters; --model all runs all 6 models including M6b

## Task Commits

1. **Task 1: M6b parameter recovery (model_recovery.py + script 11)** - `9dd6bdd` (feat)
2. **Task 2: M6b downstream pipeline (scripts 14, 15, 16)** - `53afd10` (feat)

## Files Created/Modified

- `scripts/fitting/model_recovery.py` — WMRL_M6B_BOUNDS/WMRL_M6B_PARAMS import; get_param_names/sample_parameters/compute_recovery_metrics/plot_recovery_scatter/plot_distribution_comparison all add wmrl_m6b elif; generate_synthetic_participant has dual-kernel simulation with stick-breaking decode; CLI choices updated
- `scripts/11_run_model_recovery.py` — wmrl_m6b added to argparse choices and 'all' expansion
- `scripts/14_compare_models.py` — M6b added to patterns dict; --m6b argparse arg; M6b load block in main()
- `scripts/15_analyze_mle_by_trauma.py` — WMRL_M6B_PARAMS list; kappa_total/kappa_share in PARAM_NAMES; M6b load path detection; M6b merge block; load_data() return tuple extended to 8; MODEL_CONFIG conditional entry; argparse choices and 'all' expansion
- `scripts/16_regress_parameters_on_scales.py` — kappa_total/kappa_share column renames in load_integrated_data(); format_label() entries; argparse choices and 'all' expansion; wmrl_m6b param_cols dispatch

## Decisions Made

- **Dual kernel copy optimization**: The hybrid_probs array is only copied once even when both kernels fire. First kernel triggers the copy (if not already done); second kernel reuses the same copy. The `modified` flag tracks whether normalization is needed. Avoids one redundant array copy per trial vs naive implementation.
- **Shared last_actions initialization**: M6b initializes `last_actions = {}` in the same `if model in ('wmrl_m6a', 'wmrl_m6b'):` branch as M6a, keeping the block initialization clean and symmetric.
- **load_data() return tuple extension**: Script 15's load_data() now returns an 8-tuple. Only one caller (main()), so the update is safe and no backward-compat issue.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- M6b is fully integrated into all pipeline scripts (11, 14, 15, 16)
- Parameter recovery functional: quick test (N=2, 1 dataset, n_starts=3) completes without error; recovery CSV has all 8 true/recovered columns
- Full parameter recovery (N=50, 10 datasets) should be run on cluster before Phase 11 (M4): `python scripts/11_run_model_recovery.py --model wmrl_m6b --n-subjects 50 --n-datasets 10 --n-jobs 8`
- Phase 10 (M6b) is complete — both plans (10-01 core, 10-02 downstream) done
- Ready for Phase 11 (M4 LBA joint choice+RT model)

**Blockers for next phase:**
- Full parameter recovery gate (r >= 0.80) not yet run for M6b — must run on cluster before Phase 11 begins

---
*Phase: 10-m6b-dual-perseveration*
*Completed: 2026-04-03*
