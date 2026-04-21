---
phase: 09-m6a-stimulus-specific-perseveration
plan: 02
subsystem: fitting
tags: [jax, mle, wmrl, perseveration, kappa_s, parameter-recovery, pipeline, model-comparison]

# Dependency graph
requires:
  - phase: 09-m6a-stimulus-specific-perseveration
    plan: 01
    provides: M6a JAX likelihood, fit_mle.py integration, WMRL_M6A_BOUNDS/PARAMS
  - phase: 08-m5-rl-forgetting
    provides: M5 pipeline extension pattern (elif model == 'wmrl_m5' dispatch blocks)
provides:
  - M6a parameter recovery with per-stimulus last_actions dict in model_recovery.py
  - wmrl_m6a in scripts/11_run_model_recovery.py argparse + 'all' expansion
  - M6a in AIC/BIC comparison table via script 14 (auto-detection + --m6a arg)
  - M6a trauma group analysis with kappa_s parameter in script 15
  - M6a continuous regression with kappa_s in script 16
  - n_starts and n_jobs args exposed in script 11 and run_parameter_recovery()
affects:
  - 10-m6b-dual-perseveration (M6b pipeline follows same pattern)
  - 11-m4-lba (integrate M6a in joint comparison table)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "M6a synthetic generation: per-stimulus dict (last_actions={}) vs M3/M5 global scalar"
    - "Dict resets to {} at top of each block iteration (block boundary reset)"
    - "last_actions.get(stimulus) is not None: first-presentation uniform fallback"
    - "Downstream dispatch: copy M3 branch (no phi_rl), NOT M5 branch for M6a"
    - "compute_diagnostics=False in recovery loop avoids boundary-value ZeroDivisionError"
    - "Guard empty DataFrame columns before filtering (n<5/n<10 skip thresholds)"

key-files:
  created:
    - output/recovery/wmrl_m6a/recovery_results.csv
    - output/recovery/wmrl_m6a/recovery_metrics.csv
  modified:
    - scripts/fitting/model_recovery.py
    - scripts/11_run_model_recovery.py
    - scripts/14_compare_models.py
    - scripts/15_analyze_mle_by_trauma.py
    - scripts/16_regress_parameters_on_scales.py

key-decisions:
  - "M6a dispatch branches in model_recovery.py copy M3 pattern (no phi_rl) -- NOT M5"
  - "compute_diagnostics=False in run_parameter_recovery() loop avoids ZeroDivisionError at parameter bounds"
  - "n_starts and n_jobs exposed as params in run_parameter_recovery() and script 11 argparse"
  - "Guard corr_df/ols_df column access in script 15 summary for small-N robustness"

patterns-established:
  - "All downstream elif dispatch blocks: add wmrl_m6a after wmrl_m5, before else/raise"
  - "Script 15 load_data() returns wmrl_m6a as extra tuple member (parallel to wmrl_m5)"

# Metrics
duration: 12min
completed: 2026-04-02
---

# Phase 9 Plan 02: M6a Pipeline Integration Summary

**M6a wired into all downstream scripts: parameter recovery (per-stimulus last_actions dict), model comparison (AIC/BIC table with auto-detection), trauma group analysis (kappa_s parameter), and continuous regression (kappa_s on trauma scales)**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-04-02T21:52:32Z
- **Completed:** 2026-04-02T22:04:34Z
- **Tasks:** 2 of 2
- **Files modified:** 5

## Accomplishments

- Implemented per-stimulus `last_actions = {}` dict in `generate_synthetic_participant()` for M6a, resetting at each block boundary -- structurally distinct from M3/M5 global scalar
- Wired M6a into all dispatch blocks in `model_recovery.py`: `get_param_names`, `sample_parameters`, `compute_recovery_metrics`, `plot_recovery_scatter`, `plot_distribution_comparison`
- Quick parameter recovery test (N=2, 1 dataset, 3 starts) completes and writes `output/recovery/wmrl_m6a/` CSV files
- M6a auto-detected and included in AIC/BIC comparison table in script 14 (`output/mle/wmrl_m6a_individual_fits.csv`)
- Script 15 accepts `--model wmrl_m6a`, loads data defensively, reports `kappa_s` parameter associations
- Script 16 accepts `--model wmrl_m6a`, renames `kappa_s -> kappa_s_mean`, includes in regression output

## Task Commits

Each task was committed atomically:

1. **Task 1: M6a parameter recovery** - `481f799` (feat)
2. **Task 2: M6a pipeline scripts (14, 15, 16)** - `07af5e3` (feat)
3. **Bug fix: script 15 empty DataFrame guard** - `192fd0a` (fix)

**Plan metadata:** (docs commit pending)

## Files Created/Modified

- `scripts/fitting/model_recovery.py` - Added WMRL_M6A imports, get_param_names, sample_parameters, per-stimulus last_actions dict generation, all dispatch blocks, argparse choices, n_starts/n_jobs params, compute_diagnostics=False
- `scripts/11_run_model_recovery.py` - Added wmrl_m6a to choices and 'all', added --n-starts/--n-jobs args, pass-through to run_parameter_recovery
- `scripts/14_compare_models.py` - Added 'M6a' to find_mle_files patterns dict, added --m6a argparse arg, added load block
- `scripts/15_analyze_mle_by_trauma.py` - Added WMRL_M6A_PARAMS, kappa_s display name, M6a defensive loading, M6a merge, MODEL_CONFIG entry, argparse choices, load_data return, empty-DataFrame guards
- `scripts/16_regress_parameters_on_scales.py` - Added wmrl_m6a to choices/'all', kappa_s->kappa_s_mean rename, kappa_s_mean format_label, wmrl_m6a param_cols elif branch

## Decisions Made

- `compute_diagnostics=False` passed in `run_parameter_recovery()`: diagnostics are unnecessary during recovery and calling `params_to_unconstrained` when optimizer returns a boundary value (e.g., capacity=7.0 exactly) causes `ZeroDivisionError` in the logit transform. Disabling avoids this without any loss of recovery functionality.
- `n_starts` and `n_jobs` exposed as params in `run_parameter_recovery()` and script 11 to support the plan's verification command (`--n-starts 10 --n-jobs 4`) which had no prior wiring.
- Script 15 `load_data()` now returns 7-tuple (added `wmrl_m6a`), which required updating all call sites.
- Empty DataFrame guards added in script 15 summary section: `spearman_correlations()` and `ols_regression()` have n<5 and n<10 thresholds that produce empty results for small pilot fits, causing `KeyError` on column access. These are pre-existing bugs that only manifested for M6a (which currently has 3 participants).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added --n-starts/--n-jobs to script 11 and run_parameter_recovery()**

- **Found during:** Task 1 (verification command uses `--n-starts 10 --n-jobs 4`)
- **Issue:** Plan verification command used args that didn't exist in script 11 or model_recovery.py
- **Fix:** Added `--n-starts` and `--n-jobs` to argparse in both files; wired `n_starts` into `fit_participant_mle()` call; `n_jobs` accepted but documentation only (no parallel impl yet)
- **Files modified:** `scripts/11_run_model_recovery.py`, `scripts/fitting/model_recovery.py`
- **Verification:** `python scripts/11_run_model_recovery.py --help` shows both args
- **Committed in:** `481f799` (Task 1 commit)

**2. [Rule 1 - Bug] Added compute_diagnostics=False to avoid ZeroDivisionError at parameter bounds**

- **Found during:** Task 1 (recovery test with N=2)
- **Issue:** `fit_participant_mle` defaults to `compute_diagnostics=True`, which calls `params_to_unconstrained()`. When optimizer returns a value at the exact upper bound (e.g., phi=0.999), `bounded_to_unbounded` computes `(0.999-0.001)/(0.999-0.001)=1.0` then `logit(1.0)` → `ZeroDivisionError`
- **Fix:** Pass `compute_diagnostics=False` in `run_parameter_recovery()` -- not needed for recovery, avoids boundary issue
- **Files modified:** `scripts/fitting/model_recovery.py`
- **Verification:** Recovery test completes without ZeroDivisionError
- **Committed in:** `481f799` (Task 1 commit)

**3. [Rule 1 - Bug] Guard empty DataFrame in script 15 summary section for small N**

- **Found during:** Task 2 (script 15 verification with n=3)
- **Issue:** `spearman_correlations()` returns `pd.DataFrame([])` (no columns) when all param-predictor pairs have n<5. Summary section `corr_df[corr_df['model'] == ...]` raises `KeyError: 'model'`
- **Fix:** Check `'model' in corr_df.columns` and `'significant' in corr_df.columns` before filtering; similarly for `ols_df['p']`
- **Files modified:** `scripts/15_analyze_mle_by_trauma.py`
- **Verification:** `python scripts/15_analyze_mle_by_trauma.py --model wmrl_m6a` completes cleanly
- **Committed in:** `192fd0a` (fix commit after Task 2)

---

**Total deviations:** 3 auto-fixed (1 blocking, 2 bugs)
**Impact on plan:** All fixes necessary for correctness and robustness. No scope creep.

## Issues Encountered

The `compute_diagnostics=True` default in `fit_participant_mle` is a latent bug for any model where the optimizer reaches an exact parameter boundary. For the recovery use-case this is simply bypassed with `compute_diagnostics=False`. For the full fitting pipeline (script 12), this is mitigated by LHS sampling that avoids exact boundaries, but boundary-value robustness should be addressed in a future cleanup.

## Next Phase Readiness

- M6a pipeline fully integrated: recovery, comparison, trauma analysis, regression
- Quick recovery (N=2, 3 starts) verified working end-to-end
- Full parameter recovery (N=50, 10 datasets, r>=0.80 gate) NOT yet run -- should be done on cluster before Phase 10 (M6b): `python scripts/11_run_model_recovery.py --model wmrl_m6a --n-subjects 50 --n-datasets 10 --n-jobs 8`
- Phase 10 (M6b dual perseveration) can now proceed using M6a as the base pattern

---
*Phase: 09-m6a-stimulus-specific-perseveration*
*Completed: 2026-04-02*
