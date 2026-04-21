---
phase: 15-m3-hierarchical-poc-level2
plan: "02"
subsystem: bayesian-fitting
tags:
  - numpyro
  - convergence-gate
  - shrinkage-diagnostics
  - waic-loo
  - schema-parity
  - wmrl_m3

dependency-graph:
  requires:
    - "15-01"  # wmrl_m3_hierarchical_model, prepare_stacked_participant_data
    - "13-05"  # bayesian_diagnostics.py (compute_pointwise_log_lik), bayesian_summary_writer.py
  provides:
    - run_inference_with_bump in numpyro_models.py
    - wmrl_m3 path in fit_bayesian.py (BAYESIAN_IMPLEMENTED + convergence gate)
    - compute_shrinkage_report / write_shrinkage_report / filter_padding_from_loglik
  affects:
    - "15-03"  # Wave 3 validation tests consume these APIs
    - "16-*"   # Choice-only family extension reuses run_inference_with_bump pattern

tech-stack:
  added: []
  patterns:
    - "Convergence auto-bump: Python retry loop over target_accept_probs (0.80->0.95->0.99)"
    - "Convergence gate: max_rhat<1.01 AND min_ess>400 AND n_div==0 blocks file writes"
    - "Shrinkage formula: 1 - var_indiv / (var_group_mean + 1e-10)"
    - "Padding filter: mask==0 positions set to NaN before az.waic()/az.loo()"

key-files:
  created: []
  modified:
    - scripts/fitting/numpyro_models.py
    - scripts/fitting/fit_bayesian.py
    - scripts/fitting/bayesian_diagnostics.py

decisions:
  - id: "D1"
    context: "filter_padding_from_loglik returns NumPy array (not JAX DeviceArray)"
    decision: "Convert to np.array() before masking because ArviZ 0.23.4 operates on NumPy-backed data"
    rationale: "Avoids JAX->NumPy conversion errors in az.add_groups() downstream"
  - id: "D2"
    context: "save_results() imports filter_padding_from_loglik via local import inside function"
    decision: "Use local import to avoid circular import risk; all three new diagnostics functions imported together"
    rationale: "Module-level import of bayesian_diagnostics in fit_bayesian.py would create a second import block; local import inside wmrl_m3 branch is cleaner"
  - id: "D3"
    context: "Legacy qlearning/wmrl mu_beta print block"
    decision: "Wrap in 'if model in (qlearning, wmrl)' guard and add 'if mu_beta in samples' safety check"
    rationale: "Plan said not to fix the mu_beta bug for legacy paths, but wrapping avoids KeyError crash when running wmrl_m3"

metrics:
  duration: "6 minutes"
  completed: "2026-04-12"
---

# Phase 15 Plan 02: Convergence CLI Shrinkage Summary

**One-liner:** wmrl_m3 CLI path with 0.80->0.95->0.99 convergence auto-bump, HIER-07 gate, shrinkage diagnostics, and NaN-based padding filter for WAIC/LOO.

## What Was Built

### Task 1a: run_inference_with_bump (numpyro_models.py)

Added after the existing `run_inference()` function. Implements a Python retry loop over
`target_accept_probs = (0.80, 0.95, 0.99)`:

- Creates a fresh NUTS kernel and MCMC object for each acceptance probability level.
- Runs MCMC, reads `mcmc.get_extra_fields()["diverging"].sum()` for divergence count.
- Prints `[convergence-gate] target_accept_prob=X.XX divergences=N` after each run.
- Returns immediately on zero divergences (first level that passes).
- Falls through and returns last run when all levels exhausted (downstream gate flags it).

### Task 1b: fit_bayesian.py wmrl_m3 extension

**Imports added:** `wmrl_m3_hierarchical_model`, `prepare_stacked_participant_data`,
`run_inference_with_bump`, `compute_pointwise_log_lik`, `build_inference_data_with_loglik`,
`write_bayesian_summary`.

**BAYESIAN_IMPLEMENTED** updated to `{'qlearning', 'wmrl', 'wmrl_m3'}`.

**fit_model() wmrl_m3 branch:**
- Calls `prepare_stacked_participant_data()` for stacked JAX arrays.
- Loads `less_total_events` column from `output/summary_participant_metrics.csv`, z-scores it,
  converts to `jnp.array(float32)`. Falls back to `covariate_lec=None` with warning if CSV or
  column is missing.
- Aligns LEC covariate to `sorted(participant_data_stacked.keys())` for correctness.
- Calls `run_inference_with_bump()` with the stacked model args.
- Prints group-level `{param}_mu_pr` and `kappa_sigma_pr` summaries.
- Returns `(mcmc, participant_data_stacked)`.

**save_results() wmrl_m3 branch:**
- Computes `compute_pointwise_log_lik(mcmc, pdata_stacked, "wmrl_m3")`.
- Calls `filter_padding_from_loglik()` to set mask==0 positions to NaN.
- Builds InferenceData with `build_inference_data_with_loglik()`.
- **CONVERGENCE GATE (HIER-07):** Computes `az.summary()` for 7 M3 params, checks
  `max_rhat < 1.01 AND min_ess_bulk > 400 AND num_divergences == 0`. If gate fails, prints
  diagnostic message and returns early — NO files written.
- If gate passes: writes schema-parity CSV via `write_bayesian_summary()`, NetCDF posterior,
  WAIC, LOO, and shrinkage report.

**save_results() signature:** `(mcmc, data, model, output_dir, save_plots=True, participant_data_stacked=None)`

**main():** `mcmc, extra = fit_model(...)` then passes `participant_data_stacked=extra` for
`model == 'wmrl_m3'`, `None` for legacy models.

### Task 2: Shrinkage diagnostics and padding filter (bayesian_diagnostics.py)

**`compute_shrinkage_report(idata, param_names) -> dict[str, float]`**
- For each param: flatten posterior draws to `(total_draws, n_participants)`.
- `var_indiv = np.var(flat)` (across ALL draws AND participants).
- `var_group_mean = np.var(flat.mean(axis=1))` (variance of per-draw group mean).
- `shrinkage = 1.0 - var_indiv / (var_group_mean + 1e-10)`.
- Returns `dict[str, float]`.

**`write_shrinkage_report(shrinkage, output_path, *, threshold=0.3) -> Path`**
- Writes markdown table: Parameter | Shrinkage | Status.
- Status: "identified" if >= 0.3, "WARNING: poorly identified" if < 0.3.
- Summary count of identified vs poorly-identified.
- Interpretation note about using poorly-identified params as descriptive only.

**`filter_padding_from_loglik(pointwise_loglik, participant_data_stacked) -> np.ndarray`**
- Converts JAX array to NumPy float32.
- For each participant (sorted order), flattens `masks_stacked` to 1-D.
- Sets `result[:, :, idx, padding_indices] = np.nan` for mask==0 positions.
- Returns NumPy array (ArviZ-compatible).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Legacy mu_beta print block crashes for wmrl_m3**

- **Found during:** Task 1b
- **Issue:** The existing `fit_model()` print block accessed `samples['mu_beta']` unconditionally.
  This would crash with KeyError when `model == 'wmrl_m3'` if the code ever fell through to that block.
- **Fix:** Wrapped the print block in `if model in ('qlearning', 'wmrl'):` guard, added
  `if 'mu_beta' in samples:` safety check inside. Kept the wmrl_m3 path entirely separate
  with its own group-param printing logic.
- **Files modified:** `scripts/fitting/fit_bayesian.py`
- **Commit:** 3cb0b27

None of the other deviations required architectural changes. Plan executed exactly as specified.

## Verification Results

| Check | Result |
|-------|--------|
| `run_inference_with_bump` importable | PASS |
| `compute_shrinkage_report, write_shrinkage_report, filter_padding_from_loglik` importable | PASS |
| `fit_bayesian.py --model wmrl_m3 --help` prints without error | PASS |
| `save_results()` signature has `participant_data_stacked` | PASS |
| 58 fast tests (not slow) | 58/58 PASS |

## Next Phase Readiness

Phase 15-03 (Wave 3: validation tests) can proceed immediately. The APIs it needs are:
- `run_inference_with_bump` — in `numpyro_models.py`
- `wmrl_m3_hierarchical_model` + `prepare_stacked_participant_data` — from 15-01
- `compute_shrinkage_report`, `write_shrinkage_report`, `filter_padding_from_loglik` — now in `bayesian_diagnostics.py`
- Full CLI path `fit_bayesian.py --model wmrl_m3` — operational

No blockers. Shrinkage test fixtures can use synthetic posterior data without running full MCMC.
