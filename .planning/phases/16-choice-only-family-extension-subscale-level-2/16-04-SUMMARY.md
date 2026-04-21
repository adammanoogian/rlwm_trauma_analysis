---
phase: 16-choice-only-family-extension-subscale-level-2
plan: "04"
subsystem: fitting
tags: [numpyro, jax, hierarchical-bayes, nuts, hbayesdm, slurm, choice-only]

requires:
  - phase: 16-02
    provides: qlearning_hierarchical_model_stacked, wmrl_hierarchical_model_stacked
  - phase: 16-03
    provides: wmrl_m5_hierarchical_model, wmrl_m6a_hierarchical_model, wmrl_m6b_hierarchical_model
  - phase: 15-02
    provides: run_inference_with_bump, filter_padding_from_loglik, compute_shrinkage_report

provides:
  - STACKED_MODEL_DISPATCH routing all 6 choice-only models
  - _fit_stacked_model() shared helper for stacked hierarchical inference
  - _load_lec_covariate() shared LEC covariate loading/z-scoring
  - _L2_LEC_SUPPORTED frozenset for M1/M2 LEC guard
  - save_results convergence gate (HIER-07) for all 6 models
  - 5 SLURM scripts (M1/M2/M5/M6a/M6b) for cluster execution
  - Fix: build_inference_data_with_loglik overwrites stale log_likelihood group

affects: [phase-17, phase-18, cluster-submission]

tech-stack:
  added: []
  patterns:
    - "_L2_LEC_SUPPORTED frozenset guards models that raise NotImplementedError on non-None covariate_lec"
    - "build_inference_data_with_loglik deletes existing log_likelihood group before add_groups (az.from_numpyro populates per-participant scalar log-probs from numpyro.factor sites)"
    - "STACKED_MODEL_DISPATCH table — single dict routes model string to NumPyro model fn"
    - "_fit_stacked_model encapsulates: prepare_stacked, LEC load, run_inference_with_bump, group-level printing"

key-files:
  created:
    - cluster/13_bayesian_m1.slurm
    - cluster/13_bayesian_m2.slurm
    - cluster/13_bayesian_m5.slurm
    - cluster/13_bayesian_m6a.slurm
    - cluster/13_bayesian_m6b.slurm
  modified:
    - scripts/fitting/fit_bayesian.py
    - scripts/fitting/bayesian_diagnostics.py

key-decisions:
  - "_L2_LEC_SUPPORTED frozenset (locked): {wmrl_m3, wmrl_m5, wmrl_m6a, wmrl_m6b} pass covariate_lec; M1/M2 receive None. Guard lives in _fit_stacked_model, not in model functions."
  - "log_likelihood group overwrite (locked): build_inference_data_with_loglik deletes pre-existing log_likelihood before add_groups; az.from_numpyro picks up scalar per-participant log-probs from numpyro.factor sites which have wrong shape for WAIC/LOO"
  - "Legacy qlearning/wmrl code paths removed: both now go through STACKED_MODEL_DISPATCH (stacked non-centered convention). No backward compat concern — both models were newly added in 16-02."
  - "EXPECTED_PARAMETERIZATION used in save_results for parameterization_version stamp — no hardcoded strings in fit_bayesian.py"

patterns-established:
  - "STACKED_MODEL_DISPATCH: new models added by adding one line to dict; no other code changes"
  - "save_results convergence gate: model in STACKED_MODEL_DISPATCH is the single predicate"
  - "SLURM scripts: identical conda/JAX activation block; only job-name, log-files, time, and --model flag differ"

duration: 22min
completed: 2026-04-13
---

# Phase 16 Plan 04: fit_bayesian Dispatch + SLURM Scripts Summary

**STACKED_MODEL_DISPATCH table routes all 6 choice-only hierarchical models through _fit_stacked_model, with convergence gate + WAIC/LOO + shrinkage for each; 5 SLURM scripts created for cluster submission**

## Performance

- **Duration:** 22 min
- **Started:** 2026-04-13T08:19:42Z
- **Completed:** 2026-04-13T08:41:42Z
- **Tasks:** 3
- **Files modified:** 2 (+ 5 created)

## Accomplishments

- Refactored fit_bayesian.py with `STACKED_MODEL_DISPATCH` dict routing all 6 choice-only models; extracted `_fit_stacked_model()` helper and `_load_lec_covariate()` for shared logic
- All 6 models (qlearning, wmrl, wmrl_m3, wmrl_m5, wmrl_m6a, wmrl_m6b) pass 1-chain/10W/20S smoke test on 5-participant subset without Python exceptions
- Created 5 SLURM scripts (M1=4h, M2/M5/M6a=6h, M6b=8h) using the M3 template pattern
- Fixed two bugs found during smoke testing: (1) LEC covariate guard for M1/M2, (2) log_likelihood group overwrite in build_inference_data_with_loglik

## Task Commits

Each task was committed atomically:

1. **Task 1: Refactor fit_bayesian.py with _fit_stacked_model helper** - `b57aa08` (feat)
2. **Task 2: Smoke test all new models with synthetic data** - `6f99419` (feat, includes two bug fixes)
3. **Task 3: Create SLURM scripts for 5 new models** - `a91ad1d` (feat)

**Plan metadata:** (this commit)

## Files Created/Modified

- `scripts/fitting/fit_bayesian.py` — refactored: STACKED_MODEL_DISPATCH, _fit_stacked_model, _load_lec_covariate, _L2_LEC_SUPPORTED, simplified fit_model/save_results/main
- `scripts/fitting/bayesian_diagnostics.py` — bug fix: delete existing log_likelihood group before add_groups in build_inference_data_with_loglik
- `cluster/13_bayesian_m1.slurm` — M1 (qlearning) cluster script, 4h
- `cluster/13_bayesian_m2.slurm` — M2 (wmrl) cluster script, 6h
- `cluster/13_bayesian_m5.slurm` — M5 (wmrl_m5) cluster script, 6h
- `cluster/13_bayesian_m6a.slurm` — M6a (wmrl_m6a) cluster script, 6h
- `cluster/13_bayesian_m6b.slurm` — M6b (wmrl_m6b, winning model) cluster script, 8h

## Decisions Made

- **_L2_LEC_SUPPORTED frozenset (locked):** `{wmrl_m3, wmrl_m5, wmrl_m6a, wmrl_m6b}` pass covariate_lec to model; M1/M2 receive None because `qlearning_hierarchical_model_stacked` and `wmrl_hierarchical_model_stacked` raise NotImplementedError if covariate_lec is not None. Guard lives in `_fit_stacked_model`, not in model functions.
- **log_likelihood group overwrite (locked):** `build_inference_data_with_loglik` deletes pre-existing `log_likelihood` before `add_groups`; `az.from_numpyro` picks up scalar per-participant log-probs from `numpyro.factor` sites (shape `(chains, samples)`) which have the wrong shape for WAIC/LOO. The proper pointwise array from `compute_pointwise_log_lik` must replace it.
- **EXPECTED_PARAMETERIZATION used in save_results:** `parameterization_version` stamp uses `EXPECTED_PARAMETERIZATION.get(model, "v4.0-K[2,6]-phiapprox")` — no hardcoded strings in fit_bayesian.py.
- **Legacy qlearning/wmrl code paths removed:** Both models now go through STACKED_MODEL_DISPATCH (stacked non-centered convention, committed in 16-02). No backward compatibility concern.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] LEC covariate passed to M1/M2 despite NotImplementedError guard in model**

- **Found during:** Task 2 (smoke test of qlearning)
- **Issue:** `_fit_stacked_model` called `_load_lec_covariate` for all models. M1 (`qlearning_hierarchical_model_stacked`) raises `NotImplementedError` if `covariate_lec is not None`. MCMC crashed before completing.
- **Fix:** Added `_L2_LEC_SUPPORTED = frozenset({"wmrl_m3", "wmrl_m5", "wmrl_m6a", "wmrl_m6b"})`. In `_fit_stacked_model`, LEC loading skipped if `model not in _L2_LEC_SUPPORTED`; `covariate_lec=None` passed to model.
- **Files modified:** `scripts/fitting/fit_bayesian.py`
- **Verification:** qlearning smoke test passes without NotImplementedError
- **Committed in:** `6f99419` (Task 2 commit)

**2. [Rule 1 - Bug] build_inference_data_with_loglik fails with "log_likelihood group already exists"**

- **Found during:** Task 2 (all models after LEC fix)
- **Issue:** `az.from_numpyro(mcmc)` automatically creates a `log_likelihood` group from `numpyro.factor` sites inside the participant for-loop. These per-participant scalar log-probs have shape `(chains, samples_per_chain)` — wrong shape for WAIC/LOO. Subsequent `idata.add_groups(log_likelihood=...)` raised `ValueError: ['log_likelihood'] group(s) already exists`.
- **Fix:** In `build_inference_data_with_loglik`, added: `if "log_likelihood" in idata._groups: del idata["log_likelihood"]` before `add_groups`. The proper shape `(chains, samples, n_participants, n_trials)` pointwise array from `compute_pointwise_log_lik` overwrites the stale group.
- **Files modified:** `scripts/fitting/bayesian_diagnostics.py`
- **Verification:** All 6 models reach CONVERGENCE GATE FAILED (expected) without ArviZ ValueError
- **Committed in:** `6f99419` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug)
**Impact on plan:** Both fixes necessary for any stacked model to reach WAIC/LOO computation. No scope creep.

## Issues Encountered

- `conda run -n ds_env python -c "..."` with multiline code blocks fails on Windows (conda assertion error on newlines in arguments). Worked around by writing temporary `.py` scripts.

## Next Phase Readiness

- All 6 choice-only models can now be submitted to cluster via their respective SLURM scripts
- Cluster submission sequence: `sbatch cluster/13_bayesian_m{1,2,3,5,6a,6b}.slurm`
- Phase 17 (M4 Hierarchical LBA) can use STACKED_MODEL_DISPATCH pattern as template for M4 dispatch; M4 will require a separate dispatch path due to float64 and LBA likelihood differences
- Phase 18 (Integration + Comparison) ready to consume all 6 posterior NetCDF files

---
*Phase: 16-choice-only-family-extension-subscale-level-2*
*Completed: 2026-04-13*
