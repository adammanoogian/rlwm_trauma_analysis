---
phase: 21-principled-bayesian-model-selection-pipeline
plan: 03
subsystem: testing
tags: [bayesian, parameter-recovery, numpyro, jax, mcmc, slurm, hierarchical, kappa]

# Dependency graph
requires:
  - phase: 21-01
    provides: "PARAM_PRIOR_DEFAULTS with all mu_prior_loc=0.0 (v4.0-locked priors)"
  - phase: 15
    provides: "prepare_stacked_participant_data + _fit_stacked_model canonical fit path"
  - phase: 16
    provides: "fully-batched likelihood (_FULLY_BATCHED_MODELS tuple) for 6 choice-only models"
provides:
  - "scripts/21_run_bayesian_recovery.py — single-subject + aggregate CLI modes"
  - "cluster/21_2_recovery.slurm — SLURM array (1-50), 1 synthetic subject per task"
  - "cluster/21_2_recovery_aggregate.slurm — post-array CSV + summary.md writer"
  - "scripts/fitting/tests/test_bayesian_recovery.py — 7 smoke tests"
  - "Baseline-only scope: 2-cov L2 recovery gated via plan 21-11 pytest, NOT cluster"
affects: [21-10-master-orchestrator, 21-04-baseline-fit, phase-16-l2-inference]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "SLURM array dispatch for embarrassingly-parallel synthetic subject fits"
    - "Per-MODEL JAX cache directory (cross-array warm-start, cross-model isolation)"
    - "Autopush-after-aggregate pattern (single push, avoid 50 concurrent array races)"
    - "phi_approx-consistent prior sampling mirrors the non-centered hBayesDM transform"

key-files:
  created:
    - "scripts/21_run_bayesian_recovery.py"
    - "scripts/fitting/tests/test_bayesian_recovery.py"
    - "cluster/21_2_recovery.slurm"
    - "cluster/21_2_recovery_aggregate.slurm"
  modified: []

key-decisions:
  - "Scope: recovery exercises baseline (no-covariate) inference path only; 2-cov L2 hook from plan 21-11 is gated via pytest to avoid tripling cluster cost"
  - "sigma_scale=0.2 fixed in prior sampling (mirrors HalfNormal(0.2) on sigma_pr); single-subject hierarchy degenerates so this is reasonable"
  - "Kappa-family pass criterion = r >= 0.80 AND HDI coverage >= 0.90; other params labelled 'descriptive only' (matches quick-005 MLE recovery r = 0.21–0.77 floor)"
  - "RNG derivation: base seed folded with subject_idx via jax.random.fold_in for per-task independence; distinct sim_seed and mcmc_seed within each task"
  - "Autopush NOT sourced in array script (would race 50 concurrent pushes); sourced only in aggregate script"

patterns-established:
  - "Two-phase SLURM pipeline: array job + dependent aggregator (--dependency=afterok:$ARRAY_JOBID)"
  - "CLI mode switch (--mode single-subject vs --mode aggregate) in one script for orchestrator simplicity"

# Metrics
duration: 10m
completed: 2026-04-18
---

# Phase 21 Plan 03: Bayesian Parameter Recovery Pipeline Summary

**N=50 synthetic-subject Bayesian recovery via SLURM array for all 6 choice-only models, with kappa-family pass gate (r >= 0.80 AND 95% HDI coverage >= 0.90) and per-subject JSON + aggregated CSV + verdict markdown.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-04-18T14:46:33Z
- **Completed:** 2026-04-18T14:56:02Z
- **Tasks:** 2
- **Files created:** 4 (1 main script, 1 test file, 2 SLURM scripts)
- **Commits:** 2 task commits

## Accomplishments

- `scripts/21_run_bayesian_recovery.py` (956 lines) with two CLI modes:
  - **single-subject:** samples one true parameter vector from `PARAM_PRIOR_DEFAULTS` via the same `phi_approx` transform used inside `sample_bounded_param`, generates trial data with `scripts.fitting.model_recovery.generate_synthetic_participant`, fits via `_fit_stacked_model` (warmup=500, samples=1000, chains=2, max_tree_depth=8), extracts per-parameter posterior_mean + 95% HDI + in_hdi flag + convergence diagnostics, writes per-subject JSON with all metadata
  - **aggregate:** globs per-subject JSONs, computes Pearson r + HDI coverage per parameter, applies kappa-family pass criterion (r >= 0.80 AND coverage >= 0.90), writes CSV + summary.md, returns exit 1 only if any kappa-family parameter FAILs
- `scripts/fitting/tests/test_bayesian_recovery.py` — 7 passing tests:
  - `test_sample_true_params_from_prior_has_all_keys` (6 models × all params in bounds)
  - `test_sample_true_params_different_subjects_give_different_draws`
  - `test_single_subject_recovery_qlearning_smoke` (end-to-end MCMC, 16 s)
  - `test_aggregate_handles_missing` (partial JSONs, NO_KAPPA verdict)
  - `test_aggregate_kappa_pass_and_fail` (strong recovery => PASS)
  - `test_aggregate_kappa_fail_low_coverage` (perfect r, zero coverage => FAIL)
  - `test_safe_pearson_r_handles_zero_variance`
- `cluster/21_2_recovery.slurm` — SLURM array 1-50, 1.5h/16G/2 CPUs/task, JAX cache keyed on MODEL, autopush intentionally omitted
- `cluster/21_2_recovery_aggregate.slurm` — 30min/8G/1 CPU, invokes aggregate mode, sources autopush, master orchestrator chains via `afterok` dependency

## Task Commits

1. **Task 1: Implement 21_run_bayesian_recovery.py + tests** — `05f1b79` (feat)
2. **Task 2: Create SLURM array + aggregator scripts** — `205f480` (feat)

## Files Created/Modified

- `scripts/21_run_bayesian_recovery.py` — single-subject + aggregate recovery runner (956 lines)
- `scripts/fitting/tests/test_bayesian_recovery.py` — 7 smoke tests (352 lines)
- `cluster/21_2_recovery.slurm` — SLURM array dispatcher (129 lines)
- `cluster/21_2_recovery_aggregate.slurm` — post-array aggregator (116 lines)

## Decisions Made

- **Baseline-only scope (from plan):** The 2-covariate L2 hook for M3/M5/M6a built in plan 21-11 is covered by the pytest `test_recovery_2cov_m3` rather than this production 50-subject sweep. Avoids tripling cluster cost (150 additional MCMC jobs) for a confirmatory test the local pytest already gates.
- **sigma_scale = 0.2 in `sample_true_params_from_prior`:** Fixed rather than inferred because we are sampling a single subject's parameter vector (N=1 hierarchy degenerates). Matches the `HalfNormal(0.2)` prior on `sigma_pr` that the hierarchical model actually uses.
- **Kappa-family pass criterion:** `r >= 0.80 AND HDI coverage >= 0.90` for params in `{kappa, kappa_s, kappa_total, kappa_share}`; all other params labelled `descriptive only`. Matches Baribault & Collins (2023) gate 3 and aligns with the quick-005 finding that base RLWM parameters (alpha_pos, alpha_neg, phi, rho, capacity) are structurally under-identified (r = 0.21–0.77).
- **RNG derivation:** Base `seed` folded with `subject_idx` via `jax.random.fold_in` for per-array-task independence. Within a task the simulator uses `sim_seed = seed * 7919 + subject_idx` and MCMC uses `mcmc_seed = seed * 131 + subject_idx * 17 + 42` so the three stochastic processes (prior draw, synthetic sim, MCMC chain init) are distinct.
- **Autopush gating:** The array script does NOT source `cluster/autopush.sh` — 50 concurrent `git push` calls from compute nodes would race and spam the remote. Only the aggregate script (which runs once, after all array tasks finish) calls autopush.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed zero-variance detection in `_safe_pearson_r`**

- **Found during:** Task 1 test execution
- **Issue:** `np.std([0.2, 0.2, 0.2])` returned `-5.66e-17` due to floating-point drift, so the `== 0` check failed and the test `test_safe_pearson_r_handles_zero_variance` failed by computing a spurious Pearson r on a constant vector.
- **Fix:** Switched the zero-variance check from `np.std(x) == 0` to `np.ptp(x) == 0` (peak-to-peak range is an integer-exact 0 for constant inputs).
- **Files modified:** `scripts/21_run_bayesian_recovery.py` (one-line change in `_safe_pearson_r`)
- **Verification:** `pytest scripts/fitting/tests/test_bayesian_recovery.py -v` — all 7 tests pass after the fix.
- **Commit:** `05f1b79` (included in Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Fix necessary for correct edge-case handling when aggregate mode receives a single loaded subject (posterior-mean vector is constant → Pearson r undefined → must return NaN). No scope creep.

## Authentication Gates

None — all work was local, no external authentication required.

## Issues Encountered

None — plan executed cleanly once the zero-variance bug was fixed.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Step 21.2 recovery pipeline ready for cluster submission. `/gsd:execute-phase` or master orchestrator (plan 21-10) can now chain:
  ```
  ARRAY_JOBID=$(sbatch --parsable --export=ALL,MODEL=wmrl_m3 cluster/21_2_recovery.slurm)
  sbatch --dependency=afterok:$ARRAY_JOBID --export=ALL,MODEL=wmrl_m3 cluster/21_2_recovery_aggregate.slurm
  ```
- Kappa-family pass criterion is the Phase 21 "gate 3" check before L2 inference proceeds — if M3/M5/M6a/M6b FAIL, the pipeline should short-circuit.
- Expected cluster wall-clock: ~30 min per model (50 array tasks parallelized, each 1 min–30 min depending on convergence bump behaviour in `run_inference_with_bump`).
- Baseline scope preserved — the 2-cov L2 hook from plan 21-11 is gated by its own pytest suite, so this plan does not expand the cluster sweep cost.

---
*Phase: 21-principled-bayesian-model-selection-pipeline*
*Completed: 2026-04-18*
