---
phase: 13-infrastructure-repair-hierarchical-scaffolding
plan: 05
subsystem: infra
tags: [jax, numpyro, arviz, bayesian, waic, loo, slurm, gpu, schema-parity]

# Dependency graph
requires:
  - phase: 13-01
    provides: numpyro_models.py at canonical path, imports resolved
  - phase: 13-03
    provides: return_pointwise=True on all 6 block likelihood functions and 6 stacked wrappers

provides:
  - bayesian_diagnostics.py: compute_pointwise_log_lik() for WAIC/LOO via ArviZ
  - bayesian_summary_writer.py: schema-parity CSV writer for downstream script compatibility
  - cluster/13_bayesian_gpu.slurm: SLURM script with JAX cache for Bayesian GPU fitting
  - test_bayesian_summary.py: 15 tests validating schema parity and converged logic
  - test_compile_gate.py: @pytest.mark.slow compile-time gate test (warm < 60s)
  - fixtures/qlearning_bayesian_reference.csv: canonical schema reference for tests

affects:
  - 15-m3-hierarchical-poc: needs bayesian_diagnostics.py for WAIC/LOO comparison
  - 16-choice-only-extension: needs bayesian_summary_writer.py via --source bayesian flag
  - 18-integration-comparison-manuscript: needs WAIC/LOO from InferenceData with log_likelihood group

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "compute_pointwise_log_lik dispatch: model_name -> stacked likelihood fn + jax.vmap over (chains, samples)"
    - "schema-parity: Bayesian CSV uses _hdi_low/_hdi_high/_sd in place of _se/_ci_*; no grad_norm/hessian"
    - "converged = max_rhat < 1.01 AND min_ess_bulk > 400 AND num_divergences == 0"
    - "SLURM JAX cache: JAX_COMPILATION_CACHE_DIR=/scratch/${PROJECT:-fc37}/$USER/.jax_cache_gpu"

key-files:
  created:
    - scripts/fitting/bayesian_diagnostics.py
    - scripts/fitting/bayesian_summary_writer.py
    - cluster/13_bayesian_gpu.slurm
    - scripts/fitting/tests/test_bayesian_summary.py
    - scripts/fitting/tests/test_compile_gate.py
    - scripts/fitting/tests/fixtures/qlearning_bayesian_reference.csv
  modified: []

key-decisions:
  - "compute_pointwise_log_lik returns (chains, samples, participants, n_blocks*max_trials); padded trials have log_prob=0.0 inherited from mask"
  - "Posterior MEAN used for <param> point-estimate columns (matches MLE semantics for downstream consumers)"
  - "95% HDI for _hdi_low/_hdi_high via az.hdi(); posterior STD for _sd"
  - "M6b: kappa_total/kappa_share decoded inside compute_pointwise_log_lik before calling wmrl_m6b_multiblock_likelihood_stacked(kappa, kappa_s)"
  - "SLURM: time=06:00:00, mem=32G; NUMPYRO_HOST_DEVICE_COUNT=1; JAX cache block copied verbatim from 12_mle_gpu.slurm"
  - "compile gate marked @pytest.mark.slow; measures warm (second) invocation only"

patterns-established:
  - "Bayesian downstream: load_bayesian_fits() validates parameterization_version presence; raises ValueError on legacy v3.0 files"
  - "Schema column ordering: participant_id -> params -> info criteria -> HDI/SD -> convergence -> n_trials/converged/at_bounds -> parameterization_version"

# Metrics
duration: 7min
completed: 2026-04-12
---

# Phase 13 Plan 05: Bayesian Diagnostics, Schema-Parity Writer & Cluster Script Summary

**Bayesian diagnostics (WAIC/LOO via ArviZ), schema-parity CSV writer enabling --source bayesian flag, and JAX-cached GPU SLURM script — completing Phase 13 Wave 2 infrastructure.**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-04-12T09:04:38Z
- **Completed:** 2026-04-12T09:11:34Z
- **Tasks:** 2/2
- **Files created:** 6

## Accomplishments

- `compute_pointwise_log_lik()` dispatches over 6 model families, vmaps over (chains, samples_per_chain), and returns shape `(chains, samples, participants, n_blocks*max_trials)` ready for `az.waic()` / `az.loo()`.
- `write_bayesian_summary()` outputs a schema-parity CSV replacing Hessian columns with `_hdi_low/_hdi_high/_sd`; validated by 15 pytest tests all passing.
- `cluster/13_bayesian_gpu.slurm` forks 12_mle_gpu.slurm with the JAX cache block copied verbatim, time/mem adjusted for Bayesian fitting, and `NUMPYRO_HOST_DEVICE_COUNT=1` added.

## Task Commits

1. **Task 1: bayesian_diagnostics.py** - `ac11784` (feat)
2. **Task 2: bayesian_summary_writer, tests, SLURM script** - `fd7ad11` (feat)

## Files Created/Modified

- `scripts/fitting/bayesian_diagnostics.py` - compute_pointwise_log_lik() + build_inference_data_with_loglik()
- `scripts/fitting/bayesian_summary_writer.py` - write_bayesian_summary() schema-parity CSV writer
- `cluster/13_bayesian_gpu.slurm` - SLURM script with JAX cache, NUTS/NumPyro Bayesian fitting
- `scripts/fitting/tests/test_bayesian_summary.py` - 15 schema-parity and converged-logic tests
- `scripts/fitting/tests/test_compile_gate.py` - @pytest.mark.slow warm-compile gate test
- `scripts/fitting/tests/fixtures/qlearning_bayesian_reference.csv` - canonical schema reference CSV

## Decisions Made

- Posterior MEAN used for `<param>` columns (not median) — matches MLE point-estimate semantics for downstream scripts 15/16/17.
- M6b decodes `kappa_total*kappa_share -> kappa` and `kappa_total*(1-kappa_share) -> kappa_s` inside `compute_pointwise_log_lik` before passing to `wmrl_m6b_multiblock_likelihood_stacked`.
- `converged = max_rhat < 1.01 AND min_ess_bulk > 400 AND num_divergences == 0` — strict thresholds matching ArviZ defaults.
- Compile gate tests marked `@pytest.mark.slow` to exclude from fast CI; measure second (warm) invocation only.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

Phase 15 (M3 Hierarchical POC) can now:
1. Run `compute_pointwise_log_lik()` after MCMC to get per-trial log-lik array.
2. Call `build_inference_data_with_loglik()` to attach `log_likelihood` group to ArviZ InferenceData.
3. Run `az.waic(idata)` and `az.loo(idata)` without warnings.
4. Write schema-parity CSV via `write_bayesian_summary()` for downstream scripts.
5. Submit cluster jobs via `cluster/13_bayesian_gpu.slurm`.

No blockers for Phase 15.

---
*Phase: 13-infrastructure-repair-hierarchical-scaffolding*
*Completed: 2026-04-12*
