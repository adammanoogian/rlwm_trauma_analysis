---
phase: 17-m4-hierarchical-lba
plan: 02
subsystem: fitting
tags: [jax, numpyro, lba, mcmc, hierarchical, float64, checkpoint-resume, pareto-k]

# Dependency graph
requires:
  - phase: 17-01
    provides: wmrl_m4_hierarchical_model and prepare_stacked_participant_data_m4 in numpyro_models.py
  - phase: 16-04
    provides: bayesian_diagnostics (build_inference_data_with_loglik, filter_padding, shrinkage), bayesian_summary_writer
  - phase: 13-05
    provides: bayesian_diagnostics infrastructure pattern
provides:
  - scripts/13_fit_bayesian_m4.py -- self-contained M4 Bayesian fitting pipeline script
  - Float64 process isolation (M4H-01) implemented and verified
  - Checkpoint-resume pattern (M4H-04) with jax.device_get safety
  - Convergence gate (max_rhat<1.01, min_ess>400, 0 divergences) wired to output writes
  - Pareto-k gating (M4H-05) with 5% threshold, PASS/FALLBACK verdict, JSON + MD reports
  - Participant-level M4 log-lik via vmap over posterior samples
affects:
  - 17-03 (SLURM script for cluster submission uses this as entry point)
  - 18 (Phase 18 integration reads wmrl_m4_pareto_k_report.json for LOO reliability verdict)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Float64 isolation: import jax + jax.config.update as FIRST executable lines (before all other imports)"
    - "Checkpoint-resume: mcmc.warmup() -> jax.device_get -> pickle; mcmc.post_warmup_state = loaded_state"
    - "Participant-level log-lik: jax.vmap over flat samples for M4 (no per-trial pointwise in LBA)"
    - "Pareto-k gating: az.loo(idata, pointwise=True) -> frac(k>0.7) > 0.05 triggers fallback branch"
    - "Self-contained pipeline script pattern: no dispatch through fit_bayesian.py"

key-files:
  created:
    - scripts/13_fit_bayesian_m4.py
  modified: []

key-decisions:
  - "Participant-level (not trial-level) log-lik for M4 LOO: wmrl_m4_multiblock_likelihood_stacked returns scalar NLL per participant, no return_pointwise path exists for LBA"
  - "_build_inference_data_m4 uses 3D log-lik (chains, samples, n_participants) not 4D: ArviZ add_groups dims=['participant'] only"
  - "jax.device_get before pickle is mandatory: JAX arrays on accelerator device cannot be serialized directly"
  - "chain_method='vectorized' locked: 'parallel' mode requires os.fork() which is unavailable on Windows/SLURM"
  - "Convergence gate early-return writes run metadata JSON even on failure (diagnostic value)"
  - "Pareto-k error path: LOO failure (e.g. ArviZ version mismatch) caught with except; fallback report written regardless"

patterns-established:
  - "M4H-01: import jax; jax.config.update('jax_enable_x64', True) as first executable pair"
  - "M4H-04: warmup checkpoint lives at output/bayesian/m4_warmup_state.pkl; presence drives resume branch"

# Metrics
duration: 8min
completed: 2026-04-13
---

# Phase 17 Plan 02: M4 Hierarchical LBA Fitting Script Summary

**Self-contained M4 Bayesian pipeline script with float64 isolation, checkpoint-resume, convergence gate, and Pareto-k LOO reliability gating**

## Performance

- **Duration:** 8 min
- **Started:** 2026-04-13T16:35:19Z
- **Completed:** 2026-04-13T16:43:36Z
- **Tasks:** 1/1
- **Files modified:** 1

## Accomplishments

- Created `scripts/13_fit_bayesian_m4.py` (857 lines) as the complete M4 hierarchical Bayesian pipeline script
- Implemented float64 process isolation: `import jax; jax.config.update("jax_enable_x64", True)` as first executable lines (M4H-01)
- Checkpoint-resume pattern with `jax.device_get` safety before pickle serialization (M4H-04)
- Convergence gate (max_rhat < 1.01, min_ess > 400, n_divergences == 0) blocks output writes on failure; run metadata JSON written regardless
- Pareto-k gating writes PASS/FALLBACK verdict to markdown + JSON for Phase 18 consumption (M4H-05)
- Participant-level log-lik computation via `jax.vmap` over posterior draws (LBA has no per-trial pointwise path)

## Task Commits

1. **Task 1: Create scripts/13_fit_bayesian_m4.py** - `6a632eb` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `scripts/13_fit_bayesian_m4.py` - Self-contained M4 hierarchical Bayesian fitting script (857 lines)

## Decisions Made

- **Participant-level log-lik for M4 LOO:** `wmrl_m4_multiblock_likelihood_stacked` returns a scalar total NLL per participant; the LBA likelihood does not expose per-trial log-probs via `return_pointwise`. A custom `compute_m4_pointwise_loglik` function computes shape `(chains, samples_per_chain, n_participants)` using `jax.vmap` over flat posterior samples.
- **3D log-lik in InferenceData:** `_build_inference_data_m4` builds a custom function (not reusing `build_inference_data_with_loglik` from bayesian_diagnostics) since the existing function assumes 4D shape `(chains, samples, participants, trials)`. M4 uses 3D `(chains, samples, participants)` with `dims=["participant"]`.
- **chain_method='vectorized' not 'parallel':** Locked per plan spec (M4H-03). `parallel` mode requires process forking; `vectorized` is the correct NumPyro approach for single-process multi-chain.
- **Error handling in Pareto-k gate:** LOO computation wrapped in try/except so that a computational failure (e.g. ArviZ version issue) does not prevent the rest of the outputs being written. Fallback report written regardless.

## Deviations from Plan

None - plan executed exactly as written.

The plan specified that `compute_pointwise_log_lik` from bayesian_diagnostics does NOT support M4, and directed implementing inline. This was followed precisely: `compute_m4_pointwise_loglik` is implemented inline in `13_fit_bayesian_m4.py`. The `_build_inference_data_m4` helper was also implemented inline since the existing `build_inference_data_with_loglik` hardcodes 4D shape assumptions.

## Issues Encountered

None.

## Next Phase Readiness

- `scripts/13_fit_bayesian_m4.py` is ready for cluster submission via SLURM (Phase 17 Plan 03)
- The script accepts `--checkpoint-dir` to point at a cluster scratch directory for warmup state persistence across job restarts
- Phase 18 can consume `output/bayesian/wmrl_m4_pareto_k_report.json` to determine whether M4 LOO is included in cross-model comparison or the MLE AIC track is used instead
- **Pareto-k > 0.7 fallback is near-certain:** As noted in STATE.md blockers, participant-level LOO for LBA will likely trigger the fallback branch in production; this is expected behavior, not a bug

---
*Phase: 17-m4-hierarchical-lba*
*Completed: 2026-04-13*
