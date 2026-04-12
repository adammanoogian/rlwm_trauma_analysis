---
phase: 13-infrastructure-repair-hierarchical-scaffolding
plan: 03
subsystem: infra
tags: [jax, likelihoods, pointwise, waic, loo, arviz, hierarchical]

# Dependency graph
requires:
  - phase: 13-infrastructure-repair-hierarchical-scaffolding
    provides: "Plans 01-02: numpyro scaffolding and K-parameterization research"
provides:
  - "return_pointwise flag on all 6 choice-only *_block_likelihood functions"
  - "return_pointwise propagation on all 6 *_multiblock_likelihood_stacked wrappers"
  - "Per-trial log-prob arrays shape (n_blocks * max_trials,) for WAIC/LOO"
  - "30 unit tests covering scalar-default and pointwise paths for all model families"
affects:
  - 13-04 (numpyro_models.py integration)
  - 13-05 (bayesian_diagnostics.py - primary consumer of pointwise path)
  - 18-integration-comparison (az.waic() / az.loo() computation)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Opt-in pointwise via keyword-only bool flag after * separator - no break to existing callers"
    - "lax.scan for pointwise path (uniform output shape); lax.fori_loop for scalar path (original)"
    - "static_argnames=(return_pointwise,) added to JIT wrappers to allow specialization"

key-files:
  created:
    - scripts/fitting/tests/test_pointwise_loglik.py
  modified:
    - scripts/fitting/jax_likelihoods.py

key-decisions:
  - "Use Python if/else on return_pointwise (static bool), NOT jax.lax.cond - JIT specializes on static bool"
  - "lax.scan in pointwise path collects per-block arrays; lax.fori_loop in scalar path (unchanged MLE)"
  - "Flat reshape: all_block_probs.reshape(-1) gives (n_blocks * MAX_TRIALS_PER_BLOCK,) to az.from_dict"
  - "Padding positions have log_prob = 0.0 (masked in lax.scan output) - must filter before WAIC/LOO"
  - "Default return_pointwise=False preserves 100% backward compatibility with all MLE objective fns"

patterns-established:
  - "Pointwise API pattern: *_block_likelihood(... mask=mask, *, return_pointwise=False)"
  - "Stacked wrapper pointwise: lax.scan over blocks, reshape to flat; scalar: lax.fori_loop unchanged"
  - "Test pattern: parametrize over all 6 model families, verify both scalar and tuple paths"

# Metrics
duration: 11min
completed: 2026-04-12
---

# Phase 13 Plan 03: Pointwise Log-Likelihood Return Path Summary

**Exposed lax.scan per-trial log-probs via return_pointwise flag on all 6 JAX block likelihood functions and 6 stacked wrappers, with 30 parametric tests covering scalar-default and pointwise paths for every model family**

## Performance

- **Duration:** 11 min
- **Started:** 2026-04-12T08:46:24Z
- **Completed:** 2026-04-12T08:57:29Z
- **Tasks:** 2
- **Files modified:** 2 (jax_likelihoods.py modified, test_pointwise_loglik.py created)

## Accomplishments

- Added `return_pointwise: bool = False` keyword-only param to all 6 `*_block_likelihood` functions. Default returns scalar float unchanged; `True` returns `(total_log_lik, per_trial_log_probs)` tuple with shape `(MAX_TRIALS_PER_BLOCK,)`.
- Updated all 6 `*_multiblock_likelihood_stacked` wrappers to propagate `return_pointwise`. When `True`, uses `lax.scan` to collect per-block arrays and returns flat `(n_blocks * MAX_TRIALS_PER_BLOCK,)` array. When `False`, original `lax.fori_loop` path is preserved verbatim.
- Updated 2 JIT wrappers with `static_argnames=("return_pointwise",)` so JAX specializes separate traces for each branch.
- Created 30 tests covering: scalar default, tuple shape, sum equality, padding zeros, all 6 block functions, all 6 stacked wrappers. All pass. All 3 prior MLE tests unchanged and passing.

## Task Commits

Each task was committed atomically:

1. **Task 1: Add return_pointwise flag to all 6 block_likelihood functions** - `dd2654b` (feat)
2. **Task 2: Update stacked wrappers and add unit tests** - `6ff383a` (feat)

## Files Created/Modified

- `scripts/fitting/jax_likelihoods.py` - Added `return_pointwise` param + updated return logic to 6 block functions and 6 stacked wrappers; updated 2 JIT wrappers with `static_argnames`
- `scripts/fitting/tests/test_pointwise_loglik.py` - 30 parametric tests for all 6 model families, both paths, sum equality, padding zeros

## Decisions Made

- Used Python `if return_pointwise:` (NOT `jax.lax.cond`) because `return_pointwise` is a static Python bool. JIT compiles separate traces for each value via `static_argnames`, so the branch is compile-time not runtime.
- `lax.scan` chosen for pointwise path because it produces uniform-shape outputs `(n_blocks, MAX_TRIALS_PER_BLOCK)` that can be trivially reshaped. `lax.fori_loop` cannot accumulate outputs, so only the scalar path uses it.
- Flat reshape `all_block_probs.reshape(-1)` gives `arviz.from_dict()` the `(n_obs,)` array it expects for WAIC/LOO. Padding positions (log_prob=0.0) must be filtered by Plan 05 `bayesian_diagnostics.py` before feeding to `az.waic()`.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 05 (`bayesian_diagnostics.py`) can now call `*_multiblock_likelihood_stacked(return_pointwise=True)` to collect per-trial log-probs for `az.waic()` / `az.loo()`.
- Caller must filter out padding positions (mask == 0 entries, where log_prob = 0.0) before feeding to arviz to avoid inflating effective parameter count.
- All MLE callers (`fit_mle.py`, objective functions) are unchanged - they use the default `return_pointwise=False` path.

---
*Phase: 13-infrastructure-repair-hierarchical-scaffolding*
*Completed: 2026-04-12*
