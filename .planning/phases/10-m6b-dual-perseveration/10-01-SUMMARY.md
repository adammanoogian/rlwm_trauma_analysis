---
phase: 10-m6b-dual-perseveration
plan: 01
subsystem: modelling
tags: [jax, mle, perseveration, stick-breaking, dual-kernel, lax-scan]

# Dependency graph
requires:
  - phase: 09-m6a-stimulus-specific-perseveration
    provides: wmrl_m6a_block_likelihood, per-stimulus last_actions carry pattern, M6a fitting pipeline
  - phase: 08-m5-rl-forgetting
    provides: M5 8-param pattern (how to add 8th parameter to M3 pipeline)
provides:
  - wmrl_m6b_block_likelihood (dual carry: global last_action + per-stimulus last_actions)
  - wmrl_m6b_multiblock_likelihood and _stacked variants
  - WMRL_M6B_BOUNDS, WMRL_M6B_PARAMS, transform functions (jax_unconstrained/bounded)
  - Full MLE fitting pipeline for wmrl_m6b (bounded, jax, gpu objectives, all dispatch points)
  - Stick-breaking reparameterization (kappa_total, kappa_share) for constraint enforcement
  - Verified equivalences: kappa_share=1.0 == M3, kappa_share=0.0 == M6a (0.0e+00 diff)
affects:
  - 10-02 (downstream: model_recovery.py, scripts 14/15/16/11 — M6b registration)
  - 11-m4-lba (model comparison context — M6b is final choice-only extension before M4)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Stick-breaking reparameterization: kappa = kappa_total * kappa_share decodes in objective functions, NOT in transform layer"
    - "Dual carry in lax.scan: 6-element tuple (Q, WM, WM_0, log_lik, last_action_scalar, last_actions_array)"
    - "Effective-weight gating: eff_kappa = jnp.where(has_global, kappa, 0.0) — no use_m2_path branch"
    - "All four fit_all_gpu dispatch points require explicit elif — no silent else fallthrough"
    - "param_cols dispatch in fit_all_participants separate from param_cols dispatch in main()"

key-files:
  created: []
  modified:
    - scripts/fitting/jax_likelihoods.py
    - scripts/fitting/mle_utils.py
    - scripts/fitting/fit_mle.py

key-decisions:
  - "Stick-breaking decode in objective functions only (not in transform): kappa = kappa_total * kappa_share; kappa_s = kappa_total * (1 - kappa_share)"
  - "Likelihood signature takes decoded kappa/kappa_s (not kappa_total/kappa_share) — consistent with M3/M6a"
  - "Dual carry: both scalar last_action (M3-style) and array last_actions (M6a-style) tracked independently"
  - "Effective-weight gating (eff_kappa/eff_kappa_s) handles all 4 kernel availability cases without branching"
  - "kappa_share lower bound 0.0 (not 0.001) — allows full reduction to M3 (share=1) or M6a (share=0)"

patterns-established:
  - "M6b is the template for adding dual perseveration kernels: copy M6a pattern for stim-specific, add M3 scalar carry alongside"
  - "Downstream (10-02) follows M6a extension pattern: add wmrl_m6b elif everywhere wmrl_m6a appears"

# Metrics
duration: 21min
completed: 2026-04-03
---

# Phase 10 Plan 01: M6b Dual Perseveration — Core Likelihood + MLE Fitting Summary

**Dual-perseveration M6b model with JAX lax.scan dual carry (global + per-stimulus kernels), stick-breaking constraint (kappa_total/kappa_share), and full MLE pipeline — 46/46 participants fit, kappa_share=1.0 matches M3 exactly, kappa_share=0.0 matches M6a exactly**

## Performance

- **Duration:** 21 min
- **Started:** 2026-04-03T08:51:54Z
- **Completed:** 2026-04-03T09:13:39Z
- **Tasks:** 2/2
- **Files modified:** 3

## Accomplishments

- M6b likelihood functions with dual carry (6-element lax.scan) — NLL verified finite, negative, padding-equivalent
- Stick-breaking reparameterization enforces kappa + kappa_s <= 1 by construction; verified kappa_share=1.0 gives 0.0e+00 diff vs M3, kappa_share=0.0 gives 0.0e+00 diff vs M6a
- Full MLE pipeline: all imports, 3 objective functions (bounded/jax/gpu), 4 fit_all_gpu dispatch points, fit_participant_mle, prepare_participant_data, diagnostics, argparse, param_cols — 46/46 converged, CSV has correct kappa_total/kappa_share columns

## Task Commits

1. **Task 1: M6b JAX likelihood + parameter registration** - `c9d12a1` (feat)
2. **Task 2: M6b MLE fitting pipeline integration** - `ebae37d` (feat)

## Files Created/Modified

- `scripts/fitting/jax_likelihoods.py` — wmrl_m6b_block_likelihood (dual carry), _multiblock_likelihood, _stacked, 4 inline tests (smoke, M3 equiv, M6a equiv, padding equiv)
- `scripts/fitting/mle_utils.py` — WMRL_M6B_BOUNDS, WMRL_M6B_PARAMS, jax_unconstrained/bounded transforms, all 8 dispatch function branches
- `scripts/fitting/fit_mle.py` — imports, warmup, 3 objective functions, 4 fit_all_gpu dispatch points, fit_participant_mle, prepare_participant_data (2 locations), fit_all_participants param_cols, diagnostics Hessian dispatch, main() param_cols + argparse

## Decisions Made

- **Stick-breaking decode in objectives only**: `wmrl_m6b_block_likelihood` takes `kappa` and `kappa_s` directly (decoded); the three objective functions decode `kappa = kappa_total * kappa_share` before calling likelihood. This keeps likelihood interface consistent with M3/M6a and makes tests cleaner.
- **Effective-weight gating over use_m2_path branch**: Instead of the M3/M6a `use_m2_path = jnp.logical_or(...)` pattern, M6b uses `eff_kappa = jnp.where(has_global, kappa, 0.0)` approach to handle dual kernels without branching. Cleaner for 4 availability cases.
- **Both lower bounds 0.0**: kappa_total and kappa_share use `(0.0, 1.0)` bounds (not 0.001) to allow exact reductions to M3 (share=1.0), M6a (share=0.0), and M2 (total=0.0).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Missing param_cols dispatch in fit_all_participants**
- **Found during:** Task 2 verification (wmrl_m6b fit run)
- **Issue:** The `fit_all_participants` function has its OWN `param_cols` dispatch (separate from `main()`) that controlled CSV column ordering. Plan documented the dispatch in `main()` but missed this second instance. Without it, `else: param_cols = []` meant all parameter columns were absent from the CSV.
- **Fix:** Added `elif model == 'wmrl_m6b': param_cols = WMRL_M6B_PARAMS` in `fit_all_participants` column ordering block.
- **Files modified:** scripts/fitting/fit_mle.py
- **Verification:** Re-ran fitting; CSV now shows all 8 parameter columns correctly.
- **Committed in:** ebae37d (part of Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Fix essential for correct CSV output. No scope creep.

## Issues Encountered

None beyond the missing param_cols dispatch (documented above as deviation).

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- M6b core likelihood and MLE fitting fully implemented and verified
- Ready for downstream integration: model_recovery.py, scripts 14/15/16/11 (Phase 10 Plan 02)
- Fitting results in `output/mle/wmrl_m6b_individual_fits.csv` (46 participants, 100% convergence with n_starts=5)
- Inline test results: kappa_share=1.0 vs M3 diff=0.00e+00; kappa_share=0.0 vs M6a diff=0.00e+00

**Blockers for next phase:**
- None — all equivalences verified, pipeline functional

---
*Phase: 10-m6b-dual-perseveration*
*Completed: 2026-04-03*
