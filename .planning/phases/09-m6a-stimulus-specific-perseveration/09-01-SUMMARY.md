---
phase: 09-m6a-stimulus-specific-perseveration
plan: 01
subsystem: fitting
tags: [jax, mle, wmrl, perseveration, kappa_s, per-stimulus, lax-scan]

# Dependency graph
requires:
  - phase: 08-m5-rl-forgetting
    provides: M5 fitting pipeline pattern (dispatch blocks, objective functions, transforms)
  - phase: 07-wmrl-m3
    provides: M3 block likelihood with global last_action carry (base for M6a)
provides:
  - wmrl_m6a_block_likelihood with per-stimulus last_actions int32 array carry
  - wmrl_m6a_multiblock_likelihood and wmrl_m6a_multiblock_likelihood_stacked
  - WMRL_M6A_BOUNDS, WMRL_M6A_PARAMS, jax_unconstrained_to_params_wmrl_m6a, jax_bounded_to_unconstrained_wmrl_m6a
  - Full MLE fitting pipeline for wmrl_m6a (bounded/jax/gpu objectives, CLI)
  - Inline tests verifying per-stimulus tracking, padding equivalence, smoke test
affects:
  - 09-02 (parameter recovery for M6a -- not yet planned)
  - 10-m6b-dual-perseveration (M6b builds on M6a pattern)
  - 12-m4-lba (integrate M6a into comparison table)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "M6a carry pattern: replace scalar last_action with (num_stimuli,) int32 array in lax.scan"
    - "Stimulus sentinel: -1 = never seen in block, enables first-presentation uniform fallback"
    - "jnp.maximum(last_action_s, 0) clamp paired with use_m2_path gate prevents -1 indexing"
    - "last_actions.at[stimulus].set() update unconditional on valid (outside use_m2_path branch)"
    - "Model extension pattern: copy M3 (not M5), add elif wmrl_m6a after elif wmrl_m5 everywhere"

key-files:
  created: []
  modified:
    - scripts/fitting/jax_likelihoods.py
    - scripts/fitting/mle_utils.py
    - scripts/fitting/fit_mle.py

key-decisions:
  - "M6a branches from M3 (not M5): no phi_rl, 7 parameters (kappa_s replaces kappa)"
  - "kappa_s lower bound is 0.0 (same as M3's kappa) to allow M2-equivalent behavior"
  - "last_actions array update is unconditional on valid -- happens whether or not kernel was applied"
  - "All four fit_all_gpu dispatch points explicitly handle wmrl_m6a (no silent fallthrough to M5)"

patterns-established:
  - "Model extension: add all dispatch elifs in order after wmrl_m5, before else/raise"
  - "prepare_participant_data tuple updated alongside fit_all_gpu set_sizes tuple"

# Metrics
duration: 45min
completed: 2026-04-02
---

# Phase 9 Plan 01: M6a Stimulus-Specific Perseveration Summary

**M6a JAX likelihood with per-stimulus int32 last_actions carry replacing global scalar, WMRL_M6A_BOUNDS/PARAMS registration, and full fit_mle.py pipeline integration with all four fit_all_gpu dispatch points explicitly handled**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-04-02T20:00:00Z (approx)
- **Completed:** 2026-04-02T23:47:00Z (approx)
- **Tasks:** 2 of 2
- **Files modified:** 3

## Accomplishments

- Implemented `wmrl_m6a_block_likelihood` with `last_actions = jnp.full((num_stimuli,), -1, dtype=jnp.int32)` carry, per-stimulus indexing, and unconditional update
- Verified structural difference from M3: M6a gives different NLL (diff=0.693147) because stimulus 1's first presentation has no kernel in M6a but does in M3 (global carry)
- Full fitting pipeline integrated: `python scripts/12_fit_mle.py --model wmrl_m6a` runs end-to-end, outputs `wmrl_m6a_individual_fits.csv` with `kappa_s` column

## Task Commits

Each task was committed atomically:

1. **Task 1: M6a JAX likelihood + parameter registration** - `499962f` (feat)
2. **Task 2: M6a fitting pipeline (fit_mle.py CLI integration)** - `d11411f` (feat)

## Files Created/Modified

- `scripts/fitting/jax_likelihoods.py` - Added wmrl_m6a_block_likelihood (per-stimulus last_actions carry), wmrl_m6a_multiblock_likelihood, wmrl_m6a_multiblock_likelihood_stacked, three test functions
- `scripts/fitting/mle_utils.py` - Added WMRL_M6A_BOUNDS, WMRL_M6A_PARAMS, both transform functions, extended all dispatch functions with wmrl_m6a elif
- `scripts/fitting/fit_mle.py` - Added imports, three objective functions, warmup branch, fit_participant_mle branch, all four fit_all_gpu dispatch points, prepare_participant_data, param_cols, argparse

## Decisions Made

- M6a uses 7 parameters (same as M3), `kappa_s` replaces `kappa` -- no phi_rl, not based on M5
- `kappa_s` lower bound is `0.0` (not `0.001`) matching M3's `kappa` -- allows M2-equivalent at kappa_s=0
- The `last_actions.at[stimulus].set(...)` update is always executed for valid trials, regardless of whether the kernel was applied -- this is critical for correct tracking
- The `jnp.maximum(last_action_s, 0)` clamp is paired with `use_m2_path` gate to handle sentinel -1 without bad indexing

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Added prepare_participant_data set_sizes handling for wmrl_m6a**

- **Found during:** Task 2 (fit_mle.py integration)
- **Issue:** `prepare_participant_data` had `model in ('wmrl', 'wmrl_m3', 'wmrl_m5')` for set_sizes collection, causing `ValueError: set_sizes_blocks required for WM-RL M6a model` on first fit attempt
- **Fix:** Extended both the set_sizes condition and the pad_blocks_to_max condition to include `'wmrl_m6a'`
- **Files modified:** `scripts/fitting/fit_mle.py`
- **Verification:** `python scripts/12_fit_mle.py --model wmrl_m6a` ran successfully after fix
- **Committed in:** `d11411f` (Task 2 commit)

**2. [Rule 2 - Missing Critical] Added wmrl_m6a to Hessian diagnostics objective dispatch**

- **Found during:** Task 2 (fit_mle.py integration)
- **Issue:** Plan specification mentioned four dispatch points in fit_all_gpu but did not explicitly call out the Hessian objective dispatch (separate from fit_all_gpu). Without this, `--compute-diagnostics` would fail for wmrl_m6a
- **Fix:** Added `elif model == 'wmrl_m6a': objective_fn = _make_jax_objective_wmrl_m6a(...)` in the Hessian section
- **Files modified:** `scripts/fitting/fit_mle.py`
- **Committed in:** `d11411f` (Task 2 commit)

---

**Total deviations:** 2 auto-fixed (1 blocking, 1 missing critical)
**Impact on plan:** Both fixes necessary for correctness. No scope creep.

## Issues Encountered

None beyond the two auto-fixed deviations above.

## Next Phase Readiness

- M6a fitting pipeline fully functional and verified on real data (3 participants, 100% convergence)
- Per-stimulus tracking verified structurally different from M3 global tracking (NLL diff=0.693147)
- Padding equivalence verified (PASSED)
- Downstream scripts (14, 15, 16, model_recovery, script 11) NOT yet updated for M6a -- these are deferred to a separate plan per the v3.0 roadmap (Phase 9 plan 02 or integration plan)
- Parameter recovery (r >= 0.80 gate) not yet run -- should be done before proceeding to Phase 10 (M6b)

---
*Phase: 09-m6a-stimulus-specific-perseveration*
*Completed: 2026-04-02*
