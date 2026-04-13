---
phase: 17-m4-hierarchical-lba
plan: 01
subsystem: fitting
tags: [numpyro, jax, lba, hierarchical-bayes, m4, rt, float64, non-centered]

# Dependency graph
requires:
  - phase: 16-choice-only-family-extension-subscale-level-2
    provides: hierarchical model structure (sample_bounded_param, PARAM_PRIOR_DEFAULTS, phi_approx, prepare_stacked_participant_data)
  - phase: 11-m4-lba-joint-choice-rt
    provides: wmrl_m4_multiblock_likelihood_stacked, preprocess_rt_block, lba_likelihood.py
provides:
  - prepare_stacked_participant_data_m4: RT-aware data prep returning rts_stacked (float64) with combined RT-outlier+padding mask
  - wmrl_m4_hierarchical_model: 10-parameter NumPyro model with non-centered log(b-A) LBA reparameterization
  - Unit tests confirming RT data prep correctness and NUTS compilation with all 10 parameters finite
affects:
  - 17-02: fit_bayesian.py dispatch integration (STACKED_MODEL_DISPATCH for wmrl_m4)
  - 17-03: cluster SLURM scripts for M4 hierarchical fits

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Lazy import pattern: lba_likelihood imported inside function body to prevent float64 contamination in choice-only paths"
    - "Non-centered log-scale LBA params: log_X_mu_pr + log_X_sigma_pr * log_X_z -> exp() for v_scale, A, delta"
    - "b = A + delta decode inside participant for-loop, not in likelihood (M4H-02 pattern)"
    - "Probit-bounded t0: 0.05 + 0.25 * phi_approx(t0_mu_pr + t0_sigma_pr * t0_z)"

key-files:
  created:
    - scripts/fitting/tests/test_m4_hierarchical.py
  modified:
    - scripts/fitting/numpyro_models.py

key-decisions:
  - "Lazy import lba_likelihood inside function bodies (not module-level) to prevent float64 from propagating into choice-only models"
  - "delta = b - A sampled via log-normal, decoded as b = A + delta inside for-loop, guaranteeing b > A (M4H-02)"
  - "No epsilon parameter in M4 hierarchical model (LBA handles decision noise; epsilon is a softmax-noise term only)"
  - "RT padding value is 0.5s (not 0.0) to avoid t_star <= 0 in masked calls to LBA likelihood"

patterns-established:
  - "M4 data prep extends prepare_stacked_participant_data with rts_stacked (float64) and combined masks"
  - "All 10 M4 parameters use non-centered parameterization: 6 RLWM via sample_bounded_param, 4 LBA manually"

# Metrics
duration: 11min
completed: 2026-04-13
---

# Phase 17 Plan 01: M4 Hierarchical LBA Summary

**NumPyro M4 hierarchical model with non-centered log(b-A) LBA reparameterization and RT-aware float64 data prep; all 10 parameters compile under NUTS with finite posteriors**

## Performance

- **Duration:** 11 min
- **Started:** 2026-04-13T16:19:50Z
- **Completed:** 2026-04-13T16:30:52Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- `prepare_stacked_participant_data_m4`: RT-aware data preparation function that stacks RTs in float64 seconds, calls `preprocess_rt_block` per block, pads to `MAX_TRIALS_PER_BLOCK` with 0.5 (safe value), and ANDs the padding mask with the RT-outlier mask into `masks_stacked`
- `wmrl_m4_hierarchical_model`: full 10-parameter hierarchical NumPyro model; 6 RLWM params via `sample_bounded_param`, 4 LBA params (v_scale, A, delta, t0) with non-centered log-scale/probit reparameterization; `b = A + delta` decode inside participant for-loop guarantees b > A
- 3 unit tests all passing: RT data prep with outlier masking, sorted participant ordering, and NUTS smoke compilation with all 10 parameter keys present and finite

## Task Commits

Each task was committed atomically:

1. **Task 1: Add prepare_stacked_participant_data_m4 and wmrl_m4_hierarchical_model** - `5c16483` (feat)
2. **Task 2: Unit tests for M4 data prep and model smoke test** - `205d4ba` (test)

**Plan metadata:** see final commit below

## Files Created/Modified

- `scripts/fitting/numpyro_models.py` - Added two new functions at end of file (313 insertions); no existing functions modified
- `scripts/fitting/tests/test_m4_hierarchical.py` - New test file with 3 tests (223 lines)

## Decisions Made

- **Lazy import for lba_likelihood**: `wmrl_m4_multiblock_likelihood_stacked` and `preprocess_rt_block` are imported inside function bodies, not at module level. This prevents float64 from activating in choice-only model import paths.
- **RT padding value = 0.5s**: Padding positions use 0.5s rather than 0.0s to prevent `t_star = rt - t0 <= 0` in the LBA PDF when those positions are evaluated (even though they're masked out). Provides a safe default value.
- **No epsilon in M4**: M4 hierarchical model has 10 parameters (6 RLWM + 4 LBA), not 11. Epsilon is a softmax noise term; M4 uses LBA decision dynamics directly.
- **delta = b - A parameterization (M4H-02)**: Sampling `delta` via log-normal rather than `b` directly ensures `b > A` by construction without inequality constraints.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `prepare_stacked_participant_data_m4` and `wmrl_m4_hierarchical_model` are production-ready for Plan 02 (fit_bayesian.py dispatch integration)
- Plan 02 needs to add `"wmrl_m4"` to `STACKED_MODEL_DISPATCH` in `fit_bayesian.py` and wire `prepare_stacked_participant_data_m4` into the data loading path
- Smoke test confirmed NUTS compiles correctly (93s JIT compilation on CPU; will be faster on GPU cluster)

---
*Phase: 17-m4-hierarchical-lba*
*Completed: 2026-04-13*
