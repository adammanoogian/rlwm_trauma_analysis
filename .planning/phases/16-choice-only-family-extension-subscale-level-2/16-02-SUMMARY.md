---
phase: 16-choice-only-family-extension-subscale-level-2
plan: 02
subsystem: fitting
tags: [numpyro, hierarchical-bayesian, hBayesDM, non-centered, q-learning, wmrl, jax]

# Dependency graph
requires:
  - phase: 15-m3-hierarchical-poc
    provides: wmrl_m3_hierarchical_model template, sample_bounded_param, PARAM_PRIOR_DEFAULTS, hBayesDM non-centered convention, prepare_stacked_participant_data

provides:
  - qlearning_hierarchical_model_stacked in numpyro_models.py (HIER-02)
  - wmrl_hierarchical_model_stacked in numpyro_models.py (HIER-03)
  - q_learning_multiblock_likelihood_stacked and wmrl_multiblock_likelihood_stacked imported at module level

affects:
  - 16-03 (M5 wmrl_m5_hierarchical_model_stacked)
  - 16-04 (M6a wmrl_m6a_hierarchical_model_stacked)
  - 16-05 (M6b wmrl_m6b_hierarchical_model_stacked)
  - fit_bayesian.py dispatch table for qlearning and wmrl model keys

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Stacked-format hierarchical model pattern: sorted participant IDs, sample_bounded_param loop, numpyro.factor per participant"
    - "Q-learning vs WM-RL distinction: Q-learning never receives set_sizes_stacked; WM-RL always does"
    - "Forward-compat covariate_lec=None guard: NotImplementedError for models without natural L2 target"

key-files:
  created: []
  modified:
    - scripts/fitting/numpyro_models.py

key-decisions:
  - "Q-learning (M1) raises NotImplementedError on non-None covariate_lec — no natural L2 target parameter in this release"
  - "WM-RL (M2) raises NotImplementedError on non-None covariate_lec — no perseveration parameter for L2 regression"
  - "Both functions use sorted(participant_data_stacked.keys()) for participant ordering (alignment constraint)"

patterns-established:
  - "All stacked-format models follow: sorted IDs, sample_bounded_param loop over param list, numpyro.factor per participant"
  - "Q-learning signature omits set_sizes_stacked entirely from the likelihood call"
  - "WM-RL signature passes set_sizes_stacked=pdata['set_sizes_stacked'] to the likelihood call"

# Metrics
duration: 8min
completed: 2026-04-12
---

# Phase 16 Plan 02: Choice-Only Family Extension — M1 and M2 Stacked Models Summary

**M1 (Q-learning) and M2 (WM-RL) ported to hBayesDM non-centered stacked-format hierarchical models using sample_bounded_param; Q-learning confirmed to never receive set_sizes_stacked**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-04-12T~session-start
- **Completed:** 2026-04-12
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added `qlearning_hierarchical_model_stacked` (HIER-02): 3-parameter Q-learning model using hBayesDM non-centered convention with stacked pre-padded arrays
- Added `wmrl_hierarchical_model_stacked` (HIER-03): 6-parameter WM-RL model with same convention; correctly passes `set_sizes_stacked` to likelihood
- Added imports for `q_learning_multiblock_likelihood_stacked` and `wmrl_multiblock_likelihood_stacked` at module level
- Legacy `qlearning_hierarchical_model` and `wmrl_hierarchical_model` verified untouched

## Task Commits

Each task was committed atomically:

1. **Tasks 1+2: qlearning_hierarchical_model_stacked and wmrl_hierarchical_model_stacked** - `d61877d` (feat)

**Plan metadata:** (see final commit below)

## Files Created/Modified

- `scripts/fitting/numpyro_models.py` — Added 2 new stacked-format hierarchical model functions and 2 new likelihood imports

## Decisions Made

- Both `qlearning_hierarchical_model_stacked` and `wmrl_hierarchical_model_stacked` accept `covariate_lec=None` for forward compatibility but raise `NotImplementedError` on non-None input, since neither M1 nor M2 has a natural Level-2 regression target parameter (no kappa analogue).
- Participant ordering uses `sorted(participant_data_stacked.keys())` in both functions, consistent with Phase 15 constraint.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None. Import verification required using `conda run -n ds_env` since the default Python environment lacks JAX.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- HIER-02 and HIER-03 requirements complete
- M5, M6a, M6b stacked models (HIER-04, HIER-05, HIER-06) are the next mechanical extension in 16-03 through 16-05
- `fit_bayesian.py` dispatch table will need to add `"qlearning"` and `"wmrl"` keys pointing to the new stacked functions when 16-06 wires CLI paths

---
*Phase: 16-choice-only-family-extension-subscale-level-2*
*Completed: 2026-04-12*
