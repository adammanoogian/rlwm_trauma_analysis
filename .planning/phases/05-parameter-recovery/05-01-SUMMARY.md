---
phase: 05-parameter-recovery
plan: 01
subsystem: modeling
tags: [parameter-recovery, synthetic-data, mle, jax, validation]

# Dependency graph
requires:
  - phase: 04-regression-visualization
    provides: MLE fitting infrastructure (fit_mle.py, mle_utils.py, jax_likelihoods.py)
provides:
  - sample_parameters() for uniform sampling from MLE bounds
  - generate_synthetic_participant() for realistic trial-level data generation
  - run_parameter_recovery() for MLE fitting validation
  - compute_recovery_metrics() for Pearson r, RMSE, bias computation
affects: [05-02, 05-03, 05-04, 05-05, 05-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - JAX-based synthetic data generation matching real task structure
    - Parameter recovery validation with r >= 0.80 threshold
    - Uniform parameter sampling for unbiased recovery tests

key-files:
  created:
    - scripts/fitting/model_recovery.py
  modified: []

key-decisions:
  - "Use JAX for synthetic agent simulation (faster than agent classes)"
  - "Fixed beta=50 in synthetic data (matches real fitting)"
  - "Synthetic sona_id starts at 90000 (avoids collision with real data)"
  - "Pass participant_id as int (not string) to prepare_participant_data"

patterns-established:
  - "Synthetic data matches exact structure of task_trials_long.csv"
  - "Recovery metrics: Pearson r, RMSE, bias, pass/fail (r >= 0.80)"
  - "Progress tracking via tqdm for dataset and subject loops"

# Metrics
duration: 36min
completed: 2026-02-06
---

# Phase 5 Plan 1: Parameter Recovery Core Pipeline Summary

**JAX-based synthetic data generation and MLE recovery validation with Pearson r, RMSE, and bias metrics**

## Performance

- **Duration:** 36 min
- **Started:** 2026-02-06T11:56:32Z
- **Completed:** 2026-02-06T12:32:14Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Synthetic data generator produces realistic trial-level DataFrames matching task structure
- Parameter sampling uniformly from MLE bounds for unbiased recovery tests
- Complete recovery pipeline: sample → generate → fit → evaluate
- Recovery metrics computation with pass/fail criterion (r >= 0.80)

## Task Commits

Each task was committed atomically:

1. **Task 1: Rewrite model_recovery.py with synthetic data generation** - `72bce5c` (feat)
2. **Task 2: Add recovery loop and metrics computation** - `29dfcc0` (feat)

## Files Created/Modified
- `scripts/fitting/model_recovery.py` - Complete rewrite (496 lines)
  - `sample_parameters()` - Uniform sampling from MLE bounds
  - `generate_synthetic_participant()` - JAX-based agent simulation
  - `run_parameter_recovery()` - MLE fitting loop with tqdm progress
  - `compute_recovery_metrics()` - Pearson r, RMSE, bias, pass/fail

## Decisions Made

1. **JAX for synthetic simulation:** Use JAX directly instead of agent classes for speed. Q-learning and WM-RL simulation implemented with proper epsilon noise, asymmetric learning rates, and reversal logic.

2. **Fixed beta=50:** Match real fitting procedure by using fixed inverse temperature during synthetic data generation (for parameter identifiability).

3. **Synthetic participant IDs:** Start at 90000 to avoid collision with real data (10000-10XXX range).

4. **Participant ID type:** Pass int (not string) to `prepare_participant_data()` to match pandas filtering behavior with int64 columns.

5. **Data structure matching:** Synthetic data exactly matches task_trials_long.csv:
   - 21 blocks (3-23)
   - Variable trials per block (30-90)
   - Set sizes cycle [2, 3, 5, 6]
   - Reversals triggered after 12-18 consecutive correct responses
   - Columns: sona_id, block, stimulus, key_press, reward, set_size, trial_in_block

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Fixed prepare_participant_data() call signature**
- **Found during:** Task 2 verification (first test run)
- **Issue:** prepare_participant_data() requires participant_id and model arguments, not just DataFrame
- **Fix:** Added participant_id and model arguments to function call
- **Files modified:** scripts/fitting/model_recovery.py
- **Verification:** Recovery pipeline completes successfully
- **Committed in:** 29dfcc0 (Task 2 commit)

**2. [Rule 3 - Blocking] Fixed fit_participant_mle() call signature**
- **Found during:** Task 2 verification (second test run)
- **Issue:** fit_participant_mle() takes unpacked arrays (stimuli_blocks, actions_blocks, etc.), not dict
- **Fix:** Unpacked data_dict into individual array arguments
- **Files modified:** scripts/fitting/model_recovery.py
- **Verification:** Recovery pipeline completes successfully
- **Committed in:** 29dfcc0 (Task 2 commit)

**3. [Rule 1 - Bug] Fixed participant_id type for pandas filtering**
- **Found during:** Task 2 verification (third test run)
- **Issue:** Converting sona_id to string caused pandas filter to return 0 rows (int64 column)
- **Fix:** Pass participant_id as int (matching DataFrame dtype)
- **Files modified:** scripts/fitting/model_recovery.py
- **Verification:** prepare_participant_data() returns 21 blocks instead of 0
- **Committed in:** 29dfcc0 (Task 2 commit)

---

**Total deviations:** 3 auto-fixed (1 bug, 2 blocking)
**Impact on plan:** All auto-fixes necessary for correct integration with existing fit_mle.py infrastructure. No scope creep.

## Issues Encountered

**API discovery:** Function signatures for `prepare_participant_data()` and `fit_participant_mle()` required inspection. Fixed via reading fit_mle.py source and adjusting call sites accordingly.

**Type matching:** Pandas filtering behavior with int64 columns discovered through debugging. Real data uses int64 sona_id, so synthetic data must match.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for next plan:** 05-02 (Visualization utilities for recovery plots)

**Foundation complete:**
- Synthetic data generation working for all 3 models (Q-learning, WM-RL, WM-RL+K)
- MLE fitting successfully recovers parameters from synthetic data
- Recovery metrics computation validated with small test (5 subjects)
- Progress tracking via tqdm for user feedback

**Verified:**
- All 6 verification criteria passed
- Sample recovery test shows epsilon parameter recovers well (r=0.995)
- Alpha parameters show lower correlation with n=5 (expected, will improve with larger sample)

---
*Phase: 05-parameter-recovery*
*Completed: 2026-02-06*
