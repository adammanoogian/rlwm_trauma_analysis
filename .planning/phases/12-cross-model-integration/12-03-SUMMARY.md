---
phase: 12-cross-model-integration
plan: 03
subsystem: fitting
tags: [model-recovery, perseveration, synthetic-data, convex-combination, epsilon]

# Dependency graph
requires:
  - phase: 12-cross-model-integration (12-01, 12-02)
    provides: M6a/M6b integration into pipeline, model reference docs
  - phase: 09-m6a-stim-specific (09-01, 09-02)
    provides: M6a likelihood with per-stimulus perseveration
  - phase: 10-m6b-dual-perseveration (10-01, 10-02)
    provides: M6b likelihood with dual perseveration (stick-breaking)
provides:
  - Corrected generate_synthetic_participant() matching jax_likelihoods.py formulas exactly
  - M6a/M6b parameter recovery unblocked (perseveration signal now present in synthetic data)
  - M3/M5 kappa recovery bias eliminated (convex combination replaces additive renormalization)
affects: [INTG-04, M6-11, cluster parameter recovery runs for M6a/M6b]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Epsilon-then-perseveration ordering for all non-M4 models"
    - "Convex combination (1-w)*P + w*Ck for all perseveration models"

key-files:
  created: []
  modified:
    - scripts/fitting/model_recovery.py

key-decisions:
  - "Q-learning epsilon moved inline to Q-learning branch (was in now-removed action selection block)"
  - "All perseveration models use convex combination, not additive renormalization"
  - "Epsilon applied BEFORE perseveration for all non-M4 models, matching jax_likelihoods.py"

patterns-established:
  - "Synthetic generation formula order: softmax -> epsilon noise -> perseveration convex combination -> action sampling"
  - "M6b effective-weight gating pattern: eff_kappa = kappa if has_global else 0.0"

# Metrics
duration: 6min
completed: 2026-04-03
---

# Phase 12 Plan 03: Fix Synthetic Generation Perseveration Bugs (Gap Closure)

**Fixed two bugs in generate_synthetic_participant() making M6a/M6b perseveration reachable and M3/M5 formula match jax_likelihoods.py convex combination**

## Performance

- **Duration:** 6 min
- **Started:** 2026-04-03T15:21:35Z
- **Completed:** 2026-04-03T15:27:52Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments
- Fixed M6a/M6b elif branches that were unreachable (siblings of `if last_action is not None:` instead of independent branches)
- Changed M3/M5 from additive renormalization (`P[a] += kappa; P /= sum`) to convex combination `(1-kappa)*P_noisy + kappa*Ck`
- Corrected epsilon-then-perseveration ordering for all non-M4 models
- Removed double epsilon application in non-M4 action selection
- Verified M6a kappa_s recovery: range 0.3275, r=1.000 (N=2 smoke test)
- Verified M6b kappa_total recovery: range 0.1546, r=1.000 (N=2 smoke test)
- Verified M3(kappa=0)==M2 algebraic identity preserved
- Verified M4 RT generation unaffected

## Task Commits

Each task was committed atomically:

1. **Task 1: Fix perseveration and epsilon ordering** - `c8360d2` (fix)
2. **Task 2: Smoke-test M6a/M6b parameter recovery** - validation only, no code changes

## Files Created/Modified
- `scripts/fitting/model_recovery.py` - Rewrote perseveration + epsilon + action selection block in generate_synthetic_participant()

## Decisions Made
- Q-learning epsilon application moved inline to Q-learning branch (deviation Rule 1 auto-fix: without this, Q-learning would lose epsilon noise after removing double-application in action selection)
- All perseveration models now use identical convex combination pattern, matching jax_likelihoods.py exactly

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Q-learning epsilon noise lost after action selection refactor**
- **Found during:** Task 1 (perseveration rewrite)
- **Issue:** Removing epsilon from action selection block also removed Q-learning's epsilon (Q-learning branch sets action_probs = rl_probs without epsilon)
- **Fix:** Added epsilon noise inline in Q-learning branch: `action_probs = epsilon / NUM_ACTIONS + (1.0 - epsilon) * rl_probs`
- **Files modified:** scripts/fitting/model_recovery.py
- **Verification:** M3(kappa=0)==M2 identity still holds; Q-learning generation produces valid trials
- **Committed in:** c8360d2 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Auto-fix necessary for correctness. Without it, Q-learning synthetic data would have had no epsilon noise. No scope creep.

## Issues Encountered
None -- both recovery runs completed without errors.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- M6a/M6b parameter recovery unblocked -- full cluster runs can proceed
- Remaining pending todos from STATE.md:
  - Run M6a full recovery: `python scripts/11_run_model_recovery.py --model wmrl_m6a --n-subjects 50 --n-datasets 10 --n-jobs 8`
  - Run M6b full recovery: `python scripts/11_run_model_recovery.py --model wmrl_m6b --n-subjects 50 --n-datasets 10 --n-jobs 8`
  - Run full cross-model recovery: `python scripts/11_run_model_recovery.py --mode cross-model --model all --n-subjects 50 --n-datasets 10 --n-jobs 8`

---
*Phase: 12-cross-model-integration (gap closure)*
*Completed: 2026-04-03*
