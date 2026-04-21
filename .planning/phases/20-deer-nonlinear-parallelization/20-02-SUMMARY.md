---
phase: 20-deer-nonlinear-parallelization
plan: 02
subsystem: fitting
tags: [jax, pscan, vectorization, likelihood, vmap, parallel-scan]

# Dependency graph
requires:
  - phase: 20-01
    provides: precompute_last_action_global and precompute_last_actions_per_stimulus functions
provides:
  - 12 fully vectorized pscan likelihood variants (6 block + 6 multiblock)
  - 9 vectorized agreement tests confirming numerical equivalence
affects: [20-03-docs-update, bayesian-fitting-gpu]

# Tech tracking
tech-stack:
  added: []
  patterns: [vectorized-phase2-policy, precomputed-perseveration-arrays, vmap-softmax-batched]

key-files:
  created: []
  modified:
    - scripts/fitting/jax_likelihoods.py
    - scripts/fitting/tests/test_pscan_likelihoods.py

key-decisions:
  - "Modified pscan functions in-place rather than creating new variants"
  - "Used jax.vmap for batched softmax/epsilon rather than manual broadcasting"
  - "Preserved axis=-1 renormalization for batched base_probs in WM-RL models"

patterns-established:
  - "Vectorized Phase 2: precompute perseveration arrays, then vmap policy computation"
  - "Precompute helpers called inside block function (not hoisted to multiblock level)"

# Metrics
duration: 10min
completed: 2026-04-14
---

# Phase 20 Plan 02: Vectorize Phase 2 in Pscan Likelihoods Summary

**Replaced sequential lax.scan Phase 2 with vectorized array ops in all 12 pscan likelihood variants using jax.vmap and precomputed perseveration arrays**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-04-14T14:36:34Z
- **Completed:** 2026-04-14T14:46:56Z
- **Tasks:** 2/2
- **Files modified:** 2

## Accomplishments
- Eliminated all sequential `policy_step` lax.scan calls from the 6 block-level pscan functions (M1/M2/M3/M5/M6a/M6b)
- Used `precompute_last_action_global` (M3/M5) and `precompute_last_actions_per_stimulus` (M6a/M6b) from Plan 20-01 to vectorize perseveration
- All 30 non-slow tests pass (21 existing + 9 new), confirming < 1e-4 relative error agreement
- Net reduction of 57 lines (169 added, 226 removed) due to simpler vectorized code

## Task Commits

Each task was committed atomically:

1. **Task 1: Vectorize Phase 2 in all 6 block-level pscan functions** - `59235c9` (feat)
2. **Task 2: Add vectorized-specific agreement tests** - `d316a72` (test)

## Files Created/Modified
- `scripts/fitting/jax_likelihoods.py` - Vectorized Phase 2 in all 6 `*_block_likelihood_pscan()` functions
- `scripts/fitting/tests/test_pscan_likelihoods.py` - Added TestVectorizedPhase2 class with 9 test cases

## Decisions Made
- Modified pscan functions in-place rather than creating new function variants, since the pscan path is already the "parallel" path
- Called precompute helpers inside each block function rather than hoisting to multiblock level -- keeps the block function self-contained and correct when called standalone
- Used `jax.vmap(softmax_policy, in_axes=(0, None))` for batched softmax rather than reimplementing a batched version

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- All 12 pscan variants now use fully vectorized Phase 2 (associative scan Phase 1 + vectorized policy Phase 2)
- Ready for Plan 20-03: documentation update to reflect vectorized architecture
- The pscan path is now fully parallel-ready for GPU acceleration (no sequential bottleneck remains)

---
*Phase: 20-deer-nonlinear-parallelization*
*Completed: 2026-04-14*
