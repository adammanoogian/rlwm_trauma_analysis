---
phase: 20-deer-nonlinear-parallelization
plan: 01
subsystem: fitting
tags: [deer, parallel-scan, perseveration, jax, precomputation, vectorized-policy]

# Dependency graph
requires:
  - phase: 19-associative-scan-parallelization
    provides: Phase 1 parallel scan primitives (affine_scan, Q/WM scan), 12 pscan likelihood variants
provides:
  - DEER research document with NO-GO recommendation and Unifying Framework analysis
  - precompute_last_action_global() for M3/M5 global perseveration
  - precompute_last_actions_per_stimulus() for M6a/M6b per-stimulus perseveration
  - Unit tests verifying exact agreement with sequential scan carry
affects: [20-02 vectorized policy variants, 20-03 documentation update]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Perseveration precomputation: parameter-independent data preprocessing outside MCMC loop"
    - "Research-first design: thorough algorithm analysis before implementation"

key-files:
  created:
    - docs/DEER_NONLINEAR_PARALLELIZATION.md
  modified:
    - scripts/fitting/jax_likelihoods.py
    - scripts/fitting/tests/test_pscan_likelihoods.py

key-decisions:
  - "NO-GO on DEER: discrete state, non-differentiable argmax, tiny D, short T make DEER inapplicable"
  - "GO on vectorized policy: precompute last_action from observed data, then broadcast"
  - "Perseveration carry is a phantom dependency: in likelihood evaluation, actions are observed data"

patterns-established:
  - "Parameter-independent precomputation: data-only arrays computed once before MCMC"
  - "Agreement testing: precomputed arrays verified against sequential lax.scan carry"

# Metrics
duration: 10min
completed: 2026-04-14
---

# Phase 20 Plan 01: DEER Research + Precomputation Summary

**DEER NO-GO research document with Unifying Framework analysis, plus two perseveration precomputation functions enabling vectorized Phase 2 policy**

## Performance

- **Duration:** 10 min
- **Started:** 2026-04-14T14:23:17Z
- **Completed:** 2026-04-14T14:33:36Z
- **Tasks:** 2/2
- **Files modified:** 3

## Accomplishments
- DEER research document covering all 5 required subsections: algorithm, convergence, alternatives, Unifying Framework, NO-GO recommendation
- Two precomputation utility functions that extract perseveration carry from observed data (parameter-independent)
- 7 unit tests confirming exact agreement with sequential lax.scan carry for both global and per-stimulus cases

## Task Commits

Each task was committed atomically:

1. **Task 1: Create DEER research document** - `c96eb6b` (docs)
2. **Task 2: Implement precomputation functions + unit tests** - `d015f27` (feat)

## Files Created/Modified
- `docs/DEER_NONLINEAR_PARALLELIZATION.md` - 449-line research document covering DEER algorithm, convergence analysis, Unifying Framework taxonomy, and NO-GO recommendation
- `scripts/fitting/jax_likelihoods.py` - Added `precompute_last_action_global()` and `precompute_last_actions_per_stimulus()` after Phase 19 scan helpers
- `scripts/fitting/tests/test_pscan_likelihoods.py` - Added 7 tests: 3 global, 2 per-stimulus, 2 agreement with sequential scan

## Decisions Made
- NO-GO on DEER: The RLWM perseveration carry is an implementation artifact, not a mathematical sequential dependency. In likelihood evaluation, actions are observed data, so last_action[t] = actions[t-1] is precomputable.
- Used lax.scan (not pure array shift) for both precomputation functions to correctly handle masked/padded trials where invalid trials do not update last_action.
- Added an extra test (test_precompute_last_action_global_all_masked) beyond the 6 specified, covering the edge case where all trials are masked.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
- Conda environment discovery: the `rlwm` env name did not exist; tests run successfully under `ds_env` which has JAX 0.4.31.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Precomputation functions ready for use by Plan 20-02 (vectorized policy likelihood variants)
- Plan 20-02 will replace Phase 2 lax.scan with vectorized array ops using the precomputed last_action arrays
- All 7 precompute tests pass; no blockers

---
*Phase: 20-deer-nonlinear-parallelization*
*Completed: 2026-04-14*
