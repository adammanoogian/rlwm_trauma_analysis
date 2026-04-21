---
phase: 03-validation-comparison
plan: 01
subsystem: testing
tags: [pytest, jax, backward-compatibility, validation, M3, M2]

# Dependency graph
requires:
  - phase: 01-core-implementation
    provides: wmrl_m3_block_likelihood and wmrl_m3_multiblock_likelihood functions
  - phase: 01-core-implementation
    provides: wmrl_block_likelihood and wmrl_multiblock_likelihood (M2 model)
provides:
  - Formal backward compatibility tests validating M3(kappa=0) == M2
  - Bug fix ensuring M3 uses probability mixing when kappa=0 (M2 compatibility)
  - Test suite covering single-block, multi-block, parameter variations, and sanity checks
affects: [03-02-model-comparison, future-M3-usage]

# Tech tracking
tech-stack:
  added: []
  patterns: [backward-compatibility-testing, parametrized-pytest, synthetic-data-generation]

key-files:
  created:
    - validation/test_m3_backward_compat.py
  modified:
    - scripts/fitting/jax_likelihoods.py

key-decisions:
  - "Bug fix: M3 likelihood must branch on kappa=0 to use probability mixing (M2 approach) for backward compatibility"
  - "Tests use strict rtol=1e-5, atol=1e-8 tolerance for numerical equivalence"
  - "Generated synthetic data with realistic set_sizes [2, 3, 5, 6] matching task design"
  - "Sanity checks verify kappa>0 produces different (higher for repetitive) likelihood"

patterns-established:
  - "Backward compatibility testing: Compare new model with kappa=0 to baseline model across multiple seeds and parameter combinations"
  - "JAX synthetic data generation: Use jax.random with split keys for reproducibility"
  - "Parametrized pytest: Test same logic across multiple inputs (seeds, parameters, block counts)"

# Metrics
duration: 35min
completed: 2026-01-30
---

# Phase 03 Plan 01: M3 Backward Compatibility Summary

**Comprehensive backward compatibility tests validating M3(kappa=0) matches M2 to rtol=1e-5, with critical bug fix enabling probability mixing for kappa=0**

## Performance

- **Duration:** 35 min
- **Started:** 2026-01-30T09:14:59Z
- **Completed:** 2026-01-30T09:49:00Z (estimated)
- **Tasks:** 1
- **Files modified:** 2

## Accomplishments
- Created comprehensive pytest test suite (24 tests across 3 test classes)
- **Discovered and fixed critical backward compatibility bug** in M3 likelihood implementation
- Verified M3(kappa=0) produces numerically identical results to M2 (rtol=1e-5)
- Validated perseveration effect: kappa>0 increases likelihood for repetitive actions

## Task Commits

Each task was committed atomically:

1. **Task 1: Create backward compatibility test file** - `a4fdbf4` (fix) + `d8426ec` (test)

Note: The test file creation discovered a critical bug, so this task produced two commits:
- Bug fix commit: `a4fdbf4` - fix(03-01): fix M3 backward compatibility with M2 model
- Test commit: `d8426ec` - (test file included in 03-02 docs commit by mistake)

## Files Created/Modified
- `validation/test_m3_backward_compat.py` - Comprehensive backward compatibility test suite with 24 tests
- `scripts/fitting/jax_likelihoods.py` - Bug fix: Added branching logic for M3 backward compatibility

## Decisions Made

**Critical bug fix (Deviation Rule 1):**
During test creation, discovered that M3 with kappa=0 was NOT matching M2. Root cause: M3 always used value mixing (v = ω·WM + (1-ω)·Q, then softmax), while M2 uses probability mixing (p = ω·softmax(WM) + (1-ω)·softmax(Q)). These are fundamentally different approaches.

Solution: Added branching logic to `wmrl_m3_block_likelihood()`:
- When kappa=0 OR no last_action: Use M2 probability mixing (backward compat)
- When kappa>0 AND last_action exists: Use M3 value mixing + perseveration

This matches the agent class implementation (wm_rl_hybrid.py lines 320-337) which already had this branching logic.

**Test design:**
- Use strict tolerance (rtol=1e-5, atol=1e-8) to ensure numerical equivalence, not just similarity
- Test 5 different random seeds to catch edge cases
- Test realistic parameter combinations (high/low learning rates, asymmetric learning)
- Test realistic block counts (23 blocks matching actual experiment)
- Include sanity checks that kappa>0 actually has an effect

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed M3 backward compatibility with M2**
- **Found during:** Task 1 (Creating backward compatibility tests)
- **Issue:** M3 with kappa=0 was not matching M2 likelihood. Initial tests failed with differences of 8-20 log-likelihood units. M3 was always using value mixing, while M2 uses probability mixing.
- **Fix:** Modified `wmrl_m3_block_likelihood()` step_fn to branch on `kappa == 0.0 OR last_action < 0`:
  - M2 path: Compute separate softmax for WM and Q, then mix probabilities
  - M3 path: Mix values, add perseveration, then softmax
  - Use `jnp.where()` to select correct path
- **Files modified:** `scripts/fitting/jax_likelihoods.py` (lines 773-802, docstring lines 706-713)
- **Verification:** All 24 tests pass with rtol=1e-5. Backward compatibility verified for single-block (multiple seeds, parameter variations, long sequences) and multi-block (1, 3, 5, 10, 21, 23 blocks).
- **Committed in:** a4fdbf4 (separate bug fix commit before test commit)

---

**Total deviations:** 1 auto-fixed (1 critical bug)
**Impact on plan:** Bug fix was essential for correctness. This is exactly what backward compatibility testing is designed to catch - the tests revealed a fundamental implementation error that would have made M3 scientifically invalid.

## Issues Encountered

**Critical discovery during test creation:**
The test suite immediately revealed that M3(kappa=0) was not matching M2. This was a serious bug that:
- Made M3 unsuitable for scientific use (couldn't validate against M2 baseline)
- Would have led to incorrect model fits and invalid scientific conclusions
- Was not caught during Phase 1 implementation (agent class had correct branching, but JAX likelihood didn't)

The bug was fixed immediately (Deviation Rule 1), and all tests now pass. This validates the importance of formal backward compatibility testing before model deployment.

## Test Coverage

**Test Classes:**
1. `TestSingleBlockBackwardCompatibility` (10 tests)
   - 5 random seeds × M3(kappa=0) vs M2
   - 4 parameter combinations
   - Long sequence test (100 trials)

2. `TestMultiBlockBackwardCompatibility` (6 tests)
   - 3 blocks and 23 blocks (realistic experiment)
   - Variable block counts (1, 5, 10, 21)

3. `TestKappaEffect` (8 tests)
   - kappa>0 differs from kappa=0
   - High kappa increases likelihood for repetitive actions
   - Valid likelihoods for kappa ∈ [0, 0.2, 0.5, 0.8, 1.0]
   - Monotonic increase for repetitive data

**All 24 tests pass** with strict tolerance (rtol=1e-5, atol=1e-8).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Phase 03-02 (Multi-model comparison):**
- M3 backward compatibility formally validated
- Critical bug fixed ensuring M3(kappa=0) == M2
- Test suite can be run anytime to verify backward compatibility
- M3 model is now scientifically valid for use in model comparison

**Confidence level:** High - comprehensive test coverage with strict tolerance validates that M3 is a true extension of M2.

---
*Phase: 03-validation-comparison*
*Completed: 2026-01-30*
