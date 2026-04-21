---
phase: 11-m4-lba-joint-choice-rt
plan: "01"
subsystem: fitting
tags: [jax, lba, float64, likelihood, joint-choice-rt, brown-heathcote-2008, mcdougle-collins-2021]

# Dependency graph
requires:
  - phase: 10-m6b-dual-perseveration
    provides: M6b fit-all-participants pattern; downstream script integration for new models
provides:
  - "lba_pdf: Brown & Heathcote (2008) defective PDF in float64 with 1e-300 clamp"
  - "lba_cdf: analytic CDF clipped to [0,1] in float64"
  - "lba_sf / lba_log_sf: survivor and log-survivor with 1e-300 floor"
  - "lba_joint_log_lik: per-trial log P(choice=i, RT=t) via vmap, no Python loops"
  - "preprocess_rt_block: outlier filter (150-2000ms) + ms-to-seconds conversion"
  - "validate_t0_constraint: standalone diagnostic utility"
  - "FIXED_S=0.1 constant"
affects:
  - 11-02 (M4 block likelihood in lax.scan)
  - 11-03 (mle_utils M4 parameter bounds + fit_mle dispatch)
  - 11-04 (model_recovery RT simulation)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "float64 LBA module: jax_enable_x64 at module top before any JAX ops"
    - "vmap for multi-accumulator vectorization (no Python loops in JIT path)"
    - "clamp-then-log pattern: jnp.maximum(x, 1e-300) before log"
    - "composite LBA survivor via 1-F_i (not jss.norm.logsf, which only handles single normal)"

key-files:
  created:
    - scripts/fitting/lba_likelihood.py
  modified: []

key-decisions:
  - "lba_joint_log_lik uses log_sf_sum - log_sf_chosen pattern (avoids per-accumulator branch)"
  - "test_lba_joint_log_lik asserts finiteness + fast>slow ordering (not negativity; log-density can be positive when PDF > 1)"
  - "validate_t0_constraint is standalone diagnostic only; structural t0 bounds belong in WMRL_M4_BOUNDS (Plan 02/03)"

patterns-established:
  - "LBA density isolation: new float64 module rather than adding to jax_likelihoods.py (avoids float32 contamination of M1-M6b)"
  - "Inline smoke tests: __main__ block with 8 tests covering dtype, boundary, complement, joint LL, negative drift, RT preprocessing"

# Metrics
duration: 4min
completed: 2026-04-03
---

# Phase 11 Plan 01: LBA Density Module Summary

**Brown & Heathcote (2008) LBA density/CDF/survivor in a standalone float64 JAX module with vmap-based joint log-likelihood and RT preprocessing utilities**

## Performance

- **Duration:** 4 min
- **Started:** 2026-04-03T11:38:17Z
- **Completed:** 2026-04-03T11:42:24Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Created `scripts/fitting/lba_likelihood.py` as a self-contained float64 module; `jax_enable_x64=True` set at module load before any JAX operations
- Implemented `lba_pdf`, `lba_cdf`, `lba_sf`, `lba_log_sf` per Brown & Heathcote (2008) formula verified from Fleming (2012) MATLAB LBA_tpdf.m
- Implemented `lba_joint_log_lik` using `jax.vmap` across accumulators — no Python loops in the JIT path; follows McDougle & Collins (2021) Eq. 9
- Added `preprocess_rt_block` (150-2000ms filter + ms-to-seconds) and `validate_t0_constraint` (standalone diagnostic)
- All 8 inline smoke tests pass; `jax_likelihoods.py` (M1-M6b) unaffected

## Task Commits

Each task was committed atomically:

1. **Task 1: LBA density, CDF, survivor functions (float64)** - `506aa89` (feat)

**Plan metadata:** (to be added in final commit)

## Files Created/Modified

- `scripts/fitting/lba_likelihood.py` - Complete LBA density module with float64, density/CDF/survivor/joint-LL functions, RT preprocessing utilities, and 8 inline smoke tests

## Decisions Made

- `test_lba_joint_log_lik` changed from asserting `ll < 0` to asserting finiteness + `fast_chosen_ll > slow_chosen_ll`. Rationale: a defective LBA PDF can exceed 1.0 (it integrates to less than 1 over time, not over all values), so its log can be positive. The original plan assertion was mathematically incorrect; the corrected test is a stronger check (ordering) while remaining correct.
- `validate_t0_constraint` receives a flat array of filtered RTs (not a list of blocks) to keep the utility simple. The plan's docstring example showed a single-block array, which is what this plan implements. Multi-block validation (looping over blocks) can be added in Plan 03 when it's wired into the fitting pipeline.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed incorrect `assert float(ll) < 0` in test_lba_joint_log_lik**

- **Found during:** Task 1 (verification run)
- **Issue:** Plan specified `assert float(ll) < 0` but LBA defective PDFs can exceed 1.0, making log-density positive. For v=3.0, A=0.5, b=1.0, t=0.3, the PDF is ~6.0 so log-density is ~1.79. The assertion was mathematically wrong.
- **Fix:** Replaced with finiteness check plus ordering check (fast chosen accumulator must have higher joint LL than slow chosen accumulator), which is both correct and more diagnostic.
- **Files modified:** scripts/fitting/lba_likelihood.py
- **Verification:** `python scripts/fitting/lba_likelihood.py` — all 8 tests pass including the corrected test
- **Committed in:** 506aa89 (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 — bug in test spec)
**Impact on plan:** The fix corrects a mathematically invalid assertion to one that is both correct and stronger. No scope change.

## Issues Encountered

None beyond the test assertion fix above.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- `lba_likelihood.py` is ready to import in Plan 02 (M4 block likelihood in `lax.scan`)
- Plan 02 will use `lba_joint_log_lik` inside the scan body, replacing M3's `log(noisy_probs[action])` step
- `preprocess_rt_block` will be called in Plan 03 (`prepare_participant_data` for `model == 'wmrl_m4'`)
- `validate_t0_constraint` should be called in Plan 03 at data-prep time; structural t0 bounds in `WMRL_M4_BOUNDS` are the primary protection

---
*Phase: 11-m4-lba-joint-choice-rt*
*Completed: 2026-04-03*
