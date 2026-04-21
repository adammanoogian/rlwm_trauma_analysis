---
phase: 13-infrastructure-repair-hierarchical-scaffolding
plan: 04
subsystem: infra
tags: [numpyro, jax, hierarchical-bayes, non-centered-parameterization, phi-approx, hBayesDM]

# Dependency graph
requires:
  - phase: 13-01
    provides: numpyro_models.py resurrected, deps pinned
  - phase: 13-02
    provides: K bounds [2,6] locked, K_PARAMETERIZATION.md written

provides:
  - scripts/fitting/numpyro_helpers.py with phi_approx, sample_bounded_param, sample_capacity, PARAM_PRIOR_DEFAULTS, sample_model_params
  - config.py EXPECTED_PARAMETERIZATION dict (7 models) and load_fits_with_validation()
  - 10-test suite for transforms, MCMC recovery, and validation error handling

affects: [13-05, 15-hier-poc, 16-choice-family, 17-m4-hierarchical, 18-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "hBayesDM non-centered convention: mu_pr ~ Normal, sigma_pr ~ HalfNormal(0.2), z ~ Normal(0,1), theta = lower + (upper-lower)*Phi_approx(mu_pr + sigma_pr*z)"
    - "Phi_approx = jax.scipy.stats.norm.cdf (NOT expit, NOT polynomial approximation)"
    - "parameterization_version string column in fit CSVs as fail-loud validation gate"
    - "PARAM_PRIOR_DEFAULTS catalogue with shifted mu_pr_loc for biologically-informed priors"

key-files:
  created:
    - scripts/fitting/numpyro_helpers.py
    - scripts/fitting/tests/test_numpyro_helpers.py
  modified:
    - config.py

key-decisions:
  - "phi_approx = jax.scipy.stats.norm.cdf (standard normal CDF, no custom approximation)"
  - "float32 phi_approx saturates at ~±6 sigma; test_phi_approx_bounds uses ±5 not ±10"
  - "PARAM_PRIOR_DEFAULTS excludes LBA-only params (v_scale, A, delta, t0) — handled separately in M4 model"
  - "load_fits_with_validation raises with expected vs actual in error message"

patterns-established:
  - "All bounded params sampled via sample_bounded_param (single implementation, no reinvention per model)"
  - "capacity always via sample_capacity wrapper (enforces [2,6] convention, documents Senta 2025)"
  - "parameterization_version validated on CSV load — downstream scripts must call load_fits_with_validation"

# Metrics
duration: 7min
completed: 2026-04-12
---

# Phase 13 Plan 04: Non-Centered Parameterization Helpers Summary

**hBayesDM-convention numpyro_helpers.py with phi_approx=norm.cdf, sample_bounded_param for 11 params, K in [2,6] via sample_capacity, and config.py parameterization_version validation with fail-loud error on stale v3.0 fits**

## Performance

- **Duration:** 7 min
- **Started:** 2026-04-12T09:03:11Z
- **Completed:** 2026-04-12T09:10:01Z
- **Tasks:** 2
- **Files modified:** 3

## Accomplishments

- `numpyro_helpers.py` created with the hBayesDM non-centered transform convention used across all 11 RLWM parameters; phi_approx wraps jax.scipy.stats.norm.cdf directly (no polynomial approximation or expit)
- `sample_capacity` enforces K in [2, 6] matching Senta, Bishop, Collins (2025) PLOS Comp Biol bounds; `sample_model_params` enables one-call parameter sampling from MODEL_REGISTRY entries
- `config.py` extended with `EXPECTED_PARAMETERIZATION` dict and `load_fits_with_validation()` that raises ValueError with expected vs actual values on mismatch — prevents stale v3.0 (K in [1,7]) fits from contaminating v4.0 hierarchical analyses
- 10 tests pass: phi_approx values/bounds, range correctness for all params, MCMC recovery (alpha_pos and K within 5% rel error), PARAM_PRIOR_DEFAULTS completeness, and 3 validation error-handling tests

## Task Commits

Each task was committed atomically:

1. **Task 1: Create numpyro_helpers.py** - `0031071` (feat)
2. **Task 2: Add parameterization validation to config.py and write unit tests** - `2707bf8` (feat)

**Plan metadata:** (committed below)

## Files Created/Modified

- `scripts/fitting/numpyro_helpers.py` - phi_approx, sample_bounded_param, sample_capacity, PARAM_PRIOR_DEFAULTS, sample_model_params
- `scripts/fitting/tests/test_numpyro_helpers.py` - 10 tests (8 fast, 2 slow MCMC recovery)
- `config.py` - EXPECTED_PARAMETERIZATION dict and load_fits_with_validation() function

## Decisions Made

- **phi_approx = jax.scipy.stats.norm.cdf**: locked. Not expit (wrong shape), not polynomial approximation (no need in JAX). Named for grep-ability.
- **float32 saturation at ±6 sigma**: test_phi_approx_bounds uses ±5 (phi_approx(5) ~ 0.9999997, phi_approx(-5) ~ 2.9e-7) instead of ±10 (which saturates to exact 0.0/1.0 in float32).
- **LBA params excluded from PARAM_PRIOR_DEFAULTS**: v_scale, A, delta, t0 are not bounded [0,1] or [2,6] — they require log-scale or non-standard transforms handled inside the M4 hierarchical model separately. test_param_prior_defaults_completeness hard-codes this exclusion set.
- **load_fits_with_validation raises with expected vs actual**: error message always includes both values per project convention ("Error messages must include expected vs actual values").

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed test_phi_approx_bounds for float32 saturation**

- **Found during:** Task 2 test execution
- **Issue:** Plan specified `phi_approx(-10.0) > 0` and `phi_approx(10.0) < 1.0` but float32 norm.cdf saturates to exactly 1.0 at ±7+ sigma
- **Fix:** Changed test to use ±5.0 which remains non-degenerate in float32 (phi_approx(5) = 0.9999997, phi_approx(-5) = 2.9e-7)
- **Files modified:** scripts/fitting/tests/test_numpyro_helpers.py
- **Verification:** test_phi_approx_bounds passes after fix
- **Committed in:** 2707bf8 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - bug in test assertion)
**Impact on plan:** Necessary correctness fix for float32 arithmetic. No scope creep.

## Issues Encountered

None beyond the float32 saturation deviation documented above.

## Next Phase Readiness

- `numpyro_helpers.py` is the foundational import for Plans 13-05 (bayesian_diagnostics), Phase 15 (M3 hierarchical POC), and all subsequent hierarchical models in Phases 16-17
- `load_fits_with_validation()` in config.py is ready for scripts 15/16 to adopt when loading individual-level MLE fits
- Phase 14 (MLE refit with K in [2,6]) must stamp `parameterization_version = "v4.0-K[2,6]-phiapprox"` on output CSVs for downstream validation to pass
- No blockers for Phase 13 Wave 2 completion

---
*Phase: 13-infrastructure-repair-hierarchical-scaffolding*
*Completed: 2026-04-12*
