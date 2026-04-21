---
phase: 16-choice-only-family-extension-subscale-level-2
plan: 03
subsystem: fitting
tags: [numpyro, hierarchical-bayes, jax, hBayesDM, wmrl, M5, M6a, M6b, stick-breaking, non-centered]

# Dependency graph
requires:
  - phase: 15-m3-hierarchical-poc-subscale-level-2
    provides: wmrl_m3_hierarchical_model template, sample_bounded_param, PARAM_PRIOR_DEFAULTS, hBayesDM convention
  - phase: 16-choice-only-family-extension-subscale-level-2
    provides: 16-01 collinearity audit, 16-02 M1/M2 stacked models (prerequisite commit context)
provides:
  - wmrl_m5_hierarchical_model (8 params: 7 via loop + kappa manual, HIER-04)
  - wmrl_m6a_hierarchical_model (7 params: kappa_s replaces kappa, HIER-05)
  - wmrl_m6b_hierarchical_model (8 params: kappa_total/kappa_share stick-breaking decode, HIER-06)
  - All three models support optional covariate_lec L2 regression
affects:
  - fit_bayesian.py dispatch table (must register wmrl_m5/m6a/m6b dispatch)
  - bayesian_diagnostics.py (pointwise log-lik dispatch for M5/M6a/M6b)
  - Phase 17 M4 hierarchical (uses M6b as complexity template)
  - Phase 18 WAIC/LOO comparison table

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "M5/M6a pattern: extend M3 by adding phi_rl to sample_bounded_param loop or swapping kappa->kappa_s"
    - "M6b stick-breaking: kappa_total/kappa_share sampled independently; decoded per-participant BEFORE likelihood call"
    - "Dual L2 regression: two beta coefficients (beta_lec_kappa_total, beta_lec_kappa_share) for M6b"
    - "PARAM_PRIOR_DEFAULTS drives all mu_prior_loc: phi_rl=-0.8, kappa_s=-2.0, kappa_total=-2.0, kappa_share=0.0"

key-files:
  created: []
  modified:
    - scripts/fitting/numpyro_models.py

key-decisions:
  - "M5 kappa stays manually sampled (same pattern as M3) — not moved into sample_bounded_param loop — to preserve L2 regression hook"
  - "M6b decodes kappa=kappa_total*kappa_share and kappa_s=kappa_total*(1-kappa_share) inside the for-loop, not in the likelihood"
  - "M6b likelihood receives decoded kappa/kappa_s, not kappa_total/kappa_share directly"
  - "phi_rl uses mu_prior_loc=-0.8 matching PARAM_PRIOR_DEFAULTS (same as phi)"
  - "kappa_share uses mu_prior_loc=0.0 (group-mean share near 0.5 a priori on probit scale)"

patterns-established:
  - "kappa decode pattern: locked in STATE.md; kappa=kappa_total*kappa_share; kappa_s=kappa_total*(1-kappa_share)"
  - "All manually-sampled parameters use .expand([n_participants]) not numpyro.plate"
  - "All three new models follow sorted(participant_data_stacked.keys()) for participant ordering"

# Metrics
duration: 9min
completed: 2026-04-13
---

# Phase 16 Plan 03: M5/M6a/M6b Hierarchical Models Summary

**Three hierarchical NumPyro models added — M5 (phi_rl), M6a (kappa_s), M6b (stick-breaking dual perseveration) — all following the hBayesDM non-centered M3 template with optional LEC-total L2 regression**

## Performance

- **Duration:** 9 min
- **Started:** 2026-04-13T08:03:02Z
- **Completed:** 2026-04-13T08:11:32Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added `wmrl_m5_hierarchical_model` (HIER-04): 8 params total — 7 via `sample_bounded_param` (alpha_pos, alpha_neg, phi, rho, capacity, epsilon, phi_rl) plus kappa sampled manually with L2 shift
- Added `wmrl_m6a_hierarchical_model` (HIER-05): 7 params — kappa_s replaces kappa; uses same manual sampling pattern with `kappa_s_mu_pr` / `kappa_s_sigma_pr` / `kappa_s_z` sites
- Added `wmrl_m6b_hierarchical_model` (HIER-06): 8 params — 6 standard via loop + kappa_total and kappa_share each sampled manually; decoded per-participant inside the for-loop before passing to likelihood
- Added imports for `wmrl_m5_multiblock_likelihood_stacked`, `wmrl_m6a_multiblock_likelihood_stacked`, `wmrl_m6b_multiblock_likelihood_stacked` in the module-level import block

## Task Commits

Tasks 1 and 2 were both implemented in the file already committed as part of the sequential session context:

1. **Task 1: Add wmrl_m5_hierarchical_model and wmrl_m6a_hierarchical_model** - `d61877d` (feat)
2. **Task 2: Add wmrl_m6b_hierarchical_model** - `d61877d` (feat, same commit — both tasks touched numpyro_models.py)

_Note: Both tasks were committed together as d61877d during session execution; the working tree was clean when this plan ran, confirming all changes are persisted._

## Files Created/Modified

- `scripts/fitting/numpyro_models.py` — Added imports for M5/M6a/M6b stacked likelihoods; added three new hierarchical model functions

## Decisions Made

- M5 keeps kappa manually sampled (outside `sample_bounded_param`) to maintain L2 regression hook, consistent with M3 template pattern
- M6b per-participant decode is the correct location for the stick-breaking transform — putting it in the likelihood would hide the latent parameterization from MCMC and break posterior interpretation
- phi_rl and phi share the same `mu_prior_loc=-0.8` (both are forgetting rates; symmetric prior justification)
- kappa_share gets `mu_prior_loc=0.0` (no prior preference for global vs. stimulus-specific split a priori)

## Deviations from Plan

None — plan executed exactly as written. The three model functions and their imports were added in strict accordance with the M3 template and PARAM_PRIOR_DEFAULTS values specified in the plan.

## Issues Encountered

None. The Edit tool initially reported the file had been modified since last read (a linter or prior session had already applied the same changes), but all verification checks confirmed the correct implementation was in place and importable.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- All three choice-only family extension hierarchical models (HIER-04, HIER-05, HIER-06) are complete and importable
- `fit_bayesian.py` dispatch table must be updated in the next plan to route `--model wmrl_m5 / wmrl_m6a / wmrl_m6b` to the new functions
- `bayesian_diagnostics.py` compute_pointwise_log_lik dispatch needs M5/M6a/M6b entries (stacked likelihood calls)
- M6b kappa decode in diagnostics: `kappa_total*kappa_share -> kappa`, `kappa_total*(1-kappa_share) -> kappa_s` must mirror the per-participant decode used here

---
*Phase: 16-choice-only-family-extension-subscale-level-2*
*Completed: 2026-04-13*
