---
phase: 16-choice-only-family-extension-subscale-level-2
plan: 05
subsystem: bayesian-fitting
tags: [numpyro, hierarchical-bayes, level2-regression, wmrl-m6b, subscale, slurm]

# Dependency graph
requires:
  - phase: 16-01
    provides: level2_design.py with build_level2_design_matrix and COVARIATE_NAMES (4 predictors)
  - phase: 16-03
    provides: wmrl_m6b_hierarchical_model with manual kappa_total/kappa_share sampling pattern
  - phase: 16-04
    provides: STACKED_MODEL_DISPATCH, _fit_stacked_model, fit_model refactored signatures
provides:
  - wmrl_m6b_hierarchical_model_subscale in numpyro_models.py (32 beta sites, 4 covariates x 8 params)
  - --subscale CLI flag in fit_bayesian.py routing to subscale model
  - _load_subscale_design_matrix helper in fit_bayesian.py
  - cluster/13_bayesian_m6b_subscale.slurm with 12h wall-clock and 48G memory
affects:
  - 16-06-and-beyond (subscale model is now dispatchable from CLI and SLURM)
  - phase 18 (manuscript will reference 32 beta sites from subscale posterior)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Manual mu_pr/sigma_pr/z sampling for all 8 params enables uniform multi-covariate L2 shift pattern"
    - "beta_{cov_name}_{param_name} site naming convention for subscale L2 coefficients"
    - "_load_subscale_design_matrix helper: load -> align -> build_level2_design_matrix -> jnp.array(float32)"
    - "subscale=True flag gates model_fn swap inside fit_model before dispatch to _fit_stacked_model"

key-files:
  created:
    - cluster/13_bayesian_m6b_subscale.slurm
  modified:
    - scripts/fitting/numpyro_models.py
    - scripts/fitting/fit_bayesian.py

key-decisions:
  - "32 beta sites (8 params x 4 covariates) not 40 (plan said ~40): hyperarousal residual excluded per 16-01 decision"
  - "All 8 M6b params bypass sample_bounded_param to support multi-covariate L2 shifts uniformly"
  - "beta site naming: beta_{cov_name}_{param_name} — outer loop over params, inner over covariates"
  - "subscale guard: ValueError if --subscale used with model != wmrl_m6b"
  - "beta_* summary print covers all beta_ sites (not just beta_lec_*), supporting subscale 32-site output"
  - "SLURM uses --time=12:00:00 and --mem=48G (vs 8h/32G for standard M6b)"

patterns-established:
  - "subscale=False default in fit_model/(_fit_stacked_model) preserves backward compat with all 6 existing models"
  - "subscale path builds model_args with covariate_matrix/covariate_names (not covariate_lec)"
  - "covariate_names fallback: if None and matrix provided, names as cov0, cov1, etc."

# Metrics
duration: 20min
completed: 2026-04-12
---

# Phase 16 Plan 05: Subscale Level-2 M6b Summary

**wmrl_m6b_hierarchical_model_subscale: 32-beta hierarchical M6b (8 params x 4 covariates) with --subscale CLI dispatch and 12h SLURM script**

## Performance

- **Duration:** ~20 min
- **Started:** 2026-04-12T00:00:00Z
- **Completed:** 2026-04-12T00:20:00Z
- **Tasks:** 2
- **Files modified:** 3 (numpyro_models.py, fit_bayesian.py, + 1 created)

## Accomplishments

- `wmrl_m6b_hierarchical_model_subscale` implemented: all 8 M6b params sampled manually (bypassing `sample_bounded_param`) with multi-covariate L2 shifts applied on the unconstrained probit scale before Phi_approx transform
- 32 beta sites named `beta_{cov_name}_{param_name}` — 8 params x 4 covariates from `level2_design.py`
- `--subscale` CLI flag added to `fit_bayesian.py` with guard (wmrl_m6b only), plus `_load_subscale_design_matrix` helper that calls `build_level2_design_matrix` as single source of truth
- `cluster/13_bayesian_m6b_subscale.slurm` created with `--time=12:00:00` and `--mem=48G`

## Task Commits

Each task was committed atomically:

1. **Task 1: Implement wmrl_m6b_hierarchical_model_subscale** - `4ec4170` (feat)
2. **Task 2: Add subscale dispatch to fit_bayesian.py** - `b0f7c6c` (feat)

## Files Created/Modified

- `scripts/fitting/numpyro_models.py` - Added `wmrl_m6b_hierarchical_model_subscale` at end of file (before `__main__`)
- `scripts/fitting/fit_bayesian.py` - Added `_load_subscale_design_matrix`, `subscale` param to `_fit_stacked_model`/`fit_model`, `--subscale` CLI arg, beta_* display updated
- `cluster/13_bayesian_m6b_subscale.slurm` - Created; 12h wall-clock, 48G memory, runs `--subscale` flag

## Decisions Made

- **32 beta sites (not 40):** Plan said "~40 beta sites" (8 params x 5 covariates). Actual is 32 (8 x 4) because hyperarousal residual is excluded per the 16-01 design lock. The CRITICAL UPDATE in the prompt was honored exactly.
- **beta site naming convention locked:** `beta_{cov_name}_{param_name}` (e.g., `beta_lec_total_kappa_total`, `beta_iesr_intr_resid_alpha_pos`). Outer loop over params, inner loop over covariate_names.
- **subscale guard uses ValueError not NotImplementedError:** The model exists in STACKED_MODEL_DISPATCH; the subscale flag is a variant selector, not a missing-implementation error.
- **beta_* HDI print expanded:** `_fit_stacked_model` now prints all `beta_`-prefixed sites (not just `beta_lec_*`), so the 32-site subscale output is fully displayed post-MCMC.
- **covariate_names fallback:** If `covariate_matrix` is provided but `covariate_names` is None, names default to `cov0, cov1, ...` with a ValueError check on length mismatch.

## Deviations from Plan

None — plan executed as specified, adjusted for the confirmed 4-covariate (32 beta site) design from the CRITICAL UPDATE.

## Issues Encountered

None. The implementation followed the existing M6b manual-sampling pattern (kappa_total/kappa_share) and extended it uniformly to all 8 params. The `beta_{cov}_` prefix change from `beta_lec_` in the standard model was the only structural difference.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- L2-05 subscale M6b infrastructure complete and ready for cluster submission
- Run `sbatch cluster/13_bayesian_m6b_subscale.slurm` on Monash M3 to execute the 32-beta fit
- Beta sites are in ArviZ posterior NetCDF for downstream ArviZ extraction in Phase 18 manuscript
- If divergences persist at 0.95 target_accept_prob, use 0.99 level via `run_inference_with_bump` (already supported by the bump pattern) or upgrade to horseshoe prior (L2-08)

---
*Phase: 16-choice-only-family-extension-subscale-level-2*
*Completed: 2026-04-12*
