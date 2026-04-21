---
phase: 16-choice-only-family-extension-subscale-level-2
plan: "07"
subsystem: bayesian-results-analysis
tags: [forest-plot, bayesian-comparison, arviz, level2-regression, horseshoe]

dependency_graph:
  requires:
    - 16-04  # _L2_LEC_SUPPORTED models, STACKED_MODEL_DISPATCH
    - 16-05  # wmrl_m6b_hierarchical_model_subscale, 32 beta sites
  provides:
    - L2-07: Forest plot script for M6b Level-2 posterior coefficients
    - Success-Criterion-6: az.compare with LOO + stacking weights for all 6 choice-only models
    - L2-08: Horseshoe prior decision documented (deferred with justification)
  affects:
    - 16-08  # Phase summary / integration
    - 18     # Manuscript Phase: forest plots referenced in results section

tech_stack:
  added: []
  patterns:
    - lazy-arviz-import       # _ARVIZ_AVAILABLE guard for environment portability
    - dynamic-beta-discovery  # discover_beta_vars() finds all beta_ sites at runtime
    - graceful-missing-file   # scripts handle absent NetCDF with warning + early return

key_files:
  created:
    - scripts/18_bayesian_level2_effects.py
  modified:
    - scripts/14_compare_models.py

decisions:
  - id: L2-08
    decision: Horseshoe prior DEFERRED
    rationale: >
      The wmrl_m6b subscale model with 32 beta sites has not been run on the cluster.
      Normal(0,1) priors on beta coefficients have not been tested. It is premature to
      add regularized horseshoe priors before observing whether flat priors produce
      convergence issues (divergences, Rhat > 1.01) or implausibly diffuse posteriors.
      The decision gate after cluster job completion: inspect max_rhat for beta_ sites,
      posterior SD relative to prior scale, and number of 95% HDI exculsions. If all
      32 beta sites exclude zero, horseshoe is more appropriate as a skeptical prior.
    status: deferred
    gate: After wmrl_m6b subscale cluster job completes

metrics:
  duration: "4 min"
  completed: "2026-04-13"
---

# Phase 16 Plan 07: Bayesian Level-2 Forest Plots and Model Comparison Summary

**One-liner:** Forest plot script for M6b L2 posterior betas (32 sites) + az.compare stacking weights in script 14.

---

## Objective

Create `scripts/18_bayesian_level2_effects.py` (L2-07) for forest plot visualization of hierarchical Bayesian Level-2 regression coefficients, add `--bayesian-comparison` flag to `scripts/14_compare_models.py` (Success Criterion 6), and document the L2-08 horseshoe prior decision.

---

## Tasks Completed

| # | Name | Commit | Key Files |
|---|------|--------|-----------|
| 1 | Create scripts/18_bayesian_level2_effects.py | d9f64e6 | scripts/18_bayesian_level2_effects.py |
| 2 | Add --bayesian-comparison flag to scripts/14_compare_models.py | c8fb65d | scripts/14_compare_models.py |

---

## Implementation Details

### Task 1: scripts/18_bayesian_level2_effects.py

New numbered pipeline script following all project conventions:

- **CLI:** `--model`, `--posterior-path`, `--output-dir`, `--hdi-prob`
- **Dynamic discovery:** `discover_beta_vars()` finds all `beta_`-prefixed sites at runtime — handles arbitrary numbers of beta sites (32 for subscale, or any other count)
- **Grouped forest plots:**
  - `{model}_forest_lec5.png` — LEC-5 total coefficients
  - `{model}_forest_iesr_residuals.png` — IES-R total + intrusion/avoidance residuals
  - `{model}_forest_all_l2.png` — All beta coefficients
- **Coefficient summary CSV:** `output/bayesian/level2/{model}_l2_coefficient_summary.csv` with columns: variable, mean, sd, hdi_low, hdi_high, hdi_excludes_zero
- **Graceful handling:** Prints warning and returns early if NetCDF file is missing
- **L2-08 documentation:** Printed to stdout with full reasoning for deferral
- **Lazy arviz import:** `_ARVIZ_AVAILABLE` guard for environment portability

### Task 2: --bayesian-comparison in scripts/14_compare_models.py

Added without disrupting existing MLE comparison logic:

- **`BAYESIAN_NETCDF_MAP`:** Module-level dict mapping M1/M2/M3/M5/M6a/M6b to NetCDF paths
- **`_load_bayesian_compare_dict()`:** Loads each posterior with graceful skip on missing file or missing `log_likelihood` group
- **`_pareto_k_summary()`:** Computes `az.loo(pointwise=True)` for each model; reports % of observations with Pareto-k > 0.7
- **`run_bayesian_comparison()`:** Orchestrates loading → `az.compare(ic='loo', method='stacking')` → verdict → write `stacking_weights.md`
- **Verdict logic:** M6b weight >= 0.5 → "M6b is preferred"; otherwise "INCONCLUSIVE"
- **Output:** `output/bayesian/level2/stacking_weights.md` with LOO table, Pareto-k diagnostics, and plain-language verdict
- **Flag is independent:** `--bayesian-comparison` can be combined with MLE comparison or used standalone

---

## Verification

- [x] `scripts/18_bayesian_level2_effects.py --help` prints help text
- [x] `scripts/18_bayesian_level2_effects.py` syntax is valid (ast.parse)
- [x] `scripts/14_compare_models.py --help` recognizes `--bayesian-comparison`
- [x] `grep az.compare scripts/14_compare_models.py` returns match
- [x] Forest plot script handles missing NetCDF gracefully (warning + return)
- [x] Bayesian comparison writes `stacking_weights.md` with verdict
- [x] L2-08 horseshoe decision documented in script output

---

## Deviations from Plan

None — plan executed exactly as written.

The plan mentioned "~40 beta sites" in the objective; the actual count is 32 (8 params x 4 covariates) as locked in 16-01. The `discover_beta_vars()` function handles whatever count exists at runtime, so no code change was needed.

---

## L2-08 Horseshoe Prior Decision

**Status: DEFERRED with justification**

**Reasoning:**
The subscale model (`wmrl_m6b_hierarchical_model_subscale`) with 32 beta sites has not yet been run on the cluster. The Normal(0,1) beta priors have not been tested on real data. Key concerns before enabling horseshoe:

1. **Convergence not yet observed:** No posterior is available to inspect `max_rhat` for beta_ sites. Adding horseshoe to an untested model adds complexity without evidence it is needed.

2. **False positive rate unclear:** The Normal(0,1) prior on the probit scale implies ~68% of the prior mass within [-1, +1] probit units (moderate effect). Whether this is too diffuse for N=160 participants requires empirical observation.

3. **Horseshoe implementation overhead:** The regularized horseshoe (Piironen & Vehtari, 2017) requires a local shrinkage parameter per coefficient and a global shrinkage parameter, doubling the number of hierarchical hyperparameters. This should not be added until simpler priors demonstrably fail.

**Decision gate for enabling L2-08:**
After the cluster job completes and `18_bayesian_level2_effects.py` is run:
- If `max_rhat > 1.01` for any beta_ site: consider horseshoe
- If posterior SD of beta_ sites >> 1 on the probit scale: flat prior is not regularizing
- If all 32 sites exclude zero at 95% HDI: horseshoe as skeptical prior is appropriate

---

## Next Phase Readiness

The final plan in Phase 16 (if any) can safely depend on:
- `scripts/18_bayesian_level2_effects.py` existing and runnable after cluster job
- `scripts/14_compare_models.py --bayesian-comparison` working once posteriors exist
- L2-08 decision gate documented and ready to evaluate post-cluster

No blockers for Phase 17 (M4 Hierarchical LBA) or Phase 18 (Integration, Comparison, Manuscript).
