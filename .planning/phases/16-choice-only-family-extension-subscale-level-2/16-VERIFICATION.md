---
phase: 16-choice-only-family-extension-subscale-level-2
verified: 2026-04-12T00:00:00Z
status: passed
score: 7/7 must-haves verified
---

# Phase 16: Choice-Only Family Extension + Subscale Level-2 Verification Report

**Phase Goal:** Mechanically extend the Phase 15 M3 template to the rest of the choice-only family (M1, M2, M5, M6a, M6b) and lock the full subscale Level-2 parameterization on the winning model M6b. Includes collinearity audit (Pitfall 3) and permutation null test (Pitfall 2) infrastructure.
**Verified:** 2026-04-12T00:00:00Z
**Status:** PASSED
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | All 5 new choice-only hierarchical models exist and are wired into STACKED_MODEL_DISPATCH | VERIFIED | numpyro_models.py defines all 5 at lines 1111/1211/1321/1478/1627; all imported and dispatched in fit_bayesian.py lines 51-81 |
| 2 | IES-R subscale collinearity audit completed with PASS verdict and Gram-Schmidt residualization | VERIFIED | output/bayesian/level2/ies_r_collinearity_audit.md exists; full design condition number 11.32 < 30; PASS verdict; _residualize() implements Gram-Schmidt at level2_design.py line 114 |
| 3 | Full M6b subscale Level-2 model with 32 beta coefficients | VERIFIED | wmrl_m6b_hierarchical_model_subscale (line 1835); 8 params x 4 covariates = 32 sites; --subscale flag in fit_bayesian.py dispatches to it |
| 4 | Permutation null test: 50-shuffle infrastructure ready | VERIFIED | cluster/13_bayesian_permutation.slurm has --array=0-49; _run_permutation_shuffle() (line 698) is substantive; aggregate_permutation_results.py (213 lines) has p-value logic |
| 5 | Forest plots for M6b Level-2 posterior via script 18 | VERIFIED | scripts/18_bayesian_level2_effects.py (454 lines); discover_beta_vars(), make_forest_plot() defined and wired in main() |
| 6 | az.compare with stacking weights in script 14 | VERIFIED | run_bayesian_comparison() at line 639; az.compare(ic='loo', method='stacking') at line 674; --bayesian-comparison flag at line 818 |
| 7 | L2-08 horseshoe prior decision documented | VERIFIED | Script 18 prints L2-08 decision block (lines 391-438) with STATUS: DEFERRED and full reasoning |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| scripts/fitting/numpyro_models.py | 5 new hierarchical model functions + subscale variant | VERIFIED | 2054 lines; 6 new functions; 82 numpyro.sample/factor calls; no stubs |
| scripts/fitting/fit_bayesian.py | STACKED_MODEL_DISPATCH, _fit_stacked_model, --subscale, --permutation-shuffle | VERIFIED | 1018 lines; all 4 elements present and wired |
| scripts/fitting/level2_design.py | build_level2_design_matrix with Gram-Schmidt, run_collinearity_audit | VERIFIED | 568 lines; _residualize() at line 114; COVARIATE_NAMES = 4 predictors; both functions exported |
| scripts/18_bayesian_level2_effects.py | Forest plot generation for M6b Level-2 posterior | VERIFIED | 454 lines; discover_beta_vars, make_forest_plot, compute_coefficient_summary all present |
| scripts/14_compare_models.py | --bayesian-comparison flag and run_bayesian_comparison() | VERIFIED | 1116 lines; function at line 639; flag at line 818 |
| scripts/fitting/aggregate_permutation_results.py | Permutation null test aggregation with p-value | VERIFIED | 213 lines; load_shuffle_results() and full aggregation; 5% nominal alpha verdict |
| output/bayesian/level2/ies_r_collinearity_audit.md | Collinearity audit with PASS verdict | VERIFIED | File exists; condition number 11.32 < 30; PASS verdict |
| cluster/13_bayesian_m{1,2,3,5,6a,6b}.slurm | One SLURM script per model | VERIFIED | All 6 files confirmed; each passes correct --model flag |
| cluster/13_bayesian_m6b_subscale.slurm | M6b subscale SLURM with --subscale flag | VERIFIED | File exists; passes --subscale at line 113 |
| cluster/13_bayesian_permutation.slurm | Permutation array job with --array=0-49 | VERIFIED | --array=0-49 at line 25; passes --permutation-shuffle SLURM_ARRAY_TASK_ID |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| fit_bayesian.py | numpyro_models.py | import + STACKED_MODEL_DISPATCH | WIRED | 6 models imported at lines 47-56; dispatched in dict at lines 75-82 |
| fit_bayesian.py | level2_design.py | _load_subscale_design_matrix() | WIRED | Imports build_level2_design_matrix + COVARIATE_NAMES at lines 248-250 |
| wmrl_m6b_hierarchical_model_subscale | beta sites (32) | double loop over param_names x covariate_names | WIRED | Lines 1971-1976; numpyro.sample per (param, cov) pair |
| 18_bayesian_level2_effects.py | forest plots | discover_beta_vars() -> make_forest_plot() in main() | WIRED | Lines 338-361 call make_forest_plot 3 times |
| 14_compare_models.py | run_bayesian_comparison() | --bayesian-comparison flag | WIRED | Lines 1085-1088 dispatch on args.bayesian_comparison |
| cluster/13_bayesian_permutation.slurm | _run_permutation_shuffle() | --permutation-shuffle flag | WIRED | Line 93 passes --permutation-shuffle SLURM_ARRAY_TASK_ID |

### Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|----------------|
| HIER-02: Q-learning stacked hierarchical model | SATISFIED | None |
| HIER-03: WM-RL stacked hierarchical model | SATISFIED | None |
| HIER-04: M5 (WM-RL+phi_rl) hierarchical model | SATISFIED | None |
| HIER-05: M6a (WM-RL+kappa_s) hierarchical model | SATISFIED | None |
| HIER-06: M6b (WM-RL+dual) hierarchical model | SATISFIED | None |
| L2-02: IES-R subscale collinearity audit | SATISFIED | None |
| L2-03: Orthogonalized subscale parameterization | SATISFIED | Deviation accepted: hyperarousal dropped (linear dependence) |
| L2-04: LEC-5 subcategory predictors | SATISFIED (deviation) | Data not available; deviation documented; accepted per phase spec |
| L2-05: Full subscale Level-2 fit on M6b (~32 coefficients) | SATISFIED | Infrastructure complete; cluster execution pending |
| L2-06: Permutation null test infrastructure | SATISFIED | None |
| L2-07: Forest plots via 18_bayesian_level2_effects.py | SATISFIED | None |
| L2-08: Horseshoe prior decision documented | SATISFIED (P2 optional, deferred) | Deferred with documented gate condition |

### Known Deviations (Accepted Per Phase Spec)

1. 4 predictors instead of 5-6: IES-R subscales sum exactly to IES-R total (rank deficiency); hyperarousal dropped. Condition number 11.32 < 30. Documented in audit report.
2. 32 beta coefficients instead of 48: Follows from 4 predictors x 8 params.
3. Cluster jobs not yet run: Infrastructure complete. Convergence gate requires cluster execution. Accepted per success criteria.
4. L2-08 horseshoe deferred: P2 optional; deferral documented with gate condition.

### Anti-Patterns Found

None. No TODO/FIXME/placeholder patterns in any verified file. No empty return stubs. All model functions contain substantive numpyro sampling code.

### Human Verification Required

None required for infrastructure verification. Post-cluster checks (after cluster execution):
- Inspect Rhat diagnostics for all 6 models
- Verify 32 beta sites appear in M6b subscale posterior NetCDF
- Review forest plots for visual correctness

## Gaps Summary

No gaps. All 7 observable truths verified, all 10 artifacts pass all three levels, all 12 requirements satisfied (L2-04 and L2-08 with documented deviations accepted per phase spec).

---

_Verified: 2026-04-12T00:00:00Z_
_Verifier: Claude (gsd-verifier)_
