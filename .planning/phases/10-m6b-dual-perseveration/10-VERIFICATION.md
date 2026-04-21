---
phase: 10-m6b-dual-perseveration
verified: 2026-04-03T09:38:08Z
status: human_needed
score: 3/4 must-haves verified (1 deferred to cluster compute)
human_verification:
  - test: Run full parameter recovery for M6b and M6a on cluster
    expected: All 8 M6b parameters achieve r >= 0.80 across N=50 subjects, 10 synthetic datasets
    why_human: Full recovery requires cluster compute. The N=2 quick test confirmed code runs without error but r values with 2 data points are meaningless. Run: python scripts/11_run_model_recovery.py --model wmrl_m6b --n-subjects 50 --n-datasets 10 --n-jobs 8
---

# Phase 10: M6b Dual Perseveration Verification Report

**Phase Goal:** Users can fit a dual-perseveration model combining global and stimulus-specific kernels with a constraint that their sum stays at or below 1
**Verified:** 2026-04-03T09:38:08Z
**Status:** human_needed (3/4 automated checks pass; parameter recovery gate deferred to cluster)
**Re-verification:** No - initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Running python scripts/12_fit_mle.py --model wmrl_m6b completes without error and writes fit results | VERIFIED | output/mle/wmrl_m6b_individual_fits.csv exists with 46 rows, correct 8-param columns (kappa_total, kappa_share), AIC/BIC, convergence flags |
| 2 | Stick-breaking reparameterization enforces kappa + kappa_s <= 1 for all feasible parameters | VERIFIED | kappa_total in [0,1], kappa_share in [0,1]; decode: kappa = kappa_total * kappa_share, kappa_s = kappa_total * (1 - kappa_share); sum = kappa_total <= 1 by construction. Inline tests verify kappa_share=1.0 matches M3 and kappa_share=0.0 matches M6a exactly (diff < 1e-6). |
| 3 | Parameter recovery passes r >= 0.80 for all M6a (7 params) and M6b (8 params) parameters | NEEDS HUMAN | Recovery infrastructure exists and runs. N=2 smoke test in output/recovery/wmrl_m6b/ confirmed. Full N=50/10-dataset run not yet performed. r values from N=2 are statistically meaningless. |
| 4 | M6b appears in the choice-only AIC/BIC comparison table alongside M1-M3, M5, and M6a | VERIFIED | Script 14 auto-detects wmrl_m6b_individual_fits.csv from default output/mle/ dir. M6b registered in patterns dict alongside M1-M3, M5, M6a. CSV file exists with 46/46 converged. |

**Score:** 3/4 truths verified (1 needs cluster compute)
### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| scripts/fitting/jax_likelihoods.py | wmrl_m6b_block_likelihood + multiblock variants | VERIFIED | 3610 lines; wmrl_m6b_block_likelihood at line 2341 with full lax.scan dual-carry (6-element tuple). 4 inline tests at lines 3361-3537. All tests pass when run. |
| scripts/fitting/mle_utils.py | WMRL_M6B_BOUNDS, WMRL_M6B_PARAMS, transform functions | VERIFIED | 1321 lines; WMRL_M6B_BOUNDS at line 82, WMRL_M6B_PARAMS at line 108, transform functions at lines 333-375. All dispatch functions updated. |
| scripts/fitting/fit_mle.py | CLI --model wmrl_m6b, 3 objective functions, all dispatch points | VERIFIED | 2777 lines; wmrl_m6b in argparse choices at line 2487; _make_jax_objective_wmrl_m6b (490), _make_bounded_objective_wmrl_m6b (785), _gpu_objective_wmrl_m6b (991); param_cols dispatch at line 2342. |
| scripts/fitting/model_recovery.py | M6b parameter recovery with dual-kernel synthetic generation | VERIFIED | 1336 lines; wmrl_m6b in all 5 dispatch functions. Dual-kernel simulation with stick-breaking decode at lines 225-230. |
| scripts/11_run_model_recovery.py | wmrl_m6b in argparse choices and all expansion | VERIFIED | wmrl_m6b in choices at line 134; in all expansion at line 154. |
| scripts/14_compare_models.py | M6b in AIC/BIC patterns dict, --m6b argparse | VERIFIED | M6b in patterns dict at line 548; --m6b arg at line 583; auto-detection from default output/mle/. |
| scripts/15_analyze_mle_by_trauma.py | M6b load path, MODEL_CONFIG, param display names | VERIFIED | wmrl_m6b_path detection at lines 154-157; MODEL_CONFIG conditional entry at line 813; kappa_total/kappa_share in PARAM_NAMES. |
| scripts/16_regress_parameters_on_scales.py | M6b param_cols dispatch, kappa renames, all expansion | VERIFIED | wmrl_m6b param_cols at line 793; all expansion at line 735; argparse choices at line 681. |
| output/mle/wmrl_m6b_individual_fits.csv | 46 participants, 8 parameter columns, AIC/BIC | VERIFIED | 47 lines (header + 46 rows); all 8 M6b params plus nll, aic, bic, aicc, pseudo_r2, converged. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| fit_mle.py --model wmrl_m6b | wmrl_m6b_multiblock_likelihood_stacked | 3 objective functions | WIRED | All 3 objectives call stacked likelihood with stick-breaking decode applied before the call |
| _make_jax_objective_wmrl_m6b | stick-breaking constraint | kappa = kappa_total * kappa_share | WIRED | Lines 527-529; identical decode in bounded (821-822) and GPU (1008-1010) objectives |
| model_recovery.py | dual-kernel synthetic generation | generate_synthetic_participant | WIRED | Stick-breaking decode at lines 227-230; both global last_action and per-stimulus last_actions dict maintained independently |
| script 14 | wmrl_m6b_individual_fits.csv | auto-detect in find_mle_files() | WIRED | patterns dict at line 548 includes M6b; file exists at output/mle/wmrl_m6b_individual_fits.csv |
| script 15 load_data() | wmrl_m6b CSV | defensive load mle/ then output/ | WIRED | Lines 154-157; returns 8-tuple including wmrl_m6b; conditional MODEL_CONFIG entry |

### Requirements Coverage

| Requirement | Status | Notes |
|-------------|--------|-------|
| M6-07: JAX likelihood with dual kernels | SATISFIED | wmrl_m6b_block_likelihood with 6-element dual carry at line 2341 |
| M6-08: Stick-breaking reparameterization | SATISFIED | kappa_total * kappa_share decode in all 3 objectives; enforces sum <= 1 by construction |
| M6-09: MLE bounds, transforms, param names | SATISFIED | WMRL_M6B_BOUNDS, WMRL_M6B_PARAMS, jax_unconstrained/bounded transforms all present and dispatched |
| M6-10: CLI --model wmrl_m6b | SATISFIED | argparse choices at line 2487; all internal dispatch points cover wmrl_m6b |
| M6-11: Parameter recovery r >= 0.80 for M6a and M6b | NEEDS HUMAN | Infrastructure complete; N=2 smoke test confirms code runs; full N=50 cluster run required |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| scripts/fitting/jax_likelihoods.py | 328-339 | Placeholder comment and hardcoded values | INFO | In q_learning_step function which is defined but never called (zero call sites). Dead code from early development. No impact on M6b or any active fitting path. |

### Human Verification Required

#### 1. Full Parameter Recovery (M6b and M6a)

**Test:** Run on cluster: python scripts/11_run_model_recovery.py --model wmrl_m6b --n-subjects 50 --n-datasets 10 --n-jobs 8. Also run for M6a if not already done at N=50.

**Expected:** All 8 M6b parameters achieve Pearson r >= 0.80 in recovery metrics at output/recovery/wmrl_m6b/recovery_metrics.csv. Column pass_fail shows PASS for all rows.

**Why human:** Requires cluster compute. The N=2 quick test in output/recovery/wmrl_m6b/ confirms the pipeline runs end-to-end without error, but r values with 2 data points are meaningless (forced to exactly +/-1.0, p-values all 1.0 due to degenerate regression). The r >= 0.80 threshold can only be evaluated with N >= 20 subjects.

### Gaps Summary

No structural gaps were found. All code artifacts exist, are substantive, and are wired into the pipeline. The only outstanding item is the full parameter recovery run (r >= 0.80 gate) which was explicitly deferred to cluster compute and noted as a blocker in SUMMARY.md for phase 10-02. This is expected behavior, not a structural gap.

The placeholder comment at jax_likelihoods.py:328 is in a dead function (q_learning_step) with zero call sites and has no impact on any active code path.

---

_Verified: 2026-04-03T09:38:08Z_
_Verifier: Claude (gsd-verifier)_
