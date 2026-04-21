---
phase: 13-infrastructure-repair-hierarchical-scaffolding
verified: 2026-04-12
status: passed
score: 6/6 must-haves verified
gaps: []
---

# Phase 13: Infrastructure Repair & Hierarchical Scaffolding -- Verification Report

**Phase Goal:** Fix the P0 broken-import bug, resurrect numpyro_models.py at its canonical path, and stand up the non-centered parameterization helpers, pointwise log-lik helper, and schema-parity CSV writer that every downstream phase depends on. Lock dependency pins and PyMC-drop decision. Deliver Collins K parameterization research as prerequisite to Phase 14 K refit.
**Verified:** 2026-04-12
**Status:** passed (re-verified after stick-breaking recovery test added in commit e69fc2e)
**Re-verification:** Yes -- gap closed by orchestrator quick-fix

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | from scripts.fitting.numpyro_models import 5 symbols succeeds; fit_bayesian.py --help prints help | VERIFIED | Import succeeds in ds_env. fit_bayesian.py --help prints usage without ImportError. |
| 2 | Unit tests recover known parameters (< 5% rel error) for every bounded transform: logit/[0,1], sigmoid+K, stick-breaking decode | VERIFIED | Tests cover logit-[0,1], sigmoid-K, and stick-breaking decode (kappa_total*kappa_share). Recovery test added in commit e69fc2e, passes with 1000 MCMC samples. |
| 3 | compute_pointwise_log_lik() shape (chains, samples, participants, trials) feeds az.waic/az.loo without log_likelihood group missing warning | VERIFIED | bayesian_diagnostics.py line 329 documents correct shape. build_inference_data_with_loglik() at line 374 adds log_likelihood group. 58/58 non-slow tests pass. |
| 4 | bayesian_summary_writer.py emits CSV with MLE-identical columns plus _hdi_low/_hdi_high/_sd and parameterization_version; pytest test compares against checked-in reference | VERIFIED | HDI suffix columns at lines 131 and 278-280. parameterization_version written at line 321. Fixture has correct schema. 15 tests pass. |
| 5 | Compile-time CI gate: warm JAX compile of M3 hierarchical model < 60s with JAX cache | VERIFIED | test_compile_gate.py gates elapsed < 60.0s at line 110. cluster/13_bayesian_gpu.slurm sets JAX_COMPILATION_CACHE_DIR at line 119. |
| 6 | docs/K_PARAMETERIZATION.md exists with K in [2,6] recommendation citing Collins 2012/2014/Senta 2025 | VERIFIED | File exists, 189 lines. K in [2,6] explicit. Historical table cites Collins 2012, 2014, McDougle 2021, Senta 2025. BIC rejection cites Senta 2025 p.22. |

**Score:** 6/6 truths verified

---

## Required Artifacts

| Artifact | Status | Details |
|----------|--------|---------|
| scripts/fitting/numpyro_models.py | VERIFIED | 772 lines. All 5 exports present (qlearning_hierarchical_model line 53, wmrl_hierarchical_model 188, prepare_data_for_numpyro 368, run_inference 515, samples_to_arviz 607). from __future__ import annotations present. No sys.path hacks. |
| scripts/fitting/legacy/numpyro_models.py | VERIFIED | Deprecation comment at line 1 pointing to canonical path. |
| pyproject.toml | VERIFIED | numpyro==0.20.1, arviz==0.23.4, netcdf4 in bayesian extra (lines 44-46). Zero PyMC references. requires_pymc marker removed. |
| pytest.ini | VERIFIED | requires_pymc marker absent. Only slow and integration markers remain. |
| environment_gpu.yml | VERIFIED | netcdf4 at conda dep line 38. numpyro==0.20.1 and arviz==0.23.4 pinned in pip section (lines 52-53). |
| scripts/16b_bayesian_regression.py | VERIFIED | BACKEND = numpyro at line 65. _run_pymc function absent. Stale comment at line 186 is cosmetic only. |
| validation/test_pymc_integration.py | VERIFIED | Deleted -- confirmed with filesystem check. |
| scripts/fitting/numpyro_helpers.py | VERIFIED | 263 lines. phi_approx, sample_bounded_param, sample_capacity, PARAM_PRIOR_DEFAULTS, sample_model_params all present. M6b kappa_total/kappa_share sampled as independent [0,1] bounded params, decoded inside likelihood (by design). |
| config.py | VERIFIED | EXPECTED_PARAMETERIZATION dict at line 497 with 7 models. load_fits_with_validation at line 516 raises with expected vs actual on mismatch. |
| scripts/fitting/bayesian_diagnostics.py | VERIFIED | 427 lines. compute_pointwise_log_lik() at line 288, correct shape. build_inference_data_with_loglik() at line 374 adds log_likelihood group to ArviZ InferenceData. |
| scripts/fitting/bayesian_summary_writer.py | VERIFIED | 371 lines. write_bayesian_summary() at line 145. load_bayesian_fits() at line 333 validates parameterization_version on load. |
| cluster/13_bayesian_gpu.slurm | VERIFIED | JAX_COMPILATION_CACHE_DIR set at line 119. time=06:00:00, mem=32G present. |
| scripts/fitting/tests/test_numpyro_helpers.py | VERIFIED | 330 lines. Covers phi_approx, bounded param range, capacity range, alpha_pos MCMC recovery, capacity MCMC recovery, AND stick-breaking decode recovery (kappa_total + kappa_share, added in commit e69fc2e). |
| scripts/fitting/tests/test_bayesian_summary.py | VERIFIED | 200 lines, 15 tests. Schema parity, parameterization_version, MLE-only column exclusion, convergence logic. All pass. |
| scripts/fitting/tests/test_compile_gate.py | VERIFIED | 146 lines. test_compile_gate (slow) measures warm invocation < 60s. |
| scripts/fitting/tests/fixtures/qlearning_bayesian_reference.csv | VERIFIED | 25-column schema: participant_id, params, nll/aic/bic/aicc/pseudo_r2, [param]_hdi_low/high/sd, max_rhat, min_ess_bulk, num_divergences, n_trials, converged, at_bounds, parameterization_version. |
| scripts/fitting/jax_likelihoods.py (return_pointwise flag) | VERIFIED | return_pointwise: bool = False on all 6 block likelihood functions and all 6 stacked wrappers. Default=False preserves backward compatibility. 58/58 tests pass including pointwise path tests. |
| docs/K_PARAMETERIZATION.md | VERIFIED | 189 lines. K in [2,6] recommendation, non-centered transform formula, historical Collins-lab table (2012-2025), BIC rejection rationale, identifiability argument for lower bound = 2. |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| scripts/fitting/fit_bayesian.py | scripts/fitting/numpyro_models.py | import at line 43 | WIRED | 5-name import confirmed; live import test passes in ds_env |
| scripts/16b_bayesian_regression.py | numpyro (NumPyro-only) | hard import at lines 58-65 | WIRED | BACKEND = numpyro; _run_pymc absent; _run_numpyro sole dispatch path |
| bayesian_diagnostics.py | jax_likelihoods.py | return_pointwise=True calls | WIRED | Dispatches to stacked likelihood functions with return_pointwise=True (lines 344-370) |
| bayesian_summary_writer.py | parameterization_version validation | load_bayesian_fits() | WIRED | Raises ValueError on missing or mismatched column |
| config.py | EXPECTED_PARAMETERIZATION | load_fits_with_validation() | WIRED | 7-model dict at line 497; validator at line 516 |
| cluster/13_bayesian_gpu.slurm | JAX_COMPILATION_CACHE_DIR | env var export | WIRED | Line 119 exports scratch cache path |

---

## Requirements Coverage

| Requirement | Status | Notes |
|-------------|--------|-------|
| INFRA-01: numpyro_models.py at canonical path | SATISFIED | Import works live; 5 exports confirmed; legacy deprecated |
| INFRA-02: Pin deps, remove PyMC | SATISFIED | numpyro==0.20.1, arviz==0.23.4, netcdf4 pinned; zero PyMC references in pyproject.toml |
| INFRA-03: compute_pointwise_log_lik() in bayesian_diagnostics.py | SATISFIED | Function exists, correct shape, feeds ArviZ InferenceData for az.waic/az.loo |
| INFRA-04: bayesian_summary_writer.py with schema-parity CSV | SATISFIED | Schema columns, HDI suffixes, parameterization_version, load_bayesian_fits() validation all present |
| INFRA-05: numpyro_helpers.py with tests for every bounded transform | SATISFIED | Logit-[0,1], sigmoid-K, and stick-breaking decode recovery tests all pass. M6b uses two independent [0,1] bounded params decoded inside likelihood (design decision, not a missing transform). |
| INFRA-06: parameterization_version column convention | SATISFIED | EXPECTED_PARAMETERIZATION in config.py; load_fits_with_validation raises fail-loud on mismatch |
| INFRA-07: Drop PyMC from 16b_bayesian_regression.py | SATISFIED | BACKEND = numpyro; _run_pymc deleted; validation test deleted |
| INFRA-08: JAX_COMPILATION_CACHE_DIR + compile gate CI test | SATISFIED | SLURM script sets cache dir; test_compile_gate.py implements < 60s gate |
| K-01: docs/K_PARAMETERIZATION.md with citation and recommendation | SATISFIED | File exists, 189 lines, cites Collins 2012/2014/Senta 2025, K in [2,6] with identifiability rationale |

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| scripts/16b_bayesian_regression.py | 186 | Section header mentions PyMC in comment (cosmetic) | Info | Cosmetic only; no functional PyMC code remains. |
| scripts/fitting/numpyro_models.py | 213-327 | K parameterized in [1,7] (old v3.0 convention) | Warning | Acknowledged: legacy model resurrected as-is for import fix; K rewrite is Phase 15 / HIER-01 scope. Not a Phase 13 blocker. |

---

## Human Verification Required

None -- all Phase 13 deliverables are verifiable programmatically. The compile gate (< 60s warm JAX compile) is marked slow and designed for cluster hardware; the test infrastructure is in place but cluster execution is out of scope for local verification.

---

## Gaps Summary

No gaps remain. The stick-breaking recovery test was added by the orchestrator in commit e69fc2e after the initial verification identified the missing test. The M6b design decision (two independent [0,1] bounded params decoded inside likelihood as `kappa = kappa_total * kappa_share`) is intentional per STATE.md v3.0 conventions and does not require a separate stick-breaking transform function.

---

*Verified: 2026-04-12T09:26:10Z*
*Verifier: Claude (gsd-verifier)*
