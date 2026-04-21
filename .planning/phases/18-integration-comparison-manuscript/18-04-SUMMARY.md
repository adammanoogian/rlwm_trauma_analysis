---
phase: 18
plan: 04
subsystem: documentation
tags: [model-reference, hierarchical-bayesian, non-centered, level2-regression, waic-loo, schema-parity]

dependency-graph:
  requires:
    - "13-04: numpyro_helpers.py sample_bounded_param and PARAM_PRIOR_DEFAULTS established"
    - "15-01: M3 hierarchical POC, non-centered parameterization confirmed working"
    - "16-01: 4-predictor design matrix locked, N=160 confirmed, condition number 11.3"
    - "16-03 to 16-06: All 6 choice-only hierarchical models implemented"
    - "17-01 to 17-03: M4 hierarchical LBA model and integration tests"
    - "13-05: bayesian_diagnostics.py compute_pointwise_log_lik, filter_padding_from_loglik"
  provides:
    - "Single-source reference doc for v4.0 hierarchical Bayesian infrastructure"
    - "DOC-01 requirement: MODEL_REFERENCE.md Hierarchical Bayesian Pipeline section"
    - "Corrected winning model note (M5 -> M6b)"
  affects:
    - "18-05: Manuscript revision can reference MODEL_REFERENCE.md section 11 directly"
    - "Future contributors: onboarding reference for hierarchical pipeline"

tech-stack:
  added: []
  patterns:
    - "hBayesDM non-centered parameterization (Ahn et al., 2017)"
    - "numpyro.factor pattern for likelihood attachment"
    - "WAIC/LOO with Pareto-k gating for M4 separate track"
    - "Schema-parity CSV for --source mle|bayesian flag"

key-files:
  created: []
  modified:
    - docs/03_methods_reference/MODEL_REFERENCE.md

decisions:
  - "Winning model note corrected from M5 (dAIC=435.6 over M3) to M6b (unit Akaike weight) per quick-006 results"
  - "Section 11 added with 5 subsections; See Also renumbered to section 12"
  - "K_PARAMETERIZATION.md cross-referenced in 3 locations: winning model note, PARAM_PRIOR_DEFAULTS table, and See Also"

metrics:
  duration: "~4 minutes"
  completed: "2026-04-13"
---

# Phase 18 Plan 04: MODEL_REFERENCE Documentation Summary

**One-liner:** Added Hierarchical Bayesian Pipeline section (5 subsections) to MODEL_REFERENCE.md documenting v4.0 non-centered parameterization, Level-2 regression, numpyro.factor pattern, WAIC/LOO workflow, and schema-parity CSV; corrected winning model from M5 to M6b.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Fix outdated winning model note | eb3ccc3 | docs/03_methods_reference/MODEL_REFERENCE.md |
| 2 | Add Hierarchical Bayesian Pipeline section (DOC-01) | e9bec83 | docs/03_methods_reference/MODEL_REFERENCE.md |

## What Was Built

### Task 1: Winning Model Note Correction

Replaced the stale M5 winning model note (which referenced the pre-refit dAIC=435.6 result) with the accurate M6b note reflecting the quick-006 post-refit results:

- M6b has lowest aggregate AIC and BIC across N=154 participants
- Effectively unit Akaike weight
- Cross-reference to `docs/K_PARAMETERIZATION.md` for K bounds

### Task 2: Hierarchical Bayesian Pipeline Section

Added section 11 with 5 subsections to `docs/03_methods_reference/MODEL_REFERENCE.md`, providing the DOC-01 single-source reference for the entire v4.0 hierarchical infrastructure:

**11.1 Non-Centered Parameterization:** Documents the hBayesDM convention, `sample_bounded_param()` implementation, `Phi_approx = jax.scipy.stats.norm.cdf`, and the full `PARAM_PRIOR_DEFAULTS` table with 10 parameters, their `mu_prior_loc`, bounds, and rationale. Cross-references `docs/K_PARAMETERIZATION.md` for K bounds.

**11.2 Level-2 Regression Structure:** Documents the locked 4-predictor design matrix `[lec_total, iesr_total, iesr_intr_resid, iesr_avd_resid]`, Gram-Schmidt orthogonalization (hyperarousal excluded due to rank deficiency), condition number 11.3, the Level-2 shift equation on the probit scale, and N=160 complete-data participants.

**11.3 NumPyro Factor Pattern and Pointwise Log-Likelihood:** Documents `numpyro.factor(f"obs_{participant_id}", -nll_i)` pattern, `compute_pointwise_log_lik()` producing shape `(chains, samples, participants, n_blocks * max_trials)`, and `filter_padding_from_loglik()` pre-processing.

**11.4 WAIC/LOO Workflow:** Documents primary (LOO-CV with PSIS) and secondary (WAIC) metrics, `az.compare` stacking over 6 choice-only models, M4 Pareto-k gating with the 5% threshold, and the convergence gate `max_rhat < 1.01 AND min_ess_bulk > 400 AND num_divergences == 0`.

**11.5 Schema-Parity CSV:** Documents the `output/bayesian/{model}_individual_fits.csv` column schema including `{param}_hdi_low/high/sd`, convergence columns, and how this enables `--source mle|bayesian` flag in scripts 15/16/17 with zero analysis-logic changes.

Also renumbered the existing `## See Also` to `## 12. See Also` and added a `docs/K_PARAMETERIZATION.md` entry.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Winning model corrected to M6b | quick-006 refit results: M6b aggregate AIC 143324.93 vs M5 143897.82; unit Akaike weight |
| Section numbering: 11 for Hierarchical Pipeline, 12 for See Also | Follows existing numbered section pattern (1-10) |
| K_PARAMETERIZATION.md cross-referenced in 3 locations | Ensures traceability from winning model note, parameter table, and reference section |

## Verification Results

All plan verification checks passed:

1. `grep "Hierarchical Bayesian Pipeline" MODEL_REFERENCE.md` — section 11 confirmed at line 1318
2. `grep "M6b" MODEL_REFERENCE.md | head -5` — winning model note updated at line 23
3. `grep "K_PARAMETERIZATION" MODEL_REFERENCE.md` — 4 cross-references confirmed (lines 23, 1348, 1357, 1451)
4. `grep "sample_bounded_param" MODEL_REFERENCE.md` — confirmed at line 1338
5. `grep "Schema-Parity" MODEL_REFERENCE.md` — confirmed at line 1428

## Deviations from Plan

None - plan executed exactly as written.

## Next Phase Readiness

Phase 18 Plan 05 (manuscript revision) can proceed. MODEL_REFERENCE.md section 11 provides the authoritative reference for all hierarchical methodology claims in the manuscript.

No blockers or concerns for Phase 18-05.
