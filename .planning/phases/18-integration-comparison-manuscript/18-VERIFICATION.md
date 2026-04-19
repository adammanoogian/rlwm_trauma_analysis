---
phase: 18-integration-comparison-manuscript
verified: 2026-04-13T21:00:00Z
re_verified: 2026-04-19
status: passed
score: 14/14 must-haves verified
gaps: []
resolution_history:
  - date: 2026-04-19
    action: "Re-verified during v4.0 milestone audit; truth #14 (vmap + numpyro.factor in paper.qmd) confirmed satisfied at HEAD"
    evidence: "grep -n 'numpyro.factor\\|jax.vmap' manuscript/paper.qmd returns lines 512 and 513"
    prior_status: gaps_found
---

# Phase 18: Integration Comparison Manuscript Verification Report

Phase Goal: Wire the Bayesian fits from Phases 15-17 into the existing downstream pipeline via the schema-parity --source mle|bayesian flag, freeze 16b as deprecated, produce MLE-vs-Bayesian reliability scatterplots, and rewrite the manuscript sections around the joint hierarchical narrative.
Verified: 2026-04-13T21:00:00Z
Re-verified: 2026-04-19 (v4.0 milestone audit — truth #14 satisfied at HEAD)
Status: passed
Re-verification: Yes — 2026-04-19 re-verification flipped status from gaps_found to passed

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Script 15 has --source flag routing to output/bayesian/ | VERIFIED | add_argument(--source) at line 838; fits_dir routing at 852-859; load_data(fits_dir) at 866 |
| 2 | Script 16 has --source flag routing to output/regressions/bayesian/ | VERIFIED | --source at line 719; route logic at 733-737; base_fits_dir bayesian at 775 |
| 3 | Script 17 has --source flag routing to output/bayesian/model_comparison/ | VERIFIED | --source at line 285; fits_dir routing at 290-295; load_per_participant_aic(fits_dir) at 308 |
| 4 | Script 16b has deprecation docstring and no PyMC imports | VERIFIED | deprecated:: at line 6; numpyro/jax-only imports at 79-82 |
| 5 | Script 18b generates scatter per param/model with 45-degree line and M6b shrinkage | VERIFIED | plot_mle_vs_bayes_reliability() at 53; 45-deg line at 128; highlight_shrinkage for wmrl_m6b at 278 |
| 6 | Script 14 --bayesian-comparison produces stacking weights CSV for 6 choice-only models | VERIFIED | BAYESIAN_NETCDF_MAP covers M1/M2/M3/M5/M6a/M6b at 96-103; stacking_weights.csv at 746 |
| 7 | M4 in separate section with Pareto-k gating NOT in az.compare dict | VERIFIED | M4 Separate Track block at 792; m4_comparison.csv at 864; M4 never in compare_dict |
| 8 | WAIC computed as secondary metric alongside LOO | VERIFIED | az.waic() loop at 755; waic_summary.csv at 775; WAIC appended to Markdown |
| 9 | MODEL_REFERENCE.md has Hierarchical Bayesian Pipeline section with 5 subsections | VERIFIED | Section 11 at 1318; subsections 11.1-11.5; K_PARAMETERIZATION.md cross-ref 4x; M6b winning at 23 |
| 10 | numpyro.factor and compute_pointwise_log_lik documented in MODEL_REFERENCE.md | VERIFIED | numpyro.factor at 1393; compute_pointwise_log_lik() at 1398 |
| 11 | Manuscript methods NumPyro hierarchical Bayesian zero PyMC Level-2 regression | VERIFIED | NumPyro at 473; grep PyMC count = 0; Level-2 at 481-510; LOO-CV at 486 |
| 12 | Manuscript results has stacking-weight table and Level-2 forest plot section | VERIFIED | Stacking weights table at 615; Hierarchical Level-2 section at 933 |
| 13 | Manuscript Limitations covers M4 Pareto-k K identifiability M6b shrinkage | VERIFIED | sec-limitations at 1117; M4 Pareto-k at 1133; K bounded range at 1139; shrinkage at 1122-1126 |
| 14 | Manuscript methods mentions vmap and numpyro.factor by name | VERIFIED (2026-04-19 re-verification) | `grep -n "numpyro.factor\|jax.vmap" manuscript/paper.qmd` returns `numpyro.factor` at line 512 and `jax.vmap` at line 513; previously FAILED at 2026-04-13 initial verification, resolved in subsequent paper.qmd edit |

Score: 14/14 truths verified (re-verified 2026-04-19)
### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| scripts/15_analyze_mle_by_trauma.py | --source flag + path routing | VERIFIED | 1041 lines; flag + bayesian routing + load_data() refactor |
| scripts/16_regress_parameters_on_scales.py | --source flag + path routing | VERIFIED | 1140 lines; flag + output/regressions/bayesian routing |
| scripts/17_analyze_winner_heterogeneity.py | --source flag + path routing | VERIFIED | 361 lines; flag + fits_dir rename throughout |
| scripts/14_compare_models.py | --bayesian-comparison with CSV + M4 + WAIC | VERIFIED | 1251 lines; all CMP artifacts implemented |
| scripts/16b_bayesian_regression.py | Deprecation docstring NumPyro-only runnable | VERIFIED | Deprecation at line 6; no PyMC imports |
| scripts/18b_mle_vs_bayes_reliability.py | New scatter plot script | VERIFIED | 319 lines; created 2026-04-13; core function and main() wired |
| docs/03_methods_reference/MODEL_REFERENCE.md | Section 11 Hierarchical Bayesian Pipeline | VERIFIED | Section 11 at 1318 with all 5 subsections |
| manuscript/paper.qmd | Revised methods/results/limitations | VERIFIED (2026-04-19 re-verification) | Methods has NumPyro/Level-2/LOO at 473; vmap at 513; numpyro.factor at 512 — all ROADMAP criterion 7 terms present |

### Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| scripts/15 | output/bayesian/*_individual_fits.csv | fits_dir passed to load_data() | WIRED |
| scripts/16 | output/bayesian/*_individual_fits.csv | base_fits_dir = Path(output/bayesian) at 775 | WIRED |
| scripts/17 | output/bayesian/*_individual_fits.csv | fits_dir at 291-295; passed to load_per_participant_aic | WIRED |
| scripts/14 | output/bayesian/level2/stacking_weights.csv | comparison.to_csv(csv_path) at 747 | WIRED |
| scripts/14 | output/bayesian/wmrl_m4_posterior.nc | M4 track loads NetCDF at 795-798 | WIRED |
| scripts/18b | output/mle/*_individual_fits.csv | MLE CSV for x-axis | WIRED |
| scripts/18b | output/bayesian/*_individual_fits.csv | Bayesian CSV for y-axis | WIRED |
| manuscript/paper.qmd | output/bayesian/level2/stacking_weights.csv | Results code cell at 615 with graceful fallback | WIRED |

### Requirements Coverage

| Requirement | Status | Notes |
|-------------|--------|-------|
| CMP-01: az.compare stacking weights over 6 choice-only models | SATISFIED | All 6 in BAYESIAN_NETCDF_MAP; az.compare at 674 |
| CMP-02: M4 separate track NOT in az.compare dict | SATISFIED | M4 track at 792-870 |
| CMP-03: WAIC + LOO reported | SATISFIED | az.waic loop at 753; waic_summary.csv at 775 |
| CMP-04: --bayesian-comparison mode; 4 output files; MLE default unchanged | SATISFIED | Flag at 953; stacking_weights.md/csv + waic_summary.csv + m4_comparison.csv |
| MIG-01: Script 15 --source flag | SATISFIED | Flag at 838; full path routing and load_data() refactor |
| MIG-02: Script 16 --source flag | SATISFIED | Flag at 719; output/regressions/bayesian routing |
| MIG-03: Script 17 --source flag | SATISFIED | Flag at 285; fits_dir routing in main() |
| MIG-04: 16b deprecation docstring runnable no PyMC | SATISFIED | Deprecation at 6; NumPyro-only |
| MIG-05: MLE-vs-Bayesian reliability scatterplots | SATISFIED | 18b script with shrinkage arrows for M6b |
| DOC-01: MODEL_REFERENCE.md Hierarchical Bayesian Pipeline | SATISFIED | Section 11 complete |
| DOC-02: Manuscript methods NumPyro vmap numpyro.factor Level-2 WAIC/LOO | SATISFIED (2026-04-19 re-verification) | All 5 terms present in paper.qmd Methods (lines 473-513) |
| DOC-03: Manuscript results stacking-weight table Level-2 forest plots | SATISFIED | Both present with graceful fallback |
| DOC-04: Manuscript limitations M4 Pareto-k K identifiability M6b shrinkage | SATISFIED | All documented at 1117-1151 |

### Anti-Patterns Found

No blocker anti-patterns. Two graceful-fallback try/except cells in paper.qmd (lines 615 and 963) are intentional design for missing cluster outputs.

### Runtime-Dependent Outputs (Not Implementation Gaps)

The following do not yet exist because cluster MCMC runs have not completed. These are data prerequisites, not code gaps:

- output/bayesian/{model}_individual_fits.csv: requires scripts/13_fit_bayesian.py on cluster
- output/bayesian/level2/stacking_weights.csv: requires scripts/14_compare_models.py --bayesian-comparison
- output/bayesian/figures/mle_vs_bayes/*.png: requires scripts/18b_mle_vs_bayes_reliability.py after fits

### Gaps Summary

No gaps. Re-verified 2026-04-19 during v4.0 milestone audit: the single gap from the 2026-04-13 initial verification (ROADMAP criterion 7 — vmap + numpyro.factor mentions in paper.qmd) has been satisfied at HEAD. `grep -n "numpyro.factor\|jax.vmap" manuscript/paper.qmd` returns `numpyro.factor` at line 512 and `jax.vmap` at line 513, inside the Model Fitting paragraph. All 14/14 Observable Truths now pass.

---
_Verified: 2026-04-13T21:00:00Z_
_Verifier: Claude (gsd-verifier)_
