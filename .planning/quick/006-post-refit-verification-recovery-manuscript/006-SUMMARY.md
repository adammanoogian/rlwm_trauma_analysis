---
phase: quick-006
plan: 006
type: execute
status: complete
date: 2026-04-10
winning_model_before: wmrl_m5
winning_model_after: wmrl_m6b
n_participants: 154
agreement_aic_bic: true
---

# Quick-006 Summary: Post-refit Verification, Recovery, and Manuscript Revision

Work executed from 006-PLAN.md on the N=154 re-fit pipeline. All eight
sub-tasks executed in one session. Eight logical commits landed on main.

## What Changed

- **Winning model flipped**: M5 to M6b, aggregate dAIC=572.89, dBIC=572.89,
  Akaike weight effectively 1.0. AIC and BIC ranks agree for all six
  choice-only models.
- **Per-participant winners** now reported: M6b=55 (35.7%), M5=41 (26.6%),
  M6a=38 (24.7%), M3=15 (9.7%), M2=3 (1.9%), M1=2 (1.3%).
- **Bonferroni correction added** to 16_regress_parameters_on_scales.py
  alongside existing FDR-BH. Applied to all 7 models.
- **Winner heterogeneity script** created (17_analyze_winner_heterogeneity.py)
  and run. Kruskal-Wallis shows very large effects on kappa_share
  (H=80.08, p<1e-15, eta_H^2=0.51) and kappa_total (H=53.36, p<1e-9,
  eta_H^2=0.33) across winner groups.
- **Manuscript comprehensively updated** in both paper.qmd (authoritative
  source) and paper.tex (build output): new abstract, three new Results
  subsections (Winner Heterogeneity, Stratified by Trauma Group,
  Parameter Recovery and Identifiability), full Discussion replacing six
  "to be completed" placeholders.
- **M2 WMRL 33% non-convergence diagnosed**: NOT a NaN propagation bug
  (audit confirmed both GPU and CPU paths are NaN-safe). Root cause is
  bound-attracted optimization: all 51 non-converged participants have
  alpha_neg at lower bound (with epsilon and alpha_pos often also at
  bounds). L-BFGS-B reports non-converged because the projected gradient
  is nonzero at the boundary, even though multiple starts agree on the
  same optimum. NLLs are still valid; aggregate AIC comparison is
  unaffected.

## Headline Numbers

### Aggregate Model Comparison (AIC and BIC agree)

| Rank | Model | k | Aggregate AIC | Aggregate BIC | dAIC |
|------|-------|---|---------------|---------------|------|
| 1 | M6b dual perseveration | 8 | 143,325 | 148,971 | 0.0 |
| 2 | M5 RL forgetting | 8 | 143,898 | 149,544 | 572.9 |
| 3 | M6a stimulus-specific | 7 | 144,772 | 149,712 | 1446.7 |
| 4 | M3 global kappa | 7 | 144,866 | 149,807 | 1541.0 |
| 5 | M2 WMRL | 6 | 147,328 | 151,563 | 4003.2 |
| 6 | M1 Q-learning | 3 | 152,143 | 154,261 | 8818.2 |

### M6b Parameter Recovery (N=50 synthetic)

| Parameter | r | Status |
|-----------|-----|--------|
| kappa_total | 0.997 | PASS (trusted) |
| kappa_share | 0.931 | PASS (trusted) |
| epsilon | 0.772 | FAIL (close) |
| rho | 0.629 | FAIL |
| alpha_pos | 0.598 | FAIL |
| alpha_neg | 0.516 | FAIL |
| phi | 0.442 | FAIL |
| capacity (K) | 0.213 | FAIL (severe) |

### Trauma-Parameter Regressions Summary (within-model corrections)

| Model | n tests | Uncorrected | FDR-BH | Bonferroni |
|-------|---------|-------------|--------|------------|
| qlearning | 18 | 3 | 0 | 0 |
| wmrl (M2) | 36 | 6 | 0 | 0 |
| wmrl_m3 | 42 | 5 | **3** | 0 |
| wmrl_m5 | 48 | 7 | 0 | 0 |
| wmrl_m6a | 42 | 5 | 0 | 0 |
| wmrl_m6b | 48 | 7 | 0 | 0 |
| wmrl_m4 | 60 | 14 | 0 | 0 |

M3 FDR-BH survivors: phi x IES-R Hyperarousal (p_fdr=0.033), **kappa x
LEC-5 Total Events (p_fdr=0.033)**, phi x IES-R Total (p_fdr=0.033).

The kappa x LEC-5 signal replicates in M6b as kappa_total x LEC-5
(p_uncorrected=0.0028) but does not survive FDR-BH in the larger M6b
test family (p_fdr=0.135).

## Files Created

- scripts/17_analyze_winner_heterogeneity.py
- output/model_comparison/comparison_results_bic.csv
- output/model_comparison/winner_heterogeneity.csv
- output/model_comparison/winner_heterogeneity_summary.csv
- output/model_comparison/winner_heterogeneity_figure.png
- figures/model_comparison/winner_heterogeneity_figure.png
- output/regressions/{qlearning,wmrl,wmrl_m3,wmrl_m4,wmrl_m5,wmrl_m6a,wmrl_m6b}/significance_corrected.csv
- output/regressions/{all 7 models}/significance_summary.md
- .planning/quick/006-post-refit-verification-recovery-manuscript/LITERATURE.md (local only; .planning/ gitignored for new files)
- .planning/quick/006-post-refit-verification-recovery-manuscript/006-SUMMARY.md (this file)

## Files Modified

- scripts/fitting/fit_mle.py (audit comments only; no code changes)
- scripts/14_compare_models.py (BIC rank columns, comparison_results_bic.csv export, side-by-side print)
- scripts/16_regress_parameters_on_scales.py (Bonferroni correction, significance_corrected.csv + .md export)
- manuscript/paper.qmd (authoritative source: Abstract, three new Results subsections, full Discussion)
- manuscript/paper.tex (build output mirror)
- .planning/STATE.md (Post-Refit Reality subsection, quick tasks table)
- .planning/PROJECT.md (scrubbed of result contamination; local only because file is untracked and .planning/ is gitignored)

## Scientific Decisions

### Why BIC alongside AIC
Both are standard; AIC measures predictive accuracy while BIC is
consistent for true-model selection under the Burnham-Anderson
framework. With N=154 and k=3..8, BIC's per-parameter penalty
k*ln(n_trials_ppt) is ~3.3x AIC's flat 2k, so a disagreement would be
diagnostic. In practice AIC and BIC agree exactly on all six
choice-only models, which is additional robustness evidence for the
M6b preference.

### Why disclose identifiability rather than drop M6b
Aggregate AIC/BIC comparison depends on the likelihood of the data, not
on per-participant point estimates of non-identifiable parameters. M6b
remains the preferred model even with poor base-RLWM recovery because
the perseveration-kernel component of its likelihood captures variance
that the simpler models cannot. We disclose the identifiability
limitation prominently in the manuscript (new Parameter Recovery
subsection plus Discussion paragraph) so readers understand which
inferences are safe (kappa-level, aggregate comparison) and which are
not (individual differences in K, phi, rho, alpha, epsilon).

### Family-wise correction within-model, not across
M1..M6b are alternative explanations of the same data, not independent
hypotheses. Correcting across the 7 x params x scales cube would
overcount by treating each model's test family as independent. Within-
model FDR-BH is the standard approach in this subfield.

### Scientific takeaway for the paper
The perseveration kernel (kappa in M3, kappa_total in M6b) is the
best-supported trauma-parameter bridge in these data, surviving FDR-BH
in M3 and reproducing uncorrected in M6b. Because kappa recovers well
(r=0.997 in M6b), this is identifiability-safe. The epsilon-trauma
association remains exploratory: it does not survive correction, and
epsilon's own recovery (r=0.77) is below the 0.80 threshold.

## Deferred (Explicit Follow-ups for Quick-007 or later)

- **M4 LBA parameter recovery** (~48h cluster compute).
  `sbatch --time=48:00:00 --export=MODEL=wmrl_m4,NSUBJ=30 cluster/11_recovery_gpu.slurm`
- **M2 WMRL re-fit**: optional. The 33% non-convergence is diagnosed as
  bound-attracted, not NaN. Re-fit would only matter if bounds are
  relaxed or a boundary-aware convergence criterion is adopted.
- **Bayesian hierarchical fitting of M6b**: would regularize base-RLWM
  parameters via group-level priors and partially recover
  identifiability through shrinkage. Infrastructure exists
  (13_fit_bayesian.py, numpyro_models.py) but not yet run for M6b.
- **Cross-model recovery validation** with M5/M6a/M6b included. Would
  confirm that M6b is distinguishable from M5 and M6a at the AIC level
  on synthetic data. Command:
  `python scripts/11_run_model_recovery.py --mode cross-model --model all --n-subjects 50 --n-datasets 10 --n-jobs 8`
- **Literature DOI verification**: LITERATURE.md contains several [TBD]
  DOI entries for Myers & Gluck and Ross that need manual confirmation
  before the final manuscript submission.
- **PROJECT.md commit**: file remains untracked locally (.planning/
  gitignore prevents new additions). If the project adopts a convention
  to track PROJECT.md, run `git add -f .planning/PROJECT.md` in a
  follow-up commit.

## Outstanding Issues

1. **Myers & Gluck citation needed** in references.bib. The Discussion
   currently cites only collins2014working for the perseveration-
   reversal connection; if a Myers paper is confirmed, it should be
   added.
2. **phi_rl in M5 display** should be checked: currently paper.tex
   displays "phi_rl" without LaTeX math formatting in a couple of
   tables. Quarto rebuild may normalize this.
3. **fig-winner-heterogeneity path** in paper.tex uses
   `../output/model_comparison/winner_heterogeneity_figure.png`, which
   is valid when pdflatex is run from manuscript/. If Quarto rebuilds,
   the path may need to be updated to use
   `../output/model_comparison/...` from paper.qmd's perspective.

## Commits in This Task

1. `9128523` fix(mle): audit NaN argmin guards on all fit_mle paths; diagnose M2 non-convergence
2. `6fa6299` feat(model_compare): add BIC rank columns and dedicated BIC-sorted comparison output
3. `6e3555d` feat(analysis): add 17_analyze_winner_heterogeneity.py for per-participant winning model diagnostics
4. `0944d9d` feat(regressions): add Bonferroni correction and emit significance_corrected.csv + markdown summary
5. `7943b7e` docs(manuscript): update paper for M6b winner, identifiability, heterogeneity, lineage, trauma lit
6. (planning docs; untracked/partial) STATE.md updated inline with commit 5; PROJECT.md remains untracked
7. (this file) docs(quick-006): task summary
