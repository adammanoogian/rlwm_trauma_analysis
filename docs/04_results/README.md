# docs/04_results/ — Pipeline Results Index

Every top-level result category produced by the pipeline, including
supplementary and orphaned artifacts not shown in manuscript/paper.qmd.
One row per result with producer + canonical output path + provenance.
Placeholders are used where artifacts do not yet exist (e.g. Bayesian
posteriors blocked on cluster runs).

## Behavioral

| Result | Producer | Output path | Status |
|---|---|---|---|
| Task performance plots | scripts/02_behav_analyses/02_visualize_task_performance.py | figures/behavioral_analysis/ | available |
| Trauma-group behavioral stats | scripts/02_behav_analyses/03_analyze_trauma_groups.py | figures/trauma_groups/, output/summary_by_trauma.csv | available |
| Statistical analyses (ANOVA, descriptives) | scripts/02_behav_analyses/04_run_statistical_analyses.py | output/statistical_analyses/ | available |
| Scale distributions | scripts/legacy/analysis/trauma_scale_distributions.py | figures/scale_distributions.png, figures/trauma_scale_analysis/ | available after pipeline rerun |

## Model fitting (MLE)

| Result | Producer | Output path | Status |
|---|---|---|---|
| Individual MLE fits (all 7 models) | scripts/04_model_fitting/a_mle/fit_mle.py | output/mle/{model}_individual_fits.csv | available |
| Model comparison (AIC/BIC) | scripts/06_fit_analyses/compare_models.py | output/model_comparison/ | available |
| Winner heterogeneity | scripts/06_fit_analyses/analyze_winner_heterogeneity.py | output/model_comparison/winner_heterogeneity*.csv, figures/model_comparison/winner_heterogeneity_figure.png | available |

## Trauma associations (MLE)

| Result | Producer | Output path | Status |
|---|---|---|---|
| Parameter-trauma correlations | scripts/06_fit_analyses/analyze_mle_by_trauma.py | output/regressions/{model}/ | available |
| FDR/Bonferroni-corrected regressions | scripts/06_fit_analyses/regress_parameters_on_scales.py | output/regressions/{model}/significance_*.{csv,md} | available |

## Bayesian (blocked on cluster)

| Result | Producer | Output path | Status |
|---|---|---|---|
| Hierarchical posteriors (6 choice-only models) | scripts/04_model_fitting/b_bayesian/fit_bayesian.py | output/bayesian/{model}_posterior.nc | _placeholder — cluster refit pending_ |
| M4 LBA posterior | scripts/04_model_fitting/b_bayesian/fit_bayesian.py --model wmrl_m4 | output/bayesian/wmrl_m4_posterior.nc | _placeholder — cluster refit pending_ |
| Pscan benchmarks | cluster/13_bayesian_pscan.slurm | output/bayesian/pscan_benchmark.json | available |
| M6b posterior diagnostics | scripts/legacy/visualization/plot_posterior_diagnostics.py | figures/m6b_posterior_diagnostics.png | _placeholder — needs posterior first_ |
| M6b posterior vs MLE | validation/compare_posterior_to_mle.py | figures/m6b_posterior_vs_mle.png | _placeholder — needs posterior first_ |
| Level-2 stacking weights | scripts/06_fit_analyses/compare_models.py --bayesian-comparison | output/bayesian/level2/stacking_weights.csv | _placeholder — needs posterior first_ |
| Level-2 forest plots | scripts/06_fit_analyses/bayesian_level2_effects.py | output/bayesian/figures/m6b_forest_lec5.png | _placeholder — needs posterior first_ |

Entries marked _placeholder_ will be filled in after the next cluster
Bayesian fit completes (see .planning/STATE.md for current blocker).
