# Stage folder name starts with a digit so standard dotted-path imports do
# NOT work (`scripts.06_fit_analyses.*` is illegal — dotted names cannot
# start with a digit). None of the files in this package are imported as
# Python modules by other code; they are all CLIs invoked via
# `python scripts/06_fit_analyses/0N_<name>.py ...`.
#
# File ordering (Scheme D, plan 29-04b) follows paper-read order:
#   01_compare_models.py              — MLE AIC/BIC + optional WAIC/LOO
#   02_compute_loo_stacking.py        — Bayesian LOO-PSIS + stacking + RFX-BMS
#   03_model_averaging.py             — posterior model averaging across winners
#   04_analyze_mle_by_trauma.py       — per-model parameter-trauma scatter
#   05_regress_parameters_on_scales.py — FDR/Bonferroni-corrected regressions
#   06_analyze_winner_heterogeneity.py — per-participant winning-model table
#   07_bayesian_level2_effects.py     — forest-plot rendering backend
#   08_manuscript_tables.py           — final paper tables + Figure 1 forest
"""Fit-result analyses (06_fit_analyses): paper-read order 01-08."""
