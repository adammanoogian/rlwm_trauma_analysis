---
phase: quick-005
plan: 01
subsystem: pipeline-analysis-manuscript
tags: [model-comparison, trauma-analysis, regressions, quarto, manuscript-figures]
dependency-graph:
  requires: [v3.0 MLE fits for all 7 models]
  provides: [Updated pipeline outputs (N=154), manuscript model overview + distribution figures]
  affects: [manuscript rendering, future manuscript sections]
tech-stack:
  added: [tcolorbox, fontawesome5]
  patterns: [MODEL_REGISTRY-driven figure generation]
file-tracking:
  key-files:
    created:
      - manuscript/outputs/fig_model_overview.pdf
      - manuscript/outputs/fig_param_distributions.pdf
    modified:
      - manuscript/paper.qmd
      - manuscript/arxiv_template.tex
      - output/model_comparison/comparison_results.csv
      - output/model_comparison/participant_wins.csv
      - output/mle/spearman_correlations.csv
      - output/mle/group_comparison_stats.csv
      - output/regressions/*/regression_results_simple.csv
      - output/regressions/*/regression_results_multiple.csv
      - figures/model_comparison/*.png
      - figures/mle_trauma_analysis/*.png
      - figures/regressions/**/*.png
decisions: []
metrics:
  duration: ~15 min
  completed: 2026-04-08
---

# Quick Task 005: Re-run Pipeline Analyses and Update Quarto Manuscript

Regenerated all downstream pipeline outputs (scripts 14-16) with full 154-participant dataset and added two new manuscript figures.

## Tasks Completed

### Task 1: Re-run pipeline scripts 14, 15, 16

All three downstream analysis scripts ran successfully with 154 participants (previously stale at 14).

**Script 14 (Model Comparison):**
- M6b (dual perseveration) confirmed as winning model by both AIC and BIC
- dAIC = 572.9 over second-best (M5), Akaike weight = 100%
- Per-participant AIC: M6b wins 55/154 (35.7%), M5 wins 41 (26.6%), M6a wins 38 (24.7%)
- Stratified analysis: 45 participants matched to trauma groups

**Script 15 (Parameter-Trauma Analysis):**
- All 7 models analyzed (N=154 per model)
- Spearman correlations, group comparisons, OLS regressions regenerated
- 288 files updated (CSVs + figures)

**Script 16 (Scale Regressions):**
- Simple and multiple regressions for all 7 models
- N=154 per model in regression outputs

### Task 2: Add model overview chart and parameter distributions to manuscript

**Figure A: Model Parameter Overview (fig-model-architecture)**
- Heatmap grid showing which parameters belong to which model
- Built programmatically from MODEL_REGISTRY (not hardcoded)
- Parameters grouped by category (learning rates, WM, perseveration, RL forgetting, noise, LBA)
- M4 separated visually with dashed line
- Saved as PDF to manuscript/outputs/

**Figure B: Parameter Distributions (fig-param-distributions)**
- 2x2 violin plot grid showing alpha_neg, phi, rho, K across choice-only models
- Shows how parameter estimates shift with model complexity
- Reference lines for bounded parameters (phi, rho in [0,1])

**Template Fixes (Deviation Rule 3 - Blocking):**
- Added tcolorbox and fontawesome5 to arxiv_template.tex (required for Quarto callout rendering)
- Fixed inline math escaping: `$-$` replaced with en-dash, alpha subscripts braced with `{}`

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] LaTeX tcolorbox package missing from template**
- Found during: Task 2 verification (quarto render)
- Issue: arxiv_template.tex lacked tcolorbox and fontawesome5 packages needed by Quarto callout blocks
- Fix: Added `\usepackage[most]{tcolorbox}`, `\usepackage{fontawesome5}`, and callout color definitions
- Files modified: manuscript/arxiv_template.tex
- Commit: 6b045a4

**2. [Rule 3 - Blocking] Inline math escaping broke pdflatex compilation**
- Found during: Task 2 verification (quarto render)
- Issue: `$-$` (negative sign in math) was rendered as literal `\$-\$` by Pandoc, and `$\alpha_-$` subscript was mis-escaped
- Fix: Replaced `$-$` with en-dash `--`, braced alpha subscripts as `$\alpha_{-}$` and `$\alpha_{+}$`
- Files modified: manuscript/paper.qmd
- Commit: 6b045a4

## Commits

| Hash | Message |
|------|---------|
| 1fd9bc7 | results(quick-005): regenerate pipeline outputs for 154 participants |
| 6b045a4 | feat(quick-005): add model overview chart and parameter distribution figures to manuscript |

## Verification

- participant_wins.csv: total=154 (confirmed)
- spearman_correlations.csv: n=154 (confirmed)
- regression CSVs: N=154 (confirmed)
- fig_model_overview.pdf: 37.5 KB (confirmed)
- fig_param_distributions.pdf: 48.2 KB (confirmed)
- quarto render paper.qmd --to pdf: completed without errors
