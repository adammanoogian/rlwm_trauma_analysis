---
phase: 18
plan: 03
subsystem: analysis-scripts
tags: [deprecation, mle, bayesian, reliability, shrinkage, visualization]
requires: []
provides:
  - scripts/16b_bayesian_regression.py (deprecation docstring)
  - scripts/18b_mle_vs_bayes_reliability.py (MLE-vs-Bayesian reliability scatterplots)
affects:
  - 18-04 (MODEL_REFERENCE.md update may reference 18b as supplementary figure generator)
  - 18-05 (manuscript revision may cite 16b as deprecated supplementary path)
tech-stack:
  added: []
  patterns:
    - deprecation-docstring
    - mle-vs-bayes-reliability-scatter
    - shrinkage-arrow-visualization
key-files:
  created:
    - scripts/18b_mle_vs_bayes_reliability.py
  modified:
    - scripts/16b_bayesian_regression.py
decisions:
  - "16b deprecation docstring only (no code deletion): keeps script runnable for exploratory use"
  - "highlight_shrinkage=True for wmrl_m6b only (winning model), arrow from identity line to posterior mean"
  - "strict=False in zip() for shrinkage arrows: tolerates Python < 3.10 strict keyword absence"
metrics:
  duration: "~5 min"
  completed: "2026-04-13"
---

# Phase 18 Plan 03: 16b Deprecation + MLE-vs-Bayes Reliability Scatterplots Summary

**One-liner:** Froze 16b with a `.. deprecated::` docstring referencing the L2 hierarchical pipeline, and created 18b to generate MLE-vs-Bayesian posterior-mean scatter plots with M6b shrinkage arrows.

## Objective

MIG-04: Mark the old post-hoc regression script (16b) as deprecated while keeping it runnable.
MIG-05: Create standalone script 18b that generates one scatter per (parameter, model) cell comparing MLE point estimates to Bayesian posterior means, with shrinkage arrows for the winning model (M6b).

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add deprecation docstring to script 16b | 94bb986 | scripts/16b_bayesian_regression.py |
| 2 | Create MLE-vs-Bayesian reliability scatterplot script | 216e1aa | scripts/18b_mle_vs_bayes_reliability.py |

## What Was Built

### Task 1: Deprecation Docstring (16b)

Added `.. deprecated::` RST directive to the module docstring of `scripts/16b_bayesian_regression.py`. The notice:
- Names the Level-2 hierarchical pipeline (scripts 13 + 18) as the successor
- Clarifies 16b's retained role: fast-preview / supplementary path on MLE point estimates
- Does NOT break any imports or code paths — script remains fully runnable

### Task 2: Reliability Scatterplot Script (18b)

Created `scripts/18b_mle_vs_bayes_reliability.py` with:

**`plot_mle_vs_bayes_reliability()`** core function:
- Inner-joins MLE and Bayesian DataFrames on `participant_id`
- Plots scatter (alpha=0.5, s=20, steelblue) of MLE vs posterior mean
- Draws dashed 45-degree identity reference line
- Annotates Pearson r in upper-left corner
- When `highlight_shrinkage=True`: draws `annotate` arrows from `(mle_val, mle_val)` on the identity line toward `(mle_val, bayes_val)` in steelblue (lw=0.3) to visualise shrinkage direction
- Saves to `output_dir/{model_name}_{param}.png` at dpi=200

**`main()`** orchestrator:
- `--model` argument (default `all`) iterates over MODEL_REGISTRY keys
- `--output-dir` argument (default `output/bayesian/figures/mle_vs_bayes`)
- Loads `output/mle/{model}_individual_fits.csv` and `output/bayesian/{model}_individual_fits.csv`
- Graceful skip with stderr warning when either CSV is missing
- Sets `highlight_shrinkage=True` automatically for `wmrl_m6b` (winning model)
- Prints summary count and output directory on completion

## Verification

All checks passed:
1. `python -c "import ast; ast.parse(...)"` — both scripts parse as valid Python
2. `python scripts/18b_mle_vs_bayes_reliability.py --help` — shows `--model` and `--output-dir`
3. `grep -rn "deprecated" scripts/16b_bayesian_regression.py` — line 6 match in docstring
4. `grep -n "highlight_shrinkage" scripts/18b_mle_vs_bayes_reliability.py` — function signature, docstring, implementation, and call site

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Deprecation docstring only; no code deletion | 16b is retained as fast-preview path; code deletion would break users who rely on it for exploratory checks before running full pipeline |
| highlight_shrinkage for M6b only | M6b is the winning model; shrinkage visualisation is most scientifically meaningful for the parameters being reported in the manuscript |
| Graceful skip (stderr warning) for missing CSVs | Bayesian individual fits do not yet exist locally (cluster job pending); script must be runnable in partial-data states |
| Output to `output/bayesian/figures/mle_vs_bayes/` | Namespaced under `bayesian/` to reflect that y-axis data is Bayesian; segregated from Level-2 forest plots in `bayesian/level2/` |

## Deviations from Plan

None - plan executed exactly as written.

## Next Phase Readiness

Plan 18-04 (MODEL_REFERENCE.md update) can proceed immediately. No blockers introduced by this plan.

The new 18b script will produce figures once cluster Bayesian fits complete (output/bayesian/{model}_individual_fits.csv files generated by scripts/13_fit_bayesian.py).
