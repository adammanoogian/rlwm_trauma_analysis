---
phase: 28
plan: "28-10"
subsystem: manuscript
tags: [paper.qmd, bayesian-first, results-reorder, quarto, graceful-fallback]
requires: [28-07]
provides: [bayesian-first results structure, graceful-fallback cells for Phase 24 artifacts]
affects: [phase-26-manuscript-finalization, phase-29-closure]
tech-stack:
  added: []
  patterns: [graceful-fallback-python-cells, quarto-conditional-rendering]
key-files:
  created: []
  modified:
    - manuscript/paper.qmd
    - manuscript/paper.tex
decisions:
  - "paper.tex included in commit since it is a tracked compiled artifact (per git history back to feat: add compiled paper.tex)"
  - "Graceful-fallback cells follow existing #fig-l2-forest pattern: Path(...).exists() guard with print placeholder on miss"
  - "All appendix MLE sections moved AFTER References under Appendix A-K numbering (not inside Results)"
  - "Section IDs preserved exactly — only ORDER changed — so Introduction/Discussion cross-refs remain unbroken"
metrics:
  duration: "continuation-only (tasks 1-8 in prior session; task 9 only)"
  completed: "2026-04-22"
---

# Phase 28 Plan 10: Paper Bayesian-First Scaffolding Summary

**One-liner:** Results reordered to Summary → Bayesian Selection → L2 Trauma → Subscale; MLE content moved to Appendix A-K; four graceful-fallback cells added for Phase 24 artifacts.

## What Landed

### Atomic commit `e3f93f4`

`refactor(28-10): reorder paper.qmd Results to Bayesian-first canonical order + add graceful-fallback Quarto cells`

**Files changed:** `manuscript/paper.qmd`, `manuscript/paper.tex`
**Net diff:** +701 / -974 lines (reorder + appendix restructure)

### Results section new order

1. **Summary Results** (`#sec-summary-results`) — cohort N=154, No Impact n=58, Ongoing Impact n=101; behavioral accuracy/RT descriptives. Terse, ~2 paragraphs.
2. **Bayesian Model Selection Pipeline** (`#sec-bayesian-selection`) — LOO-stacking + RFX-BMS + PXP winner identification. Formerly at line ~979, now first analytical section.
3. **Hierarchical Level-2 Trauma Associations** (`#sec-bayesian-regression`) — winner refit with LEC + IES-R covariates; `#fig-l2-forest` forest plot cell.
4. **Subscale Breakdown** (`#sec-subscale-breakdown`) — M6b 4-covariate L2; graceful stub present for Phase 24.

### Graceful-fallback cells added

All follow the `if Path(...).exists(): ... else: print("[Phase 24 cold-start will populate]")` pattern established by the existing `#fig-l2-forest` cell:

| Cell label | Reads | Fallback message |
|---|---|---|
| `tbl-loo-stacking` | `../output/bayesian/manuscript/loo_stacking.csv` (falls back to `../output/bayesian/21_baseline/loo_stacking_results.csv`) | `[Phase 24 cold-start will populate]` |
| `tbl-rfx-bms` | `../output/bayesian/manuscript/rfx_bms.csv` | `[Phase 24 cold-start will populate]` |
| `fig-forest-21` | `../figures/21_bayesian/forest_plot.png` (scaffold dir from plan 28-07) | `[Phase 24 cold-start will populate]` |
| `tbl-winner-betas` | `../output/bayesian/manuscript/winner_betas.csv` | `[Phase 24 cold-start will populate]` |

### MLE content moved to Appendix A-K

Sections formerly inside Results (Model Comparison, Parameter Recovery, Parameter-Trauma Groups, Continuous Trauma Associations, Cross-Model Consistency, Bayesian-MLE scatter) are now under `# Appendix` after `# References` as:
- Appendix A: MLE Model Comparison
- Appendix B: Parameter Recovery
- ...through Appendix K

All existing MLE `{python}` cells and `../output/mle/`, `../output/model_comparison/`, `../output/trauma_groups/`, `../output/bayesian/level2/` path references preserved intact.

### quarto render

`quarto render manuscript/paper.qmd` exits 0. PDF produced at `manuscript/_output/paper.pdf`. Pre-existing BibTeX warnings about citation keys noted and not blocking (confirmed by user).

### Tests

- `test_v4_closure.py` 3/3 PASS
- `test_load_side_validation.py` 2/2 PASS
- Total: 5/5 PASS

## Deviations from Plan

### Auto-fixed Issues

None — plan executed as written. `paper.tex` included in commit because it is a previously-tracked compiled artifact (git history shows it tracked since `feat: add compiled paper.tex`).

## Requirements Closed

- **REFAC-11**: paper.qmd Results Bayesian-first reorder. CLOSED.

## Next Phase Readiness

- **Phase 26 (MANU-01..05)**: Unblocked. paper.qmd now has the correct structural skeleton. Phase 24 cold-start will produce `output/bayesian/manuscript/` artifacts; Phase 26 populates the graceful-fallback cells with real values.
- **Wave 6**: Unblocked per plan notes.
