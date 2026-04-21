---
wave: 5
depends_on: [28-07, 28-08, 28-09]
files_modified:
  - manuscript/paper.qmd
autonomous: false
---

# 28-10 paper.qmd Bayesian-First Structural Scaffolding

## Goal

Reorder the `manuscript/paper.qmd` Results section so Bayesian model selection comes FIRST (summary → Bayesian fits → L2 trauma regression → subscale breakdown → appendix with MLE + recovery + Bayesian↔MLE scatter), and add graceful-fallback Quarto `{python}` code cells for the cross-refs that currently render as broken links (`@tbl-loo-stacking`, `@tbl-rfx-bms`, `@fig-forest-21`, `@tbl-winner-betas`). Do NOT populate real data — Phase 26 (MANU-01..05) does that after Phase 24 produces the real Bayesian artifacts. This plan only creates the structural skeleton with placeholders.

**Non-autonomous.** A human should skim the reordered Results section before committing to catch prose discontinuities (e.g., paragraph transitions that no longer make sense after reordering).

## Must Haves

- [ ] `manuscript/paper.qmd` Results section (`# Results {#sec-results}`) reordered to canonical Bayesian-first structure:
  1. Summary results (cohort, behavior, group descriptives — from current "Model Comparison" intro, terse)
  2. Bayesian model fitting & selection (formerly `#sec-bayesian-selection` at line 979 — moved to the top of Results)
  3. Hierarchical Level-2 trauma regression (formerly `#sec-level2-trauma` or similar — winner refit with LEC + IES-R covariates)
  4. Subscale breakdown (M6b 4-covariate subscale L2)
  5. Appendix sections: MLE Model Comparison, Parameter Recovery, Parameter-Trauma relationships, Bayesian↔MLE scatter, Continuous Trauma Associations (moved into Appendix block)
- [ ] Graceful-fallback `{python}` code cells added for the following cross-refs, modeled on the existing `#fig-l2-forest` pattern at paper.qmd lines 1019–1047:
  - `@tbl-loo-stacking` → reads `../output/bayesian/manuscript/loo_stacking.csv` (or falls back to `../output/bayesian/21_baseline/loo_stacking_results.csv`); if missing, renders a single-row placeholder "`[Phase 24 cold-start will populate]`"
  - `@tbl-rfx-bms` → reads `../output/bayesian/manuscript/rfx_bms.csv`; placeholder if missing
  - `@fig-forest-21` → loads `../figures/21_bayesian/forest_plot.png`; placeholder markdown block if missing (the scaffold dir exists per plan 28-07)
  - `@tbl-winner-betas` → reads `../output/bayesian/manuscript/winner_betas.csv`; placeholder if missing
- [ ] All existing working cross-refs (`@tbl-model-comparison`, `@tbl-stacking-weights`, `@fig-winner-heterogeneity`, `@fig-posterior-vs-mle`, etc.) remain functional — no path changes from plan 28-07 (which explicitly preserved `../output/mle`, `../output/model_comparison`, `../output/trauma_groups`, `../output/bayesian/level2`).
- [ ] `quarto render manuscript/paper.qmd` from repo root exits 0 and produces `manuscript/_output/paper.pdf`. Warnings about missing optional figures are acceptable (the graceful-fallback cells catch them).
- [ ] Prose continuity preserved — if any inline `{python}` ref (e.g., `{python} winner_display`) becomes unreachable because its computing cell was moved, either (a) move the computing cell along with it or (b) replace the inline ref with a placeholder literal like `[winner]`. Do NOT leave orphaned `{python}` refs.
- [ ] No section IDs (`{#sec-*}`) renamed — only their ORDER in the document changes, so existing cross-refs from Introduction/Discussion still resolve.
- [ ] `pytest scripts/fitting/tests/test_v4_closure.py` still passes 3/3 (plan changes only manuscript, not fitting code).
- [ ] Atomic commit: `refactor(28-10): reorder paper.qmd Results to Bayesian-first + add graceful-fallback cells for Phase 24 artifacts`.

## Tasks

<tasks>
  <task id="1">
    <title>Read paper.qmd in full and map current Results sub-sections</title>
    <detail>Read `manuscript/paper.qmd` in full (1301 lines). Construct a section outline: for each `##` or `###` header inside `# Results {#sec-results}`, record (line range, section ID, dependent `{python}` cells, dependent `@tbl-*`/`@fig-*` cross-refs, downstream prose that references this section). This outline IS the reordering plan — do not skip this step.</detail>
  </task>

  <task id="2">
    <title>Identify the target Bayesian-first section order</title>
    <detail>Per roadmap Phase 28 paper-restructure scope, target order is:
      1. **Summary results** (cohort N, behavioral accuracy, RT descriptives) — pull terse material from current intro + behavioral paragraphs; keep under 2 paragraphs.
      2. **Bayesian model fitting** — current `#sec-bayesian-selection` (line 979) promoted to first analytical section; include LOO-stacking + RFX-BMS + PXP selection; winner(s) identified with posterior summaries.
      3. **Hierarchical Level-2 trauma regression** — current `#fig-l2-forest` cell + surrounding prose (lines 988–1047); winner refit with LEC + IES-R covariates.
      4. **Subscale breakdown** — M6b 4-covariate L2 section if it exists; otherwise add a stub that the `21_7_scale_audit` output will populate.
      5. **Appendix (moved from main Results)**: MLE AIC/BIC table (current `@tbl-model-comparison`), Winner Heterogeneity, Parameter Recovery, Parameter Estimates, Parameter-Trauma Groups, Continuous Trauma Associations, Cross-Model Consistency, Bayesian↔MLE scatter.
      Write the target order out explicitly in the plan execution log before touching paper.qmd.</detail>
  </task>

  <task id="3">
    <title>Add graceful-fallback {python} cells for missing cross-refs</title>
    <detail>Use the existing `#fig-l2-forest` cell at lines 1019–1047 as the template. The pattern is:
      ```python
      #| label: fig-l2-forest
      #| fig-cap: "..."
      import pandas as pd
      from pathlib import Path
      p = Path("../output/bayesian/level2/wmrl_m6b_forest.png")
      if p.exists():
          from IPython.display import Image
          Image(filename=str(p))
      else:
          print("[Phase 24 cold-start will populate]")
      ```
      Create analogous cells for:
        - `#| label: tbl-loo-stacking` — reads `../output/bayesian/manuscript/loo_stacking.csv`, falls back to `../output/bayesian/21_baseline/loo_stacking_results.csv`; renders as markdown table with graceful placeholder.
        - `#| label: tbl-rfx-bms` — reads `../output/bayesian/manuscript/rfx_bms.csv`.
        - `#| label: fig-forest-21` — loads `../figures/21_bayesian/forest_plot.png` (scaffold dir exists per plan 28-07).
        - `#| label: tbl-winner-betas` — reads `../output/bayesian/manuscript/winner_betas.csv`.
      Place these cells in their logically-correct sections of the reordered Results.</detail>
  </task>

  <task id="4">
    <title>Reorder Results sub-sections</title>
    <detail>Execute the reorder per task 2's target order. Use `Edit` tool with large unique blocks (heading + body) rather than line ranges. Move each `## Sub-section` with all its body + figures + prose intact. When a paragraph transition no longer reads correctly (e.g., "As shown in the previous section..." when the referenced section is now in the appendix), rewrite the transitional sentence minimally.</detail>
  </task>

  <task id="5">
    <title>Create Appendix container for moved MLE content</title>
    <detail>If the paper.qmd doesn't already have a clear Appendix H2 inside Results (it currently has `# Appendix` at the end, per 28-RESEARCH.md §paper.qmd current state), move the MLE-centric subsections into `# Appendix` with sub-sub-sections `## Appendix A: MLE Model Comparison`, `## Appendix B: Parameter Recovery`, etc. Preserve all existing MLE-related `{python}` cells — they already work with `../output/mle/` paths which plan 28-07 preserved.</detail>
  </task>

  <task id="6">
    <title>Verify no orphaned inline {python} refs</title>
    <detail>Grep paper.qmd for `{python}` inline refs (e.g., `{python} winner_display`). Each must still have a computing cell visible earlier in the document. If any cell was moved but its consumer stayed, either move the cell OR inline a placeholder literal.</detail>
  </task>

  <task id="7">
    <title>Run quarto render and iterate on errors</title>
    <detail>`cd manuscript && quarto render paper.qmd` (or `quarto render manuscript/paper.qmd` from repo root). Expected result: exit 0, PDF produced at `manuscript/_output/paper.pdf`. Warnings are acceptable (missing optional figures will be filled by Phase 26). If exit ≠ 0, iterate: most common errors will be (a) unresolved cross-refs (add the graceful-fallback cell), (b) missing file paths in non-graceful cells (wrap them in `if Path(...).exists()`), (c) duplicate cell labels (rename).</detail>
  </task>

  <task id="8">
    <title>Human skim</title>
    <detail>The plan is marked autonomous=false. After tasks 1-7 produce a rendered PDF, ask the user to skim the reordered Results for prose flow — especially paragraph transitions near section boundaries. Apply any prose fixups the user suggests as a follow-up commit on the same plan (or as additional edits staged in the same commit).</detail>
  </task>

  <task id="9">
    <title>Atomic commit</title>
    <detail>`refactor(28-10): reorder paper.qmd Results to Bayesian-first + add graceful-fallback cells for Phase 24 artifacts`. Body: list the 5 sub-sections in the new order, the 4 new graceful-fallback cells, and note that Phase 26 will populate real values.</detail>
  </task>
</tasks>

## Verification

```bash
# Quarto renders successfully
quarto render manuscript/paper.qmd
test -f manuscript/_output/paper.pdf

# New graceful-fallback cells exist
grep -n "label: tbl-loo-stacking" manuscript/paper.qmd
grep -n "label: tbl-rfx-bms" manuscript/paper.qmd
grep -n "label: fig-forest-21" manuscript/paper.qmd
grep -n "label: tbl-winner-betas" manuscript/paper.qmd

# Bayesian section is now near the TOP of Results (line number should be much earlier than 979)
grep -n "#sec-bayesian-selection\|Bayesian Model" manuscript/paper.qmd | head -3

# Existing working cross-refs still present
grep -n "label: tbl-model-comparison\|label: fig-l2-forest\|label: fig-winner-heterogeneity" manuscript/paper.qmd

# Path invariant from plan 28-07 preserved
grep -n "\\.\\./output/mle\\|\\.\\./output/model_comparison\\|\\.\\./output/trauma_groups\\|\\.\\./output/bayesian/level2" manuscript/paper.qmd

# Closure invariant
pytest scripts/fitting/tests/test_v4_closure.py -v
```

## Requirement IDs

Closes: **REFAC-11**.
