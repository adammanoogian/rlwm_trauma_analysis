---
phase: quick-005
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - output/model_comparison/comparison_results.csv
  - output/model_comparison/participant_wins.csv
  - output/model_comparison/stratified_results.csv
  - output/model_comparison/model_group_crosstab.csv
  - output/mle/group_comparison_stats.csv
  - output/mle/spearman_correlations.csv
  - output/mle/ols_regression_results.csv
  - output/regressions/wmrl_m6b/regression_results_simple.csv
  - output/regressions/wmrl_m6b/regression_results_multiple.csv
  - figures/model_comparison/*.png
  - figures/mle_trauma_analysis/*.png
  - figures/regressions/**/*.png
  - manuscript/paper.qmd
  - manuscript/figures/plot_utils.py
autonomous: true

must_haves:
  truths:
    - "Model comparison outputs reflect all 154 participants (not 14)"
    - "Parameter-trauma analyses run for all 7 models with 154 subjects"
    - "Regression outputs regenerated for all 7 models"
    - "Manuscript includes model parameter overview chart showing M1-M6b progression"
    - "Manuscript includes parameter correlation/spread plots for trauma-relevant params"
    - "Manuscript renders with quarto without errors"
  artifacts:
    - path: "output/model_comparison/comparison_results.csv"
      provides: "AIC/BIC comparison for 6 choice-only models"
      contains: "M6b"
    - path: "output/model_comparison/participant_wins.csv"
      provides: "Per-participant AIC wins"
      contains: "154"
    - path: "output/mle/spearman_correlations.csv"
      provides: "Spearman correlations for all models"
    - path: "manuscript/paper.qmd"
      provides: "Updated manuscript with new figures"
  key_links:
    - from: "scripts/14_compare_models.py"
      to: "output/model_comparison/"
      via: "regenerated CSV + PNG outputs"
    - from: "scripts/15_analyze_mle_by_trauma.py"
      to: "output/mle/ and figures/mle_trauma_analysis/"
      via: "group comparison stats, correlations, figures"
    - from: "manuscript/paper.qmd"
      to: "output/model_comparison/comparison_results.csv"
      via: "pd.read_csv in setup cell"
---

<objective>
Re-run analysis pipeline scripts 14-16 to regenerate all outputs with the full 154-participant dataset (currently stale at 14 participants), then update the Quarto manuscript with: (a) a model parameter overview chart showing which parameters each model adds in the M1-M6b progression, (b) parameter correlation/spread plots for trauma-relevant parameters, and (c) embedded figures from regenerated pipeline output.

Purpose: The MLE fits for all 7 models completed on the cluster (154 subjects, 50 starts), but downstream analysis scripts (14, 15, 16) were last run with only 14 participants. Regenerating these outputs is required before the manuscript figures and tables can display correct values.

Output: Regenerated comparison CSVs, correlation CSVs, regression CSVs, all associated figures, and an updated paper.qmd with new manuscript figures.
</objective>

<execution_context>
@C:\Users\aman0087\.claude/get-shit-done/workflows/execute-plan.md
@C:\Users\aman0087\.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@config.py (MODEL_REGISTRY with all 7 model definitions)
@manuscript/paper.qmd (current manuscript -- 772 lines)
@manuscript/figures/plot_utils.py (GROUP_COLORS, MODEL_DISPLAY_NAMES, PARAM_DISPLAY_NAMES)
</context>

<tasks>

<task type="auto">
  <name>Task 1: Re-run pipeline scripts 14, 15, 16</name>
  <files>
    output/model_comparison/comparison_results.csv
    output/model_comparison/participant_wins.csv
    output/model_comparison/stratified_results.csv
    output/mle/group_comparison_stats.csv
    output/mle/spearman_correlations.csv
    figures/model_comparison/*.png
    figures/mle_trauma_analysis/*.png
    output/regressions/*/regression_results_simple.csv
    output/regressions/*/regression_results_multiple.csv
    figures/regressions/**/*.png
  </files>
  <action>
Run the three downstream analysis scripts in order. Each script is self-contained and reads from output/mle/ MLE fit CSVs.

1. Model comparison (script 14):
```bash
cd C:/Users/aman0087/Documents/Github/rlwm_trauma_analysis
python scripts/14_compare_models.py
```
Verify output: `output/model_comparison/participant_wins.csv` should show `total` column = 154 (not 14). Check `output/model_comparison/comparison_results.csv` has all 6 choice-only models.

2. Parameter-trauma analysis (script 15, all models):
```bash
python scripts/15_analyze_mle_by_trauma.py --model all
```
Verify: `output/mle/spearman_correlations.csv` and `output/mle/group_comparison_stats.csv` both updated. Figures regenerated in `figures/mle_trauma_analysis/`.

3. Scale regressions (script 16, all models):
```bash
python scripts/16_regress_parameters_on_scales.py --model all
```
Verify: `output/regressions/wmrl_m6b/regression_results_simple.csv` and `regression_results_multiple.csv` exist with N>100 in the N column.

After all three complete, spot-check key numbers:
- `participant_wins.csv`: total should be 154
- `comparison_results.csv`: M6b should be first row (lowest aggregate_aic)
- `spearman_correlations.csv`: n column should be ~147-154 (not ~14)
  </action>
  <verify>
```bash
# Verify participant count in model comparison
head -2 output/model_comparison/participant_wins.csv
# Verify N in correlations
head -3 output/mle/spearman_correlations.csv
# Verify regression N
head -2 output/regressions/wmrl_m6b/regression_results_simple.csv
```
All should show N approximately 147-154, not 14.
  </verify>
  <done>All three pipeline scripts complete without errors. participant_wins.csv shows total=154, spearman_correlations.csv shows n~147-154, regression CSVs show N>100. Figures regenerated in figures/ directories.</done>
</task>

<task type="auto">
  <name>Task 2: Add model parameter overview chart and parameter spread plots to manuscript</name>
  <files>
    manuscript/paper.qmd
    manuscript/figures/plot_utils.py
  </files>
  <action>
Add two new figure-generating code cells to paper.qmd. Place them in the Methods section (after the model parameter glossary callout, before the model fitting subsection).

**Figure A: Model Parameter Overview Chart (fig-model-architecture)**

Create a code cell that generates a visual chart showing which parameters each model adds in the M1-M6b progression. Implementation:

1. Build a matrix: rows = parameters (alpha_pos, alpha_neg, phi, rho, K, kappa, kappa_s, kappa_total, kappa_share, phi_rl, epsilon, v_scale, A, delta, t0), columns = models (M1, M2, M3, M5, M6a, M6b, M4). Cell = 1 if model has parameter, 0 otherwise.
2. Use `plt.imshow()` or `plt.pcolormesh()` to render as a heatmap grid. Color filled cells a muted blue, empty cells white. Add thin grid lines.
3. Annotate: On each filled cell, place a small checkmark or filled circle. On empty cells, leave blank.
4. Y-axis: parameter display names from PARAM_DISPLAY_NAMES. X-axis: model short names (M1, M2, M3, M5, M6a, M6b, M4). Separate M4 visually with a vertical dashed line or gap.
5. Group parameters by category using horizontal separators or subtle background shading:
   - Learning rates: alpha_pos, alpha_neg
   - WM: phi, rho, K
   - Perseveration: kappa, kappa_s, kappa_total, kappa_share
   - RL forgetting: phi_rl
   - Noise: epsilon
   - LBA (M4 only): v_scale, A, delta, t0
6. Figure size: (TEXT_WIDTH, 4.5). Save to OUTPUT_DIR / "fig_model_overview.pdf".
7. Use fig-cap: "Model parameter overview. Each column shows the free parameters for one model. Models are ordered by complexity within the choice-only family (M1--M6b); M4 (joint choice+RT, dashed separator) includes additional LBA accumulator parameters."

Read MODEL_REGISTRY from config.py to build the matrix programmatically -- do NOT hardcode which parameters belong to which model.

**Figure B: Parameter Distributions by Model (fig-param-distributions)**

Add a second code cell that shows violin/box plots of key trauma-relevant parameters across all choice-only models. This lets readers see how parameter estimates shift as model complexity increases. Implementation:

1. Load individual fits for all choice-only models from the `fits` dict (already loaded in setup cell).
2. Select shared WM parameters that appear in M2+: phi, rho, capacity. Also show alpha_neg (shared across all models).
3. Create a 2x2 grid of subplots, one per parameter.
4. For each parameter subplot: plot violin plots (one per model that has that parameter), colored by model. Use a sequential or categorical colormap (e.g., tab10). X-axis: model short names. Y-axis: parameter value.
5. Add horizontal reference lines at parameter bounds if meaningful (e.g., phi in [0,1], rho in [0,1]).
6. Figure size: (TEXT_WIDTH, 4.5). Save to OUTPUT_DIR / "fig_param_distributions.pdf".
7. Use fig-cap: "Distribution of key parameter estimates across choice-only models. Violins show the full distribution across 154 participants. Shared parameters (phi, rho, K, alpha_neg) are plotted for each model that includes them."

Place both figures in a new subsection or directly within the Models subsection, after the parameter glossary callout and before the "Model Fitting" subsection. Use the Quarto label/fig-cap pattern consistent with the rest of the manuscript.

**Important implementation notes:**
- Import MODEL_REGISTRY, CHOICE_ONLY_MODELS from config (already imported in setup cell).
- Use PARAM_DISPLAY_NAMES from plot_utils for axis labels.
- Use MODEL_DISPLAY_NAMES or MODEL_REGISTRY short_names for x-axis labels.
- Follow the existing pattern: `try: ... except Exception as e: print(f"... ({e})")` for graceful failure.
- Save PDFs to OUTPUT_DIR (manuscript/outputs/) matching existing figure pattern.
  </action>
  <verify>
```bash
cd C:/Users/aman0087/Documents/Github/rlwm_trauma_analysis/manuscript
quarto render paper.qmd --to pdf 2>&1 | tail -20
```
Manuscript renders without errors. Check that `manuscript/outputs/fig_model_overview.pdf` and `manuscript/outputs/fig_param_distributions.pdf` are generated.
  </verify>
  <done>paper.qmd contains two new figure cells (model parameter overview chart and parameter distribution violins). Both figures render in the PDF output. The overview chart programmatically reads MODEL_REGISTRY. The distribution plot shows phi, rho, K, alpha_neg across choice-only models.</done>
</task>

</tasks>

<verification>
1. Run `head -2 output/model_comparison/participant_wins.csv` -- total column shows 154
2. Run `head -3 output/mle/spearman_correlations.csv` -- n column shows ~147-154
3. Run `ls manuscript/outputs/fig_model_overview.pdf manuscript/outputs/fig_param_distributions.pdf` -- both exist
4. Run `quarto render manuscript/paper.qmd --to pdf` -- completes without errors
5. Visually inspect the PDF: model overview chart shows parameter grid, distribution violins show 4 parameters across models
</verification>

<success_criteria>
- All 3 pipeline scripts (14, 15, 16) ran successfully with 154 participants
- Model comparison outputs updated (participant_wins.csv total=154)
- Spearman correlations and regressions regenerated for all 7 models
- Manuscript has model parameter overview chart (programmatic from MODEL_REGISTRY)
- Manuscript has parameter distribution violin plots (4 key params x choice-only models)
- Quarto renders paper.qmd to PDF without errors
</success_criteria>

<output>
After completion, create `.planning/quick/005-rerun-pipeline-analyses-update-quarto-manuscript/005-SUMMARY.md`
</output>
