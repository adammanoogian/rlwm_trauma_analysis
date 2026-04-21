---
phase: quick-003
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - manuscript/paper.qmd
  - manuscript/figures/plot_utils.py
autonomous: true

must_haves:
  truths:
    - "If winning model changes (e.g., M6b beats M5), paper re-renders with correct winner everywhere"
    - "All participant counts, group names, parameter names derive from data files or config.py"
    - "Model comparison table reads from comparison_results.csv, not performance_summary.json"
    - "Parameter-by-group violin plot works with actual group names from group_assignments.csv"
    - "Methods text shows n_starts=50, correct group names, correct MIN_TRIALS_THRESHOLD"
  artifacts:
    - path: "manuscript/paper.qmd"
      provides: "Fully data-driven Quarto manuscript"
    - path: "manuscript/figures/plot_utils.py"
      provides: "Updated plot utilities with actual group colors keyed by real group names"
  key_links:
    - from: "manuscript/paper.qmd setup cell"
      to: "output/model_comparison/comparison_results.csv"
      via: "pd.read_csv to determine winning_model"
      pattern: "comparison_results\\.csv"
    - from: "manuscript/paper.qmd setup cell"
      to: "config.py MODEL_REGISTRY"
      via: "sys.path.insert + import"
      pattern: "MODEL_REGISTRY"
    - from: "manuscript/paper.qmd setup cell"
      to: "output/trauma_groups/group_assignments.csv"
      via: "pd.read_csv for group names and counts"
      pattern: "group_assignments\\.csv"
---

<objective>
Softcode the Quarto manuscript (manuscript/paper.qmd) so that the winning model, participant counts, group names, fitting parameters, and all model references are derived from data files and config.py rather than hardcoded. If the winning model changes after re-fitting, the paper should automatically update on re-render.

Purpose: The paper currently hardcodes M5 as the winner, uses wrong group names ("control", "exposed", "symptomatic" instead of actual "No Ongoing Impact", "Ongoing Impact"), says "10 random restarts" (actual: 50), and has a broken model comparison table. Making everything data-driven ensures the manuscript stays correct as analysis evolves.

Output: Updated manuscript/paper.qmd and manuscript/figures/plot_utils.py
</objective>

<execution_context>
@C:\Users\aman0087\.claude/get-shit-done/workflows/execute-plan.md
@C:\Users\aman0087\.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@manuscript/paper.qmd
@manuscript/figures/plot_utils.py
@config.py
</context>

<tasks>

<task type="auto">
  <name>Task 1: Update plot_utils.py group colors and add reverse-lookup helper</name>
  <files>manuscript/figures/plot_utils.py</files>
  <action>
Update manuscript/figures/plot_utils.py to support actual group names from data:

1. **Replace GROUP_COLORS keys** with actual group names from group_assignments.csv. The data has TWO groups (not three):
   - `"Trauma Exposure - No Ongoing Impact"` (29 participants)
   - `"Trauma Exposure - Ongoing Impact"` (24 participants)

   Replace the existing GROUP_COLORS dict with:
   ```python
   GROUP_COLORS: dict[str, str] = {
       "Trauma Exposure - No Ongoing Impact": "#06A77D",  # Green
       "Trauma Exposure - Ongoing Impact": "#D62246",     # Red
   }
   ```

   Add short labels for axis tick labels:
   ```python
   GROUP_SHORT_LABELS: dict[str, str] = {
       "Trauma Exposure - No Ongoing Impact": "No Impact",
       "Trauma Exposure - Ongoing Impact": "Ongoing Impact",
   }
   ```

2. **Add a reverse-lookup dict** from MODEL_REGISTRY short_name to internal key:
   ```python
   # Reverse lookup: short_name -> internal key (e.g., "M6b" -> "wmrl_m6b")
   SHORT_NAME_TO_KEY: dict[str, str] = {
       "M1": "qlearning",
       "M2": "wmrl",
       "M3": "wmrl_m3",
       "M4": "wmrl_m4",
       "M5": "wmrl_m5",
       "M6a": "wmrl_m6a",
       "M6b": "wmrl_m6b",
   }
   ```

3. **Export the new names** from the module (GROUP_SHORT_LABELS, SHORT_NAME_TO_KEY).

4. Keep `from __future__ import annotations` at top. Keep all existing exports unchanged (apply_manuscript_style, GROUP_COLORS, MODEL_DISPLAY_NAMES, PARAM_DISPLAY_NAMES, COLUMN_WIDTH, TEXT_WIDTH). Update docstrings to reflect 2 groups.

Do NOT change MANUSCRIPT_STYLE, COLUMN_WIDTH, TEXT_WIDTH, MODEL_DISPLAY_NAMES, PARAM_DISPLAY_NAMES, or apply_manuscript_style(). Only change GROUP_COLORS and add the two new dicts.
  </action>
  <verify>
Run: `python -c "import sys; sys.path.insert(0, 'manuscript/figures'); from plot_utils import GROUP_COLORS, GROUP_SHORT_LABELS, SHORT_NAME_TO_KEY; print('GROUP_COLORS:', GROUP_COLORS); print('SHORT_LABELS:', GROUP_SHORT_LABELS); print('SHORT_NAME_TO_KEY:', SHORT_NAME_TO_KEY)"`
Expect: Two-group color dict, short labels, and 7-entry reverse lookup printed without errors.
  </verify>
  <done>plot_utils.py exports GROUP_COLORS with 2 actual group names, GROUP_SHORT_LABELS for axis labels, and SHORT_NAME_TO_KEY for comparison_results.csv mapping.</done>
</task>

<task type="auto">
  <name>Task 2: Rewrite paper.qmd setup cell and all code cells to be fully data-driven</name>
  <files>manuscript/paper.qmd</files>
  <action>
Rewrite ALL Python code cells in manuscript/paper.qmd to derive everything from data. The prose sections need only surgical inline-Python fixes (no rewriting). Work through these sections in order:

**A. Setup cell (lines 35-84) -- COMPLETE REWRITE:**

Replace the entire setup cell with one that:

1. Adds BOTH `manuscript/figures` AND the project root to sys.path:
   ```python
   sys.path.insert(0, str(Path("figures")))
   sys.path.insert(0, str(Path("..")))
   ```

2. Imports from plot_utils: add GROUP_SHORT_LABELS and SHORT_NAME_TO_KEY to existing imports.

3. Imports from config: `from config import MODEL_REGISTRY, CHOICE_ONLY_MODELS, MIN_TRIALS_THRESHOLD`

4. Defines paths:
   ```python
   MLE_DIR = Path("../output/mle")
   COMPARISON_DIR = Path("../output/model_comparison")
   GROUPS_DIR = Path("../output/trauma_groups")
   FIGURES_DIR = Path("../figures")
   OUTPUT_DIR = Path("outputs")
   OUTPUT_DIR.mkdir(exist_ok=True)
   ```

5. Loads comparison_results.csv to determine winner programmatically:
   ```python
   df_comparison = pd.read_csv(COMPARISON_DIR / "comparison_results.csv")
   # Winner is first row (lowest aggregate_aic). Column "model" has short names (M6b, M5...)
   winner_short = df_comparison.iloc[0]["model"]
   winning_model = SHORT_NAME_TO_KEY[winner_short]
   winner_display = MODEL_DISPLAY_NAMES[winning_model]
   winner_n_params = MODEL_REGISTRY[winning_model]["n_params"]
   winner_params = MODEL_REGISTRY[winning_model]["params"]
   
   # dAIC vs second-best
   if len(df_comparison) >= 2:
       daic_vs_second = df_comparison.iloc[1]["delta_aic"] - df_comparison.iloc[0]["delta_aic"]
   else:
       daic_vs_second = float("nan")
   # Also store dBIC
   if len(df_comparison) >= 2:
       dbic_vs_second = df_comparison.iloc[1]["delta_bic"] - df_comparison.iloc[0]["delta_bic"]
   else:
       dbic_vs_second = float("nan")
   ```

   IMPORTANT: daic_vs_second should be the absolute difference (second_best delta_aic which equals the gap since winner's delta_aic=0). Simplify to:
   ```python
   daic_vs_second = df_comparison.iloc[1]["delta_aic"]
   dbic_vs_second = df_comparison.iloc[1]["delta_bic"]
   ```

6. Loads group_assignments.csv for group info:
   ```python
   df_groups = pd.read_csv(GROUPS_DIR / "group_assignments.csv")
   group_col = "hypothesis_group"
   groups = sorted(df_groups[group_col].unique())
   group_counts = df_groups[group_col].value_counts().to_dict()
   ```

7. Loads individual fits for the winning model:
   ```python
   fits_winner_path = MLE_DIR / MODEL_REGISTRY[winning_model]["csv_filename"]
   df_winner = pd.read_csv(fits_winner_path)
   n_participants = len(df_winner)
   ```

8. Loads all model fits (for model comparison cell):
   ```python
   fits = {}
   for model_key in MODEL_REGISTRY:
       fpath = MLE_DIR / MODEL_REGISTRY[model_key]["csv_filename"]
       if fpath.exists():
           fits[model_key] = pd.read_csv(fpath)
   n_models = len(fits)
   ```

9. Sets fitting parameters as variables:
   ```python
   n_starts = 50
   optimizer_name = "L-BFGS-B"
   min_trials = MIN_TRIALS_THRESHOLD  # from config.py
   ```

**B. Model comparison table (lines 203-243) -- REWRITE cell body:**

Replace the entire try block. Read from comparison_results.csv directly instead of performance_summary.json:

```python
try:
    # Filter to choice-only models
    choice_only_short = [MODEL_REGISTRY[k]["short_name"] for k in CHOICE_ONLY_MODELS]
    df_cmp = df_comparison[df_comparison["model"].isin(choice_only_short)].copy()

    # Map short names to display names
    short_to_display = {MODEL_REGISTRY[k]["short_name"]: MODEL_REGISTRY[k]["display_name"] for k in MODEL_REGISTRY}
    short_to_nparams = {MODEL_REGISTRY[k]["short_name"]: MODEL_REGISTRY[k]["n_params"] for k in MODEL_REGISTRY}

    df_cmp["Model"] = df_cmp["model"].map(short_to_display)
    df_cmp["k"] = df_cmp["model"].map(short_to_nparams)
    df_cmp["Aggregate AIC"] = df_cmp["aggregate_aic"].round(1)
    df_cmp["Aggregate BIC"] = df_cmp["aggregate_bic"].round(1)
    df_cmp["dAIC"] = df_cmp["delta_aic"].round(1)

    df_display = df_cmp[["Model", "k", "Aggregate AIC", "Aggregate BIC", "dAIC"]].reset_index(drop=True)

    from IPython.display import display
    display(df_display.style.hide(axis="index"))
except Exception as e:
    print(f"Data not yet available. Run scripts/14_compare_models.py first. ({e})")
```

**C. Winning model section header and text (lines 245-256):**

Change section header from `### Winning Model: M5` to:
```
### Winning Model: `{python} winner_display` {#sec-winning-model}
```

Replace the hardcoded paragraph (lines 247-256) with inline Python references. The paragraph should become:
```
The winning model among choice-only models was `{python} winner_display`
(`{python} winner_n_params` free parameters), which beat the second-best model by
$\Delta\text{AIC} = $ `{python} f"{daic_vs_second:.1f}"` and
$\Delta\text{BIC} = $ `{python} f"{dbic_vs_second:.1f}"`, constituting
very strong evidence [@daw2011model].
```

Remove the M5-specific description paragraph (lines 253-256 about phi_RL). Replace with a generic statement that can work for any winner:
```
See @sec-model-table for the complete parameter listing of `{python} winner_display`.
```

**D. Parameter-by-group violin plot (lines 260-318) -- REWRITE cell body:**

Fix the broken cell. Key changes:
- Merge df_winner with df_groups on participant_id/sona_id (the group_assignments uses "sona_id", individual_fits uses "participant_id" -- these are the same IDs)
- Get param_cols from MODEL_REGISTRY[winning_model]["params"]
- Use actual group names from data (groups variable from setup)
- Use GROUP_COLORS and GROUP_SHORT_LABELS for coloring and tick labels

Replace the try block:
```python
try:
    # Merge fits with group assignments
    df_plot = df_winner.merge(
        df_groups[["sona_id", group_col]],
        left_on="participant_id",
        right_on="sona_id",
        how="inner",
    )

    param_cols = [p for p in winner_params if p in df_plot.columns]
    n_params = len(param_cols)
    ncols = 4
    nrows = (n_params + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(TEXT_WIDTH, nrows * 1.8))
    axes = axes.flatten()

    for i, param in enumerate(param_cols):
        ax = axes[i]
        group_data = [df_plot.loc[df_plot[group_col] == g, param].dropna().values for g in groups]
        parts = ax.violinplot(
            group_data,
            positions=range(len(groups)),
            showmedians=True,
            showextrema=False,
        )
        for j, (body, g) in enumerate(zip(parts["bodies"], groups)):
            body.set_facecolor(GROUP_COLORS[g])
            body.set_alpha(0.7)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.5)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([GROUP_SHORT_LABELS.get(g, g) for g in groups], fontsize=7)
        ax.set_ylabel(PARAM_DISPLAY_NAMES.get(param, param))
        ax.set_title(PARAM_DISPLAY_NAMES.get(param, param))

    for j in range(n_params, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig_parameters_by_group.pdf", bbox_inches="tight")
    plt.show()
except Exception as e:
    print(f"Data not yet available. Run scripts/15_analyze_mle_by_trauma.py first. ({e})")
```

Update the fig-cap to use inline Python -- Quarto fig-cap cannot contain inline Python, so use a generic caption:
```
#| fig-cap: "Model parameters by trauma group. Violins show kernel density estimates; central marks are medians."
```

**E. Correlation heatmap (lines 322-344) -- Fix path:**

Replace line 329:
```python
corr_path = Path(f"../figures/mle_trauma_analysis/correlation_heatmap_{winning_model}.png")
```

Update the tbl-cap similarly -- make it generic (no "M5"):
```
#| fig-cap: "Spearman correlations between winning model parameters and IES-R subscale scores (Intrusion, Avoidance, Hyperarousal). Color encodes correlation strength; asterisks indicate Bonferroni-corrected significance (p < 0.05)."
```

**F. Regression table (lines 348-376) -- Fix path:**

Replace line 354:
```python
reg_dir = Path(f"../output/regressions/{winning_model}")
```

Update tbl-cap to be generic (replace "M5" with "winning model"):
```
#| tbl-cap: "OLS regression of winning model parameters on IES-R subscales (Intrusion, Avoidance, Hyperarousal), controlling for age and sex. Standardized beta coefficients shown. Bold: p < 0.05 after Bonferroni correction."
```

**G. Methods text inline fixes (scattered through lines 120-196):**

Line 125-126 (participants section): `{python} 400` should become `{python} min_trials`

Line 129-133 (group descriptions): Replace the three-group description:
```
Participants were classified into two groups based on LEC-5 endorsement
and IES-R total score:
**`{python} groups[0]`** (n = `{python} group_counts[groups[0]]`)
and **`{python} groups[1]`** (n = `{python} group_counts[groups[1]]`).
```

Line 179: Replace "10 random restarts" with:
```
`{python} n_starts` random restarts per participant
```

**H. Conclusion inline fixes (lines 411-419):**

Line 417: Replace "(M5: WM-RL + $\phi_{\mathrm{RL}}$)" with:
```
(`{python} winner_display`)
```

**I. Appendix model table (lines 432-470) -- Derive from MODEL_REGISTRY:**

Replace the hardcoded dict with code that builds it from MODEL_REGISTRY:

```python
try:
    rows = []
    # Order: M1, M2, M3, M5, M6a, M6b, M4 (choice-only first, then M4)
    model_order = ["qlearning", "wmrl", "wmrl_m3", "wmrl_m5", "wmrl_m6a", "wmrl_m6b", "wmrl_m4"]
    for key in model_order:
        info = MODEL_REGISTRY[key]
        rows.append({
            "Model": info["display_name"],
            "Free Parameters": ", ".join(PARAM_DISPLAY_NAMES.get(p, p) for p in info["params"]),
            "k": info["n_params"],
            "Type": "Choice+RT" if not info["is_choice_only"] else "Choice only",
        })

    df_params = pd.DataFrame(rows)
    from IPython.display import display
    display(df_params.style.hide(axis="index"))
except Exception as e:
    print(f"Error building model table: {e}")
```

**J. Exclusion criteria section (lines 472-480):**

Line 475: Replace `{python} 400` with `{python} min_trials`

**IMPORTANT NOTES for implementation:**
- The comparison_results.csv `model` column uses SHORT names (M1, M2, M3, M5, M6a, M6b) -- NOT internal keys. Use SHORT_NAME_TO_KEY to convert.
- The individual_fits CSVs use `participant_id`; group_assignments.csv uses `sona_id`. These are the same IDs -- merge on them.
- The individual_fits CSVs use `capacity` (not `K`) as the column name for WM capacity. MODEL_REGISTRY also uses `capacity`.
- PARAM_DISPLAY_NAMES has `"K"` as the key for capacity display name, but the CSV column is `capacity`. Add `"capacity": r"$K$"` to PARAM_DISPLAY_NAMES in plot_utils.py (Task 1).
- Do NOT change any prose that doesn't contain hardcoded model references. Leave Discussion placeholders as-is.
- Keep all YAML frontmatter, format settings, and LaTeX preamble unchanged.
- Keep `from __future__ import annotations` -- wait, this is a .qmd file not a .py module. Do NOT add `from __future__ import annotations` to the Python cells (it would break inline expressions).
  </action>
  <verify>
1. Grep for hardcoded "wmrl_m5" in paper.qmd (should find ZERO outside of model_order list in appendix):
   `grep -n "wmrl_m5" manuscript/paper.qmd`

2. Grep for hardcoded group names (should find ZERO instances of "control", "exposed", "symptomatic"):
   `grep -n '"control"\|"exposed"\|"symptomatic"' manuscript/paper.qmd`

3. Grep for "10 random" (should find ZERO):
   `grep -n "10 random" manuscript/paper.qmd`

4. Grep for "mean_aic" from old performance_summary.json approach (should find ZERO):
   `grep -n "mean_aic\|mean_bic" manuscript/paper.qmd`

5. Check the setup cell sets winning_model from comparison_results.csv:
   `grep -n "comparison_results" manuscript/paper.qmd`

6. Verify MODEL_REGISTRY import:
   `grep -n "MODEL_REGISTRY" manuscript/paper.qmd`
  </verify>
  <done>
- winning_model determined programmatically from comparison_results.csv (currently resolves to wmrl_m6b)
- All inline references use variables (winner_display, n_participants, n_starts, min_trials, groups, group_counts)
- Model comparison table reads from comparison_results.csv directly
- Violin plot merges fits with group_assignments.csv using actual group names
- Correlation heatmap and regression paths use f-string with winning_model
- Appendix model table derived from MODEL_REGISTRY
- Zero hardcoded model names, group names, or fitting parameters remain
  </done>
</task>

</tasks>

<verification>
1. `grep -c "wmrl_m5" manuscript/paper.qmd` should return a small number (only in model_order list in appendix and CHOICE_ONLY_MODELS reference)
2. `grep -c '"control"\|"exposed"\|"symptomatic"' manuscript/paper.qmd` should return 0
3. `grep -c "10 random" manuscript/paper.qmd` should return 0
4. `grep -c "comparison_results" manuscript/paper.qmd` should return >= 1
5. `grep -c "MODEL_REGISTRY" manuscript/paper.qmd` should return >= 1
6. `grep -c "group_assignments" manuscript/paper.qmd` should return >= 1
7. `python -c "import sys; sys.path.insert(0, 'manuscript/figures'); from plot_utils import GROUP_COLORS, GROUP_SHORT_LABELS, SHORT_NAME_TO_KEY; assert len(GROUP_COLORS) == 2; assert len(SHORT_NAME_TO_KEY) == 7; print('OK')"` prints OK
</verification>

<success_criteria>
- The paper is fully data-driven: changing comparison_results.csv winner propagates everywhere
- No hardcoded model names in prose (only in model_order list and MODEL_REGISTRY references)
- No hardcoded group names ("control"/"exposed"/"symptomatic" replaced with data-driven names)
- Model comparison table reads from comparison_results.csv (not performance_summary.json)
- Violin plot correctly merges fits with group_assignments.csv
- All figure/regression paths use f-string with winning_model variable
- Methods text shows n_starts=50 (not 10), correct group count (2, not 3)
</success_criteria>

<output>
After completion, create `.planning/quick/003-quarto-softcoded-winning-model/003-SUMMARY.md`
</output>
