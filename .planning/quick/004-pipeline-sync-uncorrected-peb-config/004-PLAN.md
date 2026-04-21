---
phase: quick-004
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/15_analyze_mle_by_trauma.py
  - scripts/16_regress_parameters_on_scales.py
  - manuscript/paper.qmd
  - scripts/fitting/fit_bayesian.py
  - scripts/13_fit_bayesian.py
autonomous: true

must_haves:
  truths:
    - "Script 15 loads survey data from summary_participant_metrics.csv (169 rows), not participant_surveys.csv (49 rows)"
    - "Script 16 loads survey data from summary_participant_metrics.csv as primary source"
    - "After merge, scripts 15/16 produce N matching the number of fitted participants (approx 154), not 21-49"
    - "Output CSVs still contain both uncorrected and corrected p-value columns"
    - "Paper.qmd narrative leads with uncorrected p-values and frames corrections as sensitivity analysis"
    - "Script 13 argparse accepts all 7 MODEL_REGISTRY model keys"
  artifacts:
    - path: "scripts/15_analyze_mle_by_trauma.py"
      provides: "Survey data loaded from summary_participant_metrics.csv with column rename"
      contains: "summary_participant_metrics.csv"
    - path: "scripts/16_regress_parameters_on_scales.py"
      provides: "Survey data loaded from summary_participant_metrics.csv as primary"
      contains: "summary_participant_metrics.csv"
    - path: "manuscript/paper.qmd"
      provides: "Narrative text leading with uncorrected stats"
      contains: "uncorrected"
    - path: "scripts/fitting/fit_bayesian.py"
      provides: "MODEL_REGISTRY-based model choices in argparse"
      contains: "MODEL_REGISTRY"
  key_links:
    - from: "scripts/15_analyze_mle_by_trauma.py"
      to: "output/summary_participant_metrics.csv"
      via: "pd.read_csv in load_data()"
      pattern: "summary_participant_metrics\\.csv"
    - from: "scripts/16_regress_parameters_on_scales.py"
      to: "output/summary_participant_metrics.csv"
      via: "pd.read_csv in load_data()"
      pattern: "summary_participant_metrics\\.csv"
    - from: "scripts/fitting/fit_bayesian.py"
      to: "config.py MODEL_REGISTRY"
      via: "import and argparse choices"
      pattern: "MODEL_REGISTRY"
---

<objective>
Fix stale survey data source in scripts 15/16 (participant_surveys.csv -> summary_participant_metrics.csv), update manuscript narrative to lead with uncorrected p-values, and add MODEL_REGISTRY support to Bayesian fitting script 13.

Purpose: The stale data source causes N to drop from ~154 to ~21-49 in downstream analyses, invalidating results. The manuscript should report uncorrected stats as primary (standard in computational psychiatry). The Bayesian argparse should accept all 7 models even though only M1/M2 have full implementations.
Output: Updated scripts 15, 16, paper.qmd, fit_bayesian.py, 13_fit_bayesian.py
</objective>

<execution_context>
@C:\Users\aman0087\.claude/get-shit-done/workflows/execute-plan.md
@C:\Users\aman0087\.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/STATE.md
@config.py (MODEL_REGISTRY, ALL_MODELS, CHOICE_ONLY_MODELS)
@scripts/15_analyze_mle_by_trauma.py
@scripts/16_regress_parameters_on_scales.py
@scripts/fitting/fit_bayesian.py
@scripts/13_fit_bayesian.py
@manuscript/paper.qmd
</context>

<tasks>

<task type="auto">
  <name>Task 1: Fix survey data source in script 15</name>
  <files>scripts/15_analyze_mle_by_trauma.py</files>
  <action>
In `load_data()` (line ~134), change the survey data source:

BEFORE:
```python
surveys = pd.read_csv(OUTPUT_DIR / "participant_surveys.csv")
```

AFTER:
```python
surveys = pd.read_csv(PROJECT_ROOT / "output" / "summary_participant_metrics.csv")
```

Then add column rename right after loading to map the new column names to the names the rest of the script expects:
```python
surveys = surveys.rename(columns={
    'less_total_events': 'lec_total',
    'less_personal_events': 'lec_personal',
})
```

This is critical: the rest of the script uses `TRAUMA_PREDICTORS = ['lec_total', 'lec_personal', 'ies_intrusion', 'ies_avoidance', 'ies_hyperarousal']` and `ols_regression_extended` also references `'lec_total'`. By renaming at load time, all downstream references remain valid.

The `ies_total`, `ies_intrusion`, `ies_avoidance`, `ies_hyperarousal` columns already have matching names in both files — no rename needed for those.

Also add a print statement after loading: `print(f"Loaded survey data: {len(surveys)} participants from summary_participant_metrics.csv")`
  </action>
  <verify>
Run: `python scripts/15_analyze_mle_by_trauma.py --model wmrl_m5 --dry-run 2>&1 | head -20` (or if no --dry-run, just run it and check the participant count lines in output). The merge should report ~154 participants, not 49 or 21. If full run is too slow, add a temporary early return after load_data() prints, verify N, then remove it.

Alternative quick check: `python -c "import pandas as pd; df = pd.read_csv('output/summary_participant_metrics.csv'); print(len(df), 'rows'); print([c for c in df.columns if 'less' in c or 'ies' in c or 'lec' in c])"`
  </verify>
  <done>Script 15 load_data() reads from summary_participant_metrics.csv, renames less_* columns to lec_*, and the inner merge with MLE fits produces N >= 100 (approximately 154).</done>
</task>

<task type="auto">
  <name>Task 2: Fix survey data source in script 16</name>
  <files>scripts/16_regress_parameters_on_scales.py</files>
  <action>
In the `load_data()` function (lines ~187-199), replace the primary survey loading logic:

BEFORE (lines 187-199):
```python
surveys_path = Path('output/mle/participant_surveys.csv')
if surveys_path.exists():
    participant_data = pd.read_csv(surveys_path)
    rename_map = {
        'lec_total': 'lec_total_events',
        'lec_personal': 'lec_personal_events'
    }
    participant_data = participant_data.rename(columns=rename_map)
else:
    participant_data = pd.read_csv('output/summary_participant_metrics_all.csv')
```

AFTER:
```python
participant_data = pd.read_csv(Path('output/summary_participant_metrics.csv'))
# Rename columns to match expected names used throughout this script
rename_map = {
    'less_total_events': 'lec_total_events',
    'less_personal_events': 'lec_personal_events',
}
participant_data = participant_data.rename(columns=rename_map)
print(f"  Loaded survey data: {len(participant_data)} participants from summary_participant_metrics.csv")
```

Note the difference from script 15: script 16 uses `lec_total_events` / `lec_personal_events` (with _events suffix) throughout, while script 15 uses `lec_total` / `lec_personal` (without _events suffix). Match each script's internal convention.

Remove the fallback to `summary_participant_metrics_all.csv` — it is no longer needed.
  </action>
  <verify>
Run: `python scripts/16_regress_parameters_on_scales.py --model wmrl_m5 2>&1 | head -30` and verify the participant count line says ~154, not 21 or 49.
  </verify>
  <done>Script 16 load_data() reads from summary_participant_metrics.csv as sole source, renames less_* to lec_*_events, merge produces N >= 100.</done>
</task>

<task type="auto">
  <name>Task 3: Update manuscript narrative for uncorrected p-values as primary</name>
  <files>manuscript/paper.qmd</files>
  <action>
The output CSVs already contain both uncorrected and corrected columns — no script changes needed for the data. The change is purely narrative in paper.qmd.

1. **Section "Parameter-Trauma Group Relationships" (line ~346):** Change the sentence describing the method from "Mann-Whitney U tests with Bonferroni correction" to "Mann-Whitney U tests (uncorrected; Bonferroni-corrected p-values reported for sensitivity)". Remove the corrected alpha computation from the narrative text.

2. **Line ~423-434 results text:** Change from "No parameter showed a statistically significant group difference after Bonferroni correction" to report the uncorrected results as primary. Something like:
   "At the uncorrected level ($\alpha = .05$), [report any p < .05 if they exist, or 'no parameter reached significance']. These results remained non-significant after Bonferroni correction for `{python} winner_n_params` comparisons (@tbl-group-comparisons)."

3. **Section "Continuous Trauma Associations" (line ~438-441):** Change "After family-wise error correction, no correlations reached significance" to: "Spearman rank correlations between winning-model parameters and continuous trauma measures are reported at uncorrected thresholds, with family-wise error (FWE) corrected p-values for sensitivity."

4. **Line ~500-501:** Change "This finding did not survive correction for multiple comparisons" to something more neutral like "This association did not survive FWE correction ($p_{\text{FWE}} = $ ...)."

5. **Section "Regression Analyses" (line ~531-535):** Change "after FDR correction (all $q > .05$)" to report uncorrected results as primary and FDR as sensitivity.

6. **In @tbl-group-comparisons (lines ~408-409):** The table already shows both p and p(corrected) — keep as-is. The p column is already the uncorrected value. No change needed to the table itself.

Keep all the inline Python expressions that read from data — they are already correct and reference both `p_uncorrected` and `p_bonferroni`/`p_fwe`.
  </action>
  <verify>
Read through the changed sections in paper.qmd to confirm: (a) uncorrected is described as primary in all three analysis sections, (b) corrected is framed as sensitivity, (c) no Python code was broken, (d) all inline expressions still valid.
  </verify>
  <done>Manuscript narrative reports uncorrected p-values as primary statistical test across group comparisons, correlations, and regressions, with corrections noted as sensitivity analyses.</done>
</task>

<task type="auto">
  <name>Task 4: Add MODEL_REGISTRY support to Bayesian fitting scripts</name>
  <files>scripts/fitting/fit_bayesian.py, scripts/13_fit_bayesian.py</files>
  <action>
**In scripts/fitting/fit_bayesian.py:**

1. Add import at top: `from config import MODEL_REGISTRY, ALL_MODELS`

2. In `main()` argparse (line ~291), change:
   ```python
   parser.add_argument('--model', type=str, required=True,
                       choices=['qlearning', 'wmrl'],
                       help='Model to fit')
   ```
   to:
   ```python
   parser.add_argument('--model', type=str, required=True,
                       choices=ALL_MODELS,
                       help='Model to fit (only qlearning and wmrl have full Bayesian implementations)')
   ```

3. In `fit_model()` (line ~146-148), extend the model dispatch to handle unimplemented models gracefully:
   ```python
   BAYESIAN_IMPLEMENTED = {'qlearning', 'wmrl'}
   if model not in BAYESIAN_IMPLEMENTED:
       raise NotImplementedError(
           f"Bayesian fitting for '{model}' is not yet implemented. "
           f"Implemented models: {sorted(BAYESIAN_IMPLEMENTED)}. "
           f"Use scripts/12_fit_mle.py for MLE fitting of this model."
       )
   model_name = "Q-LEARNING" if model == 'qlearning' else "WM-RL"
   model_fn = qlearning_hierarchical_model if model == 'qlearning' else wmrl_hierarchical_model
   ```

**In scripts/13_fit_bayesian.py:**

4. Update the docstring "Models Available" section to list all 7 models and note which have full Bayesian implementations:
   ```
   Models Available:
       - qlearning: Q-learning (M1) [full Bayesian]
       - wmrl: WM-RL (M2) [full Bayesian]
       - wmrl_m3: WM-RL+kappa (M3) [MLE only — Bayesian not yet implemented]
       - wmrl_m5: WM-RL+phi_rl (M5) [MLE only]
       - wmrl_m6a: WM-RL+kappa_s (M6a) [MLE only]
       - wmrl_m6b: WM-RL+dual (M6b) [MLE only]
       - wmrl_m4: RLWM-LBA (M4) [MLE only]
   ```

Do NOT implement full NumPyro models for M3-M6b — that requires separate hierarchical prior definitions and is a much larger task. The argparse acceptance + clear NotImplementedError is sufficient for now.
  </action>
  <verify>
Run: `python scripts/13_fit_bayesian.py --help` and confirm all 7 models appear in --model choices.
Run: `python scripts/13_fit_bayesian.py --model wmrl_m5 --data output/task_trials_long.csv 2>&1 | head -5` and confirm it raises NotImplementedError with a helpful message.
  </verify>
  <done>Script 13 argparse accepts all 7 MODEL_REGISTRY models. Unimplemented models raise NotImplementedError with clear message directing to MLE. Docstring updated.</done>
</task>

</tasks>

<verification>
1. Script 15 loads from summary_participant_metrics.csv and reports N >= 100 after merge
2. Script 16 loads from summary_participant_metrics.csv and reports N >= 100 after merge  
3. Neither script references participant_surveys.csv
4. Output CSVs from scripts 15/16 still contain both corrected and uncorrected p-value columns
5. paper.qmd narrative leads with uncorrected stats in all three analysis sections
6. `python scripts/13_fit_bayesian.py --help` shows all 7 models
7. Unimplemented Bayesian models raise NotImplementedError
</verification>

<success_criteria>
- Scripts 15/16 use summary_participant_metrics.csv (169 rows) instead of participant_surveys.csv (49 rows)
- After merge with MLE fits, N is approximately 154 (not 21 or 49)
- Manuscript narrative reports uncorrected p-values as primary, corrections as sensitivity
- Bayesian script 13 accepts all MODEL_REGISTRY keys; unimplemented models fail with helpful message
- No regressions: existing output CSV column structure preserved (both corrected and uncorrected columns)
</success_criteria>

<output>
After completion, create `.planning/quick/004-pipeline-sync-uncorrected-peb-config/004-SUMMARY.md`
</output>
