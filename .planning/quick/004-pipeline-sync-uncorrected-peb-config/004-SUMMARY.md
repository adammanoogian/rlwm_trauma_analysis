---
phase: quick-004
plan: 01
subsystem: pipeline-sync
tags: [data-loading, manuscript, bayesian, survey-data, p-values]
requires: []
provides:
  - scripts 15/16 load survey data from summary_participant_metrics.csv (169 rows)
  - manuscript narrative leads with uncorrected p-values, corrections as sensitivity
  - Bayesian script 13 accepts all 7 MODEL_REGISTRY model keys
affects:
  - downstream analysis N counts (now ~154 instead of 21-49)
  - manuscript compile output
tech-stack:
  added: []
  patterns:
    - rename-at-load: translate column names from data file conventions to script conventions at read time
key-files:
  created: []
  modified:
    - scripts/15_analyze_mle_by_trauma.py
    - scripts/16_regress_parameters_on_scales.py
    - manuscript/paper.qmd
    - scripts/fitting/fit_bayesian.py
    - scripts/13_fit_bayesian.py
decisions:
  - "Survey data source: summary_participant_metrics.csv is authoritative (169 rows); participant_surveys.csv is stale (49 rows)"
  - "Column rename at load time: less_* -> lec_* keeps all downstream references valid without further changes"
  - "Script 15 uses lec_total/lec_personal (no _events suffix); script 16 uses lec_total_events/lec_personal_events (with _events suffix)"
  - "Uncorrected alpha=0.05 is primary reporting threshold; Bonferroni/FWE/FDR reported as sensitivity analyses"
  - "Bayesian unimplemented models raise NotImplementedError with message directing to MLE script"
metrics:
  duration: 12 min
  completed: 2026-04-07
---

# Quick Task 004: Pipeline Sync — Uncorrected P-values, Survey Data Fix, Bayesian Config Summary

Fix stale survey data source in scripts 15/16, update manuscript narrative to lead with uncorrected p-values, add MODEL_REGISTRY support to Bayesian fitting.

## What Was Done

### Task 1: Fix survey data source in script 15

`scripts/15_analyze_mle_by_trauma.py` `load_data()` previously loaded from
`output/mle/participant_surveys.csv` (49 rows — a stale early-pipeline file).
Changed to `output/summary_participant_metrics.csv` (169 rows, complete dataset).
Added column rename at load time:

```python
surveys = surveys.rename(columns={
    'less_total_events': 'lec_total',
    'less_personal_events': 'lec_personal',
})
```

This keeps all downstream references to `TRAUMA_PREDICTORS = ['lec_total', 'lec_personal', ...]`
valid without modifying any other code in the script.

### Task 2: Fix survey data source in script 16

`scripts/16_regress_parameters_on_scales.py` `load_integrated_data()` previously
attempted `output/mle/participant_surveys.csv` with fallback to `summary_participant_metrics_all.csv`.
Replaced with single unconditional load from `output/summary_participant_metrics.csv`.
Column rename maps `less_*` to `lec_*_events` (with `_events` suffix, matching script 16's
internal convention in `merge_cols`):

```python
rename_map = {
    'less_total_events': 'lec_total_events',
    'less_personal_events': 'lec_personal_events',
}
```

### Task 3: Update manuscript narrative for uncorrected p-values

Updated `manuscript/paper.qmd` across five locations:

1. **Statistical Analysis section**: Reframed from "Bonferroni corrected" to
   "uncorrected alpha=0.05; corrections reported as sensitivity".
2. **Group comparisons method text**: Removed corrected alpha calculation; now
   mentions Bonferroni only as sensitivity.
3. **Group results text**: Changed "No parameter showed significant difference
   after Bonferroni correction" to "At the uncorrected level, no parameter showed
   significant difference. Results remained non-significant after Bonferroni correction."
4. **Correlations section**: "After FWE correction, no correlations reached
   significance" changed to "reported at uncorrected thresholds; FWE for sensitivity;
   no correlations reached the uncorrected significance threshold".
5. **Regression section**: "after FDR correction (all q > .05)" changed to
   "all p > .05; FDR-corrected q-values confirmed".

The inline Python expressions that read `p_uncorrected`, `p_bonferroni`, `p_fwe`
from CSV files were not changed — the data already contains both corrected and
uncorrected columns.

### Task 4: MODEL_REGISTRY support for Bayesian fitting

Updated `scripts/fitting/fit_bayesian.py`:
- Added `from __future__ import annotations`
- Added `ALL_MODELS` to import from config
- Changed argparse `choices=['qlearning', 'wmrl']` to `choices=ALL_MODELS`
- Added `BAYESIAN_IMPLEMENTED` guard in `fit_model()`:
  ```python
  BAYESIAN_IMPLEMENTED = {'qlearning', 'wmrl'}
  if model not in BAYESIAN_IMPLEMENTED:
      raise NotImplementedError(
          f"Bayesian fitting for '{model}' is not yet implemented. ..."
      )
  ```

Updated `scripts/13_fit_bayesian.py` docstring to list all 7 models with
their Bayesian implementation status.

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| summary_participant_metrics.csv as survey source | 169 rows vs 49 in stale participant_surveys.csv; merger produces ~154 (matching fitted participants) |
| Column rename at load time | Keeps downstream code unchanged; single authoritative rename point |
| Script 15 vs 16 column suffix convention | Script 15 uses lec_total (no _events); script 16 uses lec_total_events (with _events) — each script's internal convention preserved |
| Uncorrected as primary reporting | Standard in computational psychiatry; corrections as sensitivity reduces false negatives in exploratory analysis |
| NotImplementedError for unimplemented Bayesian | Cleaner than silent wrong output; message directs to MLE fallback |

## Deviations from Plan

None — plan executed exactly as written.

## Next Phase Readiness

- Scripts 15/16 will now produce N~154 in downstream analyses (was 21-49)
- Manuscript can be recompiled with corrected N counts
- Bayesian script ready for future M3+ implementation (guard in place)
