---
phase: quick-003
plan: 01
subsystem: manuscript
tags: [quarto, matplotlib, pandas, model-comparison, data-driven]

# Dependency graph
requires:
  - phase: quick-001
    provides: Quarto manuscript scaffold (paper.qmd, plot_utils.py)
  - phase: quick-002
    provides: MODEL_REGISTRY in config.py, comparison_results.csv from script 14
provides:
  - Fully data-driven paper.qmd (winning model determined from comparison_results.csv)
  - plot_utils.py with actual group names from group_assignments.csv
affects: [manuscript rendering, any future manuscript updates]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Setup cell imports config.py MODEL_REGISTRY for all model metadata
    - SHORT_NAME_TO_KEY lookup converts comparison_results.csv short names to internal keys
    - GROUP_SHORT_LABELS for compact axis tick labels from full group names

key-files:
  created: []
  modified:
    - manuscript/paper.qmd
    - manuscript/figures/plot_utils.py

key-decisions:
  - "Group names keyed by full hypothesis_group string (not short aliases) to match data column exactly"
  - "capacity entry added to PARAM_DISPLAY_NAMES (CSV column name for WM capacity, separate from K)"
  - "SHORT_NAME_TO_KEY lives in plot_utils.py (alongside GROUP_COLORS) not in paper.qmd directly"
  - "daic_vs_second = df_comparison.iloc[1]['delta_aic'] (second row's delta since winner is 0)"

patterns-established:
  - "comparison_results.csv row 0 = winning model; SHORT_NAME_TO_KEY converts to internal key"
  - "Violin plots merge individual_fits (participant_id) with group_assignments (sona_id) on id columns"

# Metrics
duration: 15min
completed: 2026-04-07
---

# Quick Task 003: Quarto Softcoded Winning Model Summary

**Manuscript paper.qmd fully data-driven: winning model, group names, n_starts, and model table all derive from comparison_results.csv, config.py MODEL_REGISTRY, and group_assignments.csv**

## Performance

- **Duration:** ~15 min
- **Started:** 2026-04-07
- **Completed:** 2026-04-07
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Replaced hardcoded `winning_model = "wmrl_m5"` with programmatic lookup from comparison_results.csv (currently resolves to wmrl_m6b)
- Fixed group names: replaced "control"/"exposed"/"symptomatic" with actual `hypothesis_group` values from group_assignments.csv
- Fixed model comparison table: reads comparison_results.csv directly instead of broken performance_summary.json approach
- Fixed violin plot: merges individual_fits with group_assignments on participant_id/sona_id
- Fixed "10 random restarts" to use `n_starts = 50` variable
- Replaced hardcoded appendix model table dict with MODEL_REGISTRY-derived code
- Added GROUP_SHORT_LABELS, SHORT_NAME_TO_KEY, and `capacity` to PARAM_DISPLAY_NAMES in plot_utils.py

## Task Commits

1. **Task 1: Update plot_utils.py group colors and add reverse-lookup helper** - `231f75a` (feat)
2. **Task 2: Rewrite paper.qmd setup cell and all code cells to be fully data-driven** - `d7ea897` (feat)

**Plan metadata:** (see final commit below)

## Files Created/Modified

- `manuscript/figures/plot_utils.py` - Replaced GROUP_COLORS keys with actual group names, added GROUP_SHORT_LABELS and SHORT_NAME_TO_KEY dicts, added "capacity" to PARAM_DISPLAY_NAMES
- `manuscript/paper.qmd` - Rewrote setup cell, model comparison table, winning model section, violin plot, correlation heatmap path, regression path, conclusion, and appendix model table to be fully data-driven

## Decisions Made

- GROUP_COLORS keys use full `hypothesis_group` strings (not short aliases) to match data column exactly without any translation layer
- `capacity` added to PARAM_DISPLAY_NAMES alongside `K` since MODEL_REGISTRY uses `capacity` as the param name but old PARAM_DISPLAY_NAMES only had `K`
- SHORT_NAME_TO_KEY lives in plot_utils.py (not paper.qmd inline) so it is available to any future figure scripts
- dAIC vs second-best taken directly as `df_comparison.iloc[1]["delta_aic"]` since winner's delta_aic = 0 by construction

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing Critical] Added `capacity` entry to PARAM_DISPLAY_NAMES**
- **Found during:** Task 1 review of plan notes
- **Issue:** MODEL_REGISTRY uses `capacity` as param name for WM capacity, but PARAM_DISPLAY_NAMES only had `K`. Violin plot cell uses `PARAM_DISPLAY_NAMES.get(param, param)` so `capacity` would display as raw string without this fix.
- **Fix:** Added `"capacity": r"$K$"` entry to PARAM_DISPLAY_NAMES in plot_utils.py
- **Files modified:** manuscript/figures/plot_utils.py
- **Verification:** Import check passes; both `K` and `capacity` map to `$K$`
- **Committed in:** 231f75a (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 missing critical)
**Impact on plan:** Essential for correct parameter display in violin plot. No scope creep.

## Issues Encountered

None.

## Next Phase Readiness

- Manuscript will auto-update when comparison_results.csv changes (e.g., after cluster re-fit)
- Violin plot will work once cluster re-fit completes and wmrl_m6b_individual_fits.csv exists
- Regression and correlation heatmap paths use winning_model f-string, will resolve to wmrl_m6b paths

---
*Phase: quick-003*
*Completed: 2026-04-07*
