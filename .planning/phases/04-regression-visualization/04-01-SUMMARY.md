---
phase: 04-regression-visualization
plan: 01
subsystem: visualization
tags: [matplotlib, seaborn, regression, plotting, M3, kappa, perseveration]

# Dependency graph
requires:
  - phase: 03-model-comparison
    provides: wmrl_m3_individual_fits.csv with M3 parameters
provides:
  - Shared plotting utility (plotting_utils.py) for color-by visualization
  - Script 15 support for all three models (M1, M2, M3) with --model flag
  - Color-by functionality for scatter/regression plots via --color-by flag
  - All plot types (violin, heatmap, forest, scatter) generated for all models
affects: [05-parameter-recovery, 07-publication-polish, 16_regress_parameters_on_scales]

# Tech tracking
tech-stack:
  added: [scripts/utils/plotting_utils.py]
  patterns: [shared color palette utility, argparse CLI flags, model config dict pattern]

key-files:
  created:
    - scripts/utils/plotting_utils.py
  modified:
    - scripts/15_analyze_mle_by_trauma.py

key-decisions:
  - "Use TRAUMA_GROUP_COLORS constant in plotting_utils matching Script 15's existing colors"
  - "color-by is visual overlay only - does not change which analyses run"
  - "Model-specific figure filenames (e.g., correlation_heatmap_wmrl_m3.png) to avoid overwrites"
  - "WM-RL+K display name for M3 model in plots and output"

patterns-established:
  - "Shared plotting utility pattern: get_color_palette() + add_colored_scatter() reusable across scripts"
  - "MODEL_CONFIG dict pattern for looping over multiple models in main()"
  - "Optional color_by parameter defaults to None (preserves legacy behavior)"

# Metrics
duration: 19min
completed: 2026-02-05
---

# Phase 04 Plan 01: Regression Visualization Foundation Summary

**Shared color-by plotting utility and Script 15 extended with M3 support, --model flag, --color-by flag, and all plot types for all models**

## Performance

- **Duration:** 19 min
- **Started:** 2026-02-05T20:33:58Z
- **Completed:** 2026-02-05T20:52:58Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments
- Created reusable plotting_utils.py with color palette generation and colored scatter plotting
- Extended Script 15 to support M3 (wmrl_m3) alongside M1 and M2
- Added --model CLI flag (qlearning | wmrl | wmrl_m3 | all) for selective analysis
- Added --color-by CLI flag for flexible scatter plot coloring by any categorical variable
- All plot types (violin, heatmap, forest, scatter) now generated for all three models

## Task Commits

Each task was committed atomically:

1. **Task 1: Create shared plotting utility (plotting_utils.py)** - `297a8ba` (feat)
2. **Task 2: Update Script 15 with M3, --model, --color-by, and full plot coverage** - `5e22f4b` (feat)

## Files Created/Modified
- `scripts/utils/plotting_utils.py` - Shared color palette generation (get_color_palette) and colored scatter plotting (add_colored_scatter) with TRAUMA_GROUP_COLORS constant
- `scripts/15_analyze_mle_by_trauma.py` - Added M3 support, argparse CLI (--model, --color-by), MODEL_CONFIG dict pattern, extended plotting functions with params/color_by parameters

## Decisions Made

**1. TRAUMA_GROUP_COLORS in plotting_utils**
- Defined in utility module to match Script 15's existing GROUP_COLORS
- Allows scripts to pass custom_colors=TRAUMA_GROUP_COLORS when color_by='hypothesis_group'
- Preserves existing trauma group color mapping

**2. color-by is visual overlay only**
- Does NOT change which statistical analyses run (group comparisons still use hypothesis_group)
- Only affects scatter/regression plot point coloring
- Defaults to None to preserve legacy behavior when flag not specified

**3. Model-specific figure filenames**
- Pattern: `{plot_type}_{model_key}.png` (e.g., correlation_heatmap_wmrl_m3.png)
- Prevents overwrites when running different models
- All three models get all plot types in --model all mode

**4. WM-RL+K display name for M3**
- Model name: 'WM-RL+K' (shows kappa addition clearly)
- File suffix: 'wmrl_m3' (matches MLE fit filename convention)
- Parameter name: kappa → κ (Greek letter in plots)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Installed missing matplotlib/seaborn packages**
- **Found during:** Task 1 verification (import test)
- **Issue:** System Python lacked matplotlib, seaborn, scipy, statsmodels, scikit-learn
- **Fix:** Ran pip install for all required visualization packages
- **Files modified:** None (system environment)
- **Verification:** Import succeeded without errors
- **Note:** This is expected in Windows environment without conda rlwm environment active

---

**Total deviations:** 1 auto-fixed (1 blocking - environment setup)
**Impact on plan:** Environment setup necessary for execution. No code changes required.

## Issues Encountered

**Environment activation:**
- rlwm conda environment not created on Windows system
- Used system Python 3.13 with pip-installed packages instead
- All scripts executed successfully with correct package versions
- Note: Cluster/Linux execution will use conda environment per environment.yml

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

**Ready for Plan 02 (Script 16):**
- Shared plotting_utils.py available for import
- Color-by pattern established and verified
- M3 parameter definitions confirmed in all CSVs
- MODEL_CONFIG pattern documented for Script 16 to follow

**Outputs verified:**
- All three models appear in output/mle/group_comparison_stats.csv
- All 12 figure files generated (4 plot types × 3 models)
- --color-by flag functional with hypothesis_group
- CSV structure supports M3 kappa parameter

**No blockers or concerns.**

---
*Phase: 04-regression-visualization*
*Completed: 2026-02-05*
