---
phase: 04-regression-visualization
plan: 02
subsystem: visualization
tags: [matplotlib, seaborn, regression, plotting, M3, kappa, perseveration, color-by]

# Dependency graph
requires:
  - phase: 04-regression-visualization
    plan: 01
    provides: plotting_utils.py for color-by functionality
  - phase: 03-model-comparison
    provides: wmrl_m3_individual_fits.csv with M3 parameters
provides:
  - Script 16 support for all three models (M1, M2, M3) with --model flag
  - Model-specific output subdirectories (output/regressions/{model}/)
  - Color-by functionality via --color-by flag
  - Structured CSV output with Section column grouping regressions
affects: [07-publication-polish]

# Tech tracking
tech-stack:
  added: []
  patterns: [model loop pattern in main(), auto-detect params path from model name, structured CSV with Section column]

key-files:
  created: []
  modified:
    - scripts/16_regress_parameters_on_scales.py

key-decisions:
  - "Remove --params argument, auto-detect from model name (output/mle/{model}_individual_fits.csv)"
  - "Model-specific subdirectories prevent overwrites when --model all"
  - "Section column in CSV groups each scale x parameter regression for readability"
  - "Skip DataFrame console display due to Windows Unicode encoding issues (CSV saved correctly)"

patterns-established:
  - "Auto-detect params path pattern: Path(f'output/mle/{model}_individual_fits.csv')"
  - "Model loop with subdirectories: for model in models_to_run: model_output_dir = base_output_dir / model"
  - "Structured CSV with Section column for grouping: row['Section'] = f'{format_label(param)} ~ {format_label(pred)}'"

# Metrics
duration: 18min
completed: 2026-02-05
---

# Phase 04 Plan 02: Script 16 Regression Analysis Extension Summary

**Script 16 extended with M3 support, --model all, --color-by capability, model-specific subdirectories, and structured CSV output**

## Performance

- **Duration:** 18 min
- **Started:** 2026-02-05T22:06:04Z
- **Completed:** 2026-02-05T22:24:46Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments
- Extended Script 16 to support M3 (wmrl_m3) with kappa perseveration parameter
- Added --model flag with choices [qlearning, wmrl, wmrl_m3, all] for flexible analysis
- Implemented model-specific output subdirectories (output/regressions/{model}/)
- Added --color-by CLI flag for scatter plot coloring by any categorical variable
- Removed --params argument, replaced with auto-detection from model name
- Merged trauma groups and demographics data for --color-by support
- Added Section column to CSV output for structured grouping of regressions
- Fixed Unicode encoding issues for Windows console compatibility

## Task Commits

Each task was committed atomically:

1. **Task 1: Add M3 support, --model all, --color-by, and model subdirectories to Script 16** - `7abac09` (feat)

## Files Created/Modified
- `scripts/16_regress_parameters_on_scales.py` - Added M3 support, argparse CLI (--model, --color-by), model loop with subdirectories, plotting_utils imports, structured CSV with Section column, demographics/groups merging for color-by, Unicode fixes

## Decisions Made

**1. Auto-detect params path from model name**
- Pattern: `params_path = Path(f'output/mle/{model}_individual_fits.csv')`
- Eliminates need for --params argument (was hardcoded default)
- Enables --model all to loop over multiple models seamlessly

**2. Model-specific subdirectories**
- Structure: `output/regressions/{model}/regression_results_simple.csv`
- Prevents overwrites when running --model all
- Each model gets its own CSV files and PNG figures

**3. Section column in CSV output**
- Format: `"alpha+ (Positive Learning Rate) ~ IES-R Total Score"`
- Groups each scale x parameter regression together when sorted
- Improves readability for large regression output tables (REGR-01 requirement)

**4. Skip DataFrame console display**
- Windows console (cp1252 encoding) cannot render Greek letters (β, R²)
- CSV files save correctly with UTF-8 encoding
- Users can view CSV files directly without encoding errors

## Deviations from Plan

None - plan executed exactly as written. All required features implemented and verified.

## Issues Encountered

**Windows Console Unicode Encoding:**
- Windows console uses cp1252 encoding by default
- Greek letters (β, κ, R², ×) in print statements cause UnicodeEncodeError
- Fixed by replacing Greek letters with ASCII equivalents in print statements
- CSV files remain unaffected (use UTF-8 encoding)
- Note: This is expected behavior in Windows environment

## User Setup Required

None - no external service configuration required.

## Verification Results

**1. Q-learning model:**
- ✓ Completed without errors
- ✓ Created `output/regressions/qlearning/` subdirectory
- ✓ Generated 18 scatter plots (3 params × 6 predictors)
- ✓ Created regression_results_simple.csv with Section column
- ✓ Created regression_matrix_all.png
- ✓ FDR correction applied across all regressions

**2. WM-RL M3 model:**
- ✓ Completed without errors
- ✓ Created `output/regressions/wmrl_m3/` subdirectory
- ✓ kappa_mean parameter appears in all analyses
- ✓ Generated 42 scatter plots (7 params × 6 predictors including kappa)
- ✓ CSV contains "kappa (Perseveration) ~ ..." sections

**3. --model all:**
- ✓ Successfully loops over all three models (qlearning, wmrl, wmrl_m3)
- ✓ No overwrites between models (separate subdirectories)
- ✓ Completed message shows all three models processed

**4. --color-by hypothesis_group:**
- ✓ Validates column exists before plotting
- ✓ Generates color palette using TRAUMA_GROUP_COLORS
- ✓ Applies colored scatter to all regression plots
- ✓ No legend in matrix cells (show_legend=False)

**5. Structured CSV output:**
- ✓ Section column present in all regression_results_simple.csv files
- ✓ Rows sorted by Section for grouped display
- ✓ Format: `"{param_label} ~ {predictor_label}"`

## Next Phase Readiness

**Phase 4 Complete:**
- Script 15 and Script 16 both support M3, --model all, --color-by
- Shared plotting_utils.py used by both scripts
- All plot types generated for all models
- Model-specific output organization prevents overwrites
- Structured CSV output with clear sectioning (REGR-01)

**Ready for checkpoint:human-verify:**
- All auto tasks complete
- User verification needed for full pipeline and visual outputs

**No blockers or concerns.**

---
*Phase: 04-regression-visualization*
*Completed: 2026-02-05*
