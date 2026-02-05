# Phase 4: Regression Visualization - Context

**Gathered:** 2026-02-05
**Status:** Ready for planning

<domain>
## Phase Boundary

Enhanced visualization and output organization for continuous regression analysis in Scripts 15-16. Adding `--color-by` grouping to all scatter/regression plots, adding M3 model support, and restructuring output by model. Existing plot styling and color palette stay the same.

</domain>

<decisions>
## Implementation Decisions

### Model handling
- Add `--model all` and `wmrl_m3` choices to Script 16's `--model` flag
- `--model all` loops over all fitted models (qlearning, wmrl, wmrl_m3) and generates output for each
- Script 15 already loops both models internally; extend to include M3
- Output files get model-specific names to avoid overwrites across runs

### Output organization
- Model subdirectories: `output/regressions/qlearning/`, `output/regressions/wmrl/`, `output/regressions/wmrl_m3/`
- Each subdirectory contains its own regression CSVs and plot PNGs
- Script 16 output restructured with clear sections grouping each scale x parameter regression

### Color-by mechanism
- `--color-by <variable>` is a color overlay only — does NOT change which analyses run
- Trauma group comparisons still come from existing grouping logic
- `--color-by` accepts any categorical column from participant data (trauma_group, gender, etc.)
- Applied to all scatter/regression plots in both Scripts 15 and 16

### Plot coverage
- All plot types generated for all models (correlation heatmaps, forest plots, scatter plots)
- Currently Script 15 only generates some plots for WM-RL; extend to Q-learning and M3 too
- Q-learning has fewer parameters (3 vs 6-7) but still gets all applicable plot types

### Visual style
- Keep current visualization style, color palette, and plot formatting
- No changes to existing plot aesthetics beyond adding color-by support

### Claude's Discretion
- Exact implementation of --color-by infrastructure (shared utility vs per-script)
- Legend placement and sizing when color-by is active
- How to handle color-by variables with many categories (>5 groups)
- Subplot layout adjustments when Q-learning has fewer parameters than WM-RL/M3

</decisions>

<specifics>
## Specific Ideas

- "Keep most of the current visualization / color output the same. Just make it more aware of the models"
- The emphasis is on continuous scale regressions (Script 16) over discrete trauma groups
- Less emphasis on tight group comparisons, more on the continuous relationships

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 04-regression-visualization*
*Context gathered: 2026-02-05*
