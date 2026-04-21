---
phase: 18-integration-comparison-manuscript
plan: 01
subsystem: analysis-pipeline
tags: [argparse, path-routing, schema-parity, bayesian, mle, migration]

# Dependency graph
requires:
  - phase: 13-infrastructure-repair-hierarchical-scaffolding
    provides: schema-parity CSV pattern (INFRA-04): Bayesian individual_fits CSVs match MLE column layout
  - phase: 15-m3-hierarchical-poc-level2
    provides: output/bayesian/*_individual_fits.csv files produced by hierarchical fitting
  - phase: 16-choice-only-family-extension-subscale-level-2
    provides: output/bayesian/*_individual_fits.csv for all 6 choice-only models
provides:
  - --source mle|bayesian flag on scripts 15, 16, 17 with backward-compatible defaults
  - Path routing: --source bayesian sends output to output/bayesian/analysis/, output/regressions/bayesian/, output/bayesian/model_comparison/
  - No analysis logic changed — pure path-routing layer
affects:
  - phase: 18-integration-comparison-manuscript (plans 02-05 can now call scripts 15/16/17 with --source bayesian)
  - future cluster jobs that run full pipeline on Bayesian fits

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "--source flag pattern: single argparse choice drives all path routing; analysis logic untouched"
    - "load_data(fits_dir) refactor: module-level constant as default, caller overrides for Bayesian"
    - "args.source default detection: compare against hardcoded default string before overriding"

key-files:
  created: []
  modified:
    - scripts/15_analyze_mle_by_trauma.py
    - scripts/16_regress_parameters_on_scales.py
    - scripts/17_analyze_winner_heterogeneity.py

key-decisions:
  - "MLE source retains legacy fallback to output/ root; Bayesian source has no fallback (files expected in output/bayesian/)"
  - "Script 16 detects unchanged defaults by string comparison against hardcoded default values before overriding"
  - "Script 17 mle_dir renamed to fits_dir throughout main() for source-agnostic semantics"
  - "Module-level FIGURES_DIR and OUTPUT_DIR constants untouched; local variables shadow them in main() when --source bayesian"

patterns-established:
  - "Schema-parity migration pattern: downstream scripts get --source flag, no statistical logic rewrite needed"

# Metrics
duration: 7min
completed: 2026-04-13
---

# Phase 18 Plan 01: --source mle|bayesian Flag Migration Summary

**Schema-parity flag flip for scripts 15, 16, 17: single --source argument routes all I/O paths to Bayesian outputs with zero analysis logic changes**

## Performance

- **Duration:** ~7 min
- **Started:** 2026-04-13T18:52:56Z
- **Completed:** 2026-04-13T19:00:23Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Added `--source {mle,bayesian}` argparse flag to all three downstream analysis scripts
- Script 15: refactored `load_data()` to accept `fits_dir` parameter; main() resolves `figures_dir` and `analysis_output_dir` from `--source`
- Script 16: default `--output-dir` and `--figures-dir` auto-redirect to `bayesian/` subdirectory when `--source bayesian`; model CSV lookup uses `output/bayesian/` or `output/mle/`
- Script 17: added `import argparse`; renamed `mle_dir` to `fits_dir`; full path routing in main()
- All three scripts print `Source: MLE` or `Source: BAYESIAN` banner on startup
- All three default to MLE with identical behavior to pre-change code (backward compatible)

## Task Commits

Each task was committed atomically:

1. **Task 1: Add --source flag to script 15 (analyze_mle_by_trauma)** - `7b368f0` (feat)
2. **Task 2: Add --source flag to script 16 (regress_parameters_on_scales)** - `192b87a` (feat)
3. **Task 3: Add --source flag to script 17 (analyze_winner_heterogeneity)** - `5bc54f5` (feat)

## Files Created/Modified

- `scripts/15_analyze_mle_by_trauma.py` - `load_data(fits_dir)` parameter, `--source` flag, path routing in main()
- `scripts/16_regress_parameters_on_scales.py` - `--source` flag, default path override, CSV lookup routing
- `scripts/17_analyze_winner_heterogeneity.py` - `import argparse`, `--source` flag, `mle_dir` renamed to `fits_dir`

## Decisions Made

- **MLE legacy fallback preserved, Bayesian has no fallback (locked):** When `--source mle`, scripts check `output/mle/` then fall back to `output/` root for files written by older pipeline runs. When `--source bayesian`, files must exist in `output/bayesian/` — no fallback to prevent silently loading stale MLE data.
- **Script 16 default detection by string comparison (locked):** After `args = parser.parse_args()`, compare `args.output_dir == 'output/regressions'` to detect unchanged default. Avoids sentinel value complexity; works correctly because argparse returns the literal default string when user doesn't specify.
- **Module-level constants untouched (locked):** `OUTPUT_DIR`, `FIGURES_DIR` in scripts 15 and 17 remain as MLE defaults. Local variables `fits_dir`, `figures_dir`, `analysis_output_dir` in `main()` shadow them for Bayesian routing. Ensures any code that imports these constants (e.g., notebooks) still gets the MLE path.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- MIG-01, MIG-02, MIG-03 requirements satisfied
- Scripts 15/16/17 ready to accept Bayesian fit CSVs from Phase 15-16 output directories
- Plan 18-02 (Bayesian model comparison and WAIC/LOO) can now build on this routing infrastructure

---
*Phase: 18-integration-comparison-manuscript*
*Completed: 2026-04-13*
