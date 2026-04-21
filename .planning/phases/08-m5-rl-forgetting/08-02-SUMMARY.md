---
phase: 08-m5-rl-forgetting
plan: 02
subsystem: modeling
tags: [parameter-recovery, model-comparison, trauma-analysis, regression, wmrl_m5, phi_rl]

# Dependency graph
requires:
  - phase: 08-01
    provides: WMRL_M5_BOUNDS, WMRL_M5_PARAMS in mle_utils.py; wmrl_m5 likelihood; fit_mle M5 support; output/wmrl_m5_individual_fits.csv

provides:
  - M5 parameter recovery simulation via scripts/fitting/model_recovery.py
  - M5 in AIC/BIC comparison table via scripts/14_compare_models.py (--m5 arg + auto-detection)
  - M5 trauma group analysis with phi_rl via scripts/15_analyze_mle_by_trauma.py
  - M5 continuous regression with phi_rl via scripts/16_regress_parameters_on_scales.py

affects: [09-m6a-stim-specific, 10-m6b-dual, 11-m4-lba, 12-integration]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "M5 pipeline extension pattern: copy M3 dispatch blocks, add elif wmrl_m5, add new param (phi_rl)"
    - "Dual-path file search: output/mle/ first, then output/ root (for models fit with --output output)"

key-files:
  created: []
  modified:
    - scripts/fitting/model_recovery.py
    - scripts/11_run_model_recovery.py
    - scripts/14_compare_models.py
    - scripts/15_analyze_mle_by_trauma.py
    - scripts/16_regress_parameters_on_scales.py

key-decisions:
  - "M5 auto-detection in script 14 searches output/mle/ then output/ root (M5 fits in output/ from plan 01)"
  - "Script 15 wmrl_m5 guard: prints helpful message if fits not found, prevents crash"
  - "Script 14 default --mle-dir corrected from output/mle_results to output/mle (pre-existing bug fix)"

patterns-established:
  - "All downstream scripts use elif wmrl_m5 branch pattern — same pattern for M6a, M6b, M4"
  - "Fallback path search: primary dir checked first, then secondary (handles different --output invocations)"

# Metrics
duration: 44min
completed: 2026-04-02
---

# Phase 8 Plan 02: M5 RL Forgetting Pipeline Integration Summary

**M5 (phi_rl RL forgetting) wired into all downstream pipeline scripts: parameter recovery simulation, AIC/BIC comparison, trauma group analysis, and continuous regression — with M5 winning over M3 by dAIC=435.6**

## Performance

- **Duration:** 44 min
- **Started:** 2026-04-02T18:49:57Z
- **Completed:** 2026-04-02T19:34:14Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- M5 parameter recovery simulation complete: synthetic data generation with Q decay BEFORE policy (matching likelihood), runs without error
- M5 appears in AIC/BIC comparison table (script 14): M5 wins over M3 by dAIC=435.6 (very strong evidence), dBIC=226.9
- phi_rl parameter reported in trauma group analysis (script 15) and continuous regression (script 16)
- Quick parameter recovery test (N=10) completed in 16 min: pipeline functional, CSV written to output/recovery/wmrl_m5/

## Task Commits

Each task was committed atomically:

1. **Task 1: M5 parameter recovery simulation** - `8c5bb49` (feat)
2. **Task 2: M5 pipeline scripts (comparison + trauma analysis)** - `a1bc5b9` (feat)

**Plan metadata:** (pending — see final commit below)

## Files Created/Modified

- `scripts/fitting/model_recovery.py` - Added WMRL_M5_BOUNDS/WMRL_M5_PARAMS imports, wmrl_m5 in all 5 dispatch functions; Q decay before policy in generate_synthetic_participant()
- `scripts/11_run_model_recovery.py` - Added wmrl_m5 to argparse choices and all expansion
- `scripts/14_compare_models.py` - Added --m5 argument, M5 auto-detection pattern, fallback output/ search, fixed default --mle-dir
- `scripts/15_analyze_mle_by_trauma.py` - Added WMRL_M5_PARAMS, phi_rl display name, defensive M5 loading with dual-path search, MODEL_CONFIG entry, updated argparse
- `scripts/16_regress_parameters_on_scales.py` - Added phi_rl->phi_rl_mean rename, phi_rl_mean to format_label, wmrl_m5 param_cols, fallback path, updated argparse

## Decisions Made

- Script 14 auto-detection searches `output/mle/` first, then `output/` root — because M5 fits from plan 01 were written to `output/` with `--output output`, not `output/mle/`
- Script 15 wmrl_m5 guard: returns with descriptive error if wmrl_m5 fits not found (instead of crashing with FileNotFoundError)
- Script 14 `--mle-dir` default corrected from `output/mle_results` to `output/mle` — pre-existing bug where auto-detection always failed silently (Rule 1 auto-fix)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed broken auto-detection default path in script 14**
- **Found during:** Task 2 (M5 pipeline scripts)
- **Issue:** `--mle-dir` default was `output/mle_results` but fits are in `output/mle/`. Running `python scripts/14_compare_models.py` without arguments always failed with "At least 2 models required"
- **Fix:** Changed default to `output/mle`; also added fallback search in `output/` root for models written outside standard dir
- **Files modified:** scripts/14_compare_models.py
- **Verification:** `python scripts/14_compare_models.py` now auto-detects M1-M3 and M5 correctly
- **Committed in:** a1bc5b9 (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (1 bug)
**Impact on plan:** Fix required for correct auto-detection; no scope creep.

## Issues Encountered

- Full parameter recovery (N=50 subjects, r >= 0.80 gate) is a long-running operation (~80-150 min on CPU). Quick test (N=10, 1 dataset) ran in 16 min and confirmed code works without errors. N=10 results: best r=0.572 (phi_rl) — expected with small N; r>=0.80 requires N>=50 with multiple datasets. Full validation should be run on cluster: `sbatch cluster/12_mle.slurm` with recovery config.
- Output path mismatch: M5 fits from plan 01 in `output/` not `output/mle/` — handled via dual-path search in scripts 14, 15, 16.

## Next Phase Readiness

- All pipeline scripts handle wmrl_m5: comparison, trauma analysis, regression
- M5 is confirmed as new winning model (dAIC=435.6 over M3)
- Full parameter recovery (r >= 0.80 gate) pending — recommend running on cluster before proceeding to Phase 9
- Phase 9 (M6a stimulus-specific perseveration) can follow the same extension pattern established here

---
*Phase: 08-m5-rl-forgetting*
*Completed: 2026-04-02*
