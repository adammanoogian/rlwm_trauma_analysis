---
phase: 11-m4-lba-joint-choice-rt
plan: "03"
subsystem: fitting
tags: [jax, lba, model-recovery, parameter-recovery, rt-simulation, downstream-pipeline, model-comparison, trauma-analysis]

# Dependency graph
requires:
  - phase: 11-m4-lba-joint-choice-rt
    plan: "02"
    provides: wmrl_m4_block_likelihood, WMRL_M4_PARAMS/WMRL_M4_BOUNDS, fit_mle.py M4 pipeline, prepare_participant_data RT extraction

provides:
  - "generate_synthetic_participant: LBA race RT simulation for wmrl_m4 (k~Uniform(0,A), t=(b-k)/v, winner=argmin, RT=t_winner+t0)"
  - "Convex combination perseveration for M4 synthetic generation (matches likelihood exactly)"
  - "run_parameter_recovery end-to-end for wmrl_m4 with rts_blocks passthrough"
  - "Script 14: M4 in SEPARATE TRACK (joint choice+RT not comparable to choice-only AIC)"
  - "Script 15: wmrl_m4 analysis with WMRL_M4_PARAMS (v_scale, A, delta, t0) + load_data() 9-tuple"
  - "Script 16: wmrl_m4 regression dispatch with v_scale/A/delta/t0 renamed to _mean suffix"
  - "Script 11: argparse accepts wmrl_m4; 'all' list includes wmrl_m4"

affects:
  - 11-04 (full M4 pipeline now complete; cluster run for r>=0.80 gate ready)
  - Future analysis runs (scripts 14/15/16 ready for real M4 fits)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "LBA race simulation: k~Uniform(0,A) via numpy rng (not JAX); t=(b-k)/max(v,1e-6); winner=argmin"
    - "Convex combination perseveration: (1-kappa)*hybrid + kappa*Ck for M4 (not additive renorm used by M3/M5)"
    - "Separate comparison track: pop 'M4' from fits_dict before AIC comparison; report standalone"
    - "9-tuple return from load_data(): adds wmrl_m4 at end; all callers updated"

key-files:
  created: []
  modified:
    - scripts/fitting/model_recovery.py
    - scripts/11_run_model_recovery.py
    - scripts/14_compare_models.py
    - scripts/15_analyze_mle_by_trauma.py
    - scripts/16_regress_parameters_on_scales.py

key-decisions:
  - "M4 LBA race uses numpy rng (not JAX random) for start points -- consistent with other numpy operations in generate_synthetic_participant"
  - "Convex combination perseveration for M4 (not additive renorm): ensures generation exactly matches likelihood formula, prevents kappa recovery bias"
  - "RT stored in milliseconds in synthetic DataFrame (real data is in ms; preprocessing converts to seconds)"
  - "M4 separate track in script 14: pop from fits_dict BEFORE comparison; no accidental AIC cross-contamination"
  - "Script 15 load_data() grows to 9-tuple (adds wmrl_m4 at end); M4 loaded defensively with fallback path"
  - "Script 16 adds v_scale/A/delta/t0 rename to _mean suffix in load_integrated_data; M4 param_cols uses v_scale_mean/A_mean/delta_mean/t0_mean"

patterns-established:
  - "M4 downstream pattern: wherever M6b has elif branch, M4 needs one too (get_param_names, sample_parameters, compute_recovery_metrics, plot_recovery_scatter, plot_distribution_comparison)"
  - "RT simulation uses numpy rng (already initialized in generate_synthetic_participant) not JAX random"

# Metrics
duration: 12min
completed: 2026-04-03
---

# Phase 11 Plan 03: M4 Downstream Pipeline Integration Summary

**LBA race simulation in model_recovery.py with convex combination perseveration, M4 in separate choice+RT comparison track in script 14, and WMRL_M4_PARAMS (v_scale, A, delta, t0) dispatched in scripts 15 and 16**

## Performance

- **Duration:** 12 min
- **Started:** 2026-04-03T12:03:07Z
- **Completed:** 2026-04-03T12:15:02Z
- **Tasks:** 2
- **Files modified:** 5

## Accomplishments

- Implemented LBA race simulation in `generate_synthetic_participant` for wmrl_m4: k~Uniform(0,A) via numpy rng, t=(b-k)/v_safe, winner=argmin(t), RT=t[winner]+t0, stored in milliseconds
- Used convex combination perseveration for M4 `(1-kappa)*hybrid + kappa*Ck` matching M4 likelihood exactly (M3/M5 continue using additive renormalization)
- Added `rts_blocks=data_dict.get('rts_blocks')` passthrough in `run_parameter_recovery` so M4 synthetic data flows into fit pipeline
- Script 14: pop M4 from `fits_dict` before AIC comparison; report in separate `M4 (Joint Choice+RT) - SEPARATE TRACK` section with per-param mean/SEM summary
- Scripts 15/16: added WMRL_M4_PARAMS constant and dispatch branches; `load_data()` extended to 9-tuple; `load_integrated_data` adds v_scale/A/delta/t0 renaming
- Quick test verified: N=2 recovery completes end-to-end, synthetic data has `rt` column (316-840ms range), 10 params (no epsilon) in results

## Task Commits

Each task was committed atomically:

1. **Task 1: M4 parameter recovery with LBA race RT simulation** - `461f377` (feat)
2. **Task 2: M4 separate comparison track + trauma analysis scripts** - `4d2c20f` (feat)

**Plan metadata:** (to be added in final commit)

## Files Created/Modified

- `scripts/fitting/model_recovery.py` - Added WMRL_M4 imports; wmrl_m4 in all dispatch functions; LBA race simulation with convex perseveration; rts_blocks passthrough; RT stored in ms
- `scripts/11_run_model_recovery.py` - Added wmrl_m4 to argparse choices and 'all' model list
- `scripts/14_compare_models.py` - Added M4 pattern in find_mle_files; pop M4 before AIC comparison; M4 separate track reporting section; --m4 argparse flag
- `scripts/15_analyze_mle_by_trauma.py` - WMRL_M4_PARAMS constant; v_scale/A/delta/t0 in PARAM_NAMES; load_data() 9-tuple with M4; MODEL_CONFIG entry; argparse choices
- `scripts/16_regress_parameters_on_scales.py` - wmrl_m4 in choices and all-models list; M4 param_cols dispatch; v_scale/A/delta/t0 rename in load_integrated_data

## Decisions Made

- **Convex combination for M4 generation:** `(1-kappa)*hybrid + kappa*Ck` matches M4 likelihood formula exactly. M3/M5 use additive renormalization (their likelihoods do too). Using the wrong formula would bias kappa recovery.
- **numpy rng for LBA start points:** generate_synthetic_participant already uses numpy rng for reversal thresholds and rewards; consistent to use it for LBA start points rather than mixing JAX random.
- **RT in milliseconds in synthetic DataFrame:** Real data is in ms; `prepare_participant_data` converts ms→s. Storing in ms means synthetic data passes through the same preprocessing path as real data.
- **M4 pop from fits_dict:** Cleanest approach -- `fits_dict.pop('M4', None)` before any comparison logic ensures M4 never accidentally appears in AIC table even if comparison logic is later extended.

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None beyond standard implementation.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- M4 parameter recovery pipeline is complete and ready for full cluster run (N>=30, n_datasets>=10) to validate r>=0.80 gate
- Full cluster validation command: `python scripts/11_run_model_recovery.py --model wmrl_m4 --n-subjects 50 --n-datasets 10 --n-jobs 8`
- Scripts 14/15/16 ready to analyze real M4 fits once `python scripts/12_fit_mle.py --model wmrl_m4` has run on full dataset
- Plan 11-04 (final phase: documentation and cluster job scripts) is the last remaining plan in phase 11

---
*Phase: 11-m4-lba-joint-choice-rt*
*Completed: 2026-04-03*
