---
phase: 23-tech-debt-sweep-pre-flight-cleanup
plan: 04
subsystem: infra
tags: [arviz, netcdf, config, validation, pytest, bayesian, posterior]

# Dependency graph
requires:
  - phase: 23-tech-debt-sweep-pre-flight-cleanup
    provides: "23-01/02/03 completed: legacy/ deleted, K-bounds guard, 16b residue scrubbed"
  - phase: 13-infra-06
    provides: "load_fits_with_validation (CSV read-side wrapper precedent)"
provides:
  - "config.load_netcdf_with_validation(path, model) — validated single entry point for all NetCDF posterior loads"
  - "15 az.from_netcdf call sites rewired across 13 consumer files"
  - "scripts/fitting/tests/test_load_side_validation.py — grep invariant CI guard"
  - "CLEAN-04 closed"
affects:
  - "24-cold-start-pipeline-execution"
  - "25-reproducibility-regression"
  - "27-closure"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single validated entry point for all NetCDF posterior loads (mirrors load_fits_with_validation for CSV side)"
    - "Grep invariant test pattern: enumerate consumer files, assert no banned call pattern exists in active code"
    - "Soft-warn on missing parameterization_version attr (v4.0 state); hard-raise deferred to v5.1 write-side retrofit"

key-files:
  created:
    - "scripts/fitting/tests/test_load_side_validation.py"
  modified:
    - "config.py"
    - "scripts/14_compare_models.py"
    - "scripts/18_bayesian_level2_effects.py"
    - "scripts/21_fit_with_l2.py"
    - "scripts/21_compute_loo_stacking.py"
    - "scripts/21_baseline_audit.py"
    - "scripts/21_model_averaging.py"
    - "scripts/21_scale_audit.py"
    - "scripts/visualization/plot_posterior_diagnostics.py"
    - "scripts/visualization/plot_group_parameters.py"
    - "scripts/visualization/plot_model_comparison.py"
    - "scripts/visualization/quick_arviz_plots.py"
    - "scripts/simulations/generate_data.py"
    - "validation/compare_posterior_to_mle.py"

key-decisions:
  - "Soft-warn (not hard-raise) on missing parameterization_version attr: v4.0 NetCDF write side (fit_bayesian.py:732) does not emit this attr; retrofitting write+read together is a v5.1 item. Phase 23 delivers the infrastructure (wrapper + grep invariant) — hard enforcement lands when write-side attr is added."
  - "For scripts with dual-model CLIs (plot_group_parameters, plot_model_comparison): hardcode 'qlearning'/'wmrl' at the two call sites rather than adding a --model flag; these scripts already take --qlearning/--wmrl path args."
  - "grep invariant test excludes tests/ subtree to prevent self-referential false positives from docstrings in the guard file itself."
  - "14_compare_models.py line 599: model key derived from path stem (Path(path_str).stem.replace('_posterior', '')) since the loop var 'name' is a display name (M1, M2, ...) not an internal id."

patterns-established:
  - "NetCDF validation pattern: load_netcdf_with_validation(path, model) for all downstream consumers"
  - "Grep invariant test: hardcoded enumerated file list + pattern assertions to prevent future regressions"

# Metrics
duration: 45min
completed: 2026-04-19
---

# Phase 23 Plan 04: NetCDF Validation Wrapper Summary

**Single-entry NetCDF validation wrapper added to config.py; 15 bare az.from_netcdf call sites rewired across 13 consumer files; grep invariant CI guard installed — CLEAN-04 closed**

## Performance

- **Duration:** ~45 min
- **Started:** 2026-04-19T00:00:00Z
- **Completed:** 2026-04-19T00:45:00Z
- **Tasks:** 2
- **Files modified:** 14 (13 consumer scripts + config.py) + 1 new test file

## Accomplishments

- Added `load_netcdf_with_validation(path, model)` to config.py immediately after `load_fits_with_validation` — validates file existence, non-zero size, valid InferenceData, and `posterior` group; soft-warns on missing `parameterization_version` attr (v4.0 state)
- Rewired all 15 `az.from_netcdf(...)` call sites across 13 files to use the wrapper; every NetCDF posterior load now has a single validated entry point
- Created `scripts/fitting/tests/test_load_side_validation.py` with two pytest guards: (1) enumerated consumer files must not contain bare NetCDF calls, (2) no `.py` in scripts/ or validation/ may contain `xr.open_dataset(`
- All tests green (5/5 including existing MLE suite); v4.0 closure guard exits 0; two `--help` smoke tests pass

## Task Commits

1. **Task 1: Add load_netcdf_with_validation to config.py** — `2ac5430` (chore)
2. **Task 2: Wire wrapper into all consumers + install guard** — `2ece61c` (chore)

**Plan metadata:** see SUMMARY.md creation commit

## Files Created/Modified

- `config.py` — Added `load_netcdf_with_validation(path, model) -> az.InferenceData` + `TYPE_CHECKING` import for `az`
- `scripts/14_compare_models.py` — Added import; 2 sites rewired (line 599 derives model from path stem; line 798 uses `"wmrl_m4"`)
- `scripts/18_bayesian_level2_effects.py` — Added import; 1 site rewired (`args.model`)
- `scripts/21_fit_with_l2.py` — Added import; 1 site rewired (`model` param of `_verify_l2_posteriors_ready`)
- `scripts/21_compute_loo_stacking.py` — Added import; 1 site rewired (loop var `model`)
- `scripts/21_baseline_audit.py` — Extended existing `from config import`; 1 site rewired (`model` param)
- `scripts/21_model_averaging.py` — Added import; 1 site rewired (loop var `winner`)
- `scripts/21_scale_audit.py` — Added import; 2 sites rewired (function param `winner`)
- `scripts/visualization/plot_posterior_diagnostics.py` — Added sys.path bootstrap + import; 1 site rewired (`model_key` from `infer_model_from_path`)
- `scripts/visualization/plot_group_parameters.py` — Added `from __future__`, sys.path bootstrap, import; updated `load_posterior_samples` signature to accept `model: str`; updated 2 callers in main()
- `scripts/visualization/plot_model_comparison.py` — Same treatment as plot_group_parameters
- `scripts/visualization/quick_arviz_plots.py` — Added `from __future__`, sys.path bootstrap, import; 1 site rewired (model derived from path stem before load)
- `scripts/simulations/generate_data.py` — Extended existing import; 1 site rewired (`model_type` param)
- `validation/compare_posterior_to_mle.py` — Added sys.path bootstrap + import; updated `_posterior_means` signature to accept `model: str`; threaded through from `compare()`
- `scripts/fitting/tests/test_load_side_validation.py` (NEW) — 2-function pytest grep invariant guard

## Decisions Made

- **Soft-warn on absent `parameterization_version`**: v4.0 NetCDF write side does not emit this attr. Flip to hard-raise is a v5.1 item once `fit_bayesian.py:732` is retrofitted.
- **Dual-model visualization scripts**: `plot_group_parameters.py` and `plot_model_comparison.py` take `--qlearning` / `--wmrl` path args; hardcoded `"qlearning"` and `"wmrl"` model strings at the two call sites rather than adding a `--model` flag.
- **14_compare_models.py line 599**: loop var `name` is a display key ("M1", "M2", ...) not an internal model id. Derive internal key from `Path(path_str).stem.replace("_posterior", "")`.
- **Grep invariant self-exclusion**: the test guard itself lives in `tests/` and contains the patterns as docstring/string literals. Excluding `tests/` from the `xr.open_dataset` walk prevents false-positive self-detection.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] grep invariant test flagged its own docstrings**

- **Found during:** Task 2 Sub-step C (pytest run of new guard test)
- **Issue:** `test_no_bare_xr_open_dataset_anywhere` walks `scripts/` which includes `scripts/fitting/tests/test_load_side_validation.py` — the test file mentions `xr.open_dataset(` in docstrings/f-strings, causing it to report itself as a violation.
- **Fix:** Extended `excluded_subtrees` tuple from `("__pycache__", "tests/fixtures")` to `("__pycache__", "tests/fixtures", "tests/")` — test files are logically excluded from the enforcement scope (per plan's "Files explicitly NOT modified" section which notes test files may legitimately use bare NetCDF).
- **Files modified:** `scripts/fitting/tests/test_load_side_validation.py`
- **Committed in:** `2ece61c` (Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Minor self-reference fix. No scope change; guard correctly enforces zero bare calls outside test trees.

## Issues Encountered

None — all 15 call sites rewired cleanly. Both `--help` smoke tests and all 5 pytest tests pass. Closure guard exits 0.

## v5.1 Deferral

**Write-side `parameterization_version` attr retrofit** (`scripts/fitting/fit_bayesian.py:732`): `idata.to_netcdf(...)` does not currently write `parameterization_version` to the NetCDF attrs. The read-side wrapper emits a `DeprecationWarning` when the attr is absent (current v4.0 state). Full hard enforcement (flip to `ValueError`) is deferred to a v5.1 plan that will:
1. Add `idata.attrs["parameterization_version"] = EXPECTED_PARAMETERIZATION[model]` in `fit_bayesian.py` before `idata.to_netcdf(...)`.
2. Flip the `warnings.warn(DeprecationWarning)` to `raise ValueError` in `load_netcdf_with_validation`.

## Next Phase Readiness

- CLEAN-04 fully closed; all 15 NetCDF load paths validated.
- Phase 24 cold-start pipeline execution can now produce posteriors knowing that any stale or mismatched `.nc` file will surface at load time rather than propagate silently through analysis.
- No blockers for Phase 24.

---
*Phase: 23-tech-debt-sweep-pre-flight-cleanup*
*Completed: 2026-04-19*
