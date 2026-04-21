---
phase: 14
plan: "01"
subsystem: mle-fitting
tags:
  - capacity-bounds
  - parameterization-version
  - identifiability
  - k-refit
requires:
  - "13-02"  # K bounds [2,6] confirmed via K-01 identifiability research
  - "13-04"  # EXPECTED_PARAMETERIZATION defined in config.py with validation gate
provides:
  - "K-02: K bounds constrained to [2,6] in all 6 WM BOUNDS dicts in mle_utils.py"
  - "K-03 partial: parameterization_version stamp in fit_mle.py main() — cluster refit is separate execution"
affects:
  - "14-02"  # GPU LBA batching — will produce fits with new bounds
  - "14-03"  # K-03 cluster refit — requires these bounds to be in effect
  - "15-01"  # M3 hierarchical POC — load_fits_with_validation() will reject stale fits

tech-stack:
  added: []
  patterns:
    - "BOUNDS dict constraint pattern: capacity key updated to (2.0, 6.0) in all 6 WM model dicts"
    - "Version stamp injection: EXPECTED_PARAMETERIZATION[args.model] applied to fits_df before CSV save"

key-files:
  created: []
  modified:
    - "scripts/fitting/mle_utils.py"
    - "scripts/fitting/fit_mle.py"
    - "config.py"

key-decisions:
  - "K bounds (2.0, 6.0) applied uniformly to all 6 capacity-bearing BOUNDS dicts"
  - "Version stamp injected in main() before to_csv (not inside fit_all_participants or fit_all_gpu)"
  - "QLEARNING_BOUNDS unchanged — M1 has no capacity parameter"
  - "WM_CAPACITY_MU_PRIOR = 4.0 and WM_CAPACITY_SIGMA_PRIOR = 1.5 unchanged (prior mean centered in [2,6])"

patterns-established:
  - "Downstream validation gate: fits_df['parameterization_version'] = EXPECTED_PARAMETERIZATION[args.model]"

duration: "~7 min"
completed: "2026-04-12"
---

# Phase 14 Plan 01: K-Bounds and Version Stamp Summary

K bounds constrained from (1.0, 7.0) to (2.0, 6.0) in all 6 WM BOUNDS dicts in mle_utils.py, and parameterization_version column stamped on MLE CSV output in fit_mle.py main() before CSV save.

## Accomplishments

1. **K-02 complete:** All 6 WM capacity BOUNDS dicts in `scripts/fitting/mle_utils.py` updated from `(1.0, 7.0)` to `(2.0, 6.0)`:
   - WMRL_BOUNDS, WMRL_M3_BOUNDS, WMRL_M5_BOUNDS, WMRL_M6A_BOUNDS, WMRL_M6B_BOUNDS, WMRL_M4_BOUNDS
   - QLEARNING_BOUNDS unchanged (M1 has no capacity parameter)

2. **config.py documentation constants updated:** `ModelParams.WM_CAPACITY_MIN = 2` and `WM_CAPACITY_MAX = 6` — consistent with new bounds.

3. **K-03 partial (stamp injection ready):** `fit_mle.py` now imports `EXPECTED_PARAMETERIZATION` from `config` and stamps `fits_df['parameterization_version']` in `main()` before `fits_df.to_csv()`. Both CPU and GPU paths converge to `main()` before the single save, so both get the stamp.

4. **LHS sampler verified:** `sample_lhs_starts` produces capacity values within [2.0, 6.0] for all 6 WM models (100 draws, seed=42). Range observed: [2.002, 5.995] across models.

5. **15/16 tests pass** (1 pre-existing failure in test_compile_gate — JAX scan shape issue in fixture, confirmed pre-existing by stash check).

## Task Commits

| Task | Name | Commit | Files Modified |
|------|------|--------|----------------|
| 1 | Constrain K bounds [1,7] to [2,6] | df9c20a | scripts/fitting/mle_utils.py, config.py |
| 2 | Stamp parameterization_version in fit_mle.py | 3190e37 | scripts/fitting/fit_mle.py |
| 3 | Verify LHS sampler respects new bounds | — (read-only verification) | — |

## Files Modified

| File | Change |
|------|--------|
| `scripts/fitting/mle_utils.py` | 6 capacity bounds updated (1.0,7.0) → (2.0,6.0) |
| `config.py` | WM_CAPACITY_MIN=1→2, WM_CAPACITY_MAX=7→6 |
| `scripts/fitting/fit_mle.py` | Import EXPECTED_PARAMETERIZATION; stamp fits_df before to_csv |

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| K lower bound = 2 | Structural identifiability: K<2 confounded with rho at ns=2 (smallest set-size). Matching Senta et al. (2025). |
| K upper bound = 6 | Task max set-size = 6; K>6 non-identified (no blocks with ns>6 to distinguish K=6 from K=7). |
| Stamp in main() not fit_all_* | main() is the single convergence point before to_csv; avoids duplicate logic in CPU and GPU code paths. |
| WM_CAPACITY_MU_PRIOR unchanged | Prior mean 4.0 is centered in [2,6]; sigma 1.5 gives reasonable spread. No prior change needed. |

## Deviations from Plan

None — plan executed exactly as written.

## Issues

- **Pre-existing test failure:** `test_compile_gate.py::test_compile_gate` fails due to a JAX `lax.scan` shape error in the test fixture (`scan got value with no leading axis`). Confirmed pre-existing by stash check against the previous commit. Not caused by this plan's changes. Filed as a pre-existing issue for Phase 15 to investigate.

## Next Phase Readiness

- **14-02 (GPU LBA batching):** Ready. Updated bounds in place; GPU path will produce correctly bounded K fits.
- **14-03 (K-03 cluster refit):** Ready. Stamp injection in fit_mle.py means cluster fits will carry `parameterization_version = "v4.0-K[2,6]-phiapprox"` in output CSVs, enabling `load_fits_with_validation()` downstream.
- **15-01 (M3 hierarchical POC):** Will need to address the pre-existing test_compile_gate failure before or during Phase 15 execution.
