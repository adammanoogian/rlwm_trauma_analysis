---
phase: 14
plan: 03
subsystem: cluster-ops
tags: [slurm, model-comparison, k-refit, gpu, cluster]
requires: ["14-01", "14-02"]
provides: ["SLURM/comparison compatibility verified", "Cluster refit commands documented"]
affects: ["15-M3-hierarchical-poc"]
tech-stack:
  added: []
  patterns: []
key-files:
  created: []
  modified: []
key-decisions:
  - "14_compare_models.py does NOT use load_fits_with_validation — correct for transition period (mixed-vintage CSVs)"
  - "SLURM script passes --compute-diagnostics (Hessian metrics included in refit)"
  - "No SLURM script changes needed — K bounds + version stamp propagate through mle_utils.py/fit_mle.py"
patterns-established: []
duration: "partial — Task 1 ~7 min, Task 2 pending cluster"
completed: "pending"
status: "partial — Task 2 checkpoint:human-action awaiting cluster refit"
---

# Phase 14 Plan 03: SLURM/Comparison Compatibility Summary

**SLURM and comparison scripts verified compatible with K bounds + version stamp — cluster refit submitted, awaiting results.**

## Status: PARTIAL

Task 1 complete (read-only verification). Task 2 (checkpoint:human-action) deferred — cluster refit submitted, results pending.

## Accomplishments

1. **SLURM script compatible:** `cluster/12_mle_gpu.slurm` dispatches all 7 models, calls `scripts/fitting/fit_mle.py` directly (K bounds + version stamp propagate), no hardcoded capacity bounds or version strings. `--compute-diagnostics` flag is on.

2. **Comparison script compatible:** `scripts/14_compare_models.py` uses `pd.read_csv` passthrough — extra `parameterization_version` column is loaded but never touched. No KeyError risk, no numeric aggregation contamination. Does NOT use `load_fits_with_validation()` — correct for transition period.

## Task Status

| Task | Name | Status | Notes |
|------|------|--------|-------|
| 1 | Verify SLURM + comparison script compatibility | Complete | Read-only, no commits |
| 2 | Execute cluster refit and verification | PENDING | Checkpoint:human-action — cluster jobs submitted |

## Pending Cluster Actions

When cluster results are available, report:
1. Did all 7 models refit with parameterization_version?
2. K recovery r for M3 and M6b (target: r >= 0.50)?
3. M4 GPU wall time for N=154 (target: < 12h)?
4. Is M6b still rank 1 on AIC?

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| No `load_fits_with_validation` in compare script | Script must handle mixed-vintage CSVs during transition; validation belongs on single-model downstream consumers |
| Keep `--compute-diagnostics` on | Hessian metrics useful for Phase 18 MLE-vs-Bayesian comparison |
| No SLURM script changes | Code changes in mle_utils.py + fit_mle.py propagate transparently |

## Deviations from Plan

None — Task 1 executed as specified. Task 2 deferred by user request (cluster jobs running).

---
*Phase: 14-collins-k-refit-gpu-lba-batching*
*Task 1 completed: 2026-04-12*
*Task 2: pending cluster refit*
