---
phase: 16-choice-only-family-extension-subscale-level-2
plan: "06"
subsystem: fitting
tags: [numpyro, jax, permutation-test, slurm, bayesian, mcmc, lec-covariate, l2-regression]

# Dependency graph
requires:
  - phase: 16-04
    provides: _fit_stacked_model helper, _load_lec_covariate, STACKED_MODEL_DISPATCH
  - phase: 15-02
    provides: run_inference_with_bump, wmrl_m3_hierarchical_model with beta_lec_kappa site

provides:
  - "--permutation-shuffle INT CLI argument in fit_bayesian.py"
  - "_run_permutation_shuffle() helper with deterministic np.random.default_rng(shuffle_idx)"
  - "cluster/13_bayesian_permutation.slurm SLURM array job (--array=0-49)"
  - "scripts/fitting/aggregate_permutation_results.py with PASS/FAIL verdict at 5% alpha"

affects:
  - "Phase 18: Integration — permutation summary is L2-06 validation gate for manuscript"
  - "cluster submission: 50 tasks x 4h = 200 CPU-hours needed"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Permutation path short-circuits before save_results(): no CSV/NetCDF/shrinkage written"
    - "np.random.default_rng(shuffle_idx) for deterministic per-shuffle permutation RNG"
    - "az.hdi() on flat beta_arr for 95% HDI excludes-zero test"
    - "JSON compact output per shuffle, aggregated post-hoc via standalone script"

key-files:
  created:
    - cluster/13_bayesian_permutation.slurm
    - scripts/fitting/aggregate_permutation_results.py
  modified:
    - scripts/fitting/fit_bayesian.py

key-decisions:
  - "Permutation shuffles participant-level LEC covariate labels (not within-participant data)"
  - "Reduced MCMC budget hardcoded in SLURM (warmup=500/samples=1000); CLI flags allow override"
  - "JSON output per shuffle (not full CSV/NetCDF) — compact format sufficient for FPR check"
  - "parser.error() guard: --permutation-shuffle only accepted with --model wmrl_m3"
  - "SLURM uses %A_%a log naming (master job ID + array task ID)"

patterns-established:
  - "Permutation test pattern: shuffle covariate with np.random.default_rng(idx), run reduced MCMC, save JSON"
  - "Aggregation script pattern: glob JSONs, count HDI-excludes-zero, write Markdown PASS/FAIL verdict"

# Metrics
duration: 5min
completed: 2026-04-13
---

# Phase 16 Plan 06: Permutation Null Test Infrastructure Summary

**Permutation null test infrastructure for M3 LEC covariate (L2-06): --permutation-shuffle CLI arg, 50-task SLURM array job, and JSON aggregation script with 5% FPR PASS/FAIL verdict**

## Performance

- **Duration:** 5 min
- **Started:** 2026-04-13T08:46:09Z
- **Completed:** 2026-04-13T08:51:39Z
- **Tasks:** 2
- **Files modified:** 3 (1 modified, 2 created)

## Accomplishments

- Added `--permutation-shuffle INT` to fit_bayesian.py with wmrl_m3-only guard and `_run_permutation_shuffle()` helper that deterministically shuffles LEC participant labels via `np.random.default_rng(shuffle_idx)`, runs reduced MCMC, extracts `beta_lec_kappa` 95% HDI, and saves compact JSON to `output/bayesian/permutation/`
- Created `cluster/13_bayesian_permutation.slurm` as a 50-task array job (`--array=0-49`) with 4h time limit, half-budget warmup/samples, and correct `$SLURM_ARRAY_TASK_ID` plumbing
- Created `scripts/fitting/aggregate_permutation_results.py` that globs JSON results, counts HDI-excludes-zero hits, and writes a Markdown summary with per-shuffle table and explicit PASS/FAIL verdict at 5% alpha

## Task Commits

1. **Task 1: Add --permutation-shuffle CLI arg and _run_permutation_shuffle** - `8d6f68f` (feat)
2. **Task 2: Create SLURM array job and aggregation script** - `62bb898` (feat)

**Plan metadata:** (created below)

## Files Created/Modified

- `scripts/fitting/fit_bayesian.py` - Added `--permutation-shuffle INT` arg, validation guard, `_run_permutation_shuffle()` function, `import json` + `import time`
- `cluster/13_bayesian_permutation.slurm` - SLURM array job: `--array=0-49`, 4h/32G/4CPU, passes `$SLURM_ARRAY_TASK_ID` as `--permutation-shuffle`
- `scripts/fitting/aggregate_permutation_results.py` - Standalone aggregation: loads shuffle JSONs, PASS/FAIL at 5% FPR, writes `permutation_summary.md`

## Decisions Made

- **Permutation shuffles covariate labels only:** `np.random.default_rng(shuffle_idx).permutation(covariate_lec)` randomizes participant-level LEC alignment while keeping all behavioral data intact. This tests the null that the kappa-LEC association is above chance.
- **Reduced MCMC budget in SLURM:** `--warmup 500 --samples 1000` (half of standard). Sufficient for a HDI-excludes-zero check; each shuffle does not need convergence-grade posterior for FPR estimation. Wall time per task ≤ 4h.
- **JSON-only output per shuffle:** No CSV/NetCDF/shrinkage written for permutation runs. The `beta_lec_kappa` mean, std, HDI bounds, `excludes_zero` flag, divergences, and wall_clock_seconds are all the information needed downstream.
- **`parser.error()` guard at validation time:** The guard runs before data loading so the error is immediate; avoids 10+ minutes of MCMC before discovering wrong model.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required. Cluster submission is a manual step:

```bash
sbatch cluster/13_bayesian_permutation.slurm
# After all 50 tasks complete:
python scripts/fitting/aggregate_permutation_results.py
```

## Next Phase Readiness

- Permutation infrastructure is ready for cluster submission (L2-06 requirement)
- After 50 shuffles complete, `aggregate_permutation_results.py` provides the PASS/FAIL verdict for the manuscript
- No blockers for remaining Phase 16 plans

---
*Phase: 16-choice-only-family-extension-subscale-level-2*
*Completed: 2026-04-13*
