---
phase: 18-integration-comparison-manuscript
plan: 02
subsystem: model-comparison
tags: [arviz, bayesian, LOO, WAIC, stacking-weights, M4-LBA, Pareto-k]

# Dependency graph
requires:
  - phase: 17-m4-hierarchical-lba
    provides: wmrl_m4_posterior.nc with pointwise log-likelihood for Pareto-k LOO
  - phase: 16-choice-only-family-extension-subscale-level-2
    provides: choice-only model posteriors (M1/M2/M3/M5/M6a/M6b) for az.compare

provides:
  - CSV stacking-weight table (stacking_weights.csv) alongside existing Markdown
  - WAIC secondary metric per model written to waic_summary.csv
  - M4 separate comparison track with Pareto-k gating and m4_comparison.csv
  - Complete --bayesian-comparison mode (CMP-01 through CMP-04)

affects:
  - 18-04-PLAN (MODEL_REFERENCE.md update — LOO/WAIC comparison section)
  - 18-05-PLAN (manuscript revision — Bayesian comparison results)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "M4 separate track pattern: load independently, gate on Pareto-k > 5%, write m4_comparison.csv, never enter az.compare dict"
    - "Dual metric pattern: LOO primary (az.compare stacking), WAIC secondary (az.waic loop)"
    - "Machine-readable companion: every .md output has a parallel .csv for downstream scripting"

key-files:
  created: []
  modified:
    - scripts/14_compare_models.py

key-decisions:
  - "WAIC computed per-model via az.waic() loop; not via az.compare (LOO remains primary)"
  - "M4 Pareto-k threshold: 5% (conservative; near-certain to trigger fallback in production)"
  - "m4_comparison.csv single-row DataFrame with elpd_loo, pareto_k_pct, loo_reliable, elpd_waic, incommensurable_with_choice_only"
  - "WAIC section appended to stacking_weights.md only when waic_results non-empty (graceful skip on error)"

patterns-established:
  - "CMP-04 artifact layout: output/bayesian/level2/stacking_weights.csv + stacking_weights.md + waic_summary.csv + m4_comparison.csv"

# Metrics
duration: 5min
completed: 2026-04-13
---

# Phase 18 Plan 02: Bayesian Comparison Extensions Summary

**az.compare stacking weights + WAIC secondary metric + M4 Pareto-k separate track, all writing to output/bayesian/level2/ with CSV+Markdown dual output**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-04-13T18:53:59Z
- **Completed:** 2026-04-13T18:58:36Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- CMP-01 already implemented; added CSV companion (`stacking_weights.csv`) for machine-readable downstream consumption
- CMP-03: WAIC computed per model via `az.waic()` loop; `waic_summary.csv` written; WAIC section appended to Markdown
- CMP-02: M4 separate track with Pareto-k gating — loads `wmrl_m4_posterior.nc` independently, computes LOO pointwise, gates on `k > 0.7` fraction, writes `m4_comparison.csv`; M4 never enters `compare_dict` fed to `az.compare`
- CMP-04: complete: `--bayesian-comparison` now produces 4 files (stacking_weights.md, stacking_weights.csv, waic_summary.csv, m4_comparison.csv); default MLE mode untouched

## Task Commits

Each task was committed atomically:

1. **Task 1 + 2: CSV output, WAIC metric, M4 separate track** - `bdd5458` (feat)

**Plan metadata:** (will be added in final commit)

## Files Created/Modified

- `scripts/14_compare_models.py` - Extended `run_bayesian_comparison()` with CSV output, WAIC loop, and M4 separate track

## Decisions Made

- Tasks 1 and 2 both modify `run_bayesian_comparison()` in sequence; committed as a single atomic feat commit since the full function modification was one logical unit
- M4 Pareto-k fallback threshold is 5% (per plan spec); near-certain to trigger in production given LBA joint likelihood structure
- `waic_val_m4` initialized to `float("nan")` when WAIC fails so `m4_comparison.csv` always writes a row (no KeyError downstream)

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- 18-02 complete: CMP-01..04 all satisfied
- 18-03 (deprecation + reliability plots) can proceed
- 18-04 (MODEL_REFERENCE.md update) needs the LOO/WAIC/M4 section documented
- 18-05 (manuscript revision) needs Bayesian comparison results from cluster job; blocked on cluster execution

---
*Phase: 18-integration-comparison-manuscript*
*Completed: 2026-04-13*
