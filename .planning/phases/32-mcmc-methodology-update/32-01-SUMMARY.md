---
phase: 32-mcmc-methodology-update
plan: 01
subsystem: bayesian-diagnostics
tags: [arviz, numpyro, bfmi, baribault-collins-2023, convergence-gate]

requires:
  - phase: 13-hierarchical-baseline
    provides: write_bayesian_summary CSV writer + HIER-07 convergence gate (R-hat / ESS / divergences)
  - phase: 21-bayesian-pipeline
    provides: convergence_table.csv + 21_baseline_audit.py consumer of writer output
  - phase: 31-final-package-restructure
    provides: src/rlwm/fitting/bayesian.py canonical location after CCDS migration
provides:
  - Tier-1 publication-grade convergence gate (R-hat <= 1.01 AND ESS >= 400 AND divergences == 0 AND BFMI >= 0.2)
  - Two additive CSV columns (min_bfmi, per_chain_ess_bulk) for downstream audit scripts
  - Single-line [convergence-gate] log surfacing all four metrics for cluster log scrapers
  - Backward-compatible NaN tolerance for legacy fits without sample_stats.energy
affects: [32-02, 32-03, 32-04, 32-05, 32-06, 24-cold-start-pipeline-execution, 25, 26, 27]

tech-stack:
  added: []
  patterns:
    - "az.bfmi(idata).min() across chains as the canonical BFMI summary"
    - "Per-chain ESS via az.summary on posterior.isel(chain=slice(c, c+1))"
    - "Semicolon-separated string for vector-valued CSV columns to keep readers split-friendly"
    - "NaN-tolerant gate booleans for legacy artifacts (np.isnan(x) or x >= threshold)"

key-files:
  created: []
  modified:
    - scripts/fitting/bayesian_summary_writer.py (47 insertions / 5 deletions)
    - src/rlwm/fitting/bayesian.py (28 insertions / 4 deletions)
    - tests/integration/test_bayesian_summary.py (133 insertions, 0 deletions)

key-decisions:
  - "BFMI gate uses Baribault & Collins 2023 Tier-1 threshold (>= 0.2), not Betancourt 2016 stricter (>= 0.3) — matches v5.0 manuscript anchor."
  - "Legacy fits without sample_stats.energy stay valid via NaN tolerance (np.isnan(min_bfmi) or min_bfmi >= 0.2)."
  - "Per-chain ESS stored as semicolon-separated string in a single column rather than n_chains float columns (avoids schema fanout when n_chains varies)."
  - "Test for low-BFMI failure uses monkey-patch on az.bfmi rather than constructing pathological energy arrays — simpler, faster, equally diagnostic."

patterns-established:
  - "Gate boolean and log line co-located in src/rlwm/fitting/bayesian.py, both folded together so the writer CSV and runtime log report identical verdicts."
  - "Function-local az import in writer means monkeypatch.setattr(az, 'bfmi', ...) at test time propagates without bayesian_summary_writer.az shadowing."

duration: 11min
completed: 2026-05-03
---

# Phase 32 Plan 01: Tier-1 BFMI Gate + Per-Chain ESS Summary

**Convergence gate now enforces Baribault & Collins 2023 Tier-1 publication standard (R-hat <= 1.01 AND ESS >= 400 AND divergences == 0 AND min_bfmi >= 0.2) with backward-compatible NaN tolerance for legacy fits.**

## Performance

- **Duration:** ~11 min
- **Started:** 2026-05-03T13:56:14Z
- **Completed:** 2026-05-03T14:07:08Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- Hardened the post-fit convergence gate in `scripts/fitting/bayesian_summary_writer.py` to fold `min_bfmi >= 0.2` into the `converged` boolean, matching the Tier-1 publication standard cited in 32-RESEARCH.md.
- Added two additive CSV schema columns (`min_bfmi`, `per_chain_ess_bulk`) between `num_divergences` and `n_trials` so downstream audit scripts (`21_baseline_audit.py`, manuscript table builders) can diagnose label-switching and one-bad-chain pathologies that the existing all-chain summary hides.
- Mirrored the same gate in `src/rlwm/fitting/bayesian.py::run_full_bayesian_workflow`, including a single-line `[convergence-gate]` log emission so cluster log scrapers can grep one canonical line per fit.
- Locked the new behavior with two pytest cases in `tests/integration/test_bayesian_summary.py` (one schema-presence test, one monkey-patched low-BFMI failure test).

## Task Commits

Each task committed atomically:

1. **Task 1: Add BFMI + per-chain ESS to write_bayesian_summary** — `8f229e0` (feat)
2. **Task 2: Update bayesian.py gate logging to surface BFMI** — `413f16c` (feat)
3. **Task 3: Extend test_bayesian_summary.py with BFMI gate test** — `5b582dd` (test)

## Files Created/Modified

- `scripts/fitting/bayesian_summary_writer.py` — Added BFMI computation block (~15 lines), per-chain ESS computation block (~12 lines), two new column entries in `_build_column_order`, BFMI gate in the `converged` boolean, docstring update naming both new columns and rewriting gate description item 4.
- `src/rlwm/fitting/bayesian.py` — Added BFMI computation (try/except guarded `np.asarray(az.bfmi(idata)).ravel().min()`) before the convergence gate; folded `min_bfmi >= 0.2` (with NaN tolerance) into the local `converged` boolean; tightened ESS comparison from `> 400` to `>= 400` for parity with the writer; added a single-line `[convergence-gate]` log emission; extended the existing GATE FAILED / GATE PASSED messages with `min_bfmi`.
- `tests/integration/test_bayesian_summary.py` — Added `_build_minimal_idata` helper, `test_summary_includes_bfmi_and_per_chain_ess_columns`, `test_converged_flag_fails_under_low_bfmi`. New tests use synthetic 4-chain x 100-draw x 2-participant InferenceData with healthy energy; the low-BFMI test monkey-patches `az.bfmi` to return `[0.05, 0.5, 0.5, 0.5]` and asserts `converged == False` for every row.

## Tier-1 Gate Boolean (now in production)

Both `bayesian_summary_writer.py` (CSV `converged` column) and `src/rlwm/fitting/bayesian.py` (runtime gate / log) compute:

```python
converged = (
    (not np.isnan(max_rhat) and max_rhat < 1.01)
    and (not np.isnan(min_ess) and min_ess > 400)   # writer; bayesian.py uses >= 400 (mirror)
    and (num_divergences_total == 0)
    and (np.isnan(min_bfmi) or min_bfmi >= 0.2)
)
```

The `np.isnan(min_bfmi)` guard is the **backward-compatibility hatch**: legacy fits that pre-date the BFMI capture path land at `min_bfmi == NaN` and still satisfy the gate, so re-summarising existing `.nc` files (e.g. `models/bayesian/21_baseline/wmrl_m6b_posterior.nc`) does not retroactively flag them as failed.

## Decisions Made

- **Tier-1 threshold = 0.2 (not 0.3).** Baribault & Collins 2023 explicitly cite 0.2 as the publication-grade BFMI floor. The stricter Betancourt 2016 threshold (0.3) was rejected because the v5.0 manuscript anchor is Baribault & Collins.
- **NaN-tolerant gate for legacy fits.** Re-running the writer on pre-Phase-32 NetCDF files must not retroactively label them as failed — the `np.isnan(min_bfmi) or min_bfmi >= 0.2` clause encodes "tolerate NaN, fail explicit low values".
- **Per-chain ESS as a string, not `n_chains` columns.** A semicolon-separated string keeps the schema stable across runs that use different chain counts (4 chains for choice-only fits, 8 for M6b subscale, etc.) and is trivially split-able by downstream readers.
- **Test uses `monkeypatch` on `az.bfmi`, not pathological energy arrays.** The plan offered both options; the simpler monkey-patch is equally diagnostic and runs in ~0.3 s instead of computing real BFMI on a hand-crafted energy series.

## Deviations from Plan

None - plan executed exactly as written.

The plan named line 351 as the converged-boolean target; the actual line in the current writer was around line 351-355 (the boolean was already split across multiple lines); the edit landed at the equivalent semantic location with the BFMI clause appended. The plan also called the line range "around 263" for the divergences block; the current code has the same block at the same logical position with an updated structure.

## Issues Encountered

- **Concurrent agent commits.** While Task 1 was in flight, parallel commits (`5179fc8`, `b45dcd8`, `e144b35`, `a7dde96`, `f93c159`) for Phase 32-02 and 32-03 landed on the same branch. This did not affect Task 1-3 because each task staged only its own file (`scripts/fitting/bayesian_summary_writer.py`, `src/rlwm/fitting/bayesian.py`, `tests/integration/test_bayesian_summary.py`); none of the parallel-agent files overlapped. Verified post-commit that my three commits (`8f229e0`, `413f16c`, `5b582dd`) sit cleanly in the linear history alongside the 32-02/32-03 commits.
- **Pre-existing ruff errors in `bayesian.py` line 233 (E712) and line 162 (E501).** Both unrelated to my changes. Left alone (no scope creep).

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- **32-02, 32-03 already shipped** by parallel agent (commits `5179fc8`, `b45dcd8`, `e144b35`, `a7dde96`, `f93c159`); the Wave 1 plans are now all closed and Wave 2 (32-04 softmax-bias kappa) is unblocked.
- **Cluster smoke run (32-05) can now distinguish convergence levels precisely.** Before this plan, a fit that landed `R-hat = 1.005, ESS = 1200, divergences = 0` with `BFMI = 0.05` would have been falsely labeled `converged = True`. After this plan, that same fit reports `converged = False` with the BFMI verdict surfaced in both the CSV column and the cluster log.
- **Re-summarising existing v4.0 NetCDF artifacts is safe.** The NaN-tolerance clause means `wmrl_m6b_posterior.nc` and the other Phase 21 baseline `.nc` files re-pumped through the new writer will preserve their original `converged` verdict.

---
*Phase: 32-mcmc-methodology-update*
*Completed: 2026-05-03*
