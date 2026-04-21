---
phase: 21-principled-bayesian-model-selection-pipeline
plan: 04
subsystem: infra
tags: [bayesian, hierarchical, numpyro, mcmc, slurm, output-routing, fit-bayesian-patch]

# Dependency graph
requires:
  - phase: 21-01
    provides: "Prior-predictive runner pattern (JAX_PLATFORMS=cpu, ds_env conda ladder, autopush hook)"
  - phase: 21-03
    provides: "SLURM + conda ladder template + convergence-gate-surface pattern"
  - phase: 16
    provides: "STACKED_MODEL_DISPATCH, save_results convergence gate, _fit_stacked_model"
provides:
  - "scripts/21_fit_baseline.py — thin wrapper forcing --output-subdir 21_baseline + load-bearing exit-1 shim"
  - "cluster/21_3_fit_baseline.slurm — 10h/48G/4-CPU per-MODEL submission template"
  - "scripts/fitting/fit_bayesian.py — save_results + main() patched with output_subdir kwarg/CLI flag"
  - "scripts/fitting/bayesian_diagnostics.py — run_posterior_predictive_check patched with ppc_output_dir kwarg (Approach A)"
  - "scripts/fitting/bayesian_summary_writer.py — write_bayesian_summary patched with output_subdir kwarg"
  - "Backward-compat: Phase 16 callers (no --output-subdir flag) still write to output/bayesian/ root"
affects: [21-05-convergence-audit, 21-06-loo-stacking, 21-07-winner-l2-refit, 21-10-master-orchestrator]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Keyword-only output_subdir argument plumbed through 3 modules (save_results -> write_bayesian_summary / run_posterior_predictive_check)"
    - "Thin orchestrator over fit_bayesian.main() via sys.argv rewrite (no MCMC re-implementation)"
    - "Load-bearing post-fit expected.exists() shim turns silent convergence-gate None-return into SLURM exit 1"
    - "Per-MODEL JAX compilation cache (cross-model isolation, cross-run warm-start)"

key-files:
  created:
    - "scripts/21_fit_baseline.py"
    - "cluster/21_3_fit_baseline.slurm"
  modified:
    - "scripts/fitting/fit_bayesian.py (save_results +output_subdir, main() +--output-subdir CLI flag, PPC call passes ppc_output_dir=bayesian_dir)"
    - "scripts/fitting/bayesian_diagnostics.py (run_posterior_predictive_check +ppc_output_dir kwarg)"
    - "scripts/fitting/bayesian_summary_writer.py (write_bayesian_summary +output_subdir kwarg)"

key-decisions:
  - "Approach A for PPC routing: add ppc_output_dir kwarg to run_posterior_predictive_check, preserving backward compat when None (not Approach B full refactor)"
  - "sys.argv rewrite over argparse-pass-through: the 21_fit_baseline.py wrapper replaces sys.argv and calls fit_bayesian.main() directly — no MCMC re-implementation, no ArgumentParser inheritance complexity"
  - "Load-bearing convergence-gate shim: explicit expected.exists() check + sys.exit(1) after fit_main() returns — converts silent gate-fail None-return into real SLURM failure"
  - "10h SLURM ceiling absorbs compile + auto-bump retries; M1 ~2h, M2-M5 ~4-6h, M6b ~6-8h"
  - "Per-MODEL JAX cache (.jax_cache_21_baseline_\${MODEL}) allows 6 parallel SLURM jobs without cache-dir collisions"

patterns-established:
  - "Pattern: output-subdir plumbing through the Bayesian stack (save_results -> write_bayesian_summary / run_posterior_predictive_check) is the canonical way to add a new output namespace without overwriting Phase 16 artifacts"
  - "Pattern: thin wrappers over fit_bayesian.main() (21_fit_baseline.py model) should force the output_subdir via sys.argv rewrite and verify artifacts via post-fit exists-check"

# Metrics
duration: ~45min (Task 1 on cluster environment; Task 2 files landed; orchestrator completed Task 2 commit + SUMMARY after executor rate-limit)
completed: 2026-04-18
---

# Phase 21 Plan 04: Baseline Hierarchical Fit Runner (NO L2) Summary

**Thin orchestrator over fit_bayesian.main() routing all 6 choice-only model baseline fits to output/bayesian/21_baseline/ — patches 3 Bayesian modules with output_subdir plumbing + adds load-bearing convergence-gate-surface shim so SLURM --dependency=afterok chains cannot silently skip gate failures.**

## Performance

- **Duration:** ~45 min (Task 1 ~30 min executor work; Task 2 ~10 min executor; ~5 min orchestrator cleanup after executor rate-limit)
- **Started:** 2026-04-18T17:00:00Z (approx, executor spawn time)
- **Completed:** 2026-04-18T (orchestrator finish)
- **Tasks:** 2/2
- **Files created:** 2
- **Files modified:** 3 (fit_bayesian.py, bayesian_diagnostics.py, bayesian_summary_writer.py) + STATE.md

## Accomplishments

- `save_results` in `scripts/fitting/fit_bayesian.py` now accepts `output_subdir: str | None = None` (keyword-only, default None for Phase 16 backward-compat). When set, routes `{model}_posterior.nc`, `{model}_individual_fits.csv`, `{model}_shrinkage_report.md`, `{model}_ppc_results.csv` to `output/bayesian/<output_subdir>/` instead of the legacy `output/bayesian/` root.
- `run_posterior_predictive_check` in `scripts/fitting/bayesian_diagnostics.py` gained keyword-only `ppc_output_dir: Path | None = None` (Approach A). When None, falls back to `output_dir / "bayesian"` — preserves Phase 16 behavior. `save_results` calls it with `ppc_output_dir=bayesian_dir` so the PPC CSV tracks the posterior NetCDF subdir.
- `write_bayesian_summary` in `scripts/fitting/bayesian_summary_writer.py` also gained `output_subdir` to route the per-participant fit CSV consistently.
- `fit_bayesian.py main()` gained `--output-subdir OUTPUT_SUBDIR` CLI flag (default None) — threaded into `save_results`.
- `scripts/21_fit_baseline.py` (225 lines): thin argparse wrapper requiring `--model` from 6 choices (qlearning, wmrl, wmrl_m3, wmrl_m5, wmrl_m6a, wmrl_m6b); rewrites `sys.argv` with `--output-subdir 21_baseline` forced; calls `fit_bayesian.main()`; checks `output/bayesian/21_baseline/{model}_posterior.nc` exists post-fit; prints `[CONVERGENCE GATE FAIL]` + `sys.exit(1)` if missing. Load-bearing: without this, `save_results` returning None on gate failure leaves SLURM at exit 0 and downstream dependency chains silently skip.
- `cluster/21_3_fit_baseline.slurm` (159 lines): 10h/48G/4-CPU/comp partition submission; MODEL/WARMUP/SAMPLES/CHAINS/SEED/MAX_TREE_DEPTH env vars; ds_env → `/scratch/fc37` conda ladder; per-MODEL `JAX_COMPILATION_CACHE_DIR`; `NUMPYRO_HOST_DEVICE_COUNT=4`, `JAX_PLATFORMS=cpu`; post-run expected-artifacts echo; `source cluster/autopush.sh`; `exit $EXIT_CODE`. Submit 6 times in parallel (cross-model isolation via subdir).

## Task Commits

1. **Task 1: Patch save_results + run_posterior_predictive_check + write_bayesian_summary with output_subdir plumbing + CLI flag** — `1b39606` (feat)
2. **Task 2: 21_fit_baseline.py thin orchestrator + 21_3_fit_baseline.slurm cluster submission** — `8d27407` (feat)

_Note: Task 2 was executed and files landed on disk before the spawning agent hit its rate limit; the orchestrator committed the files, wrote this SUMMARY, and updated STATE.md._

## Files Created/Modified

**Created:**
- `scripts/21_fit_baseline.py` — thin wrapper + convergence-gate shim
- `cluster/21_3_fit_baseline.slurm` — per-MODEL SLURM submission

**Modified:**
- `scripts/fitting/fit_bayesian.py` — `save_results` (+output_subdir, line ~577), `main()` (+--output-subdir CLI, line ~1097), PPC call (pass ppc_output_dir=bayesian_dir, line ~776)
- `scripts/fitting/bayesian_diagnostics.py` — `run_posterior_predictive_check` (+ppc_output_dir, line ~645)
- `scripts/fitting/bayesian_summary_writer.py` — `write_bayesian_summary` (+output_subdir, line ~155)

## Decisions Made

- **Approach A for PPC routing (preferred over Approach B):** Added `ppc_output_dir` kwarg to `run_posterior_predictive_check` rather than refactoring the function to use a pre-computed path. Backward-compat is automatic when None; Phase 16 callers unchanged.
- **`sys.argv` rewrite in 21_fit_baseline.py:** Simpler than ArgumentParser inheritance or parameter pass-through. The baseline fit is a constrained subset of fit_bayesian.main() — the wrapper documents intent and enforces the 21_baseline routing at a single point.
- **Load-bearing convergence-gate shim:** Explicit `expected.exists()` + `sys.exit(1)` check after fit_main() — not a nice-to-have. `save_results` returns None on gate failure (R-hat/ESS/divergences), which makes fit_main() exit 0. Without the shim, SLURM would see success and downstream `--dependency=afterok` chains would continue. This is the difference between "pipeline blocks on failed convergence" (Phase 21 SC #2) and "pipeline silently skips a failed model."
- **10h SLURM ceiling:** Absorbs first-compile time (~5-15 min on CPU for M6b fully-batched vmap) + 3 auto-bump retries (target_accept 0.80 → 0.95 → 0.99). Measured Phase 16 wall times were M1 ~2h, M2-M5 ~4-6h, M6b ~6-8h. 10h gives margin.
- **Per-MODEL JAX cache directory:** 6 models submitted in parallel can warm their own caches without collision. Phase 16 used per-model directories for the same reason.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

- **Executor hit user-level rate limit mid-Task-2:** After Task 1 committed (`1b39606`) and Task 2 files landed on disk (uncommitted), the spawning agent received "You've hit your limit · resets 6pm (Europe/Berlin)" and returned before committing Task 2, writing SUMMARY.md, or updating STATE.md. Orchestrator verified the on-disk Task 2 artifacts matched the plan spec (argparse, 6-model choice list, `--output-subdir 21_baseline` forced via sys.argv rewrite, load-bearing exit-1 shim, SLURM header with 10h/48G/4-CPU, env-var parametrization, ds_env conda ladder, per-MODEL JAX cache, autopush, exit code forwarding), then committed Task 2 (`8d27407`), wrote this SUMMARY, and updated STATE.md. No re-spawn of a fresh executor was done because it would have risked reverting Task 1's landed `save_results`/`run_posterior_predictive_check`/`write_bayesian_summary` modifications.

## Verification

- [x] `python -c "from scripts.fitting.fit_bayesian import save_results; import inspect; assert 'output_subdir' in inspect.signature(save_results).parameters"` passes
- [x] `python -c "from scripts.fitting.bayesian_diagnostics import run_posterior_predictive_check; import inspect; assert 'ppc_output_dir' in inspect.signature(run_posterior_predictive_check).parameters"` passes
- [x] `python -c "from scripts.fitting.bayesian_summary_writer import write_bayesian_summary; import inspect; assert 'output_subdir' in inspect.signature(write_bayesian_summary).parameters"` passes
- [x] `python scripts/fitting/fit_bayesian.py --help | grep output-subdir` shows the flag
- [x] `python scripts/21_fit_baseline.py --help` shows all 6 model choices + default budget flags
- [x] `grep "21_baseline" cluster/21_3_fit_baseline.slurm` returns ≥1 match
- [x] `grep "output_subdir\|ppc_output_dir" scripts/fitting/fit_bayesian.py scripts/fitting/bayesian_diagnostics.py scripts/fitting/bayesian_summary_writer.py` shows all 3 files patched

## Next Phase Readiness

- **Unblocks 21-05 (convergence + PPC audit):** Now reads `output/bayesian/21_baseline/{model}_posterior.nc` + `{model}_ppc_results.csv` without collision with Phase 16 outputs.
- **Unblocks 21-06 (LOO + stacking + RFX-BMS):** Has 6 non-circular baseline posteriors to rank.
- **Unblocks 21-07 (winner L2 refit):** Will re-submit with `--output-subdir 21_l2` and force `covariate_lec` / `covariate_iesr` based on winner(s).
- **Unblocks 21-10 (master orchestrator):** All upstream SLURM scripts now share the `output/bayesian/21_*` subdirectory convention.

---
*Phase: 21-principled-bayesian-model-selection-pipeline*
*Completed: 2026-04-18*
