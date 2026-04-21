---
phase: 19-associative-scan-likelihood
plan: 03
subsystem: fitting
tags: [jax, pscan, numpyro, benchmark, slurm, gpu, a100, mcmc, bayesian]

# Dependency graph
requires:
  - phase: 19-associative-scan-likelihood (plan 02)
    provides: 12 pscan multiblock likelihood functions for all 6 choice-only models
  - phase: 15-m3-hierarchical-poc
    provides: hierarchical model functions in numpyro_models.py, fit_bayesian.py pipeline
  - phase: 16-choice-only-family-extension
    provides: 6 hierarchical model functions (M1-M6b) in numpyro_models.py
provides:
  - "--use-pscan CLI flag in fit_bayesian.py for transparent sequential/pscan switching"
  - "use_pscan kwarg on all 7 hierarchical model functions (6 standard + subscale)"
  - "Micro-benchmark script (validation/benchmark_parallel_scan.py) for per-call timing"
  - "GPU SLURM job (cluster/13_bayesian_pscan.slurm) with 3-stage A/B protocol"
  - "PSCAN-06 automated comparison: group-mean rel error < 5%, WAIC/LOO diff < 1.0"
affects:
  - Phase 20 (DEER): will use the same pscan flag and benchmarking infrastructure
  - Future GPU cluster runs: sbatch cluster/13_bayesian_pscan.slurm

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "use_pscan=False default everywhere: opt-in parallel scan, backward-compatible"
    - "Likelihood function dispatch via local variable: _lik_fn = pscan if use_pscan else sequential"
    - "pscan output file suffix: _pscan appended to NetCDF filename for A/B comparison"
    - "3-stage SLURM benchmark: micro-bench -> sequential MCMC -> pscan MCMC -> posterior comparison"

key-files:
  created:
    - validation/benchmark_parallel_scan.py
    - cluster/13_bayesian_pscan.slurm
  modified:
    - scripts/fitting/numpyro_models.py
    - scripts/fitting/fit_bayesian.py

key-decisions:
  - "use_pscan kwarg passed through model_args dict (mcmc.run(**model_args)), not as a separate mechanism: simplest integration with run_inference_with_bump"
  - "NetCDF pscan outputs use _pscan filename suffix rather than separate directory: keeps A/B files adjacent for easy comparison"
  - "Benchmark calls likelihood functions directly (no outer jax.jit wrapper) because return_pointwise Python branching is not JIT-compatible"
  - "CPU pscan ~3.7x slower than sequential (expected): associative scan has higher constant factor but O(log T) depth; speedup requires GPU parallel hardware"

patterns-established:
  - "All hierarchical model functions accept use_pscan: bool = False as last keyword argument before **kwargs"
  - "Pscan dispatch uses conditional assignment: _lik_fn = pscan_variant if use_pscan else sequential_variant"
  - "Benchmark JSON at output/bayesian/pscan_benchmark.json with device, timing, agreement, compilation metrics"

# Metrics
duration: 31min
completed: 2026-04-14
---

# Phase 19 Plan 03: PScan Integration, Benchmarking, and A/B Protocol Summary

**--use-pscan flag wired into Bayesian pipeline for all 6 choice-only models with micro-benchmark, GPU SLURM job, and automated PSCAN-06 posterior comparison protocol**

## Performance

- **Duration:** 31 min
- **Started:** 2026-04-14T09:12:17Z
- **Completed:** 2026-04-14T09:42:58Z
- **Tasks:** 2
- **Files modified:** 4 (2 modified, 2 created)

## Accomplishments
- `--use-pscan` CLI flag in `fit_bayesian.py` transparently switches all 6 choice-only models from sequential `lax.scan` to O(log T) `associative_scan` likelihoods
- `use_pscan` kwarg added to all 7 hierarchical model functions (6 standard + M6b subscale) with `False` default preserving backward compatibility
- Micro-benchmark validates NLL agreement (0.0 relative error on M3 synthetic data) and measures per-call timing
- GPU SLURM script implements complete 3-stage A/B benchmark protocol (PSCAN-05 + PSCAN-06)

## Task Commits

1. **Task 1: Add use_pscan kwarg to numpyro models and --use-pscan CLI flag** - `bd6d635` (feat)
2. **Task 2: Create micro-benchmark and GPU SLURM script** - `8460e22` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified
- `scripts/fitting/numpyro_models.py` - Added 6 pscan likelihood imports; `use_pscan: bool = False` kwarg to 7 hierarchical model functions; conditional dispatch to pscan or sequential likelihood in each for-loop
- `scripts/fitting/fit_bayesian.py` - Added `--use-pscan` CLI flag; propagated through `fit_model` -> `_fit_stacked_model` -> `model_args`; `_pscan` suffix on NetCDF filename when enabled
- `validation/benchmark_parallel_scan.py` - Standalone micro-benchmark measuring sequential vs pscan wall time, JIT compilation overhead, NLL agreement; writes JSON to `output/bayesian/pscan_benchmark.json`
- `cluster/13_bayesian_pscan.slurm` - 3-stage GPU A100 SLURM job: micro-benchmark (Stage 1), sequential vs pscan MCMC (Stage 2), posterior comparison with PSCAN-06 criteria (Stage 3)

## Decisions Made

**model_args dict for use_pscan propagation:** Rather than modifying the NUTS kernel or MCMC object, `use_pscan` is passed as a regular model kwarg through the existing `model_args` dict that gets unpacked as `**model_args` in `mcmc.run()`. This is the simplest integration path since NumPyro passes all `run()` kwargs to the model function.

**No outer jax.jit in benchmark:** The likelihood functions use Python `if return_pointwise:` branching internally, which is incompatible with JIT tracing. The benchmark calls functions directly and relies on their internal JIT compilation (via `lax.fori_loop`/`lax.scan`). This is consistent with how the test suite and MCMC inference call these functions.

**CPU baseline as expected:** PScan is ~3.7x slower on CPU (825ms vs 222ms for M3 with 17 blocks x 100 trials). This is the expected behavior -- associative scan has 2x the work of sequential scan but O(log T) depth. Speedup requires GPU parallel execution units, which is exactly what the SLURM script benchmarks.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] JIT tracing error from outer jax.jit wrapper in benchmark**
- **Found during:** Task 2 (benchmark script execution)
- **Issue:** Initial benchmark wrapped likelihood functions in `jax.jit()`, but `return_pointwise=False` caused `TracerBoolConversionError` because Python `if` on a traced value is not JIT-compatible.
- **Fix:** Removed outer `jax.jit` wrapper; call functions directly (they use internal `lax.fori_loop`/`lax.scan` which handles JIT compilation).
- **Files modified:** validation/benchmark_parallel_scan.py
- **Verification:** Benchmark runs successfully, produces correct timing and NLL agreement
- **Committed in:** 8460e22 (part of Task 2 commit)

---

**Total deviations:** 1 auto-fixed (Rule 1 - Bug)
**Impact on plan:** Fix was necessary for benchmark to execute. No scope creep.

## Issues Encountered
None beyond the JIT tracing issue documented above.

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- `--use-pscan` flag is ready for GPU cluster testing via `sbatch cluster/13_bayesian_pscan.slurm`
- Phase 19 is complete (all 3 plans done): primitives (19-01), 12 pscan likelihoods (19-02), pipeline integration + benchmarking (19-03)
- GPU benchmark will determine whether pscan provides meaningful speedup for MCMC workloads
- PSCAN-06 posterior agreement protocol is fully automated in the SLURM script
- All existing sequential code paths are unchanged (backward compatible)

---
*Phase: 19-associative-scan-likelihood*
*Completed: 2026-04-14*
