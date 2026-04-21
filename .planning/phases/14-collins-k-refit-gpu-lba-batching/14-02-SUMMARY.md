---
phase: 14
plan: 02
subsystem: gpu-fitting
tags: [jax, m4, lba, float64, gpu, mle, smoke-test]
requires: ["14-01"]
provides: ["fit_all_gpu_m4 named wrapper", "M4 GPU smoke test", "m4_synthetic_data_small fixture"]
affects: ["14-03", "17-M4-hierarchical"]
tech-stack:
  added: []
  patterns: ["float64-before-jax pattern", "GPU M4 wrapper delegation pattern"]
key-files:
  created:
    - scripts/fitting/tests/test_gpu_m4.py
  modified:
    - scripts/fitting/fit_mle.py
    - scripts/fitting/tests/conftest.py
key-decisions:
  - "fit_all_gpu_m4 enables float64 FIRST (before any JAX ops), then triggers lba_likelihood import"
  - "Wrapper delegates entirely to fit_all_gpu(model='wmrl_m4') — no code duplication"
  - "Smoke test marked @pytest.mark.slow; n_starts=5 for fast JIT validation"
  - "conftest.py fixture uses simulate_wmrl_block + uniform RTs (smoke test, not parameter recovery)"
patterns-established:
  - "Named GPU entry point pattern: float64 enable -> lba_likelihood import -> delegate to fit_all_gpu"
duration: "6 min"
completed: "2026-04-12"
---

# Phase 14 Plan 02: GPU M4 Wrapper Summary

**One-liner:** Named `fit_all_gpu_m4` wrapper enforces float64 initialization order before delegating to `fit_all_gpu(model='wmrl_m4')`, with smoke test validating finite NLL and K in [2.0, 6.0].

## Performance

- Execution time: ~6 minutes
- Tasks completed: 2/2
- Test regressions: 0 (61/61 existing tests pass)
- New tests: 1 smoke test (marked `@pytest.mark.slow`)

## Accomplishments

1. **Task 1 - fit_all_gpu_m4 wrapper** (35f2173): Added named entry point in `fit_mle.py` after `fit_all_gpu`. The function sets `jax_enable_x64=True` as first statement, triggers lazy import of `lba_likelihood.wmrl_m4_multiblock_likelihood_stacked` to ensure float64 is globally active, then delegates to `fit_all_gpu(data=data, model="wmrl_m4", ...)`. NumPy-style docstring, Python 3.10+ type hints. Satisfies GPU-01 requirement.

2. **Task 2 - M4 fixture and smoke test** (c533a61): Added `m4_synthetic_data_small` fixture to `conftest.py` (5 participants x 3 blocks x 30 trials, with RT column from `np.random.default_rng`). Created `test_gpu_m4.py` with `test_fit_all_gpu_m4_smoke` validating: 5 participants returned, NLL finite for all, capacity in [2.0, 6.0]. Test is `@pytest.mark.slow` for skipping in fast CI.

## Task Commits

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Create fit_all_gpu_m4 wrapper | 35f2173 | scripts/fitting/fit_mle.py |
| 2 | Add M4 fixture and smoke test | c533a61 | scripts/fitting/tests/conftest.py, scripts/fitting/tests/test_gpu_m4.py |

## Files Modified

- `scripts/fitting/fit_mle.py` — Added `fit_all_gpu_m4` function (lines 1708-1761)
- `scripts/fitting/tests/conftest.py` — Added `m4_synthetic_data_small` fixture
- `scripts/fitting/tests/test_gpu_m4.py` — New file: smoke test for GPU M4 fitting path

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| float64 enable FIRST in wrapper | `jax_enable_x64` cannot be toggled after first JAX array materialises; wrapper must be called before any other JAX ops in process |
| lba_likelihood lazy import in wrapper | Mirrors `main()` pattern (line 2830); module-level `jax.config.update` in lba_likelihood reinforces float64 globally |
| Delegate to fit_all_gpu, no code duplication | M4 branch already complete inside fit_all_gpu; wrapper only adds the initialization sequence |
| n_starts=5 in smoke test | Smallest practical value for JIT compilation validation; parameter recovery not the goal here |
| Uniform RTs in fixture | Smoke test only needs positive finite RTs; exact distribution irrelevant for testing finite NLL assertion |

## Deviations from Plan

None — plan executed exactly as written.

## Issues

None.

## Next Phase Readiness

- **14-03 (GPU benchmarking / cluster refit script)** — Ready. `fit_all_gpu_m4` is the callable entry point needed for the SLURM cluster refit.
- **Phase 17 (M4 Hierarchical)** — `fit_all_gpu_m4` provides the MLE prior that seeds the hierarchical model, and the smoke test gives a regression baseline.
- **GPU-01 requirement** — Complete: named callable `fit_all_gpu_m4` exists, float64 ordering is correct, smoke test validates finite NLL + K bounds.
