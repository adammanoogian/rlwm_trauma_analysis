# Phase 14: Collins K Refit + GPU LBA Batching - Research

**Researched:** 2026-04-12
**Domain:** MLE parameter bounds change (K [1,7]→[2,6]) + GPU vmap for M4 LBA fitting
**Confidence:** HIGH — all findings are from direct codebase inspection

---

## Summary

Phase 14 has two parallel tracks: (1) tighten K bounds from [1,7] to [2,6] in `mle_utils.py` across all 7 models and add `parameterization_version` stamping to the MLE CSV writer, then refit on the cluster; (2) verify that `fit_all_gpu` already handles M4 via its own `wmrl_m4` branch and document what `fit_all_gpu_m4` means in context.

**K-02 (bounds change) is a small targeted edit**: every `*_BOUNDS` dict in `mle_utils.py` has `'capacity': (1.0, 7.0)` — six models plus M4. Changing all six to `(2.0, 6.0)` and M4's to `(2.0, 6.0)` is the complete code change for K-02. The `sample_lhs_starts` and all `jax_bounded_to_unconstrained_*` / `jax_unconstrained_to_params_*` functions use these dicts at runtime — no additional changes are needed in those functions.

**K-03 (refit + `parameterization_version` column)**: The existing `main()` in `fit_mle.py` saves `fits_df.to_csv(fits_path, index=False)` without injecting `parameterization_version`. A single line must be added before the save to stamp the column. `config.py` already has `EXPECTED_PARAMETERIZATION` and `load_fits_with_validation()` which validates this column downstream. The SLURM script `cluster/12_mle_gpu.slurm` already dispatches all 7 models as parallel GPU jobs — no new SLURM script needed.

**GPU-01/02/03**: `fit_all_gpu()` in `fit_mle.py` already contains a complete `model == 'wmrl_m4'` branch with 7-arg vmap (adds `rts`), float64-aware data preparation, and the `_gpu_objective_wmrl_m4` objective. The function is already called when `--use-gpu` is passed and GPU is detected. The requirement name `fit_all_gpu_m4` in the phase description refers to validating/exposing this existing path as a named function, not building it from scratch. The real work is: (a) verifying the M4 GPU path produces finite outputs on a 5-participant synthetic batch in float64, and (b) benchmarking wall time for N=154.

**Primary recommendation:** K-02 is 7 one-line tuple edits. K-03 is one `df['parameterization_version'] = ...` line before the CSV save. GPU-01 needs a named wrapper `fit_all_gpu_m4` that calls `fit_all_gpu(..., model='wmrl_m4')` after enabling float64, with a 5-participant smoke test. GPU-02 and GPU-03 are execution tasks (cluster runs + timing verification), not code tasks.

---

## Standard Stack

### Core (already in use — no new dependencies)

| Library | Purpose | Notes |
|---------|---------|-------|
| JAX / jaxlib | Autodiff, vmap, jit | float64 enabled via `jax.config.update("jax_enable_x64", True)` |
| jaxopt | `LBFGS` solver for GPU vmap path | Already imported in `fit_mle.py` |
| scipy.optimize | `L-BFGS-B` for CPU sequential path | Already in use |
| scipy.stats.qmc | Latin Hypercube Sampling | `sample_lhs_starts()` in `mle_utils.py` |
| pandas | CSV I/O | `fits_df.to_csv()` is the save point |

### No new dependencies required for Phase 14.

---

## Architecture Patterns

### Pattern 1: Bounds Dicts Drive Everything

All six `*_BOUNDS` dicts in `mle_utils.py` are the single source of truth for K bounds. They are consumed by:
- `sample_lhs_starts()` — LHS sampling in bounded space
- `jax_bounded_to_unconstrained_*()` — forward transforms for starting points
- `jax_unconstrained_to_params_*()` — inverse transforms for result extraction
- `fit_participant_mle()` — via `ScipyBoundedMinimize` bounds argument

Changing `'capacity': (1.0, 7.0)` → `'capacity': (2.0, 6.0)` in all six choice-model dicts and the M4 dict propagates automatically to every consumer. No function signatures change.

**Affected dicts (all in `scripts/fitting/mle_utils.py`):**
- `WMRL_BOUNDS` (line 37-44)
- `WMRL_M3_BOUNDS` (line 47-55)
- `WMRL_M5_BOUNDS` (line 58-67)
- `WMRL_M6A_BOUNDS` (line 70-78)
- `WMRL_M6B_BOUNDS` (line 81-90)
- `WMRL_M4_BOUNDS` (line 93-106)

Note: `QLEARNING_BOUNDS` has no `capacity` key — M1 (Q-learning) is not affected.

### Pattern 2: `parameterization_version` Stamping

`config.py` already defines `EXPECTED_PARAMETERIZATION` (lines ~497-505) with the expected version string per model:
```python
EXPECTED_PARAMETERIZATION: dict[str, str] = {
    "qlearning": "v4.0-phiapprox",
    "wmrl": "v4.0-K[2,6]-phiapprox",
    "wmrl_m3": "v4.0-K[2,6]-phiapprox",
    "wmrl_m5": "v4.0-K[2,6]-phiapprox",
    "wmrl_m6a": "v4.0-K[2,6]-phiapprox",
    "wmrl_m6b": "v4.0-K[2,6]-phiapprox-stickbreaking",
    "wmrl_m4": "v4.0-K[2,6]-phiapprox-lba",
}
```
`load_fits_with_validation()` in `config.py` raises `ValueError` if the column is missing or mismatched.

The injection point in `fit_mle.py` `main()` is just before `fits_df.to_csv(fits_path, index=False)` (~line 2965). The column must use the appropriate version string per model. Use `from config import EXPECTED_PARAMETERIZATION` and inject `fits_df['parameterization_version'] = EXPECTED_PARAMETERIZATION[args.model]`.

The GPU path `fit_all_gpu()` also returns a `df` (line 1653) — the stamping must happen after both the CPU and GPU paths return, i.e., in `main()` before save, not inside `fit_all_gpu`. This is correct because `main()` calls `fit_all_participants()` which may call `fit_all_gpu()` internally.

### Pattern 3: `fit_all_gpu` M4 Branch Already Exists

`fit_all_gpu()` in `fit_mle.py` already contains:
- Data prep with `rts_batch` for `model == 'wmrl_m4'` (lines 1294-1296, 1306)
- Separate `_run_one` / `_run_all` vmap for 7-arg M4 signature (lines 1381-1389)
- Execution branch for M4 (lines 1413-1416)
- Result extraction with `WMRL_M4_PARAMS` (lines 1478-1481)

The `_gpu_objective_wmrl_m4` function (lines 1178-1218) does `from scripts.fitting.lba_likelihood import wmrl_m4_multiblock_likelihood_stacked` as a lazy import inside the function. This is intentional — `lba_likelihood.py` calls `jax.config.update("jax_enable_x64", True)` at import time (line 10-11 of that file), so it must be imported after float64 is enabled in `main()`.

**GPU-01 deliverable**: a named function `fit_all_gpu_m4(data, n_starts=50, seed=42, verbose=True, compute_diagnostics=False)` that:
1. Calls `jax.config.update("jax_enable_x64", True)` at the top
2. Delegates to `fit_all_gpu(..., model='wmrl_m4', ...)`
3. Is tested with a 5-participant synthetic M4 recovery batch

### Pattern 4: Float64 Enabling Sequence for M4

The current `main()` CLI handles this correctly (lines 2827-2830):
```python
if args.model == 'wmrl_m4':
    jax.config.update("jax_enable_x64", True)
    from scripts.fitting.lba_likelihood import preprocess_rt_block  # triggers float64
```
The `fit_all_gpu_m4` wrapper must replicate this pattern — call `jax.config.update("jax_enable_x64", True)` BEFORE any JAX operations. This is a hard ordering requirement because JAX's x64 flag cannot be changed after the first JAX array is materialized in the process.

The `_gpu_objective_wmrl_m4` objective does a lazy `from scripts.fitting.lba_likelihood import ...` on every call — this is fine with JIT because it only runs at trace time, not execution time.

### Pattern 5: Test Pattern for GPU Smoke Test

Existing tests in `scripts/fitting/tests/` use synthetic data fixtures from `conftest.py`. The GPU smoke test for M4 should:
1. Generate 5 synthetic participants with M4 parameters (kappa included, no epsilon)
2. Call `fit_all_gpu_m4(data, n_starts=10, seed=42)` on a DataFrame of 5 participants
3. Assert: (a) no NaN in output `nll` column, (b) all `capacity` values in [2.0, 6.0], (c) all `capacity` values are float64 dtype (or at minimum finite floats)
4. Should run in < 5 minutes even on CPU (n_starts=10, n=5 is tiny)

The conftest does not yet have an M4 fixture. One must be added. The existing `simulate_wmrl_block()` in conftest can serve as a template — M4 requires additionally sampling RTs from a simple distribution (e.g., `np.random.uniform(0.3, 0.8)` in seconds).

### Pattern 6: SLURM Dispatch Pattern

`cluster/12_mle_gpu.slurm` already:
- Has `MODEL_COUNT` check to dispatch N single-model jobs in parallel (lines 162-187)
- Passes `--use-gpu` flag (line 213)
- Passes `--compute-diagnostics` (line 214)
- Has `--time=12:00:00` (line 35)
- Uses `rlwm_gpu` conda environment with `module load miniforge3`

For the K-03 refit run, no new SLURM script is needed. Submit `cluster/12_mle_gpu.slurm` which will dispatch all 7 models. The only change needed before the cluster run is the code changes in K-02 (bounds) and K-03 (version stamp).

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| LHS starting points in new [2,6] K range | Custom sampler | `sample_lhs_starts()` in `mle_utils.py` | Already reads bounds from `*_BOUNDS` dicts; changing the dict is sufficient |
| Bounds clamping in transforms | Explicit clipping | `bounded_to_unbounded()` already clips at `1e-8` on normalized scale | No change needed |
| Version validation | Custom dict lookup | `load_fits_with_validation()` in `config.py` | Already implemented; just add the column before save |
| M4 GPU objective | New function | `_gpu_objective_wmrl_m4` + `fit_all_gpu()` M4 branch | Already complete; just expose as named wrapper |
| RT synthetic data for M4 tests | Complex simulator | `np.random.uniform(0.3, 0.8)` uniform distribution | Good enough for a smoke test; exact distribution doesn't matter for checking finite NLL |

---

## Common Pitfalls

### Pitfall 1: Float64 Enable Too Late for GPU Path
**What goes wrong:** `fit_all_gpu_m4` creates a JAX array (e.g., from `sample_lhs_starts`) before calling `jax.config.update("jax_enable_x64", True)`. All arrays are silently float32, LBA CDF saturates at ~±6 sigma, and NLL returns NaN for reasonable RTs.
**How to avoid:** `jax.config.update("jax_enable_x64", True)` must be the FIRST statement in `fit_all_gpu_m4`, before any `jnp.array()` call. Also import `lba_likelihood` (or at minimum call `jax.config.update`) before `sample_lhs_starts`.
**Warning signs:** All `nll` values are NaN or identical; `rts_batch.dtype` is float32 instead of float64.

### Pitfall 2: Forgetting `parameterization_version` in the GPU Path Return Value
**What goes wrong:** `fit_all_gpu()` returns `df` without the column. `main()` stamps it afterward. But if `main()` calls `fit_all_participants()` which internally calls `fit_all_gpu()`, and the stamp is only added in `main()`, the GPU path return value before `main()` saves it still lacks the column. Since stamping happens in `main()` before `to_csv`, this is actually correct — but must be verified the stamp happens before BOTH `fits_df.to_csv()` calls (there is only one in `main()`).

### Pitfall 3: K Bounds Change Breaks Warmup Dummies
**What goes wrong:** `warmup_jax_compilation()` (~lines 161-245 of `fit_mle.py`) creates dummy data with hardcoded `capacity=4.0`. At K bounds [2,6], `capacity=4.0` is still within bounds, so this is fine. But the warmup for `wmrl_m4` passes `capacity=4.0` to `wmrl_m4_multiblock_likelihood` — this also remains valid at [2,6].
**How to avoid:** No action needed. 4.0 is in [2,6].

### Pitfall 4: `WMRL_BOUNDS` (M2) Not Updated
**What goes wrong:** Only M3-M6b dicts are updated but `WMRL_BOUNDS` (used by M2) is overlooked. M2 silently continues using K [1,7].
**How to avoid:** There are 6 dicts with `capacity` bounds (WMRL, WMRL_M3, WMRL_M5, WMRL_M6A, WMRL_M6B, WMRL_M4). Update all 6.

### Pitfall 5: `parameterization_version` Version String Mismatch
**What goes wrong:** M6b uses `"v4.0-K[2,6]-phiapprox-stickbreaking"` and M4 uses `"v4.0-K[2,6]-phiapprox-lba"` — not the simple `"v4.0-K[2,6]-phiapprox"` of M2/M3/M5/M6a. If a flat string is used for all models, `load_fits_with_validation()` raises for M6b and M4.
**How to avoid:** Use `EXPECTED_PARAMETERIZATION[model]` (imported from `config`), not a hardcoded string.

### Pitfall 6: GPU K Correlation Test on Wrong Recovery Dataset
**What goes wrong:** K-03 requires K recovery r >= 0.50 on N=50 synthetic participants re-generated with K in [2,6]. Using the existing quick-005/quick-006 recovery outputs (which used K in [1,7]) will show the old r=0.21 and appear to fail.
**How to avoid:** Re-run parameter recovery from scratch with the new K bounds. `scripts/11_run_model_recovery.py` is the right script.

### Pitfall 7: lba_likelihood Lazy Import Inside _gpu_objective_wmrl_m4 at JIT Trace Time
**What goes wrong:** The line `from scripts.fitting.lba_likelihood import wmrl_m4_multiblock_likelihood_stacked` inside `_gpu_objective_wmrl_m4` runs at Python trace time, not JAX execution time. If `lba_likelihood` is not importable (missing file, wrong path), the error appears during the vmap JIT compilation phase, not at import time, which makes it hard to diagnose.
**How to avoid:** Add a `from scripts.fitting import lba_likelihood` smoke import at the top of `fit_mle.py` guarded by `if False:` or test with `python -c "from scripts.fitting.lba_likelihood import wmrl_m4_multiblock_likelihood_stacked"` before the cluster run.

---

## Code Examples

### K-02: Changing capacity bounds in all 7 *_BOUNDS dicts

In `scripts/fitting/mle_utils.py`, change all six occurrences:
```python
# Before (6 dicts):
'capacity': (1.0, 7.0),

# After (6 dicts):
'capacity': (2.0, 6.0),
```
Dicts: `WMRL_BOUNDS`, `WMRL_M3_BOUNDS`, `WMRL_M5_BOUNDS`, `WMRL_M6A_BOUNDS`, `WMRL_M6B_BOUNDS`, `WMRL_M4_BOUNDS`.

Also update `ModelParams` in `config.py` for documentation consistency:
```python
# Before:
WM_CAPACITY_MIN = 1
WM_CAPACITY_MAX = 7

# After:
WM_CAPACITY_MIN = 2
WM_CAPACITY_MAX = 6
```
Note: `ModelParams.WM_CAPACITY_MIN/MAX` are documentation constants, not used in the tight optimization loop — they do not need to match for correctness, but consistency avoids confusion.

### K-03: Stamping parameterization_version in main()

In `scripts/fitting/fit_mle.py` `main()`, just before `fits_df.to_csv(fits_path, index=False)`:
```python
from config import EXPECTED_PARAMETERIZATION
fits_df['parameterization_version'] = EXPECTED_PARAMETERIZATION[args.model]
fits_df.to_csv(fits_path, index=False)
```
The import can be at the top of the file or inline. The `EXPECTED_PARAMETERIZATION` dict already has correct strings for all 7 models.

### GPU-01: fit_all_gpu_m4 wrapper function

Add to `scripts/fitting/fit_mle.py` after the `fit_all_gpu` function definition:
```python
def fit_all_gpu_m4(
    data: pd.DataFrame,
    n_starts: int = 50,
    seed: int = 42,
    verbose: bool = True,
    compute_diagnostics: bool = False,
) -> tuple[pd.DataFrame, dict, list[dict]]:
    """
    GPU-vectorized MLE fitting for M4 (RLWM-LBA joint choice+RT).

    Wraps fit_all_gpu with float64 enabled for LBA numerical stability.
    Must be called before any other JAX operations in the process.

    Returns same format as fit_all_gpu / fit_all_participants.
    """
    jax.config.update("jax_enable_x64", True)
    # Trigger lba_likelihood import (sets float64 globally)
    from scripts.fitting.lba_likelihood import wmrl_m4_multiblock_likelihood_stacked  # noqa: F401
    return fit_all_gpu(
        data=data,
        model='wmrl_m4',
        n_starts=n_starts,
        seed=seed,
        verbose=verbose,
        compute_diagnostics=compute_diagnostics,
    )
```

### Smoke test for GPU-01

Test function (add to `scripts/fitting/tests/test_mle_quick.py` or a new `test_gpu_m4.py`):
```python
def test_fit_all_gpu_m4_smoke(m4_synthetic_data_small):
    """Verify fit_all_gpu_m4 returns finite NLL without NaN or dtype errors."""
    import jax
    from scripts.fitting.fit_mle import fit_all_gpu_m4

    jax.config.update("jax_enable_x64", True)
    df, _, _ = fit_all_gpu_m4(m4_synthetic_data_small, n_starts=10, seed=42, verbose=False)

    assert len(df) == 5
    assert df['nll'].notna().all(), "NLL should not contain NaN"
    assert df['nll'].apply(lambda x: np.isfinite(float(x))).all()
    assert (df['capacity'] >= 2.0).all() and (df['capacity'] <= 6.0).all()
```
The `m4_synthetic_data_small` fixture returns a DataFrame with 5 synthetic participants formatted as `[sona_id, block, stimulus, key_press, reward, set_size, rt]`.

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| K in [1, 7] (v3.0 MLE) | K in [2, 6] (v4.0) | Phase 14 K-02 | Removes non-identified region; matches Senta et al. 2025 |
| No parameterization_version column | `parameterization_version` column stamped on every MLE CSV | Phase 14 K-03 | Enables `load_fits_with_validation()` to reject stale fits |
| M4 GPU path implicit in fit_all_gpu | Explicit `fit_all_gpu_m4` wrapper | Phase 14 GPU-01 | Callable entry point for tests and downstream scripts |

---

## Open Questions

1. **Does `fit_all_gpu` M4 branch work without modification on GPU?**
   - What we know: The M4 vmap branch exists in `fit_all_gpu()` and handles 7-arg data, float64. The `_gpu_objective_wmrl_m4` lazily imports `lba_likelihood` which sets float64.
   - What's unclear: Whether the lazy import inside a vmapped+jitted function causes issues at trace time. The import runs once at Python trace, then JIT compiles. This should be fine, but has not been cluster-tested.
   - Recommendation: Smoke test locally with 5 participants (CPU with x64 enabled) before the cluster run.

2. **How long does the M4 GPU refit take on A100 vs. ~48h CPU baseline?**
   - What we know: The SLURM script comment says "M4 (LBA, float64): ~12-24 hours for 154 participants (GPU, 50 starts)". Success criterion requires < 12h on A100.
   - What's unclear: Whether the comment was empirically measured or estimated. A100 has better float64 throughput than V100 (the likely source of the estimate).
   - Recommendation: Run a 10-participant timing test first (`--limit 10 --use-gpu`) to get a real extrapolation before committing the full run.

3. **K recovery improvement from r=0.21 to r>=0.50 — is the bound change sufficient?**
   - What we know: The r=0.21 baseline was from K in [1,7] recovery. The fundamental identifiability argument (K<2 confounded with rho) suggests [2,6] should help. K-01 research confirms this.
   - What's unclear: Whether r>=0.50 is achievable with N=50 synthetic participants and 20 random starts. The current recovery script uses `--n-starts 20`; using 50 starts may help.
   - Recommendation: Run recovery with `--n-starts 50` for K-03 verification to match the real-data fitting settings.

---

## Sources

### Primary (HIGH confidence)
- Direct code inspection of `scripts/fitting/mle_utils.py` — bounds dicts, transform functions, sample_lhs_starts
- Direct code inspection of `scripts/fitting/fit_mle.py` — fit_all_gpu, _gpu_objective_wmrl_m4, main() save path
- Direct code inspection of `scripts/fitting/lba_likelihood.py` — float64 enforcement pattern
- Direct code inspection of `config.py` — EXPECTED_PARAMETERIZATION, load_fits_with_validation
- Direct code inspection of `cluster/12_mle_gpu.slurm` — SLURM dispatch pattern
- Direct code inspection of `docs/K_PARAMETERIZATION.md` — K-01 findings, version string
- `output/mle/wmrl_m3_individual_fits.csv` — confirmed: no `parameterization_version` column, capacity range [1.0, 7.0], n=154

### Secondary (MEDIUM confidence)
- `STATE.md` and `CLAUDE.md` project notes on phase structure and v4.0 decisions

---

## Metadata

**Confidence breakdown:**
- K-02 (bounds change): HIGH — direct code inspection, 6 dict locations identified
- K-03 (version stamp): HIGH — injection point identified, EXPECTED_PARAMETERIZATION already coded
- GPU-01 (fit_all_gpu_m4 wrapper): HIGH — M4 branch already in fit_all_gpu, wrapper is trivial
- GPU-02/03 (cluster execution + timing): MEDIUM — timing estimates from SLURM comment, not measured empirically

**Research date:** 2026-04-12
**Valid until:** Until fit_mle.py or mle_utils.py is significantly refactored (stable: 90 days)
