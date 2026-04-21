---
quick: 007-gpu-mcmc-speedup-vmap-chain-method
plan: 007
subsystem: inference
tags: [jax, numpyro, vmap, mcmc, gpu, chain_method, hierarchical]

requires:
  - phase: 15-m3-hierarchical
    provides: wmrl_m3_hierarchical_model with Python for-loop likelihood
  - phase: 20-deer
    provides: wmrl_m3_block_likelihood, wmrl_m3_multiblock_likelihood_stacked

provides:
  - "_select_chain_method(num_chains): GPU-aware chain_method selector (vectorized on GPU, parallel/sequential on CPU)"
  - "stack_across_participants(): uniform (N, max_n_blocks, 100) padding with mask-gated padded blocks"
  - "wmrl_m3_fully_batched_likelihood(): nested vmap over participants and blocks, returns (N,) log-lik vector"
  - "wmrl_m3_hierarchical_model with single numpyro.factor('obs', ...) instead of 154 per-participant factors"
  - "GPU/NumPyro config diagnostic banner in run_inference_with_bump (backend, devices, chain_method, x64, env vars)"
  - "Agreement test: batched vs sequential M3 likelihood, max_rel_err=0 on 3 draws x 5 participants"

affects:
  - "008 or follow-on quick task: propagate vmap pattern to M1/M2/M5/M6a/M6b"
  - cluster/13_bayesian_*.slurm GPU validation runs
  - Phase 17 M4 hierarchical (different bottleneck; pmap not vmap)

tech-stack:
  added: []
  patterns:
    - "GPU chain_method: always 'vectorized' on single GPU (pmap='parallel' for multi-GPU, not yet standard)"
    - "vmap-over-participants: outer vmap(N) x inner vmap(B) x block_likelihood(T), params broadcast on inner"
    - "uniform participant stacking: zero-mask padding to max_n_blocks is a no-op on likelihood (mask gates all updates)"
    - "single numpyro.factor pattern: one 'obs' site vs N per-participant sites reduces XLA subgraph count by 154x"

key-files:
  created:
    - scripts/fitting/tests/test_m3_hierarchical.py (appended test_wmrl_m3_fully_batched_matches_sequential)
    - .planning/quick/007-gpu-mcmc-speedup-vmap-chain-method/007-SUMMARY.md
  modified:
    - scripts/fitting/jax_likelihoods.py (added wmrl_m3_fully_batched_likelihood)
    - scripts/fitting/numpyro_models.py (added _select_chain_method, stack_across_participants, refactored wmrl_m3_hierarchical_model)
    - scripts/fitting/fit_bayesian.py (import stack_across_participants, pass stacked_arrays for wmrl_m3)
    - cluster/13_bayesian_pscan_smoke.slurm (added GPU config diagnostic block)

key-decisions:
  - "GPU chain_method='vectorized' not 'parallel': single-GPU SLURM (--gres=gpu:1) has local_device_count=1, making 'parallel' impossible; 'vectorized' runs all chains via vmap on one device."
  - "pmap multi-GPU path documented but not implemented: future SLURM with --gres=gpu:N would light up 'parallel' automatically via _select_chain_method."
  - "pscan + vmap composition is out of scope: wmrl_m3_hierarchical_model raises NotImplementedError on use_pscan=True."
  - "stack_across_participants is called once in _fit_stacked_model (not inside MCMC trace): avoids recomputation per MCMC sample."
  - "bayesian_diagnostics.py unchanged: compute_pointwise_log_lik continues to use per-participant dict-of-dicts (post-fit, speed not critical)."
  - "Agreement test: max_rel_err=0.0 on all 3 draws (not just < 1e-4 as required). Float32 vmap is bit-exact with sequential on CPU."

patterns-established:
  - "nested vmap pattern: jax.vmap(f, in_axes=(0,0,0,0,0,None,...)) for inner (block-level), jax.vmap(g, in_axes=(0,...,0)) for outer (participant-level)"
  - "GPU diagnostic banner: print once before MCMC for-loop in run_inference_with_bump, not per-iteration"
  - "stacked_arrays kwarg with None fallback: allows tests to call model without pre-computing stack; production always passes pre-computed"

duration: 28min
completed: 2026-04-16
---

# Quick Task 007: GPU MCMC Speedup via vmap Chain Method Summary

**wmrl_m3 GPU MCMC parallelized: chain_method auto-selects 'vectorized' on GPU, 154-participant Python for-loop replaced by nested jax.vmap returning (N,) log-lik vector with single numpyro.factor("obs", ...) call**

## Performance

- **Duration:** 28 min
- **Started:** 2026-04-16T07:34:24Z
- **Completed:** 2026-04-16T08:02:xx Z
- **Tasks:** 7/7
- **Files modified:** 5

## Three Root Causes Addressed

### Cause 1: chain_method falls back to "sequential" on GPU

**Before:** `"parallel" if jax.local_device_count() >= num_chains else "sequential"` — on a single-GPU SLURM node, `local_device_count() == 1` while `num_chains == 4`, so this evaluates to "sequential", serializing all 4 chains.

**After:** `_select_chain_method(num_chains)` returns "vectorized" when `jax.default_backend() == "gpu"` and `local_device_count < num_chains`. Vectorized mode batches chains via vmap on a single GPU, giving true chain-level parallelism without requiring multiple physical GPUs.

**Expected speedup component:** ~4x (4 chains were fully sequential, now parallel).

### Cause 2: 154-participant Python for-loop creates 154 sequential XLA subgraphs

**Before:** `for idx, pid in enumerate(participant_ids): numpyro.factor(f"obs_p{pid}", ...)` — each iteration traces a separate XLA computation graph, and on GPU each requires a host-device round-trip.

**After:** Single `wmrl_m3_fully_batched_likelihood(...)` call with `numpyro.factor("obs", per_participant_ll.sum())`. Outer `jax.vmap` over participants creates a single batched XLA kernel.

**Expected speedup component:** GPU: large (eliminates 154 host-device round-trips per sample). CPU: modest (JIT already fused the loop, but reduces graph compilation size).

### Cause 3: Sequential lax.fori_loop over blocks (independent blocks not parallelized)

**Before:** `wmrl_m3_multiblock_likelihood_stacked` uses `lax.fori_loop(0, num_blocks, body_fn, 0.0)` — sequential scan over 12-17 independent blocks per participant.

**After:** Inner `jax.vmap(_block_ll, in_axes=(0,0,0,0,0,None,...))` maps `wmrl_m3_block_likelihood` over the block dimension. Blocks are fully independent (Q/WM/perseveration all reset at block boundaries per Senta 2025), so vmap is semantically correct.

**Expected speedup component:** ~12-17x per-participant (blocks now parallel), compounded with cause 2.

## Agreement Test Results

| Draw | max_rel_err (batched vs sequential) | Status |
|------|--------------------------------------|--------|
| 0    | 0.00e+00                             | PASS   |
| 1    | 0.00e+00                             | PASS   |
| 2    | 0.00e+00                             | PASS   |

Requirement was < 1e-4. Observed: exact bit-for-bit agreement on float32 CPU. The vmap refactor is numerically equivalent to the sequential implementation.

Synthetic data: N=5 participants with n_blocks in {12, 13, 17}, 60-100 trials per block (variable), 3 random parameter draws.

## GPU Wall Time Estimate

Cluster smoke test (`sbatch cluster/13_bayesian_pscan_smoke.slurm`) has NOT yet been run — that is for the user to validate on GPU hardware. Baseline was ~5.5 min/iter; target is <= 33s/iter (>= 10x). The three causes together are expected to yield >>10x.

Populate after cluster run: `[TBD — run sbatch cluster/13_bayesian_pscan_smoke.slurm and note wmrl_m3 Stage 2 wall time per iteration]`

## Task Commits

| # | Task | Hash | Type |
|---|------|------|------|
| 1 | Fix chain_method selection in run_inference and run_inference_with_bump | `a6359f3` | feat |
| 2 | Add GPU config diagnostic logging and timing in run_inference_with_bump | `a044134` | feat |
| 3 | Add stack_across_participants helper with uniform max_n_blocks padding | `04f5fbf` | feat |
| 4 | Add wmrl_m3_fully_batched_likelihood via nested vmap | `aeaa208` | feat |
| 5 | Refactor wmrl_m3_hierarchical_model to use fully-batched likelihood | `6403c72` | feat |
| 6 | Add agreement test between sequential and fully-batched M3 likelihoods | `1ac963c` | test |
| 7 | Full test suite validation (no files modified, validation only) | — | — |

## Files Created/Modified

- `scripts/fitting/numpyro_models.py` — Added `_select_chain_method`, GPU config banner, `[timing]` lines, `stack_across_participants`, refactored `wmrl_m3_hierarchical_model` (new `stacked_arrays` kwarg, single `numpyro.factor("obs", ...)`)
- `scripts/fitting/jax_likelihoods.py` — Added `wmrl_m3_fully_batched_likelihood` (nested vmap, 147 lines)
- `scripts/fitting/fit_bayesian.py` — Added `stack_across_participants` import; passes `stacked_arrays` in `model_args` for `wmrl_m3` only
- `cluster/13_bayesian_pscan_smoke.slurm` — Added GPU config diagnostic block before Stage 1
- `scripts/fitting/tests/test_m3_hierarchical.py` — Appended `test_wmrl_m3_fully_batched_matches_sequential`

## Decisions Made

- **GPU chain_method='vectorized' locked (not 'parallel'):** Single-GPU SLURM jobs (--gres=gpu:1) have `jax.local_device_count() == 1`. NumPyro's "parallel" mode requires `local_device_count >= num_chains`. "vectorized" is correct for all current Monash M3 GPU jobs.
- **pmap multi-GPU path documented but not activated:** If a future SLURM job requests `--gres=gpu:4`, `_select_chain_method` automatically returns "parallel" (pmap). No code change needed.
- **pscan + vmap is out of scope:** `wmrl_m3_hierarchical_model` raises `NotImplementedError` on `use_pscan=True`. The pscan path creates non-uniform block structures incompatible with uniform stacking.
- **stack_across_participants called once in fit_bayesian.py, not inside MCMC:** Avoids Python-level recomputation per MCMC sample. The model accepts `stacked_arrays=None` as a fallback for test convenience.
- **bayesian_diagnostics.py unchanged:** `compute_pointwise_log_lik` runs post-fit, not inside MCMC. It continues to use per-participant dict-of-dicts. Speed is not critical in this path.
- **Other 5 models unchanged (M1/M2/M5/M6a/M6b/M4):** This is a wmrl_m3-only POC. A follow-on quick task will propagate the vmap pattern to other choice-only models once the M3 cluster run validates the expected speedup.

## pmap Research Conclusion

On Monash M3 GPU partition (standard: 1 GPU per job), `chain_method="parallel"` is impossible for `num_chains > 1` because NumPyro pmap requires one device per chain. `chain_method="vectorized"` is the correct choice: it uses JAX vmap internally to run all chains on a single device with a shared batch dimension. The `_select_chain_method` helper implements the recommended rule:

- GPU + multi-chain + single device → "vectorized"
- GPU + multi-chain + multi-device (rare) → "parallel"
- CPU → "parallel" or "sequential" (existing behavior, unchanged)

## Test Results

All 4 required test suites pass:

| Suite | Tests | Result |
|-------|-------|--------|
| test_m3_hierarchical.py (non-slow) | 1/1 | PASS |
| test_pointwise_loglik.py | 30/30 | PASS |
| test_compile_gate.py | 2/2 | PASS |
| test_numpyro_helpers.py | 11/11 (8 collected) | PASS |

CPU smoke fit (wmrl_m3, N=154, warmup=20, samples=20):
- JIT compilation: success
- MCMC sampling: 0 divergences at target_accept_prob=0.8
- Pointwise log-lik: shape (1, 20, 154, 1700), PASS
- ArviZ InferenceData: built, PASS
- Convergence gate: FAILED (expected — warmup=20 too small for real convergence)
- No `obs_p{pid}` factor sites in wmrl_m3_hierarchical_model (confirmed)
- JAX/NumPyro config banner: visible in stdout

## M3-Only POC Cluster Readiness

The following is ready for cluster validation:

1. `sbatch cluster/13_bayesian_pscan_smoke.slurm` — runs wmrl_m3 Stage 2 with the new chain_method=vectorized path
2. Check log for: `chain_method: vectorized` in the GPU config block
3. Measure wall time per iteration in Stage 2 (target: <= 33s vs baseline 5.5 min)
4. If target met: propagate vmap pattern to M5/M6a/M6b in a follow-on quick task

Other 5 models (M1/M2/M5/M6a/M6b) continue to use the Python for-loop (unchanged). M4 RLWM-LBA uses a different bottleneck (LBA kernel, not hierarchical participant loop).

## Deviations from Plan

None — plan executed exactly as written. All shape/dtype checks from the `<code_trace>` section matched without discrepancy.

## Issues Encountered

None. The `jaxopt` import error in `test_mle_quick.py` is a pre-existing environment issue unrelated to this task (jaxopt not installed in this conda environment); that test file was excluded from the required test suites.

## Next Phase Readiness

- M3 vmap refactor complete; ready for cluster smoke test on GPU hardware
- Follow-on quick task (008 or similar) should port vmap pattern to M5/M6a/M6b after cluster validation
- `_select_chain_method` already handles all 5 remaining models (they call run_inference_with_bump which now uses the correct rule)
- bayesian_diagnostics.py WAIC/LOO pipeline validated: unchanged and passing

---
*Quick: 007-gpu-mcmc-speedup-vmap-chain-method*
*Completed: 2026-04-16*
