---
quick: 007-gpu-mcmc-speedup-vmap-chain-method
plan: 007
type: execute
wave: 1
depends_on: []
files_modified:
  - scripts/fitting/numpyro_models.py
  - scripts/fitting/jax_likelihoods.py
  - scripts/fitting/bayesian_diagnostics.py
  - scripts/fitting/tests/test_m3_hierarchical.py
  - cluster/13_bayesian_pscan_smoke.slurm
autonomous: true

must_haves:
  truths:
    - "GPU smoke test for wmrl_m3 completes MCMC warmup iterations at >= 10x the rate seen in the 20-03 baseline (baseline: ~5.5 min/iter; target: <= 33s/iter)."
    - "run_inference_with_bump picks chain_method='vectorized' whenever a GPU device is detected, regardless of num_chains."
    - "SLURM smoke test log contains a GPU config block printing jax.devices(), jax.default_backend(), NUMPYRO_HOST_DEVICE_COUNT, chain_method, XLA_PYTHON_CLIENT_PREALLOCATE, and JAX_COMPILATION_CACHE_DIR before the first MCMC call."
    - "wmrl_m3_hierarchical_model accumulates the likelihood in a single numpyro.factor('obs', total_ll) call over all participants via vmap, replacing the 154-iteration Python for-loop."
    - "For wmrl_m3 only, the new fully-batched likelihood's total NLL agrees with the sequential for-loop implementation to relative error < 1e-4 across 3 random parameter draws for N=5 synthetic participants with 12-17 blocks each."
    - "Pointwise log-lik pipeline (compute_pointwise_log_lik -> build_inference_data_with_loglik -> filter_padding_from_loglik -> az.waic) still produces a valid InferenceData for wmrl_m3 and WAIC/LOO executes without errors."
    - "JAX vmap/pmap semantics and NumPyro chain_method options are documented in the <research_findings> section of this plan (task 0) with concrete references to official JAX and NumPyro docs."
    - "Argument shapes and dtypes for every parameter in the wmrl_m3 data path (stimuli through covariate_lec) are explicitly traced in the <code_trace> section with before/after shapes for the vmap refactor."
    - "pscan behavior is unchanged by this plan: --use-pscan remains an explicit flag, default path is sequential, and the existing pscan smoke test from cluster/13_bayesian_pscan_smoke.slurm still runs."
  artifacts:
    - path: "scripts/fitting/numpyro_models.py"
      provides: "Fixed chain_method logic in run_inference_with_bump and run_inference; GPU config logging; refactored wmrl_m3_hierarchical_model using vmap'd likelihood; uniform-padded prepare_stacked_participant_data with stack_across_participants helper."
      contains: "stack_across_participants"
    - path: "scripts/fitting/jax_likelihoods.py"
      provides: "New wmrl_m3_fully_batched_likelihood(stimuli, actions, rewards, set_sizes, masks, alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon, ...) taking (N, B, T) arrays and per-participant parameter arrays, returning (N,) log-lik vector via nested vmap over participants and blocks."
      contains: "wmrl_m3_fully_batched_likelihood"
    - path: "scripts/fitting/bayesian_diagnostics.py"
      provides: "compute_pointwise_log_lik and filter_padding_from_loglik updated to handle both legacy variable-block format and new uniform-padded-across-participants format (N, max_n_blocks, 100) with padded-block mask values of 0."
      contains: "uniform_across_participants"
    - path: "scripts/fitting/tests/test_m3_hierarchical.py"
      provides: "Agreement test between sequential for-loop likelihood and fully-batched vmap'd likelihood for wmrl_m3 at relative error < 1e-4."
      contains: "test_wmrl_m3_fully_batched_matches_sequential"
    - path: "cluster/13_bayesian_pscan_smoke.slurm"
      provides: "GPU config diagnostic block printed before MCMC kickoff; chain_method visible in logs."
      contains: "GPU config diagnostic"
  key_links:
    - from: "run_inference_with_bump"
      to: "MCMC(..., chain_method=...)"
      via: "chain_method decision rule"
      pattern: "jax\\.default_backend\\(\\) == ['\"]gpu['\"].*vectorized"
    - from: "wmrl_m3_hierarchical_model"
      to: "wmrl_m3_fully_batched_likelihood"
      via: "single numpyro.factor call"
      pattern: "numpyro\\.factor\\(['\"]obs['\"]"
    - from: "prepare_stacked_participant_data"
      to: "stack_across_participants"
      via: "uniform-padded (N, max_n_blocks, 100) tensors"
      pattern: "stack_across_participants"
    - from: "wmrl_m3_fully_batched_likelihood"
      to: "wmrl_m3_block_likelihood"
      via: "jax.vmap(jax.vmap(block_likelihood))"
      pattern: "jax\\.vmap\\(.*jax\\.vmap"
---

<objective>
GPU MCMC for the wmrl_m3 hierarchical model is running ~30x slower than CPU
(~5.5 min/iter vs 9-28 s/iter on CPU). Three mechanical causes have been
identified in code: (1) chain_method falls back to "sequential" on GPU because
`jax.local_device_count() < num_chains`, serializing chains; (2) the
hierarchical model iterates 154 participants in a Python for-loop, creating
154 sequential sub-graphs per sample; (3) within each participant,
`lax.fori_loop` scans blocks sequentially even though blocks are fully
independent (Q/WM/perseveration all reset at block boundaries per Senta 2025).

This plan fixes all three for wmrl_m3 as a proof of concept. Other five
choice-only models (M1/M2/M5/M6a/M6b) are deliberately left untouched; a
follow-on quick task will propagate the pattern once the M3 cluster run
demonstrates the expected >= 10x GPU speedup.

Purpose:
- Restore GPU MCMC to a productive throughput for hierarchical Bayesian fits.
- Validate the vmap-over-participants pattern on the simplest non-trivial
  hierarchical model (M3) before porting to M5/M6a/M6b.
- Improve observability so future GPU regressions surface in a single log
  block rather than requiring inspection of iteration timestamps.

Output:
- Modified numpyro_models.py, jax_likelihoods.py, bayesian_diagnostics.py.
- New agreement test in test_m3_hierarchical.py.
- Updated smoke test SLURM with GPU config logging.
- Research findings and code trace documented in this plan (no separate docs).
</objective>

<execution_context>
@C:\Users\aman0087\.claude/get-shit-done/workflows/execute-plan.md
@C:\Users\aman0087\.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/STATE.md

@scripts/fitting/numpyro_models.py
@scripts/fitting/jax_likelihoods.py
@scripts/fitting/bayesian_diagnostics.py
@scripts/fitting/fit_bayesian.py
@scripts/fitting/tests/test_m3_hierarchical.py
@cluster/13_bayesian_pscan_smoke.slurm
@docs/03_methods_reference/MODEL_REFERENCE.md
</context>

<research_findings>

This section is the research deliverable for user request #2. Findings are
documented inline rather than in a separate file per the task instructions.
Citations reference the NumPyro documentation at
https://num.pyro.ai/en/stable/mcmc.html and the JAX documentation at
https://jax.readthedocs.io/.

## NumPyro chain_method options (numpyro.infer.MCMC.__init__)

Per NumPyro source (numpyro/infer/mcmc.py::MCMC) and docs:

- "parallel"   - runs chains on separate host devices via `pmap`. Requires
                 `jax.local_device_count() >= num_chains`. On CPU, the device
                 count is controlled by `numpyro.set_host_device_count(N)`
                 (must be called before any JAX computation). On GPU, each
                 physical GPU counts as one device; pmap will assign one
                 chain per GPU. In single-GPU SLURM jobs (the Monash M3 norm),
                 `jax.local_device_count() == 1`, so "parallel" with
                 num_chains > 1 would raise: requires multiple devices.

- "vectorized" - runs chains in parallel on a SINGLE device by treating the
                 chain index as an extra batch dimension. Uses vmap
                 internally. Works on any backend (CPU/GPU/TPU) with a single
                 device. Memory cost scales linearly in num_chains. On GPU,
                 this is the correct choice for executing 2-4 chains in
                 parallel on one physical GPU.

- "sequential" - runs chains one after the other on a single device. No
                 parallelism. Used as a safe fallback. Memory cost is O(1)
                 in num_chains but wall time is O(num_chains). This is the
                 current code path on GPU, and is the primary root cause of
                 the observed ~2x loss on top of the vmap loss.

Conclusion: on GPU, chain_method MUST be "vectorized" to get chain-level
parallelism on a single device. The existing rule
`"parallel" if jax.local_device_count() >= num_chains else "sequential"`
is only valid on CPU where set_host_device_count(N) exposes N devices.

## NUMPYRO_HOST_DEVICE_COUNT behavior

- On CPU, `NUMPYRO_HOST_DEVICE_COUNT=N` (equivalent to
  `numpyro.set_host_device_count(N)` called before any JAX op) makes JAX
  report N "CPU devices", enabling chain_method="parallel" with N chains.
- On GPU, this env var has NO effect on the number of visible GPU devices.
  GPU device count is controlled by the CUDA driver and `CUDA_VISIBLE_DEVICES`.
  Setting `NUMPYRO_HOST_DEVICE_COUNT` on GPU is harmless but misleading.
- Current `run_inference_with_bump` calls `numpyro.set_host_device_count(num_chains)`
  unconditionally. On GPU this call is a no-op for GPU devices but may still
  reconfigure the CPU backend's internal thread pool. Safe to leave but
  irrelevant to the GPU path.

Conclusion: on GPU, the smoke-test SLURM's `export NUMPYRO_HOST_DEVICE_COUNT=1`
is harmless but unnecessary. We will log it for diagnostic purposes but the
chain_method decision must be based on `jax.default_backend()`, not device
count.

## jax.vmap in_axes semantics for dict-of-arrays

Per jax.readthedocs.io/en/latest/_autosummary/jax.vmap.html:

- `in_axes` can be a single int (maps that axis on every positional arg), a
  tuple of ints matching positional args, or a pytree matching the argument
  structure.
- For a function `f(a, b, c)` where only `a` should be mapped and `b, c` are
  broadcast constants, use `in_axes=(0, None, None)`.
- `None` means "do not map; broadcast this argument across all vmapped calls".
- For a dict argument, in_axes can be a dict with the same keys whose values
  are the per-key axis specs.
- Nested vmap composes: `jax.vmap(jax.vmap(f, in_axes=A_inner), in_axes=A_outer)`
  runs the outer batch dimension first (outermost loop), then the inner batch
  dimension. Composition is correct for our pattern: outer = participants,
  inner = blocks.

For wmrl_m3_fully_batched_likelihood we use an inner vmap over blocks
(`in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None, None)`) and an
outer vmap over participants (with per-participant parameters mapped on axis 0
and per-participant data arrays mapped on axis 0 of the leading participant
dimension).

## jax.pmap vs vmap for chains across multiple GPUs

Per jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html:

- `pmap` distributes work across multiple physical devices; it is the
  correct primitive when each chain should run on its own GPU.
- `vmap` parallelizes on a SINGLE device using SIMD-style batching.
- NumPyro's chain_method="parallel" wraps pmap; "vectorized" wraps vmap.

Cluster applicability (Monash M3 — user request #4):
- Standard GPU partition: 1 GPU per job (A100 or P100). chain_method="parallel"
  impossible; must use "vectorized".
- Multi-GPU jobs would require `--gres=gpu:4` or similar; subject to partition
  limits. If ever used, `chain_method="parallel"` with `num_chains=4` gives
  one chain per GPU (cleanest parallelism; no memory interference between
  chains). However current SLURM scripts (13_bayesian_m3.slurm,
  13_bayesian_pscan_smoke.slurm) request `--gres=gpu:1`.

Recommendation: do NOT implement pmap-based multi-GPU path in this task.
Document the option, keep the decision rule
`"vectorized" if gpu else ("parallel" if local_device_count >= num_chains else "sequential")`.
The pmap path would light up automatically if a future SLURM script requests
multiple GPUs and jax.local_device_count() returns >= num_chains.

## Recommended chain_method decision rule (implemented in task 1)

```
backend = jax.default_backend()
n_devices = jax.local_device_count()

if backend == "gpu":
    # Single-GPU: vmap chains on one device. Multi-GPU: pmap across devices.
    if n_devices >= num_chains:
        chain_method = "parallel"     # rare: --gres=gpu:N with N >= chains
    else:
        chain_method = "vectorized"   # the common case
elif backend == "tpu":
    chain_method = "vectorized" if n_devices < num_chains else "parallel"
else:  # cpu
    # Respects numpyro.set_host_device_count(num_chains) already called above.
    chain_method = "parallel" if n_devices >= num_chains else "sequential"
```

</research_findings>

<code_trace>

This section is the code-tracing deliverable for user request #3. It documents
the exact shape and dtype of every argument at every site in the wmrl_m3
data path, BEFORE and AFTER the vmap refactor.

## Before: current sequential loop (numpyro_models.py:1118-1139)

Data prep (prepare_stacked_participant_data, lines 896-994):
- Input: `data_df` with columns sona_id, block, stimulus, key_press, reward,
  set_size.
- Output: `dict[participant_id, dict[str, jnp.ndarray]]` where each inner
  dict has:
    - stimuli_stacked   (n_blocks_i, 100) int32       # n_blocks_i in {12..17}
    - actions_stacked   (n_blocks_i, 100) int32
    - rewards_stacked   (n_blocks_i, 100) float32
    - set_sizes_stacked (n_blocks_i, 100) float32
    - masks_stacked     (n_blocks_i, 100) float32     # 1.0 real, 0.0 pad

Parameter sampling (lines 1065-1106):
- `sampled[param]` has shape (N,) where N = len(participant_data_stacked)
  for each of {alpha_pos, alpha_neg, phi, rho, capacity, epsilon, kappa}.
- `covariate_lec` shape (N,) float32 or None.

Likelihood loop (lines 1118-1139):
- For each idx, pid:
    - `pdata = participant_data_stacked[pid]` has variable n_blocks_i.
    - `log_lik = _m3_lik_fn(..., alpha_pos=sampled['alpha_pos'][idx], ...)`
      where each parameter is a SCALAR (shape ()).
    - `numpyro.factor(f"obs_p{pid}", log_lik)` — creates 154 separate factor
      sites named obs_p{pid_0}, obs_p{pid_1}, ...

Inner: wmrl_m3_multiblock_likelihood_stacked (jax_likelihoods.py:2086-2170):
- Inputs:
    - stimuli_stacked, actions_stacked, rewards_stacked, set_sizes_stacked,
      masks_stacked — shapes (n_blocks_i, 100). Variable n_blocks_i.
    - alpha_pos, ..., kappa, epsilon — all scalars.
- `num_blocks = stimuli_stacked.shape[0]` = n_blocks_i (variable 12..17).
- `lax.fori_loop(0, num_blocks, body_fn, 0.0)` scans blocks SEQUENTIALLY.
- body_fn calls wmrl_m3_block_likelihood(stimuli=stimuli_stacked[block_idx],
  ...) — block likelihood takes (100,) slices and scalar params.
- Returns scalar total_ll.

Innermost: wmrl_m3_block_likelihood (jax_likelihoods.py:1482+):
- Inputs:
    - stimuli, actions: (100,) int32
    - rewards, set_sizes, mask: (100,) float32
    - alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon: scalars
- Returns scalar log-lik (or (scalar, (100,)) pointwise tuple).
- Q/WM/last_action all reset at function entry (Q_init, WM_init, last_action=-1).
  CONFIRMED: no cross-block state leakage.

Bottleneck summary (154 participants, avg 17 blocks, 100 trials):
- Graph nodes: 154 (outer Python loop)
    x 17 (sequential lax.fori_loop over blocks)
    x 100 (inner lax.scan over trials)
  = 261,800 sequential scan steps per MCMC likelihood evaluation.
- On CPU the Python-level work is amortized by JIT. On GPU the 154 separate
  numpyro.factor sites each force a separate XLA subgraph and host-device
  round-trip per participant. This is the dominant GPU cost.

## After: fully-batched path (new code in this plan)

Data prep (modified prepare_stacked_participant_data):
- Unchanged per-participant output (n_blocks_i, 100) — dict-of-dicts preserved
  for backward compat with bayesian_diagnostics.py.
- NEW: `stack_across_participants(participant_data_stacked)` helper returns
  a single dict:
    - stimuli    (N, max_n_blocks, 100) int32
    - actions    (N, max_n_blocks, 100) int32
    - rewards    (N, max_n_blocks, 100) float32
    - set_sizes  (N, max_n_blocks, 100) float32
    - masks      (N, max_n_blocks, 100) float32      # PADDED BLOCKS ALL 0.0
  where max_n_blocks = max over participants (likely 17).
- Participants with fewer blocks (12) get (17 - 12) = 5 zero-mask blocks
  appended. Because mask=0 zeroes out both the likelihood contribution and
  the Q/WM/perseveration updates within the existing block_likelihood, the
  padded blocks contribute EXACTLY 0.0 to total_ll.
- Participant ordering is `sorted(participant_data_stacked.keys())` — same
  as existing ordering used by covariate_lec alignment.

Parameter sampling (unchanged):
- `sampled[param]` shape (N,).
- `kappa` shape (N,), `covariate_lec` shape (N,) or None.

Batched likelihood call (new wmrl_m3_hierarchical_model body):
- Single call:
    total_ll_per_participant = wmrl_m3_fully_batched_likelihood(
        stimuli=stacked["stimuli"],       # (N, B, 100)
        actions=stacked["actions"],       # (N, B, 100)
        rewards=stacked["rewards"],       # (N, B, 100)
        set_sizes=stacked["set_sizes"],   # (N, B, 100)
        masks=stacked["masks"],           # (N, B, 100)
        alpha_pos=sampled["alpha_pos"],   # (N,)
        alpha_neg=sampled["alpha_neg"],   # (N,)
        phi=sampled["phi"],               # (N,)
        rho=sampled["rho"],               # (N,)
        capacity=sampled["capacity"],     # (N,)
        kappa=sampled["kappa"],           # (N,)
        epsilon=sampled["epsilon"],       # (N,)
        num_stimuli=6,                    # static
        num_actions=3,                    # static
        q_init=0.5,                       # static
        wm_init=1/3,                      # static
    )
    # Returns (N,) log-lik vector.
    numpyro.factor("obs", total_ll_per_participant.sum())
- ONE factor site instead of 154.

wmrl_m3_fully_batched_likelihood (new function in jax_likelihoods.py):
- Implementation:
    _block = lambda stim, act, rew, ss, mask, ap, an, ph, rh, cap, k, e: (
        wmrl_m3_block_likelihood(
            stimuli=stim, actions=act, rewards=rew, set_sizes=ss, mask=mask,
            alpha_pos=ap, alpha_neg=an, phi=ph, rho=rh, capacity=cap,
            kappa=k, epsilon=e,
            num_stimuli=num_stimuli, num_actions=num_actions,
            q_init=q_init, wm_init=wm_init,
            return_pointwise=False,
        )
    )
    # Inner vmap: map over blocks axis; params broadcast within participant.
    _over_blocks = jax.vmap(
        _block,
        in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None, None),
        out_axes=0,
    )
    # Block-level vmap returns (B,) per-block log-liks; sum to get participant-level.
    _per_participant = lambda *args: _over_blocks(*args).sum()
    # Outer vmap: map over participants axis; everything maps on axis 0.
    _over_participants = jax.vmap(
        _per_participant,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        out_axes=0,
    )
    return _over_participants(
        stimuli, actions, rewards, set_sizes, masks,
        alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon,
    )

- Shape flow:
    outer vmap unpacks axis 0 of each input
      -> inner vmap sees (B, 100) per data arg, scalar per param arg
      -> inner vmap unpacks axis 0 of each data arg
        -> _block sees (100,) per data arg, scalars per param
        -> returns scalar log-lik per block
      -> inner returns (B,) vector
      -> _per_participant sums to scalar
    -> outer returns (N,) vector.

- Critical dtype preservation: stimuli/actions stay int32 through both vmaps
  (numpy integer dtype is preserved by vmap). Params stay float32 (numpyro
  defaults to float32 under choice-only models; float64 only for M4).

## Pointwise log-lik path (bayesian_diagnostics.py)

Current: `compute_pointwise_log_lik` iterates participants (line 351) and
internally calls the stacked likelihood with `return_pointwise=True`. This
path is used for WAIC/LOO and runs POST-fit (not inside MCMC), so speed is
not critical. It must continue to work with the per-participant dict-of-dicts
format that bayesian_diagnostics consumes.

Decision: `prepare_stacked_participant_data` returns BOTH the legacy
per-participant dict (for bayesian_diagnostics) AND a cached stacked-across-
participants dict (for the hierarchical model). Concretely:

- `prepare_stacked_participant_data` unchanged signature, unchanged per-
  participant output.
- NEW top-level helper `stack_across_participants(participant_data_stacked)`
  called inside `fit_bayesian.py::_fit_stacked_model` just before the
  `mcmc.run(...)` call. Result passed as a new kwarg to the hierarchical
  model (see task 5).
- `compute_pointwise_log_lik` and `filter_padding_from_loglik` NOT modified.
  They continue to operate on the per-participant dict.

This preserves backward compatibility for M1/M2/M5/M6a/M6b (all still use the
Python for-loop). Only wmrl_m3 adopts the stacked-across-participants path.

</code_trace>

<tasks>

<task type="auto">
  <name>Task 1: Fix chain_method selection in run_inference and run_inference_with_bump</name>
  <files>scripts/fitting/numpyro_models.py</files>
  <action>
    Replace the two chain_method decision sites in numpyro_models.py
    (lines 605-607 in `run_inference` and lines 703-705 in
    `run_inference_with_bump`) with a single shared helper.

    Add a module-level helper function near the top of the file (below the
    imports but above `run_inference`):

        def _select_chain_method(num_chains: int) -> str:
            """Select chain_method based on JAX backend and device count.

            GPU: always "vectorized" unless multiple physical GPUs exist
            (then "parallel" across GPUs via pmap).
            CPU: "parallel" if set_host_device_count exposed enough devices,
            else "sequential".
            """
            import jax
            backend = jax.default_backend()
            n_devices = jax.local_device_count()
            if backend == "gpu":
                if n_devices >= num_chains:
                    return "parallel"
                return "vectorized"
            if backend == "tpu":
                return "parallel" if n_devices >= num_chains else "vectorized"
            # cpu
            return "parallel" if n_devices >= num_chains else "sequential"

    Then replace both inline rules with `_chain_method = _select_chain_method(num_chains)`.

    After the MCMC object is constructed in each function, add a single log
    line:

        print(f"   Backend: {jax.default_backend()} | chain_method: {_chain_method}")

    Do NOT remove the `numpyro.set_host_device_count(num_chains)` call in
    `run_inference_with_bump` — it remains correct for CPU. It is a no-op for
    GPU device count but configures the CPU backend safely.

    Parameters passed:
    - `_select_chain_method(num_chains: int) -> str`
    - `num_chains` comes from the `run_inference_with_bump(..., num_chains=4)`
      kwarg (default 4).
    - `jax.default_backend()` takes no arguments; returns 'cpu'/'gpu'/'tpu'.
    - `jax.local_device_count()` takes no arguments; returns int.
  </action>
  <verify>
    python -c "
    import jax, scripts.fitting.numpyro_models as m
    print('backend =', jax.default_backend())
    print('devices =', jax.local_device_count())
    print('method  =', m._select_chain_method(2))
    print('method4 =', m._select_chain_method(4))
    "
    # On CPU dev box: should print 'sequential' (1 device, 2 chains) unless
    # NUMPYRO_HOST_DEVICE_COUNT was set.
    # On GPU (run in smoke test): should print 'vectorized' for num_chains=2.

    python -m pytest scripts/fitting/tests/test_m3_hierarchical.py -q
    # Existing tests must still pass.
  </verify>
  <done>
    Both `run_inference` and `run_inference_with_bump` call
    `_select_chain_method(num_chains)` instead of inline rules. GPU +
    num_chains=2 resolves to "vectorized". CPU with 1 device + num_chains=2
    resolves to "sequential" (matches current behavior). CPU with
    `set_host_device_count(4)` + num_chains=4 resolves to "parallel" (matches
    current behavior). All existing M3 hierarchical tests pass.
  </done>
</task>

<task type="auto">
  <name>Task 2: Add GPU config diagnostic logging at MCMC kickoff and in smoke test</name>
  <files>scripts/fitting/numpyro_models.py, cluster/13_bayesian_pscan_smoke.slurm</files>
  <action>
    PART A (numpyro_models.py): In `run_inference_with_bump`, extend the
    existing startup banner (lines 686-694) with a self-contained GPU config
    block:

        import os
        print("=" * 60)
        print(">> JAX / NumPyro configuration")
        print(f"   backend            : {jax.default_backend()}")
        print(f"   devices            : {jax.devices()}")
        print(f"   local_device_count : {jax.local_device_count()}")
        print(f"   x64 enabled        : {jax.config.jax_enable_x64}")
        print(f"   chain_method       : {_select_chain_method(num_chains)}")
        print(f"   num_chains         : {num_chains}")
        for env_var in [
            "NUMPYRO_HOST_DEVICE_COUNT",
            "XLA_PYTHON_CLIENT_PREALLOCATE",
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            "JAX_COMPILATION_CACHE_DIR",
            "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES",
            "CUDA_VISIBLE_DEVICES",
        ]:
            print(f"   {env_var:32s} = {os.environ.get(env_var, '<unset>')}")
        print("=" * 60)

    Add this block immediately AFTER the existing "Starting MCMC sampling
    with convergence auto-bump..." line (line 686) and BEFORE the per-run
    for loop.

    Add timing checkpoints inside the for-loop over `target_accept_probs`
    (around line 697). Specifically:

        import time
        t0_compile = time.perf_counter()
        mcmc.run(rng_key, **model_args)
        t1_run = time.perf_counter()
        print(f"[timing] target_accept_prob={tap:.2f} wall={t1_run - t0_compile:.1f}s")

    This captures compile + sample wall time per target_accept_prob level.
    Leave the existing convergence-gate print line untouched.

    PART B (cluster/13_bayesian_pscan_smoke.slurm): After the existing "Verify
    GPU access" python block (lines 79-92), add a second diagnostic block
    that prints the same env-var table:

        python -c "
        import os
        import jax
        print('=' * 60)
        print('SMOKE TEST: JAX / NumPyro environment snapshot')
        print('=' * 60)
        print('backend:', jax.default_backend())
        print('devices:', jax.devices())
        print('local_device_count:', jax.local_device_count())
        print('x64:', jax.config.jax_enable_x64)
        for v in [
            'NUMPYRO_HOST_DEVICE_COUNT',
            'XLA_PYTHON_CLIENT_PREALLOCATE',
            'XLA_PYTHON_CLIENT_MEM_FRACTION',
            'JAX_COMPILATION_CACHE_DIR',
            'CUDA_VISIBLE_DEVICES',
        ]:
            print(f'{v:32s} = {os.environ.get(v, \"<unset>\")}')
        print('=' * 60)
        "

    Parameters passed:
    - `jax.default_backend()` -> str
    - `jax.devices()` -> list[Device]
    - `jax.local_device_count()` -> int
    - `jax.config.jax_enable_x64` -> bool
    - `os.environ.get(name, default)` -> str
    - `time.perf_counter()` -> float seconds
  </action>
  <verify>
    # PART A: run a trivial hierarchical fit locally and confirm banner prints.
    python -c "
    import jax, numpyro, numpyro.distributions as dist
    from scripts.fitting.numpyro_models import run_inference_with_bump
    def model():
        numpyro.sample('x', dist.Normal(0, 1))
    mcmc = run_inference_with_bump(
        model, model_args={}, num_warmup=10, num_samples=10, num_chains=1,
        target_accept_probs=(0.8,),
    )
    "
    # Expected: '>> JAX / NumPyro configuration' banner visible in stdout.

    # PART B: validate SLURM file syntax only (can't run sbatch locally).
    bash -n cluster/13_bayesian_pscan_smoke.slurm
    # Exit 0 = syntactically valid.
  </verify>
  <done>
    Startup log contains one banner block with backend, devices, chain_method,
    x64, num_chains, and 6 env vars. Each MCMC run prints a [timing] line
    with wall time. Smoke test SLURM file prints the same env-var snapshot
    after the GPU verification step. `bash -n` on the SLURM file succeeds.
  </done>
</task>

<task type="auto">
  <name>Task 3: Add stack_across_participants helper with uniform max_n_blocks padding</name>
  <files>scripts/fitting/numpyro_models.py</files>
  <action>
    Add a new module-level helper function immediately after
    `prepare_stacked_participant_data` (around line 995):

        def stack_across_participants(
            participant_data_stacked: dict[Any, dict[str, jnp.ndarray]],
        ) -> dict[str, jnp.ndarray]:
            """Stack per-participant arrays into (N, max_n_blocks, max_trials) tensors.

            Pads participants with fewer blocks to max_n_blocks by appending
            zero-mask blocks. Because mask=0 contributes exactly 0.0 to the
            block likelihood (both the log-prob term and the Q/WM updates are
            gated on mask), padded blocks leave total_ll invariant.

            Participant order follows sorted(participant_data_stacked.keys())
            — same order used by covariate_lec downstream.

            Parameters
            ----------
            participant_data_stacked : dict
                Output of prepare_stacked_participant_data. Per-participant
                arrays have shape (n_blocks_i, MAX_TRIALS_PER_BLOCK=100).

            Returns
            -------
            dict[str, jnp.ndarray]
                Keys (all shape (N, max_n_blocks, 100)):
                    * stimuli    : int32
                    * actions    : int32
                    * rewards    : float32
                    * set_sizes  : float32
                    * masks      : float32  (padded blocks are entirely 0.0)
                Plus:
                    * participant_ids : list[Any]  (ordered)
                    * n_blocks_per_ppt : jnp.ndarray shape (N,) int32
            """
            from scripts.fitting.jax_likelihoods import MAX_TRIALS_PER_BLOCK

            ppt_ids = sorted(participant_data_stacked.keys())
            max_n_blocks = max(
                participant_data_stacked[pid]["stimuli_stacked"].shape[0]
                for pid in ppt_ids
            )
            max_trials = MAX_TRIALS_PER_BLOCK  # 100

            def _pad_blocks(arr: jnp.ndarray, fill_value: float) -> jnp.ndarray:
                n_blocks_i = arr.shape[0]
                pad_blocks = max_n_blocks - n_blocks_i
                if pad_blocks == 0:
                    return arr
                pad_shape = (pad_blocks, max_trials)
                pad_arr = jnp.full(pad_shape, fill_value, dtype=arr.dtype)
                return jnp.concatenate([arr, pad_arr], axis=0)

            stacked = {
                "stimuli": jnp.stack([
                    _pad_blocks(participant_data_stacked[pid]["stimuli_stacked"], 0)
                    for pid in ppt_ids
                ]),
                "actions": jnp.stack([
                    _pad_blocks(participant_data_stacked[pid]["actions_stacked"], 0)
                    for pid in ppt_ids
                ]),
                "rewards": jnp.stack([
                    _pad_blocks(participant_data_stacked[pid]["rewards_stacked"], 0.0)
                    for pid in ppt_ids
                ]),
                "set_sizes": jnp.stack([
                    _pad_blocks(participant_data_stacked[pid]["set_sizes_stacked"], 6.0)
                    for pid in ppt_ids
                ]),
                "masks": jnp.stack([
                    _pad_blocks(participant_data_stacked[pid]["masks_stacked"], 0.0)
                    for pid in ppt_ids
                ]),
            }
            stacked["participant_ids"] = ppt_ids
            stacked["n_blocks_per_ppt"] = jnp.array(
                [participant_data_stacked[pid]["stimuli_stacked"].shape[0]
                 for pid in ppt_ids],
                dtype=jnp.int32,
            )
            return stacked

    Notes on fill values:
    - stimuli: 0 (any valid stimulus index 0..num_stimuli-1 is fine since
      mask=0 disables the update).
    - actions: 0 (same reasoning).
    - rewards: 0.0.
    - set_sizes: 6.0 (safe default since mask=0 disables updates but the
      softmax still evaluates numerically; 6.0 is within the valid range).
    - masks: 0.0 (the load-bearing value).

    Parameters passed:
    - `stack_across_participants(participant_data_stacked)`:
        input dict keys per participant contain jnp.ndarray of shape
        (n_blocks_i, 100).
    - Output keys stimuli/actions/rewards/set_sizes/masks have shape
      (N, max_n_blocks, 100). N = len(participant_ids). max_n_blocks is the
      max over participants (expected 17 for the production dataset).
  </action>
  <verify>
    python -c "
    import jax.numpy as jnp
    import pandas as pd
    from scripts.fitting.numpyro_models import (
        prepare_stacked_participant_data,
        stack_across_participants,
    )

    # Synthetic: 3 participants with variable blocks (2, 3, 2)
    rows = []
    for pid, n_blocks in [('A', 2), ('B', 3), ('C', 2)]:
        for b in range(n_blocks):
            for t in range(10):
                rows.append({
                    'sona_id': pid, 'block': b, 'stimulus': t % 3,
                    'key_press': t % 3, 'reward': float(t % 2),
                    'set_size': 3.0,
                })
    df = pd.DataFrame(rows)
    pdata = prepare_stacked_participant_data(df)
    stacked = stack_across_participants(pdata)
    print('N          :', len(stacked['participant_ids']))
    print('max_blocks :', stacked['stimuli'].shape[1])
    print('n_blocks   :', stacked['n_blocks_per_ppt'])
    print('masks shape:', stacked['masks'].shape)
    assert stacked['stimuli'].shape == (3, 3, 100)
    assert stacked['masks'].shape == (3, 3, 100)
    # Participant 'A' with 2 real blocks -> block index 2 should be all-zero mask
    assert float(stacked['masks'][0, 2].sum()) == 0.0
    # Participant 'B' with 3 real blocks -> all 3 blocks have real trials
    assert float(stacked['masks'][1, 0].sum()) == 10.0  # 10 real trials in block 0
    print('PASS')
    "
  </verify>
  <done>
    `stack_across_participants` returns a dict with shape (N, max_n_blocks, 100)
    per data key. Padded blocks have masks_stacked identically 0.0. Participant
    order matches `sorted(participant_data_stacked.keys())`. Verification
    script prints "PASS".
  </done>
</task>

<task type="auto">
  <name>Task 4: Add wmrl_m3_fully_batched_likelihood via nested vmap over participants and blocks</name>
  <files>scripts/fitting/jax_likelihoods.py</files>
  <action>
    Add a new function to jax_likelihoods.py, placed directly AFTER
    `wmrl_m3_multiblock_likelihood_stacked` (around line 2171, before the
    existing `wmrl_block_likelihood_jit` JIT registration block):

        def wmrl_m3_fully_batched_likelihood(
            stimuli: jnp.ndarray,          # (N, B, T) int32
            actions: jnp.ndarray,          # (N, B, T) int32
            rewards: jnp.ndarray,          # (N, B, T) float32
            set_sizes: jnp.ndarray,        # (N, B, T) float32
            masks: jnp.ndarray,            # (N, B, T) float32
            alpha_pos: jnp.ndarray,        # (N,) float32
            alpha_neg: jnp.ndarray,        # (N,) float32
            phi: jnp.ndarray,              # (N,) float32
            rho: jnp.ndarray,              # (N,) float32
            capacity: jnp.ndarray,         # (N,) float32
            kappa: jnp.ndarray,            # (N,) float32
            epsilon: jnp.ndarray,          # (N,) float32
            num_stimuli: int = 6,
            num_actions: int = 3,
            q_init: float = 0.5,
            wm_init: float = 1.0 / 3.0,
        ) -> jnp.ndarray:
            """Fully-batched WM-RL M3 log-likelihood via nested vmap.

            Pattern:
              outer vmap over participants (axis 0 of every input)
                -> inner vmap over blocks (axis 0 of per-participant data)
                  -> wmrl_m3_block_likelihood on (T,) slices and scalar params
                -> sum over blocks -> scalar per participant
              -> (N,) vector returned

            CRITICAL: this uses the SAME block likelihood as the sequential
            path (wmrl_m3_block_likelihood). Q, WM, and perseveration state
            all reset at block entry, so blocks are independent and vmap is
            correct per Senta 2025 (see MODEL_REFERENCE.md:212, 306, 665, 717).

            Padded blocks (mask entirely 0.0) contribute 0.0 because the
            inner scan gates every likelihood and state update on mask[t].

            Parameters
            ----------
            stimuli, actions, rewards, set_sizes, masks : (N, B, T) arrays
                Per-participant stacked data. Dimension 0 is participant,
                dimension 1 is block, dimension 2 is trial. B = max_n_blocks.
            alpha_pos, ..., epsilon : (N,) arrays
                Per-participant parameter values.
            num_stimuli, num_actions, q_init, wm_init : static
                Same as wmrl_m3_block_likelihood.

            Returns
            -------
            jnp.ndarray
                Shape (N,) float — total log-likelihood per participant.
            """
            def _block_ll(
                stim, act, rew, ss, mask,
                ap, an, ph, rh, cap, k, e,
            ):
                # Scalar log-lik for a single (participant, block).
                return wmrl_m3_block_likelihood(
                    stimuli=stim,
                    actions=act,
                    rewards=rew,
                    set_sizes=ss,
                    alpha_pos=ap,
                    alpha_neg=an,
                    phi=ph,
                    rho=rh,
                    capacity=cap,
                    kappa=k,
                    epsilon=e,
                    num_stimuli=num_stimuli,
                    num_actions=num_actions,
                    q_init=q_init,
                    wm_init=wm_init,
                    mask=mask,
                    return_pointwise=False,
                )

            # Inner vmap: over blocks. Data args on axis 0, params broadcast.
            _over_blocks = jax.vmap(
                _block_ll,
                in_axes=(0, 0, 0, 0, 0, None, None, None, None, None, None, None),
                out_axes=0,
            )

            def _participant_ll(
                stim, act, rew, ss, mask,
                ap, an, ph, rh, cap, k, e,
            ):
                block_lls = _over_blocks(
                    stim, act, rew, ss, mask, ap, an, ph, rh, cap, k, e,
                )
                return block_lls.sum()

            # Outer vmap: over participants. Everything on axis 0.
            _over_participants = jax.vmap(
                _participant_ll,
                in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                out_axes=0,
            )
            return _over_participants(
                stimuli, actions, rewards, set_sizes, masks,
                alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon,
            )

    Do NOT add a new jax.jit decoration here. NumPyro will trace and compile
    the whole model graph; top-level jit on the likelihood would fight with
    NumPyro's own tracing.

    Parameters passed (summary):
    - wmrl_m3_fully_batched_likelihood takes 5 data args (N, B, T), 7
      parameter args (N,), and 4 static kwargs.
    - Inner call to wmrl_m3_block_likelihood receives (T,)-shape data slices
      and scalar parameters; matches the existing signature exactly.
    - Static kwargs (num_stimuli, num_actions, q_init, wm_init) are closed
      over by the inner lambdas and appear as Python closure variables in
      the vmap'd function — NOT as mapped args.
  </action>
  <verify>
    python -c "
    import jax.numpy as jnp
    from scripts.fitting.jax_likelihoods import (
        wmrl_m3_fully_batched_likelihood,
        wmrl_m3_block_likelihood,
    )

    N, B, T = 3, 2, 10
    stim = jnp.zeros((N, B, T), dtype=jnp.int32)
    act  = jnp.zeros((N, B, T), dtype=jnp.int32)
    rew  = jnp.ones((N, B, T),  dtype=jnp.float32) * 0.5
    ss   = jnp.ones((N, B, T),  dtype=jnp.float32) * 3.0
    mask = jnp.ones((N, B, T),  dtype=jnp.float32)
    ap = jnp.full((N,), 0.3, jnp.float32)
    an = jnp.full((N,), 0.2, jnp.float32)
    ph = jnp.full((N,), 0.1, jnp.float32)
    rh = jnp.full((N,), 0.5, jnp.float32)
    cap= jnp.full((N,), 3.0, jnp.float32)
    k  = jnp.full((N,), 0.2, jnp.float32)
    e  = jnp.full((N,), 0.05, jnp.float32)

    out = wmrl_m3_fully_batched_likelihood(
        stim, act, rew, ss, mask, ap, an, ph, rh, cap, k, e,
    )
    print('shape:', out.shape, 'dtype:', out.dtype)
    assert out.shape == (N,)
    assert out.dtype == jnp.float32
    # Compare first participant against direct block call
    direct = wmrl_m3_block_likelihood(
        stimuli=stim[0, 0], actions=act[0, 0], rewards=rew[0, 0],
        set_sizes=ss[0, 0], alpha_pos=ap[0], alpha_neg=an[0],
        phi=ph[0], rho=rh[0], capacity=cap[0], kappa=k[0], epsilon=e[0],
        mask=mask[0, 0],
    ) + wmrl_m3_block_likelihood(
        stimuli=stim[0, 1], actions=act[0, 1], rewards=rew[0, 1],
        set_sizes=ss[0, 1], alpha_pos=ap[0], alpha_neg=an[0],
        phi=ph[0], rho=rh[0], capacity=cap[0], kappa=k[0], epsilon=e[0],
        mask=mask[0, 1],
    )
    rel_err = abs(float(out[0]) - float(direct)) / (abs(float(direct)) + 1e-9)
    print(f'rel_err[0] = {rel_err:.2e}')
    assert rel_err < 1e-5, f'Agreement failed: {rel_err}'
    print('PASS')
    "
  </verify>
  <done>
    `wmrl_m3_fully_batched_likelihood` returns (N,) float32 log-lik vector.
    For N=3, B=2, T=10 smoke test, participant 0's output matches the sum
    of two direct block likelihoods at relative error < 1e-5. Function is
    importable from jax_likelihoods.
  </done>
</task>

<task type="auto">
  <name>Task 5: Refactor wmrl_m3_hierarchical_model to use fully-batched likelihood with single numpyro.factor</name>
  <files>scripts/fitting/numpyro_models.py, scripts/fitting/fit_bayesian.py</files>
  <action>
    PART A (numpyro_models.py): Replace the body of `wmrl_m3_hierarchical_model`
    (lines 1108-1139, the "Likelihood via numpyro.factor" for-loop block)
    with a single fully-batched call.

    Add a new kwarg to the function signature:

        def wmrl_m3_hierarchical_model(
            participant_data_stacked: dict,
            covariate_lec: jnp.ndarray | None = None,
            num_stimuli: int = 6,
            num_actions: int = 3,
            q_init: float = 0.5,
            wm_init: float = 1.0 / 3.0,
            use_pscan: bool = False,
            stacked_arrays: dict | None = None,   # NEW
        ) -> None:

    Semantics of `stacked_arrays`:
    - If provided, it must be the output of `stack_across_participants()`:
      dict with keys stimuli/actions/rewards/set_sizes/masks (N, B, 100),
      participant_ids (ordered list), n_blocks_per_ppt (N,).
    - If None, call `stack_across_participants(participant_data_stacked)`
      internally. (Convenient for tests; production code in fit_bayesian.py
      will pass it explicitly to avoid recomputing inside MCMC.)
    - If `use_pscan=True`, raise `NotImplementedError("pscan + vmap combo "
      "is out of scope for this quick task; pass use_pscan=False.")`.

    Replace the for-loop (lines 1113-1139) with:

        if use_pscan:
            raise NotImplementedError(
                "wmrl_m3_hierarchical_model: use_pscan + fully-batched vmap "
                "path is not implemented in quick-007. Use use_pscan=False, "
                "or revert to the sequential for-loop model (not exposed)."
            )

        if stacked_arrays is None:
            stacked_arrays = stack_across_participants(participant_data_stacked)

        # Import here to keep the top-of-file imports narrow.
        from scripts.fitting.jax_likelihoods import wmrl_m3_fully_batched_likelihood

        per_participant_ll = wmrl_m3_fully_batched_likelihood(
            stimuli=stacked_arrays["stimuli"],
            actions=stacked_arrays["actions"],
            rewards=stacked_arrays["rewards"],
            set_sizes=stacked_arrays["set_sizes"],
            masks=stacked_arrays["masks"],
            alpha_pos=sampled["alpha_pos"],
            alpha_neg=sampled["alpha_neg"],
            phi=sampled["phi"],
            rho=sampled["rho"],
            capacity=sampled["capacity"],
            kappa=sampled["kappa"],
            epsilon=sampled["epsilon"],
            num_stimuli=num_stimuli,
            num_actions=num_actions,
            q_init=q_init,
            wm_init=wm_init,
        )
        numpyro.factor("obs", per_participant_ll.sum())

    PART B (fit_bayesian.py): In `_fit_stacked_model` around line 378 where
    model_args is built for wmrl_m3:

        from scripts.fitting.numpyro_models import stack_across_participants
        ...
        if model_name == "wmrl_m3":
            stacked_arrays = stack_across_participants(participant_data_stacked)
            model_args = {
                "participant_data_stacked": participant_data_stacked,
                "covariate_lec": covariate_lec,
                "stacked_arrays": stacked_arrays,
                # ...existing kwargs (num_stimuli, num_actions, q_init, wm_init,
                # use_pscan=False)...
            }

    Leave model_args unchanged for all other models (qlearning, wmrl,
    wmrl_m5, wmrl_m6a, wmrl_m6b, wmrl_m4). They continue to use the
    per-participant for-loop path.

    Remove the outdated docstring line in wmrl_m3_hierarchical_model:
      "Likelihood is accumulated via numpyro.factor in a Python for-loop over
       participants (matches existing qlearning/wmrl models; vmap not applicable here)."
    Replace with:
      "Likelihood is accumulated via a single numpyro.factor('obs', ...) call
       using wmrl_m3_fully_batched_likelihood (nested vmap over participants
       and blocks). Blocks are independent per Senta 2025 (Q/WM/perseveration
       reset at block boundaries)."

    Parameters passed (Task 5 summary):
    - wmrl_m3_hierarchical_model now accepts `stacked_arrays` (dict) kwarg.
    - stacked_arrays keys feed into wmrl_m3_fully_batched_likelihood exactly:
      stimuli/actions/rewards/set_sizes/masks -> (N, B, 100) arrays;
      plus per-participant parameter arrays sampled["..."] shape (N,).
    - `covariate_lec` shape (N,) remains hooked into kappa_unc computation on
      line 1099; unchanged by this task.
    - `numpyro.factor("obs", scalar)` replaces 154 `numpyro.factor(f"obs_p{pid}", ...)`.
  </action>
  <verify>
    # Compilation smoke test (no real MCMC)
    python -c "
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import MCMC, NUTS
    import pandas as pd
    from scripts.fitting.numpyro_models import (
        prepare_stacked_participant_data,
        stack_across_participants,
        wmrl_m3_hierarchical_model,
    )

    # Small synthetic dataset: 3 ppts, 2 blocks each, 10 trials
    rows = []
    for pid in ['A', 'B', 'C']:
        for b in range(2):
            for t in range(10):
                rows.append({'sona_id': pid, 'block': b, 'stimulus': t%3,
                             'key_press': t%3, 'reward': float(t%2),
                             'set_size': 3.0})
    df = pd.DataFrame(rows)
    pdata = prepare_stacked_participant_data(df)
    stacked = stack_across_participants(pdata)

    kernel = NUTS(wmrl_m3_hierarchical_model, target_accept_prob=0.8)
    mcmc = MCMC(kernel, num_warmup=5, num_samples=5, num_chains=1,
                progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0),
             participant_data_stacked=pdata,
             stacked_arrays=stacked)
    # Must have a single 'obs' factor site, not per-participant sites
    samples = mcmc.get_samples()
    assert 'alpha_pos' in samples
    print('PASS - single obs factor, samples ok')
    "
  </verify>
  <done>
    `wmrl_m3_hierarchical_model` has `stacked_arrays` kwarg, raises
    NotImplementedError on use_pscan=True, calls
    `wmrl_m3_fully_batched_likelihood` once, and emits a single
    `numpyro.factor("obs", ...)` call. `_fit_stacked_model` for wmrl_m3
    computes stacked_arrays once and passes it through model_args. Compile
    smoke test runs 5 warmup + 5 samples on CPU without error. Other 5
    models' model_args are unchanged.
  </done>
</task>

<task type="auto">
  <name>Task 6: Add agreement test between sequential and fully-batched M3 likelihoods</name>
  <files>scripts/fitting/tests/test_m3_hierarchical.py</files>
  <action>
    Append a new test function to
    `scripts/fitting/tests/test_m3_hierarchical.py`. Do NOT modify existing
    tests.

    Test:

        def test_wmrl_m3_fully_batched_matches_sequential():
            """Fully-batched vmap'd M3 likelihood agrees with sequential.

            Generates N=5 synthetic participants with variable n_blocks
            (12 or 17) and 100-trial blocks. For 3 random parameter draws,
            computes total NLL both ways and asserts relative error < 1e-4.

            This is the correctness gate for the vmap refactor (Task 4/5).
            """
            import numpy as np
            import jax
            import jax.numpy as jnp
            import pandas as pd
            from scripts.fitting.numpyro_models import (
                prepare_stacked_participant_data,
                stack_across_participants,
            )
            from scripts.fitting.jax_likelihoods import (
                wmrl_m3_fully_batched_likelihood,
                wmrl_m3_multiblock_likelihood_stacked,
            )

            rng = np.random.default_rng(42)

            # N=5 participants; n_blocks in {12, 17}; T = full MAX_TRIALS_PER_BLOCK=100
            ppt_configs = [("P0", 12), ("P1", 17), ("P2", 12), ("P3", 17), ("P4", 13)]
            rows = []
            for pid, n_blocks in ppt_configs:
                for b in range(n_blocks):
                    trials_in_block = int(rng.integers(60, 100))
                    for t in range(trials_in_block):
                        rows.append({
                            "sona_id": pid,
                            "block": b,
                            "stimulus": int(rng.integers(0, 6)),
                            "key_press": int(rng.integers(0, 3)),
                            "reward": float(rng.integers(0, 2)),
                            "set_size": float(rng.choice([2.0, 3.0, 6.0])),
                        })
            df = pd.DataFrame(rows)

            pdata = prepare_stacked_participant_data(df)
            stacked = stack_across_participants(pdata)
            N = len(stacked["participant_ids"])
            assert N == 5

            for draw_idx in range(3):
                # Random but reasonable parameter draws
                key = jax.random.PRNGKey(1000 + draw_idx)
                k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
                alpha_pos = jax.random.uniform(k1, (N,), minval=0.1, maxval=0.9)
                alpha_neg = jax.random.uniform(k2, (N,), minval=0.1, maxval=0.9)
                phi       = jax.random.uniform(k3, (N,), minval=0.05, maxval=0.5)
                rho       = jax.random.uniform(k4, (N,), minval=0.2, maxval=0.9)
                capacity  = jax.random.uniform(k5, (N,), minval=2.0, maxval=6.0)
                kappa     = jax.random.uniform(k6, (N,), minval=0.0, maxval=0.5)
                epsilon   = jax.random.uniform(k7, (N,), minval=0.01, maxval=0.1)

                # Path A: fully-batched
                batched_ll = wmrl_m3_fully_batched_likelihood(
                    stimuli=stacked["stimuli"],
                    actions=stacked["actions"],
                    rewards=stacked["rewards"],
                    set_sizes=stacked["set_sizes"],
                    masks=stacked["masks"],
                    alpha_pos=alpha_pos, alpha_neg=alpha_neg, phi=phi,
                    rho=rho, capacity=capacity, kappa=kappa, epsilon=epsilon,
                )
                # Shape check
                assert batched_ll.shape == (N,)

                # Path B: sequential per-participant
                seq_lls = []
                for idx, pid in enumerate(stacked["participant_ids"]):
                    pp = pdata[pid]
                    ll_i = wmrl_m3_multiblock_likelihood_stacked(
                        stimuli_stacked=pp["stimuli_stacked"],
                        actions_stacked=pp["actions_stacked"],
                        rewards_stacked=pp["rewards_stacked"],
                        set_sizes_stacked=pp["set_sizes_stacked"],
                        masks_stacked=pp["masks_stacked"],
                        alpha_pos=float(alpha_pos[idx]),
                        alpha_neg=float(alpha_neg[idx]),
                        phi=float(phi[idx]),
                        rho=float(rho[idx]),
                        capacity=float(capacity[idx]),
                        kappa=float(kappa[idx]),
                        epsilon=float(epsilon[idx]),
                    )
                    seq_lls.append(float(ll_i))

                batched_np = np.array(batched_ll)
                seq_np = np.array(seq_lls, dtype=np.float32)

                # Per-participant relative error
                rel_err = np.abs(batched_np - seq_np) / (np.abs(seq_np) + 1e-9)
                max_rel_err = float(rel_err.max())
                print(f"draw {draw_idx}: max_rel_err = {max_rel_err:.2e}")
                assert max_rel_err < 1e-4, (
                    f"Agreement failed: draw={draw_idx}, "
                    f"expected max_rel_err < 1e-4, got {max_rel_err:.2e}. "
                    f"batched={batched_np}, seq={seq_np}"
                )

    Parameters passed:
    - prepare_stacked_participant_data(df) -> dict[pid, dict[str, jnp.ndarray]]
    - stack_across_participants(pdata) -> dict with (N, B, 100) keys
    - wmrl_m3_fully_batched_likelihood: (N, B, 100) data + (N,) params
    - wmrl_m3_multiblock_likelihood_stacked: (B_i, 100) data + scalar params
    - All test assertions operate on numpy arrays.
  </action>
  <verify>
    python -m pytest scripts/fitting/tests/test_m3_hierarchical.py::test_wmrl_m3_fully_batched_matches_sequential -v -s
    # Should print 3 "draw N: max_rel_err = ..." lines with all < 1e-4, then PASS.
  </verify>
  <done>
    New test in test_m3_hierarchical.py passes on 3 random draws across 5
    participants with variable n_blocks (12/13/17). Per-participant relative
    error is < 1e-4 for every draw. Existing tests in the file still pass
    (no regressions).
  </done>
</task>

<task type="auto">
  <name>Task 7: Run full test suite and targeted M3 smoke on CPU to validate end-to-end pipeline</name>
  <files>(validation only; no files modified)</files>
  <action>
    Run the existing test suites in order:

    1. M3 hierarchical tests (correctness of vmap + refactor):
       python -m pytest scripts/fitting/tests/test_m3_hierarchical.py -v

    2. Pointwise log-lik tests (ensures diagnostics pipeline unaffected):
       python -m pytest scripts/fitting/tests/test_pointwise_loglik.py -v

    3. Compile gate (ensures all models still compile):
       python -m pytest scripts/fitting/tests/test_compile_gate.py -v

    4. NumPyro helpers (sample_bounded_param regression check):
       python -m pytest scripts/fitting/tests/test_numpyro_helpers.py -v

    5. Minimal wmrl_m3 fit on a tiny subset (CPU only, smoke path):
       python scripts/fitting/fit_bayesian.py \
           --model wmrl_m3 \
           --data output/task_trials_long.csv \
           --chains 1 \
           --warmup 20 \
           --samples 20 \
           --max-tree-depth 4 \
           --seed 42 \
           --output output/quick_007_smoke

       Expected: produces a run_metadata.json and a posterior.nc file under
       output/quick_007_smoke/. Non-zero exit is acceptable IF it is the
       convergence gate (not a crash). A crash during model tracing/compile
       is a failure.

    6. Inspect the stdout log:
       - Confirm the JAX / NumPyro configuration banner prints (from task 2).
       - Confirm "chain_method:" appears in the banner with the expected value
         ("sequential" on 1-device CPU with 1 chain; "parallel" on CPU with
          set_host_device_count(num_chains) if configured).
       - Confirm NO "obs_p..." factor sites appear in `mcmc.print_summary()`
         output (only "obs" and the sampled parameters).

    If any of steps 1-4 fail, STOP and fix the regression before proceeding.
    If step 5 crashes during compile (exit != 0 AND no run_metadata.json
    written), STOP and diagnose. If step 5 exits non-zero but writes metadata
    and posterior, the convergence gate tripped — acceptable for this smoke
    run since warmup=20 is too small for real convergence.

    Parameters passed: N/A (this task only runs existing scripts).
  </action>
  <verify>
    # Full test pass + smoke fit completed
    test $? -eq 0 || { echo "Step 1-4 test failed — revert and debug"; exit 1; }
    # Step 5 smoke check:
    test -f output/quick_007_smoke/wmrl_m3_run_metadata.json && \
        echo "Smoke fit produced metadata file" || \
        { echo "Smoke fit crashed during compile/trace — FAIL"; exit 1; }
  </verify>
  <done>
    Steps 1-4 all pass with no regressions. Step 5 produces
    output/quick_007_smoke/wmrl_m3_run_metadata.json (even if convergence
    gate tripped). Stdout shows the GPU config banner and "chain_method:"
    line. `mcmc.print_summary()` output contains an "obs" factor but no
    per-participant `obs_p{pid}` sites.
  </done>
</task>

</tasks>

<verification>

Run the full verification sequence after all tasks complete:

1. Unit tests: `python -m pytest scripts/fitting/tests/ -v -k "m3 or hierarchical or pointwise or helpers"`.

2. Syntactic check of SLURM: `bash -n cluster/13_bayesian_pscan_smoke.slurm`.

3. Smoke end-to-end: `python scripts/fitting/fit_bayesian.py --model wmrl_m3 --data output/task_trials_long.csv --chains 1 --warmup 20 --samples 20 --max-tree-depth 4 --output output/quick_007_smoke`.

4. Grep for leftover references to the old per-participant factor:
   `grep -rn "obs_p{" scripts/fitting/numpyro_models.py` (should NOT match
   inside wmrl_m3_hierarchical_model; other models unchanged).

5. Grep for the new batched function:
   `grep -n "wmrl_m3_fully_batched_likelihood" scripts/fitting/jax_likelihoods.py scripts/fitting/numpyro_models.py` (should show 1 definition + 1 call site).

6. Grep for the chain_method selector: `grep -n "_select_chain_method" scripts/fitting/numpyro_models.py` (should show 1 definition + 2 usages).

7. Cluster validation (deferred — user runs after plan commit):
   `sbatch cluster/13_bayesian_pscan_smoke.slurm`. Check the new GPU config
   block at top of the log; check chain_method="vectorized"; check wmrl_m3
   wall time per iteration in Stage 2.

</verification>

<success_criteria>

- `_select_chain_method(num_chains)` resolves to "vectorized" when
  `jax.default_backend() == "gpu"` and `jax.local_device_count() < num_chains`,
  and to "parallel" when `local_device_count >= num_chains`.
- GPU config banner printed once per `run_inference_with_bump` call with all
  7 fields (backend, devices, local_device_count, x64, chain_method,
  num_chains) plus 6 env var values.
- Per-target-accept-prob [timing] wall-time line appears for each MCMC run.
- `stack_across_participants` returns (N, max_n_blocks, 100) tensors with
  `masks` = 0.0 on padded blocks and `n_blocks_per_ppt` recorded.
- `wmrl_m3_fully_batched_likelihood` returns shape (N,) float32 and matches
  the sequential loop at max rel err < 1e-4 for the 3-draw agreement test.
- `wmrl_m3_hierarchical_model` uses exactly one `numpyro.factor("obs", ...)`
  call (grep confirms no `obs_p{pid}` remaining in that function).
- `use_pscan=True` on the wmrl_m3 hierarchical model raises
  NotImplementedError with a message referencing quick-007.
- `_fit_stacked_model` in fit_bayesian.py passes `stacked_arrays` into
  model_args for model wmrl_m3 only (other 5 models unchanged).
- Existing tests in test_m3_hierarchical.py, test_pointwise_loglik.py,
  test_compile_gate.py, test_numpyro_helpers.py continue to pass.
- CPU end-to-end smoke fit for wmrl_m3 produces run_metadata.json (no compile
  crash).
- Research findings and code trace are documented in this PLAN.md (sections
  <research_findings> and <code_trace>); no separate research file created.
- `cluster/13_bayesian_pscan_smoke.slurm` prints a second "SMOKE TEST: JAX /
  NumPyro environment snapshot" block before Stage 1.

</success_criteria>

<output>

After completion, create
`.planning/quick/007-gpu-mcmc-speedup-vmap-chain-method/007-SUMMARY.md`
following the standard summary template. Document:

- Which three root causes were addressed and how.
- Agreement test results (max relative error per draw).
- Whether the M3-only POC is ready for cluster validation.
- Explicit note: other 5 models (M1/M2/M5/M6a/M6b/M4) are UNCHANGED and
  continue to use the Python for-loop. Follow-on quick task will propagate.
- pmap research conclusion: single-GPU SLURM is the constraint; chain_method
  "vectorized" is correct for the Monash M3 workload; "parallel" path
  remains available if multi-GPU SLURM becomes standard.
- Observed GPU wall time per iteration after the cluster smoke run (populate
  this AFTER `sbatch cluster/13_bayesian_pscan_smoke.slurm` returns).

</output>
