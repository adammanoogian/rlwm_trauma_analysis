#!/usr/bin/env python3
"""
GPU Performance Diagnostic Script

Run this to identify where the bottleneck is in MLE fitting.
"""
import sys
import time
import os

# Force unbuffered output for SLURM jobs
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

sys.path.insert(0, '.')

def log(msg):
    """Print with timestamp and flush."""
    import builtins
    builtins.print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

log("=" * 55)
log("GPU PERFORMANCE DIAGNOSTIC")
log("=" * 55)

log("Importing JAX...")
import jax
import jax.numpy as jnp

log("Importing jaxopt...")
from jaxopt import LBFGS

log("Importing numpy...")
import numpy as np

devices = jax.devices()
log(f"JAX devices: {devices}")
gpu_devices = [d for d in devices if d.platform == "gpu"]
log(f"GPU available: {len(gpu_devices) > 0}")

# Import our likelihood functions
log("Importing likelihood functions...")
from jax_likelihoods import (
    q_learning_multiblock_likelihood,
    q_learning_multiblock_likelihood_stacked,
    q_learning_block_likelihood,
)
from mle_utils import jax_unconstrained_to_params_qlearning

# Create synthetic data matching real data structure
log("Creating synthetic data (17 blocks x 100 trials)...")
np.random.seed(42)
n_blocks = 17
n_trials = 100

stimuli_blocks = [jnp.array(np.random.randint(0, 6, n_trials), dtype=jnp.int32) for _ in range(n_blocks)]
actions_blocks = [jnp.array(np.random.randint(0, 3, n_trials), dtype=jnp.int32) for _ in range(n_blocks)]
rewards_blocks = [jnp.array(np.random.binomial(1, 0.7, n_trials), dtype=jnp.float32) for _ in range(n_blocks)]
masks_blocks = [jnp.ones(n_trials, dtype=jnp.float32) for _ in range(n_blocks)]

# Pre-stack for objective
log("Stacking data...")
stimuli_stacked = jnp.stack(stimuli_blocks)
actions_stacked = jnp.stack(actions_blocks)
rewards_stacked = jnp.stack(rewards_blocks)
masks_stacked = jnp.stack(masks_blocks)

log(f"Data shapes: stimuli={stimuli_stacked.shape}, masks={masks_stacked.shape}")

# =============================================================================
# TEST 1: Raw likelihood function speed (STACKED version - optimized)
# =============================================================================
log("=" * 55)
log("TEST 1: Single likelihood evaluation (stacked version)")
log("=" * 55)

def objective_raw(x):
    alpha_pos, alpha_neg, epsilon = jax_unconstrained_to_params_qlearning(x)
    # Use stacked version directly - no list conversion overhead
    return -q_learning_multiblock_likelihood_stacked(
        stimuli_stacked=stimuli_stacked,
        actions_stacked=actions_stacked,
        rewards_stacked=rewards_stacked,
        masks_stacked=masks_stacked,
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
        epsilon=epsilon,
    )

objective_jit = jax.jit(objective_raw)
x0 = jnp.array([0.0, 0.0, -2.0])

# First call (JIT compilation)
log("First call (includes JIT compilation)...")
start = time.time()
result = objective_jit(x0)
jax.block_until_ready(result)
compile_time = time.time() - start
log(f"  Time: {compile_time:.3f}s")

# Subsequent calls (should be fast)
log("Next 100 calls (compiled)...")
start = time.time()
for _ in range(100):
    result = objective_jit(x0)
    jax.block_until_ready(result)
eval_time = time.time() - start
log(f"  Total: {eval_time:.3f}s ({eval_time/100*1000:.2f}ms per call)")
log()

# =============================================================================
# TEST 2: LBFGS solver with jit=True (jaxopt internal JIT - RECOMMENDED)
# =============================================================================
log("=" * 55)
log("TEST 2: LBFGS solver with jit=True (jaxopt handles JIT internally)")
log("=" * 55)

solver = LBFGS(fun=objective_jit, maxiter=100, tol=1e-6, jit=True)

log("First optimization (includes internal JIT compilation)...")
start = time.time()
x0 = jnp.array([0.0, 0.0, -2.0])
params, state = solver.run(x0)
jax.block_until_ready(params)
internal_jit_compile = time.time() - start
log(f"  Time: {internal_jit_compile:.2f}s")

log("Next 10 optimizations (should reuse cached compilation)...")
start = time.time()
for i in range(10):
    x0 = jnp.array([np.random.randn(), np.random.randn(), -2.0])
    params, state = solver.run(x0)
    jax.block_until_ready(params)
internal_jit_time = time.time() - start
log(f"  Total: {internal_jit_time:.2f}s ({internal_jit_time/10:.2f}s per start)")
log()

# =============================================================================
# TEST 3: LBFGS solver with jax.jit(solver.run) wrapper
# NOTE: This often fails with TracerBoolConversionError - that's expected!
# =============================================================================
log("=" * 55)
log("TEST 3: LBFGS solver WITH jax.jit(solver.run) wrapper")
log("=" * 55)
log("(This approach usually fails - testing to confirm)")

try:
    solver2 = LBFGS(fun=objective_jit, maxiter=100, tol=1e-6, jit=False)
    jit_run = jax.jit(solver2.run)

    start = time.time()
    x0 = jnp.array([0.0, 0.0, -2.0])
    params, state = jit_run(x0)
    jax.block_until_ready(params)
    solver_compile_time = time.time() - start
    log(f"  Surprisingly worked! Time: {solver_compile_time:.2f}s")
    jit_time = 0.0  # Placeholder
except Exception as e:
    log(f"  FAILED (as expected): {type(e).__name__}")
    log(f"  This confirms jax.jit(solver.run) doesn't work with jaxopt")
    solver_compile_time = float('inf')
    jit_time = float('inf')
log()

# =============================================================================
# TEST 4: VECTORIZED optimization with vmap (NEW - should be fastest!)
# =============================================================================
log("=" * 55)
log("TEST 4: VECTORIZED optimization with jax.vmap(solver.run)")
log("=" * 55)

solver3 = LBFGS(fun=objective_jit, maxiter=100, tol=1e-6, jit=True)
vmap_run = jax.vmap(solver3.run)

# Generate batch of starting points
n_starts_batch = 20
x0_batch = jnp.array([[np.random.randn(), np.random.randn(), -2.0] for _ in range(n_starts_batch)])

log(f"Running {n_starts_batch} optimizations IN PARALLEL with vmap...")
log("First batch (includes vmap compilation)...")
start = time.time()
all_params, all_states = vmap_run(x0_batch)
jax.block_until_ready(all_params)
vmap_first = time.time() - start
log(f"  Time for {n_starts_batch} parallel starts: {vmap_first:.2f}s")

log("Second batch (compiled)...")
x0_batch2 = jnp.array([[np.random.randn(), np.random.randn(), -2.0] for _ in range(n_starts_batch)])
start = time.time()
all_params2, all_states2 = vmap_run(x0_batch2)
jax.block_until_ready(all_params2)
vmap_second = time.time() - start
log(f"  Time for {n_starts_batch} parallel starts: {vmap_second:.2f}s")
log(f"  Effective per-start: {vmap_second/n_starts_batch:.3f}s")
log()

# =============================================================================
# SUMMARY
# =============================================================================
log("=" * 55)
log("SUMMARY")
log("=" * 55)
log(f"Likelihood compile time:      {compile_time:.2f}s")
log(f"Likelihood per-call:          {eval_time/100*1000:.2f}ms")
log()
log(f"Sequential (jit=True):")
log(f"  First call (compile):       {internal_jit_compile:.2f}s")
log(f"  Per start (sequential):     {internal_jit_time/10:.2f}s")
log(f"  20 starts would take:       {internal_jit_compile + 19*internal_jit_time/10:.1f}s")
log()
log(f"VMAP (parallel - RECOMMENDED):")
log(f"  First batch (compile):      {vmap_first:.2f}s")
log(f"  20 starts in parallel:      {vmap_second:.2f}s")
log(f"  Effective per-start:        {vmap_second/n_starts_batch:.3f}s")
log()

# Calculate speedup
sequential_time = internal_jit_compile + 19 * internal_jit_time/10
vmap_time = vmap_first  # First batch includes compile, use that for comparison
speedup = sequential_time / vmap_second if vmap_second > 0 else float('inf')
log(f"VMAP SPEEDUP: {speedup:.1f}x faster than sequential!")
log()

# Expected performance for full fitting with vmap
log("EXTRAPOLATION (20 starts per participant, 45 participants) WITH VMAP:")
per_participant_vmap = vmap_first  # First includes compile
subsequent_vmap = vmap_second
total_time_vmap = per_participant_vmap + 44 * subsequent_vmap
log(f"  First participant:  {per_participant_vmap:.1f}s (includes compile)")
log(f"  Each subsequent:    {subsequent_vmap:.1f}s (20 parallel starts)")
log(f"  Total estimated:    {total_time_vmap:.0f}s = {total_time_vmap/60:.1f} min")
log("=" * 55)
