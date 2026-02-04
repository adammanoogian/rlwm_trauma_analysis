#!/usr/bin/env python3
"""
GPU Performance Diagnostic Script

Run this to identify where the bottleneck is in MLE fitting.
"""
import sys
import time
sys.path.insert(0, '.')

import jax
import jax.numpy as jnp
from jaxopt import LBFGS
import numpy as np

# Force GPU
print("=" * 60)
print("GPU PERFORMANCE DIAGNOSTIC")
print("=" * 60)

devices = jax.devices()
print(f"JAX devices: {devices}")
gpu_devices = [d for d in devices if d.platform == "gpu"]
print(f"GPU available: {len(gpu_devices) > 0}")
print()

# Import our likelihood functions
from jax_likelihoods import (
    q_learning_multiblock_likelihood,
    q_learning_multiblock_likelihood_stacked,
    q_learning_block_likelihood,
)
from mle_utils import jax_unconstrained_to_params_qlearning

# Create synthetic data matching real data structure
print("Creating synthetic data (17 blocks x 100 trials)...")
np.random.seed(42)
n_blocks = 17
n_trials = 100

stimuli_blocks = [jnp.array(np.random.randint(0, 6, n_trials), dtype=jnp.int32) for _ in range(n_blocks)]
actions_blocks = [jnp.array(np.random.randint(0, 3, n_trials), dtype=jnp.int32) for _ in range(n_blocks)]
rewards_blocks = [jnp.array(np.random.binomial(1, 0.7, n_trials), dtype=jnp.float32) for _ in range(n_blocks)]
masks_blocks = [jnp.ones(n_trials, dtype=jnp.float32) for _ in range(n_blocks)]

# Pre-stack for objective
stimuli_stacked = jnp.stack(stimuli_blocks)
actions_stacked = jnp.stack(actions_blocks)
rewards_stacked = jnp.stack(rewards_blocks)
masks_stacked = jnp.stack(masks_blocks)

print(f"Data shapes: stimuli={stimuli_stacked.shape}, masks={masks_stacked.shape}")
print()

# =============================================================================
# TEST 1: Raw likelihood function speed (STACKED version - optimized)
# =============================================================================
print("=" * 60)
print("TEST 1: Single likelihood evaluation (stacked version)")
print("=" * 60)

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
print("First call (includes JIT compilation)...")
start = time.time()
result = objective_jit(x0)
jax.block_until_ready(result)
compile_time = time.time() - start
print(f"  Time: {compile_time:.3f}s")

# Subsequent calls (should be fast)
print("Next 100 calls (compiled)...")
start = time.time()
for _ in range(100):
    result = objective_jit(x0)
    jax.block_until_ready(result)
eval_time = time.time() - start
print(f"  Total: {eval_time:.3f}s ({eval_time/100*1000:.2f}ms per call)")
print()

# =============================================================================
# TEST 2: LBFGS solver with jit=True (jaxopt internal JIT - RECOMMENDED)
# =============================================================================
print("=" * 60)
print("TEST 2: LBFGS solver with jit=True (jaxopt handles JIT internally)")
print("=" * 60)

solver = LBFGS(fun=objective_jit, maxiter=100, tol=1e-6, jit=True)

print("First optimization (includes internal JIT compilation)...")
start = time.time()
x0 = jnp.array([0.0, 0.0, -2.0])
params, state = solver.run(x0)
jax.block_until_ready(params)
internal_jit_compile = time.time() - start
print(f"  Time: {internal_jit_compile:.2f}s")

print("Next 10 optimizations (should reuse cached compilation)...")
start = time.time()
for i in range(10):
    x0 = jnp.array([np.random.randn(), np.random.randn(), -2.0])
    params, state = solver.run(x0)
    jax.block_until_ready(params)
internal_jit_time = time.time() - start
print(f"  Total: {internal_jit_time:.2f}s ({internal_jit_time/10:.2f}s per start)")
print()

# =============================================================================
# TEST 3: LBFGS solver with jax.jit(solver.run) wrapper
# NOTE: This often fails with TracerBoolConversionError - that's expected!
# =============================================================================
print("=" * 60)
print("TEST 3: LBFGS solver WITH jax.jit(solver.run) wrapper")
print("=" * 60)
print("(This approach usually fails - testing to confirm)")

try:
    solver2 = LBFGS(fun=objective_jit, maxiter=100, tol=1e-6, jit=False)
    jit_run = jax.jit(solver2.run)

    start = time.time()
    x0 = jnp.array([0.0, 0.0, -2.0])
    params, state = jit_run(x0)
    jax.block_until_ready(params)
    solver_compile_time = time.time() - start
    print(f"  Surprisingly worked! Time: {solver_compile_time:.2f}s")
    jit_time = 0.0  # Placeholder
except Exception as e:
    print(f"  FAILED (as expected): {type(e).__name__}")
    print(f"  This confirms jax.jit(solver.run) doesn't work with jaxopt")
    solver_compile_time = float('inf')
    jit_time = float('inf')
print()

# =============================================================================
# TEST 4: VECTORIZED optimization with vmap (NEW - should be fastest!)
# =============================================================================
print("=" * 60)
print("TEST 4: VECTORIZED optimization with jax.vmap(solver.run)")
print("=" * 60)

solver3 = LBFGS(fun=objective_jit, maxiter=100, tol=1e-6, jit=True)
vmap_run = jax.vmap(solver3.run)

# Generate batch of starting points
n_starts_batch = 20
x0_batch = jnp.array([[np.random.randn(), np.random.randn(), -2.0] for _ in range(n_starts_batch)])

print(f"Running {n_starts_batch} optimizations IN PARALLEL with vmap...")
print("First batch (includes vmap compilation)...")
start = time.time()
all_params, all_states = vmap_run(x0_batch)
jax.block_until_ready(all_params)
vmap_first = time.time() - start
print(f"  Time for {n_starts_batch} parallel starts: {vmap_first:.2f}s")

print("Second batch (compiled)...")
x0_batch2 = jnp.array([[np.random.randn(), np.random.randn(), -2.0] for _ in range(n_starts_batch)])
start = time.time()
all_params2, all_states2 = vmap_run(x0_batch2)
jax.block_until_ready(all_params2)
vmap_second = time.time() - start
print(f"  Time for {n_starts_batch} parallel starts: {vmap_second:.2f}s")
print(f"  Effective per-start: {vmap_second/n_starts_batch:.3f}s")
print()

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Likelihood compile time:      {compile_time:.2f}s")
print(f"Likelihood per-call:          {eval_time/100*1000:.2f}ms")
print()
print(f"Sequential (jit=True):")
print(f"  First call (compile):       {internal_jit_compile:.2f}s")
print(f"  Per start (sequential):     {internal_jit_time/10:.2f}s")
print(f"  20 starts would take:       {internal_jit_compile + 19*internal_jit_time/10:.1f}s")
print()
print(f"VMAP (parallel - RECOMMENDED):")
print(f"  First batch (compile):      {vmap_first:.2f}s")
print(f"  20 starts in parallel:      {vmap_second:.2f}s")
print(f"  Effective per-start:        {vmap_second/n_starts_batch:.3f}s")
print()

# Calculate speedup
sequential_time = internal_jit_compile + 19 * internal_jit_time/10
vmap_time = vmap_first  # First batch includes compile, use that for comparison
speedup = sequential_time / vmap_second if vmap_second > 0 else float('inf')
print(f"VMAP SPEEDUP: {speedup:.1f}x faster than sequential!")
print()

# Expected performance for full fitting with vmap
print("EXTRAPOLATION (20 starts per participant, 45 participants) WITH VMAP:")
per_participant_vmap = vmap_first  # First includes compile
subsequent_vmap = vmap_second
total_time_vmap = per_participant_vmap + 44 * subsequent_vmap
print(f"  First participant:  {per_participant_vmap:.1f}s (includes compile)")
print(f"  Each subsequent:    {subsequent_vmap:.1f}s (20 parallel starts)")
print(f"  Total estimated:    {total_time_vmap:.0f}s = {total_time_vmap/60:.1f} min")
print("=" * 60)
