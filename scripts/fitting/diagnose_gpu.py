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
from jax_likelihoods import q_learning_multiblock_likelihood, q_learning_block_likelihood
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
# TEST 1: Raw likelihood function speed
# =============================================================================
print("=" * 60)
print("TEST 1: Single likelihood evaluation")
print("=" * 60)

def objective_raw(x):
    alpha_pos, alpha_neg, epsilon = jax_unconstrained_to_params_qlearning(x)
    return -q_learning_multiblock_likelihood(
        stimuli_blocks=list(stimuli_stacked),
        actions_blocks=list(actions_stacked),
        rewards_blocks=list(rewards_stacked),
        alpha_pos=alpha_pos,
        alpha_neg=alpha_neg,
        epsilon=epsilon,
        masks_blocks=list(masks_stacked)
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
# TEST 3: LBFGS solver with jax.jit(solver.run) wrapper (may cause long compile)
# =============================================================================
print("=" * 60)
print("TEST 3: LBFGS solver WITH jax.jit(solver.run) wrapper")
print("=" * 60)

solver2 = LBFGS(fun=objective_jit, maxiter=100, tol=1e-6, jit=False)  # Disable internal JIT
jit_run = jax.jit(solver2.run)

print("First optimization (full solver compilation - may be slow!)...")
start = time.time()
x0 = jnp.array([0.0, 0.0, -2.0])
params, state = jit_run(x0)
jax.block_until_ready(params)
solver_compile_time = time.time() - start
print(f"  Time: {solver_compile_time:.2f}s")

print("Next 5 optimizations (compiled)...")
start = time.time()
for i in range(5):
    x0 = jnp.array([np.random.randn(), np.random.randn(), -2.0])
    params, state = jit_run(x0)
    jax.block_until_ready(params)
jit_time = time.time() - start
print(f"  Total: {jit_time:.2f}s ({jit_time/5:.2f}s per start)")
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
print(f"jaxopt jit=True (recommended):")
print(f"  First call (compile):       {internal_jit_compile:.2f}s")
print(f"  Subsequent (per start):     {internal_jit_time/10:.2f}s")
print()
print(f"jax.jit(solver.run) wrapper:")
print(f"  First call (compile):       {solver_compile_time:.2f}s")
print(f"  Subsequent (per start):     {jit_time/5:.2f}s")
print()

# Determine which is better
if internal_jit_compile < solver_compile_time:
    print("RECOMMENDATION: Use jit=True (jaxopt internal) - faster compilation")
    best_compile = internal_jit_compile
    best_per_start = internal_jit_time/10
else:
    print("RECOMMENDATION: Use jax.jit(solver.run) - faster per-start")
    best_compile = solver_compile_time
    best_per_start = jit_time/5
print()

# Expected performance for full fitting
print("EXTRAPOLATION (20 starts per participant, 45 participants):")
per_participant = best_compile + 19 * best_per_start
total_time = per_participant + 44 * 20 * best_per_start
print(f"  First participant:  {best_compile:.1f}s (compile) + {19*best_per_start:.1f}s = {per_participant:.1f}s")
print(f"  Each subsequent:    {20 * best_per_start:.1f}s")
print(f"  Total estimated:    {total_time:.0f}s = {total_time/60:.1f} min")
print("=" * 60)
