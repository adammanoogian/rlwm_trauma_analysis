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
# TEST 2: LBFGS solver without JIT
# =============================================================================
print("=" * 60)
print("TEST 2: LBFGS solver WITHOUT jax.jit(solver.run)")
print("=" * 60)

solver = LBFGS(fun=objective_jit, maxiter=100, tol=1e-6)

print("Running 3 optimization starts (no JIT on solver)...")
start = time.time()
for i in range(3):
    x0 = jnp.array([np.random.randn(), np.random.randn(), -2.0])
    params, state = solver.run(x0)
no_jit_time = time.time() - start
print(f"  Total: {no_jit_time:.2f}s ({no_jit_time/3:.2f}s per start)")
print()

# =============================================================================
# TEST 3: LBFGS solver WITH JIT
# =============================================================================
print("=" * 60)
print("TEST 3: LBFGS solver WITH jax.jit(solver.run)")
print("=" * 60)

jit_run = jax.jit(solver.run)

print("First optimization (includes JIT compilation of solver)...")
start = time.time()
x0 = jnp.array([0.0, 0.0, -2.0])
params, state = jit_run(x0)
jax.block_until_ready(params)
solver_compile_time = time.time() - start
print(f"  Time: {solver_compile_time:.2f}s")

print("Next 10 optimizations (compiled)...")
start = time.time()
for i in range(10):
    x0 = jnp.array([np.random.randn(), np.random.randn(), -2.0])
    params, state = jit_run(x0)
    jax.block_until_ready(params)
jit_time = time.time() - start
print(f"  Total: {jit_time:.2f}s ({jit_time/10:.2f}s per start)")
print()

# =============================================================================
# SUMMARY
# =============================================================================
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Likelihood compile time:  {compile_time:.2f}s")
print(f"Likelihood per-call:      {eval_time/100*1000:.2f}ms")
print(f"Solver WITHOUT JIT:       {no_jit_time/3:.2f}s per start")
print(f"Solver compile time:      {solver_compile_time:.2f}s")
print(f"Solver WITH JIT:          {jit_time/10:.2f}s per start")
print()

speedup = (no_jit_time/3) / (jit_time/10) if jit_time > 0 else float('inf')
print(f"JIT solver speedup: {speedup:.1f}x")
print()

# Expected performance for full fitting
print("EXTRAPOLATION (20 starts per participant, 45 participants):")
per_participant_jit = solver_compile_time + 19 * (jit_time/10)
print(f"  First participant:  {solver_compile_time:.1f}s (compile) + {19*(jit_time/10):.1f}s = {per_participant_jit:.1f}s")
print(f"  Each subsequent:    {20 * (jit_time/10):.1f}s")
print(f"  Total estimated:    {per_participant_jit + 44 * 20 * (jit_time/10):.0f}s = {(per_participant_jit + 44 * 20 * (jit_time/10))/60:.1f} min")
print("=" * 60)
