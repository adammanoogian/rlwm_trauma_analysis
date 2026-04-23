#!/usr/bin/env python3
"""
JIT Warmup Script for MLE Fitting

Triggers JAX JIT compilation with minimal memory overhead.
Run this with limited CPUs (e.g., srun --cpus-per-task=2) to prevent
LLVM from spawning too many compilation threads.

The compiled kernels are saved to JAX_COMPILATION_CACHE_DIR and
reused by subsequent fitting runs.

Why this works:
--------------
LLVM's execution engine spawns threads equal to std::thread::hardware_concurrency().
By running the warmup phase with only 2 CPUs visible (via srun --cpus-per-task=2),
LLVM sees 2 CPUs and spawns only 2 compilation threads, preventing memory exhaustion.

The main fitting phase then runs with full CPU allocation, but since the kernels
are already cached, no LLVM compilation occurs and memory stays stable.

Usage:
------
# Standalone test
python scripts/fitting/warmup_jit.py --model all

# In SLURM (two-phase approach)
srun --cpus-per-task=2 --exact python scripts/fitting/warmup_jit.py --model $MODEL
"""

from __future__ import annotations

import os
import sys
import time

# Add project root to path for imports
sys.path.insert(0, '.')


def warmup_jit_compilation(model: str = 'all', verbose: bool = True):
    """
    Trigger JIT compilation for specified model(s).

    This function creates synthetic data with shapes matching real experimental
    data, then calls each likelihood function once to trigger JIT compilation.
    The compiled XLA kernels are automatically cached to disk by JAX.

    Parameters
    ----------
    model : str
        Which model(s) to warmup: 'all', or a specific model name
    verbose : bool
        Whether to print progress messages
    """
    import jax
    import jax.numpy as jnp

    # Import likelihood functions and constants
    from rlwm.fitting.core import (
        MAX_BLOCKS,
        MAX_TRIALS_PER_BLOCK,
        NUM_ACTIONS,
    )
    from rlwm.fitting.models.qlearning import q_learning_multiblock_likelihood
    from rlwm.fitting.models.wmrl import wmrl_multiblock_likelihood
    from rlwm.fitting.models.wmrl_m3 import wmrl_m3_multiblock_likelihood
    from rlwm.fitting.models.wmrl_m5 import wmrl_m5_multiblock_likelihood
    from rlwm.fitting.models.wmrl_m6a import wmrl_m6a_multiblock_likelihood
    from rlwm.fitting.models.wmrl_m6b import wmrl_m6b_multiblock_likelihood

    if verbose:
        print("=" * 60)
        print("JAX JIT Warmup for MLE Fitting")
        print("=" * 60)
        print(f"JAX devices: {jax.devices()}")
        cache_dir = os.environ.get('JAX_COMPILATION_CACHE_DIR', 'Not set (using default)')
        print(f"JAX compilation cache: {cache_dir}")
        print(f"Data shape: {MAX_BLOCKS} blocks x {MAX_TRIALS_PER_BLOCK} trials")
        print()

    # Create synthetic data matching real data shapes
    # Using lists of arrays (one per block) as required by multiblock functions
    n_blocks = MAX_BLOCKS
    n_trials = MAX_TRIALS_PER_BLOCK
    n_actions = NUM_ACTIONS
    num_stimuli = 6  # Max stimuli in task

    # Create synthetic trial data for all blocks
    stimuli_blocks = [jnp.zeros(n_trials, dtype=jnp.int32) for _ in range(n_blocks)]
    actions_blocks = [jnp.zeros(n_trials, dtype=jnp.int32) for _ in range(n_blocks)]
    rewards_blocks = [jnp.zeros(n_trials, dtype=jnp.float32) for _ in range(n_blocks)]
    set_sizes_blocks = [jnp.full(n_trials, 3, dtype=jnp.int32) for _ in range(n_blocks)]
    masks_blocks = [jnp.ones(n_trials, dtype=jnp.float32) for _ in range(n_blocks)]

    ALL_CHOICE_MODELS = ['qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b']
    models_to_warmup = ALL_CHOICE_MODELS if model == 'all' else [model]

    for model_name in models_to_warmup:
        if verbose:
            print(f"Warming up {model_name}...", end=' ', flush=True)

        start = time.time()

        if model_name == 'qlearning':
            # Q-learning: alpha_pos, alpha_neg, epsilon
            _ = q_learning_multiblock_likelihood(
                stimuli_blocks=stimuli_blocks,
                actions_blocks=actions_blocks,
                rewards_blocks=rewards_blocks,
                alpha_pos=0.3,
                alpha_neg=0.3,
                epsilon=0.1,
                num_stimuli=num_stimuli,
                num_actions=n_actions,
                masks_blocks=masks_blocks,
            )

        elif model_name == 'wmrl':
            # WM-RL: alpha_pos, alpha_neg, phi, rho, K, epsilon
            _ = wmrl_multiblock_likelihood(
                stimuli_blocks=stimuli_blocks,
                actions_blocks=actions_blocks,
                rewards_blocks=rewards_blocks,
                set_sizes_blocks=set_sizes_blocks,
                alpha_pos=0.3,
                alpha_neg=0.3,
                phi=0.8,
                rho=0.5,
                capacity=3.0,
                epsilon=0.1,
                num_stimuli=num_stimuli,
                num_actions=n_actions,
                masks_blocks=masks_blocks,
            )

        elif model_name == 'wmrl_m3':
            # WM-RL M3: alpha_pos, alpha_neg, phi, rho, K, kappa, epsilon
            _ = wmrl_m3_multiblock_likelihood(
                stimuli_blocks=stimuli_blocks,
                actions_blocks=actions_blocks,
                rewards_blocks=rewards_blocks,
                set_sizes_blocks=set_sizes_blocks,
                alpha_pos=0.3,
                alpha_neg=0.3,
                phi=0.8,
                rho=0.5,
                capacity=3.0,
                kappa=0.1,
                epsilon=0.1,
                num_stimuli=num_stimuli,
                num_actions=n_actions,
                masks_blocks=masks_blocks,
            )

        elif model_name == 'wmrl_m5':
            # WM-RL M5: M3 + phi_rl (RL forgetting)
            _ = wmrl_m5_multiblock_likelihood(
                stimuli_blocks=stimuli_blocks,
                actions_blocks=actions_blocks,
                rewards_blocks=rewards_blocks,
                set_sizes_blocks=set_sizes_blocks,
                alpha_pos=0.3,
                alpha_neg=0.3,
                phi=0.8,
                rho=0.5,
                capacity=3.0,
                kappa=0.1,
                phi_rl=0.05,
                epsilon=0.1,
                num_stimuli=num_stimuli,
                num_actions=n_actions,
                masks_blocks=masks_blocks,
            )

        elif model_name == 'wmrl_m6a':
            # WM-RL M6a: per-stimulus perseveration (kappa_s)
            _ = wmrl_m6a_multiblock_likelihood(
                stimuli_blocks=stimuli_blocks,
                actions_blocks=actions_blocks,
                rewards_blocks=rewards_blocks,
                set_sizes_blocks=set_sizes_blocks,
                alpha_pos=0.3,
                alpha_neg=0.3,
                phi=0.8,
                rho=0.5,
                capacity=3.0,
                kappa_s=0.1,
                epsilon=0.1,
                num_stimuli=num_stimuli,
                num_actions=n_actions,
                masks_blocks=masks_blocks,
            )

        elif model_name == 'wmrl_m6b':
            # WM-RL M6b: dual perseveration (decoded kappa + kappa_s)
            kappa_total = 0.3
            kappa_share = 0.5
            _ = wmrl_m6b_multiblock_likelihood(
                stimuli_blocks=stimuli_blocks,
                actions_blocks=actions_blocks,
                rewards_blocks=rewards_blocks,
                set_sizes_blocks=set_sizes_blocks,
                alpha_pos=0.3,
                alpha_neg=0.3,
                phi=0.8,
                rho=0.5,
                capacity=3.0,
                kappa=kappa_total * kappa_share,
                kappa_s=kappa_total * (1 - kappa_share),
                epsilon=0.1,
                num_stimuli=num_stimuli,
                num_actions=n_actions,
                masks_blocks=masks_blocks,
            )

        elif model_name == 'wmrl_m4':
            # M4 requires float64 + separate likelihood module
            jax.config.update("jax_enable_x64", True)
            from scripts.fitting.lba_likelihood import wmrl_m4_multiblock_likelihood
            rts_blocks = [jnp.full(n_trials, 0.5, dtype=jnp.float64) for _ in range(n_blocks)]
            _ = wmrl_m4_multiblock_likelihood(
                stimuli_blocks=stimuli_blocks,
                actions_blocks=actions_blocks,
                rewards_blocks=rewards_blocks,
                set_sizes_blocks=set_sizes_blocks,
                rts_blocks=rts_blocks,
                masks_blocks=masks_blocks,
                alpha_pos=0.3,
                alpha_neg=0.3,
                phi=0.8,
                rho=0.5,
                capacity=3.0,
                kappa=0.1,
                v_scale=5.0,
                A=0.2,
                b=0.3,
                t0=0.15,
                num_stimuli=num_stimuli,
                num_actions=n_actions,
            )

        elapsed = time.time() - start
        if verbose:
            print(f"done ({elapsed:.1f}s)")

    if verbose:
        print()
        print("=" * 60)
        print("JIT warmup complete. Compiled kernels cached to disk.")
        print("=" * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='JIT warmup for MLE fitting - triggers JAX compilation with limited CPUs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Warmup all models (default)
  python scripts/fitting/warmup_jit.py

  # Warmup specific model
  python scripts/fitting/warmup_jit.py --model wmrl_m3

  # In SLURM with limited CPUs (prevents LLVM memory exhaustion)
  srun --cpus-per-task=2 --exact python scripts/fitting/warmup_jit.py --model $MODEL
        """
    )
    parser.add_argument(
        '--model',
        default='all',
        choices=['all', 'qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b', 'wmrl_m4'],
        help='Model(s) to warmup (default: all choice-only models; M4 must be specified explicitly)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()
    warmup_jit_compilation(model=args.model, verbose=not args.quiet)
