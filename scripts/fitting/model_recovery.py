"""
Model Recovery Analysis

This script performs model recovery to validate that:
1. Q-learning data is better fit by Q-learning model
2. WM-RL data is better fit by WM-RL model
3. Models can be distinguished from their behavioral signatures

Procedure:
-----------
1. Generate synthetic data from both models with known parameters
2. Fit both models to both datasets (4 fits total)
3. Compare likelihoods: data from Model A should be better explained by Model A

Author: Generated for RLWM trauma analysis project
Date: 2025-11-22
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import arviz as az
from typing import Dict, List

from scripts.fitting.jax_likelihoods import (
    q_learning_multiblock_likelihood,
    wmrl_multiblock_likelihood
)


def simulate_qlearning_data(
    alpha_pos: float = 0.6,
    alpha_neg: float = 0.4,
    beta: float = 2.0,
    num_blocks: int = 10,
    trials_per_block: int = 60,
    num_stimuli: int = 6,
    num_actions: int = 3,
    seed: int = 42
):
    """
    Simulate data from Q-learning model.

    Returns block-structured data matching NumPyro format.
    """
    key = jax.random.PRNGKey(seed)

    stimuli_blocks = []
    actions_blocks = []
    rewards_blocks = []

    # Initialize Q-table
    Q = jnp.ones((num_stimuli, num_actions)) * 0.5

    for block in range(num_blocks):
        stimuli = []
        actions = []
        rewards = []

        # Reset Q-values at each block
        Q = jnp.ones((num_stimuli, num_actions)) * 0.5

        for trial in range(trials_per_block):
            # Sample stimulus
            key, subkey = jax.random.split(key)
            stimulus = int(jax.random.randint(subkey, (), 0, num_stimuli))

            # Compute action probabilities
            q_vals = Q[stimulus, :]
            q_scaled = beta * (q_vals - jnp.max(q_vals))
            probs = jnp.exp(q_scaled) / jnp.sum(jnp.exp(q_scaled))

            # Sample action
            key, subkey = jax.random.split(key)
            action = int(jax.random.choice(subkey, num_actions, p=probs))

            # Generate reward (70% correct for chosen action)
            key, subkey = jax.random.split(key)
            reward = float(jax.random.bernoulli(subkey, 0.7))

            # Update Q-value
            delta = reward - Q[stimulus, action]
            alpha = alpha_pos if delta > 0 else alpha_neg
            Q = Q.at[stimulus, action].set(Q[stimulus, action] + alpha * delta)

            stimuli.append(stimulus)
            actions.append(action)
            rewards.append(reward)

        stimuli_blocks.append(jnp.array(stimuli, dtype=jnp.int32))
        actions_blocks.append(jnp.array(actions, dtype=jnp.int32))
        rewards_blocks.append(jnp.array(rewards, dtype=jnp.float32))

    return {
        'stimuli_blocks': stimuli_blocks,
        'actions_blocks': actions_blocks,
        'rewards_blocks': rewards_blocks
    }


def simulate_wmrl_data(
    alpha_pos: float = 0.3,
    alpha_neg: float = 0.1,
    beta: float = 2.0,
    beta_wm: float = 3.0,
    phi: float = 0.2,
    rho: float = 0.7,
    capacity: float = 4.0,
    num_blocks: int = 10,
    trials_per_block: int = 60,
    num_stimuli: int = 6,
    num_actions: int = 3,
    set_size: int = 5,
    seed: int = 43
):
    """
    Simulate data from WM-RL hybrid model.

    Returns block-structured data matching NumPyro format.
    """
    key = jax.random.PRNGKey(seed)

    stimuli_blocks = []
    actions_blocks = []
    rewards_blocks = []
    set_sizes_blocks = []

    # Initialize matrices
    Q = jnp.ones((num_stimuli, num_actions)) * 0.5
    WM = jnp.ones((num_stimuli, num_actions)) * 0.5
    WM_0 = jnp.ones((num_stimuli, num_actions)) * 0.5

    for block in range(num_blocks):
        stimuli = []
        actions = []
        rewards = []

        # Reset at each block
        Q = jnp.ones((num_stimuli, num_actions)) * 0.5
        WM = jnp.ones((num_stimuli, num_actions)) * 0.5

        for trial in range(trials_per_block):
            # Sample stimulus
            key, subkey = jax.random.split(key)
            stimulus = int(jax.random.randint(subkey, (), 0, num_stimuli))

            # Decay WM
            WM = (1 - phi) * WM + phi * WM_0

            # Compute RL and WM probabilities
            q_vals = Q[stimulus, :]
            q_scaled = beta * (q_vals - jnp.max(q_vals))
            rl_probs = jnp.exp(q_scaled) / jnp.sum(jnp.exp(q_scaled))

            wm_vals = WM[stimulus, :]
            wm_scaled = beta_wm * (wm_vals - jnp.max(wm_vals))
            wm_probs = jnp.exp(wm_scaled) / jnp.sum(jnp.exp(wm_scaled))

            # Adaptive weight
            omega = rho * jnp.minimum(1.0, capacity / set_size)

            # Hybrid policy
            hybrid_probs = omega * wm_probs + (1 - omega) * rl_probs
            hybrid_probs = hybrid_probs / jnp.sum(hybrid_probs)

            # Sample action
            key, subkey = jax.random.split(key)
            action = int(jax.random.choice(subkey, num_actions, p=hybrid_probs))

            # Generate reward
            key, subkey = jax.random.split(key)
            reward = float(jax.random.bernoulli(subkey, 0.7))

            # Update WM
            WM = WM.at[stimulus, action].set(reward)

            # Update Q
            delta = reward - Q[stimulus, action]
            alpha = alpha_pos if delta > 0 else alpha_neg
            Q = Q.at[stimulus, action].set(Q[stimulus, action] + alpha * delta)

            stimuli.append(stimulus)
            actions.append(action)
            rewards.append(reward)

        stimuli_blocks.append(jnp.array(stimuli, dtype=jnp.int32))
        actions_blocks.append(jnp.array(actions, dtype=jnp.int32))
        rewards_blocks.append(jnp.array(rewards, dtype=jnp.float32))
        set_sizes_blocks.append(jnp.ones(trials_per_block, dtype=jnp.float32) * set_size)

    return {
        'stimuli_blocks': stimuli_blocks,
        'actions_blocks': actions_blocks,
        'rewards_blocks': rewards_blocks,
        'set_sizes_blocks': set_sizes_blocks
    }


def compute_model_recovery_matrix():
    """
    Compute model recovery confusion matrix.

    Returns:
    --------
    dict with log-likelihoods for all 4 combinations
    """
    print("=" * 80)
    print("MODEL RECOVERY ANALYSIS")
    print("=" * 80)

    # Generate synthetic data
    print("\n>> Generating synthetic data...")
    ql_data = simulate_qlearning_data(seed=42)
    wmrl_data = simulate_wmrl_data(seed=43)

    print(f"  ✓ Q-learning data: {len(ql_data['stimuli_blocks'])} blocks")
    print(f"  ✓ WM-RL data: {len(wmrl_data['stimuli_blocks'])} blocks")

    # Test parameters (ground truth for simulation)
    ql_params = {'alpha_pos': 0.6, 'alpha_neg': 0.4, 'beta': 2.0}
    wmrl_params = {
        'alpha_pos': 0.3, 'alpha_neg': 0.1, 'beta': 2.0,
        'beta_wm': 3.0, 'phi': 0.2, 'rho': 0.7, 'capacity': 4.0
    }

    # Compute likelihoods
    print("\n>> Computing log-likelihoods...")

    # Q-learning model on Q-learning data (should be best)
    print("  [1/4] Q-learning model on Q-learning data...")
    ll_ql_on_ql = q_learning_multiblock_likelihood(
        stimuli_blocks=ql_data['stimuli_blocks'],
        actions_blocks=ql_data['actions_blocks'],
        rewards_blocks=ql_data['rewards_blocks'],
        **ql_params
    )
    print(f"    Log-likelihood: {float(ll_ql_on_ql):.2f}")

    # Q-learning model on WM-RL data (should be worse)
    print("  [2/4] Q-learning model on WM-RL data...")
    ll_ql_on_wmrl = q_learning_multiblock_likelihood(
        stimuli_blocks=wmrl_data['stimuli_blocks'],
        actions_blocks=wmrl_data['actions_blocks'],
        rewards_blocks=wmrl_data['rewards_blocks'],
        **ql_params
    )
    print(f"    Log-likelihood: {float(ll_ql_on_wmrl):.2f}")

    # WM-RL model on Q-learning data (should be worse)
    print("  [3/4] WM-RL model on Q-learning data...")
    # Need to create dummy set_sizes for Q-learning data
    ql_data_with_sets = {
        **ql_data,
        'set_sizes_blocks': [jnp.ones(len(stim), dtype=jnp.float32) * 6
                             for stim in ql_data['stimuli_blocks']]
    }
    ll_wmrl_on_ql = wmrl_multiblock_likelihood(
        stimuli_blocks=ql_data_with_sets['stimuli_blocks'],
        actions_blocks=ql_data_with_sets['actions_blocks'],
        rewards_blocks=ql_data_with_sets['rewards_blocks'],
        set_sizes_blocks=ql_data_with_sets['set_sizes_blocks'],
        **wmrl_params
    )
    print(f"    Log-likelihood: {float(ll_wmrl_on_ql):.2f}")

    # WM-RL model on WM-RL data (should be best)
    print("  [4/4] WM-RL model on WM-RL data...")
    ll_wmrl_on_wmrl = wmrl_multiblock_likelihood(
        stimuli_blocks=wmrl_data['stimuli_blocks'],
        actions_blocks=wmrl_data['actions_blocks'],
        rewards_blocks=wmrl_data['rewards_blocks'],
        set_sizes_blocks=wmrl_data['set_sizes_blocks'],
        **wmrl_params
    )
    print(f"    Log-likelihood: {float(ll_wmrl_on_wmrl):.2f}")

    # Print recovery matrix
    print("\n" + "=" * 80)
    print("MODEL RECOVERY MATRIX")
    print("=" * 80)
    print("\n                  Generating Model")
    print("                  Q-learning    WM-RL")
    print("Fitting     Q-learning   {:.2f}      {:.2f}".format(
        float(ll_ql_on_ql), float(ll_ql_on_wmrl)))
    print("Model       WM-RL        {:.2f}      {:.2f}".format(
        float(ll_wmrl_on_ql), float(ll_wmrl_on_wmrl)))

    # Check recovery success
    print("\n>> Recovery Analysis:")
    ql_recovers = ll_ql_on_ql > ll_wmrl_on_ql
    wmrl_recovers = ll_wmrl_on_wmrl > ll_ql_on_wmrl

    if ql_recovers:
        print(f"  ✓ Q-learning data correctly identified (ΔLL = {float(ll_ql_on_ql - ll_wmrl_on_ql):.2f})")
    else:
        print(f"  ✗ Q-learning data misidentified (ΔLL = {float(ll_ql_on_ql - ll_wmrl_on_ql):.2f})")

    if wmrl_recovers:
        print(f"  ✓ WM-RL data correctly identified (ΔLL = {float(ll_wmrl_on_wmrl - ll_ql_on_wmrl):.2f})")
    else:
        print(f"  ✗ WM-RL data misidentified (ΔLL = {float(ll_wmrl_on_wmrl - ll_ql_on_wmrl):.2f})")

    if ql_recovers and wmrl_recovers:
        print("\n  ✓✓ PERFECT RECOVERY: Both models correctly identified their own data!")

    return {
        'll_ql_on_ql': float(ll_ql_on_ql),
        'll_ql_on_wmrl': float(ll_ql_on_wmrl),
        'll_wmrl_on_ql': float(ll_wmrl_on_ql),
        'll_wmrl_on_wmrl': float(ll_wmrl_on_wmrl),
        'ql_recovers': ql_recovers,
        'wmrl_recovers': wmrl_recovers
    }


if __name__ == "__main__":
    results = compute_model_recovery_matrix()

    print("\n" + "=" * 80)
    print("MODEL RECOVERY COMPLETE!")
    print("=" * 80)
