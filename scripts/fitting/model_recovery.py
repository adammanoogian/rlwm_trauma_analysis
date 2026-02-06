"""
Parameter Recovery for RLWM Models

This module validates MLE fitting quality by testing whether the fitting procedure
can accurately recover known parameter values from synthetic data.

Procedure (Senta et al., 2025):
1. Sample true parameters uniformly from MLE bounds
2. Generate synthetic trial-level data matching task structure
3. Fit synthetic data via MLE (same procedure as real data)
4. Compare recovered vs true parameters
5. Compute recovery metrics: Pearson r, RMSE, bias
6. Pass/fail criterion: r >= 0.80 per parameter

Author: Generated for RLWM trauma analysis project
Date: 2026-02-06
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.stats import pearsonr
from tqdm import tqdm
from typing import Dict, List, Optional

from scripts.fitting.fit_mle import fit_participant_mle, prepare_participant_data
from scripts.fitting.mle_utils import (
    QLEARNING_BOUNDS, WMRL_BOUNDS, WMRL_M3_BOUNDS,
    QLEARNING_PARAMS, WMRL_PARAMS, WMRL_M3_PARAMS
)


# =============================================================================
# Constants (matching real data structure)
# =============================================================================

NUM_BLOCKS = 21  # Blocks 3-23 in real data
BLOCK_OFFSET = 3  # First block is numbered 3
NUM_STIMULI = 3  # Maximum 3 stimuli per block
NUM_ACTIONS = 3  # Always 3 actions
SET_SIZES = [2, 3, 5, 6]  # Cycle through these
FIXED_BETA = 50.0  # Fixed inverse temperature (matching fit_mle.py)

# Trials per block range (matching real data variability)
MIN_TRIALS_PER_BLOCK = 30
MAX_TRIALS_PER_BLOCK = 90

# Reversal trigger (12-18 consecutive correct responses)
MIN_CORRECT_FOR_REVERSAL = 12
MAX_CORRECT_FOR_REVERSAL = 18


# =============================================================================
# Parameter Sampling
# =============================================================================

def sample_parameters(model: str, n_subjects: int, seed: int) -> List[Dict[str, float]]:
    """
    Sample parameter sets uniformly from MLE bounds.

    Parameters
    ----------
    model : str
        Model name ('qlearning', 'wmrl', 'wmrl_m3')
    n_subjects : int
        Number of parameter sets to sample
    seed : int
        Random seed for reproducibility

    Returns
    -------
    List[Dict[str, float]]
        List of parameter dictionaries sampled from bounds
    """
    rng = np.random.default_rng(seed)

    # Get bounds for model
    if model == 'qlearning':
        bounds = QLEARNING_BOUNDS
        param_names = QLEARNING_PARAMS
    elif model == 'wmrl':
        bounds = WMRL_BOUNDS
        param_names = WMRL_PARAMS
    elif model == 'wmrl_m3':
        bounds = WMRL_M3_BOUNDS
        param_names = WMRL_M3_PARAMS
    else:
        raise ValueError(f"Unknown model: {model}")

    # Sample uniformly from bounds
    params_list = []
    for _ in range(n_subjects):
        params = {}
        for param_name in param_names:
            low, high = bounds[param_name]
            params[param_name] = rng.uniform(low, high)
        params_list.append(params)

    return params_list


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_participant(
    params: Dict[str, float],
    model: str,
    seed: int
) -> pd.DataFrame:
    """
    Generate synthetic trial-level data matching task_trials_long.csv structure.

    Uses JAX for speed. Implements Q-learning and WM-RL agent simulation
    with epsilon noise, asymmetric learning rates, and reversal logic.

    Parameters
    ----------
    params : Dict[str, float]
        True parameter values for simulation
    model : str
        Model name ('qlearning', 'wmrl', 'wmrl_m3')
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Trial-level data with columns:
        sona_id, block, stimulus, key_press, reward, set_size, trial_in_block
    """
    key = jax.random.PRNGKey(seed)
    rng = np.random.default_rng(seed)

    # Extract parameters
    alpha_pos = params['alpha_pos']
    alpha_neg = params['alpha_neg']
    epsilon = params['epsilon']

    # Model-specific parameters
    if model in ['wmrl', 'wmrl_m3']:
        phi = params['phi']
        rho = params['rho']
        capacity = params['capacity']
        if model == 'wmrl_m3':
            kappa = params['kappa']
        else:
            kappa = 0.0
    else:
        phi = rho = capacity = kappa = None

    # Initialize data collection
    all_trials = []
    sona_id = 90000 + seed  # Synthetic participant IDs start at 90000

    for block_idx in range(NUM_BLOCKS):
        block_num = block_idx + BLOCK_OFFSET  # Blocks 3-23
        set_size = SET_SIZES[block_idx % len(SET_SIZES)]
        n_trials_block = rng.integers(MIN_TRIALS_PER_BLOCK, MAX_TRIALS_PER_BLOCK + 1)

        # Initialize Q-table and WM matrix at start of block
        Q = np.ones((NUM_STIMULI, NUM_ACTIONS)) * 0.5

        if model in ['wmrl', 'wmrl_m3']:
            WM = np.ones((NUM_STIMULI, NUM_ACTIONS)) * (1.0 / NUM_ACTIONS)  # Baseline = 0.333
            wm_baseline = 1.0 / NUM_ACTIONS
        else:
            WM = wm_baseline = None

        # Reversal tracking
        consecutive_correct = 0
        reversal_threshold = rng.integers(MIN_CORRECT_FOR_REVERSAL, MAX_CORRECT_FOR_REVERSAL + 1)
        reward_mapping = rng.permutation(NUM_ACTIONS)  # Initial stimulus-action mapping

        # Last action for perseveration
        last_action = None

        for trial_in_block in range(1, n_trials_block + 1):
            # Sample stimulus (uniform random)
            key, subkey = jax.random.split(key)
            stimulus = int(jax.random.randint(subkey, (), 0, NUM_STIMULI))

            # Compute action probabilities
            if model in ['wmrl', 'wmrl_m3']:
                # WM-RL hybrid
                # Apply WM decay first
                WM = (1 - phi) * WM + phi * wm_baseline

                # RL component (softmax)
                q_vals = Q[stimulus, :]
                q_scaled = FIXED_BETA * (q_vals - np.max(q_vals))
                rl_probs = np.exp(q_scaled) / np.sum(np.exp(q_scaled))

                # WM component (softmax)
                wm_vals = WM[stimulus, :]
                wm_scaled = FIXED_BETA * (wm_vals - np.max(wm_vals))
                wm_probs = np.exp(wm_scaled) / np.sum(np.exp(wm_scaled))

                # Adaptive weight
                omega = rho * np.minimum(1.0, capacity / set_size)

                # Hybrid policy
                hybrid_probs = omega * wm_probs + (1 - omega) * rl_probs

                # Perseveration (M3 only)
                if model == 'wmrl_m3' and last_action is not None:
                    hybrid_probs = hybrid_probs.copy()
                    hybrid_probs[last_action] += kappa
                    hybrid_probs = hybrid_probs / np.sum(hybrid_probs)

                action_probs = hybrid_probs
            else:
                # Q-learning (softmax)
                q_vals = Q[stimulus, :]
                q_scaled = FIXED_BETA * (q_vals - np.max(q_vals))
                rl_probs = np.exp(q_scaled) / np.sum(np.exp(q_scaled))
                action_probs = rl_probs

            # Apply epsilon noise
            noisy_probs = epsilon / NUM_ACTIONS + (1 - epsilon) * action_probs

            # Sample action
            key, subkey = jax.random.split(key)
            action = int(jax.random.choice(subkey, NUM_ACTIONS, p=jnp.array(noisy_probs)))

            # Generate reward based on current mapping
            correct_action = reward_mapping[stimulus]
            if action == correct_action:
                # 70% reward probability for correct action
                key, subkey = jax.random.split(key)
                reward = float(jax.random.bernoulli(subkey, 0.7))
                if reward > 0.5:
                    consecutive_correct += 1
                else:
                    consecutive_correct = 0
            else:
                # 30% reward probability for incorrect action
                key, subkey = jax.random.split(key)
                reward = float(jax.random.bernoulli(subkey, 0.3))
                consecutive_correct = 0

            # Update Q-value
            delta = reward - Q[stimulus, action]
            alpha = alpha_pos if delta > 0 else alpha_neg
            Q[stimulus, action] = Q[stimulus, action] + alpha * delta

            # Update WM (immediate overwrite)
            if model in ['wmrl', 'wmrl_m3']:
                WM[stimulus, action] = reward

            # Store trial
            all_trials.append({
                'sona_id': sona_id,
                'block': block_num,
                'stimulus': stimulus,
                'key_press': action,
                'reward': reward,
                'set_size': set_size,
                'trial_in_block': trial_in_block
            })

            # Update last action for perseveration
            last_action = action

            # Check for reversal
            if consecutive_correct >= reversal_threshold:
                # Trigger reversal: shuffle reward mapping
                reward_mapping = rng.permutation(NUM_ACTIONS)
                consecutive_correct = 0
                reversal_threshold = rng.integers(MIN_CORRECT_FOR_REVERSAL, MAX_CORRECT_FOR_REVERSAL + 1)

    return pd.DataFrame(all_trials)


# =============================================================================
# Recovery Pipeline
# =============================================================================

def run_parameter_recovery(
    model: str,
    n_subjects: int,
    n_datasets: int,
    seed: int,
    use_gpu: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run complete parameter recovery pipeline.

    For each dataset:
        1. Sample true parameters
        2. Generate synthetic data
        3. Fit via MLE (same procedure as real data)
        4. Store true vs recovered parameters

    Parameters
    ----------
    model : str
        Model name ('qlearning', 'wmrl', 'wmrl_m3')
    n_subjects : int
        Number of synthetic participants per dataset
    n_datasets : int
        Number of independent recovery datasets
    seed : int
        Base random seed
    use_gpu : bool
        Whether to use GPU for fitting (default: False)
    verbose : bool
        Whether to show progress bars (default: True)

    Returns
    -------
    pd.DataFrame
        Results with columns:
        dataset, subject, true_{param}, recovered_{param}, nll, converged
    """
    results = []

    # Outer loop: datasets
    dataset_iter = range(n_datasets)
    if verbose:
        dataset_iter = tqdm(dataset_iter, desc=f"Recovery ({model})", unit="dataset")

    for dataset_idx in dataset_iter:
        # Sample parameter sets for this dataset
        dataset_seed = seed + dataset_idx * 10000
        params_list = sample_parameters(model, n_subjects, dataset_seed)

        # Inner loop: subjects
        subject_iter = range(n_subjects)
        if verbose:
            subject_iter = tqdm(subject_iter, desc=f"  Dataset {dataset_idx+1}", leave=False, unit="subj")

        for subject_idx in subject_iter:
            true_params = params_list[subject_idx]
            subject_seed = dataset_seed + subject_idx

            # Generate synthetic data
            df_synthetic = generate_synthetic_participant(true_params, model, subject_seed)

            # Prepare data for fitting (same as real data)
            participant_id = int(df_synthetic['sona_id'].iloc[0])
            data_dict = prepare_participant_data(df_synthetic, participant_id, model)

            # Fit via MLE with n_starts=50 (matching real fitting)
            fit_result = fit_participant_mle(
                stimuli_blocks=data_dict['stimuli_blocks'],
                actions_blocks=data_dict['actions_blocks'],
                rewards_blocks=data_dict['rewards_blocks'],
                set_sizes_blocks=data_dict.get('set_sizes_blocks'),
                masks_blocks=data_dict.get('masks_blocks'),
                model=model,
                n_starts=50,
                seed=subject_seed,
                verbose=False  # Suppress per-subject output
            )

            # Store results
            result = {
                'dataset': dataset_idx,
                'subject': subject_idx,
                'nll': fit_result['nll'],
                'converged': fit_result['converged']
            }

            # Add true and recovered parameters
            for param_name in true_params.keys():
                result[f'true_{param_name}'] = true_params[param_name]
                result[f'recovered_{param_name}'] = fit_result[param_name]

            results.append(result)

    return pd.DataFrame(results)


# =============================================================================
# Recovery Metrics
# =============================================================================

def compute_recovery_metrics(results_df: pd.DataFrame, model: str) -> pd.DataFrame:
    """
    Compute parameter recovery metrics.

    For each parameter:
    - Pearson r (correlation between true and recovered)
    - RMSE (root mean squared error)
    - Bias (mean error: recovered - true)
    - Pass/fail (r >= 0.80)

    Parameters
    ----------
    results_df : pd.DataFrame
        Recovery results from run_parameter_recovery()
    model : str
        Model name ('qlearning', 'wmrl', 'wmrl_m3')

    Returns
    -------
    pd.DataFrame
        Metrics with columns: parameter, pearson_r, p_value, rmse, bias, pass_fail
    """
    # Get parameter names for model
    if model == 'qlearning':
        param_names = QLEARNING_PARAMS
    elif model == 'wmrl':
        param_names = WMRL_PARAMS
    elif model == 'wmrl_m3':
        param_names = WMRL_M3_PARAMS
    else:
        raise ValueError(f"Unknown model: {model}")

    metrics = []

    for param in param_names:
        true_col = f'true_{param}'
        recovered_col = f'recovered_{param}'

        # Extract values
        true_vals = results_df[true_col].values
        recovered_vals = results_df[recovered_col].values

        # Pearson r
        r, p_value = pearsonr(true_vals, recovered_vals)

        # RMSE
        rmse = np.sqrt(np.mean((true_vals - recovered_vals) ** 2))

        # Bias
        bias = np.mean(recovered_vals - true_vals)

        # Pass/fail (r >= 0.80)
        pass_fail = 'PASS' if r >= 0.80 else 'FAIL'

        metrics.append({
            'parameter': param,
            'pearson_r': r,
            'p_value': p_value,
            'rmse': rmse,
            'bias': bias,
            'pass_fail': pass_fail
        })

    return pd.DataFrame(metrics)


# =============================================================================
# Testing
# =============================================================================

if __name__ == '__main__':
    print("Testing parameter recovery module...")

    # Test 1: Parameter sampling
    print("\n1. Testing parameter sampling:")
    params_list = sample_parameters('wmrl_m3', n_subjects=3, seed=42)
    print(f"   Sampled {len(params_list)} parameter sets")
    print(f"   First params: {params_list[0]}")
    assert len(params_list) == 3
    assert len(params_list[0]) == 7  # M3 has 7 parameters
    print("   ✓ PASSED")

    # Test 2: Synthetic data generation
    print("\n2. Testing synthetic data generation:")
    df = generate_synthetic_participant(params_list[0], 'wmrl_m3', seed=42)
    print(f"   Generated {len(df)} trials across {df['block'].nunique()} blocks")
    print(f"   Columns: {list(df.columns)}")
    assert df['block'].nunique() == NUM_BLOCKS
    assert set(df.columns) >= {'sona_id', 'block', 'stimulus', 'key_press', 'reward', 'set_size', 'trial_in_block'}
    print("   ✓ PASSED")

    # Test 3: Small recovery test (Q-learning, fast)
    print("\n3. Testing recovery pipeline (Q-learning, n=2):")
    results = run_parameter_recovery('qlearning', n_subjects=2, n_datasets=1, seed=42, verbose=False)
    print(f"   Results shape: {results.shape}")
    print(f"   Columns: {list(results.columns)}")
    assert results.shape[0] == 2  # 2 subjects
    assert 'true_alpha_pos' in results.columns
    assert 'recovered_alpha_pos' in results.columns
    print("   ✓ PASSED")

    # Test 4: Metrics computation
    print("\n4. Testing metrics computation:")
    metrics = compute_recovery_metrics(results, 'qlearning')
    print(f"   Metrics:\n{metrics}")
    assert 'pearson_r' in metrics.columns
    assert 'pass_fail' in metrics.columns
    assert len(metrics) == 3  # Q-learning has 3 parameters
    print("   ✓ PASSED")

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
