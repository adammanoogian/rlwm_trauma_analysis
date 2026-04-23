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

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

from rlwm.fitting.mle import fit_participant_mle, prepare_participant_data
from scripts.fitting.mle_utils import (
    QLEARNING_BOUNDS,
    QLEARNING_PARAMS,
    WMRL_BOUNDS,
    WMRL_M3_BOUNDS,
    WMRL_M3_PARAMS,
    WMRL_M4_BOUNDS,
    WMRL_M4_PARAMS,
    WMRL_M5_BOUNDS,
    WMRL_M5_PARAMS,
    WMRL_M6A_BOUNDS,
    WMRL_M6A_PARAMS,
    WMRL_M6B_BOUNDS,
    WMRL_M6B_PARAMS,
    WMRL_PARAMS,
)
from scripts.utils.plotting import plot_behavioral_comparison

# =============================================================================
# Constants (matching real data structure)
# =============================================================================

NUM_BLOCKS = 21  # Blocks 3-23 in real data
BLOCK_OFFSET = 3  # First block is numbered 3
MAX_STIMULI = 6  # Q/WM table dimension: must match likelihood default (num_stimuli=6)
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
# Parameter Sampling and Loading
# =============================================================================

def get_param_names(model: str) -> list[str]:
    """Get parameter names for a model."""
    if model == 'qlearning':
        return QLEARNING_PARAMS
    elif model == 'wmrl':
        return WMRL_PARAMS
    elif model == 'wmrl_m3':
        return WMRL_M3_PARAMS
    elif model == 'wmrl_m5':
        return WMRL_M5_PARAMS
    elif model == 'wmrl_m6a':
        return WMRL_M6A_PARAMS
    elif model == 'wmrl_m6b':
        return WMRL_M6B_PARAMS
    elif model == 'wmrl_m4':
        return WMRL_M4_PARAMS
    else:
        raise ValueError(f"Unknown model: {model}")

def sample_parameters(model: str, n_subjects: int, seed: int) -> list[dict[str, float]]:
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
    list[dict[str, float]]
        list of parameter dictionaries sampled from bounds
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
    elif model == 'wmrl_m5':
        bounds = WMRL_M5_BOUNDS
        param_names = WMRL_M5_PARAMS
    elif model == 'wmrl_m6a':
        bounds = WMRL_M6A_BOUNDS
        param_names = WMRL_M6A_PARAMS
    elif model == 'wmrl_m6b':
        bounds = WMRL_M6B_BOUNDS
        param_names = WMRL_M6B_PARAMS
    elif model == 'wmrl_m4':
        bounds = WMRL_M4_BOUNDS
        param_names = WMRL_M4_PARAMS
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

def load_fitted_params(fitted_params_path: str, model: str) -> list[dict]:
    """
    Load fitted parameters from MLE results CSV.

    Parameters
    ----------
    fitted_params_path : str
        Path to CSV with fitted parameters (e.g., wmrl_m3_individual_fits.csv)
    model : str
        Model name to get parameter names

    Returns
    -------
    list[dict]
        list of parameter dictionaries, one per participant
        Each dict includes 'sona_id' and all model parameters
    """
    df = pd.read_csv(fitted_params_path)

    # Get parameter names for model
    param_names = get_param_names(model)

    params_list = []

    # Determine ID column name (could be 'sona_id' or 'participant_id')
    id_col = 'sona_id' if 'sona_id' in df.columns else 'participant_id'

    for _, row in df.iterrows():
        params = {p: row[p] for p in param_names if p in df.columns}
        params['sona_id'] = row[id_col]  # Always use 'sona_id' key internally
        params_list.append(params)

    return params_list

# =============================================================================
# Synthetic Data Generation
# =============================================================================

def generate_synthetic_participant(
    params: dict[str, float],
    model: str,
    seed: int
) -> pd.DataFrame:
    """
    Generate synthetic trial-level data matching task_trials_long.csv structure.

    Uses numpy for simulation. Implements Q-learning and WM-RL agent simulation
    with epsilon noise, asymmetric learning rates, and reversal logic.

    Each block presents stimuli drawn uniformly from {0, ..., set_size-1},
    matching the real task structure. Q and WM tables have shape (6, 3)
    to match the likelihood default (num_stimuli=6), ensuring fitted indices
    are identical to what prepare_participant_data passes to the likelihood.

    Parameters
    ----------
    params : dict[str, float]
        True parameter values for simulation
    model : str
        Model name ('qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b', 'wmrl_m4')
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
    # M4 has no epsilon parameter; all other models do
    epsilon = params.get('epsilon', 0.0)

    # Model-specific parameters
    if model in ['wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b', 'wmrl_m4']:
        phi = params['phi']
        rho = params['rho']
        capacity = params['capacity']
        if model in ('wmrl_m3', 'wmrl_m5', 'wmrl_m4'):
            kappa = params['kappa']
        elif model == 'wmrl_m6b':
            # Stick-breaking decode: kappa = kappa_total * kappa_share
            kappa_total = params['kappa_total']
            kappa_share = params['kappa_share']
            kappa = kappa_total * kappa_share
            kappa_s = kappa_total * (1.0 - kappa_share)
        else:
            kappa = 0.0
        if model == 'wmrl_m5':
            phi_rl = params['phi_rl']
        else:
            phi_rl = 0.0
        if model == 'wmrl_m6a':
            kappa_s = params['kappa_s']
        elif model != 'wmrl_m6b':
            kappa_s = 0.0
        # M4 LBA parameters
        if model == 'wmrl_m4':
            v_scale = params['v_scale']
            A = params['A']
            delta = params['delta']
            t0 = params['t0']
            b = A + delta  # Reparameterization decode
    else:
        phi = rho = capacity = kappa = phi_rl = kappa_s = None

    # Initialize data collection
    all_trials = []
    sona_id = 90000 + seed  # Synthetic participant IDs start at 90000

    for block_idx in range(NUM_BLOCKS):
        block_num = block_idx + BLOCK_OFFSET  # Blocks 3-23
        set_size = SET_SIZES[block_idx % len(SET_SIZES)]
        n_trials_block = rng.integers(MIN_TRIALS_PER_BLOCK, MAX_TRIALS_PER_BLOCK + 1)

        # Initialize Q-table and WM matrix at start of block.
        # Shape is (MAX_STIMULI, NUM_ACTIONS) = (6, 3), matching the likelihood
        # default (num_stimuli=6). Only indices 0..set_size-1 are ever written,
        # but the full 6-row table is required so fitted indices are identical.
        Q = np.ones((MAX_STIMULI, NUM_ACTIONS)) * 0.5

        if model in ['wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b', 'wmrl_m4']:
            WM = np.ones((MAX_STIMULI, NUM_ACTIONS)) * (1.0 / NUM_ACTIONS)  # Baseline = 0.333
            wm_baseline = 1.0 / NUM_ACTIONS
        else:
            WM = wm_baseline = None

        # Reversal tracking
        consecutive_correct = 0
        reversal_threshold = rng.integers(MIN_CORRECT_FOR_REVERSAL, MAX_CORRECT_FOR_REVERSAL + 1)
        reward_mapping = rng.integers(0, NUM_ACTIONS, size=set_size)  # Random action per stimulus

        # Last action for perseveration
        # M3/M5: global scalar only; M6a: per-stimulus dict only; M6b: BOTH global + per-stimulus
        last_action = None
        if model in ('wmrl_m6a', 'wmrl_m6b'):
            last_actions = {}  # dict: stimulus -> last action (None = never seen in this block)

        for trial_in_block in range(1, n_trials_block + 1):
            # Sample stimulus uniformly from the set_size stimuli present in
            # this block. This matches the real task structure where set_size
            # determines how many unique stimuli (0..set_size-1) appear.
            stimulus = rng.integers(0, set_size)

            # Compute action probabilities
            if model in ['wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b', 'wmrl_m4']:
                # WM-RL hybrid
                # Apply WM decay first
                WM = (1 - phi) * WM + phi * wm_baseline

                # M5: RL forgetting (decay Q toward uniform baseline BEFORE policy)
                if model == 'wmrl_m5':
                    Q0 = 1.0 / NUM_ACTIONS  # = 0.333
                    Q = (1 - phi_rl) * Q + phi_rl * Q0

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

                # =============================================================
                # EPSILON + PERSEVERATION
                # Order and formula MUST match jax_likelihoods.py exactly.
                # Non-M4: epsilon FIRST, then perseveration (convex combination)
                # M4: no epsilon, perseveration feeds into LBA race
                # =============================================================

                if model != 'wmrl_m4':
                    # Step 1: Apply epsilon noise FIRST (before perseveration)
                    # Matches: apply_epsilon_noise(base_probs, epsilon, num_actions)
                    noisy_probs = epsilon / NUM_ACTIONS + (1.0 - epsilon) * hybrid_probs
                else:
                    # M4: no epsilon parameter; hybrid goes directly to perseveration
                    noisy_probs = hybrid_probs.copy()

                # Step 2: Apply perseveration SECOND (convex combination)
                if model in ('wmrl_m3', 'wmrl_m5'):
                    # Global perseveration: (1-kappa)*P_noisy + kappa*Ck
                    # Ref: jax_likelihoods.py line 1156
                    if last_action is not None:
                        Ck = np.zeros(NUM_ACTIONS)
                        Ck[last_action] = 1.0
                        noisy_probs = (1.0 - kappa) * noisy_probs + kappa * Ck

                elif model == 'wmrl_m4':
                    # M4 global perseveration: (1-kappa)*hybrid + kappa*Ck
                    # Ref: lba_likelihood.py (already correct in current code)
                    if last_action is not None:
                        Ck = np.zeros(NUM_ACTIONS)
                        Ck[last_action] = 1.0
                        noisy_probs = (1.0 - kappa) * noisy_probs + kappa * Ck

                elif model == 'wmrl_m6a':
                    # Per-stimulus perseveration: (1-kappa_s)*P_noisy + kappa_s*Ck_stim
                    # Ref: jax_likelihoods.py line 2135
                    if last_actions.get(stimulus) is not None:
                        Ck = np.zeros(NUM_ACTIONS)
                        Ck[last_actions[stimulus]] = 1.0
                        noisy_probs = (1.0 - kappa_s) * noisy_probs + kappa_s * Ck

                elif model == 'wmrl_m6b':
                    # Dual perseveration: three-way blend with effective-weight gating
                    # Ref: jax_likelihoods.py lines 2461-2484
                    eff_kappa = kappa if last_action is not None else 0.0
                    eff_kappa_s = kappa_s if last_actions.get(stimulus) is not None else 0.0
                    if eff_kappa > 0.0 or eff_kappa_s > 0.0:
                        Ck_global = np.zeros(NUM_ACTIONS)
                        Ck_stim = np.zeros(NUM_ACTIONS)
                        if eff_kappa > 0.0:
                            Ck_global[last_action] = 1.0
                        if eff_kappa_s > 0.0:
                            Ck_stim[last_actions[stimulus]] = 1.0
                        noisy_probs = (
                            (1.0 - eff_kappa - eff_kappa_s) * noisy_probs
                            + eff_kappa * Ck_global
                            + eff_kappa_s * Ck_stim
                        )

                action_probs = noisy_probs
            else:
                # Q-learning (softmax + epsilon noise)
                q_vals = Q[stimulus, :]
                q_scaled = FIXED_BETA * (q_vals - np.max(q_vals))
                rl_probs = np.exp(q_scaled) / np.sum(np.exp(q_scaled))
                # Apply epsilon noise (matches jax_likelihoods.py qlearning_block_likelihood)
                action_probs = epsilon / NUM_ACTIONS + (1.0 - epsilon) * rl_probs

            # Action selection
            if model == 'wmrl_m4':
                # LBA race simulation (McDougle & Collins 2021)
                # Drift rates proportional to hybrid policy
                v_all = v_scale * action_probs
                # Start points: k_i ~ Uniform(0, A)
                k = rng.uniform(0, A, size=NUM_ACTIONS)
                # Time to threshold: t_i = (b - k_i) / v_i
                # Clamp drift to avoid zero/negative (ensures finite positive race time)
                v_safe = np.maximum(v_all, 1e-6)
                t_race = (b - k) / v_safe
                # Winner = accumulator that reaches threshold first
                action = int(np.argmin(t_race))
                rt_sim = float(t_race[action] + t0)
            else:
                # Epsilon already applied above (before perseveration)
                # Sample action from final action_probs
                key, subkey = jax.random.split(key)
                action = int(jax.random.choice(subkey, NUM_ACTIONS, p=jnp.array(action_probs)))
                rt_sim = None

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
            if model in ['wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b', 'wmrl_m4']:
                WM[stimulus, action] = reward

            # Store trial
            trial_dict = {
                'sona_id': sona_id,
                'block': block_num,
                'stimulus': stimulus,
                'key_press': action,
                'reward': reward,
                'set_size': set_size,
                'trial_in_block': trial_in_block
            }
            if model == 'wmrl_m4':
                # Store RT in milliseconds (matching real data format;
                # preprocessing converts ms -> s before fitting)
                trial_dict['rt'] = rt_sim * 1000.0
            all_trials.append(trial_dict)

            # Update last action for perseveration
            last_action = action
            # M6a and M6b: update per-stimulus tracking
            if model in ('wmrl_m6a', 'wmrl_m6b'):
                last_actions[stimulus] = action

            # Check for reversal
            if consecutive_correct >= reversal_threshold:
                # Trigger reversal: new random correct actions from {0,1,2}
                reward_mapping = rng.integers(0, NUM_ACTIONS, size=set_size)
                consecutive_correct = 0
                reversal_threshold = rng.integers(MIN_CORRECT_FOR_REVERSAL, MAX_CORRECT_FOR_REVERSAL + 1)

    return pd.DataFrame(all_trials)

# =============================================================================
# Posterior Predictive Check
# =============================================================================

def compare_behavior(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compare behavioral metrics between real and synthetic data.

    Computes per Wilson & Collins (2019) / Palminteri et al. (2017):
    - Overall accuracy
    - Accuracy by set size
    - Learning curve (early vs late blocks)
    - Post-reversal accuracy (if detectable)

    Parameters
    ----------
    real_data : pd.DataFrame
        Real trial data (task_trials_long.csv format)
    synthetic_data : pd.DataFrame
        Synthetic trial data (same format)

    Returns
    -------
    pd.DataFrame
        Comparison table with source (real/synthetic) and metrics
    """
    metrics = []

    for source, data in [('real', real_data), ('synthetic', synthetic_data)]:
        row = {'source': source}

        # Overall accuracy
        row['overall_accuracy'] = data['reward'].mean()

        # Accuracy by set size
        for ss in [2, 3, 5, 6]:
            ss_data = data[data['set_size'] == ss]
            if len(ss_data) > 0:
                row[f'accuracy_ss{ss}'] = ss_data['reward'].mean()
            else:
                row[f'accuracy_ss{ss}'] = np.nan

        # Learning curve: early (first 7 blocks) vs late (last 7 blocks)
        blocks = sorted(data['block'].unique())
        if len(blocks) >= 14:
            early_blocks = blocks[:7]
            late_blocks = blocks[-7:]
            row['accuracy_early'] = data[data['block'].isin(early_blocks)]['reward'].mean()
            row['accuracy_late'] = data[data['block'].isin(late_blocks)]['reward'].mean()
            row['learning_effect'] = row['accuracy_late'] - row['accuracy_early']
        else:
            row['accuracy_early'] = np.nan
            row['accuracy_late'] = np.nan
            row['learning_effect'] = np.nan

        # Post-reversal accuracy (trials 1-5 after reversal)
        # This requires identifying reversal points - simplified version:
        # Use first 5 trials of each block as proxy for post-reversal
        first_trials = data.groupby(['sona_id', 'block']).head(5)
        row['post_reversal_accuracy'] = first_trials['reward'].mean()

        metrics.append(row)

    comparison_df = pd.DataFrame(metrics)

    # Add difference row
    real_row = comparison_df[comparison_df['source'] == 'real'].iloc[0]
    syn_row = comparison_df[comparison_df['source'] == 'synthetic'].iloc[0]

    diff_row = {'source': 'difference'}
    for col in comparison_df.columns:
        if col != 'source' and not pd.isna(real_row[col]) and not pd.isna(syn_row[col]):
            diff_row[col] = syn_row[col] - real_row[col]

    comparison_df = pd.concat([comparison_df, pd.DataFrame([diff_row])], ignore_index=True)

    return comparison_df

def run_posterior_predictive_check(
    model: str,
    fitted_params_path: str,
    real_data_path: str,
    output_dir: Path,
    figures_dir: Path,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Generate synthetic data from fitted params and compare to real behavior.

    Workflow:
    1. Load fitted parameters from MLE results
    2. Generate synthetic data for each participant using their fitted params
    3. Compute behavioral metrics on both real and synthetic data
    4. Compare and generate overlay plots

    Parameters
    ----------
    model : str
        Model name ('qlearning', 'wmrl', 'wmrl_m3')
    fitted_params_path : str
        Path to fitted parameters CSV
    real_data_path : str
        Path to real trial data
    output_dir : Path
        Directory for output CSVs
    figures_dir : Path
        Directory for figures
    verbose : bool
        Show progress bars

    Returns
    -------
    pd.DataFrame
        Behavioral comparison metrics (real vs synthetic)
    """
    # 1. Load fitted params
    print(f"Loading fitted parameters from: {fitted_params_path}")
    fitted_params_list = load_fitted_params(fitted_params_path, model)
    print(f"  Found {len(fitted_params_list)} participants")

    # 2. Generate synthetic data for each participant
    synthetic_dfs = []
    iterator = tqdm(fitted_params_list, desc="Generating synthetic data") if verbose else fitted_params_list

    for params in iterator:
        sona_id = params['sona_id']
        # Use participant ID as seed for reproducibility
        seed = int(sona_id) if isinstance(sona_id, (int, float)) else hash(str(sona_id)) % 2**31

        syn_df = generate_synthetic_participant(
            params=params,
            model=model,
            seed=seed
        )
        syn_df['sona_id'] = sona_id  # Keep original participant ID
        synthetic_dfs.append(syn_df)

    synthetic_data = pd.concat(synthetic_dfs, ignore_index=True)

    # 3. Save synthetic data
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_path = output_dir / 'synthetic_trials.csv'
    synthetic_data.to_csv(synthetic_path, index=False)
    print(f"Saved synthetic data to: {synthetic_path}")

    # 4. Load real data and compare
    print(f"Loading real data from: {real_data_path}")
    real_data = pd.read_csv(real_data_path)

    # 5. Compute behavioral comparison
    comparison_df = compare_behavior(real_data, synthetic_data)

    # 6. Save comparison
    comparison_path = output_dir / 'behavioral_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Saved comparison to: {comparison_path}")

    # 7. Generate overlay plots
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_behavioral_comparison(real_data, synthetic_data, figures_dir, model_name=model.upper())

    return comparison_df

# =============================================================================
# Recovery Pipeline
# =============================================================================

def run_parameter_recovery(
    model: str,
    n_subjects: int,
    n_datasets: int,
    seed: int,
    use_gpu: bool = False,
    verbose: bool = True,
    n_starts: int = 20,
    n_jobs: int = 1
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
        Model name ('qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b', 'wmrl_m4')
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
    n_starts : int
        Number of random restarts per participant. Default 20 is adequate for
        recovery validation; 50 is overkill and triples runtime for WM-RL models.
    n_jobs : int
        Number of parallel jobs for CPU fitting (default: 1)

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

            # Fit via MLE
            fit_result = fit_participant_mle(
                stimuli_blocks=data_dict['stimuli_blocks'],
                actions_blocks=data_dict['actions_blocks'],
                rewards_blocks=data_dict['rewards_blocks'],
                set_sizes_blocks=data_dict.get('set_sizes_blocks'),
                masks_blocks=data_dict.get('masks_blocks'),
                rts_blocks=data_dict.get('rts_blocks'),  # None for choice-only models
                model=model,
                n_starts=n_starts,
                seed=subject_seed,
                compute_diagnostics=False,  # Not needed for recovery
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
    elif model == 'wmrl_m5':
        param_names = WMRL_M5_PARAMS
    elif model == 'wmrl_m6a':
        param_names = WMRL_M6A_PARAMS
    elif model == 'wmrl_m6b':
        param_names = WMRL_M6B_PARAMS
    elif model == 'wmrl_m4':
        param_names = WMRL_M4_PARAMS
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
# Visualization (wired in main())
# =============================================================================

def plot_recovery_scatter(
    results_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    model: str,
    output_dir: Path
) -> None:
    """
    Generate scatter plots showing true vs recovered parameters.

    For each parameter, creates individual scatter plot with:
    - Identity line (y=x, dashed black)
    - Regression line (solid red)
    - Annotations (r, RMSE, Bias)
    - PASS/FAIL badge based on r >= 0.80 threshold

    Also creates combined all_parameters_recovery.png with subplots.

    Parameters
    ----------
    results_df : pd.DataFrame
        Recovery results from run_parameter_recovery()
    metrics_df : pd.DataFrame
        Metrics from compute_recovery_metrics()
    model : str
        Model name for parameter list
    output_dir : Path
        Directory to save figures
    """
    from scripts.utils.plotting import plot_scatter_with_annotations

    # Get parameter names
    if model == 'qlearning':
        param_names = QLEARNING_PARAMS
    elif model == 'wmrl':
        param_names = WMRL_PARAMS
    elif model == 'wmrl_m3':
        param_names = WMRL_M3_PARAMS
    elif model == 'wmrl_m5':
        param_names = WMRL_M5_PARAMS
    elif model == 'wmrl_m6a':
        param_names = WMRL_M6A_PARAMS
    elif model == 'wmrl_m6b':
        param_names = WMRL_M6B_PARAMS
    elif model == 'wmrl_m4':
        param_names = WMRL_M4_PARAMS
    else:
        raise ValueError(f"Unknown model: {model}")

    n_params = len(param_names)

    # Create combined figure
    ncols = 3
    nrows = int(np.ceil(n_params / ncols))
    fig_combined, axes_combined = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    axes_combined = axes_combined.flatten() if n_params > 1 else [axes_combined]

    for idx, param in enumerate(param_names):
        # Extract data
        true_vals = results_df[f'true_{param}'].values
        recovered_vals = results_df[f'recovered_{param}'].values

        # Get metrics for this parameter
        param_metrics = metrics_df[metrics_df['parameter'] == param].iloc[0]
        annotations = {
            'r': param_metrics['pearson_r'],
            'RMSE': param_metrics['rmse'],
            'Bias': param_metrics['bias']
        }

        # Individual plot
        fig_ind, ax_ind = plt.subplots(figsize=(6, 6))
        plot_scatter_with_annotations(
            ax_ind, true_vals, recovered_vals,
            annotations=annotations,
            pass_threshold=0.80,
            pass_key='r'
        )
        ax_ind.set_xlabel(f'True {param}', fontsize=12)
        ax_ind.set_ylabel(f'Recovered {param}', fontsize=12)
        ax_ind.set_title(f'{param.capitalize()} Recovery', fontsize=14, fontweight='bold')
        fig_ind.tight_layout()
        fig_ind.savefig(output_dir / f'{param}_recovery.png', dpi=300, bbox_inches='tight')
        plt.close(fig_ind)

        # Add to combined plot
        ax_combined = axes_combined[idx]
        plot_scatter_with_annotations(
            ax_combined, true_vals, recovered_vals,
            annotations=annotations,
            pass_threshold=0.80,
            pass_key='r',
            s=30  # Smaller points for combined view
        )
        ax_combined.set_xlabel(f'True {param}', fontsize=10)
        ax_combined.set_ylabel(f'Recovered {param}', fontsize=10)
        ax_combined.set_title(f'{param.capitalize()}', fontsize=11, fontweight='bold')

    # Hide unused subplots
    for idx in range(n_params, len(axes_combined)):
        axes_combined[idx].axis('off')

    fig_combined.suptitle(f'{model.upper()} Parameter Recovery', fontsize=16, fontweight='bold')
    fig_combined.tight_layout()
    fig_combined.savefig(output_dir / 'all_parameters_recovery.png', dpi=300, bbox_inches='tight')
    plt.close(fig_combined)

def plot_distribution_comparison(
    results_df: pd.DataFrame,
    real_params_path: str | None,
    model: str,
    output_dir: Path
) -> None:
    """
    Generate KDE distribution plots comparing recovered vs real fitted parameters.

    For each parameter, creates overlapping KDE plots showing:
    - Recovered parameters (from synthetic data recovery)
    - Real fitted parameters (from actual participant data)

    This provides sanity check that synthetic data has realistic parameter
    distributions matching real data.

    Parameters
    ----------
    results_df : pd.DataFrame
        Recovery results from run_parameter_recovery()
    real_params_path : str or None
        Path to real fitted parameters CSV. If None, tries default location.
    model : str
        Model name for parameter list
    output_dir : Path
        Directory to save figures
    """
    from scripts.utils.plotting import plot_kde_comparison

    # Get parameter names
    if model == 'qlearning':
        param_names = QLEARNING_PARAMS
    elif model == 'wmrl':
        param_names = WMRL_PARAMS
    elif model == 'wmrl_m3':
        param_names = WMRL_M3_PARAMS
    elif model == 'wmrl_m5':
        param_names = WMRL_M5_PARAMS
    elif model == 'wmrl_m6a':
        param_names = WMRL_M6A_PARAMS
    elif model == 'wmrl_m6b':
        param_names = WMRL_M6B_PARAMS
    elif model == 'wmrl_m4':
        param_names = WMRL_M4_PARAMS
    else:
        raise ValueError(f"Unknown model: {model}")

    # Try to load real fitted parameters
    if real_params_path is None:
        real_params_path = f'output/mle/{model}_individual_fits.csv'

    try:
        df_real = pd.read_csv(real_params_path)
        has_real_data = True
    except FileNotFoundError:
        print(f"Warning: Real fitted params not found at {real_params_path}")
        print("         Skipping distribution comparison plots.")
        has_real_data = False
        return

    # Create combined figure
    n_params = len(param_names)
    ncols = 3
    nrows = int(np.ceil(n_params / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten() if n_params > 1 else [axes]

    for idx, param in enumerate(param_names):
        ax = axes[idx]

        # Extract distributions
        recovered_vals = results_df[f'recovered_{param}'].values
        real_vals = df_real[param].values

        distributions = {
            'Recovered': recovered_vals,
            'Real Fitted': real_vals
        }

        colors = {
            'Recovered': '#1f77b4',  # Blue
            'Real Fitted': '#ff7f0e'  # Orange
        }

        # Plot KDE comparison
        plot_kde_comparison(ax, distributions, colors=colors)

        ax.set_xlabel(param.capitalize(), fontsize=11)
        ax.set_ylabel('Density', fontsize=11)
        ax.set_title(f'{param.capitalize()} Distribution', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)

    # Hide unused subplots
    for idx in range(n_params, len(axes)):
        axes[idx].axis('off')

    fig.suptitle(f'{model.upper()} Parameter Distributions', fontsize=16, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_dir / 'parameter_distributions.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

# =============================================================================
# Main CLI
# =============================================================================

def main():
    """
    Command-line interface for parameter recovery analysis.

    Runs complete recovery pipeline:
    1. Sample true parameters uniformly from MLE bounds
    2. Generate synthetic trial-level data
    3. Fit via MLE (same procedure as real data)
    4. Compute recovery metrics (r, RMSE, bias)
    5. Save results to CSV
    6. Generate visualization plots

    Pass/fail criterion: Pearson r >= 0.80 per parameter (Senta et al., 2025)
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Parameter recovery analysis for RLWM models (Senta et al. methodology)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Q-learning recovery with 50 subjects, 10 datasets
  python scripts/fitting/model_recovery.py --model qlearning --n-subjects 50 --n-datasets 10

  # WM-RL M3 recovery with GPU acceleration
  python scripts/fitting/model_recovery.py --model wmrl_m3 --n-subjects 100 --n-datasets 10 --use-gpu

  # Quick test with 10 subjects, 1 dataset
  python scripts/fitting/model_recovery.py --model wmrl --n-subjects 10 --n-datasets 1 --seed 42
        """
    )
    parser.add_argument('--mode', type=str, default='recovery',
                        choices=['recovery', 'ppc'],
                        help='recovery: sample params, validate fitting | ppc: use fitted params, validate model')
    parser.add_argument('--model', type=str, required=True,
                        choices=['qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b', 'wmrl_m4'],
                        help='Model to test recovery')
    parser.add_argument('--n-subjects', type=int, default=50,
                        help='Number of synthetic subjects per dataset (default: 50)')
    parser.add_argument('--n-datasets', type=int, default=3,
                        help='Number of independent datasets (default: 3)')
    parser.add_argument('--n-starts', type=int, default=20,
                        help='Random restarts per subject during fitting (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU acceleration if available')
    parser.add_argument('--output-dir', type=str, default='output/recovery',
                        help='Output directory for results (default: output/recovery)')
    parser.add_argument('--figures-dir', type=str, default='figures/recovery',
                        help='Output directory for figures (default: figures/recovery)')
    parser.add_argument('--fitted-params', type=str, default=None,
                        help='Path to fitted params CSV (auto-detected from --model if not specified)')
    parser.add_argument('--real-data', type=str, default='output/task_trials_long.csv',
                        help='Path to real trial data for behavioral comparison')
    parser.add_argument('--real-params', type=str, default=None,
                        help='Path to real fitted parameters CSV for distribution comparison')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')
    args = parser.parse_args()

    # Auto-detect fitted params path if PPC mode and not specified
    if args.mode == 'ppc' and args.fitted_params is None:
        args.fitted_params = f'output/mle/{args.model}_individual_fits.csv'

    # Create output directories
    if args.mode == 'recovery':
        output_dir = Path(args.output_dir) / args.model
        figures_dir = Path(args.figures_dir) / args.model
    else:  # ppc mode
        output_dir = Path('output/ppc') / args.model
        figures_dir = Path('figures/ppc') / args.model

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Branch on mode
    if args.mode == 'recovery':
        # ===== PARAMETER RECOVERY MODE =====
        # Print configuration
        if not args.quiet:
            print("=" * 80)
            print("PARAMETER RECOVERY ANALYSIS")
            print("=" * 80)
            print(f"Model:        {args.model}")
            print(f"N subjects:   {args.n_subjects}")
            print(f"N datasets:   {args.n_datasets}")
            print(f"Seed:         {args.seed}")
            print(f"Output dir:   {output_dir}")
            print(f"Figures dir:  {figures_dir}")

            # Check GPU availability
            if args.use_gpu:
                try:
                    devices = jax.devices('gpu')
                    print(f"GPU:          {len(devices)} device(s) available")
                except RuntimeError:
                    print("GPU:          Requested but not available, using CPU")
            else:
                print("GPU:          Not requested (using CPU)")

            print("=" * 80)
            print()

        # Run recovery
        if not args.quiet:
            print("Running parameter recovery pipeline...")

        results_df = run_parameter_recovery(
            model=args.model,
            n_subjects=args.n_subjects,
            n_datasets=args.n_datasets,
            seed=args.seed,
            use_gpu=args.use_gpu,
            verbose=not args.quiet,
            n_starts=args.n_starts,
        )

        # Compute metrics
        if not args.quiet:
            print("\nComputing recovery metrics...")

        metrics_df = compute_recovery_metrics(results_df, args.model)

        # Save CSVs
        results_path = output_dir / 'recovery_results.csv'
        metrics_path = output_dir / 'recovery_metrics.csv'

        results_df.to_csv(results_path, index=False)
        metrics_df.to_csv(metrics_path, index=False)

        if not args.quiet:
            print(f"Saved results:  {results_path}")
            print(f"Saved metrics:  {metrics_path}")

        # Generate plots
        if not args.quiet:
            print("\nGenerating visualization plots...")

        plot_recovery_scatter(results_df, metrics_df, args.model, figures_dir)

        if not args.quiet:
            print(f"Saved scatter plots: {figures_dir}/")

        plot_distribution_comparison(results_df, args.real_params, args.model, figures_dir)

        if not args.quiet:
            print(f"Saved distribution plots: {figures_dir}/")

        # Print summary table
        if not args.quiet:
            print("\n" + "=" * 80)
            print("RECOVERY METRICS SUMMARY")
            print("=" * 80)
            print(metrics_df.to_string(index=False))
            print("=" * 80)

        # Determine exit code
        all_passed = (metrics_df['pass_fail'] == 'PASS').all()

        if all_passed:
            if not args.quiet:
                print("\n✓ ALL PARAMETERS PASSED (r >= 0.80)")
                print("  Recovery quality is excellent.")
            return 0
        else:
            failed_params = metrics_df[metrics_df['pass_fail'] == 'FAIL']['parameter'].tolist()
            if not args.quiet:
                print(f"\n✗ SOME PARAMETERS FAILED: {', '.join(failed_params)}")
                print("  Recovery quality needs improvement.")
            return 1

    elif args.mode == 'ppc':
        # ===== POSTERIOR PREDICTIVE CHECK MODE =====
        # Print configuration
        if not args.quiet:
            print("=" * 80)
            print("POSTERIOR PREDICTIVE CHECK")
            print("=" * 80)
            print(f"Model:         {args.model}")
            print(f"Fitted params: {args.fitted_params}")
            print(f"Real data:     {args.real_data}")
            print(f"Output dir:    {output_dir}")
            print(f"Figures dir:   {figures_dir}")
            print("=" * 80)
            print()

        # Run PPC
        comparison = run_posterior_predictive_check(
            model=args.model,
            fitted_params_path=args.fitted_params,
            real_data_path=args.real_data,
            output_dir=output_dir,
            figures_dir=figures_dir,
            verbose=not args.quiet
        )

        # Print summary
        print("\n" + "=" * 80)
        print("POSTERIOR PREDICTIVE CHECK RESULTS")
        print("=" * 80)
        print(comparison.to_string(index=False))
        print("=" * 80)

        return 0

# =============================================================================
# Model Recovery Check
# =============================================================================

def run_model_recovery_check(
    synthetic_data_path: str,
    generative_model: str,
    output_dir: Path,
    comparison_models: list[str] | None = None,
    use_gpu: bool = False,
    n_jobs: int = 1,
    n_starts: int = 50,
    verbose: bool = True
) -> dict:
    """
    Fit all models to synthetic data and check if generative model wins.

    This is the "model recovery" aspect of PPC - verifying that when we generate
    data from model X, model X wins the model comparison on that data.

    Per Senta et al. (2025), proper model recovery requires:
    1. Generate data from known model with known parameters
    2. Fit ALL models to this synthetic data
    3. Compare via AIC/BIC
    4. Generative model should win (lowest AIC)

    Note: M4 (RLWM-LBA) is excluded from the default comparison set because
    its joint choice+RT likelihood is incommensurable with choice-only AIC.

    Parameters
    ----------
    synthetic_data_path : str
        Path to synthetic trial data CSV
    generative_model : str
        Model that generated the synthetic data
    output_dir : Path
        Directory to save MLE results
    comparison_models : list[str] or None
        Models to fit and compare. Defaults to all 6 choice-only models:
        ['qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b'].
        M4 excluded because joint likelihood AIC is incommensurable with choice-only.
    use_gpu : bool
        Use GPU acceleration (default: False)
    n_jobs : int
        Number of parallel jobs for CPU fitting (default: 1)
    n_starts : int
        Number of random starts for MLE optimization (default: 50)
    verbose : bool
        Show progress (default: True)

    Returns
    -------
    dict with:
        - generative_model: str (model that generated data)
        - winning_model: str (model with lowest AIC)
        - generative_wins: bool (winning_model == generative_model)
        - aic_scores: dict[str, float] (AIC per model)
        - bic_scores: dict[str, float] (BIC per model)
        - confusion_entry: (generative, winning) tuple for confusion matrix
    """
    import subprocess

    # Default: all 6 choice-only models (M4 excluded — joint likelihood incommensurable)
    CHOICE_ONLY_MODELS = ['qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b']
    if comparison_models is None:
        models = CHOICE_ONLY_MODELS
    else:
        models = list(comparison_models)

    # Ensure generative model is in the comparison set
    if generative_model not in models:
        models = [generative_model] + models
    mle_results_dir = output_dir / 'mle_results'
    mle_results_dir.mkdir(parents=True, exist_ok=True)

    # Fit each model to synthetic data
    for model in models:
        if verbose:
            print(f"\nFitting {model} to synthetic data...")

        cmd = [
            'python', 'scripts/04_model_fitting/a_mle/fit_mle.py',
            '--model', model,
            '--data', str(synthetic_data_path),
            '--output', str(mle_results_dir),
            '--n-starts', str(n_starts)
        ]

        if use_gpu:
            cmd.append('--use-gpu')
        elif n_jobs > 1:
            cmd.extend(['--n-jobs', str(n_jobs)])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Warning: Fitting {model} may have issues")
            if verbose and result.stderr:
                print(f"  stderr: {result.stderr[:200]}")

    # Compare models
    if verbose:
        print("\nComparing models...")

    # Load AIC/BIC from each model's results
    aic_scores = {}
    bic_scores = {}

    for model in models:
        results_path = mle_results_dir / f'{model}_individual_fits.csv'
        if results_path.exists():
            df = pd.read_csv(results_path)
            # Sum AIC/BIC across all participants
            aic_scores[model] = df['aic'].sum() if 'aic' in df.columns else float('inf')
            bic_scores[model] = df['bic'].sum() if 'bic' in df.columns else float('inf')
        else:
            aic_scores[model] = float('inf')
            bic_scores[model] = float('inf')

    # Determine winner (lowest AIC)
    winning_model = min(aic_scores, key=aic_scores.get)
    generative_wins = winning_model == generative_model

    result = {
        'generative_model': generative_model,
        'winning_model': winning_model,
        'generative_wins': generative_wins,
        'aic_scores': aic_scores,
        'bic_scores': bic_scores,
        'confusion_entry': (generative_model, winning_model)
    }

    # Print summary
    if verbose:
        print("\n" + "="*60)
        print("MODEL RECOVERY RESULTS")
        print("="*60)
        print(f"Generative model: {generative_model}")
        print(f"Winning model:    {winning_model}")
        print("\nAIC Scores:")
        for model, aic in sorted(aic_scores.items(), key=lambda x: x[1]):
            marker = " <- WINNER" if model == winning_model else ""
            gen_marker = " (generative)" if model == generative_model else ""
            print(f"  {model}: {aic:.1f}{gen_marker}{marker}")
        print(f"\nModel Recovery: {'PASS' if generative_wins else 'FAIL'}")
        print("="*60)

    return result


# Default choice-only models for cross-model recovery.
# M4 (RLWM-LBA) excluded: joint choice+RT likelihood AIC is incommensurable
# with choice-only models.
CHOICE_ONLY_MODELS = ['qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b']


def run_cross_model_recovery(
    generating_models: list[str] | None = None,
    n_subjects: int = 50,
    n_datasets: int = 10,
    seed: int = 42,
    use_gpu: bool = False,
    n_jobs: int = 1,
    n_starts: int = 50,
    verbose: bool = True
) -> dict:
    """
    Cross-model recovery: generate from each model, fit all choice-only models,
    build confusion matrix (rows=generator, cols=winner).

    For each generating model and each dataset:
    1. Sample parameters and generate synthetic data
    2. Fit all 6 choice-only models to that data
    3. Determine AIC winner
    4. Tally winner in confusion matrix

    M4 (RLWM-LBA) is excluded from both generation and comparison because its
    joint choice+RT likelihood is incommensurable with choice-only AIC.

    Parameters
    ----------
    generating_models : list[str] or None
        Models to generate data from. Defaults to CHOICE_ONLY_MODELS.
    n_subjects : int
        Number of synthetic subjects per dataset (default: 50)
    n_datasets : int
        Number of independent datasets per generating model (default: 10)
    seed : int
        Base random seed (default: 42)
    use_gpu : bool
        Use GPU for fitting (default: False)
    n_jobs : int
        Parallel jobs for CPU fitting (default: 1)
    n_starts : int
        Random starts for MLE optimization (default: 50)
    verbose : bool
        Show progress output (default: True)

    Returns
    -------
    dict with:
        - confusion_matrix: pd.DataFrame (rows=generator, cols=winner, values=win count)
        - per_generator_results: list of dicts (one per generating model)
        - all_pass: bool (True if every generator won plurality of its datasets)
    """
    import tempfile

    if generating_models is None:
        generating_models = list(CHOICE_ONLY_MODELS)

    comparison_models = list(CHOICE_ONLY_MODELS)

    # Initialize confusion matrix
    confusion = pd.DataFrame(
        0, index=generating_models, columns=comparison_models
    )
    confusion.index.name = 'generator'
    confusion.columns.name = 'winner'

    per_generator_results = []

    for gen_model in generating_models:
        if verbose:
            print(f"\n{'='*70}")
            print(f"CROSS-MODEL RECOVERY: Generating from {gen_model.upper()}")
            print(f"{'='*70}")

        gen_wins = 0
        gen_total = 0
        dataset_results = []

        for ds_idx in range(n_datasets):
            dataset_seed = seed + hash(gen_model) % 10000 + ds_idx * 10000
            if verbose:
                print(f"\n  Dataset {ds_idx + 1}/{n_datasets} (seed={dataset_seed})")

            # 1. Sample parameters and generate synthetic data
            params_list = sample_parameters(gen_model, n_subjects, dataset_seed)
            synthetic_dfs = []
            for subj_idx, params in enumerate(params_list):
                subj_seed = dataset_seed + subj_idx
                df = generate_synthetic_participant(params, gen_model, subj_seed)
                synthetic_dfs.append(df)

            synthetic_data = pd.concat(synthetic_dfs, ignore_index=True)

            # 2. Save to temp directory
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                data_path = tmp_path / f'{gen_model}_synthetic_data.csv'
                synthetic_data.to_csv(data_path, index=False)

                output_dir = tmp_path / 'recovery' / gen_model / f'dataset_{ds_idx}'

                # 3. Fit all choice-only models and determine winner
                result = run_model_recovery_check(
                    synthetic_data_path=str(data_path),
                    generative_model=gen_model,
                    output_dir=output_dir,
                    comparison_models=comparison_models,
                    use_gpu=use_gpu,
                    n_jobs=n_jobs,
                    n_starts=n_starts,
                    verbose=verbose
                )

                # 4. Tally in confusion matrix
                winner = result['winning_model']
                if winner in confusion.columns:
                    confusion.loc[gen_model, winner] += 1
                gen_total += 1
                if result['generative_wins']:
                    gen_wins += 1
                dataset_results.append(result)

        gen_pass = gen_wins > gen_total / 2  # Plurality: wins more than half
        per_generator_results.append({
            'model': gen_model,
            'wins': gen_wins,
            'total': gen_total,
            'pass': gen_pass,
            'dataset_results': dataset_results
        })

        if verbose:
            status = "PASS" if gen_pass else "FAIL"
            print(f"\n  {gen_model}: {gen_wins}/{gen_total} datasets won ({status})")

    all_pass = all(r['pass'] for r in per_generator_results)

    # Print confusion matrix
    if verbose:
        print(f"\n{'='*70}")
        print("CROSS-MODEL CONFUSION MATRIX")
        print("(rows = generating model, columns = AIC winner)")
        print(f"{'='*70}")
        print(confusion.to_string())
        print(f"\n{'='*70}")
        print("PER-GENERATOR SUMMARY")
        print(f"{'='*70}")
        for r in per_generator_results:
            status = "PASS" if r['pass'] else "FAIL"
            print(f"  {r['model']}: {r['wins']}/{r['total']} ({status})")
        overall = "ALL PASS" if all_pass else "SOME FAILED"
        print(f"\nOverall: {overall}")
        print(f"{'='*70}")

    return {
        'confusion_matrix': confusion,
        'per_generator_results': per_generator_results,
        'all_pass': all_pass
    }


# =============================================================================
# Testing
# =============================================================================

def run_tests():
    """Run module tests (when called directly without arguments)."""
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

if __name__ == '__main__':
    import sys
    if len(sys.argv) == 1:
        # No arguments - run tests
        run_tests()
    else:
        # Has arguments - run CLI
        sys.exit(main())
