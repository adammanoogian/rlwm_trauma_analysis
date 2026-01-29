"""
Maximum Likelihood Estimation for RLWM Models

Following Senta et al. (2025) methodology:
- Individual fits with 20 random starting points
- scipy.optimize.minimize with L-BFGS-B
- AIC/BIC for model comparison
- Group statistics: mean +/- SEM across participants

Usage:
    python scripts/fitting/fit_mle.py --model qlearning --data output/task_trials_long.csv
    python scripts/fitting/fit_mle.py --model wmrl --data output/task_trials_long.csv
    python scripts/fitting/fit_mle.py --model wmrl_m3 --data output/task_trials_long.csv
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from functools import partial
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import config for exclusions
from config import EXCLUDED_PARTICIPANTS

# Import JAX likelihood functions
from scripts.fitting.jax_likelihoods import (
    q_learning_multiblock_likelihood,
    wmrl_multiblock_likelihood,
    wmrl_m3_multiblock_likelihood,
    prepare_block_data,
)

# Import MLE utilities
from scripts.fitting.mle_utils import (
    params_to_unconstrained,
    unconstrained_to_params,
    sample_random_start,
    get_default_params,
    compute_aic,
    compute_bic,
    compute_aicc,
    get_n_params,
    summarize_all_parameters,
    check_convergence,
    check_at_bounds,
    QLEARNING_PARAMS,
    WMRL_PARAMS,
    WMRL_M3_PARAMS,
)

# Suppress JAX warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# Objective Functions
# =============================================================================

def _objective_qlearning(
    x: np.ndarray,
    stimuli_blocks: List[np.ndarray],
    actions_blocks: List[np.ndarray],
    rewards_blocks: List[np.ndarray],
) -> float:
    """
    Negative log-likelihood objective for Q-learning model.

    Args:
        x: Unconstrained parameter array [alpha_pos, alpha_neg, epsilon]
        stimuli_blocks: List of stimulus arrays per block
        actions_blocks: List of action arrays per block
        rewards_blocks: List of reward arrays per block

    Returns:
        Negative log-likelihood (for minimization)

    Note: Beta is fixed at 50 inside the likelihood function.
    """
    params = unconstrained_to_params(x, 'qlearning')

    try:
        log_lik = q_learning_multiblock_likelihood(
            stimuli_blocks=stimuli_blocks,
            actions_blocks=actions_blocks,
            rewards_blocks=rewards_blocks,
            alpha_pos=params['alpha_pos'],
            alpha_neg=params['alpha_neg'],
            epsilon=params['epsilon']
        )
        return -float(log_lik)  # Negative for minimization
    except Exception as e:
        return np.inf  # Return high value on error


def _objective_wmrl(
    x: np.ndarray,
    stimuli_blocks: List[np.ndarray],
    actions_blocks: List[np.ndarray],
    rewards_blocks: List[np.ndarray],
    set_sizes_blocks: List[np.ndarray],
) -> float:
    """
    Negative log-likelihood objective for WM-RL model.

    Args:
        x: Unconstrained parameter array [alpha_pos, alpha_neg, phi, rho, capacity, epsilon]
        stimuli_blocks: List of stimulus arrays per block
        actions_blocks: List of action arrays per block
        rewards_blocks: List of reward arrays per block
        set_sizes_blocks: List of set size arrays per block

    Returns:
        Negative log-likelihood (for minimization)

    Note: Beta is fixed at 50 inside the likelihood function.
    """
    params = unconstrained_to_params(x, 'wmrl')

    try:
        log_lik = wmrl_multiblock_likelihood(
            stimuli_blocks=stimuli_blocks,
            actions_blocks=actions_blocks,
            rewards_blocks=rewards_blocks,
            set_sizes_blocks=set_sizes_blocks,
            alpha_pos=params['alpha_pos'],
            alpha_neg=params['alpha_neg'],
            phi=params['phi'],
            rho=params['rho'],
            capacity=params['capacity'],
            epsilon=params['epsilon']
        )
        return -float(log_lik)
    except Exception as e:
        return np.inf


def _objective_wmrl_m3(
    x: np.ndarray,
    stimuli_blocks: List[np.ndarray],
    actions_blocks: List[np.ndarray],
    rewards_blocks: List[np.ndarray],
    set_sizes_blocks: List[np.ndarray],
) -> float:
    """
    Negative log-likelihood objective for WM-RL M3 model (with kappa perseveration).

    Args:
        x: Unconstrained parameter array [alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon]
        stimuli_blocks: List of stimulus arrays per block
        actions_blocks: List of action arrays per block
        rewards_blocks: List of reward arrays per block
        set_sizes_blocks: List of set size arrays per block

    Returns:
        Negative log-likelihood (for minimization)

    Note: Beta is fixed at 50 inside the likelihood function.
    """
    params = unconstrained_to_params(x, 'wmrl_m3')

    try:
        log_lik = wmrl_m3_multiblock_likelihood(
            stimuli_blocks=stimuli_blocks,
            actions_blocks=actions_blocks,
            rewards_blocks=rewards_blocks,
            set_sizes_blocks=set_sizes_blocks,
            alpha_pos=params['alpha_pos'],
            alpha_neg=params['alpha_neg'],
            phi=params['phi'],
            rho=params['rho'],
            capacity=params['capacity'],
            kappa=params['kappa'],
            epsilon=params['epsilon']
        )
        return -float(log_lik)
    except Exception as e:
        return np.inf


# =============================================================================
# Single Participant Fitting
# =============================================================================

def fit_participant_mle(
    stimuli_blocks: List[np.ndarray],
    actions_blocks: List[np.ndarray],
    rewards_blocks: List[np.ndarray],
    set_sizes_blocks: Optional[List[np.ndarray]] = None,
    model: str = 'qlearning',
    n_starts: int = 20,
    seed: Optional[int] = None,
) -> Dict:
    """
    Fit a single participant using MLE with multiple random starts.

    Following Senta et al. (2025): 20 random starting points, keep best result.
    Note: Beta is fixed at 50 inside the likelihood functions.

    Args:
        stimuli_blocks: List of stimulus arrays per block
        actions_blocks: List of action arrays per block
        rewards_blocks: List of reward arrays per block
        set_sizes_blocks: List of set size arrays per block (WM-RL/M3 only)
        model: 'qlearning', 'wmrl', or 'wmrl_m3'
        n_starts: Number of random starting points
        seed: Random seed for reproducibility

    Returns:
        Dictionary with fitted parameters, NLL, AIC, BIC, convergence info
    """
    rng = np.random.default_rng(seed)

    # Count total trials
    n_trials = sum(len(s) for s in stimuli_blocks)

    # Create objective function with data bound
    # Note: Beta is fixed at 50 inside the likelihood functions
    if model == 'qlearning':
        objective = partial(
            _objective_qlearning,
            stimuli_blocks=stimuli_blocks,
            actions_blocks=actions_blocks,
            rewards_blocks=rewards_blocks,
        )
        n_params = 3
    elif model == 'wmrl':
        if set_sizes_blocks is None:
            raise ValueError("set_sizes_blocks required for WM-RL model")
        objective = partial(
            _objective_wmrl,
            stimuli_blocks=stimuli_blocks,
            actions_blocks=actions_blocks,
            rewards_blocks=rewards_blocks,
            set_sizes_blocks=set_sizes_blocks,
        )
        n_params = 6
    elif model == 'wmrl_m3':
        if set_sizes_blocks is None:
            raise ValueError("set_sizes_blocks required for WM-RL M3 model")
        objective = partial(
            _objective_wmrl_m3,
            stimuli_blocks=stimuli_blocks,
            actions_blocks=actions_blocks,
            rewards_blocks=rewards_blocks,
            set_sizes_blocks=set_sizes_blocks,
        )
        n_params = 7
    else:
        raise ValueError(f"Unknown model: {model}")

    # Run optimization from multiple starting points
    results = []
    for i in range(n_starts):
        x0 = sample_random_start(model, rng)

        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'disp': False}
            )
            results.append(result)
        except Exception as e:
            continue  # Skip failed optimizations

    # Check convergence
    convergence_info = check_convergence(results)

    if not results or convergence_info['best_nll'] == np.inf:
        # All optimizations failed
        if model == 'qlearning':
            param_names = QLEARNING_PARAMS
        elif model == 'wmrl':
            param_names = WMRL_PARAMS
        elif model == 'wmrl_m3':
            param_names = WMRL_M3_PARAMS
        else:
            param_names = []
        return {
            **{p: np.nan for p in param_names},
            'nll': np.nan,
            'aic': np.nan,
            'bic': np.nan,
            'aicc': np.nan,
            'n_trials': n_trials,
            'converged': False,
            'n_successful_starts': 0,
            'at_bounds': []
        }

    # Get best result
    best_result = min(results, key=lambda r: r.fun if r.success else np.inf)
    best_params = unconstrained_to_params(best_result.x, model)
    best_nll = best_result.fun

    # Compute information criteria
    k = get_n_params(model)
    aic = compute_aic(best_nll, k)
    bic = compute_bic(best_nll, k, n_trials)
    aicc = compute_aicc(best_nll, k, n_trials)

    # Check if parameters hit bounds
    at_bounds = check_at_bounds(best_params, model)

    return {
        **best_params,
        'nll': best_nll,
        'aic': aic,
        'bic': bic,
        'aicc': aicc,
        'n_trials': n_trials,
        'converged': convergence_info['converged'],
        'n_successful_starts': convergence_info['n_successful'],
        'at_bounds': at_bounds
    }


# =============================================================================
# Multi-Participant Fitting
# =============================================================================

def prepare_participant_data(
    data: pd.DataFrame,
    participant_id: str,
    model: str = 'qlearning'
) -> Dict:
    """
    Prepare data for a single participant.

    Args:
        data: Full DataFrame with all participants
        participant_id: ID of participant to extract
        model: Model type (determines whether to include set_sizes)

    Returns:
        Dictionary with block-organized arrays
    """
    pdata = data[data['sona_id'] == participant_id].copy()

    stimuli_blocks = []
    actions_blocks = []
    rewards_blocks = []
    set_sizes_blocks = []

    for block_num in sorted(pdata['block'].unique()):
        block_data = pdata[pdata['block'] == block_num]

        stimuli_blocks.append(block_data['stimulus'].values.astype(np.int32))
        actions_blocks.append(block_data['key_press'].values.astype(np.int32))
        rewards_blocks.append(block_data['reward'].values.astype(np.float32))

        if model in ('wmrl', 'wmrl_m3') and 'set_size' in block_data.columns:
            set_sizes_blocks.append(block_data['set_size'].values.astype(np.int32))

    result = {
        'stimuli_blocks': stimuli_blocks,
        'actions_blocks': actions_blocks,
        'rewards_blocks': rewards_blocks,
    }

    if model in ('wmrl', 'wmrl_m3'):
        result['set_sizes_blocks'] = set_sizes_blocks

    return result


def fit_all_participants(
    data: pd.DataFrame,
    model: str = 'qlearning',
    n_starts: int = 20,
    seed: int = 42,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Fit all participants using MLE.

    Args:
        data: DataFrame with columns [sona_id, block, stimulus, key_press, reward, set_size]
        model: 'qlearning', 'wmrl', or 'wmrl_m3'
        n_starts: Number of random starting points per participant
        seed: Random seed
        verbose: Show progress bar

    Returns:
        DataFrame with fitted parameters for each participant
    """
    import time

    participants = data['sona_id'].unique()
    n_participants = len(participants)
    results = []

    # Track timing for ETA estimation
    fit_times = []
    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting MLE fitting: {model.upper()}")
        print(f"{'='*60}")
        print(f"Participants: {n_participants}")
        print(f"Random starts per participant: {n_starts}")
        print(f"{'='*60}\n")

    # Use tqdm for progress bar with detailed stats
    iterator = tqdm(
        enumerate(participants),
        total=n_participants,
        desc=f'Fitting {model}',
        unit='participant',
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    ) if verbose else enumerate(participants)

    for i, pid in iterator:
        fit_start = time.time()

        # Prepare data
        pdata = prepare_participant_data(data, pid, model)
        n_trials = sum(len(s) for s in pdata['stimuli_blocks'])

        # Fit
        fit_result = fit_participant_mle(
            stimuli_blocks=pdata['stimuli_blocks'],
            actions_blocks=pdata['actions_blocks'],
            rewards_blocks=pdata['rewards_blocks'],
            set_sizes_blocks=pdata.get('set_sizes_blocks'),
            model=model,
            n_starts=n_starts,
            seed=seed + i,  # Different seed per participant
        )

        # Track timing
        fit_time = time.time() - fit_start
        fit_times.append(fit_time)

        # Add participant ID
        fit_result['participant_id'] = pid

        results.append(fit_result)

        # Update tqdm postfix with current stats
        if verbose and hasattr(iterator, 'set_postfix'):
            avg_time = np.mean(fit_times)
            remaining = (n_participants - i - 1) * avg_time
            iterator.set_postfix({
                'trials': n_trials,
                'NLL': f"{fit_result['nll']:.1f}" if not np.isnan(fit_result['nll']) else 'NaN',
                'conv': '✓' if fit_result['converged'] else '✗',
                'avg': f"{avg_time:.1f}s"
            })

    # Print summary timing
    if verbose:
        total_time = time.time() - start_time
        avg_time = np.mean(fit_times)
        print(f"\n{'='*60}")
        print(f"Fitting Complete!")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
        print(f"Average time per participant: {avg_time:.2f} seconds")
        print(f"Converged: {sum(r['converged'] for r in results)}/{n_participants}")
        print(f"{'='*60}\n")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Reorder columns
    if model == 'qlearning':
        param_cols = QLEARNING_PARAMS
    elif model == 'wmrl':
        param_cols = WMRL_PARAMS
    elif model == 'wmrl_m3':
        param_cols = WMRL_M3_PARAMS
    else:
        param_cols = []

    cols = ['participant_id'] + param_cols
    cols += ['nll', 'aic', 'bic', 'aicc', 'n_trials', 'converged', 'n_successful_starts', 'at_bounds']
    df = df[[c for c in cols if c in df.columns]]

    return df


# =============================================================================
# Group Statistics
# =============================================================================

def compute_group_summary(
    fits_df: pd.DataFrame,
    model: str
) -> pd.DataFrame:
    """
    Compute group-level summary statistics.

    Following Senta et al.: mean +/- SEM for each parameter.

    Args:
        fits_df: DataFrame with individual fit results
        model: 'qlearning' or 'wmrl'

    Returns:
        DataFrame with group statistics
    """
    # Only include converged fits
    converged_df = fits_df[fits_df['converged'] == True]

    if len(converged_df) == 0:
        print("Warning: No converged fits!")
        converged_df = fits_df  # Use all if none converged

    summary = summarize_all_parameters(converged_df, model)

    # Convert to DataFrame format
    rows = []
    for param, stats in summary.items():
        rows.append({
            'parameter': param,
            'mean': stats['mean'],
            'sd': stats['sd'],
            'se': stats['se'],
            'ci_lower': stats['ci_lower'],
            'ci_upper': stats['ci_upper'],
            'n': stats['n']
        })

    # Add fit quality metrics
    rows.append({
        'parameter': 'nll',
        'mean': converged_df['nll'].mean(),
        'sd': converged_df['nll'].std(),
        'se': converged_df['nll'].std() / np.sqrt(len(converged_df)),
        'ci_lower': np.nan,
        'ci_upper': np.nan,
        'n': len(converged_df)
    })

    for metric in ['aic', 'bic']:
        rows.append({
            'parameter': metric,
            'mean': converged_df[metric].mean(),
            'sd': converged_df[metric].std(),
            'se': converged_df[metric].std() / np.sqrt(len(converged_df)),
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'n': len(converged_df)
        })

    return pd.DataFrame(rows)


# =============================================================================
# Main CLI
# =============================================================================

def load_and_prepare_data(filepath: str, exclude_practice: bool = True) -> pd.DataFrame:
    """Load and prepare data for fitting."""
    data = pd.read_csv(filepath)

    # Required columns
    required = ['sona_id', 'block', 'stimulus', 'key_press']
    for col in required:
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    # Create reward column if needed
    if 'reward' not in data.columns and 'correct' in data.columns:
        data['reward'] = data['correct'].astype(float)

    # Exclude practice blocks (blocks 1-2)
    if exclude_practice:
        data = data[data['block'] >= 3].copy()

    # Map key_press if needed (J=0, K=1, L=2)
    if data['key_press'].dtype == object or data['key_press'].max() > 2:
        key_map = {'j': 0, 'k': 1, 'l': 2, 'J': 0, 'K': 1, 'L': 2}
        if data['key_press'].dtype == object:
            data['key_press'] = data['key_press'].map(key_map)

    return data


def main():
    parser = argparse.ArgumentParser(
        description='MLE fitting for RLWM models (Senta et al. methodology)'
    )
    parser.add_argument('--model', type=str, required=True,
                        choices=['qlearning', 'wmrl', 'wmrl_m3'],
                        help='Model to fit (qlearning=M1, wmrl=M2, wmrl_m3=M3)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to trial data CSV')
    parser.add_argument('--output', type=str, default='output/mle/',
                        help='Output directory')
    parser.add_argument('--n-starts', type=int, default=20,
                        help='Number of random starting points (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"MLE Fitting: {args.model.upper()}")
    print(f"{'='*60}")
    print(f"Data: {args.data}")
    print(f"Random starts: {args.n_starts}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    data = load_and_prepare_data(args.data)
    
    # Exclude participants based on data quality
    initial_n = data['sona_id'].nunique()
    data = data[~data['sona_id'].isin(EXCLUDED_PARTICIPANTS)].copy()
    n_participants = data['sona_id'].nunique()
    n_excluded = initial_n - n_participants
    
    n_trials = len(data)
    print(f"  Total participants in data: {initial_n}")
    if n_excluded > 0:
        print(f"  Excluded participants: {n_excluded} (insufficient data/duplicates)")
    print(f"  Final sample: {n_participants}")
    print(f"  Total trials: {n_trials}")
    print()

    # Fit all participants
    fits_df = fit_all_participants(
        data,
        model=args.model,
        n_starts=args.n_starts,
        seed=args.seed,
        verbose=not args.quiet
    )

    # Compute group summary
    summary_df = compute_group_summary(fits_df, args.model)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fits_path = output_dir / f'{args.model}_individual_fits.csv'
    summary_path = output_dir / f'{args.model}_group_summary.csv'

    fits_df.to_csv(fits_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")

    # Print summary
    n_converged = fits_df['converged'].sum()
    print(f"\nConvergence: {n_converged}/{n_participants} ({100*n_converged/n_participants:.1f}%)")

    print(f"\nGroup Statistics (n={n_converged}):")
    print("-" * 50)
    if args.model == 'qlearning':
        param_names = QLEARNING_PARAMS
    elif args.model == 'wmrl':
        param_names = WMRL_PARAMS
    elif args.model == 'wmrl_m3':
        param_names = WMRL_M3_PARAMS
    else:
        param_names = []
    for param in param_names:
        row = summary_df[summary_df['parameter'] == param].iloc[0]
        print(f"  {param:12s}: {row['mean']:.3f} +/- {row['se']:.3f}  (SD={row['sd']:.3f})")

    print(f"\nModel Fit:")
    print("-" * 50)
    for metric in ['nll', 'aic', 'bic']:
        row = summary_df[summary_df['parameter'] == metric].iloc[0]
        print(f"  {metric.upper():12s}: {row['mean']:.2f} +/- {row['se']:.2f}")

    # Check for parameters at bounds
    all_at_bounds = []
    for bounds_list in fits_df['at_bounds']:
        if isinstance(bounds_list, list):
            all_at_bounds.extend(bounds_list)

    if all_at_bounds:
        from collections import Counter
        bounds_counts = Counter(all_at_bounds)
        print(f"\nWarning: Parameters hitting bounds:")
        for param, count in bounds_counts.items():
            print(f"  {param}: {count} participants")

    print(f"\n{'='*60}")
    print(f"Results saved to:")
    print(f"  Individual fits: {fits_path}")
    print(f"  Group summary:   {summary_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
