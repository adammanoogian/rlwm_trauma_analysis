"""
Maximum Likelihood Estimation for RLWM Models

Following Senta et al. (2025) methodology:
- Individual fits with 20 random starting points
- jaxopt.LBFGS for fast optimization with analytical gradients
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

# Optional memory tracking
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jaxopt import LBFGS
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
    jax_unconstrained_to_params_qlearning,
    jax_unconstrained_to_params_wmrl,
    jax_unconstrained_to_params_wmrl_m3,
    QLEARNING_PARAMS,
    WMRL_PARAMS,
    WMRL_M3_PARAMS,
)

# Suppress JAX warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# =============================================================================
# JAX Compilation Warmup (for parallel execution)
# =============================================================================

def warmup_jax_compilation(model: str, verbose: bool = True):
    """
    Pre-compile JAX functions before spawning parallel workers.

    This populates the JAX persistent cache (JAX_COMPILATION_CACHE_DIR)
    so worker processes can read cached compilations instead of
    each compiling independently. This dramatically reduces overhead
    when using joblib for parallel fitting.

    Args:
        model: 'qlearning', 'wmrl', or 'wmrl_m3'
        verbose: Print warmup status
    """
    if verbose:
        print(f"Warming up JAX compilation for {model}...")

    # Create dummy data matching typical participant shapes
    key = jax.random.PRNGKey(0)
    n_blocks = 7
    n_trials = 50

    stimuli_blocks = []
    actions_blocks = []
    rewards_blocks = []
    set_sizes_blocks = []

    for _ in range(n_blocks):
        key, k1, k2, k3 = jax.random.split(key, 4)
        stimuli_blocks.append(jax.random.randint(k1, (n_trials,), 0, 6))
        actions_blocks.append(jax.random.randint(k2, (n_trials,), 0, 3))
        rewards_blocks.append(jax.random.bernoulli(k3, 0.7, (n_trials,)).astype(jnp.float32))
        set_sizes_blocks.append(jnp.full((n_trials,), 4, dtype=jnp.int32))

    # Call likelihood function to trigger JIT compilation
    if model == 'qlearning':
        q_learning_multiblock_likelihood(
            stimuli_blocks, actions_blocks, rewards_blocks,
            alpha_pos=0.3, alpha_neg=0.1, epsilon=0.05
        )
    elif model == 'wmrl':
        wmrl_multiblock_likelihood(
            stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks,
            alpha_pos=0.3, alpha_neg=0.1, phi=0.1, rho=0.7, capacity=4.0, epsilon=0.05
        )
    elif model == 'wmrl_m3':
        wmrl_m3_multiblock_likelihood(
            stimuli_blocks, actions_blocks, rewards_blocks, set_sizes_blocks,
            alpha_pos=0.3, alpha_neg=0.1, phi=0.1, rho=0.7, capacity=4.0,
            kappa=0.3, epsilon=0.05
        )

    if verbose:
        print(f"  JAX compilation cached for {model}\n")


# =============================================================================
# JAX-Compatible Objective Functions (for jaxopt with automatic differentiation)
# =============================================================================

def _make_jax_objective_qlearning(
    stimuli_blocks: List[jnp.ndarray],
    actions_blocks: List[jnp.ndarray],
    rewards_blocks: List[jnp.ndarray],
):
    """
    Create a JAX-compatible objective function for Q-learning.

    Returns a pure function that takes unconstrained parameters and returns NLL.
    This function is JIT-compilable and supports automatic differentiation.
    """
    def objective(x: jnp.ndarray) -> float:
        alpha_pos, alpha_neg, epsilon = jax_unconstrained_to_params_qlearning(x)
        log_lik = q_learning_multiblock_likelihood(
            stimuli_blocks=stimuli_blocks,
            actions_blocks=actions_blocks,
            rewards_blocks=rewards_blocks,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            epsilon=epsilon
        )
        return -log_lik  # Negative for minimization

    return objective


def _make_jax_objective_wmrl(
    stimuli_blocks: List[jnp.ndarray],
    actions_blocks: List[jnp.ndarray],
    rewards_blocks: List[jnp.ndarray],
    set_sizes_blocks: List[jnp.ndarray],
):
    """
    Create a JAX-compatible objective function for WM-RL.

    Returns a pure function that takes unconstrained parameters and returns NLL.
    """
    def objective(x: jnp.ndarray) -> float:
        alpha_pos, alpha_neg, phi, rho, capacity, epsilon = jax_unconstrained_to_params_wmrl(x)
        log_lik = wmrl_multiblock_likelihood(
            stimuli_blocks=stimuli_blocks,
            actions_blocks=actions_blocks,
            rewards_blocks=rewards_blocks,
            set_sizes_blocks=set_sizes_blocks,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            phi=phi,
            rho=rho,
            capacity=capacity,
            epsilon=epsilon
        )
        return -log_lik

    return objective


def _make_jax_objective_wmrl_m3(
    stimuli_blocks: List[jnp.ndarray],
    actions_blocks: List[jnp.ndarray],
    rewards_blocks: List[jnp.ndarray],
    set_sizes_blocks: List[jnp.ndarray],
):
    """
    Create a JAX-compatible objective function for WM-RL M3.

    Returns a pure function that takes unconstrained parameters and returns NLL.
    """
    def objective(x: jnp.ndarray) -> float:
        alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon = jax_unconstrained_to_params_wmrl_m3(x)
        log_lik = wmrl_m3_multiblock_likelihood(
            stimuli_blocks=stimuli_blocks,
            actions_blocks=actions_blocks,
            rewards_blocks=rewards_blocks,
            set_sizes_blocks=set_sizes_blocks,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
            phi=phi,
            rho=rho,
            capacity=capacity,
            kappa=kappa,
            epsilon=epsilon
        )
        return -log_lik

    return objective


# =============================================================================
# Single Participant Fitting (using jaxopt.LBFGS with analytical gradients)
# =============================================================================

class _JaxoptResult:
    """Simple wrapper to make jaxopt results compatible with convergence checking."""
    def __init__(self, x, fun, success):
        self.x = x
        self.fun = fun
        self.success = success


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

    Uses jaxopt.LBFGS for fast optimization with analytical gradients via JAX autodiff.
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

    # Convert data to JAX arrays for efficient computation
    stimuli_jax = [jnp.array(s, dtype=jnp.int32) for s in stimuli_blocks]
    actions_jax = [jnp.array(a, dtype=jnp.int32) for a in actions_blocks]
    rewards_jax = [jnp.array(r, dtype=jnp.float32) for r in rewards_blocks]

    # Create JAX-compatible objective function
    # Note: Beta is fixed at 50 inside the likelihood functions
    if model == 'qlearning':
        objective = _make_jax_objective_qlearning(
            stimuli_jax, actions_jax, rewards_jax
        )
        n_params = 3
    elif model == 'wmrl':
        if set_sizes_blocks is None:
            raise ValueError("set_sizes_blocks required for WM-RL model")
        set_sizes_jax = [jnp.array(s, dtype=jnp.int32) for s in set_sizes_blocks]
        objective = _make_jax_objective_wmrl(
            stimuli_jax, actions_jax, rewards_jax, set_sizes_jax
        )
        n_params = 6
    elif model == 'wmrl_m3':
        if set_sizes_blocks is None:
            raise ValueError("set_sizes_blocks required for WM-RL M3 model")
        set_sizes_jax = [jnp.array(s, dtype=jnp.int32) for s in set_sizes_blocks]
        objective = _make_jax_objective_wmrl_m3(
            stimuli_jax, actions_jax, rewards_jax, set_sizes_jax
        )
        n_params = 7
    else:
        raise ValueError(f"Unknown model: {model}")

    # Create jaxopt LBFGS solver with JIT-compiled objective
    # jaxopt computes gradients automatically via JAX autodiff
    solver = LBFGS(fun=objective, maxiter=1000, tol=1e-6)

    # Run optimization from multiple starting points
    results = []
    for i in range(n_starts):
        x0 = jnp.array(sample_random_start(model, rng))

        try:
            # jaxopt returns (params, state) tuple
            result_params, state = solver.run(x0)

            # Extract final objective value
            final_nll = float(objective(result_params))

            # Check if optimization succeeded (no NaN/inf)
            success = jnp.isfinite(final_nll)

            # Wrap in result object for compatibility with check_convergence
            result = _JaxoptResult(
                x=np.array(result_params),
                fun=final_nll,
                success=bool(success)
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


def _fit_single_participant_worker(args: Tuple) -> Dict:
    """
    Picklable worker function for parallel fitting.

    This wrapper is needed because joblib requires picklable functions.
    The function receives all arguments as a tuple and unpacks them.

    Args:
        args: Tuple of (pid, data_dict, model, n_starts, seed)

    Returns:
        Fit result dictionary with participant_id added
    """
    pid, data_dict, model, n_starts, seed = args

    result = fit_participant_mle(
        stimuli_blocks=data_dict['stimuli_blocks'],
        actions_blocks=data_dict['actions_blocks'],
        rewards_blocks=data_dict['rewards_blocks'],
        set_sizes_blocks=data_dict.get('set_sizes_blocks'),
        model=model,
        n_starts=n_starts,
        seed=seed,
    )
    result['participant_id'] = pid
    return result


def fit_all_participants(
    data: pd.DataFrame,
    model: str = 'qlearning',
    n_starts: int = 20,
    seed: int = 42,
    verbose: bool = True,
    n_jobs: int = 1
) -> Tuple[pd.DataFrame, Dict]:
    """
    Fit all participants using MLE.

    Args:
        data: DataFrame with columns [sona_id, block, stimulus, key_press, reward, set_size]
        model: 'qlearning', 'wmrl', or 'wmrl_m3'
        n_starts: Number of random starting points per participant
        seed: Random seed
        verbose: Show progress bar
        n_jobs: Number of parallel workers (1 = sequential, -1 = all cores)

    Returns:
        Tuple of (DataFrame with fitted parameters, timing_info dict)
    """
    import time
    from joblib import Parallel, delayed

    participants = data['sona_id'].unique()
    n_participants = len(participants)

    # Track timing for ETA estimation
    start_time = time.time()

    if verbose:
        print(f"\n{'='*60}")
        print(f"Starting MLE fitting: {model.upper()}")
        print(f"{'='*60}")
        print(f"Participants: {n_participants}")
        print(f"Random starts per participant: {n_starts}")
        print(f"Parallel workers: {n_jobs if n_jobs > 0 else 'all available cores'}")
        print(f"{'='*60}\n")

    # Prepare all participant data upfront (needed for parallel execution)
    if verbose:
        print("Preparing participant data...")
    participant_args = []
    for i, pid in enumerate(participants):
        pdata = prepare_participant_data(data, pid, model)
        participant_args.append((pid, pdata, model, n_starts, seed + i))

    # Execute fitting
    if n_jobs == 1:
        # Sequential execution (existing behavior with tqdm progress)
        results = []
        fit_times = []

        iterator = tqdm(
            participant_args,
            total=n_participants,
            desc=f'Fitting {model}',
            unit='participant',
            ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        ) if verbose else participant_args

        for args in iterator:
            fit_start = time.time()
            result = _fit_single_participant_worker(args)
            fit_time = time.time() - fit_start
            fit_times.append(fit_time)
            results.append(result)

            # Update tqdm postfix with current stats
            if verbose and hasattr(iterator, 'set_postfix'):
                n_trials = sum(len(s) for s in args[1]['stimuli_blocks'])
                avg_time = np.mean(fit_times)
                iterator.set_postfix({
                    'trials': n_trials,
                    'NLL': f"{result['nll']:.1f}" if not np.isnan(result['nll']) else 'NaN',
                    'conv': '✓' if result['converged'] else '✗',
                    'avg': f"{avg_time:.1f}s"
                })
    else:
        # Parallel execution using joblib

        # Warmup JAX compilation in main process (populates disk cache)
        # This ensures workers can read cached compilations instead of
        # each independently compiling, reducing overhead from ~30s to ~5s
        warmup_jax_compilation(model, verbose=verbose)

        if verbose:
            print(f"Running parallel fitting with {n_jobs} workers...")

        # joblib handles the parallel execution
        # verbose=10 shows progress; verbose=0 is silent
        results = Parallel(
            n_jobs=n_jobs,
            verbose=10 if verbose else 0,
            backend='loky'  # Process-based backend for true parallelism
        )(
            delayed(_fit_single_participant_worker)(args)
            for args in participant_args
        )

        # For parallel execution, we can't track individual times
        # Use total time / n_participants as estimate
        fit_times = []

    # Calculate timing (always, for timing_info return)
    total_time = time.time() - start_time

    # Print summary timing
    if verbose:
        print(f"\n{'='*60}")
        print(f"Fitting Complete!")
        print(f"{'='*60}")
        print(f"Total time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
        if fit_times:
            avg_time = np.mean(fit_times)
            print(f"Average time per participant: {avg_time:.2f} seconds")
        else:
            print(f"Average time per participant: {total_time/n_participants:.2f} seconds (parallel)")
        print(f"Converged: {sum(r['converged'] for r in results)}/{n_participants}")
        if n_jobs != 1:
            speedup = (total_time / n_participants * n_participants) / total_time
            print(f"Effective parallelization: {n_jobs} workers")
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

    # Build timing info dict for extrapolation
    timing_info = {
        'total_time': total_time,
        'fit_times': fit_times,
        'jit_time': fit_times[0] if fit_times else None,
        'steady_state_times': fit_times[1:] if len(fit_times) > 1 else [],
        'n_jobs': n_jobs,
    }

    return df, timing_info


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
    parser.add_argument('--include-practice', action='store_true',
                        help='Include practice blocks (1-2) in fitting. Default excludes practice.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit to first N participants (for timing/testing)')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Number of parallel workers (default: 1 = sequential)')
    parser.add_argument('--use-gpu', action='store_true',
                        help='Use GPU acceleration if available (requires JAX CUDA)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # GPU detection
    gpu_available = False
    gpu_devices = []
    if args.use_gpu:
        try:
            devices = jax.devices()
            gpu_devices = [d for d in devices if d.platform == 'gpu']
            gpu_available = len(gpu_devices) > 0
            if gpu_available:
                # JAX will automatically use GPU when available
                print(f"GPU detected: {gpu_devices}")
            else:
                print("WARNING: --use-gpu specified but no GPU found. Using CPU.")
                print(f"Available devices: {devices}")
        except Exception as e:
            print(f"WARNING: GPU detection failed: {e}")
            print("Falling back to CPU.")

    print(f"\n{'='*60}")
    print(f"MLE Fitting: {args.model.upper()}")
    print(f"{'='*60}")
    print(f"Data: {args.data}")
    print(f"Random starts: {args.n_starts}")
    print(f"Seed: {args.seed}")
    print(f"Output: {args.output}")
    print(f"Include practice: {args.include_practice}")
    print(f"Parallel workers: {args.n_jobs if args.n_jobs > 0 else 'all available cores'}")
    if args.use_gpu:
        print(f"GPU acceleration: {'enabled' if gpu_available else 'requested but unavailable'}")
    if args.limit:
        print(f"Limit: {args.limit} participants (testing mode)")
    print(f"{'='*60}\n")

    # Load data
    print("Loading data...")
    exclude_practice = not args.include_practice
    data = load_and_prepare_data(args.data, exclude_practice=exclude_practice)
    
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

    # Apply limit filter if specified
    full_participant_count = n_participants
    if args.limit is not None:
        unique_pids = data['sona_id'].unique()[:args.limit]
        data = data[data['sona_id'].isin(unique_pids)].copy()
        n_participants = len(unique_pids)
        print(f"  Limited to first {args.limit} participants (testing mode)")
    print()

    # Fit all participants
    fits_df, timing_info = fit_all_participants(
        data,
        model=args.model,
        n_starts=args.n_starts,
        seed=args.seed,
        verbose=not args.quiet,
        n_jobs=args.n_jobs
    )

    # Compute group summary
    summary_df = compute_group_summary(fits_df, args.model)

    # Timing analysis and extrapolation (when using --limit)
    if timing_info and args.limit:
        print(f"\n{'='*60}")
        print("TIMING ANALYSIS & EXTRAPOLATION")
        print(f"{'='*60}")

        jit_time = timing_info['jit_time']
        steady_times = timing_info['steady_state_times']

        if jit_time is not None:
            print(f"JIT compilation (1st participant): {jit_time:.1f}s")

        if steady_times:
            steady_mean = np.mean(steady_times)
            steady_std = np.std(steady_times)
            print(f"Steady-state per participant: {steady_mean:.1f}s ± {steady_std:.1f}s")

            # Extrapolate to full dataset
            total_n = full_participant_count
            estimated = jit_time + (total_n - 1) * steady_mean
            print(f"\nExtrapolated time for {total_n} participants: {estimated/60:.1f} min")
        elif jit_time is not None:
            # Only 1 participant tested
            total_n = full_participant_count
            # Conservative estimate: assume JIT time includes ~50% overhead
            steady_estimate = jit_time * 0.6
            estimated = jit_time + (total_n - 1) * steady_estimate
            print(f"\nNote: Only 1 participant tested (includes JIT overhead)")
            print(f"Rough estimate for {total_n} participants: {estimated/60:.1f} min")
            print(f"(Run with --limit 3 for more accurate extrapolation)")

        # Memory usage if psutil available
        if HAS_PSUTIL:
            process = psutil.Process()
            mem_mb = process.memory_info().rss / 1024 / 1024
            print(f"\nPeak memory usage: {mem_mb:.0f} MB")

        print(f"{'='*60}")

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
