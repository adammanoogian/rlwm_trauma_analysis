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
import matplotlib.pyplot as plt
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
    from scripts.utils.plotting_utils import plot_scatter_with_annotations

    # Get parameter names
    if model == 'qlearning':
        param_names = QLEARNING_PARAMS
    elif model == 'wmrl':
        param_names = WMRL_PARAMS
    elif model == 'wmrl_m3':
        param_names = WMRL_M3_PARAMS
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
    real_params_path: Optional[str],
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
    from scripts.utils.plotting_utils import plot_kde_comparison

    # Get parameter names
    if model == 'qlearning':
        param_names = QLEARNING_PARAMS
    elif model == 'wmrl':
        param_names = WMRL_PARAMS
    elif model == 'wmrl_m3':
        param_names = WMRL_M3_PARAMS
    else:
        raise ValueError(f"Unknown model: {model}")

    # Try to load real fitted parameters
    if real_params_path is None:
        real_params_path = f'output/mle_results/{model}_individual_fits.csv'

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
                        choices=['qlearning', 'wmrl', 'wmrl_m3'],
                        help='Model to test recovery')
    parser.add_argument('--n-subjects', type=int, default=50,
                        help='Number of synthetic subjects per dataset (default: 50)')
    parser.add_argument('--n-datasets', type=int, default=10,
                        help='Number of independent datasets (default: 10)')
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
        args.fitted_params = f'output/mle_results/{args.model}_individual_fits.csv'

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
            verbose=not args.quiet
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
