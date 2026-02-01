"""
Prior-Based Parameter Space Exploration

Sample parameters from prior distributions and explore how different
parameter combinations affect model behavior. Creates comprehensive
heatmap visualizations for all parameter pairs.

Requirements:
    pip install tqdm  # For progress bars

Defaults (from actual experimental task):
    - Set sizes: [2, 3, 5, 6] (all set sizes from actual task)
    - Trials: 45 (median from actual task; range: 30-90)
    - Reversals: 12-18 consecutive correct (from config)

Usage:
    # Serial execution with defaults (1 CPU, uses actual task parameters)
    python tests/examples/explore_prior_parameter_space.py --model qlearning --n-samples 200

    # Parallel execution (use all CPUs, RECOMMENDED)
    python tests/examples/explore_prior_parameter_space.py --model both --n-samples 200 --n-jobs -1

    # Parallel with specific number of CPUs
    python tests/examples/explore_prior_parameter_space.py --model wmrl --n-samples 300 --n-jobs 4

    # Quick test run (fewer samples, fewer trials)
    python tests/examples/explore_prior_parameter_space.py --model qlearning --n-samples 20 --num-trials 30 --n-jobs -1

    # Match exact task structure (all 4 set sizes)
    python tests/examples/explore_prior_parameter_space.py --model both --n-samples 200 --set-sizes 2 3 5 6 --n-jobs -1
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import argparse
from itertools import combinations
from multiprocessing import Pool, cpu_count
from functools import partial
import time

try:
    from tqdm.auto import tqdm
except ImportError:
    print("WARNING: tqdm not installed. Progress bars will not be shown.")
    print("Install with: pip install tqdm")
    # Fallback to no progress bar
    def tqdm(iterable, **kwargs):
        return iterable

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from environments.rlwm_env import create_rlwm_env
from models.q_learning import QLearningAgent
from models.wm_rl_hybrid import WMRLHybridAgent
from scripts.simulations.unified_simulator import simulate_agent_fixed
from scripts.analysis.plotting_utils import setup_plot_style, save_figure, get_color_palette
from config import TaskParams


# ============================================================================
# PRIOR DISTRIBUTIONS (matching PyMC priors)
# ============================================================================

def sample_qlearning_params(n_samples: int, seed: int = None) -> pd.DataFrame:
    """
    Sample Q-learning parameters from prior distributions.

    Parameters
    ----------
    n_samples : int
        Number of parameter sets to sample
    seed : int, optional
        Random seed

    Returns
    -------
    pd.DataFrame
        Sampled parameters with columns: alpha_pos, alpha_neg, beta
    """
    if seed is not None:
        np.random.seed(seed)

    params = pd.DataFrame({
        'alpha_pos': np.random.beta(2, 2, n_samples),      # Beta(2, 2) ~ uniform-ish on [0,1]
        'alpha_neg': np.random.beta(2, 2, n_samples),      # Beta(2, 2)
        'beta': np.random.gamma(2, 1, n_samples),          # Gamma(2, 1) ~ mean=2, allows high values
    })

    return params


def sample_wmrl_params(n_samples: int, seed: int = None) -> pd.DataFrame:
    """
    Sample WM-RL parameters from prior distributions.

    Parameters
    ----------
    n_samples : int
        Number of parameter sets to sample
    seed : int, optional
        Random seed

    Returns
    -------
    pd.DataFrame
        Sampled parameters with columns: alpha_pos, alpha_neg, beta, beta_wm,
        capacity, phi, rho
    """
    if seed is not None:
        np.random.seed(seed)

    params = pd.DataFrame({
        'alpha_pos': np.random.beta(2, 2, n_samples),
        'alpha_neg': np.random.beta(2, 2, n_samples),
        'beta': np.random.gamma(2, 1, n_samples),
        'beta_wm': np.random.gamma(2, 1, n_samples),
        'capacity': np.random.randint(2, 7, n_samples),    # DiscreteUniform(2, 6)
        'phi': np.random.beta(2, 2, n_samples),            # WM decay rate
        'rho': np.random.beta(2, 2, n_samples),            # Base WM reliance
    })

    return params


# ============================================================================
# SIMULATION (with parallelization support)
# ============================================================================

def _simulate_single_param_set(
    args: Tuple[int, pd.Series, str, List[int], int, int, int]
) -> List[Dict]:
    """
    Worker function to simulate a single parameter set across conditions.

    This function is designed to be pickled for multiprocessing.

    Parameters
    ----------
    args : Tuple
        (idx, param_row, model_type, set_sizes, num_trials, num_reps, seed)

    Returns
    -------
    List[Dict]
        Results for this parameter set across all set sizes
    """
    idx, row, model_type, set_sizes, num_trials, num_reps, seed = args

    agent_class = QLearningAgent if model_type == 'qlearning' else WMRLHybridAgent
    results = []

    for set_size in set_sizes:
        accuracies = []

        for rep in range(num_reps):
            # Create environment
            env = create_rlwm_env(
                set_size=set_size,
                phase_type='main_task',
                max_trials_per_block=num_trials,
                seed=seed + idx * 1000 + rep
            )

            # Prepare parameters
            params = {
                'num_stimuli': 6,
                'num_actions': 3,
                'gamma': 0.0,
                'q_init': 0.5,
            }

            # Add model-specific parameters
            if model_type == 'qlearning':
                params.update({
                    'alpha_pos': row['alpha_pos'],
                    'alpha_neg': row['alpha_neg'],
                    'beta': row['beta'],
                })
            else:  # wmrl
                params.update({
                    'alpha_pos': row['alpha_pos'],
                    'alpha_neg': row['alpha_neg'],
                    'beta': row['beta'],
                    'beta_wm': row['beta_wm'],
                    'capacity': int(row['capacity']),
                    'phi': row['phi'],
                    'rho': row['rho'],
                    'wm_init': 0.0,
                })

            # Run simulation
            result = simulate_agent_fixed(
                agent_class=agent_class,
                params=params,
                env=env,
                num_trials=num_trials,
                seed=seed + idx * 1000 + rep
            )

            accuracies.append(result.accuracy)

        # Store results
        result_row = row.to_dict()
        result_row.update({
            'set_size': set_size,
            'accuracy_mean': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'num_trials': num_trials,
            'num_reps': num_reps,
        })
        results.append(result_row)

    return results


def run_simulations_with_samples(
    model_type: str,
    param_samples: pd.DataFrame,
    set_sizes: List[int] = [3, 5],
    num_trials: int = 50,
    num_reps: int = 3,
    seed: int = 42,
    n_jobs: int = 1
) -> pd.DataFrame:
    """
    Run simulations for each sampled parameter set (parallelized).

    Parameters
    ----------
    model_type : str
        'qlearning' or 'wmrl'
    param_samples : pd.DataFrame
        Sampled parameters
    set_sizes : List[int]
        Set sizes to test
    num_trials : int
        Trials per simulation
    num_reps : int
        Repetitions per condition
    seed : int
        Base random seed
    n_jobs : int
        Number of parallel jobs (1 = serial, -1 = all CPUs)

    Returns
    -------
    pd.DataFrame
        Results with all parameters and accuracy metrics
    """
    n_samples = len(param_samples)

    # Determine number of jobs
    if n_jobs == -1:
        n_jobs = cpu_count()
    elif n_jobs < 1:
        n_jobs = 1

    print(f"  Using {n_jobs} parallel job(s)")

    # Prepare arguments for each parameter set
    job_args = [
        (idx, row, model_type, set_sizes, num_trials, num_reps, seed)
        for idx, row in param_samples.iterrows()
    ]

    # Run simulations with progress bar
    if n_jobs == 1:
        # Serial execution with progress bar
        all_results = []
        for args in tqdm(job_args, desc="  Simulating", unit="param_set"):
            all_results.extend(_simulate_single_param_set(args))
    else:
        # Parallel execution with progress bar
        print(f"  Starting parallel simulations...")
        with Pool(processes=n_jobs) as pool:
            # Use imap_unordered for progress tracking
            results_nested = list(tqdm(
                pool.imap_unordered(_simulate_single_param_set, job_args),
                total=len(job_args),
                desc="  Simulating",
                unit="param_set"
            ))

        # Flatten results
        all_results = [item for sublist in results_nested for item in sublist]

    return pd.DataFrame(all_results)


# ============================================================================
# VISUALIZATION: COMPREHENSIVE HEATMAPS
# ============================================================================

def create_pairwise_heatmaps(
    results: pd.DataFrame,
    model_type: str,
    param_pairs: List[Tuple[str, str]] = None,
    save_dir: Path = None
) -> None:
    """
    Create heatmaps for all pairwise parameter combinations.

    Parameters
    ----------
    results : pd.DataFrame
        Simulation results with parameters and accuracy
    model_type : str
        'qlearning' or 'wmrl'
    param_pairs : List[Tuple[str, str]], optional
        Specific parameter pairs to plot. If None, plots all important pairs
    save_dir : Path, optional
        Directory to save figures
    """
    setup_plot_style()

    # Define parameter pairs to visualize
    if param_pairs is None:
        if model_type == 'qlearning':
            param_pairs = [
                ('alpha_pos', 'alpha_neg'),
                ('alpha_pos', 'beta'),
                ('alpha_neg', 'beta'),
            ]
        else:  # wmrl
            param_pairs = [
                ('alpha_pos', 'alpha_neg'),
                ('alpha_pos', 'beta'),
                ('capacity', 'rho'),
                ('capacity', 'phi'),
                ('rho', 'phi'),
                ('beta', 'beta_wm'),
                ('alpha_pos', 'capacity'),
                ('beta', 'rho'),
            ]

    n_pairs = len(param_pairs)
    n_cols = 3
    n_rows = int(np.ceil(n_pairs / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_pairs > 1 else [axes]

    # Average across set sizes for overall view
    results_avg = results.groupby([col for col in results.columns if col not in [
        'set_size', 'accuracy_mean', 'accuracy_std', 'num_trials', 'num_reps'
    ]])['accuracy_mean'].mean().reset_index()

    for idx, (param_x, param_y) in enumerate(param_pairs):
        ax = axes[idx]

        # Create bins for continuous parameters
        if param_x == 'capacity':
            param_x_plot = param_x
        else:
            results_avg[f'{param_x}_binned'] = pd.cut(results_avg[param_x], bins=10)
            param_x_plot = f'{param_x}_binned'

        if param_y == 'capacity':
            param_y_plot = param_y
        else:
            results_avg[f'{param_y}_binned'] = pd.cut(results_avg[param_y], bins=10)
            param_y_plot = f'{param_y}_binned'

        # Create pivot table
        try:
            pivot = results_avg.pivot_table(
                index=param_y_plot,
                columns=param_x_plot,
                values='accuracy_mean',
                aggfunc='mean'
            )

            # Plot heatmap
            sns.heatmap(
                pivot,
                ax=ax,
                cmap='RdYlGn',
                vmin=0.3,
                vmax=1.0,
                cbar_kws={'label': 'Accuracy'},
                annot=False,
                fmt='.2f'
            )

            # Labels
            param_labels = {
                'alpha_pos': 'Alpha+ (Pos PE LR)',
                'alpha_neg': 'Alpha- (Neg PE LR)',
                'beta': 'Beta (Inv Temp)',
                'beta_wm': 'Beta WM (WM Inv Temp)',
                'capacity': 'Capacity (K)',
                'phi': 'Phi (Decay)',
                'rho': 'Rho (WM Reliance)',
            }

            ax.set_xlabel(param_labels.get(param_x, param_x), fontweight='bold')
            ax.set_ylabel(param_labels.get(param_y, param_y), fontweight='bold')
            ax.set_title(f'{param_labels.get(param_x, param_x)} × {param_labels.get(param_y, param_y)}',
                        fontweight='bold')

            # Simplify tick labels
            if param_x != 'capacity':
                ax.set_xticklabels([])
            if param_y != 'capacity':
                ax.set_yticklabels([])

        except Exception as e:
            ax.text(0.5, 0.5, f'Error:\n{str(e)}', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(f'{param_x} × {param_y} (Error)', fontweight='bold')

    # Remove extra subplots
    for idx in range(n_pairs, len(axes)):
        fig.delaxes(axes[idx])

    model_name = 'Q-Learning' if model_type == 'qlearning' else 'WM-RL'
    fig.suptitle(f'{model_name}: Parameter Space Exploration (Prior Sampling)',
                fontsize=16, fontweight='bold', y=0.995)
    fig.tight_layout()

    if save_dir:
        save_figure(fig, f'{model_type}_prior_parameter_heatmaps', subdir='parameter_exploration')

    plt.close(fig)


def create_marginal_distributions(
    results: pd.DataFrame,
    model_type: str,
    save_dir: Path = None
) -> None:
    """
    Plot marginal effects of each parameter on accuracy with separate lines per set size.

    Parameters
    ----------
    results : pd.DataFrame
        Simulation results
    model_type : str
        'qlearning' or 'wmrl'
    save_dir : Path, optional
        Directory to save figure
    """
    setup_plot_style()

    # Get parameter columns
    if model_type == 'qlearning':
        params = ['alpha_pos', 'alpha_neg', 'beta']
    else:
        params = ['alpha_pos', 'alpha_neg', 'beta', 'beta_wm', 'capacity', 'phi', 'rho']

    n_params = len(params)
    n_cols = 3
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    param_labels = {
        'alpha_pos': 'Alpha+ (Positive PE LR)',
        'alpha_neg': 'Alpha- (Negative PE LR)',
        'beta': 'Beta (Inverse Temperature)',
        'beta_wm': 'Beta WM (WM Inverse Temp)',
        'capacity': 'Capacity (K)',
        'phi': 'Phi (WM Decay Rate)',
        'rho': 'Rho (Base WM Reliance)',
    }

    # Get set sizes and colors
    set_sizes = sorted(results['set_size'].unique())
    colors = get_color_palette('set_size')

    for idx, param in enumerate(params):
        ax = axes[idx]

        # Plot separate line for each set size
        for ss in set_sizes:
            ss_data = results[results['set_size'] == ss].copy()

            # Bin parameter values
            if param == 'capacity':
                grouped = ss_data.groupby('capacity')['accuracy_mean'].agg(['mean', 'std'])
                ax.plot(grouped.index, grouped['mean'], 'o-',
                       linewidth=2, markersize=6, color=colors[ss],
                       label=f'Set Size {ss}', alpha=0.8)
                ax.fill_between(
                    grouped.index,
                    grouped['mean'] - grouped['std'],
                    grouped['mean'] + grouped['std'],
                    alpha=0.2,
                    color=colors[ss]
                )
            else:
                ss_data[f'{param}_binned'] = pd.cut(ss_data[param], bins=15)
                grouped = ss_data.groupby(f'{param}_binned', observed=False)['accuracy_mean'].agg(['mean', 'std', 'count'])

                # Filter out bins with no data
                grouped = grouped[grouped['count'] > 0]

                if len(grouped) > 0:
                    bin_centers = [iv.mid for iv in grouped.index]

                    ax.plot(bin_centers, grouped['mean'], 'o-',
                           linewidth=2, markersize=4, color=colors[ss],
                           label=f'Set Size {ss}', alpha=0.8)
                    ax.fill_between(
                        bin_centers,
                        grouped['mean'] - grouped['std'],
                        grouped['mean'] + grouped['std'],
                        alpha=0.2,
                        color=colors[ss]
                    )

        ax.set_xlabel(param_labels.get(param, param), fontweight='bold')
        ax.set_ylabel('Accuracy', fontweight='bold')
        ax.set_title(f'Effect of {param_labels.get(param, param)}', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1/3, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=8, loc='best')

    # Remove extra subplots
    for idx in range(n_params, len(axes)):
        fig.delaxes(axes[idx])

    model_name = 'Q-Learning' if model_type == 'qlearning' else 'WM-RL'
    fig.suptitle(f'{model_name}: Marginal Parameter Effects by Set Size (Prior Sampling)',
                fontsize=16, fontweight='bold', y=0.995)
    fig.tight_layout()

    if save_dir:
        save_figure(fig, f'{model_type}_prior_marginal_effects', subdir='parameter_exploration')

    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Start total timer
    total_start_time = time.time()

    parser = argparse.ArgumentParser(description='Explore parameter space using prior sampling')
    parser.add_argument('--model', type=str, choices=['qlearning', 'wmrl', 'both'],
                       default='both', help='Which model to explore')
    parser.add_argument('--n-samples', type=int, default=200,
                       help='Number of parameter sets to sample from priors')
    parser.add_argument('--set-sizes', type=int, nargs='+', default=TaskParams.SET_SIZES,
                       help=f'Set sizes to test (default: {TaskParams.SET_SIZES} from actual task)')
    parser.add_argument('--num-trials', type=int, default=TaskParams.TRIALS_PER_BLOCK_MEDIAN,
                       help=f'Trials per simulation (default: {TaskParams.TRIALS_PER_BLOCK_MEDIAN}, median from actual task)')
    parser.add_argument('--num-reps', type=int, default=3,
                       help='Repetitions per condition for averaging')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel jobs (1=serial, -1=all CPUs)')

    args = parser.parse_args()

    print("=" * 80)
    print("PRIOR-BASED PARAMETER SPACE EXPLORATION")
    print("=" * 80)
    print()
    print(f"Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Prior samples: {args.n_samples}")
    print(f"  Set sizes: {args.set_sizes}")
    print(f"  Trials per simulation: {args.num_trials}")
    print(f"  Repetitions: {args.num_reps}")
    print(f"  Parallel jobs: {args.n_jobs} ({'all CPUs' if args.n_jobs == -1 else f'{args.n_jobs} job(s)'})")
    print(f"  Random seed: {args.seed}")
    print()

    # Output directory
    output_dir = project_root / 'output' / 'parameter_exploration'
    output_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = ['qlearning', 'wmrl'] if args.model == 'both' else [args.model]

    for model in models_to_run:
        print("-" * 80)
        print(f"EXPLORING {model.upper()} PARAMETER SPACE")
        print("-" * 80)
        print()

        # Sample parameters from priors
        print(f"Sampling {args.n_samples} parameter sets from prior distributions...")
        if model == 'qlearning':
            param_samples = sample_qlearning_params(args.n_samples, seed=args.seed)
            print(f"  Sampled: alpha_pos, alpha_neg, beta")
        else:
            param_samples = sample_wmrl_params(args.n_samples, seed=args.seed)
            print(f"  Sampled: alpha_pos, alpha_neg, beta, beta_wm, capacity, phi, rho")
        print()

        # Show sample statistics
        print("Prior sample statistics:")
        print(param_samples.describe())
        print()

        # Run simulations with timing
        print(f"Running simulations for {args.n_samples} parameter sets...")
        start_time = time.time()

        results = run_simulations_with_samples(
            model_type=model,
            param_samples=param_samples,
            set_sizes=args.set_sizes,
            num_trials=args.num_trials,
            num_reps=args.num_reps,
            seed=args.seed,
            n_jobs=args.n_jobs
        )

        elapsed_time = time.time() - start_time
        print(f"  Complete! Elapsed time: {elapsed_time/60:.2f} minutes ({elapsed_time:.1f} seconds)")
        print(f"  Average time per parameter set: {elapsed_time/args.n_samples:.2f} seconds")
        print()

        # Save results
        output_file = output_dir / f'{model}_prior_exploration_n{args.n_samples}_seed{args.seed}.csv'
        results.to_csv(output_file, index=False)
        print(f"Saved results: {output_file}")
        print()

        # Create visualizations
        print("Creating visualizations...")
        print("  1. Pairwise parameter heatmaps...")
        create_pairwise_heatmaps(results, model, save_dir=output_dir)
        print(f"     ✓ Saved: figures/parameter_exploration/{model}_prior_parameter_heatmaps.png")

        print("  2. Marginal parameter effects...")
        create_marginal_distributions(results, model, save_dir=output_dir)
        print(f"     ✓ Saved: figures/parameter_exploration/{model}_prior_marginal_effects.png")
        print()

        # Summary statistics
        print("Performance summary:")
        print(f"  Mean accuracy: {results['accuracy_mean'].mean():.3f} ± {results['accuracy_mean'].std():.3f}")
        print(f"  Max accuracy: {results['accuracy_mean'].max():.3f}")
        print(f"  Min accuracy: {results['accuracy_mean'].min():.3f}")
        print()

        # Best parameters
        best_idx = results['accuracy_mean'].idxmax()
        best_row = results.loc[best_idx]
        print("Best parameter combination found:")
        for param in param_samples.columns:
            print(f"  {param} = {best_row[param]:.3f}")
        print(f"  → Accuracy = {best_row['accuracy_mean']:.3f}")
        print()

    # Total time
    total_elapsed_time = time.time() - total_start_time

    print("=" * 80)
    print("EXPLORATION COMPLETE!")
    print("=" * 80)
    print()
    print(f"Results saved to: {output_dir}")
    print(f"Figures saved to: figures/parameter_exploration/")
    print()
    print(f"Total runtime: {total_elapsed_time/60:.2f} minutes ({total_elapsed_time:.1f} seconds)")
    print()


if __name__ == "__main__":
    main()
