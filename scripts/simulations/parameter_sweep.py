"""
Parameter Space Exploration for RLWM Models

Tools for running parameter sweeps to understand:
- How parameters affect performance
- Interaction effects between parameters and task variables (set size)
- Model predictions across parameter ranges

Usage:
    # Q-learning sweep
    python simulations/parameter_sweep.py --model qlearning --num-trials 100 --num-reps 5

    # WM-RL sweep
    python simulations/parameter_sweep.py --model wmrl --num-trials 50 --num-reps 3

    # Custom parameter ranges (in script or Python)
    from simulations.parameter_sweep import ParameterSweep
    sweep = ParameterSweep(model_type='qlearning')
    results = sweep.sweep_qlearning_parameters(
        alpha_range=[0.1, 0.3, 0.5],
        beta_range=[1, 3, 5]
    )
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Literal
import itertools
from tqdm import tqdm
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
# Ensure `src/` is importable (so `import rlwm` works without installation).
src_root = project_root / "src"
if src_root.exists():
    sys.path.insert(0, str(src_root))

from rlwm.envs.rlwm_env import create_rlwm_env
from rlwm.models.q_learning import QLearningAgent
from rlwm.models.wm_rl_hybrid import WMRLHybridAgent
from scripts.simulations.unified_simulator import simulate_agent_fixed
try:
    # Prefer the versioned output root if present in this project.
    from config import OUTPUT_VERSION_DIR  # type: ignore
except Exception:  # pragma: no cover
    OUTPUT_VERSION_DIR = None


class ParameterSweep:
    """Run systematic parameter sweeps for RLWM models."""

    def __init__(
        self,
        model_type: str = 'qlearning',
        output_dir: Optional[Path] = None,
        seed: int = 42
    ):
        """
        Initialize parameter sweep.

        Parameters
        ----------
        model_type : str
            'qlearning' or 'wmrl'
        output_dir : Path, optional
            Directory to save results. If None, uses OUTPUT_VERSION_DIR/parameter_sweeps
        seed : int
            Random seed
        """
        self.model_type = model_type
        if output_dir is None:
            if OUTPUT_VERSION_DIR is not None:
                output_dir = Path(OUTPUT_VERSION_DIR) / 'parameter_sweeps'
            else:
                output_dir = project_root / 'output' / 'parameter_sweeps'
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        # Keep an internal accumulator for convenience, but sweep methods return fresh DataFrames.
        self.results: List[Dict[str, float]] = []

    def _agent_class(self):
        if self.model_type == 'qlearning':
            return QLearningAgent
        if self.model_type == 'wmrl':
            return WMRLHybridAgent
        raise ValueError(f"Unknown model type: {self.model_type}")

    def run_single_condition(
        self,
        params: Dict[str, float],
        set_size: int,
        num_trials: int = 100,
        num_reps: int = 5
    ) -> Dict[str, float]:
        """
        Run model with specific parameters and task configuration.

        Parameters
        ----------
        params : dict
            Model parameters
        set_size : int
            Task set size (2, 3, 5, or 6)
        num_trials : int
            Trials per run
        num_reps : int
            Number of repetitions to average

        Returns
        -------
        dict
            Average performance metrics
        """
        metrics_list = []

        for rep in range(num_reps):
            # Create environment
            env = create_rlwm_env(
                set_size=set_size,
                phase_type='main_task',
                max_trials_per_block=num_trials,
                seed=self.seed + rep
            )

            agent_class = self._agent_class()

            # Run simulation using unified simulator
            result = simulate_agent_fixed(
                agent_class=agent_class,
                params=params,
                env=env,
                num_trials=num_trials,
                seed=self.seed + rep
            )

            metrics_list.append({
                'accuracy': result.accuracy,
                'total_reward': np.sum(result.rewards),
                'num_trials': len(result.stimuli)
            })

        # Average across reps
        avg_metrics = {
            'accuracy': np.mean([m['accuracy'] for m in metrics_list]),
            'accuracy_std': np.std([m['accuracy'] for m in metrics_list]),
            'total_reward': np.mean([m['total_reward'] for m in metrics_list]),
        }

        return avg_metrics

    def sweep_qlearning_parameters(
        self,
        alpha_pos_range: List[float] = [0.1, 0.2, 0.3, 0.5, 0.7],
        alpha_neg_range: List[float] = [0.05, 0.1, 0.2],
        beta_range: List[float] = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
        set_sizes: List[int] = [2, 3, 5, 6],
        num_trials: int = 100,
        num_reps: int = 5
    ) -> pd.DataFrame:
        """
        Sweep Q-learning parameters across task conditions.

        Parameters
        ----------
        alpha_pos_range : list
            Learning rates for positive PE to test
        alpha_neg_range : list
            Learning rates for negative PE to test
        beta_range : list
            Inverse temperatures to test
        set_sizes : list
            Set sizes to test
        num_trials : int
            Trials per condition
        num_reps : int
            Repetitions per condition

        Returns
        -------
        pd.DataFrame
            Results dataframe
        """
        print(f"Running Q-learning parameter sweep")
        print(f"  Alpha+ values: {len(alpha_pos_range)}")
        print(f"  Alpha- values: {len(alpha_neg_range)}")
        print(f"  Beta values: {len(beta_range)}")
        print(f"  Set sizes: {set_sizes}")
        print(f"  Total conditions: {len(alpha_pos_range) * len(alpha_neg_range) * len(beta_range) * len(set_sizes)}")

        results: List[Dict[str, float]] = []

        # Grid search
        conditions = list(itertools.product(alpha_pos_range, alpha_neg_range, beta_range, set_sizes))

        for alpha_pos, alpha_neg, beta, set_size in tqdm(conditions, desc="Parameter sweep"):
            params = {
                'num_stimuli': 6,
                'num_actions': 3,
                'alpha_pos': alpha_pos,
                'alpha_neg': alpha_neg,
                'beta': beta,
                'gamma': 0.0,
                'q_init': 0.5
            }

            metrics = self.run_single_condition(
                params=params,
                set_size=set_size,
                num_trials=num_trials,
                num_reps=num_reps
            )

            # Store results
            result = {
                'model': 'qlearning',
                'alpha_pos': alpha_pos,
                'alpha_neg': alpha_neg,
                'beta': beta,
                'set_size': set_size,
                **metrics
            }
            results.append(result)

        # Save results
        df = pd.DataFrame(results)
        self.results = results
        output_file = self.output_dir / f'qlearning_sweep_seed{self.seed}.csv'
        df.to_csv(output_file, index=False)
        print(f"\nSaved results to: {output_file}")

        return df

    def sweep_wmrl_parameters(
        self,
        alpha_pos_range: List[float] = [0.2, 0.3],
        alpha_neg_range: List[float] = [0.1],
        beta_range: List[float] = [2.0, 3.0],
        beta_wm_range: List[float] = [3.0, 5.0],
        capacity_range: List[int] = [3, 4, 5],
        phi_range: List[float] = [0.05, 0.1, 0.2],
        rho_range: List[float] = [0.5, 0.7, 0.9],
        set_sizes: List[int] = [2, 3, 5, 6],
        num_trials: int = 100,
        num_reps: int = 3
    ) -> pd.DataFrame:
        """
        Sweep WM-RL parameters (matrix-based architecture).

        Parameters
        ----------
        alpha_pos_range, alpha_neg_range : list
            RL learning rate parameters
        beta_range : list
            RL inverse temperature
        beta_wm_range : list
            WM inverse temperature
        capacity_range : list
            WM capacity values
        phi_range : list
            WM global decay rates
        rho_range : list
            Base WM reliance values
        set_sizes : list
            Task set sizes
        num_trials, num_reps : int
            Simulation parameters

        Returns
        -------
        pd.DataFrame
            Results dataframe
        """
        print(f"Running WM-RL parameter sweep")
        print(f"  Alpha+ values: {len(alpha_pos_range)}")
        print(f"  Alpha- values: {len(alpha_neg_range)}")
        print(f"  Beta (RL) values: {len(beta_range)}")
        print(f"  Beta (WM) values: {len(beta_wm_range)}")
        print(f"  Capacity values: {len(capacity_range)}")
        print(f"  Phi values: {len(phi_range)}")
        print(f"  Rho values: {len(rho_range)}")
        total_conditions = (len(alpha_pos_range) * len(alpha_neg_range) * len(beta_range) *
                           len(beta_wm_range) * len(capacity_range) * len(phi_range) *
                           len(rho_range) * len(set_sizes))
        print(f"  Total conditions: {total_conditions}")

        results: List[Dict[str, float]] = []

        # Grid search (all combinations)
        conditions = list(itertools.product(
            alpha_pos_range, alpha_neg_range, beta_range, beta_wm_range,
            capacity_range, phi_range, rho_range, set_sizes
        ))

        for alpha_pos, alpha_neg, beta, beta_wm, capacity, phi, rho, set_size in tqdm(conditions, desc="WM-RL sweep"):
            params = {
                'num_stimuli': 6,
                'num_actions': 3,
                'alpha_pos': alpha_pos,
                'alpha_neg': alpha_neg,
                'beta': beta,
                'beta_wm': beta_wm,
                'capacity': capacity,
                'phi': phi,
                'rho': rho,
                'gamma': 0.0,
                'q_init': 0.5,
                'wm_init': 0.0
            }

            metrics = self.run_single_condition(
                params=params,
                set_size=set_size,
                num_trials=num_trials,
                num_reps=num_reps
            )

            result = {
                'model': 'wmrl',
                'alpha_pos': alpha_pos,
                'alpha_neg': alpha_neg,
                'beta': beta,
                'beta_wm': beta_wm,
                'capacity': capacity,
                'phi': phi,
                'rho': rho,
                'set_size': set_size,
                **metrics
            }
            results.append(result)

        # Save
        df = pd.DataFrame(results)
        self.results = results
        output_file = self.output_dir / f'wmrl_sweep_seed{self.seed}.csv'
        df.to_csv(output_file, index=False)
        print(f"\nSaved results to: {output_file}")

        return df


def _require_columns(df: pd.DataFrame, cols: Sequence[str], *, context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for {context}: {missing}. Available: {list(df.columns)}")


def visualize_qlearning_sweep(
    sweep_results: pd.DataFrame,
    output_dir: Path,
    *,
    metric: Literal['accuracy', 'total_reward'] = 'accuracy',
    collapse_over: Sequence[str] = ('alpha_neg',),
):
    """
    Create visualizations of Q-learning parameter sweep.

    Parameters
    ----------
    sweep_results : pd.DataFrame
        Results from parameter sweep
    output_dir : Path
        Output directory for figures
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    _require_columns(
        sweep_results,
        cols=['alpha_pos', 'alpha_neg', 'beta', 'set_size', metric],
        context='qlearning sweep visualization',
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = sweep_results.copy()
    # For plotting, it’s often helpful to reduce dimensionality (e.g., average over alpha_neg).
    group_cols = ['alpha_pos', 'beta', 'set_size']
    for c in collapse_over:
        if c in group_cols:
            raise ValueError(f"`collapse_over` cannot include required grouping column: {c}")
    collapse_cols = [c for c in collapse_over if c in df.columns]
    if collapse_cols:
        df = df.groupby(group_cols, as_index=False)[metric].mean()

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. Alpha+ effect on metric (by set size), averaging over beta
    ax = axes[0, 0]
    for set_size in sorted(df['set_size'].unique()):
        subset = df[df['set_size'] == set_size]
        grouped = subset.groupby('alpha_pos')[metric].mean()
        ax.plot(grouped.index, grouped.values, marker='o', linewidth=2, label=f'Set size {set_size}')
    ax.set_xlabel('Positive learning rate (α+)', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Effect of α+ on {metric.replace("_", " ")}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    if metric == 'accuracy':
        ax.axhline(1/3, color='red', linestyle='--', alpha=0.5, label='Chance')

    # 2. Beta effect on metric (by set size), averaging over alpha_pos
    ax = axes[0, 1]
    for set_size in sorted(df['set_size'].unique()):
        subset = df[df['set_size'] == set_size]
        grouped = subset.groupby('beta')[metric].mean()
        ax.plot(grouped.index, grouped.values, marker='o', linewidth=2, label=f'Set size {set_size}')
    ax.set_xlabel('Inverse Temperature (β)', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Effect of β on {metric.replace("_", " ")}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 3. Heatmap: alpha_pos × beta (averaged across set sizes)
    ax = axes[1, 0]
    pivot = df.groupby(['alpha_pos', 'beta'])[metric].mean().reset_index()
    pivot_table = pivot.pivot(index='alpha_pos', columns='beta', values=metric)
    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='viridis', ax=ax,
                cbar_kws={'label': metric.replace('_', ' ').title()})
    ax.set_title(f'{metric.replace("_", " ").title()} heatmap: α+ × β\n(averaged across set sizes)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Inverse Temperature (β)', fontsize=12)
    ax.set_ylabel('Positive learning rate (α+)', fontsize=12)

    # 4. Set size effect (by alpha_pos)
    ax = axes[1, 1]
    alphas_to_plot = sorted(df['alpha_pos'].unique())[:4]
    for alpha_pos in alphas_to_plot:
        subset = df[df['alpha_pos'] == alpha_pos]
        if len(subset) > 0:
            grouped = subset.groupby('set_size')[metric].mean()
            ax.plot(grouped.index, grouped.values, marker='o', linewidth=2, label=f'α+={alpha_pos}')
    ax.set_xlabel('Set Size', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Set size effect by α+ (collapsed)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'qlearning_sweep_viz.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}")
    plt.close()


def visualize_wmrl_sweep(
    sweep_results: pd.DataFrame,
    output_dir: Path,
    *,
    metric: Literal['accuracy', 'total_reward'] = 'accuracy',
    collapse_over: Sequence[str] = ('alpha_pos', 'alpha_neg', 'beta', 'beta_wm', 'phi'),
):
    """
    Create visualizations of WM-RL parameter sweep.

    Parameters
    ----------
    sweep_results : pd.DataFrame
        Results from parameter sweep
    output_dir : Path
        Output directory for figures
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    _require_columns(
        sweep_results,
        cols=['capacity', 'rho', 'set_size', metric],
        context='wmrl sweep visualization',
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = sweep_results.copy()
    group_cols = ['capacity', 'rho', 'set_size']
    collapse_cols = [c for c in collapse_over if c in df.columns and c not in group_cols]
    if collapse_cols:
        df = df.groupby(group_cols, as_index=False)[metric].mean()

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # 1. Capacity effect by set size
    ax = axes[0, 0]
    for set_size in sorted(df['set_size'].unique()):
        subset = df[df['set_size'] == set_size]
        grouped = subset.groupby('capacity')[metric].mean()
        ax.plot(grouped.index, grouped.values, marker='o', linewidth=2, label=f'Set size {set_size}')
    ax.set_xlabel('WM Capacity', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Capacity effect on {metric.replace("_", " ")}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 2. Rho (WM reliance) effect
    ax = axes[0, 1]
    for set_size in sorted(df['set_size'].unique()):
        subset = df[df['set_size'] == set_size]
        grouped = subset.groupby('rho')[metric].mean()
        ax.plot(grouped.index, grouped.values, marker='o', linewidth=2, label=f'Set size {set_size}')
    ax.set_xlabel('WM reliance (ρ)', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'ρ effect on {metric.replace("_", " ")}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # 3. Heatmap: capacity × rho (averaged)
    ax = axes[1, 0]
    pivot = df.groupby(['capacity', 'rho'])[metric].mean().reset_index()
    pivot_table = pivot.pivot(index='capacity', columns='rho', values=metric)
    sns.heatmap(pivot_table, annot=True, fmt='.2f', cmap='viridis', ax=ax,
                cbar_kws={'label': metric.replace('_', ' ').title()})
    ax.set_title(f'{metric.replace("_", " ").title()}: capacity × ρ\n(averaged)', fontsize=14, fontweight='bold')
    ax.set_xlabel('WM reliance (ρ)', fontsize=12)
    ax.set_ylabel('WM Capacity', fontsize=12)

    # 4. Set size effect by capacity
    ax = axes[1, 1]
    capacities_to_plot = sorted(df['capacity'].unique())[:3]
    for capacity in capacities_to_plot:
        subset = df[df['capacity'] == capacity]
        if len(subset) > 0:
            grouped = subset.groupby('set_size')[metric].mean()
            ax.plot(grouped.index, grouped.values, marker='o', linewidth=2, label=f'Capacity={capacity}')
    ax.set_xlabel('Set Size', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title('Set size effect by capacity (collapsed)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    output_file = output_dir / 'wmrl_sweep_viz.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved visualization to: {output_file}")
    plt.close()


def main(argv: Optional[Sequence[str]] = None):
    """Run parameter sweep from command line.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse (defaults to `sys.argv[1:]`). Passing an explicit list
        is useful in notebooks where IPython injects extra flags.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Run parameter sweep for RLWM models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, default='qlearning',
                        choices=['qlearning', 'wmrl'],
                        help='Model type to sweep')
    parser.add_argument('--num-trials', type=int, default=100,
                        help='Number of trials per condition')
    parser.add_argument('--num-reps', type=int, default=5,
                        help='Number of repetitions per condition')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: OUTPUT_VERSION_DIR/parameter_sweeps if available)')

    args = parser.parse_args(argv)

    # Create sweep object
    sweep = ParameterSweep(
        model_type=args.model,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        seed=args.seed
    )

    # Run sweep
    if args.model == 'qlearning':
        print("\n" + "="*80)
        print("Q-LEARNING PARAMETER SWEEP")
        print("="*80 + "\n")

        results = sweep.sweep_qlearning_parameters(
            num_trials=args.num_trials,
            num_reps=args.num_reps
        )

        print("\nGenerating visualizations...")
        visualize_qlearning_sweep(results, sweep.output_dir)

    elif args.model == 'wmrl':
        print("\n" + "="*80)
        print("WM-RL HYBRID PARAMETER SWEEP")
        print("="*80 + "\n")

        results = sweep.sweep_wmrl_parameters(
            num_trials=args.num_trials,
            num_reps=args.num_reps
        )

        print("\nGenerating visualizations...")
        visualize_wmrl_sweep(results, sweep.output_dir)

    print("\n" + "="*80)
    print("PARAMETER SWEEP COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {sweep.output_dir}")


if __name__ == "__main__":
    # In notebooks, `__name__ == "__main__"` but IPython injects arguments (e.g. `-f ...json`),
    # which breaks argparse. Avoid auto-running the CLI in that context.
    if "ipykernel" not in sys.modules:
        main()
