#!/usr/bin/env python
"""
09_1: Simulate Model Predictions
================================

Generate model predictions on human behavioral data for comparison.

This script simulates how Q-learning and WM-RL models would perform
on actual human trial sequences, useful for:
- Model validation (do predicted patterns match human data?)
- Understanding model behavior with reasonable parameters
- Generating stimulus-based learning curves
- Multi-run Monte Carlo analysis

Note: This is DIFFERENT from 09_generate_synthetic_data.py which creates
completely new synthetic participants. This script uses HUMAN data as the
trial sequence and simulates model responses.

CONSOLIDATED from:
    - scripts/analysis/simulate_model_performance.py (single run)
    - scripts/analysis/simulate_model_performance_multi_run.py (Monte Carlo)

Inputs:
    - output/task_trials_long.csv (human behavioral data)

Outputs:
    - output/model_performance/<model>_predictions_simulated.csv
    - output/model_performance/<model>_predictions_stimulus_based_n<N>.csv
    - figures/model_performance/<model>_stimulus_learning_curve.png
    - figures/model_performance/<model>_performance_analysis.png

Usage:
    # Single run with default parameters
    python scripts/09_1_simulate_model_predictions.py --mode single

    # Multi-run Monte Carlo (recommended for averaging)
    python scripts/09_1_simulate_model_predictions.py --mode multi --n-runs 20

    # Specific models
    python scripts/09_1_simulate_model_predictions.py --models qlearning wmrl

    # Custom parameters (JSON format)
    python scripts/09_1_simulate_model_predictions.py --params '{"alpha_pos": 0.5}'

Next Steps:
    - Compare predictions to human data
    - Run 09_generate_synthetic_data.py for full synthetic datasets
    - Use 10_run_parameter_sweep.py for systematic exploration
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from config import (
    AnalysisParams,
    OUTPUT_DIR,
    FIGURES_DIR,
    TaskParams
)
from models.q_learning import QLearningAgent
from models.wm_rl_hybrid import WMRLHybridAgent


# =============================================================================
# DEFAULT PARAMETERS
# =============================================================================

DEFAULT_PARAMS = {
    'qlearning': {
        'alpha_pos': 0.6,   # Moderate positive learning rate
        'alpha_neg': 0.3,   # Lower negative learning rate (common asymmetry)
        'beta': 2.5         # Moderate exploration/exploitation
    },
    'wmrl': {
        'alpha_pos': 0.5,
        'alpha_neg': 0.3,
        'beta': 2.0,
        'beta_wm': 3.0,     # Higher inverse temp for WM (more confident)
        'capacity': 4,      # Moderate WM capacity
        'phi': 0.2,         # Some decay
        'rho': 0.7          # Moderate WM reliance
    }
}


# =============================================================================
# SINGLE RUN SIMULATION
# =============================================================================

def generate_predictions_single_run(
    behavioral_data: pd.DataFrame,
    model_name: str,
    params: Dict,
    track_stimulus_encounters: bool = False,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate trial-by-trial predictions from model for a single run.

    Parameters
    ----------
    behavioral_data : pd.DataFrame
        Behavioral data with columns: subject_id/sona_id, block, trial,
        set_size, stimulus, response, correct
    model_name : str
        'qlearning' or 'wmrl'
    params : dict
        Model parameters
    track_stimulus_encounters : bool
        If True, track encounters per stimulus (for multi-run analysis)
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Predictions with performance metrics
    """
    np.random.seed(seed)
    predictions = []

    # Standardize column names
    data = behavioral_data.copy()
    if 'sona_id' in data.columns:
        data['subject_id'] = data['sona_id']
    if 'key_answer' in data.columns:
        data['response'] = data['key_answer']

    # Group by subject and block
    for (subject_id, block_id), block_data in data.groupby(['subject_id', 'block']):
        block_data = block_data.sort_values('trial').reset_index(drop=True)

        # Initialize agent
        if model_name == 'qlearning':
            agent = QLearningAgent(
                alpha_pos=params['alpha_pos'],
                alpha_neg=params['alpha_neg'],
                beta=params['beta']
            )
        elif model_name == 'wmrl':
            agent = WMRLHybridAgent(
                alpha_pos=params['alpha_pos'],
                alpha_neg=params['alpha_neg'],
                beta=params['beta'],
                beta_wm=params.get('beta_wm', params['beta']),
                capacity=int(params['capacity']),
                phi=params['phi'],
                rho=params['rho']
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Track encounters (for stimulus-based analysis)
        stimulus_encounter_count = defaultdict(int)
        stimulus_reversal_occurred = defaultdict(bool)
        stimulus_encounters_since_reversal = defaultdict(int)
        stimulus_correct_streak = defaultdict(int)

        # Track block-level reversals
        correct_streak = 0
        reversal_trial = None
        trials_since_reversal = 0

        # Simulate block trial by trial
        for idx, row in block_data.iterrows():
            # Ensure values are proper Python ints
            # Note: stimulus in data is 1-indexed, models expect 0-indexed
            stimulus = int(float(row['stimulus'])) - 1
            set_size = int(float(row['set_size']))
            actual_correct = int(float(row['correct']))

            # Stimulus encounter tracking
            stimulus_encounter_count[stimulus] += 1
            encounter_num = stimulus_encounter_count[stimulus]

            # Get model's choice
            if model_name == 'wmrl':
                model_choice, _ = agent.choose_action(stimulus, set_size)
                model_choice = int(model_choice)
            else:
                model_choice = int(agent.choose_action(stimulus))

            # Determine if model was correct
            if actual_correct == 1:
                rewarded_action = int(float(row['response']))
            else:
                rewarded_action = 1 - int(float(row['response']))

            model_correct = int(model_choice == rewarded_action)

            # Update agent with actual feedback
            if model_name == 'wmrl':
                agent.update(stimulus, model_choice, model_correct, set_size)
            else:
                agent.update(stimulus, model_choice, model_correct)

            # Track reversals
            if actual_correct:
                correct_streak += 1
                stimulus_correct_streak[stimulus] += 1

                # Block-level reversal detection
                if TaskParams.REVERSAL_MIN <= correct_streak <= TaskParams.REVERSAL_MAX:
                    if reversal_trial is None:
                        reversal_trial = idx
                        trials_since_reversal = 0

                # Stimulus-level reversal detection
                if TaskParams.REVERSAL_MIN <= stimulus_correct_streak[stimulus] <= TaskParams.REVERSAL_MAX:
                    if not stimulus_reversal_occurred[stimulus]:
                        stimulus_reversal_occurred[stimulus] = True
                        stimulus_encounters_since_reversal[stimulus] = 0
            else:
                correct_streak = 0
                stimulus_correct_streak[stimulus] = 0

            # Build prediction record
            pred = {
                'subject_id': subject_id,
                'block': block_id,
                'trial': row['trial'],
                'trial_num': idx + 1,
                'set_size': set_size,
                'stimulus': stimulus,
                'model_choice': model_choice,
                'correct': model_correct,
                'trials_since_reversal': trials_since_reversal,
                'is_post_reversal': reversal_trial is not None
            }

            # Add stimulus-based tracking if requested
            if track_stimulus_encounters:
                pred['encounter_num'] = encounter_num
                pred['encounters_since_reversal'] = stimulus_encounters_since_reversal[stimulus]
                pred['stimulus_post_reversal'] = stimulus_reversal_occurred[stimulus]

            predictions.append(pred)

            # Increment counters
            if reversal_trial is not None:
                trials_since_reversal += 1
            if stimulus_reversal_occurred[stimulus]:
                stimulus_encounters_since_reversal[stimulus] += 1

    return pd.DataFrame(predictions)


# =============================================================================
# MULTI-RUN SIMULATION (MONTE CARLO)
# =============================================================================

def run_multiple_simulations(
    behavioral_data: pd.DataFrame,
    model_name: str,
    params: Dict,
    n_runs: int = 10,
    base_seed: int = 42
) -> pd.DataFrame:
    """
    Run multiple simulations and combine results.

    Parameters
    ----------
    behavioral_data : pd.DataFrame
        Behavioral data
    model_name : str
        Model name
    params : dict
        Model parameters
    n_runs : int
        Number of simulation runs
    base_seed : int
        Base random seed (incremented for each run)

    Returns
    -------
    pd.DataFrame
        Combined predictions from all runs with run_id column
    """
    all_predictions = []

    print(f"  Running {n_runs} simulations...")
    for run_id in tqdm(range(n_runs), desc="  Simulations"):
        run_seed = base_seed + run_id
        predictions = generate_predictions_single_run(
            behavioral_data,
            model_name,
            params,
            track_stimulus_encounters=True,
            seed=run_seed
        )
        predictions['run_id'] = run_id
        all_predictions.append(predictions)

    return pd.concat(all_predictions, ignore_index=True)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_stimulus_learning_curves(
    predictions_df: pd.DataFrame,
    model_name: str,
    save_dir: Path
) -> None:
    """
    Plot learning curves based on encounters per stimulus.

    X-axis: Encounter number with stimulus (1st time, 2nd time, etc.)
    Y-axis: Accuracy
    Lines: Different set sizes
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = AnalysisParams.COLORS_SET_SIZE
    set_sizes = sorted(predictions_df['set_size'].unique())

    fig, ax = plt.subplots(figsize=(12, 6))

    for ss in set_sizes:
        ss_data = predictions_df[predictions_df['set_size'] == ss].copy()

        # Group by encounter number
        grouped = ss_data.groupby('encounter_num')['correct'].agg(
            ['mean', 'sem', 'count']
        ).reset_index()

        # Only plot encounters with sufficient data
        grouped = grouped[grouped['count'] >= 5]

        if len(grouped) == 0:
            continue

        ax.plot(
            grouped['encounter_num'],
            grouped['mean'] * 100,
            'o-',
            color=colors[ss],
            label=f'Set Size {ss}',
            linewidth=2,
            markersize=4,
            alpha=0.8
        )

        ax.fill_between(
            grouped['encounter_num'],
            (grouped['mean'] - grouped['sem']) * 100,
            (grouped['mean'] + grouped['sem']) * 100,
            color=colors[ss],
            alpha=0.2
        )

    ax.set_xlabel('Encounter with Stimulus (Nth time)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Chance')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{model_name}: Learning by Stimulus Encounters', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    save_path = save_dir / f'{model_name.lower().replace(" ", "_")}_stimulus_learning_curve.png'
    plt.savefig(save_path, dpi=AnalysisParams.FIG_DPI, format=AnalysisParams.FIG_FORMAT, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()


def plot_combined_performance_analysis(
    predictions_df: pd.DataFrame,
    model_name: str,
    n_trials_threshold: int,
    save_dir: Path
) -> None:
    """Create combined two-panel figure with stimulus-based analyses."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = AnalysisParams.COLORS_SET_SIZE
    set_sizes = sorted(predictions_df['set_size'].unique())

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.3)

    # ===== LEFT: Learning curve by stimulus encounters =====
    ax1 = fig.add_subplot(gs[0, 0])

    encounter_col = 'encounter_num' if 'encounter_num' in predictions_df.columns else 'trial_num'

    for ss in set_sizes:
        ss_data = predictions_df[predictions_df['set_size'] == ss].copy()
        grouped = ss_data.groupby(encounter_col)['correct'].agg(['mean', 'sem', 'count']).reset_index()
        grouped = grouped[grouped['count'] >= 5]

        if len(grouped) == 0:
            continue

        ax1.plot(
            grouped[encounter_col],
            grouped['mean'] * 100,
            'o-',
            color=colors[ss],
            label=f'Set Size {ss}',
            linewidth=2,
            markersize=4,
            alpha=0.8
        )

        ax1.fill_between(
            grouped[encounter_col],
            (grouped['mean'] - grouped['sem']) * 100,
            (grouped['mean'] + grouped['sem']) * 100,
            color=colors[ss],
            alpha=0.2
        )

    xlabel = 'Encounter with Stimulus (Nth time)' if encounter_col == 'encounter_num' else 'Trial Number'
    ax1.set_xlabel(xlabel, fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{model_name}: Learning Curves', fontsize=12, fontweight='bold')

    # ===== RIGHT: Performance by position =====
    ax2 = fig.add_subplot(gs[0, 1])

    def categorize_position(row):
        if not row['is_post_reversal']:
            if row['trial_num'] < n_trials_threshold:
                return 'Early Block'
            else:
                return 'Late Block'
        else:
            if row['trials_since_reversal'] < n_trials_threshold:
                return 'Early Post-Rev'
            else:
                return 'Late Post-Rev'

    df = predictions_df.copy()
    df['position'] = df.apply(categorize_position, axis=1)

    position_order = ['Early Block', 'Late Block', 'Early Post-Rev', 'Late Post-Rev']
    grouped = df.groupby(['set_size', 'position'])['correct'].agg(['mean', 'sem']).reset_index()

    n_positions = len(position_order)
    n_set_sizes = len(set_sizes)
    bar_width = 0.8 / n_set_sizes
    x = np.arange(n_positions)

    for i, ss in enumerate(set_sizes):
        ss_data = grouped[grouped['set_size'] == ss].copy()
        ss_data = ss_data.set_index('position').reindex(position_order).reset_index()
        ss_data['mean'] = ss_data['mean'].fillna(0)
        ss_data['sem'] = ss_data['sem'].fillna(0)

        offset = (i - n_set_sizes/2 + 0.5) * bar_width
        ax2.bar(
            x + offset,
            ss_data['mean'] * 100,
            bar_width,
            label=f'Set Size {ss}',
            color=colors[ss],
            alpha=0.8,
            yerr=ss_data['sem'] * 100,
            capsize=4
        )

    ax2.set_xlabel('Trial Position', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_ylim([0, 110])
    ax2.set_xticks(x)
    ax2.set_xticklabels(position_order, fontsize=9, rotation=15, ha='right')
    ax2.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_title(f'{model_name}: Performance by Position', fontsize=12, fontweight='bold')

    plt.suptitle(f'{model_name} Performance Analysis', fontsize=14, fontweight='bold', y=1.02)

    save_path = save_dir / f'{model_name.lower().replace(" ", "_")}_performance_analysis.png'
    plt.savefig(save_path, dpi=AnalysisParams.FIG_DPI, format=AnalysisParams.FIG_FORMAT, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Simulate model predictions on human behavioral data'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='output/task_trials_long.csv',
        help='Path to behavioral data'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='multi',
        choices=['single', 'multi'],
        help='Simulation mode: single run or multi-run Monte Carlo'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['qlearning', 'wmrl'],
        help='Models to simulate: qlearning, wmrl, or both'
    )
    parser.add_argument(
        '--n-runs',
        type=int,
        default=20,
        help='Number of simulation runs for multi mode'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=4,
        help='Trial threshold for early/late classification'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--params',
        type=str,
        default=None,
        help='Custom parameters as JSON string (overrides defaults)'
    )

    args = parser.parse_args()

    # Paths
    data_path = project_root / args.data
    output_dir = OUTPUT_DIR / 'model_performance'
    figure_dir = FIGURES_DIR / 'model_performance'

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MODEL PREDICTION SIMULATION")
    print("=" * 80)
    print(f"\nMode: {args.mode}")
    print(f"Models: {args.models}")
    if args.mode == 'multi':
        print(f"Runs: {args.n_runs}")

    # Load behavioral data
    print(f"\nLoading behavioral data: {data_path}")
    behavioral_data = pd.read_csv(data_path)

    # Standardize column names
    if 'sona_id' in behavioral_data.columns:
        behavioral_data['subject_id'] = behavioral_data['sona_id']
    if 'key_answer' in behavioral_data.columns:
        behavioral_data['response'] = behavioral_data['key_answer']

    print(f"  Loaded {len(behavioral_data)} trials")
    print(f"  Subjects: {behavioral_data['subject_id'].nunique()}")
    print(f"  Blocks: {behavioral_data['block'].nunique()}")
    print(f"  Set sizes: {sorted(behavioral_data['set_size'].unique())}")

    # Parse custom params if provided
    custom_params = json.loads(args.params) if args.params else {}

    # Process each model
    for model_name in args.models:
        if model_name not in DEFAULT_PARAMS:
            print(f"\nSkipping unknown model: {model_name}")
            continue

        print("\n" + "=" * 80)
        print(f"SIMULATING {model_name.upper()} MODEL")
        print("=" * 80)

        # Merge default and custom params
        params = DEFAULT_PARAMS[model_name].copy()
        params.update(custom_params)

        print(f"\nUsing parameter values:")
        for param, value in params.items():
            print(f"  {param} = {value}")

        # Generate predictions
        if args.mode == 'single':
            predictions_df = generate_predictions_single_run(
                behavioral_data,
                model_name,
                params,
                track_stimulus_encounters=False,
                seed=args.seed
            )
            filename_suffix = 'simulated'
        else:  # multi
            predictions_df = run_multiple_simulations(
                behavioral_data,
                model_name,
                params,
                n_runs=args.n_runs,
                base_seed=args.seed
            )
            filename_suffix = f'stimulus_based_n{args.n_runs}'

        # Calculate performance
        overall_acc = predictions_df['correct'].mean()
        print(f"\n  Overall accuracy: {overall_acc:.3f}")

        print(f"\n  Performance by set size:")
        for ss in sorted(predictions_df['set_size'].unique()):
            ss_acc = predictions_df[predictions_df['set_size'] == ss]['correct'].mean()
            print(f"    Set Size {ss}: {ss_acc:.3f}")

        # Save predictions
        predictions_file = output_dir / f'{model_name}_predictions_{filename_suffix}.csv'
        predictions_df.to_csv(predictions_file, index=False)
        print(f"\n  [OK] Saved predictions: {predictions_file}")

        # Create visualizations
        print(f"\nCreating visualizations...")

        model_display_name = {
            'qlearning': 'Q-Learning',
            'wmrl': 'WM-RL Hybrid'
        }.get(model_name, model_name)

        # Combined analysis plot
        plot_combined_performance_analysis(
            predictions_df,
            model_display_name,
            n_trials_threshold=args.threshold,
            save_dir=figure_dir
        )

        # Stimulus learning curves (multi-run only)
        if args.mode == 'multi' and 'encounter_num' in predictions_df.columns:
            plot_stimulus_learning_curves(
                predictions_df,
                model_display_name,
                save_dir=figure_dir
            )

        # Performance summary by position
        def categorize_trial(row):
            if not row['is_post_reversal']:
                if row['trial_num'] < args.threshold:
                    return 'Early Block'
                else:
                    return 'Late Block'
            else:
                if row['trials_since_reversal'] < args.threshold:
                    return 'Early Post-Reversal'
                else:
                    return 'Late Post-Reversal'

        predictions_df['position'] = predictions_df.apply(categorize_trial, axis=1)

        print(f"\n  Performance by trial position:")
        for pos in ['Early Block', 'Late Block', 'Early Post-Reversal', 'Late Post-Reversal']:
            pos_data = predictions_df[predictions_df['position'] == pos]
            if len(pos_data) > 0:
                acc = pos_data['correct'].mean()
                print(f"    {pos:20s}: {acc:.3f} (n={len(pos_data)})")

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print(f"\nFigures saved to: {figure_dir}")
    print(f"Predictions saved to: {output_dir}")
    print("\nGenerated outputs:")
    print("  - Performance analysis plots")
    if args.mode == 'multi':
        print("  - Stimulus-based learning curves")
        print(f"  - Averaged over {args.n_runs} runs")
    print()
    print("Next steps:")
    print("  - Compare predictions with human data")
    print("  - Run 09_generate_synthetic_data.py for full synthetic datasets")
    print("  - Run 10_run_parameter_sweep.py for systematic exploration")
    print("=" * 80)


if __name__ == '__main__':
    main()
