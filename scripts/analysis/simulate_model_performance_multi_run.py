"""
Simulate model predictions with multiple runs and stimulus-based learning curves.

This script:
1. Tracks encounters PER STIMULUS (not trials per block)
2. Runs multiple simulations to capture average performance + variability
3. Plots learning curves showing "trials per stimulus" averaged across all stimuli
4. Categorizes performance by "Nth encounter with stimulus" (not trial position)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import sys
from collections import defaultdict
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from models.q_learning import QLearningAgent
from models.wm_rl_hybrid import WMRLHybridAgent
from config import (
    TaskParams,
    OUTPUT_DIR, FIGURES_DIR
)


def generate_model_predictions_single_run(
    behavioral_data: pd.DataFrame,
    model_name: str,
    params: dict,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate trial-by-trial predictions from model for a single run.

    Now tracks encounters PER STIMULUS instead of trial position.

    Parameters
    ----------
    behavioral_data : pd.DataFrame
        Behavioral data with columns: subject_id, block, trial, set_size,
        stimulus, response, correct
    model_name : str
        'qlearning' or 'wmrl'
    params : dict
        Model parameters
    seed : int
        Random seed

    Returns
    -------
    predictions_df : pd.DataFrame
        Predictions with stimulus-based encounter tracking
    """
    np.random.seed(seed)
    predictions = []

    # Group by subject and block
    for (subject_id, block_id), block_data in behavioral_data.groupby(['subject_id', 'block']):
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

        # Track encounters per stimulus
        stimulus_encounter_count = defaultdict(int)
        stimulus_reversal_occurred = defaultdict(bool)
        stimulus_encounters_since_reversal = defaultdict(int)
        stimulus_correct_streak = defaultdict(int)

        # Simulate block trial by trial
        for idx, row in block_data.iterrows():
            # Ensure values are proper Python ints, not numpy types
            # Note: stimulus in data is 1-indexed, but models expect 0-indexed
            stimulus = int(float(row['stimulus'])) - 1
            set_size = int(float(row['set_size']))
            actual_correct = int(float(row['correct']))

            # Increment encounter count for this stimulus
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

            # Track reversals PER STIMULUS
            if actual_correct:
                stimulus_correct_streak[stimulus] += 1
                # Check if reversal should occur for this stimulus
                if (TaskParams.REVERSAL_MIN <= stimulus_correct_streak[stimulus] <= TaskParams.REVERSAL_MAX):
                    if not stimulus_reversal_occurred[stimulus]:
                        stimulus_reversal_occurred[stimulus] = True
                        stimulus_encounters_since_reversal[stimulus] = 0
            else:
                stimulus_correct_streak[stimulus] = 0

            # Increment encounters since reversal for this stimulus
            if stimulus_reversal_occurred[stimulus]:
                stimulus_encounters_since_reversal[stimulus] += 1

            # Record prediction
            predictions.append({
                'subject_id': subject_id,
                'block': block_id,
                'trial': row['trial'],
                'trial_num': idx + 1,
                'set_size': set_size,
                'stimulus': stimulus,
                'encounter_num': encounter_num,  # NEW: Nth time seeing this stimulus
                'model_choice': model_choice,
                'correct': model_correct,
                'encounters_since_reversal': stimulus_encounters_since_reversal[stimulus],
                'is_post_reversal': stimulus_reversal_occurred[stimulus]
            })

    return pd.DataFrame(predictions)


def run_multiple_simulations(
    behavioral_data: pd.DataFrame,
    model_name: str,
    params: dict,
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
    all_predictions : pd.DataFrame
        Combined predictions from all runs with run_id column
    """
    all_predictions = []

    print(f"  Running {n_runs} simulations...")
    for run_id in tqdm(range(n_runs), desc="  Simulations"):
        run_seed = base_seed + run_id
        predictions = generate_model_predictions_single_run(
            behavioral_data,
            model_name,
            params,
            seed=run_seed
        )
        predictions['run_id'] = run_id
        all_predictions.append(predictions)

    return pd.concat(all_predictions, ignore_index=True)


def plot_stimulus_learning_curves(
    predictions_df: pd.DataFrame,
    model_name: str,
    save_dir: Path
):
    """
    Plot learning curves based on encounters per stimulus.

    X-axis: Encounter number with stimulus (1st time, 2nd time, etc.)
    Y-axis: Accuracy
    Lines: Different set sizes
    Averaged across all stimuli and all runs

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with encounter_num column
    model_name : str
        Model name for title
    save_dir : Path
        Directory to save figure
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from config import AnalysisParams

    colors = AnalysisParams.COLORS_SET_SIZE
    set_sizes = sorted(predictions_df['set_size'].unique())

    fig, ax = plt.subplots(figsize=(12, 6))

    # Group by encounter number and set size, then average
    for ss in set_sizes:
        ss_data = predictions_df[predictions_df['set_size'] == ss].copy()

        # Group by encounter number, compute mean and SEM
        grouped = ss_data.groupby('encounter_num')['correct'].agg(['mean', 'sem', 'count']).reset_index()

        # Only plot encounters with sufficient data
        grouped = grouped[grouped['count'] >= 5]

        if len(grouped) == 0:
            continue

        # Plot line
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

        # Add confidence band
        ax.fill_between(
            grouped['encounter_num'],
            (grouped['mean'] - grouped['sem']) * 100,
            (grouped['mean'] + grouped['sem']) * 100,
            color=colors[ss],
            alpha=0.2
        )

    # Styling
    ax.set_xlabel('Encounter with Stimulus (Nth time)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Chance')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'{model_name}: Learning by Stimulus Encounters', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save
    save_path = save_dir / f'{model_name.lower().replace(" ", "_")}_stimulus_learning_curve.png'
    plt.savefig(save_path, dpi=AnalysisParams.FIG_DPI, format=AnalysisParams.FIG_FORMAT, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()


def plot_performance_by_stimulus_encounter(
    predictions_df: pd.DataFrame,
    model_name: str,
    n_encounters_threshold: int,
    save_dir: Path
):
    """
    Plot performance by stimulus encounter position.

    Categories:
    1. Early stimulus experience (< N encounters with THIS stimulus)
    2. Late stimulus experience (>= N encounters, pre-reversal)
    3. Early post-reversal (< N encounters since reversal)
    4. Late post-reversal (>= N encounters since reversal)

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions
    model_name : str
        Model name
    n_encounters_threshold : int
        Threshold for early/late
    save_dir : Path
        Save directory
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from config import AnalysisParams

    colors = AnalysisParams.COLORS_SET_SIZE
    set_sizes = sorted(predictions_df['set_size'].unique())

    # Categorize by stimulus encounter position
    def categorize_encounter(row):
        if not row['is_post_reversal']:
            if row['encounter_num'] < n_encounters_threshold:
                return 'Early Stimulus\nExperience'
            else:
                return 'Late Stimulus\nExperience'
        else:
            if row['encounters_since_reversal'] < n_encounters_threshold:
                return 'Early\nPost-Reversal'
            else:
                return 'Late\nPost-Reversal'

    df = predictions_df.copy()
    df['encounter_position'] = df.apply(categorize_encounter, axis=1)

    # Define position order
    position_order = [
        'Early Stimulus\nExperience',
        'Late Stimulus\nExperience',
        'Early\nPost-Reversal',
        'Late\nPost-Reversal'
    ]

    # Group by set size and position
    grouped = df.groupby(['set_size', 'encounter_position'])['correct'].agg(['mean', 'sem', 'count']).reset_index()

    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 7))

    n_positions = len(position_order)
    n_set_sizes = len(set_sizes)
    bar_width = 0.8 / n_set_sizes
    x = np.arange(n_positions)

    # Plot bars for each set size
    for i, ss in enumerate(set_sizes):
        ss_data = grouped[grouped['set_size'] == ss].copy()
        ss_data = ss_data.set_index('encounter_position').reindex(position_order).reset_index()
        ss_data['mean'] = ss_data['mean'].fillna(0)
        ss_data['sem'] = ss_data['sem'].fillna(0)

        offset = (i - n_set_sizes/2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            ss_data['mean'] * 100,
            bar_width,
            label=f'Set Size {ss}',
            color=colors[ss],
            alpha=0.8,
            yerr=ss_data['sem'] * 100,
            capsize=4
        )

        # Add value labels
        for bar, acc in zip(bars, ss_data['mean'] * 100):
            height = bar.get_height()
            if height > 0:
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 2,
                    f'{acc:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold'
                )

    # Styling
    ax.set_xlabel('Stimulus Encounter Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 110])
    ax.set_xticks(x)
    ax.set_xticklabels(position_order, fontsize=10)
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title(
        f'{model_name}: Performance by Stimulus Encounter Position\n(Threshold = {n_encounters_threshold} encounters)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    plt.tight_layout()

    # Save
    save_path = save_dir / f'{model_name.lower().replace(" ", "_")}_stimulus_encounter_performance.png'
    plt.savefig(save_path, dpi=AnalysisParams.FIG_DPI, format=AnalysisParams.FIG_FORMAT, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()


def plot_combined_stimulus_analysis(
    predictions_df: pd.DataFrame,
    model_name: str,
    n_encounters_threshold: int,
    save_dir: Path
):
    """Create combined two-panel figure with stimulus-based analyses."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from config import AnalysisParams

    colors = AnalysisParams.COLORS_SET_SIZE
    set_sizes = sorted(predictions_df['set_size'].unique())

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.3)

    # ===== LEFT: Learning curve by stimulus encounters =====
    ax1 = fig.add_subplot(gs[0, 0])

    for ss in set_sizes:
        ss_data = predictions_df[predictions_df['set_size'] == ss].copy()
        grouped = ss_data.groupby('encounter_num')['correct'].agg(['mean', 'sem', 'count']).reset_index()
        grouped = grouped[grouped['count'] >= 5]

        if len(grouped) == 0:
            continue

        ax1.plot(
            grouped['encounter_num'],
            grouped['mean'] * 100,
            'o-',
            color=colors[ss],
            label=f'Set Size {ss}',
            linewidth=2,
            markersize=4,
            alpha=0.8
        )

        ax1.fill_between(
            grouped['encounter_num'],
            (grouped['mean'] - grouped['sem']) * 100,
            (grouped['mean'] + grouped['sem']) * 100,
            color=colors[ss],
            alpha=0.2
        )

    ax1.set_xlabel('Encounter with Stimulus (Nth time)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{model_name}: Stimulus Learning Curves', fontsize=12, fontweight='bold')

    # ===== RIGHT: Performance by encounter position =====
    ax2 = fig.add_subplot(gs[0, 1])

    def categorize_encounter(row):
        if not row['is_post_reversal']:
            if row['encounter_num'] < n_encounters_threshold:
                return 'Early Stim'
            else:
                return 'Late Stim'
        else:
            if row['encounters_since_reversal'] < n_encounters_threshold:
                return 'Early Post-Rev'
            else:
                return 'Late Post-Rev'

    df = predictions_df.copy()
    df['position'] = df.apply(categorize_encounter, axis=1)

    position_order = ['Early Stim', 'Late Stim', 'Early Post-Rev', 'Late Post-Rev']
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

    ax2.set_xlabel('Stimulus Encounter Position', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_ylim([0, 110])
    ax2.set_xticks(x)
    ax2.set_xticklabels(position_order, fontsize=9, rotation=15, ha='right')
    ax2.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_title(f'{model_name}: Performance by Encounter Position', fontsize=12, fontweight='bold')

    plt.suptitle(f'{model_name} Performance Analysis (Stimulus-Based)', fontsize=14, fontweight='bold', y=1.02)

    # Save
    save_path = save_dir / f'{model_name.lower().replace(" ", "_")}_stimulus_performance_analysis.png'
    plt.savefig(save_path, dpi=AnalysisParams.FIG_DPI, format=AnalysisParams.FIG_FORMAT, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Simulate model predictions with stimulus-based learning curves'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='output/task_trials_long.csv',
        help='Path to behavioral data'
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
        help='Number of simulation runs for averaging'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=4,
        help='Encounter threshold for early/late classification'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Base random seed'
    )

    args = parser.parse_args()

    # Paths
    data_path = project_root / args.data
    output_dir = OUTPUT_DIR / 'model_performance'
    figure_dir = FIGURES_DIR / 'model_performance'

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MODEL PERFORMANCE SIMULATION (STIMULUS-BASED, MULTI-RUN)")
    print("=" * 80)

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
    print(f"  Number of runs: {args.n_runs}")

    # Define reasonable parameter values
    model_params = {
        'qlearning': {
            'alpha_pos': 0.6,
            'alpha_neg': 0.3,
            'beta': 2.5
        },
        'wmrl': {
            'alpha_pos': 0.5,
            'alpha_neg': 0.3,
            'beta': 2.0,
            'beta_wm': 3.0,
            'capacity': 4,
            'phi': 0.2,
            'rho': 0.7
        }
    }

    # Process each model
    for model_name in args.models:
        if model_name not in model_params:
            print(f"\nSkipping unknown model: {model_name}")
            continue

        print("\n" + "=" * 80)
        print(f"SIMULATING {model_name.upper()} MODEL")
        print("=" * 80)

        params = model_params[model_name]
        print(f"\nUsing parameter values:")
        for param, value in params.items():
            print(f"  {param} = {value}")

        # Run multiple simulations
        all_predictions = run_multiple_simulations(
            behavioral_data,
            model_name,
            params,
            n_runs=args.n_runs,
            base_seed=args.seed
        )

        # Calculate performance
        overall_acc = all_predictions['correct'].mean()
        print(f"\n  Overall accuracy: {overall_acc:.3f}")

        print(f"\n  Performance by set size:")
        for ss in sorted(all_predictions['set_size'].unique()):
            ss_acc = all_predictions[all_predictions['set_size'] == ss]['correct'].mean()
            print(f"    Set Size {ss}: {ss_acc:.3f}")

        # Save predictions
        predictions_file = output_dir / f'{model_name}_predictions_stimulus_based_n{args.n_runs}.csv'
        all_predictions.to_csv(predictions_file, index=False)
        print(f"\n  [OK] Saved predictions: {predictions_file}")

        # Create visualizations
        print(f"\nCreating visualizations...")

        model_display_name = {
            'qlearning': 'Q-Learning',
            'wmrl': 'WM-RL Hybrid'
        }.get(model_name, model_name)

        # Combined analysis
        plot_combined_stimulus_analysis(
            all_predictions,
            model_display_name,
            n_encounters_threshold=args.threshold,
            save_dir=figure_dir
        )

        # Individual plots
        plot_stimulus_learning_curves(
            all_predictions,
            model_display_name,
            save_dir=figure_dir
        )

        plot_performance_by_stimulus_encounter(
            all_predictions,
            model_display_name,
            n_encounters_threshold=args.threshold,
            save_dir=figure_dir
        )

        # Performance summary
        def categorize_encounter(row):
            if not row['is_post_reversal']:
                if row['encounter_num'] < args.threshold:
                    return 'Early Stimulus Experience'
                else:
                    return 'Late Stimulus Experience'
            else:
                if row['encounters_since_reversal'] < args.threshold:
                    return 'Early Post-Reversal'
                else:
                    return 'Late Post-Reversal'

        all_predictions['position'] = all_predictions.apply(categorize_encounter, axis=1)

        print(f"\n  Performance by stimulus encounter position:")
        for pos in ['Early Stimulus Experience', 'Late Stimulus Experience',
                    'Early Post-Reversal', 'Late Post-Reversal']:
            pos_data = all_predictions[all_predictions['position'] == pos]
            if len(pos_data) > 0:
                acc = pos_data['correct'].mean()
                print(f"    {pos:30s}: {acc:.3f} (n={len(pos_data)})")

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE")
    print("=" * 80)
    print(f"\nFigures saved to: {figure_dir}")
    print(f"Predictions saved to: {output_dir}")
    print("\nGenerated plots:")
    print("  - Stimulus-based learning curves (encounters per stimulus)")
    print("  - Performance by stimulus encounter position")
    print(f"  - Averaged over {args.n_runs} runs")
    print("=" * 80)


if __name__ == '__main__':
    main()
