"""
Visualize human behavioral data with stimulus-based learning curves.

This script generates the same visualizations as the model predictions,
but using actual human behavioral data.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from config import TaskParams, OUTPUT_DIR, FIGURES_DIR, AnalysisParams


def process_human_data_stimulus_based(trials_df):
    """
    Process human data to track encounters per stimulus.

    Parameters
    ----------
    trials_df : pd.DataFrame
        Task trials data

    Returns
    -------
    pd.DataFrame
        Processed data with encounter tracking
    """
    processed = []

    # Process each participant and block
    for (subject_id, block_id), block_data in trials_df.groupby(['sona_id', 'block']):
        block_data = block_data.sort_values('trial_in_block').reset_index(drop=True)

        # Track encounters per stimulus
        stimulus_encounter_count = defaultdict(int)
        stimulus_reversal_occurred = defaultdict(bool)
        stimulus_encounters_since_reversal = defaultdict(int)
        stimulus_correct_streak = defaultdict(int)

        for idx, row in block_data.iterrows():
            # Get stimulus (convert from 1-indexed to 0-indexed for consistency)
            stimulus = int(float(row['stimulus'])) - 1
            set_size = int(float(row['set_size']))
            correct = int(float(row['correct']))

            # Increment encounter count
            stimulus_encounter_count[stimulus] += 1
            encounter_num = stimulus_encounter_count[stimulus]

            # Track reversals per stimulus
            if correct:
                stimulus_correct_streak[stimulus] += 1
                if (TaskParams.REVERSAL_MIN <= stimulus_correct_streak[stimulus] <= TaskParams.REVERSAL_MAX):
                    if not stimulus_reversal_occurred[stimulus]:
                        stimulus_reversal_occurred[stimulus] = True
                        stimulus_encounters_since_reversal[stimulus] = 0
            else:
                stimulus_correct_streak[stimulus] = 0

            # Increment encounters since reversal
            if stimulus_reversal_occurred[stimulus]:
                stimulus_encounters_since_reversal[stimulus] += 1

            # Record
            processed.append({
                'subject_id': subject_id,
                'block': block_id,
                'trial': row['trial_in_block'],
                'set_size': set_size,
                'stimulus': stimulus,
                'encounter_num': encounter_num,
                'correct': correct,
                'encounters_since_reversal': stimulus_encounters_since_reversal[stimulus],
                'is_post_reversal': stimulus_reversal_occurred[stimulus],
                'rt': row.get('rt', np.nan)
            })

    return pd.DataFrame(processed)


def plot_human_stimulus_learning_curves(
    data_df: pd.DataFrame,
    save_dir: Path
):
    """
    Plot stimulus-based learning curves for human data.

    Parameters
    ----------
    data_df : pd.DataFrame
        Processed human data with encounter tracking
    save_dir : Path
        Directory to save figure
    """
    colors = AnalysisParams.COLORS_SET_SIZE
    set_sizes = sorted(data_df['set_size'].unique())
    n_participants = data_df['subject_id'].nunique()

    fig, ax = plt.subplots(figsize=(12, 6))

    for ss in set_sizes:
        ss_data = data_df[data_df['set_size'] == ss].copy()

        # Group by encounter number and calculate mean, std, sem
        grouped = ss_data.groupby('encounter_num')['correct'].agg(['mean', 'std', 'count']).reset_index()
        grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])

        # Only plot encounters with sufficient data
        grouped = grouped[grouped['count'] >= 3]

        if len(grouped) == 0:
            continue

        # Plot line with error bars (SEM)
        ax.errorbar(
            grouped['encounter_num'],
            grouped['mean'] * 100,
            yerr=grouped['sem'] * 100,
            fmt='o-',
            color=colors[ss],
            label=f'Set Size {ss}',
            linewidth=2,
            markersize=5,
            alpha=0.8,
            capsize=3,
            capthick=1.5
        )

    # Styling
    ax.set_xlabel('Encounter with Stimulus (Nth time)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Chance')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Human Performance: Stimulus Learning Curves (N={n_participants})', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save
    save_path = save_dir / 'human_stimulus_learning_curve.png'
    plt.savefig(save_path, dpi=AnalysisParams.FIG_DPI, format=AnalysisParams.FIG_FORMAT, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()


def plot_human_encounter_performance(
    data_df: pd.DataFrame,
    n_encounters_threshold: int,
    save_dir: Path
):
    """
    Plot performance by stimulus encounter position for human data.

    Parameters
    ----------
    data_df : pd.DataFrame
        Processed human data
    n_encounters_threshold : int
        Threshold for early/late
    save_dir : Path
        Save directory
    """
    colors = AnalysisParams.COLORS_SET_SIZE
    set_sizes = sorted(data_df['set_size'].unique())
    n_participants = data_df['subject_id'].nunique()

    # Categorize by stimulus encounter position
    def categorize_encounter(row):
        if not row['is_post_reversal']:
            if row['encounter_num'] < n_encounters_threshold:
                return 'Early Stim\nExperience'
            else:
                return 'Late Stim\nExperience'
        else:
            if row['encounters_since_reversal'] < n_encounters_threshold:
                return 'Early\nPost-Reversal'
            else:
                return 'Late\nPost-Reversal'

    df = data_df.copy()
    df['encounter_position'] = df.apply(categorize_encounter, axis=1)

    # Define position order
    position_order = [
        'Early Stim\nExperience',
        'Late Stim\nExperience',
        'Early\nPost-Reversal',
        'Late\nPost-Reversal'
    ]

    # Group by set size and position
    grouped = df.groupby(['set_size', 'encounter_position'])['correct'].agg(['mean', 'std', 'count']).reset_index()
    grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])

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
            yerr=ss_data['sem'] * 100,
            label=f'Set Size {ss}',
            color=colors[ss],
            alpha=0.8,
            capsize=4,
            error_kw={'linewidth': 1.5}
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
        f'Human Performance: Performance by Stimulus Encounter Position (N={n_participants})\n(Threshold = {n_encounters_threshold} encounters)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    plt.tight_layout()

    # Save
    save_path = save_dir / 'human_stimulus_encounter_performance.png'
    plt.savefig(save_path, dpi=AnalysisParams.FIG_DPI, format=AnalysisParams.FIG_FORMAT, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()


def plot_human_combined_analysis(
    data_df: pd.DataFrame,
    n_encounters_threshold: int,
    save_dir: Path
):
    """Create combined two-panel figure for human data."""
    colors = AnalysisParams.COLORS_SET_SIZE
    set_sizes = sorted(data_df['set_size'].unique())
    n_participants = data_df['subject_id'].nunique()

    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.3)

    # ===== LEFT: Learning curve by stimulus encounters =====
    ax1 = fig.add_subplot(gs[0, 0])

    for ss in set_sizes:
        ss_data = data_df[data_df['set_size'] == ss].copy()
        grouped = ss_data.groupby('encounter_num')['correct'].agg(['mean', 'std', 'count']).reset_index()
        grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
        grouped = grouped[grouped['count'] >= 3]

        if len(grouped) == 0:
            continue

        ax1.errorbar(
            grouped['encounter_num'],
            grouped['mean'] * 100,
            yerr=grouped['sem'] * 100,
            fmt='o-',
            color=colors[ss],
            label=f'Set Size {ss}',
            linewidth=2,
            markersize=5,
            alpha=0.8,
            capsize=3,
            capthick=1.5
        )

    ax1.set_xlabel('Encounter with Stimulus (Nth time)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Human: Stimulus Learning Curves', fontsize=12, fontweight='bold')

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

    df = data_df.copy()
    df['position'] = df.apply(categorize_encounter, axis=1)

    position_order = ['Early Stim', 'Late Stim', 'Early Post-Rev', 'Late Post-Rev']
    grouped = df.groupby(['set_size', 'position'])['correct'].agg(['mean']).reset_index()

    n_positions = len(position_order)
    n_set_sizes = len(set_sizes)
    bar_width = 0.8 / n_set_sizes
    x = np.arange(n_positions)

    for i, ss in enumerate(set_sizes):
        ss_data = grouped[grouped['set_size'] == ss].copy()
        ss_data = ss_data.set_index('position').reindex(position_order).reset_index()
        ss_data['mean'] = ss_data['mean'].fillna(0)

        offset = (i - n_set_sizes/2 + 0.5) * bar_width
        ax2.bar(
            x + offset,
            ss_data['mean'] * 100,
            bar_width,
            label=f'Set Size {ss}',
            color=colors[ss],
            alpha=0.8
        )

    ax2.set_xlabel('Stimulus Encounter Position', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_ylim([0, 110])
    ax2.set_xticks(x)
    ax2.set_xticklabels(position_order, fontsize=9, rotation=15, ha='right')
    ax2.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_title('Human: Performance by Encounter Position', fontsize=12, fontweight='bold')

    plt.suptitle(f'Human Performance Analysis (Stimulus-Based, N={n_participants})', fontsize=14, fontweight='bold', y=1.02)

    # Save
    save_path = save_dir / 'human_stimulus_performance_analysis.png'
    plt.savefig(save_path, dpi=AnalysisParams.FIG_DPI, format=AnalysisParams.FIG_FORMAT, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize human performance with stimulus-based learning curves'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='output/task_trials_long.csv',
        help='Path to task trials data'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=4,
        help='Encounter threshold for early/late classification'
    )

    args = parser.parse_args()

    # Paths
    data_path = project_root / args.data
    output_dir = OUTPUT_DIR / 'behavioral_summary'
    figure_dir = FIGURES_DIR / 'behavioral_summary'

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("HUMAN PERFORMANCE VISUALIZATION (STIMULUS-BASED)")
    print("=" * 80)

    # Load data
    print(f"\nLoading task trials: {data_path}")
    trials_df = pd.read_csv(data_path)
    print(f"  Loaded {len(trials_df)} trials")
    print(f"  Participants: {trials_df['sona_id'].nunique()}")

    # Filter to main task blocks only (exclude practice blocks 1-2)
    from config import DataParams
    trials_df = trials_df[trials_df['block'] >= DataParams.MAIN_TASK_START_BLOCK].copy()
    print(f"  After filtering to main task blocks (>={DataParams.MAIN_TASK_START_BLOCK}): {len(trials_df)} trials")

    # Process to track stimulus encounters
    print("\nProcessing stimulus-based encounter tracking...")
    processed_df = process_human_data_stimulus_based(trials_df)

    # Calculate performance
    overall_acc = processed_df['correct'].mean()
    print(f"  Overall accuracy: {overall_acc:.3f}")

    print(f"\n  Performance by set size:")
    for ss in sorted(processed_df['set_size'].unique()):
        ss_acc = processed_df[processed_df['set_size'] == ss]['correct'].mean()
        print(f"    Set Size {ss}: {ss_acc:.3f}")

    # Save processed data
    processed_file = output_dir / 'human_stimulus_based_data.csv'
    processed_df.to_csv(processed_file, index=False)
    print(f"\n  [OK] Saved processed data: {processed_file}")

    # Create visualizations
    print(f"\nCreating visualizations...")

    plot_human_combined_analysis(
        processed_df,
        n_encounters_threshold=args.threshold,
        save_dir=figure_dir
    )

    plot_human_stimulus_learning_curves(
        processed_df,
        save_dir=figure_dir
    )

    plot_human_encounter_performance(
        processed_df,
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

    processed_df['position'] = processed_df.apply(categorize_encounter, axis=1)

    print(f"\n  Performance by stimulus encounter position:")
    for pos in ['Early Stimulus Experience', 'Late Stimulus Experience',
                'Early Post-Reversal', 'Late Post-Reversal']:
        pos_data = processed_df[processed_df['position'] == pos]
        if len(pos_data) > 0:
            acc = pos_data['correct'].mean()
            print(f"    {pos:30s}: {acc:.3f} (n={len(pos_data)})")

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nFigures saved to: {figure_dir}")
    print(f"Processed data saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
