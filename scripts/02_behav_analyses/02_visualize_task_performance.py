#!/usr/bin/env python
"""
06: Visualize Task Performance
==============================

Creates stimulus-based learning curves and performance visualizations
from human behavioral data.

This script generates the same visualizations as the model predictions,
but using actual human behavioral data.

Inputs:
    - data/processed/task_trials_long.csv (or custom via --data)
    - reports/tables/trauma_groups/group_assignments.csv (optional via --trauma-groups)

Outputs:
    - reports/figures/behavioral_summary/human_stimulus_learning_curve.png
    - reports/figures/behavioral_summary/human_stimulus_encounter_performance.png
    - reports/figures/behavioral_summary/human_stimulus_performance_analysis.png
    - reports/tables/behavioral_summary/human_stimulus_based_data.csv

Usage:
    # Basic visualization
    python scripts/02_behav_analyses/02_visualize_task_performance.py

    # With custom data file
    python scripts/02_behav_analyses/02_visualize_task_performance.py --data data/processed/task_trials_long_all.csv

    # With trauma group comparisons
    python scripts/02_behav_analyses/02_visualize_task_performance.py --trauma-groups reports/tables/trauma_groups/group_assignments.csv

    # Custom encounter threshold
    python scripts/02_behav_analyses/02_visualize_task_performance.py --threshold 5

Next Steps:
    - Run 03_analyze_trauma_groups.py to create trauma groupings
    - Run 04_run_statistical_analyses.py for statistical tests
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from collections import defaultdict
import argparse

# Add project root to path (parents[2] = project root; parents[1] = scripts/)
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from config import (
    TaskParams,
    DataParams,
    PROCESSED_DIR,
    REPORTS_FIGURES_DIR,
    REPORTS_TABLES_BEHAVIORAL,
    AnalysisParams,
)


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

    # Filter out practice blocks (block < 3)
    trials_df = trials_df[trials_df['block'] >= 3].copy()

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

            # Track correct streak for reversal detection
            if correct:
                stimulus_correct_streak[stimulus] += 1
                # Check if reversal threshold reached - mark NEXT trial as post-reversal
                if (TaskParams.REVERSAL_MIN <= stimulus_correct_streak[stimulus] <= TaskParams.REVERSAL_MAX):
                    if not stimulus_reversal_occurred[stimulus]:
                        # Reversal triggered! Next encounter will be post-reversal
                        stimulus_reversal_occurred[stimulus] = True
                        stimulus_encounters_since_reversal[stimulus] = 0
                        stimulus_correct_streak[stimulus] = 0  # Reset streak after reversal
            else:
                stimulus_correct_streak[stimulus] = 0

            # Determine if current trial is post-reversal
            is_post_reversal_trial = stimulus_reversal_occurred[stimulus] and stimulus_encounters_since_reversal[stimulus] > 0

            # Increment encounters since reversal (for next trial)
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
                'is_post_reversal': is_post_reversal_trial,
                'rt': row.get('rt', np.nan)
            })

    return pd.DataFrame(processed)


def plot_human_stimulus_learning_curves(
    data_df: pd.DataFrame,
    save_dir: Path,
    by_trauma_group: bool = False
):
    """
    Plot stimulus-based learning curves for human data.

    Parameters
    ----------
    data_df : pd.DataFrame
        Processed human data with encounter tracking
    save_dir : Path
        Directory to save figure
    by_trauma_group : bool
        If True, create separate plots for each trauma group
    """
    colors = AnalysisParams.COLORS_SET_SIZE
    set_sizes = sorted(data_df['set_size'].unique())
    n_participants = data_df['subject_id'].nunique()

    # Trauma group colors
    trauma_colors = {
        'No Trauma': '#2ecc71',
        'Trauma - No Ongoing Impact': '#f39c12',
        'Trauma - Ongoing Impact': '#e74c3c'
    }

    if by_trauma_group and 'trauma_group' in data_df.columns:
        # Create subplots for each set size
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for idx, ss in enumerate(set_sizes):
            ax = axes[idx]
            ss_data = data_df[data_df['set_size'] == ss].copy()

            # Plot each trauma group
            for group in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
                group_data = ss_data[ss_data['trauma_group'] == group]

                if len(group_data) == 0:
                    continue

                # Group by encounter
                grouped = group_data.groupby('encounter_num')['correct'].agg(['mean', 'std', 'count']).reset_index()
                grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])
                grouped = grouped[grouped['count'] >= 3]

                if len(grouped) == 0:
                    continue

                # Plot
                ax.errorbar(
                    grouped['encounter_num'],
                    grouped['mean'] * 100,
                    yerr=grouped['sem'] * 100,
                    fmt='o-',
                    color=trauma_colors[group],
                    label=group,
                    linewidth=2.5,
                    markersize=6,
                    alpha=0.8,
                    capsize=3,
                    capthick=1.5
                )

            ax.set_xlabel('Encounter with Stimulus (Nth time)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
            ax.set_ylim([0, 105])
            ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

            # Add reversal marker
            ax.axvline(12, color='red', linestyle=':', linewidth=2, alpha=0.7)
            if idx == 0:  # Only add text label on first subplot
                ax.text(12.2, 100, 'Reversals\nStart', fontsize=8, color='red', va='top', fontweight='bold')

            ax.legend(loc='lower right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Set Size {ss}', fontsize=13, fontweight='bold')

        plt.suptitle(f'Stimulus Learning Curves by Trauma Group (N={n_participants})',
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        # Save
        save_path = save_dir / 'human_stimulus_learning_curve_by_trauma_group.png'
        plt.savefig(save_path, dpi=AnalysisParams.FIG_DPI, format=AnalysisParams.FIG_FORMAT, bbox_inches='tight')
        print(f"  [OK] Saved: {save_path}")
        plt.close()

    else:
        # Original plot (all participants together)
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

        # Add reversal marker
        ax.axvline(12, color='red', linestyle=':', linewidth=2, alpha=0.7)
        ax.text(12.2, 100, 'Reversals Start', fontsize=9, color='red', va='top', fontweight='bold')

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
        Threshold for early/late (default=4)
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
            return 'Post-Reversal'

    df = data_df.copy()
    df['encounter_position'] = df.apply(categorize_encounter, axis=1)

    # Define position order
    position_order = [
        'Early Stim\nExperience',
        'Late Stim\nExperience',
        'Post-Reversal'
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


def plot_encounter_performance_by_trauma_group(
    data_df: pd.DataFrame,
    n_encounters_threshold: int,
    save_dir: Path
):
    """
    Plot performance by stimulus encounter position, separated by trauma group.

    Parameters
    ----------
    data_df : pd.DataFrame
        Processed human data with trauma_group column
    n_encounters_threshold : int
        Threshold for early/late (default=4)
    save_dir : Path
        Save directory
    """
    if 'trauma_group' not in data_df.columns:
        print("  [SKIP] No trauma group data available")
        return

    # Trauma group colors
    trauma_colors = {
        'No Trauma': '#2ecc71',
        'Trauma - No Ongoing Impact': '#f39c12',
        'Trauma - Ongoing Impact': '#e74c3c'
    }

    # Categorize by stimulus encounter position
    def categorize_encounter(row):
        if not row['is_post_reversal']:
            if row['encounter_num'] < n_encounters_threshold:
                return 'Early Stim\nExperience'
            else:
                return 'Late Stim\nExperience'
        else:
            return 'Post-Reversal'

    df = data_df.copy()
    df['encounter_position'] = df.apply(categorize_encounter, axis=1)

    # Define position order
    position_order = [
        'Early Stim\nExperience',
        'Late Stim\nExperience',
        'Post-Reversal'
    ]

    # Group by trauma group and position
    grouped = df.groupby(['trauma_group', 'encounter_position'])['correct'].agg(['mean', 'std', 'count']).reset_index()
    grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])

    # Set up figure
    fig, ax = plt.subplots(figsize=(12, 7))

    n_positions = len(position_order)
    trauma_groups = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    n_groups = len(trauma_groups)
    bar_width = 0.8 / n_groups
    x = np.arange(n_positions)

    # Plot bars for each trauma group
    for i, group in enumerate(trauma_groups):
        group_data = grouped[grouped['trauma_group'] == group].copy()
        group_data = group_data.set_index('encounter_position').reindex(position_order).reset_index()
        group_data['mean'] = group_data['mean'].fillna(0)
        group_data['sem'] = group_data['sem'].fillna(0)

        offset = (i - n_groups/2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            group_data['mean'] * 100,
            bar_width,
            yerr=group_data['sem'] * 100,
            label=group,
            color=trauma_colors[group],
            alpha=0.8,
            capsize=4,
            error_kw={'linewidth': 1.5}
        )

        # Add value labels
        for bar, acc in zip(bars, group_data['mean'] * 100):
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
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_title(
        f'Performance by Encounter Position and Trauma Group\n(Threshold = {n_encounters_threshold} encounters)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )

    plt.tight_layout()

    # Save
    save_path = save_dir / 'encounter_performance_by_trauma_group.png'
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

    # Add reversal marker
    ax1.axvline(12, color='red', linestyle=':', linewidth=2, alpha=0.7)
    ax1.text(12.2, 100, 'Reversals Start', fontsize=9, color='red', va='top', fontweight='bold')

    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Stimulus Learning Curves', fontsize=12, fontweight='bold')

    # ===== RIGHT: Performance by encounter position =====
    ax2 = fig.add_subplot(gs[0, 1])

    def categorize_encounter(row):
        if not row['is_post_reversal']:
            if row['encounter_num'] < n_encounters_threshold:
                return 'Early Stim\nExperience'
            else:
                return 'Late Stim\nExperience'
        else:
            return 'Post-Reversal'

    df = data_df.copy()
    df['position'] = df.apply(categorize_encounter, axis=1)

    position_order = ['Early Stim\nExperience', 'Late Stim\nExperience', 'Post-Reversal']
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
            capsize=3,
            error_kw={'elinewidth': 1.5}
        )

    ax2.set_xlabel('Stimulus Encounter Position', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_ylim([0, 110])
    ax2.set_xticks(x)
    ax2.set_xticklabels(position_order, fontsize=9, rotation=15, ha='right')
    ax2.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_title('Performance by Encounter Position', fontsize=12, fontweight='bold')

    plt.suptitle(f'Performance Analysis (Stimulus-Based, N={n_participants})', fontsize=14, fontweight='bold', y=1.02)

    # Save
    save_path = save_dir / 'human_stimulus_performance_analysis.png'
    plt.savefig(save_path, dpi=AnalysisParams.FIG_DPI, format=AnalysisParams.FIG_FORMAT, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize human performance with stimulus-based learning curves'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=str(PROCESSED_DIR / 'task_trials_long_all_participants.csv'),
        help='Path to task trials data'
    )
    parser.add_argument(
        '--threshold',
        type=int,
        default=4,
        help='Encounter threshold for early/late classification'
    )
    parser.add_argument(
        '--trauma-groups',
        type=str,
        default=None,
        help='Path to trauma group assignments CSV (optional, for trauma group analysis)'
    )

    args = parser.parse_args()

    # Paths
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = project_root / data_path
    output_dir = REPORTS_TABLES_BEHAVIORAL
    figure_dir = REPORTS_FIGURES_DIR / 'behavioral_summary'

    # Use trauma_groups directory if trauma groups provided
    if args.trauma_groups:
        figure_dir = REPORTS_FIGURES_DIR / 'trauma_groups'

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

    # Load trauma groups if provided
    if args.trauma_groups:
        print(f"\nLoading trauma groups: {args.trauma_groups}")
        trauma_groups_df = pd.read_csv(args.trauma_groups)
        trauma_groups_df = trauma_groups_df[trauma_groups_df['hypothesis_group'] != 'Excluded_Low_High'].copy()
        print(f"  Loaded trauma groups for {len(trauma_groups_df)} participants (excluding Low-High)")

        # Merge with trials
        trials_df = trials_df.merge(
            trauma_groups_df[['sona_id', 'hypothesis_group']],
            on='sona_id',
            how='inner'
        )
        print(f"  After merging: {len(trials_df)} trials with trauma group labels")

    # Filter to main task blocks only (exclude practice blocks 1-2)
    trials_df = trials_df[trials_df['block'] >= DataParams.MAIN_TASK_START_BLOCK].copy()
    print(f"  After filtering to main task blocks (>={DataParams.MAIN_TASK_START_BLOCK}): {len(trials_df)} trials")

    # Process to track stimulus encounters
    print("\nProcessing stimulus-based encounter tracking...")
    processed_df = process_human_data_stimulus_based(trials_df)

    # Add trauma group column if it was in original data
    if 'hypothesis_group' in trials_df.columns:
        # Map subject_id to trauma group
        trauma_map = trials_df[['sona_id', 'hypothesis_group']].drop_duplicates().set_index('sona_id')['hypothesis_group'].to_dict()
        processed_df['trauma_group'] = processed_df['subject_id'].map(trauma_map)

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
        save_dir=figure_dir,
        by_trauma_group=args.trauma_groups is not None
    )

    plot_human_encounter_performance(
        processed_df,
        n_encounters_threshold=args.threshold,
        save_dir=figure_dir
    )

    # Create trauma group encounter performance plot if trauma groups provided
    if args.trauma_groups is not None:
        plot_encounter_performance_by_trauma_group(
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
            return 'Post-Reversal'

    processed_df['position'] = processed_df.apply(categorize_encounter, axis=1)

    print(f"\n  Performance by stimulus encounter position:")
    for pos in ['Early Stimulus Experience', 'Late Stimulus Experience', 'Post-Reversal']:
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
