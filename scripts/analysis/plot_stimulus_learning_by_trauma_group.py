"""
Stimulus-Based Learning Curves by Trauma Group

Creates learning curves showing accuracy as a function of encounter number
with each stimulus, broken down by trauma group and set size.

Similar to the stimulus learning analysis but grouped by trauma.

Usage:
    python scripts/analysis/plot_stimulus_learning_by_trauma_group.py \
        --data output/task_trials_long_all_participants.csv \
        --output-dir figures/trauma_groups

Author: Generated for RLWM trauma analysis
Date: 2025-11-24
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import sys
warnings.filterwarnings('ignore')

# Import plotting config
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from plotting_config import PlotConfig

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_data_with_groups(
    data_path: Path,
    groups_path: Path,
    min_block: int = 3
) -> pd.DataFrame:
    """
    Load trial data and merge with trauma groups.

    Parameters
    ----------
    data_path : Path
        Path to trial data CSV
    groups_path : Path
        Path to trauma group assignments
    min_block : int
        Minimum block number to include (default: 3, excludes training)
    """
    print(f"\n>> Loading trial data: {data_path}")
    df_trials = pd.read_csv(data_path)
    n_participants_raw = df_trials['sona_id'].nunique()
    n_trials_raw = len(df_trials)

    # Filter out training blocks
    if min_block is not None:
        df_trials = df_trials[df_trials['block'] >= min_block].copy()
        print(f"  ✓ Loaded {n_trials_raw} trials from {n_participants_raw} participants")
        print(f"  ✓ Filtered to blocks >= {min_block}: {len(df_trials)} trials remain (excluded training)")

    print(f"\n>> Loading trauma groups: {groups_path}")
    df_groups = pd.read_csv(groups_path)

    # Filter to hypothesis groups only
    df_groups = df_groups[df_groups['hypothesis_group'] != 'Excluded_Low_High'].copy()
    print(f"  ✓ Using {len(df_groups)} participants in trauma groups")

    # Merge
    df_merged = df_trials.merge(
        df_groups[['sona_id', 'hypothesis_group', 'lec_total_events', 'ies_total']],
        on='sona_id',
        how='inner'
    )

    print(f"  ✓ Merged: {len(df_merged)} trials with trauma group labels")

    # Verify no set_size = 4
    if 4 in df_merged['set_size'].values:
        n_size4 = (df_merged['set_size'] == 4).sum()
        print(f"  WARNING: Found {n_size4} trials with set_size=4 (training data may not be fully excluded)")
    else:
        print(f"  ✓ Verified: No set_size=4 trials (training data successfully excluded)")

    return df_merged


def compute_stimulus_encounters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute encounter number for each stimulus presentation within each block.

    For each participant, block, and stimulus, label the 1st encounter, 2nd encounter, etc.
    This resets for each block, matching the task structure.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level data

    Returns
    -------
    pd.DataFrame
        Data with 'encounter_number' column added
    """
    print("\n>> Computing stimulus encounter numbers...")

    df = df.copy()

    # Sort by participant, block, stimulus, and trial order
    df = df.sort_values(['sona_id', 'block', 'stimulus', 'trial_in_block'])

    # Compute cumulative encounter count for each participant-block-stimulus triplet
    df['encounter_number'] = df.groupby(['sona_id', 'block', 'stimulus']).cumcount() + 1

    max_encounters = df['encounter_number'].max()
    print(f"  ✓ Computed encounter numbers (max: {max_encounters} encounters per block)")

    return df


def plot_learning_curves_by_group(df: pd.DataFrame, output_dir: Path):
    """
    Create stimulus learning curves faceted by trauma group and set size.

    Similar to the reference image but with trauma groups.
    """
    print("\n>> Creating stimulus learning curves by trauma group...")

    # Group colors
    group_colors = {
        'No Trauma': '#2ecc71',
        'Trauma - No Ongoing Impact': '#f39c12',
        'Trauma - Ongoing Impact': '#e74c3c'
    }

    # Set size colors (for line styles or markers)
    set_size_styles = {
        2: {'marker': 'o', 'linestyle': '-'},
        3: {'marker': 's', 'linestyle': '-'},
        5: {'marker': '^', 'linestyle': '-'},
        6: {'marker': 'D', 'linestyle': '-'}
    }

    # Create figure with subplots for each set size
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    set_sizes = sorted(df['set_size'].unique())

    for idx, set_size in enumerate(set_sizes):
        ax = axes[idx]

        # Filter to this set size
        df_size = df[df['set_size'] == set_size].copy()

        # Plot each trauma group
        for group in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
            df_group = df_size[df_size['hypothesis_group'] == group]

            if len(df_group) == 0:
                continue

            # Aggregate by encounter number
            learning_curve = df_group.groupby('encounter_number')['correct'].agg(['mean', 'sem', 'count']).reset_index()

            # Only plot if we have at least 5 participants contributing
            learning_curve = learning_curve[learning_curve['count'] >= 5]

            if len(learning_curve) == 0:
                continue

            # Plot
            ax.plot(
                learning_curve['encounter_number'],
                learning_curve['mean'] * 100,  # Convert to percentage
                color=group_colors[group],
                label=group,
                linewidth=2.5,
                marker=set_size_styles[set_size]['marker'],
                markersize=6,
                alpha=0.8
            )

            # Add error bars (SEM)
            ax.fill_between(
                learning_curve['encounter_number'],
                (learning_curve['mean'] - learning_curve['sem']) * 100,
                (learning_curve['mean'] + learning_curve['sem']) * 100,
                color=group_colors[group],
                alpha=0.2
            )

        # Formatting
        ax.set_xlabel('Encounter with Stimulus (Nth time)', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
        ax.set_title(f'Set Size {int(set_size)}', fontsize=PlotConfig.SUBTITLE_SIZE, fontweight='bold')
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_ylim(0, 100)
        ax.tick_params(axis='both', which='major', labelsize=PlotConfig.TICK_LABEL_SIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=PlotConfig.LEGEND_SIZE, framealpha=0.9)

    plt.suptitle('Stimulus Learning Curves by Trauma Group',
                 fontsize=PlotConfig.TITLE_SIZE, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    output_path = output_dir / 'stimulus_learning_curves_by_trauma_group.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")

    plt.close()


def plot_combined_learning_curves(df: pd.DataFrame, output_dir: Path):
    """
    Create a single panel with all set sizes and trauma groups.

    Uses different line styles for set sizes and colors for trauma groups.
    """
    print("\n>> Creating combined stimulus learning curves...")

    # Group colors
    group_colors = {
        'No Trauma': '#2ecc71',
        'Trauma - No Ongoing Impact': '#f39c12',
        'Trauma - Ongoing Impact': '#e74c3c'
    }

    # Set size line styles
    set_size_styles = {
        2: '-',
        3: '--',
        5: '-.',
        6: ':'
    }

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each combination
    for set_size in sorted(df['set_size'].unique()):
        df_size = df[df['set_size'] == set_size].copy()

        for group in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
            df_group = df_size[df_size['hypothesis_group'] == group]

            if len(df_group) == 0:
                continue

            # Aggregate by encounter
            learning_curve = df_group.groupby('encounter_number')['correct'].agg(['mean', 'sem', 'count']).reset_index()
            learning_curve = learning_curve[learning_curve['count'] >= 5]

            if len(learning_curve) == 0:
                continue

            # Plot
            label = f'{group} (SS={int(set_size)})'
            ax.plot(
                learning_curve['encounter_number'],
                learning_curve['mean'] * 100,
                color=group_colors[group],
                linestyle=set_size_styles[set_size],
                label=label,
                linewidth=2,
                alpha=0.7
            )

    # Formatting
    ax.set_xlabel('Encounter with Stimulus (Nth time)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
    ax.set_title('Stimulus Learning Curves: All Groups × Set Sizes',
                 fontsize=15, fontweight='bold', pad=20)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='Chance')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=9, framealpha=0.9, ncol=2)

    plt.tight_layout()

    # Save
    output_path = output_dir / 'stimulus_learning_curves_combined.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")

    plt.close()


def plot_early_vs_late_learning(df: pd.DataFrame, output_dir: Path):
    """
    Compare early learning (encounters 1-3) vs late learning (encounters 10+) by trauma group.
    """
    print("\n>> Creating early vs late learning comparison...")

    # Define early and late periods
    df_early = df[df['encounter_number'] <= 3].copy()
    df_late = df[df['encounter_number'] >= 10].copy()

    # Group colors
    group_colors = {
        'No Trauma': '#2ecc71',
        'Trauma - No Ongoing Impact': '#f39c12',
        'Trauma - Ongoing Impact': '#e74c3c'
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Early learning
    ax = axes[0]
    early_stats = df_early.groupby(['hypothesis_group', 'set_size'])['correct'].agg(['mean', 'sem']).reset_index()

    for group in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        group_data = early_stats[early_stats['hypothesis_group'] == group]
        ax.plot(
            group_data['set_size'],
            group_data['mean'] * 100,
            'o-',
            color=group_colors[group],
            label=group,
            linewidth=2.5,
            markersize=8
        )
        ax.fill_between(
            group_data['set_size'],
            (group_data['mean'] - group_data['sem']) * 100,
            (group_data['mean'] + group_data['sem']) * 100,
            color=group_colors[group],
            alpha=0.2
        )

    ax.set_xlabel('Set Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Early Learning (Encounters 1-3)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Late learning
    ax = axes[1]
    late_stats = df_late.groupby(['hypothesis_group', 'set_size'])['correct'].agg(['mean', 'sem']).reset_index()

    for group in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        group_data = late_stats[late_stats['hypothesis_group'] == group]
        if len(group_data) > 0:
            ax.plot(
                group_data['set_size'],
                group_data['mean'] * 100,
                'o-',
                color=group_colors[group],
                label=group,
                linewidth=2.5,
                markersize=8
            )
            ax.fill_between(
                group_data['set_size'],
                (group_data['mean'] - group_data['sem']) * 100,
                (group_data['mean'] + group_data['sem']) * 100,
                color=group_colors[group],
                alpha=0.2
            )

    ax.set_xlabel('Set Size', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Late Learning (Encounters 10+)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle('Early vs Late Learning by Trauma Group',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Save
    output_path = output_dir / 'stimulus_learning_early_vs_late.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")

    plt.close()


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics for stimulus-based learning."""
    print("\n" + "=" * 80)
    print("STIMULUS LEARNING STATISTICS BY TRAUMA GROUP")
    print("=" * 80)

    for group in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        group_data = df[df['hypothesis_group'] == group]

        print(f"\n{group}:")
        print(f"  N participants: {group_data['sona_id'].nunique()}")
        print(f"  N unique stimuli: {group_data['stimulus'].nunique()}")
        print(f"  Max encounters per stimulus: {group_data['encounter_number'].max()}")

        # Early vs late performance
        early_acc = group_data[group_data['encounter_number'] <= 3]['correct'].mean()
        late_acc = group_data[group_data['encounter_number'] >= 10]['correct'].mean()

        print(f"  Early learning (1-3 encounters): {early_acc:.3f}")
        print(f"  Late learning (10+ encounters): {late_acc:.3f}")
        if not np.isnan(late_acc) and not np.isnan(early_acc):
            improvement = late_acc - early_acc
            print(f"  Learning gain: {improvement:.3f} ({improvement*100:.1f}%)")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Create stimulus-based learning curves by trauma group'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='output/task_trials_long_all_participants.csv',
        help='Path to trial data CSV'
    )
    parser.add_argument(
        '--trauma-groups',
        type=str,
        default='output/trauma_groups/group_assignments.csv',
        help='Path to trauma group assignments'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='figures/trauma_groups',
        help='Output directory for figures'
    )
    parser.add_argument(
        '--min-block',
        type=int,
        default=3,
        help='Minimum block number to include (default: 3, excludes training)'
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STIMULUS LEARNING CURVES BY TRAUMA GROUP")
    print("=" * 80)

    # Load data
    df = load_data_with_groups(
        Path(args.data),
        Path(args.trauma_groups),
        min_block=args.min_block
    )

    # Compute encounter numbers
    df = compute_stimulus_encounters(df)

    # Print statistics
    print_summary_statistics(df)

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_learning_curves_by_group(df, output_dir)
    plot_combined_learning_curves(df, output_dir)
    plot_early_vs_late_learning(df, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}/")
    print("\nGenerated visualizations:")
    print("  1. stimulus_learning_curves_by_trauma_group.png - Separate panels per set size")
    print("  2. stimulus_learning_curves_combined.png - All groups and set sizes overlaid")
    print("  3. stimulus_learning_early_vs_late.png - Early vs late learning comparison")
    print()


if __name__ == '__main__':
    main()
