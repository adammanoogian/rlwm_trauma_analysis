"""
Analyze Learning Trajectories by Trauma Group

Creates visualizations showing how learning curves differ across
trauma groups (hypothesis-driven A/B/C classification).

Visualizations:
1. All participants' learning curves colored by trauma group
2. Group-specific panels showing mean ± SEM with individual trajectories
3. Performance by block and set size (cognitive load)
4. Reaction time trajectories by group

Usage:
    python scripts/analysis/analyze_learning_by_trauma_group.py \
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
warnings.filterwarnings('ignore')

# Import plotting config
import sys
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
        Minimum block number to include (default: 3, excludes training blocks 1-2)
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
    else:
        print(f"  ✓ Loaded {len(df_trials)} trials from {df_trials['sona_id'].nunique()} participants")

    print(f"\n>> Loading trauma groups: {groups_path}")
    df_groups = pd.read_csv(groups_path)

    # Filter to hypothesis groups only
    df_groups = df_groups[df_groups['hypothesis_group'] != 'Excluded_Low_High'].copy()
    print(f"  ✓ Using {len(df_groups)} participants in groups A/B/C")

    # Merge
    df_merged = df_trials.merge(df_groups[['sona_id', 'hypothesis_group', 'lec_total_events', 'ies_total']],
                                  on='sona_id',
                                  how='inner')

    print(f"  ✓ Merged: {len(df_merged)} trials with trauma group labels")

    # Verify no set_size = 4 (sanity check for training data exclusion)
    if 4 in df_merged['set_size'].values:
        n_size4 = (df_merged['set_size'] == 4).sum()
        print(f"  WARNING: Found {n_size4} trials with set_size=4 (training data may not be fully excluded)")
    else:
        print(f"  ✓ Verified: No set_size=4 trials (training data successfully excluded)")

    return df_merged


def compute_rolling_accuracy(
    df: pd.DataFrame,
    window: int = 20,
    group_col: str = 'sona_id'
) -> pd.DataFrame:
    """
    Compute rolling accuracy for each participant.

    Parameters
    ----------
    df : pd.DataFrame
        Trial-level data
    window : int
        Rolling window size (number of trials)
    group_col : str
        Column to group by (e.g., 'sona_id')

    Returns
    -------
    pd.DataFrame
        Data with rolling accuracy column
    """
    print(f"\n>> Computing rolling accuracy (window={window})...")

    df = df.copy()
    df['correct_binary'] = df['correct'].astype(int)

    # Compute rolling accuracy per participant
    df['rolling_accuracy'] = df.groupby(group_col)['correct_binary'].transform(
        lambda x: x.rolling(window=window, min_periods=1, center=False).mean()
    )

    return df


def plot_all_learning_curves(df: pd.DataFrame, output_dir: Path):
    """
    Plot all individual learning curves colored by trauma group.

    This creates a "spaghetti plot" showing individual trajectories.
    """
    print("\n>> Creating all-participants learning curves...")

    # Group colors
    group_colors = {
        'No Trauma': '#2ecc71',
        'Trauma - No Ongoing Impact': '#f39c12',
        'Trauma - Ongoing Impact': '#e74c3c'
    }

    group_labels = {
        'No Trauma': 'No Trauma',
        'Trauma - No Ongoing Impact': 'Trauma - No Ongoing Impact',
        'Trauma - Ongoing Impact': 'Trauma - Ongoing Impact'
    }

    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot each participant
    for group in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        group_data = df[df['hypothesis_group'] == group]

        for participant in group_data['sona_id'].unique():
            pdata = group_data[group_data['sona_id'] == participant]
            pdata = pdata.sort_values('trial_in_experiment')

            ax.plot(
                pdata['trial_in_experiment'],
                pdata['rolling_accuracy'],
                color=group_colors[group],
                alpha=0.3,
                linewidth=1
            )

    # Add group means
    for group in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        group_data = df[df['hypothesis_group'] == group]

        # Bin trials and compute mean per bin
        bins = np.arange(0, group_data['trial_in_experiment'].max() + 50, 50)
        group_data['trial_bin'] = pd.cut(group_data['trial_in_experiment'], bins=bins)

        binned_means = group_data.groupby('trial_bin')['rolling_accuracy'].mean()
        bin_centers = [(interval.left + interval.right) / 2 for interval in binned_means.index]

        ax.plot(
            bin_centers,
            binned_means.values,
            color=group_colors[group],
            linewidth=3,
            label=group_labels[group],
            alpha=0.9
        )

    # Formatting
    ax.set_xlabel('Trial Number', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Rolling Accuracy (20-trial window)', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
    ax.set_title('Learning Trajectories by Trauma Group\n(Individual lines + group means)',
                 fontsize=PlotConfig.TITLE_SIZE, fontweight='bold', pad=PlotConfig.PAD)
    ax.legend(loc='lower right', fontsize=PlotConfig.LEGEND_SIZE, framealpha=0.9)
    ax.tick_params(axis='both', which='major', labelsize=PlotConfig.TICK_LABEL_SIZE)
    ax.grid(True, alpha=PlotConfig.GRID_ALPHA)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')

    plt.tight_layout()

    # Save
    output_path = output_dir / 'learning_curves_all_participants.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")

    plt.close()


def plot_group_specific_curves(df: pd.DataFrame, output_dir: Path):
    """
    Create separate panels for each trauma group showing individual and mean trajectories.
    """
    print("\n>> Creating group-specific learning curve panels...")

    group_colors = {
        'No Trauma': '#2ecc71',
        'Trauma - No Ongoing Impact': '#f39c12',
        'Trauma - Ongoing Impact': '#e74c3c'
    }

    group_labels = {
        'No Trauma': 'No Trauma',
        'Trauma - No Ongoing Impact': 'Trauma - No Ongoing Impact',
        'Trauma - Ongoing Impact': 'Trauma - Ongoing Impact'
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

    for idx, group in enumerate(['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']):
        ax = axes[idx]
        group_data = df[df['hypothesis_group'] == group]

        n_participants = group_data['sona_id'].nunique()

        # Plot individual trajectories
        for participant in group_data['sona_id'].unique():
            pdata = group_data[group_data['sona_id'] == participant]
            pdata = pdata.sort_values('trial_in_experiment')

            ax.plot(
                pdata['trial_in_experiment'],
                pdata['rolling_accuracy'],
                color=group_colors[group],
                alpha=0.4,
                linewidth=1.5
            )

        # Compute and plot group mean ± SEM
        bins = np.arange(0, group_data['trial_in_experiment'].max() + 50, 50)
        group_data['trial_bin'] = pd.cut(group_data['trial_in_experiment'], bins=bins)

        binned_stats = group_data.groupby('trial_bin')['rolling_accuracy'].agg(['mean', 'sem'])
        bin_centers = [(interval.left + interval.right) / 2 for interval in binned_stats.index]

        # Mean line
        ax.plot(
            bin_centers,
            binned_stats['mean'].values,
            color=group_colors[group],
            linewidth=4,
            label=f'Group Mean (n={n_participants})',
            alpha=1.0
        )

        # SEM shading
        ax.fill_between(
            bin_centers,
            binned_stats['mean'] - binned_stats['sem'],
            binned_stats['mean'] + binned_stats['sem'],
            color=group_colors[group],
            alpha=0.2
        )

        # Formatting
        ax.set_xlabel('Trial Number', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Rolling Accuracy', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
        ax.set_title(group_labels[group], fontsize=PlotConfig.SUBTITLE_SIZE, fontweight='bold')
        ax.legend(loc='lower right', fontsize=PlotConfig.LEGEND_SIZE)
        ax.tick_params(axis='both', which='major', labelsize=PlotConfig.TICK_LABEL_SIZE)
        ax.grid(True, alpha=PlotConfig.GRID_ALPHA)
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

    plt.suptitle('Learning Trajectories: Detailed by Trauma Group',
                 fontsize=PlotConfig.TITLE_SIZE, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save
    output_path = output_dir / 'learning_curves_by_group_panels.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")

    plt.close()


def plot_performance_by_load(df: pd.DataFrame, output_dir: Path):
    """
    Plot accuracy and RT by set size (cognitive load) and trauma group.
    """
    print("\n>> Creating performance by cognitive load plots...")

    group_colors = {
        'No Trauma': '#2ecc71',
        'Trauma - No Ongoing Impact': '#f39c12',
        'Trauma - Ongoing Impact': '#e74c3c'
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Panel 1: Accuracy by set size
    ax = axes[0, 0]
    for group in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        group_data = df[df['hypothesis_group'] == group]

        # Compute mean accuracy per set size
        acc_by_size = group_data.groupby('set_size')['correct'].agg(['mean', 'sem']).reset_index()

        ax.plot(
            acc_by_size['set_size'],
            acc_by_size['mean'],
            'o-',
            color=group_colors[group],
            linewidth=2,
            markersize=8,
            label=group.replace('_', ' ')
        )

        ax.fill_between(
            acc_by_size['set_size'],
            acc_by_size['mean'] - acc_by_size['sem'],
            acc_by_size['mean'] + acc_by_size['sem'],
            color=group_colors[group],
            alpha=0.2
        )

    ax.set_xlabel('Set Size (Number of Stimuli)', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
    ax.set_title('Accuracy by Cognitive Load', fontsize=PlotConfig.SUBTITLE_SIZE, fontweight='bold')
    ax.legend(fontsize=PlotConfig.LEGEND_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=PlotConfig.TICK_LABEL_SIZE)
    ax.grid(True, alpha=PlotConfig.GRID_ALPHA)
    ax.set_ylim(0, 1)

    # Panel 2: RT by set size
    ax = axes[0, 1]
    for group in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        group_data = df[df['hypothesis_group'] == group]

        rt_by_size = group_data.groupby('set_size')['rt'].agg(['mean', 'sem']).reset_index()

        ax.plot(
            rt_by_size['set_size'],
            rt_by_size['mean'],
            'o-',
            color=group_colors[group],
            linewidth=2,
            markersize=8,
            label=group.replace('_', ' ')
        )

        ax.fill_between(
            rt_by_size['set_size'],
            rt_by_size['mean'] - rt_by_size['sem'],
            rt_by_size['mean'] + rt_by_size['sem'],
            color=group_colors[group],
            alpha=0.2
        )

    ax.set_xlabel('Set Size (Number of Stimuli)', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Reaction Time (ms)', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
    ax.set_title('RT by Cognitive Load', fontsize=PlotConfig.SUBTITLE_SIZE, fontweight='bold')
    ax.legend(fontsize=PlotConfig.LEGEND_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=PlotConfig.TICK_LABEL_SIZE)
    ax.grid(True, alpha=PlotConfig.GRID_ALPHA)

    # Panel 3: Accuracy by block
    ax = axes[1, 0]
    for group in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        group_data = df[df['hypothesis_group'] == group]

        acc_by_block = group_data.groupby('block')['correct'].agg(['mean', 'sem']).reset_index()

        ax.plot(
            acc_by_block['block'],
            acc_by_block['mean'],
            'o-',
            color=group_colors[group],
            linewidth=2,
            markersize=6,
            alpha=0.7,
            label=group.replace('_', ' ')
        )

    ax.set_xlabel('Block Number', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
    ax.set_title('Accuracy Across Blocks', fontsize=PlotConfig.SUBTITLE_SIZE, fontweight='bold')
    ax.legend(fontsize=PlotConfig.LEGEND_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=PlotConfig.TICK_LABEL_SIZE)
    ax.grid(True, alpha=PlotConfig.GRID_ALPHA)
    ax.set_ylim(0, 1)

    # Panel 4: RT by block
    ax = axes[1, 1]
    for group in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        group_data = df[df['hypothesis_group'] == group]

        rt_by_block = group_data.groupby('block')['rt'].agg(['mean', 'sem']).reset_index()

        ax.plot(
            rt_by_block['block'],
            rt_by_block['mean'],
            'o-',
            color=group_colors[group],
            linewidth=2,
            markersize=6,
            alpha=0.7,
            label=group.replace('_', ' ')
        )

    ax.set_xlabel('Block Number', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
    ax.set_ylabel('Reaction Time (ms)', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
    ax.set_title('RT Across Blocks', fontsize=PlotConfig.SUBTITLE_SIZE, fontweight='bold')
    ax.legend(fontsize=PlotConfig.LEGEND_SIZE)
    ax.tick_params(axis='both', which='major', labelsize=PlotConfig.TICK_LABEL_SIZE)
    ax.grid(True, alpha=PlotConfig.GRID_ALPHA)

    plt.suptitle('Performance by Cognitive Load and Time',
                 fontsize=PlotConfig.TITLE_SIZE, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Save
    output_path = output_dir / 'performance_by_load_and_time.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")

    plt.close()


def print_summary_statistics(df: pd.DataFrame):
    """Print summary statistics by trauma group."""
    print("\n" + "=" * 80)
    print("BEHAVIORAL PERFORMANCE BY TRAUMA GROUP")
    print("=" * 80)

    for group in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        group_data = df[df['hypothesis_group'] == group]

        n_participants = group_data['sona_id'].nunique()
        n_trials = len(group_data)

        overall_acc = group_data['correct'].mean()
        overall_rt = group_data['rt'].mean()

        print(f"\n{group} (n={n_participants} participants, {n_trials} trials):")
        print(f"  Overall accuracy: {overall_acc:.3f}")
        print(f"  Overall RT: {overall_rt:.1f} ms")

        # By set size
        print(f"  Accuracy by set size:")
        for size in sorted(group_data['set_size'].unique()):
            acc = group_data[group_data['set_size'] == size]['correct'].mean()
            print(f"    Set {size}: {acc:.3f}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Analyze learning trajectories by trauma group'
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
        '--window',
        type=int,
        default=20,
        help='Rolling window size for smoothing (trials)'
    )
    parser.add_argument(
        '--min-block',
        type=int,
        default=3,
        help='Minimum block number to include (default: 3, excludes training blocks 1-2)'
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ANALYZING LEARNING TRAJECTORIES BY TRAUMA GROUP")
    print("=" * 80)

    # Load data
    df = load_data_with_groups(
        Path(args.data),
        Path(args.trauma_groups),
        min_block=args.min_block
    )

    # Compute rolling accuracy
    df = compute_rolling_accuracy(df, window=args.window)

    # Print summary statistics
    print_summary_statistics(df)

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    plot_all_learning_curves(df, output_dir)
    plot_group_specific_curves(df, output_dir)
    plot_performance_by_load(df, output_dir)

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}/")
    print("\nGenerated visualizations:")
    print("  1. learning_curves_all_participants.png - Spaghetti plot with group means")
    print("  2. learning_curves_by_group_panels.png - Separate panels per group")
    print("  3. performance_by_load_and_time.png - Load effects and temporal dynamics")
    print()


if __name__ == '__main__':
    main()
