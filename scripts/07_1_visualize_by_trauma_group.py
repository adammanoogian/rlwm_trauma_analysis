#!/usr/bin/env python
"""
07_1: Visualize by Trauma Group
===============================

Comprehensive trauma group visualizations and comparisons.

This CONSOLIDATED script combines functionality from 8 files:
1. visualize_learning_curves_by_group.py - Learning curves by trauma group
2. visualize_load_by_trauma_group.py - Load × Trauma Group interaction
3. visualize_reversal_by_trauma_group.py - Reversal learning by group
4. visualize_rt_by_load_and_group.py - RT by load and group
5. visualize_feedback_learning.py - Feedback effects visualization
6. analyze_learning_by_trauma_group.py - Learning trajectory analysis
7. analyze_parameters_by_trauma_group.py - Model parameter comparisons
8. plot_stimulus_learning_by_trauma_group.py - Stimulus-specific learning

Inputs:
    - output/summary_participant_metrics.csv
    - output/task_trials_long.csv (or task_trials_long_all_participants.csv)
    - output/trauma_groups/group_assignments.csv (optional)
    - output/mle/*_individual_fits.csv (for parameter analysis)

Outputs:
    - figures/trauma_groups/learning_curves_by_trauma_group.png
    - figures/trauma_groups/load_by_trauma_group_anova.png
    - figures/trauma_groups/reversal_metrics_by_trauma_group.png
    - figures/trauma_groups/rt_by_load_and_trauma_group.png
    - figures/feedback_learning/feedback_learning_by_setsize*.png
    - figures/trauma_groups/stimulus_learning_curves_by_trauma_group.png

Usage:
    # Run all visualizations
    python scripts/07_1_visualize_by_trauma_group.py

    # Run specific visualization types
    python scripts/07_1_visualize_by_trauma_group.py --learning     # Learning curves
    python scripts/07_1_visualize_by_trauma_group.py --load         # Load × Group
    python scripts/07_1_visualize_by_trauma_group.py --reversal     # Reversal learning
    python scripts/07_1_visualize_by_trauma_group.py --rt           # RT analysis
    python scripts/07_1_visualize_by_trauma_group.py --feedback     # Feedback effects
    python scripts/07_1_visualize_by_trauma_group.py --params       # Parameter comparisons
    python scripts/07_1_visualize_by_trauma_group.py --stimulus     # Stimulus learning

    # Combine multiple flags
    python scripts/07_1_visualize_by_trauma_group.py --learning --load --rt

Next Steps:
    - Run 08_run_statistical_analyses.py for ANOVAs
    - Run model fitting (12-14) then 15_analyze_mle_by_trauma.py
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import sys
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from config import OUTPUT_DIR, FIGURES_DIR, TaskParams, AnalysisParams

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# ============================================================================
# CONSTANTS
# ============================================================================

GROUP_COLORS = {
    'No Trauma': '#2ecc71',                    # Green
    'Trauma - No Ongoing Impact': '#545847',   # Dark Gray-Green
    'Trauma - Ongoing Impact': '#C1CFDA'       # Light Blue-Gray
}

GROUP_MARKERS = {
    'No Trauma': 'o',
    'Trauma - No Ongoing Impact': 's',
    'Trauma - Ongoing Impact': '^'
}

GROUP_ORDER = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']

SAVE_DIR = Path('figures/trauma_groups')

# ============================================================================
# DATA LOADING & GROUPING
# ============================================================================

def create_trauma_groups(df):
    """
    Create 3 trauma groups based on LESS endorsement and IES-R cutoff.

    Groups:
    - No Trauma: LESS = 0 (no trauma events)
    - Trauma - No Ongoing Impact: LESS >= 1 AND IES-R < 24
    - Trauma - Ongoing Impact: LESS >= 1 AND IES-R >= 24
    """
    IES_CUTOFF = 24

    def assign_group(row):
        less = row.get('less_total_events', row.get('lec_total_events', 0))
        ies = row.get('ies_total', 0)

        if pd.isna(less) or pd.isna(ies):
            return 'Excluded'

        if less == 0:
            return 'No Trauma'
        elif less >= 1 and ies < IES_CUTOFF:
            return 'Trauma - No Ongoing Impact'
        elif less >= 1 and ies >= IES_CUTOFF:
            return 'Trauma - Ongoing Impact'
        else:
            return 'Excluded'

    df['trauma_group'] = df.apply(assign_group, axis=1)

    # Print group sizes
    print("\nTrauma Group Assignment:")
    for group in GROUP_ORDER:
        n = (df['trauma_group'] == group).sum()
        if n > 0:
            group_data = df[df['trauma_group'] == group]
            print(f"  {group}: n={n}")

    return df


def load_summary_data():
    """Load summary participant metrics and add trauma groups."""
    summary_path = Path('output/summary_participant_metrics.csv')
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary data not found at {summary_path}")

    df = pd.read_csv(summary_path)
    print(f"Loaded summary data: {len(df)} participants")

    df = create_trauma_groups(df)
    return df


def load_trials_data():
    """Load task trials data."""
    trials_path = Path('output/task_trials_long.csv')
    if not trials_path.exists():
        trials_path = Path('output/task_trials_long_all_participants.csv')

    if not trials_path.exists():
        raise FileNotFoundError(f"Task trials not found")

    df = pd.read_csv(trials_path)
    # Filter to main task
    df = df[df['block'] >= 3].copy()
    print(f"Loaded task trials: {len(df)} trials (main task only)")
    return df


# ============================================================================
# PART 1: LEARNING CURVES
# ============================================================================

def plot_learning_curves_by_group(summary_df, trials_df, save_dir):
    """Plot learning curves across blocks for each trauma group."""
    print("\n>> Creating learning curves by trauma group...")

    # Calculate block accuracy
    trials_df = trials_df.copy()
    trials_df['block_renumbered'] = trials_df['block'] - 2  # Start from 1

    block_accuracy = []
    for sona_id in trials_df['sona_id'].unique():
        p_trials = trials_df[trials_df['sona_id'] == sona_id]

        for block_num in p_trials['block_renumbered'].unique():
            block_trials = p_trials[p_trials['block_renumbered'] == block_num]
            if len(block_trials) > 0:
                block_accuracy.append({
                    'sona_id': sona_id,
                    'block': int(block_num),
                    'accuracy': block_trials['correct'].mean(),
                    'n_trials': len(block_trials)
                })

    block_df = pd.DataFrame(block_accuracy)

    # Merge with trauma groups
    trauma_groups = summary_df[['sona_id', 'trauma_group']].copy()
    block_df = block_df.merge(trauma_groups, on='sona_id', how='left')

    # Plot
    fig, ax = plt.subplots(figsize=(14, 7))

    plot_df = block_df[block_df['trauma_group'].isin(GROUP_ORDER)].copy()

    for group in GROUP_ORDER:
        group_data = plot_df[plot_df['trauma_group'] == group]

        if len(group_data) == 0:
            continue

        grouped = group_data.groupby('block')['accuracy'].agg(['mean', 'sem', 'count']).reset_index()
        grouped['mean'] = grouped['mean'] * 100
        grouped['sem'] = grouped['sem'] * 100

        ax.errorbar(
            grouped['block'],
            grouped['mean'],
            yerr=grouped['sem'],
            label=f"{group} (n={group_data['sona_id'].nunique()})",
            color=GROUP_COLORS[group],
            marker=GROUP_MARKERS[group],
            markersize=8,
            linewidth=2.5,
            capsize=4,
            capthick=2,
            alpha=0.9
        )

    ax.set_xlabel('Block Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Learning Curves Across Blocks by Trauma Group', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim([0, 100])
    ax.legend(title='Trauma Group', title_fontsize=12, fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(50, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)

    plt.tight_layout()

    save_path = save_dir / 'learning_curves_by_trauma_group.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================
# PART 2: LOAD × TRAUMA GROUP
# ============================================================================

def plot_load_by_trauma_group(summary_df, trials_df, save_dir):
    """Create Load × Trauma Group interaction plot."""
    print("\n>> Creating Load × Trauma Group visualization...")

    # Calculate load-specific accuracy
    load_accuracy = []
    for sona_id in trials_df['sona_id'].unique():
        p_trials = trials_df[trials_df['sona_id'] == sona_id]

        if 'load_condition' not in p_trials.columns:
            continue

        low_load = p_trials[p_trials['load_condition'] == 'low']
        high_load = p_trials[p_trials['load_condition'] == 'high']

        load_accuracy.append({
            'sona_id': sona_id,
            'accuracy_low': low_load['correct'].mean() if len(low_load) > 0 else np.nan,
            'accuracy_high': high_load['correct'].mean() if len(high_load) > 0 else np.nan
        })

    load_df = pd.DataFrame(load_accuracy)

    # Merge with trauma groups
    df = summary_df.merge(load_df, on='sona_id', how='inner')
    df = df[df['trauma_group'].isin(GROUP_ORDER)].copy()

    # Create long format
    long_data = []
    for _, row in df.iterrows():
        long_data.append({
            'sona_id': row['sona_id'],
            'trauma_group': row['trauma_group'],
            'load': 'Low Load',
            'accuracy': row['accuracy_low'] * 100
        })
        long_data.append({
            'sona_id': row['sona_id'],
            'trauma_group': row['trauma_group'],
            'load': 'High Load',
            'accuracy': row['accuracy_high'] * 100
        })

    long_df = pd.DataFrame(long_data)

    # Plot bars
    fig, ax = plt.subplots(figsize=(12, 7))

    summary_data = long_df.groupby(['trauma_group', 'load'])['accuracy'].agg(['mean', 'sem']).reset_index()

    x_positions = np.arange(2)
    bar_width = 0.25

    for i, group in enumerate(GROUP_ORDER):
        group_data = summary_data[summary_data['trauma_group'] == group]

        low_row = group_data[group_data['load'] == 'Low Load']
        high_row = group_data[group_data['load'] == 'High Load']

        means = [
            low_row['mean'].values[0] if len(low_row) > 0 else 0,
            high_row['mean'].values[0] if len(high_row) > 0 else 0
        ]
        sems = [
            low_row['sem'].values[0] if len(low_row) > 0 else 0,
            high_row['sem'].values[0] if len(high_row) > 0 else 0
        ]

        offset = (i - 1) * bar_width
        ax.bar(x_positions + offset, means, bar_width, yerr=sems,
               label=group, color=GROUP_COLORS[group], alpha=0.8,
               capsize=5, edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Load Condition', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy by Load Condition and Trauma Group\n2×3 ANOVA Design',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Low Load\n(Set Size 2-3)', 'High Load\n(Set Size 5-6)'], fontsize=12)
    ax.set_ylim([0, 100])
    ax.legend(title='Trauma Group', title_fontsize=12, fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.axhline(50, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)

    plt.tight_layout()

    save_path = save_dir / 'load_by_trauma_group_anova.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================
# PART 3: REVERSAL LEARNING
# ============================================================================

def plot_reversal_by_trauma_group(summary_df, save_dir):
    """Plot reversal learning metrics by trauma group."""
    print("\n>> Creating reversal learning visualization...")

    df = summary_df[summary_df['trauma_group'].isin(GROUP_ORDER)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Metrics to plot
    metrics = [
        ('performance_drop_post_reversal', 'Performance Drop (%)', 'Perseveration:\nAccuracy Drop Post-Reversal'),
        ('adaptation_rate_post_reversal', 'Adaptation Rate (%)', 'Cognitive Flexibility:\nRecovery Speed'),
        ('n_reversals', 'Number of Reversals', 'Reversal Exposure:\nReversals Encountered')
    ]

    for idx, (col, ylabel, title) in enumerate(metrics):
        ax = axes[idx]

        if col not in df.columns:
            ax.text(0.5, 0.5, f'Column {col} not found', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(title, fontsize=13, fontweight='bold')
            continue

        group_stats = []
        for group in GROUP_ORDER:
            group_data = df[df['trauma_group'] == group][col].dropna()
            if len(group_data) > 0:
                multiplier = 100 if 'rate' in col or 'drop' in col else 1
                group_stats.append({
                    'group': group,
                    'mean': group_data.mean() * multiplier,
                    'sem': group_data.sem() * multiplier,
                    'n': len(group_data)
                })

        if group_stats:
            stats_df = pd.DataFrame(group_stats)
            x_pos = np.arange(len(stats_df))

            colors = [GROUP_COLORS[g] for g in stats_df['group']]
            ax.bar(x_pos, stats_df['mean'], yerr=stats_df['sem'],
                   color=colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=1.2)

            ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
            ax.set_xlabel('Trauma Group', fontsize=12, fontweight='bold')
            ax.set_title(title, fontsize=13, fontweight='bold')
            ax.set_xticks(x_pos)
            ax.set_xticklabels([s.replace(' - ', '\n') for s in stats_df['group']], fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')

            for i, row in stats_df.iterrows():
                y_pos = row['mean'] + row['sem'] + max(row['mean'], 1) * 0.05
                ax.text(i, y_pos, f"n={row['n']}", ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()

    save_path = save_dir / 'reversal_metrics_by_trauma_group.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================
# PART 4: RT BY LOAD AND GROUP
# ============================================================================

def plot_rt_by_load_and_group(summary_df, trials_df, save_dir):
    """Plot RT by load condition and trauma group."""
    print("\n>> Creating RT visualization...")

    # Calculate RT by load
    rt_data = []
    for sona_id in trials_df['sona_id'].unique():
        p_trials = trials_df[trials_df['sona_id'] == sona_id]

        if 'load_condition' not in p_trials.columns or 'rt' not in p_trials.columns:
            continue

        p_trials = p_trials[p_trials['rt'].notna()]

        for load in ['low', 'high']:
            load_trials = p_trials[p_trials['load_condition'] == load]
            if len(load_trials) > 0:
                rt_data.append({
                    'sona_id': sona_id,
                    'load': 'Low Load' if load == 'low' else 'High Load',
                    'mean_rt': load_trials['rt'].mean()
                })

    rt_df = pd.DataFrame(rt_data)

    # Merge with trauma groups
    trauma_groups = summary_df[['sona_id', 'trauma_group']].copy()
    rt_df = rt_df.merge(trauma_groups, on='sona_id', how='left')
    rt_df = rt_df[rt_df['trauma_group'].isin(GROUP_ORDER)].copy()

    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))

    summary_data = rt_df.groupby(['trauma_group', 'load'])['mean_rt'].agg(['mean', 'sem']).reset_index()

    x_positions = np.arange(2)
    bar_width = 0.25

    for i, group in enumerate(GROUP_ORDER):
        group_data = summary_data[summary_data['trauma_group'] == group]

        low_row = group_data[group_data['load'] == 'Low Load']
        high_row = group_data[group_data['load'] == 'High Load']

        means = [
            low_row['mean'].values[0] if len(low_row) > 0 else 0,
            high_row['mean'].values[0] if len(high_row) > 0 else 0
        ]
        sems = [
            low_row['sem'].values[0] if len(low_row) > 0 else 0,
            high_row['sem'].values[0] if len(high_row) > 0 else 0
        ]

        n_participants = rt_df[rt_df['trauma_group'] == group]['sona_id'].nunique()

        offset = (i - 1) * bar_width
        ax.bar(x_positions + offset, means, bar_width, yerr=sems,
               label=f'{group} (n={n_participants})', color=GROUP_COLORS[group],
               alpha=0.8, capsize=5, edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Mean Reaction Time (ms)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Load Condition', fontsize=14, fontweight='bold')
    ax.set_title('Reaction Time by Load Condition and Trauma Group',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Low Load\n(Set Size 2-3)', 'High Load\n(Set Size 5-6)'], fontsize=12)
    ax.legend(title='Trauma Group', title_fontsize=12, fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()

    save_path = save_dir / 'rt_by_load_and_trauma_group.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================
# PART 5: FEEDBACK LEARNING
# ============================================================================

def plot_feedback_learning(trials_df, save_dir):
    """Plot feedback learning effects."""
    print("\n>> Creating feedback learning visualization...")

    feedback_dir = Path('figures/feedback_learning')
    feedback_dir.mkdir(parents=True, exist_ok=True)

    # Process feedback learning data
    processed = []

    for (subject_id, block_id), block_data in trials_df.groupby(['sona_id', 'block']):
        block_data = block_data.sort_values('trial_in_block').reset_index(drop=True)

        stimulus_encounter_count = defaultdict(int)

        for idx in range(len(block_data) - 1):
            current_row = block_data.iloc[idx]
            next_row = block_data.iloc[idx + 1]

            stimulus = int(float(current_row['stimulus'])) - 1
            set_size = int(float(current_row['set_size']))
            current_correct = int(float(current_row['correct']))
            next_correct = int(float(next_row['correct']))

            stimulus_encounter_count[stimulus] += 1
            encounter_num = stimulus_encounter_count[stimulus]

            feedback_type = 'positive' if current_correct == 1 else 'negative'

            processed.append({
                'subject_id': subject_id,
                'set_size_n': set_size,
                'encounter_num': encounter_num,
                'feedback_type': feedback_type,
                'correct_n_plus_1': next_correct
            })

    data_df = pd.DataFrame(processed)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    negative_df = data_df[data_df['feedback_type'] == 'negative']
    set_sizes = sorted(negative_df['set_size_n'].unique())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(set_sizes)))

    # Panel 1: Recovery after incorrect
    ax = axes[0]
    for ss, color in zip(set_sizes, colors):
        subset = negative_df[negative_df['set_size_n'] == ss]
        grouped = subset.groupby('encounter_num')['correct_n_plus_1'].agg(['mean', 'sem', 'count']).reset_index()
        grouped = grouped[grouped['count'] >= 5]

        if len(grouped) > 0:
            ax.errorbar(grouped['encounter_num'], grouped['mean'] * 100, yerr=grouped['sem'] * 100,
                       label=f'Set Size {ss}', marker='o', linewidth=2, markersize=6,
                       capsize=4, color=color, alpha=0.8)

    ax.set_xlabel('Encounter Number with Stimulus', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy at Trial n+1 (%)', fontsize=12, fontweight='bold')
    ax.set_title('Recovery After Incorrect Response', fontsize=13, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([0, 100])
    ax.axhline(50, color='gray', linestyle=':', alpha=0.5)

    # Panel 2: Comparison
    ax = axes[1]
    comparison_data = []

    for ss in set_sizes:
        neg_acc = data_df[(data_df['feedback_type'] == 'negative') &
                          (data_df['set_size_n'] == ss)]['correct_n_plus_1'].mean() * 100
        pos_acc = data_df[(data_df['feedback_type'] == 'positive') &
                          (data_df['set_size_n'] == ss)]['correct_n_plus_1'].mean() * 100
        comparison_data.append({'set_size': ss, 'negative': neg_acc, 'positive': pos_acc})

    comp_df = pd.DataFrame(comparison_data)

    x = np.arange(len(set_sizes))
    width = 0.35

    ax.bar(x - width/2, comp_df['negative'], width, label='After Incorrect', color='#e74c3c', alpha=0.8)
    ax.bar(x + width/2, comp_df['positive'], width, label='After Correct', color='#2ecc71', alpha=0.8)

    ax.set_xlabel('Set Size at Trial n', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy at Trial n+1 (%)', fontsize=12, fontweight='bold')
    ax.set_title('Accuracy After Feedback Type', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'SS {ss}' for ss in set_sizes])
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_ylim([0, 100])
    ax.axhline(50, color='gray', linestyle=':', alpha=0.5)

    plt.tight_layout()

    save_path = feedback_dir / 'feedback_learning_by_setsize.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================
# PART 6: STIMULUS LEARNING BY GROUP
# ============================================================================

def plot_stimulus_learning_by_group(summary_df, trials_df, save_dir):
    """Plot stimulus-based learning curves by trauma group."""
    print("\n>> Creating stimulus learning curves by group...")

    # Compute encounter numbers
    trials_df = trials_df.copy()
    trials_df = trials_df.sort_values(['sona_id', 'block', 'stimulus', 'trial_in_block'])
    trials_df['encounter_number'] = trials_df.groupby(['sona_id', 'block', 'stimulus']).cumcount() + 1

    # Merge with trauma groups
    trauma_groups = summary_df[['sona_id', 'trauma_group']].copy()
    trials_df = trials_df.merge(trauma_groups, on='sona_id', how='inner')
    trials_df = trials_df[trials_df['trauma_group'].isin(GROUP_ORDER)].copy()

    # Plot by set size
    set_sizes = sorted(trials_df['set_size'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, set_size in enumerate(set_sizes):
        if idx >= 4:
            break

        ax = axes[idx]
        df_size = trials_df[trials_df['set_size'] == set_size]

        for group in GROUP_ORDER:
            df_group = df_size[df_size['trauma_group'] == group]

            if len(df_group) == 0:
                continue

            curve = df_group.groupby('encounter_number')['correct'].agg(['mean', 'sem', 'count']).reset_index()
            curve = curve[curve['count'] >= 5]

            if len(curve) == 0:
                continue

            ax.plot(curve['encounter_number'], curve['mean'] * 100,
                   color=GROUP_COLORS[group], label=group, linewidth=2.5,
                   marker=GROUP_MARKERS[group], markersize=6, alpha=0.8)

            ax.fill_between(curve['encounter_number'],
                           (curve['mean'] - curve['sem']) * 100,
                           (curve['mean'] + curve['sem']) * 100,
                           color=GROUP_COLORS[group], alpha=0.2)

        ax.set_xlabel('Encounter with Stimulus (Nth time)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Set Size {int(set_size)}', fontsize=12, fontweight='bold')
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=9)

    # Hide unused axes
    for idx in range(len(set_sizes), 4):
        axes[idx].axis('off')

    plt.suptitle('Stimulus Learning Curves by Trauma Group', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    save_path = save_dir / 'stimulus_learning_curves_by_trauma_group.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


# ============================================================================
# PART 7: PARAMETER ANALYSIS (if MLE data available)
# ============================================================================

def plot_parameters_by_group(save_dir):
    """Plot MLE parameters by trauma group if available."""
    print("\n>> Creating parameter analysis visualization...")

    # Try to load MLE fits
    mle_dir = Path('output/mle')

    param_files = {
        'qlearning': mle_dir / 'qlearning_individual_fits.csv',
        'wmrl': mle_dir / 'wmrl_individual_fits.csv',
        'wmrl_m3': mle_dir / 'wmrl_m3_individual_fits.csv'
    }

    available_models = {k: v for k, v in param_files.items() if v.exists()}

    if not available_models:
        print("  [SKIP] No MLE fits found. Run model fitting first.")
        return

    # Load summary for trauma groups
    summary_df = load_summary_data()

    for model_name, param_file in available_models.items():
        print(f"  Processing {model_name}...")

        params_df = pd.read_csv(param_file)

        # Merge with trauma groups
        merged_df = params_df.merge(summary_df[['sona_id', 'trauma_group']],
                                    left_on='participant_id', right_on='sona_id', how='inner')
        merged_df = merged_df[merged_df['trauma_group'].isin(GROUP_ORDER)].copy()

        if len(merged_df) == 0:
            print(f"    No matched participants for {model_name}")
            continue

        # Identify parameter columns
        param_cols = [c for c in params_df.columns if c not in ['participant_id', 'sona_id',
                                                                  'neg_loglik', 'aic', 'bic', 'n_trials']]

        if len(param_cols) == 0:
            continue

        # Create violin plots
        n_params = min(len(param_cols), 6)
        fig, axes = plt.subplots(1, n_params, figsize=(5 * n_params, 6))
        if n_params == 1:
            axes = [axes]

        for idx, param in enumerate(param_cols[:n_params]):
            ax = axes[idx]

            plot_data = merged_df[['trauma_group', param]].dropna()

            positions = []
            data_by_group = []

            for i, group in enumerate(GROUP_ORDER):
                group_values = plot_data[plot_data['trauma_group'] == group][param].values
                if len(group_values) > 0:
                    positions.append(i + 1)
                    data_by_group.append(group_values)

            if len(data_by_group) > 0:
                parts = ax.violinplot(data_by_group, positions=positions, showmeans=True, widths=0.7)

                for pc, pos in zip(parts['bodies'], positions):
                    group = GROUP_ORDER[pos - 1]
                    pc.set_facecolor(GROUP_COLORS[group])
                    pc.set_alpha(0.7)

                # Add jittered points
                for pos, values in zip(positions, data_by_group):
                    group = GROUP_ORDER[pos - 1]
                    jitter = np.random.normal(0, 0.05, len(values))
                    ax.scatter(pos + jitter, values, c=GROUP_COLORS[group], s=50, alpha=0.6,
                              edgecolors='black', linewidth=0.5, zorder=3)

            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(['No\nTrauma', 'Trauma\nNo Impact', 'Trauma\nOngoing'], fontsize=9)
            ax.set_ylabel(param.replace('_', ' ').title(), fontsize=11, fontweight='bold')
            ax.set_title(f'{param.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'{model_name.upper()} Parameters by Trauma Group', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = save_dir / f'{model_name}_parameters_by_group.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {save_path}")
        plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Visualize by trauma group (comprehensive analysis)'
    )
    parser.add_argument('--learning', action='store_true', help='Learning curves')
    parser.add_argument('--load', action='store_true', help='Load × Group interaction')
    parser.add_argument('--reversal', action='store_true', help='Reversal learning')
    parser.add_argument('--rt', action='store_true', help='RT analysis')
    parser.add_argument('--feedback', action='store_true', help='Feedback effects')
    parser.add_argument('--params', action='store_true', help='Parameter comparisons')
    parser.add_argument('--stimulus', action='store_true', help='Stimulus learning')

    args = parser.parse_args()

    # If no specific flags, run all
    run_all = not any([args.learning, args.load, args.reversal, args.rt,
                       args.feedback, args.params, args.stimulus])

    # Create output directory
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TRAUMA GROUP VISUALIZATIONS")
    print("=" * 80)

    # Load data
    summary_df = load_summary_data()
    trials_df = load_trials_data()

    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    if run_all or args.learning:
        plot_learning_curves_by_group(summary_df, trials_df, SAVE_DIR)

    if run_all or args.load:
        plot_load_by_trauma_group(summary_df, trials_df, SAVE_DIR)

    if run_all or args.reversal:
        plot_reversal_by_trauma_group(summary_df, SAVE_DIR)

    if run_all or args.rt:
        plot_rt_by_load_and_group(summary_df, trials_df, SAVE_DIR)

    if run_all or args.feedback:
        plot_feedback_learning(trials_df, SAVE_DIR)

    if run_all or args.stimulus:
        plot_stimulus_learning_by_group(summary_df, trials_df, SAVE_DIR)

    if run_all or args.params:
        plot_parameters_by_group(SAVE_DIR)

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)
    print(f"\nFigures saved to: {SAVE_DIR}/")
    print("\nNext steps:")
    print("  - Run 08_run_statistical_analyses.py for ANOVAs")
    print("  - Run model fitting (12-14) then 15_analyze_mle_by_trauma.py")


if __name__ == '__main__':
    main()
