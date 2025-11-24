"""
Visualize learning curves by trauma group across blocks.

Shows accuracy progression across blocks (1-21) for each trauma group.

Usage:
    python scripts/analysis/visualize_learning_curves_by_group.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300


def create_trauma_groups(df):
    """
    Create 3 trauma groups based on LEC and IES-R scores.
    
    Groups:
    - Group 1 (No Trauma): LEC = 0 (no trauma events)
    - Group 2 (Trauma - No Ongoing Impact): LEC >= 1 AND IES-R 0-23
    - Group 3 (Trauma - Ongoing Impact): LEC >= 1 AND IES-R >= 24
    
    Parameters
    ----------
    df : pd.DataFrame
        Participant data with lec_total_events and ies_total columns
        
    Returns
    -------
    pd.DataFrame
        Data with trauma_group column added
    """
    # Thresholds
    LEC_THRESHOLD = 0  # 0 = no trauma, 1+ = trauma present
    IES_THRESHOLD = 24  # 0-23 = no long-term impact, 24+ = long-term impact
    
    print(f"\n=== Trauma Group Classification ===")
    print(f"LEC-5 Threshold: {LEC_THRESHOLD} (0 = no trauma)")
    print(f"IES-R Threshold: {IES_THRESHOLD} (0-23 = no impact, 24+ = impact)\n")
    
    # Assign groups
    def assign_group(row):
        lec = row['lec_total_events']
        ies = row['ies_total']
        
        if lec == 0:
            return 'No Trauma'
        elif lec >= 1 and ies < IES_THRESHOLD:
            return 'Trauma - No Ongoing Impact'
        elif lec >= 1 and ies >= IES_THRESHOLD:
            return 'Trauma - Ongoing Impact'
        else:
            return 'Excluded'
    
    df['trauma_group'] = df.apply(assign_group, axis=1)
    
    # Print group sizes
    print("Group Sizes:")
    for group in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        n = (df['trauma_group'] == group).sum()
        if n > 0:
            group_data = df[df['trauma_group'] == group]
            lec_mean = group_data['lec_total_events'].mean()
            ies_mean = group_data['ies_total'].mean()
            print(f"  {group}: n={n}")
            print(f"    LEC M={lec_mean:.2f}, IES-R M={ies_mean:.2f}")
    
    n_excluded = (df['trauma_group'] == 'Excluded').sum()
    if n_excluded > 0:
        print(f"  Excluded: n={n_excluded}")
    
    return df


def calculate_block_accuracy(trials_df, trauma_df):
    """
    Calculate accuracy per block for each participant and assign trauma groups.
    
    12 blocks total:
    - Blocks 1-3: Set size 2
    - Blocks 4-6: Set size 3  
    - Blocks 7-9: Set size 5
    - Blocks 10-12: Set size 6
    
    Parameters
    ----------
    trials_df : pd.DataFrame
        Task trials data
    trauma_df : pd.DataFrame
        Summary data with trauma scores
        
    Returns
    -------
    pd.DataFrame
        Block-level accuracy with trauma groups
    """
    # Filter to main task blocks (3-14 in raw data = blocks 1-12 in analysis)
    trials_df = trials_df[(trials_df['block'] >= 3) & (trials_df['block'] <= 14)].copy()
    
    # Renumber blocks: raw block 3->1, 4->2, ..., 14->12
    trials_df['block_renumbered'] = trials_df['block'] - 2
    
    # Calculate accuracy per participant per block
    block_accuracy = []
    
    for sona_id in trials_df['sona_id'].unique():
        p_trials = trials_df[trials_df['sona_id'] == sona_id]
        
        for block_num in p_trials['block_renumbered'].unique():
            block_trials = p_trials[p_trials['block_renumbered'] == block_num]
            
            if len(block_trials) > 0:
                # Determine set size based on block number
                if 1 <= block_num <= 3:
                    set_size = 2
                elif 4 <= block_num <= 6:
                    set_size = 3
                elif 7 <= block_num <= 9:
                    set_size = 5
                elif 10 <= block_num <= 12:
                    set_size = 6
                else:
                    set_size = None
                
                block_accuracy.append({
                    'sona_id': sona_id,
                    'block': int(block_num),
                    'set_size': set_size,
                    'accuracy': block_trials['correct'].mean(),
                    'n_trials': len(block_trials)
                })
    
    block_df = pd.DataFrame(block_accuracy)
    
    # Merge with trauma groups
    trauma_groups = trauma_df[['sona_id', 'trauma_group']].copy()
    block_df = block_df.merge(trauma_groups, on='sona_id', how='left')
    
    return block_df


def plot_learning_curves_by_group(block_df, save_dir):
    """
    Plot learning curves showing accuracy across blocks for each trauma group.
    
    Parameters
    ----------
    block_df : pd.DataFrame
        Block-level accuracy data with trauma groups
    save_dir : Path
        Directory to save figures
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Define group order and colors
    group_order = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    colors = ['#2ecc71', '#545847', '#C1CFDA']  # Green, Dark Gray-Green, Light Blue-Gray
    markers = ['o', 's', '^']
    
    # Filter out excluded participants
    plot_df = block_df[block_df['trauma_group'].isin(group_order)].copy()
    
    # Plot for each group
    for group, color, marker in zip(group_order, colors, markers):
        group_data = plot_df[plot_df['trauma_group'] == group]
        
        if len(group_data) == 0:
            continue
        
        # Calculate mean and SEM per block
        grouped = group_data.groupby('block')['accuracy'].agg(['mean', 'sem', 'count']).reset_index()
        
        # Convert to percentage
        grouped['mean'] = grouped['mean'] * 100
        grouped['sem'] = grouped['sem'] * 100
        
        # Plot
        ax.errorbar(
            grouped['block'],
            grouped['mean'],
            yerr=grouped['sem'],
            label=f"{group} (n={group_data['sona_id'].nunique()})",
            color=color,
            marker=marker,
            markersize=8,
            linewidth=2.5,
            capsize=4,
            capthick=2,
            alpha=0.9
        )
    
    # Styling
    ax.set_xlabel('Block Number', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('Learning Curves Across 12 Blocks by Trauma Group\n(Blocks 1-3: SS2 | 4-6: SS3 | 7-9: SS5 | 10-12: SS6)',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Set x-axis to show blocks 1-12
    ax.set_xticks(range(1, 13))
    ax.set_xlim([0.5, 12.5])
    ax.set_ylim([0, 100])
    
    # Add vertical lines to separate set size sections
    for x in [3.5, 6.5, 9.5]:
        ax.axvline(x, color='gray', linestyle='--', linewidth=1, alpha=0.4)
    
    # Legend
    ax.legend(
        title='Trauma Group',
        title_fontsize=12,
        fontsize=11,
        loc='lower right',
        framealpha=0.95,
        edgecolor='black'
    )
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Chance line
    ax.axhline(50, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, label='Chance (50%)')
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / 'learning_curves_by_trauma_group.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


def plot_learning_curves_dual_panel(block_df, save_dir):
    """
    Create dual-panel plot: one for early blocks, one for late blocks.
    
    Parameters
    ----------
    block_df : pd.DataFrame
        Block-level accuracy data
    save_dir : Path
        Directory to save figures
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    group_order = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    colors = ['#2ecc71', '#545847', '#C1CFDA']  # Green, Dark Gray-Green, Light Blue-Gray
    markers = ['o', 's', '^']
    
    plot_df = block_df[block_df['trauma_group'].isin(group_order)].copy()
    
    # Determine early and late blocks
    all_blocks = sorted(plot_df['block'].unique())
    mid_point = len(all_blocks) // 2
    early_blocks = all_blocks[:mid_point]
    late_blocks = all_blocks[mid_point:]
    
    # Left panel: Early blocks
    ax = axes[0]
    for group, color, marker in zip(group_order, colors, markers):
        group_data = plot_df[(plot_df['trauma_group'] == group) & 
                             (plot_df['block'].isin(early_blocks))]
        
        if len(group_data) == 0:
            continue
        
        grouped = group_data.groupby('block')['accuracy'].agg(['mean', 'sem']).reset_index()
        grouped['mean'] = grouped['mean'] * 100
        grouped['sem'] = grouped['sem'] * 100
        
        ax.errorbar(grouped['block'], grouped['mean'], yerr=grouped['sem'],
                   label=group, color=color, marker=marker, markersize=8,
                   linewidth=2.5, capsize=4, alpha=0.9)
    
    ax.set_xlabel('Block Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Early Blocks', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(50, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    
    # Right panel: Late blocks
    ax = axes[1]
    for group, color, marker in zip(group_order, colors, markers):
        group_data = plot_df[(plot_df['trauma_group'] == group) & 
                             (plot_df['block'].isin(late_blocks))]
        
        if len(group_data) == 0:
            continue
        
        grouped = group_data.groupby('block')['accuracy'].agg(['mean', 'sem']).reset_index()
        grouped['mean'] = grouped['mean'] * 100
        grouped['sem'] = grouped['sem'] * 100
        
        ax.errorbar(grouped['block'], grouped['mean'], yerr=grouped['sem'],
                   label=group, color=color, marker=marker, markersize=8,
                   linewidth=2.5, capsize=4, alpha=0.9)
    
    ax.set_xlabel('Block Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Late Blocks', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 100])
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(50, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    ax.legend(title='Trauma Group', title_fontsize=11, fontsize=10,
             loc='lower right', framealpha=0.95)
    
    plt.suptitle('Learning Curves: Early vs Late Blocks by Trauma Group',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = save_dir / 'learning_curves_early_vs_late.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def print_learning_statistics(block_df):
    """Print learning statistics for each group."""
    print("\n" + "="*80)
    print("LEARNING STATISTICS BY GROUP")
    print("="*80)
    
    group_order = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    
    for group in group_order:
        group_data = block_df[block_df['trauma_group'] == group]
        
        if len(group_data) == 0:
            continue
        
        # Get first and last blocks
        blocks = sorted(group_data['block'].unique())
        if len(blocks) < 2:
            continue
        
        first_block = blocks[0]
        last_block = blocks[-1]
        
        # Calculate mean accuracy for first and last blocks
        first_acc = group_data[group_data['block'] == first_block]['accuracy'].mean() * 100
        last_acc = group_data[group_data['block'] == last_block]['accuracy'].mean() * 100
        
        improvement = last_acc - first_acc
        
        print(f"\n{group}:")
        print(f"  Block {first_block} accuracy: {first_acc:.2f}%")
        print(f"  Block {last_block} accuracy: {last_acc:.2f}%")
        print(f"  Improvement: {improvement:+.2f}%")
        
        # Overall mean
        overall_mean = group_data['accuracy'].mean() * 100
        print(f"  Overall mean accuracy: {overall_mean:.2f}%")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("LEARNING CURVES BY TRAUMA GROUP")
    print("="*80)
    
    # Load summary data (has trauma scores)
    summary_path = Path('output/summary_participant_metrics.csv')
    if not summary_path.exists():
        print(f"\nERROR: Summary data not found at {summary_path}")
        print("Run the data pipeline first: python run_data_pipeline.py")
        return
    
    summary_df = pd.read_csv(summary_path)
    print(f"\nLoaded summary data: {len(summary_df)} participants")
    
    # Load task trials
    trials_path = Path('output/task_trials_long_all_participants.csv')
    if not trials_path.exists():
        print(f"\nERROR: Task trials not found at {trials_path}")
        print("Run the data pipeline first: python run_data_pipeline.py")
        return
    
    trials_df = pd.read_csv(trials_path)
    print(f"Loaded task trials: {len(trials_df)} trials")
    
    # Create trauma groups
    summary_df = create_trauma_groups(summary_df)
    
    # Calculate block-level accuracy
    print("\n" + "="*80)
    print("CALCULATING BLOCK-LEVEL ACCURACY")
    print("="*80)
    
    block_df = calculate_block_accuracy(trials_df, summary_df)
    print(f"\nCalculated accuracy for {len(block_df)} block observations")
    print(f"Participants: {block_df['sona_id'].nunique()}")
    print(f"Blocks: {sorted(block_df['block'].unique())}")
    
    # Print learning statistics
    print_learning_statistics(block_df)
    
    # Create output directory
    save_dir = Path('figures/trauma_groups')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_learning_curves_by_group(block_df, save_dir)
    plot_learning_curves_dual_panel(block_df, save_dir)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nFigures saved to: {save_dir}/")
    print()


if __name__ == '__main__':
    main()
