"""
Visualize reaction time distributions by load and trauma group.

Shows mean RT by load condition (low vs high) for each trauma group.

Usage:
    python scripts/analysis/visualize_rt_by_load_and_group.py
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


def calculate_rt_by_load(trials_df, trauma_df):
    """
    Calculate mean RT by load condition for each participant.
    
    Parameters
    ----------
    trials_df : pd.DataFrame
        Task trials data
    trauma_df : pd.DataFrame
        Summary data with trauma scores
        
    Returns
    -------
    pd.DataFrame
        RT data by load condition with trauma groups
    """
    # Filter to main task blocks and completed trials only
    trials_df = trials_df[(trials_df['block'] >= 3) & (trials_df['rt'].notna())].copy()
    
    # Calculate RT by load for each participant
    rt_data = []
    
    for sona_id in trials_df['sona_id'].unique():
        p_trials = trials_df[trials_df['sona_id'] == sona_id]
        
        # Low load trials
        low_load = p_trials[p_trials['load_condition'] == 'low']
        if len(low_load) > 0:
            rt_data.append({
                'sona_id': sona_id,
                'load': 'Low Load',
                'mean_rt': low_load['rt'].mean(),
                'median_rt': low_load['rt'].median(),
                'n_trials': len(low_load)
            })
        
        # High load trials
        high_load = p_trials[p_trials['load_condition'] == 'high']
        if len(high_load) > 0:
            rt_data.append({
                'sona_id': sona_id,
                'load': 'High Load',
                'mean_rt': high_load['rt'].mean(),
                'median_rt': high_load['rt'].median(),
                'n_trials': len(high_load)
            })
    
    rt_df = pd.DataFrame(rt_data)
    
    # Merge with trauma groups
    trauma_groups = trauma_df[['sona_id', 'trauma_group']].copy()
    rt_df = rt_df.merge(trauma_groups, on='sona_id', how='left')
    
    return rt_df


def plot_rt_by_load_and_group(rt_df, save_dir):
    """
    Create grouped bar plot: Load × Trauma Group for mean RT.
    
    Parameters
    ----------
    rt_df : pd.DataFrame
        RT data with load and trauma group
    save_dir : Path
        Directory to save figures
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define group order and colors
    group_order = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    colors = ['#2ecc71', '#545847', '#C1CFDA']  # Green, Dark Gray-Green, Light Blue-Gray
    
    # Filter to valid groups
    plot_df = rt_df[rt_df['trauma_group'].isin(group_order)].copy()
    
    # Calculate means and SEMs
    summary_data = plot_df.groupby(['trauma_group', 'load'])['mean_rt'].agg(['mean', 'sem']).reset_index()
    
    # Set up positions
    x_positions = np.arange(2)  # Low Load, High Load
    bar_width = 0.25
    
    # Plot bars for each trauma group
    for i, group in enumerate(group_order):
        group_data = summary_data[summary_data['trauma_group'] == group]
        
        if len(group_data) == 0:
            continue
        
        # Get values in order: Low Load, High Load
        low_load_row = group_data[group_data['load'] == 'Low Load']
        high_load_row = group_data[group_data['load'] == 'High Load']
        
        means = [
            low_load_row['mean'].values[0] if len(low_load_row) > 0 else 0,
            high_load_row['mean'].values[0] if len(high_load_row) > 0 else 0
        ]
        sems = [
            low_load_row['sem'].values[0] if len(low_load_row) > 0 else 0,
            high_load_row['sem'].values[0] if len(high_load_row) > 0 else 0
        ]
        
        # Offset positions for each group
        offset = (i - 1) * bar_width
        positions = x_positions + offset
        
        # Count participants in this group
        n_participants = plot_df[plot_df['trauma_group'] == group]['sona_id'].nunique()
        
        ax.bar(positions, means, bar_width,
               yerr=sems,
               label=f'{group} (n={n_participants})',
               color=colors[i],
               alpha=0.8,
               capsize=5,
               edgecolor='black',
               linewidth=1.2)
    
    # Styling
    ax.set_ylabel('Mean Reaction Time (ms)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Load Condition', fontsize=14, fontweight='bold')
    ax.set_title('Reaction Time by Load Condition and Trauma Group',
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Low Load\n(Set Size 2-3)', 'High Load\n(Set Size 5-6)'], 
                       fontsize=12)
    ax.set_ylim([0, ax.get_ylim()[1] * 1.1])
    
    # Add legend
    ax.legend(title='Trauma Group',
              title_fontsize=12,
              fontsize=11,
              loc='upper left',
              framealpha=0.95,
              edgecolor='black')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / 'rt_by_load_and_trauma_group.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


def plot_rt_distributions(rt_df, save_dir):
    """
    Create violin plot showing RT distributions.
    
    Parameters
    ----------
    rt_df : pd.DataFrame
        RT data
    save_dir : Path
        Directory to save figures
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    group_order = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    colors = ['#2ecc71', '#545847', '#C1CFDA']  # Green, Dark Gray-Green, Light Blue-Gray
    
    plot_df = rt_df[rt_df['trauma_group'].isin(group_order)].copy()
    
    # Left panel: Low Load
    ax = axes[0]
    low_load_data = plot_df[plot_df['load'] == 'Low Load']
    
    if len(low_load_data) > 0:
        sns.violinplot(data=low_load_data, x='trauma_group', y='mean_rt',
                      order=group_order, palette=colors, ax=ax, alpha=0.6)
        sns.stripplot(data=low_load_data, x='trauma_group', y='mean_rt',
                     order=group_order, color='black', ax=ax, alpha=0.5, size=6)
    
    ax.set_xlabel('Trauma Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean RT (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Low Load RT Distribution', fontsize=14, fontweight='bold')
    ax.set_xticklabels([g.replace(' ', '\n') for g in group_order], rotation=0, ha='center')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Right panel: High Load
    ax = axes[1]
    high_load_data = plot_df[plot_df['load'] == 'High Load']
    
    if len(high_load_data) > 0:
        sns.violinplot(data=high_load_data, x='trauma_group', y='mean_rt',
                      order=group_order, palette=colors, ax=ax, alpha=0.6)
        sns.stripplot(data=high_load_data, x='trauma_group', y='mean_rt',
                     order=group_order, color='black', ax=ax, alpha=0.5, size=6)
    
    ax.set_xlabel('Trauma Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean RT (ms)', fontsize=12, fontweight='bold')
    ax.set_title('High Load RT Distribution', fontsize=14, fontweight='bold')
    ax.set_xticklabels([g.replace(' ', '\n') for g in group_order], rotation=0, ha='center')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    plt.suptitle('Reaction Time Distributions by Load and Trauma Group',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    save_path = save_dir / 'rt_distributions_by_group.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def print_rt_statistics(rt_df):
    """Print RT statistics for each group and load condition."""
    print("\n" + "="*80)
    print("RT STATISTICS BY GROUP AND LOAD")
    print("="*80)
    
    group_order = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    load_order = ['Low Load', 'High Load']
    
    for group in group_order:
        group_data = rt_df[rt_df['trauma_group'] == group]
        
        if len(group_data) == 0:
            continue
        
        print(f"\n{group}:")
        for load in load_order:
            load_data = group_data[group_data['load'] == load]
            
            if len(load_data) > 0:
                mean_rt = load_data['mean_rt'].mean()
                sem_rt = load_data['mean_rt'].sem()
                median_rt = load_data['mean_rt'].median()
                n = len(load_data)
                
                print(f"  {load}:")
                print(f"    Mean RT: {mean_rt:.2f} ms (SEM = {sem_rt:.2f})")
                print(f"    Median RT: {median_rt:.2f} ms")
                print(f"    n = {n}")
        
        # Calculate RT difference (High - Low)
        low_rt = group_data[group_data['load'] == 'Low Load']['mean_rt']
        high_rt = group_data[group_data['load'] == 'High Load']['mean_rt']
        
        if len(low_rt) > 0 and len(high_rt) > 0:
            # Match participants for paired comparison
            low_participants = group_data[group_data['load'] == 'Low Load']['sona_id'].values
            high_participants = group_data[group_data['load'] == 'High Load']['sona_id'].values
            common_participants = set(low_participants) & set(high_participants)
            
            if len(common_participants) > 0:
                differences = []
                for pid in common_participants:
                    low = group_data[(group_data['sona_id'] == pid) & 
                                    (group_data['load'] == 'Low Load')]['mean_rt'].values[0]
                    high = group_data[(group_data['sona_id'] == pid) & 
                                     (group_data['load'] == 'High Load')]['mean_rt'].values[0]
                    differences.append(high - low)
                
                mean_diff = np.mean(differences)
                sem_diff = stats.sem(differences)
                print(f"  RT Difference (High - Low): {mean_diff:.2f} ms (SEM = {sem_diff:.2f})")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("REACTION TIME BY LOAD AND TRAUMA GROUP")
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
    
    # Calculate RT by load
    print("\n" + "="*80)
    print("CALCULATING RT BY LOAD CONDITION")
    print("="*80)
    
    rt_df = calculate_rt_by_load(trials_df, summary_df)
    print(f"\nCalculated RT for {len(rt_df)} observations")
    print(f"Participants: {rt_df['sona_id'].nunique()}")
    
    # Print statistics
    print_rt_statistics(rt_df)
    
    # Create output directory
    save_dir = Path('figures/trauma_groups')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_rt_by_load_and_group(rt_df, save_dir)
    plot_rt_distributions(rt_df, save_dir)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nFigures saved to: {save_dir}/")
    print()


if __name__ == '__main__':
    main()
