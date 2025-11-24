"""
Visualize 2x3 Load × Trauma Group interaction for ANOVA.

Creates bar plot showing:
- 2 Load conditions (Low vs High)
- 3 Trauma groups (No Trauma, Trauma-No Impact, Trauma-Ongoing Impact)

Usage:
    python scripts/analysis/visualize_load_by_trauma_group.py
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
            return 'Excluded'  # Should not occur with this logic
    
    df['trauma_group'] = df.apply(assign_group, axis=1)
    
    # Print group sizes
    print("Group Sizes:")
    for group in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        n = (df['trauma_group'] == group).sum()
        group_data = df[df['trauma_group'] == group]
        if len(group_data) > 0:
            lec_mean = group_data['lec_total_events'].mean()
            ies_mean = group_data['ies_total'].mean()
            print(f"  {group}: n={n}")
            print(f"    LEC M={lec_mean:.2f}, IES-R M={ies_mean:.2f}")
    
    n_excluded = (df['trauma_group'] == 'Excluded').sum()
    if n_excluded > 0:
        print(f"  Excluded: n={n_excluded}")
    
    return df, LEC_THRESHOLD, IES_THRESHOLD


def prepare_data_for_anova(df):
    """
    Reshape data from wide to long format for ANOVA visualization.
    
    Parameters
    ----------
    df : pd.DataFrame
        Wide-format data with accuracy_low_load and accuracy_high_load columns
        
    Returns
    -------
    pd.DataFrame
        Long-format data with columns: sona_id, trauma_group, load, accuracy
    """
    # Exclude inconsistent cases
    df = df[df['trauma_group'] != 'Excluded'].copy()
    
    # Create long-format data
    long_data = []
    
    for _, row in df.iterrows():
        # Low load
        long_data.append({
            'sona_id': row['sona_id'],
            'trauma_group': row['trauma_group'],
            'load': 'Low Load',
            'accuracy': row['accuracy_low_load'] * 100,  # Convert to percentage
            'lec_total': row['lec_total_events'],
            'ies_total': row['ies_total']
        })
        
        # High load
        long_data.append({
            'sona_id': row['sona_id'],
            'trauma_group': row['trauma_group'],
            'load': 'High Load',
            'accuracy': row['accuracy_high_load'] * 100,
            'lec_total': row['lec_total_events'],
            'ies_total': row['ies_total']
        })
    
    return pd.DataFrame(long_data)


def plot_load_by_trauma_group(long_df, save_dir):
    """
    Create 2x3 bar plot: Load × Trauma Group.
    
    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format data
    save_dir : Path
        Directory to save figure
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Define group order and colors
    group_order = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    colors = ['#2ecc71', '#545847', '#C1CFDA']  # Green, Dark Gray-Green, Light Blue-Gray
    
    # Calculate means and SEMs
    summary_data = long_df.groupby(['trauma_group', 'load'])['accuracy'].agg(['mean', 'sem']).reset_index()
    
    # Set up positions
    x_positions = np.arange(2)  # Low Load, High Load
    bar_width = 0.25
    
    # Plot bars for each trauma group
    for i, (group, color) in enumerate(zip(group_order, colors)):
        group_data = summary_data[summary_data['trauma_group'] == group]
        
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
        
        ax.bar(positions, means, bar_width, 
               yerr=sems,
               label=group,
               color=color,
               alpha=0.8,
               capsize=5,
               edgecolor='black',
               linewidth=1.2)
    
    # Styling
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Load Condition', fontsize=14, fontweight='bold')
    ax.set_title('Accuracy by Load Condition and Trauma Group\n2×3 ANOVA Design',
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks(x_positions)
    ax.set_xticklabels(['Low Load\n(Set Size 2-3)', 'High Load\n(Set Size 5-6)'], 
                       fontsize=12)
    ax.set_ylim([0, 100])
    
    # Add legend
    ax.legend(title='Trauma Group', 
              title_fontsize=12,
              fontsize=11,
              loc='upper right',
              framealpha=0.95,
              edgecolor='black')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax.set_axisbelow(True)
    
    # Add chance line
    ax.axhline(50, color='gray', linestyle=':', linewidth=1.5, alpha=0.6, label='Chance (50%)')
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / 'load_by_trauma_group_anova.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


def plot_interaction_lines(long_df, save_dir):
    """
    Create interaction plot with lines connecting low to high load.
    
    Parameters
    ----------
    long_df : pd.DataFrame
        Long-format data
    save_dir : Path
        Directory to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Define group order and colors
    group_order = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    colors = ['#2ecc71', '#545847', '#C1CFDA']  # Green, Dark Gray-Green, Light Blue-Gray
    markers = ['o', 's', '^']
    
    # Calculate means and SEMs
    summary_data = long_df.groupby(['trauma_group', 'load'])['accuracy'].agg(['mean', 'sem']).reset_index()
    
    # Plot lines for each group
    for group, color, marker in zip(group_order, colors, markers):
        group_data = summary_data[summary_data['trauma_group'] == group]
        
        # Sort by load (alphabetically: High Load, Low Load)
        group_data = group_data.sort_values('load')
        
        ax.errorbar(group_data['load'], 
                   group_data['mean'],
                   yerr=group_data['sem'],
                   label=group,
                   color=color,
                   marker=marker,
                   markersize=10,
                   linewidth=2.5,
                   capsize=6,
                   capthick=2,
                   alpha=0.9)
    
    # Styling
    ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Load Condition', fontsize=14, fontweight='bold')
    ax.set_title('Load × Trauma Group Interaction\nAccuracy Performance',
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.set_ylim([0, 100])
    ax.set_xlim([-0.3, 1.3])
    
    # Legend
    ax.legend(title='Trauma Group',
              title_fontsize=12,
              fontsize=11,
              loc='best',
              framealpha=0.95,
              edgecolor='black')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Chance line
    ax.axhline(50, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / 'load_by_trauma_interaction_lines.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def print_descriptive_stats(long_df):
    """Print descriptive statistics for each cell of the design."""
    print("\n" + "="*80)
    print("DESCRIPTIVE STATISTICS: 2×3 Design")
    print("="*80)
    
    group_order = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    load_order = ['Low Load', 'High Load']
    
    for group in group_order:
        print(f"\n{group}:")
        for load in load_order:
            subset = long_df[(long_df['trauma_group'] == group) & (long_df['load'] == load)]
            
            if len(subset) > 0:
                mean_acc = subset['accuracy'].mean()
                sem_acc = subset['accuracy'].sem()
                n = len(subset)
                
                print(f"  {load}: M = {mean_acc:.2f}%, SEM = {sem_acc:.2f}, n = {n}")


def calculate_effect_sizes(long_df):
    """Calculate load effect (difference) for each trauma group."""
    print("\n" + "="*80)
    print("LOAD EFFECT SIZES (High Load - Low Load)")
    print("="*80)
    
    group_order = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    
    for group in group_order:
        group_data = long_df[long_df['trauma_group'] == group]
        
        low_load = group_data[group_data['load'] == 'Low Load']['accuracy']
        high_load = group_data[group_data['load'] == 'High Load']['accuracy']
        
        # Calculate difference scores (paired)
        # Need to match participants
        participants = group_data['sona_id'].unique()
        differences = []
        
        for pid in participants:
            p_data = group_data[group_data['sona_id'] == pid]
            low = p_data[p_data['load'] == 'Low Load']['accuracy'].values[0]
            high = p_data[p_data['load'] == 'High Load']['accuracy'].values[0]
            differences.append(high - low)
        
        mean_diff = np.mean(differences)
        sem_diff = stats.sem(differences)
        
        print(f"\n{group}:")
        print(f"  Mean difference: {mean_diff:.2f}% (SEM = {sem_diff:.2f})")
        print(f"  Interpretation: {'Accuracy decreased' if mean_diff < 0 else 'Accuracy increased'} by {abs(mean_diff):.2f}% from low to high load")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("LOAD × TRAUMA GROUP VISUALIZATION")
    print("="*80)
    
    # Load summary data (has trauma scores)
    summary_path = Path('output/summary_participant_metrics.csv')
    if not summary_path.exists():
        print(f"\nERROR: Summary data not found at {summary_path}")
        print("Run the data pipeline first: python run_data_pipeline.py")
        return
    
    df = pd.read_csv(summary_path)
    print(f"\nLoaded summary data: {len(df)} participants")
    
    # Load task trials to calculate load-specific accuracy
    trials_path = Path('output/task_trials_long_all_participants.csv')
    if not trials_path.exists():
        print(f"\nERROR: Task trials not found at {trials_path}")
        print("Run the data pipeline first: python run_data_pipeline.py")
        return
    
    trials_df = pd.read_csv(trials_path)
    print(f"Loaded task trials: {len(trials_df)} trials")
    
    # Filter out practice blocks
    trials_df = trials_df[trials_df['block'] >= 3].copy()
    
    # Calculate load-specific accuracy for each participant
    load_accuracy = []
    for sona_id in trials_df['sona_id'].unique():
        p_trials = trials_df[trials_df['sona_id'] == sona_id]
        
        low_load_trials = p_trials[p_trials['load_condition'] == 'low']
        high_load_trials = p_trials[p_trials['load_condition'] == 'high']
        
        load_accuracy.append({
            'sona_id': sona_id,
            'accuracy_low_load': low_load_trials['correct'].mean() if len(low_load_trials) > 0 else np.nan,
            'accuracy_high_load': high_load_trials['correct'].mean() if len(high_load_trials) > 0 else np.nan
        })
    
    load_df = pd.DataFrame(load_accuracy)
    
    # Merge with summary data
    df = df.merge(load_df, on='sona_id', how='inner')
    print(f"Merged data: {len(df)} participants with both trauma scores and load accuracy")
    
    # Required columns
    required_cols = ['sona_id', 'lec_total_events', 'ies_total', 
                     'accuracy_low_load', 'accuracy_high_load']
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"\nERROR: Missing required columns: {missing}")
        return
    
    # Remove participants with missing data
    df_clean = df.dropna(subset=required_cols)
    print(f"After removing missing data: {len(df_clean)} participants")
    
    # Create trauma groups
    df_grouped, lec_med, ies_med = create_trauma_groups(df_clean)
    
    # Prepare long-format data
    print("\n" + "="*80)
    print("PREPARING DATA FOR ANOVA")
    print("="*80)
    
    long_df = prepare_data_for_anova(df_grouped)
    print(f"\nLong-format data: {len(long_df)} observations")
    print(f"Unique participants: {long_df['sona_id'].nunique()}")
    
    # Print descriptive statistics
    print_descriptive_stats(long_df)
    
    # Calculate effect sizes
    calculate_effect_sizes(long_df)
    
    # Create output directory
    save_dir = Path('figures/trauma_groups')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_load_by_trauma_group(long_df, save_dir)
    plot_interaction_lines(long_df, save_dir)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nFigures saved to: {save_dir}/")
    print("\nNext steps:")
    print("  1. Review visualizations")
    print("  2. Run 2×3 mixed ANOVA in your statistical software")
    print("  3. Check for Load × Group interaction effect")
    print()


if __name__ == '__main__':
    main()
