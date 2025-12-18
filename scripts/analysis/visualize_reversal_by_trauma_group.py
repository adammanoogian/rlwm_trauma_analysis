"""
Visualize post-reversal performance by trauma group (Perseveration analysis).

Shows:
- Performance drop immediately after reversals
- Adaptation rate (recovery speed)
- Number of reversals experienced

Usage:
    python scripts/analysis/visualize_reversal_by_trauma_group.py
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
    Create 3 trauma groups based on LESS and IES-R scores.
    
    Groups:
    - Group 1 (No Trauma): LESS = 0 (no trauma events)
    - Group 2 (Trauma - No Ongoing Impact): LESS >= 1 AND IES-R 0-23
    - Group 3 (Trauma - Ongoing Impact): LESS >= 1 AND IES-R >= 24
    """
    LESS_THRESHOLD = 0
    IES_THRESHOLD = 24
    
    print(f"\n=== Trauma Group Classification ===")
    print(f"LESS Threshold: {LESS_THRESHOLD} (0 = no trauma)")
    print(f"IES-R Threshold: {IES_THRESHOLD} (0-23 = no impact, 24+ = impact)\n")
    
    def assign_group(row):
        less = row['less_total_events']
        ies = row['ies_total']
        
        if less == 0:
            return 'No Trauma'
        elif less >= 1 and ies < IES_THRESHOLD:
            return 'Trauma - No Ongoing Impact'
        elif less >= 1 and ies >= IES_THRESHOLD:
            return 'Trauma - Ongoing Impact'
        else:
            return 'Excluded'
    
    df['trauma_group'] = df.apply(assign_group, axis=1)
    
    # Print group sizes
    print("Group Sizes:")
    for group in ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        n = (df['trauma_group'] == group).sum()
        group_data = df[df['trauma_group'] == group]
        if len(group_data) > 0:
            less_mean = group_data['less_total_events'].mean()
            ies_mean = group_data['ies_total'].mean()
            print(f"  {group}: n={n}")
            print(f"    LESS M={less_mean:.2f}, IES-R M={ies_mean:.2f}")
    
    return df


def plot_reversal_metrics_by_group(df, save_dir):
    """
    Create bar plots showing reversal metrics by trauma group.
    
    Plots:
    1. Performance drop post-reversal (perseveration)
    2. Adaptation rate (recovery speed)
    3. Number of reversals
    """
    # Filter valid groups
    df = df[df['trauma_group'] != 'Excluded'].copy()
    
    # Define group order and colors
    group_order = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    colors = ['#2ecc71', '#545847', '#C1CFDA']
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Performance Drop Post-Reversal
    ax = axes[0]
    group_stats = []
    for group in group_order:
        group_data = df[df['trauma_group'] == group]['performance_drop_post_reversal']
        group_data = group_data.dropna()
        if len(group_data) > 0:
            group_stats.append({
                'group': group,
                'mean': group_data.mean() * 100,  # Convert to percentage
                'sem': group_data.sem() * 100,
                'n': len(group_data)
            })
    
    if group_stats:
        stats_df = pd.DataFrame(group_stats)
        x_pos = np.arange(len(stats_df))
        
        ax.bar(x_pos, stats_df['mean'], 
               yerr=stats_df['sem'],
               color=colors[:len(stats_df)],
               alpha=0.8,
               capsize=5,
               edgecolor='black',
               linewidth=1.2)
        
        ax.set_ylabel('Performance Drop (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trauma Group', fontsize=12, fontweight='bold')
        ax.set_title('Perseveration:\nAccuracy Drop Post-Reversal', 
                     fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.replace(' - ', '\n') for s in stats_df['group']], 
                           fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_axisbelow(True)
        
        # Add sample sizes
        for i, row in stats_df.iterrows():
            ax.text(i, row['mean'] + row['sem'] + 2, f"n={row['n']}", 
                   ha='center', fontsize=9, fontweight='bold')
    
    # 2. Adaptation Rate
    ax = axes[1]
    group_stats = []
    for group in group_order:
        group_data = df[df['trauma_group'] == group]['adaptation_rate_post_reversal']
        group_data = group_data.dropna()
        if len(group_data) > 0:
            group_stats.append({
                'group': group,
                'mean': group_data.mean() * 100,
                'sem': group_data.sem() * 100,
                'n': len(group_data)
            })
    
    if group_stats:
        stats_df = pd.DataFrame(group_stats)
        x_pos = np.arange(len(stats_df))
        
        ax.bar(x_pos, stats_df['mean'], 
               yerr=stats_df['sem'],
               color=colors[:len(stats_df)],
               alpha=0.8,
               capsize=5,
               edgecolor='black',
               linewidth=1.2)
        
        ax.set_ylabel('Adaptation Rate (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trauma Group', fontsize=12, fontweight='bold')
        ax.set_title('Cognitive Flexibility:\nRecovery Speed Post-Reversal', 
                     fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.replace(' - ', '\n') for s in stats_df['group']], 
                           fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_axisbelow(True)
        ax.axhline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.6)
        
        # Add sample sizes
        for i, row in stats_df.iterrows():
            y_pos = row['mean'] + row['sem'] + 2 if row['mean'] >= 0 else row['mean'] - row['sem'] - 2
            ax.text(i, y_pos, f"n={row['n']}", 
                   ha='center', fontsize=9, fontweight='bold')
    
    # 3. Number of Reversals
    ax = axes[2]
    group_stats = []
    for group in group_order:
        group_data = df[df['trauma_group'] == group]['n_reversals']
        group_data = group_data.dropna()
        if len(group_data) > 0:
            group_stats.append({
                'group': group,
                'mean': group_data.mean(),
                'sem': group_data.sem(),
                'n': len(group_data)
            })
    
    if group_stats:
        stats_df = pd.DataFrame(group_stats)
        x_pos = np.arange(len(stats_df))
        
        ax.bar(x_pos, stats_df['mean'], 
               yerr=stats_df['sem'],
               color=colors[:len(stats_df)],
               alpha=0.8,
               capsize=5,
               edgecolor='black',
               linewidth=1.2)
        
        ax.set_ylabel('Number of Reversals', fontsize=12, fontweight='bold')
        ax.set_xlabel('Trauma Group', fontsize=12, fontweight='bold')
        ax.set_title('Reversal Exposure:\nReversals Encountered', 
                     fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([s.replace(' - ', '\n') for s in stats_df['group']], 
                           fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_axisbelow(True)
        
        # Add sample sizes
        for i, row in stats_df.iterrows():
            ax.text(i, row['mean'] + row['sem'] + 0.5, f"n={row['n']}", 
                   ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    save_path = save_dir / 'reversal_metrics_by_trauma_group.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved: {save_path}")
    plt.close()


def print_reversal_statistics(df):
    """Print detailed statistics for reversal metrics by group."""
    print("\n" + "="*80)
    print("REVERSAL PERFORMANCE STATISTICS BY TRAUMA GROUP")
    print("="*80)
    
    df = df[df['trauma_group'] != 'Excluded'].copy()
    group_order = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    
    for group in group_order:
        group_data = df[df['trauma_group'] == group]
        
        if len(group_data) == 0:
            continue
        
        print(f"\n{group} (n={len(group_data)}):")
        
        # Performance drop
        perf_drop = group_data['performance_drop_post_reversal'].dropna()
        if len(perf_drop) > 0:
            print(f"  Performance Drop Post-Reversal:")
            print(f"    Mean: {perf_drop.mean()*100:.2f}% (SEM = {perf_drop.sem()*100:.2f})")
            print(f"    Range: {perf_drop.min()*100:.2f}% to {perf_drop.max()*100:.2f}%")
        
        # Adaptation rate
        adapt = group_data['adaptation_rate_post_reversal'].dropna()
        if len(adapt) > 0:
            print(f"  Adaptation Rate (Recovery):")
            print(f"    Mean: {adapt.mean()*100:.2f}% (SEM = {adapt.sem()*100:.2f})")
            print(f"    Range: {adapt.min()*100:.2f}% to {adapt.max()*100:.2f}%")
        
        # Number of reversals
        n_rev = group_data['n_reversals'].dropna()
        if len(n_rev) > 0:
            print(f"  Reversals Encountered:")
            print(f"    Mean: {n_rev.mean():.2f} (SEM = {n_rev.sem():.2f})")
            print(f"    Range: {int(n_rev.min())} to {int(n_rev.max())}")


def main():
    """Main execution."""
    print("\n" + "="*80)
    print("REVERSAL LEARNING BY TRAUMA GROUP")
    print("="*80)
    
    # Load summary data
    summary_path = Path('output/summary_participant_metrics.csv')
    if not summary_path.exists():
        print(f"\nERROR: Summary data not found at {summary_path}")
        print("Run the data pipeline first to calculate reversal metrics")
        return
    
    df = pd.read_csv(summary_path)
    print(f"\nLoaded summary data: {len(df)} participants")
    
    # Check for required columns
    required_cols = ['sona_id', 'less_total_events', 'ies_total',
                     'performance_drop_post_reversal', 'adaptation_rate_post_reversal', 
                     'n_reversals']
    
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"\nERROR: Missing required columns: {missing}")
        print("These metrics should be calculated by the data pipeline.")
        print("Check that calculate_reversal_metrics() is being called in scoring_functions.py")
        return
    
    # Create trauma groups
    df = create_trauma_groups(df)
    
    # Print statistics
    print_reversal_statistics(df)
    
    # Create output directory
    save_dir = Path('figures/trauma_groups')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    
    plot_reversal_metrics_by_group(df, save_dir)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nFigures saved to: {save_dir}/")
    print("\nInterpretation:")
    print("  - Higher performance drop = More perseveration (difficulty switching)")
    print("  - Lower adaptation rate = Slower recovery from reversal")
    print("  - These metrics assess cognitive flexibility and rule learning")
    print()


if __name__ == '__main__':
    main()
