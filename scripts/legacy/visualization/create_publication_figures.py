"""
Create publication-ready figures for behavioral results section.

This script generates three main figures:
- Figure 1: Learning curves across the task
- Figure 2: Working memory load effects (accuracy + RT)
- Figure 3: Trauma severity correlations (scatter plots)

And formats the three main tables for publication.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from config import OUTPUT_DIR, FIGURES_DIR
from plotting_config import PlotConfig

# Color scheme for trauma groups
COLORS = {
    'low_load': '#2E86AB',  # Blue
    'high_load': '#F18F01',  # Orange
    'no_impact': '#06A77D',  # Green
    'ongoing_impact': '#D62246'  # Red
}

# Get font sizes
FONT_SIZES = PlotConfig.get_fontsize_dict()
DPI = PlotConfig.DPI_PRINT


def apply_publication_style():
    """Apply publication-quality style to plots."""
    PlotConfig.apply_defaults()


def save_publication_figure(fig, path):
    """Save figure with publication settings."""
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    print(f"  Saved: {path.name}")


def load_data():
    """Load all required data files."""
    # Load trial-level data
    trials_data = pd.read_csv(OUTPUT_DIR / 'task_trials_long_all_participants.csv')
    
    # Load summary data with groups
    summary_data = pd.read_csv(OUTPUT_DIR / 'statistical_analyses' / 'data_summary_with_groups.csv')
    
    # Load demographics
    demographics = pd.read_csv(OUTPUT_DIR / 'parsed_demographics.csv')
    
    # Load descriptive tables
    table1 = pd.read_csv(OUTPUT_DIR / 'descriptives' / 'table1_demographics.csv')
    table2 = pd.read_csv(OUTPUT_DIR / 'descriptives' / 'table2_trauma_scores.csv')
    table3 = pd.read_csv(OUTPUT_DIR / 'descriptives' / 'table3_task_performance.csv')
    
    return trials_data, summary_data, demographics, table1, table2, table3


def create_figure1_learning_curves(trials_data, save_dir):
    """
    Figure 1: Learning Curves Across the Task
    
    Shows accuracy over trial bins for both load conditions.
    Demonstrates above-chance learning.
    """
    apply_publication_style()
    
    # Need to merge with summary data to get trauma groups
    summary_df = pd.read_csv(OUTPUT_DIR / 'statistical_analyses' / 'data_summary_with_groups.csv')
    
    # Merge to get trauma groups
    df = trials_data.merge(summary_df[['sona_id', 'trauma_group']], on='sona_id', how='left')
    
    # Filter participants without trauma group
    df = df[df['trauma_group'].notna()].copy()
    
    # Exclude practice blocks (blocks 1 and 2)
    df = df[df['block'] > 2].copy()
    
    # Renumber blocks to start from 1 (block 3 becomes 1, etc.)
    df['experimental_block'] = df.groupby('sona_id')['block'].rank(method='dense').astype(int)
    
    # Keep only first 12 experimental blocks (the standard task length)
    df = df[df['experimental_block'] <= 12].copy()
    
    # Calculate accuracy by load and experimental block
    learning_data = df.groupby(['load_condition', 'experimental_block'])['correct'].agg(['mean', 'sem']).reset_index()
    learning_data['mean'] *= 100  # Convert to percentage
    learning_data['sem'] *= 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot learning curves
    for load, color in zip(['low', 'high'], [COLORS['low_load'], COLORS['high_load']]):
        data = learning_data[learning_data['load_condition'] == load]
        
        ax.plot(data['experimental_block'], data['mean'], 
                color=color, linewidth=2.5, 
                label=f'{load.capitalize()} Load',
                marker='o', markersize=6)
        
        ax.fill_between(data['experimental_block'],
                        data['mean'] - data['sem'],
                        data['mean'] + data['sem'],
                        color=color, alpha=0.2)
    
    # Add chance line
    ax.axhline(y=33.33, color='gray', linestyle='--', linewidth=1.5, 
               label='Chance (33.3%)', alpha=0.6)
    
    # Formatting
    ax.set_xlabel('Experimental Block', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Accuracy (%)', fontsize=FONT_SIZES['label'])
    ax.set_title('Learning Across the Task', fontsize=FONT_SIZES['title'], fontweight='bold')
    ax.legend(fontsize=FONT_SIZES['legend'], frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim([25, 95])
    ax.tick_params(labelsize=FONT_SIZES['tick'])
    
    plt.tight_layout()
    save_publication_figure(fig, save_dir / 'Figure1_learning_curves.png')
    plt.close()
    
    print("✓ Figure 1: Learning curves created")


def create_figure2_load_effects(summary_data, save_dir):
    """
    Figure 2: Working Memory Load Effects on Performance
    
    Two-panel figure showing:
    - Panel A: Accuracy by load and trauma group
    - Panel B: RT by load and trauma group
    """
    apply_publication_style()
    
    # Prepare data
    df = summary_data[summary_data['trauma_group'].notna()].copy()
    
    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: Accuracy
    ax = axes[0]
    
    # Prepare data for plotting
    acc_data = []
    for group in ['Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        for load in ['low', 'high']:
            group_load_data = df[df['trauma_group'] == group]
            col = f'accuracy_{load}'
            acc_data.append({
                'Group': 'No Ongoing Impact' if 'No Ongoing' in group else 'Ongoing Impact',
                'Load': load.capitalize(),
                'Accuracy': group_load_data[col].mean() * 100,
                'SEM': group_load_data[col].sem() * 100
            })
    
    acc_df = pd.DataFrame(acc_data)
    
    # Plot bars
    x = np.arange(2)  # Two load conditions
    width = 0.35
    
    for i, group in enumerate(['No Ongoing Impact', 'Ongoing Impact']):
        group_data = acc_df[acc_df['Group'] == group]
        means = group_data['Accuracy'].values
        sems = group_data['SEM'].values
        
        color = COLORS['no_impact'] if group == 'No Ongoing Impact' else COLORS['ongoing_impact']
        ax.bar(x + i*width, means, width, yerr=sems, 
               label=group, color=color, capsize=8, alpha=0.8, 
               error_kw={'linewidth': 2, 'capthick': 2})
    
    ax.set_ylabel('Accuracy (%)', fontsize=FONT_SIZES['label'])
    ax.set_xlabel('Working Memory Load', fontsize=FONT_SIZES['label'])
    ax.set_title('A. Accuracy by Load', fontsize=FONT_SIZES['title'], fontweight='bold', loc='left')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(['Low', 'High'])
    ax.legend(fontsize=FONT_SIZES['legend'], frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim([60, 90])
    ax.tick_params(labelsize=FONT_SIZES['tick'])
    
    # Panel B: RT
    ax = axes[1]
    
    # Prepare RT data (using median ± SD)
    rt_data = []
    for group in ['Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        for load in ['low', 'high']:
            group_load_data = df[df['trauma_group'] == group]
            col = f'rt_{load}'
            rt_data.append({
                'Group': 'No Ongoing Impact' if 'No Ongoing' in group else 'Ongoing Impact',
                'Load': load.capitalize(),
                'RT': group_load_data[col].median(),
                'SD': group_load_data[col].std()
            })
    
    rt_df = pd.DataFrame(rt_data)
    
    # Plot bars
    for i, group in enumerate(['No Ongoing Impact', 'Ongoing Impact']):
        group_data = rt_df[rt_df['Group'] == group]
        means = group_data['RT'].values
        sds = group_data['SD'].values
        
        color = COLORS['no_impact'] if group == 'No Ongoing Impact' else COLORS['ongoing_impact']
        ax.bar(x + i*width, means, width, yerr=sds,
               label=group, color=color, capsize=8, alpha=0.8, 
               error_kw={'linewidth': 2, 'capthick': 2})
    
    ax.set_ylabel('Median Reaction Time (ms)', fontsize=FONT_SIZES['label'])
    ax.set_xlabel('Working Memory Load', fontsize=FONT_SIZES['label'])
    ax.set_title('B. Reaction Time by Load', fontsize=FONT_SIZES['title'], fontweight='bold', loc='left')
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(['Low', 'High'])
    ax.legend(fontsize=FONT_SIZES['legend'], frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim([350, 850])  # Expanded to show full error bars with SD
    ax.tick_params(labelsize=FONT_SIZES['tick'])
    
    plt.tight_layout()
    save_publication_figure(fig, save_dir / 'Figure2_load_effects.png')
    plt.close()
    
    print("✓ Figure 2: Load effects created")


def create_figure3_trauma_correlations(summary_data, save_dir):
    """
    Figure 3: Trauma Severity and Task Performance
    
    Two-panel figure showing scatter plots:
    - Panel A: LESS Total vs. Overall Accuracy
    - Panel B: IES-R Total vs. Overall Accuracy
    """
    apply_publication_style()
    
    # Prepare data
    df = summary_data[summary_data['trauma_group'].notna()].copy()
    df['overall_accuracy'] = (df['accuracy_low'] + df['accuracy_high']) / 2 * 100
    
    # Create figure with two panels
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel A: LESS Total
    ax = axes[0]
    
    for group, color in zip(['Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact'],
                            [COLORS['no_impact'], COLORS['ongoing_impact']]):
        group_data = df[df['trauma_group'] == group]
        label = 'No Ongoing Impact' if 'No Ongoing' in group else 'Ongoing Impact'
        
        ax.scatter(group_data['less_total_events'], 
                  group_data['overall_accuracy'],
                  color=color, s=80, alpha=0.6, label=label, edgecolors='white', linewidth=0.5)
    
    # Add regression line for all data
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['less_total_events'], df['overall_accuracy']
    )
    x_line = np.linspace(df['less_total_events'].min(), df['less_total_events'].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'k--', linewidth=1.5, alpha=0.5)
    
    # Add statistics text
    ax.text(0.05, 0.95, f'r = {r_value:.3f}\np = {p_value:.3f}',
            transform=ax.transAxes, fontsize=FONT_SIZES['legend'],
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('LESS Total Events', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Overall Accuracy (%)', fontsize=FONT_SIZES['label'])
    ax.set_title('A. Trauma Exposure', fontsize=FONT_SIZES['title'], fontweight='bold', loc='left')
    ax.legend(fontsize=FONT_SIZES['legend'], frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=FONT_SIZES['tick'])
    
    # Panel B: IES-R Total
    ax = axes[1]
    
    for group, color in zip(['Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact'],
                            [COLORS['no_impact'], COLORS['ongoing_impact']]):
        group_data = df[df['trauma_group'] == group]
        label = 'No Ongoing Impact' if 'No Ongoing' in group else 'Ongoing Impact'
        
        ax.scatter(group_data['ies_total'], 
                  group_data['overall_accuracy'],
                  color=color, s=80, alpha=0.6, label=label, edgecolors='white', linewidth=0.5)
    
    # Add regression line for all data
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        df['ies_total'], df['overall_accuracy']
    )
    x_line = np.linspace(df['ies_total'].min(), df['ies_total'].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'k--', linewidth=1.5, alpha=0.5)
    
    # Add statistics text
    ax.text(0.05, 0.95, f'r = {r_value:.3f}\np = {p_value:.3f}',
            transform=ax.transAxes, fontsize=FONT_SIZES['legend'],
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('IES-R Total Score', fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Overall Accuracy (%)', fontsize=FONT_SIZES['label'])
    ax.set_title('B. Current Symptom Severity', fontsize=FONT_SIZES['title'], fontweight='bold', loc='left')
    ax.legend(fontsize=FONT_SIZES['legend'], frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=FONT_SIZES['tick'])
    
    plt.tight_layout()
    save_publication_figure(fig, save_dir / 'Figure3_trauma_correlations.png')
    plt.close()
    
    print("✓ Figure 3: Trauma correlations created")


def format_table_for_publication(table_df, table_name, save_dir):
    """Format and save table as publication-ready CSV and formatted text."""
    # Save CSV
    csv_path = save_dir / f'{table_name}.csv'
    table_df.to_csv(csv_path, index=False)
    
    # Create formatted text version
    txt_path = save_dir / f'{table_name}.txt'
    with open(txt_path, 'w') as f:
        f.write(table_df.to_string(index=False))
    
    print(f"✓ {table_name} formatted and saved")


def create_enhanced_table1(demographics, summary_data, save_dir):
    """Create enhanced Table 1 with demographics: N, Age, LESS, and IES-R statistics."""
    
    # Merge demographics with trauma groups for age data
    df = summary_data.merge(demographics, on='sona_id', how='left')
    df = df[df['trauma_group'].notna()]
    
    # Calculate statistics by group
    results = []
    
    for group in ['Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']:
        group_data = df[df['trauma_group'] == group]
        
        # N
        n = len(group_data)
        
        # Age statistics (mean ± SD, min-max)
        age_data = group_data['age_years'].dropna()
        if len(age_data) > 0:
            age_mean = age_data.mean()
            age_sd = age_data.std()
            age_min = age_data.min()
            age_max = age_data.max()
            age_str = f"{age_mean:.1f} ± {age_sd:.1f} ({age_min:.0f}-{age_max:.0f})"
        else:
            age_str = "Not available"
        
        # LESS Total statistics
        less_mean = group_data['less_total_events'].mean()
        less_sd = group_data['less_total_events'].std()
        less_min = group_data['less_total_events'].min()
        less_max = group_data['less_total_events'].max()
        less_str = f"{less_mean:.1f} ± {less_sd:.1f} ({less_min:.0f}-{less_max:.0f})"
        
        # IES-R Total statistics
        iesr_mean = group_data['ies_total'].mean()
        iesr_sd = group_data['ies_total'].std()
        iesr_min = group_data['ies_total'].min()
        iesr_max = group_data['ies_total'].max()
        iesr_str = f"{iesr_mean:.1f} ± {iesr_sd:.1f} ({iesr_min:.0f}-{iesr_max:.0f})"
        
        results.append({
            'Trauma Group': group.replace('Trauma - ', ''),
            'N': n,
            'Age (years)': age_str,
            'LESS Total': less_str,
            'IES-R Total': iesr_str
        })
    
    table1_enhanced = pd.DataFrame(results)
    
    # Save
    csv_path = save_dir / 'Table1_demographics.csv'
    table1_enhanced.to_csv(csv_path, index=False)
    
    txt_path = save_dir / 'Table1_demographics.txt'
    with open(txt_path, 'w') as f:
        f.write("Table 1: Participant Demographics and Trauma Characteristics\n")
        f.write("="*80 + "\n\n")
        f.write("Values are presented as Mean ± SD (Min-Max)\n\n")
        f.write(table1_enhanced.to_string(index=False))
    
    print(f"✓ Table 1 (demographics with Age, LESS, IES-R) created")


def create_supplementary_table_s1(save_dir):
    """Create Supplementary Table S1: Complete Regression Analyses."""
    
    # Parse all regression files
    regression_files = {
        'LESS Total': OUTPUT_DIR / 'statistical_analyses' / 'regression_accuracy_overall_LESS_Total.txt',
        'IES-R Total': OUTPUT_DIR / 'statistical_analyses' / 'regression_accuracy_overall_IES_Total.txt',
    }
    
    # Read IES-R subscales regression
    subscales_file = OUTPUT_DIR / 'statistical_analyses' / 'regression_accuracy_overall_IES_Subscales.txt'
    
    regression_data = []
    
    # Parse LESS Total
    with open(regression_files['LESS Total'], 'r') as f:
        content = f.read()
        # Extract coefficient, t-stat, p-value from the line with less_total_events
        for line in content.split('\n'):
            if 'less_total_events' in line.lower():
                parts = line.split()
                if len(parts) >= 5:
                    coef = parts[1]
                    t_val = parts[3]
                    p_val = parts[4]
                    regression_data.append({
                        'Predictor': 'LESS Total Events',
                        'β': coef,
                        't': t_val,
                        'p': p_val
                    })
                    break
    
    # Parse IES-R Total
    with open(regression_files['IES-R Total'], 'r') as f:
        content = f.read()
        for line in content.split('\n'):
            if 'ies_total' in line.lower():
                parts = line.split()
                if len(parts) >= 5:
                    coef = parts[1]
                    t_val = parts[3]
                    p_val = parts[4]
                    regression_data.append({
                        'Predictor': 'IES-R Total Score',
                        'β': coef,
                        't': t_val,
                        'p': p_val
                    })
                    break
    
    # Parse IES-R Subscales
    with open(subscales_file, 'r') as f:
        content = f.read()
        for predictor, search_term in [('IES-R Intrusion', 'ies_intrusion'),
                                        ('IES-R Avoidance', 'ies_avoidance'),
                                        ('IES-R Hyperarousal', 'ies_hyperarousal')]:
            for line in content.split('\n'):
                if search_term in line.lower():
                    parts = line.split()
                    if len(parts) >= 5:
                        coef = parts[1]
                        t_val = parts[3]
                        p_val = parts[4]
                        regression_data.append({
                            'Predictor': predictor,
                            'β': coef,
                            't': t_val,
                            'p': p_val
                        })
                        break
    
    supp_table = pd.DataFrame(regression_data)
    
    # Save
    csv_path = save_dir / 'SupplementaryTable_S1_regressions.csv'
    supp_table.to_csv(csv_path, index=False)
    
    txt_path = save_dir / 'SupplementaryTable_S1_regressions.txt'
    with open(txt_path, 'w') as f:
        f.write("Supplementary Table S1: Regression Analyses Predicting Overall Accuracy\n")
        f.write("="*80 + "\n\n")
        f.write(supp_table.to_string(index=False))
    
    print("✓ Supplementary Table S1 (complete) created")


def main():
    """Main function to create all publication figures and tables."""
    print("\n" + "="*60)
    print("Creating Publication-Ready Figures and Tables")
    print("="*60 + "\n")
    
    # Create output directory
    output_dir = FIGURES_DIR / 'publication'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    trials_data, summary_data, demographics, table1, table2, table3 = load_data()
    print("✓ Data loaded\n")
    
    # Create figures
    print("Creating figures...")
    create_figure1_learning_curves(trials_data, output_dir)
    create_figure2_load_effects(summary_data, output_dir)
    create_figure3_trauma_correlations(summary_data, output_dir)
    print()
    
    # Format tables
    print("Formatting tables...")
    create_enhanced_table1(demographics, summary_data, output_dir)
    format_table_for_publication(table2, 'Table2_trauma_scores', output_dir)
    format_table_for_publication(table3, 'Table3_task_performance', output_dir)
    create_supplementary_table_s1(output_dir)
    print()
    
    print("="*60)
    print(f"✓ All figures and tables saved to:")
    print(f"  {output_dir}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
