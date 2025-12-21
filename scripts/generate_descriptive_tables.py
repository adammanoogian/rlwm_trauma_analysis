"""
Generate descriptive statistics tables for thesis.

Creates formatted tables for:
1. Demographics by trauma group
2. Task performance metrics by group and load
3. Trauma scale scores
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent))

from analysis.visualize_load_by_trauma_group import create_trauma_groups


def calculate_demographics_by_group(summary_df, trials_df):
    """
    Calculate demographic statistics by trauma group.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary participant data with trauma scores
    trials_df : pd.DataFrame
        Trial-level data (to extract demographics if needed)
        
    Returns
    -------
    pd.DataFrame
        Demographics table by trauma group
    """
    # Create trauma groups
    df_grouped, _, _ = create_trauma_groups(summary_df)
    df_clean = df_grouped[df_grouped['trauma_group'] != 'Excluded'].copy()
    
    # Demographic columns to analyze
    demo_cols = {
        'age': 'Age (years)',
        'gender': 'Gender',
        'education_status': 'Education Status',
        'country': 'Country of Residence',
        'primary_language': 'Primary Language',
        'relationship_status': 'Relationship Status',
        'living_arrangement': 'Living Arrangement',
        'screen_time': 'Daily Screen Time (hours)'
    }
    
    results = []
    groups = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    
    # Overall N
    for group in groups:
        n = (df_clean['trauma_group'] == group).sum()
        results.append({
            'Variable': 'N',
            'Group': group,
            'Value': str(n)
        })
    
    # Continuous variables (age, screen_time)
    continuous_vars = ['age', 'screen_time']
    for var in continuous_vars:
        if var in df_clean.columns:
            for group in groups:
                group_data = df_clean[df_clean['trauma_group'] == group][var].dropna()
                if len(group_data) > 0:
                    mean_val = group_data.mean()
                    sd_val = group_data.std()
                    results.append({
                        'Variable': demo_cols.get(var, var),
                        'Group': group,
                        'Value': f"{mean_val:.2f} ± {sd_val:.2f}"
                    })
    
    # Categorical variables
    categorical_vars = ['gender', 'education_status', 'country', 'primary_language', 
                       'relationship_status', 'living_arrangement']
    
    for var in categorical_vars:
        if var in df_clean.columns:
            for group in groups:
                group_data = df_clean[df_clean['trauma_group'] == group][var].dropna()
                if len(group_data) > 0:
                    # Get frequency of most common category
                    value_counts = group_data.value_counts()
                    if len(value_counts) > 0:
                        # Store counts for each category
                        for category, count in value_counts.items():
                            pct = (count / len(group_data)) * 100
                            results.append({
                                'Variable': f"{demo_cols.get(var, var)} - {category}",
                                'Group': group,
                                'Value': f"{count} ({pct:.1f}%)"
                            })
    
    # Convert to wide format
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        pivot_df = results_df.pivot(index='Variable', columns='Group', values='Value')
        # Only include groups that actually exist in the data
        existing_groups = [g for g in groups if g in pivot_df.columns]
        pivot_df = pivot_df[existing_groups]
        return pivot_df
    
    return pd.DataFrame()


def calculate_trauma_scores_by_group(summary_df):
    """
    Calculate trauma scale scores by group.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary participant data with trauma scores
        
    Returns
    -------
    pd.DataFrame
        Trauma scores table
    """
    df_grouped, _, _ = create_trauma_groups(summary_df)
    df_clean = df_grouped[df_grouped['trauma_group'] != 'Excluded'].copy()
    
    groups = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    
    trauma_vars = {
        'less_total_events': 'LESS Total Events',
        'less_personal_events': 'LESS Personal Events',
        'ies_total': 'IES-R Total',
        'ies_intrusion': 'IES-R Intrusion',
        'ies_avoidance': 'IES-R Avoidance',
        'ies_hyperarousal': 'IES-R Hyperarousal'
    }
    
    results = []
    
    for var, label in trauma_vars.items():
        if var in df_clean.columns:
            for group in groups:
                group_data = df_clean[df_clean['trauma_group'] == group][var].dropna()
                if len(group_data) > 0:
                    mean_val = group_data.mean()
                    sd_val = group_data.std()
                    results.append({
                        'Variable': label,
                        'Group': group,
                        'M': mean_val,
                        'SD': sd_val,
                        'Value': f"{mean_val:.2f} ± {sd_val:.2f}"
                    })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        pivot_df = results_df.pivot(index='Variable', columns='Group', values='Value')
        # Only include groups that actually exist in the data
        existing_groups = [g for g in groups if g in pivot_df.columns]
        pivot_df = pivot_df[existing_groups]
        return pivot_df
    
    return pd.DataFrame()


def calculate_task_performance_by_group_and_load(summary_df, trials_df):
    """
    Calculate task performance metrics by group and load.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary participant data
    trials_df : pd.DataFrame
        Trial-level data
        
    Returns
    -------
    pd.DataFrame
        Task performance table
    """
    # Filter to experimental blocks
    exp_trials = trials_df[trials_df['block'] >= 3].copy()
    
    # Add trauma groups to summary
    df_grouped, _, _ = create_trauma_groups(summary_df)
    df_clean = df_grouped[df_grouped['trauma_group'] != 'Excluded'].copy()
    
    # Merge trauma group info with trials
    exp_trials = exp_trials.merge(
        df_clean[['sona_id', 'trauma_group']], 
        on='sona_id', 
        how='inner'
    )
    
    groups = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    loads = ['low', 'high']
    
    metrics = {
        'accuracy': 'Accuracy (%)',
        'rt': 'Median RT (ms)'
    }
    
    results = []
    
    for metric, label in metrics.items():
        for load in loads:
            load_trials = exp_trials[exp_trials['load_condition'] == load]
            
            for group in groups:
                group_trials = load_trials[load_trials['trauma_group'] == group]
                
                if metric == 'accuracy':
                    # Calculate mean accuracy
                    by_participant = group_trials.groupby('sona_id')['correct'].mean()
                    if len(by_participant) > 0:
                        mean_val = by_participant.mean() * 100
                        sd_val = by_participant.std() * 100
                        results.append({
                            'Metric': label,
                            'Load': 'Low Load' if load == 'low' else 'High Load',
                            'Group': group,
                            'Value': f"{mean_val:.2f} ± {sd_val:.2f}"
                        })
                
                elif metric == 'rt':
                    # Calculate median RT per participant, then mean across participants
                    by_participant = group_trials.groupby('sona_id')['rt'].median()
                    if len(by_participant) > 0:
                        mean_val = by_participant.mean()
                        sd_val = by_participant.std()
                        results.append({
                            'Metric': label,
                            'Load': 'Low Load' if load == 'low' else 'High Load',
                            'Group': group,
                            'Value': f"{mean_val:.0f} ± {sd_val:.0f}"
                        })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        # Create multi-index table
        pivot_df = results_df.pivot_table(
            index=['Metric', 'Load'], 
            columns='Group', 
            values='Value',
            aggfunc='first'
        )
        # Only include groups that actually exist in the data
        existing_groups = [g for g in groups if g in pivot_df.columns]
        pivot_df = pivot_df[existing_groups]
        return pivot_df
    
    return pd.DataFrame()


def calculate_additional_metrics_by_group(summary_df):
    """
    Calculate additional task metrics by trauma group.
    
    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary participant data
        
    Returns
    -------
    pd.DataFrame
        Additional metrics table
    """
    df_grouped, _, _ = create_trauma_groups(summary_df)
    df_clean = df_grouped[df_grouped['trauma_group'] != 'Excluded'].copy()
    
    groups = ['No Trauma', 'Trauma - No Ongoing Impact', 'Trauma - Ongoing Impact']
    
    metrics = {
        'feedback_sensitivity': 'Feedback Sensitivity (Δ Accuracy)',
        'learning_slope': 'Learning Slope (Δ per block)',
        'performance_drop_post_reversal': 'Perseveration (% drop)',
        'adaptation_rate_post_reversal': 'Adaptation Rate (% recovery)',
        'n_reversals': 'Number of Reversals'
    }
    
    results = []
    
    for var, label in metrics.items():
        if var in df_clean.columns:
            for group in groups:
                group_data = df_clean[df_clean['trauma_group'] == group][var].dropna()
                if len(group_data) > 0:
                    mean_val = group_data.mean()
                    sd_val = group_data.std()
                    
                    # Format based on metric type
                    if 'n_reversals' in var:
                        value_str = f"{mean_val:.2f} ± {sd_val:.2f}"
                    elif 'slope' in var:
                        value_str = f"{mean_val:.4f} ± {sd_val:.4f}"
                    else:
                        value_str = f"{mean_val:.2f} ± {sd_val:.2f}"
                    
                    results.append({
                        'Metric': label,
                        'Group': group,
                        'Value': value_str
                    })
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        pivot_df = results_df.pivot(index='Metric', columns='Group', values='Value')
        # Only include groups that actually exist in the data
        existing_groups = [g for g in groups if g in pivot_df.columns]
        pivot_df = pivot_df[existing_groups]
        return pivot_df
    
    return pd.DataFrame()


def generate_all_descriptive_tables(output_dir='output/descriptives'):
    """
    Generate all descriptive statistics tables.
    
    Parameters
    ----------
    output_dir : str or Path
        Directory to save tables
    """
    print("\n" + "="*80)
    print("GENERATING DESCRIPTIVE STATISTICS TABLES")
    print("="*80)
    
    # Load data
    summary_path = Path('output/summary_participant_metrics.csv')
    trials_path = Path('output/task_trials_long_all_participants.csv')
    
    if not summary_path.exists():
        print(f"\nERROR: Summary data not found at {summary_path}")
        print("Run the data pipeline first: python run_data_pipeline.py")
        return
    
    if not trials_path.exists():
        print(f"\nERROR: Task trials not found at {trials_path}")
        print("Run the data pipeline first: python run_data_pipeline.py")
        return
    
    summary_df = pd.read_csv(summary_path)
    trials_df = pd.read_csv(trials_path)
    
    print(f"\nLoaded {len(summary_df)} participants, {len(trials_df)} trials")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate tables
    print("\n" + "="*80)
    print("Table 1: Demographics by Trauma Group")
    print("="*80)
    
    demographics_table = calculate_demographics_by_group(summary_df, trials_df)
    if len(demographics_table) > 0:
        print(demographics_table)
        save_path = output_path / 'table1_demographics.csv'
        demographics_table.to_csv(save_path)
        print(f"\nSaved: {save_path}")
    else:
        print("No demographic data available")
    
    print("\n" + "="*80)
    print("Table 2: Trauma Scores by Group")
    print("="*80)
    
    trauma_table = calculate_trauma_scores_by_group(summary_df)
    if len(trauma_table) > 0:
        print(trauma_table)
        save_path = output_path / 'table2_trauma_scores.csv'
        trauma_table.to_csv(save_path)
        print(f"\nSaved: {save_path}")
    else:
        print("No trauma score data available")
    
    print("\n" + "="*80)
    print("Table 3: Task Performance by Group and Load")
    print("="*80)
    
    performance_table = calculate_task_performance_by_group_and_load(summary_df, trials_df)
    if len(performance_table) > 0:
        print(performance_table)
        save_path = output_path / 'table3_task_performance.csv'
        performance_table.to_csv(save_path)
        print(f"\nSaved: {save_path}")
    else:
        print("No performance data available")
    
    print("\n" + "="*80)
    print("Table 4: Additional Metrics by Group")
    print("="*80)
    
    additional_table = calculate_additional_metrics_by_group(summary_df)
    if len(additional_table) > 0:
        print(additional_table)
        save_path = output_path / 'table4_additional_metrics.csv'
        additional_table.to_csv(save_path)
        print(f"\nSaved: {save_path}")
    else:
        print("No additional metrics available")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nAll tables saved to: {output_path}/")
    print()


if __name__ == '__main__':
    generate_all_descriptive_tables()
