"""
Analyze feedback sensitivity and perseveration behavioral metrics.

This script computes and analyzes feedback sensitivity (win-stay/lose-shift strategies)
and perseveration (choice repetition after errors) in relation to trauma exposure
and symptom severity.

Author: Analysis Pipeline
Date: February 2026
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import EXCLUDED_PARTICIPANTS


# ============================================================================
# Configuration
# ============================================================================

# Input files
BEHAVIORAL_DATA = project_root / "output" / "statistical_analyses" / "data_summary_with_groups.csv"
TRIAL_DATA = project_root / "output" / "task_trials_long_all_participants.csv"

# Output directory
OUTPUT_DIR = project_root / "output" / "statistical_analyses"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Trauma variables
TRAUMA_VARS = [
    'less_total_events',
    'ies_total',
    'ies_intrusion',
    'ies_avoidance',
    'ies_hyperarousal'
]


# ============================================================================
# Helper Functions
# ============================================================================

def compute_feedback_sensitivity(df_participant):
    """
    Compute feedback sensitivity: (win-stay + lose-shift) / 2
    
    Win-stay: P(repeat choice | previous correct)
    Lose-shift: P(switch choice | previous incorrect)
    
    Parameters
    ----------
    df_participant : pd.DataFrame
        Trial data for one participant
        
    Returns
    -------
    float or np.nan
        Feedback sensitivity score [0-1], or NaN if insufficient data
    """
    # Ensure sorted by trial
    df = df_participant.sort_values('trial').copy()
    
    # Need previous trial info - exclude first trial and practice
    df = df[~df['is_practice']].copy()
    if len(df) < 2:
        return np.nan
    
    # Get previous trial outcomes and choices
    df['prev_correct'] = df['correct'].shift(1)
    df['prev_choice'] = df['choice'].shift(1)
    df['same_choice'] = (df['choice'] == df['prev_choice']).astype(float)
    
    # Remove first trial (no previous trial)
    df = df.iloc[1:].copy()
    
    if len(df) == 0:
        return np.nan
    
    # Win-stay: stayed after correct
    correct_trials = df[df['prev_correct'] == 1]
    if len(correct_trials) > 0:
        win_stay = correct_trials['same_choice'].mean()
    else:
        win_stay = np.nan
    
    # Lose-shift: switched after incorrect
    incorrect_trials = df[df['prev_correct'] == 0]
    if len(incorrect_trials) > 0:
        lose_shift = 1 - incorrect_trials['same_choice'].mean()  # 1 - stay rate = shift rate
    else:
        lose_shift = np.nan
    
    # Average (both must be available)
    if pd.notna(win_stay) and pd.notna(lose_shift):
        return (win_stay + lose_shift) / 2
    else:
        return np.nan


def compute_perseveration_index(df_participant):
    """
    Compute perseveration: choice repetition rate after incorrect trials.
    
    Measures tendency to repeat choices despite negative feedback.
    
    Parameters
    ----------
    df_participant : pd.DataFrame
        Trial data for one participant
        
    Returns
    -------
    float or np.nan
        Perseveration index [0-1], or NaN if insufficient data
    """
    # Ensure sorted by trial
    df = df_participant.sort_values('trial').copy()
    
    # Exclude practice trials
    df = df[~df['is_practice']].copy()
    if len(df) < 2:
        return np.nan
    
    # Get previous trial info
    df['prev_correct'] = df['correct'].shift(1)
    df['prev_choice'] = df['choice'].shift(1)
    df['same_choice'] = (df['choice'] == df['prev_choice']).astype(float)
    
    # Remove first trial
    df = df.iloc[1:].copy()
    
    # Perseveration = repetition after errors
    error_trials = df[df['prev_correct'] == 0]
    
    if len(error_trials) > 0:
        return error_trials['same_choice'].mean()
    else:
        return np.nan


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("=" * 80)
    print("FEEDBACK SENSITIVITY & PERSEVERATION ANALYSIS")
    print("=" * 80)
    print()
    
    # ========================================================================
    # Load Data
    # ========================================================================
    
    print("Loading behavioral data...")
    df_behavioral = pd.read_csv(BEHAVIORAL_DATA)
    print(f"  Loaded {len(df_behavioral)} participants")
    
    print("\nLoading trial-level data...")
    df_trials = pd.read_csv(TRIAL_DATA)
    print(f"  Loaded {len(df_trials)} trials from {df_trials['sona_id'].nunique()} participants")
    
    # Rename columns for consistency
    df_trials = df_trials.rename(columns={
        'sona_id': 'participant_id',
        'key_press': 'choice',
        'trial_in_experiment': 'trial'
    })
    
    # ========================================================================
    # Compute Metrics for Each Participant
    # ========================================================================
    
    print("\nComputing feedback sensitivity and perseveration...")
    
    metrics_list = []
    
    for participant_id in df_behavioral['sona_id'].unique():
        df_p = df_trials[df_trials['participant_id'] == participant_id].copy()
        
        if len(df_p) == 0:
            print(f"  Warning: No trial data for participant {participant_id}")
            continue
        
        # Compute metrics
        feedback_sens = compute_feedback_sensitivity(df_p)
        persev = compute_perseveration_index(df_p)
        
        metrics_list.append({
            'participant_id': participant_id,
            'feedback_sensitivity': feedback_sens,
            'perseveration_index': persev
        })
    
    df_metrics = pd.DataFrame(metrics_list)
    
    # Merge with behavioral data
    df_full = df_behavioral.merge(
        df_metrics,
        left_on='sona_id',
        right_on='participant_id',
        how='left'
    )
    
    print(f"\n  Computed for {df_metrics['feedback_sensitivity'].notna().sum()} participants (feedback)")
    print(f"  Computed for {df_metrics['perseveration_index'].notna().sum()} participants (perseveration)")
    
    # ========================================================================
    # Descriptive Statistics
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)
    
    descriptives = []
    
    for metric in ['feedback_sensitivity', 'perseveration_index']:
        data = df_full[metric].dropna()
        
        desc = {
            'Metric': metric.replace('_', ' ').title(),
            'N': len(data),
            'Mean': data.mean(),
            'SD': data.std(),
            'Median': data.median(),
            'IQR': data.quantile(0.75) - data.quantile(0.25),
            'Min': data.min(),
            'Max': data.max()
        }
        descriptives.append(desc)
        
        print(f"\n{metric.replace('_', ' ').title()}:")
        print(f"  N = {desc['N']}")
        print(f"  Mean = {desc['Mean']:.3f} (SD = {desc['SD']:.3f})")
        print(f"  Median = {desc['Median']:.3f} (IQR = {desc['IQR']:.3f})")
        print(f"  Range = [{desc['Min']:.3f}, {desc['Max']:.3f}]")
    
    df_descriptives = pd.DataFrame(descriptives)
    
    # Save
    desc_file = OUTPUT_DIR / "feedback_perseveration_descriptives.csv"
    df_descriptives.to_csv(desc_file, index=False)
    print(f"\nDescriptives saved to: {desc_file}")
    
    # ========================================================================
    # Trauma Correlations
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("TRAUMA CORRELATIONS (SPEARMAN)")
    print("=" * 80)
    
    correlation_results = []
    
    for metric in ['feedback_sensitivity', 'perseveration_index']:
        print(f"\n{metric.replace('_', ' ').title()}:")
        print("-" * 60)
        
        for trauma_var in TRAUMA_VARS:
            # Get complete cases
            mask = df_full[metric].notna() & df_full[trauma_var].notna()
            
            if mask.sum() < 3:
                print(f"  {trauma_var}: Insufficient data")
                continue
            
            # Spearman correlation
            rho, p = stats.spearmanr(
                df_full.loc[mask, trauma_var],
                df_full.loc[mask, metric]
            )
            
            # Simple linear regression for beta
            from scipy.stats import linregress
            
            # Standardize both variables for standardized beta
            x_std = stats.zscore(df_full.loc[mask, trauma_var])
            y_std = stats.zscore(df_full.loc[mask, metric])
            
            slope, intercept, r_value, p_reg, stderr = linregress(x_std, y_std)
            
            result = {
                'Metric': metric,
                'Trauma_Variable': trauma_var,
                'N': mask.sum(),
                'Spearman_rho': rho,
                'p_value': p,
                'Beta_standardized': slope,
                'R_squared': r_value**2
            }
            correlation_results.append(result)
            
            # Print
            sig_marker = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {trauma_var:25s}: ρ = {rho:6.3f}, β = {slope:6.3f}, p = {p:.4f} {sig_marker} (N = {mask.sum()})")
    
    df_correlations = pd.DataFrame(correlation_results)
    
    # Save
    corr_file = OUTPUT_DIR / "feedback_perseveration_trauma_correlations.csv"
    df_correlations.to_csv(corr_file, index=False)
    print(f"\nCorrelations saved to: {corr_file}")
    
    # ========================================================================
    # Summary Statistics for Results Section
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("SUMMARY FOR RESULTS SECTION")
    print("=" * 80)
    
    # Feedback sensitivity
    fs_correlations = df_correlations[df_correlations['Metric'] == 'feedback_sensitivity']
    max_beta_fs = fs_correlations['Beta_standardized'].abs().max()
    min_p_fs = fs_correlations['p_value'].min()
    max_p_fs = fs_correlations['p_value'].max()
    
    print("\nFeedback Sensitivity:")
    print(f"  |β| range: < {max_beta_fs:.2f}")
    print(f"  p-value range: {min_p_fs:.3f} – {max_p_fs:.3f}")
    print(f"  All non-significant: {(fs_correlations['p_value'] > 0.05).all()}")
    
    # Perseveration
    pers_correlations = df_correlations[df_correlations['Metric'] == 'perseveration_index']
    max_beta_pers = pers_correlations['Beta_standardized'].abs().max()
    min_p_pers = pers_correlations['p_value'].min()
    max_p_pers = pers_correlations['p_value'].max()
    
    print("\nPerseveration Index:")
    print(f"  |β| range: < {max_beta_pers:.2f}")
    print(f"  p-value range: {min_p_pers:.3f} – {max_p_pers:.3f}")
    print(f"  All non-significant: {(pers_correlations['p_value'] > 0.05).all()}")
    
    # Overall summary
    all_ps = df_correlations['p_value']
    all_betas = df_correlations['Beta_standardized'].abs()
    
    print("\nOverall Summary:")
    print(f"  All |β|s < {all_betas.max():.2f}")
    print(f"  All ps > {all_ps.min():.3f}")
    print(f"  Smallest p-value: {all_ps.min():.4f} ({df_correlations.loc[all_ps.idxmin(), 'Metric']} ~ {df_correlations.loc[all_ps.idxmin(), 'Trauma_Variable']})")
    
    # ========================================================================
    # Save Combined Dataset
    # ========================================================================
    
    # Save full behavioral data with metrics
    output_file = OUTPUT_DIR / "data_summary_with_feedback_perseveration.csv"
    df_full.to_csv(output_file, index=False)
    print(f"\nFull dataset saved to: {output_file}")
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
