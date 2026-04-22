"""
Base Model Exhaustiveness Analysis
===================================

Extracts all legitimate insights from Q-learning and WM-RL models.
No new variants. No reopening model space.

Outputs:
(a) Model comparison table (AIC/BIC/AICc, Δ values)
(b) Parameter distributions (histograms, ranges, medians/IQRs)
(c) Parameter ↔ behavioral metric mapping
(d) Parameter ↔ trauma analyses

Usage:
    python scripts/analysis/analyze_base_models.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import sys

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import EXCLUDED_PARTICIPANTS

# Set plotting style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300

OUTPUT_DIR = Path('output/base_model_analysis')
FIGURES_DIR = Path('figures/base_model_analysis')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)


def deanonymize_participant_ids(df, id_column='participant_id'):
    """
    Convert anonymized participant IDs back to real IDs using mapping file.
    Handles both anonymized (anon_XXXXX) and already numeric IDs.
    """
    import json
    
    # Load mapping
    mapping_file = Path('data/participant_id_mapping.json')
    if not mapping_file.exists():
        print("Warning: No participant ID mapping file found. Using IDs as-is.")
        return df
    
    with open(mapping_file, 'r') as f:
        mapping_data = json.load(f)
    
    # Create reverse mapping from anonymized ID to real ID
    # The mapping file has filename -> assigned_id
    # We need to map the anonymized format back to assigned_id
    
    # For now, just strip 'anon_' prefix and convert to int where possible
    def convert_id(pid):
        if isinstance(pid, str):
            if pid.startswith('anon_'):
                # This is anonymized - try to extract the number
                try:
                    return int(pid.replace('anon_', ''))
                except:
                    return pid
            else:
                # Try to convert to int
                try:
                    return int(pid)
                except:
                    return pid
        return pid
    
    df[id_column] = df[id_column].apply(convert_id)
    return df


def load_model_fits():
    """Load Q-learning and WM-RL fitted parameters."""
    print("\n" + "="*80)
    print("LOADING FITTED PARAMETERS")
    print("="*80)
    
    ql_fits = pd.read_csv('output/mle/qlearning_individual_fits.csv')
    wmrl_fits = pd.read_csv('output/mle/wmrl_individual_fits.csv')
    
    # De-anonymize IDs
    ql_fits = deanonymize_participant_ids(ql_fits)
    wmrl_fits = deanonymize_participant_ids(wmrl_fits)
    
    # Apply exclusions
    ql_fits = ql_fits[~ql_fits['participant_id'].isin(EXCLUDED_PARTICIPANTS)].copy()
    wmrl_fits = wmrl_fits[~wmrl_fits['participant_id'].isin(EXCLUDED_PARTICIPANTS)].copy()
    
    print(f"\nQ-learning: {len(ql_fits)} participants")
    print(f"WM-RL: {len(wmrl_fits)} participants")
    
    # Check convergence
    ql_converged = ql_fits['converged'].sum()
    wmrl_converged = wmrl_fits['converged'].sum()
    
    print(f"\nConverged fits:")
    print(f"  Q-learning: {ql_converged}/{len(ql_fits)} ({100*ql_converged/len(ql_fits):.1f}%)")
    print(f"  WM-RL: {wmrl_converged}/{len(wmrl_fits)} ({100*wmrl_converged/len(wmrl_fits):.1f}%)")
    
    return ql_fits, wmrl_fits


def load_behavioral_data():
    """Load behavioral summary data."""
    behavioral = pd.read_csv('output/statistical_analyses/data_summary_with_groups.csv')
    behavioral = behavioral[~behavioral['sona_id'].isin(EXCLUDED_PARTICIPANTS)].copy()
    return behavioral


# =============================================================================
# (a) MODEL COMPARISON TABLE
# =============================================================================

def create_model_comparison_table(ql_fits, wmrl_fits):
    """
    Compare Q-learning vs WM-RL using AIC, BIC, AICc.
    Calculate Δ values (difference from best model for each participant).
    """
    print("\n" + "="*80)
    print("(a) MODEL COMPARISON")
    print("="*80)
    
    # Use all fits (supervisor's fits - convergence status may be unreliable)
    ql_conv = ql_fits.copy()
    wmrl_conv = wmrl_fits.copy()
    
    print(f"Note: Using all {len(ql_conv)} fits (convergence flag may not reflect actual optimization quality)")
    
    # Merge on participant_id
    comparison = ql_conv[['participant_id', 'aic', 'bic', 'aicc']].merge(
        wmrl_conv[['participant_id', 'aic', 'bic', 'aicc']],
        on='participant_id',
        suffixes=('_ql', '_wmrl')
    )
    
    # Calculate Δ values (difference from best model)
    for metric in ['aic', 'bic', 'aicc']:
        comparison[f'delta_{metric}_ql'] = comparison[f'{metric}_ql'] - comparison[[f'{metric}_ql', f'{metric}_wmrl']].min(axis=1)
        comparison[f'delta_{metric}_wmrl'] = comparison[f'{metric}_wmrl'] - comparison[[f'{metric}_ql', f'{metric}_wmrl']].min(axis=1)
    
    # Count wins
    ql_wins_aic = (comparison['aic_ql'] < comparison['aic_wmrl']).sum()
    wmrl_wins_aic = (comparison['aic_wmrl'] < comparison['aic_ql']).sum()
    
    ql_wins_bic = (comparison['bic_ql'] < comparison['bic_wmrl']).sum()
    wmrl_wins_bic = (comparison['bic_wmrl'] < comparison['bic_ql']).sum()
    
    # Summary statistics
    summary = pd.DataFrame({
        'Model': ['Q-learning', 'WM-RL'],
        'N_params': [3, 6],
        'AIC_mean': [comparison['aic_ql'].mean(), comparison['aic_wmrl'].mean()],
        'AIC_sd': [comparison['aic_ql'].std(), comparison['aic_wmrl'].std()],
        'BIC_mean': [comparison['bic_ql'].mean(), comparison['bic_wmrl'].mean()],
        'BIC_sd': [comparison['bic_ql'].std(), comparison['bic_wmrl'].std()],
        'AICc_mean': [comparison['aicc_ql'].mean(), comparison['aicc_wmrl'].mean()],
        'AICc_sd': [comparison['aicc_ql'].std(), comparison['aicc_wmrl'].std()],
        'AIC_wins': [ql_wins_aic, wmrl_wins_aic],
        'BIC_wins': [ql_wins_bic, wmrl_wins_bic],
    })
    
    # Calculate delta means
    summary['Delta_AIC_mean'] = [comparison['delta_aic_ql'].mean(), comparison['delta_aic_wmrl'].mean()]
    summary['Delta_BIC_mean'] = [comparison['delta_bic_ql'].mean(), comparison['delta_bic_wmrl'].mean()]
    
    # Save
    summary.to_csv(OUTPUT_DIR / 'model_comparison_summary.csv', index=False)
    comparison.to_csv(OUTPUT_DIR / 'model_comparison_individual.csv', index=False)
    
    print("\nModel Comparison Summary:")
    print(summary.to_string(index=False))
    
    print(f"\n** Interpretation **")
    print(f"AIC favors: {'Q-learning' if ql_wins_aic > wmrl_wins_aic else 'WM-RL'} ({max(ql_wins_aic, wmrl_wins_aic)}/{len(comparison)} participants)")
    print(f"BIC favors: {'Q-learning' if ql_wins_bic > wmrl_wins_bic else 'WM-RL'} ({max(ql_wins_bic, wmrl_wins_bic)}/{len(comparison)} participants)")
    
    # Paired t-test
    t_aic, p_aic = stats.ttest_rel(comparison['aic_ql'], comparison['aic_wmrl'])
    t_bic, p_bic = stats.ttest_rel(comparison['bic_ql'], comparison['bic_wmrl'])
    
    print(f"\nPaired t-tests (QL vs WMRL):")
    print(f"  AIC: t({len(comparison)-1}) = {t_aic:.3f}, p = {p_aic:.4f}")
    print(f"  BIC: t({len(comparison)-1}) = {t_bic:.3f}, p = {p_bic:.4f}")
    
    return summary, comparison


# =============================================================================
# (b) PARAMETER DISTRIBUTIONS
# =============================================================================

def plot_parameter_distributions(ql_fits, wmrl_fits):
    """
    Plot histograms, ranges, medians/IQRs for all parameters.
    """
    print("\n" + "="*80)
    print("(b) PARAMETER DISTRIBUTIONS")
    print("="*80)
    
    # Use all fits
    ql_conv = ql_fits.copy()
    wmrl_conv = wmrl_fits.copy()
    
    # Q-learning parameters
    ql_params = ['alpha_pos', 'alpha_neg', 'epsilon']
    
    # WM-RL parameters
    wmrl_params = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'epsilon']
    
    # Summary statistics
    stats_list = []
    
    # Q-learning
    for param in ql_params:
        values = ql_conv[param]
        stats_list.append({
            'Model': 'Q-learning',
            'Parameter': param,
            'Median': values.median(),
            'IQR': values.quantile(0.75) - values.quantile(0.25),
            'Min': values.min(),
            'Max': values.max(),
            'Mean': values.mean(),
            'SD': values.std()
        })
    
    # WM-RL
    for param in wmrl_params:
        values = wmrl_conv[param]
        stats_list.append({
            'Model': 'WM-RL',
            'Parameter': param,
            'Median': values.median(),
            'IQR': values.quantile(0.75) - values.quantile(0.25),
            'Min': values.min(),
            'Max': values.max(),
            'Mean': values.mean(),
            'SD': values.std()
        })
    
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(OUTPUT_DIR / 'parameter_distributions.csv', index=False)
    
    print("\nParameter Distribution Summary:")
    print(stats_df.to_string(index=False))
    
    # Plot Q-learning distributions
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for i, param in enumerate(ql_params):
        ax = axes[i]
        ql_conv[param].hist(bins=20, ax=ax, edgecolor='black', alpha=0.7)
        ax.axvline(ql_conv[param].median(), color='red', linestyle='--', linewidth=2, label='Median')
        ax.set_xlabel(param)
        ax.set_ylabel('Count')
        ax.set_title(f'{param}\nMedian={ql_conv[param].median():.3f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'qlearning_parameter_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot WM-RL distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, param in enumerate(wmrl_params):
        ax = axes[i]
        wmrl_conv[param].hist(bins=20, ax=ax, edgecolor='black', alpha=0.7)
        ax.axvline(wmrl_conv[param].median(), color='red', linestyle='--', linewidth=2, label='Median')
        ax.set_xlabel(param)
        ax.set_ylabel('Count')
        ax.set_title(f'{param}\nMedian={wmrl_conv[param].median():.3f}')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'wmrl_parameter_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved distribution plots to {FIGURES_DIR}/")
    
    return stats_df


# =============================================================================
# (c) PARAMETER ↔ BEHAVIORAL METRIC MAPPING
# =============================================================================

def analyze_parameter_behavior_mapping(ql_fits, wmrl_fits, behavioral):
    """
    Map parameters to behavioral metrics:
    - capacity ↔ set-size effect
    - α+ ↔ learning curve slope
    - α- ↔ feedback sensitivity
    - ε ↔ RT / variability
    """
    print("\n" + "="*80)
    print("(c) PARAMETER ↔ BEHAVIORAL MAPPING")
    print("="*80)
    
    # Merge WM-RL fits with behavioral data
    wmrl_conv = wmrl_fits.copy()
    
    # Ensure participant_id is consistent type
    if 'sona_id' in behavioral.columns:
        behavioral = behavioral.rename(columns={'sona_id': 'participant_id'})
    
    # Convert participant_id to string for matching (handles both anonymized and numeric IDs)
    wmrl_conv['participant_id'] = wmrl_conv['participant_id'].astype(str)
    behavioral['participant_id'] = behavioral['participant_id'].astype(str)
    
    merged = wmrl_conv.merge(behavioral, on='participant_id', how='inner')
    
    print(f"Merged {len(merged)} participants with both model fits and behavioral data")
    
    # Calculate behavioral metrics
    # Set-size effect: accuracy_low - accuracy_high
    merged['set_size_effect'] = merged['accuracy_low'] - merged['accuracy_high']
    
    # RT variability: could use std if available, or just mean RT
    # For now, use mean RT as proxy
    merged['rt_mean'] = merged['mean_rt_overall']
    
    # Correlations
    correlations = []
    
    # capacity ↔ set-size effect
    r, p = stats.pearsonr(merged['capacity'], merged['set_size_effect'])
    correlations.append({
        'Parameter': 'capacity',
        'Behavioral_Metric': 'set_size_effect',
        'r': r,
        'p': p,
        'n': len(merged)
    })
    
    # alpha_pos ↔ overall accuracy (learning efficiency)
    r, p = stats.pearsonr(merged['alpha_pos'], merged['accuracy_overall'])
    correlations.append({
        'Parameter': 'alpha_pos',
        'Behavioral_Metric': 'accuracy_overall',
        'r': r,
        'p': p,
        'n': len(merged)
    })
    
    # epsilon ↔ RT
    r, p = stats.pearsonr(merged['epsilon'], merged['rt_mean'])
    correlations.append({
        'Parameter': 'epsilon',
        'Behavioral_Metric': 'mean_rt',
        'r': r,
        'p': p,
        'n': len(merged)
    })
    
    corr_df = pd.DataFrame(correlations)
    corr_df.to_csv(OUTPUT_DIR / 'parameter_behavior_correlations.csv', index=False)
    
    print("\nParameter ↔ Behavior Correlations:")
    print(corr_df.to_string(index=False))
    
    # Scatterplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # capacity vs set-size effect
    ax = axes[0]
    ax.scatter(merged['capacity'], merged['set_size_effect'], alpha=0.6)
    ax.set_xlabel('WM Capacity')
    ax.set_ylabel('Set-Size Effect (Acc Low - Acc High)')
    ax.set_title(f'Capacity ↔ Set-Size Effect\nr={corr_df.iloc[0]["r"]:.3f}, p={corr_df.iloc[0]["p"]:.3f}')
    
    # alpha_pos vs accuracy
    ax = axes[1]
    ax.scatter(merged['alpha_pos'], merged['accuracy_overall'], alpha=0.6)
    ax.set_xlabel('Learning Rate α+')
    ax.set_ylabel('Overall Accuracy')
    ax.set_title(f'α+ ↔ Accuracy\nr={corr_df.iloc[1]["r"]:.3f}, p={corr_df.iloc[1]["p"]:.3f}')
    
    # epsilon vs RT
    ax = axes[2]
    ax.scatter(merged['epsilon'], merged['rt_mean'], alpha=0.6)
    ax.set_xlabel('Noise ε')
    ax.set_ylabel('Mean RT (ms)')
    ax.set_title(f'ε ↔ RT\nr={corr_df.iloc[2]["r"]:.3f}, p={corr_df.iloc[2]["p"]:.3f}')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'parameter_behavior_scatterplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved scatterplots to {FIGURES_DIR}/parameter_behavior_scatterplots.png")
    
    return corr_df


# =============================================================================
# (d) PARAMETER ↔ TRAUMA ANALYSES
# =============================================================================

def analyze_parameter_trauma_associations(ql_fits, wmrl_fits, behavioral):
    """
    Regress parameters on:
    - LESS total
    - IES-R total
    - IES-R subscales (exploratory)
    
    Create forest plot.
    """
    print("\n" + "="*80)
    print("(d) PARAMETER ↔ TRAUMA ANALYSES")
    print("="*80)
    
    # Use WM-RL (richer model)
    wmrl_conv = wmrl_fits.copy()
    
    # Ensure participant_id is consistent type
    if 'sona_id' in behavioral.columns:
        behavioral = behavioral.rename(columns={'sona_id': 'participant_id'})
    
    # Convert to string for matching
    wmrl_conv['participant_id'] = wmrl_conv['participant_id'].astype(str)
    behavioral['participant_id'] = behavioral['participant_id'].astype(str)
    
    merged = wmrl_conv.merge(behavioral, on='participant_id', how='inner')
    
    print(f"Merged {len(merged)} participants with both model fits and trauma/behavioral data")
    
    # Parameters to test
    params = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'epsilon']
    
    # Predictors
    predictors = ['less_total_events', 'ies_total', 'ies_intrusion', 'ies_avoidance', 'ies_hyperarousal']
    
    # Run regressions
    results = []
    
    for param in params:
        for predictor in predictors:
            # Simple linear regression
            X = merged[predictor].values
            y = merged[param].values
            
            # Remove NaNs
            mask = ~(np.isnan(X) | np.isnan(y))
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) < 10:  # Skip if too few data points
                continue
            
            # Standardize
            X_std = (X_clean - X_clean.mean()) / X_clean.std()
            
            # Fit
            slope, intercept, r_value, p_value, std_err = stats.linregress(X_std, y_clean)
            
            results.append({
                'Parameter': param,
                'Predictor': predictor,
                'beta': slope,
                'se': std_err,
                'r': r_value,
                'p': p_value,
                'n': len(X_clean),
                'ci_lower': slope - 1.96 * std_err,
                'ci_upper': slope + 1.96 * std_err
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'parameter_trauma_regressions.csv', index=False)
    
    print("\nParameter-Trauma Regression Results:")
    print(results_df.to_string(index=False))
    
    # Forest plot for LESS and IES-R total
    main_results = results_df[results_df['Predictor'].isin(['less_total_events', 'ies_total'])].copy()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = []
    y_labels = []
    counter = 0
    
    for predictor in ['less_total_events', 'ies_total']:
        pred_data = main_results[main_results['Predictor'] == predictor]
        
        for _, row in pred_data.iterrows():
            y_pos.append(counter)
            y_labels.append(f"{row['Parameter']} ← {predictor}")
            
            # Point estimate
            ax.plot(row['beta'], counter, 'o', color='black', markersize=8)
            
            # CI
            ax.plot([row['ci_lower'], row['ci_upper']], [counter, counter], 'k-', linewidth=2)
            
            # Significance marker
            if row['p'] < 0.05:
                ax.plot(row['beta'], counter, '*', color='red', markersize=12)
            
            counter += 1
        
        counter += 0.5  # Space between predictors
    
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Standardized β (95% CI)')
    ax.set_title('Parameter ↔ Trauma Associations\n(* p < 0.05)')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'parameter_trauma_forest_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved forest plot to {FIGURES_DIR}/parameter_trauma_forest_plot.png")
    
    return results_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*80)
    print("BASE MODEL EXHAUSTIVENESS ANALYSIS")
    print("="*80)
    print("\nModel space: Q-learning vs WM-RL (LOCKED)")
    print("\nOutputs:")
    print("  (a) Model comparison table")
    print("  (b) Parameter distributions")
    print("  (c) Parameter ↔ behavioral mapping [SKIPPED - ID mismatch]")
    print("  (d) Parameter ↔ trauma analyses [SKIPPED - ID mismatch]")
    print("="*80)
    
    # Load data
    ql_fits, wmrl_fits = load_model_fits()
    behavioral = load_behavioral_data()
    
    # (a) Model comparison
    model_comparison, individual_comparison = create_model_comparison_table(ql_fits, wmrl_fits)
    
    # (b) Parameter distributions
    param_stats = plot_parameter_distributions(ql_fits, wmrl_fits)
    
    # Check if we can merge with behavioral data
    # Convert IDs for merging test
    test_wmrl = wmrl_fits.copy()
    test_wmrl['participant_id'] = test_wmrl['participant_id'].astype(str)
    test_behavioral = behavioral.copy()
    if 'sona_id' in test_behavioral.columns:
        test_behavioral = test_behavioral.rename(columns={'sona_id': 'participant_id'})
    test_behavioral['participant_id'] = test_behavioral['participant_id'].astype(str)
    
    merged_test = test_wmrl.merge(test_behavioral, on='participant_id', how='inner')
    
    if len(merged_test) < 10:
        print("\n" + "="*80)
        print("WARNING: Participant ID Mismatch")
        print("="*80)
        print(f"\nOnly {len(merged_test)} participants could be matched between model fits and behavioral data.")
        print("This suggests the model fits use a different ID system than the current dataset.")
        print("\nSkipping analyses (c) and (d) which require behavioral/trauma data linkage.")
        print("Contact your supervisor for:")
        print("  - Updated model fits using current participant IDs, OR")
        print("  - ID mapping file to link fit IDs to SONA IDs")
        print("="*80)
    else:
        # (c) Parameter-behavior mapping
        param_behavior_corr = analyze_parameter_behavior_mapping(ql_fits, wmrl_fits, behavioral)
        
        # (d) Parameter-trauma analyses
        param_trauma_results = analyze_parameter_trauma_associations(ql_fits, wmrl_fits, behavioral)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE (Partial)")
    print("="*80)
    print(f"\nOutputs saved to:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  {FIGURES_DIR}/")
    print("\nCompleted:")
    print("  ✓ (a) Model comparison table")
    print("  ✓ (b) Parameter distributions")
    if len(merged_test) < 10:
        print("\nSkipped due to ID mismatch:")
        print("  ✗ (c) Parameter ↔ behavioral mapping")
        print("  ✗ (d) Parameter ↔ trauma analyses")
    print("\n** BASE MODEL SPACE IS NOW EXHAUSTED (for available data) **")
    print("="*80)


if __name__ == '__main__':
    main()
