"""
Computational Modeling Analysis Pipeline: M1 Q-learning vs M2 WM-RL Hybrid

Analyzes parameter estimates from maximum likelihood fitting:
- Model comparison (AIC/BIC)
- Parameter descriptives and distributions
- Parameter → behavior regressions
- Parameter → trauma regressions
- FDR correction for multiple comparisons

Author: Psychology Honours Thesis
Date: February 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.stats import shapiro, wilcoxon, ttest_rel, spearmanr
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# ============================================================================
# FILE PATHS - READ THIS CAREFULLY!
# ============================================================================
# 
# CURRENT STATUS (February 2026):
# Using N=47 MATCHED files (Q-learning restored from commit daa3fc8, missing participant 10079)
# 
# WHEN SUPERVISOR PROVIDES COMPLETE N=48 REFITS:
# 1. Comment out the "TEMPORARY" paths below
# 2. Uncomment the "FINAL" paths below  
# 3. Run this script again to get complete N=48 results
# 
# ============================================================================

# --- TEMPORARY: N=47 matched datasets (CURRENTLY USING) ---
BEHAVIORAL_SUMMARY = "output/mle/behavioral_summary_matched_with_metrics.csv"
M1_PARAMS = "output/mle/qlearning_individual_fits_matched.csv"
M2_PARAMS = "output/mle/wmrl_individual_fits_matched.csv"

# --- FINAL: N=48 complete datasets (UNCOMMENT WHEN REFITTING COMPLETE) ---
# BEHAVIORAL_SUMMARY = "output/statistical_analyses/data_summary_with_groups.csv"
# M1_PARAMS = "output/mle/qlearning_individual_fits.csv"
# M2_PARAMS = "output/mle/wmrl_individual_fits.csv"

# Output directories
OUTPUT_DIR = Path("output/modelling_base_models")
FIGURES_DIR = Path("figures/modelling_base_models")

# Create output directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Parameter lists
M1_PARAM_NAMES = ['alpha_pos', 'alpha_neg', 'epsilon']
M2_PARAM_NAMES = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'epsilon']

# Behavioral metrics
BEHAVIORAL_METRICS = [
    'accuracy_overall',
    'mean_rt_overall',
    'set_size_effect_accuracy',
    'set_size_effect_rt',
    'learning_slope',
    'feedback_sensitivity',
    'perseveration_index'
]

# Trauma variables
TRAUMA_VARS = {
    'less_total_events': 'LESS Total',
    'ies_total': 'IES-R Total',
    'ies_intrusion': 'IES-R Intrusion',
    'ies_avoidance': 'IES-R Avoidance',
    'ies_hyperarousal': 'IES-R Hyperarousal'
}

# Parameter display names
PARAM_LABELS = {
    'alpha_pos': 'α₊ (positive LR)',
    'alpha_neg': 'α₋ (negative LR)',
    'epsilon': 'ε (noise)',
    'phi': 'φ (WM decay)',
    'rho': 'ρ (WM reliance)',
    'capacity': 'K (WM capacity)'
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def check_id_integrity(behavioral_df, m1_df, m2_df):
    """Check participant ID matching across datasets."""
    print("\n" + "="*70)
    print("PARTICIPANT ID INTEGRITY CHECKS")
    print("="*70)
    
    beh_ids = set(behavioral_df['participant_id'])
    m1_ids = set(m1_df['participant_id'])
    m2_ids = set(m2_df['participant_id'])
    
    print(f"\nDataset sizes:")
    print(f"  Behavioral: N = {len(beh_ids)}")
    print(f"  M1 (Q-learning): N = {len(m1_ids)}")
    print(f"  M2 (WM-RL): N = {len(m2_ids)}")
    
    # Check for duplicates
    for name, df in [("Behavioral", behavioral_df), ("M1", m1_df), ("M2", m2_df)]:
        dupes = df['participant_id'].duplicated().sum()
        if dupes > 0:
            print(f"\n  ERROR: {name} has {dupes} duplicate IDs!")
            print(f"  Duplicates: {df[df['participant_id'].duplicated()]['participant_id'].values}")
            raise ValueError(f"Duplicate participant IDs found in {name}")
    
    # Check intersection
    all_three = beh_ids & m1_ids & m2_ids
    print(f"\nParticipants in all 3 datasets: N = {len(all_three)}")
    
    missing_from_m1 = beh_ids - m1_ids
    missing_from_m2 = beh_ids - m2_ids
    missing_from_beh = (m1_ids | m2_ids) - beh_ids
    
    if missing_from_m1:
        print(f"\n  WARNING: {len(missing_from_m1)} participants in behavioral but not M1:")
        print(f"    {sorted(missing_from_m1)}")
    
    if missing_from_m2:
        print(f"\n  WARNING: {len(missing_from_m2)} participants in behavioral but not M2:")
        print(f"    {sorted(missing_from_m2)}")
    
    if missing_from_beh:
        print(f"\n  WARNING: {len(missing_from_beh)} participants in M1/M2 but not behavioral:")
        print(f"    {sorted(missing_from_beh)}")
    
    if len(all_three) != len(beh_ids):
        raise ValueError(
            f"ID mismatch! Expected N={len(beh_ids)} in all datasets, "
            f"but only {len(all_three)} participants overlap."
        )
    
    print("\n✓ ID integrity check passed!")
    return sorted(all_three)


def test_normality(data, name="Data", alpha=0.05):
    """Test normality using Shapiro-Wilk."""
    stat, p = shapiro(data)
    is_normal = p > alpha
    return is_normal, p


def compute_standardized_beta(X, y):
    """Compute standardized regression coefficients."""
    # Standardize X and y
    X_std = (X - X.mean()) / X.std()
    y_std = (y - y.mean()) / y.std()
    
    # Add intercept
    X_std_const = sm.add_constant(X_std)
    
    # Fit model
    model = sm.OLS(y_std, X_std_const).fit()
    
    return model


def create_forest_plot(results_df, outcome_var, param_names, save_path, model_name):
    """Create forest plot for regression results."""
    fig, ax = plt.subplots(figsize=(8, len(param_names) * 0.6 + 1))
    
    # Extract data
    betas = []
    cis_lower = []
    cis_upper = []
    labels = []
    
    for param in param_names:
        row = results_df[results_df['parameter'] == param]
        if len(row) == 0:
            continue
        
        beta = row['beta_std'].values[0]
        ci = row['CI_95'].values[0]
        
        # Parse CI string "[lower, upper]"
        ci_vals = ci.strip('[]').split(',')
        lower = float(ci_vals[0])
        upper = float(ci_vals[1])
        
        betas.append(beta)
        cis_lower.append(beta - lower)
        cis_upper.append(upper - beta)
        labels.append(PARAM_LABELS.get(param, param))
    
    # Reverse for plotting (top to bottom)
    y_pos = np.arange(len(labels))[::-1]
    
    # Plot
    ax.errorbar(betas, y_pos, xerr=[cis_lower, cis_upper], 
                fmt='o', markersize=8, capsize=5, capthick=2,
                color='darkblue', ecolor='steelblue', linewidth=2)
    
    # Add vertical line at 0
    ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Standardized β (95% CI)', fontsize=11)
    ax.set_title(f'{model_name}: {outcome_var}', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def model_comparison(m1_df, m2_df, common_ids):
    """Compare M1 vs M2 using AIC and BIC."""
    print("\n" + "="*70)
    print("MODEL COMPARISON: M1 (Q-Learning) vs M2 (WM-RL)")
    print("="*70)
    
    # Filter to common IDs and sort
    m1_sub = m1_df[m1_df['participant_id'].isin(common_ids)].sort_values('participant_id').reset_index(drop=True)
    m2_sub = m2_df[m2_df['participant_id'].isin(common_ids)].sort_values('participant_id').reset_index(drop=True)
    
    # Verify matching order
    assert (m1_sub['participant_id'] == m2_sub['participant_id']).all(), "ID mismatch!"
    
    # Extract AIC and BIC (using lowercase column names from MLE output)
    m1_aic = m1_sub['aic'].values
    m1_bic = m1_sub['bic'].values
    m2_aic = m2_sub['aic'].values
    m2_bic = m2_sub['bic'].values
    
    # Compute deltas (M2 - M1; negative means M2 is better)
    delta_aic = m2_aic - m1_aic
    delta_bic = m2_bic - m1_bic
    
    # Summary statistics
    sum_m1_aic = m1_aic.sum()
    sum_m2_aic = m2_aic.sum()
    sum_m1_bic = m1_bic.sum()
    sum_m2_bic = m2_bic.sum()
    
    # Win counts (negative delta = M2 wins)
    aic_m2_wins = (delta_aic < 0).sum()
    bic_m2_wins = (delta_bic < 0).sum()
    
    print(f"\nAggregate fit indices:")
    print(f"  M1 Σ AIC = {sum_m1_aic:.1f}")
    print(f"  M2 Σ AIC = {sum_m2_aic:.1f}")
    print(f"  M1 Σ BIC = {sum_m1_bic:.1f}")
    print(f"  M2 Σ BIC = {sum_m2_bic:.1f}")
    
    print(f"\nWin counts (out of N={len(common_ids)}):")
    print(f"  AIC: M2 wins {aic_m2_wins} ({aic_m2_wins/len(common_ids)*100:.1f}%)")
    print(f"  BIC: M2 wins {bic_m2_wins} ({bic_m2_wins/len(common_ids)*100:.1f}%)")
    
    # Statistical test on deltas
    print(f"\nPaired comparison (M2 - M1):")
    print(f"  Δ AIC: M = {delta_aic.mean():.2f}, SD = {delta_aic.std():.2f}")
    print(f"  Δ BIC: M = {delta_bic.mean():.2f}, SD = {delta_bic.std():.2f}")
    
    # Test normality
    aic_normal, aic_p = test_normality(delta_aic, "Δ AIC")
    bic_normal, bic_p = test_normality(delta_bic, "Δ BIC")
    
    print(f"\n  Normality tests:")
    print(f"    Δ AIC: {'Normal' if aic_normal else 'Non-normal'} (Shapiro-Wilk p = {aic_p:.4f})")
    print(f"    Δ BIC: {'Normal' if bic_normal else 'Non-normal'} (Shapiro-Wilk p = {bic_p:.4f})")
    
    # Choose test
    if aic_normal:
        aic_stat, aic_test_p = ttest_rel(m2_aic, m1_aic)
        aic_test_name = "Paired t-test"
    else:
        aic_stat, aic_test_p = wilcoxon(m2_aic, m1_aic)
        aic_test_name = "Wilcoxon signed-rank"
    
    if bic_normal:
        bic_stat, bic_test_p = ttest_rel(m2_bic, m1_bic)
        bic_test_name = "Paired t-test"
    else:
        bic_stat, bic_test_p = wilcoxon(m2_bic, m1_bic)
        bic_test_name = "Wilcoxon signed-rank"
    
    print(f"\n  Statistical tests:")
    print(f"    AIC ({aic_test_name}): stat = {aic_stat:.2f}, p = {aic_test_p:.4f}")
    print(f"    BIC ({bic_test_name}): stat = {bic_stat:.2f}, p = {bic_test_p:.4f}")
    
    # Save results
    comparison_df = pd.DataFrame({
        'participant_id': m1_sub['participant_id'],
        'M1_AIC': m1_aic,
        'M2_AIC': m2_aic,
        'delta_AIC': delta_aic,
        'M1_BIC': m1_bic,
        'M2_BIC': m2_bic,
        'delta_BIC': delta_bic
    })
    comparison_df.to_csv(OUTPUT_DIR / 'model_comparison_per_participant.csv', index=False)
    
    summary_df = pd.DataFrame({
        'Metric': ['Sum_M1_AIC', 'Sum_M2_AIC', 'Sum_M1_BIC', 'Sum_M2_BIC',
                   'M2_wins_AIC', 'M2_wins_BIC', 'Mean_delta_AIC', 'Mean_delta_BIC',
                   'AIC_test', 'AIC_p', 'BIC_test', 'BIC_p'],
        'Value': [sum_m1_aic, sum_m2_aic, sum_m1_bic, sum_m2_bic,
                  aic_m2_wins, bic_m2_wins, delta_aic.mean(), delta_bic.mean(),
                  aic_test_name, aic_test_p, bic_test_name, bic_test_p]
    })
    summary_df.to_csv(OUTPUT_DIR / 'model_comparison_summary.csv', index=False)
    
    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Histogram of deltas
    axes[0].hist(delta_aic, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    axes[0].set_xlabel('Δ AIC (M2 - M1)', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Distribution of Δ AIC\n(negative = M2 better)', fontsize=12)
    axes[0].legend()
    
    axes[1].hist(delta_bic, bins=20, alpha=0.7, color='coral', edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    axes[1].set_xlabel('Δ BIC (M2 - M1)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Distribution of Δ BIC\n(negative = M2 better)', fontsize=12)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'model_comparison_deltas.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Win counts bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(2)
    width = 0.35
    
    m1_wins_aic = len(common_ids) - aic_m2_wins
    m1_wins_bic = len(common_ids) - bic_m2_wins
    
    ax.bar(x - width/2, [m1_wins_aic, m1_wins_bic], width, label='M1 wins', color='lightcoral')
    ax.bar(x + width/2, [aic_m2_wins, bic_m2_wins], width, label='M2 wins', color='steelblue')
    
    ax.set_ylabel('Number of participants', fontsize=11)
    ax.set_title('Model Comparison Win Counts', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['AIC', 'BIC'])
    ax.legend()
    ax.axhline(len(common_ids)/2, color='gray', linestyle=':', linewidth=1)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'model_comparison_wins.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return comparison_df


def parameter_descriptives(param_df, param_names, model_name):
    """Compute and visualize parameter descriptives."""
    print(f"\n{'='*70}")
    print(f"PARAMETER DESCRIPTIVES: {model_name}")
    print("="*70)
    
    stats_list = []
    
    for param in param_names:
        values = param_df[param].values
        
        stats_dict = {
            'parameter': param,
            'median': np.median(values),
            'IQR': np.percentile(values, 75) - np.percentile(values, 25),
            'mean': np.mean(values),
            'SD': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
        stats_list.append(stats_dict)
        
        print(f"\n{PARAM_LABELS.get(param, param)}:")
        print(f"  Median = {stats_dict['median']:.3f}, IQR = {stats_dict['IQR']:.3f}")
        print(f"  Mean = {stats_dict['mean']:.3f}, SD = {stats_dict['SD']:.3f}")
        print(f"  Range = [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")
    
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(OUTPUT_DIR / f'{model_name}_parameter_descriptives.csv', index=False)
    
    # Distribution plots
    n_params = len(param_names)
    fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(12, 6))
    axes = axes.flatten()
    
    for i, param in enumerate(param_names):
        values = param_df[param].values
        axes[i].hist(values, bins=15, alpha=0.7, color='steelblue', edgecolor='black')
        axes[i].axvline(np.median(values), color='red', linestyle='--', linewidth=2, label='Median')
        axes[i].set_xlabel(PARAM_LABELS.get(param, param), fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)
        axes[i].legend(fontsize=8)
    
    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'{model_name} Parameter Distributions', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'{model_name}_parameter_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Correlation matrix
    corr_matrix = param_df[param_names].corr(method='spearman')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Ticks and labels
    ax.set_xticks(np.arange(len(param_names)))
    ax.set_yticks(np.arange(len(param_names)))
    ax.set_xticklabels([PARAM_LABELS.get(p, p) for p in param_names], rotation=45, ha='right')
    ax.set_yticklabels([PARAM_LABELS.get(p, p) for p in param_names])
    
    # Annotate correlations
    for i in range(len(param_names)):
        for j in range(len(param_names)):
            text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=9)
    
    plt.colorbar(im, ax=ax, label='Spearman ρ')
    plt.title(f'{model_name} Parameter Correlations', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'{model_name}_parameter_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats_df


def parameter_behavior_analysis(merged_df, param_names, behavioral_metrics, model_name):
    """Analyze parameter → behavior relationships."""
    print(f"\n{'='*70}")
    print(f"PARAMETER → BEHAVIOR ANALYSIS: {model_name}")
    print("="*70)
    
    bivariate_results = []
    
    # Bivariate regressions
    for param in param_names:
        for behavior in behavioral_metrics:
            # Check for missing data
            valid_mask = merged_df[[param, behavior]].notna().all(axis=1)
            if valid_mask.sum() < 10:
                continue
            
            X = merged_df.loc[valid_mask, param].values.reshape(-1, 1)
            y = merged_df.loc[valid_mask, behavior].values
            
            # Standardize
            X_std = (X - X.mean()) / X.std()
            y_std = (y - y.mean()) / y.std()
            
            # Add intercept
            X_const = sm.add_constant(X_std)
            
            # Fit
            model = sm.OLS(y_std, X_const).fit()
            
            beta_std = model.params[1]
            se = model.bse[1]
            t_val = model.tvalues[1]
            p_val = model.pvalues[1]
            ci = model.conf_int()[1]
            r_squared = model.rsquared
            
            bivariate_results.append({
                'parameter': param,
                'behavior': behavior,
                'beta_std': beta_std,
                'SE': se,
                't': t_val,
                'p': p_val,
                'R²': r_squared,
                'CI_95': f'[{ci[0]:.3f}, {ci[1]:.3f}]',
                'N': valid_mask.sum()
            })
    
    bivariate_df = pd.DataFrame(bivariate_results)
    
    # FDR correction
    p_values = bivariate_df['p'].values
    reject, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    bivariate_df['q_FDR'] = p_corrected
    bivariate_df['significant_FDR'] = reject
    
    # Sort by p-value
    bivariate_df = bivariate_df.sort_values('p').reset_index(drop=True)
    
    bivariate_df.to_csv(OUTPUT_DIR / f'{model_name}_parameter_behavior_bivariate.csv', index=False)
    
    print(f"\nBivariate regressions completed: {len(bivariate_df)} tests")
    print(f"Significant after FDR correction: {reject.sum()}")
    
    if reject.sum() > 0:
        print("\nTop 5 strongest associations:")
        top5 = bivariate_df.head(5)
        for _, row in top5.iterrows():
            sig_marker = "*" if row['significant_FDR'] else ""
            print(f"  {PARAM_LABELS.get(row['parameter'], row['parameter'])} → {row['behavior']}: "
                  f"β = {row['beta_std']:.3f}, p = {row['p']:.4f}, q = {row['q_FDR']:.4f} {sig_marker}")
    
    # Forest plots for each behavioral metric
    for behavior in behavioral_metrics:
        behavior_subset = bivariate_df[bivariate_df['behavior'] == behavior]
        if len(behavior_subset) > 0:
            save_path = FIGURES_DIR / f'{model_name}_forest_{behavior}.png'
            create_forest_plot(behavior_subset, behavior, param_names, save_path, model_name)
    
    # Multivariable regressions
    print("\nMultivariable regressions:")
    multivariable_results = []
    
    for behavior in behavioral_metrics:
        # Check for missing data
        cols_needed = param_names + [behavior]
        valid_mask = merged_df[cols_needed].notna().all(axis=1)
        
        if valid_mask.sum() < len(param_names) + 5:  # Need sufficient N
            continue
        
        X = merged_df.loc[valid_mask, param_names].values
        y = merged_df.loc[valid_mask, behavior].values
        
        # Standardize
        X_std = (X - X.mean(axis=0)) / X.std(axis=0)
        y_std = (y - y.mean()) / y.std()
        
        # Add intercept
        X_const = sm.add_constant(X_std)
        
        # Fit
        model = sm.OLS(y_std, X_const).fit()
        
        for i, param in enumerate(param_names):
            multivariable_results.append({
                'behavior': behavior,
                'parameter': param,
                'beta_std': model.params[i+1],
                'SE': model.bse[i+1],
                't': model.tvalues[i+1],
                'p': model.pvalues[i+1],
                'R²_adj': model.rsquared_adj,
                'N': valid_mask.sum()
            })
        
        print(f"  {behavior}: R² = {model.rsquared:.3f}, R²_adj = {model.rsquared_adj:.3f}, "
              f"F = {model.fvalue:.2f}, p = {model.f_pvalue:.4f}")
    
    multivariable_df = pd.DataFrame(multivariable_results)
    multivariable_df.to_csv(OUTPUT_DIR / f'{model_name}_parameter_behavior_multivariable.csv', index=False)
    
    return bivariate_df, multivariable_df


def parameter_trauma_analysis(merged_df, param_names, trauma_vars, model_name):
    """Analyze parameter ~ trauma relationships."""
    print(f"\n{'='*70}")
    print(f"PARAMETER ← TRAUMA ANALYSIS: {model_name}")
    print("="*70)
    
    all_results = []
    
    for param in param_names:
        for trauma_var, trauma_label in trauma_vars.items():
            # Check if trauma variable exists
            if trauma_var not in merged_df.columns:
                continue
            
            # 1) Univariate: param ~ trauma_var
            valid_mask = merged_df[[param, trauma_var]].notna().all(axis=1)
            if valid_mask.sum() < 10:
                continue
            
            y = merged_df.loc[valid_mask, param].values
            X = merged_df.loc[valid_mask, trauma_var].values.reshape(-1, 1)
            
            # Standardize
            y_std = (y - y.mean()) / y.std()
            X_std = (X - X.mean()) / X.std()
            X_const = sm.add_constant(X_std)
            
            model = sm.OLS(y_std, X_const).fit()
            
            all_results.append({
                'parameter': param,
                'predictor': trauma_label,
                'model_type': 'univariate',
                'beta_std': model.params[1],
                'SE': model.bse[1],
                't': model.tvalues[1],
                'p': model.pvalues[1],
                'R²': model.rsquared,
                'N': valid_mask.sum()
            })
    
    # Also do LESS + IESR_total together
    if 'less_total_events' in merged_df.columns and 'ies_total' in merged_df.columns:
        for param in param_names:
            valid_mask = merged_df[[param, 'less_total_events', 'ies_total']].notna().all(axis=1)
            if valid_mask.sum() < 10:
                continue
            
            y = merged_df.loc[valid_mask, param].values
            X = merged_df.loc[valid_mask, ['less_total_events', 'ies_total']].values
            
            # Standardize
            y_std = (y - y.mean()) / y.std()
            X_std = (X - X.mean(axis=0)) / X.std(axis=0)
            X_const = sm.add_constant(X_std)
            
            model = sm.OLS(y_std, X_const).fit()
            
            all_results.append({
                'parameter': param,
                'predictor': 'LESS Total',
                'model_type': 'bivariate',
                'beta_std': model.params[1],
                'SE': model.bse[1],
                't': model.tvalues[1],
                'p': model.pvalues[1],
                'R²': model.rsquared,
                'N': valid_mask.sum()
            })
            
            all_results.append({
                'parameter': param,
                'predictor': 'IES-R Total',
                'model_type': 'bivariate',
                'beta_std': model.params[2],
                'SE': model.bse[2],
                't': model.tvalues[2],
                'p': model.pvalues[2],
                'R²': model.rsquared,
                'N': valid_mask.sum()
            })
    
    results_df = pd.DataFrame(all_results)
    
    # FDR correction (only on univariate tests for simplicity)
    univariate_mask = results_df['model_type'] == 'univariate'
    p_values_uni = results_df.loc[univariate_mask, 'p'].values
    
    if len(p_values_uni) > 0:
        reject, p_corrected, _, _ = multipletests(p_values_uni, alpha=0.05, method='fdr_bh')
        results_df.loc[univariate_mask, 'q_FDR'] = p_corrected
        results_df.loc[univariate_mask, 'significant_FDR'] = reject
    
    results_df = results_df.sort_values('p').reset_index(drop=True)
    results_df.to_csv(OUTPUT_DIR / f'{model_name}_parameter_trauma.csv', index=False)
    
    print(f"\nCompleted {len(results_df)} trauma → parameter tests")
    
    if 'significant_FDR' in results_df.columns:
        n_sig = results_df['significant_FDR'].sum()
        print(f"Significant after FDR correction: {n_sig}")
        
        if n_sig > 0:
            print("\nSignificant associations:")
            sig_results = results_df[results_df['significant_FDR'] == True]
            for _, row in sig_results.iterrows():
                print(f"  {row['predictor']} → {PARAM_LABELS.get(row['parameter'], row['parameter'])}: "
                      f"β = {row['beta_std']:.3f}, p = {row['p']:.4f}, q = {row['q_FDR']:.4f}")
        else:
            print("  (No significant trauma associations after FDR correction)")
    
    # Forest plots by trauma variable
    for trauma_var, trauma_label in trauma_vars.items():
        trauma_subset = results_df[(results_df['predictor'] == trauma_label) & 
                                   (results_df['model_type'] == 'univariate')]
        if len(trauma_subset) > 0:
            # Create simple forest plot
            fig, ax = plt.subplots(figsize=(8, len(param_names) * 0.6 + 1))
            
            betas = []
            ses = []
            labels = []
            
            for param in param_names:
                row = trauma_subset[trauma_subset['parameter'] == param]
                if len(row) == 0:
                    continue
                beta = row['beta_std'].values[0]
                se = row['SE'].values[0]
                betas.append(beta)
                ses.append(se * 1.96)  # 95% CI
                labels.append(PARAM_LABELS.get(param, param))
            
            if len(betas) > 0:
                y_pos = np.arange(len(labels))[::-1]
                ax.errorbar(betas, y_pos, xerr=ses, fmt='o', markersize=8, 
                           capsize=5, capthick=2, color='darkgreen', 
                           ecolor='lightgreen', linewidth=2)
                ax.axvline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(labels)
                ax.set_xlabel('Standardized β (95% CI)', fontsize=11)
                ax.set_title(f'{model_name}: {trauma_label} → Parameters', fontsize=12, fontweight='bold')
                ax.grid(axis='x', alpha=0.3, linestyle=':')
                
                plt.tight_layout()
                plt.savefig(FIGURES_DIR / f'{model_name}_trauma_{trauma_var}.png', 
                           dpi=300, bbox_inches='tight')
                plt.close()
    
    return results_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Run complete modeling analysis pipeline."""
    print("\n" + "="*70)
    print("COMPUTATIONAL MODELING ANALYSIS PIPELINE")
    print("M1 (Q-Learning) vs M2 (WM-RL Hybrid)")
    print("="*70)
    
    # Load data
    print("\nLoading datasets...")
    behavioral_df = pd.read_csv(BEHAVIORAL_SUMMARY)
    m1_df = pd.read_csv(M1_PARAMS)
    m2_df = pd.read_csv(M2_PARAMS)
    
    # Handle different column names (behavioral uses 'sona_id')
    if 'sona_id' in behavioral_df.columns and 'participant_id' not in behavioral_df.columns:
        behavioral_df['participant_id'] = behavioral_df['sona_id']
    
    print(f"  Behavioral: {len(behavioral_df)} participants")
    print(f"  M1 parameters: {len(m1_df)} participants")
    print(f"  M2 parameters: {len(m2_df)} participants")
    
    # ID integrity checks
    common_ids = check_id_integrity(behavioral_df, m1_df, m2_df)
    
    # Filter all datasets to common IDs
    behavioral_df = behavioral_df[behavioral_df['participant_id'].isin(common_ids)].reset_index(drop=True)
    m1_df = m1_df[m1_df['participant_id'].isin(common_ids)].reset_index(drop=True)
    m2_df = m2_df[m2_df['participant_id'].isin(common_ids)].reset_index(drop=True)
    
    # 1. Model comparison
    comparison_df = model_comparison(m1_df, m2_df, common_ids)
    
    # 2. Parameter descriptives M1
    m1_stats = parameter_descriptives(m1_df, M1_PARAM_NAMES, 'M1_QLearning')
    
    # 3. Parameter descriptives M2
    m2_stats = parameter_descriptives(m2_df, M2_PARAM_NAMES, 'M2_WMRL')
    
    # 4. Merge behavioral + parameters for regression analyses
    m1_merged = behavioral_df.merge(m1_df, on='participant_id', how='inner')
    m2_merged = behavioral_df.merge(m2_df, on='participant_id', how='inner')
    
    # 5. Parameter → behavior (M1)
    m1_biv_beh, m1_multi_beh = parameter_behavior_analysis(
        m1_merged, M1_PARAM_NAMES, BEHAVIORAL_METRICS, 'M1_QLearning'
    )
    
    # 6. Parameter → behavior (M2)
    m2_biv_beh, m2_multi_beh = parameter_behavior_analysis(
        m2_merged, M2_PARAM_NAMES, BEHAVIORAL_METRICS, 'M2_WMRL'
    )
    
    # 7. Parameter ← trauma (M1)
    m1_trauma = parameter_trauma_analysis(
        m1_merged, M1_PARAM_NAMES, TRAUMA_VARS, 'M1_QLearning'
    )
    
    # 8. Parameter ← trauma (M2)
    m2_trauma = parameter_trauma_analysis(
        m2_merged, M2_PARAM_NAMES, TRAUMA_VARS, 'M2_WMRL'
    )
    
    # Print summary report
    print("\n" + "="*70)
    print("ANALYSIS SUMMARY")
    print("="*70)
    
    print(f"\n✓ Model comparison complete")
    print(f"  - AIC favors: {'M2' if comparison_df['delta_AIC'].mean() < 0 else 'M1'}")
    print(f"  - BIC favors: {'M2' if comparison_df['delta_BIC'].mean() < 0 else 'M1'}")
    
    print(f"\n✓ Parameter descriptives complete")
    print(f"  - M1: {len(M1_PARAM_NAMES)} parameters analyzed")
    print(f"  - M2: {len(M2_PARAM_NAMES)} parameters analyzed")
    
    print(f"\n✓ Parameter → behavior regressions complete")
    m1_sig_beh = m1_biv_beh['significant_FDR'].sum() if 'significant_FDR' in m1_biv_beh.columns else 0
    m2_sig_beh = m2_biv_beh['significant_FDR'].sum() if 'significant_FDR' in m2_biv_beh.columns else 0
    print(f"  - M1: {m1_sig_beh} significant associations (FDR-corrected)")
    print(f"  - M2: {m2_sig_beh} significant associations (FDR-corrected)")
    
    print(f"\n✓ Parameter ← trauma regressions complete")
    m1_sig_trauma = m1_trauma['significant_FDR'].sum() if 'significant_FDR' in m1_trauma.columns else 0
    m2_sig_trauma = m2_trauma['significant_FDR'].sum() if 'significant_FDR' in m2_trauma.columns else 0
    print(f"  - M1: {m1_sig_trauma} significant associations (FDR-corrected)")
    print(f"  - M2: {m2_sig_trauma} significant associations (FDR-corrected)")
    
    print(f"\n✓ All outputs saved to:")
    print(f"  - Tables: {OUTPUT_DIR}")
    print(f"  - Figures: {FIGURES_DIR}")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
