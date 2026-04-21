"""
Trauma Scale Distribution and Covariation Analysis
===================================================

Per supervisor's request: 
"Perhaps as an exploratory analysis consider looking at some of the subscales 
instead of the main scale. Check the distribution of the scales against each 
other to see how they covary."

STEP 1: Check distributions
STEP 2: Check covariation (correlation matrix)
STEP 3: Exploratory regressions with subscales

Usage:
    python scripts/analysis/trauma_scale_distributions.py
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

OUTPUT_DIR = Path('output/trauma_scale_analysis')
FIGURES_DIR = Path('figures/trauma_scale_analysis')
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# Canonical path expected by manuscript/paper.qmd (resolves to manuscript/figures/ at render time
# via the ../figures/scale_distributions.png include on the paper.qmd side).
CANONICAL_FIGURE_PATH = Path('figures/scale_distributions.png')  # paper.qmd expects this location


def load_data():
    """Load behavioral data with trauma scales."""
    print("\n" + "="*80)
    print("LOADING DATA")
    print("="*80)
    
    behavioral = pd.read_csv('output/statistical_analyses/data_summary_with_groups.csv')
    behavioral = behavioral[~behavioral['sona_id'].isin(EXCLUDED_PARTICIPANTS)].copy()
    
    print(f"\nN = {len(behavioral)} participants")
    
    return behavioral


# =============================================================================
# STEP 1: CHECK DISTRIBUTIONS
# =============================================================================

def analyze_scale_distributions(data):
    """
    Analyze distribution of each scale:
    - Histogram
    - Mean, SD, Range
    - Skewness
    
    Looking for: floor effects, restricted range, variance differences
    """
    print("\n" + "="*80)
    print("STEP 1: SCALE DISTRIBUTIONS")
    print("="*80)
    
    # Define scales
    scales = {
        'LEC-5 Total Events': 'lec_total_events',
        'IES-R Total': 'ies_total',
        'IES-R Intrusion': 'ies_intrusion',
        'IES-R Avoidance': 'ies_avoidance',
        'IES-R Hyperarousal': 'ies_hyperarousal'
    }
    
    # Calculate statistics
    stats_list = []
    
    for label, col in scales.items():
        values = data[col].dropna()
        
        # Basic stats
        stats_dict = {
            'Scale': label,
            'N': len(values),
            'Mean': values.mean(),
            'SD': values.std(),
            'Median': values.median(),
            'Min': values.min(),
            'Max': values.max(),
            'Range': values.max() - values.min(),
            'Skewness': stats.skew(values),
            'Kurtosis': stats.kurtosis(values)
        }
        
        # Check for floor/ceiling effects
        # Floor effect: >25% at minimum
        floor_pct = (values == values.min()).sum() / len(values) * 100
        # Ceiling effect: >25% at maximum
        ceiling_pct = (values == values.max()).sum() / len(values) * 100
        
        stats_dict['Floor_%'] = floor_pct
        stats_dict['Ceiling_%'] = ceiling_pct
        
        stats_list.append(stats_dict)
    
    stats_df = pd.DataFrame(stats_list)
    stats_df.to_csv(OUTPUT_DIR / 'scale_distribution_statistics.csv', index=False)
    
    print("\nScale Distribution Statistics:")
    print(stats_df.to_string(index=False))
    
    # Identify issues
    print("\n" + "-"*80)
    print("DISTRIBUTION CHARACTERISTICS:")
    print("-"*80)
    
    for _, row in stats_df.iterrows():
        print(f"\n{row['Scale']}:")
        
        # Floor effect
        if row['Floor_%'] > 25:
            print(f"  ⚠ FLOOR EFFECT: {row['Floor_%']:.1f}% at minimum")
        
        # Ceiling effect
        if row['Ceiling_%'] > 25:
            print(f"  ⚠ CEILING EFFECT: {row['Ceiling_%']:.1f}% at maximum")
        
        # Skewness
        if abs(row['Skewness']) > 1:
            direction = "positive" if row['Skewness'] > 0 else "negative"
            print(f"  ⚠ HIGH SKEWNESS: {row['Skewness']:.2f} ({direction})")
        
        # Restricted range (relative to SD)
        cv = row['SD'] / row['Mean'] if row['Mean'] > 0 else 0
        print(f"  Coefficient of variation: {cv:.2f}")
        
        if cv < 0.3:
            print(f"  ⚠ RESTRICTED VARIANCE")
    
    # Compare variances
    print("\n" + "-"*80)
    print("VARIANCE COMPARISON:")
    print("-"*80)
    
    ies_subscales = stats_df[stats_df['Scale'].str.contains('IES-R') & 
                              ~stats_df['Scale'].str.contains('Total')]
    
    max_var_scale = ies_subscales.loc[ies_subscales['SD'].idxmax(), 'Scale']
    max_var_value = ies_subscales['SD'].max()
    
    print(f"\nAmong IES-R subscales, {max_var_scale} shows greatest variability (SD={max_var_value:.2f})")
    
    # Plot histograms
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (label, col) in enumerate(scales.items()):
        ax = axes[i]
        
        values = data[col].dropna()
        
        # Histogram
        ax.hist(values, bins=20, edgecolor='black', alpha=0.7)
        
        # Add mean and median lines
        ax.axvline(values.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={values.mean():.1f}')
        ax.axvline(values.median(), color='blue', linestyle='--', linewidth=2, label=f'Median={values.median():.1f}')
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{label}\nSkew={stats.skew(values):.2f}, SD={values.std():.2f}')
        ax.legend(fontsize=9)
    
    # Remove extra subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    CANONICAL_FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(CANONICAL_FIGURE_PATH, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved histogram figure to {CANONICAL_FIGURE_PATH}")
    
    return stats_df, scales


# =============================================================================
# STEP 2: CHECK COVARIATION
# =============================================================================

def analyze_scale_covariation(data, scales):
    """
    Create correlation matrix showing how scales covary.
    
    Using Spearman correlation (robust to non-normality).
    
    Looking for:
    - LESS vs IES-R independence
    - Whether IES-R total is driven by one subscale
    - Collinearity among subscales
    """
    print("\n" + "="*80)
    print("STEP 2: SCALE COVARIATION")
    print("="*80)
    
    # Extract scale columns
    scale_cols = list(scales.values())
    scale_data = data[scale_cols].copy()
    scale_data.columns = [k for k in scales.keys()]
    
    # Spearman correlation matrix
    corr_spearman = scale_data.corr(method='spearman')
    
    # Pearson for comparison
    corr_pearson = scale_data.corr(method='pearson')
    
    # Save correlations
    corr_spearman.to_csv(OUTPUT_DIR / 'correlation_matrix_spearman.csv')
    corr_pearson.to_csv(OUTPUT_DIR / 'correlation_matrix_pearson.csv')
    
    print("\nSpearman Correlation Matrix:")
    print(corr_spearman.round(3).to_string())
    
    # Compute p-values
    n = len(scale_data)
    pvals = pd.DataFrame(np.zeros_like(corr_spearman), 
                         columns=corr_spearman.columns,
                         index=corr_spearman.index)
    
    for i, col1 in enumerate(scale_data.columns):
        for j, col2 in enumerate(scale_data.columns):
            if i != j:
                # Drop NaN pairs
                mask = ~(scale_data[col1].isna() | scale_data[col2].isna())
                r, p = stats.spearmanr(scale_data.loc[mask, col1], scale_data.loc[mask, col2])
                pvals.iloc[i, j] = p
            else:
                pvals.iloc[i, j] = np.nan
    
    pvals.to_csv(OUTPUT_DIR / 'correlation_pvalues.csv')
    
    # Key findings
    print("\n" + "-"*80)
    print("KEY COVARIATION PATTERNS:")
    print("-"*80)
    
    # LEC-5 vs IES-R
    lec_iesr_corr = corr_spearman.loc['LEC-5 Total Events', 'IES-R Total']
    lec_iesr_p = pvals.loc['LEC-5 Total Events', 'IES-R Total']
    print(f"\n1. LEC-5 Total Events ↔ IES-R Total: r={lec_iesr_corr:.3f}, p={lec_iesr_p:.4f}")
    if abs(lec_iesr_corr) < 0.3:
        print("   → LEC-5 and IES-R are largely INDEPENDENT")
    elif abs(lec_iesr_corr) > 0.6:
        print("   → LEC-5 and IES-R are highly correlated")
    else:
        print("   → LEC-5 and IES-R show moderate correlation")
    
    # IES-R total vs subscales
    print("\n2. IES-R Total driven by:")
    for subscale in ['IES-R Intrusion', 'IES-R Avoidance', 'IES-R Hyperarousal']:
        r = corr_spearman.loc['IES-R Total', subscale]
        p = pvals.loc['IES-R Total', subscale]
        print(f"   {subscale}: r={r:.3f}, p={p:.4f}")
        if r > 0.8:
            print(f"      → STRONG contributor (r > .80)")
    
    # Subscale intercorrelations
    print("\n3. IES-R Subscale intercorrelations:")
    subscales = ['IES-R Intrusion', 'IES-R Avoidance', 'IES-R Hyperarousal']
    for i, sub1 in enumerate(subscales):
        for sub2 in subscales[i+1:]:
            r = corr_spearman.loc[sub1, sub2]
            p = pvals.loc[sub1, sub2]
            print(f"   {sub1} ↔ {sub2}: r={r:.3f}, p={p:.4f}")
            if r > 0.7:
                print(f"      → HIGH collinearity (may not be independent)")
    
    # Plot correlation heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_spearman, dtype=bool))
    
    # Plot
    sns.heatmap(corr_spearman, mask=mask, annot=True, fmt='.3f', 
                cmap='coolwarm', center=0, vmin=-1, vmax=1,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax)
    
    ax.set_title('Spearman Correlation Matrix\n(Trauma Scales)', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved correlation heatmap to {FIGURES_DIR}/correlation_heatmap.png")
    
    return corr_spearman, pvals


# =============================================================================
# STEP 3: EXPLORATORY REGRESSIONS WITH SUBSCALES
# =============================================================================

def exploratory_subscale_regressions(behavioral, wmrl_fits):
    """
    Exploratory regressions: Parameter ~ Subscale
    
    For each WM-RL parameter:
    - Parameter ~ Intrusion
    - Parameter ~ Avoidance  
    - Parameter ~ Hyperarousal
    
    NO totals. NO combined models. NO interactions.
    Clearly labeled as EXPLORATORY.
    """
    print("\n" + "="*80)
    print("STEP 3: EXPLORATORY SUBSCALE REGRESSIONS")
    print("="*80)
    print("\nThese are EXPLORATORY analyses only.")
    print("Testing whether individual IES-R subscales predict model parameters.")
    print("-"*80)
    
    # Check if we have model fits
    wmrl_file = Path('output/mle/wmrl_individual_fits.csv')
    if not wmrl_file.exists():
        print("\nWARNING: WM-RL fits not found. Skipping regression analyses.")
        return None
    
    # Load and de-anonymize fits
    wmrl = pd.read_csv(wmrl_file)
    
    # De-anonymize IDs
    def convert_id(pid):
        if isinstance(pid, str):
            if pid.startswith('anon_'):
                try:
                    return int(pid.replace('anon_', ''))
                except:
                    return pid
            else:
                try:
                    return int(pid)
                except:
                    return pid
        return pid
    
    wmrl['participant_id'] = wmrl['participant_id'].apply(convert_id)
    wmrl = wmrl[~wmrl['participant_id'].isin(EXCLUDED_PARTICIPANTS)].copy()
    
    # Merge with behavioral data
    behavioral['participant_id'] = behavioral['sona_id']
    merged = wmrl.merge(behavioral[['participant_id', 'ies_intrusion', 'ies_avoidance', 
                                     'ies_hyperarousal']], 
                        on='participant_id', how='inner')
    
    print(f"\nN = {len(merged)} participants with both model fits and subscale data")
    
    if len(merged) < 10:
        print("WARNING: Too few participants for reliable regression. Results may be unstable.")
        print("This suggests ID mismatch between model fits and behavioral data.")
        return None
    
    # Parameters to test
    params = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'epsilon']
    
    # Subscales to test
    subscales = ['ies_intrusion', 'ies_avoidance', 'ies_hyperarousal']
    subscale_labels = {
        'ies_intrusion': 'IES-R Intrusion',
        'ies_avoidance': 'IES-R Avoidance',
        'ies_hyperarousal': 'IES-R Hyperarousal'
    }
    
    # Run simple regressions
    results = []
    
    for param in params:
        for subscale in subscales:
            # Get data
            X = merged[subscale].values
            y = merged[param].values
            
            # Remove NaNs
            mask = ~(np.isnan(X) | np.isnan(y))
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) < 10:
                continue
            
            # Standardize predictor
            X_std = (X_clean - X_clean.mean()) / X_clean.std()
            
            # Simple linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(X_std, y_clean)
            
            results.append({
                'Parameter': param,
                'Predictor': subscale_labels[subscale],
                'beta': slope,
                'se': std_err,
                'r': r_value,
                'r_squared': r_value**2,
                'p': p_value,
                'n': len(X_clean),
                'ci_lower': slope - 1.96 * std_err,
                'ci_upper': slope + 1.96 * std_err,
                'significant': 'Yes' if p_value < 0.05 else 'No'
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'exploratory_subscale_regressions.csv', index=False)
    
    print("\nExploratory Regression Results:")
    print(results_df.to_string(index=False))
    
    # Summary
    sig_results = results_df[results_df['p'] < 0.05]
    
    print("\n" + "-"*80)
    print("SUMMARY:")
    print("-"*80)
    print(f"\nTotal tests: {len(results_df)}")
    print(f"Significant (p < .05): {len(sig_results)}")
    
    if len(sig_results) > 0:
        print("\nSignificant associations:")
        for _, row in sig_results.iterrows():
            print(f"  {row['Parameter']} ← {row['Predictor']}: β={row['beta']:.3f}, p={row['p']:.4f}")
    else:
        print("\nNo significant associations found.")
        print("This suggests IES-R subscales do not strongly predict model parameters.")
    
    # Forest plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 10))
    
    for idx, subscale_col in enumerate(subscales):
        ax = axes[idx]
        subscale_label = subscale_labels[subscale_col]
        
        # Get results for this subscale
        sub_results = results_df[results_df['Predictor'] == subscale_label].copy()
        sub_results = sub_results.sort_values('Parameter')
        
        y_pos = range(len(sub_results))
        
        for i, (_, row) in enumerate(sub_results.iterrows()):
            # Point estimate
            color = 'red' if row['p'] < 0.05 else 'black'
            ax.plot(row['beta'], i, 'o', color=color, markersize=8)
            
            # CI
            ax.plot([row['ci_lower'], row['ci_upper']], [i, i], 
                   color=color, linewidth=2, alpha=0.7)
        
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub_results['Parameter'])
        ax.set_xlabel('Standardized β (95% CI)')
        ax.set_title(f'{subscale_label}\n(Red = p < .05)')
        ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'exploratory_subscale_forest_plots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSaved forest plots to {FIGURES_DIR}/exploratory_subscale_forest_plots.png")
    
    return results_df


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "="*80)
    print("TRAUMA SCALE DISTRIBUTION AND COVARIATION ANALYSIS")
    print("="*80)
    print("\nPer supervisor's request:")
    print('"Check the distribution of the scales against each other to see how they covary"')
    print("="*80)
    
    # Load data
    data = load_data()
    
    # STEP 1: Check distributions
    stats_df, scales = analyze_scale_distributions(data)
    
    # STEP 2: Check covariation
    corr_matrix, pvals = analyze_scale_covariation(data, scales)
    
    # STEP 3: Exploratory regressions
    regression_results = exploratory_subscale_regressions(data, None)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to:")
    print(f"  {OUTPUT_DIR}/")
    print(f"  {FIGURES_DIR}/")
    
    print("\n" + "="*80)
    print("INTERPRETATION GUIDANCE")
    print("="*80)
    
    print("\nFor your manuscript, you can now state:")
    print('"Subscales differed in variance and distributional shape,')
    print(' with [SUBSCALE] showing the greatest spread."')
    
    print("\nThis justifies:")
    print("  1. Examining subscales separately")
    print("  2. Interpreting null results in context of scale properties")
    print("  3. Focusing on subscales with better psychometric properties")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
