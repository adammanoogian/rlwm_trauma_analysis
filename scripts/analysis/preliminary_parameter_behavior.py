"""
Preliminary Parameter-Behavior Alignment Analysis
WM-RL parameters predicting behavioral task metrics

WARNING: Uses N=47 overlap only (pre-exclusion fits, missing participant 10079)
Results are PRELIMINARY - for exploration only, NOT publication!

Checks:
1. Do fitted parameters explain observed behavioral performance?
2. Which parameters show strongest behavior associations?
3. Are expected relationships present (e.g., learning rates → accuracy)?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')

# Configuration
BEHAVIORAL_SUMMARY = "output/statistical_analyses/data_summary_with_groups.csv"
WMRL_PARAMS = "output/mle/wmrl_individual_fits.csv"
EXCLUDED_IDS = [10044, 10073]  # Known duplicates
OUTPUT_DIR = Path("output/preliminary")
FIGURES_DIR = Path("figures/preliminary")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Variables
PARAM_NAMES = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'epsilon']
PARAM_LABELS = {
    'alpha_pos': 'α₊',
    'alpha_neg': 'α₋',
    'epsilon': 'ε',
    'phi': 'φ',
    'rho': 'ρ',
    'capacity': 'K'
}

BEHAVIORAL_METRICS = [
    'accuracy_overall',
    'rt_median_overall', 
    'learning_slope',
    'set_size_effect_accuracy',
    'feedback_sensitivity',
    'perseveration_index'
]

BEHAVIOR_LABELS = {
    'accuracy_overall': 'Overall Accuracy',
    'rt_median_overall': 'Median RT',
    'learning_slope': 'Learning Slope',
    'set_size_effect_accuracy': 'Set Size Effect',
    'feedback_sensitivity': 'Feedback Sensitivity',
    'perseveration_index': 'Perseveration'
}

print("="*70)
print("PRELIMINARY PARAMETER-BEHAVIOR ALIGNMENT")
print("WM-RL Parameters Predicting Task Performance")
print("="*70)
print("\nWARNING: N=47 overlap only - PRELIMINARY results")
print("="*70)

# Load data
behavioral_df = pd.read_csv(BEHAVIORAL_SUMMARY)
params_df = pd.read_csv(WMRL_PARAMS)

print(f"\nLoaded behavioral: N = {len(behavioral_df)}")
print(f"Loaded WM-RL parameters: N = {len(params_df)}")

# Remove known duplicates from parameters
params_df = params_df[~params_df['participant_id'].isin(EXCLUDED_IDS)].copy()
print(f"After excluding duplicates {EXCLUDED_IDS}: N = {len(params_df)}")

# Merge on common IDs
merged_df = behavioral_df.merge(params_df, on='participant_id', how='inner', suffixes=('', '_param'))
print(f"\nMerged dataset: N = {len(merged_df)}")

if len(merged_df) < len(behavioral_df):
    missing_from_params = set(behavioral_df['participant_id']) - set(merged_df['participant_id'])
    print(f"\nWARNING: {len(missing_from_params)} participants in behavioral but not in parameters:")
    print(f"  Missing IDs: {sorted(missing_from_params)}")

# Check which behavioral metrics are available
available_behaviors = [b for b in BEHAVIORAL_METRICS if b in merged_df.columns]
print(f"\nAvailable behavioral metrics: {len(available_behaviors)}")
for b in available_behaviors:
    print(f"  - {BEHAVIOR_LABELS.get(b, b)}")

print("\n" + "="*70)
print("BIVARIATE CORRELATIONS")
print("="*70)

# Compute all bivariate correlations
correlation_results = []

for param in PARAM_NAMES:
    for behavior in available_behaviors:
        # Remove missing data
        valid_mask = merged_df[[param, behavior]].notna().all(axis=1)
        
        if valid_mask.sum() < 10:
            continue
        
        x = merged_df.loc[valid_mask, param].values
        y = merged_df.loc[valid_mask, behavior].values
        
        # Spearman correlation
        rho, p_val = spearmanr(x, y)
        
        correlation_results.append({
            'parameter': param,
            'behavior': behavior,
            'rho': rho,
            'p': p_val,
            'N': valid_mask.sum()
        })

corr_df = pd.DataFrame(correlation_results)

# FDR correction
reject, p_corrected, _, _ = multipletests(corr_df['p'].values, alpha=0.05, method='fdr_bh')
corr_df['q_FDR'] = p_corrected
corr_df['significant_FDR'] = reject

# Sort by p-value
corr_df = corr_df.sort_values('p').reset_index(drop=True)

# Save
corr_df.to_csv(OUTPUT_DIR / 'wmrl_parameter_behavior_correlations_preliminary.csv', index=False)

print(f"\nCompleted {len(corr_df)} bivariate correlations")
print(f"Significant after FDR correction: {reject.sum()}")

if reject.sum() > 0:
    print("\nSignificant associations (FDR q < 0.05):")
    sig_results = corr_df[corr_df['significant_FDR'] == True]
    for _, row in sig_results.iterrows():
        direction = "+" if row['rho'] > 0 else "-"
        print(f"  {PARAM_LABELS[row['parameter']]} {direction} {BEHAVIOR_LABELS.get(row['behavior'], row['behavior'])}: "
              f"ρ = {row['rho']:.3f}, p = {row['p']:.4f}, q = {row['q_FDR']:.4f}")
else:
    print("\n(No significant correlations after FDR correction)")

print("\nTop 10 strongest associations (uncorrected):")
for _, row in corr_df.head(10).iterrows():
    direction = "+" if row['rho'] > 0 else "-"
    sig = "*" if row['significant_FDR'] else ""
    print(f"  {PARAM_LABELS[row['parameter']]} {direction} {BEHAVIOR_LABELS.get(row['behavior'], row['behavior'])}: "
          f"ρ = {row['rho']:.3f}, p = {row['p']:.4f} {sig}")

# Heatmap of correlations
print("\n" + "="*70)
print("CREATING CORRELATION HEATMAP")
print("="*70)

# Pivot to matrix format
corr_matrix = pd.DataFrame(index=PARAM_NAMES, columns=available_behaviors, dtype=float)
p_matrix = pd.DataFrame(index=PARAM_NAMES, columns=available_behaviors, dtype=float)

for _, row in corr_df.iterrows():
    corr_matrix.loc[row['parameter'], row['behavior']] = row['rho']
    p_matrix.loc[row['parameter'], row['behavior']] = row['p']

# Plot
fig, ax = plt.subplots(figsize=(10, 6))

im = ax.imshow(corr_matrix.values.astype(float), cmap='RdBu_r', vmin=-0.6, vmax=0.6, aspect='auto')

# Labels
ax.set_xticks(np.arange(len(available_behaviors)))
ax.set_yticks(np.arange(len(PARAM_NAMES)))
ax.set_xticklabels([BEHAVIOR_LABELS.get(b, b) for b in available_behaviors], 
                   rotation=45, ha='right', fontsize=10)
ax.set_yticklabels([PARAM_LABELS[p] for p in PARAM_NAMES], fontsize=11)

# Annotate with correlations and significance stars
for i, param in enumerate(PARAM_NAMES):
    for j, behavior in enumerate(available_behaviors):
        rho = corr_matrix.loc[param, behavior]
        p = p_matrix.loc[param, behavior]
        
        if pd.notna(rho):
            # Add significance stars
            stars = ''
            if p < 0.001:
                stars = '***'
            elif p < 0.01:
                stars = '**'
            elif p < 0.05:
                stars = '*'
            
            text_color = 'white' if abs(rho) > 0.4 else 'black'
            ax.text(j, i, f'{rho:.2f}{stars}',
                   ha='center', va='center', color=text_color, fontsize=9)

plt.colorbar(im, ax=ax, label="Spearman's ρ", shrink=0.8)
plt.title("WM-RL Parameters × Behavioral Metrics\n(Preliminary N=47; * p<.05, ** p<.01, *** p<.001)", 
         fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'wmrl_parameter_behavior_heatmap_preliminary.png', dpi=300, bbox_inches='tight')
print("✓ Saved correlation heatmap")
plt.close()

# Multivariable regressions
print("\n" + "="*70)
print("MULTIVARIABLE REGRESSIONS")
print("="*70)

multi_results = []

for behavior in available_behaviors:
    # Check for sufficient data
    cols_needed = PARAM_NAMES + [behavior]
    valid_mask = merged_df[cols_needed].notna().all(axis=1)
    
    if valid_mask.sum() < len(PARAM_NAMES) + 10:
        print(f"\nSkipping {behavior}: insufficient data (N={valid_mask.sum()})")
        continue
    
    X = merged_df.loc[valid_mask, PARAM_NAMES].values
    y = merged_df.loc[valid_mask, behavior].values
    
    # Standardize
    X_std = (X - X.mean(axis=0)) / X.std(axis=0)
    y_std = (y - y.mean()) / y.std()
    
    # Add intercept
    X_const = sm.add_constant(X_std)
    
    # Fit
    model = sm.OLS(y_std, X_const).fit()
    
    print(f"\n{BEHAVIOR_LABELS.get(behavior, behavior)}:")
    print(f"  R² = {model.rsquared:.3f}, R²_adj = {model.rsquared_adj:.3f}")
    print(f"  F({model.df_model:.0f}, {model.df_resid:.0f}) = {model.fvalue:.2f}, p = {model.f_pvalue:.4f}")
    print(f"  N = {valid_mask.sum()}")
    
    # Extract parameter coefficients
    print(f"  Standardized coefficients:")
    for i, param in enumerate(PARAM_NAMES):
        beta = model.params[i+1]
        se = model.bse[i+1]
        t = model.tvalues[i+1]
        p = model.pvalues[i+1]
        
        sig = ''
        if p < 0.001:
            sig = '***'
        elif p < 0.01:
            sig = '**'
        elif p < 0.05:
            sig = '*'
        
        print(f"    {PARAM_LABELS[param]:>3s}: β = {beta:>6.3f}, SE = {se:.3f}, t = {t:>6.2f}, p = {p:.4f} {sig}")
        
        multi_results.append({
            'behavior': behavior,
            'parameter': param,
            'beta_std': beta,
            'SE': se,
            't': t,
            'p': p,
            'R²': model.rsquared,
            'R²_adj': model.rsquared_adj,
            'N': valid_mask.sum()
        })

multi_df = pd.DataFrame(multi_results)
multi_df.to_csv(OUTPUT_DIR / 'wmrl_parameter_behavior_multivariable_preliminary.csv', index=False)

# Forest plots
print("\n" + "="*70)
print("CREATING FOREST PLOTS")
print("="*70)

for behavior in available_behaviors:
    behavior_subset = corr_df[corr_df['behavior'] == behavior]
    
    if len(behavior_subset) == 0:
        continue
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Prepare data
    params_to_plot = []
    rhos = []
    
    for param in PARAM_NAMES:
        row = behavior_subset[behavior_subset['parameter'] == param]
        if len(row) > 0:
            params_to_plot.append(PARAM_LABELS[param])
            rhos.append(row['rho'].values[0])
    
    y_pos = np.arange(len(params_to_plot))[::-1]
    
    # Plot
    colors = ['darkgreen' if r > 0 else 'darkred' for r in rhos]
    ax.barh(y_pos, rhos, color=colors, alpha=0.6, edgecolor='black', linewidth=1.5)
    ax.axvline(0, color='black', linestyle='-', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(params_to_plot, fontsize=11)
    ax.set_xlabel("Spearman's ρ", fontsize=11, fontweight='bold')
    ax.set_title(f"{BEHAVIOR_LABELS.get(behavior, behavior)}\n(Preliminary N=47)", 
                fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f'wmrl_forest_{behavior}_preliminary.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"✓ Saved {len(available_behaviors)} forest plots")

# Expected relationships check
print("\n" + "="*70)
print("EXPECTED RELATIONSHIPS CHECK")
print("="*70)

print("\nTheoretical predictions:")
print("  1. Higher learning rates → better accuracy?")
print("  2. Higher WM capacity → better performance in high load?")
print("  3. Higher noise (ε) → worse performance?")

expected_checks = [
    ('alpha_pos', 'accuracy_overall', '+', 'Higher α₊ → better accuracy'),
    ('alpha_neg', 'accuracy_overall', '+', 'Higher α₋ → better accuracy'),
    ('capacity', 'set_size_effect_accuracy', '-', 'Higher K → smaller set size cost'),
    ('epsilon', 'accuracy_overall', '-', 'Higher ε → worse accuracy'),
    ('phi', 'learning_slope', '?', 'WM decay effect on learning (exploratory)')
]

print("\nActual results:")
for param, behavior, expected_dir, description in expected_checks:
    row = corr_df[(corr_df['parameter'] == param) & (corr_df['behavior'] == behavior)]
    
    if len(row) > 0:
        rho = row['rho'].values[0]
        p = row['p'].values[0]
        q = row['q_FDR'].values[0]
        
        actual_dir = '+' if rho > 0 else '-'
        match = '✓' if expected_dir == '?' or actual_dir == expected_dir else '✗'
        sig = '*' if row['significant_FDR'].values[0] else ''
        
        print(f"  {match} {description}")
        print(f"     Expected: {expected_dir}, Actual: ρ = {rho:.3f}, p = {p:.4f}, q = {q:.4f} {sig}")
    else:
        print(f"  ? {description} (data not available)")

print("\n" + "="*70)
print("KEY FINDINGS SUMMARY")
print("="*70)

# Strongest parameter predictor
strongest_param_counts = corr_df.groupby('parameter')['significant_FDR'].sum().sort_values(ascending=False)
if strongest_param_counts.max() > 0:
    print(f"\nMost predictive parameter: {PARAM_LABELS[strongest_param_counts.index[0]]} "
          f"({strongest_param_counts.iloc[0]} significant associations)")

# Most explained behavior
strongest_behavior_counts = corr_df.groupby('behavior')['significant_FDR'].sum().sort_values(ascending=False)
if strongest_behavior_counts.max() > 0:
    print(f"Most explained behavior: {BEHAVIOR_LABELS.get(strongest_behavior_counts.index[0], strongest_behavior_counts.index[0])} "
          f"({strongest_behavior_counts.iloc[0]} significant parameters)")

# Overall parameter-behavior alignment
total_tests = len(corr_df)
total_sig = corr_df['significant_FDR'].sum()
print(f"\nOverall alignment: {total_sig}/{total_tests} ({total_sig/total_tests*100:.1f}%) significant associations")

if total_sig > 0:
    print("  → Parameters DO explain behavioral variance")
else:
    print("  → Weak parameter-behavior alignment (may improve with N=48)")

print("\n" + "="*70)
print("OUTPUTS SAVED")
print("="*70)
print(f"\nTables: {OUTPUT_DIR}")
print(f"Figures: {FIGURES_DIR}")

print("\n" + "="*70)
print("IMPORTANT LIMITATIONS")
print("="*70)
print("• N=47 (not final N=48 - missing participant 10079)")
print("• WM-RL only (cannot compare to Q-learning)")
print("• Pre-exclusion parameter fits")
print("• Results are PRELIMINARY - rerun on N=48 when available")
print("="*70 + "\n")
