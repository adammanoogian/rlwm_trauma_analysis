"""
Preliminary WM-RL Parameter Descriptives
(Using N=49 pre-exclusion fits - FOR EXPLORATORY PURPOSES ONLY)

This provides a quick look at WM-RL parameter distributions to understand
the fitted values while waiting for complete N=48 refits.

WARNING: These are pre-exclusion fits (N=49) and should NOT be used for 
publication. Results are preliminary only.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
WMRL_PARAMS = "output/mle/wmrl_individual_fits.csv"
EXCLUDED_IDS = [10044, 10073]  # Known duplicates to exclude
OUTPUT_DIR = Path("output/preliminary")
FIGURES_DIR = Path("figures/preliminary")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Parameter info
PARAM_NAMES = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'epsilon']
PARAM_LABELS = {
    'alpha_pos': 'α₊ (positive LR)',
    'alpha_neg': 'α₋ (negative LR)',
    'epsilon': 'ε (noise)',
    'phi': 'φ (WM decay)',
    'rho': 'ρ (WM reliance)',
    'capacity': 'K (WM capacity)'
}

PARAM_DESCRIPTIONS = {
    'alpha_pos': 'Learning rate for better-than-expected outcomes (higher = faster learning from rewards)',
    'alpha_neg': 'Learning rate for worse-than-expected outcomes (higher = faster learning from losses)',
    'epsilon': 'Random exploration / lapse rate (higher = more random choices)',
    'phi': 'WM decay rate (higher = faster forgetting of specific outcomes)',
    'rho': 'WM reliance weight (higher = more trust in recent memory vs long-term learning)',
    'capacity': 'WM capacity (max number of stimulus-action pairs remembered)'
}

print("="*70)
print("PRELIMINARY WM-RL PARAMETER DESCRIPTIVES")
print("="*70)
print("\nWARNING: Using pre-exclusion N=49 fits")
print("These results are PRELIMINARY and for exploration only")
print("="*70)

# Load data
df = pd.read_csv(WMRL_PARAMS)
print(f"\nLoaded {len(df)} participants")

# Exclude known duplicates
df_clean = df[~df['participant_id'].isin(EXCLUDED_IDS)].copy()
print(f"After excluding duplicates {EXCLUDED_IDS}: N = {len(df_clean)}")

# Check for any parameters at bounds
print("\n" + "="*70)
print("PARAMETER BOUNDARY CHECKS")
print("="*70)

if 'at_bounds' in df_clean.columns:
    # Count how many participants have each parameter at bounds
    at_bounds_counts = {}
    for param in PARAM_NAMES:
        count = df_clean['at_bounds'].str.contains(param, na=False).sum()
        at_bounds_counts[param] = count
        if count > 0:
            print(f"\n{PARAM_LABELS[param]}: {count} participants at boundary")

# Descriptive statistics
print("\n" + "="*70)
print("PARAMETER DESCRIPTIVE STATISTICS")
print("="*70)

stats_list = []

for param in PARAM_NAMES:
    values = df_clean[param].values
    
    stats_dict = {
        'parameter': param,
        'label': PARAM_LABELS[param],
        'mean': np.mean(values),
        'SD': np.std(values),
        'median': np.median(values),
        'IQR': np.percentile(values, 75) - np.percentile(values, 25),
        'min': np.min(values),
        'max': np.max(values),
        'Q25': np.percentile(values, 25),
        'Q75': np.percentile(values, 75)
    }
    stats_list.append(stats_dict)
    
    print(f"\n{PARAM_LABELS[param]}:")
    print(f"  {PARAM_DESCRIPTIONS[param]}")
    print(f"  Mean = {stats_dict['mean']:.3f}, SD = {stats_dict['SD']:.3f}")
    print(f"  Median = {stats_dict['median']:.3f}, IQR = {stats_dict['IQR']:.3f}")
    print(f"  Range = [{stats_dict['min']:.3f}, {stats_dict['max']:.3f}]")

stats_df = pd.DataFrame(stats_list)
stats_df.to_csv(OUTPUT_DIR / 'wmrl_parameter_descriptives_preliminary.csv', index=False)

# Visualizations
print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

# 1. Distribution plots
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for i, param in enumerate(PARAM_NAMES):
    values = df_clean[param].values
    
    axes[i].hist(values, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    axes[i].axvline(np.median(values), color='red', linestyle='--', linewidth=2, label='Median')
    axes[i].axvline(np.mean(values), color='orange', linestyle=':', linewidth=2, label='Mean')
    
    axes[i].set_xlabel(PARAM_LABELS[param], fontsize=11, fontweight='bold')
    axes[i].set_ylabel('Frequency', fontsize=10)
    axes[i].legend(fontsize=9)
    axes[i].set_title(f"{PARAM_LABELS[param]}\nM={np.mean(values):.2f}, SD={np.std(values):.2f}", 
                     fontsize=10)

plt.suptitle('WM-RL Parameter Distributions (Preliminary N=47)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'wmrl_parameter_distributions_preliminary.png', dpi=300, bbox_inches='tight')
print("✓ Saved parameter distributions")
plt.close()

# 2. Correlation matrix
from scipy.stats import spearmanr

corr_matrix = df_clean[PARAM_NAMES].corr(method='spearman')

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

# Labels
labels_short = [PARAM_LABELS[p] for p in PARAM_NAMES]
ax.set_xticks(np.arange(len(PARAM_NAMES)))
ax.set_yticks(np.arange(len(PARAM_NAMES)))
ax.set_xticklabels(labels_short, rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(labels_short, fontsize=10)

# Annotate
for i in range(len(PARAM_NAMES)):
    for j in range(len(PARAM_NAMES)):
        text = ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}',
                      ha='center', va='center', 
                      color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black',
                      fontsize=9, fontweight='bold' if abs(corr_matrix.iloc[i, j]) > 0.3 else 'normal')

plt.colorbar(im, ax=ax, label='Spearman ρ', shrink=0.8)
plt.title('WM-RL Parameter Correlations (Preliminary)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'wmrl_parameter_correlations_preliminary.png', dpi=300, bbox_inches='tight')
print("✓ Saved parameter correlations")
plt.close()

# 3. Fit quality metrics
if 'nll' in df_clean.columns and 'aic' in df_clean.columns:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    axes[0].hist(df_clean['nll'], bins=20, alpha=0.7, color='coral', edgecolor='black')
    axes[0].set_xlabel('Negative Log-Likelihood', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].set_title('Model Fit Quality', fontsize=11, fontweight='bold')
    
    axes[1].hist(df_clean['aic'], bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1].set_xlabel('AIC', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)
    axes[1].set_title('Akaike Information Criterion', fontsize=11, fontweight='bold')
    
    axes[2].hist(df_clean['bic'], bins=20, alpha=0.7, color='plum', edgecolor='black')
    axes[2].set_xlabel('BIC', fontsize=11)
    axes[2].set_ylabel('Frequency', fontsize=11)
    axes[2].set_title('Bayesian Information Criterion', fontsize=11, fontweight='bold')
    
    plt.suptitle('WM-RL Fit Quality Metrics (Preliminary)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'wmrl_fit_quality_preliminary.png', dpi=300, bbox_inches='tight')
    print("✓ Saved fit quality metrics")
    plt.close()

# 4. Learning rate comparison (α+ vs α-)
fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(df_clean['alpha_pos'], df_clean['alpha_neg'], 
          alpha=0.6, s=100, color='steelblue', edgecolor='black', linewidth=0.5)

# Add diagonal line
max_alpha = max(df_clean['alpha_pos'].max(), df_clean['alpha_neg'].max())
ax.plot([0, max_alpha], [0, max_alpha], 'r--', linewidth=2, alpha=0.5, label='α₊ = α₋')

ax.set_xlabel('α₊ (Positive Learning Rate)', fontsize=12, fontweight='bold')
ax.set_ylabel('α₋ (Negative Learning Rate)', fontsize=12, fontweight='bold')
ax.set_title('Learning Rate Asymmetry (Preliminary)', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.3, linestyle=':')

# Add annotations
n_positive_bias = (df_clean['alpha_pos'] > df_clean['alpha_neg']).sum()
n_negative_bias = (df_clean['alpha_neg'] > df_clean['alpha_pos']).sum()
n_symmetric = (df_clean['alpha_pos'] == df_clean['alpha_neg']).sum()

textstr = f'Positive bias: {n_positive_bias} ({n_positive_bias/len(df_clean)*100:.1f}%)\n'
textstr += f'Negative bias: {n_negative_bias} ({n_negative_bias/len(df_clean)*100:.1f}%)\n'
textstr += f'Symmetric: {n_symmetric} ({n_symmetric/len(df_clean)*100:.1f}%)'

ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'wmrl_learning_asymmetry_preliminary.png', dpi=300, bbox_inches='tight')
print("✓ Saved learning rate asymmetry plot")
plt.close()

# Summary
print("\n" + "="*70)
print("INTERPRETATION GUIDE")
print("="*70)

print("\nWhat these parameters tell us about learning:")

alpha_pos_mean = df_clean['alpha_pos'].mean()
alpha_neg_mean = df_clean['alpha_neg'].mean()

if alpha_pos_mean > alpha_neg_mean:
    print(f"\n• Learning Rates: On average, α₊ > α₋ ({alpha_pos_mean:.3f} vs {alpha_neg_mean:.3f})")
    print("  → Participants learn MORE from rewards than losses (optimism bias)")
else:
    print(f"\n• Learning Rates: On average, α₋ > α₊ ({alpha_neg_mean:.3f} vs {alpha_pos_mean:.3f})")
    print("  → Participants learn MORE from losses than rewards (negativity bias)")

phi_mean = df_clean['phi'].mean()
print(f"\n• WM Decay (φ): Mean = {phi_mean:.3f}")
if phi_mean > 0.5:
    print("  → High decay = Working memory fades quickly, relies more on RL")
else:
    print("  → Low decay = Working memory persists longer")

rho_mean = df_clean['rho'].mean()
print(f"\n• WM Reliance (ρ): Mean = {rho_mean:.3f}")
if rho_mean > 0.5:
    print("  → High reliance = Prefers using recent memory when available")
else:
    print("  → Low reliance = Prefers long-term RL values even with memory")

capacity_mean = df_clean['capacity'].mean()
print(f"\n• WM Capacity (K): Mean = {capacity_mean:.1f} items")
print(f"  → Can remember approximately {capacity_mean:.1f} stimulus-action pairs")

epsilon_mean = df_clean['epsilon'].mean()
print(f"\n• Noise (ε): Mean = {epsilon_mean:.3f}")
print(f"  → Random choices occur {epsilon_mean*100:.1f}% of the time")

print("\n" + "="*70)
print("OUTPUTS SAVED")
print("="*70)
print(f"\nTables: {OUTPUT_DIR}")
print(f"Figures: {FIGURES_DIR}")

print("\n" + "="*70)
print("REMEMBER: These are PRELIMINARY results on N=47 pre-exclusion data")
print("Do NOT use for publication - wait for complete N=48 refits!")
print("="*70 + "\n")
