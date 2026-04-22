"""
Create parameter-behavior correlation heatmap (Figure 8).
FDR-corrected significance with Benjamini-Hochberg procedure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from plotting_config import PlotConfig

# Apply plotting defaults
PlotConfig.apply_defaults()

# ============================================================================
# LOAD DATA
# ============================================================================

# Load Table 7 correlations
table7 = pd.read_csv('output/modelling_base_models/Table7_parameter_behavior_correlations.csv')

# Filter to M2 only
m2_data = table7[table7['Model'] == 'M2'].copy()

# Fix parameter names (replace Unicode subscripts with regular +/-)
m2_data['Parameter'] = m2_data['Parameter'].replace({'α₊': 'α+', 'α₋': 'α-'})

print("=" * 80)
print("CREATING PARAMETER-BEHAVIOR CORRELATION HEATMAP (M2)")
print("=" * 80)
print(f"\nTotal tests: {len(m2_data)}")

# ============================================================================
# APPLY FDR CORRECTION
# ============================================================================

print("\nApplying Benjamini-Hochberg FDR correction...")

# Apply FDR correction across all 42 tests
reject, pvals_corrected, _, _ = multipletests(
    m2_data['p'], 
    alpha=0.05, 
    method='fdr_bh'
)

m2_data['q_fdr'] = pvals_corrected
m2_data['fdr_sig'] = reject

# Count significant results
n_sig_uncorrected = (m2_data['p'] < 0.05).sum()
n_sig_fdr = m2_data['fdr_sig'].sum()

print(f"  Uncorrected p < .05: {n_sig_uncorrected} / {len(m2_data)}")
print(f"  FDR-corrected q < .05: {n_sig_fdr} / {len(m2_data)}")

print("\nFDR-significant correlations:")
sig_results = m2_data[m2_data['fdr_sig']][['Parameter', 'Behavioral Metric', 'ρ', 'p', 'q_fdr']]
for _, row in sig_results.iterrows():
    print(f"  {row['Parameter']} → {row['Behavioral Metric']}: ρ={row['ρ']:.3f}, q={row['q_fdr']:.4f}")

# ============================================================================
# CREATE CORRELATION MATRIX
# ============================================================================

# Pivot to matrix format
corr_matrix = m2_data.pivot(
    index='Parameter', 
    columns='Behavioral Metric', 
    values='ρ'
)

# Pivot FDR significance
fdr_matrix = m2_data.pivot(
    index='Parameter',
    columns='Behavioral Metric',
    values='fdr_sig'
)

# Define parameter order (top to bottom)
param_order = ['α+', 'α-', 'φ', 'ρ', 'K', 'ε']
corr_matrix = corr_matrix.reindex(param_order)
fdr_matrix = fdr_matrix.reindex(param_order)

# Define behavioral metric order (left to right)
metric_order = [
    'Overall Accuracy',
    'Mean RT',
    'Set-Size Effect (Accuracy)',
    'Set-Size Effect (RT)',
    'Learning Slope',
    'Feedback Sensitivity',
    'Perseveration Index'
]
corr_matrix = corr_matrix[metric_order]
fdr_matrix = fdr_matrix[metric_order]

print(f"\nHeatmap dimensions: {corr_matrix.shape[0]} parameters × {corr_matrix.shape[1]} metrics")

# ============================================================================
# CREATE HEATMAP
# ============================================================================

print("\nCreating heatmap...")

# Get font sizes
FONT_SIZES = PlotConfig.get_fontsize_dict()

# Create figure
fig, ax = plt.subplots(figsize=(14, 7))

# Create heatmap
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='RdBu_r',
    center=0,
    vmin=-1,
    vmax=1,
    cbar_kws={'label': 'Spearman ρ', 'shrink': 0.8},
    linewidths=1.5,
    linecolor='black',
    ax=ax,
    annot_kws={'fontsize': FONT_SIZES['tick']}
)

# Add FDR significance markers
for i, param in enumerate(param_order):
    for j, metric in enumerate(metric_order):
        if fdr_matrix.loc[param, metric]:
            # Add asterisk for FDR-significant cells
            ax.text(j + 0.5, i + 0.2, '*', 
                   ha='center', va='center',
                   fontsize=FONT_SIZES['title'], 
                   fontweight='bold',
                   color='black')

# Formatting
ax.set_xlabel('Behavioral Metric', fontsize=FONT_SIZES['label'], fontweight='bold')
ax.set_ylabel('Model Parameter (WM-RL)', fontsize=FONT_SIZES['label'], fontweight='bold')
ax.set_title('Parameter-Behavior Correlations (M2 Model)', 
             fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)

# Rotate x-axis labels for readability
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=FONT_SIZES['tick'])
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=FONT_SIZES['tick'])

# Adjust layout
plt.tight_layout()

# Save
output_dir = Path('figures/modelling_base_models')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'Figure8_parameter_behavior_heatmap.png', 
            dpi=PlotConfig.DPI_PRINT, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved to {output_dir / 'Figure8_parameter_behavior_heatmap.png'}")

# ============================================================================
# SAVE FDR-CORRECTED TABLE
# ============================================================================

print("\nSaving FDR-corrected results...")

# Save updated table with FDR q-values
m2_data_export = m2_data[['Parameter', 'Behavioral Metric', 'ρ', 'p', 'q_fdr', 'fdr_sig']].copy()
m2_data_export['p'] = m2_data_export['p'].apply(lambda x: f'{x:.4f}')
m2_data_export['q_fdr'] = m2_data_export['q_fdr'].apply(lambda x: f'{x:.4f}')
m2_data_export['ρ'] = m2_data_export['ρ'].apply(lambda x: f'{x:.3f}')

output_table = Path('output/modelling_base_models/Table7_M2_FDR_corrected.csv')
m2_data_export.to_csv(output_table, index=False)

print(f"  ✓ Saved to {output_table}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("HEATMAP GENERATION COMPLETE")
print("=" * 80)

print(f"\nFigure: figures/modelling_base_models/Figure8_parameter_behavior_heatmap.png")
print(f"Table:  output/modelling_base_models/Table7_M2_FDR_corrected.csv")

print(f"\nSignificance summary:")
print(f"  * = FDR q < .05 (Benjamini-Hochberg)")
print(f"  {n_sig_fdr} correlations survived FDR correction")

print("\nCaption suggestion:")
print("─" * 80)
print("""
Figure 8. Parameter-behavior correlation heatmap for the WM-RL hybrid model.
Cells display Spearman rank correlation coefficients (ρ) between model parameters
and behavioral metrics. Asterisks indicate correlations surviving FDR correction
(q < .05, Benjamini-Hochberg). Decision noise (ε) showed the strongest associations
with task performance, while WM-specific parameters (φ, ρ, K) demonstrated weaker
or non-significant relationships with behavior. N = 47 (N = 46 for Learning Slope,
N = 45 for Perseveration Index due to data constraints).
""")
print("─" * 80)
