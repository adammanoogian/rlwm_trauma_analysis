"""
Generate all computational modeling figures for thesis.
APA 7 formatted for publication.

Figures:
- Figure 4: ΔAIC/ΔBIC distributions  
- Figure 5: Parameter distributions
- Figure 6: Key parameter-behavior correlations
- Figure 7: Forest plot of parameter-trauma effects
- Supplementary Figure S3: Trauma scale histograms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from pathlib import Path
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from plotting_config import PlotConfig

# Apply plotting defaults
PlotConfig.apply_defaults()

# Load data
m1 = pd.read_csv('output/mle/qlearning_individual_fits_matched.csv')
m2_full = pd.read_csv('output/mle/wmrl_individual_fits.csv')
m2 = m2_full[m2_full['participant_id'].isin(m1['participant_id'])].copy()
beh = pd.read_csv('output/mle/behavioral_summary_matched_with_metrics.csv')

# Sort for alignment
m1 = m1.sort_values('participant_id').reset_index(drop=True)
m2 = m2.sort_values('participant_id').reset_index(drop=True)
beh = beh.sort_values('participant_id').reset_index(drop=True)

# Output directory
output_dir = Path('figures/modelling_base_models')
output_dir.mkdir(parents=True, exist_ok=True)

# Get font sizes
FONT_SIZES = PlotConfig.get_fontsize_dict()
DPI = PlotConfig.DPI_PRINT

print("=" * 80)
print("GENERATING COMPUTATIONAL MODELING FIGURES (APA 7)")
print("=" * 80)

# ============================================================================
# FIGURE 4: ΔAIC/ΔBIC DISTRIBUTIONS
# ============================================================================

print("\nFigure 4: ΔAIC/ΔBIC Distributions")

delta_aic = m2['aic'] - m1['aic']
delta_bic = m2['bic'] - m1['bic']

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: ΔAIC
ax = axes[0]
ax.hist(delta_aic, bins=15, color='#2E86AB', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
ax.axvline(delta_aic.median(), color='darkblue', linestyle='-', linewidth=2, 
           label=f'Median = {delta_aic.median():.1f}')
ax.set_xlabel('ΔAIC (M2 - M1)', fontsize=FONT_SIZES['label'])
ax.set_ylabel('Frequency', fontsize=FONT_SIZES['label'])
ax.set_title('A. ΔAIC Distribution', fontsize=FONT_SIZES['title'], 
             fontweight='bold', loc='left')
ax.legend(fontsize=FONT_SIZES['legend'], frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=FONT_SIZES['tick'])

# Panel B: ΔBIC
ax = axes[1]
ax.hist(delta_bic, bins=15, color='#F18F01', alpha=0.7, edgecolor='black')
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
ax.axvline(delta_bic.median(), color='darkorange', linestyle='-', linewidth=2, 
           label=f'Median = {delta_bic.median():.1f}')
ax.set_xlabel('ΔBIC (M2 - M1)', fontsize=FONT_SIZES['label'])
ax.set_ylabel('Frequency', fontsize=FONT_SIZES['label'])
ax.set_title('B. ΔBIC Distribution', fontsize=FONT_SIZES['title'], 
             fontweight='bold', loc='left')
ax.legend(fontsize=FONT_SIZES['legend'], frameon=False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=FONT_SIZES['tick'])

plt.tight_layout()
plt.savefig(output_dir / 'Figure4_model_comparison_deltas.png', 
            dpi=DPI, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved to {output_dir / 'Figure4_model_comparison_deltas.png'}")

# ============================================================================
# FIGURE 5: PARAMETER DISTRIBUTIONS
# ============================================================================

print("\nFigure 5: Parameter Distributions")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# M1 parameters with panel labels
params_m1 = [
    ('alpha_pos', r'$\alpha_+$', axes[0, 0], 'A'),
    ('alpha_neg', r'$\alpha_-$', axes[0, 1], 'B'),
    ('epsilon', 'ε', axes[0, 2], 'C')
]

for param, label, ax, panel_label in params_m1:
    ax.hist(m1[param], bins=15, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax.axvline(m1[param].median(), color='darkblue', linestyle='--', linewidth=2)
    ax.set_xlabel(label, fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Frequency', fontsize=FONT_SIZES['label'])
    ax.set_title(f'{label} (M1)', fontsize=FONT_SIZES['title'], fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=FONT_SIZES['tick'])
    # Add panel label
    ax.text(0.05, 0.95, panel_label, transform=ax.transAxes, 
            fontsize=FONT_SIZES['title'], fontweight='bold', va='top')

# M2 unique parameters with panel labels
params_m2 = [
    ('phi', 'φ', axes[1, 0], 'D'),
    ('rho', 'ρ', axes[1, 1], 'E'),
    ('capacity', 'K', axes[1, 2], 'F')
]

for param, label, ax, panel_label in params_m2:
    ax.hist(m2[param], bins=15, color='#F18F01', alpha=0.7, edgecolor='black')
    ax.axvline(m2[param].median(), color='darkorange', linestyle='--', linewidth=2)
    if param == 'capacity':
        ax.axvline(4, color='red', linestyle=':', linewidth=2, label='Theoretical\nWM capacity\n(K = 4)')
        ax.legend(fontsize=FONT_SIZES['legend']-2, frameon=False)
    ax.set_xlabel(label, fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Frequency', fontsize=FONT_SIZES['label'])
    ax.set_title(f'{label} (M2)', fontsize=FONT_SIZES['title'], fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=FONT_SIZES['tick'])
    # Add panel label
    ax.text(0.05, 0.95, panel_label, transform=ax.transAxes, 
            fontsize=FONT_SIZES['title'], fontweight='bold', va='top')

plt.tight_layout()
plt.subplots_adjust(hspace=0.5, bottom=0.08)
plt.savefig(output_dir / 'Figure5_parameter_distributions.png', 
            dpi=DPI, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved to {output_dir / 'Figure5_parameter_distributions.png'}")

# ============================================================================
# FIGURE 6: KEY PARAMETER-BEHAVIOR CORRELATIONS
# ============================================================================

print("\nFigure 6: Key Parameter-Behavior Correlations")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ε → accuracy (M2)
ax = axes[0]
rho, p = spearmanr(m2['epsilon'], beh['accuracy_overall'])
ax.scatter(m2['epsilon'], beh['accuracy_overall'], 
           s=50, alpha=0.6, color='#2E86AB', edgecolors='black', linewidth=0.5)
# Add regression line
z = np.polyfit(m2['epsilon'], beh['accuracy_overall'], 1)
p_fit = np.poly1d(z)
x_line = np.linspace(m2['epsilon'].min(), m2['epsilon'].max(), 100)
ax.plot(x_line, p_fit(x_line), "r--", linewidth=2, alpha=0.6)
ax.set_xlabel('ε (Decision Noise)', fontsize=FONT_SIZES['label'])
ax.set_ylabel('Overall Accuracy', fontsize=FONT_SIZES['label'])
ax.set_title(f'A. ε → Accuracy\nρ = {rho:.3f}, p < .001', 
             fontsize=FONT_SIZES['title'], fontweight='bold', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=FONT_SIZES['tick'])

# φ → accuracy
ax = axes[1]
rho, p = spearmanr(m2['phi'], beh['accuracy_overall'])
ax.scatter(m2['phi'], beh['accuracy_overall'], 
           s=50, alpha=0.6, color='#F18F01', edgecolors='black', linewidth=0.5)
z = np.polyfit(m2['phi'], beh['accuracy_overall'], 1)
p_fit = np.poly1d(z)
x_line = np.linspace(m2['phi'].min(), m2['phi'].max(), 100)
ax.plot(x_line, p_fit(x_line), "r--", linewidth=2, alpha=0.6)
ax.set_xlabel('φ (WM Decay)', fontsize=FONT_SIZES['label'])
ax.set_ylabel('Overall Accuracy', fontsize=FONT_SIZES['label'])
ax.set_title(f'B. φ → Accuracy\nρ = {rho:.3f}, p = {p:.3f}', 
             fontsize=FONT_SIZES['title'], fontweight='bold', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=FONT_SIZES['tick'])

# ε → feedback sensitivity
ax = axes[2]
rho, p = spearmanr(m2['epsilon'], beh['feedback_sensitivity'])
ax.scatter(m2['epsilon'], beh['feedback_sensitivity'], 
           s=50, alpha=0.6, color='#06A77D', edgecolors='black', linewidth=0.5)
z = np.polyfit(m2['epsilon'], beh['feedback_sensitivity'], 1)
p_fit = np.poly1d(z)
x_line = np.linspace(m2['epsilon'].min(), m2['epsilon'].max(), 100)
ax.plot(x_line, p_fit(x_line), "r--", linewidth=2, alpha=0.6)
ax.set_xlabel('ε (Decision Noise)', fontsize=FONT_SIZES['label'])
ax.set_ylabel('Feedback Sensitivity', fontsize=FONT_SIZES['label'])
ax.set_title(f'C. ε → Feedback Sensitivity\nρ = {rho:.3f}, p = {p:.3f}', 
             fontsize=FONT_SIZES['title'], fontweight='bold', loc='left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=FONT_SIZES['tick'])

plt.tight_layout()
plt.savefig(output_dir / 'Figure6_key_parameter_behavior_correlations.png', 
            dpi=DPI, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved to {output_dir / 'Figure6_key_parameter_behavior_correlations.png'}")

# ============================================================================
# FIGURE 7: FOREST PLOT OF PARAMETER-TRAUMA EFFECTS
# ============================================================================

print("\nFigure 7: Forest Plot of Parameter-Trauma Effects")

# Calculate all correlations with 95% CIs
trauma_vars = [
    ('less_total_events', 'LESS Total'),
    ('ies_total', 'IES-R Total')
]

forest_data = []

# M2 parameters (primary analysis)
for param, param_label in [('alpha_pos', 'α+'), ('alpha_neg', 'α-'), 
                            ('phi', 'φ'), ('epsilon', 'ε')]:
    for trauma, trauma_label in trauma_vars:
        rho, p = spearmanr(m2[param], beh[trauma])
        
        # Calculate 95% CI using Fisher z-transformation
        n = len(m2)
        z = np.arctanh(rho)
        se = 1 / np.sqrt(n - 3)
        ci_lower = np.tanh(z - 1.96 * se)
        ci_upper = np.tanh(z + 1.96 * se)
        
        forest_data.append({
            'param': param_label,
            'trauma': trauma_label,
            'rho': rho,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'label': f'{param_label} ~ {trauma_label}'
        })

forest_df = pd.DataFrame(forest_data)

# Create forest plot
fig, ax = plt.subplots(figsize=(10, 8))

y_positions = np.arange(len(forest_df))

# Plot CIs
for i, row in forest_df.iterrows():
    color = '#2E86AB' if 'LESS' in row['trauma'] else '#F18F01'
    ax.plot([row['ci_lower'], row['ci_upper']], [i, i], 
            linewidth=2, color=color, alpha=0.7)
    ax.scatter(row['rho'], i, s=100, color=color, zorder=3, 
               edgecolors='black', linewidth=1)

# Add zero line
ax.axvline(0, color='red', linestyle='--', linewidth=2, alpha=0.5)

# Formatting
ax.set_yticks(y_positions)
ax.set_yticklabels(forest_df['label'], fontsize=FONT_SIZES['tick'])
ax.set_xlabel('Spearman ρ (95% CI)', fontsize=FONT_SIZES['label'])
ax.set_title('Parameter-Trauma Associations (WM-RL Model)', 
             fontsize=FONT_SIZES['title'], fontweight='bold')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.tick_params(labelsize=FONT_SIZES['tick'])
ax.set_xlim([-0.5, 0.5])

# Add legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E86AB', 
           markersize=10, label='LESS Total', markeredgecolor='black'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#F18F01', 
           markersize=10, label='IES-R Total', markeredgecolor='black')
]
ax.legend(handles=legend_elements, fontsize=FONT_SIZES['legend'], 
          frameon=False, loc='lower right')

plt.tight_layout()
plt.savefig(output_dir / 'Figure7_forest_plot_parameter_trauma.png', 
            dpi=DPI, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved to {output_dir / 'Figure7_forest_plot_parameter_trauma.png'}")

# ============================================================================
# SUPPLEMENTARY FIGURE S3: TRAUMA SCALE HISTOGRAMS
# ============================================================================

print("\nSupplementary Figure S3: Trauma Scale Histograms")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

trauma_plots = [
    ('less_total_events', 'LESS Total Events', axes[0, 0], 'A'),
    ('ies_total', 'IES-R Total Score', axes[0, 1], 'B'),
    ('ies_intrusion', 'IES-R Intrusion', axes[0, 2], 'C'),
    ('ies_avoidance', 'IES-R Avoidance', axes[1, 0], 'D'),
    ('ies_hyperarousal', 'IES-R Hyperarousal', axes[1, 1], 'E')
]

for var, label, ax, panel_label in trauma_plots:
    ax.hist(beh[var], bins=15, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax.axvline(beh[var].median(), color='darkblue', linestyle='--', linewidth=2, 
               label=f'Median = {beh[var].median():.1f}')
    ax.set_xlabel(label, fontsize=FONT_SIZES['label'])
    ax.set_ylabel('Frequency', fontsize=FONT_SIZES['label'])
    ax.set_title(f'{panel_label}. {label}', fontsize=FONT_SIZES['title'], fontweight='bold')
    ax.legend(fontsize=FONT_SIZES['legend']-2, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=FONT_SIZES['tick'])

# Remove empty subplot
axes[1, 2].remove()

plt.tight_layout()
plt.savefig(output_dir / 'FigureS3_trauma_scale_histograms.png', 
            dpi=DPI, bbox_inches='tight')
plt.close()

print(f"  ✓ Saved to {output_dir / 'FigureS3_trauma_scale_histograms.png'}")

print("\n" + "=" * 80)
print("ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 80)
print(f"\nOutput directory: {output_dir}")
print("\nGenerated files:")
print("  • Figure4_model_comparison_deltas.png")
print("  • Figure5_parameter_distributions.png")
print("  • Figure6_key_parameter_behavior_correlations.png")
print("  • Figure7_forest_plot_parameter_trauma.png")
print("  • FigureS3_trauma_scale_histograms.png")
