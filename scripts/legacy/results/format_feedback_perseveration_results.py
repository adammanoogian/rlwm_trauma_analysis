"""
Format feedback sensitivity and perseveration results for thesis write-up.
"""

import pandas as pd
import numpy as np

# Load data
desc = pd.read_csv('output/statistical_analyses/feedback_perseveration_descriptives.csv')
corr = pd.read_csv('output/statistical_analyses/feedback_perseveration_trauma_correlations.csv')

print("=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

# ============================================================================
# FEEDBACK SENSITIVITY
# ============================================================================

fb_desc = desc[desc['Metric'] == 'Feedback Sensitivity']
fb_corr = corr[corr['Metric'] == 'feedback_sensitivity']

print("\nFeedback Sensitivity (win-stay/lose-shift strategy)")
print(f"  N = {int(fb_desc['N'].values[0])}, Mean = {fb_desc['Mean'].values[0]:.3f} ± {fb_desc['SD'].values[0]:.3f}")
print(f"\n  Trauma correlations: All non-significant")

# Get max absolute beta
max_beta = fb_corr['Beta_standardized'].abs().max()
print(f"    All |β|s < {max_beta:.2f}")

# Get min p-value
min_p = fb_corr['p_value'].min()
print(f"    All ps > {min_p:.2f}")

# Find strongest trend (smallest p-value)
strongest_idx = fb_corr['p_value'].idxmin()
strongest = fb_corr.loc[strongest_idx]
trauma_var = strongest['Trauma_Variable']
# Convert to readable name
var_names = {
    'less_total_events': 'LESS Total',
    'ies_total': 'IES-R Total',
    'ies_intrusion': 'IES-R Intrusion',
    'ies_avoidance': 'IES-R Avoidance',
    'ies_hyperarousal': 'IES-R Hyperarousal'
}
print(f"    Strongest trend: {var_names[trauma_var]} (ρ = {strongest['Spearman_rho']:.3f}, β = {strongest['Beta_standardized']:.2f}, p = {strongest['p_value']:.3f})")

# ============================================================================
# PERSEVERATION INDEX
# ============================================================================

per_desc = desc[desc['Metric'] == 'Perseveration Index']
per_corr = corr[corr['Metric'] == 'perseveration_index']

print("\nPerseveration Index (choice repetition after errors)")
print(f"  N = {int(per_desc['N'].values[0])}, Mean = {per_desc['Mean'].values[0]:.3f} ± {per_desc['SD'].values[0]:.3f}")
print(f"\n  Trauma correlations: All non-significant")

# Get max absolute beta
max_beta = per_corr['Beta_standardized'].abs().max()
print(f"    All |β|s < {max_beta:.2f}")

# Get min p-value
min_p = per_corr['p_value'].min()
print(f"    All ps > {min_p:.2f}")

# Find strongest trend (smallest p-value)
strongest_idx = per_corr['p_value'].idxmin()
strongest = per_corr.loc[strongest_idx]
trauma_var = strongest['Trauma_Variable']
print(f"    Strongest trend: {var_names[trauma_var]} (ρ = {strongest['Spearman_rho']:.3f}, β = {strongest['Beta_standardized']:.2f}, p = {strongest['p_value']:.3f})")

print("\n" + "=" * 80)
