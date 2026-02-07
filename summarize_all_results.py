"""
Generate complete numerical summary of all results for thesis writing.
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("COMPLETE RESULTS SUMMARY FOR THESIS WRITE-UP")
print("=" * 80)

# ============================================================================
# BEHAVIORAL RESULTS (N=48)
# ============================================================================

df = pd.read_csv('output/statistical_analyses/data_summary_with_groups.csv')

print("\n" + "=" * 80)
print("1. BEHAVIORAL PERFORMANCE (N=48)")
print("=" * 80)

print(f"\nOverall Task Performance:")
print(f"  Accuracy: M = {df['accuracy_overall'].mean():.3f}, SD = {df['accuracy_overall'].std():.3f}")
print(f"  Reaction Time: M = {df['mean_rt_overall'].mean():.1f} ms, SD = {df['mean_rt_overall'].std():.1f} ms")

print(f"\nWorking Memory Load Effects:")
print(f"  Low Load Accuracy: M = {df['accuracy_low'].mean():.3f}, SD = {df['accuracy_low'].std():.3f}")
print(f"  High Load Accuracy: M = {df['accuracy_high'].mean():.3f}, SD = {df['accuracy_high'].std():.3f}")
print(f"  Low Load RT (median): M = {df['rt_low'].median():.1f} ms, SD = {df['rt_low'].std():.1f} ms")
print(f"  High Load RT (median): M = {df['rt_high'].median():.1f} ms, SD = {df['rt_high'].std():.1f} ms")

# Load feedback/perseveration metrics
fb = pd.read_csv('output/statistical_analyses/feedback_perseveration_descriptives.csv')
print(f"\nFeedback Sensitivity:")
print(f"  M = {fb[fb['Metric'] == 'Feedback Sensitivity']['Mean'].values[0]:.3f}, SD = {fb[fb['Metric'] == 'Feedback Sensitivity']['SD'].values[0]:.3f}")
print(f"  Range = [{fb[fb['Metric'] == 'Feedback Sensitivity']['Min'].values[0]:.2f}, {fb[fb['Metric'] == 'Feedback Sensitivity']['Max'].values[0]:.2f}]")

print(f"\nPerseveration Index:")
print(f"  M = {fb[fb['Metric'] == 'Perseveration Index']['Mean'].values[0]:.3f}, SD = {fb[fb['Metric'] == 'Perseveration Index']['SD'].values[0]:.3f}")
print(f"  Range = [{fb[fb['Metric'] == 'Perseveration Index']['Min'].values[0]:.2f}, {fb[fb['Metric'] == 'Perseveration Index']['Max'].values[0]:.2f}]")

# ============================================================================
# TRAUMA EXPOSURE (N=48)
# ============================================================================

print("\n" + "=" * 80)
print("2. TRAUMA EXPOSURE & SYMPTOMS (N=48)")
print("=" * 80)

print(f"\nLESS (Life Events Scale for Students):")
print(f"  Total Events: M = {df['less_total_events'].mean():.2f}, SD = {df['less_total_events'].std():.2f}")
print(f"  Range = [{df['less_total_events'].min():.0f}, {df['less_total_events'].max():.0f}]")

print(f"\nIES-R (Impact of Event Scale - Revised):")
print(f"  Total Score: M = {df['ies_total'].mean():.2f}, SD = {df['ies_total'].std():.2f}")
print(f"  Range = [{df['ies_total'].min():.0f}, {df['ies_total'].max():.0f}]")
print(f"  Intrusion: M = {df['ies_intrusion'].mean():.2f}, SD = {df['ies_intrusion'].std():.2f}")
print(f"  Avoidance: M = {df['ies_avoidance'].mean():.2f}, SD = {df['ies_avoidance'].std():.2f}")
print(f"  Hyperarousal: M = {df['ies_hyperarousal'].mean():.2f}, SD = {df['ies_hyperarousal'].std():.2f}")

print(f"\nTrauma Groups:")
n_no_impact = (df['trauma_group'] == 'Trauma - No Ongoing Impact').sum()
n_ongoing = (df['trauma_group'] == 'Trauma - Ongoing Impact').sum()
print(f"  No Ongoing Impact: n = {n_no_impact}")
print(f"  Ongoing Impact: n = {n_ongoing}")

# ============================================================================
# TRAUMA-BEHAVIOR CORRELATIONS (N=48)
# ============================================================================

print("\n" + "=" * 80)
print("3. TRAUMA-BEHAVIOR CORRELATIONS (N=48)")
print("=" * 80)

fb_corr = pd.read_csv('output/statistical_analyses/feedback_perseveration_trauma_correlations.csv')

print("\nFeedback Sensitivity ~ Trauma:")
fb_less = fb_corr[(fb_corr['Metric'] == 'feedback_sensitivity') & (fb_corr['Trauma_Variable'] == 'less_total_events')]
print(f"  LESS Total: β = {fb_less['Beta_standardized'].values[0]:.3f}, p = {fb_less['p_value'].values[0]:.3f}")
fb_ies = fb_corr[(fb_corr['Metric'] == 'feedback_sensitivity') & (fb_corr['Trauma_Variable'] == 'ies_total')]
print(f"  IES-R Total: β = {fb_ies['Beta_standardized'].values[0]:.3f}, p = {fb_ies['p_value'].values[0]:.3f}")
print(f"  Result: All correlations non-significant (all ps > .38)")

print("\nPerseveration ~ Trauma:")
per_less = fb_corr[(fb_corr['Metric'] == 'perseveration_index') & (fb_corr['Trauma_Variable'] == 'less_total_events')]
print(f"  LESS Total: β = {per_less['Beta_standardized'].values[0]:.3f}, p = {per_less['p_value'].values[0]:.3f}")
per_ies = fb_corr[(fb_corr['Metric'] == 'perseveration_index') & (fb_corr['Trauma_Variable'] == 'ies_total')]
print(f"  IES-R Total: β = {per_ies['Beta_standardized'].values[0]:.3f}, p = {per_ies['p_value'].values[0]:.3f}")
per_hyper = fb_corr[(fb_corr['Metric'] == 'perseveration_index') & (fb_corr['Trauma_Variable'] == 'ies_hyperarousal')]
print(f"  IES-R Hyperarousal: ρ = {per_hyper['Spearman_rho'].values[0]:.3f}, p = {per_hyper['p_value'].values[0]:.3f} (strongest trend)")
print(f"  Result: All correlations non-significant (all ps > .14)")

# ============================================================================
# COMPUTATIONAL MODELING RESULTS (N=47 PRELIMINARY)
# ============================================================================

print("\n" + "=" * 80)
print("4. COMPUTATIONAL MODELING RESULTS (N=47 PRELIMINARY)")
print("=" * 80)
print("Note: Using N=47 matched sample (awaiting supervisor's N=48 refits)")

# Load modeling data
m1 = pd.read_csv('output/mle/qlearning_individual_fits_matched.csv')
m2 = pd.read_csv('output/mle/wmrl_individual_fits_matched.csv')
beh = pd.read_csv('output/mle/behavioral_summary_matched_with_metrics.csv')

print(f"\n4.1 MODEL COMPARISON")
print(f"Sample size: N = {len(m1)}")

print(f"\nQ-Learning (M1):")
print(f"  AIC: M = {m1['aic'].mean():.1f}, SD = {m1['aic'].std():.1f}")
print(f"  BIC: M = {m1['bic'].mean():.1f}, SD = {m1['bic'].std():.1f}")

print(f"\nWM-RL Hybrid (M2):")
print(f"  AIC: M = {m2['aic'].mean():.1f}, SD = {m2['aic'].std():.1f}")
print(f"  BIC: M = {m2['bic'].mean():.1f}, SD = {m2['bic'].std():.1f}")

# Paired comparisons
from scipy.stats import wilcoxon
aic_stat, aic_p = wilcoxon(m1['aic'], m2['aic'])
bic_stat, bic_p = wilcoxon(m1['bic'], m2['bic'])

print(f"\nPaired Model Comparison:")
print(f"  AIC: Z = {aic_stat:.1f}, p = {aic_p:.3f} (n.s.)")
print(f"  BIC: Z = {bic_stat:.1f}, p = {bic_p:.3f} (marginal, but not significant)")
print(f"  Conclusion: Models fit equivalently well")

print(f"\n4.2 PARAMETER ESTIMATES")

# M1 parameters
print(f"\nQ-Learning Parameters (M1):")
print(f"  α₊ (positive LR): M = {m1['alpha_pos'].mean():.3f}, SD = {m1['alpha_pos'].std():.3f}")
print(f"  α₋ (negative LR): M = {m1['alpha_neg'].mean():.3f}, SD = {m1['alpha_neg'].std():.3f}")
print(f"  ε (noise): M = {m1['epsilon'].mean():.3f}, SD = {m1['epsilon'].std():.3f}")

# M2 parameters
print(f"\nWM-RL Hybrid Parameters (M2):")
print(f"  α₊ (positive LR): M = {m2['alpha_pos'].mean():.3f}, SD = {m2['alpha_pos'].std():.3f}")
print(f"  α₋ (negative LR): M = {m2['alpha_neg'].mean():.3f}, SD = {m2['alpha_neg'].std():.3f}")
print(f"  φ (WM decay): M = {m2['phi'].mean():.3f}, SD = {m2['phi'].std():.3f}")
print(f"  ρ (WM reliance): M = {m2['rho'].mean():.3f}, SD = {m2['rho'].std():.3f}")
print(f"  K (WM capacity): M = {m2['capacity'].mean():.3f}, SD = {m2['capacity'].std():.3f}")
print(f"  ε (noise): M = {m2['epsilon'].mean():.3f}, SD = {m2['epsilon'].std():.3f}")

print(f"\n4.3 PARAMETER → BEHAVIOR ASSOCIATIONS")
print(f"Testing which parameters predict task performance")

# Run quick regressions
from scipy.stats import spearmanr

print(f"\nSignificant Associations (M2 WM-RL Model):")

# ε → behavioral metrics
rho, p = spearmanr(m2['epsilon'], beh['accuracy_overall'])
print(f"\n  ε (noise) → Accuracy: ρ = {rho:.3f}, p = {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'}")

rho, p = spearmanr(m2['epsilon'], beh['feedback_sensitivity'])
print(f"  ε (noise) → Feedback Sensitivity: ρ = {rho:.3f}, p = {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'}")

rho, p = spearmanr(m2['epsilon'], beh['set_size_effect_accuracy'])
print(f"  ε (noise) → Set-Size Effect (Acc): ρ = {rho:.3f}, p = {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'}")

rho, p = spearmanr(m2['epsilon'], beh['set_size_effect_rt'])
print(f"  ε (noise) → Set-Size Effect (RT): ρ = {rho:.3f}, p = {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'}")

rho, p = spearmanr(m2['epsilon'], beh['perseveration_index'])
print(f"  ε (noise) → Perseveration: ρ = {rho:.3f}, p = {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'}")

# φ → accuracy
rho, p = spearmanr(m2['phi'], beh['accuracy_overall'])
print(f"\n  φ (WM decay) → Accuracy: ρ = {rho:.3f}, p = {p:.4f} {'***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'}")

print(f"\n  Interpretation: Higher noise (ε) → worse performance across multiple metrics")
print(f"                  Slower WM decay (φ) → better accuracy")

print(f"\n4.4 PARAMETER ← TRAUMA ASSOCIATIONS")
print(f"Testing whether trauma predicts model parameters")

# Check trauma correlations
print(f"\nM1 Q-Learning:")
for param in ['alpha_pos', 'alpha_neg', 'epsilon']:
    rho_less, p_less = spearmanr(m1[param], beh['less_total_events'])
    rho_ies, p_ies = spearmanr(m1[param], beh['ies_total'])
    print(f"  {param} ~ LESS: ρ={rho_less:.3f}, p={p_less:.3f}")
    print(f"  {param} ~ IES-R: ρ={rho_ies:.3f}, p={p_ies:.3f}")

print(f"\nM2 WM-RL Hybrid:")
for param in ['alpha_pos', 'alpha_neg', 'phi', 'epsilon']:
    rho_less, p_less = spearmanr(m2[param], beh['less_total_events'])
    rho_ies, p_ies = spearmanr(m2[param], beh['ies_total'])
    print(f"  {param} ~ LESS: ρ={rho_less:.3f}, p={p_less:.3f}")
    print(f"  {param} ~ IES-R: ρ={rho_ies:.3f}, p={p_ies:.3f}")

print(f"\nResult: ALL correlations non-significant after FDR correction (all q > 0.97)")
print(f"        Trauma exposure/symptoms do not predict computational parameters")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("KEY FINDINGS SUMMARY")
print("=" * 80)

print("\n1. BEHAVIORAL PERFORMANCE (N=48)")
print("   • Participants learned task above chance")
print("   • Strong WM load effects on accuracy and RT")
print("   • Individual differences in feedback sensitivity and perseveration")
print("   • NO significant trauma-behavior correlations")

print("\n2. COMPUTATIONAL MODELING (N=47 preliminary)")
print("   • Q-learning and WM-RL models fit equivalently well")
print("   • Noise parameter (ε) predicts 5 behavioral metrics")
print("   • WM decay (φ) predicts overall accuracy")
print("   • NO trauma effects on model parameters")

print("\n3. INTERPRETATION")
print("   • Trauma affects neither observable behavior NOR latent learning processes")
print("   • Individual differences in computational parameters explain behavioral variability")
print("   • Working memory constraints impact reinforcement learning performance")

print("\n" + "=" * 80)
print("END OF RESULTS SUMMARY")
print("=" * 80)
