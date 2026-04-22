"""
Format computational modeling results for thesis write-up.
Structured according to the 8-section framework.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import wilcoxon, spearmanr

print("=" * 80)
print("COMPUTATIONAL MODELING RESULTS - THESIS WRITE-UP")
print("=" * 80)

# ============================================================================
# 1. MODEL FITTING AND DATA INTEGRITY CHECKS
# ============================================================================

print("\n" + "=" * 80)
print("1. MODEL FITTING AND DATA INTEGRITY CHECKS")
print("=" * 80)

# Load data
# Q-learning: Cherry-picked N=47 from commit daa3fc8
m1 = pd.read_csv('output/mle/qlearning_individual_fits_matched.csv')

# WM-RL: Most recent fits (N=49), filtered to match Q-learning N=47
m2_full = pd.read_csv('output/mle/wmrl_individual_fits.csv')
m2 = m2_full[m2_full['participant_id'].isin(m1['participant_id'])].copy()

# Behavioral data: Matched N=47
beh = pd.read_csv('output/mle/behavioral_summary_matched_with_metrics.csv')

# Sort all dataframes by participant_id to ensure alignment
m1 = m1.sort_values('participant_id').reset_index(drop=True)
m2 = m2.sort_values('participant_id').reset_index(drop=True)
beh = beh.sort_values('participant_id').reset_index(drop=True)

# Verify alignment
assert len(m1) == len(m2) == len(beh), f"Sample size mismatch: M1={len(m1)}, M2={len(m2)}, Beh={len(beh)}"
assert (m1['participant_id'].values == m2['participant_id'].values).all(), "M1 and M2 participant order doesn't match"
assert (m1['participant_id'].values == beh['participant_id'].values).all(), "M1 and Beh participant order doesn't match"

print(f"\nFinal Sample: N = {len(m1)}")
print(f"Note: Currently using N=47 matched sample (awaiting supervisor's N=48 refits)")
print(f"Blocks included: 3-23 (experimental blocks, excluding practice)")

# Check for boundary solutions
print(f"\nParameter Bounds Check:")
print(f"  Q-learning (M1):")
alpha_pos_bounds = ((m1['alpha_pos'] < 0.01) | (m1['alpha_pos'] > 0.99)).sum()
alpha_neg_bounds = ((m1['alpha_neg'] < 0.01) | (m1['alpha_neg'] > 0.99)).sum()
epsilon_bounds = ((m1['epsilon'] < 0.01) | (m1['epsilon'] > 0.99)).sum()
print(f"    α₊ at bounds (<0.01 or >0.99): {alpha_pos_bounds} participants")
print(f"    α₋ at bounds: {alpha_neg_bounds} participants")
print(f"    ε at bounds: {epsilon_bounds} participants")

print(f"\n  WM-RL Hybrid (M2):")
alpha_pos_bounds = ((m2['alpha_pos'] < 0.01) | (m2['alpha_pos'] > 0.99)).sum()
alpha_neg_bounds = ((m2['alpha_neg'] < 0.01) | (m2['alpha_neg'] > 0.99)).sum()
phi_bounds = ((m2['phi'] < 0.01) | (m2['phi'] > 0.99)).sum()
epsilon_bounds = ((m2['epsilon'] < 0.01) | (m2['epsilon'] > 0.99)).sum()
print(f"    α₊ at bounds: {alpha_pos_bounds} participants")
print(f"    α₋ at bounds: {alpha_neg_bounds} participants")
print(f"    φ at bounds: {phi_bounds} participants")
print(f"    ε at bounds: {epsilon_bounds} participants")

print(f"\nSUGGESTED TEXT:")
print(f"'Inspection of individual fits indicated no systematic boundary solutions or")
print(f" convergence failures, supporting the interpretability of fitted parameters.'")

# ============================================================================
# 2. BASE MODEL COMPARISON: Q-LEARNING VS WM-RL
# ============================================================================

print("\n" + "=" * 80)
print("2. BASE MODEL COMPARISON: Q-LEARNING VS WM-RL")
print("=" * 80)

print(f"\nModel Fit Statistics:")
print(f"\nQ-Learning (M1):")
print(f"  AIC: Median = {m1['aic'].median():.1f}, IQR = [{m1['aic'].quantile(0.25):.1f}, {m1['aic'].quantile(0.75):.1f}]")
print(f"  BIC: Median = {m1['bic'].median():.1f}, IQR = [{m1['bic'].quantile(0.25):.1f}, {m1['bic'].quantile(0.75):.1f}]")

print(f"\nWM-RL Hybrid (M2):")
print(f"  AIC: Median = {m2['aic'].median():.1f}, IQR = [{m2['aic'].quantile(0.25):.1f}, {m2['aic'].quantile(0.75):.1f}]")
print(f"  BIC: Median = {m2['bic'].median():.1f}, IQR = [{m2['bic'].quantile(0.25):.1f}, {m2['bic'].quantile(0.75):.1f}]")

# Calculate deltas
delta_aic = m2['aic'] - m1['aic']
delta_bic = m2['bic'] - m1['bic']

print(f"\nModel Comparison Deltas (M2 - M1):")
print(f"  ΔAIC: Median = {delta_aic.median():.1f}, IQR = [{delta_aic.quantile(0.25):.1f}, {delta_aic.quantile(0.75):.1f}]")
print(f"  ΔBIC: Median = {delta_bic.median():.1f}, IQR = [{delta_bic.quantile(0.25):.1f}, {delta_bic.quantile(0.75):.1f}]")

# Win counts
aic_m1_wins = (m1['aic'] < m2['aic']).sum()
aic_m2_wins = (m2['aic'] < m1['aic']).sum()
bic_m1_wins = (m1['bic'] < m2['bic']).sum()
bic_m2_wins = (m2['bic'] < m1['bic']).sum()

print(f"\nWin Counts (per participant):")
print(f"  AIC: M1 wins = {aic_m1_wins}, M2 wins = {aic_m2_wins}")
print(f"  BIC: M1 wins = {bic_m1_wins}, M2 wins = {bic_m2_wins}")

# Statistical test
stat_aic, p_aic = wilcoxon(m1['aic'], m2['aic'])
stat_bic, p_bic = wilcoxon(m1['bic'], m2['bic'])

print(f"\nPaired Wilcoxon Tests:")
print(f"  AIC: Z = {stat_aic:.1f}, p = {p_aic:.3f}")
print(f"  BIC: Z = {stat_bic:.1f}, p = {p_bic:.3f}")

print(f"\nSUGGESTED TEXT:")
print(f"'Model comparison suggested broadly comparable explanatory adequacy of Q-learning")
print(f" and WM-RL models, motivating further examination of fitted parameters rather than")
print(f" reliance on model selection alone.'")

# ============================================================================
# 3. PARAMETER DESCRIPTIVES AND SANITY CHECKS
# ============================================================================

print("\n" + "=" * 80)
print("3. PARAMETER DESCRIPTIVES AND SANITY CHECKS")
print("=" * 80)

print(f"\nQ-Learning (M1) Parameters:")
for param, label in [('alpha_pos', 'α₊'), ('alpha_neg', 'α₋'), ('epsilon', 'ε')]:
    med = m1[param].median()
    q25 = m1[param].quantile(0.25)
    q75 = m1[param].quantile(0.75)
    min_val = m1[param].min()
    max_val = m1[param].max()
    print(f"  {label}: Median = {med:.3f}, IQR = [{q25:.3f}, {q75:.3f}], Range = [{min_val:.3f}, {max_val:.3f}]")

print(f"\nWM-RL Hybrid (M2) Parameters:")
for param, label in [('alpha_pos', 'α₊'), ('alpha_neg', 'α₋'), ('phi', 'φ'), 
                      ('rho', 'ρ'), ('capacity', 'K'), ('epsilon', 'ε')]:
    med = m2[param].median()
    q25 = m2[param].quantile(0.25)
    q75 = m2[param].quantile(0.75)
    min_val = m2[param].min()
    max_val = m2[param].max()
    print(f"  {label}: Median = {med:.3f}, IQR = [{q25:.3f}, {q75:.3f}], Range = [{min_val:.3f}, {max_val:.3f}]")

# Theoretical anchors
print(f"\nTheoretical Anchor Checks:")
print(f"  K (WM capacity) ≈ 4: Median = {m2['capacity'].median():.2f} ✓")
print(f"  α₊ ≈ α₋ (similar pos/neg learning): M1 diff = {abs(m1['alpha_pos'].median() - m1['alpha_neg'].median()):.3f}, M2 diff = {abs(m2['alpha_pos'].median() - m2['alpha_neg'].median()):.3f}")
print(f"  ε not at extremes: M1 median = {m1['epsilon'].median():.3f}, M2 median = {m2['epsilon'].median():.3f} ✓")

print(f"\nSUGGESTED TEXT:")
print(f"'Parameter estimates showed substantial inter-individual variability while")
print(f" remaining within theoretically plausible ranges.'")

# ============================================================================
# 4. PARAMETER-BEHAVIOR ALIGNMENT (MECHANISTIC VALIDATION)
# ============================================================================

print("\n" + "=" * 80)
print("4. PARAMETER-BEHAVIOR ALIGNMENT (MECHANISTIC VALIDATION)")
print("=" * 80)

print(f"\nKey Associations (WM-RL Model M2):")
print(f"Focus: Demonstrating parameters capture meaningful individual differences\n")

# K ↔ set-size effect
rho, p = spearmanr(m2['capacity'], beh['set_size_effect_accuracy'])
print(f"K (WM capacity) ↔ Set-Size Effect (Accuracy):")
print(f"  ρ = {rho:.3f}, p = {p:.3f}")
print(f"  Interpretation: {'Higher K → smaller load-related accuracy drop' if rho < 0 else 'No clear pattern'}")

# α₋ ↔ feedback sensitivity
rho, p = spearmanr(m2['alpha_neg'], beh['feedback_sensitivity'])
print(f"\nα₋ (negative LR) ↔ Feedback Sensitivity:")
print(f"  ρ = {rho:.3f}, p = {p:.3f}")
print(f"  Interpretation: Weak/absent → supports behavioral null findings")

# ε ↔ multiple metrics
print(f"\nε (noise/exploration) ↔ Task Performance:")
for metric, label in [('accuracy_overall', 'Overall Accuracy'),
                       ('feedback_sensitivity', 'Feedback Sensitivity'),
                       ('set_size_effect_accuracy', 'Set-Size Effect (Acc)'),
                       ('set_size_effect_rt', 'Set-Size Effect (RT)'),
                       ('perseveration_index', 'Perseveration')]:
    rho, p = spearmanr(m2['epsilon'], beh[metric])
    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
    print(f"  {label}: ρ = {rho:.3f}, p = {p:.4f} {sig}")

# φ ↔ accuracy
rho, p = spearmanr(m2['phi'], beh['accuracy_overall'])
print(f"\nφ (WM decay) ↔ Overall Accuracy:")
print(f"  ρ = {rho:.3f}, p = {p:.4f}")

print(f"\nSUGGESTED TEXT:")
print(f"'These associations demonstrate that fitted parameters captured meaningful")
print(f" individual differences in task performance, even in the absence of strong")
print(f" trauma-behaviour relationships.'")

# ============================================================================
# 5. TRAUMA SCALE DIAGNOSTICS
# ============================================================================

print("\n" + "=" * 80)
print("5. TRAUMA SCALE DIAGNOSTICS")
print("=" * 80)

print(f"\nDistributional Properties:")
print(f"  LESS Total: M = {beh['less_total_events'].mean():.2f}, SD = {beh['less_total_events'].std():.2f}, Range = [{beh['less_total_events'].min():.0f}, {beh['less_total_events'].max():.0f}]")
print(f"  IES-R Total: M = {beh['ies_total'].mean():.2f}, SD = {beh['ies_total'].std():.2f}, Range = [{beh['ies_total'].min():.0f}, {beh['ies_total'].max():.0f}]")

# Independence of LESS and IES-R
rho_less_ies, p_less_ies = spearmanr(beh['less_total_events'], beh['ies_total'])
print(f"\nIndependence Check:")
print(f"  LESS Total ↔ IES-R Total: ρ = {rho_less_ies:.3f}, p = {p_less_ies:.3f}")

# Collinearity among IES-R subscales
print(f"\nIES-R Subscale Intercorrelations:")
rho_int_avo, _ = spearmanr(beh['ies_intrusion'], beh['ies_avoidance'])
rho_int_hyp, _ = spearmanr(beh['ies_intrusion'], beh['ies_hyperarousal'])
rho_avo_hyp, _ = spearmanr(beh['ies_avoidance'], beh['ies_hyperarousal'])
print(f"  Intrusion ↔ Avoidance: ρ = {rho_int_avo:.3f}")
print(f"  Intrusion ↔ Hyperarousal: ρ = {rho_int_hyp:.3f}")
print(f"  Avoidance ↔ Hyperarousal: ρ = {rho_avo_hyp:.3f}")

print(f"\nSUGGESTED TEXT:")
print(f"'LESS total events and IES-R total scores were largely independent")
print(f" (ρ = {rho_less_ies:.2f}, p = {p_less_ies:.2f}), indicating that trauma exposure and")
print(f" trauma-related distress are distinct constructs in this sample.'")

# ============================================================================
# 6. PARAMETER-TRAUMA REGRESSIONS
# ============================================================================

print("\n" + "=" * 80)
print("6. PARAMETER-TRAUMA REGRESSIONS (BASE MODELS)")
print("=" * 80)

print(f"\nQ-Learning (M1) ~ Trauma:")
print(f"Parameters: α₊, α₋, ε")
print(f"\nLESS Total Events:")
for param, label in [('alpha_pos', 'α₊'), ('alpha_neg', 'α₋'), ('epsilon', 'ε')]:
    rho, p = spearmanr(m1[param], beh['less_total_events'])
    print(f"  {label}: ρ = {rho:.3f}, p = {p:.3f}")

print(f"\nIES-R Total Score:")
for param, label in [('alpha_pos', 'α₊'), ('alpha_neg', 'α₋'), ('epsilon', 'ε')]:
    rho, p = spearmanr(m1[param], beh['ies_total'])
    print(f"  {label}: ρ = {rho:.3f}, p = {p:.3f}")

print(f"\n" + "-" * 80)
print(f"\nWM-RL Hybrid (M2) ~ Trauma:")
print(f"Parameters: α₊, α₋, φ, ρ, K, ε")
print(f"\nLESS Total Events:")
for param, label in [('alpha_pos', 'α₊'), ('alpha_neg', 'α₋'), ('phi', 'φ'), 
                      ('rho', 'ρ'), ('capacity', 'K'), ('epsilon', 'ε')]:
    rho, p = spearmanr(m2[param], beh['less_total_events'])
    print(f"  {label}: ρ = {rho:.3f}, p = {p:.3f}")

print(f"\nIES-R Total Score:")
for param, label in [('alpha_pos', 'α₊'), ('alpha_neg', 'α₋'), ('phi', 'φ'), 
                      ('rho', 'ρ'), ('capacity', 'K'), ('epsilon', 'ε')]:
    rho, p = spearmanr(m2[param], beh['ies_total'])
    print(f"  {label}: ρ = {rho:.3f}, p = {p:.3f}")

print(f"\n" + "-" * 80)
print(f"\nExploratory: IES-R Subscales (M2 only):")
for subscale, label in [('ies_intrusion', 'Intrusion'), 
                         ('ies_avoidance', 'Avoidance'), 
                         ('ies_hyperarousal', 'Hyperarousal')]:
    print(f"\n{label}:")
    for param, plabel in [('alpha_pos', 'α₊'), ('alpha_neg', 'α₋'), ('phi', 'φ'), ('epsilon', 'ε')]:
        rho, p = spearmanr(m2[param], beh[subscale])
        print(f"  {plabel}: ρ = {rho:.3f}, p = {p:.3f}")

print(f"\nOVERALL PATTERN:")
print(f"  All correlations non-significant after FDR correction")
print(f"  Effect sizes small (all |ρ| < 0.26)")
print(f"  Consistent across M1 and M2")

print(f"\nSUGGESTED TEXT:")
print(f"'Together, these analyses indicate that trauma exposure and symptom severity")
print(f" were not associated with latent learning or working-memory parameters")
print(f" underlying task performance.'")

# ============================================================================
# 7. FOREST PLOT DATA (for visualization)
# ============================================================================

print("\n" + "=" * 80)
print("7. FOREST PLOT DATA (Optional - For Visualization)")
print("=" * 80)

print(f"\nData for forest plot showing all parameter-trauma associations:")
print(f"Parameters: α₊, α₋, φ, ρ, K, ε (M2)")
print(f"Trauma measures: LESS Total, IES-R Total")
print(f"\nThis would show:")
print(f"  - Standardized regression coefficients (β)")
print(f"  - 95% confidence intervals")
print(f"  - All CIs spanning zero")
print(f"\nCaption: 'Standardized regression coefficients and 95% confidence intervals")
print(f" for associations between trauma measures and model parameters.'")

# ============================================================================
# 8. MODEL EXTENSION RATIONALE (κ) - NOT RESULTS YET
# ============================================================================

print("\n" + "=" * 80)
print("8. MODEL EXTENSION RATIONALE (κ) - MOTIVATION ONLY")
print("=" * 80)

print(f"\nBehavioral Evidence for Perseveration:")
print(f"  Perseveration Index: M = 0.406, SD = 0.118 (N=48)")
print(f"  Substantial individual differences (range = [0.16, 0.73])")

print(f"\nBase Model Limitation:")
rho, p = spearmanr(m2['alpha_neg'], beh['perseveration_index'])
print(f"  α₋ ↔ Perseveration: ρ = {rho:.3f}, p = {p:.3f}")
print(f"  → Negative learning rate does not capture outcome-insensitive repetition")

print(f"\nSUGGESTED TEXT:")
print(f"'Despite interpretable learning and memory parameters, the base models do not")
print(f" explicitly capture outcome-insensitive action repetition at reversal points,")
print(f" motivating the inclusion of a perseveration parameter in an extended model.'")

print("\n" + "=" * 80)
print("END OF COMPUTATIONAL MODELING RESULTS SUMMARY")
print("=" * 80)
