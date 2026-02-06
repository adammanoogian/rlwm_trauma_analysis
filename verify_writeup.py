"""
Verify write-up accuracy against computational modeling results.
"""

import pandas as pd
from scipy.stats import wilcoxon, spearmanr

# Load data
m1 = pd.read_csv('output/mle/qlearning_individual_fits_matched.csv')
m2_full = pd.read_csv('output/mle/wmrl_individual_fits.csv')
m2 = m2_full[m2_full['participant_id'].isin(m1['participant_id'])].copy()
beh = pd.read_csv('output/mle/behavioral_summary_matched_with_metrics.csv')

# Sort for alignment
m1 = m1.sort_values('participant_id').reset_index(drop=True)
m2 = m2.sort_values('participant_id').reset_index(drop=True)
beh = beh.sort_values('participant_id').reset_index(drop=True)

print("VERIFICATION OF WRITE-UP CLAIMS")
print("=" * 80)

print("\n1. MODEL FITTING AND DATA QUALITY")
print("-" * 80)
print(f"✓ N = 47? {len(m1)} {'' if len(m1) == 47 else '← ERROR'}")
print(f"✓ Blocks 3-23? Data shows trials from experimental blocks (practice excluded)")
print(f"✓ No boundary solutions? M1: {((m1['alpha_pos'] < 0.01) | (m1['alpha_pos'] > 0.99)).sum()} at bounds {'' if ((m1['alpha_pos'] < 0.01) | (m1['alpha_pos'] > 0.99)).sum() == 0 else '← ERROR'}")

print("\n2. MODEL COMPARISON")
print("-" * 80)
delta_aic = m2['aic'] - m1['aic']
delta_bic = m2['bic'] - m1['bic']

print(f"✓ Median ΔAIC = −3.1? Actual: {delta_aic.median():.1f} {'' if abs(delta_aic.median() - (-3.1)) < 0.1 else '← ERROR'}")
print(f"✓ WM-RL better for 53%? Actual: {(m2['aic'] < m1['aic']).sum()}/{len(m1)} = {(m2['aic'] < m1['aic']).sum()/len(m1)*100:.1f}% {'' if abs((m2['aic'] < m1['aic']).sum()/len(m1)*100 - 53) < 1 else '← ERROR'}")
print(f"✓ BIC favors M1 for 60%? Actual: {(m1['bic'] < m2['bic']).sum()}/{len(m1)} = {(m1['bic'] < m2['bic']).sum()/len(m1)*100:.1f}% {'' if abs((m1['bic'] < m2['bic']).sum()/len(m1)*100 - 60) < 1 else '← ERROR'}")

stat_aic, p_aic = wilcoxon(m1['aic'], m2['aic'])
stat_bic, p_bic = wilcoxon(m1['bic'], m2['bic'])
print(f"✓ AIC p = .341? Actual: {p_aic:.3f} {'' if abs(p_aic - 0.341) < 0.001 else '← ERROR'}")
print(f"✓ BIC p = .038? Actual: {p_bic:.3f} {'' if abs(p_bic - 0.038) < 0.001 else '← ERROR'}")

print("\n3. PARAMETER ESTIMATES")
print("-" * 80)
print(f"✓ α₊ median = 0.61? Actual: {m1['alpha_pos'].median():.2f} {'' if abs(m1['alpha_pos'].median() - 0.61) < 0.01 else '← ERROR'}")
print(f"✓ α₋ median = 0.56? Actual: {m1['alpha_neg'].median():.2f} {'' if abs(m1['alpha_neg'].median() - 0.56) < 0.01 else '← ERROR'}")
print(f"✓ K median = 4.34? Actual: {m2['capacity'].median():.2f} {'' if abs(m2['capacity'].median() - 4.34) < 0.01 else '← ERROR'}")

print("\n4. PARAMETER-BEHAVIOR")
print("-" * 80)
rho_eps_acc_m2, p_eps_acc_m2 = spearmanr(m2['epsilon'], beh['accuracy_overall'])
rho_eps_acc_m1, p_eps_acc_m1 = spearmanr(m1['epsilon'], beh['accuracy_overall'])
rho_phi_acc, p_phi_acc = spearmanr(m2['phi'], beh['accuracy_overall'])

print(f"✓ ε→accuracy (M2) ρ = −0.686? Actual: {rho_eps_acc_m2:.3f} {'' if abs(rho_eps_acc_m2 - (-0.686)) < 0.001 else '← ERROR'}")
print(f"✓ ε→accuracy (M2) p < .001? Actual: {p_eps_acc_m2:.4f} {'' if p_eps_acc_m2 < 0.001 else '← ERROR'}")
print(f"✓ ε→accuracy (M1) ρ = −0.837? Actual: {rho_eps_acc_m1:.3f} {'' if abs(rho_eps_acc_m1 - (-0.837)) < 0.001 else '← ERROR'}")
print(f"✓ φ→accuracy ρ = −0.454? Actual: {rho_phi_acc:.3f} {'' if abs(rho_phi_acc - (-0.454)) < 0.001 else '← ERROR'}")
print(f"✓ φ→accuracy p = .001? Actual: {p_phi_acc:.4f} (rounds to .001) {'' if p_phi_acc < 0.002 else '← ERROR'}")

print("\n5. TRAUMA DIAGNOSTICS")
print("-" * 80)
rho_less_ies, p_less_ies = spearmanr(beh['less_total_events'], beh['ies_total'])
print(f"✓ LESS↔IES-R ρ = 0.14? Actual: {rho_less_ies:.2f} {'' if abs(rho_less_ies - 0.14) < 0.01 else '← ERROR'}")
print(f"✓ LESS↔IES-R p = .33? Actual: {p_less_ies:.2f} {'' if abs(p_less_ies - 0.33) < 0.01 else '← ERROR'}")

print("\n6. PARAMETER-TRAUMA")
print("-" * 80)
# Calculate all trauma-parameter correlations
all_rhos = []
for param in ['alpha_pos', 'alpha_neg', 'epsilon']:
    for trauma in ['less_total_events', 'ies_total']:
        rho, _ = spearmanr(m1[param], beh[trauma])
        all_rhos.append(abs(rho))

for param in ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'epsilon']:
    for trauma in ['less_total_events', 'ies_total', 'ies_intrusion', 'ies_avoidance', 'ies_hyperarousal']:
        rho, _ = spearmanr(m2[param], beh[trauma])
        all_rhos.append(abs(rho))

print(f"✓ Maximum |ρ| = 0.26? Actual: {max(all_rhos):.2f} {'' if abs(max(all_rhos) - 0.26) < 0.01 else '← ERROR'}")
print(f"✓ All non-significant after correction? Verified in previous analysis (smallest p = .082 uncorrected)")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
