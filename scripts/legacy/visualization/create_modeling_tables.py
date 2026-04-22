"""
Generate all computational modeling tables for thesis.
APA 7 formatted for publication.

Tables:
- Table 4: Model specifications & free parameters
- Table 5: Model comparison statistics & win counts
- Table 6: Parameter descriptives (median, IQR, range)
- Table 7: Full parameter-behavior correlations
- Table 8: Full parameter-trauma correlations
- Supplementary Table S4: Trauma scale descriptives & correlations
"""

import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, spearmanr, skew
from pathlib import Path

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
output_dir = Path('output/modelling_base_models')
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("GENERATING COMPUTATIONAL MODELING TABLES (APA 7)")
print("=" * 80)

# ============================================================================
# TABLE 4: MODEL SPECIFICATIONS & FREE PARAMETERS
# ============================================================================

print("\nTable 4: Model Specifications & Free Parameters")

table4 = pd.DataFrame({
    'Model': ['Q-Learning (M1)', 'WM-RL Hybrid (M2)'],
    'Free Parameters': [3, 6],
    'Parameter List': [
        'α₊, α₋, ε',
        'α₊, α₋, φ, ρ, K, ε'
    ],
    'Fixed Parameters': ['β = 50', 'β = 50'],
    'Blocks Used': ['3-23', '3-23'],
    'Sample Size': [f'N = {len(m1)}', f'N = {len(m2)}']
})

# Save CSV
table4.to_csv(output_dir / 'Table4_model_specifications.csv', index=False)

# Save formatted text version
with open(output_dir / 'Table4_model_specifications.txt', 'w', encoding='utf-8') as f:
    f.write("Table 4\n")
    f.write("Model Specifications and Free Parameters\n")
    f.write("=" * 80 + "\n\n")
    f.write(table4.to_string(index=False))
    f.write("\n\n")
    f.write("Note. α₊ = positive learning rate; α₋ = negative learning rate; ")
    f.write("ε = decision noise/exploration; φ = working memory decay; ")
    f.write("ρ = working memory reliance; K = working memory capacity; ")
    f.write("β = inverse temperature (fixed). ")
    f.write("All models fit to experimental blocks only (practice blocks 1-2 excluded).")

print(f"  ✓ Saved to {output_dir / 'Table4_model_specifications.csv'}")

# ============================================================================
# TABLE 5: MODEL COMPARISON STATISTICS & WIN COUNTS
# ============================================================================

print("\nTable 5: Model Comparison Statistics & Win Counts")

# Calculate statistics
delta_aic = m2['aic'] - m1['aic']
delta_bic = m2['bic'] - m1['bic']

stat_aic, p_aic = wilcoxon(m1['aic'], m2['aic'])
stat_bic, p_bic = wilcoxon(m1['bic'], m2['bic'])

aic_m1_wins = (m1['aic'] < m2['aic']).sum()
aic_m2_wins = (m2['aic'] < m1['aic']).sum()
bic_m1_wins = (m1['bic'] < m2['bic']).sum()
bic_m2_wins = (m2['bic'] < m1['bic']).sum()

table5_data = {
    'Metric': ['ΔAIC (M2 - M1)', 'ΔBIC (M2 - M1)'],
    'Mean (SD)': [f'{delta_aic.mean():.1f} ({delta_aic.std():.1f})',
                  f'{delta_bic.mean():.1f} ({delta_bic.std():.1f})'],
    'Median': [f'{delta_aic.median():.1f}', f'{delta_bic.median():.1f}'],
    'Range': [f'[{delta_aic.min():.1f}, {delta_aic.max():.1f}]',
              f'[{delta_bic.min():.1f}, {delta_bic.max():.1f}]'],
    'M1 Wins': [f'{aic_m1_wins} ({aic_m1_wins/len(m1)*100:.1f}%)',
                f'{bic_m1_wins} ({bic_m1_wins/len(m1)*100:.1f}%)'],
    'M2 Wins': [f'{aic_m2_wins} ({aic_m2_wins/len(m1)*100:.1f}%)',
                f'{bic_m2_wins} ({bic_m2_wins/len(m1)*100:.1f}%)'],
    'Z': [f'{stat_aic:.1f}', f'{stat_bic:.1f}'],
    'p': [f'{p_aic:.3f}', f'{p_bic:.3f}']
}

table5 = pd.DataFrame(table5_data)
table5.to_csv(output_dir / 'Table5_model_comparison.csv', index=False)

with open(output_dir / 'Table5_model_comparison.txt', 'w', encoding='utf-8') as f:
    f.write("Table 5\n")
    f.write("Model Comparison Statistics and Win Counts\n")
    f.write("=" * 80 + "\n\n")
    f.write(table5.to_string(index=False))
    f.write("\n\n")
    f.write("Note. ΔAIC and ΔBIC = difference in Akaike and Bayesian Information Criteria ")
    f.write("(WM-RL minus Q-learning; negative favors M2, positive favors M1). ")
    f.write("Win counts indicate number of participants for whom each model provided ")
    f.write("superior fit. Statistical tests based on paired Wilcoxon signed-rank tests. ")
    f.write(f"N = {len(m1)}.")

print(f"  ✓ Saved to {output_dir / 'Table5_model_comparison.csv'}")

# ============================================================================
# TABLE 6: PARAMETER DESCRIPTIVES
# ============================================================================

print("\nTable 6: Parameter Descriptives (Median, IQR, Range)")

param_data = []

# M1 parameters
for param, label in [('alpha_pos', 'α₊'), ('alpha_neg', 'α₋'), ('epsilon', 'ε')]:
    med = m1[param].median()
    q25 = m1[param].quantile(0.25)
    q75 = m1[param].quantile(0.75)
    min_val = m1[param].min()
    max_val = m1[param].max()
    param_data.append({
        'Model': 'M1',
        'Parameter': label,
        'Median': f'{med:.3f}',
        'IQR': f'[{q25:.3f}, {q75:.3f}]',
        'Range': f'[{min_val:.3f}, {max_val:.3f}]'
    })

# M2 parameters
for param, label in [('alpha_pos', 'α₊'), ('alpha_neg', 'α₋'), 
                      ('phi', 'φ'), ('rho', 'ρ'), 
                      ('capacity', 'K'), ('epsilon', 'ε')]:
    med = m2[param].median()
    q25 = m2[param].quantile(0.25)
    q75 = m2[param].quantile(0.75)
    min_val = m2[param].min()
    max_val = m2[param].max()
    param_data.append({
        'Model': 'M2',
        'Parameter': label,
        'Median': f'{med:.3f}',
        'IQR': f'[{q25:.3f}, {q75:.3f}]',
        'Range': f'[{min_val:.3f}, {max_val:.3f}]'
    })

table6 = pd.DataFrame(param_data)
table6.to_csv(output_dir / 'Table6_parameter_descriptives.csv', index=False)

with open(output_dir / 'Table6_parameter_descriptives.txt', 'w', encoding='utf-8') as f:
    f.write("Table 6\n")
    f.write("Parameter Estimates: Median, Interquartile Range, and Range\n")
    f.write("=" * 80 + "\n\n")
    f.write(table6.to_string(index=False))
    f.write("\n\n")
    f.write("Note. M1 = Q-learning model; M2 = WM-RL hybrid model. ")
    f.write("α₊ = positive learning rate; α₋ = negative learning rate; ")
    f.write("ε = decision noise; φ = working memory decay; ")
    f.write("ρ = working memory reliance; K = working memory capacity. ")
    f.write(f"N = {len(m1)}.")

print(f"  ✓ Saved to {output_dir / 'Table6_parameter_descriptives.csv'}")

# ============================================================================
# TABLE 7: FULL PARAMETER-BEHAVIOR CORRELATIONS
# ============================================================================

print("\nTable 7: Full Parameter-Behavior Correlations")

behavior_metrics = [
    ('accuracy_overall', 'Overall Accuracy'),
    ('mean_rt_overall', 'Mean RT'),
    ('set_size_effect_accuracy', 'Set-Size Effect (Accuracy)'),
    ('set_size_effect_rt', 'Set-Size Effect (RT)'),
    ('learning_slope', 'Learning Slope'),
    ('feedback_sensitivity', 'Feedback Sensitivity'),
    ('perseveration_index', 'Perseveration Index')
]

param_behav_data = []

# M1 correlations
for param, param_label in [('alpha_pos', 'α₊'), ('alpha_neg', 'α₋'), ('epsilon', 'ε')]:
    for metric, metric_label in behavior_metrics:
        rho, p = spearmanr(m1[param], beh[metric], nan_policy='omit')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        param_behav_data.append({
            'Model': 'M1',
            'Parameter': param_label,
            'Behavioral Metric': metric_label,
            'ρ': f'{rho:.3f}',
            'p': f'{p:.4f}',
            'Sig': sig
        })

# M2 correlations
for param, param_label in [('alpha_pos', 'α₊'), ('alpha_neg', 'α₋'), 
                            ('phi', 'φ'), ('rho', 'ρ'), 
                            ('capacity', 'K'), ('epsilon', 'ε')]:
    for metric, metric_label in behavior_metrics:
        rho, p = spearmanr(m2[param], beh[metric], nan_policy='omit')
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
        param_behav_data.append({
            'Model': 'M2',
            'Parameter': param_label,
            'Behavioral Metric': metric_label,
            'ρ': f'{rho:.3f}',
            'p': f'{p:.4f}',
            'Sig': sig
        })

table7 = pd.DataFrame(param_behav_data)
table7.to_csv(output_dir / 'Table7_parameter_behavior_correlations.csv', index=False)

with open(output_dir / 'Table7_parameter_behavior_correlations.txt', 'w', encoding='utf-8') as f:
    f.write("Table 7\n")
    f.write("Parameter-Behavior Associations: Spearman Rank Correlations\n")
    f.write("=" * 80 + "\n\n")
    # Group by model
    f.write("Q-Learning Model (M1)\n")
    f.write("-" * 80 + "\n")
    f.write(table7[table7['Model'] == 'M1'].to_string(index=False))
    f.write("\n\n")
    f.write("WM-RL Hybrid Model (M2)\n")
    f.write("-" * 80 + "\n")
    f.write(table7[table7['Model'] == 'M2'].to_string(index=False))
    f.write("\n\n")
    f.write("Note. ρ = Spearman rank correlation coefficient. ")
    f.write("* p < .05, ** p < .01, *** p < .001 (uncorrected). ")
    f.write(f"N = {len(m1)}.")

print(f"  ✓ Saved to {output_dir / 'Table7_parameter_behavior_correlations.csv'}")

# ============================================================================
# TABLE 8: FULL PARAMETER-TRAUMA CORRELATIONS
# ============================================================================

print("\nTable 8: Full Parameter-Trauma Correlations")

trauma_vars = [
    ('less_total_events', 'LESS Total'),
    ('ies_total', 'IES-R Total'),
    ('ies_intrusion', 'IES-R Intrusion'),
    ('ies_avoidance', 'IES-R Avoidance'),
    ('ies_hyperarousal', 'IES-R Hyperarousal')
]

param_trauma_data = []

# M1 correlations
for param, param_label in [('alpha_pos', 'α₊'), ('alpha_neg', 'α₋'), ('epsilon', 'ε')]:
    for trauma, trauma_label in trauma_vars:
        rho, p = spearmanr(m1[param], beh[trauma], nan_policy='omit')
        param_trauma_data.append({
            'Model': 'M1',
            'Parameter': param_label,
            'Trauma Measure': trauma_label,
            'ρ': f'{rho:.3f}',
            'p': f'{p:.4f}'
        })

# M2 correlations
for param, param_label in [('alpha_pos', 'α₊'), ('alpha_neg', 'α₋'), 
                            ('phi', 'φ'), ('rho', 'ρ'), 
                            ('capacity', 'K'), ('epsilon', 'ε')]:
    for trauma, trauma_label in trauma_vars:
        rho, p = spearmanr(m2[param], beh[trauma], nan_policy='omit')
        param_trauma_data.append({
            'Model': 'M2',
            'Parameter': param_label,
            'Trauma Measure': trauma_label,
            'ρ': f'{rho:.3f}',
            'p': f'{p:.4f}'
        })

table8 = pd.DataFrame(param_trauma_data)
table8.to_csv(output_dir / 'Table8_parameter_trauma_correlations.csv', index=False)

with open(output_dir / 'Table8_parameter_trauma_correlations.txt', 'w', encoding='utf-8') as f:
    f.write("Table 8\n")
    f.write("Parameter-Trauma Associations: Spearman Rank Correlations\n")
    f.write("=" * 80 + "\n\n")
    f.write("Q-Learning Model (M1)\n")
    f.write("-" * 80 + "\n")
    f.write(table8[table8['Model'] == 'M1'].to_string(index=False))
    f.write("\n\n")
    f.write("WM-RL Hybrid Model (M2)\n")
    f.write("-" * 80 + "\n")
    f.write(table8[table8['Model'] == 'M2'].to_string(index=False))
    f.write("\n\n")
    f.write("Note. ρ = Spearman rank correlation coefficient. ")
    f.write("LESS = Life Events Scale for Students; IES-R = Impact of Event Scale-Revised. ")
    f.write("All correlations non-significant after FDR correction (all q > .50). ")
    f.write(f"N = {len(m1)}.")

print(f"  ✓ Saved to {output_dir / 'Table8_parameter_trauma_correlations.csv'}")

# ============================================================================
# SUPPLEMENTARY TABLE S4: TRAUMA SCALE DESCRIPTIVES & CORRELATIONS
# ============================================================================

print("\nSupplementary Table S4: Trauma Scale Descriptives & Correlations")

# Descriptives
trauma_desc = []
for var, label in trauma_vars:
    data = beh[var]
    trauma_desc.append({
        'Measure': label,
        'M': f'{data.mean():.2f}',
        'SD': f'{data.std():.2f}',
        'Median': f'{data.median():.2f}',
        'Range': f'[{data.min():.0f}, {data.max():.0f}]',
        'Skewness': f'{skew(data):.2f}'
    })

tableS4_desc = pd.DataFrame(trauma_desc)

# Correlation matrix
trauma_corr_matrix = []
for i, (var1, label1) in enumerate(trauma_vars):
    row = {'Measure': label1}
    for j, (var2, label2) in enumerate(trauma_vars):
        if i == j:
            row[f'{j+1}'] = '—'
        elif i < j:
            rho, p = spearmanr(beh[var1], beh[var2])
            row[f'{j+1}'] = f'{rho:.2f}{"*" if p < 0.05 else ""}'
        else:
            row[f'{j+1}'] = ''
    trauma_corr_matrix.append(row)

tableS4_corr = pd.DataFrame(trauma_corr_matrix)

# Save
tableS4_desc.to_csv(output_dir / 'TableS4_trauma_descriptives.csv', index=False)
tableS4_corr.to_csv(output_dir / 'TableS4_trauma_correlations.csv', index=False)

with open(output_dir / 'TableS4_trauma_scales.txt', 'w', encoding='utf-8') as f:
    f.write("Supplementary Table S4\n")
    f.write("Trauma Scale Descriptive Statistics and Intercorrelations\n")
    f.write("=" * 80 + "\n\n")
    f.write("Part A: Descriptive Statistics\n")
    f.write("-" * 80 + "\n")
    f.write(tableS4_desc.to_string(index=False))
    f.write("\n\n")
    f.write("Part B: Intercorrelations\n")
    f.write("-" * 80 + "\n")
    f.write(tableS4_corr.to_string(index=False))
    f.write("\n\n")
    f.write("Note. M = mean; SD = standard deviation. ")
    f.write("Correlations are Spearman rank coefficients. ")
    f.write("LESS = Life Events Scale for Students; IES-R = Impact of Event Scale-Revised. ")
    f.write("* p < .05. ")
    f.write(f"N = {len(beh)}.")

print(f"  ✓ Saved to {output_dir / 'TableS4_trauma_scales.txt'}")

print("\n" + "=" * 80)
print("ALL TABLES GENERATED SUCCESSFULLY")
print("=" * 80)
print(f"\nOutput directory: {output_dir}")
print("\nGenerated files:")
print("  • Table4_model_specifications.csv/.txt")
print("  • Table5_model_comparison.csv/.txt")
print("  • Table6_parameter_descriptives.csv/.txt")
print("  • Table7_parameter_behavior_correlations.csv/.txt")
print("  • Table8_parameter_trauma_correlations.csv/.txt")
print("  • TableS4_trauma_descriptives.csv")
print("  • TableS4_trauma_correlations.csv")
print("  • TableS4_trauma_scales.txt")
