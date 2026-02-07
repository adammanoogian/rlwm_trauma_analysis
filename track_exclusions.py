"""
Track participant exclusions through the analysis pipeline.
"""
import pandas as pd
import numpy as np

print("="*80)
print("PARTICIPANT EXCLUSION TRACKING")
print("="*80)

# Load all datasets
demographics = pd.read_csv('output/demographics_complete.csv')
summary = pd.read_csv('output/summary_participant_metrics.csv')
stats_data = pd.read_csv('output/statistical_analyses/data_summary_with_groups.csv')
long_data = pd.read_csv('output/statistical_analyses/data_long_format.csv')

# Get unique participant sets
demo_ids = set(demographics['sona_id'])
summary_ids = set(summary['sona_id'])
stats_ids = set(stats_data['sona_id'])
anova_ids = set(long_data['sona_id'].unique())

print(f"\n{'Stage':<30} {'N':<10}")
print("-" * 40)
print(f"{'1. Demographics extracted':<30} {len(demo_ids):<10}")
print(f"{'2. Behavioral summary':<30} {len(summary_ids):<10}")
print(f"{'3. Statistical data file':<30} {len(stats_ids):<10}")
print(f"{'4. ANOVA analysis':<30} {len(anova_ids):<10}")

# Stage 1 → 2: Demographics to Behavioral Summary
excluded_stage1 = sorted(demo_ids - summary_ids)
if excluded_stage1:
    print(f"\n{'='*80}")
    print(f"EXCLUDED: Demographics → Behavioral Summary (N={len(excluded_stage1)})")
    print(f"{'='*80}")
    for pid in excluded_stage1:
        print(f"\n  Participant {pid}:")
        print(f"    Reason: No task data file found")

# Stage 2 → 3: Behavioral Summary to Statistical Data
excluded_stage2 = sorted(summary_ids - stats_ids)
if excluded_stage2:
    print(f"\n{'='*80}")
    print(f"EXCLUDED: Behavioral Summary → Statistical Data (N={len(excluded_stage2)})")
    print(f"{'='*80}")
    for pid in excluded_stage2:
        row = summary[summary['sona_id'] == pid].iloc[0]
        trials = row['n_trials_total']
        acc = row['accuracy_overall']
        print(f"\n  Participant {pid}:")
        print(f"    Trials: {int(trials)} (expected 807-1077)")
        print(f"    Accuracy: {acc:.1%}")
        if trials < 100:
            print(f"    Reason: Extremely low trial count (<100)")
        elif trials < 500:
            print(f"    Reason: Insufficient trials (<500)")
        elif pd.isna(row.get('ies_total', None)) or pd.isna(row.get('less_total_events', None)):
            print(f"    Reason: Missing questionnaire data")
        else:
            print(f"    Reason: Unknown - check data quality")

# Stage 3 → 4: Statistical Data to ANOVA
excluded_stage3 = sorted(stats_ids - anova_ids)
if excluded_stage3:
    print(f"\n{'='*80}")
    print(f"EXCLUDED: Statistical Data → ANOVA (N={len(excluded_stage3)})")
    print(f"{'='*80}")
    for pid in excluded_stage3:
        row = stats_data[stats_data['sona_id'] == pid].iloc[0]
        trials = row['n_trials_total']
        acc_low = row['accuracy_low']
        acc_high = row['accuracy_high']
        print(f"\n  Participant {pid}:")
        print(f"    Total trials: {int(trials)}")
        print(f"    Low-load accuracy: {acc_low:.1%}")
        print(f"    High-load accuracy: {acc_high:.1%}")
        
        # Check for missing load data
        if pd.isna(acc_low) or pd.isna(acc_high):
            print(f"    Reason: Missing load-specific accuracy data")
        elif acc_low == 0 or acc_high == 0:
            print(f"    Reason: Zero accuracy in one load condition")
        else:
            print(f"    Reason: Unknown - possibly duplicate or data quality")

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Total started: {len(demo_ids)}")
print(f"Final in ANOVA: {len(anova_ids)}")
print(f"Total excluded: {len(demo_ids) - len(anova_ids)}")
print()

all_excluded = sorted((demo_ids - anova_ids))
print(f"All excluded IDs: {all_excluded}")

# Show trial counts for all participants in final sample
print(f"\n{'='*80}")
print("RETAINED PARTICIPANTS - TRIAL COUNTS")
print(f"{'='*80}")
retained_summary = summary[summary['sona_id'].isin(anova_ids)].sort_values('n_trials_total')
print(f"\n{'ID':<10} {'Trials':<10} {'Accuracy':<10}")
print("-" * 30)
for _, row in retained_summary.iterrows():
    trials = int(row['n_trials_total'])
    acc = row['accuracy_overall']
    flag = "⚠️" if trials < 600 else ""
    print(f"{int(row['sona_id']):<10} {trials:<10} {acc:>8.1%}  {flag}")
    
print("\n⚠️ = Less than 600 trials (~12 blocks)")
