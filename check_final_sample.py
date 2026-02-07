import pandas as pd

# Load data
demo = pd.read_csv('output/demographics_complete.csv')
stats = pd.read_csv('output/statistical_analyses/data_summary_with_groups.csv')

print(f"Total demographics: {len(demo)}")
print(f"Total in analysis: {len(stats)}")
print(f"Excluded: {len(demo) - len(stats)}")
print()

# Find excluded IDs
demo_ids = set(demo['sona_id'])
stats_ids = set(stats['sona_id'])
excluded_ids = sorted(demo_ids - stats_ids)

print(f"Excluded participant IDs: {excluded_ids}")
print()

# Check trial counts for all participants
print("Trial counts for all participants:")
stats_sorted = stats.sort_values('n_trials_total')
for _, row in stats_sorted.iterrows():
    trials = row['n_trials_total']
    acc = row['accuracy_overall']
    status = "⚠️ LOW" if trials < 500 else "✓"
    print(f"  {status} ID {int(row['sona_id'])}: {int(trials)} trials, {acc:.1%} accuracy")
