import pandas as pd

demo = pd.read_csv('output/demographics_complete.csv')
stats = pd.read_csv('output/statistical_analyses/data_summary_with_groups.csv')
summary = pd.read_csv('output/behavioral_summary.csv')

print(f'Demographics N: {len(demo)}')
print(f'Statistical analysis N: {len(stats)}')
print(f'Behavioral summary N: {len(summary)}')
print()

demo_ids = set(demo['sona_id'])
stats_ids = set(stats['sona_id'])
summary_ids = set(summary['sona_id'])

missing_from_stats = demo_ids - stats_ids
missing_from_summary = demo_ids - summary_ids

print(f'Missing from statistical analysis: {sorted(missing_from_stats)}')
print(f'Missing from behavioral summary: {sorted(missing_from_summary)}')
print()

# Check if these participants are in the trauma grouping
if len(missing_from_stats) > 0:
    print("\nChecking trauma group assignment for excluded participants:")
    for pid in sorted(missing_from_stats):
        demo_row = demo[demo['sona_id'] == pid]
        if pid in summary_ids:
            summary_row = summary[summary['sona_id'] == pid]
            print(f"  ID {pid}: In demographics, in behavioral summary")
            if 'ies_total' in summary_row.columns:
                ies = summary_row['ies_total'].values[0]
                less = summary_row.get('less_total_events', [None]).values[0]
                print(f"    IES-R: {ies}, LESS: {less}")
        else:
            print(f"  ID {pid}: In demographics, NOT in behavioral summary")
