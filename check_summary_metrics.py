import pandas as pd

df = pd.read_csv('output/summary_participant_metrics.csv')
print(f'Total participants: {len(df)}')
print(f'Has 10079: {(df["sona_id"] == 10079).any()}')

if (df['sona_id'] == 10079).any():
    p = df[df['sona_id'] == 10079].iloc[0]
    print(f'\nParticipant 10079:')
    print(f'  n_trials_total: {p.get("n_trials_total", "N/A")}')
    print(f'  accuracy: {p.get("accuracy_overall", "N/A")}')
    
print(f'\nAll SONA IDs: {sorted(df["sona_id"].unique())}')
