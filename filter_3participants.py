import pandas as pd

# Load full data
df = pd.read_csv('output/task_trials_long_all_participants.csv')

# Filter to 3 participants (one from each trauma group)
participants = [10001, 10002, 9187]
df_subset = df[df['sona_id'].isin(participants)]

# Save subset
df_subset.to_csv('output/task_trials_3participants.csv', index=False)
print(f'Saved {len(df_subset)} trials from {df_subset["sona_id"].nunique()} participants')
print(f'Participants: {sorted(df_subset["sona_id"].unique())}')
