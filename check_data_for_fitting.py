import pandas as pd
from config import EXCLUDED_PARTICIPANTS

df = pd.read_csv('output/task_trials_long_all_participants.csv')

print(f'Total participants: {df["sona_id"].nunique()}')
print(f'Excluded IDs present: {set(df["sona_id"]) & set(EXCLUDED_PARTICIPANTS)}')
print(f'Blocks range: {df["block"].min()}-{df["block"].max()}')
print(f'Total trials: {len(df)}')
print(f'\nParticipants: {sorted(df["sona_id"].unique())[:10]}...')
