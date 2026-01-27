"""Quick check of parsed data structure."""
import pandas as pd
df = pd.read_csv('output/task_trials_long_all_participants.csv')

print('Columns:', list(df.columns))
print()

# Check for NaN values in key columns
print('NaN counts:')
for col in ['stimulus', 'key_press', 'correct', 'block', 'reward']:
    if col in df.columns:
        print(f'  {col}: {df[col].isna().sum()}')

print()
print('Stimulus range:', df['stimulus'].min(), '-', df['stimulus'].max())
print('key_press unique values:', sorted(df['key_press'].dropna().unique()))
print('reward values:', df['reward'].unique() if 'reward' in df.columns else 'N/A')
print('set_size values:', sorted(df['set_size'].dropna().unique()))

print()
print('Total trials:', len(df))
print('Participants:', df['sona_id'].nunique())
