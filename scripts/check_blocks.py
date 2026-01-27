"""Check block distribution in parsed data."""
import pandas as pd
df = pd.read_csv('output/task_trials_long_all_participants.csv')

print('Block distribution:')
block_counts = df.groupby('block').size()
print(block_counts)

print()
print('Blocks >= 3 (will be used for fitting):')
df_fit = df[df['block'] >= 3]
print(f'  Trials: {len(df_fit)}')
print(f'  Participants: {df_fit["sona_id"].nunique()}')

# Check set_size by block
print()
print('Set size by block (first 10):')
ss_by_block = df.groupby('block')['set_size'].first()
print(ss_by_block.head(10))
