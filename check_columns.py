import pandas as pd

df = pd.read_csv('output/task_trials_long_all_participants.csv')
print('Columns in file:')
print(list(df.columns))

print(f'\nkey_press exists: {"key_press" in df.columns}')
if 'key_press' in df.columns:
    print(f'key_press NaN count: {df["key_press"].isna().sum()} / {len(df)}')
    print(f'key_press non-NaN count: {df["key_press"].notna().sum()}')
    print(f'Sample key_press values: {df["key_press"].dropna().unique()[:10]}')

    print(f'\nBlocks >= 3 sample:')
    df_main = df[df["block"] >= 3]
    print(f'  Trials: {len(df_main)}')
    print(f'  key_press NaN: {df_main["key_press"].isna().sum()}')
    print(f'  key_press non-NaN: {df_main["key_press"].notna().sum()}')
    if df_main["key_press"].notna().sum() > 0:
        print(f'\nFirst 10 rows with key_press:')
        print(df_main[['sona_id', 'block', 'stimulus', 'key_press', 'correct']].head(10))
