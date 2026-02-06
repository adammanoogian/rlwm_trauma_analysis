import pandas as pd
import json

# Load the unmapped file
df = pd.read_csv('data/rlwm_trauma_PARTICIPANT_SESSION_2026-01-27_13h51.52.432.csv')

# Count task trials
task_trials = df[df['trial_type'] == 'rlwm-trial']
print(f'Total trials: {len(task_trials)}')

# Check sona_id
if 'sona_id' in df.columns:
    sona_ids = df['sona_id'].dropna().unique()
    print(f'Sona IDs in file: {sona_ids}')

# Add to mapping
mapping = json.load(open('data/participant_id_mapping.json'))
filename = 'rlwm_trauma_PARTICIPANT_SESSION_2026-01-27_13h51.52.432.csv'

mapping[filename] = {
    'assigned_id': 10079,
    'n_trials': len(task_trials),
    'filename': filename
}

# Save updated mapping
json.dump(mapping, open('data/participant_id_mapping.json', 'w'), indent=2)
print(f'\nAdded {filename} to mapping as participant 10079')
print(f'Trial count: {len(task_trials)}')
