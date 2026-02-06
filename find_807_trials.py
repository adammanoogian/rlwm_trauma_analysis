import pandas as pd
import json
from pathlib import Path

# Load current mapping
mapping = json.load(open('data/participant_id_mapping.json'))

# Check all CSV files for 807 trials
data_path = Path('data')
results = []

print("Checking all CSV files for trial counts...")
for csv_file in sorted(data_path.glob('*.csv')):
    try:
        df = pd.read_csv(csv_file)
        task_trials = df[df['trial_type'] == 'rlwm-trial'] if 'trial_type' in df.columns else pd.DataFrame()
        n_trials = len(task_trials)
        
        # Get mapping info if exists
        mapped_id = mapping.get(csv_file.name, {}).get('assigned_id', 'unmapped')
        
        results.append({
            'file': csv_file.name,
            'n_trials': n_trials,
            'mapped_id': mapped_id
        })
        
        # Print 807-trial files
        if n_trials == 807:
            print(f"\n807 TRIALS: {csv_file.name}")
            print(f"  Mapped ID: {mapped_id}")
            
    except Exception as e:
        print(f"Error reading {csv_file.name}: {e}")

# Show all 807-trial files
print("\n" + "="*80)
print("ALL FILES WITH 807 TRIALS:")
print("="*80)
trial_807 = [r for r in results if r['n_trials'] == 807]
for r in trial_807:
    print(f"{r['mapped_id']:>8} : {r['file']}")

print(f"\nTotal files with 807 trials: {len(trial_807)}")

# Check if any are unmapped
unmapped_807 = [r for r in trial_807 if r['mapped_id'] == 'unmapped']
if unmapped_807:
    print(f"\nUNMAPPED files with 807 trials: {len(unmapped_807)}")
    for r in unmapped_807:
        print(f"  {r['file']}")
