"""
Search for files with ~807 trials regardless of SONA ID.
This will help identify if participant 10079 has a different ID in their raw file.
"""

import pandas as pd
from pathlib import Path
import json

print("=" * 80)
print("REVERSE MAPPING SEARCH: Finding files with ~807 trials")
print("=" * 80)

# Target: 807 trials, 0.812 accuracy, 613.7 RT
TARGET_TRIALS = 807
TARGET_ACC = 0.812
TARGET_RT = 613.7

# Load behavioral summary
beh_summary = pd.read_csv('output/statistical_analyses/data_summary_with_groups.csv')
p10079 = beh_summary[beh_summary['sona_id'] == 10079].iloc[0]

print(f"\nTarget participant 10079:")
print(f"  Trials: {p10079['n_trials_total']}")
print(f"  Accuracy: {p10079['accuracy_overall']:.3f}")
print(f"  Mean RT: {p10079['mean_rt_overall']:.1f}")

# Load current mapping
mapping = json.load(open('data/participant_id_mapping.json'))
print(f"\nCurrent mapping has {len(mapping)} files")

# Search experiment server folder
folder = Path('d:/THESIS/rlwm_trauma/data')
csv_files = list(folder.glob('rlwm_trauma_PARTICIPANT_SESSION_*.csv'))

print(f"\n" + "=" * 80)
print(f"Searching {len(csv_files)} files for ~807 trials...")
print("=" * 80)

candidates = []

for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file)
        
        # Count task trials
        if 'trial_type' in df.columns:
            task_trials = df[df['trial_type'] == 'rlwm-trial']
            n_trials = len(task_trials)
            
            # Look for files with ~807 trials (within 10 trials)
            if 800 <= n_trials <= 815:
                # Get SONA ID from file
                raw_sona_id = df['sona_id'].iloc[0] if 'sona_id' in df.columns else None
                
                # Check if in mapping
                mapped_info = mapping.get(csv_file.name, {})
                mapped_id = mapped_info.get('assigned_id', 'unmapped')
                
                # Calculate stats
                if len(task_trials) > 0 and 'correct' in task_trials.columns and 'rt' in task_trials.columns:
                    acc = task_trials['correct'].mean()
                    rt = task_trials['rt'].mean()
                    
                    # Check if stats match 10079
                    acc_diff = abs(acc - TARGET_ACC)
                    rt_diff = abs(rt - TARGET_RT)
                    trial_diff = abs(n_trials - TARGET_TRIALS)
                    
                    is_likely_match = (acc_diff < 0.02 and rt_diff < 20)
                    
                    candidates.append({
                        'file': csv_file.name,
                        'raw_sona_id': raw_sona_id,
                        'mapped_id': mapped_id,
                        'n_trials': n_trials,
                        'accuracy': acc,
                        'mean_rt': rt,
                        'acc_diff': acc_diff,
                        'rt_diff': rt_diff,
                        'likely_match': is_likely_match
                    })
                    
    except Exception as e:
        pass

print(f"\nFound {len(candidates)} files with ~807 trials:")
print("=" * 80)

# Sort by how well they match
candidates.sort(key=lambda x: x['acc_diff'] + x['rt_diff']/1000)

for c in candidates:
    marker = "✓✓✓ LIKELY MATCH" if c['likely_match'] else ""
    print(f"\n{c['file']} {marker}")
    print(f"  Raw SONA ID: {c['raw_sona_id']}")
    print(f"  Mapped ID: {c['mapped_id']}")
    print(f"  Trials: {c['n_trials']} (diff: {abs(c['n_trials'] - TARGET_TRIALS)})")
    print(f"  Accuracy: {c['accuracy']:.3f} (diff: {c['acc_diff']:.4f})")
    print(f"  Mean RT: {c['mean_rt']:.1f} (diff: {c['rt_diff']:.1f})")

# Find the best match
if candidates:
    best = candidates[0]
    if best['likely_match']:
        print("\n" + "=" * 80)
        print("LIKELY SOLUTION FOUND!")
        print("=" * 80)
        print(f"\nFile: {best['file']}")
        print(f"Raw SONA ID in file: {best['raw_sona_id']}")
        print(f"Currently mapped to: {best['mapped_id']}")
        print(f"Should be mapped to: 10079")
        print(f"\nStatistics match very closely:")
        print(f"  Trials: {best['n_trials']} vs {TARGET_TRIALS} expected")
        print(f"  Accuracy: {best['accuracy']:.3f} vs {TARGET_ACC:.3f} expected")
        print(f"  RT: {best['mean_rt']:.1f} vs {TARGET_RT:.1f} expected")
        
        if best['mapped_id'] != 'unmapped' and best['mapped_id'] != 10079:
            print(f"\n⚠ WARNING: This file is currently mapped to participant {best['mapped_id']}")
            print(f"   This could be a duplicate or incorrect mapping!")
    else:
        print("\nNo strong statistical match found among 807-trial files.")
else:
    print("\nNo files found with ~807 trials.")
