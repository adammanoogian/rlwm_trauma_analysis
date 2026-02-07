"""
Comprehensive search for participant 10079's raw data file.
Searches the experiment server folder for the missing participant.
"""

import pandas as pd
from pathlib import Path
import json

print("=" * 80)
print("SEARCHING FOR PARTICIPANT 10079")
print("=" * 80)

# Target participant info from behavioral summary
beh_summary = pd.read_csv('output/statistical_analyses/data_summary_with_groups.csv')
p10079 = beh_summary[beh_summary['sona_id'] == 10079].iloc[0]

print("\nTarget participant (from behavioral summary):")
print(f"  SONA ID: 10079")
print(f"  Expected trials: {p10079['n_trials_total']}")
print(f"  Accuracy: {p10079['accuracy_overall']:.3f}")
print(f"  Mean RT: {p10079['mean_rt_overall']:.1f}")

# Search both data folders
folders_to_search = [
    Path('d:/THESIS/rlwm_trauma/data'),  # Experiment server
    Path('data')  # Analysis folder
]

all_results = []

for folder in folders_to_search:
    print(f"\n" + "=" * 80)
    print(f"SEARCHING: {folder}")
    print("=" * 80)
    
    if not folder.exists():
        print(f"  Folder does not exist!")
        continue
    
    csv_files = list(folder.glob('rlwm_trauma_PARTICIPANT_SESSION_*.csv'))
    print(f"  Found {len(csv_files)} CSV files")
    
    for i, csv_file in enumerate(csv_files, 1):
        try:
            # Read file
            df = pd.read_csv(csv_file)
            
            # Check for SONA ID 10079
            if 'sona_id' in df.columns:
                sona_ids = df['sona_id'].dropna().unique()
                
                if 10079 in sona_ids:
                    print(f"\n✓✓✓ FOUND PARTICIPANT 10079! ✓✓✓")
                    print(f"  File: {csv_file.name}")
                    print(f"  Full path: {csv_file}")
                    
                    # Count task trials
                    if 'trial_type' in df.columns:
                        task_trials = df[df['trial_type'] == 'rlwm-trial']
                        print(f"  Task trials: {len(task_trials)}")
                        
                        if len(task_trials) > 0 and 'correct' in task_trials.columns:
                            acc = task_trials['correct'].mean()
                            rt = task_trials['rt'].mean()
                            print(f"  Accuracy: {acc:.3f}")
                            print(f"  Mean RT: {rt:.1f}")
                            
                            # Check if stats match
                            trial_match = abs(len(task_trials) - p10079['n_trials_total']) <= 5
                            acc_match = abs(acc - p10079['accuracy_overall']) < 0.01
                            rt_match = abs(rt - p10079['mean_rt_overall']) < 10
                            
                            if trial_match and acc_match and rt_match:
                                print(f"  ✓ Statistics MATCH expected values!")
                            else:
                                print(f"  ⚠ Statistics DO NOT match:")
                                if not trial_match:
                                    print(f"    Trials: {len(task_trials)} vs expected {p10079['n_trials_total']}")
                                if not acc_match:
                                    print(f"    Accuracy: {acc:.3f} vs expected {p10079['accuracy_overall']:.3f}")
                                if not rt_match:
                                    print(f"    RT: {rt:.1f} vs expected {p10079['mean_rt_overall']:.1f}")
                    
                    all_results.append({
                        'file': csv_file,
                        'folder': folder,
                        'match': True
                    })
                    
        except Exception as e:
            # Skip files that can't be read
            if i % 20 == 0:
                print(f"  Checked {i}/{len(csv_files)} files...", end='\r')
    
    print(f"  Checked all {len(csv_files)} files")

print("\n" + "=" * 80)
print("SEARCH COMPLETE")
print("=" * 80)

if all_results:
    print(f"\n✓ Found {len(all_results)} file(s) containing SONA ID 10079:")
    for r in all_results:
        print(f"\n  {r['file'].name}")
        print(f"  Location: {r['folder']}")
        
    # Provide next steps
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Verify the file above is the correct one")
    print("2. Add to participant_id_mapping.json if needed")
    print("3. Regenerate trial data: python scripts/01_parse_raw_data.py")
    print("4. Rerun feedback/perseveration analysis")
    
else:
    print("\n✗ Participant 10079 NOT FOUND in any CSV file")
    print("\nPossible reasons:")
    print("  - File was deleted or never synced")
    print("  - SONA ID in the raw file is different (mapping issue)")
    print("  - Data was entered manually in behavioral summary")
    print("\nRecommendation: Contact your supervisor for the missing data file")
