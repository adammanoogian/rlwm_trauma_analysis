import pandas as pd
from pathlib import Path

# Load behavioral summary to see 10079's details
df_summary = pd.read_csv('output/statistical_analyses/data_summary_with_groups.csv')
p10079 = df_summary[df_summary['sona_id'] == 10079].iloc[0]

print("Participant 10079 from behavioral summary:")
print(f"  SONA ID: {p10079['sona_id']}")
print(f"  Trials: {p10079['n_trials_total']}")
print(f"  Accuracy: {p10079['accuracy_overall']:.3f}")
print(f"  Mean RT: {p10079['mean_rt_overall']:.1f}")

# Now check ALL raw CSV files in the experiment data folder for matching statistics
print("\n" + "="*80)
print("Searching experiment data folder for matching participant...")
print("="*80)

exp_data_path = Path('d:/THESIS/rlwm_trauma/data')

matches = []
for csv_file in sorted(exp_data_path.glob('*.csv')):
    try:
        df = pd.read_csv(csv_file)
        
        # Filter to task trials
        if 'trial_type' in df.columns:
            task_df = df[df['trial_type'] == 'rlwm-trial'].copy()
            
            if len(task_df) > 0:
                # Calculate accuracy and RT
                n_trials = len(task_df)
                
                if 'correct' in task_df.columns and 'rt' in task_df.columns:
                    accuracy = task_df['correct'].mean()
                    mean_rt = task_df['rt'].mean()
                    
                    # Check if matches 10079's stats (within tolerance)
                    trial_match = abs(n_trials - p10079['n_trials_total']) <= 5
                    acc_match = abs(accuracy - p10079['accuracy_overall']) < 0.01
                    rt_match = abs(mean_rt - p10079['mean_rt_overall']) < 10
                    
                    if trial_match and acc_match and rt_match:
                        sona_id = df['sona_id'].iloc[0] if 'sona_id' in df.columns else 'unknown'
                        print(f"\n✓ MATCH FOUND: {csv_file.name}")
                        print(f"  SONA ID in file: {sona_id}")
                        print(f"  Trials: {n_trials} (expected {p10079['n_trials_total']})")
                        print(f"  Accuracy: {accuracy:.3f} (expected {p10079['accuracy_overall']:.3f})")
                        print(f"  Mean RT: {mean_rt:.1f} (expected {p10079['mean_rt_overall']:.1f})")
                        
                        matches.append({
                            'file': csv_file.name,
                            'sona_id': sona_id,
                            'n_trials': n_trials
                        })
                        
    except Exception as e:
        pass  # Skip problematic files

if not matches:
    print("\nNo exact matches found. Checking for files with ~807 trials...")
    for csv_file in sorted(exp_data_path.glob('*.csv')):
        try:
            df = pd.read_csv(csv_file)
            if 'trial_type' in df.columns:
                task_df = df[df['trial_type'] == 'rlwm-trial']
                if 800 <= len(task_df) <= 815:
                    sona_id = df['sona_id'].iloc[0] if 'sona_id' in df.columns else 'unknown'
                    print(f"\n  {csv_file.name}")
                    print(f"    SONA ID: {sona_id}, Trials: {len(task_df)}")
        except:
            pass

print("\n" + "="*80)
print(f"Total matches: {len(matches)}")
