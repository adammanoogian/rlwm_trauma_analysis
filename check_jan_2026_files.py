import pandas as pd

# Check the three January 2026 files
files = [
    'data/rlwm_trauma_PARTICIPANT_SESSION_2026-01-03_21h00.04.905.csv',
    'data/rlwm_trauma_PARTICIPANT_SESSION_2026-01-04_19h13.44.594.csv',
    'data/rlwm_trauma_PARTICIPANT_SESSION_2026-01-27_13h51.52.432.csv'
]

print("Checking January 2026 files for participant 10079:")
print("=" * 80)

for file in files:
    try:
        df = pd.read_csv(file)
        
        # Get SONA IDs
        sona_ids = df['sona_id'].dropna().unique() if 'sona_id' in df.columns else []
        
        # Count task trials
        task_trials = df[df['trial_type'] == 'rlwm-trial'] if 'trial_type' in df.columns else pd.DataFrame()
        
        print(f"\n{file.split('/')[-1]}")
        print(f"  SONA IDs: {list(sona_ids)}")
        print(f"  Task trials: {len(task_trials)}")
        
        if len(task_trials) > 0 and 'correct' in task_trials.columns:
            acc = task_trials['correct'].mean()
            rt = task_trials['rt'].mean()
            print(f"  Accuracy: {acc:.3f}")
            print(f"  Mean RT: {rt:.1f}")
            
            # Check if matches 10079
            if 800 <= len(task_trials) <= 815:
                if abs(acc - 0.812) < 0.02 and abs(rt - 613.7) < 20:
                    print(f"  ✓✓✓ LIKELY MATCH FOR 10079!")
                    
    except Exception as e:
        print(f"\n{file.split('/')[-1]}: Error - {e}")

print("\n" + "=" * 80)
print("CONCLUSION:")
print("If any file above shows ~807 trials with 81% accuracy and ~614 RT,")
print("that's participant 10079 (regardless of SONA ID in the file).")
