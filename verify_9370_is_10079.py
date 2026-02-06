import pandas as pd

# Load the file
df = pd.read_csv('data/rlwm_trauma_PARTICIPANT_SESSION_2026-01-27_13h51.52.432.csv')

# Get task trials (categorize-html type)
task_trials = df[df['trial_type'] == 'categorize-html'].copy()

print("File: rlwm_trauma_PARTICIPANT_SESSION_2026-01-27_13h51.52.432.csv")
print("=" * 80)
print(f"SONA ID in file: 9370")
print(f"Task trials: {len(task_trials)}")

if len(task_trials) > 0:
    # Calculate accuracy and RT
    if 'correct' in task_trials.columns:
        accuracy = task_trials['correct'].mean()
        print(f"Accuracy: {accuracy:.3f}")
    
    if 'rt' in task_trials.columns:
        mean_rt = task_trials['rt'].mean()
        print(f"Mean RT: {mean_rt:.1f}")
    
    print("\n" + "=" * 80)
    print("Expected for participant 10079:")
    print(f"  Trials: 807")
    print(f"  Accuracy: 0.812")
    print(f"  Mean RT: 613.7")
    
    print("\n" + "=" * 80)
    if len(task_trials) == 807 and abs(accuracy - 0.812) < 0.01 and abs(mean_rt - 613.7) < 10:
        print("✓✓✓ CONFIRMED: SONA ID 9370 = Participant 10079")
        print("\nThis is a remapping issue!")
        print("The participant entered SONA ID 9370 in the experiment,")
        print("but should be mapped to participant ID 10079 in your analysis.")
    else:
        print("Statistics don't match - might not be the right participant")
