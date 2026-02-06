import pandas as pd

beh = pd.read_csv('output/mle/behavioral_summary_matched_with_metrics.csv')

print("Exclusion criteria from config.py:")
print("  MIN_TRIALS = 50")
print("  MIN_ACCURACY = 0.3")
print()

problem_pids = [10001, 10053, 10062]

print("Participant  n_trials  accuracy  Meets_MIN_TRIALS  Meets_MIN_ACC")
print("-" * 70)
for pid in problem_pids:
    p = beh[beh['participant_id']==pid].iloc[0]
    trials = p['n_trials_total']
    acc = p['accuracy_overall']
    print(f"{pid:11}  {trials:8.0f}   {acc:7.3f}  {str(trials>=50):16}  {str(acc>=0.3):13}")

print()
print("All three participants MEET the documented exclusion criteria.")
print()
print("However, participant 10062 has 807 trials (unusual - possible duplicate?)")
print("Let me check the full sample distribution...")
print()

stats = beh[['participant_id', 'n_trials_total', 'accuracy_overall']].sort_values('n_trials_total')
print(f"Trial count range: {stats['n_trials_total'].min():.0f} to {stats['n_trials_total'].max():.0f}")
print(f"Mean trials: {stats['n_trials_total'].mean():.1f} (SD = {stats['n_trials_total'].std():.1f})")
print(f"Median trials: {stats['n_trials_total'].median():.0f}")
print()
print("Participants with >500 trials:")
print(stats[stats['n_trials_total'] > 500])
