import pandas as pd

# Load MLE fits
mle = pd.read_csv('output/mle/qlearning_individual_fits.csv')
# Load behavioral data
behav = pd.read_csv('output/statistical_analyses/data_summary_with_groups.csv')

mle_ids = set(mle['participant_id'].unique())
behav_ids = set(behav['sona_id'].unique())

print(f"MLE fit IDs (N={len(mle_ids)}): {sorted(list(mle_ids))[:10]}...")
print(f"Behavioral IDs (N={len(behav_ids)}): {sorted(list(behav_ids))[:10]}...")
print(f"\nMatching IDs: {len(mle_ids & behav_ids)}/{len(behav_ids)}")
print(f"MLE IDs not in behavioral: {sorted(mle_ids - behav_ids)}")
print(f"Behavioral IDs not in MLE: {sorted(behav_ids - mle_ids)}")
