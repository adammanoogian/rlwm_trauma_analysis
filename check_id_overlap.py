import json
import pandas as pd

# Load participant ID mapping
with open('data/participant_id_mapping.json') as f:
    mapping = json.load(f)

assigned_ids = sorted(set([v['assigned_id'] for v in mapping.values()]))
print(f"IDs in mapping.json: {len(assigned_ids)}")
print(f"First 20: {assigned_ids[:20]}")

# Load MLE fits
mle = pd.read_csv('output/mle/qlearning_individual_fits.csv')
mle_ids = sorted(mle['participant_id'].astype(str).unique())
print(f"\nIDs in MLE fits: {len(mle_ids)}")
print(f"First 20: {mle_ids[:20]}")

# Load group assignments
groups = pd.read_csv('output/trauma_groups/group_assignments.csv')
group_ids = sorted(groups['sona_id'].astype(int).unique())
print(f"\nIDs in group_assignments: {len(group_ids)}")
print(f"First 20: {group_ids[:20]}")

# Check overlaps
overlap_mapping_groups = set(assigned_ids).intersection(set(group_ids))
print(f"\nOverlap (mapping ∩ groups): {len(overlap_mapping_groups)}")

# Convert MLE numeric IDs to int for comparison
mle_numeric = []
for x in mle_ids:
    if not x.startswith('anon'):
        mle_numeric.append(int(x))
        
overlap_mle_groups = set(mle_numeric).intersection(set(group_ids))
print(f"Overlap (MLE numeric ∩ groups): {len(overlap_mle_groups)}")
print(f"Overlapping IDs: {sorted(overlap_mle_groups)}")

# Check which group IDs are NOT in MLE
missing_from_mle = set(group_ids) - set(mle_numeric) - set(assigned_ids)
print(f"\nGroup IDs with no MLE fits: {len(set(group_ids) - set(assigned_ids))}")

print(f"\nSummary:")
print(f"  - mapping.json has {len(assigned_ids)} assigned IDs")
print(f"  - MLE fits has {len(mle_ids)} participants ({len(mle_numeric)} numeric, {len(mle_ids)-len(mle_numeric)} anonymous)")
print(f"  - group_assignments has {len(group_ids)} participants")
print(f"  - {len(overlap_mapping_groups)} participants match between mapping and groups")
print(f"  - Need to fit {len(set(group_ids) - set(assigned_ids))} more participants")
