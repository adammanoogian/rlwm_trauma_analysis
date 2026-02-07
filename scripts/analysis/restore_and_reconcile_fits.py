"""
Restore and Reconcile Model Fits from Git History

This script:
1. Loads restored Q-learning fits (N=49 pre-exclusion from commit b686af8)
2. Loads current WM-RL fits (N=49 pre-exclusion)
3. Excludes duplicates (10044, 10073)
4. Checks ID overlap with behavioral N=48
5. Creates matched M1/M2 datasets for analysis

Output: Clean Q-learning and WM-RL parameter files ready for analysis pipeline
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
BEHAVIORAL_SUMMARY = "output/statistical_analyses/data_summary_with_groups.csv"
QLEARNING_RESTORED = "output/mle/qlearning_individual_fits_restored.csv"
WMRL_CURRENT = "output/mle/wmrl_individual_fits.csv"
EXCLUDED_IDS = [10044, 10073]  # Known duplicates

OUTPUT_DIR = Path("output/mle")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("RESTORING AND RECONCILING MODEL FITS")
print("="*70)

# Load data
print("\nLoading datasets...")
behavioral_df = pd.read_csv(BEHAVIORAL_SUMMARY)
qlearning_df = pd.read_csv(QLEARNING_RESTORED)
wmrl_df = pd.read_csv(WMRL_CURRENT)

print(f"  Behavioral summary: N = {len(behavioral_df)}")
print(f"  Q-learning restored (commit daa3fc8): N = {len(qlearning_df)}")
print(f"  WM-RL current: N = {len(wmrl_df)}")

# Check for participant_id column - behavioral uses 'sona_id'
if 'sona_id' in behavioral_df.columns and 'participant_id' not in behavioral_df.columns:
    behavioral_df['participant_id'] = behavioral_df['sona_id']
    print("\n(Renamed 'sona_id' to 'participant_id' in behavioral data)")

# Check for participant_id column
print("\n" + "="*70)
print("CHECKING PARTICIPANT ID FORMATS")
print("="*70)

print("\nBehavioral IDs (first 10):")
print(behavioral_df['participant_id'].head(10).tolist())

print("\nQ-learning IDs (first 10):")
print(qlearning_df['participant_id'].head(10).tolist())

print("\nWM-RL IDs (first 10):")
print(wmrl_df['participant_id'].head(10).tolist())

# Check if Q-learning uses anonymized IDs or numeric IDs
qlearning_has_anon = qlearning_df['participant_id'].astype(str).str.contains('anon_').any()
wmrl_has_anon = wmrl_df['participant_id'].astype(str).str.contains('anon_').any()

print(f"\nQ-learning uses anonymized IDs: {qlearning_has_anon}")
print(f"WM-RL uses anonymized IDs: {wmrl_has_anon}")

if qlearning_has_anon and not wmrl_has_anon:
    print("\n⚠ WARNING: Q-learning has anonymized IDs but WM-RL has numeric IDs")
    print("   This is from before ID remapping (commit daa3fc8)")
    print("   Need to check if we can map them...")
    
    # Check if there's a mapping file
    mapping_file = Path("data/participant_id_mapping.json")
    if mapping_file.exists():
        print(f"\n✓ Found mapping file: {mapping_file}")
        import json
        with open(mapping_file) as f:
            id_mapping = json.load(f)
        
        # Try to remap Q-learning IDs
        print("\nAttempting to remap Q-learning anonymized IDs to numeric IDs...")
        
        # Create reverse mapping (anon_id -> numeric_id)
        reverse_mapping = {v: k for k, v in id_mapping.items()}
        
        # Also check for numeric IDs in the Q-learning data
        numeric_ids_in_qlearning = qlearning_df[~qlearning_df['participant_id'].astype(str).str.contains('anon_', na=False)]
        print(f"  Found {len(numeric_ids_in_qlearning)} numeric IDs already in Q-learning data")
        
        # For anonymized IDs, try to map
        qlearning_df['participant_id_original'] = qlearning_df['participant_id']
        qlearning_df['participant_id'] = qlearning_df['participant_id'].apply(
            lambda x: int(reverse_mapping.get(str(x), x)) if str(x).startswith('anon_') and str(x) in reverse_mapping 
            else (int(x) if str(x).isdigit() else x)
        )
        
        # Check how many we successfully mapped
        still_anon = qlearning_df['participant_id'].astype(str).str.contains('anon_').sum()
        print(f"  After remapping: {still_anon} IDs still anonymized")
        
        if still_anon > 0:
            print(f"\n  ⚠ Could not remap {still_anon} participants:")
            unmapped = qlearning_df[qlearning_df['participant_id'].astype(str).str.contains('anon_')]
            print(f"    {unmapped['participant_id'].tolist()}")
            print("\n  These will be excluded from analysis")
            
            # Remove unmapped IDs
            qlearning_df = qlearning_df[~qlearning_df['participant_id'].astype(str).str.contains('anon_')].copy()
            print(f"\n  Q-learning after removing unmapped: N = {len(qlearning_df)}")

# Exclude known duplicates
print("\n" + "="*70)
print("EXCLUDING DUPLICATE PARTICIPANTS")
print("="*70)

print(f"\nExcluding IDs: {EXCLUDED_IDS}")

qlearning_clean = qlearning_df[~qlearning_df['participant_id'].isin(EXCLUDED_IDS)].copy()
wmrl_clean = wmrl_df[~wmrl_df['participant_id'].isin(EXCLUDED_IDS)].copy()

print(f"  Q-learning after exclusions: N = {len(qlearning_clean)}")
print(f"  WM-RL after exclusions: N = {len(wmrl_clean)}")

# Check overlap
print("\n" + "="*70)
print("CHECKING ID OVERLAP")
print("="*70)

beh_ids = set(behavioral_df['participant_id'])
q_ids = set(qlearning_clean['participant_id'])
w_ids = set(wmrl_clean['participant_id'])

all_three = beh_ids & q_ids & w_ids
print(f"\nParticipants in all 3 datasets: N = {len(all_three)}")

in_beh_not_q = beh_ids - q_ids
in_beh_not_w = beh_ids - w_ids
in_q_not_beh = q_ids - beh_ids
in_w_not_beh = w_ids - beh_ids

if in_beh_not_q:
    print(f"\n⚠ In behavioral but NOT Q-learning: N = {len(in_beh_not_q)}")
    print(f"   IDs: {sorted(in_beh_not_q)}")

if in_beh_not_w:
    print(f"\n⚠ In behavioral but NOT WM-RL: N = {len(in_beh_not_w)}")
    print(f"   IDs: {sorted(in_beh_not_w)}")

if in_q_not_beh:
    print(f"\n⚠ In Q-learning but NOT behavioral: N = {len(in_q_not_beh)}")
    print(f"   IDs: {sorted(in_q_not_beh)}")

if in_w_not_beh:
    print(f"\n⚠ In WM-RL but NOT behavioral: N = {len(in_w_not_beh)}")
    print(f"   IDs: {sorted(in_w_not_beh)}")

# Filter to common IDs only
print("\n" + "="*70)
print("CREATING MATCHED DATASETS")
print("="*70)

print(f"\nFiltering all datasets to common N = {len(all_three)}")

behavioral_matched = behavioral_df[behavioral_df['participant_id'].isin(all_three)].copy()
qlearning_matched = qlearning_clean[qlearning_clean['participant_id'].isin(all_three)].copy()
wmrl_matched = wmrl_clean[wmrl_clean['participant_id'].isin(all_three)].copy()

# Sort by participant_id
behavioral_matched = behavioral_matched.sort_values('participant_id').reset_index(drop=True)
qlearning_matched = qlearning_matched.sort_values('participant_id').reset_index(drop=True)
wmrl_matched = wmrl_matched.sort_values('participant_id').reset_index(drop=True)

print(f"  Behavioral matched: N = {len(behavioral_matched)}")
print(f"  Q-learning matched: N = {len(qlearning_matched)}")
print(f"  WM-RL matched: N = {len(wmrl_matched)}")

# Verify perfect alignment
assert (behavioral_matched['participant_id'].values == qlearning_matched['participant_id'].values).all()
assert (behavioral_matched['participant_id'].values == wmrl_matched['participant_id'].values).all()
print("\n✓ All datasets perfectly aligned")

# Save matched datasets
print("\n" + "="*70)
print("SAVING MATCHED DATASETS")
print("="*70)

qlearning_matched.to_csv(OUTPUT_DIR / 'qlearning_individual_fits_matched.csv', index=False)
wmrl_matched.to_csv(OUTPUT_DIR / 'wmrl_individual_fits_matched.csv', index=False)
behavioral_matched.to_csv(OUTPUT_DIR / 'behavioral_summary_matched.csv', index=False)

print(f"\n✓ Saved matched Q-learning fits: {OUTPUT_DIR / 'qlearning_individual_fits_matched.csv'}")
print(f"✓ Saved matched WM-RL fits: {OUTPUT_DIR / 'wmrl_individual_fits_matched.csv'}")
print(f"✓ Saved matched behavioral data: {OUTPUT_DIR / 'behavioral_summary_matched.csv'}")

# Summary statistics
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nFinal matched sample: N = {len(all_three)}")
print(f"\nParticipant IDs:")
print(f"  {sorted(all_three)}")

print(f"\nExcluded from original behavioral N={len(beh_ids)}:")
if len(beh_ids - all_three) > 0:
    print(f"  {sorted(beh_ids - all_three)}")
else:
    print("  None")

print("\n" + "="*70)
print("NEXT STEPS")
print("="*70)

if len(all_three) >= 45:  # Reasonable minimum
    print(f"\n✓ Good overlap! N={len(all_three)} is sufficient for analysis")
    print("\nYou can now run the analysis pipeline:")
    print("  python scripts/analysis/analysis_modelling_base_models.py")
    print("\nRemember to update the file paths in the script to use:")
    print("  - output/mle/qlearning_individual_fits_matched.csv")
    print("  - output/mle/wmrl_individual_fits_matched.csv")
    print("  - output/mle/behavioral_summary_matched.csv")
else:
    print(f"\n⚠ WARNING: Only N={len(all_three)} overlap - may be insufficient")
    print("  Consider waiting for supervisor to rerun complete N=48 fits")

print("\n" + "="*70 + "\n")
