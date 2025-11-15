"""
Parse data from all participants including those with anonymous/missing sona_ids.

This script handles participants who:
- Completed full experiment
- Completed partial experiment (>=100 trials)
- Have missing or unknown sona_ids (assigns anonymous IDs)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Load ID mapping
mapping_file = Path('data/participant_id_mapping.json')
with open(mapping_file, 'r') as f:
    id_mapping = json.load(f)

print("=" * 80)
print("PARSING ALL PARTICIPANTS (INCLUDING PARTIAL DATA)")
print("=" * 80)
print(f"\nUsing ID mapping from: {mapping_file}")
print(f"Total participants to process: {len(id_mapping)}\n")

# Process each participant
all_task_trials = []
all_demographics = []
all_survey1 = []
all_survey2 = []

for filename, info in id_mapping.items():
    assigned_id = info['assigned_id']
    filepath = Path('data') / filename

    print(f"Processing ID {assigned_id} ({filename[:50]}...)")

    try:
        df = pd.read_csv(filepath, low_memory=False)

        # Override sona_id with assigned ID
        df['sona_id'] = assigned_id

        # Extract task trials (categorize-html type)
        if 'trial_type' in df.columns:
            task_data = df[df['trial_type'] == 'categorize-html'].copy()

            if len(task_data) > 0:
                # Add trial info
                task_data['trial_in_experiment'] = range(1, len(task_data) + 1)

                # Use existing block column if available, otherwise try to extract
                if 'block' not in task_data.columns or task_data['block'].isna().all():
                    # Try to extract block from internal_node_id if available
                    if 'internal_node_id' in task_data.columns:
                        # internal_node_id typically contains block info like "0.0.3.0" where 3rd number is block
                        def extract_block(node_id):
                            if pd.isna(node_id):
                                return np.nan
                            parts = str(node_id).split('.')
                            if len(parts) >= 3:
                                return float(parts[2])
                            return np.nan

                        task_data['block'] = task_data['internal_node_id'].apply(extract_block)

                    # If still not extracted, estimate from trial number (45-50 trials per block)
                    if 'block' not in task_data.columns or task_data['block'].isna().all():
                        task_data['block'] = (task_data['trial_in_experiment'] // 50) + 3  # Start at block 3

                # Extract key columns
                task_cols = ['sona_id', 'trial_in_experiment', 'block', 'time_elapsed',
                           'rt', 'stimulus', 'response', 'correct']
                task_subset = task_data[[ c for c in task_cols if c in task_data.columns]].copy()

                all_task_trials.append(task_subset)
                print(f"  Task trials: {len(task_data)}")

        # Extract demographics
        demo_data = df[df['section'] == 'demographics_text']
        if len(demo_data) > 0:
            # Simple extraction - would need more sophisticated parsing for full demographics
            demo_row = {'sona_id': assigned_id}
            all_demographics.append(demo_row)
            print(f"  Demographics: Found")

        # Extract Survey 1 (LEC-5)
        survey1_data = df[df['section'] == 'survey1']
        if len(survey1_data) > 0:
            survey1_row = {'sona_id': assigned_id}
            # Would need to parse response column properly
            all_survey1.append(survey1_row)
            print(f"  Survey 1: Found")

        # Extract Survey 2 (IES-R)
        survey2_data = df[df['section'] == 'survey2']
        if len(survey2_data) > 0:
            survey2_row = {'sona_id': assigned_id}
            # Would need to parse response column properly
            all_survey2.append(survey2_row)
            print(f"  Survey 2: Found")

    except Exception as e:
        print(f"  ERROR: {e}")

print()
print("=" * 80)
print("COMBINING DATA")
print("=" * 80)

# Combine all task trials
if all_task_trials:
    task_df = pd.concat(all_task_trials, ignore_index=True)

    # Add additional columns
    task_df['trial_in_block'] = task_df.groupby(['sona_id', 'block']).cumcount() + 1

    # Infer set size from stimulus (if stimulus is 1-6, max stimulus in block = set size)
    if 'stimulus' in task_df.columns:
        task_df['stimulus'] = task_df['stimulus'].fillna(-1).astype(int)
        task_df['set_size'] = task_df.groupby(['sona_id', 'block'])['stimulus'].transform('max')
        task_df['set_size'] = task_df['set_size'].replace(-1, np.nan)

    # Add load condition
    if 'set_size' in task_df.columns:
        task_df['load_condition'] = task_df['set_size'].apply(
            lambda x: 'low' if x <= 3 else 'high' if x >= 4 else np.nan
        )

    # Ensure correct column exists
    if 'correct' not in task_df.columns and 'response' in task_df.columns:
        # Would need to infer from feedback, for now mark as unknown
        task_df['correct'] = np.nan

    print(f"\\nCombined task trials: {len(task_df)} trials")
    print(f"Participants: {task_df['sona_id'].nunique()}")
    print(f"Blocks: {sorted(task_df['block'].dropna().unique())}")

    # Save
    output_path = Path('output') / 'task_trials_long_all_participants.csv'
    task_df.to_csv(output_path, index=False)
    print(f"\\n[SAVED] {output_path}")

    # Summary statistics
    print("\\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    trials_per_participant = task_df.groupby('sona_id').size()
    print(f"\\nTrials per participant:")
    for sona_id, n_trials in trials_per_participant.items():
        est_blocks = n_trials / 50
        status = "Complete" if n_trials >= 900 else "Partial"
        print(f"  ID {sona_id}: {n_trials:4d} trials (~{est_blocks:4.1f} blocks) - {status}")

    print(f"\\nOverall:")
    print(f"  Total trials: {len(task_df)}")
    print(f"  Mean trials per participant: {trials_per_participant.mean():.1f}")
    print(f"  Complete participants (>=900 trials): {(trials_per_participant >= 900).sum()}")
    print(f"  Partial participants (100-899 trials): {((trials_per_participant >= 100) & (trials_per_participant < 900)).sum()}")

else:
    print("No task data found!")

print("\\n" + "=" * 80)
print("PARSING COMPLETE")
print("=" * 80)
print("\\nNext steps:")
print("  1. Review output/task_trials_long_all_participants.csv")
print("  2. Run compute_summary_metrics.py to calculate performance metrics")
print("  3. Generate visualizations with all 9 participants")
print("=" * 80)
