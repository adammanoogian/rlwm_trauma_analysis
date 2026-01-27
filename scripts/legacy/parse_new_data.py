"""
Parse participant data from external rlwm_trauma/data directory.

This script handles all participant CSV files from the external data folder,
extracts task trial data, and saves combined output for MLE fitting.

Usage:
    python scripts/parse_new_data.py

Output:
    output/task_trials_long_all_participants.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import sys

# Configuration
DATA_DIR = Path('C:/Users/aman0087/Documents/Github/rlwm_trauma/data')
OUTPUT_DIR = Path('output')
OUTPUT_DIR.mkdir(exist_ok=True)

# Minimum trials to include a participant (exclude incomplete sessions)
MIN_TRIALS = 100


def parse_single_file(filepath: Path) -> pd.DataFrame | None:
    """
    Parse task trials from a single participant CSV file.

    Returns DataFrame with task trials or None if parsing fails.
    """
    try:
        df = pd.read_csv(filepath, low_memory=False)

        if 'trial_type' not in df.columns:
            return None

        # Extract task trials (categorize-html type)
        task_data = df[df['trial_type'] == 'categorize-html'].copy()

        if len(task_data) == 0:
            return None

        # Get participant ID (use sona_id if valid, otherwise use filename hash)
        sona_id = None
        if 'sona_id' in df.columns:
            sona_values = df['sona_id'].dropna().unique()
            # Filter out empty strings and invalid values
            valid_sona = [s for s in sona_values if s and str(s).strip() and str(s) != 'nan']
            if valid_sona:
                sona_id = str(valid_sona[0]).strip()

        # If no valid sona_id, generate from filename
        if not sona_id or sona_id == '':
            # Use hash of filename for consistent ID
            filename_hash = abs(hash(filepath.name)) % 100000
            sona_id = f"anon_{filename_hash}"

        task_data['sona_id'] = sona_id
        task_data['source_file'] = filepath.name

        # Add trial numbering
        task_data['trial_in_experiment'] = range(1, len(task_data) + 1)

        # Handle block column
        if 'block' not in task_data.columns or task_data['block'].isna().all():
            # Estimate from trial number (~50 trials per block)
            task_data['block'] = (task_data['trial_in_experiment'] // 50) + 3

        # Handle key_press mapping
        if 'key_answer' in task_data.columns and 'key_press' not in task_data.columns:
            task_data['key_press'] = task_data['key_answer']

        # Select key columns
        cols_to_keep = ['sona_id', 'trial_in_experiment', 'block', 'stimulus',
                        'key_press', 'correct', 'rt', 'time_elapsed', 'set_size',
                        'load_condition', 'source_file']
        cols_available = [c for c in cols_to_keep if c in task_data.columns]

        return task_data[cols_available].copy()

    except Exception as e:
        print(f"  Error parsing {filepath.name}: {e}")
        return None


def main():
    print("=" * 80)
    print("PARSING NEW PARTICIPANT DATA")
    print("=" * 80)
    print(f"\nSource directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Find all CSV files
    csv_files = sorted(DATA_DIR.glob('rlwm_trauma_*.csv'))
    print(f"\nFound {len(csv_files)} participant files")

    if len(csv_files) == 0:
        print("ERROR: No participant files found!")
        return

    # Parse each file
    all_trials = []
    participant_info = []

    print("\nParsing files...")
    for filepath in tqdm(csv_files, desc='Parsing', unit='file'):
        task_df = parse_single_file(filepath)

        if task_df is not None and len(task_df) >= MIN_TRIALS:
            all_trials.append(task_df)
            participant_info.append({
                'filename': filepath.name,
                'sona_id': task_df['sona_id'].iloc[0],
                'n_trials': len(task_df)
            })

    print(f"\nSuccessfully parsed: {len(all_trials)} participants (>={MIN_TRIALS} trials)")

    if not all_trials:
        print("ERROR: No valid participant data found!")
        return

    # Combine all data
    combined_df = pd.concat(all_trials, ignore_index=True)

    # Add derived columns
    combined_df['trial_in_block'] = combined_df.groupby(['sona_id', 'block']).cumcount() + 1

    # Ensure correct data types
    if 'stimulus' in combined_df.columns:
        combined_df['stimulus'] = pd.to_numeric(combined_df['stimulus'], errors='coerce').fillna(-1).astype(int)
        # Convert to 0-indexed (stimulus 1-6 becomes 0-5)
        combined_df['stimulus'] = combined_df['stimulus'] - 1
        combined_df.loc[combined_df['stimulus'] < 0, 'stimulus'] = -1  # Keep invalid as -1

    if 'correct' in combined_df.columns:
        combined_df['correct'] = pd.to_numeric(combined_df['correct'], errors='coerce')

    if 'block' in combined_df.columns:
        combined_df['block'] = pd.to_numeric(combined_df['block'], errors='coerce').astype(int)

    # Add reward column (convert correct to float 0.0/1.0)
    if 'correct' in combined_df.columns:
        combined_df['reward'] = combined_df['correct'].astype(float)

    # Handle key_press: convert to int, handle -1 as invalid
    if 'key_press' in combined_df.columns:
        combined_df['key_press'] = pd.to_numeric(combined_df['key_press'], errors='coerce').fillna(-1).astype(int)

    # === DATA CLEANING FILTERS ===
    # 1. Remove practice trials (blocks 1-2, keep experimental blocks >= 3)
    n_before = len(combined_df)
    combined_df = combined_df[combined_df['block'] >= 3].copy()
    n_practice = n_before - len(combined_df)
    if n_practice > 0:
        print(f"\nRemoved {n_practice} practice trials (blocks 1-2)")

    # 2. Filter out invalid trials (key_press == -1 or stimulus < 0)
    n_before = len(combined_df)
    combined_df = combined_df[(combined_df['key_press'] >= 0) & (combined_df['stimulus'] >= 0)]
    n_removed = n_before - len(combined_df)
    if n_removed > 0:
        print(f"Removed {n_removed} invalid trials (no response or invalid stimulus)")

    # Infer set_size if not present (stimulus is now 0-indexed, so max+1 = set_size)
    if 'set_size' not in combined_df.columns or combined_df['set_size'].isna().all():
        combined_df['set_size'] = combined_df.groupby(['sona_id', 'block'])['stimulus'].transform('max') + 1

    # Add load_condition if not present
    if 'load_condition' not in combined_df.columns or combined_df['load_condition'].isna().all():
        combined_df['load_condition'] = combined_df['set_size'].apply(
            lambda x: 'low' if x <= 3 else 'high' if x >= 4 else 'unknown'
        )

    # Save combined data
    output_path = OUTPUT_DIR / 'task_trials_long_all_participants.csv'
    combined_df.to_csv(output_path, index=False)
    print(f"\n[SAVED] {output_path}")

    # Save participant info
    info_df = pd.DataFrame(participant_info)
    info_path = OUTPUT_DIR / 'participant_info.csv'
    info_df.to_csv(info_path, index=False)
    print(f"[SAVED] {info_path}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    n_participants = combined_df['sona_id'].nunique()
    trials_per_participant = combined_df.groupby('sona_id').size()

    print(f"\nTotal participants: {n_participants}")
    print(f"Total trials: {len(combined_df):,}")
    print(f"\nTrials per participant:")
    print(f"  Mean: {trials_per_participant.mean():.1f}")
    print(f"  Min: {trials_per_participant.min()}")
    print(f"  Max: {trials_per_participant.max()}")

    # Complete vs partial
    complete = (trials_per_participant >= 900).sum()
    partial = ((trials_per_participant >= MIN_TRIALS) & (trials_per_participant < 900)).sum()
    print(f"\nComplete participants (>=900 trials): {complete}")
    print(f"Partial participants ({MIN_TRIALS}-899 trials): {partial}")

    # Blocks
    if 'block' in combined_df.columns:
        blocks = sorted(combined_df['block'].unique())
        print(f"\nBlocks: {blocks}")

    # Accuracy summary
    if 'correct' in combined_df.columns:
        acc = combined_df['correct'].mean()
        print(f"\nOverall accuracy: {acc:.2%}")

        # Participant-level performance check
        participant_acc = combined_df.groupby('sona_id')['correct'].mean()
        print(f"\nParticipant-level accuracy:")
        print(f"  Mean: {participant_acc.mean():.2%}")
        print(f"  Std:  {participant_acc.std():.2%}")
        print(f"  Min:  {participant_acc.min():.2%}")
        print(f"  Max:  {participant_acc.max():.2%}")

        # Show lowest 25% of performers (bottom quartile)
        q25_cutoff = participant_acc.quantile(0.25)
        lowest_quartile = participant_acc[participant_acc <= q25_cutoff].sort_values()
        print(f"\n  Bottom 25% (n={len(lowest_quartile)}, cutoff={q25_cutoff:.2%}):")
        for pid, p_acc in lowest_quartile.items():
            n_trials = len(combined_df[combined_df['sona_id'] == pid])
            print(f"    {pid}: {p_acc:.2%} ({n_trials} trials)")

    print("\n" + "=" * 80)
    print("PARSING COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
