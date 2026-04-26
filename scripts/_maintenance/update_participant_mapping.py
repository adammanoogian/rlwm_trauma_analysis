"""
Update participant_id_mapping.json to include all CSV files in data/raw/.

This script scans data/raw/ for all CSV files matching the experiment pattern
and assigns anonymous IDs to new participants, preserving existing mappings.
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Resolve to project root so `from config import ...` works regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import DATA_RAW_DIR

def count_task_trials(filepath):
    """Count number of task trials (categorize-html) in a CSV file."""
    try:
        df = pd.read_csv(filepath, low_memory=False)
        if 'trial_type' in df.columns:
            task_data = df[df['trial_type'] == 'categorize-html']
            return len(task_data)
        return 0
    except Exception as e:
        print(f"  Error reading {filepath.name}: {e}")
        return 0

def main():
    data_dir = DATA_RAW_DIR  # CCDS tier (Phase 31-02; was Path('data') flat pre-migration)
    mapping_file = data_dir / 'participant_id_mapping.json'

    print("=" * 80)
    print("UPDATING PARTICIPANT ID MAPPING")
    print("=" * 80)

    # Load existing mapping or create new one
    if mapping_file.exists():
        with open(mapping_file, 'r') as f:
            existing_mapping = json.load(f)
        print(f"\nLoaded existing mapping: {len(existing_mapping)} participants")

        # Get highest existing ID
        existing_ids = [info['assigned_id'] for info in existing_mapping.values()]
        next_id = max(existing_ids) + 1 if existing_ids else 10000
    else:
        existing_mapping = {}
        next_id = 10000
        print("\nNo existing mapping found, creating new one")

    # Find all CSV files matching the experiment pattern
    csv_files = sorted(data_dir.glob('rlwm_trauma_PARTICIPANT_SESSION_*.csv'))

    print(f"\nFound {len(csv_files)} CSV files in data directory")
    print(f"Next available ID: {next_id}")
    print()

    # Track changes
    new_files = []
    updated_mapping = existing_mapping.copy()

    # Process each file
    for csv_file in csv_files:
        filename = csv_file.name

        if filename in updated_mapping:
            # File already mapped, verify trial count
            existing_trials = updated_mapping[filename]['n_trials']
            current_trials = count_task_trials(csv_file)

            if current_trials != existing_trials:
                print(f"⚠️  Updated trial count for {filename}: {existing_trials} → {current_trials}")
                updated_mapping[filename]['n_trials'] = current_trials
        else:
            # New file, add to mapping
            n_trials = count_task_trials(csv_file)

            updated_mapping[filename] = {
                'assigned_id': next_id,
                'n_trials': n_trials,
                'filename': filename
            }

            status = "Complete" if n_trials >= 900 else "Partial"
            print(f"✓ Added ID {next_id}: {filename[:50]}... ({n_trials} trials - {status})")

            new_files.append(filename)
            next_id += 1

    # Save updated mapping
    with open(mapping_file, 'w') as f:
        json.dump(updated_mapping, f, indent=2)

    print()
    print("=" * 80)
    print("MAPPING UPDATE COMPLETE")
    print("=" * 80)
    print(f"\nTotal participants: {len(updated_mapping)}")
    print(f"New participants added: {len(new_files)}")
    print(f"\n[SAVED] {mapping_file}")

    # Summary statistics
    all_trials = [info['n_trials'] for info in updated_mapping.values()]
    complete = sum(1 for n in all_trials if n >= 900)
    partial = sum(1 for n in all_trials if 100 <= n < 900)

    print(f"\nParticipant breakdown:")
    print(f"  Complete (>=900 trials): {complete}")
    print(f"  Partial (100-899 trials): {partial}")
    print(f"  Total: {len(updated_mapping)}")

    print("\n" + "=" * 80)
    print("NEXT STEP: Re-run the data pipeline")
    print("=" * 80)
    print("\nRun: python run_data_pipeline.py")
    print()

if __name__ == '__main__':
    main()
