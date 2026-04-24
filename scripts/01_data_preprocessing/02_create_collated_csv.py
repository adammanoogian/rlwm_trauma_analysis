"""
Create collated participant dataset with all individual responses.

This script:
1. Loads parsed data files (demographics, survey1, survey2, task) from data/interim/
2. Merges all data into a wide-format dataset (one row per participant)
3. Includes all demographic variables, survey responses, and task summary stats
4. Saves the collated dataset to data/interim/collated_participant_data.csv

Usage:
    python scripts/01_data_preprocessing/02_create_collated_csv.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Add utils to path
sys.path.append(str(Path(__file__).resolve().parents[1] / 'utils'))

from scoring import calculate_all_task_metrics

# Import config CCDS constants
# (CCDS constants landed in plan 31-01; physical files moved in plan 31-02)
from config import INTERIM_DIR, DataParams


def main():
    print("=" * 60)
    print("STEP 2: Creating Collated Participant Dataset")
    print("=" * 60)
    print()

    # Paths — CCDS interim tier (parsed products are gitignored PII)
    demographics_path = DataParams.PARSED_DEMOGRAPHICS
    survey1_path = DataParams.PARSED_SURVEY1
    survey2_path = DataParams.PARSED_SURVEY2
    task_path = DataParams.PARSED_TASK_TRIALS

    # Check if parsed files exist
    for path in [demographics_path, survey1_path, survey2_path, task_path]:
        if not Path(path).exists():
            print(f"ERROR: Required file not found: {path}")
            print("Please run 01_parse_raw_data.py first")
            sys.exit(1)

    print("Loading parsed data files...")

    # Load parsed data
    demographics = pd.read_csv(demographics_path)
    print(f"[OK] Loaded demographics: {len(demographics)} participants")

    survey1 = pd.read_csv(survey1_path)
    print(f"[OK] Loaded Survey 1: {len(survey1)} participants")

    survey2 = pd.read_csv(survey2_path)
    print(f"[OK] Loaded Survey 2: {len(survey2)} participants")

    task_trials = pd.read_csv(task_path)
    print(f"[OK] Loaded task trials: {len(task_trials)} trials")
    print()

    # Calculate task summary metrics per participant
    print("-" * 60)
    print("Calculating task performance metrics...")

    task_metrics_list = []

    for sona_id in task_trials['sona_id'].unique():
        participant_trials = task_trials[task_trials['sona_id'] == sona_id]
        metrics = calculate_all_task_metrics(participant_trials)
        metrics['sona_id'] = sona_id
        task_metrics_list.append(metrics)

    task_metrics = pd.DataFrame(task_metrics_list)
    print(f"[OK] Calculated metrics for {len(task_metrics)} participants")
    print(f"  Metrics calculated: {len(task_metrics.columns) - 1}")  # -1 for sona_id
    print()

    # Merge all data
    print("-" * 60)
    print("Merging all data sources...")

    # Start with demographics
    collated = demographics.copy()
    print(f"Starting with demographics: {len(collated)} rows")

    # Merge Survey 1
    collated = collated.merge(survey1, on='sona_id', how='outer')
    print(f"After merging Survey 1: {len(collated)} rows, {len(collated.columns)} columns")

    # Merge Survey 2
    collated = collated.merge(survey2, on='sona_id', how='outer')
    print(f"After merging Survey 2: {len(collated)} rows, {len(collated.columns)} columns")

    # Merge task metrics
    collated = collated.merge(task_metrics, on='sona_id', how='left')
    print(f"After merging task metrics: {len(collated)} rows, {len(collated.columns)} columns")
    print()

    # Organize column order
    print("-" * 60)
    print("Organizing columns...")

    # Define column order groups
    id_cols = ['sona_id']

    demographic_cols = [col for col in collated.columns if col in [
        'age_years', 'country', 'primary_language', 'gender', 'education',
        'relationship_status', 'living_arrangement', 'screen_time'
    ]]

    survey1_cols = [col for col in collated.columns if col.startswith('s1_item')]

    survey2_cols = [col for col in collated.columns if col.startswith('s2_item') and not col.endswith(('_any_exposure', '_personal'))]

    task_cols = [col for col in collated.columns if col not in id_cols + demographic_cols + survey1_cols + survey2_cols]

    # Reorder columns
    ordered_cols = id_cols + demographic_cols + survey1_cols + survey2_cols + task_cols
    collated = collated[ordered_cols]

    print(f"Column organization:")
    print(f"  - ID: 1 column")
    print(f"  - Demographics: {len(demographic_cols)} columns")
    print(f"  - Survey 1 (LEC-5): {len(survey1_cols)} columns")
    print(f"  - Survey 2 (IES-R): {len(survey2_cols)} columns")
    print(f"  - Task metrics: {len(task_cols)} columns")
    print(f"  - TOTAL: {len(collated.columns)} columns")
    print()

    # Data summary
    print("-" * 60)
    print("Data Summary:")
    print(f"  Total participants: {len(collated)}")
    print(f"  Complete cases (no missing data): {collated.dropna().shape[0]}")
    print(f"  Missing data percentage: {collated.isna().sum().sum() / (collated.shape[0] * collated.shape[1]) * 100:.1f}%")
    print()

    # Save collated data — CCDS interim tier (gitignored PII)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DataParams.COLLATED_DATA
    collated.to_csv(output_path, index=False)
    print("-" * 60)
    print(f"[OK] SAVED: {output_path}")
    print()

    # Display sample
    print("-" * 60)
    print("Sample of collated data (first 3 rows, first 10 columns):")
    print(collated.iloc[:3, :10].to_string())
    print()

    print("=" * 60)
    print("STEP 2 COMPLETE: Collated dataset created successfully")
    print("=" * 60)
    print()
    print("Next step: Run 03_create_task_trials_csv.py to create the task trials dataset")
    print()


if __name__ == '__main__':
    main()
