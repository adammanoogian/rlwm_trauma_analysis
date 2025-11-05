"""
Parse raw jsPsych experiment data and extract demographics, surveys, and task data.

This script:
1. Loads all raw CSV files from the data/ directory
2. Extracts and parses demographics data
3. Extracts and parses Survey 1 (LEC-5) with multi-select responses
4. Extracts and parses Survey 2 (IES-R) from JSON
5. Extracts task trial data
6. Saves intermediate cleaned data files

Usage:
    python scripts/01_parse_raw_data.py
"""

import os
import sys
import pandas as pd
import numpy as np
import glob

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from data_cleaning import (
    extract_demographics,
    extract_survey1_data,
    extract_survey2_data,
    extract_task_trials,
    validate_data
)


def main():
    print("=" * 60)
    print("STEP 1: Parsing Raw jsPsych Data")
    print("=" * 60)
    print()

    # Paths
    data_dir = 'data'
    output_dir = 'output'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Check if data directory exists
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found at {data_dir}")
        print("Please ensure the data/ directory exists with CSV files.")
        sys.exit(1)

    # Find all CSV files in the data directory
    csv_files = glob.glob(os.path.join(data_dir, '*.csv'))
    
    if len(csv_files) == 0:
        print(f"ERROR: No CSV files found in {data_dir}/")
        print("Please add CSV files to the data/ directory.")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s) in {data_dir}/")
    for csv_file in csv_files:
        print(f"  - {os.path.basename(csv_file)}")
    print()

    # Load and concatenate all CSV files
    print("Loading and combining all CSV files...")
    dfs = []
    for csv_file in csv_files:
        try:
            temp_df = pd.read_csv(csv_file)
            dfs.append(temp_df)
            print(f"  Loaded {os.path.basename(csv_file)}: {len(temp_df)} rows")
        except Exception as e:
            print(f"  WARNING: Could not load {os.path.basename(csv_file)}: {e}")
    
    if len(dfs) == 0:
        print("ERROR: Could not load any CSV files successfully.")
        sys.exit(1)
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"\nCombined dataset: {len(df)} rows, {len(df.columns)} columns")
    print()

    # Display basic info
    if 'sona_id' in df.columns:
        n_participants = df['sona_id'].nunique()
        print(f"Number of unique participants: {n_participants}")

    if 'section' in df.columns:
        print(f"\nSections found: {df['section'].unique()}")
    print()

    # Extract demographics
    print("-" * 60)
    print("Extracting demographics...")
    demographics = extract_demographics(df)
    print(f"Demographics extracted for {len(demographics)} participants")
    print(f"Columns: {list(demographics.columns)}")
    print()

    # Extract Survey 1
    print("-" * 60)
    print("Extracting and parsing Survey 1 (LEC-5)...")
    survey1 = extract_survey1_data(df)
    print(f"Survey 1 extracted for {len(survey1)} participants")
    print(f"Columns: {len(survey1.columns)} columns")
    print()

    # Extract Survey 2
    print("-" * 60)
    print("Extracting and parsing Survey 2 (IES-R)...")
    survey2 = extract_survey2_data(df)
    print(f"Survey 2 extracted for {len(survey2)} participants")
    print(f"Columns: {list(survey2.columns)}")
    print()

    # Extract task trials
    print("-" * 60)
    print("Extracting task trials...")
    task_trials = extract_task_trials(df)
    print(f"Task trials extracted: {len(task_trials)} trials")

    if len(task_trials) > 0:
        n_task_participants = task_trials['sona_id'].nunique()
        print(f"Participants with task data: {n_task_participants}")

        if 'block' in task_trials.columns:
            print(f"Blocks: {sorted(task_trials['block'].unique())}")

        if 'set_size' in task_trials.columns:
            print(f"Set sizes: {sorted(task_trials['set_size'].unique())}")
    print()

    # Validate data
    print("-" * 60)
    print("Validating data...")
    validation = validate_data(demographics)
    print(f"Total participants: {validation['n_participants']}")

    if validation['warnings']:
        print("\nWarnings:")
        for warning in validation['warnings']:
            print(f"  - {warning}")
    else:
        print("No validation issues found")
    print()

    # Save intermediate files
    print("-" * 60)
    print("Saving intermediate cleaned data files...")

    demographics.to_csv(os.path.join(output_dir, 'parsed_demographics.csv'), index=False)
    print(f"[SAVED] {output_dir}/parsed_demographics.csv")

    survey1.to_csv(os.path.join(output_dir, 'parsed_survey1.csv'), index=False)
    print(f"[SAVED] {output_dir}/parsed_survey1.csv")

    survey2.to_csv(os.path.join(output_dir, 'parsed_survey2.csv'), index=False)
    print(f"[SAVED] {output_dir}/parsed_survey2.csv")

    task_trials.to_csv(os.path.join(output_dir, 'parsed_task_trials.csv'), index=False)
    print(f"[SAVED] {output_dir}/parsed_task_trials.csv")

    print()
    print("=" * 60)
    print("STEP 1 COMPLETE: Raw data parsing finished successfully")
    print("=" * 60)
    print()
    print("Next step: Run 02_create_collated_csv.py to create the collated dataset")
    print()


if __name__ == '__main__':
    main()
