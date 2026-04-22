"""
Create long-format task trials dataset.

This script:
1. Loads parsed task trial data
2. Cleans and formats trial-level data
3. Adds useful derived columns (e.g., trial_in_experiment)
4. Saves clean trial-by-trial dataset in long format

Usage:
    python scripts/03_create_task_trials_csv.py
"""

import os
import sys
import pandas as pd
import numpy as np


def main():
    print("=" * 60)
    print("STEP 3: Creating Task Trials Dataset (Long Format)")
    print("=" * 60)
    print()

    # Paths
    output_dir = 'output'
    task_path = os.path.join(output_dir, 'parsed_task_trials.csv')

    # Check if parsed file exists
    if not os.path.exists(task_path):
        print(f"ERROR: Required file not found: {task_path}")
        print("Please run 01_parse_raw_data.py first")
        sys.exit(1)

    print(f"Loading task trials from {task_path}...")
    task_trials = pd.read_csv(task_path)
    print(f"[OK] Loaded {len(task_trials)} trials")
    print()

    # Basic info
    print("-" * 60)
    print("Task Trials Summary:")
    print(f"  Total trials: {len(task_trials)}")
    print(f"  Participants: {task_trials['sona_id'].nunique()}")

    if 'block' in task_trials.columns:
        print(f"  Blocks: {sorted(task_trials['block'].unique())}")
        trials_per_participant = len(task_trials) / task_trials['sona_id'].nunique()
        print(f"  Average trials per participant: {trials_per_participant:.1f}")

    if 'set_size' in task_trials.columns:
        print(f"  Set sizes: {sorted(task_trials['set_size'].unique())}")

    if 'load_condition' in task_trials.columns:
        print(f"  Load conditions: {sorted(task_trials['load_condition'].unique())}")

    print()

    # Add derived columns
    print("-" * 60)
    print("Adding derived columns...")

    # Sort by participant and trial order
    task_trials = task_trials.sort_values(['sona_id', 'trial_index']).reset_index(drop=True)

    # Add trial number within experiment (across all blocks)
    task_trials['trial_in_experiment'] = task_trials.groupby('sona_id').cumcount() + 1

    # Add trial number within block
    if 'block' in task_trials.columns and 'trial' not in task_trials.columns:
        task_trials['trial_in_block'] = task_trials.groupby(['sona_id', 'block']).cumcount() + 1
    elif 'trial' in task_trials.columns:
        task_trials['trial_in_block'] = task_trials['trial']

    # Add binary timeout indicator
    if 'key_press' in task_trials.columns:
        task_trials['timeout'] = task_trials['key_press'].isna().astype(int)

    # Add response time bins for analysis
    if 'rt' in task_trials.columns:
        # Categorize RT into fast/medium/slow
        task_trials['rt_category'] = pd.cut(
            task_trials['rt'],
            bins=[0, 500, 1000, np.inf],
            labels=['fast', 'medium', 'slow'],
            include_lowest=True
        )

    print("[OK] Added derived columns:")
    print("  - trial_in_experiment (cumulative trial number)")
    print("  - trial_in_block (trial within current block)")
    if 'key_press' in task_trials.columns:
        print("  - timeout (1 if no response, 0 if responded)")
    if 'rt' in task_trials.columns:
        print("  - rt_category (fast/medium/slow)")
    print()

    # Organize columns
    print("-" * 60)
    print("Organizing columns...")

    # Define preferred column order
    priority_cols = [
        'sona_id',
        'trial_in_experiment',
        'block',
        'trial_in_block',
        'set_size',
        'load_condition',
        'stimulus',
        'key_press',
        'key_answer',
        'correct',
        'rt',
        'rt_category',
        'timeout',
        'time_elapsed',
        'trial_index',
        'phase_type',
        'set',
        'reversal_crit',
        'counter'
    ]

    # Order columns (keep priority ones first, then others)
    available_priority = [col for col in priority_cols if col in task_trials.columns]
    other_cols = [col for col in task_trials.columns if col not in available_priority]
    task_trials = task_trials[available_priority + other_cols]

    print(f"[OK] Organized {len(task_trials.columns)} columns")
    print()

    # Performance statistics
    print("-" * 60)
    print("Performance Statistics:")

    if 'correct' in task_trials.columns:
        overall_acc = task_trials['correct'].mean() * 100
        print(f"  Overall accuracy: {overall_acc:.1f}%")

    if 'timeout' in task_trials.columns:
        timeout_rate = task_trials['timeout'].mean() * 100
        print(f"  Timeout rate: {timeout_rate:.1f}%")

    if 'rt' in task_trials.columns:
        completed_trials = task_trials[task_trials['key_press'].notna()]
        mean_rt = completed_trials['rt'].mean()
        median_rt = completed_trials['rt'].median()
        print(f"  Mean RT (completed trials): {mean_rt:.0f} ms")
        print(f"  Median RT (completed trials): {median_rt:.0f} ms")

    if 'load_condition' in task_trials.columns and 'correct' in task_trials.columns:
        print("\n  Accuracy by load condition:")
        for load in sorted(task_trials['load_condition'].unique()):
            load_acc = task_trials[task_trials['load_condition'] == load]['correct'].mean() * 100
            print(f"    {load}: {load_acc:.1f}%")

    if 'set_size' in task_trials.columns and 'correct' in task_trials.columns:
        print("\n  Accuracy by set size:")
        for ss in sorted(task_trials['set_size'].unique()):
            ss_acc = task_trials[task_trials['set_size'] == ss]['correct'].mean() * 100
            print(f"    Set size {ss}: {ss_acc:.1f}%")

    print()

    # Save task trials
    output_path = os.path.join(output_dir, 'task_trials_long.csv')
    task_trials.to_csv(output_path, index=False)
    print("-" * 60)
    print(f"[OK] SAVED: {output_path}")
    print(f"  {len(task_trials)} trials × {len(task_trials.columns)} columns")
    print()

    # Display sample
    print("-" * 60)
    print("Sample of task trials data (first 5 trials, key columns):")
    display_cols = [col for col in ['sona_id', 'trial_in_experiment', 'block', 'trial_in_block',
                                     'set_size', 'correct', 'rt', 'timeout'] if col in task_trials.columns]
    print(task_trials[display_cols].head().to_string())
    print()

    print("=" * 60)
    print("STEP 3 COMPLETE: Task trials dataset created successfully")
    print("=" * 60)
    print()
    print("Next step: Run 04_create_summary_csv.py to create the summary dataset")
    print()


if __name__ == '__main__':
    main()
