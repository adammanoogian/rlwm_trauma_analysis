"""
Create summary dataset with derived metrics and scale scores.

This script:
1. Loads parsed data files (demographics, survey1, survey2, task)
2. Calculates all derived metrics:
   - LEC-5 summary scores
   - IES-R total and subscale scores
   - All task performance metrics
3. Creates one-row-per-participant summary dataset
4. Saves the summary dataset

Usage:
    python scripts/04_create_summary_csv.py
"""

import os
import sys
import pandas as pd
import numpy as np

# Add project root to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import EXCLUDED_PARTICIPANTS, DataParams

# Add utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))

from scoring_functions import (
    score_less,
    score_ies_r,
    calculate_all_task_metrics
)


def main():
    print("=" * 60)
    print("STEP 4: Creating Summary Dataset with Derived Metrics")
    print("=" * 60)
    print()

    # Paths
    output_dir = 'output'
    demographics_path = os.path.join(output_dir, 'parsed_demographics.csv')
    survey1_path = os.path.join(output_dir, 'parsed_survey1.csv')
    survey2_path = os.path.join(output_dir, 'parsed_survey2.csv')
    task_path = os.path.join(output_dir, 'parsed_task_trials.csv')

    # Check if parsed files exist
    for path in [demographics_path, survey1_path, survey2_path, task_path]:
        if not os.path.exists(path):
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

    # ==========================================================================
    # DATA QUALITY FILTERING
    # ==========================================================================
    print("-" * 60)
    print("Checking data quality thresholds...")
    print(f"  MIN_BLOCKS: {DataParams.MIN_BLOCKS}")
    print(f"  MIN_TRIALS: {DataParams.MIN_TRIALS}")
    print()

    # Get participant block counts (main task only, blocks >= 3)
    main_task_trials = task_trials[task_trials['block'] >= DataParams.MAIN_TASK_START_BLOCK]
    block_counts = main_task_trials.groupby('sona_id')['block'].nunique()
    trial_counts = main_task_trials.groupby('sona_id').size()

    # Identify participants with insufficient data
    insufficient_blocks = block_counts[block_counts < DataParams.MIN_BLOCKS].index.tolist()
    insufficient_trials = trial_counts[trial_counts < DataParams.MIN_TRIALS].index.tolist()

    # Combine with existing exclusions
    quality_exclusions = set(insufficient_blocks) | set(insufficient_trials)
    new_exclusions = quality_exclusions - set(EXCLUDED_PARTICIPANTS)

    if new_exclusions:
        print(f"WARNING: {len(new_exclusions)} participants flagged for insufficient data:")
        for pid in sorted(new_exclusions):
            n_blocks = block_counts.get(pid, 0)
            n_trials = trial_counts.get(pid, 0)
            print(f"  {pid}: {n_trials} trials, {n_blocks} blocks")
        print()
        print("Consider adding these to EXCLUDED_PARTICIPANTS in config.py")
        print()

    # Report on existing exclusions found in data
    existing_exclusions_in_data = set(EXCLUDED_PARTICIPANTS) & set(task_trials['sona_id'].unique())
    if existing_exclusions_in_data:
        print(f"Existing exclusions found in data: {len(existing_exclusions_in_data)}")
        for pid in sorted(existing_exclusions_in_data):
            n_trials = trial_counts.get(pid, 0)
            n_blocks = block_counts.get(pid, 0)
            print(f"  {pid}: {n_trials} trials, {n_blocks} blocks (already excluded)")
        print()

    # Apply exclusions to task_trials for metrics calculation
    n_before = task_trials['sona_id'].nunique()
    task_trials_filtered = task_trials[~task_trials['sona_id'].isin(EXCLUDED_PARTICIPANTS)]
    n_after = task_trials_filtered['sona_id'].nunique()
    print(f"After exclusions: {n_after} participants (excluded {n_before - n_after})")
    print()

    # Calculate LESS scores
    print("-" * 60)
    print("Calculating LESS (Survey 1) summary scores...")
    survey1_scored = score_less(survey1)
    less_summary = survey1_scored[['sona_id', 'less_total_events', 'less_personal_events']]
    print(f"[OK] Calculated LESS scores for {len(less_summary)} participants")
    print(f"  Mean total events: {less_summary['less_total_events'].mean():.2f}")
    print(f"  Mean personal events: {less_summary['less_personal_events'].mean():.2f}")
    print()

    # Calculate IES-R scores
    print("-" * 60)
    print("Calculating IES-R (Survey 2) subscale scores...")
    survey2_scored = score_ies_r(survey2)
    ies_summary = survey2_scored[['sona_id', 'ies_total', 'ies_intrusion', 'ies_avoidance', 'ies_hyperarousal']]
    print(f"[OK] Calculated IES-R scores for {len(ies_summary)} participants")
    print(f"  Mean total score: {ies_summary['ies_total'].mean():.2f}")
    print(f"  Mean intrusion: {ies_summary['ies_intrusion'].mean():.2f}")
    print(f"  Mean avoidance: {ies_summary['ies_avoidance'].mean():.2f}")
    print(f"  Mean hyperarousal: {ies_summary['ies_hyperarousal'].mean():.2f}")
    print()

    # Calculate task metrics
    print("-" * 60)
    print("Calculating task performance metrics...")
    print("  (This may take a moment...)")
    print()

    task_metrics_list = []

    # Use original task_trials (not filtered) to calculate metrics for ALL participants
    # We'll add an exclusion flag separately
    for sona_id in task_trials['sona_id'].unique():
        participant_trials = task_trials[task_trials['sona_id'] == sona_id]
        metrics = calculate_all_task_metrics(participant_trials)
        metrics['sona_id'] = sona_id
        # Add exclusion flag for downstream scripts
        metrics['excluded'] = sona_id in EXCLUDED_PARTICIPANTS
        task_metrics_list.append(metrics)

    task_metrics = pd.DataFrame(task_metrics_list)
    print(f"[OK] Calculated task metrics for {len(task_metrics)} participants")
    print(f"  Metrics per participant: {len(task_metrics.columns) - 1}")
    print()

    # Display key task metrics
    print("  Key task performance statistics:")
    if 'accuracy_overall' in task_metrics.columns:
        print(f"    Mean overall accuracy: {task_metrics['accuracy_overall'].mean():.3f}")
    if 'mean_rt_overall' in task_metrics.columns:
        print(f"    Mean RT: {task_metrics['mean_rt_overall'].mean():.0f} ms")
    if 'accuracy_low_load' in task_metrics.columns:
        print(f"    Mean accuracy (low load): {task_metrics['accuracy_low_load'].mean():.3f}")
    if 'accuracy_high_load' in task_metrics.columns:
        print(f"    Mean accuracy (high load): {task_metrics['accuracy_high_load'].mean():.3f}")
    if 'learning_slope' in task_metrics.columns:
        print(f"    Mean learning slope: {task_metrics['learning_slope'].mean():.4f}")
    if 'n_reversals' in task_metrics.columns:
        print(f"    Mean reversals detected: {task_metrics['n_reversals'].mean():.1f}")
    print()

    # Merge all summary data
    print("-" * 60)
    print("Merging all summary data...")

    # Start with demographics
    summary = demographics.copy()
    print(f"Starting with demographics: {len(summary)} rows")

    # Merge LESS scores
    summary = summary.merge(less_summary, on='sona_id', how='left')
    print(f"After merging LESS scores: {len(summary)} rows, {len(summary.columns)} columns")

    # Merge IES-R scores
    summary = summary.merge(ies_summary, on='sona_id', how='left')
    print(f"After merging IES-R scores: {len(summary)} rows, {len(summary.columns)} columns")

    # Merge task metrics
    summary = summary.merge(task_metrics, on='sona_id', how='left')
    print(f"After merging task metrics: {len(summary)} rows, {len(summary.columns)} columns")
    print()

    # Organize columns
    print("-" * 60)
    print("Organizing columns...")

    # Define column order groups
    id_cols = ['sona_id']

    demographic_cols = [col for col in summary.columns if col in [
        'age_years', 'country', 'primary_language', 'gender', 'education',
        'relationship_status', 'living_arrangement', 'screen_time'
    ]]

    lec_cols = [col for col in summary.columns if col.startswith('lec_')]
    ies_cols = [col for col in summary.columns if col.startswith('ies_')]

    # Task metrics (all remaining columns)
    task_metric_cols = [col for col in summary.columns
                        if col not in id_cols + demographic_cols + lec_cols + ies_cols]

    # Reorder columns
    ordered_cols = id_cols + demographic_cols + lec_cols + ies_cols + task_metric_cols
    summary = summary[ordered_cols]

    print(f"Column organization:")
    print(f"  - ID: 1 column")
    print(f"  - Demographics: {len(demographic_cols)} columns")
    print(f"  - LEC-5 summary: {len(lec_cols)} columns")
    print(f"  - IES-R summary: {len(ies_cols)} columns")
    print(f"  - Task metrics: {len(task_metric_cols)} columns")
    print(f"  - TOTAL: {len(summary.columns)} columns")
    print()

    # Data quality summary
    print("-" * 60)
    print("Data Quality Summary:")
    print(f"  Total participants: {len(summary)}")

    # Check completeness
    participants_with_all_data = summary.dropna(subset=lec_cols + ies_cols + ['accuracy_overall']).shape[0]
    print(f"  Participants with complete survey & task data: {participants_with_all_data}")

    # Missing data by section
    if len(lec_cols) > 0:
        missing_lec = summary[lec_cols].isna().any(axis=1).sum()
        print(f"  Missing LEC-5 data: {missing_lec} participants")

    if len(ies_cols) > 0:
        missing_ies = summary[ies_cols].isna().any(axis=1).sum()
        print(f"  Missing IES-R data: {missing_ies} participants")

    if 'accuracy_overall' in summary.columns:
        missing_task = summary['accuracy_overall'].isna().sum()
        print(f"  Missing task data: {missing_task} participants")

    print()

    # Save summary data
    output_path = os.path.join(output_dir, 'summary_participant_metrics.csv')
    summary.to_csv(output_path, index=False)
    print("-" * 60)
    print(f"[OK] SAVED: {output_path}")
    print(f"  {len(summary)} participants × {len(summary.columns)} columns")
    print()

    # Display sample
    print("-" * 60)
    print("Sample of summary data (first 3 participants, key columns):")
    display_cols = [col for col in [
        'sona_id', 'age_years', 'gender',
        'less_total_events', 'ies_total',
        'accuracy_overall', 'mean_rt_overall',
        'accuracy_low_load', 'accuracy_high_load'
    ] if col in summary.columns]

    print(summary[display_cols].head(3).to_string())
    print()

    print("=" * 60)
    print("STEP 4 COMPLETE: Summary dataset created successfully")
    print("=" * 60)
    print()
    print("All pipeline steps complete! Final outputs:")
    print(f"  1. {output_dir}/collated_participant_data.csv")
    print(f"  2. {output_dir}/task_trials_long.csv")
    print(f"  3. {output_dir}/summary_participant_metrics.csv")
    print()


if __name__ == '__main__':
    main()
