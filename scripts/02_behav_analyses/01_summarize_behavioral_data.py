#!/usr/bin/env python
"""
05: Summarize Behavioral Data
=============================

Summarizes task performance, demographics, and scale metrics from processed data.
Performs deviation checks against expected data structure.

This script:
1. Loads processed data (derivatives only, no modifications to raw data)
2. Summarizes task performance, demographics, and scale metrics
3. Checks for deviations from expected data structure
4. Generates reports and visualizations

Inputs:
    - output/collated_participant_data.csv
    - output/summary_participant_metrics.csv
    - output/task_trials_long.csv
    - output/parsed_demographics.csv
    - output/parsed_survey1.csv (LEC-5)
    - output/parsed_survey2.csv (IES-R)

Outputs:
    - output/behavioral_summary/data_summary_report.txt

Usage:
    python scripts/02_behav_analyses/01_summarize_behavioral_data.py

Next Steps:
    - Run 02_visualize_task_performance.py for learning curves
    - Run 03_analyze_trauma_groups.py for trauma grouping analysis
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from config import TaskParams, DataParams, OUTPUT_DIR, FIGURES_DIR, AnalysisParams


def load_processed_data():
    """Load all processed data files (derivatives only)."""
    print("Loading processed data files...")

    data = {}

    # Load collated participant data
    collated_path = OUTPUT_DIR / 'collated_participant_data.csv'
    if collated_path.exists():
        data['collated'] = pd.read_csv(collated_path)
        print(f"  [OK] Loaded collated data: {len(data['collated'])} participants")
    else:
        print(f"  [WARN] Collated data not found: {collated_path}")
        data['collated'] = None

    # Load summary metrics
    summary_path = OUTPUT_DIR / 'summary_participant_metrics.csv'
    if summary_path.exists():
        data['summary'] = pd.read_csv(summary_path)
        print(f"  [OK] Loaded summary metrics: {len(data['summary'])} participants")
    else:
        print(f"  [WARN] Summary metrics not found: {summary_path}")
        data['summary'] = None

    # Load task trials
    trials_path = OUTPUT_DIR / 'task_trials_long.csv'
    if trials_path.exists():
        data['trials'] = pd.read_csv(trials_path)
        print(f"  [OK] Loaded task trials: {len(data['trials'])} trials")
    else:
        print(f"  [WARN] Task trials not found: {trials_path}")
        data['trials'] = None

    # Load demographics
    demo_path = OUTPUT_DIR / 'parsed_demographics.csv'
    if demo_path.exists():
        data['demographics'] = pd.read_csv(demo_path)
        print(f"  [OK] Loaded demographics: {len(data['demographics'])} participants")
    else:
        print(f"  [WARN] Demographics not found: {demo_path}")
        data['demographics'] = None

    # Load surveys
    survey1_path = OUTPUT_DIR / 'parsed_survey1.csv'
    if survey1_path.exists():
        data['survey1'] = pd.read_csv(survey1_path)
        print(f"  [OK] Loaded Survey 1 (LEC-5): {len(data['survey1'])} participants")
    else:
        print(f"  [WARN] Survey 1 not found: {survey1_path}")
        data['survey1'] = None

    survey2_path = OUTPUT_DIR / 'parsed_survey2.csv'
    if survey2_path.exists():
        data['survey2'] = pd.read_csv(survey2_path)
        print(f"  [OK] Loaded Survey 2 (IES-R): {len(data['survey2'])} participants")
    else:
        print(f"  [WARN] Survey 2 not found: {survey2_path}")
        data['survey2'] = None

    return data


def check_for_deviations(data):
    """
    Check for deviations from expected data structure.

    Parameters
    ----------
    data : dict
        Dictionary containing loaded data

    Returns
    -------
    dict
        Deviation report
    """
    print("\nChecking for deviations from expected structure...")

    deviations = {
        'errors': [],
        'warnings': [],
        'info': []
    }

    trials = data.get('trials')
    summary = data.get('summary')

    if trials is None:
        deviations['errors'].append("Task trials data not found")
        return deviations

    # Check number of participants
    n_participants = trials['sona_id'].nunique()
    deviations['info'].append(f"Number of participants: {n_participants}")

    # Check set sizes
    expected_set_sizes = set(TaskParams.SET_SIZES)
    actual_set_sizes = set(trials['set_size'].dropna().unique())
    if actual_set_sizes != expected_set_sizes:
        deviations['warnings'].append(
            f"Set sizes mismatch - Expected: {expected_set_sizes}, Found: {actual_set_sizes}"
        )
    else:
        deviations['info'].append(f"Set sizes match expected: {expected_set_sizes}")

    # Check blocks
    actual_blocks = sorted(trials['block'].dropna().unique())
    deviations['info'].append(f"Blocks found: {actual_blocks}")

    # Check if practice blocks are excluded (should start at block 3)
    if len(actual_blocks) > 0 and min(actual_blocks) < DataParams.MAIN_TASK_START_BLOCK:
        deviations['warnings'].append(
            f"Practice blocks detected (blocks < {DataParams.MAIN_TASK_START_BLOCK}). "
            f"Consider filtering to main task blocks only."
        )
    else:
        deviations['info'].append("Practice blocks properly excluded")

    # Check trials per block
    trials_per_block = trials.groupby(['sona_id', 'block']).size()
    deviations['info'].append(
        f"Trials per block - Mean: {trials_per_block.mean():.1f}, "
        f"Median: {trials_per_block.median():.0f}, "
        f"Range: [{trials_per_block.min()}, {trials_per_block.max()}]"
    )

    if trials_per_block.min() < TaskParams.TRIALS_PER_BLOCK_MIN:
        deviations['warnings'].append(
            f"Some blocks have < {TaskParams.TRIALS_PER_BLOCK_MIN} trials"
        )

    if trials_per_block.max() > TaskParams.TRIALS_PER_BLOCK_MAX:
        deviations['warnings'].append(
            f"Some blocks have > {TaskParams.TRIALS_PER_BLOCK_MAX} trials"
        )

    # Check for missing data
    if 'correct' in trials.columns:
        n_missing_correct = trials['correct'].isna().sum()
        if n_missing_correct > 0:
            deviations['warnings'].append(
                f"{n_missing_correct} trials missing 'correct' values"
            )

    # Check stimulus range
    if 'stimulus' in trials.columns:
        stimuli = trials['stimulus'].dropna().unique()
        deviations['info'].append(f"Stimuli: {sorted(stimuli)}")
        if max(stimuli) > TaskParams.MAX_STIMULI:
            deviations['errors'].append(
                f"Stimulus IDs exceed maximum ({TaskParams.MAX_STIMULI})"
            )

    # Check overall accuracy
    if summary is not None and 'accuracy_overall' in summary.columns:
        acc = summary['accuracy_overall'].values[0]
        deviations['info'].append(f"Overall accuracy: {acc:.3f}")

        if acc < 0.5:
            deviations['warnings'].append("Overall accuracy < 50% (chance level)")
        elif acc > 0.95:
            deviations['warnings'].append("Overall accuracy > 95% (ceiling effect)")

    return deviations


def print_deviation_report(deviations):
    """Print deviation report in formatted manner."""
    print("\n" + "=" * 80)
    print("DEVIATION REPORT")
    print("=" * 80)

    if deviations['errors']:
        print("\nERRORS:")
        for error in deviations['errors']:
            print(f"  [ERROR] {error}")

    if deviations['warnings']:
        print("\nWARNINGS:")
        for warning in deviations['warnings']:
            print(f"  [WARN] {warning}")

    if deviations['info']:
        print("\nINFO:")
        for info in deviations['info']:
            print(f"  [INFO] {info}")

    if not deviations['errors'] and not deviations['warnings']:
        print("\n  [OK] No errors or warnings detected")

    print("=" * 80)


def generate_summary_report(data, output_dir):
    """Generate text summary report."""
    print("\nGenerating summary report...")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("BEHAVIORAL DATA SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Participant info
    trials = data.get('trials')
    summary = data.get('summary')
    demographics = data.get('demographics')

    if trials is not None:
        n_participants = trials['sona_id'].nunique()
        n_trials = len(trials)

        report_lines.append(f"PARTICIPANTS: {n_participants}")
        report_lines.append(f"TOTAL TRIALS: {n_trials}")
        report_lines.append("")

    # Demographics
    if demographics is not None and len(demographics) > 0:
        report_lines.append("DEMOGRAPHICS:")
        if 'age_years' in demographics.columns:
            ages = demographics['age_years'].dropna()
            if len(ages) > 0:
                report_lines.append(f"  Age: {ages.values}")

        if 'gender' in demographics.columns:
            gender_counts = demographics['gender'].value_counts()
            report_lines.append(f"  Gender: {dict(gender_counts)}")

        if 'country' in demographics.columns:
            countries = demographics['country'].dropna().unique()
            report_lines.append(f"  Countries: {list(countries)}")

        report_lines.append("")

    # Task performance
    if summary is not None and len(summary) > 0:
        report_lines.append("TASK PERFORMANCE:")

        metrics = [
            ('Overall Accuracy', 'accuracy_overall'),
            ('Low Load Accuracy', 'accuracy_low_load'),
            ('High Load Accuracy', 'accuracy_high_load'),
            ('Set Size 2 Accuracy', 'accuracy_setsize_2'),
            ('Set Size 3 Accuracy', 'accuracy_setsize_3'),
            ('Set Size 5 Accuracy', 'accuracy_setsize_5'),
            ('Set Size 6 Accuracy', 'accuracy_setsize_6'),
            ('Mean RT (ms)', 'mean_rt_overall'),
            ('Learning Slope', 'learning_slope'),
            ('Learning Improvement', 'learning_improvement_early_to_late'),
        ]

        for label, col in metrics:
            if col in summary.columns:
                val = summary[col].values[0]
                if pd.notna(val):
                    report_lines.append(f"  {label:30s}: {val:.4f}")

        report_lines.append("")

    # Survey scores
    if summary is not None and len(summary) > 0:
        report_lines.append("SCALE METRICS:")

        scale_metrics = [
            ('LEC-5 Total Events', 'lec_total_events'),
            ('LEC-5 Personal Events', 'lec_personal_events'),
            ('IES-R Total', 'ies_total'),
            ('IES-R Intrusion', 'ies_intrusion'),
            ('IES-R Avoidance', 'ies_avoidance'),
            ('IES-R Hyperarousal', 'ies_hyperarousal'),
        ]

        for label, col in scale_metrics:
            if col in summary.columns:
                val = summary[col].values[0]
                if pd.notna(val):
                    report_lines.append(f"  {label:30s}: {val}")

        report_lines.append("")

    report_lines.append("=" * 80)

    # Print to console
    report_text = "\n".join(report_lines)
    print(report_text)

    # Save to file
    report_path = output_dir / 'data_summary_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)

    print(f"\n[OK] Saved report: {report_path}")

    return report_text


def main():
    # Create output directories
    output_dir = OUTPUT_DIR / 'behavioral_summary'
    figure_dir = FIGURES_DIR / 'behavioral_summary'
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BEHAVIORAL DATA SUMMARY AND VALIDATION")
    print("=" * 80)
    print()

    # Load data
    data = load_processed_data()

    # Check for deviations
    deviations = check_for_deviations(data)
    print_deviation_report(deviations)

    # Generate summary report
    report = generate_summary_report(data, output_dir)

    print("\n" + "=" * 80)
    print("SUMMARY COMPLETE")
    print("=" * 80)
    print(f"\nOutput saved to: {output_dir}")
    print("\nNext steps:")
    print("  - Run 02_visualize_task_performance.py to create stimulus-based learning curves")
    print("  - Run 03_analyze_trauma_groups.py for trauma grouping analysis")
    print("=" * 80)


if __name__ == '__main__':
    main()
