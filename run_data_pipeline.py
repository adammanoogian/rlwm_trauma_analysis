#!/usr/bin/env python3
"""
Complete Data Processing and Analysis Pipeline
Runs all data parsing, cleaning, and behavioral analysis (excludes model fitting/simulation)

By default, syncs data from experiment folder before processing.
Use --no-sync to skip data synchronization.
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_command(step_num, total_steps, description, command):
    """Run a command and handle errors."""
    print(f"\n[{step_num}/{total_steps}] {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Step {step_num} failed!")
        print(f"Command: {command}")
        return False

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Complete data processing and analysis pipeline for RLWM trauma study'
    )
    parser.add_argument(
        '--no-sync',
        action='store_true',
        help='Skip syncing data from experiment folder (default: sync is enabled)'
    )
    args = parser.parse_args()

    print("=" * 50)
    print("RLWM Trauma Analysis - Data Pipeline")
    print("=" * 50)

    # Define pipeline steps
    steps = []

    # Add sync step (unless --no-sync flag is set)
    if not args.no_sync:
        steps.append(("Syncing data from experiment folder", "python sync_experiment_data.py"))
    else:
        print("\nSkipping data sync (--no-sync flag set)")

    # Add remaining pipeline steps
    steps.extend([
        ("Updating participant ID mapping", "python scripts/update_participant_mapping.py"),
        ("Parsing raw jsPsych data", "python scripts/01_parse_raw_data.py"),
        ("Creating collated participant data", "python scripts/02_create_collated_csv.py"),
        ("Creating task trials CSV", "python scripts/03_create_task_trials_csv.py"),
        ("Creating summary CSV", "python scripts/04_create_summary_csv.py"),
        ("Parsing all participants (including partial data)", "python scripts/parse_all_participants.py"),
        ("Generating human performance visualizations", "python scripts/analysis/visualize_human_performance.py --data output/task_trials_long_all_participants.csv"),
        ("Generating scale distributions", "python scripts/analysis/visualize_scale_distributions.py"),
        ("Generating scale correlations", "python scripts/analysis/visualize_scale_correlations.py"),
        ("Creating summary report", "python scripts/analysis/summarize_behavioral_data.py"),
    ])

    total_steps = len(steps)

    # Run each step
    for i, (description, command) in enumerate(steps, 1):
        success = run_command(i, total_steps, description, command)
        if not success:
            print("\n" + "=" * 50)
            print("Pipeline FAILED at step", i)
            print("=" * 50)
            sys.exit(1)

    # Success message
    print("\n" + "=" * 50)
    print("Pipeline Complete!")
    print("=" * 50)
    print("\nOutputs generated in:")
    print("  - output/*.csv")
    print("  - figures/behavioral_summary/*.png")
    print("  - output/behavioral_summary/data_summary_report.txt")

if __name__ == "__main__":
    main()
