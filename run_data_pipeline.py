#!/usr/bin/env python3
"""
Complete Data Processing and Analysis Pipeline (Steps 01-08)

Runs all data parsing, cleaning, and behavioral analysis.
Model fitting (steps 09-16) is handled separately via cluster scripts.

By default, syncs data from the experiment folder before processing.
Use --no-sync to skip data synchronization.

Usage:
    python run_data_pipeline.py              # Full pipeline with data sync
    python run_data_pipeline.py --no-sync    # Skip data sync (data already present)
    python run_data_pipeline.py --from 5     # Resume from step 5
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def run_command(step_num, total_steps, description, command):
    """Run a command and handle errors."""
    print(f"\n{'=' * 60}")
    print(f"[{step_num}/{total_steps}] {description}")
    print(f"  Command: {command}")
    print("=" * 60)
    try:
        subprocess.run(command, shell=True, check=True, capture_output=False, text=True)
        print(f"  -> Step {step_num} complete.")
        return True
    except subprocess.CalledProcessError:
        print(f"  -> ERROR: Step {step_num} failed!")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Complete data processing and analysis pipeline for RLWM trauma study"
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip syncing data from experiment folder",
    )
    parser.add_argument(
        "--from",
        type=int,
        default=1,
        dest="from_step",
        help="Resume from this step number (default: 1)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("RLWM Trauma Analysis — Data Pipeline (Steps 01-08)")
    print("=" * 60)

    # ----------------------------------------------------------------
    # Define pipeline steps
    # ----------------------------------------------------------------
    steps = []

    # Step 0 (optional): Sync data from experiment folder
    if not args.no_sync:
        steps.append((
            "Syncing data from experiment folder",
            "python scripts/utils/sync_experiment_data.py",
        ))
    else:
        print("\nSkipping data sync (--no-sync flag set)")

    # Steps 01-04: Data Processing
    steps.extend([
        (
            "01 — Parsing raw jsPsych data",
            "python scripts/data_processing/01_parse_raw_data.py",
        ),
        (
            "02 — Creating collated participant data",
            "python scripts/data_processing/02_create_collated_csv.py",
        ),
        (
            "03 — Creating task trials CSV",
            "python scripts/data_processing/03_create_task_trials_csv.py",
        ),
        (
            "04 — Creating summary CSV",
            "python scripts/data_processing/04_create_summary_csv.py",
        ),
    ])

    # Steps 05-08: Behavioral Analysis
    steps.extend([
        (
            "05 — Summarizing behavioral data",
            "python scripts/behavioral/05_summarize_behavioral_data.py",
        ),
        (
            "06 — Visualizing task performance",
            "python scripts/behavioral/06_visualize_task_performance.py",
        ),
        (
            "07 — Analyzing trauma groups",
            "python scripts/behavioral/07_analyze_trauma_groups.py",
        ),
        (
            "08 — Running statistical analyses",
            "python scripts/behavioral/08_run_statistical_analyses.py",
        ),
    ])

    total_steps = len(steps)

    # ----------------------------------------------------------------
    # Run each step
    # ----------------------------------------------------------------
    for i, (description, command) in enumerate(steps, 1):
        if i < args.from_step:
            print(f"\n  [Skipping step {i}: {description}]")
            continue

        success = run_command(i, total_steps, description, command)
        if not success:
            print(f"\nPipeline FAILED at step {i}.")
            print(f"Fix the issue and resume with: python run_data_pipeline.py --from {i}")
            sys.exit(1)

    # ----------------------------------------------------------------
    # Success
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Pipeline Complete! (Steps 01-08)")
    print("=" * 60)
    print("\nOutputs:")
    print("  output/task_trials_long.csv          — Main task trial data")
    print("  output/summary_participant_metrics.csv — Participant summaries")
    print("  output/trauma_groups/                 — Group assignments")
    print("  output/descriptives/                  — Descriptive tables")
    print("  output/statistical_analyses/          — ANOVA & regression")
    print("  figures/behavioral_summary/           — Performance plots")
    print("  figures/trauma_groups/                — Group visualizations")
    print()
    print("Next steps (run on cluster):")
    print("  sbatch cluster/13_full_pipeline.slurm   # Steps 03-16")
    print("  # or individually:")
    print("  sbatch cluster/12_mle_gpu.slurm         # MLE fitting only")


if __name__ == "__main__":
    main()
