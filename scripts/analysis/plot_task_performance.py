"""
Generate Task Performance Plots

Main script to load behavioral data and generate all performance visualization plots.

Usage:
    python scripts/analysis/plot_task_performance.py
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import DataParams
from scripts.analysis.behavioral_plots import (
    plot_accuracy_by_setsize,
    plot_learning_curves,
    plot_post_reversal_learning
)


def load_task_data() -> pd.DataFrame:
    """
    Load task trial data from CSV.

    Returns
    -------
    pd.DataFrame
        Trial-level task data
    """
    data_path = project_root / 'output' / 'task_trials_long.csv'

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            "Please run the data processing scripts first (01-04)."
        )

    print(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)

    print(f"Loaded {len(data)} trials from {data['sona_id'].nunique()} participants")

    return data


def filter_main_task_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to main task blocks only (exclude practice).

    Parameters
    ----------
    data : pd.DataFrame
        Full task data

    Returns
    -------
    pd.DataFrame
        Filtered data (main task blocks only)
    """
    # Filter to main task phase
    main_data = data[data['phase_type'] == 'main_task'].copy()

    print(f"Filtered to {len(main_data)} main task trials from {main_data['sona_id'].nunique()} participants")

    return main_data


def main():
    """
    Main function to generate all task performance plots.
    """
    print("=" * 80)
    print("RLWM TRAUMA ANALYSIS: TASK PERFORMANCE PLOTS")
    print("=" * 80)
    print()

    # Load data
    data = load_task_data()

    # Filter to main task
    main_data = filter_main_task_data(data)

    print()
    print("-" * 80)
    print("Generating plots...")
    print("-" * 80)
    print()

    # Plot 1: Accuracy by Set Size
    print("1. Creating accuracy by set size plot...")
    try:
        plot_accuracy_by_setsize(main_data, save=True, show=False)
        print("   ✓ Saved: figures/behavioral_analysis/accuracy_by_setsize.png")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print()

    # Plot 2: Learning Curves
    print("2. Creating learning curves plot...")
    try:
        plot_learning_curves(main_data, max_trials=20, save=True, show=False)
        print("   ✓ Saved: figures/behavioral_analysis/learning_curves_by_setsize.png")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print()

    # Plot 3: Post-Reversal Learning
    print("3. Creating post-reversal learning plot...")
    try:
        plot_post_reversal_learning(main_data, max_trials_post_reversal=15, save=True, show=False)
        print("   ✓ Saved: figures/behavioral_analysis/post_reversal_learning.png")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print()
    print("=" * 80)
    print("COMPLETE! All plots generated successfully.")
    print("=" * 80)
    print()
    print("Output directory: figures/behavioral_analysis/")
    print()


if __name__ == "__main__":
    main()
