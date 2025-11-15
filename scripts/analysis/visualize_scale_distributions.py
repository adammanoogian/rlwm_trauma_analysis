"""
Visualize distributions of scale metrics (LEC-5, IES-R).

Note: With N=1 participant, this creates a template visualization
that will be more informative with additional participants.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from config import OUTPUT_DIR, FIGURES_DIR, AnalysisParams


def plot_scale_distributions(summary_df, save_dir):
    """
    Plot distributions of scale metrics.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary metrics data
    save_dir : Path
        Directory to save figures
    """
    n_participants = len(summary_df)

    # Extract scale columns
    lec_cols = ['lec_total_events', 'lec_personal_events', 'lec_sum_exposures']
    ies_cols = ['ies_total', 'ies_intrusion', 'ies_avoidance', 'ies_hyperarousal']

    # Check which columns exist
    available_lec = [c for c in lec_cols if c in summary_df.columns]
    available_ies = [c for c in ies_cols if c in summary_df.columns]

    if not available_lec and not available_ies:
        print("  [WARN] No scale metrics found in data")
        return

    # Create figure
    n_plots = len(available_lec) + len(available_ies)
    n_cols = min(3, n_plots)
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    plot_idx = 0

    # Plot LEC-5 metrics
    for col in available_lec:
        ax = axes[plot_idx]
        data = summary_df[col].dropna()

        if n_participants == 1:
            # Single participant - show value as bar
            ax.bar([col.replace('lec_', '').replace('_', ' ').title()],
                   [data.values[0]],
                   color='steelblue',
                   alpha=0.7)
            ax.set_ylabel('Count/Score', fontweight='bold')
            ax.set_title(f'LEC-5: {col.replace("lec_", "").replace("_", " ").title()}',
                        fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Add value label
            ax.text(0, data.values[0] + 0.5, f'{data.values[0]:.1f}',
                   ha='center', fontweight='bold')

        else:
            # Multiple participants - histogram
            ax.hist(data, bins=15, color='steelblue', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Score', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(f'LEC-5: {col.replace("lec_", "").replace("_", " ").title()}',
                        fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Add stats
            mean_val = data.mean()
            std_val = data.std()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.2f}')
            ax.legend()

        plot_idx += 1

    # Plot IES-R metrics
    for col in available_ies:
        ax = axes[plot_idx]
        data = summary_df[col].dropna()

        if n_participants == 1:
            # Single participant - show value as bar
            ax.bar([col.replace('ies_', '').title()],
                   [data.values[0]],
                   color='coral',
                   alpha=0.7)
            ax.set_ylabel('Score', fontweight='bold')
            ax.set_title(f'IES-R: {col.replace("ies_", "").title()}',
                        fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Add value label
            ax.text(0, data.values[0] + 0.5, f'{data.values[0]:.1f}',
                   ha='center', fontweight='bold')

        else:
            # Multiple participants - histogram
            ax.hist(data, bins=15, color='coral', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Score', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(f'IES-R: {col.replace("ies_", "").title()}',
                        fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Add stats
            mean_val = data.mean()
            std_val = data.std()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.2f}')
            ax.legend()

        plot_idx += 1

    # Hide unused axes
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Scale Metric Distributions (N={n_participants})', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Save
    save_path = save_dir / 'scale_distributions.png'
    plt.savefig(save_path, dpi=AnalysisParams.FIG_DPI, format=AnalysisParams.FIG_FORMAT, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()


def plot_task_performance_distributions(summary_df, save_dir):
    """
    Plot distributions of task performance metrics.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary metrics data
    save_dir : Path
        Directory to save figures
    """
    n_participants = len(summary_df)

    # Performance metrics to plot
    perf_metrics = {
        'Accuracy Overall': 'accuracy_overall',
        'Accuracy Low Load': 'accuracy_low_load',
        'Accuracy High Load': 'accuracy_high_load',
        'Mean RT (ms)': 'mean_rt_overall',
        'Learning Slope': 'learning_slope',
        'Learning Improvement': 'learning_improvement_early_to_late'
    }

    # Filter to available columns
    available_metrics = {k: v for k, v in perf_metrics.items() if v in summary_df.columns}

    if not available_metrics:
        print("  [WARN] No performance metrics found")
        return

    # Create figure
    n_plots = len(available_metrics)
    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (label, col) in enumerate(available_metrics.items()):
        ax = axes[idx]
        data = summary_df[col].dropna()

        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label, fontweight='bold')
            continue

        if n_participants == 1:
            # Single participant - show value as bar
            ax.bar([label], [data.values[0]], color='mediumseagreen', alpha=0.7)
            ax.set_ylabel('Value', fontweight='bold')
            ax.set_title(label, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Add value label
            ax.text(0, data.values[0] + abs(data.values[0])*0.05, f'{data.values[0]:.3f}',
                   ha='center', fontweight='bold')

        else:
            # Multiple participants - histogram
            ax.hist(data, bins=15, color='mediumseagreen', alpha=0.7, edgecolor='black')
            ax.set_xlabel('Value', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(label, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

            # Add stats
            mean_val = data.mean()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.3f}')
            ax.legend()

    # Hide unused axes
    for idx in range(len(available_metrics), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Task Performance Distributions (N={n_participants})', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Save
    save_path = save_dir / 'performance_distributions.png'
    plt.savefig(save_path, dpi=AnalysisParams.FIG_DPI, format=AnalysisParams.FIG_FORMAT, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize scale metric distributions'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='output/summary_participant_metrics.csv',
        help='Path to summary metrics data'
    )

    args = parser.parse_args()

    # Paths
    data_path = project_root / args.data
    figure_dir = FIGURES_DIR / 'behavioral_summary'

    figure_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SCALE DISTRIBUTION VISUALIZATION")
    print("=" * 80)

    # Load data
    print(f"\nLoading summary metrics: {data_path}")
    summary_df = pd.read_csv(data_path)
    print(f"  Loaded {len(summary_df)} participants")

    if len(summary_df) == 1:
        print("\n  [NOTE] Only 1 participant - visualizations will show individual values")
        print("          Distributions will be more informative with additional participants")

    # Create visualizations
    print(f"\nCreating visualizations...")

    plot_scale_distributions(summary_df, figure_dir)
    plot_task_performance_distributions(summary_df, figure_dir)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nFigures saved to: {figure_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
