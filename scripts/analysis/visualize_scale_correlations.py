"""
Visualize correlations between scale metrics and task performance.

Note: With N=1 participant, correlations cannot be computed.
This creates a template visualization that will be populated
when additional participants are added.
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


def plot_scale_correlations(summary_df, save_dir):
    """
    Plot correlation matrix of scale metrics and task performance.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary metrics data
    save_dir : Path
        Directory to save figures
    """
    n_participants = len(summary_df)

    if n_participants < 2:
        print("  [WARN] Need at least 2 participants to compute correlations")
        print("          Creating template visualization...")

        # Create template figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Template text
        ax.text(0.5, 0.5,
               f'Correlation Matrix\n\n'
               f'Current: N={n_participants} participant\n'
               f'Required: N>=2 participants\n\n'
               f'This visualization will show correlations between:\n'
               f'- LEC-5 trauma exposure metrics\n'
               f'- IES-R PTSD symptom scores\n'
               f'- Task performance measures\n\n'
               f'Add more participants to enable correlation analysis',
               ha='center',
               va='center',
               fontsize=12,
               transform=ax.transAxes)

        ax.axis('off')
        plt.title('Scale & Performance Correlation Matrix (Template)', fontsize=14, fontweight='bold')

        # Save
        save_path = save_dir / 'scale_correlations.png'
        plt.savefig(save_path, dpi=AnalysisParams.FIG_DPI, format=AnalysisParams.FIG_FORMAT, bbox_inches='tight')
        print(f"  [OK] Saved template: {save_path}")
        plt.close()
        return

    # Select relevant columns for correlation
    scale_cols = []

    # LEC-5 columns
    lec_cols = ['lec_total_events', 'lec_personal_events', 'lec_sum_exposures']
    scale_cols.extend([c for c in lec_cols if c in summary_df.columns])

    # IES-R columns
    ies_cols = ['ies_total', 'ies_intrusion', 'ies_avoidance', 'ies_hyperarousal']
    scale_cols.extend([c for c in ies_cols if c in summary_df.columns])

    # Task performance columns
    perf_cols = [
        'accuracy_overall',
        'accuracy_low_load',
        'accuracy_high_load',
        'mean_rt_overall',
        'learning_slope',
        'learning_improvement_early_to_late',
        'performance_drop_post_reversal',
        'adaptation_rate_post_reversal'
    ]
    scale_cols.extend([c for c in perf_cols if c in summary_df.columns])

    # Filter to available columns with data
    available_cols = [c for c in scale_cols if c in summary_df.columns and summary_df[c].notna().sum() >= 2]

    if len(available_cols) < 2:
        print("  [WARN] Insufficient data for correlation matrix")
        return

    # Compute correlation matrix
    corr_data = summary_df[available_cols].dropna()
    corr_matrix = corr_data.corr()

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Correlation Coefficient'},
        vmin=-1,
        vmax=1,
        ax=ax
    )

    # Format labels
    labels = [c.replace('_', ' ').replace('lec ', 'LEC-5 ').replace('ies ', 'IES-R ').title()
             for c in available_cols]
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels, rotation=0)

    plt.title(f'Scale & Performance Correlation Matrix (N={n_participants})', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()

    # Save
    save_path = save_dir / 'scale_correlations.png'
    plt.savefig(save_path, dpi=AnalysisParams.FIG_DPI, format=AnalysisParams.FIG_FORMAT, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()


def plot_trauma_performance_relationships(summary_df, save_dir):
    """
    Plot scatterplots of trauma metrics vs performance.

    Parameters
    ----------
    summary_df : pd.DataFrame
        Summary metrics data
    save_dir : Path
        Directory to save figures
    """
    n_participants = len(summary_df)

    if n_participants < 2:
        print("  [INFO] Scatter plots require N>=2 participants (skipping)")
        return

    # Define relationships to plot
    relationships = [
        ('lec_total_events', 'ies_total', 'LEC-5 Total Events', 'IES-R Total'),
        ('lec_total_events', 'accuracy_overall', 'LEC-5 Total Events', 'Overall Accuracy'),
        ('lec_personal_events', 'accuracy_overall', 'LEC-5 Personal Events', 'Overall Accuracy'),
        ('ies_total', 'accuracy_overall', 'IES-R Total', 'Overall Accuracy'),
        ('ies_total', 'learning_slope', 'IES-R Total', 'Learning Slope'),
        ('lec_total_events', 'performance_drop_post_reversal', 'LEC-5 Total', 'Performance Drop Post-Reversal'),
        ('ies_total', 'adaptation_rate_post_reversal', 'IES-R Total', 'Adaptation Rate Post-Reversal'),
    ]

    # Filter to available relationships
    available_rels = []
    for x_col, y_col, x_label, y_label in relationships:
        if x_col in summary_df.columns and y_col in summary_df.columns:
            if summary_df[[x_col, y_col]].notna().all(axis=1).sum() >= 2:
                available_rels.append((x_col, y_col, x_label, y_label))

    if not available_rels:
        print("  [INFO] No valid relationships for scatter plots (skipping)")
        return

    # Create figure
    n_plots = len(available_rels)
    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (x_col, y_col, x_label, y_label) in enumerate(available_rels):
        ax = axes[idx]

        # Get data
        plot_data = summary_df[[x_col, y_col]].dropna()

        if len(plot_data) < 2:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{y_label} vs {x_label}', fontweight='bold')
            continue

        # Scatter plot
        ax.scatter(plot_data[x_col], plot_data[y_col], alpha=0.6, s=100, color='steelblue')

        # Add trendline if enough data
        if len(plot_data) >= 3:
            z = np.polyfit(plot_data[x_col], plot_data[y_col], 1)
            p = np.poly1d(z)
            x_line = np.linspace(plot_data[x_col].min(), plot_data[x_col].max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)

            # Compute correlation
            corr = plot_data[x_col].corr(plot_data[y_col])
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        ax.set_xlabel(x_label, fontweight='bold')
        ax.set_ylabel(y_label, fontweight='bold')
        ax.set_title(f'{y_label} vs {x_label}', fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for idx in range(len(available_rels), len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Trauma & Performance Relationships (N={n_participants})', fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Save
    save_path = save_dir / 'trauma_performance_scatterplots.png'
    plt.savefig(save_path, dpi=AnalysisParams.FIG_DPI, format=AnalysisParams.FIG_FORMAT, bbox_inches='tight')
    print(f"  [OK] Saved: {save_path}")
    plt.close()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Visualize correlations between scale metrics and performance'
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
    print("SCALE CORRELATION VISUALIZATION")
    print("=" * 80)

    # Load data
    print(f"\nLoading summary metrics: {data_path}")
    summary_df = pd.read_csv(data_path)
    print(f"  Loaded {len(summary_df)} participants")

    if len(summary_df) == 1:
        print("\n  [NOTE] Only 1 participant - cannot compute correlations")
        print("          Will create template visualization")

    # Create visualizations
    print(f"\nCreating visualizations...")

    plot_scale_correlations(summary_df, figure_dir)
    plot_trauma_performance_relationships(summary_df, figure_dir)

    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)
    print(f"\nFigures saved to: {figure_dir}")

    if len(summary_df) < 2:
        print("\nNote: Correlation analyses require at least 2 participants.")
        print("      Add more participants to enable full correlation visualization.")

    print("=" * 80)


if __name__ == '__main__':
    main()
