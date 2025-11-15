"""
Visualization functions for model performance analysis.

This module provides functions to visualize:
1. Trial-by-trial accuracy (learning curves)
2. Performance by trial position (early vs late, pre vs post reversal)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, List
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from config import AnalysisParams


def get_color_palette(palette_type):
    """Get color palette from config."""
    if palette_type == 'set_size':
        return AnalysisParams.COLORS_SET_SIZE
    elif palette_type == 'load':
        return AnalysisParams.COLORS_LOAD
    elif palette_type == 'phase':
        return AnalysisParams.COLORS_PHASE
    else:
        raise ValueError(f"Unknown palette type: {palette_type}")


FIGURE_DPI = AnalysisParams.FIG_DPI
FIGURE_FORMAT = AnalysisParams.FIG_FORMAT


def plot_learning_curves(
    predictions_df: pd.DataFrame,
    trial_type: str = 'since_start',
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 6)
) -> None:
    """
    Plot trial-by-trial accuracy as line graph with separate lines per set size.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Model predictions with columns:
        - trial_num: trial number in block
        - trials_since_reversal: trials since last reversal
        - set_size: set size condition
        - correct: whether model predicted correctly (0/1)
        - accuracy: rolling/binned accuracy
    trial_type : str
        'since_start' or 'since_reversal' - x-axis type
    save_path : Path, optional
        Path to save figure
    title : str, optional
        Plot title (auto-generated if None)
    figsize : tuple
        Figure size (width, height)
    """
    # Set up figure
    fig, ax = plt.subplots(figsize=figsize)

    # Get color palette for set sizes
    colors = get_color_palette('set_size')
    set_sizes = sorted(predictions_df['set_size'].unique())

    # Determine x-axis column
    if trial_type == 'since_start':
        x_col = 'trial_num'
        x_label = 'Trial Since Block Start'
    elif trial_type == 'since_reversal':
        x_col = 'trials_since_reversal'
        x_label = 'Trials Since Reversal'
    else:
        raise ValueError(f"trial_type must be 'since_start' or 'since_reversal', got {trial_type}")

    # Plot line for each set size
    for ss in set_sizes:
        ss_data = predictions_df[predictions_df['set_size'] == ss].copy()

        # Group by x-axis and compute mean accuracy
        grouped = ss_data.groupby(x_col)['correct'].agg(['mean', 'sem']).reset_index()
        grouped = grouped.rename(columns={'mean': 'accuracy'})

        # Plot line with error band
        ax.plot(
            grouped[x_col],
            grouped['accuracy'] * 100,  # Convert to percentage
            'o-',
            color=colors[ss],
            label=f'Set Size {ss}',
            linewidth=2,
            markersize=4,
            alpha=0.8
        )

        # Add confidence band (±1 SEM)
        if 'sem' in grouped.columns:
            ax.fill_between(
                grouped[x_col],
                (grouped['accuracy'] - grouped['sem']) * 100,
                (grouped['accuracy'] + grouped['sem']) * 100,
                color=colors[ss],
                alpha=0.2
            )

    # Styling
    ax.set_xlabel(x_label, fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 105])
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Chance')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Title
    if title is None:
        title = f'Learning Curves: Accuracy by {x_label}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"  [OK] Saved: {save_path}")

    plt.close()


def plot_performance_by_trial_position(
    predictions_df: pd.DataFrame,
    n_trials_threshold: int = 4,
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 7)
) -> None:
    """
    Plot bar chart of accuracy by set size and trial position.

    Trial positions:
    1. Early block (< n_trials_threshold since start)
    2. Late block (>= n_trials_threshold since start, before reversal)
    3. Early post-reversal (< n_trials_threshold since reversal)
    4. Late post-reversal (>= n_trials_threshold since reversal)

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Model predictions with columns:
        - trial_num: trial number in block
        - trials_since_reversal: trials since last reversal
        - set_size: set size condition
        - correct: whether model predicted correctly (0/1)
        - is_post_reversal: whether trial is after reversal
    n_trials_threshold : int
        Threshold for "early" vs "late" trials (default: 4)
    save_path : Path, optional
        Path to save figure
    title : str, optional
        Plot title (auto-generated if None)
    figsize : tuple
        Figure size (width, height)
    """
    # Create trial position categories
    df = predictions_df.copy()

    def categorize_trial(row):
        """Categorize trial by position."""
        if not row.get('is_post_reversal', False):
            # Pre-reversal trials
            if row['trial_num'] < n_trials_threshold:
                return 'Early Block\n(< 4 trials)'
            else:
                return 'Late Block\n(≥ 4 trials, pre-reversal)'
        else:
            # Post-reversal trials
            if row['trials_since_reversal'] < n_trials_threshold:
                return 'Early Post-Reversal\n(< 4 trials)'
            else:
                return 'Late Post-Reversal\n(≥ 4 trials)'

    df['trial_position'] = df.apply(categorize_trial, axis=1)

    # Define position order for plotting
    position_order = [
        'Early Block\n(< 4 trials)',
        'Late Block\n(≥ 4 trials, pre-reversal)',
        'Early Post-Reversal\n(< 4 trials)',
        'Late Post-Reversal\n(≥ 4 trials)'
    ]

    # Compute accuracy by set size and trial position
    grouped = df.groupby(['set_size', 'trial_position'])['correct'].agg(['mean', 'sem']).reset_index()
    grouped = grouped.rename(columns={'mean': 'accuracy'})

    # Get set sizes and colors
    set_sizes = sorted(df['set_size'].unique())
    colors = get_color_palette('set_size')

    # Set up figure
    fig, ax = plt.subplots(figsize=figsize)

    # Bar plot parameters
    n_positions = len(position_order)
    n_set_sizes = len(set_sizes)
    bar_width = 0.8 / n_set_sizes
    x = np.arange(n_positions)

    # Plot bars for each set size
    for i, ss in enumerate(set_sizes):
        ss_data = grouped[grouped['set_size'] == ss].copy()

        # Ensure all positions are represented
        ss_data = ss_data.set_index('trial_position').reindex(position_order).reset_index()
        ss_data['accuracy'] = ss_data['accuracy'].fillna(0)
        ss_data['sem'] = ss_data['sem'].fillna(0)

        # Plot bars
        offset = (i - n_set_sizes/2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            ss_data['accuracy'] * 100,  # Convert to percentage
            bar_width,
            label=f'Set Size {ss}',
            color=colors[ss],
            alpha=0.8,
            yerr=ss_data['sem'] * 100,  # Error bars
            capsize=5,
            error_kw={'linewidth': 1.5, 'alpha': 0.7}
        )

        # Add value labels on bars
        for bar, acc in zip(bars, ss_data['accuracy'] * 100):
            height = bar.get_height()
            if height > 0:  # Only label non-zero bars
                ax.text(
                    bar.get_x() + bar.get_width()/2,
                    height + 2,
                    f'{acc:.1f}%',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    fontweight='bold'
                )

    # Styling
    ax.set_xlabel('Trial Position', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 110])
    ax.set_xticks(x)
    ax.set_xticklabels(position_order, fontsize=10)
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Chance')
    ax.legend(loc='upper right', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    # Title
    if title is None:
        title = f'Performance by Trial Position (Threshold = {n_trials_threshold} trials)'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"  [OK] Saved: {save_path}")

    plt.close()


def plot_combined_performance_analysis(
    predictions_df: pd.DataFrame,
    n_trials_threshold: int = 4,
    save_dir: Optional[Path] = None,
    model_name: str = 'Model'
) -> None:
    """
    Create both performance plots in a single figure.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Model predictions
    n_trials_threshold : int
        Threshold for early/late trials
    save_dir : Path, optional
        Directory to save figures
    model_name : str
        Model name for titles
    """
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(1, 2, wspace=0.3)

    # Get colors
    colors = get_color_palette('set_size')
    set_sizes = sorted(predictions_df['set_size'].unique())

    # ===== LEFT: Learning curve (since start) =====
    ax1 = fig.add_subplot(gs[0, 0])

    for ss in set_sizes:
        ss_data = predictions_df[predictions_df['set_size'] == ss].copy()
        grouped = ss_data.groupby('trial_num')['correct'].agg(['mean', 'sem']).reset_index()
        grouped = grouped.rename(columns={'mean': 'accuracy'})

        ax1.plot(
            grouped['trial_num'],
            grouped['accuracy'] * 100,
            'o-',
            color=colors[ss],
            label=f'Set Size {ss}',
            linewidth=2,
            markersize=4,
            alpha=0.8
        )

        ax1.fill_between(
            grouped['trial_num'],
            (grouped['accuracy'] - grouped['sem']) * 100,
            (grouped['accuracy'] + grouped['sem']) * 100,
            color=colors[ss],
            alpha=0.2
        )

    ax1.set_xlabel('Trial Since Block Start', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax1.set_ylim([0, 105])
    ax1.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'{model_name}: Learning Curves', fontsize=12, fontweight='bold')

    # ===== RIGHT: Performance by position =====
    ax2 = fig.add_subplot(gs[0, 1])

    df = predictions_df.copy()

    def categorize_trial(row):
        if not row.get('is_post_reversal', False):
            if row['trial_num'] < n_trials_threshold:
                return 'Early Block'
            else:
                return 'Late Block'
        else:
            if row['trials_since_reversal'] < n_trials_threshold:
                return 'Early Post-Rev'
            else:
                return 'Late Post-Rev'

    df['trial_position'] = df.apply(categorize_trial, axis=1)

    position_order = ['Early Block', 'Late Block', 'Early Post-Rev', 'Late Post-Rev']
    grouped = df.groupby(['set_size', 'trial_position'])['correct'].agg(['mean', 'sem']).reset_index()
    grouped = grouped.rename(columns={'mean': 'accuracy'})

    n_positions = len(position_order)
    n_set_sizes = len(set_sizes)
    bar_width = 0.8 / n_set_sizes
    x = np.arange(n_positions)

    for i, ss in enumerate(set_sizes):
        ss_data = grouped[grouped['set_size'] == ss].copy()
        ss_data = ss_data.set_index('trial_position').reindex(position_order).reset_index()
        ss_data['accuracy'] = ss_data['accuracy'].fillna(0)
        ss_data['sem'] = ss_data['sem'].fillna(0)

        offset = (i - n_set_sizes/2 + 0.5) * bar_width
        ax2.bar(
            x + offset,
            ss_data['accuracy'] * 100,
            bar_width,
            label=f'Set Size {ss}',
            color=colors[ss],
            alpha=0.8,
            yerr=ss_data['sem'] * 100,
            capsize=4
        )

    ax2.set_xlabel('Trial Position', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax2.set_ylim([0, 110])
    ax2.set_xticks(x)
    ax2.set_xticklabels(position_order, fontsize=9, rotation=15, ha='right')
    ax2.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_title(f'{model_name}: Performance by Trial Position', fontsize=12, fontweight='bold')

    plt.suptitle(f'{model_name} Performance Analysis', fontsize=14, fontweight='bold', y=1.02)

    # Save
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f'{model_name.lower().replace(" ", "_")}_performance_analysis.png'
        plt.savefig(save_path, dpi=FIGURE_DPI, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"  [OK] Saved: {save_path}")

    plt.close()


if __name__ == '__main__':
    print("Visualization functions loaded.")
    print("\nAvailable functions:")
    print("  - plot_learning_curves()")
    print("  - plot_performance_by_trial_position()")
    print("  - plot_combined_performance_analysis()")
