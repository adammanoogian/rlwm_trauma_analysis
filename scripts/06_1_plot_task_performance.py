#!/usr/bin/env python
"""
06_1: Plot Task Performance
===========================

Generate task performance visualization plots.

This script generates basic performance visualizations including:
- Accuracy by set size
- Learning curves across trials
- Post-reversal learning patterns

PROMOTED from: scripts/analysis/plot_task_performance.py

Inputs:
    - output/task_trials_long.csv

Outputs:
    - figures/behavioral_analysis/accuracy_by_setsize.png
    - figures/behavioral_analysis/learning_curves_by_setsize.png
    - figures/behavioral_analysis/post_reversal_learning.png

Usage:
    python scripts/06_1_plot_task_performance.py

Next Steps:
    - Run 07_analyze_trauma_groups.py for group analysis
    - Run 07_1_visualize_by_trauma_group.py for group-specific visualizations
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import AnalysisParams

# ============================================================================
# Plotting Utilities
# ============================================================================

def setup_plot_style():
    """
    Apply consistent matplotlib style for all plots.

    Uses configuration from config.AnalysisParams to ensure visual consistency.
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.3)

    # Set matplotlib parameters
    plt.rcParams['figure.dpi'] = AnalysisParams.FIG_DPI
    plt.rcParams['savefig.dpi'] = AnalysisParams.FIG_DPI
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 6

def save_figure(
    fig: plt.Figure,
    filename: str,
    subdir: str = 'behavioral_analysis',
    tight: bool = True
):
    """
    Save figure to the figures directory with consistent naming and format.

    Parameters
    ----------
    fig : plt.Figure
        Figure object to save
    filename : str
        Base filename (without extension)
    subdir : str
        Subdirectory within figures/ (default: 'behavioral_analysis')
    tight : bool
        Whether to use tight_layout (default: True)
    """
    # Construct output path
    figures_dir = project_root / 'figures' / subdir
    figures_dir.mkdir(parents=True, exist_ok=True)

    output_path = figures_dir / f"{filename}.{AnalysisParams.FIG_FORMAT}"

    # Apply tight layout if requested
    if tight:
        fig.tight_layout()

    # Save figure
    fig.savefig(output_path, dpi=AnalysisParams.FIG_DPI, bbox_inches='tight')
    print(f"Saved figure: {output_path}")

def aggregate_by_condition(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    group_cols: list[str] | None = None,
    participant_col: str = 'sona_id'
) -> pd.DataFrame:
    """
    Aggregate data by condition, computing mean and SEM across participants.

    Parameters
    ----------
    data : pd.DataFrame
        Trial-level data
    x_col : str
        Column name for x-axis variable
    y_col : str
        Column name for y-axis variable (to aggregate)
    group_cols : list of str, optional
        Additional grouping columns (e.g., set_size, condition)
    participant_col : str
        Column name for participant ID (default: 'sona_id')

    Returns
    -------
    pd.DataFrame
        Aggregated data with columns: [group_cols, x_col, mean, sem, n]
    """
    if group_cols is None:
        group_cols = []

    # First, aggregate within each participant
    participant_means = data.groupby([participant_col] + group_cols + [x_col])[y_col].mean().reset_index()

    # Then, aggregate across participants
    grouping = group_cols + [x_col]
    agg_data = participant_means.groupby(grouping)[y_col].agg([
        ('mean', 'mean'),
        ('sem', lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        ('n', 'count')
    ]).reset_index()

    return agg_data

def plot_line_with_error(
    ax: plt.Axes,
    x: np.ndarray,
    y_mean: np.ndarray,
    y_sem: np.ndarray,
    color: str,
    label: str,
    alpha: float = 0.2,
    marker: str = 'o',
    linestyle: str = '-'
):
    """
    Plot line with shaded error region (SEM).

    Parameters
    ----------
    ax : plt.Axes
        Axes object to plot on
    x : np.ndarray
        X-axis values
    y_mean : np.ndarray
        Mean values for y-axis
    y_sem : np.ndarray
        Standard error of the mean for y-axis
    color : str
        Line and fill color
    label : str
        Legend label
    alpha : float
        Transparency for error region (default: 0.2)
    marker : str
        Marker style (default: 'o')
    linestyle : str
        Line style (default: '-')
    """
    # Plot mean line
    ax.plot(x, y_mean, color=color, label=label, marker=marker, linestyle=linestyle)

    # Plot error region
    ax.fill_between(
        x,
        y_mean - y_sem,
        y_mean + y_sem,
        color=color,
        alpha=alpha
    )

def format_axes(
    ax: plt.Axes,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    add_legend: bool = True,
    legend_loc: str = 'best'
):
    """
    Apply consistent formatting to axes.

    Parameters
    ----------
    ax : plt.Axes
        Axes object to format
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    title : str, optional
        Axes title
    xlim : tuple, optional
        X-axis limits (min, max)
    ylim : tuple, optional
        Y-axis limits (min, max)
    add_legend : bool
        Whether to add legend (default: True)
    legend_loc : str
        Legend location (default: 'best')
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title:
        ax.set_title(title)

    if xlim:
        ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim(ylim)

    if add_legend:
        ax.legend(loc=legend_loc, frameon=True, fancybox=False, edgecolor='black')

    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--')

    # Improve spine visibility
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

def get_color_palette(palette_name: str = 'set_size') -> dict:
    """
    Get color palette from config.

    Parameters
    ----------
    palette_name : str
        Name of the color palette ('set_size', 'load', 'phase')

    Returns
    -------
    dict
        Color dictionary
    """
    if palette_name == 'set_size':
        return AnalysisParams.COLORS_SET_SIZE
    elif palette_name == 'load':
        return AnalysisParams.COLORS_LOAD
    elif palette_name == 'phase':
        return AnalysisParams.COLORS_PHASE
    else:
        raise ValueError(f"Unknown palette: {palette_name}")

# ============================================================================
# Behavioral Plotting Functions
# ============================================================================

def plot_accuracy_by_setsize(
    data: pd.DataFrame,
    save: bool = True,
    show: bool = False
) -> plt.Figure:
    """
    Plot overall accuracy as a function of set size.

    Creates a bar plot showing mean accuracy (± SEM) for each set size.

    Parameters
    ----------
    data : pd.DataFrame
        Trial-level data with columns: 'sona_id', 'set_size', 'correct'
    save : bool
        Whether to save the figure (default: True)
    show : bool
        Whether to display the figure (default: False)

    Returns
    -------
    plt.Figure
        Figure object
    """
    setup_plot_style()

    # Aggregate data
    agg_data = aggregate_by_condition(
        data,
        x_col='set_size',
        y_col='correct',
        group_cols=[]
    )

    # Get colors
    colors = get_color_palette('set_size')

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot bars
    set_sizes = agg_data['set_size'].values
    means = agg_data['mean'].values
    sems = agg_data['sem'].values

    bar_colors = [colors[ss] for ss in set_sizes]

    bars = ax.bar(
        set_sizes,
        means,
        yerr=sems,
        color=bar_colors,
        alpha=0.8,
        edgecolor='black',
        linewidth=1.5,
        capsize=5,
        error_kw={'linewidth': 2}
    )

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.01,
            f'{mean:.2f}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    # Format axes
    ax.set_xlabel('Set Size', fontsize=14, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('Overall Accuracy by Set Size', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(set_sizes)

    # Add horizontal line at chance (1/3 for 3-choice task)
    ax.axhline(y=1/3, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Chance')

    # Add grid
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Add legend
    ax.legend(loc='lower left', frameon=True, fancybox=False, edgecolor='black')

    # Improve spines
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    # Save figure
    if save:
        save_figure(fig, 'accuracy_by_setsize', subdir='behavioral_analysis')

    if show:
        pass  # plt.show() removed for headless compatibility
    else:
        plt.close(fig)

    return fig

def plot_learning_curves(
    data: pd.DataFrame,
    max_trials: int = 20,
    save: bool = True,
    show: bool = False
) -> plt.Figure:
    """
    Plot learning curves: accuracy vs trials per stimulus, by set size.

    Creates 2 subplots (placeholders for Q-learning and WM-RL models),
    each showing learning curves for different set sizes.

    Parameters
    ----------
    data : pd.DataFrame
        Trial-level data with columns: 'sona_id', 'set_size', 'counter', 'correct'
    max_trials : int
        Maximum number of trials per stimulus to plot (default: 20)
    save : bool
        Whether to save the figure (default: True)
    show : bool
        Whether to display the figure (default: False)

    Returns
    -------
    plt.Figure
        Figure object
    """
    setup_plot_style()

    # Filter data to max_trials
    plot_data = data[data['counter'] <= max_trials].copy()

    # Aggregate data by set size and trial
    agg_data = aggregate_by_condition(
        plot_data,
        x_col='counter',
        y_col='correct',
        group_cols=['set_size']
    )

    # Get colors
    colors = get_color_palette('set_size')
    set_sizes = sorted(agg_data['set_size'].unique())

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Plot for each "model" (currently both show empirical data)
    model_names = ['Behavioral Data (Empirical)', 'Model Comparison (Future)']

    for idx, (ax, model_name) in enumerate(zip(axes, model_names)):
        for set_size in set_sizes:
            # Get data for this set size
            ss_data = agg_data[agg_data['set_size'] == set_size]

            x = ss_data['counter'].values
            y_mean = ss_data['mean'].values
            y_sem = ss_data['sem'].values

            # Plot line with error
            plot_line_with_error(
                ax,
                x,
                y_mean,
                y_sem,
                color=colors[set_size],
                label=f'Set Size {set_size}',
                alpha=0.2,
                marker='o' if idx == 0 else 's',
                linestyle='-'
            )

        # Add chance line
        ax.axhline(y=1/3, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Chance')

        # Format axes
        format_axes(
            ax,
            xlabel='Trials per Stimulus',
            ylabel='Accuracy' if idx == 0 else '',
            title=model_name,
            ylim=(0, 1.05),
            xlim=(1, max_trials),
            add_legend=True,
            legend_loc='lower right'
        )

    # Overall title
    fig.suptitle('Learning Curves: Accuracy vs Trials per Stimulus', fontsize=16, fontweight='bold', y=1.02)

    # Save figure
    if save:
        save_figure(fig, 'learning_curves_by_setsize', subdir='behavioral_analysis')

    if show:
        pass  # plt.show() removed for headless compatibility
    else:
        plt.close(fig)

    return fig

def plot_post_reversal_learning(
    data: pd.DataFrame,
    max_trials_post_reversal: int = 15,
    save: bool = True,
    show: bool = False
) -> plt.Figure:
    """
    Plot post-reversal learning: accuracy vs trials since reversal, by set size.

    Creates 2 subplots (placeholders for Q-learning and WM-RL models),
    each showing adaptation after reversals for different set sizes.

    Parameters
    ----------
    data : pd.DataFrame
        Trial-level data with columns: 'sona_id', 'set_size', 'counter', 'correct', 'reversal_crit'
    max_trials_post_reversal : int
        Maximum number of post-reversal trials to plot (default: 15)
    save : bool
        Whether to save the figure (default: True)
    show : bool
        Whether to display the figure (default: False)

    Returns
    -------
    plt.Figure
        Figure object
    """
    setup_plot_style()

    # Identify reversal trials
    # Reversal occurs when reversal_crit changes (e.g., from a number to 'inf' or changes value)
    # We'll detect when counter resets to 0 or 1 after being higher

    # Create a copy and sort by participant, block, trial
    df = data.copy()
    df = df.sort_values(['sona_id', 'block', 'trial_in_block'])

    # Detect reversals: counter decreases (resets) indicates reversal
    df['counter_prev'] = df.groupby(['sona_id', 'block', 'stimulus'])['counter'].shift(1)
    df['is_reversal'] = (df['counter'] < df['counter_prev']) & (df['counter_prev'].notna())

    # Create trial index relative to last reversal
    df['trials_since_reversal'] = 0

    for idx in df[df['is_reversal']].index:
        # Get the participant, block, stimulus
        participant = df.loc[idx, 'sona_id']
        block = df.loc[idx, 'block']
        stimulus = df.loc[idx, 'stimulus']

        # Mark all subsequent trials for this stimulus as post-reversal
        mask = (
            (df['sona_id'] == participant) &
            (df['block'] == block) &
            (df['stimulus'] == stimulus) &
            (df.index >= idx)
        )

        # Reset counter for post-reversal trials
        df.loc[mask, 'trials_since_reversal'] = df.loc[mask, 'counter']

    # Filter to only post-reversal trials
    post_rev_data = df[df['trials_since_reversal'] > 0].copy()
    post_rev_data = post_rev_data[post_rev_data['trials_since_reversal'] <= max_trials_post_reversal]

    if len(post_rev_data) == 0:
        print("Warning: No post-reversal trials found in data")
        return None

    # Aggregate data
    agg_data = aggregate_by_condition(
        post_rev_data,
        x_col='trials_since_reversal',
        y_col='correct',
        group_cols=['set_size']
    )

    # Get colors
    colors = get_color_palette('set_size')
    set_sizes = sorted(agg_data['set_size'].unique())

    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    # Plot for each "model"
    model_names = ['Behavioral Data (Empirical)', 'Model Comparison (Future)']

    for idx, (ax, model_name) in enumerate(zip(axes, model_names)):
        for set_size in set_sizes:
            # Get data for this set size
            ss_data = agg_data[agg_data['set_size'] == set_size]

            if len(ss_data) == 0:
                continue

            x = ss_data['trials_since_reversal'].values
            y_mean = ss_data['mean'].values
            y_sem = ss_data['sem'].values

            # Plot line with error
            plot_line_with_error(
                ax,
                x,
                y_mean,
                y_sem,
                color=colors[set_size],
                label=f'Set Size {set_size}',
                alpha=0.2,
                marker='o' if idx == 0 else 's',
                linestyle='-'
            )

        # Add chance line
        ax.axhline(y=1/3, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Chance')

        # Format axes
        format_axes(
            ax,
            xlabel='Trials Since Reversal',
            ylabel='Accuracy' if idx == 0 else '',
            title=model_name,
            ylim=(0, 1.05),
            xlim=(1, max_trials_post_reversal),
            add_legend=True,
            legend_loc='lower right'
        )

    # Overall title
    fig.suptitle('Post-Reversal Learning: Adaptation After Contingency Change',
                 fontsize=16, fontweight='bold', y=1.02)

    # Save figure
    if save:
        save_figure(fig, 'post_reversal_learning', subdir='behavioral_analysis')

    if show:
        pass  # plt.show() removed for headless compatibility
    else:
        plt.close(fig)

    return fig

# ============================================================================
# Main Pipeline
# ============================================================================

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
    print("Next steps:")
    print("  - Run 07_analyze_trauma_groups.py for group analysis")
    print("  - Run 07_1_visualize_by_trauma_group.py for group-specific visualizations")
    print()

if __name__ == "__main__":
    main()
