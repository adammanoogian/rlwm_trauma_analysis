"""
Behavioral Data Plotting Functions

Specific plotting functions for visualizing task performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import AnalysisParams, TaskParams
from scripts.analysis.plotting_utils import (
    setup_plot_style,
    save_figure,
    aggregate_by_condition,
    plot_line_with_error,
    format_axes,
    get_color_palette
)


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
        plt.show()
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
        plt.show()
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
        plt.show()
    else:
        plt.close(fig)

    return fig
