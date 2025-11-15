"""
Plotting Utilities for Behavioral Data Analysis

Generic, reusable plotting functions for creating consistent, publication-quality figures.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import AnalysisParams


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
    group_cols: Optional[List[str]] = None,
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
    title: Optional[str] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
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


def get_color_palette(palette_name: str = 'set_size') -> Dict:
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
