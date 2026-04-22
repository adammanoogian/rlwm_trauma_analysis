"""
Shared plotting utilities for consistent color-by visualization.

This module provides color palette generation and colored scatter plotting
functions that can be reused across analysis scripts (15, 16, etc.).

Functions:
    get_color_palette: Generate color mapping for categorical variable
    add_colored_scatter: Add colored scatter plot with legend
    TRAUMA_GROUP_COLORS: Predefined colors for hypothesis_group
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
from typing import Dict, Optional, List, Union


# Predefined trauma group colors (matching actual group_assignments.csv values)
# Note: All participants in this sample have trauma exposure (no "No Trauma" group)
TRAUMA_GROUP_COLORS = {
    'Trauma Exposure - No Ongoing Impact': '#F18F01',  # Orange
    'Trauma Exposure - Ongoing Impact': '#D62246',  # Red
    'Low Exposure-High Symptoms': '#6C757D',  # Gray (paradoxical, if present)
}


def get_color_palette(
    data_df: pd.DataFrame,
    color_by: str,
    custom_colors: Optional[Dict[str, str]] = None
) -> Dict[str, str]:
    """
    Generate color palette for categorical grouping.

    Args:
        data_df: DataFrame with grouping column
        color_by: Column name to color by (must exist in DataFrame)
        custom_colors: Optional dict mapping category -> color hex string.
                      If provided, uses these colors (falling back to '#808080'
                      for unmapped categories)

    Returns:
        dict: Mapping {category: hex_color_string}

    Raises:
        ValueError: If color_by column not found in DataFrame

    Examples:
        # Auto-generate palette
        palette = get_color_palette(df, 'gender')

        # Use custom colors
        palette = get_color_palette(df, 'hypothesis_group',
                                   custom_colors=TRAUMA_GROUP_COLORS)
    """
    if color_by not in data_df.columns:
        available = list(data_df.columns)
        raise ValueError(
            f"Column '{color_by}' not found in DataFrame. "
            f"Available columns: {available}"
        )

    # Get unique categories (sorted for consistency, dropping NaN)
    categories = sorted(data_df[color_by].dropna().unique())

    # If no categories found, return empty dict
    if len(categories) == 0:
        return {}

    # Use custom colors if provided
    if custom_colors is not None:
        palette = {}
        for cat in categories:
            if cat in custom_colors:
                palette[cat] = custom_colors[cat]
            else:
                # Fallback to gray for unmapped categories
                palette[cat] = '#808080'
        return palette

    # Otherwise generate palette automatically
    n_colors = len(categories)

    if n_colors <= 10:
        # Use seaborn's tab10 for up to 10 categories
        colors = sns.color_palette('tab10', n_colors=n_colors)
    else:
        # Use matplotlib colormap for many categories
        cmap = plt.cm.get_cmap('Set3', n_colors)
        colors = [cmap(i) for i in range(n_colors)]

    # Convert to hex
    palette = {
        cat: matplotlib.colors.rgb2hex(colors[i])
        for i, cat in enumerate(categories)
    }

    return palette


def add_colored_scatter(
    ax: plt.Axes,
    x: Union[str, np.ndarray],
    y: Union[str, np.ndarray],
    data_df: pd.DataFrame,
    color_by: str,
    palette: Dict[str, str],
    alpha: float = 0.7,
    s: float = 50,
    show_legend: bool = True,
    **kwargs
) -> List:
    """
    Add scatter plot with categorical coloring and legend.

    Args:
        ax: Matplotlib axes object
        x: Column name (str) or array for x-axis
        y: Column name (str) or array for y-axis
        data_df: DataFrame with data
        color_by: Column name for grouping/coloring
        palette: Color palette dict from get_color_palette()
        alpha: Point transparency (default: 0.7)
        s: Point size (default: 50)
        show_legend: Whether to add legend (default: True)
        **kwargs: Additional arguments passed to ax.scatter()

    Returns:
        List of PathCollection handles (one per category)

    Examples:
        # Basic usage
        palette = get_color_palette(df, 'gender')
        add_colored_scatter(ax, 'age', 'score', df, 'gender', palette)

        # With arrays instead of column names
        add_colored_scatter(ax, x_array, y_array, df, 'group', palette)
    """
    # Get x, y data (handle both column names and arrays)
    if isinstance(x, str):
        x_data = data_df[x].values
    else:
        x_data = np.asarray(x)

    if isinstance(y, str):
        y_data = data_df[y].values
    else:
        y_data = np.asarray(y)

    # Plot each category separately
    handles = []
    categories = sorted(data_df[color_by].dropna().unique())

    for category in categories:
        # Get mask for this category
        mask = data_df[color_by] == category

        # Plot this category
        h = ax.scatter(
            x_data[mask],
            y_data[mask],
            c=palette.get(category, '#808080'),
            label=category,
            alpha=alpha,
            s=s,
            **kwargs
        )
        handles.append(h)

    # Add legend if requested
    if show_legend:
        n_cats = len(categories)

        # Determine legend columns based on number of categories
        if n_cats <= 5:
            ncol = 1
        elif n_cats <= 10:
            ncol = 2
        elif n_cats <= 15:
            ncol = 3
        else:
            # Too many categories - skip legend
            return handles

        ax.legend(ncol=ncol, loc='best', fontsize=10)

    return handles


def plot_scatter_with_annotations(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    annotations: Dict[str, float],
    show_identity: bool = True,
    show_regression: bool = True,
    identity_color: str = 'black',
    regression_color: str = 'red',
    point_color: str = '#1f77b4',
    alpha: float = 0.6,
    s: float = 50,
    pass_threshold: Optional[float] = None,
    pass_key: str = 'r',
    **kwargs
) -> None:
    """
    Add scatter plot with identity line, regression line, and annotations.

    Generic function for comparing two related variables (e.g., true vs recovered,
    predicted vs actual). Follows existing ax-based pattern for composability.

    Args:
        ax: Matplotlib axes object
        x, y: Arrays of values to plot
        annotations: Dict of stats to display (e.g., {'r': 0.85, 'RMSE': 0.05})
        show_identity: Show y=x identity line (dashed)
        show_regression: Show regression line (solid)
        identity_color: Color for identity line (default: 'black')
        regression_color: Color for regression line (default: 'red')
        point_color: Color for scatter points (default: '#1f77b4')
        alpha: Point transparency (default: 0.6)
        s: Point size (default: 50)
        pass_threshold: If provided, adds PASS/FAIL badge based on annotations[pass_key]
        pass_key: Key in annotations dict for pass/fail check (default: 'r')
        **kwargs: Additional arguments passed to ax.scatter()

    Examples:
        # Basic usage
        fig, ax = plt.subplots()
        plot_scatter_with_annotations(
            ax, true_vals, recovered_vals,
            annotations={'r': 0.85, 'RMSE': 0.05, 'Bias': -0.01}
        )

        # With pass/fail badge
        plot_scatter_with_annotations(
            ax, true_vals, recovered_vals,
            annotations={'r': 0.92},
            pass_threshold=0.80,
            pass_key='r'
        )
    """
    # Scatter plot
    ax.scatter(x, y, c=point_color, alpha=alpha, s=s, edgecolors='white', linewidths=0.5, **kwargs)

    # Identity line (y=x)
    if show_identity:
        lims = [
            np.min([ax.get_xlim()[0], ax.get_ylim()[0]]),
            np.max([ax.get_xlim()[1], ax.get_ylim()[1]])
        ]
        ax.plot(lims, lims, '--', color=identity_color, alpha=0.7, linewidth=1.5, label='Identity')

    # Regression line
    if show_regression:
        # Fit linear regression
        coeffs = np.polyfit(x, y, 1)
        poly_fn = np.poly1d(coeffs)
        x_sorted = np.sort(x)
        ax.plot(x_sorted, poly_fn(x_sorted), '-', color=regression_color, linewidth=2, alpha=0.8, label='Regression')

    # Annotations box (top-left)
    annotation_text = []
    for key, value in annotations.items():
        if isinstance(value, float):
            if abs(value) < 0.01:
                annotation_text.append(f"{key}: {value:.4f}")
            else:
                annotation_text.append(f"{key}: {value:.3f}")
        else:
            annotation_text.append(f"{key}: {value}")

    if annotation_text:
        ax.text(
            0.05, 0.95,
            '\n'.join(annotation_text),
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
        )

    # PASS/FAIL badge (top-right)
    if pass_threshold is not None and pass_key in annotations:
        value = annotations[pass_key]
        passed = value >= pass_threshold
        badge_text = 'PASS' if passed else 'FAIL'
        badge_color = '#28a745' if passed else '#dc3545'  # Green/Red

        ax.text(
            0.95, 0.95,
            badge_text,
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            verticalalignment='top',
            horizontalalignment='right',
            color='white',
            bbox=dict(boxstyle='round', facecolor=badge_color, alpha=0.9, edgecolor='none', pad=0.5)
        )

    ax.grid(True, alpha=0.3)


def plot_kde_comparison(
    ax: plt.Axes,
    distributions: Dict[str, np.ndarray],
    colors: Optional[Dict[str, str]] = None,
    fill: bool = True,
    alpha: float = 0.3,
    linewidth: float = 2,
    **kwargs
) -> None:
    """
    Plot overlapping KDE distributions for comparison.

    Generic function for comparing same variable from different sources
    (e.g., recovered vs real fitted parameters, model A vs model B).
    Auto-generates colors if not provided.

    Args:
        ax: Matplotlib axes object
        distributions: Dict mapping label -> array of values
        colors: Optional dict mapping label -> color hex string
        fill: Whether to fill under KDE curves (default: True)
        alpha: Fill transparency (default: 0.3)
        linewidth: Line width for KDE curves (default: 2)
        **kwargs: Additional arguments passed to sns.kdeplot()

    Examples:
        # Basic usage
        fig, ax = plt.subplots()
        plot_kde_comparison(
            ax,
            {'Recovered': recovered_vals, 'Real Fitted': real_vals}
        )

        # With custom colors
        plot_kde_comparison(
            ax,
            {'Group A': vals_a, 'Group B': vals_b},
            colors={'Group A': '#1f77b4', 'Group B': '#ff7f0e'}
        )
    """
    # Auto-generate colors if not provided
    if colors is None:
        n_dists = len(distributions)
        palette = sns.color_palette('tab10', n_colors=n_dists)
        colors = {label: matplotlib.colors.rgb2hex(palette[i]) for i, label in enumerate(distributions.keys())}

    # Plot each distribution
    for label, values in distributions.items():
        color = colors.get(label, '#808080')  # Fallback to gray

        sns.kdeplot(
            data=values,
            ax=ax,
            color=color,
            fill=fill,
            alpha=alpha,
            linewidth=linewidth,
            label=label,
            **kwargs
        )

    ax.grid(True, alpha=0.3)


def plot_behavioral_comparison(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    output_dir,
    model_name: str = 'Model'
) -> None:
    """
    Generate overlay plots comparing real vs synthetic behavioral patterns.

    Creates:
    1. Set-size accuracy comparison (bar chart)
    2. Learning curve comparison (line plot)
    3. Overall accuracy distribution comparison (KDE)

    Parameters
    ----------
    real_data : pd.DataFrame
        Real trial data
    synthetic_data : pd.DataFrame
        Synthetic trial data
    output_dir : Path
        Directory to save figures
    model_name : str
        Model name for plot titles
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Set-size accuracy comparison
    fig, ax = plt.subplots(figsize=(8, 6))

    set_sizes = [2, 3, 5, 6]
    x = np.arange(len(set_sizes))
    width = 0.35

    real_acc = [real_data[real_data['set_size'] == ss]['reward'].mean() for ss in set_sizes]
    syn_acc = [synthetic_data[synthetic_data['set_size'] == ss]['reward'].mean() for ss in set_sizes]

    ax.bar(x - width/2, real_acc, width, label='Real', color='steelblue', alpha=0.8)
    ax.bar(x + width/2, syn_acc, width, label='Synthetic', color='coral', alpha=0.8)

    ax.set_xlabel('Set Size')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Accuracy by Set Size: Real vs Synthetic ({model_name})')
    ax.set_xticks(x)
    ax.set_xticklabels(set_sizes)
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'setsize_comparison.png', dpi=150)
    plt.close()

    # 2. Learning curve comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    # Compute mean accuracy per block
    real_by_block = real_data.groupby('block')['reward'].mean().reset_index()
    syn_by_block = synthetic_data.groupby('block')['reward'].mean().reset_index()

    ax.plot(real_by_block['block'], real_by_block['reward'],
            'o-', label='Real', color='steelblue', linewidth=2, markersize=5)
    ax.plot(syn_by_block['block'], syn_by_block['reward'],
            's--', label='Synthetic', color='coral', linewidth=2, markersize=5)

    ax.set_xlabel('Block')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Learning Curve: Real vs Synthetic ({model_name})')
    ax.legend()
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curve_comparison.png', dpi=150)
    plt.close()

    # 3. Per-participant accuracy distribution
    fig, ax = plt.subplots(figsize=(8, 6))

    real_subj_acc = real_data.groupby('sona_id')['reward'].mean()
    syn_subj_acc = synthetic_data.groupby('sona_id')['reward'].mean()

    sns.kdeplot(real_subj_acc, ax=ax, label='Real', color='steelblue', fill=True, alpha=0.3)
    sns.kdeplot(syn_subj_acc, ax=ax, label='Synthetic', color='coral', fill=True, alpha=0.3)

    ax.set_xlabel('Per-Participant Accuracy')
    ax.set_ylabel('Density')
    ax.set_title(f'Accuracy Distribution: Real vs Synthetic ({model_name})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_distribution_comparison.png', dpi=150)
    plt.close()

    print(f"Saved comparison plots to: {output_dir}")
