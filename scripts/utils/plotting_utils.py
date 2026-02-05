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


# Predefined trauma group colors (matching Script 15's GROUP_COLORS)
TRAUMA_GROUP_COLORS = {
    'No Trauma': '#06A77D',  # Green
    'Trauma-No Impact': '#F18F01',  # Orange
    'Trauma-Ongoing Impact': '#D62246',  # Red
    'Low Exposure-High Symptoms': '#6C757D',  # Gray (paradoxical)
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
