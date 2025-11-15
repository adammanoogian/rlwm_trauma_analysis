"""
Analysis Module for RLWM Trauma Study

This module contains functions for analyzing behavioral data and generating plots.
"""

from .plotting_utils import (
    setup_plot_style,
    save_figure,
    aggregate_by_condition,
    plot_line_with_error
)

from .behavioral_plots import (
    plot_accuracy_by_setsize,
    plot_learning_curves,
    plot_post_reversal_learning
)

__all__ = [
    'setup_plot_style',
    'save_figure',
    'aggregate_by_condition',
    'plot_line_with_error',
    'plot_accuracy_by_setsize',
    'plot_learning_curves',
    'plot_post_reversal_learning',
]
