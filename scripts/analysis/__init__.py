"""
Analysis Module for RLWM Trauma Study

This module provides shared plotting utilities used across multiple scripts.

Note: Library modules have been integrated into their parent numbered scripts.
Only plotting_utils.py remains as the true shared library (used by 4+ scripts).
"""

from .plotting_utils import (
    setup_plot_style,
    save_figure,
    aggregate_by_condition,
    plot_line_with_error
)

__all__ = [
    'setup_plot_style',
    'save_figure',
    'aggregate_by_condition',
    'plot_line_with_error',
]
