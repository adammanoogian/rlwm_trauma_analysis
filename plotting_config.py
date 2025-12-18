"""
Plotting configuration for consistent visualization across the project.

This module provides centralized control over font sizes, colors, DPI settings,
and other plotting parameters to ensure consistency across all figures.

Usage:
    from config.plotting_config import PlotConfig
    
    # Apply to matplotlib
    PlotConfig.apply_defaults()
    
    # Or use specific sizes
    plt.title('My Title', fontsize=PlotConfig.TITLE_SIZE)
"""

import matplotlib.pyplot as plt
from typing import Dict, Any


class PlotConfig:
    """Centralized plotting configuration."""
    
    # ============================================================================
    # FONT SIZES
    # ============================================================================
    
    # Title sizes
    TITLE_SIZE = 18          # Main plot title
    SUBTITLE_SIZE = 14       # Subplot titles
    SUPTITLE_SIZE = 20       # Figure super title
    
    # Axis label sizes
    AXIS_LABEL_SIZE = 16     # X and Y axis labels
    TICK_LABEL_SIZE = 14     # Tick labels on axes
    
    # Legend and annotation sizes
    LEGEND_SIZE = 13         # Legend text
    ANNOTATION_SIZE = 12     # Text annotations, quadrant labels
    
    # Small text
    SMALL_TEXT_SIZE = 10     # Small annotations, notes
    
    # ============================================================================
    # FIGURE DIMENSIONS
    # ============================================================================
    
    # Standard figure sizes (width, height) in inches
    SMALL_FIG = (8, 6)
    MEDIUM_FIG = (10, 8)
    LARGE_FIG = (12, 10)
    WIDE_FIG = (14, 6)
    TALL_FIG = (8, 12)
    
    # DPI settings
    DPI_SCREEN = 100         # For display
    DPI_PRINT = 300          # For publication/high-quality export
    
    # ============================================================================
    # COLORS
    # ============================================================================
    
    # Color palette (can be customized per project needs)
    COLOR_PRIMARY = '#2E86AB'      # Blue
    COLOR_SECONDARY = '#A23B72'    # Purple
    COLOR_ACCENT = '#F18F01'       # Orange
    COLOR_SUCCESS = '#06A77D'      # Green
    COLOR_WARNING = '#D62246'      # Red
    COLOR_NEUTRAL = '#6C757D'      # Gray
    
    # Group colors (for trauma analysis)
    GROUP_COLORS = {
        'control': '#06A77D',           # Green
        'exposed': '#F18F01',            # Orange  
        'symptomatic': '#D62246',        # Red
        'low_exposure': '#06A77D',       # Green
        'high_exposure_low_symptoms': '#F18F01',  # Orange
        'high_exposure_high_symptoms': '#D62246'  # Red
    }
    
    # ============================================================================
    # LINE AND MARKER STYLES
    # ============================================================================
    
    LINE_WIDTH = 2.0
    MARKER_SIZE = 100        # For scatter plots
    MARKER_ALPHA = 0.7
    GRID_ALPHA = 0.3
    
    # ============================================================================
    # LAYOUT
    # ============================================================================
    
    PAD = 20                 # Padding for titles
    TIGHT_LAYOUT = True      # Use tight_layout by default
    
    # ============================================================================
    # METHODS
    # ============================================================================
    
    @classmethod
    def apply_defaults(cls, use_latex: bool = False):
        """
        Apply default plotting settings to matplotlib.
        
        Parameters
        ----------
        use_latex : bool, default=False
            Whether to use LaTeX for text rendering (requires LaTeX installation)
        """
        params = {
            # Font sizes
            'font.size': cls.TICK_LABEL_SIZE,
            'axes.titlesize': cls.TITLE_SIZE,
            'axes.labelsize': cls.AXIS_LABEL_SIZE,
            'xtick.labelsize': cls.TICK_LABEL_SIZE,
            'ytick.labelsize': cls.TICK_LABEL_SIZE,
            'legend.fontsize': cls.LEGEND_SIZE,
            'figure.titlesize': cls.SUPTITLE_SIZE,
            
            # Line widths
            'lines.linewidth': cls.LINE_WIDTH,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            
            # Grid
            'axes.grid': True,
            'grid.alpha': cls.GRID_ALPHA,
            
            # Figure
            'figure.dpi': cls.DPI_SCREEN,
            'savefig.dpi': cls.DPI_PRINT,
            'savefig.bbox': 'tight',
            
            # Font
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
        }
        
        if use_latex:
            params.update({
                'text.usetex': True,
                'font.family': 'serif',
                'font.serif': ['Computer Modern Roman'],
            })
        
        plt.rcParams.update(params)
    
    @classmethod
    def get_fontsize_dict(cls) -> Dict[str, int]:
        """
        Get dictionary of font sizes for easy parameter passing.
        
        Returns
        -------
        dict
            Dictionary with keys: 'title', 'label', 'tick', 'legend', 'annotation'
        """
        return {
            'title': cls.TITLE_SIZE,
            'subtitle': cls.SUBTITLE_SIZE,
            'label': cls.AXIS_LABEL_SIZE,
            'tick': cls.TICK_LABEL_SIZE,
            'legend': cls.LEGEND_SIZE,
            'annotation': cls.ANNOTATION_SIZE,
            'small': cls.SMALL_TEXT_SIZE,
        }
    
    @classmethod
    def create_figure(cls, size: str = 'medium', **kwargs) -> tuple:
        """
        Create a figure with standard settings.
        
        Parameters
        ----------
        size : str, default='medium'
            Size preset: 'small', 'medium', 'large', 'wide', 'tall'
        **kwargs
            Additional arguments passed to plt.subplots()
        
        Returns
        -------
        fig, ax
            Matplotlib figure and axes objects
        """
        size_map = {
            'small': cls.SMALL_FIG,
            'medium': cls.MEDIUM_FIG,
            'large': cls.LARGE_FIG,
            'wide': cls.WIDE_FIG,
            'tall': cls.TALL_FIG,
        }
        
        figsize = size_map.get(size, cls.MEDIUM_FIG)
        
        fig, ax = plt.subplots(figsize=figsize, **kwargs)
        return fig, ax


# ============================================================================
# PRESET CONFIGURATIONS FOR SPECIFIC PLOT TYPES
# ============================================================================

class ScatterPlotConfig:
    """Configuration specifically for scatter plots."""
    SIZE = PlotConfig.MARKER_SIZE
    ALPHA = PlotConfig.MARKER_ALPHA
    EDGE_WIDTH = 1.5
    EDGE_COLOR = 'white'


class TimeSeriesConfig:
    """Configuration for time series / line plots."""
    LINE_WIDTH = PlotConfig.LINE_WIDTH
    MARKER_SIZE = 8
    ERROR_ALPHA = 0.2


class BarPlotConfig:
    """Configuration for bar plots."""
    WIDTH = 0.8
    EDGE_WIDTH = 1.0
    EDGE_COLOR = 'black'
    ERROR_WIDTH = 2.0
    CAPSIZE = 5


class HeatmapConfig:
    """Configuration for heatmaps / correlation matrices."""
    ANNOT_SIZE = PlotConfig.ANNOTATION_SIZE
    CBAR_LABEL_SIZE = PlotConfig.AXIS_LABEL_SIZE
    CMAP = 'RdBu_r'
    CENTER = 0
