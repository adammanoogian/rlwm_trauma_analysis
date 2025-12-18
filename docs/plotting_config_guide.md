# Plotting Configuration Usage Guide

## Overview

The `plotting_config.py` module provides centralized control over font sizes, colors, figure dimensions, and other plotting parameters to ensure consistency across all visualizations.

## Quick Start

### Import the configuration

```python
from plotting_config import PlotConfig, ScatterPlotConfig
```

### Using font sizes

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))

# Apply consistent font sizes
ax.set_title('My Title', fontsize=PlotConfig.TITLE_SIZE, fontweight='bold')
ax.set_xlabel('X Label', fontsize=PlotConfig.AXIS_LABEL_SIZE)
ax.set_ylabel('Y Label', fontsize=PlotConfig.AXIS_LABEL_SIZE)
ax.legend(fontsize=PlotConfig.LEGEND_SIZE)
ax.tick_params(labelsize=PlotConfig.TICK_LABEL_SIZE)

# Add annotations
ax.text(0.5, 0.5, 'Annotation', fontsize=PlotConfig.ANNOTATION_SIZE)
```

### Using scatter plot settings

```python
ax.scatter(x, y, 
           s=ScatterPlotConfig.SIZE,
           alpha=ScatterPlotConfig.ALPHA,
           edgecolors=ScatterPlotConfig.EDGE_COLOR,
           linewidth=ScatterPlotConfig.EDGE_WIDTH)
```

## Available Configuration Classes

### PlotConfig (Base Configuration)

**Font Sizes:**
- `TITLE_SIZE = 18` - Main plot titles
- `SUBTITLE_SIZE = 14` - Subplot titles
- `AXIS_LABEL_SIZE = 16` - X/Y axis labels
- `TICK_LABEL_SIZE = 14` - Tick labels
- `LEGEND_SIZE = 13` - Legend text
- `ANNOTATION_SIZE = 12` - Text annotations
- `SMALL_TEXT_SIZE = 10` - Small notes

**Figure Dimensions:**
- `SMALL_FIG = (8, 6)`
- `MEDIUM_FIG = (10, 8)`
- `LARGE_FIG = (12, 10)`
- `WIDE_FIG = (14, 6)`
- `TALL_FIG = (8, 12)`

**DPI Settings:**
- `DPI_SCREEN = 100` - Display
- `DPI_PRINT = 300` - High-quality export

**Colors:**
- `COLOR_PRIMARY`, `COLOR_SECONDARY`, `COLOR_ACCENT`
- `GROUP_COLORS` - Dictionary for trauma group colors

**Other:**
- `PAD = 20` - Title padding
- `GRID_ALPHA = 0.3` - Grid transparency
- `LINE_WIDTH = 2.0` - Default line width

### ScatterPlotConfig

- `SIZE = 100` - Marker size
- `ALPHA = 0.7` - Marker transparency
- `EDGE_WIDTH = 1.5` - Marker edge width
- `EDGE_COLOR = 'white'` - Marker edge color

### TimeSeriesConfig

- `LINE_WIDTH = 2.0`
- `MARKER_SIZE = 8`
- `ERROR_ALPHA = 0.2` - Error band transparency

### BarPlotConfig

- `WIDTH = 0.8` - Bar width
- `EDGE_WIDTH = 1.0` - Bar edge width
- `ERROR_WIDTH = 2.0` - Error bar width
- `CAPSIZE = 5` - Error bar cap size

### HeatmapConfig

- `ANNOT_SIZE = 12` - Annotation size
- `CBAR_LABEL_SIZE = 16` - Colorbar label size
- `CMAP = 'RdBu_r'` - Default colormap
- `CENTER = 0` - Center value for diverging colormap

## Advanced Usage

### Apply all defaults to matplotlib

```python
from plotting_config import PlotConfig

# Apply all default settings at once
PlotConfig.apply_defaults()

# Now all plots will use the configured sizes automatically
fig, ax = plt.subplots()
ax.set_title('Title')  # Will use default title size
```

### Create figures with standard sizes

```python
# Create a medium-sized figure with standard settings
fig, ax = PlotConfig.create_figure(size='medium')

# Available sizes: 'small', 'medium', 'large', 'wide', 'tall'
fig, ax = PlotConfig.create_figure(size='wide', nrows=1, ncols=2)
```

### Get font size dictionary

```python
# Get all font sizes as a dictionary
fontsizes = PlotConfig.get_fontsize_dict()
# Returns: {'title': 18, 'label': 16, 'tick': 14, 'legend': 13, ...}

# Use in function calls
plt.text(x, y, 'Label', fontsize=fontsizes['annotation'])
```

## Modifying Default Values

To change defaults for your entire project, edit `plotting_config.py`:

```python
class PlotConfig:
    # Make titles even larger
    TITLE_SIZE = 20
    
    # Change default colormap
    COLOR_PRIMARY = '#3498db'
    
    # Adjust marker sizes
    MARKER_SIZE = 150
```

## Integration Status

Currently integrated in:
- ✅ `scripts/analysis/trauma_grouping_analysis.py` - hypothesis_groups_scatter.png

**Not yet integrated** (ready for future updates):
- `scripts/analysis/visualize_human_performance.py`
- `scripts/analysis/visualize_scale_correlations.py`
- `scripts/visualization/quick_arviz_plots.py`
- Other visualization scripts

## Example: Before and After

**Before (hardcoded values):**
```python
ax.set_xlabel('X Label', fontsize=14, fontweight='bold')
ax.set_ylabel('Y Label', fontsize=14, fontweight='bold')
ax.set_title('Title', fontsize=16, fontweight='bold', pad=20)
ax.legend(fontsize=10)
ax.scatter(x, y, s=150, alpha=0.7, edgecolors='black', linewidth=1.5)
```

**After (using PlotConfig):**
```python
ax.set_xlabel('X Label', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
ax.set_ylabel('Y Label', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
ax.set_title('Title', fontsize=PlotConfig.TITLE_SIZE, fontweight='bold', pad=PlotConfig.PAD)
ax.legend(fontsize=PlotConfig.LEGEND_SIZE)
ax.scatter(x, y, s=ScatterPlotConfig.SIZE, alpha=ScatterPlotConfig.ALPHA,
           edgecolors=ScatterPlotConfig.EDGE_COLOR, linewidth=ScatterPlotConfig.EDGE_WIDTH)
```

## Benefits

1. **Consistency**: All plots use the same font sizes and styling
2. **Easy updates**: Change one value to update all plots
3. **Readability**: Clear semantic names (TITLE_SIZE vs magic number 16)
4. **Flexibility**: Override on a per-plot basis when needed
5. **Documentation**: Settings are self-documenting with descriptive names
