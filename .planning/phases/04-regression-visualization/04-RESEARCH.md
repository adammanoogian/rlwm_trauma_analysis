# Phase 4: Regression Visualization - Research

**Researched:** 2026-02-05
**Domain:** Python scientific visualization (matplotlib, seaborn) for regression analysis
**Confidence:** HIGH

## Summary

This phase extends existing regression analysis scripts (15-16) to support:
1. Model M3 (WM-RL with perseveration parameter κ)
2. Flexible color-by grouping for all scatter/regression plots
3. Organized output structure with model-specific subdirectories

The codebase follows a well-established pattern: numbered pipeline scripts (01-16) that use shared plotting configuration and parameter definitions. Script 15 currently supports Q-learning and WM-RL (M2), while Script 16 has hardcoded model selection. Both use hardcoded GROUP_COLORS for trauma groups.

**Primary recommendation:** Create shared utility function for color-by grouping, extend both scripts to handle M3 using existing parameter definitions, organize output by model subdirectories.

## Standard Stack

The established libraries/tools for this domain:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| matplotlib | 3.x | Base plotting | Python standard, full control |
| seaborn | 0.x | Statistical visualization | Built on matplotlib, consistent styling |
| pandas | 1.x/2.x | Data manipulation | DataFrame operations for merging parameters + scales |
| numpy | 1.x | Numerical operations | Array operations for statistics |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| statsmodels | 0.x | Regression models | Already used in both scripts for OLS |
| scipy.stats | 1.x | Statistical tests | Correlation, p-values |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| matplotlib | plotly | Interactive but adds dependency, inconsistent with existing figures |
| seaborn | native matplotlib | More control but loses statistical defaults |

**Installation:**
```bash
# Already installed in project (verified from imports)
# No new dependencies required
```

## Architecture Patterns

### Current Script 15 Structure
```
scripts/15_analyze_mle_by_trauma.py
├── load_data()              # Loads MLE fits + surveys + groups
├── Statistical functions    # Mann-Whitney, Spearman, OLS
│   ├── mann_whitney_with_effect_size()
│   ├── group_comparisons()
│   ├── spearman_correlations()
│   ├── ols_regression()
│   └── ols_regression_extended()
├── Plotting functions       # All hardcode GROUP_COLORS
│   ├── plot_parameters_by_group()      # Violin + swarm plots
│   ├── plot_correlation_heatmap()      # Spearman rho matrix (WM-RL only)
│   ├── plot_forest_group_means()       # Error bars by group (WM-RL only)
│   └── plot_key_scatter()              # Scatter with regression lines (WM-RL only)
└── main()                   # Runs Q-learning, then WM-RL, saves to output/mle/
```

**Key observations:**
- Line 86-92: GROUP_COLORS hardcoded dict matching plotting_config.py
- Lines 493-544: plot_parameters_by_group() uses GROUP_COLORS directly
- Lines 614-663: plot_key_scatter() uses GROUP_COLORS for point colors
- Lines 547-576: plot_correlation_heatmap() only for WM-RL (param order hardcoded)
- Lines 579-611: plot_forest_group_means() only for WM-RL
- Lines 769-785: Only WM-RL gets heatmap, forest plot, scatter plots
- No M3 support: WMRL_PARAMS list excludes kappa

### Current Script 16 Structure
```
scripts/16_regress_parameters_on_scales.py
├── load_integrated_data()              # Loads parameters + merges with surveys
├── run_simple_regression()             # OLS with statsmodels
├── run_multiple_regression()           # Multiple predictors
├── Formatting functions
│   ├── format_pvalue()                 # Add significance stars
│   └── format_label()                  # Human-readable parameter names
├── Plotting functions
│   ├── plot_regression_scatter()       # Individual param ~ predictor (no color-by)
│   └── plot_regression_matrix()        # Grid of all regressions (color by p-value)
├── create_regression_table()           # CSV output with FDR correction
└── main()                              # CLI with --model qlearning|wmrl
```

**Key observations:**
- Lines 611-614: --model flag only accepts 'qlearning' or 'wmrl'
- Lines 433-494: plot_regression_scatter() has NO color-by grouping (uniform scatter)
- Lines 520-597: plot_regression_matrix() colors by significance, not by group
- Lines 655-660: param_cols selection hardcoded for qlearning vs wmrl
- No output subdirectories: saves to flat output/regressions/
- No M3 support in --model choices

### M3 Parameter Definitions (from mle_utils.py)

**File:** `scripts/fitting/mle_utils.py`

```python
# Lines 45-54: WMRL_M3_BOUNDS
WMRL_M3_BOUNDS = {
    'alpha_pos': (0.001, 0.999),
    'alpha_neg': (0.001, 0.999),
    'phi': (0.001, 0.999),
    'rho': (0.001, 0.999),
    'capacity': (1.0, 7.0),
    'kappa': (0.0, 1.0),      # Perseveration parameter
    'epsilon': (0.001, 0.999),
}

# Lines 57-61: WMRL_M3_PARAMS (CRITICAL: Order matches likelihood signature)
WMRL_M3_PARAMS = ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'epsilon']
```

**M3 fit file pattern:**
- Filename: `output/mle/wmrl_m3_individual_fits.csv`
- Columns verified (from head command above):
  - participant_id, alpha_pos, alpha_neg, phi, rho, capacity, kappa, epsilon
  - nll, aic, bic, aicc, pseudo_r2, n_trials, converged
  - n_successful_starts, n_near_best, at_bounds

**Display name for kappa:**
- Line 106-112 (Script 15): PARAM_NAMES dict does NOT include kappa
- Need to add: `'kappa': r'$\kappa$'` (Greek kappa for perseveration)

## Shared Utilities Assessment

### Existing: plotting_config.py (PROJECT_ROOT/plotting_config.py)

**Structure:**
```python
class PlotConfig:
    # Font sizes (lines 28-42)
    TITLE_SIZE = 18
    AXIS_LABEL_SIZE = 16
    TICK_LABEL_SIZE = 14
    LEGEND_SIZE = 13

    # Colors (lines 64-79)
    GROUP_COLORS = {
        'control': '#06A77D',
        'exposed': '#F18F01',
        'symptomatic': '#D62246',
        # ... (different keys than Script 15!)
    }

    @classmethod
    def apply_defaults(cls, use_latex=False):
        # Apply matplotlib rcParams
```

**Key observations:**
- Lines 72-79: GROUP_COLORS dict exists but with DIFFERENT KEYS than Script 15
  - PlotConfig keys: 'control', 'exposed', 'symptomatic'
  - Script 15 keys: 'No Trauma', 'Trauma-No Impact', 'Trauma-Ongoing Impact'
- Script 15 redefines GROUP_COLORS (line 86-92) with project-specific keys
- Scripts 15 and 16 both import PlotConfig for sizing only, not colors

### No Existing Color-By Utility

**What scripts/utils/ contains:**
- data_cleaning.py - Data quality checks
- scoring_functions.py - Scale scoring
- statistical_tests.py - Statistical test helpers
- update_participant_mapping.py - ID mapping
- sync_experiment_data.py - Data sync
- remap_mle_ids.py - ID remapping

**None provide color-by functionality.**

### Recommended: New Utility in scripts/utils/

**Why create shared utility:**
1. Both scripts need identical color-by logic (avoid duplication)
2. Scripts 15 and 16 have different plot types but same grouping needs
3. Future scripts may need color-by (e.g., Script 17+ in phase planning)
4. Centralized legend/palette generation reduces maintenance

**Where to place:**
- `scripts/utils/plotting_utils.py` - New file for shared plot helpers

**What it should provide:**
```python
# scripts/utils/plotting_utils.py

def get_color_palette(data_df, color_by, custom_colors=None):
    """
    Generate color palette for categorical grouping.

    Args:
        data_df: DataFrame with grouping column
        color_by: Column name to color by
        custom_colors: Optional dict mapping category -> color hex

    Returns:
        dict: {category: color_hex}
    """
    pass

def add_colored_scatter(ax, x, y, data_df, color_by, palette, **kwargs):
    """
    Add scatter plot with group coloring and legend.

    Args:
        ax: matplotlib axes
        x, y: Column names or arrays
        data_df: DataFrame
        color_by: Column name for grouping
        palette: Color palette dict from get_color_palette()
        **kwargs: Passed to ax.scatter()
    """
    pass
```

## Don't Hand-Roll

Problems that look simple but have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Automatic color palette for N categories | Manual color lists | matplotlib.cm.get_cmap() with .N colors | Handles arbitrary N, colorblind-safe options |
| Legend for many categories | Manual legend entries | ax.legend(ncol=2, loc='best') with bbox_to_anchor | Matplotlib auto-positions multi-column legends |
| Parameter label formatting | String replacement | Existing PARAM_NAMES dict + format_label() | Already handles Greek letters, parenthetical descriptions |
| Model parameter selection | If/else chains | Existing get_n_params(), model_utils lookups | Centralized in mle_utils.py |

**Key insight:** Matplotlib handles edge cases (overlapping legends, many categories, colorblind palettes) better than custom solutions. Use seaborn's categorical color palettes for automatic distinct colors.

## Common Pitfalls

### Pitfall 1: Inconsistent Color Mapping Across Plots
**What goes wrong:** Same category gets different colors in different plots
**Why it happens:** Regenerating palette per plot without consistent ordering
**How to avoid:**
- Generate palette ONCE at start of script
- Pass same palette dict to all plotting functions
- Use OrderedDict or sort categories alphabetically
**Warning signs:** User reports "red means X in this plot but Y in that plot"

### Pitfall 2: Legend Overflow with Many Categories
**What goes wrong:** Legend covers data with >10 categories
**Why it happens:** Default legend positioning assumes few categories
**How to avoid:**
- Use ncol=2 or ncol=3 for >5 categories
- Use bbox_to_anchor=(1.05, 1) to place outside plot
- Consider dropping legend if >15 categories (use colorbar instead)
**Warning signs:** Plots are unreadable, legend overlaps scatter points

### Pitfall 3: Hardcoded Parameter Lists Break with M3
**What goes wrong:** Script 15 plots fail when M3 has 7 parameters (WMRL_PARAMS has 6)
**Why it happens:** Violin plots assume fixed 3x2 grid (line 503: `ncols = 3`)
**How to avoid:**
- Use `len(params)` to determine grid size dynamically
- Already implemented in Script 15 (lines 502-503)
- Verify subplot layout math: `nrows = int(np.ceil(n_params / ncols))`
**Warning signs:** IndexError when accessing axes[row, col]

### Pitfall 4: Model Name Mismatch in File Paths
**What goes wrong:** Script looks for 'wmrl_m3' but files are 'wmrl-m3' or 'WMRL_M3'
**Why it happens:** Inconsistent naming conventions across modules
**How to avoid:**
- Use exact string from mle_utils.py: 'wmrl_m3' (lowercase, underscore)
- Verified pattern: `output/mle/wmrl_m3_individual_fits.csv` (line from glob output)
- Never use hyphens or uppercase in model identifiers
**Warning signs:** FileNotFoundError even though file exists

### Pitfall 5: Color-By Column Not in Merged Data
**What goes wrong:** User specifies --color-by gender but gender not merged into analysis DataFrame
**Why it happens:** Scripts load MLE fits + surveys separately, merge may drop columns
**How to avoid:**
- Script 16 already loads participant_surveys.csv (line 159)
- Extend merge_cols list to include demographics (age, gender, education)
- Validate color_by column exists before plotting: `if color_by not in df.columns: raise ValueError(...)`
**Warning signs:** KeyError on df[color_by]

## Code Examples

Verified patterns from existing scripts:

### Pattern 1: Loading Model Parameters (Script 15, lines 115-153)
```python
# Source: scripts/15_analyze_mle_by_trauma.py

def load_data() -> tuple:
    """Load and merge MLE fits with survey/group data."""
    # Load survey data
    surveys = pd.read_csv(OUTPUT_DIR / "participant_surveys.csv")
    groups = pd.read_csv(PROJECT_ROOT / "output" / "trauma_groups" / "group_assignments.csv")

    # Convert IDs to string for consistent merging
    surveys['sona_id'] = surveys['sona_id'].astype(str)
    groups['sona_id'] = groups['sona_id'].astype(str)

    # Load MLE fits
    qlearning = pd.read_csv(OUTPUT_DIR / "qlearning_individual_fits.csv")
    wmrl = pd.read_csv(OUTPUT_DIR / "wmrl_individual_fits.csv")

    # Convert participant_id to string
    qlearning['participant_id'] = qlearning['participant_id'].astype(str)
    wmrl['participant_id'] = wmrl['participant_id'].astype(str)

    # Merge with surveys
    qlearning = qlearning.merge(
        surveys, left_on='participant_id', right_on='sona_id', how='inner'
    )
    wmrl = wmrl.merge(
        surveys, left_on='participant_id', right_on='sona_id', how='inner'
    )

    return qlearning, wmrl, surveys, groups
```

**Extension for M3:**
```python
# Add to load_data():
wmrl_m3 = pd.read_csv(OUTPUT_DIR / "wmrl_m3_individual_fits.csv")
wmrl_m3['participant_id'] = wmrl_m3['participant_id'].astype(str)
wmrl_m3 = wmrl_m3.merge(surveys, left_on='participant_id', right_on='sona_id', how='inner')
return qlearning, wmrl, wmrl_m3, surveys, groups
```

### Pattern 2: Color-By Grouping (NEW - Recommended Implementation)
```python
# Source: NEW - scripts/utils/plotting_utils.py (to be created)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_color_palette(data_df, color_by, custom_colors=None):
    """
    Generate consistent color palette for categorical grouping.

    Args:
        data_df: DataFrame with grouping column
        color_by: Column name to color by (must be categorical)
        custom_colors: Optional dict {category: color_hex}

    Returns:
        dict: {category: color_hex}

    Raises:
        ValueError: If color_by column not in DataFrame
    """
    if color_by not in data_df.columns:
        raise ValueError(f"Column '{color_by}' not found. Available: {list(data_df.columns)}")

    # Get unique categories (sorted for consistency)
    categories = sorted(data_df[color_by].dropna().unique())

    # Use custom colors if provided
    if custom_colors is not None:
        return {cat: custom_colors.get(cat, '#808080') for cat in categories}

    # Otherwise generate palette
    n_colors = len(categories)

    if n_colors <= 10:
        # Use seaborn's tab10 for up to 10 categories
        colors = sns.color_palette('tab10', n_colors=n_colors)
    else:
        # Use continuous colormap for many categories
        cmap = plt.cm.get_cmap('hsv', n_colors)
        colors = [cmap(i) for i in range(n_colors)]

    # Convert to hex
    palette = {cat: plt.matplotlib.colors.rgb2hex(colors[i])
               for i, cat in enumerate(categories)}

    return palette


def add_colored_scatter(ax, x, y, data_df, color_by, palette,
                        alpha=0.7, s=50, show_legend=True, **kwargs):
    """
    Add scatter plot with categorical coloring.

    Args:
        ax: Matplotlib axes object
        x: Column name or array for x-axis
        y: Column name or array for y-axis
        data_df: DataFrame with data
        color_by: Column name for grouping
        palette: Color palette dict from get_color_palette()
        alpha: Point transparency (default: 0.7)
        s: Point size (default: 50)
        show_legend: Whether to add legend (default: True)
        **kwargs: Additional arguments for ax.scatter()

    Returns:
        List of PathCollection objects (one per category)
    """
    # Get x, y data
    if isinstance(x, str):
        x_data = data_df[x].values
    else:
        x_data = x

    if isinstance(y, str):
        y_data = data_df[y].values
    else:
        y_data = y

    # Group by category and plot
    handles = []
    categories = sorted(data_df[color_by].dropna().unique())

    for category in categories:
        mask = data_df[color_by] == category
        h = ax.scatter(x_data[mask], y_data[mask],
                      c=palette[category], label=category,
                      alpha=alpha, s=s, **kwargs)
        handles.append(h)

    # Add legend
    if show_legend:
        n_cats = len(categories)
        ncol = 1 if n_cats <= 5 else 2 if n_cats <= 10 else 3

        if n_cats <= 15:
            ax.legend(ncol=ncol, loc='best', fontsize=10)
        else:
            # Too many categories - skip legend
            pass

    return handles
```

### Pattern 3: Dynamic Parameter Grid Layout (Script 15, lines 493-544)
```python
# Source: scripts/15_analyze_mle_by_trauma.py

def plot_parameters_by_group(df: pd.DataFrame, params: list, model_name: str,
                             figsize: tuple = None) -> plt.Figure:
    """Create violin + swarm plots of parameters by trauma group."""
    # Calculate grid size dynamically
    n_params = len(params)
    ncols = 3
    nrows = int(np.ceil(n_params / ncols))

    if figsize is None:
        figsize = (4 * ncols, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_2d(axes)  # Ensure 2D even if single row

    # Plot each parameter
    for idx, param in enumerate(params):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        # ... plotting code ...

    # Remove empty subplots
    for idx in range(n_params, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    plt.tight_layout()
    return fig
```

### Pattern 4: Model-Specific Output Directories (NEW - Recommended)
```python
# Source: NEW - Extension for both scripts

def create_output_dirs(model_name: str, base_dir: Path) -> dict:
    """
    Create model-specific output directories.

    Args:
        model_name: 'qlearning', 'wmrl', or 'wmrl_m3'
        base_dir: Base output directory (e.g., output/regressions)

    Returns:
        dict with keys: 'csv', 'figures' (both are Path objects)
    """
    dirs = {
        'csv': base_dir / model_name,
        'figures': base_dir / model_name / 'figures'
    }

    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)

    return dirs

# Usage in Script 16 main():
output_dirs = create_output_dirs(args.model, Path('output/regressions'))
# Save CSVs to output_dirs['csv']
# Save PNGs to output_dirs['figures']
```

### Pattern 5: Parameter Display Names with M3 (Extension)
```python
# Source: Extension of scripts/15_analyze_mle_by_trauma.py, lines 104-112

PARAM_NAMES = {
    'alpha_pos': r'$\alpha_+$',
    'alpha_neg': r'$\alpha_-$',
    'epsilon': r'$\varepsilon$',
    'phi': r'$\phi$',
    'rho': r'$\rho$',
    'capacity': 'K',
    'kappa': r'$\kappa$',  # ADD THIS for M3 perseveration
}
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Hardcoded trauma groups | --color-by any categorical | Phase 4 (this phase) | Flexible visualization for different analyses |
| Q-learning + WM-RL only | Q-learning + WM-RL + M3 | Phase 4 (this phase) | Supports perseveration model |
| Flat output directory | Model subdirectories | Phase 4 (this phase) | Organized outputs, no overwrites |
| Script 15 loops internally | --model all flag | Phase 4 (this phase) | Explicit control over which models to analyze |

**Deprecated/outdated:**
- Script 15 GROUP_COLORS dict (line 86-92): Will be supplemented with dynamic palette from plotting_utils
- Script 16 hardcoded --model choices (line 614): Will add 'wmrl_m3' and 'all'

## Open Questions

Things that couldn't be fully resolved:

1. **Gender column availability**
   - What we know: parsed_demographics.csv exists with 'gender' column, but current data shows all NaN
   - What's unclear: Whether gender data will be populated for final analysis
   - Recommendation: Implement --color-by gender anyway (validates column exists), warn if all NaN

2. **Default color palette for trauma groups**
   - What we know: Script 15 uses specific hex colors for 3 trauma groups
   - What's unclear: Whether to preserve these colors when --color-by trauma_group, or use auto-palette
   - Recommendation: Add custom_colors parameter to plotting_utils, pass Script 15 GROUP_COLORS when color-by is 'hypothesis_group'

3. **Script 15 --model all behavior**
   - What we know: Current main() runs qlearning then wmrl sequentially (no CLI flag)
   - What's unclear: Whether --model all should save combined CSVs or separate per model
   - Recommendation: Separate per model (avoids confusion, matches Script 16 pattern where --model selects data file)

## Sources

### Primary (HIGH confidence)
- **Direct file reads:**
  - C:\Users\aman0087\Documents\Github\rlwm_trauma_analysis\scripts\15_analyze_mle_by_trauma.py (826 lines)
  - C:\Users\aman0087\Documents\Github\rlwm_trauma_analysis\scripts\16_regress_parameters_on_scales.py (842 lines)
  - C:\Users\aman0087\Documents\Github\rlwm_trauma_analysis\scripts\fitting\mle_utils.py (1025 lines)
  - C:\Users\aman0087\Documents\Github\rlwm_trauma_analysis\plotting_config.py (234 lines)

- **CSV file structure verification (head commands):**
  - output/mle/qlearning_individual_fits.csv (columns verified)
  - output/mle/wmrl_individual_fits.csv (columns verified)
  - output/mle/wmrl_m3_individual_fits.csv (columns verified, includes kappa)
  - output/mle/participant_surveys.csv (trauma scales)
  - output/trauma_groups/group_assignments.csv (hypothesis_group column)
  - output/parsed_demographics.csv (gender, age, etc.)

- **Parameter definitions from mle_utils.py:**
  - WMRL_M3_PARAMS list (line 61): ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'epsilon']
  - WMRL_M3_BOUNDS dict (lines 45-54): kappa bounds (0.0, 1.0)

### Secondary (MEDIUM confidence)
- **Matplotlib/seaborn best practices:** Based on existing codebase patterns (not web-sourced)
- **Color palette recommendations:** Seaborn documentation (tab10, hsv palettes) - standard practice
- **Legend positioning:** Matplotlib documentation (bbox_to_anchor) - standard practice

### Tertiary (LOW confidence)
- None - all findings based on direct code analysis

## Metadata

**Confidence breakdown:**
- Script 15 architecture: HIGH - Complete code read, all functions analyzed
- Script 16 architecture: HIGH - Complete code read, all functions analyzed
- M3 parameter definitions: HIGH - Direct verification in mle_utils.py and CSV files
- Shared utilities: HIGH - Confirmed plotting_utils.py does not exist, need to create
- Available categorical columns: HIGH - Direct CSV reads confirm columns
- Color-by utility design: MEDIUM - Pattern is standard but not yet implemented in this codebase

**Research date:** 2026-02-05
**Valid until:** 30 days (stable codebase, no fast-moving dependencies)

**Implementation readiness:**
- All parameter definitions confirmed in existing code
- File naming conventions verified (wmrl_m3, not wmrl-m3)
- CSV column structure validated with head commands
- Plotting patterns established from Script 15
- No blocking unknowns - ready for planning phase
