# Folder Structure Update Summary

## Changes Made

All scripts and documentation have been updated to follow the standard folder structure:

### Output Locations

**Before:**
- Posteriors: `output/v1/qlearning_jax_posterior_*.nc`
- Figures: `output/v1/figures/`
- Summary CSVs: `output/v1/qlearning_jax_summary_*.csv`

**After:**
- Posteriors: `output/qlearning_jax_posterior_*.nc` ✅
- Figures: `figures/` ✅
- Summary CSVs: `output/qlearning_jax_summary_*.csv` ✅

---

## Updated Files

### 1. **Fitting Script**
- **File**: `scripts/fitting/fit_with_jax.py`
- **Changes**:
  - Default output directory: `OUTPUT_DIR` instead of `OUTPUT_VERSION_DIR`
  - Posteriors save directly to `output/`
  - Diagnostic plots save to `output/` (if --save-plots)

### 2. **Visualization Scripts**

#### **quick_arviz_plots.py**
- **Default output**: `figures/` instead of `output/v1/figures/`
- **All plots** (trace, posterior, forest, etc.) save to `figures/`

#### **plot_group_parameters.py**
- **Default output**: `figures/` instead of `output/v1/`
- Forest plots save to `figures/`

#### **plot_model_comparison.py**
- **Default output**: `figures/` instead of `output/v1/`
- Comparison plots save to `figures/`

### 3. **Documentation**

#### **PLOTTING_REFERENCE.md**
- Updated all example commands to use correct paths
- Removed specific file references (now uses `TIMESTAMP` placeholder)
- All examples show:
  - `output/qlearning_jax_posterior_TIMESTAMP.nc` for posteriors
  - `figures/` for all plots

#### **ANALYSIS_PIPELINE.md**
- Updated Stage 4B (JAX/NumPyro fitting) paths
- Updated Workflow 2B example commands
- All outputs now correctly reference `output/` and `figures/`

---

## Folder Structure

```
rlwm_trauma_analysis/
├── output/                           # All analysis outputs
│   ├── qlearning_jax_posterior_*.nc  # Model posteriors
│   ├── qlearning_jax_summary_*.csv   # Summary statistics
│   ├── task_trials_long.csv          # Processed data
│   ├── behavioral_summary/           # Behavioral analysis
│   ├── trauma_groups/                # Group analysis
│   └── v1/                           # Legacy (if needed)
│
├── figures/                          # All visualizations
│   ├── qlearning_jax_*_trace.png     # Diagnostic plots
│   ├── qlearning_jax_*_posterior.png # Parameter plots
│   ├── qlearning_jax_*_forest.png    # Forest plots
│   ├── behavioral_summary/           # Behavioral plots
│   └── trauma_groups/                # Group comparison plots
│
├── data/                             # Raw data (read-only)
├── scripts/                          # Analysis scripts
│   ├── fitting/                      # Model fitting
│   └── visualization/                # Plotting scripts
└── docs/                             # Documentation
```

---

## Usage

### Run Fitting
```bash
# Fit model - saves to output/
python scripts/fitting/fit_with_jax.py \
    --data output/task_trials_long_all_participants.csv \
    --chains 4 \
    --warmup 1000 \
    --samples 2000 \
    --save-plots
```

**Output:**
- `output/qlearning_jax_posterior_TIMESTAMP.nc`
- `output/qlearning_jax_summary_TIMESTAMP.csv`
- `output/qlearning_jax_trace_TIMESTAMP.png` (if --save-plots)
- `output/qlearning_jax_posterior_TIMESTAMP.png` (if --save-plots)

### Create Visualizations
```bash
# Generate all diagnostic plots - saves to figures/
python scripts/visualization/quick_arviz_plots.py \
    --posterior output/qlearning_jax_posterior_TIMESTAMP.nc

# Group parameters - saves to figures/
python scripts/visualization/plot_group_parameters.py \
    --qlearning output/qlearning_jax_posterior_TIMESTAMP.nc \
    --prefix qlearning
```

**Output:**
- `figures/qlearning_jax_TIMESTAMP_trace.png`
- `figures/qlearning_jax_TIMESTAMP_posterior.png`
- `figures/qlearning_jax_TIMESTAMP_forest.png`
- `figures/qlearning_jax_TIMESTAMP_rank.png`
- `figures/qlearning_jax_TIMESTAMP_autocorr.png`
- `figures/qlearning_jax_TIMESTAMP_energy.png`
- `figures/qlearning_jax_TIMESTAMP_pair.png`
- `figures/qlearning_group_parameters_forest.png`

---

## Benefits

✅ **Cleaner structure**: No nested `v1` subdirectories
✅ **Consistent organization**: All outputs in `output/`, all figures in `figures/`
✅ **Easier navigation**: Simpler paths, less typing
✅ **Version control ready**: Can still use `v1`, `v2` folders if needed for major versions
✅ **Documentation aligned**: All docs match actual folder structure

---

## Migration (If Needed)

If you have existing results in `output/v1/`, you can optionally move them:

```bash
# Move posteriors
mv output/v1/qlearning_jax_posterior_*.nc output/

# Move figures
mv output/v1/figures/* figures/

# Keep v1 as backup
# (or delete if you don't need it)
```

---

## Config Note

The `config.py` still has `VERSION = 'v1'` which creates `OUTPUT_VERSION_DIR` and `FIGURES_VERSION_DIR` variables. These are used by some older scripts but **not** by the JAX fitting or visualization scripts.

If you want to completely remove version subdirectories:
1. Set `VERSION = None` or `VERSION = ''` in `config.py`
2. Or just ignore the version variables (JAX scripts already do this)

---

## Questions?

See:
- `docs/PLOTTING_REFERENCE.md` - Complete plotting guide
- `docs/ANALYSIS_PIPELINE.md` - Full analysis workflow
- `docs/MODEL_REFERENCE.md` - Model specifications
