# Posterior Visualization Reference

Complete guide to visualizing fitted model posteriors.

## Quick Start: Comprehensive Diagnostics

**Activate environment first:**
```bash
conda activate ds_env
```

**Create all diagnostic plots at once:**
```bash
python scripts/visualization/quick_arviz_plots.py \
  --posterior output/qlearning_jax_posterior_TIMESTAMP.nc \
  --output-dir figures
```

**This creates 8+ plots in `figures/`:**
1. `qlearning_jax_TIMESTAMP_trace.png` - Trace plots (chains, mixing)
2. `qlearning_jax_TIMESTAMP_posterior.png` - Posterior distributions with 94% HDI
3. `qlearning_jax_TIMESTAMP_forest.png` - Forest plot (all parameters)
4. `qlearning_jax_TIMESTAMP_rank.png` - Rank plots (convergence)
5. `qlearning_jax_TIMESTAMP_autocorr.png` - Autocorrelation
6. `qlearning_jax_TIMESTAMP_energy.png` - Energy plot (NUTS diagnostic)
7. `qlearning_jax_TIMESTAMP_pair.png` - Parameter correlations
8. `qlearning_jax_TIMESTAMP_individual_forest.png` - Individual parameters
9. `qlearning_jax_TIMESTAMP_summary.csv` - Summary statistics

---

## All Visualization Scripts

All scripts live in `scripts/visualization/`. Run from the repo root.

| Script | Purpose | Primary input | Primary output |
|---|---|---|---|
| `create_modeling_figures.py` | Publication-quality MLE model figures (learning curves, set-size effects, parameter distributions) | `output/mle/` | `figures/modeling/` |
| `create_modeling_tables.py` | LaTeX/CSV tables for model results (AIC/BIC, parameter means) | `output/mle/`, `output/model_comparison/` | `output/tables/` |
| `create_parameter_behavior_heatmap.py` | Heatmap of parameter × behavioral correlations across models | `output/mle/`, `output/regressions/` | `figures/parameter_behavior_heatmap.png` |
| `create_publication_figures.py` | Composite figure panels for the manuscript | `output/mle/`, `figures/` | `figures/publication/` |
| `create_supplementary_materials.py` | Supplementary figures and tables (model recovery, parameter sweep) | `output/model_comparison/`, `output/mle/` | `figures/supplementary/`, `output/supplementary/` |
| `create_supplementary_table_s3.py` | Supplementary Table S3 (regression coefficients for all models) | `output/regressions/` | `output/tables/table_s3.csv` |
| `plot_group_parameters.py` | Forest plots of hierarchical group-level (mu) parameters | `output/bayesian/{model}_posterior.nc` | `figures/{prefix}_group_parameters_forest.png` |
| `plot_model_comparison.py` | WAIC/LOO bar chart + Akaike model weights (Bayesian comparison) | Two `.nc` posterior files | `figures/{prefix}_model_comparison_bar.png` |
| `plot_posterior_diagnostics.py` | Full MCMC diagnostic dashboard: trace, rank, energy, autocorr, pair | `output/bayesian/{model}_posterior.nc` | `figures/m6b_posterior_diagnostics.png` (or specified path) |
| `plot_wmrl_forest.py` | Forest plot of all WM-RL parameters from a single posterior | `output/bayesian/{model}_posterior.nc` | `figures/{prefix}_wmrl_forest.png` |
| `quick_arviz_plots.py` | 8+ diagnostic plots in one call (trace, posterior, forest, rank, autocorr, energy, pair, summary CSV) | Any `.nc` posterior file | `figures/{model}_{timestamp}_*.png` + summary CSV |

**When to use each:**
- **Before submitting results**: `create_publication_figures.py` + `create_modeling_tables.py`
- **After cluster Bayesian fit**: `plot_posterior_diagnostics.py`, `plot_group_parameters.py`, `plot_wmrl_forest.py`
- **For convergence check**: `quick_arviz_plots.py` (fastest, all diagnostics at once)
- **For model comparison (Bayesian)**: `plot_model_comparison.py`
- **For trauma-parameter signal**: `create_parameter_behavior_heatmap.py`

---

## Visualization Tools Overview

### 1. Group-Level Parameters (`plot_group_parameters.py`)

**Purpose**: Forest plots of population-level (μ) parameters

**Single Model:**
```bash
python scripts/visualization/plot_group_parameters.py \
  --qlearning output/qlearning_jax_posterior_TIMESTAMP.nc \
  --output-dir figures \
  --prefix qlearning
```

**Output in `figures/`:**
- `qlearning_group_parameters_forest.png`

**Features:**
- Shows μ_α+ (mean positive learning rate)
- Shows μ_α- (mean negative learning rate)
- Shows μ_β (mean inverse temperature)
- 94% HDI error bars
- Theoretical parameter bounds (shaded background)

**Two Models (Q-Learning + WM-RL):**
```bash
python scripts/visualization/plot_group_parameters.py \
  --qlearning output/qlearning_jax_posterior_TIMESTAMP.nc \
  --wmrl output/wmrl_jax_posterior_TIMESTAMP.nc \
  --output-dir figures \
  --prefix comparison
```

**Additional Output in `figures/`:**
- `comparison_group_parameters_comparison.png` - Side-by-side comparison of shared parameters

---

### 2. Model Comparison (`plot_model_comparison.py`)

**Purpose**: Compare Q-learning vs WM-RL using information criteria

**Requires both models fitted!**

```bash
python scripts/visualization/plot_model_comparison.py \
  --qlearning output/qlearning_jax_posterior_TIMESTAMP.nc \
  --wmrl output/wmrl_jax_posterior_TIMESTAMP.nc \
  --output-dir figures \
  --prefix model_comp
```

**Outputs in `figures/`:**
- `model_comp_model_comparison_bar.png` - WAIC/LOO comparison
- `model_comp_model_weights.png` - Akaike model weights
- `model_comp_goodness_of_fit.png` - Predicted vs observed (placeholder)

**Metrics Computed:**
- **WAIC** (Watanabe-Akaike Information Criterion)
- **LOO** (Leave-One-Out Cross-Validation)
- **Model weights** (higher = better)

---

### 3. ArviZ Built-in Plots (Comprehensive)

ArviZ provides 20+ plotting functions. Here are the most useful:

#### a) **Trace Plots** (Check convergence)

```python
import arviz as az
import matplotlib.pyplot as plt

idata = az.from_netcdf('output/qlearning_jax_posterior_TIMESTAMP.nc')

# Trace plots for group parameters
az.plot_trace(idata, var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'])
plt.savefig('figures/trace.png', dpi=300)
plt.show()
```

**What to look for:**
- ✅ Chains should overlap (good mixing)
- ✅ Stationary (no trends/drift)
- ❌ If chains don't overlap → poor convergence

---

#### b) **Posterior Distributions** (Parameter estimates)

```python
# Posterior distributions with HDI
az.plot_posterior(
    idata,
    var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'],
    hdi_prob=0.94
)
plt.savefig('figures/posterior.png', dpi=300)
plt.show()
```

**Shows:**
- Mean and 94% HDI
- Full posterior distribution (KDE)
- Useful for reporting parameter estimates

---

#### c) **Forest Plots** (Compare parameters)

```python
# Forest plot (all parameters at once)
az.plot_forest(
    idata,
    var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'],
    combined=True,
    hdi_prob=0.94
)
plt.savefig('figures/forest.png', dpi=300)
plt.show()
```

**Features:**
- Compact comparison
- Error bars = 94% HDI
- Good for papers/presentations

---

#### d) **Pair Plots** (Parameter correlations)

```python
# Pairwise correlations between parameters
az.plot_pair(
    idata,
    var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'],
    kind='kde',
    marginals=True
)
plt.savefig('figures/pair.png', dpi=300)
plt.show()
```

**What to look for:**
- Identify correlated parameters
- Check for multimodality
- Useful for understanding parameter space

---

#### e) **Rank Plots** (Convergence diagnostic)

```python
# Rank plots (should be uniform if converged)
az.plot_rank(idata, var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'])
plt.savefig('figures/rank.png', dpi=300)
plt.show()
```

**What to look for:**
- ✅ Uniform distribution (flat) = good convergence
- ❌ Non-uniform = convergence issues

---

#### f) **Energy Plot** (NUTS diagnostic)

```python
# Energy plot (specific to HMC/NUTS)
az.plot_energy(idata)
plt.savefig('figures/energy.png', dpi=300)
plt.show()
```

**What to look for:**
- ✅ Energy and energy transition distributions should overlap
- ❌ Separation → biased sampling

---

#### g) **Autocorrelation** (Sample independence)

```python
# Autocorrelation plots
az.plot_autocorr(idata, var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'])
plt.savefig('figures/autocorr.png', dpi=300)
plt.show()
```

**What to look for:**
- ✅ Rapid decay to 0 = good
- ❌ Slow decay = high autocorrelation (lower ESS)

---

### 4. Summary Statistics

**Quick Summary:**
```python
import arviz as az

idata = az.from_netcdf('output/qlearning_jax_posterior_TIMESTAMP.nc')
summary = az.summary(idata, var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'])
print(summary)
```

**Columns:**
- `mean`: Posterior mean
- `sd`: Standard deviation
- `hdi_3%`, `hdi_97%`: 94% highest density interval
- `r_hat`: Convergence diagnostic (should be < 1.01)
- `ess_bulk`, `ess_tail`: Effective sample size (should be > 400)

**Save to CSV:**
```python
summary.to_csv('output/summary.csv')
```

---

## Complete Python Script Examples

### Example 1: Quick Diagnostic Dashboard

```python
import arviz as az
import matplotlib.pyplot as plt

# Load posterior
idata = az.from_netcdf('output/qlearning_jax_posterior_TIMESTAMP.nc')

# Create 2x3 dashboard
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Trace plot
az.plot_trace(idata, var_names=['mu_alpha_pos'], ax=axes[0, :2], compact=False)

# Posterior
az.plot_posterior(idata, var_names=['mu_alpha_pos'], ax=axes[0, 2])

# Autocorr
az.plot_autocorr(idata, var_names=['mu_alpha_pos'], ax=axes[1, 0])

# Rank
az.plot_rank(idata, var_names=['mu_alpha_pos'], ax=axes[1, 1])

# Energy
az.plot_energy(idata, ax=axes[1, 2])

plt.tight_layout()
plt.savefig('figures/dashboard.png', dpi=300, bbox_inches='tight')
plt.show()
```

---

### Example 2: Extract Individual Parameters

```python
import arviz as az
import pandas as pd
import numpy as np

# Load posterior
idata = az.from_netcdf('output/qlearning_jax_posterior_TIMESTAMP.nc')

# Extract individual learning rates
alpha_pos_samples = idata.posterior['alpha_pos'].values  # shape: (chains, draws, participants)

# Compute mean for each participant
alpha_pos_mean = alpha_pos_samples.mean(axis=(0, 1))  # shape: (participants,)
alpha_neg_mean = idata.posterior['alpha_neg'].values.mean(axis=(0, 1))
beta_mean = idata.posterior['beta'].values.mean(axis=(0, 1))

# Create DataFrame
df_params = pd.DataFrame({
    'participant_id': range(len(alpha_pos_mean)),
    'alpha_pos': alpha_pos_mean,
    'alpha_neg': alpha_neg_mean,
    'beta': beta_mean
})

# Compute 94% HDI for each participant
from arviz import hdi

df_params['alpha_pos_hdi_low'] = [
    hdi(alpha_pos_samples[:, :, i].flatten(), hdi_prob=0.94)[0]
    for i in range(len(alpha_pos_mean))
]
df_params['alpha_pos_hdi_high'] = [
    hdi(alpha_pos_samples[:, :, i].flatten(), hdi_prob=0.94)[1]
    for i in range(len(alpha_pos_mean))
]

# Save to CSV
df_params.to_csv('output/individual_parameters.csv', index=False)
print(df_params)
```

---

### Example 3: Publication-Quality Figure

```python
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

# Load posterior
idata = az.from_netcdf('output/qlearning_jax_posterior_TIMESTAMP.nc')

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot each parameter
params = ['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta']
titles = ['α+ (Positive Learning Rate)', 'α- (Negative Learning Rate)', 'β (Inverse Temperature)']

for ax, param, title in zip(axes, params, titles):
    az.plot_posterior(idata, var_names=[param], ax=ax, hdi_prob=0.94)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Parameter Value', fontsize=14)

plt.suptitle('Group-Level Parameter Estimates (Posterior Mean ± 94% HDI)',
             fontsize=18, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig('figures/publication_figure.png', dpi=600, bbox_inches='tight')
plt.show()
```

---

## Interpretation Guide

### Group-Level Parameters (Q-Learning)

**μ_α+ (Positive Learning Rate)**
- Range: [0, 1]
- Interpretation: How quickly the group learns from correct responses
- Typical values: 0.3 - 0.7
- Higher → faster learning from rewards

**μ_α- (Negative Learning Rate)**
- Range: [0, 1]
- Interpretation: How quickly the group learns from incorrect responses
- Typical values: 0.1 - 0.5
- Often lower than α+ (asymmetric learning)
- Higher → more sensitive to negative feedback

**μ_β (Inverse Temperature)**
- Range: [0, ∞] (practical: 0-10)
- Interpretation: Exploration vs exploitation trade-off
- Low (0.5-1): More random/exploratory
- Medium (2-4): Balanced
- High (5-10): Strongly exploit best option
- Typical values: 2 - 5

---

## Convergence Diagnostics Checklist

Before trusting your results, check:

- [ ] **R-hat < 1.01** for all parameters (use `az.summary()`)
- [ ] **ESS > 400** for bulk and tail (use `az.summary()`)
- [ ] **Trace plots** show good mixing (chains overlap)
- [ ] **Rank plots** are uniform (no convergence issues)
- [ ] **Energy plot** shows overlap (if using NUTS)
- [ ] **No divergences** (check `idata.sample_stats.diverging.sum()`)
- [ ] **Posterior distributions** are smooth (not choppy)

---

## Troubleshooting

### Issue: R-hat > 1.01

**Solution:**
- Run more warmup samples (increase from 1000 to 2000)
- Increase `target_accept_prob` to 0.95
- Check for multimodality (pair plots)

### Issue: Low ESS (< 400)

**Solution:**
- Run more samples (increase from 2000 to 4000)
- Check autocorrelation (should decay quickly)
- May indicate poor mixing

### Issue: Divergences

**Solution:**
- Increase `target_accept_prob` to 0.95 or 0.99
- Increase `max_tree_depth` to 12
- Check for label switching or identification issues

### Issue: Choppy posteriors

**Solution:**
- Run more samples (need more draws)
- Check if model is converged (R-hat, ESS)
- May indicate numerical issues in likelihood

---

## Quick Commands Reference

```bash
# Activate environment
conda activate ds_env

# Quick diagnostics (all plots) - saves to figures/
python scripts/visualization/quick_arviz_plots.py \
  --posterior output/qlearning_jax_posterior_TIMESTAMP.nc

# Group parameters only - saves to figures/
python scripts/visualization/plot_group_parameters.py \
  --qlearning output/qlearning_jax_posterior_TIMESTAMP.nc \
  --prefix qlearning

# Model comparison (need both models) - saves to figures/
python scripts/visualization/plot_model_comparison.py \
  --qlearning output/qlearning_jax_posterior_TIMESTAMP.nc \
  --wmrl output/wmrl_jax_posterior_TIMESTAMP.nc
```

---

## See Also

- **ArviZ Documentation**: https://arviz-devs.github.io/arviz/
- **Gallery of Plots**: https://arviz-devs.github.io/arviz/examples/index.html
- **MODEL_REFERENCE.md**: Model specifications
- **ANALYSIS_PIPELINE.md**: Complete analysis workflow
