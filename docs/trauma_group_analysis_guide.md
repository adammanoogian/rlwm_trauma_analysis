# Trauma Group Analysis Guide

This guide explains how to analyze fitted model parameters and behavioral data based on trauma group classifications.

## Overview

The trauma group analysis pipeline consists of three main scripts:

1. **trauma_grouping_analysis.py** - Creates trauma group classifications
2. **analyze_parameters_by_trauma_group.py** - Analyzes fitted parameters by group
3. **analyze_learning_by_trauma_group.py** - Analyzes learning trajectories by group

## Workflow

### Step 1: Create Trauma Groups

```bash
python scripts/analysis/trauma_grouping_analysis.py
```

**What it does:**
- Creates hypothesis-driven groups based on median splits of LEC-5 and IES-R:
  - **Group A (Low-Low):** Low trauma exposure, low symptoms - baseline
  - **Group B (High-Low):** High trauma exposure, low symptoms - resilient profile
  - **Group C (High-High):** High trauma exposure, high symptoms - symptomatic
- Also performs hierarchical clustering for data-driven validation
- Saves group assignments to `output/trauma_groups/group_assignments.csv`

**Outputs:**
- `output/trauma_groups/group_assignments.csv` - Main output with group labels per participant
- `figures/trauma_groups/hypothesis_groups_scatter.png` - Visualization of groups
- Various other clustering visualizations

### Step 2: Fit Models (if not already done)

```bash
# Fit Q-Learning model to all participants
python scripts/fitting/fit_with_jax.py \
    --data output/task_trials_long_all_participants.csv \
    --model qlearning \
    --chains 4 --warmup 1000 --samples 2000 \
    --output output/v1/
```

**What it does:**
- Fits hierarchical Bayesian Q-Learning or WM-RL models
- Uses JAX/NumPyro for efficient MCMC sampling
- Saves posterior samples (.nc files) and summary statistics (.csv files)

**Key parameters estimated:**
- `alpha_pos`: Positive learning rate (0-1)
- `alpha_neg`: Negative learning rate (0-1)
- `beta`: Inverse temperature / decision noise (0-10)
- Plus group-level means (mu) and standard deviations (sigma)

### Step 3: Analyze Parameters by Trauma Group

```bash
python scripts/analysis/analyze_parameters_by_trauma_group.py \
    --params "output/v1/qlearning_jax_summary_*.csv" \
    --data output/task_trials_long_all_participants.csv \
    --trauma-groups output/trauma_groups/group_assignments.csv \
    --output-dir figures/trauma_groups
```

**What it does:**
- Loads fitted parameters and trauma group assignments
- Maps parameter indices [0], [1], ... to actual participant IDs
- Merges datasets and creates integrated analysis file
- Generates visualizations and statistical summaries

**Key challenge solved:**
The fitting process assigns parameters using indices, not participant IDs. This script reads the trial data to determine participant order and correctly maps indices to IDs.

**Outputs:**
- `output/trauma_groups/integrated_parameters.csv` - Merged dataset (parameters + trauma groups)
- `output/trauma_groups/parameter_summary_by_group.csv` - Descriptive statistics
- `figures/trauma_groups/parameters_by_trauma_group_violin.png` - Violin plots
- `figures/trauma_groups/parameter_scatter_matrix_by_group.png` - Pairwise relationships
- `figures/trauma_groups/trauma_parameter_correlations.png` - Correlation heatmap

**Research questions addressed:**
- Do trauma groups differ in learning rates?
- Are high-symptom individuals less sensitive to negative outcomes (lower alpha_neg)?
- Does decision noise (beta) vary with trauma exposure?
- Which trauma measures (LEC vs IES-R) most strongly correlate with parameters?

### Step 4: Analyze Learning Trajectories by Trauma Group

```bash
python scripts/analysis/analyze_learning_by_trauma_group.py \
    --data output/task_trials_long_all_participants.csv \
    --trauma-groups output/trauma_groups/group_assignments.csv \
    --output-dir figures/trauma_groups \
    --window 20
```

**What it does:**
- Computes rolling accuracy (smoothed with 20-trial window)
- Creates learning curve visualizations by trauma group
- Analyzes performance by cognitive load (set size)
- Examines temporal dynamics across blocks

**Outputs:**
- `figures/trauma_groups/learning_curves_all_participants.png` - All trajectories colored by group
- `figures/trauma_groups/learning_curves_by_group_panels.png` - Separate panels with mean ± SEM
- `figures/trauma_groups/performance_by_load_and_time.png` - Load effects (2×2 grid)

**Research questions addressed:**
- Do trauma groups differ in learning speed (slope of curve)?
- Do groups reach different asymptotic performance levels?
- Are trauma effects load-dependent (worse at high set sizes)?
- Do groups show different temporal patterns across blocks?

## Data Files Reference

### Input Files

1. **output/task_trials_long_all_participants.csv**
   - Trial-level data for all participants
   - Columns: sona_id, trial_in_experiment, block, stimulus, key_press, correct, rt, set_size, etc.

2. **output/trauma_groups/group_assignments.csv**
   - Trauma group classifications
   - Columns: sona_id, lec_total_events, ies_total, hypothesis_group, cluster_k2/k3/k4

3. **output/v1/qlearning_jax_summary_YYYYMMDD_HHMMSS.csv**
   - Fitted parameter summary statistics
   - Rows: mu_alpha_pos, alpha_pos[0], alpha_pos[1], ..., sigma_alpha_pos, z_alpha_pos[0], ...
   - Columns: mean, sd, hdi_3%, hdi_97%, mcse_mean, mcse_sd, ess_bulk, ess_tail, r_hat

4. **output/v1/qlearning_jax_posterior_YYYYMMDD_HHMMSS.nc** (optional)
   - Full posterior samples in NetCDF format
   - Used for more advanced analyses (HDI intervals, posterior predictive checks)

### Output Files

1. **output/trauma_groups/integrated_parameters.csv**
   - Merged dataset combining trauma groups + fitted parameters
   - One row per participant
   - Columns: sona_id, lec_total_events, ies_total, hypothesis_group, param_index, alpha_pos_mean, alpha_neg_mean, beta_mean

2. **output/trauma_groups/parameter_summary_by_group.csv**
   - Summary statistics of parameters by trauma group
   - Columns: group, parameter, n, mean, std, sem, min, max

## Visualization Gallery

### Parameter Analysis Visualizations

1. **parameters_by_trauma_group_violin.png**
   - Violin plots showing parameter distributions by group
   - Includes overlaid individual points (jittered)
   - Shows sample sizes for each group
   - Useful for: Identifying group differences in parameters

2. **parameter_scatter_matrix_by_group.png**
   - Pairwise scatter plots of all parameters
   - Colored by trauma group
   - Diagonal shows density distributions
   - Useful for: Understanding parameter relationships and group overlap

3. **trauma_parameter_correlations.png**
   - Heatmap of correlations between trauma measures and parameters
   - Rows: LEC Total Events, IES-R Total Score
   - Columns: alpha_pos, alpha_neg, beta
   - Useful for: Identifying which trauma dimensions relate to which parameters

### Learning Trajectory Visualizations

4. **learning_curves_all_participants.png**
   - All individual learning curves (thin lines) colored by group
   - Group means overlaid (thick lines)
   - Shows within-group heterogeneity and between-group differences
   - Useful for: Overview of learning patterns

5. **learning_curves_by_group_panels.png**
   - Three separate panels (one per group)
   - Individual trajectories + group mean ± SEM
   - Useful for: Detailed examination of each group's learning dynamics

6. **performance_by_load_and_time.png**
   - 2×2 grid:
     - Top-left: Accuracy by set size
     - Top-right: RT by set size
     - Bottom-left: Accuracy by block
     - Bottom-right: RT by block
   - Useful for: Testing load-dependent and temporal trauma effects

## Interpretation Guide

### Parameter Interpretation

**Alpha_pos (Positive Learning Rate)**
- Range: 0 to 1
- Higher values = faster learning from rewards
- Clinical hypothesis: May be reduced in PTSD (anhedonia, reduced reward sensitivity)

**Alpha_neg (Negative Learning Rate)**
- Range: 0 to 1
- Higher values = faster learning from punishments
- Clinical hypothesis: May be altered in trauma (hypervigilance vs. avoidance)

**Beta (Inverse Temperature)**
- Range: 0 to 10+
- Higher values = more deterministic choices (exploiting learned values)
- Lower values = more exploratory/noisy choices
- Clinical hypothesis: May be reduced in PTSD (cognitive flexibility deficits, attentional control)

### Group Interpretation

**Group A (Low-Low):**
- Baseline comparison group
- Low trauma exposure, low symptoms
- Expected: Typical learning parameters

**Group B (High-Low) - "Resilient":**
- High trauma exposure but low symptoms
- Theoretically interesting: What protects them?
- Hypothesis: May show preserved or even enhanced learning (compensation?)

**Group C (High-High):**
- High trauma exposure and high symptoms
- Expected: May show alterations in learning
- Hypothesis: Lower learning rates, higher decision noise

## Statistical Considerations

### Sample Size (N=17, with ~15 in groups A/B/C)

**Power considerations:**
- Underpowered for detecting small-medium effects
- Focus on effect sizes rather than p-values
- Use confidence intervals / HDI prominently
- Visualizations should show all individual data points

**Best practices:**
- Show full distributions (violin plots, not just bar charts)
- Report effect sizes (Cohen's d, Bayes factors)
- Use robust statistics (medians, MAD, Spearman correlations)
- Consider this as exploratory/hypothesis-generating

### Multiple Comparisons

When comparing 3 groups on 3 parameters (9 comparisons):
- Consider Bonferroni correction: α = 0.05/9 = 0.0056
- OR use Bayesian approach (posterior probabilities, no correction needed)
- OR report uncorrected p-values with "exploratory" caveat

### Missing Data

- Some participants may not have fitted parameters yet (sampling in progress)
- Scripts handle this gracefully (use all available data)
- Report n for each analysis clearly

## Advanced Analyses (Future Extensions)

### 1. Posterior Predictive Checks by Group
```python
# Load posterior samples
import arviz as az
posterior = az.from_netcdf('output/v1/qlearning_jax_posterior_*.nc')

# Generate predicted data for each group
# Compare observed vs predicted distributions
```

### 2. Hierarchical Model with Group as Predictor
```python
# Modify NumPyro model to include group as fixed effect
# E.g., mu_alpha_pos = beta_0 + beta_1*trauma_group
```

### 3. Model Comparison: Q-Learning vs WM-RL by Group
```python
# Do groups differ in which model fits better?
# E.g., does high cognitive load interact with trauma?
```

### 4. Symptom Subscales
```python
# Analyze IES-R subscales: intrusion, avoidance, hyperarousal
# Do specific symptoms relate to specific parameters?
```

### 5. Longitudinal / Intervention Effects
```python
# If pre-post data available: Does treatment change parameters?
# Does group moderate treatment response?
```

## Troubleshooting

### "No parameter files found"
- Check that fitting has completed: `ls output/v1/qlearning_jax_summary_*.csv`
- Verify the path/wildcard pattern matches your files

### "Parameter index exceeds participant list length"
- The participant order in trial data doesn't match fitting
- Check which participants were included in fitting
- May need to filter trial data to match fitted participants

### "Trauma groups file not found"
- Run Step 1 first: `python scripts/analysis/trauma_grouping_analysis.py`

### "Python not found"
- Use your environment: `conda activate rlwm` or similar
- Or: `python3` instead of `python`

### Visualizations look wrong / empty
- Check that groups have enough participants (n≥2 per group)
- Verify parameter values are reasonable (not all NaN)
- Check console output for warnings about missing data

## Citation

If using these analyses in publications:

```
Trauma group classification based on median splits of:
- Life Events Checklist for DSM-5 (LEC-5; Weathers et al., 2013)
- Impact of Event Scale-Revised (IES-R; Weiss & Marmar, 1997)

Bayesian hierarchical modeling using:
- JAX (Bradbury et al., 2018)
- NumPyro (Phan et al., 2019)
- ArviZ (Kumar et al., 2019)
```

## Contact / Support

For questions or issues with these scripts:
1. Check this documentation first
2. Review console output for error messages
3. Ensure all input files exist and are formatted correctly

## Changelog

**2025-11-24:** Initial creation
- Created trauma-parameter integration script
- Created learning trajectory analysis script
- Added comprehensive documentation
