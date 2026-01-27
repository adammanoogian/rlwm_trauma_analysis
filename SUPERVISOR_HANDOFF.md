# Supervisor Handoff: MLE Fitting Complete

**Date**: January 22, 2026  
**Status**: Q-learning model fitted, ready for parameter-scale regression analysis

---

## What Your Supervisor Has Done

### 1. MLE/Bayesian Model Fitting ✓

Your supervisor has fitted the **Q-learning model** using JAX/NumPyro (hierarchical Bayesian estimation, which is the modern alternative to traditional MLE):

**Fitted Models:**
- ✅ **Q-learning** (asymmetric learning rates)
  - Latest fit: `output/v1/qlearning_jax_summary_20251122_200043.csv`
  - Full posterior: `output/v1/qlearning_jax_posterior_20251122_200043.nc`

**Missing:**
- ❓ **WM-RL model** - Not found in output directory yet
  - Check with supervisor if fitting is still running or if files are elsewhere

**What Was Fitted:**
- N = 2 participants with complete data (from `task_trials_long.csv`)
- Parameters estimated per participant:
  - `alpha_pos`: Positive learning rate (learning from rewards/correct feedback)
  - `alpha_neg`: Negative learning rate (learning from punishment/incorrect feedback)
  - `beta`: Inverse temperature (exploration vs exploitation)

### 2. Automatic Trauma Group Visualizations ✓

Pre-generated figures based on median splits (exploratory):

**Location**: `figures/trauma_groups/`

Key files:
- `hypothesis_groups_scatter.png` - 3-group classification visualization
- `behavioral_by_group.png` - Performance differences by trauma group
- `parameter_summary_by_group.csv` - Descriptive stats
- `integrated_parameters.csv` - **Master file** with parameters + trauma scales

### 3. Integrated Dataset ✓

**File**: `figures/trauma_groups/integrated_parameters.csv`

Contains:
- Participant IDs
- Trauma group assignments (hypothesis-driven: A/B/C)
- Fitted model parameters (`alpha_pos_mean`, `alpha_neg_mean`, `beta_mean`)
- LEC-5 trauma exposure scores
- IES-R PTSD symptom scores

**⚠️ Current Issue**: Only 1 participant has fitted parameters (participant 9187)
- This means only N=1 completed the task AND had survey data
- You'll need to check if more participants should be included

---

## What You Need to Do Next

### Step 1: Run Regression Analyses (Priority 1) 🎯

**Your supervisor said**: "do regressions of each parameter on your scales. Follow the senta paper."

I've created a comprehensive regression script for you:

```bash
# Navigate to analysis directory
cd d:\THESIS\rlwm_trauma_analysis

# Run regression analysis
python scripts/analysis/regress_parameters_on_scales.py \
    --params output/v1/qlearning_jax_summary_20251122_200043.csv \
    --model qlearning \
    --output-dir output/regressions
```

**What this does:**
1. **Simple (univariate) regressions** for each parameter:
   - `alpha_pos ~ LEC-5 total events`
   - `alpha_pos ~ LEC-5 personal events`
   - `alpha_pos ~ IES-R total`
   - `alpha_pos ~ IES-R intrusion`
   - `alpha_pos ~ IES-R avoidance`
   - `alpha_pos ~ IES-R hyperarousal`
   - (Same for `alpha_neg` and `beta`)

2. **Multiple regression** with IES-R subscales:
   - `alpha_pos ~ intrusion + avoidance + hyperarousal`
   - (Identifies which symptom dimension is the unique predictor)

3. **Outputs**:
   - `regression_results_simple.csv` - Table of all univariate results
   - `regression_results_multiple.csv` - Multiple regression coefficients
   - `regression_matrix_all.png` - Grid of scatter plots
   - Individual scatter plots for each parameter-scale pair

### Step 2: Check Data Issues (Priority 2) ⚠️

**Problem**: Only 1-2 participants have fitted parameters

**Investigate:**

```bash
# Check which participants completed task
python -c "import pandas as pd; df = pd.read_csv('output/task_trials_long.csv'); print('Participants in task data:', df['sona_id'].nunique()); print(df['sona_id'].unique())"

# Check which have survey data
python -c "import pandas as pd; df = pd.read_csv('output/summary_participant_metrics_all.csv'); print('Participants with survey data:', len(df)); print(df[['sona_id', 'lec_total_events', 'ies_total']].head(20))"
```

**Possible solutions:**
- Use `task_trials_long_all_participants.csv` instead (has N=17)
- Re-run fitting with all participants
- Merge parameters with `summary_participant_metrics_all.csv`

### Step 3: Visualize Median Split Results (Priority 3) 📊

Your supervisor mentioned "initial figures there on median split":

```bash
# View the pre-generated figures
explorer figures\trauma_groups

# Key files to review:
# - hypothesis_groups_scatter.png
# - behavioral_by_group.png  
# - parameter_summary_by_group.csv
```

**BUT**: These may not be meaningful yet if only N=1-2 have parameters fitted.

### Step 4: Wait for WM-RL Results (Priority 4) ⏳

Once your supervisor pushes the WM-RL fitted parameters:

```bash
# Run same regression analysis for WM-RL model
python scripts/analysis/regress_parameters_on_scales.py \
    --params output/v1/wmrl_jax_summary_TIMESTAMP.csv \
    --model wmrl \
    --output-dir output/regressions_wmrl
```

This will include additional parameters:
- `wm_capacity` - Working memory capacity
- `wm_weight` - Weight given to WM vs RL system

### Step 5: Compare Models (After Step 4)

Once both models are fitted:

```bash
# Model comparison (should be auto-generated)
# Check for: output/v1/model_comparison_TIMESTAMP.csv

# Or run manually if needed
python scripts/analysis/model_comparison.py
```

---

## Interpretation Guide: "Following the Senta Paper"

Based on computational psychiatry literature (e.g., Eckstein et al., 2022; Dezfouli et al., 2019):

### What to Report in Your Results:

1. **Univariate Associations** (Simple Regressions)
   - Report: β coefficient, 95% CI, t-statistic, p-value, r, R²
   - Example: "Positive learning rate (α+) was negatively associated with IES-R intrusion symptoms (β = -0.023, 95% CI [-0.045, -0.001], t = -2.31, p = .048, r = -.52)"
   - Focus on **effect sizes** (β, r) not just p-values
   - With small N, p < .10 can be "marginally significant"

2. **Multiple Regression** (Subscale Specificity)
   - Report: Which IES-R subscale uniquely predicts parameters when controlling for others
   - Check VIF < 5 (variance inflation factor) to ensure no multicollinearity
   - Example: "In multiple regression, only hyperarousal symptoms uniquely predicted learning rate (β = -0.031, p = .023), while intrusion and avoidance did not (ps > .20)"

3. **Visualizations**
   - Scatter plots with regression lines
   - Report outliers if present
   - Consider robust regression if outliers detected

4. **Effect Size Interpretation**
   - Small: r = .10-.30, Cohen's d = 0.2
   - Medium: r = .30-.50, Cohen's d = 0.5  
   - Large: r > .50, Cohen's d = 0.8

### Hypothesis-Driven Approach (Your Supervisor's Next Step)

After exploring correlations, you'll develop hypotheses for **new parameters**:

**Examples from literature:**
1. **Decay parameter** (λ): Do trauma symptoms relate to faster forgetting?
2. **Perseveration** (ρ): Do symptoms predict difficulty updating after reversal?
3. **Separate α for reward vs punishment**: Do trauma groups show asymmetric learning?
4. **Working memory capacity** (if WM-RL better): Is capacity reduced in trauma groups?

**Workflow**:
```
1. Find correlation (e.g., IES-R → slower learning)
2. Develop hypothesis (e.g., trauma causes faster decay)
3. Add parameter to model (e.g., Q-value decay)
4. Re-fit model
5. Compare: Does new model fit better? Does new parameter mediate correlation?
6. Repeat
```

---

## Troubleshooting

### Issue: "Only 1-2 participants with fitted parameters"

**Solution A**: Re-fit using all participants dataset

```bash
# Check if supervisor used the right data file
# Should be: task_trials_long_all_participants.csv (N=17)
# Might have used: task_trials_long.csv (N=2 with survey data)

# If needed, re-run fitting:
python scripts/fitting/fit_with_jax.py \
    --data output/task_trials_long_all_participants.csv \
    --chains 4 \
    --samples 2000
```

**Solution B**: Merge parameters manually

```python
import pandas as pd

# Load fitted parameters
params = pd.read_csv('output/v1/qlearning_jax_summary_20251122_200043.csv', index_col=0)

# Load participant order (critical!)
data = pd.read_csv('output/task_trials_long.csv')
participant_order = data['sona_id'].unique()

# Extract individual parameters
records = []
for i, sona_id in enumerate(participant_order):
    records.append({
        'sona_id': sona_id,
        'alpha_pos': params.loc[f'alpha_pos[{i}]', 'mean'],
        'alpha_neg': params.loc[f'alpha_neg[{i}]', 'mean'],
        'beta': params.loc[f'beta[{i}]', 'mean']
    })

df_params = pd.DataFrame(records)

# Merge with scales
scales = pd.read_csv('output/summary_participant_metrics_all.csv')
merged = df_params.merge(scales, on='sona_id', how='inner')

print(f"Merged N = {len(merged)}")
merged.to_csv('output/parameters_with_scales.csv', index=False)
```

### Issue: "statsmodels not installed"

```bash
# Install if needed
conda install statsmodels

# Or
pip install statsmodels
```

### Issue: "Where are the WM-RL results?"

**Check:**
1. Git pull latest changes
2. Ask supervisor if fitting is still running
3. Check different output directories

---

## File Reference

### Input Data Files
- `output/task_trials_long.csv` - Task data (N=2 with surveys)
- `output/task_trials_long_all_participants.csv` - All task data (N=17)
- `output/summary_participant_metrics_all.csv` - All behavioral + scale data
- `output/trauma_groups/group_assignments.csv` - Trauma group classifications

### Fitted Parameters
- `output/v1/qlearning_jax_summary_20251122_200043.csv` - **Q-learning parameters** ⭐
- `output/v1/qlearning_jax_posterior_20251122_200043.nc` - Full posterior samples
- `output/v1/wmrl_jax_summary_*.csv` - WM-RL parameters (not found yet)

### Analysis Scripts
- `scripts/analysis/regress_parameters_on_scales.py` - **NEW: Regression analysis** ⭐
- `scripts/analysis/analyze_parameters_by_trauma_group.py` - Group comparisons
- `scripts/analysis/trauma_grouping_analysis.py` - Create trauma groups
- `scripts/fitting/fit_with_jax.py` - Model fitting script

### Output Locations
- `output/regressions/` - Regression results (you'll create this)
- `figures/trauma_groups/` - Pre-generated group visualizations
- `output/v1/figures/` - Model diagnostic plots

---

## Quick Start Commands

```bash
# 1. Check what data was used for fitting
head output/v1/qlearning_jax_summary_20251122_200043.csv

# 2. Run regression analysis
python scripts/analysis/regress_parameters_on_scales.py

# 3. View results
explorer output\regressions

# 4. Check for new WM-RL results
git pull
ls output/v1/wmrl*

# 5. Re-run with WM-RL when available
python scripts/analysis/regress_parameters_on_scales.py \
    --params output/v1/wmrl_jax_summary_TIMESTAMP.csv \
    --model wmrl
```

---

## Questions to Ask Your Supervisor

1. **Data**: "Should I use task_trials_long.csv (N=2) or task_trials_long_all_participants.csv (N=17) for the analysis?"

2. **WM-RL**: "Have you pushed the WM-RL fitted parameters? I only see Q-learning results in output/v1/"

3. **Sample size**: "I'm seeing only 1-2 participants with fitted parameters. Is this expected or should I merge with the full participant dataset?"

4. **Next parameters**: "After reviewing the correlations, what parameters should we consider adding to the model? (e.g., decay, perseveration, reward/punishment asymmetry?)"

---

## Timeline

**Now → Next Meeting:**
- [ ] Run regression analysis script
- [ ] Review correlation results
- [ ] Identify strongest parameter-scale associations
- [ ] Check for outliers in scatter plots
- [ ] Resolve data merging issue (N=1 vs N=17)

**After WM-RL Results:**
- [ ] Run same regressions for WM-RL
- [ ] Compare which model shows stronger scale associations
- [ ] Develop hypotheses for new parameters

**Iterative Modeling:**
- [ ] Propose new parameter based on correlations
- [ ] Supervisor re-fits model with new parameter
- [ ] Test if new parameter mediates scale associations
- [ ] Repeat until best model found

---

Good luck! Start with Step 1 (regressions) and resolve the data issue (Step 2), then wait for supervisor's WM-RL results.
