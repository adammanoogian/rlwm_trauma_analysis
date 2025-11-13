# RLWM Trauma Analysis Pipeline

Complete workflow for analyzing RLWM task data using reinforcement learning models and Bayesian inference.

## Overview

This pipeline integrates:
1. **Data Processing**: Clean and organize jsPsych behavioral data
2. **Model Simulation**: Generate synthetic data using RL models
3. **Model Fitting**: Fit models to human data using hierarchical Bayesian inference
4. **Model Comparison**: Evaluate which model best explains behavior
5. **Trauma Analysis**: Compare model parameters across trauma groups

## Project Structure

```
rlwm_trauma_analysis/
├── config.py                    # Central configuration
├── data/                        # Raw jsPsych CSV files
├── output/                      # Processed data
│   ├── parsed_demographics.csv
│   ├── parsed_survey1.csv       # LEC-5
│   ├── parsed_survey2.csv       # IES-R
│   ├── parsed_task_trials.csv
│   ├── collated_participant_data.csv
│   ├── task_trials_long.csv
│   └── summary_participant_metrics.csv
├── output/v1/                   # Version 1 outputs
│   ├── simulated_data.csv
│   ├── qlearning_posterior.nc
│   ├── wmrl_posterior.nc
│   └── model_comparison.csv
├── environments/                # RL environments
│   ├── rlwm_env.py
│   └── task_config.py
├── models/                      # RL agents
│   ├── q_learning.py
│   └── wm_rl_hybrid.py
├── fitting/                     # Bayesian fitting
│   ├── pymc_models.py
│   └── fit_to_data.py
├── simulations/                 # Data generation
│   └── generate_data.py
├── scripts/                     # Original data pipeline
│   ├── 01_parse_raw_data.py
│   ├── 02_create_collated_csv.py
│   ├── 03_create_task_trials_csv.py
│   ├── 04_create_summary_csv.py
│   └── utils/
├── figures/                     # Visualizations
└── docs/                        # Documentation
    ├── ENVIRONMENT_REFERENCE.md
    ├── MODEL_REFERENCE.md
    └── ANALYSIS_PIPELINE.md     # This file
```

---

## Pipeline Stages

### Stage 1: Data Cleaning (Existing)

Clean raw jsPsych data and compute behavioral metrics.

**Scripts:**
```bash
# Run existing data pipeline
python scripts/01_parse_raw_data.py
python scripts/02_create_collated_csv.py
python scripts/03_create_task_trials_csv.py
python scripts/04_create_summary_csv.py
```

**Outputs:**
- `output/parsed_*.csv`: Demographics, surveys, task trials
- `output/task_trials_long.csv`: Trial-level data (for model fitting)
- `output/summary_participant_metrics.csv`: Participant-level aggregates

**Key Variables:**
- Demographics: age, gender, race, education
- LEC-5: Trauma exposure (30 binary columns)
- IES-R: PTSD symptoms (intrusion, avoidance, hyperarousal subscales)
- Task metrics: accuracy, RT, learning slopes, reversal adaptation

---

### Stage 2: Model Development

Create RL environment and agents.

#### 2.1 Test Environment

```bash
python environments/rlwm_env.py
```

Verifies:
- Observation/action spaces
- Reward structure
- Reversal logic

#### 2.2 Test Models

```bash
# Test Q-learning
python models/q_learning.py

# Test WM-RL hybrid
python models/wm_rl_hybrid.py
```

Verifies:
- Learning dynamics
- Action selection
- Parameter effects

#### 2.3 Quick Simulation

```python
from environments.rlwm_env import create_rlwm_env
from models.q_learning import create_q_learning_agent, simulate_agent_on_env

env = create_rlwm_env(set_size=3, seed=42)
agent = create_q_learning_agent(alpha=0.3, beta=3.0, seed=42)

results = simulate_agent_on_env(agent, env, num_trials=100)
print(f"Accuracy: {results['accuracy']:.3f}")
```

---

### Stage 3: Synthetic Data Generation

Generate simulated data for validation and power analysis.

#### Option A: From Default Parameters

```bash
python simulations/generate_data.py \
    --model qlearning \
    --n-participants 100 \
    --num-blocks 21 \
    --trials-per-block 100 \
    --add-noise \
    --seed 42
```

#### Option B: From Fitted Posteriors

```bash
python simulations/generate_data.py \
    --model wmrl \
    --n-participants 100 \
    --posteriors output/v1/wmrl_posterior_20250113.nc
```

**Output:**
- `output/v1/simulated_data_qlearning_TIMESTAMP.csv`

**Use Cases:**
1. **Parameter Recovery**: Verify fitting procedure
2. **Power Analysis**: Determine sample size needed
3. **Validation**: Compare simulated vs. human distributions

---

### Stage 4: Bayesian Model Fitting

Fit models to human behavioral data.

#### 4.1 Prepare Data

Ensure `output/task_trials_long.csv` exists with columns:
- `sona_id`: Participant ID
- `block`: Block number (≥3 for main task)
- `trial`: Trial within block
- `stimulus`: Stimulus ID (will be converted to 0-indexed)
- `key_press`: Action taken (0, 1, or 2)
- `correct`: Whether response was correct (0 or 1)

#### 4.2 Fit Q-Learning

```bash
python fitting/fit_to_data.py \
    --model qlearning \
    --data output/task_trials_long.csv \
    --chains 4 \
    --samples 2000 \
    --tune 1000
```

**Estimated Runtime**: 30-60 minutes (depends on data size, hardware)

**Outputs:**
- `output/v1/qlearning_posterior_TIMESTAMP.nc`: Full posterior samples
- `output/v1/qlearning_summary_TIMESTAMP.csv`: Parameter summaries

#### 4.3 Fit WM-RL Hybrid

```bash
python fitting/fit_to_data.py \
    --model wmrl \
    --chains 4 \
    --samples 2000
```

**Estimated Runtime**: 60-120 minutes (more parameters)

#### 4.4 Fit Both and Compare

```bash
python fitting/fit_to_data.py --model both
```

Automatically compares models using WAIC and LOO.

**Output:**
- `output/v1/model_comparison_TIMESTAMP.csv`

---

### Stage 5: Posterior Analysis

Analyze fitted parameters.

#### 5.1 Load Posteriors

```python
import arviz as az
import pandas as pd

# Load posterior
trace = az.from_netcdf('output/v1/qlearning_posterior_20250113.nc')

# Summary statistics
summary = az.summary(trace)
print(summary)
```

#### 5.2 Visualize Group Parameters

```python
import arviz as az

# Plot group-level parameters
az.plot_posterior(
    trace,
    var_names=['mu_alpha', 'mu_beta'],
    figsize=(10, 4)
)
```

#### 5.3 Extract Individual Parameters

```python
# Get individual learning rates
alphas = trace.posterior['alpha'].values  # shape: (chains, draws, participants)

# Mean per participant
alpha_means = alphas.mean(axis=(0, 1))  # shape: (participants,)
```

#### 5.4 Diagnostics

```python
# Check convergence (R-hat should be < 1.01)
print(az.summary(trace, var_names=['mu_alpha', 'mu_beta']))

# Check for divergences
print(f"Divergences: {trace.sample_stats.diverging.sum().values}")

# Trace plots
az.plot_trace(trace, var_names=['mu_alpha', 'mu_beta'])
```

---

### Stage 6: Model Comparison

Determine which model best explains the data.

#### 6.1 Information Criteria

```python
import arviz as az

# Compare models
comparison = az.compare({
    'qlearning': trace_qlearning,
    'wmrl': trace_wmrl
})

print(comparison)
```

**Interpretation:**
- **rank**: Model ranking (1 = best)
- **elpd_loo**: Expected log predictive density (higher = better)
- **p_loo**: Effective number of parameters
- **weight**: Model probability

#### 6.2 Posterior Predictive Checks

Generate predictions from fitted model and compare to data:

```python
# Generate posterior predictive samples
# (Implementation depends on specific needs)
```

---

### Stage 7: Trauma Group Comparisons

Compare model parameters between trauma groups.

#### 7.1 Define Groups

```python
import pandas as pd

# Load participant data
df_participants = pd.read_csv('output/collated_participant_data.csv')

# Define groups based on IES-R cutoff
ies_cutoff = 33
df_participants['ptsd_group'] = df_participants['ies_total'] >= ies_cutoff

# Or use LEC-5 exposure
df_participants['trauma_exposed'] = df_participants['lec_personal_events'] > 0
```

#### 7.2 Compare Parameters

```python
# Extract parameters per participant
alphas = trace.posterior['alpha'].mean(axis=(0, 1))

# Add to dataframe
df_participants['alpha'] = alphas

# Compare groups
import scipy.stats as stats

ptsd_group = df_participants[df_participants['ptsd_group']]
control_group = df_participants[~df_participants['ptsd_group']]

# T-test
t_stat, p_val = stats.ttest_ind(
    ptsd_group['alpha'],
    control_group['alpha']
)

print(f"Learning rate: PTSD={ptsd_group['alpha'].mean():.3f}, "
      f"Control={control_group['alpha'].mean():.3f}, "
      f"t={t_stat:.2f}, p={p_val:.3f}")
```

#### 7.3 Regression Analysis

```python
import statsmodels.api as sm

# Predict learning rate from trauma measures
X = df_participants[['ies_intrusion', 'ies_avoidance', 'ies_hyperarousal', 'lec_personal_events']]
X = sm.add_constant(X)
y = df_participants['alpha']

model = sm.OLS(y, X).fit()
print(model.summary())
```

---

## Common Workflows

### Workflow 1: Initial Model Validation

```bash
# 1. Test environment
python environments/rlwm_env.py

# 2. Test Q-learning
python models/q_learning.py

# 3. Generate synthetic data
python simulations/generate_data.py --model qlearning --n-participants 50 --add-noise

# 4. Fit to synthetic data (parameter recovery)
python fitting/fit_to_data.py --model qlearning --data output/v1/simulated_data_qlearning_TIMESTAMP.csv

# 5. Check parameter recovery
python -c "import arviz as az; trace = az.from_netcdf('output/v1/qlearning_posterior_TIMESTAMP.nc'); print(az.summary(trace))"
```

### Workflow 2: Fit to Human Data

```bash
# 1. Ensure data is processed
python scripts/01_parse_raw_data.py
python scripts/03_create_task_trials_csv.py

# 2. Fit Q-learning
python fitting/fit_to_data.py --model qlearning

# 3. Fit WM-RL
python fitting/fit_to_data.py --model wmrl

# 4. Compare models
python fitting/fit_to_data.py --model both
```

### Workflow 3: Posterior Predictive Checks

```bash
# 1. Fit model to human data
python fitting/fit_to_data.py --model qlearning

# 2. Generate data from posterior
python simulations/generate_data.py \
    --model qlearning \
    --posteriors output/v1/qlearning_posterior_TIMESTAMP.nc \
    --n-participants 100

# 3. Compare distributions (custom analysis script)
# python scripts/analysis/compare_simulated_vs_human.py
```

---

## Analysis Checklist

### Pre-Fitting

- [ ] Data cleaned and in `output/task_trials_long.csv`
- [ ] Practice blocks filtered (block ≥ 3)
- [ ] Stimulus and action columns 0-indexed
- [ ] No missing values in key columns
- [ ] Environment tested and working
- [ ] Models tested and learning properly

### During Fitting

- [ ] MCMC sampling completes without errors
- [ ] No excessive divergences (< 1% of samples)
- [ ] R-hat < 1.01 for all parameters
- [ ] Effective sample size (ESS) > 400 per parameter
- [ ] Trace plots show good mixing

### Post-Fitting

- [ ] Posteriors saved to `.nc` files
- [ ] Parameter summaries make sense (e.g., α ∈ [0,1])
- [ ] Model comparison completed
- [ ] Posterior predictive checks performed
- [ ] Group comparisons conducted

---

## Troubleshooting

### Issue: PyMC Installation Fails

```bash
# Use conda instead of pip
conda install -c conda-forge pymc arviz
```

### Issue: Fitting Takes Too Long

- Reduce number of participants (for testing)
- Reduce samples (e.g., 1000 instead of 2000)
- Use fewer chains (2 instead of 4)
- Check for divergences (might need to simplify priors)

### Issue: High Divergences

- Increase `target_accept` to 0.99
- Reparameterize model (use non-centered parameterization)
- Check for label switching or multimodality
- Simplify priors (tighter constraints)

### Issue: Parameters Out of Range

- Check data preprocessing (0-indexing)
- Verify bounded transforms in PyMC model
- Inspect data for outliers or errors

### Issue: Models Don't Learn in Simulation

- Check environment reward structure
- Verify agent update logic
- Try stronger learning signal (higher alpha, beta)
- Inspect Q-values over trials

---

## Configuration Management

All parameters centralized in `config.py`:

```python
from config import TaskParams, ModelParams, PyMCParams

# Task parameters
print(TaskParams.SET_SIZES)          # [2, 3, 5, 6]
print(TaskParams.REVERSAL_MIN)       # 12
print(TaskParams.REWARD_CORRECT)     # 1.0

# Model defaults
print(ModelParams.ALPHA_DEFAULT)     # 0.1
print(ModelParams.WM_CAPACITY_DEFAULT)  # 4

# PyMC sampling
print(PyMCParams.NUM_CHAINS)         # 4
print(PyMCParams.NUM_SAMPLES)        # 2000
```

To modify, edit `config.py` directly.

---

## Version Control

Track major analysis versions:

```python
# In config.py
VERSION = 'v1'  # Pilot data

# Outputs automatically saved to output/v1/
```

---

## Next Steps

1. **Descriptive Analyses**: Plot learning curves, set-size effects
2. **Advanced Models**: Asymmetric learning, decay models
3. **Trauma Mediators**: Test attention, memory as mediators
4. **Neural Correlates**: If fMRI data available, relate parameters to brain activity
5. **Longitudinal**: If multiple timepoints, model parameter changes

---

## References

**Key Files:**
- Environment: `environments/rlwm_env.py`
- Models: `models/q_learning.py`, `models/wm_rl_hybrid.py`
- Fitting: `fitting/pymc_models.py`, `fitting/fit_to_data.py`
- Simulation: `simulations/generate_data.py`

**Documentation:**
- Environment API: `docs/ENVIRONMENT_REFERENCE.md`
- Model Details: `docs/MODEL_REFERENCE.md`

**Configuration:**
- Central config: `config.py`

---

## Contact & Support

For issues or questions:
- Check documentation first
- Review test scripts for examples
- Inspect config.py for parameter settings
- Use `--help` flag on command-line scripts

**Example:**
```bash
python fitting/fit_to_data.py --help
```
