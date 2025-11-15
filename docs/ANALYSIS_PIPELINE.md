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
│   ├── pymc_models.py           # Uses agent classes via unified_simulator
│   └── fit_to_data.py
├── simulations/                 # Data generation & exploration
│   ├── unified_simulator.py     # Core: fixed & sampled parameter simulation
│   ├── generate_data.py         # Synthetic data generation
│   ├── parameter_sweep.py       # Parameter space exploration
│   └── interactive_exploration.py  # Jupyter widgets
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

## Unified Architecture

**Key Design Principle**: All simulation, fitting, and parameter exploration use the **same agent implementations** from `models/`. This eliminates code duplication and ensures consistency.

### Code Flow

```
Agent Classes (models/)
    ↓
Unified Simulator (simulations/unified_simulator.py)
    ↓
    ├─→ Parameter Sweeps (parameter_sweep.py)
    ├─→ Data Generation (generate_data.py)
    └─→ PyMC Fitting (fitting/pymc_models.py)
```

### Two Simulation Modes

1. **Fixed Parameters**: `simulate_agent_fixed()`
   - Used for: Parameter sweeps, single-condition simulations
   - Example: "What accuracy do I get with α=0.3, β=2.0?"
   - Parameters are deterministic for each run

2. **Sampled Parameters**: `simulate_agent_sampled()`
   - Used for: Prior/posterior predictive checks, realistic synthetic data
   - Example: "Generate 50 participants with α ~ Beta(2,2), β ~ Gamma(2,1)"
   - Parameters vary across samples according to distributions

### Benefits

- ✅ **Single source of truth**: Agent logic defined once in `models/`
- ✅ **Guaranteed consistency**: PyMC fits the same model that parameter sweeps test
- ✅ **Easy to extend**: Add new model? Just update agent class
- ✅ **Flexible simulation**: Can simulate with both fixed and sampled parameters

### PyMC Integration

Since agent classes use pure Python (not PyTensor), we use **Metropolis sampler** (no gradients) instead of NUTS:

```python
with build_qlearning_model(data) as model:
    trace = pm.sample(draws=2000, step=pm.Metropolis())
```

Trade-off: Slower sampling than NUTS, but ensures exact consistency between simulation and fitting.

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

## Stage 8: Testing and Validation

### 8.1 Test Suite

Comprehensive pytest test suite ensures code correctness.

**Run all tests:**
```bash
pytest
```

**Run only fast tests:**
```bash
pytest -m "not slow"
```

**Run with coverage:**
```bash
pytest --cov=models --cov=fitting --cov=environments --cov-report=html
```

### 8.2 Test Categories

**Model Consistency (`test_model_consistency.py`):**
- Deterministic behavior across runs
- Parameter effects
- Valid probability distributions
- Cross-model comparisons

**Parameter Recovery (`test_parameter_recovery.py`):**
- Can we recover known parameters from synthetic data?
- Q-learning and WM-RL recovery tests
- Identifiability checks

**PyMC Integration (`test_pymc_integration.py`):**
- Model building works
- MCMC sampling succeeds
- Convergence diagnostics

**Run specific category:**
```bash
pytest tests/test_model_consistency.py -v
pytest tests/test_parameter_recovery.py -v -m "not slow"
```

### 8.3 Continuous Testing

Add pre-commit hook:
```bash
# .git/hooks/pre-commit
#!/bin/bash
pytest -m "not slow" || exit 1
```

See `tests/README.md` for complete testing documentation.

---

## Stage 9: Parameter Space Exploration

### 9.1 Systematic Parameter Sweeps

Explore how parameters affect model behavior systematically.

**Q-learning sweep:**
```bash
python simulations/parameter_sweep.py \
    --model qlearning \
    --num-trials 100 \
    --num-reps 5
```

**WM-RL sweep:**
```bash
python simulations/parameter_sweep.py \
    --model wmrl \
    --num-trials 50 \
    --num-reps 3
```

**Output:**
- CSV files with results: `output/v1/parameter_sweeps/qlearning_sweep_seed42.csv`
- Visualizations: `output/v1/parameter_sweeps/qlearning_sweep_viz.png`

### 9.2 Custom Parameter Ranges

In Python:
```python
from simulations.parameter_sweep import ParameterSweep

sweep = ParameterSweep(model_type='qlearning')

results = sweep.sweep_qlearning_parameters(
    alpha_range=[0.1, 0.3, 0.5, 0.7],
    beta_range=[1, 2, 3, 5, 10],
    set_sizes=[2, 3, 5, 6],
    num_trials=100,
    num_reps=5
)

# Results are saved automatically and returned as DataFrame
print(results.head())
```

### 9.3 Interactive Exploration (Jupyter)

For real-time parameter exploration with interactive widgets:

```python
from simulations.interactive_exploration import explore_qlearning_interactive
explore_qlearning_interactive()
```

```python
from simulations.interactive_exploration import explore_wmrl_interactive
explore_wmrl_interactive()
```

```python
from simulations.interactive_exploration import compare_models_interactive
compare_models_interactive()
```

**Features:**
- Interactive sliders for all parameters
- Real-time learning curve updates
- Q-value heatmaps
- WM buffer visualization
- Side-by-side model comparisons

### 9.4 Analysis of Parameter Sweep Results

Load and analyze sweep results:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
results = pd.read_csv('output/v1/parameter_sweeps/qlearning_sweep_seed42.csv')

# Find optimal parameters
best = results.loc[results['accuracy'].idxmax()]
print(f"Best parameters: α={best['alpha']}, β={best['beta']}")

# Plot set-size effects
for alpha in [0.1, 0.3, 0.5]:
    subset = results[results['alpha'] == alpha]
    grouped = subset.groupby('set_size')['accuracy'].mean()
    plt.plot(grouped.index, grouped.values, marker='o', label=f'α={alpha}')

plt.xlabel('Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

### 9.5 Use Cases for Parameter Sweeps

1. **Model Development:**
   - Understand parameter interactions
   - Identify sensitive vs robust parameters
   - Find reasonable parameter ranges

2. **Hypothesis Generation:**
   - Which parameters predict set-size effects?
   - When does WM-RL outperform Q-learning?
   - How does capacity limit affect performance?

3. **Prior Specification:**
   - Inform prior distributions for Bayesian fitting
   - Identify realistic parameter ranges
   - Avoid flat/uninformative priors

4. **Simulation Studies:**
   - Power analysis for detecting parameter differences
   - Sample size planning
   - Model recovery validation

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
