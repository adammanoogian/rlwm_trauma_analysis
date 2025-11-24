# RLWM Trauma Analysis Pipeline

Complete workflow for analyzing RLWM task data using reinforcement learning models and Bayesian inference.

## Overview

This pipeline integrates:
1. **Data Processing**: Clean and organize jsPsych behavioral data
2. **Model Simulation**: Generate synthetic data using RL models
3. **Model Fitting**: Fit models to human data using hierarchical Bayesian inference
   - **PyMC**: Agent-based, slower but consistent with simulations
   - **JAX/NumPyro**: Functional, 10-100x faster, gradient-based NUTS sampler (RECOMMENDED)
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
├── scripts/fitting/             # Bayesian fitting
│   ├── pymc_models.py           # PyMC: Uses agent classes via unified_simulator
│   ├── fit_to_data.py           # PyMC: Main fitting script
│   ├── jax_likelihoods.py       # JAX: Pure functional likelihoods (JIT-compiled)
│   ├── numpyro_models.py        # NumPyro: Hierarchical Bayesian models
│   └── fit_with_jax.py          # JAX/NumPyro: Main fitting script (RECOMMENDED)
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

**Key Design Principle**: The codebase provides two parallel approaches for model fitting:

1. **Agent-Based (PyMC)**: All simulation, fitting, and parameter exploration use the **same agent implementations** from `models/`. This eliminates code duplication and ensures consistency.

2. **Functional (JAX/NumPyro)**: Pure functional likelihoods for production fitting. Separate implementation optimized for speed and gradient-based sampling.

### Code Flow: Agent-Based (PyMC)

```
Agent Classes (models/)
    ↓
Unified Simulator (simulations/unified_simulator.py)
    ↓
    ├─→ Parameter Sweeps (parameter_sweep.py)
    ├─→ Data Generation (generate_data.py)
    └─→ PyMC Fitting (fitting/pymc_models.py)
```

### Code Flow: Functional (JAX/NumPyro)

```
JAX Likelihoods (scripts/fitting/jax_likelihoods.py)
    ↓
NumPyro Models (scripts/fitting/numpyro_models.py)
    ↓
Production Fitting (scripts/fitting/fit_with_jax.py)
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

### Benefits of Agent-Based Approach

- ✅ **Single source of truth**: Agent logic defined once in `models/`
- ✅ **Guaranteed consistency**: PyMC fits the same model that parameter sweeps test
- ✅ **Easy to extend**: Add new model? Just update agent class
- ✅ **Flexible simulation**: Can simulate with both fixed and sampled parameters
- ⚠️ **Trade-off**: Slower sampling (Metropolis, no gradients)

### Benefits of Functional Approach (JAX/NumPyro)

- ✅ **10-100x faster**: JIT compilation via XLA
- ✅ **Gradient-based NUTS**: Better convergence, higher ESS
- ✅ **Native JAX operations**: Automatic differentiation, parallel execution
- ✅ **Block-structured processing**: Efficient sequential operations via `jax.lax.scan()`
- ⚠️ **Trade-off**: Separate implementation from agent classes

### PyMC Integration (Agent-Based)

Since agent classes use pure Python (not PyTensor), we use **Metropolis sampler** (no gradients) instead of NUTS:

```python
with build_qlearning_model(data) as model:
    trace = pm.sample(draws=2000, step=pm.Metropolis())
```

Trade-off: Slower sampling than NUTS, but ensures exact consistency between simulation and fitting.

### JAX/NumPyro Integration (Functional)

Pure functional likelihoods enable gradient-based NUTS sampler:

```python
from scripts.fitting.numpyro_models import qlearning_hierarchical_model, run_inference

mcmc = run_inference(
    model=qlearning_hierarchical_model,
    model_args={'participant_data': participant_data},
    num_warmup=1000,
    num_samples=2000,
    num_chains=4
)
```

Trade-off: 10-100x faster but requires separate implementation from agent classes.

---

## Pipeline Stages

### Stage 1: Data Cleaning and Behavioral Analysis

Clean raw jsPsych data, compute behavioral metrics, and generate visualizations.

#### 1.1 Standard Pipeline (With Survey Data)

For participants with complete demographic and survey data:

**Scripts:**
```bash
# Parse raw jsPsych data
python scripts/01_parse_raw_data.py
python scripts/02_create_collated_csv.py
python scripts/03_create_task_trials_csv.py
python scripts/04_create_summary_csv.py
```

**Outputs:**
- `output/parsed_demographics.csv`: Demographic information
- `output/parsed_survey1.csv`: LEC-5 trauma exposure data
- `output/parsed_survey2.csv`: IES-R PTSD symptoms
- `output/parsed_task_trials.csv`: Trial-level task data
- `output/collated_participant_data.csv`: Combined participant data
- `output/task_trials_long.csv`: Trial-level data (for model fitting)
- `output/summary_participant_metrics.csv`: Participant-level aggregates

#### 1.2 Extended Pipeline (All Participants)

For datasets including participants with partial data or anonymous IDs:

**Parse all participants (including partial data):**
```bash
python scripts/parse_all_participants.py
```

This script:
- Processes all CSV files in `data/` directory
- Assigns anonymous IDs to participants without sona_id
- Includes participants with ≥100 trials (partial completion)
- Uses ID mapping from `data/participant_id_mapping.json`

**Output:**
- `output/task_trials_long_all_participants.csv`: Task data for all participants

#### 1.3 Behavioral Visualizations

**Activate environment (if using conda):**
```bash
# On Windows
conda activate ds_env

# Or use full path to Python executable
# /c/Users/USERNAME/AppData/Local/miniforge3/envs/ds_env/python.exe
```

**Generate human performance visualizations:**
```bash
python scripts/analysis/visualize_human_performance.py --data output/task_trials_long_all_participants.csv
```

**Outputs:**
- `figures/behavioral_summary/human_stimulus_performance_analysis.png`: Combined 2-panel figure
- `figures/behavioral_summary/human_stimulus_learning_curve.png`: Detailed learning curves
- `figures/behavioral_summary/human_stimulus_encounter_performance.png`: Performance by position
- `output/behavioral_summary/human_stimulus_based_data.csv`: Processed encounter data

**Generate scale distributions:**
```bash
python scripts/analysis/visualize_scale_distributions.py
```

**Outputs:**
- `figures/behavioral_summary/scale_distributions.png`: LEC-5 and IES-R distributions
- `figures/behavioral_summary/performance_distributions.png`: Task performance metrics

**Generate correlation matrices:**
```bash
python scripts/analysis/visualize_scale_correlations.py
```

**Outputs:**
- `figures/behavioral_summary/scale_correlations.png`: Correlation heatmap
- `figures/behavioral_summary/trauma_performance_scatterplots.png`: Trauma-performance relationships

**Generate summary report:**
```bash
python scripts/analysis/summarize_behavioral_data.py
```

**Output:**
- `output/behavioral_summary/data_summary_report.txt`: Comprehensive data summary

#### 1.4 Complete Behavioral Analysis Pipeline

Run all behavioral analysis steps in sequence:

```bash
# Parse all participants
python scripts/parse_all_participants.py

# Generate all visualizations
python scripts/analysis/visualize_human_performance.py --data output/task_trials_long_all_participants.csv
python scripts/analysis/visualize_scale_distributions.py
python scripts/analysis/visualize_scale_correlations.py
python scripts/analysis/summarize_behavioral_data.py
```

#### 1.5 Complete Pipeline from Raw Data (One Command)

Run the **entire data processing and analysis pipeline** with a single script (excludes model fitting/simulation).

**Two-Folder Setup:**
- **Experiment folder**: `../rlwm_trauma/data/` - Where experiment software saves new participant data
- **Analysis folder**: `rlwm_trauma_analysis/data/` - Where analysis pipeline reads data from

**Automatic Data Sync:**
By default, the pipeline automatically syncs new data from the experiment folder to the analysis folder before processing. This ensures you're always analyzing the latest data without manually copying files.

**Standard Usage (with automatic sync):**
```bash
# Python script (recommended for Windows)
python run_data_pipeline.py

# OR Bash script (Linux/Mac or Git Bash on Windows)
bash run_data_pipeline.sh
```

**Skip sync if no new data:**
```bash
python run_data_pipeline.py --no-sync
```

**Just sync data without analysis:**
```bash
python sync_experiment_data.py
```

**This pipeline will:**
1. **Sync data** from experiment folder (unless `--no-sync` is used)
2. Update participant ID mapping
3. Parse raw jsPsych data (demographics, LEC-5, IES-R, task trials)
4. Create collated participant data
5. Create task trials datasets (standard + all participants)
6. Generate summary metrics
7. Generate all behavioral visualizations
8. Generate scale distributions and correlations
9. Create comprehensive summary report

**Outputs generated:**
- `output/parsed_*.csv` - Parsed demographic and survey data
- `output/task_trials_long.csv` - Standard task trials
- `output/task_trials_long_all_participants.csv` - Extended task trials
- `output/collated_participant_data.csv` - Combined participant data
- `output/summary_participant_metrics.csv` - Behavioral metrics
- `figures/behavioral_summary/*.png` - All visualizations
- `output/behavioral_summary/data_summary_report.txt` - Summary report
- `data/sync_log.txt` - Log of all data sync operations

**Data Sync Safety Features:**
- **Read-only** access to experiment folder - never modifies source data
- Only copies files matching pattern `rlwm_trauma_PARTICIPANT_SESSION_*.csv`
- Skips files already present (compares by filename and timestamp)
- Updates existing files only if source is newer
- Logs all operations to `data/sync_log.txt`
- Validates experiment folder exists before attempting sync

**Key Variables:**
- Demographics: age, gender, race, education
- LEC-5: Trauma exposure (total_events, personal_events, sum_exposures)
- IES-R: PTSD symptoms (total, intrusion, avoidance, hyperarousal)
- Task metrics: accuracy by set size, RT, learning slopes, reversal adaptation
- Stimulus-based metrics: encounters per stimulus, pre/post-reversal performance

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

Two approaches available for fitting models to human behavioral data:

1. **PyMC (Agent-Based)**: Uses agent classes, slower but consistent with simulations
2. **JAX/NumPyro (Functional)**: Pure JAX, 10-100x faster, gradient-based NUTS sampler

**Recommendation**: Use JAX/NumPyro (Stage 4B) for production fitting. Use PyMC (Stage 4A) for validation and debugging.

#### Stage 4A: PyMC Fitting (Agent-Based)

##### 4A.1 Prepare Data

Ensure `output/task_trials_long.csv` exists with columns:
- `sona_id`: Participant ID
- `block`: Block number (≥3 for main task)
- `trial`: Trial within block
- `stimulus`: Stimulus ID (will be converted to 0-indexed)
- `key_press`: Action taken (0, 1, or 2)
- `correct`: Whether response was correct (0 or 1)

##### 4A.2 Fit Q-Learning

```bash
python fitting/fit_to_data.py \
    --model qlearning \
    --data output/task_trials_long.csv \
    --chains 4 \
    --samples 2000 \
    --tune 1000
```

**Estimated Runtime**: 2-4 hours (Metropolis sampler, no gradients)

**Outputs:**
- `output/v1/qlearning_posterior_TIMESTAMP.nc`: Full posterior samples
- `output/v1/qlearning_summary_TIMESTAMP.csv`: Parameter summaries

##### 4A.3 Fit WM-RL Hybrid

```bash
python fitting/fit_to_data.py \
    --model wmrl \
    --chains 4 \
    --samples 2000
```

**Estimated Runtime**: 4-6 hours (more parameters, Metropolis sampler)

##### 4A.4 Fit Both and Compare

```bash
python fitting/fit_to_data.py --model both
```

Automatically compares models using WAIC and LOO.

**Output:**
- `output/v1/model_comparison_TIMESTAMP.csv`

---

#### Stage 4B: JAX/NumPyro Fitting (Functional) - RECOMMENDED

Fast, gradient-based fitting using pure JAX likelihoods and NumPyro NUTS sampler.

**Key Advantages**:
- **10-100x faster** than PyMC (XLA compilation)
- **NUTS sampler** (Hamiltonian Monte Carlo with gradients)
- **Better convergence** (gradient-based exploration)
- **Higher effective sample size** per iteration

**Files**:
- `scripts/fitting/jax_likelihoods.py` - JIT-compiled likelihood functions
- `scripts/fitting/numpyro_models.py` - Hierarchical Bayesian models
- `scripts/fitting/fit_with_jax.py` - Main fitting script

##### 4B.1 Prepare Data

Same requirements as PyMC (Stage 4A.1):
- `output/task_trials_long.csv` or `output/task_trials_long_all_participants.csv`
- Columns: `sona_id`, `block`, `stimulus`, `key_press`, `correct`

##### 4B.2 Fit Q-Learning with JAX/NumPyro

**Basic Usage**:

```bash
python scripts/fitting/fit_with_jax.py \
    --data output/task_trials_long_all_participants.csv \
    --chains 4 \
    --warmup 1000 \
    --samples 2000 \
    --save-plots
```

**Quick Test** (fewer samples for validation):

```bash
python scripts/fitting/fit_with_jax.py \
    --data output/task_trials_long.csv \
    --chains 2 \
    --warmup 500 \
    --samples 1000
```

**Custom Output**:

```bash
python scripts/fitting/fit_with_jax.py \
    --data output/task_trials_long.csv \
    --output output/v2/ \
    --min-block 3 \
    --seed 42
```

**Estimated Runtime**:
- Compilation: ~30 seconds (first time only)
- Sampling: ~20-40 minutes for 2000 samples × 4 chains (vs 2-4 hours for PyMC)

**Outputs**:
- `output/qlearning_jax_posterior_TIMESTAMP.nc`: Full posterior samples (ArviZ format)
- `output/qlearning_jax_summary_TIMESTAMP.csv`: Parameter summaries with R-hat, ESS
- `output/qlearning_jax_trace_TIMESTAMP.png`: Trace plots (if --save-plots)
- `output/qlearning_jax_posterior_TIMESTAMP.png`: Posterior distributions (if --save-plots)

##### 4B.3 Programmatic Usage

```python
from scripts.fitting.numpyro_models import (
    qlearning_hierarchical_model,
    prepare_data_for_numpyro,
    run_inference,
    samples_to_arviz
)
import pandas as pd
import arviz as az

# Load data
data = pd.read_csv('output/task_trials_long_all_participants.csv')

# Prepare for NumPyro (block-structured format)
participant_data = prepare_data_for_numpyro(
    data,
    participant_col='sona_id',
    block_col='block',
    stimulus_col='stimulus',
    action_col='key_press',
    reward_col='reward'
)

# Run MCMC with NUTS sampler
mcmc = run_inference(
    model=qlearning_hierarchical_model,
    model_args={'participant_data': participant_data},
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
    seed=42,
    target_accept_prob=0.8,  # Control step size
    max_tree_depth=10        # Control trajectory length
)

# Get samples
samples = mcmc.get_samples()

# Group-level parameters
print("Group-level posteriors:")
print(f"  μ_α+ = {samples['mu_alpha_pos'].mean():.3f} ± {samples['mu_alpha_pos'].std():.3f}")
print(f"  μ_α- = {samples['mu_alpha_neg'].mean():.3f} ± {samples['mu_alpha_neg'].std():.3f}")
print(f"  μ_β  = {samples['mu_beta'].mean():.3f} ± {samples['mu_beta'].std():.3f}")

# Individual parameters (shape: [n_samples, n_participants])
print("\nIndividual posteriors:")
for i in range(samples['alpha_pos'].shape[1]):
    alpha_pos_i = samples['alpha_pos'][:, i].mean()
    alpha_neg_i = samples['alpha_neg'][:, i].mean()
    beta_i = samples['beta'][:, i].mean()
    print(f"  Participant {i}: α+={alpha_pos_i:.3f}, α-={alpha_neg_i:.3f}, β={beta_i:.3f}")

# Convert to ArviZ and save
idata = samples_to_arviz(mcmc, data)
idata.to_netcdf('output/posterior_jax.nc')

# Check diagnostics
summary = az.summary(idata)
print("\nConvergence diagnostics:")
print(summary[['mean', 'sd', 'r_hat', 'ess_bulk', 'ess_tail']])
```

##### 4B.4 Fit WM-RL Hybrid (Future)

WM-RL model is implemented in `numpyro_models.py` but not yet integrated into the main fitting script. To fit WM-RL:

```python
from scripts.fitting.numpyro_models import wmrl_hierarchical_model

# Prepare data (must include set_size column)
participant_data = prepare_data_for_numpyro(
    data,
    participant_col='sona_id',
    block_col='block',
    stimulus_col='stimulus',
    action_col='key_press',
    reward_col='reward',
    set_size_col='set_size'  # Required for WM-RL
)

# Run inference
mcmc = run_inference(
    model=wmrl_hierarchical_model,
    model_args={'participant_data': participant_data},
    num_warmup=1000,
    num_samples=2000,
    num_chains=4
)
```

##### 4B.5 Model Comparison (JAX/NumPyro)

```python
import arviz as az

# Load posteriors
idata_qlearning = az.from_netcdf('output/qlearning_jax_posterior.nc')
idata_wmrl = az.from_netcdf('output/wmrl_jax_posterior.nc')

# Compare using WAIC/LOO
comparison = az.compare({
    'Q-Learning': idata_qlearning,
    'WM-RL': idata_wmrl
})

print(comparison)
```

##### 4B.6 Performance Tips

**For Faster Sampling**:
- Use fewer chains (2 instead of 4) if compute-limited
- Reduce warmup samples (500-800) if priors are informative
- Use `target_accept_prob=0.9` for more conservative step sizes (fewer divergences)

**For Better Convergence**:
- Increase `target_accept_prob` to 0.95 if seeing divergences
- Increase `max_tree_depth` to 12 if seeing max tree depth warnings
- Check that data is properly 0-indexed (stimuli/actions)
- Verify block boundaries are correct

**Debugging**:
- Run with `test_compilation=True` in `run_inference()` to catch errors early
- Use smaller dataset first (single participant or block)
- Check likelihood values are finite: `print(log_lik)` should be negative, not NaN/-inf

##### 4B.7 Comparison: PyMC vs JAX/NumPyro

| Aspect | PyMC (4A) | JAX/NumPyro (4B) |
|--------|-----------|------------------|
| **Speed** | 2-4 hours | 20-40 minutes |
| **Sampler** | Metropolis (no gradients) | NUTS (gradient-based) |
| **Convergence** | Slower (random walk) | Faster (Hamiltonian) |
| **ESS** | Lower | Higher |
| **Code** | Uses agent classes | Pure functional JAX |
| **Consistency** | Same as simulations | Separate implementation |
| **Best for** | Validation, debugging | Production, large datasets |

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

### Workflow 2: Fit to Human Data (PyMC)

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

### Workflow 2B: Fit to Human Data (JAX/NumPyro) - RECOMMENDED

```bash
# 1. Ensure data is processed
python scripts/parse_all_participants.py  # Creates task_trials_long_all_participants.csv

# 2. Fit Q-learning with JAX/NumPyro (FAST!)
python scripts/fitting/fit_with_jax.py \
    --data output/task_trials_long_all_participants.csv \
    --chains 4 \
    --warmup 1000 \
    --samples 2000 \
    --save-plots

# Results saved to output/qlearning_jax_*

# 3. Analyze posteriors
python -c "
import arviz as az
idata = az.from_netcdf('output/qlearning_jax_posterior_TIMESTAMP.nc')
print(az.summary(idata, var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta']))
"

# 4. Create visualizations (saves to figures/)
python scripts/visualization/quick_arviz_plots.py \
    --posterior output/qlearning_jax_posterior_TIMESTAMP.nc
```

**Runtime Comparison**:
- PyMC (Workflow 2): ~2-4 hours
- JAX/NumPyro (Workflow 2B): ~20-40 minutes (5-10x faster)

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

## Stage 10: Trauma-Based Grouping Analysis

### 10.1 Overview

Analyze participant subgroups based on trauma exposure (LEC-5) and PTSD symptoms (IES-R) using two complementary approaches:
1. **Hypothesis-driven**: Theory-based 3-group classification
2. **Unsupervised**: Hierarchical clustering to discover natural groupings

### 10.2 Rationale for Methodological Choices

**Why Distribution-Based Cutoffs (Median Splits)?**

For exploratory analysis with small samples (N=17), distribution-based cutoffs are preferred over clinical thresholds for several reasons:

1. **Adequate sample sizes**: Clinical cutoffs (e.g., IES-R ≥ 33 for probable PTSD) may create severely imbalanced groups. With N=17, if only 2-3 participants exceed clinical threshold, statistical comparisons become underpowered.

2. **Relative differences matter**: In exploratory research, we're interested in *relative* differences within our sample (who has higher vs. lower trauma/symptoms) rather than absolute diagnostic classification.

3. **Sample-specific interpretation**: Median splits create balanced groups while capturing meaningful variation in your specific cohort, which is appropriate for pilot/exploratory studies.

4. **Statistical power**: Equal group sizes maximize power for group comparisons with limited N.

**Clinical Thresholds (For Reference):**
- IES-R ≥ 33: Probable PTSD diagnosis (Creamer et al., 2003)
- LEC-5: No established cutoff (it's an exposure checklist, not diagnostic)

**When to Use Clinical Cutoffs:**
- Larger samples (N > 60) where group imbalance is tolerable
- Clinical/diagnostic research questions
- Comparing to published literature using same thresholds
- Confirmatory rather than exploratory analysis

**Why Hierarchical Clustering Over K-Means?**

For this dataset, hierarchical clustering with Ward linkage is recommended:

1. **No pre-specified k**: You don't know if 2, 3, or 4 groups is optimal. Hierarchical clustering tests all simultaneously via the dendrogram.

2. **Small sample size**: With N=17, computational cost is negligible, making hierarchical's slower speed irrelevant.

3. **Deterministic results**: Unlike k-means (which uses random initialization), hierarchical clustering produces identical results every run without needing to set random seeds.

4. **Rich visualization**: The dendrogram shows not just *which* cluster each participant belongs to, but *how similar* they are to each other at different levels of the hierarchy.

5. **Flexible cluster shapes**: Doesn't assume spherical/equal-variance clusters like k-means.

6. **Individual-level interpretation**: With only 17 participants, you can examine *why* specific individuals grouped together by looking at their position in the tree.

**When K-Means Is Better:**
- Very large datasets (N > 1000) where hierarchical is too slow
- You have strong theoretical reasons to expect exactly k groups
- You want probabilistic cluster assignments (use GMM variant)

### 10.3 Run Grouping Analysis

**Hypothesis-driven + Hierarchical clustering:**
```bash
python scripts/analysis/trauma_grouping_analysis.py
```

**Outputs:**
- `figures/trauma_groups/hypothesis_groups_scatter.png`: 3-group classification visualization
- `figures/trauma_groups/hierarchical_dendrogram.png`: Full cluster hierarchy tree
- `figures/trauma_groups/cluster_comparison.png`: Comparing k=2, 3, 4 solutions
- `figures/trauma_groups/cluster_silhouette.png`: Quality metrics for each k
- `figures/trauma_groups/group_comparison_heatmap.png`: Cross-tab of methods
- `output/trauma_groups/group_assignments.csv`: All grouping solutions
- `output/trauma_groups/group_summary_stats.csv`: Descriptive stats per group
- `output/trauma_groups/clustering_metrics.csv`: Silhouette scores
- `output/trauma_groups/cutoff_values.csv`: Median values used

### 10.4 Validate Groups

Test whether groups differ on behavioral outcomes:

```bash
python scripts/analysis/validate_trauma_groups.py
```

**Statistical Tests:**
- ANOVA or Kruskal-Wallis (depending on normality)
- Post-hoc pairwise comparisons with Bonferroni correction
- Effect size calculations (eta-squared, Cohen's d)

**Outputs:**
- `figures/trauma_groups/behavioral_by_group.png`: Performance differences
- `output/trauma_groups/validation_report.txt`: Statistical results

### 10.5 Hypothesis-Driven Groups

**Group A (Low-Low)**: Low trauma exposure, low symptoms
- LEC-5 < median
- IES-R < median
- **Interpretation**: Minimal trauma history, no significant symptoms
- **Predicted behavior**: Baseline/control performance

**Group B (High-Low)**: High trauma exposure, low symptoms (Resilient)
- LEC-5 ≥ median
- IES-R < median
- **Interpretation**: Experienced trauma but no PTSD symptoms (resilience)
- **Predicted behavior**: May show trauma effects despite resilience, or compensatory performance

**Group C (High-High)**: High trauma exposure, high symptoms
- LEC-5 ≥ median
- IES-R ≥ median
- **Interpretation**: Trauma exposure with significant symptomatology
- **Predicted behavior**: Potential cognitive/learning deficits

**Note on Low LEC + High IES-R:**
This pattern (trauma symptoms without trauma exposure) is theoretically inconsistent and rare. If present, participants are flagged as "Excluded_Low_High" for review.

### 10.6 Interpreting Results

**With N=17:**
- Statistical power is limited (~80% power to detect large effects, d ≥ 1.2)
- **Focus on effect sizes** rather than p-values
- Treat as exploratory/pilot analysis
- Non-significant results ≠ no effect (may reflect insufficient power)

**Effect Size Interpretation:**
- **Small**: η² = 0.01, Cohen's d = 0.2
- **Medium**: η² = 0.06, Cohen's d = 0.5
- **Large**: η² = 0.14, Cohen's d = 0.8

**Silhouette Score Interpretation:**
- **0.7-1.0**: Strong, well-separated clusters
- **0.5-0.7**: Reasonable cluster structure
- **0.25-0.5**: Weak, overlapping clusters
- **< 0.25**: No substantial cluster structure

### 10.7 Comparing Methods

Cross-tabulation shows concordance between hypothesis-driven and clustering approaches:

- **High concordance**: Methods agree on grouping → robust, interpretable groups
- **Low concordance**: Methods disagree → data-driven structure differs from hypothesis
- **Chi-square test**: Tests whether association is significant

**Recommendations:**
1. If methods agree: Use hypothesis groups for primary analysis (more interpretable)
2. If methods disagree: Report both, explore discrepant cases
3. Always validate both against behavioral outcomes

### 10.8 Sample Size Considerations

**Current Study (N=17):**
- Appropriate for: Exploratory analysis, pilot study, effect size estimation
- Limitations: Underpowered for detecting small-medium effects, limited generalizability
- Recommendation: Treat as hypothesis-generating, plan larger replication

**For Confirmatory Analysis:**
- **N = 60-90** recommended for adequate power (80%) to detect medium effects
- Allows ~20-30 participants per group
- Enables more complex analyses (covariates, interactions, mediation)

### 10.9 Advanced Analyses (Future Directions)

**Include Subscales:**
```python
# In trauma_grouping_analysis.py, modify features:
features = [
    'lec_personal_events',  # Focus on personal trauma
    'ies_intrusion',        # Re-experiencing symptoms
    'ies_avoidance',        # Avoidance behaviors
    'ies_hyperarousal'      # Arousal symptoms
]
```

This may identify **symptom profile subtypes**:
- High intrusion, low avoidance
- High avoidance, low hyperarousal
- Balanced symptom profile

**Latent Profile Analysis (LPA):**
For larger samples (N > 100), consider LPA:
- Model-based clustering with better statistical properties
- Estimates class probabilities
- Allows covariate inclusion
- Standard in trauma research

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
