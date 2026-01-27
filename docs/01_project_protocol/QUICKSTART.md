# Quick Start: Command-Line Workflows

Complete command-line reference for running simulations, fitting models, and comparing results.

---

## Full Simulation & Comparison Pipeline

### Workflow 1: Parameter Sweeps (Explore Model Behavior)

```bash
# 1. Run a quick parameter sweep to test models
python scripts/simulations/example_parameter_sweep.py

# 2. Visualize the results
python scripts/simulations/example_visualize_sweeps.py
```

**Outputs:**
- `output/parameter_sweeps/tiny_sweep_qlearning.csv` - Q-learning results
- `output/parameter_sweeps/tiny_sweep_wmrl.csv` - WM-RL results
- `figures/parameter_sweeps/qlearning_individual.png` - Q-learning 4-panel plot
- `figures/parameter_sweeps/wmrl_individual.png` - WM-RL 4-panel plot
- `figures/parameter_sweeps/comparative_accuracy.png` - Model comparison
- `figures/parameter_sweeps/comparative_heatmaps.png` - Parameter spaces

**What this tells you:**
- How parameters affect model performance
- Which model performs better overall
- Optimal parameter ranges for each model

---

### Workflow 2: Fit Models to Behavioral Data

```bash
# PREREQUISITE: Process behavioral data first
python scripts/01_parse_raw_data.py
python scripts/02_create_collated_csv.py
python scripts/03_create_task_trials_csv.py
python scripts/04_create_summary_csv.py

# Option A: Fit both models sequentially
python scripts/fitting/fit_to_data.py --model qlearning
python scripts/fitting/fit_to_data.py --model wmrl

# Option B: Fit both models in one command
python scripts/fitting/fit_to_data.py --model both
```

**Outputs:**
- `output/fitting/qlearning_posterior_TIMESTAMP.nc` - Full Q-learning posterior
- `output/fitting/qlearning_summary_TIMESTAMP.csv` - Q-learning parameter estimates
- `output/fitting/wmrl_posterior_TIMESTAMP.nc` - Full WM-RL posterior
- `output/fitting/wmrl_summary_TIMESTAMP.csv` - WM-RL parameter estimates

**Estimated Runtime:**
- Q-learning: 30-60 minutes
- WM-RL: 60-120 minutes (more parameters)

---

### Workflow 3: Model Comparison (Find Winning Model)

```bash
# Learn about model comparison (educational tutorial)
python scripts/analysis/example_model_comparison.py

# Run the full model comparison
python scripts/analysis/run_model_comparison.py
```

**Requirements:**
- Fitted posteriors must exist in `output/fitting/`
- Behavioral data must exist at `output/task_trials_long.csv`

**Outputs:**
- `output/model_comparison/comparison_all_criteria.csv` - Full comparison table
- `figures/model_comparison/information_criteria_comparison.png` - 4-panel plot
- `figures/model_comparison/model_weights.png` - Model probabilities

**What you get:**
- **BIC, AIC, WAIC, LOO** scores for each model
- **Rankings** (1 = best model)
- **Model weights** (probability each model is best)
- **Evidence strength** (weak/positive/strong/very strong)

**How to interpret:**
```
LOWER scores = BETTER model

Δ (Delta) = difference between models:
  Δ < 2   → Weak evidence (models similar)
  Δ 2-6   → Positive evidence
  Δ 6-10  → Strong evidence
  Δ > 10  → Very strong evidence

Example output:
  model       AIC     BIC     WAIC    LOO
  Q-Learning  2547.0  2631.8  2545.2  2546.1  ← WINNER
  WM-RL       2574.4  2759.2  2568.9  2570.3

ΔBIC = 127.4 → Very strong evidence for Q-Learning
```

---

## Common Use Cases

### Use Case 1: "Which model fits my data better?"

```bash
# 1. Fit both models
python scripts/fitting/fit_to_data.py --model both

# 2. Compare them
python scripts/analysis/run_model_comparison.py
```

**Look at:**
- `output/model_comparison/comparison_all_criteria.csv`
- Check which model has rank=1 across all criteria
- Check Δ values for evidence strength

---

### Use Case 2: "What are the best parameters for each model?"

```bash
# Option A: From parameter sweeps (simulation)
python scripts/simulations/example_parameter_sweep.py
python scripts/simulations/example_visualize_sweeps.py
# Look at console output for "BEST PARAMETERS" section

# Option B: From fitted data (real participants)
python scripts/fitting/fit_to_data.py --model qlearning
# Check output/fitting/qlearning_summary_TIMESTAMP.csv
# Look at 'mean' column for each parameter
```

---

### Use Case 3: "How do parameters affect behavior?"

```bash
# Run parameter sweep
python scripts/simulations/parameter_sweep.py --model qlearning --num-trials 100 --num-reps 5

# Visualize results
python scripts/simulations/example_visualize_sweeps.py
```

**Look at:**
- `figures/parameter_sweeps/qlearning_individual.png` - See how α+, α-, β affect accuracy
- `figures/parameter_sweeps/wmrl_individual.png` - See how K, ρ, φ affect accuracy

---

### Use Case 4: "Test if my fitting procedure works (parameter recovery)"

```bash
# 1. Generate synthetic data with known parameters
python scripts/simulations/generate_data.py \
    --model qlearning \
    --n-participants 50 \
    --num-blocks 21 \
    --trials-per-block 100 \
    --add-noise \
    --seed 42

# 2. Fit the model to synthetic data
python scripts/fitting/fit_to_data.py \
    --model qlearning \
    --data output/simulated_data_qlearning_TIMESTAMP.csv

# 3. Check if recovered parameters match true parameters
# Compare output/fitting/qlearning_summary_TIMESTAMP.csv with known values
```

---

## Directory Structure (New Flat Organization)

```
output/
├── parameter_sweeps/          # Parameter sweep results
│   ├── tiny_sweep_qlearning.csv
│   ├── tiny_sweep_wmrl.csv
│   └── [other sweep files]
├── fitting/                   # Fitted model posteriors
│   ├── qlearning_posterior_TIMESTAMP.nc
│   ├── qlearning_summary_TIMESTAMP.csv
│   ├── wmrl_posterior_TIMESTAMP.nc
│   └── wmrl_summary_TIMESTAMP.csv
├── model_comparison/          # Model comparison results
│   └── comparison_all_criteria.csv
└── [behavioral data files]

figures/
├── parameter_sweeps/          # Parameter sweep visualizations
│   ├── qlearning_individual.png
│   ├── wmrl_individual.png
│   ├── comparative_accuracy.png
│   └── comparative_heatmaps.png
├── model_comparison/          # Model comparison visualizations
│   ├── information_criteria_comparison.png
│   └── model_weights.png
└── behavioral/                # Behavioral data plots
```

---

## Information Criteria Reference

### What They Measure

**AIC (Akaike Information Criterion)**
- Formula: `AIC = 2k - 2·log(L)`
- Balances fit vs complexity
- Moderate complexity penalty

**BIC (Bayesian Information Criterion)**
- Formula: `BIC = k·log(n) - 2·log(L)`
- Stronger complexity penalty than AIC
- Preferred for large datasets

**WAIC (Widely Applicable IC)**
- Fully Bayesian (uses full posterior)
- More robust for hierarchical models
- Better uncertainty quantification

**LOO (Leave-One-Out Cross-Validation)**
- Gold standard for model comparison
- Estimates out-of-sample prediction
- Most robust, but computationally expensive

### Which to Trust?

1. **If all agree:** Strong conclusion, trust the consensus
2. **If BIC differs from WAIC/LOO:** Trust WAIC/LOO (better for hierarchical models)
3. **General recommendation:**
   - PRIMARY: LOO
   - SECONDARY: WAIC
   - CONTEXT: BIC for parsimony
   - REPORT: All four for transparency

---

## Quick Diagnostics

### Check if Model Fitting Worked

```python
import arviz as az

# Load posterior
trace = az.from_netcdf('output/fitting/qlearning_posterior_TIMESTAMP.nc')

# Check convergence (R-hat should be < 1.01)
summary = az.summary(trace, var_names=['mu_alpha_pos', 'mu_alpha_neg', 'mu_beta'])
print(summary[['r_hat']])

# Check for divergences (should be 0 or very few)
print(f"Divergences: {trace.sample_stats.diverging.sum().values}")
```

### Extract Fitted Parameters

```python
import arviz as az

trace = az.from_netcdf('output/fitting/qlearning_posterior_TIMESTAMP.nc')

# Group-level means
mu_alpha_pos = trace.posterior['mu_alpha_pos'].mean().values
mu_alpha_neg = trace.posterior['mu_alpha_neg'].mean().values
mu_beta = trace.posterior['mu_beta'].mean().values

print(f"Group mean α+ = {mu_alpha_pos:.3f}")
print(f"Group mean α- = {mu_alpha_neg:.3f}")
print(f"Group mean β = {mu_beta:.3f}")

# Individual participant parameters
alphas = trace.posterior['alpha_pos'].mean(dim=['chain', 'draw']).values
print(f"Individual α+ values: {alphas}")
```

---

## Troubleshooting

### "No fitted models found"
```bash
# Run fitting first
python scripts/fitting/fit_to_data.py --model both
```

### "No behavioral data found"
```bash
# Process raw data first
python scripts/01_parse_raw_data.py
python scripts/03_create_task_trials_csv.py
```

### "No parameter sweep results found"
```bash
# Run parameter sweep first
python scripts/simulations/example_parameter_sweep.py
```

### Fitting takes too long
```bash
# Use fewer samples for testing
python scripts/fitting/fit_to_data.py --model qlearning --samples 1000 --tune 500
```

### High divergences during fitting
- Check data preprocessing (ensure 0-indexing)
- Try tighter priors in `scripts/fitting/pymc_models.py`
- Increase target_accept: add `--target-accept 0.99`

---

## Example: Complete Analysis from Scratch

```bash
# ============================================================================
# STEP 1: PROCESS BEHAVIORAL DATA
# ============================================================================
python scripts/01_parse_raw_data.py
python scripts/02_create_collated_csv.py
python scripts/03_create_task_trials_csv.py
python scripts/04_create_summary_csv.py

# ============================================================================
# STEP 2: EXPLORE MODEL BEHAVIOR (OPTIONAL BUT RECOMMENDED)
# ============================================================================
python scripts/simulations/example_parameter_sweep.py
python scripts/simulations/example_visualize_sweeps.py

# ============================================================================
# STEP 3: FIT MODELS TO DATA
# ============================================================================
python scripts/fitting/fit_to_data.py --model both

# ============================================================================
# STEP 4: COMPARE MODELS
# ============================================================================
python scripts/analysis/run_model_comparison.py

# ============================================================================
# STEP 5: CHECK RESULTS
# ============================================================================
# Look at:
#   - output/model_comparison/comparison_all_criteria.csv
#   - figures/model_comparison/*.png
#   - output/fitting/*_summary_*.csv
```

---

## File Naming Conventions

All outputs include timestamps to avoid overwriting:

```
qlearning_posterior_20250114_143022.nc
                     ^^^^^^^^^^^^^^
                     YYYYMMDD_HHMMSS

Most scripts automatically load the most recent file:
- Sorted alphabetically (which = sorted by timestamp)
- Takes last file: sorted(files)[-1]
```

---

## Command-Line Arguments Reference

### fit_to_data.py
```bash
python scripts/fitting/fit_to_data.py --help

Options:
  --model {qlearning,wmrl,both}  Which model to fit
  --data PATH                     Path to behavioral data CSV
  --chains INT                    Number of MCMC chains (default: 4)
  --samples INT                   Samples per chain (default: 2000)
  --tune INT                      Tuning samples (default: 1000)
  --target-accept FLOAT           Target acceptance rate (default: 0.9)
```

### parameter_sweep.py
```bash
python scripts/simulations/parameter_sweep.py --help

Options:
  --model {qlearning,wmrl,both}  Which model to sweep
  --num-trials INT                Trials per simulation
  --num-reps INT                  Repetitions per condition
  --seed INT                      Random seed
```

### generate_data.py
```bash
python scripts/simulations/generate_data.py --help

Options:
  --model {qlearning,wmrl}       Which model to use
  --n-participants INT           Number of participants
  --num-blocks INT               Number of task blocks
  --trials-per-block INT         Trials per block
  --add-noise                    Add behavioral noise
  --posteriors PATH              Use fitted posteriors for parameters
  --seed INT                     Random seed
```

---

## Next Steps After Finding Winning Model

Once you've identified the best model:

1. **Extract parameters** from `output/fitting/WINNER_summary_TIMESTAMP.csv`
2. **Analyze individual differences** in parameters
3. **Correlate parameters with trauma measures** (IES-R, LEC-5)
4. **Create publication figures** showing model fits
5. **Run posterior predictive checks** to validate model
6. **Test group differences** (PTSD vs control)

See `docs/ANALYSIS_PIPELINE.md` for detailed post-comparison analyses.

---

## Getting Help

```bash
# Any script with --help flag
python scripts/fitting/fit_to_data.py --help
python scripts/simulations/parameter_sweep.py --help

# Educational tutorials
python scripts/analysis/example_model_comparison.py  # Learn about IC
python scripts/simulations/example_visualize_sweeps.py  # Learn about viz

# Full documentation
docs/ANALYSIS_PIPELINE.md  # Complete pipeline reference
docs/MODEL_REFERENCE.md    # Model equations and details
```
