# RLWM Trauma Analysis Pipeline

Complete workflow for the RLWM Trauma Analysis project — from raw jsPsych data to computational model fitting and trauma-parameter regression.

## Overview

This pipeline processes raw jsPsych behavioral data from a Reinforcement Learning Working Memory (RLWM) task combined with trauma assessment surveys (LEC-5, IES-R). It covers:

1. **Data Processing** (scripts 01-04): Parse, collate, and summarize raw data
2. **Behavioral Analysis** (scripts 05-08): Descriptive stats, visualizations, trauma grouping, ANOVAs
3. **Simulations & Validation** (scripts 09-11): Synthetic data, parameter sweeps, model/parameter recovery
4. **Model Fitting** (scripts 12-14): MLE and Bayesian fitting, model comparison
5. **Results Analysis** (scripts 15-16): Parameter-trauma relationships, continuous regressions

**Sample:** N=48 participants (after exclusions from N=54 raw).

---

## Environment Setup

```bash
# Option 1: Use existing environment
conda activate ds_env

# Option 2: Create from environment.yml (recommended)
conda env create -f environment.yml
conda activate ds_env
```

For GPU-accelerated model fitting, see [cluster execution](#cluster-execution).

---

## Stage 1: Data Processing (Scripts 01-04)

Parse raw jsPsych JSON exports into analysis-ready CSV files.

```bash
python scripts/01_data_preprocessing/01_parse_raw_data.py         # Parse raw jsPsych data
python scripts/01_data_preprocessing/02_create_collated_csv.py     # Collate all participants (wide format)
python scripts/01_data_preprocessing/03_create_task_trials_csv.py  # Trial-level data (long format)
python scripts/01_data_preprocessing/04_create_summary_csv.py      # Summary metrics per participant
```

**Outputs:**
| File | Description |
|------|-------------|
| `output/parsed_demographics.csv` | Demographic information |
| `output/parsed_survey1.csv` | LEC-5 trauma exposure |
| `output/parsed_survey2.csv` | IES-R PTSD symptoms |
| `output/collated_participant_data.csv` | All participants, wide format |
| `output/task_trials_long.csv` | Main task trials (for fitting) |
| `output/task_trials_long_all.csv` | All blocks including practice |
| `output/summary_participant_metrics.csv` | Per-participant aggregates |

**Block structure:** Blocks 1-2 are practice, blocks 3-23 are main task. By default, `task_trials_long.csv` contains main task only.

---

## Stage 2: Behavioral Analysis (Scripts 05-08)

Generate descriptive statistics, visualizations, trauma group classifications, and statistical tests.

```bash
python scripts/02_behav_analyses/05_summarize_behavioral_data.py    # Behavioral summary stats
python scripts/02_behav_analyses/06_visualize_task_performance.py    # Learning curves, set-size effects
python scripts/02_behav_analyses/07_analyze_trauma_groups.py         # Trauma grouping + validation
python scripts/02_behav_analyses/08_run_statistical_analyses.py      # ANOVAs + descriptive tables
```

### Trauma Group Methodology

Script 07 creates hypothesis-driven groups based on median splits of LEC-5 (trauma exposure) and IES-R (PTSD symptoms):

| Group | LEC-5 | IES-R | Interpretation |
|-------|-------|-------|----------------|
| A (Low-Low) | < median | < median | Minimal trauma, baseline |
| B (High-Low) | >= median | < median | Exposed but resilient |
| C (High-High) | >= median | >= median | Exposed with symptoms |

**Why median splits?** With N=48, clinical cutoffs (e.g., IES-R >= 33 for probable PTSD) may create severely imbalanced groups. Median splits maximize statistical power by creating balanced groups while capturing relative variation within the sample. For confirmatory analyses with larger samples (N > 60), clinical thresholds are preferred.

The script also runs hierarchical clustering (Ward linkage) as a data-driven validation. High concordance between methods suggests robust groupings.

**Outputs:**
- `output/trauma_groups/group_assignments.csv` — group labels per participant
- `figures/trauma_groups/` — scatter plots, dendrograms, silhouette plots

---

## Stage 3: Simulations & Validation (Scripts 09-11)

Generate synthetic data and validate the fitting pipeline.

```bash
python scripts/03_model_prefitting/09_generate_synthetic_data.py  # Synthetic data generation
python scripts/03_model_prefitting/09_run_ppc.py                   # PPC analysis
python scripts/03_model_prefitting/10_run_parameter_sweep.py       # Systematic parameter exploration
python scripts/03_model_prefitting/11_run_model_recovery.py        # Parameter/model recovery
```

**Use cases:**
- **Parameter recovery:** Fit models to synthetic data with known parameters; verify recovery
- **Model recovery:** Generate data from each model; verify model comparison selects the true model
- **Parameter sweeps:** Map how parameters affect accuracy, RT, and set-size effects

---

## Stage 4: Model Fitting (Scripts 12-14)

### 4.1 MLE Fitting (Primary)

Maximum likelihood estimation with multi-start optimization. Fast, reliable point estimates.

```bash
# Q-learning (M1): alpha_pos, alpha_neg, epsilon
python scripts/04_model_fitting/a_mle/12_fit_mle.py --model qlearning

# WM-RL (M2): alpha_pos, alpha_neg, phi, rho, K, epsilon
python scripts/04_model_fitting/a_mle/12_fit_mle.py --model wmrl

# WM-RL with perseveration (M3): alpha_pos, alpha_neg, phi, rho, K, kappa, epsilon
python scripts/04_model_fitting/a_mle/12_fit_mle.py --model wmrl_m3

# WM-RL + RL forgetting (M5): alpha_pos, alpha_neg, phi, rho, K, kappa, phi_rl, epsilon
python scripts/04_model_fitting/a_mle/12_fit_mle.py --model wmrl_m5

# WM-RL + stimulus-specific perseveration (M6a): alpha_pos, alpha_neg, phi, rho, K, kappa_s, epsilon
python scripts/04_model_fitting/a_mle/12_fit_mle.py --model wmrl_m6a

# WM-RL + dual perseveration (M6b): alpha_pos, alpha_neg, phi, rho, K, kappa_total, kappa_share, epsilon
# Current winning model (AIC rank 1, Akaike weight ~1.0 across N=154)
python scripts/04_model_fitting/a_mle/12_fit_mle.py --model wmrl_m6b

# RLWM-LBA joint choice+RT (M4): alpha_pos, alpha_neg, phi, rho, K, kappa, v_scale, A, delta, t0
# NOTE: M4 AIC is NOT comparable to choice-only models (M1-M3, M5, M6a, M6b).
# M4 is the only model requiring GPU (float64 LBA likelihood).
python scripts/04_model_fitting/a_mle/12_fit_mle.py --model wmrl_m4
```

**Speed options:**

```bash
# Parallel CPU (multi-core, ~4-8x speedup)
python scripts/04_model_fitting/a_mle/12_fit_mle.py --model wmrl_m3 --n-jobs 16

# GPU-accelerated (requires rlwm_gpu environment)
python scripts/04_model_fitting/a_mle/12_fit_mle.py --model wmrl_m3 --use-gpu
```

**With practice data:**

```bash
python scripts/04_model_fitting/a_mle/12_fit_mle.py --model qlearning --data output/task_trials_long_all.csv --include-practice
```

**Outputs:** `output/mle/<model>_mle_results.csv` — per-participant parameter estimates, NLL, AIC, BIC.

### 4.2 Bayesian Fitting (Optional)

Hierarchical Bayesian models via JAX/NumPyro (NUTS sampler).

```bash
python scripts/04_model_fitting/b_bayesian/13_fit_bayesian.py --model qlearning
python scripts/04_model_fitting/b_bayesian/13_fit_bayesian.py --model wmrl --chains 4 --warmup 1000 --samples 2000
```

**Outputs:** ArviZ InferenceData (`.nc`), parameter summaries (`.csv`), trace plots.

### 4.3 Model Comparison

```bash
# Compare all fitted MLE models (AIC/BIC)
python scripts/06_fit_analyses/compare_models.py

# Compare specific models
python scripts/06_fit_analyses/compare_models.py --models qlearning wmrl wmrl_m3

# With Bayesian criteria (WAIC/LOO)
python scripts/06_fit_analyses/compare_models.py --use-waic
```

**Outputs:**
- `output/model_comparison/` — comparison tables, winning model per participant
- `figures/model_comparison/` — IC bar plots, model weight plots

### Cluster Execution

```bash
# Parallel CPU (Monash M3 cluster)
sbatch cluster/run_mle_parallel.slurm

# GPU-accelerated
sbatch cluster/run_mle_gpu.slurm

# Single model with custom settings
sbatch --export=MODEL=wmrl_m3,NJOBS=8 cluster/run_mle_parallel.slurm
```

---

## Stage 5: Results Analysis (Scripts 15-16)

Relate fitted model parameters to trauma measures.

```bash
# Parameter group comparisons (trauma groups × fitted parameters)
python scripts/06_fit_analyses/analyze_mle_by_trauma.py --model all

# Continuous regression (parameters ~ LEC-5 + IES-R subscales)
python scripts/06_fit_analyses/regress_parameters_on_scales.py --model all
```

### Stage 5b: Winner Heterogeneity (Script 17)

Analyze per-participant model-selection heterogeneity: what fraction of participants are best fit by each model, and how does this vary by trauma group.

```bash
python scripts/06_fit_analyses/analyze_winner_heterogeneity.py
```

**Inputs:** `output/model_comparison/` (from script 14)
**Outputs:** `output/model_comparison/winner_heterogeneity*.csv`, `figures/model_comparison/winner_heterogeneity_figure.png`

---

### Stage 5c: Bayesian Level-2 Effects (Script 18)

Forest plots of Level-2 regression coefficients (trauma predictors × model parameters) from the hierarchical Bayesian posterior. Runs only after cluster Bayesian fit completes.

```bash
python scripts/06_fit_analyses/bayesian_level2_effects.py
```

**Inputs:** `output/bayesian/{model}_posterior.nc` (generated after cluster Bayesian fit)
**Outputs:** `output/bayesian/figures/m6b_forest_lec5.png` and related forest plots. Gracefully skips if posterior NetCDF files are missing.

**Note:** Outputs marked as placeholder in `docs/04_results/README.md` until the cluster Bayesian fit completes.

---

### Parameter Interpretation

| Parameter | Range | Clinical Hypothesis |
|-----------|-------|-------------------|
| `alpha_pos` | 0-1 | Positive learning rate; may be reduced in PTSD (anhedonia) |
| `alpha_neg` | 0-1 | Negative learning rate; may be altered (hypervigilance vs. avoidance) |
| `epsilon` | 0-1 | Random responding / noise |
| `phi` | 0-1 | WM weight; reliance on working memory vs. RL |
| `rho` | 0-1 | WM decay; forgetting rate |
| `K` | 1-7 | WM capacity; number of items maintained |
| `stick` | 0-1 | Perseveration; tendency to repeat previous action |

### Statistical Considerations

- With N=48, focus on **effect sizes** alongside p-values
- Show full distributions (violin plots) with individual data points
- Report confidence intervals prominently
- For 3 groups × multiple parameters, consider Bonferroni correction or report as exploratory

---

## Information Criteria Reference

When comparing models, lower scores indicate better fit (penalized for complexity).

| Criterion | Formula | Best For |
|-----------|---------|----------|
| **AIC** | `2k - 2·log(L)` | Moderate complexity penalty |
| **BIC** | `k·log(n) - 2·log(L)` | Stronger penalty; preferred for large N |
| **WAIC** | Fully Bayesian | Hierarchical models; uses full posterior |
| **LOO** | Leave-one-out CV | Gold standard; out-of-sample prediction |

**Interpreting differences (delta between models):**

| Delta | Evidence |
|-------|----------|
| < 2 | Weak (models similar) |
| 2-6 | Positive |
| 6-10 | Strong |
| > 10 | Very strong |

**Priority:** LOO > WAIC > BIC > AIC. If all agree, strong conclusion. If BIC disagrees with WAIC/LOO, trust WAIC/LOO for hierarchical models.

---

## Configuration

All task parameters and model defaults are centralized in `config.py`:

```bash
python config.py  # Print current configuration
```

Key settings: set sizes `[2, 3, 5, 6]`, reversal criterion 12-18 correct, fixed beta=50 during learning, epsilon noise for random responding.

---

## Testing

```bash
# Fitting module tests
python -m pytest scripts/fitting/tests/ -v

# Likelihood self-tests
python scripts/fitting/jax_likelihoods.py
```

---

## Troubleshooting

**Data issues:**
- Ensure `output/task_trials_long.csv` exists before fitting (run scripts 01-03)
- Stimuli and actions must be 0-indexed
- Practice blocks are blocks 1-2; main task is blocks 3-23

**Fitting issues:**
- High divergences in Bayesian fitting → increase `target_accept_prob` to 0.95
- MLE stuck at boundary → check parameter bounds in `scripts/fitting/mle_utils.py`
- GPU not detected → ensure `rlwm_gpu` conda environment is active

**Model comparison:**
- "No fitted models found" → run `12_fit_mle.py` for each model first
- Conflicting IC rankings → report all four, prioritize LOO

---

## Key References

- **Task/Environment:** `docs/03_methods_reference/TASK_AND_ENVIRONMENT.md`
- **Model Math:** `docs/03_methods_reference/MODEL_REFERENCE.md`
- **Exclusions:** `docs/01_project_protocol/PARTICIPANT_EXCLUSIONS.md`
- **Senta et al. (2025):** Dual process impairments in RL and WM systems
