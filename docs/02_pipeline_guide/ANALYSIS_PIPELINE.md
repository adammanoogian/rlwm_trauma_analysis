# RLWM Trauma Analysis Pipeline

Complete workflow from raw jsPsych data to computational model fitting,
hierarchical Bayesian inference, and trauma-parameter regression.

## Overview

This pipeline processes raw jsPsych behavioral data from a Reinforcement
Learning Working Memory (RLWM) task combined with trauma assessment surveys
(LEC-5, IES-R). It follows the **Scheme D** layout (six numbered stage
folders) under the CCDS v2 directory structure.

| Stage | Folder | Description |
|-------|--------|-------------|
| 1 | `scripts/01_data_preprocessing/` | Parse, collate, and summarize raw data |
| 2 | `scripts/02_behav_analyses/` | Descriptive stats, visualizations, trauma grouping |
| 3 | `scripts/03_model_prefitting/` | Synthetic data, parameter recovery, prior predictive |
| 4 | `scripts/04_model_fitting/` | MLE and Bayesian fitting (parallel alternatives) |
| 5 | `scripts/05_post_fitting_checks/` | Baseline audit, scale audit, posterior PPC |
| 6 | `scripts/06_fit_analyses/` | Model comparison, trauma regressions, manuscript tables |

**Sample:** N=154 participants (v4.0 canonical cohort after exclusions).
See `config.get_analysis_cohort()` for the three-gate inclusion criteria.

---

## Environment Setup

```bash
# Install the rlwm package in editable mode (required for imports)
pip install -e .

# View current configuration
python config.py
```

For GPU-accelerated fitting on the Monash M3 cluster, see
[Cluster Execution](#cluster-execution).

---

## Stage 1: Data Processing

Parse raw jsPsych JSON exports into analysis-ready CSVs.

```bash
python scripts/01_data_preprocessing/01_parse_raw_data.py
python scripts/01_data_preprocessing/02_create_collated_csv.py
python scripts/01_data_preprocessing/03_create_task_trials_csv.py
python scripts/01_data_preprocessing/04_create_summary_csv.py
```

**Outputs (CCDS `data/processed/` tier):**

| File | Description |
|------|-------------|
| `data/processed/task_trials_long.csv` | Main task trials (default for fitting) |
| `data/processed/task_trials_long_all.csv` | All blocks including practice |
| `data/processed/task_trials_long_all_participants.csv` | Legacy filename (main task only) |
| `data/processed/summary_participant_metrics.csv` | Per-participant aggregates |

**Block structure:** Blocks 1-2 are practice (static, dynamic), blocks 3-23
are main task. By default, fitting uses `task_trials_long.csv` (main task only).

---

## Stage 2: Behavioral Analysis

Generate descriptive statistics, visualizations, trauma group classifications,
and statistical tests.

```bash
python scripts/02_behav_analyses/01_summarize_behavioral_data.py
python scripts/02_behav_analyses/02_visualize_task_performance.py
python scripts/02_behav_analyses/03_analyze_trauma_groups.py
python scripts/02_behav_analyses/04_run_statistical_analyses.py
```

### Trauma Groups

Participants are classified into two groups based on LEC-5 endorsement
and IES-R total score:

| Group | Criteria | Interpretation |
|-------|----------|----------------|
| Trauma-No-Ongoing-Impact | LEC-5 exposure, low IES-R | Exposed but resilient |
| Trauma-Ongoing-Impact | LEC-5 exposure, high IES-R | Exposed with symptoms |

**Outputs:** `reports/tables/trauma_groups/group_assignments.csv`,
`reports/figures/trauma_groups/`

---

## Stage 3: Pre-fit Simulations & Validation

Generate synthetic data and validate the fitting pipeline before committing
to expensive cluster runs.

```bash
python scripts/03_model_prefitting/01_generate_synthetic_data.py
python scripts/03_model_prefitting/02_run_parameter_sweep.py
python scripts/03_model_prefitting/03_run_model_recovery.py
python scripts/03_model_prefitting/04_run_prior_predictive.py
python scripts/03_model_prefitting/05_run_bayesian_recovery.py
```

**Use cases:**
- **Parameter recovery:** Fit models to synthetic data with known parameters;
  verify recovery (criterion: r >= 0.80)
- **Model recovery:** Generate data from each model; verify AIC selects the
  true generating model
- **Prior predictive:** Baribault & Collins (2023) gate — check that priors
  produce plausible behavioral patterns before fitting real data

---

## Stage 4: Model Fitting

Stage 4 uses parallel-alternative subfolders (no intra-stage numbers):

```
scripts/04_model_fitting/
├── a_mle/          # MLE point estimates via L-BFGS-B
├── b_bayesian/     # Hierarchical MCMC via NumPyro NUTS
└── c_level2/       # Winner refit with Level-2 trauma covariates
```

### 4a. MLE Fitting

Maximum likelihood estimation with 50 random restarts per participant.

```bash
# All seven models (dispatch via --model flag)
python scripts/04_model_fitting/a_mle/fit_mle.py --model qlearning
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m3
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m5
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m6a
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m6b
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m4  # joint choice+RT

# Speed options
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m5 --n-jobs 16
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m5 --use-gpu
```

**Outputs:** `models/mle/{model}_individual_fits.csv`

### 4b. Bayesian Fitting

Hierarchical Bayesian models via NumPyro NUTS. Non-centered parameterization
(hBayesDM convention). Individual-level parameters use probit-bounded priors.

```bash
# Single model
python scripts/04_model_fitting/b_bayesian/fit_bayesian.py --model wmrl_m3

# Full 9-step afterok pipeline (recommended; runs on cluster)
bash cluster/21_submit_pipeline.sh
```

The full Bayesian pipeline runs nine steps in sequence:
1. Prior predictive checks
2. Bayesian parameter recovery
3. Baseline hierarchical fits (all choice-only models)
4. Convergence audit (R-hat <= 1.05, ESS >= 400, 0 divergences)
5. PSIS-LOO + stacking weights
6. Winner Level-2 refit with trauma covariates
7. Scale-fit audit
8. Model-averaged beta coefficients
9. Manuscript tables

**Outputs:** `models/bayesian/{model}_posterior.nc`, `models/bayesian/level2/`

### 4c. Level-2 Refit

Refit the stacking winner with a Level-2 design matrix of trauma predictors
(LEC-5 total, IES-R total, residualized IES-R subscales).

```bash
python scripts/04_model_fitting/c_level2/fit_with_l2.py
```

---

## Stage 5: Post-Fitting Checks

```bash
python scripts/05_post_fitting_checks/01_baseline_audit.py
python scripts/05_post_fitting_checks/02_scale_audit.py
python scripts/05_post_fitting_checks/03_run_posterior_ppc.py
```

---

## Stage 6: Fit Analyses

Model comparison, trauma associations, and manuscript table generation.

```bash
# Model comparison (AIC/BIC for MLE; LOO/stacking for Bayesian)
python scripts/06_fit_analyses/01_compare_models.py
python scripts/06_fit_analyses/02_compute_loo_stacking.py
python scripts/06_fit_analyses/03_model_averaging.py

# Trauma associations
python scripts/06_fit_analyses/04_analyze_mle_by_trauma.py --model all
python scripts/06_fit_analyses/05_regress_parameters_on_scales.py --model all
python scripts/06_fit_analyses/06_analyze_winner_heterogeneity.py
python scripts/06_fit_analyses/07_bayesian_level2_effects.py

# Manuscript tables
python scripts/06_fit_analyses/08_manuscript_tables.py
```

**Outputs:** `reports/tables/model_comparison/`, `reports/tables/regressions/`,
`reports/figures/model_comparison/`

---

## Cluster Execution

All compute jobs >15 minutes wall-clock should run on the Monash M3 cluster.

```bash
# MLE: all models as independent GPU jobs (recommended)
bash cluster/12_submit_all_gpu.sh

# MLE: single model
sbatch --export=MODEL=wmrl_m3,NJOBS=8 cluster/12_mle.slurm

# Bayesian: consolidated template (choice-only models)
sbatch --export=ALL,MODEL=wmrl_m5 cluster/13_bayesian_choice_only.slurm

# Bayesian: GPU template (M4 LBA only)
sbatch cluster/13_bayesian_gpu.slurm

# Full Bayesian pipeline (9-step afterok chain)
bash cluster/21_submit_pipeline.sh
```

---

## Parameter Quick Reference

| Parameter | Symbol | Range | Description |
|-----------|--------|-------|-------------|
| alpha_pos | alpha+ | [0, 1] | Positive learning rate (reward sensitivity) |
| alpha_neg | alpha- | [0, 1] | Negative learning rate (error sensitivity) |
| phi | phi | [0, 1] | WM decay rate (higher = faster forgetting) |
| rho | rho | [0, 1] | WM reliability (weight of WM in the hybrid policy) |
| K | K | [2, 6] | WM capacity (number of maintained items) |
| kappa | kappa | [0, 1] | Global perseveration (repeat last action) |
| kappa_s | kappa_s | [0, 1] | Stimulus-specific perseveration |
| kappa_total | kappa_total | [0, 1] | Total perseveration budget (M6b) |
| kappa_share | kappa_share | [0, 1] | Global vs. stimulus allocation (M6b) |
| phi_rl | phi_RL | [0, 1] | RL forgetting (Q-value decay, M5 only) |
| epsilon | epsilon | [0, 1] | Random responding / attentional noise |

**Inverse temperature** beta = 50 is fixed for identifiability.

---

## Testing

```bash
# Fast tier (unit + integration, < 2 min)
python -m pytest tests/ -m "not slow and not scientific" -v

# Integration tier
python -m pytest tests/integration/ -v

# Scientific tier (parameter recovery, v4 closure — slow)
python -m pytest tests/scientific/ -v
```

---

## Key References

- **Task/Environment:** `docs/03_methods_reference/TASK_AND_ENVIRONMENT.md`
- **Model Math:** `docs/03_methods_reference/MODEL_REFERENCE.md`
- **Bayesian Architecture:** `docs/04_methods/README.md#hierarchical-bayesian-architecture`
- **Cluster Lessons:** `docs/CLUSTER_GPU_LESSONS.md`
- **Senta et al. (2025):** Dual process impairments in RL and WM systems
