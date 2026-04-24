# RLWM Trauma Analysis

Reinforcement Learning Working Memory (RLWM) pipeline — fitting hierarchical
computational models to behavioral data from 154 participants and relating
parameters to trauma exposure (LEC-5) and PTSD symptoms (IES-R). Based on
Senta et al. (2025); extends Collins & Frank (2012, 2014).

## Setup

```bash
conda env create -f environment.yml
conda activate ds_env
pip install -e .   # required: makes `import rlwm` resolve (src/ layout)
```

## Project Structure

This repository follows [Cookiecutter Data Science v2](https://cookiecutter-data-science.drivendata.org/)
conventions. See [`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md)
for the full layout.

Quick orientation:

- `data/{raw,interim,processed}/` — CCDS data tiers
- `models/{bayesian,mle,ppc,recovery}/` — fitted model artifacts
- `reports/{figures,tables}/` — manuscript figures and tables
- `scripts/0N_.../` — six-stage numbered pipeline (preprocessing → behavioral
  → prefit → fit → post-fit → analysis)
- `src/rlwm/` — installable package (`pip install -e .`)
- `tests/{unit,integration,scientific}/` — tiered test suite
- `cluster/` — SLURM jobs + master orchestrator `submit_all.sh`
- `logs/` — single unified log location (gitignored)

## Pipeline

Scripts are numbered by stage; each depends on the previous.

```bash
# Data processing (01-04)
python scripts/01_data_preprocessing/01_parse_raw_data.py
python scripts/01_data_preprocessing/02_create_collated_csv.py
python scripts/01_data_preprocessing/03_create_task_trials_csv.py
python scripts/01_data_preprocessing/04_create_summary_csv.py

# Behavioral analysis (01-04)
python scripts/02_behav_analyses/01_summarize_behavioral_data.py
python scripts/02_behav_analyses/02_visualize_task_performance.py
python scripts/02_behav_analyses/03_analyze_trauma_groups.py
python scripts/02_behav_analyses/04_run_statistical_analyses.py

# Pre-fit validation (01-05): synthetic data, parameter sweeps, recovery, PPC gates
python scripts/03_model_prefitting/01_generate_synthetic_data.py
python scripts/03_model_prefitting/02_run_parameter_sweep.py
python scripts/03_model_prefitting/03_run_model_recovery.py --mode cross-model --model all
python scripts/03_model_prefitting/04_run_prior_predictive.py --model wmrl_m3
python scripts/03_model_prefitting/05_run_bayesian_recovery.py --mode aggregate --model wmrl_m3

# MLE fitting + frequentist comparison (04_model_fitting/, 06_fit_analyses/)
for m in qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b wmrl_m4; do
  python scripts/04_model_fitting/a_mle/fit_mle.py --model $m --n-jobs 16
done
python scripts/06_fit_analyses/01_compare_models.py

# Parameter-trauma associations (06_fit_analyses/)
python scripts/06_fit_analyses/04_analyze_mle_by_trauma.py --model all
python scripts/06_fit_analyses/05_regress_parameters_on_scales.py --model all
python scripts/06_fit_analyses/06_analyze_winner_heterogeneity.py

# Hierarchical Bayesian pipeline (cluster; 9-step afterok chain)
bash cluster/21_submit_pipeline.sh
```

The Phase 21 orchestrator runs prior-predictive checks, parameter recovery,
baseline + Level-2 MCMC fits, scale audit, LOO stacking / Bayesian model
selection, model averaging, and manuscript tables as `afterok`-chained SLURM
jobs.

## Models

| Model | Free parameters | *k* | Notes |
|-------|-----------------|:---:|-------|
| **M1** Q-Learning | α₊, α₋, ε | 3 | Asymmetric delta rule |
| **M2** WM-RL | α₊, α₋, φ, ρ, *K*, ε | 6 | Collins & Frank hybrid; WM decay φ, reliability ρ, capacity *K* |
| **M3** WM-RL + κ | α₊, α₋, φ, ρ, *K*, κ, ε | 7 | Adds global perseveration kernel (Senta et al., 2025) |
| **M5** WM-RL + φ_RL | α₊, α₋, φ, ρ, *K*, κ, φ_RL, ε | 8 | M3 + RL forgetting toward 1/n_A |
| **M6a** WM-RL + κ_s | α₊, α₋, φ, ρ, *K*, κ_s, ε | 7 | Stimulus-specific perseveration |
| **M6b** WM-RL dual κ | α₊, α₋, φ, ρ, *K*, κ_total, κ_share, ε | 8 | Stick-breaking split between global and stimulus-specific — **winning model** (dAIC = 435.6 vs M3) |
| **M4** RLWM-LBA | α₊, α₋, φ, ρ, *K*, κ, v_scale, *A*, δ, *t₀* | 10 | Joint choice + RT via Linear Ballistic Accumulator; separate track (AIC not comparable) |

Inverse temperature β = 50 is fixed across all models for identifiability.
ε captures uniform random responding: `p_noisy = ε/n_A + (1−ε)·p`.
M1–M6b are compared by AIC/BIC and hierarchically by LOO-PSIS stacking weights
and random-effects BMS. M4 is evaluated on a separate track because its
joint choice-plus-RT likelihood is not commensurable with choice-only AIC.
