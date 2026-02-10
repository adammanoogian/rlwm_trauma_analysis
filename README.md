# RLWM Trauma Analysis

Reinforcement Learning Working Memory (RLWM) task analysis pipeline — fitting computational models (Q-learning, WM-RL) to behavioral data and relating parameters to trauma exposure (LEC-5) and PTSD symptoms (IES-R). Based on Senta et al. (2025).

## Setup

```bash
conda env create -f environment.yml
conda activate ds_env
```

## Pipeline

Run scripts sequentially (01-16). Each stage depends on the previous.

### Data Processing (01-04)

```bash
python scripts/01_parse_raw_data.py          # Parse raw jsPsych JSON
python scripts/02_create_collated_csv.py      # Collate participants (wide)
python scripts/03_create_task_trials_csv.py   # Trial-level data (long)
python scripts/04_create_summary_csv.py       # Summary metrics
```

### Behavioral Analysis (05-08)

```bash
python scripts/05_summarize_behavioral_data.py     # Descriptive stats
python scripts/06_visualize_task_performance.py     # Learning curves, set-size effects
python scripts/07_analyze_trauma_groups.py          # Trauma grouping + clustering
python scripts/08_run_statistical_analyses.py       # ANOVAs, descriptive tables
```

### Simulations & Validation (09-11)

```bash
python scripts/09_generate_synthetic_data.py       # Synthetic data generation
python scripts/10_run_parameter_sweep.py            # Parameter space exploration
python scripts/11_run_model_recovery.py             # Parameter & model recovery
```

### Model Fitting (12-14)

```bash
python scripts/12_fit_mle.py --model qlearning      # Q-learning (M1)
python scripts/12_fit_mle.py --model wmrl            # WM-RL (M2)
python scripts/12_fit_mle.py --model wmrl_m3         # WM-RL + perseveration (M3)
python scripts/14_compare_models.py                  # AIC/BIC model comparison
```

For parallel or GPU-accelerated fitting:

```bash
python scripts/12_fit_mle.py --model wmrl_m3 --n-jobs 16     # Multi-core
python scripts/12_fit_mle.py --model wmrl_m3 --use-gpu       # GPU (rlwm_gpu env)
```

### Results Analysis (15-16)

```bash
python scripts/15_analyze_mle_by_trauma.py --model all        # Parameter-trauma groups
python scripts/16_regress_parameters_on_scales.py --model all  # Continuous regressions
```

## Documentation

| Document | Contents |
|----------|----------|
| [ANALYSIS_PIPELINE.md](docs/02_pipeline_guide/ANALYSIS_PIPELINE.md) | Full pipeline walkthrough (stages 1-5), fitting options, IC reference |
| [MODEL_REFERENCE.md](docs/03_methods_reference/MODEL_REFERENCE.md) | Model math: Q-learning, WM-RL, parameter bounds, priors |
| [TASK_AND_ENVIRONMENT.md](docs/03_methods_reference/TASK_AND_ENVIRONMENT.md) | Task structure, environment API, block/trial layout |
| [PARTICIPANT_EXCLUSIONS.md](docs/01_project_protocol/PARTICIPANT_EXCLUSIONS.md) | Exclusion criteria, final N=48 |

## Data

### Input

Raw jsPsych CSV files go in `data/`. One file per participant session.

### Key Outputs

| File | Description |
|------|-------------|
| `output/task_trials_long.csv` | Main task trials (long format, for fitting) |
| `output/task_trials_long_all.csv` | All trials including practice blocks |
| `output/collated_participant_data.csv` | All participant data (wide format) |
| `output/summary_participant_metrics.csv` | Per-participant summary metrics |
| `output/mle/` | MLE fitting results per model |
| `output/model_comparison/` | Model comparison tables |
| `output/trauma_groups/` | Trauma group assignments |
| `figures/` | All generated visualizations |

## Models

| Model | Parameters | Description |
|-------|-----------|-------------|
| Q-learning (M1) | alpha_pos, alpha_neg, epsilon | Asymmetric RL |
| WM-RL (M2) | alpha_pos, alpha_neg, phi, rho, K, epsilon | RL + working memory |
| WM-RL+P (M3) | + stick | M2 + perseveration |

Fixed beta=50 during learning (for identifiability). Epsilon captures random responding. See [MODEL_REFERENCE.md](docs/03_methods_reference/MODEL_REFERENCE.md) for full math.

## Tests

```bash
python -m pytest scripts/fitting/tests/ -v
```

## Configuration

```bash
python config.py  # Print current settings
```
