# CLAUDE.md - Project Guidelines for AI Assistance

This file contains project-specific instructions for Claude Code when working on the RLWM Trauma Analysis project.

---

## Documentation Standards

### Keep docs/ Succinct and Standalone

1. **Merge, don't multiply**: Each major topic should have ONE authoritative document. Do not create v2/v3 versions or parallel docs on the same topic.

2. **Standalone documents**: Each doc should be self-contained. A reader should understand the topic fully without needing to read other docs.

3. **Current doc structure**:
   - `docs/TASK_AND_ENVIRONMENT.md` - Task structure, environment API, parameters
   - `docs/MODEL_REFERENCE.md` - Model mathematics, fitting, paper comparisons
   - `docs/legacy/` - Deprecated docs (for reference only)

4. **When updating docs**:
   - Update the existing document in place
   - If consolidating multiple docs, merge content into ONE file
   - Move superseded docs to `docs/legacy/`
   - Update cross-references in other files

5. **Avoid clutter**:
   - No duplicate information across docs
   - No "see other doc for details" - include the details
   - Remove outdated sections rather than marking them deprecated

---

## Model Implementation (Senta et al., 2025)

### Key Design Decisions

1. **Fixed β = 50**: Inverse temperature is NOT a free parameter during learning. This is for parameter identifiability.

2. **Epsilon noise**: Random responding is captured by ε parameter: `p_noisy = ε/nA + (1-ε)*p`

3. **No testing phase**: This task has learning only, no separate test phase.

4. **WM update**: Immediate overwrite `WM(s,a) ← r` (not prediction error update)

5. **WM baseline**: `1/nA = 0.333` (uniform probability)

### Parameter Summary

| Model | Free Parameters |
|-------|-----------------|
| Q-Learning | α₊, α₋, ε |
| WM-RL | α₊, α₋, φ, ρ, K, ε |

---

## Code Organization

### Numbered Pipeline Scripts

The analysis pipeline uses numbered scripts (01-16) for sequential processing:

```
scripts/
├── 01-04: Data Processing
│   ├── 01_parse_raw_data.py         # Parse raw jsPsych JSON files
│   ├── 02_create_collated_csv.py    # Collate all participants
│   ├── 03_create_task_trials_csv.py # Create trial-level data
│   └── 04_create_summary_csv.py     # Create participant summaries
│
├── 05-08: Behavioral Analysis
│   ├── 05_summarize_behavioral_data.py    # Behavioral summary stats
│   ├── 06_visualize_task_performance.py   # Task performance plots
│   ├── 07_analyze_trauma_groups.py        # Trauma grouping + validation
│   └── 08_run_statistical_analyses.py     # ANOVAs + descriptive tables
│
├── 09-11: Simulations & Model Validation
│   ├── 09_generate_synthetic_data.py      # Posterior predictive checks
│   ├── 10_run_parameter_sweep.py          # Systematic parameter exploration
│   └── 11_run_model_recovery.py           # Parameter recovery validation
│
├── 12-14: Model Fitting
│   ├── 12_fit_mle.py                      # MLE fitting (main CLI)
│   ├── 13_fit_bayesian.py                 # Bayesian fitting (optional)
│   └── 14_compare_models.py               # Model comparison + winning model
│
├── 15-16: Results Analysis
│   ├── 15_analyze_mle_by_trauma.py        # Parameter-trauma relationships
│   └── 16_regress_parameters_on_scales.py # Continuous scale regressions
│
├── analysis/    # Library module (imported by numbered scripts)
├── fitting/     # Library module (JAX likelihoods, MLE/Bayesian fitting)
├── simulations/ # Library module (synthetic data generation)
└── utils/       # Library module (data loading, plotting)
```

### Fitting Library Module

```
scripts/fitting/
├── jax_likelihoods.py    # Core likelihood functions (JAX)
├── numpyro_models.py     # Hierarchical Bayesian models
├── mle_utils.py          # MLE utilities (transforms, info criteria)
├── fit_mle.py            # MLE fitting implementation
├── fit_bayesian.py       # Bayesian fitting implementation
└── tests/                # Test suite
    ├── conftest.py       # Shared fixtures (synthetic data)
    ├── test_mle_quick.py # MLE parameter recovery tests
    └── test_wmrl_model.py # Bayesian model compilation tests
```

### Key Files

- `config.py` - Central configuration (task params, model defaults)
- `environments/rlwm_env.py` - Gym environment
- `models/q_learning.py` - Agent class for Q-learning
- `models/wm_rl_hybrid.py` - Agent class for WM-RL

### Data Pipeline

**Block Structure (from jsPsych):**
- Block 1: `practice_static` - Static practice (no reversals)
- Block 2: `practice_dynamic` - Dynamic practice (with reversals)
- Blocks 3-23: `main_task` - Main experimental task (21 blocks max)

**Output Files:**
| File | Description |
|------|-------------|
| `output/task_trials_long_all.csv` | All blocks with `is_practice` flag and `phase_type` |
| `output/task_trials_long.csv` | Main task only (default for fitting) |
| `output/task_trials_long_all_participants.csv` | Legacy filename (main task only) |

**Fitting with Practice Data:**
```bash
# MLE fitting with practice blocks
python scripts/fitting/fit_mle.py --model qlearning --data output/task_trials_long_all.csv --include-practice

# Bayesian fitting with practice blocks
python scripts/fitting/fit_bayesian.py --model qlearning --data output/task_trials_long_all.csv --include-practice
```

---

## Conventions

### Naming

- Use snake_case for files and functions
- Use descriptive names (e.g., `alpha_pos` not `a1`)
- Greek letters in comments, ASCII in code (e.g., `phi` not `φ`)

### Git

- Commit logical units of work
- Reference Senta et al. (2025) when implementing model features

---

## Quick Reference

### Run Full Pipeline

```bash
# Data Processing (01-04)
python scripts/01_parse_raw_data.py
python scripts/02_create_collated_csv.py
python scripts/03_create_task_trials_csv.py
python scripts/04_create_summary_csv.py

# Behavioral Analysis (05-08)
python scripts/05_summarize_behavioral_data.py
python scripts/06_visualize_task_performance.py
python scripts/07_analyze_trauma_groups.py
python scripts/08_run_statistical_analyses.py

# Model Fitting (12-14)
python scripts/12_fit_mle.py --model qlearning
python scripts/12_fit_mle.py --model wmrl
python scripts/12_fit_mle.py --model wmrl_m3
python scripts/14_compare_models.py

# Results Analysis (15-16)
python scripts/15_analyze_mle_by_trauma.py --model all
python scripts/16_regress_parameters_on_scales.py --model all
```

### Run MLE Fitting (Fast, Point Estimates)

```bash
# Q-learning (M1)
python scripts/12_fit_mle.py --model qlearning --data output/task_trials_long.csv

# WM-RL (M2)
python scripts/12_fit_mle.py --model wmrl --data output/task_trials_long.csv

# WM-RL with perseveration (M3)
python scripts/12_fit_mle.py --model wmrl_m3 --data output/task_trials_long.csv

# Parallel fitting (multi-core, ~4-8x speedup)
python scripts/12_fit_mle.py --model wmrl_m3 --data output/task_trials_long.csv --n-jobs 16

# GPU-accelerated (requires rlwm_gpu environment)
python scripts/12_fit_mle.py --model wmrl_m3 --data output/task_trials_long.csv --use-gpu
```

### Cluster Execution (Monash M3)

```bash
# CPU parallel (16 cores, ~3 min per model)
sbatch cluster/12_mle.slurm

# GPU-accelerated (requires rlwm_gpu env)
sbatch cluster/12_mle_gpu.slurm

# All 3 models as independent GPU jobs (recommended)
bash cluster/12_submit_all_gpu.sh

# Single model with custom settings
sbatch --export=MODEL=wmrl_m3,NJOBS=8 cluster/12_mle.slurm
```

### Run Bayesian Fitting (Hierarchical, Posterior Distributions)

```bash
# Q-learning
python scripts/13_fit_bayesian.py --model qlearning --data output/task_trials_long.csv

# WM-RL
python scripts/13_fit_bayesian.py --model wmrl --data output/task_trials_long.csv

# With custom MCMC settings
python scripts/13_fit_bayesian.py --model wmrl --data data.csv --chains 4 --warmup 1000 --samples 2000
```

### Model Comparison

```bash
# Compare all fitted models (AIC/BIC)
python scripts/14_compare_models.py

# Compare specific models
python scripts/14_compare_models.py --models qlearning wmrl wmrl_m3

# With Bayesian comparison (WAIC/LOO)
python scripts/14_compare_models.py --use-waic
```

### Run Tests

```bash
# Run fitting module tests
python -m pytest scripts/fitting/tests/ -v

# Run all tests (including examples)
python -m pytest tests/ -v
```

### Test Likelihoods

```bash
python scripts/fitting/jax_likelihoods.py
```

### View Config

```bash
python config.py
```
