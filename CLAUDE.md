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
| M1: Q-Learning | α₊, α₋, ε |
| M2: WM-RL | α₊, α₋, φ, ρ, K, ε |
| M3: WM-RL+kappa | α₊, α₋, φ, ρ, K, κ, ε |
| M5: WM-RL+phi_rl | α₊, α₋, φ, ρ, K, κ, φ_rl, ε |
| M6a: WM-RL+kappa_s | α₊, α₋, φ, ρ, K, κ_s, ε |
| M6b: WM-RL+dual | α₊, α₋, φ, ρ, K, κ_total, κ_share, ε |
| M4: RLWM-LBA | α₊, α₋, φ, ρ, K, κ, v_scale, A, δ, t₀ |

> M4 is the only joint choice+RT model. Its AIC is not comparable to choice-only models.
> M5 is the current winning model among choice-only models (dAIC=435.6 over M3).

---

## Code Organization

### Numbered Pipeline Scripts

The analysis pipeline uses numbered scripts grouped by stage:

Pipeline layout follows **Scheme D** (plan 29-04b): six numbered stage
folders (01-06) with intra-stage numbers that reset per stage; parallel-
alternative subfolders (04/a, 04/b, 04/c) are unnumbered. Library /
engine code uses underscore-private convention (`_engine.py`) to avoid
collision with canonical CLI entry names.

```
scripts/
├── 01_data_preprocessing/      # Strict execution order
│   ├── 01_parse_raw_data.py
│   ├── 02_create_collated_csv.py
│   ├── 03_create_task_trials_csv.py
│   └── 04_create_summary_csv.py
│
├── 02_behav_analyses/          # Strict execution order
│   ├── 01_summarize_behavioral_data.py
│   ├── 02_visualize_task_performance.py
│   ├── 03_analyze_trauma_groups.py
│   └── 04_run_statistical_analyses.py
│
├── 03_model_prefitting/        # Pre-fit simulations + gates
│   ├── 01_generate_synthetic_data.py
│   ├── 02_run_parameter_sweep.py
│   ├── 03_run_model_recovery.py
│   ├── 04_run_prior_predictive.py     # Baribault gate
│   └── 05_run_bayesian_recovery.py
│
├── 04_model_fitting/           # Parallel alternatives — no stage numbers
│   ├── a_mle/                  # Dispatch via --model CLI flag
│   │   ├── fit_mle.py                 # Thin CLI entry
│   │   └── _engine.py                 # Library (3,157 lines, private)
│   ├── b_bayesian/             # Hierarchical MCMC
│   │   ├── fit_bayesian.py            # Ad-hoc CLI entry
│   │   ├── fit_baseline.py            # Phase 21 pipeline entry
│   │   └── _engine.py                 # Library (1,173 lines, private)
│   └── c_level2/               # Winner refit with L2 design
│       └── fit_with_l2.py
│
├── 05_post_fitting_checks/     # Paper-read order: baseline -> scales -> PPC
│   ├── 01_baseline_audit.py
│   ├── 02_scale_audit.py
│   └── 03_run_posterior_ppc.py
│
├── 06_fit_analyses/            # Paper-read order: compare -> ... -> tables
│   ├── 01_compare_models.py
│   ├── 02_compute_loo_stacking.py
│   ├── 03_model_averaging.py
│   ├── 04_analyze_mle_by_trauma.py
│   ├── 05_regress_parameters_on_scales.py
│   ├── 06_analyze_winner_heterogeneity.py
│   ├── 07_bayesian_level2_effects.py
│   └── 08_manuscript_tables.py
│
├── utils/                      # Consolidated helpers (plan 29-03)
│   ├── ppc.py                  # Canonical PPC simulator (single source)
│   ├── plotting.py
│   ├── stats.py
│   ├── scoring.py
│   └── data_cleaning.py
│
├── fitting/                    # Library helpers + tests
│   ├── mle_utils.py
│   ├── bms.py
│   ├── model_recovery.py
│   ├── bayesian_diagnostics.py
│   ├── bayesian_summary_writer.py
│   ├── level2_design.py
│   ├── lba_likelihood.py
│   ├── warmup_jit.py
│   ├── compare_mle_models.py
│   └── tests/
│       ├── test_v4_closure.py
│       ├── test_load_side_validation.py
│       ├── test_loo_stacking.py
│       ├── test_bayesian_recovery.py
│       ├── test_mle_quick.py
│       ├── test_gpu_m4.py
│       └── conftest.py
│
└── legacy/                     # Archived dead folders (plan 29-04)
    ├── analysis/
    ├── results/
    ├── simulations/
    ├── statistical_analyses/
    └── visualization/
```

**Scheme D naming rules (pin for future contributors):**

- Stage folders (`0N_<descriptive>`, N = 1..6) keep numeric prefixes —
  they encode paper IMRaD order.
- Intra-stage numbers reset per stage (`0M_<descriptive>.py`, M = 1..N_in_stage)
  wherever execution order is load-bearing. Parallel-alternative subfolders
  (`04/a_mle/`, `04/b_bayesian/`, `04/c_level2/`) do NOT use numbers — they
  use canonical descriptive names.
- Library / engine code uses underscore-private convention
  (`_engine.py`, `_helpers.py`) to avoid collision with CLI entry names.
- Model fanout (M1/M2/M3/M5/M6a/M6b/M4) is via CLI `--model <name>` flag,
  NEVER via per-model script files.
- Callers that need to reuse engine code import via
  `importlib.util.spec_from_file_location` by absolute path, because Python
  dotted names cannot start with a digit (`scripts.04_model_fitting.*` is
  illegal).

### Library (src/rlwm/)

```
src/rlwm/
├── fitting/
│   ├── jax_likelihoods.py    # Core JAX likelihood functions (authoritative)
│   ├── numpyro_models.py     # Hierarchical Bayesian models (authoritative)
│   └── numpyro_helpers.py    # NumPyro sampling utilities
├── envs/
│   └── rlwm_env.py           # Gym environment
└── models/
    ├── q_learning.py         # Q-learning agent class
    └── wm_rl_hybrid.py       # WM-RL hybrid agent class
```

Install with `pip install -e .` (required to import `rlwm`).

### Key Files

- `config.py` - Central configuration (task params, model defaults)
- `src/rlwm/envs/rlwm_env.py` - Gym environment
- `src/rlwm/models/q_learning.py` - Agent class for Q-learning
- `src/rlwm/models/wm_rl_hybrid.py` - Agent class for WM-RL

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
python scripts/04_model_fitting/a_mle/fit_mle.py --model qlearning --data output/task_trials_long_all.csv --include-practice

# Bayesian fitting with practice blocks
python scripts/04_model_fitting/b_bayesian/fit_bayesian.py --model qlearning --data output/task_trials_long_all.csv --include-practice
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
python scripts/01_data_preprocessing/01_parse_raw_data.py
python scripts/01_data_preprocessing/02_create_collated_csv.py
python scripts/01_data_preprocessing/03_create_task_trials_csv.py
python scripts/01_data_preprocessing/04_create_summary_csv.py

# Behavioral Analysis (01-04)
python scripts/02_behav_analyses/01_summarize_behavioral_data.py
python scripts/02_behav_analyses/02_visualize_task_performance.py
python scripts/02_behav_analyses/03_analyze_trauma_groups.py
python scripts/02_behav_analyses/04_run_statistical_analyses.py

# Model Fitting (04_model_fitting/)
python scripts/04_model_fitting/a_mle/fit_mle.py --model qlearning
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m3
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m5
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m6a
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m6b
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m4
python scripts/06_fit_analyses/01_compare_models.py

# Post-fit Results Analysis (06_fit_analyses/)
python scripts/06_fit_analyses/04_analyze_mle_by_trauma.py --model all
python scripts/06_fit_analyses/05_regress_parameters_on_scales.py --model all
python scripts/06_fit_analyses/06_analyze_winner_heterogeneity.py
```

### Run MLE Fitting (Fast, Point Estimates)

```bash
# Q-learning (M1)
python scripts/04_model_fitting/a_mle/fit_mle.py --model qlearning --data output/task_trials_long.csv

# WM-RL (M2)
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl --data output/task_trials_long.csv

# WM-RL with perseveration (M3)
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m3 --data output/task_trials_long.csv

# M5: WM-RL + RL Forgetting
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m5 --data output/task_trials_long.csv

# M6a: WM-RL + Stimulus-Specific Perseveration
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m6a --data output/task_trials_long.csv

# M6b: WM-RL + Dual Perseveration
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m6b --data output/task_trials_long.csv

# M4: RLWM-LBA Joint Choice+RT (requires float64)
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m4 --data output/task_trials_long.csv

# Parallel fitting (multi-core, ~4-8x speedup)
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m5 --data output/task_trials_long.csv --n-jobs 16

# GPU-accelerated (requires rlwm_gpu environment)
python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m5 --data output/task_trials_long.csv --use-gpu
```

### Run Bayesian Pipeline (Full Hierarchical Fit)

```bash
# Full 9-step afterok chain (recommended; runs on cluster)
bash cluster/21_submit_pipeline.sh

# Individual steps (new canonical locations)
python scripts/03_model_prefitting/04_run_prior_predictive.py
python scripts/03_model_prefitting/05_run_bayesian_recovery.py
python scripts/04_model_fitting/b_bayesian/fit_baseline.py
python scripts/05_post_fitting_checks/01_baseline_audit.py
python scripts/06_fit_analyses/02_compute_loo_stacking.py
python scripts/04_model_fitting/c_level2/fit_with_l2.py
python scripts/05_post_fitting_checks/02_scale_audit.py
python scripts/06_fit_analyses/03_model_averaging.py
python scripts/06_fit_analyses/08_manuscript_tables.py
```

### Cluster Execution (Monash M3)

```bash
# MLE: CPU parallel (16 cores, ~3 min per model)
sbatch cluster/12_mle.slurm

# MLE: GPU-accelerated (requires rlwm_gpu env)
sbatch cluster/12_mle_gpu.slurm

# MLE: all models as independent GPU jobs (recommended)
bash cluster/12_submit_all_gpu.sh

# MLE: single model with custom settings
sbatch --export=MODEL=wmrl_m3,NJOBS=8 cluster/12_mle.slurm

# Bayesian: consolidated template (choice-only models)
sbatch --export=ALL,MODEL=wmrl_m3 cluster/13_bayesian_choice_only.slurm
sbatch --export=ALL,MODEL=wmrl_m5 cluster/13_bayesian_choice_only.slurm
sbatch --time=36:00:00 --export=ALL,MODEL=wmrl_m6b cluster/13_bayesian_choice_only.slurm

# Bayesian: GPU template (M4 LBA only)
sbatch cluster/13_bayesian_gpu.slurm
```

### Run Bayesian Fitting (Hierarchical, Posterior Distributions)

```bash
# Q-learning
python scripts/04_model_fitting/b_bayesian/fit_bayesian.py --model qlearning --data output/task_trials_long.csv

# WM-RL
python scripts/04_model_fitting/b_bayesian/fit_bayesian.py --model wmrl --data output/task_trials_long.csv

# With custom MCMC settings
python scripts/04_model_fitting/b_bayesian/fit_bayesian.py --model wmrl --data data.csv --chains 4 --warmup 1000 --samples 2000
```

### Model Comparison

```bash
# Compare choice-only models (M1-M3, M5, M6a, M6b) by AIC/BIC
python scripts/06_fit_analyses/01_compare_models.py

# Include M4 separate track
python scripts/06_fit_analyses/01_compare_models.py --m4

# Compare specific models
python scripts/06_fit_analyses/01_compare_models.py --models qlearning wmrl wmrl_m3

# With Bayesian comparison (WAIC/LOO)
python scripts/06_fit_analyses/01_compare_models.py --use-waic
```

### Cross-Model Recovery

```bash
# Validate all choice-only models are distinguishable by AIC
python scripts/03_model_prefitting/03_run_model_recovery.py --mode cross-model --model all --n-subjects 50 --n-datasets 10 --n-jobs 8
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
