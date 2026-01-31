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

### Fitting Pipeline

```
scripts/fitting/
├── jax_likelihoods.py    # Core likelihood functions (JAX)
├── numpyro_models.py     # Hierarchical Bayesian models
├── mle_utils.py          # MLE utilities (transforms, info criteria)
├── fit_mle.py            # CLI for MLE fitting (jaxopt.LBFGS)
├── fit_bayesian.py       # CLI for Bayesian fitting (NumPyro NUTS)
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

### Run MLE Fitting (Fast, Point Estimates)

```bash
# Q-learning (M1)
python scripts/fitting/fit_mle.py --model qlearning --data output/task_trials_long.csv

# WM-RL (M2)
python scripts/fitting/fit_mle.py --model wmrl --data output/task_trials_long.csv

# WM-RL with perseveration (M3)
python scripts/fitting/fit_mle.py --model wmrl_m3 --data output/task_trials_long.csv

# Parallel fitting (multi-core, ~4-8x speedup)
python scripts/fitting/fit_mle.py --model wmrl_m3 --data output/task_trials_long.csv --n-jobs 16

# GPU-accelerated (requires rlwm_gpu environment)
python scripts/fitting/fit_mle.py --model wmrl_m3 --data output/task_trials_long.csv --use-gpu
```

### Cluster Execution (Monash M3)

```bash
# Sequential (original, ~30 min per model)
sbatch cluster/run_mle.slurm

# Parallel CPU (16 cores, ~3 min per model)
sbatch cluster/run_mle_parallel.slurm

# GPU-accelerated (requires rlwm_gpu env)
sbatch cluster/run_mle_gpu.slurm

# Single model with custom settings
sbatch --export=MODEL=wmrl_m3,NJOBS=8 cluster/run_mle_parallel.slurm
```

### Run Bayesian Fitting (Hierarchical, Posterior Distributions)

```bash
# Q-learning
python scripts/fitting/fit_bayesian.py --model qlearning --data output/task_trials_long.csv

# WM-RL
python scripts/fitting/fit_bayesian.py --model wmrl --data output/task_trials_long.csv

# With custom MCMC settings
python scripts/fitting/fit_bayesian.py --model wmrl --data data.csv --chains 4 --warmup 1000 --samples 2000
```

### Run Tests

```bash
# Run fitting module tests
python -m pytest scripts/fitting/tests/ -v
```

### Test Likelihoods

```bash
python scripts/fitting/jax_likelihoods.py
```

### View Config

```bash
python config.py
```
