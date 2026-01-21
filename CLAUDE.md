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
├── fit_with_jax.py       # CLI for single model fitting
└── fit_both_models.py    # CLI for model comparison
```

### Key Files

- `config.py` - Central configuration (task params, model defaults)
- `environments/rlwm_env.py` - Gym environment
- `models/q_learning.py` - Agent class for Q-learning
- `models/wm_rl_hybrid.py` - Agent class for WM-RL

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

### Run Fitting

```bash
# Q-learning
python scripts/fitting/fit_with_jax.py --model qlearning --data output/task_trials_long.csv

# WM-RL
python scripts/fitting/fit_with_jax.py --model wmrl --data output/task_trials_long.csv
```

### Test Likelihoods

```bash
python scripts/fitting/jax_likelihoods.py
```

### View Config

```bash
python config.py
```
