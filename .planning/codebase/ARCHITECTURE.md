# Architecture

**Analysis Date:** 2026-01-28

## Pattern Overview

**Overall:** Layered, data-driven pipeline with three distinct phases: data processing, model fitting, and analysis.

**Key Characteristics:**
- Three-tier architecture: raw data → processed artifacts → fitted models/analysis
- Centralized configuration management via `config.py` controlling all task parameters and model hyperparameters
- Agent-based simulation with environment abstraction via Gymnasium
- Bayesian hierarchical inference using JAX + NumPyro for parameter estimation
- Experimental replication: exact trial sequences from jsPsych implementation loaded from sequence files

## Layers

**Configuration & Task Definition (`config.py`):**
- Purpose: Single source of truth for all task parameters, model hyperparameters, file paths, and exclusion criteria
- Location: `C:\Users\aman0087\Documents\Github\rlwm_trauma_analysis\config.py`
- Contains: TaskParams (3 actions, set sizes 2/3/5/6, reversal rules), ModelParams (α+, α-, β=50 fixed, ε), PyMCParams (MCMC settings), DataParams (file paths, exclusion thresholds)
- Depends on: numpy for random seeding
- Used by: All scripts in pipeline, models, environment, fitting routines

**Environment Layer (`environments/`):**
- Purpose: Gym-compliant environment implementing the RLWM task mechanics
- Location: `C:\Users\aman0087\Documents\Github\rlwm_trauma_analysis\environments\rlwm_env.py`
- Contains: RLWMEnv class (trial-based simulation, stimulus/set_size/block observation space, binary rewards)
- Depends on: config.py, gymnasium
- Used by: Model simulation, parameter recovery tests, behavioral validation
- Key detail: Loads exact trial sequences from jsPsych via `TaskSequenceLoader` (`environments/task_config.py`)

**Agent Models (`models/`):**
- Purpose: Cognitive models implementing decision-making algorithms
- Location:
  - `C:\Users\aman0087\Documents\Github\rlwm_trauma_analysis\models\q_learning.py` (QLearningAgent class)
  - `C:\Users\aman0087\Documents\Github\rlwm_trauma_analysis\models\wm_rl_hybrid.py` (WMRLHybridAgent class)
- Contains: Q-value matrices, softmax policies, asymmetric learning rates, WM decay, adaptive hybrid weighting
- Depends on: config.py, numpy
- Used by: Simulation scripts, likelihood functions, parameter recovery validation
- Following Senta et al. (2025): β fixed at 50 for identifiability, ε noise captures random responding

**Data Processing Pipeline (`scripts/`):**
- Purpose: Extract, parse, clean, and organize raw jsPsych data
- Location:
  - `scripts/01_parse_raw_data.py` — Extract demographics, surveys, task trials from jsPsych JSON
  - `scripts/02_create_collated_csv.py` — Merge into wide-format participant dataset
  - `scripts/03_create_task_trials_csv.py` — Create long-format trial-by-trial dataset
  - `scripts/04_create_summary_csv.py` — Calculate behavioral metrics
  - `scripts/utils/data_cleaning.py` — JSON parsing, survey extraction
  - `scripts/utils/scoring_functions.py` — LEC-5 (→LESS), IES-R subscales, task metrics
- Depends on: config.py, pandas, numpy
- Used by: Analysis scripts, model fitting
- Exclusion logic: config.EXCLUDED_PARTICIPANTS (6 participants removed for insufficient trials or duplicates)

**Likelihood & Fitting Layer (`scripts/fitting/`):**
- Purpose: Compute model likelihoods and perform Bayesian parameter inference
- Location:
  - `scripts/fitting/jax_likelihoods.py` — JAX-compiled likelihood functions (Q-learning, WM-RL)
  - `scripts/fitting/numpyro_models.py` — Hierarchical Bayesian models with NumPyro
  - `scripts/fitting/fit_with_jax.py` — CLI for single model fitting
  - `scripts/fitting/fit_both_models.py` — Model comparison workflow
- Depends on: JAX, NumPyro, pandas, config.py
- Used by: Analysis scripts
- Key detail: Block-aware processing (Q-values reset per block), epsilon noise applied post-softmax

**Analysis & Visualization (`scripts/analysis/`):**
- Purpose: Compute group statistics, model comparisons, trauma-related analysis
- Location: `scripts/analysis/` (31 files: behavioral plots, parameter analysis, regression, model comparison)
- Contains: Model comparison (BIC/AIC/WAIC/LOO), trauma group analysis, stimulus learning by load, parameter regression on trauma scales
- Depends on: JAX, NumPyro, pandas, matplotlib, ArviZ
- Used by: Research reporting

**Validation & Testing (`validation/`):**
- Purpose: Ensure models recover parameters correctly and run without error
- Location: `validation/conftest.py`, `validation/test_model_consistency.py`, `validation/test_parameter_recovery.py`
- Contains: pytest fixtures, model consistency checks, parameter recovery simulations
- Depends on: pytest, config.py, models, environments

## Data Flow

**Data Ingestion & Preparation:**
1. Raw jsPsych JSON files in `data/full_dataset/` → `01_parse_raw_data.py`
2. Extract: demographics, LEC-5 survey, IES-R survey, task trial log
3. Clean: remove excluded participants (config.EXCLUDED_PARTICIPANTS), filter invalid trials
4. Output: `output/parsed_demographics.csv`, `output/parsed_survey1.csv`, `output/parsed_survey2.csv`, `output/parsed_task_trials.csv`

**Dataset Assembly:**
5. `02_create_collated_csv.py` merges parsed files → `output/collated_participant_data.csv` (wide format, one row per participant)
6. `03_create_task_trials_csv.py` formats trial data → `output/task_trials_long.csv` (long format, one row per trial)
7. `04_create_summary_csv.py` computes metrics → `output/summary_participant_metrics.csv`

**Model Fitting:**
8. Long-format trials + participant parameters → `fit_with_jax.py` (JAX likelihood computation)
9. JAX likelihood passed to NumPyro MCMC (NUTS sampler, hierarchical priors)
10. Output: `output/mle/*.csv` (fitted posteriors), `output/model_performance/` (model comparison)

**Behavioral Analysis:**
11. Task performance plots: accuracy by block, set size, load condition
12. Trauma group comparison: learning curves, parameter distributions
13. Regression: model parameters → trauma scales (IES-R, LESS)
14. Figures: `figures/behavioral_analysis/`, `figures/mle_trauma_analysis/`

**State Management:**
- No persistent state between runs; all inputs explicitly specified in CSV files
- Config.py controls reproducibility via RANDOM_SEED = 42
- Fitted posteriors stored as NetCDF (ArviZ standard) for reproducibility

## Key Abstractions

**Trial Sequence (`environments/task_config.py` / TaskSequenceLoader):**
- Purpose: Encapsulate exact jsPsych trial structure
- Examples: `sequence0.csv` through `sequence19.csv` (20 randomizations × 23 blocks each)
- Pattern: Loads stimulus IDs, correct responses, set sizes, block assignments from CSV; caches in memory

**Agent Interface (QLearningAgent, WMRLHybridAgent):**
- Purpose: Unified decision-making interface with learn() and act() methods
- Pattern: Both agents maintain Q-matrices and update via observation (stimulus, action, reward)
- Key method: `get_action_probabilities(stimulus)` returns softmax policy with epsilon noise

**Likelihood Function (JAX):**
- Purpose: Map trial sequence + parameters → log-likelihood, JIT-compiled for speed
- Pattern:
  - `q_learning_step(carry, inputs)` advances one trial via lax.scan
  - Accumulates log-probability of observed actions given learned Q-values
  - Block boundaries reset Q-tables
- Used by: NumPyro's MCMC sampler via automatic differentiation

**Hierarchical Model (NumPyro):**
- Purpose: Estimate population and individual parameters with uncertainty
- Pattern:
  - Population level: μ, σ for each parameter
  - Individual level: non-centered parameterization for efficiency
  - Likelihood: vectorized across participants
- Priors follow Senta et al. (2025) recommendations

## Entry Points

**Data Pipeline Entry:**
- Location: `run_data_pipeline.py`
- Triggers: User runs `python run_data_pipeline.py [--no-sync]`
- Responsibilities: Orchestrates all parsing/cleaning steps in sequence with error handling

**Model Fitting Entry:**
- Location: `scripts/fitting/fit_with_jax.py`
- Triggers: User runs `python scripts/fitting/fit_with_jax.py --model [qlearning|wmrl] --data output/task_trials_long.csv`
- Responsibilities: Loads data, prepares for NumPyro, runs MCMC, saves posteriors

**Analysis Entry:**
- Location: `scripts/analysis/run_statistical_analyses.py`, `scripts/analysis/run_model_comparison.py`
- Triggers: User runs individual analysis scripts
- Responsibilities: Compute group statistics, compare models, generate figures

## Error Handling

**Strategy:** Explicit error checking with early termination; no silent failures

**Patterns:**
- Data pipeline: Checks for required input files before processing; exits with clear error message if missing (e.g., `02_create_collated_csv.py` checks for `parsed_*.csv` files)
- Participant exclusion: Hard-coded list in config.py; participants filtered in parsing step
- Exclusion criteria: MIN_TRIALS (50), MIN_ACCURACY (0.3 for data quality check)
- Model fitting: NumPyro diagnostics (R-hat < 1.05, effective sample size warnings) included in output

## Cross-Cutting Concerns

**Logging:** Console-based via print() statements; scripts output progress with [OK] tags and step counters

**Validation:**
- Task parameter validation in config.py (e.g., SET_SIZES, ACTIONS defined as module constants)
- Model parameter bounds enforced by ModelParams (ALPHA_MIN=0, ALPHA_MAX=1)
- Trial-level: rt_min=200ms, rt_max=2000ms filters for invalid response timing

**Authentication:** Not applicable (local analysis pipeline)

**Reproducibility:**
- RANDOM_SEED = 42 set in config.py
- JAX uses deterministic operations when seed fixed
- NumPyro MCMC provides chain diagnostics (R-hat, ESS) for convergence assessment
- All parameters stored as floats in output CSVs for reproducibility

---

*Architecture analysis: 2026-01-28*
