# Codebase Structure

**Analysis Date:** 2026-01-28

## Directory Layout

```
rlwm_trauma_analysis/
├── config.py                        # Central configuration (task params, model defaults, paths)
├── plotting_config.py               # Matplotlib styling
├── run_data_pipeline.py             # CLI: orchestrate data parsing → cleaning → summary
├── CLAUDE.md                        # Project guidelines for AI assistance
├── README.md                        # User documentation
├── environment.yml                  # Conda environment specification
├── requirements.txt                 # Python dependencies
├── pytest.ini                       # Test discovery and markers
│
├── data/                            # Raw input data
│   └── full_dataset/                # jsPsych JSON files from experiments
│
├── environments/                    # Task environment & configuration
│   ├── __init__.py
│   ├── rlwm_env.py                  # RLWMEnv class (Gym-compliant)
│   └── task_config.py               # TaskSequenceLoader (loads sequence CSV files)
│
├── models/                          # Agent models (Q-learning, WM-RL)
│   ├── __init__.py
│   ├── q_learning.py                # QLearningAgent class
│   └── wm_rl_hybrid.py              # WMRLHybridAgent class
│
├── scripts/                         # Pipeline and analysis scripts
│   ├── 01_parse_raw_data.py         # Parse jsPsych JSON → extract trials/surveys/demographics
│   ├── 02_create_collated_csv.py    # Merge parsed data → wide-format dataset
│   ├── 03_create_task_trials_csv.py # Format trials → long-format task data
│   ├── 04_create_summary_csv.py     # Calculate behavioral metrics
│   │
│   ├── fitting/                     # Model fitting & likelihood computation
│   │   ├── __init__.py
│   │   ├── jax_likelihoods.py       # JAX-compiled likelihood functions
│   │   ├── numpyro_models.py        # Hierarchical Bayesian models
│   │   ├── fit_with_jax.py          # CLI: fit single model
│   │   ├── fit_both_models.py       # CLI: fit & compare both models
│   │   ├── mle_utils.py             # MLE helper functions
│   │   ├── pymc_models.py           # PyMC alternative (legacy)
│   │   └── test_*.py                # Quick tests
│   │
│   ├── analysis/                    # Behavioral & statistical analysis
│   │   ├── analyze_learning_by_trauma_group.py
│   │   ├── analyze_mle_by_trauma.py
│   │   ├── analyze_parameters_by_trauma_group.py
│   │   ├── behavioral_plots.py
│   │   ├── model_comparison.py
│   │   ├── plot_*.py                # Visualization scripts
│   │   ├── regress_parameters_on_scales.py
│   │   ├── run_model_comparison.py  # CLI: model comparison
│   │   ├── run_statistical_analyses.py
│   │   └── simulate_model_performance*.py
│   │
│   ├── utils/                       # Utility functions
│   │   ├── data_cleaning.py         # JSON parsing, survey extraction
│   │   ├── scoring_functions.py     # LEC-5→LESS, IES-R→subscales, task metrics
│   │   ├── statistical_tests.py     # ANOVA, regression helper
│   │   ├── sync_experiment_data.py  # Sync from external source
│   │   └── remap_mle_ids.py         # Participant ID remapping
│   │
│   ├── simulations/                 # Model simulations
│   └── legacy/                      # Deprecated scripts
│
├── validation/                      # Test suite
│   ├── conftest.py                  # pytest fixtures
│   ├── test_model_consistency.py    # Model behavior validation
│   ├── test_parameter_recovery.py   # Parameter recovery simulations
│   └── __pycache__/
│
├── output/                          # Generated data artifacts
│   ├── parsed_demographics.csv      # Extracted demographics (step 1)
│   ├── parsed_survey1.csv           # Extracted LEC-5 (step 1)
│   ├── parsed_survey2.csv           # Extracted IES-R (step 1)
│   ├── parsed_task_trials.csv       # Extracted trial log (step 1)
│   ├── collated_participant_data.csv    # Wide-format dataset (step 2)
│   ├── task_trials_long.csv             # Long-format trials (step 3)
│   ├── summary_participant_metrics.csv  # Behavioral metrics (step 4)
│   │
│   ├── descriptives/                    # Summary tables
│   │   ├── demographic_table.csv
│   │   ├── performance_by_load.csv
│   │   └── ...
│   │
│   ├── mle/                             # Fitted model posteriors
│   │   ├── qlearning_*.csv              # Q-learning parameters (posterior means)
│   │   ├── wmrl_*.csv                   # WM-RL parameters
│   │   └── *.nc                         # NetCDF traces (ArviZ format)
│   │
│   ├── model_performance/               # Model comparison results
│   │   ├── model_comparison_*.csv
│   │   └── information_criteria.csv
│   │
│   ├── regressions/                     # Parameter regression results
│   │   └── wmrl_*/                      # Results for different WM cutoffs
│   │
│   ├── statistical_analyses/            # ANOVA, correlation results
│   │   └── *.csv
│   │
│   └── v1/                              # Versioned outputs (if VERSION='v1' in config.py)
│
├── figures/                         # Generated visualizations
│   ├── behavioral_analysis/         # Task performance plots
│   ├── behavioral_summary/          # Overview figures
│   ├── feedback_learning/           # Learning curve comparisons
│   ├── mle_trauma_analysis/         # Parameter distribution by trauma group
│   ├── model_performance/           # Model comparison plots
│   ├── parameter_exploration/       # Parameter sensitivity
│   ├── publication/                 # Publication-ready figures
│   ├── trauma_groups/               # Trauma group comparisons
│   └── v1/                          # Versioned (if VERSION='v1')
│
├── docs/                            # Project documentation
│   ├── TASK_AND_ENVIRONMENT.md      # Task mechanics, API
│   ├── MODEL_REFERENCE.md           # Model math, fitting, paper reference
│   ├── 00_current_todos/            # Current TODO items
│   ├── 01_project_protocol/         # Experimental protocol
│   ├── 02_pipeline_guide/           # Step-by-step pipeline guide
│   ├── 03_methods_reference/        # Methods & references
│   ├── 04_scientific_reports/       # Results & analysis reports
│   └── legacy/                      # Deprecated documentation
│
└── .planning/                       # GSD planning documents
    └── codebase/                    # Codebase analysis (ARCHITECTURE.md, STRUCTURE.md, etc.)
```

## Directory Purposes

**`config.py` (Root)**
- Purpose: Single source of truth for all parameters
- Contains: TaskParams, ModelParams, PyMCParams, DataParams, AnalysisParams classes
- Key functions: `get_set_size_load_condition()`, `get_phase_type()`, `sample_reversal_point()`
- Used by: Every script in the pipeline

**`data/`**
- Purpose: Raw input data (jsPsych JSON files)
- Contains: jsPsych participant data from experiments
- Key path: `data/full_dataset/` (where raw JSON files are stored)
- Generated/Committed: Committed (experiment data)

**`environments/`**
- Purpose: Task environment definition and sequence management
- Key files:
  - `rlwm_env.py`: RLWMEnv class (Gym environment with Dict observation space)
  - `task_config.py`: TaskSequenceLoader (loads exact trial sequences from CSV files)
- Imported by: Simulation scripts, validation tests

**`models/`**
- Purpose: Cognitive agent implementations
- Key files:
  - `q_learning.py`: QLearningAgent with asymmetric learning rates
  - `wm_rl_hybrid.py`: WMRLHybridAgent with distributed WM + RL hybrid
- Key methods: `get_action_probabilities(stimulus)`, `update(stimulus, action, reward)`, `reset()`
- Imported by: Fitting scripts, simulations, validation

**`scripts/`**
- Purpose: Runnable pipeline and analysis scripts
- Key files:
  - `01_parse_raw_data.py` — Entry point for data parsing (3-8 min execution)
  - `02_create_collated_csv.py` — Wide-format dataset creation
  - `03_create_task_trials_csv.py` — Long-format trial data
  - `04_create_summary_csv.py` — Behavioral metrics computation

**`scripts/fitting/`**
- Purpose: Bayesian model fitting with JAX + NumPyro
- Key files:
  - `jax_likelihoods.py` — Likelihood computation (JAX-compiled)
  - `numpyro_models.py` — Hierarchical Bayesian models
  - `fit_with_jax.py` — CLI interface for fitting
- Pattern: Load trial data → compute likelihood → MCMC sampling → save posteriors (NetCDF)

**`scripts/analysis/`**
- Purpose: Statistical analysis and visualization
- Contains: 31 analysis scripts covering learning curves, trauma group comparisons, regressions, model comparison
- Key entry points:
  - `run_statistical_analyses.py` — ANOVA, correlation tests
  - `run_model_comparison.py` — Model comparison via WAIC/LOO
  - `analyze_mle_by_trauma.py` — Parameter analysis by trauma group

**`scripts/utils/`**
- Purpose: Shared utility functions
- Key files:
  - `data_cleaning.py` — JSON parsing, survey extraction (parse_survey1_response, extract_ies_scores)
  - `scoring_functions.py` — Behavioral metrics (score_less, score_ies_r, calculate_all_task_metrics)
  - `statistical_tests.py` — Statistical helpers
  - `sync_experiment_data.py` — External data synchronization

**`validation/`**
- Purpose: Pytest-based test suite
- Key files:
  - `conftest.py` — Fixtures for trial data, agent params, participant data
  - `test_model_consistency.py` — Model behavior checks
  - `test_parameter_recovery.py` — Parameter recovery from synthetic data
- Run via: `pytest validation/` or `pytest validation/ -m slow` (for parameter recovery)

**`output/`**
- Purpose: Generated data artifacts (intermediate and final)
- Generated: Yes (created by pipeline)
- Committed: No (except manual results for reporting)
- Structure: Nested by analysis type (descriptives/, mle/, model_performance/, regressions/, statistical_analyses/)

**`figures/`**
- Purpose: Generated visualizations
- Generated: Yes (created by analysis scripts)
- Committed: No (generated during analysis)
- Structure: Organized by figure type (behavioral_analysis/, mle_trauma_analysis/, publication/)

**`docs/`**
- Purpose: Project documentation (task mechanics, model reference, protocol)
- Key files:
  - `TASK_AND_ENVIRONMENT.md` — Task parameters, environment API
  - `MODEL_REFERENCE.md` — Model equations, Senta et al. (2025) reference
- Committed: Yes
- Convention: One document per major topic (no duplication)

## Key File Locations

**Entry Points:**
- `run_data_pipeline.py` — Main data processing orchestration
- `scripts/fitting/fit_with_jax.py` — Model fitting CLI
- `scripts/analysis/run_model_comparison.py` — Model comparison CLI

**Configuration:**
- `config.py` — Central configuration
- `plotting_config.py` — Matplotlib styling
- `pytest.ini` — Test configuration

**Core Logic:**
- `models/q_learning.py` — Q-learning agent
- `models/wm_rl_hybrid.py` — WM-RL hybrid agent
- `environments/rlwm_env.py` — Gym environment
- `scripts/fitting/jax_likelihoods.py` — JAX likelihood functions
- `scripts/fitting/numpyro_models.py` — Bayesian models

**Testing:**
- `validation/conftest.py` — Test fixtures
- `validation/test_model_consistency.py` — Consistency tests
- `validation/test_parameter_recovery.py` — Parameter recovery tests

## Naming Conventions

**Files:**
- Snake_case for all files: `01_parse_raw_data.py`, `jax_likelihoods.py`
- Numbered scripts for pipeline: `01_*.py`, `02_*.py`, `03_*.py`, `04_*.py`
- Descriptive names: `analyze_learning_by_trauma_group.py` (not `analyze.py`)

**Directories:**
- Lowercase plural for collections: `scripts/`, `models/`, `figures/`
- Lowercase with underscore for multi-word: `scripts/fitting/`, `scripts/analysis/`, `scripts/utils/`
- Thematic grouping: `figures/behavioral_analysis/`, `output/statistical_analyses/`

**Classes:**
- PascalCase: `RLWMEnv`, `QLearningAgent`, `WMRLHybridAgent`, `TaskSequenceLoader`

**Functions:**
- Snake_case: `softmax_policy()`, `apply_epsilon_noise()`, `q_learning_step()`, `get_set_size_load_condition()`

**Parameters in code:**
- Greek letters: Use ASCII in code (`phi`, `rho`, `alpha_pos`), Greek in comments/docs (φ, ρ, α₊)

## Where to Add New Code

**New Feature (e.g., new analysis):**
- Primary code: `scripts/analysis/analyze_[feature].py`
- Tests: `validation/test_[feature].py`
- Configuration: Add to `config.py` if new parameters needed
- Output path: Define in `config.AnalysisParams` or use `OUTPUT_DIR / '[feature]/'`

**New Component/Module:**
- Shared utility: `scripts/utils/[module].py`
- Model variant: `models/[model_name].py` (add to models/__init__.py)
- New analysis layer: `scripts/analysis/[layer].py`

**Utilities:**
- Shared helpers: `scripts/utils/[helper].py`
- Scoring/metrics: Add to `scripts/utils/scoring_functions.py`
- Data cleaning: Add to `scripts/utils/data_cleaning.py`

**Tests:**
- Unit tests: `validation/test_[module].py`
- Fixtures (shared): Add to `validation/conftest.py`
- Mark slow tests with `@pytest.mark.slow` for conditional execution

## Special Directories

**`scripts/legacy/`**
- Purpose: Deprecated scripts (preserved for reference)
- Generated: No
- Committed: Yes (for historical reference)

**`scripts/simulations/`**
- Purpose: Model simulation and recovery scripts
- Generated: Output files (not source)
- Committed: Source code yes, outputs no

**`docs/legacy/`**
- Purpose: Superseded documentation
- Committed: Yes (for reference)
- Convention: Move docs here when consolidating into primary docs

**`.planning/codebase/`**
- Purpose: GSD codebase analysis documents
- Generated: Yes (by GSD mapper)
- Committed: Yes (for orchestrator use)

---

*Structure analysis: 2026-01-28*
