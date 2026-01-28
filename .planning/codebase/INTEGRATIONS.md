# External Integrations

**Analysis Date:** 2026-01-28

## APIs & External Services

**None.**

This project contains no external API integrations. All data is processed locally from input files.

## Data Storage

**Databases:**
- None - No database system (MySQL, PostgreSQL, MongoDB, etc.)

**File Storage:**
- Local filesystem only
- Input: `data/` - Raw jsPsych experiment output files (CSV format)
- Intermediate: `output/` - Parsed and transformed CSV files
- Analysis outputs: `output/v1/` - Versioned results with model fits
- Figures: `figures/v1/` - Versioned PNG outputs

**Configuration paths (from `config.py`):**
- `PROJECT_ROOT` - Repository root
- `DATA_DIR` - `data/` subdirectory for raw input
- `OUTPUT_DIR` - `output/` subdirectory for all outputs
- `FIGURES_DIR` - `figures/` subdirectory for generated plots
- `OUTPUT_VERSION_DIR` - `output/v1/` for versioned results (NetCDF model outputs)

**Caching:**
- None explicitly configured
- Pytest cache in `.pytest_cache/` (excluded from git)
- Python cache in `__pycache__/` (excluded from git)

## Authentication & Identity

**Auth Provider:**
- Not applicable - No external authentication required

**Implementation:**
- All code runs locally with file system access only
- No user login, API keys, or credentials needed

## Monitoring & Observability

**Error Tracking:**
- None - No external error tracking service

**Logs:**
- Standard output/stderr from CLI scripts
- No centralized logging system
- Pytest output to console with `-v` verbose flag

**Example logging in code:**
- Print statements in fitting scripts (`scripts/fitting/fit_mle.py`, `fit_with_jax.py`)
- TQDM progress bars for long-running simulations

## CI/CD & Deployment

**Hosting:**
- Not applicable - Research code repository (no deployment)

**CI Pipeline:**
- None configured
- Pytest available locally for test execution

**Run commands for testing:**
```bash
# Run all tests
pytest

# Run tests in parallel
pytest -n auto

# Run with coverage
pytest --cov=models --cov=environments

# Run specific test markers
pytest -m "not slow"
```

## Environment Configuration

**Required environment variables:**
- None explicitly required
- All configuration via Python files (`config.py`, `plotting_config.py`)

**Secrets location:**
- Not applicable - No API keys or credentials

## Data Sources

**Input Data:**
- jsPsych experiment output files (CSV format, location: `data/` directory)
- Participant behavioral data: stimulus-response pairs, reaction times, feedback
- Survey/assessment data: trauma-related scales, demographics

**Data Processing Pipeline:**
- Scripts in `scripts/` directory convert raw jsPsych output → analysis-ready CSVs
- No external data fetching or API calls

**Intermediate files (output/directory):**
- `parsed_demographics.csv` - Demographic information
- `parsed_survey1.csv` - Survey 1 responses
- `parsed_survey2.csv` - Survey 2 responses
- `parsed_task_trials.csv` - Task trial-by-trial data
- `collated_participant_data.csv` - Wide-format consolidated data
- `task_trials_long.csv` - Long-format trial data for analysis
- `summary_participant_metrics.csv` - Derived behavioral metrics

**Model fitting outputs (output/v1/ directory):**
- `task_trials_long.csv` - Input data for fitting
- Fitted posteriors (NetCDF format from ArviZ)
- Model comparison results (CSV format)

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Dependencies on External Systems

**JAX/NumPyro backend:**
- No external service dependency
- JAX compiles to XLA locally
- Optional: GPU drivers if CUDA acceleration desired (not required)

**File paths used in code:**
- `environments/rlwm_env.py` - Line 25: `from config import TaskParams`
- `models/q_learning.py` - `from config import TaskParams, ModelParams`
- `models/wm_rl_hybrid.py` - `from config import TaskParams, ModelParams`
- All scripts import from local `config.py` (no external configuration)

## Data Flow

**Typical workflow:**
1. Raw jsPsych data placed in `data/` directory
2. User runs data pipeline: `python run_data_pipeline.py`
3. Scripts in `scripts/` generate intermediate and final outputs in `output/`
4. Analysis scripts read from `output/` and generate figures in `figures/`
5. Fitting scripts (in `scripts/fitting/`) read task data and produce fitted posteriors
6. All outputs stored locally - no external upload or synchronization

**No cloud sync, no CI/CD pipeline, no external data sources.**

---

*Integration audit: 2026-01-28*
