# Technology Stack

**Analysis Date:** 2026-01-28

## Languages

**Primary:**
- Python 3.10.18 - Core language for all data processing, modeling, and analysis

## Runtime

**Environment:**
- Python 3.10.18 (via Miniforge/Conda)

**Package Manager:**
- Conda (primary environment management)
- Pip (secondary dependency installation)
- Lockfile: `environment.yml` (full spec), `requirements.txt` (core), `requirements-dev.txt` (development)

## Frameworks

**Core Data & Scientific:**
- NumPy 2.2.6 - Array operations and numerical computing
- Pandas 2.3.1 - Data manipulation, CSV I/O, DataFrames
- SciPy 1.15.2 - Statistical functions and optimization algorithms

**Reinforcement Learning & Probabilistic Programming:**
- JAX 0.4+ - JIT compilation, automatic differentiation (used in `scripts/fitting/jax_likelihoods.py`)
- NumPyro 0.13+ - Probabilistic programming, MCMC inference via numpyro models
- PyMC 5.x - Alternative Bayesian inference framework (optional, used in PyMC models)
- ArviZ 0.13+ - Posterior analysis and model diagnostics (works with both NumPyro and PyMC)
- Gymnasium (gym) - Trial-based reinforcement learning environment (see `environments/rlwm_env.py`)

**Machine Learning & Statistics:**
- Scikit-learn 1.7.1 - Statistical learning, model evaluation
- Statsmodels 0.14.5 - Statistical modeling and hypothesis testing
- Joblib 1.5.1 - Parallel processing (dependency of scikit-learn)

**Visualization:**
- Matplotlib 3.10.3 - Low-level plotting, figure generation
- Seaborn 0.13.2 - Statistical visualization (built on matplotlib)
- Pillow 11.3.0 - Image handling

**Development & Testing:**
- Pytest 7.0+ - Test framework
- Pytest-cov 4.0+ - Coverage reporting
- Pytest-xdist 3.0+ - Parallel test execution
- Pytest-timeout 2.1+ - Test timeout management
- Black 23.0+ - Code formatting
- Flake8 6.0+ - Linting
- Isort 5.12+ - Import sorting
- MyPy 1.0+ - Static type checking

**Jupyter/Interactive:**
- JupyterLab 4.4.5 - Interactive notebook environment
- IPython 8.37.0 - Interactive Python shell
- IPyWidgets 8.0+ - Interactive widgets for notebooks

**Documentation & Utilities:**
- Sphinx 7.0+ - Optional documentation generation
- TQDM 4.65+ - Progress bars for simulations
- Line-profiler 4.0+ - Line-by-line performance profiling (optional)
- Memory-profiler 0.61+ - Memory profiling (optional)

## Key Dependencies

**Critical (Direct imports in core code):**
- `numpy` - All mathematical computations in agents (`models/q_learning.py`, `models/wm_rl_hybrid.py`)
- `pandas` - Data I/O and manipulation in all pipeline scripts
- `jax` / `jax.numpy` - Likelihood computation in `scripts/fitting/jax_likelihoods.py`
- `numpyro` - Bayesian inference in `scripts/fitting/numpyro_models.py` and `fit_with_jax.py`
- `scipy.optimize.minimize` - Maximum likelihood estimation in `scripts/fitting/fit_mle.py`
- `gymnasium` - Environment API in `environments/rlwm_env.py`

**Important (Analysis and fitting):**
- `arviz` - Posterior processing and diagnostics from MCMC samples
- `matplotlib` / `seaborn` - Figure generation in analysis scripts
- `scikit-learn` - Data splitting, metrics in analysis
- `statsmodels` - Statistical tests and models
- `scipy` - Statistical distributions and functions

## Configuration

**Environment:**
- Managed via `environment.yml` (reproducible Conda environment, 217 pinned dependencies)
- Alternative minimal setup via `requirements.txt` (core packages only)
- Development tools via `requirements-dev.txt` (testing, linting, profiling)

**Build & Development:**
- `config.py` - Central configuration file at project root (tasks, models, parameters, paths)
- `plotting_config.py` - Centralized plotting defaults (fonts, colors, DPI settings)
- `pytest.ini` - Test discovery and configuration
- `.gitignore` - Version control exclusions

**No package setup file:**
- No `setup.py`, `pyproject.toml`, or `setup.cfg` present
- Project is not packaged as installable module (scripts run directly from repository)

## Platform Requirements

**Development:**
- Python 3.10+ required
- Conda or Miniconda recommended for environment reproducibility
- Platform: Windows/Linux/macOS (conda environment.yml includes platform-specific packages)

**Production:**
- Python 3.10+ runtime
- NumPy, Pandas, JAX, NumPyro for inference pipelines
- Matplotlib/Seaborn for figure generation
- Optional: CUDA-capable GPU for JAX acceleration (XLA backend)

## External Storage

**Local filesystem only:**
- Input: `data/` directory (raw jsPsych experiment output)
- Intermediate: `output/` directory (parsed CSVs)
- Analysis results: `output/v1/` versioned subdirectory
- Figures: `figures/v1/` versioned subdirectory
- No external databases, cloud storage, or APIs

## Performance & Optimization

**JAX compilation:**
- `jax.jit()` decorators for likelihood computation (see `jax_likelihoods.py`)
- Automatic differentiation for gradient-based optimization
- XLA compilation for GPU/TPU support (if available)

**Parallel processing:**
- Pytest can run tests in parallel via `-n` flag (pytest-xdist)
- NumPyro MCMC uses `numpyro.set_host_device_count(n_devices)` for multi-chain parallelization
- Joblib used internally by scikit-learn for parallelization

---

*Stack analysis: 2026-01-28*
