# Project Structure

This repository follows the [Cookiecutter Data Science v2](https://cookiecutter-data-science.drivendata.org/)
conventions for top-level layout, adapted for computational psychiatry with
hierarchical Bayesian model fitting (SLURM cluster + Quarto manuscript). The
aim is a layout that is immediately familiar to researchers coming from
CCDS-templated projects while accommodating the project-specific needs of
(a) JAX/NumPyro model fitting on a GPU cluster, (b) a tiered test suite
with scientific invariants, and (c) a single-source Quarto manuscript.

## Layout

```
rlwm_trauma_analysis/
├── CITATION.cff               # Citation metadata (CFF v1.2.0, Zenodo-compatible)
├── CLAUDE.md                  # AI-assistant project guidelines
├── README.md                  # Project overview + quickstart
├── config.py                  # Single-source Path constants + model hyperparameters
├── pyproject.toml             # Python package metadata + tool configs (ruff/mypy/pytest)
├── pytest.ini                 # Test discovery config (testpaths = tests)
├── environment.yml            # Conda environment (CPU)
├── environment_gpu.yml        # Conda environment (GPU)
│
├── data/                      # CCDS data tiers
│   ├── raw/                   # Immutable jsPsych CSVs (gitignored — sensitive)
│   ├── interim/               # Parsed + collated intermediate products (gitignored)
│   ├── processed/             # Canonical analysis-ready CSVs (tracked)
│   └── external/              # Third-party reference data (if any)
│
├── models/                    # CCDS models tier — fitted artifacts
│   ├── bayesian/              # Hierarchical posteriors (.nc gitignored, metadata tracked)
│   │   ├── 21_baseline/       # Phase 21 choice-only baseline
│   │   ├── 21_l2/             # Phase 21 level-2 regression
│   │   ├── 21_prior_predictive/
│   │   ├── 21_recovery/       # Bayesian parameter recovery
│   │   └── level2/            # Phase 16 subscale L2
│   ├── mle/                   # Maximum-likelihood individual + group fits (CSV, tracked)
│   ├── ppc/                   # Posterior predictive checks
│   ├── recovery/              # Model + parameter recovery artifacts
│   └── parameter_exploration/ # Parameter sweeps + prior-predictive sandboxes
│
├── reports/                   # CCDS reports tier — presentation-layer outputs
│   ├── figures/               # All PNG/PDF figures (manuscript + diagnostics)
│   └── tables/                # All report CSVs (descriptives, model comparison, regressions, etc.)
│
├── scripts/                   # Pipeline scripts (Scheme D from Phase 29)
│   ├── 01_data_preprocessing/
│   ├── 02_behav_analyses/
│   ├── 03_model_prefitting/
│   ├── 04_model_fitting/      # a_mle/, b_bayesian/, c_level2/
│   ├── 05_post_fitting_checks/
│   ├── 06_fit_analyses/
│   ├── fitting/               # Shared helpers for fit pipeline
│   ├── utils/                 # ppc, plotting, stats, scoring, data_cleaning
│   └── legacy/                # Archived superseded scripts
│
├── src/rlwm/                  # Installable package (src-layout per pyOpenSci)
│   ├── fitting/               # Authoritative JAX likelihoods + NumPyro models
│   ├── envs/                  # Gym environment
│   └── models/                # NumPy Gym-stateful agent classes
│
├── cluster/                   # SLURM + shell scripts for M3 cluster
│   ├── submit_all.sh          # Canonical master orchestrator (supports --dry-run)
│   └── 21_submit_pipeline.sh  # v4.0 closure-contract entry (kept as shim)
│
├── tests/                     # Consolidated test tree (Phase 31)
│   ├── unit/                  # Fast (< 1s) — import/smoke/logic
│   ├── integration/           # Medium (1-60s) — MLE smoke + structure guards
│   └── scientific/            # Slow (> 60s) — parameter recovery, v4 closure
│
├── logs/                      # Single gitignored log location (dev + SLURM)
├── docs/                      # Project documentation (this folder)
├── manuscript/                # Quarto manuscript source (paper.qmd -> paper.pdf)
└── .planning/                 # GSD planning audit trail (local-only, push-blocked)
```

## Key conventions

- **Path source of truth:** `config.py` — never hardcode paths in scripts;
  import constants (`DATA_RAW_DIR`, `PROCESSED_DIR`, `MODELS_BAYESIAN_DIR`,
  `MODELS_MLE_DIR`, `REPORTS_FIGURES_DIR`, `REPORTS_TABLES_DIR`, `LOGS_DIR`,
  etc.). Legacy aliases `OUTPUT_DIR` / `FIGURES_DIR` / `OUTPUT_VERSION_DIR`
  were removed in Phase 31 Plan 05 — `ImportError` at import time is the
  intended migration signal.
- **Test tiers:** unit (< 1s, isolated) -> integration (1-60s, cross-module)
  -> scientific (> 60s, invariants). Run `pytest -m "not slow and not
  scientific"` for CI-equivalent fast tier. Scientific tier includes the
  v4.0 closure guard (8 invariants, deterministic).
- **Data immutability:** `data/raw/` is NEVER edited in place (CCDS
  convention). All transformations produce new files in `data/interim/` or
  `data/processed/`. `data/raw/` is gitignored because participant CSVs are
  sensitive; `data/processed/` is tracked because it is reproducible from
  raw + scripts but expensive to regenerate.
- **Model/report separation:** `models/` holds fitted posteriors / MLE
  estimates (pipeline output); `reports/` holds presentation artifacts
  consumed by the manuscript (figures + tables). This matches CCDS v2
  `{models, reports}` twin top-level tiers.
- **Scheme D scripts:** stage folders `01_...` through `06_...` have numeric
  prefixes (paper IMRaD order); intra-stage scripts use per-stage reset
  numbering; `04_model_fitting/{a,b,c}_*/` subfolders use canonical
  descriptive names (no numbers) because model fanout is via CLI
  `--model <name>` flag, NEVER via per-model script files.
- **Log consolidation:** a single top-level `logs/` is the authoritative
  location for both local-dev stdout/stderr and SLURM `--output=` /
  `--error=` directives. All 18 active SLURMs pin
  `#SBATCH --output=logs/<jobname>_%j.out`. `cluster/logs/` does not exist.
- **Structure invariants are tested.** `tests/integration/test_v5_phase29_structure.py`
  asserts both Phase 29 (Scheme D scripts layout) and Phase 31 (CCDS
  top-level layout) invariants as pytest parametrized cases. Any phase
  that regresses these flips the tests red immediately.

## References

- [Cookiecutter Data Science v2](https://cookiecutter-data-science.drivendata.org/) — the canonical layout inspiration
- [pyOpenSci Python Package Guide](https://www.pyopensci.org/python-package-guide/) — src-layout + testing conventions
- [Scientific Python Development Guide](https://learn.scientific-python.org/development/) — pyproject.toml + ruff + mypy defaults
- [The Turing Way reproducible-project-template](https://github.com/the-turing-way/reproducible-project-template) — docs + manuscript layout
- [Citation File Format v1.2.0](https://citation-file-format.github.io/1.2.0/schema-guide.md) — schema for CITATION.cff

## History

- **Phase 29** (2026-04-22): Canonical Scheme D `scripts/` layout — numbered
  01..06 stage folders, parallel-alternative subfolders for model fitting,
  utils/ consolidation, dead-folder archival.
- **Phase 31** (2026-04-24): Top-level CCDS alignment — `data/{raw,interim,
  processed,external}`, `models/{bayesian,mle,ppc,recovery}`,
  `reports/{figures,tables}`, `tests/{unit,integration,scientific}`,
  unified `logs/`, legacy `output/`/`figures/`/`validation/`/`cluster/logs/`
  removed, config.py single Path source of truth, CITATION.cff added.
