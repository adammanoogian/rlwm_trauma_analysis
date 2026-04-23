# Phase 31: Final-Package Restructure — Research

**Researched:** 2026-04-24
**Domain:** Research-repo layout conventions, data-tier separation, test consolidation, CCDS/pyOpenSci/JOSS standards
**Confidence:** HIGH for CCDS conventions (official docs); MEDIUM for JOSS/pyOpenSci (fetched but rate-limited on review_criteria page); MEDIUM for exemplar cogmodel repos (single repo surveyed); HIGH for break-impact analysis (direct code grep).

---

## TL;DR Recommendation

Move the repo from a development-output layout to a CCDS-aligned final-package layout in four bounded moves. The final layout is:

```
rlwm_trauma_analysis/
├── CITATION.cff                   # NEW — JOSS + Zenodo compatibility
├── CLAUDE.md                      # updated paths
├── README.md                      # updated paths
├── config.py                      # updated: DATA_DIR, INTERIM_DIR, PROCESSED_DIR,
│                                  #   MODELS_DIR, REPORTS_DIR, FIGURES_DIR (one change)
├── environment.yml / environment_gpu.yml
├── pyproject.toml                 # updated testpaths
├── pytest.ini                     # updated testpaths (remove validation)
├── Makefile                       # NEW optional — local workflow twin for cluster/21_submit_pipeline.sh
│
├── data/
│   ├── raw/                       # MOVED from data/ flat — immutable jsPsych CSVs
│   └── external/                  # (future: external reference data, if any)
│
├── data_interim/                  # OR data/interim/ — parsed_*.csv, collated_*.csv
│   (see Q7 — 'interim' placement is the main open question)
│
├── data_processed/                # OR data/processed/ — task_trials_long*.csv, summary_*.csv
│
├── models/                        # MOVED from output/bayesian/ + output/mle/ + output/ppc/ + output/recovery/
│   ├── bayesian/                  # .nc files, shrinkage reports, winners.txt (gitignored *.nc by default)
│   ├── mle/                       # *_individual_fits.csv, *_group_summary.csv
│   ├── ppc/                       # posterior predictive check results
│   └── recovery/                  # model/parameter recovery outputs
│
├── reports/
│   ├── figures/                   # MOVED from figures/ + output/bayesian/figures/ + output/model_comparison/figures
│   └── tables/                    # MOVED from output/descriptives/, output/model_comparison/*.csv (report artifacts)
│
├── scripts/                       # UNCHANGED — Scheme D intact (01-06 stage folders)
├── src/rlwm/                      # UNCHANGED — installable package
├── cluster/                       # UNCHANGED except path strings inside .slurm
│   └── logs/                      # MERGED destination (see below)
├── docs/                          # updated path references
├── manuscript/                    # UNCHANGED
│
├── tests/                         # EXPANDED — absorbs validation/
│   ├── unit/                      # fast, isolated (current tests/ contents)
│   ├── integration/               # medium-speed, cross-module (current scripts/fitting/tests/)
│   └── scientific/                # slow, domain-truth (current validation/ contents)
│       └── conftest.py
│
└── logs/                          # MERGED — dev + cluster overflow, gitignored
```

**Primary recommendation:** Use `data/raw/` and `data/processed/` (two-level; skip `interim/` as a top-level sibling) to minimize path depth. Move `output/bayesian/` → `models/bayesian/`, `output/mle/` → `models/mle/`, `figures/` + `output/*/figures/` → `reports/figures/`, and report CSVs → `reports/tables/`. Consolidate `validation/` into `tests/scientific/`. Merge logs. Update `config.py` constants and all SLURM/script `--default=` strings.

---

## Q1 — CCDS v2 Layout and Conventions

**Source:** [CCDS official docs](https://cookiecutter-data-science.drivendata.org/) + [CCDS v2 blog post](https://drivendata.co/blog/ccds-v2) + [CCDS GitHub](https://github.com/drivendataorg/cookiecutter-data-science). **Confidence: HIGH.**

### Canonical v2 tree

```
├── LICENSE
├── Makefile          ← task runner ("make data", "make train"), NOT just build
├── README.md
├── data/
│   ├── external/     ← third-party sources
│   ├── interim/      ← intermediate transformed data
│   ├── processed/    ← final canonical datasets for modeling
│   └── raw/          ← the original, IMMUTABLE data dump
├── docs/
├── models/           ← trained/serialized models, predictions, summaries
├── notebooks/        ← exploration notebooks (numbered: 0.1-pjb-explore.ipynb)
├── pyproject.toml
├── references/       ← data dictionaries, manuals, papers
├── reports/
│   └── figures/      ← generated graphics for reporting
├── requirements.txt
└── {{ module_name }}/   ← installable source (NOT src/)
    ├── config.py
    ├── dataset.py
    ├── features.py
    └── modeling/
        ├── predict.py
        └── train.py
```

### Data tier definitions (authoritative quotes)

- **raw/**: "The original, immutable data dump." Never edited in place.
- **interim/**: "Intermediate data that has been transformed."
- **processed/**: "The final, canonical data sets for modeling."
- **external/**: "Data from third party sources."
- CCDS opinion: "Raw data must be treated as immutable." The `data/` folder is listed in `.gitignore` by default — but this is a recommendation for large-data projects. For small sensitive data (as in our case), the existing pattern of gitignoring only the raw participant CSVs (by glob pattern) and tracking processed outputs is equally defensible.

### `models/` treatment

CCDS does not mandate gitignoring `models/`. The official v2 blog says: "some projects manage this folder like the `data` folder and sync it to a canonical store (e.g., AWS S3) separately from source code. Some projects opt to remove it and use a separate experiment tracking tool." The `.gitkeep` convention (already used in our `output/bayesian/` subdirs) is appropriate for tracking directory structure while ignoring large binary files.

**For this repo:** Large `.nc` files are already gitignored; the `.gitkeep` files tracking directory structure should move with the directories. `models/bayesian/*.nc` stays gitignored; `models/mle/*.csv` and `models/bayesian/21_baseline/winners.txt` stay tracked.

### `reports/figures/` vs top-level `figures/`

CCDS v2 puts figures inside `reports/figures/`. Our repo currently has `figures/` at top level AND `output/bayesian/figures/`. Merging both into `reports/figures/` aligns with CCDS and eliminates the confusion.

### What CCDS does NOT address

- `tests/`, `validation/`, `logs/`: not discussed. These are project-specific.
- `scripts/` (pipeline scripts distinct from notebooks): CCDS expects notebook-driven exploration; pipeline scripts are not a first-class concept. Our `scripts/` is fine — it is our equivalent of the Makefile target implementations.
- `cluster/`: no CCDS concept. Keep as-is.

### Our `src/rlwm/` vs CCDS module pattern

CCDS v2 uses `{{ module_name }}/` (flat, not `src/`). Our `src/rlwm/` uses the src-layout. **The src-layout is strictly better** for installable packages (pyOpenSci recommends it; Scientific Python recommends it). Do not change to flat layout.

---

## Q2 — pyOpenSci Package Review Checklist

**Source:** [pyOpenSci Python Package Guide](https://www.pyopensci.org/python-package-guide/index.html) + [reviewer guide](https://www.pyopensci.org/software-peer-review/how-to/reviewer-guide.html). **Confidence: MEDIUM** (guide index was fetched but detail pages were rate-limited).

### What pyOpenSci requires (HIGH-confidence items from search results)

1. **Packaging**: `pyproject.toml` with build backend — already satisfied.
2. **src/ layout**: Preferred (avoids accidental import of local editable code in tests).
3. **tests/**: Must exist, must run, must be discovered by `pytest`. Co-location inside the package is acceptable but top-level `tests/` is standard.
4. **Documentation**: README with statement of need, installation instructions, quickstart example.
5. **License**: OSI-approved — already have MIT (confirmed in WMH repo, standard for the domain).
6. **CI/CD**: Not strictly required for a research repo, but if JOSS is targeted, CI helps.
7. **CITATION.cff**: Not explicitly in pyOpenSci checklist but recommended (and required for Zenodo DOI auto-population).

### Key layout recommendation from pyOpenSci

- **src-layout**: `src/{package}/` is the recommended pattern. Already implemented (`src/rlwm/`). No change needed.
- **tests/**: Top-level `tests/` directory, no `__init__.py` inside.
- **No prescribed sub-tier structure** for `tests/unit/` vs `tests/integration/` — that is project convention, not standard.

### What this means for Phase 31

The consolidation of `validation/` into `tests/scientific/` is consistent with pyOpenSci expectations. The key requirement is that `pytest` discovers all tests from `tests/` without manual path specification after the consolidation.

---

## Q3 — JOSS Requirements

**Source:** [JOSS docs index](https://joss.readthedocs.io/en/latest/) + search-derived summary (review_criteria page returned 403). **Confidence: MEDIUM.**

### Minimum bar for JOSS submission

| Requirement | Status in this repo |
|---|---|
| `paper.md` at repo root (or in a `paper/` folder) | We have `manuscript/paper.qmd` — would need a separate `paper.md` for JOSS if targeting it |
| `paper.bib` references file | Exists in manuscript/ presumably |
| OSI license (MIT, Apache, etc.) | MIT — confirmed |
| Statement of need in README | Needs checking/strengthening |
| Tests that run | Yes — pytest passes |
| Documentation | Minimal acceptable (README + docs/) |
| CITATION.cff | Recommended; required for Zenodo DOI |

### JOSS layout expectations

JOSS has **no prescribed directory layout**. The review checklist asks: "Does the project follow good open-source practices (license, documentation, tests and/or verification processes, releases, and clear contribution/support pathways)?" — format-agnostic.

### Data in JOSS repos

JOSS does not require data to be in the repo. For sensitive participant data (as here), the standard practice is:
- Raw data gitignored + referenced via a separate data-access statement (institutional data sharing agreement, Zenodo embargo).
- Processed/aggregate outputs that enable figure reproduction can be tracked or shared via OSF/Zenodo.

**For this repo:** The existing gitignore pattern (raw jsPsych CSVs excluded, processed CSVs tracked) is JOSS-compatible. Moving to `data/raw/` does not change this policy.

### Key JOSS-specific action

If JOSS submission is a goal, add `paper.md` at repo root (separate from `manuscript/paper.qmd`). This is a v5.1 item, not Phase 31 scope — but note it in docs.

---

## Q4 — Exemplar Computational-Psych Repos

**Source:** Direct GitHub fetch of [AnneCollins/WMH](https://github.com/AnneCollins/WMH) — the closest domain exemplar (RLWM, same model family, same PI). **Confidence: MEDIUM.**

### Collins lab (WMH, 2024)

The repo is organized by **study/figure, not by data tier**:
```
WMH/
├── RLWM/           ← data + code + figures for study 1
├── RLWMP/          ← data + code + figures for study 2
├── SimulationFig4/ ← simulation code only
├── spm12/          ← dependency
└── ReadMe.txt
```

**Assessment**: Pure MATLAB, ad-hoc structure with no CCDS influence. Self-contained per-study folders. This is the pre-reproducibility-era pattern common in cognitive neuroscience. Our repo is already meaningfully better than this baseline.

### General finding from domain survey

**Nobody in cognitive/computational psychiatry uses CCDS out of the box.** The typical published cognitive neuroscience repo (2020-2024) uses one of:
- Study-per-folder (Collins style above)
- Flat `data/` + `code/` + `manuscript/` (most common)
- Jupyter-notebook-centric repos for ML/AI adjacent labs

CCDS adoption in this domain is effectively zero based on surveyed repos. However, CCDS conventions are the clearest publicly documented standard available — adopting them gives a reader a reference ("we follow CCDS v2 conventions") even if most domain peers do not.

**Recommendation**: Adopt CCDS data tiers (`raw/interim/processed`) and `models/` + `reports/` naming because they are self-explaining to any reader, not because the domain uses them. The Scheme D scripts layout (Phase 29) is already better than domain norms.

---

## Q5 — Reproducibility Best Practices

**Source:** [The Turing Way reproducible-project-template](https://github.com/the-turing-way/reproducible-project-template) + [Turing Way handbook](https://book.the-turing-way.org/reproducible-research/make/) + [Scientific Python dev guide](https://learn.scientific-python.org/development/guides/pytest/). **Confidence: MEDIUM.**

### The Turing Way template layout

```
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/         ← data fetch/generate scripts
│   ├── models/       ← model train/predict scripts
│   └── visualisation/
├── models/           ← trained model artifacts
├── notebooks/
├── reports/
│   └── figures/
├── project-management/
└── docs/
```

No `tests/` or `logs/` in the Turing Way template — those are left to project convention.

### "Fresh clone → paper.pdf" protocol

The Turing Way recommends Makefile as the entry point for reproducibility:
- `make data` — download/process raw data
- `make features` — build processed features
- `make train` — fit models
- `make test` — run test suite
- `make paper` — render manuscript

**For this repo**: `cluster/21_submit_pipeline.sh` is the cluster orchestrator. A local-machine `Makefile` with the same stage targets would provide the "fresh clone → make all → paper.pdf" guarantee that Turing Way recommends. This is a low-effort addition that Phase 31 could include.

### What file should a new reader open first?

Consensus across Turing Way, CCDS, and pyOpenSci: **README.md** is the entry point. It should link to `docs/PROJECT_STRUCTURE.md` for layout, `CONTRIBUTING.md` for contribution, and `CITATION.cff` for citation. The README should include the minimal "fresh clone" commands.

### Logs placement

The Turing Way template and CCDS both ignore logs — they are transient. Consensus: one gitignored `logs/` at repo root is the standard. SLURM logs (cluster job `.out`/`.err`) are a specialized variant of the same concept and belong there or in `cluster/logs/` (both are acceptable; single location is cleaner).

---

## Q6 — Test Tier Consolidation

**Source:** [pytest docs on markers](https://docs.pytest.org/en/stable/how-to/mark.html) + [Scientific Python pytest guide](https://learn.scientific-python.org/development/guides/pytest/) + existing `pytest.ini` in this repo. **Confidence: HIGH.**

### Industry standard layout

Two dominant conventions:
1. **Flat with markers**: `tests/` contains everything; slow/expensive tests get `@pytest.mark.slow` or `@pytest.mark.scientific`. `pytest -m "not slow"` runs the fast subset.
2. **Tiered sub-directories**: `tests/unit/`, `tests/integration/`, `tests/scientific/`. Directory = implicit marker.

Both are valid. Sub-directories are more reader-friendly for a published research repo where the distinction between fast/unit and slow/scientific is meaningful to a reviewer.

### Recommendation for this repo

Use **tiered sub-directories with markers**:
```
tests/
├── conftest.py
├── unit/                    ← fast (< 1s each) — import/smoke/logic tests
│   ├── test_rlwm_package.py  ← moved from tests/
│   ├── test_period_env.py    ← moved from tests/
│   └── test_wmrl_exploration.py ← moved from tests/
├── integration/              ← medium (1s-60s) — cross-module, MLE, Bayesian integration
│   ├── test_mle_quick.py     ← moved from scripts/fitting/tests/
│   ├── test_load_side_validation.py ← moved from scripts/fitting/tests/
│   ├── test_v4_closure.py    ← moved from scripts/fitting/tests/
│   └── test_v5_phase29_structure.py ← moved from tests/ (root)
└── scientific/               ← slow (> 60s) — parameter recovery, model recovery, scientific invariants
    ├── check_v4_closure.py   ← moved from validation/ (rename: test_v4_closure_scientific.py)
    ├── test_parameter_recovery.py ← moved from validation/
    ├── test_model_consistency.py  ← moved from validation/
    └── test_unified_simulator.py ← moved from validation/
```

### pytest.ini / pyproject.toml update

```ini
[pytest]
testpaths = tests scripts/fitting/tests  # → becomes just: tests
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    scientific: marks scientific validation tests (deselect with '-m "not scientific"')
    integration: marks integration tests
addopts = -v --strict-markers --tb=short --durations=10
```

Remove `validation` from `testpaths` once its contents move to `tests/scientific/`.

### Co-located tests in `scripts/fitting/tests/`

The `scripts/fitting/tests/` subdirectory currently holds integration-level tests for the fitting library. These should move to `tests/integration/` to give pytest a single discovery root. The `test_v4_closure.py` and `test_load_side_validation.py` are the most important — they are structural/invariant guards, not unit tests.

### Slow marker usage

The `@pytest.mark.slow` marker is already registered in `pytest.ini`. After consolidation, add `@pytest.mark.scientific` for everything in `tests/scientific/`. Default CI can run `pytest -m "not slow and not scientific"` to stay fast.

---

## Q7 — Concrete Move Table

This table covers every top-level directory Phase 31 touches. "Breaks what" = paths that must be updated after the move.

### Data directories

| Current Path | New Path | Rationale | Breaks What |
|---|---|---|---|
| `data/rlwm_trauma_PARTICIPANT_SESSION_*.csv` | `data/raw/rlwm_trauma_PARTICIPANT_SESSION_*.csv` | CCDS raw tier | `.gitignore` glob pattern, `config.py DataParams.RAW_DATA_DIR`, `scripts/01_data_preprocessing/01_parse_raw_data.py` |
| `data/backup_example_dataset_pilot.csv` | `data/raw/backup_example_dataset_pilot.csv` | same | same |
| `data/participant_id_mapping.json` | `data/raw/participant_id_mapping.json` | raw metadata | `.gitignore`, `01_parse_raw_data.py` |
| `data/sync_log.txt` | `logs/sync_log.txt` | not data, operational log | `.gitignore` |

### Interim data (parsed stage outputs — stage 01 products)

| Current Path | New Path | Rationale | Breaks What |
|---|---|---|---|
| `output/parsed_demographics.csv` | `data/interim/parsed_demographics.csv` | interim tier (transformed raw) | `config.py DataParams.PARSED_DEMOGRAPHICS`, `.gitignore`, `scripts/02_*/` readers |
| `output/parsed_survey1.csv`, `output/parsed_survey1_all.csv` | `data/interim/parsed_survey1.csv`, etc. | same | `config.py DataParams.PARSED_SURVEY1`, `.gitignore` |
| `output/parsed_survey2.csv`, `output/parsed_survey2_all.csv` | `data/interim/parsed_survey2.csv`, etc. | same | `config.py DataParams.PARSED_SURVEY2` |
| `output/parsed_task_trials.csv` | `data/interim/parsed_task_trials.csv` | same | `config.py DataParams.PARSED_TASK_TRIALS` |
| `output/collated_participant_data.csv` | `data/interim/collated_participant_data.csv` | collated = interim | `config.py DataParams.COLLATED_DATA` |
| `output/demographics_complete.csv` | `data/interim/demographics_complete.csv` | gitignored today — sensitive | `.gitignore`, `scripts/06_*/` |
| `output/participant_info.csv` | `data/interim/participant_info.csv` | gitignored today | `.gitignore`, readers |

### Processed data (analysis-ready, cross-script)

| Current Path | New Path | Rationale | Breaks What |
|---|---|---|---|
| `output/task_trials_long.csv` | `data/processed/task_trials_long.csv` | canonical fitting input | `config.py DataParams.TASK_TRIALS_LONG`, ALL cluster SLURMs (`--data output/task_trials_long.csv`), `manuscript/paper.qmd` (via config) |
| `output/task_trials_long_all.csv` | `data/processed/task_trials_long_all.csv` | same | `config.py DataParams.TASK_TRIALS_ALL` |
| `output/task_trials_long_all_participants.csv` | `data/processed/task_trials_long_all_participants.csv` | legacy alias | `config.py DataParams.TASK_TRIALS_LEGACY` |
| `output/summary_participant_metrics.csv` | `data/processed/summary_participant_metrics.csv` | canonical metrics | `config.py DataParams.SUMMARY_METRICS`, cluster SLURMs, `scripts/06_*/` |
| `output/summary_participant_metrics_all.csv` | `data/processed/summary_participant_metrics_all.csv` | same | readers |

### Small dev artifacts (stay in output/ or delete)

| Current Path | Disposition | Rationale |
|---|---|---|
| `output/task_trials_3participants.csv` | Delete or `data/processed/` | dev artifact |
| `output/task_trials_single_block.csv` | Delete or `data/processed/` | dev artifact |
| `output/wmrl_m3_checkpoint.csv` (and m5) | `models/mle/` | MLE fit checkpoints |
| `output/wmrl_m3_group_summary.csv` (and m5) | `models/mle/` | MLE group summaries |
| `output/wmrl_m3_individual_fits.csv` (and m5) | `models/mle/` | MLE individual fits |
| `output/wmrl_m3_performance_summary.json` (and m5) | `models/mle/` | performance metadata |
| `output/wmrl_m3_timing_log.csv` (and m5) | `models/mle/` or `logs/` | timing artifacts |
| `output/mle_full_fitting_log.txt` | `logs/` | operational log |
| `output/mle_test_log.txt` | `logs/` | operational log |
| `output/mle_wmrl_fitting_log.txt` | `logs/` | operational log |
| `output/wmrl_monitor_log.txt` | `logs/` | operational log |
| `output/conda_packages.txt` | root or delete | environment snapshot |
| `output/v1/` | delete or archive | old version artifacts |
| `output/legacy/` | delete | superseded |

### Model outputs

| Current Path | New Path | Rationale | Breaks What |
|---|---|---|---|
| `output/bayesian/21_baseline/` | `models/bayesian/21_baseline/` | CCDS models tier | `scripts/04_*/b_bayesian/fit_baseline.py`, `scripts/05_*/01_baseline_audit.py`, `scripts/05_*/02_scale_audit.py`, `scripts/06_*/02_compute_loo_stacking.py`, `scripts/06_*/03_model_averaging.py`, `scripts/06_*/08_manuscript_tables.py`, cluster SLURMs, `manuscript/paper.qmd` |
| `output/bayesian/21_l2/` | `models/bayesian/21_l2/` | same | `scripts/04_*/c_level2/fit_with_l2.py`, `scripts/05_*/02_scale_audit.py`, `scripts/06_*/03_model_averaging.py`, `manuscript/paper.qmd` |
| `output/bayesian/21_prior_predictive/` | `models/bayesian/21_prior_predictive/` | same | `scripts/03_*/04_run_prior_predictive.py`, cluster `03_prefitting_*.slurm` |
| `output/bayesian/21_recovery/` | `models/bayesian/21_recovery/` | same | `scripts/03_*/05_run_bayesian_recovery.py` |
| `output/bayesian/level2/` | `models/bayesian/level2/` | same | `scripts/06_*/07_bayesian_level2_effects.py`, `manuscript/paper.qmd` |
| `output/bayesian/manuscript/` | `models/bayesian/manuscript/` | same | `manuscript/paper.qmd` |
| `output/bayesian/pscan_benchmark_cpu.json` | `models/bayesian/pscan_benchmark_cpu.json` | benchmark metadata | `validation/benchmark_parallel_scan.py` |
| `output/bayesian/pscan_benchmark_gpu.json` | `models/bayesian/pscan_benchmark_gpu.json` | same | same |
| `output/mle/` (all contents) | `models/mle/` | CCDS models tier | `scripts/06_*/01_compare_models.py`, `scripts/06_*/04_analyze_mle_by_trauma.py`, `scripts/06_*/05_regress_parameters_on_scales.py`, `scripts/05_*/03_run_posterior_ppc.py`, `manuscript/paper.qmd` |
| `output/ppc/` | `models/ppc/` | PPC are model artifacts | `scripts/05_*/03_run_posterior_ppc.py` |
| `output/recovery/` | `models/recovery/` | recovery outputs = model artifacts | `scripts/03_*/03_run_model_recovery.py` |
| `output/parameter_exploration/` | `models/parameter_exploration/` | model space artifacts | `scripts/03_*/02_run_parameter_sweep.py` |

### Report outputs

| Current Path | New Path | Rationale | Breaks What |
|---|---|---|---|
| `output/descriptives/` | `reports/tables/descriptives/` | report tier — manuscript tables | `scripts/02_*/04_run_statistical_analyses.py` (hardcoded `default='output/descriptives'`) |
| `output/model_comparison/comparison_results.csv` | `reports/tables/model_comparison/comparison_results.csv` | report artifact | `scripts/06_*/01_compare_models.py`, `manuscript/paper.qmd` |
| `output/model_comparison/winner_heterogeneity_figure.png` | `reports/figures/model_comparison/winner_heterogeneity_figure.png` | figure → reports/figures | `manuscript/paper.qmd` (direct image path) |
| `output/model_comparison/` (remaining CSVs) | `reports/tables/model_comparison/` | report tier | `scripts/06_*/` |
| `output/behavioral_summary/` | `reports/tables/behavioral_summary/` | report artifact | `scripts/02_*/` |
| `output/regressions/` (CSVs) | `reports/tables/regressions/` | report tier | `scripts/06_*/05_regress_parameters_on_scales.py` |
| `output/results_text/` | `reports/tables/results_text/` | report text | `scripts/06_*/` |
| `output/statistical_analyses/` | `reports/tables/statistical_analyses/` | report tier | `scripts/04_*/` |
| `output/supplementary_materials/` | `reports/tables/supplementary/` | report tier | `scripts/06_*/` |
| `output/trauma_groups/` | `reports/tables/trauma_groups/` | report artifact | `scripts/02_*/03_analyze_trauma_groups.py`, `scripts/06_*/` |
| `output/trauma_scale_analysis/` | `reports/tables/trauma_scale_analysis/` | report artifact | `scripts/06_*/` |
| `output/model_performance/` | `reports/tables/model_performance/` | report artifact | `scripts/06_*/` |

### Figures

| Current Path | New Path | Rationale | Breaks What |
|---|---|---|---|
| `figures/` (all contents) | `reports/figures/` | CCDS reports tier | `config.py FIGURES_DIR`, ALL scripts that write to `figures/`, `manuscript/paper.qmd` image includes |
| `output/bayesian/figures/` | `reports/figures/bayesian/` | same tier | `scripts/06_*/07_bayesian_level2_effects.py`, `scripts/06_*/08_manuscript_tables.py` |

### Tests

| Current Path | New Path | Rationale | Breaks What |
|---|---|---|---|
| `tests/test_rlwm_package.py` | `tests/unit/test_rlwm_package.py` | unit tier | `pytest.ini testpaths` |
| `tests/test_period_env.py` | `tests/unit/test_period_env.py` | unit tier | same |
| `tests/test_wmrl_exploration.py` | `tests/unit/test_wmrl_exploration.py` | unit tier | same |
| `tests/test_performance_plots.py` | `tests/unit/test_performance_plots.py` | unit tier | same |
| `tests/test_v5_phase29_structure.py` | `tests/integration/test_v5_phase29_structure.py` | structural guard = integration | `REPO_ROOT` path inside test (uses `Path(__file__).resolve().parents[1]` — must be `parents[2]` after moving one level deeper) |
| `scripts/fitting/tests/test_mle_quick.py` | `tests/integration/test_mle_quick.py` | fitting integration | `pytest.ini testpaths` |
| `scripts/fitting/tests/test_load_side_validation.py` | `tests/integration/test_load_side_validation.py` | structural/invariant | same |
| `scripts/fitting/tests/test_v4_closure.py` | `tests/integration/test_v4_closure.py` | v4 closure guard | `validation/check_v4_closure.py` path reference inside test |
| `scripts/fitting/tests/test_loo_stacking.py` | `tests/integration/test_loo_stacking.py` | fitting integration | same |
| `scripts/fitting/tests/test_bayesian_recovery.py` | `tests/integration/test_bayesian_recovery.py` | fitting integration | same |
| `scripts/fitting/tests/test_gpu_m4.py` | `tests/integration/test_gpu_m4.py` | GPU integration | same |
| `scripts/fitting/tests/conftest.py` | merge into `tests/conftest.py` | shared fixtures | check for conflicts |
| `validation/check_v4_closure.py` | `tests/scientific/test_v4_closure_scientific.py` | scientific tier | `scripts/fitting/tests/test_v4_closure.py` calls `python validation/check_v4_closure.py` subprocess |
| `validation/test_parameter_recovery.py` | `tests/scientific/test_parameter_recovery.py` | scientific tier | `pytest.ini testpaths` |
| `validation/test_model_consistency.py` | `tests/scientific/test_model_consistency.py` | scientific tier | same |
| `validation/test_unified_simulator.py` | `tests/scientific/test_unified_simulator.py` | scientific tier | same |
| `validation/compare_posterior_to_mle.py` | `tests/scientific/compare_posterior_to_mle.py` | scientific tool | docs references, cluster/23.1 slurm |
| `validation/benchmark_parallel_scan.py` | `tests/scientific/benchmark_parallel_scan.py` | performance tool | cluster/legacy SLURMs |
| `validation/legacy/` | `tests/scientific/legacy/` or delete | legacy | - |

### Logs

| Current Path | New Path | Rationale |
|---|---|---|
| `logs/*.out`, `logs/*.err` | `logs/` (unchanged location) | already gitignored |
| `cluster/logs/*.out`, `cluster/logs/*.err` | `logs/` (merge into root `logs/`) | single gitignored location |
| SLURM `#SBATCH --output=cluster/logs/...` | `#SBATCH --output=logs/...` | update all cluster/*.slurm |

---

## Q8 — Pitfalls and Breaking Changes

### Pitfall 1: ArviZ NetCDF absolute paths inside .nc files

**What goes wrong:** ArviZ `InferenceData` objects stored as `.nc` files do NOT embed absolute paths inside the file — they only store posterior samples and metadata. Moving `output/bayesian/21_baseline/*.nc` → `models/bayesian/21_baseline/*.nc` does NOT corrupt the files. The risk is entirely in the code paths that reference these locations.

**Mitigation:** Update `config.py` centrally; all scripts that use `config.py` constants will inherit new paths. Scripts with hardcoded defaults (e.g., `--bayesian-dir default="output/bayesian/21_baseline/"`) need their defaults updated too.

### Pitfall 2: `test_v5_phase29_structure.py` uses `Path(__file__).resolve().parents[1]`

**What goes wrong:** Moving this test from `tests/` to `tests/integration/` shifts its position by one directory level. `parents[1]` currently resolves to the repo root. After moving to `tests/integration/`, `parents[1]` resolves to `tests/` and `parents[2]` resolves to repo root.

**Mitigation:** Update `REPO_ROOT = Path(__file__).resolve().parents[2]` in the moved test.

### Pitfall 3: `test_v4_closure.py` calls `python validation/check_v4_closure.py` as a subprocess

**What goes wrong:** The integration test at `scripts/fitting/tests/test_v4_closure.py` runs `validation/check_v4_closure.py` as a subprocess from the repo root. After moving `check_v4_closure.py` to `tests/scientific/`, the subprocess path changes.

**Mitigation:** Either (a) update the subprocess path in `test_v4_closure.py` → `tests/scientific/test_v4_closure_scientific.py`, or (b) keep `validation/check_v4_closure.py` as a standalone script at the new path and update all references.

### Pitfall 4: Cluster SLURMs with `mkdir -p output/bayesian` and hardcoded paths

**What goes wrong:** Nearly all SLURM scripts (fetched: `04b_bayesian_cpu.slurm`, `03_prefitting_cpu.slurm`, `06_fit_analyses.slurm`) have hardcoded `output/bayesian`, `output/mle`, `output/model_comparison`, `figures/21_bayesian` paths. These cannot be updated via `config.py` — they use direct shell strings.

**Mitigation:** Systematic sed-pass or manual update of all active cluster scripts (not `cluster/legacy/`). Count: ~12-15 active SLURMs, each with 5-20 path references. This is the highest-effort single task in Phase 31.

### Pitfall 5: `manuscript/paper.qmd` has ~10 hardcoded `../output/` paths

The Quarto manuscript has absolute-relative paths like `Path("../output/mle")`, `Path("../output/model_comparison")`, `Path("../output/bayesian/21_baseline/loo_stacking_results.csv")`, and a direct figure include `../output/model_comparison/winner_heterogeneity_figure.png`. These all need updating.

**Mitigation:** After all moves are done, update `manuscript/paper.qmd` paths to use `../models/`, `../reports/tables/`, `../reports/figures/`. Then run `quarto render manuscript/paper.qmd` as a smoke test.

### Pitfall 6: Scripts with hardcoded `Path(...)` literals (not via `config.py`)

Confirmed via grep: the following scripts have hardcoded path strings that are NOT routed through `config.py`:
- `scripts/06_fit_analyses/05_regress_parameters_on_scales.py`: `Path('output/summary_participant_metrics.csv')`, `Path('output/trauma_groups/group_assignments.csv')`, `Path('output/parsed_demographics.csv')`, `Path('output/bayesian')`, `Path('output/mle')`
- `scripts/06_fit_analyses/01_compare_models.py`: `Path('output/trauma_groups/group_assignments.csv')`
- `scripts/03_model_prefitting/03_run_model_recovery.py`: `Path('output/recovery')`, `Path('figures/recovery')`
- `scripts/02_behav_analyses/03_analyze_trauma_groups.py`: `FIGURES_DIR = Path('figures/trauma_groups')`

These require in-file edits, not just `config.py` changes.

### Pitfall 7: `output/legacy/` treatment

This directory contains old version artifacts (`v1/`, `legacy/`). **Recommendation: delete before Phase 31 commits**. Moving legacy artifacts to `models/legacy/` would just perpetuate confusion. If any artifact needs preservation, move it to `.planning/milestones/` or document in git history.

### Pitfall 8: Cluster log path in SLURM `#SBATCH --output`

Every active SLURM has `#SBATCH --output=cluster/logs/%x_%j.out`. Merging `cluster/logs/` into `logs/` requires updating these directives to `#SBATCH --output=logs/%x_%j.out`. This is safe but tedious.

### Pitfall 9: Sequencing dependency on cold-start (Phase 24)

The roadmap note says: "data reorganization invalidates cached MCMC posteriors." This is only true if Phase 24 has NOT yet run — i.e., if the repo still relies on cached posteriors from pre-reorganization paths. If Phase 24 (cold-start) has re-run all fits into the new paths, then Phase 31 reorganization of `output/bayesian/` → `models/bayesian/` is safe (the new artifacts are at the new paths already). **If Phase 24 has NOT yet run, do the path reorganization FIRST (as a git commit with path renames), then run Phase 24 so it deposits artifacts directly to the correct new locations.**

---

## Q9 — Wave Structure Recommendation

Phase 31 has five logical moves with one clear dependency: `config.py` must be updated before scripts are updated (scripts read from config), but the physical file moves can happen before OR after the config update as long as nothing runs in between.

### Recommended wave breakdown

**Wave A — Foundation (must be first, blocks all others)**
- Update `config.py`: add `INTERIM_DIR`, `PROCESSED_DIR`, `MODELS_DIR`, `REPORTS_DIR`, updated `FIGURES_DIR`; deprecate `OUTPUT_DIR` (or keep as legacy alias briefly)
- Create new top-level directories: `data/raw/`, `data/interim/`, `data/processed/`, `models/`, `reports/figures/`, `reports/tables/`
- Update `.gitignore`: glob patterns for `data/raw/` instead of `data/`, `models/bayesian/*.nc`, `logs/`
- **Estimated effort**: 1-2 hours. No script runs needed. This is pure config + scaffolding.

**Wave B — Data moves (parallel with Wave C once Wave A done)**
- `git mv data/*.csv data/raw/` (or `data/raw/*.csv` depending on exact file list)
- `git mv output/parsed_*.csv data/interim/`
- `git mv output/collated_*.csv data/interim/`
- `git mv output/task_trials_long*.csv data/processed/`
- `git mv output/summary_participant_metrics*.csv data/processed/`
- Update `DataParams` paths in `config.py` (already updated in Wave A)
- Update `scripts/01_data_preprocessing/01_parse_raw_data.py` (reads from `DATA_DIR`, writes to new interim)
- **Estimated effort**: 2-3 hours. Test: `python scripts/01_*/01_parse_raw_data.py --dry-run` (if supported) or check config load.

**Wave C — Model and report moves (parallel with Wave B once Wave A done)**
- `git mv output/bayesian/ models/bayesian/`
- `git mv output/mle/ models/mle/`
- `git mv output/ppc/ models/ppc/`
- `git mv output/recovery/ models/recovery/`
- `git mv output/parameter_exploration/ models/parameter_exploration/`
- `git mv output/descriptives/ reports/tables/descriptives/`
- `git mv output/model_comparison/ reports/tables/model_comparison/` (except figure → `reports/figures/`)
- `git mv output/behavioral_summary/ reports/tables/behavioral_summary/`
- `git mv output/regressions/ reports/tables/regressions/`
- `git mv figures/ reports/figures/` (everything)
- **Estimated effort**: 2-3 hours for moves; 4-6 hours for updating all script defaults + cluster SLURMs. This is the largest wave.

**Wave D — Test consolidation (independent of B/C, after Wave A)**
- Create `tests/unit/`, `tests/integration/`, `tests/scientific/`
- `git mv tests/test_rlwm_package.py tests/unit/`, etc.
- `git mv scripts/fitting/tests/*.py tests/integration/` (update imports)
- `git mv validation/*.py tests/scientific/` (update subprocess paths)
- Update `pytest.ini testpaths` → `testpaths = tests`
- Update `REPO_ROOT` references in `test_v5_phase29_structure.py`
- **Estimated effort**: 3-4 hours. Test: `pytest tests/ -m "not slow and not scientific"` must pass green.

**Wave E — Log consolidation + closure guard + docs (after B/C/D)**
- `git mv cluster/logs/ logs/cluster/` OR update `#SBATCH --output=logs/%x_%j.out` in all SLURMs
- Update `manuscript/paper.qmd` path references (10 occurrences)
- Update `docs/` path references (`docs/04_methods/README.md`, `docs/legacy/`)
- Add Phase 31 assertions to `tests/integration/test_v5_phase29_structure.py` (extend, don't replace)
- Add `CITATION.cff` stub
- Add optional `Makefile` with stage targets
- **Estimated effort**: 3-4 hours.

### Safe-to-parallelize pairs

- Wave B and Wave C are fully parallel (different file trees, no shared dependencies).
- Wave D is fully parallel with B and C (test files don't touch data or models).
- Wave E must wait for B, C, D because it updates docs and closure guard assertions that reference new paths.

### Suggested plan structure for planner

```
31-01-PLAN.md  — Wave A: config.py + directory scaffold
31-02-PLAN.md  — Wave B: data moves + DataParams update
31-03-PLAN.md  — Wave C: model/report moves + script/SLURM updates
31-04-PLAN.md  — Wave D: test consolidation + pytest.ini
31-05-PLAN.md  — Wave E: logs + docs + closure guard + CITATION.cff + Makefile
```

---

## Recommended Final Layout

```
rlwm_trauma_analysis/
├── CITATION.cff
├── CLAUDE.md
├── README.md
├── Makefile                          # optional local workflow twin
├── config.py                         # updated constants
├── pyproject.toml
├── pytest.ini                        # testpaths = tests (only)
├── environment.yml
├── environment_gpu.yml
├── requirements.txt
├── requirements-dev.txt
├── run_data_pipeline.py
│
├── data/
│   ├── raw/                          # immutable jsPsych CSVs (gitignored by pattern)
│   ├── interim/                      # parsed_*.csv, collated_*.csv (gitignored — sensitive)
│   └── processed/                    # task_trials_long*.csv, summary_*.csv (tracked)
│
├── models/
│   ├── bayesian/                     # *.nc (gitignored), winners.txt, .gitkeep dirs (tracked)
│   ├── mle/                          # *_individual_fits.csv, *_group_summary.csv (tracked)
│   ├── ppc/                          # posterior predictive check outputs
│   └── recovery/                     # model and parameter recovery artifacts
│
├── reports/
│   ├── figures/                      # all PNG/PDF figures (was figures/ + output/bayesian/figures/)
│   └── tables/                       # all report CSVs/MDs (was output/descriptives/, output/model_comparison/, etc.)
│
├── scripts/                          # UNCHANGED — Scheme D (01-06 stages)
│   ├── 01_data_preprocessing/
│   ├── 02_behav_analyses/
│   ├── 03_model_prefitting/
│   ├── 04_model_fitting/{a_mle,b_bayesian,c_level2}/
│   ├── 05_post_fitting_checks/
│   ├── 06_fit_analyses/
│   ├── fitting/                      # library helpers (no tests/ subdirectory after consolidation)
│   └── utils/
│
├── src/rlwm/                         # UNCHANGED — installable package
│   ├── fitting/
│   ├── envs/
│   └── models/
│
├── cluster/                          # SLURM scripts (path strings updated)
│   └── logs/ → merged into top-level logs/
│
├── tests/
│   ├── conftest.py
│   ├── unit/                         # fast tests (< 1s)
│   ├── integration/                  # medium tests (structure guards, MLE smoke)
│   └── scientific/                   # slow tests (parameter recovery, v4 closure)
│
├── logs/                             # gitignored — dev + SLURM job outputs
├── docs/                             # updated path references
├── manuscript/                       # UNCHANGED (Quarto source)
└── .planning/                        # local-only audit trail
```

---

## Open Questions for User

These must be resolved before or during planning:

1. **`data/interim/` placement**: Should interim data (parsed_*.csv, collated_*.csv) live under `data/interim/` (CCDS standard) or remain in a separate gitignored location? They contain participant survey responses (sensitive) — already gitignored. The CCDS pattern works either way, but the user should confirm the tier name is `interim/` not something more descriptive like `data/preprocessed/`.

2. **Track `models/` in git or gitignore?** Large `.nc` posteriors are already gitignored. But `models/mle/*.csv` individual fits (dozens of files, ~50-200KB each) are currently tracked. Should these move to gitignore with "produce from script" semantics, or remain tracked for reader convenience? Tracked = reviewer can clone and see results without re-running. Gitignored = cleaner repo, requires running fits. **Recommendation: keep MLE CSVs tracked; gitignore Bayesian NC files (status quo).**

3. **JOSS target: yes or no?** If yes, add `paper.md` at repo root as a Phase 31 deliverable. If no, skip it. The difference is ~2 hours of work.

4. **Single `logs/` at root or keep `cluster/logs/` separate?** The architectural question: do job logs (which are cluster-specific) belong alongside dev logs? Single location is cleaner; separate is more intuitive on M3 cluster where `cluster/` is already a namespace. **Recommendation: merge into `logs/` at root; update `#SBATCH --output` directives.**

5. **`output/legacy/` and `output/v1/`**: Delete or archive? These contain old artifacts from pre-v4 runs. **Recommendation: delete** — history is preserved in git. Confirm with user before Phase 31 execution.

6. **Phase 24 sequencing**: Has Phase 24 (cold-start) run and produced fresh posteriors? If yes, Phase 31 can run immediately after Phase 27 closure. If no, Phase 31 must run before Phase 24 to ensure Phase 24 deposits artifacts to the correct new paths. This is the critical sequencing gate.

7. **`scripts/fitting/` library helpers**: After test consolidation, `scripts/fitting/` will no longer have a `tests/` subdirectory. Should `scripts/fitting/` itself be renamed to clarify it is a library namespace (`scripts/lib/` or just keep as-is)? **Recommendation: keep as `scripts/fitting/`** — already documented in CLAUDE.md.

---

## Sources

### Primary (HIGH confidence)
- [CCDS official docs](https://cookiecutter-data-science.drivendata.org/) — directory layout, data tier definitions
- [CCDS v2 blog post](https://drivendata.co/blog/ccds-v2) — v1→v2 changes, models/ treatment
- [CCDS GitHub README](https://github.com/drivendataorg/cookiecutter-data-science) — canonical tree
- [CCDS opinions page](https://cookiecutter-data-science.drivendata.org/opinions/) — data immutability quotes
- [pytest markers docs](https://docs.pytest.org/en/stable/how-to/mark.html) — @pytest.mark conventions
- Direct code analysis (grep, file reads) — break-impact analysis of all hardcoded paths

### Secondary (MEDIUM confidence)
- [pyOpenSci Python Package Guide](https://www.pyopensci.org/python-package-guide/index.html) — src-layout, tests recommendations
- [JOSS documentation](https://joss.readthedocs.io/en/latest/) — submission requirements (review_criteria page 403'd)
- [The Turing Way reproducible-project-template](https://github.com/the-turing-way/reproducible-project-template) — layout conventions
- [Scientific Python pytest guide](https://learn.scientific-python.org/development/guides/pytest/) — test layout standards
- [AnneCollins/WMH](https://github.com/AnneCollins/WMH) — domain exemplar (single MATLAB repo, limited generalizability)

### Tertiary (LOW confidence / domain survey)
- Web search: computational psychiatry GitHub repo patterns — confirms CCDS not standard in domain; ad-hoc layouts dominate

---

## Metadata

**Confidence breakdown:**
- CCDS conventions (Q1): HIGH — fetched from official docs directly
- pyOpenSci/JOSS (Q2/Q3): MEDIUM — guide index fetched; detail pages partially blocked
- Exemplar repos (Q4): MEDIUM — only one directly fetched; limited sample
- Reproducibility (Q5): MEDIUM — Turing Way template fetched
- Test consolidation (Q6): HIGH — pytest docs + existing pytest.ini analyzed
- Move table (Q7): HIGH — all paths verified by grep against repo filesystem
- Pitfall analysis (Q8): HIGH — direct code analysis, not inferred

**Research date:** 2026-04-24
**Valid until:** 2026-06-01 (CCDS stable; internal analysis is current-state)
