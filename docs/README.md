# RLWM Trauma Analysis - Documentation Index

**Last Updated:** 2026-01-27

Welcome to the RLWM Trauma Analysis project documentation. This folder contains all documentation organized into thematic subfolders for easy navigation.

---

## Documentation Structure

Documentation is organized into 5 main folders:

### [00_current_todos/](00_current_todos/)
**Active project tracking and work items**

Contents:
- `to-dos.md` - Current work items and project status

**Use when:**
- Checking what needs to be done
- Tracking project progress

---

### [01_project_protocol/](01_project_protocol/)
**Setup, configuration, and study protocol**

Contents:
- `QUICKSTART.md` - Getting started guide (START HERE)
- `PARTICIPANT_EXCLUSIONS.md` - Exclusion criteria and final N=48
- `plotting_config_guide.md` - Figure styling configuration

**Use when:**
- Setting up the project environment
- Understanding participant exclusion criteria
- Configuring plot aesthetics

---

### [02_pipeline_guide/](02_pipeline_guide/)
**How to run analyses and reproduce results**

Contents:
- `ANALYSIS_PIPELINE.md` - Complete analysis workflow
- `PLOTTING_REFERENCE.md` - How to create figures
- `trauma_group_analysis_guide.md` - Trauma group comparisons

**Use when:**
- Running the data pipeline
- Fitting computational models
- Creating publication figures
- Analyzing trauma groups

---

### [03_methods_reference/](03_methods_reference/)
**Formulas, model specifications, and technical details**

Contents:
- `MODEL_REFERENCE.md` - **CRITICAL** Q-learning and WM-RL model mathematics
- `TASK_AND_ENVIRONMENT.md` - Task structure and environment API
- `references/` - Key papers (Senta et al., 2025; Ehrlich et al.)

**Use when:**
- Implementing model fitting
- Understanding model parameters
- Writing methods sections
- Checking mathematical specifications

**NOTE:** Read `MODEL_REFERENCE.md` before modifying any fitting code!

---

### [04_scientific_reports/](04_scientific_reports/)
**Results, findings, and publication materials**

Contents:
- *(To be populated with results and manuscript materials)*

**Use when:**
- Preparing manuscripts
- Presenting results
- Writing progress reports

---

### [legacy/](legacy/)
**Deprecated documentation**

Contains outdated documentation kept for reference:
- Old model references
- Superseded environment docs
- Historical handoff notes

**Do not use for current work.**

---

## Quick Start Guide

### New to the Project?
1. Read `01_project_protocol/QUICKSTART.md` - Set up your environment
2. Read `02_pipeline_guide/ANALYSIS_PIPELINE.md` - Understand the workflow
3. Read `03_methods_reference/MODEL_REFERENCE.md` - Understand the models
4. Check `00_current_todos/to-dos.md` - See current status

### Running Analyses?
```bash
# Activate environment
conda activate ds_env

# Run full data pipeline
python run_data_pipeline.py --no-sync

# Fit Q-learning model
python scripts/fitting/fit_with_jax.py --model qlearning

# Run regressions
python scripts/analysis/regress_parameters_on_scales.py
```

### Key Files Outside Docs
- `config.py` - Central configuration (paths, parameters, exclusions)
- `CLAUDE.md` - AI assistant guidelines
- `run_data_pipeline.py` - Master data pipeline runner

---

## Project Overview

**Research Question:** How do trauma exposure and PTSD symptoms relate to reinforcement learning and working memory parameters?

**Models:**
- Q-Learning (asymmetric learning rates: α+, α-, ε)
- WM-RL Hybrid (adds: φ, ρ, K for working memory)

**Sample:** N=48 participants (after exclusions)

**Key Measures:**
- LEC-5: Trauma exposure
- IES-R: PTSD symptoms (intrusion, avoidance, hyperarousal)
