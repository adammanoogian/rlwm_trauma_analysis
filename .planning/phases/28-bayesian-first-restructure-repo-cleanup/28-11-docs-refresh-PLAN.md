---
wave: 5
depends_on: [28-02, 28-03, 28-04, 28-05, 28-06, 28-07, 28-08, 28-09]
files_modified:
  - CLAUDE.md  (project root — "Code Organization" and "Quick Reference" sections)
  - README.md  (pipeline block + dev setup)
  - docs/TASK_AND_ENVIRONMENT.md  (if it cross-references deleted shim paths)
  - docs/MODEL_REFERENCE.md  (if it cross-references deleted shim paths)
autonomous: true
---

# 28-11 Docs Refresh: Update CLAUDE.md + README.md + docs/ for New Structure

## Goal

Bring project-level documentation into sync with the consolidated repo layout. Update `CLAUDE.md` and `README.md` to reference the new `scripts/data_processing/`, `scripts/behavioral/`, `scripts/simulations_recovery/`, `scripts/post_mle/`, `scripts/bayesian_pipeline/`, and `src/rlwm/fitting/` locations; drop mentions of the deleted `environments/` and `models/` shim paths; ensure the Quick Reference code blocks work against the new paths.

## Must Haves

- [ ] `CLAUDE.md` "Code Organization" section (currently at lines ~90–150 per user's project CLAUDE.md shown in conversation context) reflects new directory structure:
  - `scripts/data_processing/01-04*.py` (data processing)
  - `scripts/behavioral/05-08*.py` (behavioral analysis)
  - `scripts/simulations_recovery/09-11*.py` (simulations & model validation)
  - `scripts/` 12-14*.py (model fitting) — unchanged location
  - `scripts/post_mle/15-18*.py` (post-MLE + rendering backend)
  - `scripts/bayesian_pipeline/21_*.py` (Bayesian pipeline)
  - `src/rlwm/fitting/` (authoritative JAX likelihoods + NumPyro models)
- [ ] `CLAUDE.md` "Quick Reference" `### Run Full Pipeline` block updated so every `python scripts/NN_*.py` line reflects the new path (e.g., `python scripts/data_processing/01_parse_raw_data.py`). Every listed command must run successfully from repo root.
- [ ] `CLAUDE.md` "Quick Reference" `### Cluster Execution (Monash M3)` block updated:
  - Replace individual per-model `sbatch cluster/13_bayesian_m{1..6b}.slurm` examples with the consolidated `sbatch --export=ALL,MODEL=wmrl_m6b --time=36:00:00 cluster/13_bayesian_choice_only.slurm` pattern.
- [ ] `CLAUDE.md` removes all references to `environments/` and `models/` top-level packages (shims deleted in plan 28-01). The correct paths are `src/rlwm/envs/` and `src/rlwm/models/`.
- [ ] `README.md` pipeline block updated to match the new paths.
- [ ] `README.md` dev-setup section includes `pip install -e .` as a prerequisite (plan 28-01 added a minimal note; this plan expands it into a proper setup block).
- [ ] `docs/TASK_AND_ENVIRONMENT.md` and `docs/MODEL_REFERENCE.md` scanned for any cross-references to deleted paths — updated if found.
- [ ] Grep invariant: `grep -rn "environments/\\|scripts/\\(01_parse_raw_data\\|02_create_collated_csv\\|03_create_task_trials_csv\\|04_create_summary_csv\\|05_summarize_behavioral_data\\|06_visualize_task_performance\\|07_analyze_trauma_groups\\|08_run_statistical_analyses\\|09_generate_synthetic_data\\|09_run_ppc\\|10_run_parameter_sweep\\|11_run_model_recovery\\|15_analyze_mle_by_trauma\\|16_regress_parameters_on_scales\\|17_analyze_winner_heterogeneity\\|18_bayesian_level2_effects\\|21_[a-z_]*\\)\\.py" CLAUDE.md README.md docs/` returns zero matches (except in `docs/legacy/`). Pattern covers ALL scripts moved by Plans 28-02, 28-03, 28-04, 28-05, 28-06 (i.e., `01-11`, `15-18`, `21_*`). Scripts 12/13/14 are unchanged and not included.
- [ ] `pytest scripts/fitting/tests/test_v4_closure.py` still passes 3/3 (plan changes only docs).
- [ ] Atomic commit: `docs(28-11): refresh CLAUDE.md + README.md + docs/ for consolidated repo layout`.

## Tasks

<tasks>
  <task id="1">
    <title>Read all four doc files and mark every stale path reference</title>
    <detail>Read `CLAUDE.md`, `README.md`, `docs/TASK_AND_ENVIRONMENT.md`, `docs/MODEL_REFERENCE.md` in full. Grep each for:
      - `environments/` and `environments.` (deleted shim — Plan 28-01)
      - `from models\\.` and `models/` (deleted shim — Plan 28-01)
      - `scripts/0[1-4]_` — moved to `data_processing/` (Plan 28-02): `01_parse_raw_data`, `02_create_collated_csv`, `03_create_task_trials_csv`, `04_create_summary_csv`
      - `scripts/0[5-8]_` — moved to `behavioral/` (Plan 28-03): `05_summarize_behavioral_data`, `06_visualize_task_performance`, `07_analyze_trauma_groups`, `08_run_statistical_analyses`
      - `scripts/09_` / `scripts/10_` / `scripts/11_` — moved to `simulations_recovery/` (Plan 28-04): **both** `09_generate_synthetic_data.py` AND `09_run_ppc.py`, plus `10_run_parameter_sweep.py` and `11_run_model_recovery.py`
      - `scripts/1[5-8]_` — moved to `post_mle/` (Plan 28-05): `15_analyze_mle_by_trauma`, `16_regress_parameters_on_scales`, `17_analyze_winner_heterogeneity`, `18_bayesian_level2_effects`
      - `scripts/21_[a-z_]*\\.py` — moved to `bayesian_pipeline/` (Plan 28-06): all 9 `21_*.py` scripts
      - `13_bayesian_m[1-6]\\.slurm` (except m6b_subscale — deleted per plan 28-08)
      Produce a change-list before editing.</detail>
  </task>

  <task id="2">
    <title>Update CLAUDE.md Code Organization section</title>
    <detail>Rewrite the `scripts/` tree diagram (currently at lines ~90–150 per user's project CLAUDE.md) to reflect the new 5 subdirectories. Keep the numeric-prefix convention explanation; just update the paths. Remove the explicit mention of `environments/rlwm_env.py` and `models/q_learning.py` / `models/wm_rl_hybrid.py` under "Key Files" — update those to `src/rlwm/envs/rlwm_env.py`, `src/rlwm/models/q_learning.py`, `src/rlwm/models/wm_rl_hybrid.py`.</detail>
  </task>

  <task id="3">
    <title>Update CLAUDE.md Quick Reference Run Full Pipeline block</title>
    <detail>Rewrite each `python scripts/NN_*.py` line to the new path:
      ```
      python scripts/data_processing/01_parse_raw_data.py
      python scripts/data_processing/02_create_collated_csv.py
      python scripts/data_processing/03_create_task_trials_csv.py
      python scripts/data_processing/04_create_summary_csv.py
      python scripts/behavioral/05_summarize_behavioral_data.py
      python scripts/behavioral/06_visualize_task_performance.py
      python scripts/behavioral/07_analyze_trauma_groups.py
      python scripts/behavioral/08_run_statistical_analyses.py
      python scripts/12_fit_mle.py --model ...  # unchanged location
      python scripts/14_compare_models.py        # unchanged location
      python scripts/post_mle/15_analyze_mle_by_trauma.py --model all
      python scripts/post_mle/16_regress_parameters_on_scales.py --model all
      ```
      Preserve the existing CLI flags and examples.</detail>
  </task>

  <task id="4">
    <title>Update CLAUDE.md Quick Reference Cluster Execution block</title>
    <detail>Replace the per-model SLURM examples:
      - Before: `sbatch cluster/13_bayesian_m3.slurm`
      - After: `sbatch --export=ALL,MODEL=wmrl_m3 cluster/13_bayesian_choice_only.slurm`
      - M6b override: `sbatch --time=36:00:00 --export=ALL,MODEL=wmrl_m6b cluster/13_bayesian_choice_only.slurm`
      Retain the GPU template reference (`13_bayesian_gpu.slurm`) unchanged.</detail>
  </task>

  <task id="5">
    <title>Update README.md pipeline block</title>
    <detail>Same treatment as tasks 3–4 but in README.md — most common pipeline run-through now uses the new paths.</detail>
  </task>

  <task id="6">
    <title>Expand README.md dev setup</title>
    <detail>Plan 28-01 added a minimal line about `pip install -e .`. Expand this in README.md into a proper "Setup" section:
      ```
      ## Setup
      # 1. Create conda env
      conda env create -f environment.yml
      # 2. Install package in editable mode (required to import `rlwm`)
      pip install -e .
      # 3. Run tests
      pytest
      ```
      Keep it short (5-10 lines).</detail>
  </task>

  <task id="7">
    <title>Update docs/ cross-refs if any stale paths found</title>
    <detail>For each stale path surfaced by task 1's grep of `docs/TASK_AND_ENVIRONMENT.md` and `docs/MODEL_REFERENCE.md`, rewrite to the new location. If none found, no-op.</detail>
  </task>

  <task id="8">
    <title>Atomic commit</title>
    <detail>`docs(28-11): refresh CLAUDE.md + README.md + docs/ for consolidated repo layout`.</detail>
  </task>
</tasks>

## Verification

```bash
# Stale-path grep returns zero matches in active docs
! grep -rn "^from environments\\.\\|^from models\\." CLAUDE.md README.md docs/ --include="*.md"
! grep -n "scripts/\\(01_parse_raw_data\\|02_create_collated_csv\\|03_create_task_trials_csv\\|04_create_summary_csv\\|05_summarize_behavioral_data\\|06_visualize_task_performance\\|07_analyze_trauma_groups\\|08_run_statistical_analyses\\|09_generate_synthetic_data\\|09_run_ppc\\|10_run_parameter_sweep\\|11_run_model_recovery\\|15_analyze_mle_by_trauma\\|16_regress_parameters_on_scales\\|17_analyze_winner_heterogeneity\\|18_bayesian_level2_effects\\|21_[a-z_]*\\)\\.py" CLAUDE.md README.md

# New paths present
grep -n "scripts/data_processing/01_parse_raw_data" CLAUDE.md README.md
grep -n "scripts/bayesian_pipeline/21_compute_loo_stacking" CLAUDE.md
grep -n "13_bayesian_choice_only" CLAUDE.md

# pip install -e . in README
grep -n "pip install -e \\." README.md

pytest scripts/fitting/tests/test_v4_closure.py -v
```

## Requirement IDs

Closes: **REFAC-12**.
