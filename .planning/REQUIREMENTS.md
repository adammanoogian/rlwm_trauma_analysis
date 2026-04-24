# Requirements: RLWM Trauma Analysis — v5.0

**Milestone:** v5.0 — Empirical Artifacts & Manuscript Finalization
**Defined:** 2026-04-19
**Core Value:** The model must correctly dissociate perseverative responding from learning-rate effects (α₋), enabling accurate identification of whether post-reversal failures reflect motor perseveration or outcome insensitivity in trauma populations.

**Milestone goal:** Execute the Phase 21 Bayesian selection pipeline cold-start on the cluster to produce the full empirical artifact set (LOO/stacking/BMS/forest plots/winner-beta HDIs), sweep residual v4.0 tech debt, cross-verify reproducibility against the v4.0 baseline via a seeded regression test, and finalize `paper.qmd` so `quarto render` compiles `paper.pdf` with real winner names, real Pareto-k-informed limitations, and real Level-2 effect estimates.

---

## v1 Requirements (v5.0 scope)

21 requirements across 5 categories. Every requirement maps to exactly one phase (23-27).

### EXEC — Cluster Pipeline Execution

- [ ] **EXEC-01**: Cold-start `bash cluster/21_submit_pipeline.sh` from a clean working tree completes the 9-step `afterok` chain end-to-end with zero SLURM job failures; pre-flight pytest gate on `test_numpyro_models_2cov.py` passes before any `sbatch` call
- [ ] **EXEC-02**: Every expected artifact exists on disk after the cold-start run: `output/bayesian/21_prior_predictive/{model}_prior_sim.nc` (6 files), `21_recovery/{model}_recovery.csv` (6 files), `21_baseline/{model}_posterior.nc` (6 files) + `convergence_table.csv` + `convergence_report.md`, `21_l2/{winner}_posterior.nc` + `{winner}_beta_hdi_table.csv` (winner count from 21.5), `21_l2/scale_audit_report.md` + `averaged_scale_effects.csv`, `output/bayesian/manuscript/{loo_stacking,rfx_bms,winner_betas}.{csv,md,tex}`, and winner-specific forest plot PNGs
- [ ] **EXEC-03**: SLURM accounting (JobID, wall-clock, CPU-hours, GPU-hours, max memory) logged per step in `output/bayesian/21_execution_log.md`; total GPU-hours documented to enable future-budget planning
- [ ] **EXEC-04**: Convergence gate (Baribault & Collins 2023) passes for ≥ 2 models: `R-hat ≤ 1.05 AND ESS_bulk ≥ 400 AND divergences = 0 AND BFMI ≥ 0.2` verified via `convergence_table.csv`; step 21.4 exits 0; winner determination (step 21.5) is not FORCED (auto-determined via stacking weights)

### REPRO — Reproducibility Regression

- [ ] **REPRO-01**: New `scripts/fitting/tests/test_v5_reproducibility.py` exists with seeded regression test comparing step 21.1 prior-predictive output against v4.0 baseline NetCDF; same seed ⇒ byte-identical posterior samples; different seed ⇒ group posterior means within 3 SE of v4.0 reference
- [ ] **REPRO-02**: Step 21.2 Bayesian recovery regression: Pearson r for identifiable params (kappa, kappa_total, kappa_share) meets v4.0 thresholds (≥ 0.80); 95% HDI coverage for all models within 5 pp of v4.0 baseline values; regression failure triggers exit 1
- [ ] **REPRO-03**: `python validation/check_v4_closure.py --milestone v4.0` exits 0 on v5.0 HEAD; no v4.0 closure invariants broken by tech-debt sweep or new code in Phase 23-27
- [ ] **REPRO-04**: New `validation/check_v5_closure.py` implements v5.0 invariants (Phase 23-27 VERIFICATION.md files exist, REQUIREMENTS.md row count grows from 71 to ≥ 92, v5.0 entry in MILESTONES.md, EXEC artifacts exist on disk, manuscript `paper.pdf` exists); deterministic (`diff <(python ...) <(python ...)` empty); pytest regression `test_v5_closure.py` passes

### CLEAN — Dead-Code Sweep

- [x] **CLEAN-01**: Legacy qlearning hierarchical import path removed — if `scripts/fitting/legacy/qlearning_hierarchical_model.py` (or similar) exists, it is deleted; grep `from scripts.fitting.legacy` across `scripts/` returns zero live imports; pytest passes with the removal
- [x] **CLEAN-02**: Legacy M2 K-bounds [1,7] branch removed from `scripts/fitting/mle_utils.py`; only Collins [2,6] path remains; `parameterization_version` column vocabulary no longer accepts "legacy" value; affected tests updated
- [x] **CLEAN-03**: `scripts/16b_bayesian_regression.py` deleted entirely (superseded by Phase 18 schema-parity pipeline + Phase 21 hierarchical refits); `docs/MODEL_REFERENCE.md` cross-reference updated; cluster SLURM files referencing 16b removed or updated
- [x] **CLEAN-04**: Full load-side validation audit — every `az.from_netcdf` and `xr.open_dataset` call in `scripts/{15,16,17,18,21_*}.py` and `validation/*.py` uses `config.load_netcdf_with_validation` (new NetCDF companion to the existing CSV-only `config.load_fits_with_validation`); enforced via `scripts/fitting/tests/test_load_side_validation.py` grep invariant; no silent NetCDF corruption pathway remains. Phase 23 planning grep-audit identified that the existing `load_fits_with_validation` is CSV-only (validates `parameterization_version` column in DataFrame); the new NetCDF companion follows the same validate-then-return pattern for ArviZ `InferenceData` objects.

### MANU — Manuscript Finalization

- [ ] **MANU-01**: `paper.qmd` Methods `### Bayesian Model Selection Pipeline {#sec-bayesian-selection}` subsection cites real stacking weights from `reports/tables/model_comparison/table1_loo_stacking.csv` via Quarto `{python}` inline refs; placeholder "M6b received the highest stacking weight" text replaced with `{python} winner_display`
- [ ] **MANU-02**: Winner-specific forest plots generated via `scripts/06_fit_analyses/07_bayesian_level2_effects.py` for every model in the Phase 21.5 winner set; PNGs saved to `reports/figures/bayesian/21_bayesian/` and referenced in `paper.qmd` Results via `@fig-forest-{winner}` cross-refs
- [ ] **MANU-03**: Manuscript table artefacts exist in `reports/tables/model_comparison/`: `table1_loo_stacking.{csv,md,tex}` (stacking weights), `table2_rfx_bms.{csv,md,tex}` (PXP table), `table3_winner_betas.{csv,md,tex}` (winner L2 HDIs with model_averaged_* columns); `paper.qmd` `@tbl-loo-stacking`, `@tbl-rfx-bms`, `@tbl-winner-betas` Quarto cross-refs all resolve
- [ ] **MANU-04**: Limitations section of `paper.qmd` rewritten with real Pareto-k percentages from step 21.5 `loo_stacking_results.csv` `pct_high_pareto_k` column; removes projected-from-research/PITFALLS.md placeholder text; M4 Pareto-k fallback outcome (if M4 ever appears) reported factually
- [ ] **MANU-05**: `quarto render paper.qmd` succeeds from project root — exit 0, no Quarto errors, `_output/paper.pdf` and `_output/paper.html` exist; verified in `validation/check_v5_closure.py` via subprocess wrapper with a file-existence post-check

### CLOSE — Milestone Closure

- [ ] **CLOSE-01**: `.planning/MILESTONES.md` has a new "v5.0 Empirical Artifacts & Manuscript Finalization" entry at the top with ship date, phase range (23-27), plan count, git range (start-of-v5.0 → ship-commit), key accomplishments summary mirroring v4.0 entry format
- [ ] **CLOSE-02**: Archives at `.planning/milestones/v5.0-ROADMAP.md` + `.planning/milestones/v5.0-REQUIREMENTS.md` + `.planning/milestones/v5.0-MILESTONE-AUDIT.md`; top-level `ROADMAP.md` v5.0 section collapsed into `<details>` block matching v4.0 pattern; top-level `REQUIREMENTS.md` either replaced by v5.1 scope or carries a "v5.0 archived, awaiting v5.1 scope" header
- [ ] **CLOSE-03**: `/gsd:audit-milestone` re-run on v5.0 HEAD produces `status: passed` with no critical gaps; `tech_debt` list is empty (v4.0 residual debt resolved by CLEAN family) or explicitly documents v5.1 deferrals (Phase 14 K-refit / M4 GPU / ArviZ 1.0 migration) with links
- [ ] **CLOSE-04**: `validation/check_v5_closure.py` exits 0 on ship-commit; determinism confirmed via byte-identical double-run diff; `scripts/fitting/tests/test_v5_closure.py` passes 3/3 (happy path, determinism, rejects-wrong-milestone) matching v4.0 closure-guard test pattern

### REFAC — Repo Consolidation & Paper Scaffolding (Phase 28)

Phase 28 was inserted into v5.0 to consolidate the repo and scaffold `paper.qmd` to a Bayesian-first structure before the cold-start pipeline runs in Phase 24. Executes BEFORE Phase 24 (order: 23 → 28 → 24 → 25 → 26 → 27). Code-level consolidation only — no fit execution, no real-data population of manuscript.

- [x] **REFAC-01**: Delete `environments/` and `models/` top-level backward-compat shim packages outright (every file is a pure re-export from `rlwm.*`); update all 19 call sites in `scripts/simulations/`, `tests/`, `tests/examples/`, `validation/` to import directly from `rlwm.envs.*` and `rlwm.models.*`; `tests/test_rlwm_package.py` shim-specific test methods (lines 82–114) rewritten to test the canonical `rlwm.*` paths; `pip install -e .` documented in README as dev-setup prerequisite; pre-existing `tests/test_wmrl_exploration.py` collection error resolved as a side-effect; `grep -r "from environments\." scripts/ tests/ validation/` and `grep -r "from models\." scripts/ tests/ validation/` both return zero matches after this plan
- [x] **REFAC-02**: Narrow migration of pure-library fitting math from `scripts/fitting/` to `src/rlwm/fitting/` — move `jax_likelihoods.py`, `numpyro_models.py`, and `numpyro_helpers.py` via `git mv` to preserve history; update all ~22 call-site files (enumerated in Plan 28-01): top-level scripts (`scripts/21_run_bayesian_recovery.py`, `scripts/21_run_prior_predictive.py`, `scripts/21_fit_with_l2.py`); `scripts/fitting/` orchestrators (`fit_mle.py`, `fit_bayesian.py`, `bayesian_diagnostics.py`, `warmup_jit.py`); `scripts/utils/remap_mle_ids.py`; all 11 test files under `scripts/fitting/tests/` (`test_mle_quick`, `test_m3_hierarchical`, `test_pscan_likelihoods`, `test_numpyro_helpers`, `test_compile_gate`, `test_m4_hierarchical`, `test_m4_integration`, `test_numpyro_models_2cov`, `test_pointwise_loglik`, `test_prior_predictive`, `test_wmrl_model`); 2 validation files (`validation/benchmark_parallel_scan.py` — LOAD-BEARING for pscan SLURM jobs; `validation/test_m3_backward_compat.py`); orchestrators (`fit_mle.py`, `fit_bayesian.py`, `mle_utils.py`, `bms.py`, `model_recovery.py`, `bayesian_diagnostics.py`, `bayesian_summary_writer.py`, `lba_likelihood.py`, `level2_design.py`, `warmup_jit.py`, `aggregate_permutation_results.py`, `compare_mle_models.py`) stay in `scripts/fitting/`; `grep -r "from scripts.fitting.jax_likelihoods" .` and `grep -r "from scripts.fitting.numpyro_models" .` and `grep -r "from scripts.fitting.numpyro_helpers" .` all return zero matches across `scripts/`, `tests/`, `validation/`, `src/` post-migration
- [x] **REFAC-03**: Group data-processing scripts 01–04 under `scripts/data_processing/` — `git mv` `01_parse_raw_data.py`, `02_create_collated_csv.py`, `03_create_task_trials_csv.py`, `04_create_summary_csv.py` into the new directory; update any internal cross-imports; keep numeric prefixes for ordering; no SLURM/orchestrator references exist for these → no cluster updates needed
- [x] **REFAC-04**: Group behavioral-analysis scripts 05–08 under `scripts/behavioral/` — `git mv` `05_summarize_behavioral_data.py`, `06_visualize_task_performance.py`, `07_analyze_trauma_groups.py`, `08_run_statistical_analyses.py`; verify `08_run_statistical_analyses.py` internal `from scripts.utils.statistical_tests` import still resolves from new location
- [x] **REFAC-05**: Group simulations/recovery scripts 09–11 under `scripts/simulations_recovery/` — `git mv` `09_generate_synthetic_data.py`, `09_run_ppc.py`, `10_run_parameter_sweep.py`, `11_run_model_recovery.py`; update `cluster/09_ppc_gpu.slurm` and `cluster/11_recovery_gpu.slurm` python invocations; confirm 09/11 still import `scripts.fitting.model_recovery` correctly from new location
- [x] **REFAC-06**: Group post-MLE scripts 15–18 under `scripts/post_mle/` — `git mv` `15_analyze_mle_by_trauma.py`, `16_regress_parameters_on_scales.py`, `17_analyze_winner_heterogeneity.py`, `18_bayesian_level2_effects.py`; update `scripts/21_manuscript_tables.py` line 746 `subprocess.run(["python", "scripts/18_bayesian_level2_effects.py", ...])` to reference the new path (load-bearing invariant)
- [x] **REFAC-07**: Group Phase 21 Bayesian pipeline scripts under `scripts/bayesian_pipeline/` — `git mv` all nine `21_*.py` files; update the 9 corresponding SLURM job files in `cluster/21_*.slurm` to call `python scripts/bayesian_pipeline/21_*.py`; `cluster/21_submit_pipeline.sh` afterok chain structurally unchanged (it submits SLURM files, not python scripts directly); step 21.9 subprocess call to `18_bayesian_level2_effects.py` updated consistently with REFAC-06
- [x] **REFAC-08**: `figures/` + `output/` scaffolding — move clearly-legacy dirs (`output/v1/`, `output/_tmp_param_sweep/`, `output/_tmp_param_sweep_wmrl/`, `output/modelling_base_models/`, `output/base_model_analysis/`, `figures/v1/`, `figures/feedback_learning/`) to `output/legacy/` and `figures/legacy/` via `git mv`; pre-create empty scaffolding dirs that Phase 24 will populate with `.gitkeep`: `figures/21_bayesian/`, `output/bayesian/21_baseline/`, `output/bayesian/21_l2/`, `output/bayesian/21_recovery/`, `output/bayesian/21_prior_predictive/`, `output/bayesian/manuscript/`; KEEP in place: `output/mle/`, `output/model_comparison/`, `output/trauma_groups/`, `output/bayesian/` top-level (paper.qmd relative-path references preserved — `../output/mle`, `../output/model_comparison`, `../output/trauma_groups`, `../output/bayesian/level2`); zero path changes in `paper.qmd`
- [x] **REFAC-09**: Cluster consolidation — collapse six identical per-model templates (`cluster/13_bayesian_m{1,2,3,5,6a,6b}.slurm`) into one parameterized `cluster/13_bayesian_choice_only.slurm` accepting `--export=MODEL=<name>,TIME=<HH:MM:SS>` with `TIME=${TIME:-24:00:00}` default and M6b invoked with `TIME=36:00:00`; delete the six old templates; update `cluster/21_submit_pipeline.sh` step 21.3 loop and any other submissions referencing the deleted templates; retain all 7 specialized templates (`13_bayesian_m6b_subscale.slurm`, `13_bayesian_gpu.slurm`, `13_bayesian_multigpu.slurm`, `13_bayesian_permutation.slurm`, `13_bayesian_pscan.slurm`, `13_bayesian_pscan_smoke.slurm`, `13_bayesian_fullybatched_smoke.slurm`); net template count drops from 13 to 8
- [x] **REFAC-10**: `validation/` + `tests/` pruning — delete `validation/test_fitting_quick.py` (self-skips as legacy); `git mv` `validation/check_phase_23_1_smoke.py` and `validation/diagnose_gpu.py` to `validation/legacy/`; `git mv` `tests/examples/` (5 files) to `tests/legacy/examples/`; update import statements in the kept validation/tests files (`validation/test_model_consistency.py`, `validation/test_parameter_recovery.py`, `validation/test_unified_simulator.py`, `tests/test_wmrl_exploration.py`) to use `rlwm.*` paths directly (coordinated with REFAC-01)
- [x] **REFAC-11**: `paper.qmd` Bayesian-first structural scaffolding — reorder the Results section so Bayesian model selection comes FIRST (currently buried at `#sec-bayesian-selection` line 979), followed by hierarchical Level-2 trauma regression, subscale breakdown, then MLE validation + recovery + Bayesian↔MLE scatter moved to Appendix; add graceful-fallback `{python}` code cells for `@tbl-loo-stacking`, `@tbl-rfx-bms`, `@fig-forest-21`, `@tbl-winner-betas` modeled after the existing `#fig-l2-forest` cell (lines 1019–1047) that checks file existence and renders a placeholder if missing; `quarto render manuscript/paper.qmd` from project root exits 0 and produces `manuscript/_output/paper.pdf` with placeholders in correct slots; no path changes to existing `../output/*` refs
- [x] **REFAC-12**: Docs refresh — update project-root `CLAUDE.md` "Code Organization" and "Quick Reference" sections to reflect new grouped script layout (`scripts/data_processing/`, `scripts/behavioral/`, `scripts/simulations_recovery/`, `scripts/post_mle/`, `scripts/bayesian_pipeline/`, `src/rlwm/fitting/`); update `README.md` pipeline block likewise; add `pip install -e .` to dev-setup prerequisites per REFAC-01; update `docs/` cross-references that mention deleted `environments/` or `models/` shim paths
- [x] **REFAC-13**: End-of-phase verification and grep-audit — `pytest` full suite passes (204 tests, 1 pre-existing collection error in `tests/test_wmrl_exploration.py` resolved, zero new failures); `pytest scripts/fitting/tests/test_v4_closure.py` still passes 3/3; `python validation/check_v4_closure.py --milestone v4.0` exits 0; grep invariants: `grep -r "from environments\." scripts/ tests/ validation/` zero matches, `grep -r "from models\." scripts/ tests/ validation/` zero matches, `grep -r "from scripts.fitting.jax_likelihoods" .` zero matches, `grep -r "from scripts.fitting.numpyro_models" .` zero matches, `grep -r "from scripts.fitting.numpyro_helpers" .` zero matches; `quarto render manuscript/paper.qmd` exits 0

### REFAC — Pipeline Canonical Reorganization (Phase 29)

Phase 29 was inserted into v5.0 to promote Phase 28's five-subdir grouping (`data_processing/`, `behavioral/`, `simulations_recovery/`, `post_mle/`, `bayesian_pipeline/`) into the paper-directional numbered 01–06 stage layout, consolidate cluster SLURMs into stage-numbered entry points, merge orphan docs into structured methods references, and pin the canonical shape with a pytest invariant. Executes as the final structural refactor before v5.0 closure.

- [x] **REFAC-14**: Scripts canonical reorganization — move grouped Phase-28 folders (`data_processing/`, `behavioral/`, `simulations_recovery/`, `post_mle/`, `bayesian_pipeline/`) and top-level entry scripts (`12_fit_mle.py`, `13_fit_bayesian.py`, `14_compare_models.py`) into canonical paper-directional 01–06 stage folders via `git mv` (history preserved); intra-stage renumbering under Scheme D per 29-04b (stages 01/02/03/05/06 use 01-N reset-per-stage numbering; stage 04's parallel-alternative subfolders `a_mle/`, `b_bayesian/`, `c_level2/` use canonical descriptive CLI names with library code under underscore-private `_engine.py`); update all importers across `scripts/`, `tests/`, `validation/`, `cluster/`, `manuscript/`, `docs/`, `src/` (27+ external files across the four atomic refactor waves 29-01 / 29-04 / 29-04b / 29-03); `grep -rn "from scripts\.(data_processing|behavioral|simulations_recovery|post_mle|bayesian_pipeline)\." scripts/ tests/ validation/ src/` returns zero matches outside `scripts/legacy/` and historical `.planning/`; `pytest scripts/fitting/tests/test_v4_closure.py` still passes 3/3; `python validation/check_v4_closure.py --milestone v4.0` exits 0

- [x] **REFAC-15**: Docs spare-file integration — merge `docs/HIERARCHICAL_BAYESIAN.md` → `docs/04_methods/README.md#hierarchical-bayesian-architecture`, `docs/K_PARAMETERIZATION.md` → `docs/03_methods_reference/MODEL_REFERENCE.md#k-parameterization`, and `docs/SCALES_AND_FITTING_AUDIT.md` → `docs/04_methods/README.md#scales-orthogonalization-and-audit`; `git mv` originals to `docs/legacy/`; update `manuscript/paper.qmd` line 166 caption reference to the merged SCALES location (deferred to Plan 29-06 to avoid Wave 1 parallel-write race); `docs/CLUSTER_GPU_LESSONS.md` byte-identical to Phase-29-snapshot content (enforced via committed `pre_phase29_cluster_gpu_lessons.sha256` hash manifest at repo root and sha256 invariant test `test_cluster_gpu_lessons_untouched`); `docs/PARALLEL_SCAN_LIKELIHOOD.md` left in place per CONTEXT.md user directive

- [x] **REFAC-16**: Utils consolidation — extract PPC simulator logic into canonical single-source `scripts/utils/ppc.py` (935 lines, 11 top-level defs: `simulate_from_samples` + `run_prior_ppc` + `run_posterior_ppc` + 8 private helpers) consumed by stage 03 prior-PPC (`12_run_prior_predictive.py`), stage 03 posterior-PPC (`09_run_ppc.py` deleted by 29-04b as duplicate), and stage 05 posterior-PPC (`scripts/05_post_fitting_checks/run_posterior_ppc.py`); rename `scripts/utils/plotting_utils.py` → `plotting.py`, `statistical_tests.py` → `stats.py`, `scoring_functions.py` → `scoring.py` via `git mv`; create `scripts/utils/__init__.py` package marker with module-map docstring; move `remap_mle_ids.py`, `sync_experiment_data.py`, `update_participant_mapping.py` to `scripts/_maintenance/` (zero-importer rule); `grep -rn "def run_prior_ppc\|def run_posterior_ppc\|def simulate_from_samples" scripts/ --include="*.py"` shows definitions ONLY in `scripts/utils/ppc.py` (outside `scripts/legacy/`); canonical short-name imports resolve: `from scripts.utils import ppc, plotting, stats, scoring` succeeds

- [x] **REFAC-17**: Dead-folder audit and cleanup — per-folder grep audit (excluding `.planning/` historical docs) documented in `scripts/legacy/README.md` with live-reference counts; `scripts/analysis/` (9 files), `scripts/results/` (5 files), `scripts/simulations/` (5 files), `scripts/statistical_analyses/` (1 file), `scripts/visualization/` (11 files) archived to `scripts/legacy/<folder>/` via whole-folder `git mv` (history preserved, 31 total files archived); 7 live importers rewritten to `from scripts.legacy.simulations.*` (`validation/test_unified_simulator.py`, `tests/test_wmrl_exploration.py`, 2 × `scripts/03_model_prefitting/` wrappers, 3 × `tests/legacy/examples/*`); `manuscript/paper.qmd` line 171 + `paper.tex` line 244 caption paths rewritten; `scripts/` top level contains only `01_data_preprocessing/`, `02_behav_analyses/`, `03_model_prefitting/`, `04_model_fitting/`, `05_post_fitting_checks/`, `06_fit_analyses/`, `utils/`, `fitting/` library remnant, `legacy/`, `_maintenance/`

- [x] **REFAC-18**: Cluster SLURM consolidation — update every `cluster/*.slurm` internal `python scripts/...` path to canonical 01–06 locations; create stage-numbered entry points (`cluster/01_data_preprocessing.slurm`, `02_behav_analyses.slurm`, `03_model_prefitting_{cpu,gpu}.slurm`, `04a_mle_{cpu,gpu}.slurm`, `04b_bayesian_{cpu,gpu}.slurm`, `04c_level2.slurm`, `05_post_fitting_checks.slurm`, `06_fit_analyses.slurm`); consolidate per-model templates (fold M6b subscale into `04b_bayesian_cpu.slurm` via `--export=SUBSCALE=1`; fold M4 LBA into `04b_bayesian_gpu.slurm` via `--export=MODEL=wmrl_m4` + `--time=48:00:00 --mem=96G --gres=gpu:a100:1` overrides); parameterized `STEP=` dispatch inside stage-03/05/06 SLURM bodies (case-statement routing); `cluster/submit_all.sh` master orchestrator chains all six stages via `sbatch --parsable` + `--dependency=afterok` with `--dry-run` path-validation mode; `cluster/21_submit_pipeline.sh` rewritten as one-line shim `exec bash cluster/submit_all.sh "$@"` (preserves v4.0 user-memory invocation); `bash -n cluster/*.slurm` exits 0 for every SLURM; `bash cluster/submit_all.sh --dry-run` exits 0 with all python paths resolving

- [x] **REFAC-19**: Paper.qmd + paper.tex script-path updates — rewrite every `scripts/`-prefixed path reference in `manuscript/paper.qmd` and `manuscript/paper.tex` to the new canonical 01–06 layout; grep for stale grouping-dir and top-level fit-stage patterns (`scripts/{data_processing,behavioral,simulations_recovery,post_mle,bayesian_pipeline,12_fit_mle,13_fit_bayesian,14_compare_models}`) returns zero matches (script-path rewrites absorbed by 29-04b commit 093d934 + 29-06 commit 2b26df0); `quarto render manuscript/paper.qmd` exits 0 producing `manuscript/_output/paper.pdf` (~1.04 MB); 20 graceful-fallback `{python}` cells (Phase 28 pattern) absorb missing cold-start artifacts cleanly; 5 pre-existing BibTeX warnings for missing refs (phan2019composable, hoffman2014no, ahn2017revealing, vehtari2017practical, yao2018using) deferred to Phase 26

- [x] **REFAC-20**: Phase 29 closure-guard extension — new pytest `tests/test_v5_phase29_structure.py` asserts the canonical structure shape with 31 parametrize cases across 8 test functions (6 stage folders present; 04_model_fitting/{a,b,c} sub-letters present; 10 dead folders absent from top level; `scripts/utils/ppc.py` has ≥ 2 function definitions; simulator defined ONLY in utils outside `scripts/legacy/`; 3 docs spare files merged and originals live under `docs/legacy/`; `docs/CLUSTER_GPU_LESSONS.md` sha256 matches `pre_phase29_cluster_gpu_lessons.sha256` manifest at repo root; zero `from scripts.{data_processing,behavioral,simulations_recovery,post_mle,bayesian_pipeline}` imports in active tree excluding `/legacy/`; utils canonical short names `plotting.py`/`stats.py`/`scoring.py` present, verbose names `plotting_utils.py`/`statistical_tests.py`/`scoring_functions.py` absent); `pytest tests/test_v5_phase29_structure.py -v` 31/31 PASS; `python validation/check_v4_closure.py --milestone v4.0` still exits 0; `pytest scripts/fitting/tests/test_v4_closure.py -v` still 3/3 PASS; `.planning/phases/29-pipeline-canonical-reorg/29-VERIFICATION.md` filled in with SC-evidence table + plan-level status=pass

---

## Future Requirements (v5.1+)

Deferred to later milestones; tracked but not in v5.0 scope.

### Phase 14 Cluster Execution (v4.0 carry-over)

- **K-02**: Implement constrained K bounds in `mle_utils.py` (cluster refit via `bash cluster/12_submit_all_gpu.sh` — cold start)
- **K-03**: Refit all 7 models via MLE with constrained K; `parameterization_version = "collins_k_v1"`; K recovery r ≥ 0.50 floor
- **GPU-01**: `fit_all_gpu_m4` function for M4 synthetic recovery path
- **GPU-02**: `fit_all_gpu_m4` for real-data M4 fitting
- **GPU-03**: M4 real-data fit on A100 completes in < 12h wall-clock

### Downstream Reliability (enabled by Phase 14 completion)

- **MIG-06**: MLE-vs-Bayesian reliability scatterplots regenerated with constrained-K MLE fits (currently show stale [1,7] K recovery at r = 0.21)

### Infrastructure Upgrades

- **ARVIZ1-01**: ArviZ 1.0 migration (`InferenceData` → `xarray.DataTree`)
- **WORKFLOW-01**: Simulation-based calibration (SBC) as standard pre-fit validation (Talts et al. 2018; Säilynoja et al. 2022)
- **HORSESHOE-01**: Regularized horseshoe prior as default on all Level-2 coefficient families (Piironen & Vehtari 2017)
- **PMWG-01**: Full PMwG-equivalent hierarchical LBA via Python port or subprocess to R `pmwg`

### New Cognitive Models

- **M7-01**: M7 (split `phi_WM` / `phi_RL` forgetting rates)
- **M8-01**: M8-ASYMBIAS (Senta et al. 2025 winning mechanism on our cohort)
- **M9-01**: M9-SPLIT-RHO (conditional ρ on capacity-exceedance)

---

## Out of Scope

Explicit exclusions. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Running fits on user's local machine | All cluster-heavy compute stays on Monash M3 GPU cluster; infrastructure + orchestration only |
| Rewriting scripts 15/16/17 analysis logic | Schema-parity contract (`--source mle|bayesian` flag) is load-bearing — logic changes belong in a behavioral-analysis milestone, not v5.0 |
| Phase 14 cluster execution | Explicitly deferred to v5.1 per /gsd:new-milestone scope decision — user prefers clean Phase-21-only narrative for v5.0 |
| Mobile / web UI | Project is a research pipeline, not an application |
| Retrofitting v3.0 M4 joint choice+RT track to use Phase 21 comparison framework | M4 is separate by design (Pareto-k pathology — Pitfall 5); any joint comparison rewrite is a new milestone |
| New cognitive models (M7-M9) | v5.0 is about executing existing infrastructure, not extending the model family |
| DDM as alternative to LBA | Task uses 3 choices; LBA is correct framework — decision locked in v3.0 |

---

## Traceability

Each requirement maps to exactly one phase. Populated by `gsd-roadmapper` during roadmap creation (Phase 9 of /gsd:new-milestone flow).

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLEAN-01 | Phase 23 | Complete |
| CLEAN-02 | Phase 23 | Complete |
| CLEAN-03 | Phase 23 | Complete |
| CLEAN-04 | Phase 23 | Complete |
| EXEC-01 | Phase 24 | Pending |
| EXEC-02 | Phase 24 | Pending |
| EXEC-03 | Phase 24 | Pending |
| EXEC-04 | Phase 24 | Pending |
| REPRO-01 | Phase 25 | Pending |
| REPRO-02 | Phase 25 | Pending |
| REPRO-03 | Phase 25 | Pending |
| REPRO-04 | Phase 25 | Pending |
| MANU-01 | Phase 26 | Pending |
| MANU-02 | Phase 26 | Pending |
| MANU-03 | Phase 26 | Pending |
| MANU-04 | Phase 26 | Pending |
| MANU-05 | Phase 26 | Pending |
| CLOSE-01 | Phase 27 | Pending |
| CLOSE-02 | Phase 27 | Pending |
| CLOSE-03 | Phase 27 | Pending |
| CLOSE-04 | Phase 27 | Pending |
| REFAC-01 | Phase 28 | Complete |
| REFAC-02 | Phase 28 | Complete |
| REFAC-03 | Phase 28 | Complete |
| REFAC-04 | Phase 28 | Complete |
| REFAC-05 | Phase 28 | Complete |
| REFAC-06 | Phase 28 | Complete |
| REFAC-07 | Phase 28 | Complete |
| REFAC-08 | Phase 28 | Complete |
| REFAC-09 | Phase 28 | Complete |
| REFAC-10 | Phase 28 | Complete |
| REFAC-11 | Phase 28 | Complete |
| REFAC-12 | Phase 28 | Complete |
| REFAC-13 | Phase 28 | Complete |
| REFAC-14 | Phase 29 | Complete |
| REFAC-15 | Phase 29 | Complete |
| REFAC-16 | Phase 29 | Complete |
| REFAC-17 | Phase 29 | Complete |
| REFAC-18 | Phase 29 | Complete |
| REFAC-19 | Phase 29 | Complete |
| REFAC-20 | Phase 29 | Complete |

**Coverage:**
- v5.0 requirements: 41 total (21 original + 13 REFAC Phase-28 + 7 REFAC Phase-29)
- Mapped to phases: 41
- Unmapped: 0 ✓

**Per-phase coverage:**
- Phase 23 (Tech-Debt Sweep): 4 requirements (CLEAN-01..04)
- Phase 24 (Cold-Start Execution): 4 requirements (EXEC-01..04)
- Phase 25 (Reproducibility Regression): 4 requirements (REPRO-01..04)
- Phase 26 (Manuscript Finalization): 5 requirements (MANU-01..05)
- Phase 27 (Milestone Closure): 4 requirements (CLOSE-01..04)
- Phase 28 (Repo Consolidation & Paper Scaffolding): 13 requirements (REFAC-01..13) — executes before Phase 24 per Option A-modified sequencing
- Phase 29 (Pipeline Canonical Reorganization): 7 requirements (REFAC-14..20) — executes after Phase 28 as final structural refactor before v5.0 closure

---

## Scope Decisions (from /gsd:new-milestone questioning on 2026-04-19)

1. **Endpoint = Manuscript-ready** — v5.0 closes only when `paper.qmd` auto-patched + forest plots/tables exist + limitations rewritten + `quarto render` passes. See MANU-01..MANU-05.
2. **Orchestrator = Phase 21 only** — Phase 14 (K-refit + M4 GPU) deferred to v5.1. The manuscript's primary Bayesian narrative is independent of MLE K-refit. See Deferred section.
3. **Dead-code sweep = all 4 items** — Legacy qlearning import, legacy M2 K-bounds [1,7], `scripts/16b_bayesian_regression.py`, full load-side validation audit. See CLEAN-01..CLEAN-04.
4. **Cross-verify = prior-predictive + recovery regression** — Seeded regression test against v4.0 baseline artifacts. See REPRO-01..REPRO-02. Dropped: posterior-mean replication twice-run protocol, stacking-weight stability twice-run protocol, PXP sanity check (all add cluster cost without proportional scientific value for a v4.0-code-identical refit).

---

*Requirements defined: 2026-04-19*
*Last updated: 2026-04-19 — initial definition*
