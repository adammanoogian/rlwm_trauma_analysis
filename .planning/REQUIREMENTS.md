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

- [ ] **MANU-01**: `paper.qmd` Methods `### Bayesian Model Selection Pipeline {#sec-bayesian-selection}` subsection cites real stacking weights from `output/bayesian/manuscript/loo_stacking_results.csv` via Quarto `{python}` inline refs; placeholder "M6b received the highest stacking weight" text replaced with `{python} winner_display`
- [ ] **MANU-02**: Winner-specific forest plots generated via `scripts/18_bayesian_level2_effects.py` for every model in the Phase 21.5 winner set; PNGs saved to `output/bayesian/figures/` and referenced in `paper.qmd` Results via `@fig-forest-{winner}` cross-refs
- [ ] **MANU-03**: Manuscript table artefacts exist in `output/bayesian/manuscript/`: `loo_stacking.{csv,md,tex}` (stacking weights), `rfx_bms.{csv,md,tex}` (PXP table), `winner_betas.{csv,md,tex}` (winner L2 HDIs with model_averaged_* columns); `paper.qmd` `@tbl-loo-stacking`, `@tbl-rfx-bms`, `@tbl-winner-betas` Quarto cross-refs all resolve
- [ ] **MANU-04**: Limitations section of `paper.qmd` rewritten with real Pareto-k percentages from step 21.5 `loo_stacking_results.csv` `pct_high_pareto_k` column; removes projected-from-research/PITFALLS.md placeholder text; M4 Pareto-k fallback outcome (if M4 ever appears) reported factually
- [ ] **MANU-05**: `quarto render paper.qmd` succeeds from project root — exit 0, no Quarto errors, `_output/paper.pdf` and `_output/paper.html` exist; verified in `validation/check_v5_closure.py` via subprocess wrapper with a file-existence post-check

### CLOSE — Milestone Closure

- [ ] **CLOSE-01**: `.planning/MILESTONES.md` has a new "v5.0 Empirical Artifacts & Manuscript Finalization" entry at the top with ship date, phase range (23-27), plan count, git range (start-of-v5.0 → ship-commit), key accomplishments summary mirroring v4.0 entry format
- [ ] **CLOSE-02**: Archives at `.planning/milestones/v5.0-ROADMAP.md` + `.planning/milestones/v5.0-REQUIREMENTS.md` + `.planning/milestones/v5.0-MILESTONE-AUDIT.md`; top-level `ROADMAP.md` v5.0 section collapsed into `<details>` block matching v4.0 pattern; top-level `REQUIREMENTS.md` either replaced by v5.1 scope or carries a "v5.0 archived, awaiting v5.1 scope" header
- [ ] **CLOSE-03**: `/gsd:audit-milestone` re-run on v5.0 HEAD produces `status: passed` with no critical gaps; `tech_debt` list is empty (v4.0 residual debt resolved by CLEAN family) or explicitly documents v5.1 deferrals (Phase 14 K-refit / M4 GPU / ArviZ 1.0 migration) with links
- [ ] **CLOSE-04**: `validation/check_v5_closure.py` exits 0 on ship-commit; determinism confirmed via byte-identical double-run diff; `scripts/fitting/tests/test_v5_closure.py` passes 3/3 (happy path, determinism, rejects-wrong-milestone) matching v4.0 closure-guard test pattern

### REFAC — Repo Consolidation & Paper Scaffolding (Phase 28)

Phase 28 was inserted into v5.0 to consolidate the repo and scaffold `paper.qmd` to a Bayesian-first structure before the cold-start pipeline runs in Phase 24. Executes BEFORE Phase 24 (order: 23 → 28 → 24 → 25 → 26 → 27). Code-level consolidation only — no fit execution, no real-data population of manuscript.

- [ ] **REFAC-01**: Delete `environments/` and `models/` top-level backward-compat shim packages outright (every file is a pure re-export from `rlwm.*`); update all 19 call sites in `scripts/simulations/`, `tests/`, `tests/examples/`, `validation/` to import directly from `rlwm.envs.*` and `rlwm.models.*`; `tests/test_rlwm_package.py` shim-specific test methods (lines 82–114) rewritten to test the canonical `rlwm.*` paths; `pip install -e .` documented in README as dev-setup prerequisite; pre-existing `tests/test_wmrl_exploration.py` collection error resolved as a side-effect; `grep -r "from environments\." scripts/ tests/ validation/` and `grep -r "from models\." scripts/ tests/ validation/` both return zero matches after this plan
- [ ] **REFAC-02**: Narrow migration of pure-library fitting math from `scripts/fitting/` to `src/rlwm/fitting/` — move `jax_likelihoods.py`, `numpyro_models.py`, and `numpyro_helpers.py` via `git mv` to preserve history; update all ~22 call-site files (enumerated in Plan 28-01): top-level scripts (`scripts/21_run_bayesian_recovery.py`, `scripts/21_run_prior_predictive.py`, `scripts/21_fit_with_l2.py`); `scripts/fitting/` orchestrators (`fit_mle.py`, `fit_bayesian.py`, `bayesian_diagnostics.py`, `warmup_jit.py`); `scripts/utils/remap_mle_ids.py`; all 11 test files under `scripts/fitting/tests/` (`test_mle_quick`, `test_m3_hierarchical`, `test_pscan_likelihoods`, `test_numpyro_helpers`, `test_compile_gate`, `test_m4_hierarchical`, `test_m4_integration`, `test_numpyro_models_2cov`, `test_pointwise_loglik`, `test_prior_predictive`, `test_wmrl_model`); 2 validation files (`validation/benchmark_parallel_scan.py` — LOAD-BEARING for pscan SLURM jobs; `validation/test_m3_backward_compat.py`); orchestrators (`fit_mle.py`, `fit_bayesian.py`, `mle_utils.py`, `bms.py`, `model_recovery.py`, `bayesian_diagnostics.py`, `bayesian_summary_writer.py`, `lba_likelihood.py`, `level2_design.py`, `warmup_jit.py`, `aggregate_permutation_results.py`, `compare_mle_models.py`) stay in `scripts/fitting/`; `grep -r "from scripts.fitting.jax_likelihoods" .` and `grep -r "from scripts.fitting.numpyro_models" .` and `grep -r "from scripts.fitting.numpyro_helpers" .` all return zero matches across `scripts/`, `tests/`, `validation/`, `src/` post-migration
- [ ] **REFAC-03**: Group data-processing scripts 01–04 under `scripts/data_processing/` — `git mv` `01_parse_raw_data.py`, `02_create_collated_csv.py`, `03_create_task_trials_csv.py`, `04_create_summary_csv.py` into the new directory; update any internal cross-imports; keep numeric prefixes for ordering; no SLURM/orchestrator references exist for these → no cluster updates needed
- [ ] **REFAC-04**: Group behavioral-analysis scripts 05–08 under `scripts/behavioral/` — `git mv` `05_summarize_behavioral_data.py`, `06_visualize_task_performance.py`, `07_analyze_trauma_groups.py`, `08_run_statistical_analyses.py`; verify `08_run_statistical_analyses.py` internal `from scripts.utils.statistical_tests` import still resolves from new location
- [ ] **REFAC-05**: Group simulations/recovery scripts 09–11 under `scripts/simulations_recovery/` — `git mv` `09_generate_synthetic_data.py`, `09_run_ppc.py`, `10_run_parameter_sweep.py`, `11_run_model_recovery.py`; update `cluster/09_ppc_gpu.slurm` and `cluster/11_recovery_gpu.slurm` python invocations; confirm 09/11 still import `scripts.fitting.model_recovery` correctly from new location
- [ ] **REFAC-06**: Group post-MLE scripts 15–18 under `scripts/post_mle/` — `git mv` `15_analyze_mle_by_trauma.py`, `16_regress_parameters_on_scales.py`, `17_analyze_winner_heterogeneity.py`, `18_bayesian_level2_effects.py`; update `scripts/21_manuscript_tables.py` line 746 `subprocess.run(["python", "scripts/18_bayesian_level2_effects.py", ...])` to reference the new path (load-bearing invariant)
- [ ] **REFAC-07**: Group Phase 21 Bayesian pipeline scripts under `scripts/bayesian_pipeline/` — `git mv` all nine `21_*.py` files; update the 9 corresponding SLURM job files in `cluster/21_*.slurm` to call `python scripts/bayesian_pipeline/21_*.py`; `cluster/21_submit_pipeline.sh` afterok chain structurally unchanged (it submits SLURM files, not python scripts directly); step 21.9 subprocess call to `18_bayesian_level2_effects.py` updated consistently with REFAC-06
- [ ] **REFAC-08**: `figures/` + `output/` scaffolding — move clearly-legacy dirs (`output/v1/`, `output/_tmp_param_sweep/`, `output/_tmp_param_sweep_wmrl/`, `output/modelling_base_models/`, `output/base_model_analysis/`, `figures/v1/`, `figures/feedback_learning/`) to `output/legacy/` and `figures/legacy/` via `git mv`; pre-create empty scaffolding dirs that Phase 24 will populate with `.gitkeep`: `figures/21_bayesian/`, `output/bayesian/21_baseline/`, `output/bayesian/21_l2/`, `output/bayesian/21_recovery/`, `output/bayesian/21_prior_predictive/`, `output/bayesian/manuscript/`; KEEP in place: `output/mle/`, `output/model_comparison/`, `output/trauma_groups/`, `output/bayesian/` top-level (paper.qmd relative-path references preserved — `../output/mle`, `../output/model_comparison`, `../output/trauma_groups`, `../output/bayesian/level2`); zero path changes in `paper.qmd`
- [ ] **REFAC-09**: Cluster consolidation — collapse six identical per-model templates (`cluster/13_bayesian_m{1,2,3,5,6a,6b}.slurm`) into one parameterized `cluster/13_bayesian_choice_only.slurm` accepting `--export=MODEL=<name>,TIME=<HH:MM:SS>` with `TIME=${TIME:-24:00:00}` default and M6b invoked with `TIME=36:00:00`; delete the six old templates; update `cluster/21_submit_pipeline.sh` step 21.3 loop and any other submissions referencing the deleted templates; retain all 7 specialized templates (`13_bayesian_m6b_subscale.slurm`, `13_bayesian_gpu.slurm`, `13_bayesian_multigpu.slurm`, `13_bayesian_permutation.slurm`, `13_bayesian_pscan.slurm`, `13_bayesian_pscan_smoke.slurm`, `13_bayesian_fullybatched_smoke.slurm`); net template count drops from 13 to 8
- [ ] **REFAC-10**: `validation/` + `tests/` pruning — delete `validation/test_fitting_quick.py` (self-skips as legacy); `git mv` `validation/check_phase_23_1_smoke.py` and `validation/diagnose_gpu.py` to `validation/legacy/`; `git mv` `tests/examples/` (5 files) to `tests/legacy/examples/`; update import statements in the kept validation/tests files (`validation/test_model_consistency.py`, `validation/test_parameter_recovery.py`, `validation/test_unified_simulator.py`, `tests/test_wmrl_exploration.py`) to use `rlwm.*` paths directly (coordinated with REFAC-01)
- [ ] **REFAC-11**: `paper.qmd` Bayesian-first structural scaffolding — reorder the Results section so Bayesian model selection comes FIRST (currently buried at `#sec-bayesian-selection` line 979), followed by hierarchical Level-2 trauma regression, subscale breakdown, then MLE validation + recovery + Bayesian↔MLE scatter moved to Appendix; add graceful-fallback `{python}` code cells for `@tbl-loo-stacking`, `@tbl-rfx-bms`, `@fig-forest-21`, `@tbl-winner-betas` modeled after the existing `#fig-l2-forest` cell (lines 1019–1047) that checks file existence and renders a placeholder if missing; `quarto render manuscript/paper.qmd` from project root exits 0 and produces `manuscript/_output/paper.pdf` with placeholders in correct slots; no path changes to existing `../output/*` refs
- [ ] **REFAC-12**: Docs refresh — update project-root `CLAUDE.md` "Code Organization" and "Quick Reference" sections to reflect new grouped script layout (`scripts/data_processing/`, `scripts/behavioral/`, `scripts/simulations_recovery/`, `scripts/post_mle/`, `scripts/bayesian_pipeline/`, `src/rlwm/fitting/`); update `README.md` pipeline block likewise; add `pip install -e .` to dev-setup prerequisites per REFAC-01; update `docs/` cross-references that mention deleted `environments/` or `models/` shim paths
- [ ] **REFAC-13**: End-of-phase verification and grep-audit — `pytest` full suite passes (204 tests, 1 pre-existing collection error in `tests/test_wmrl_exploration.py` resolved, zero new failures); `pytest scripts/fitting/tests/test_v4_closure.py` still passes 3/3; `python validation/check_v4_closure.py --milestone v4.0` exits 0; grep invariants: `grep -r "from environments\." scripts/ tests/ validation/` zero matches, `grep -r "from models\." scripts/ tests/ validation/` zero matches, `grep -r "from scripts.fitting.jax_likelihoods" .` zero matches, `grep -r "from scripts.fitting.numpyro_models" .` zero matches, `grep -r "from scripts.fitting.numpyro_helpers" .` zero matches; `quarto render manuscript/paper.qmd` exits 0

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
| REFAC-01 | Phase 28 | Pending |
| REFAC-02 | Phase 28 | Pending |
| REFAC-03 | Phase 28 | Pending |
| REFAC-04 | Phase 28 | Pending |
| REFAC-05 | Phase 28 | Pending |
| REFAC-06 | Phase 28 | Pending |
| REFAC-07 | Phase 28 | Pending |
| REFAC-08 | Phase 28 | Pending |
| REFAC-09 | Phase 28 | Pending |
| REFAC-10 | Phase 28 | Pending |
| REFAC-11 | Phase 28 | Pending |
| REFAC-12 | Phase 28 | Pending |
| REFAC-13 | Phase 28 | Pending |

**Coverage:**
- v5.0 requirements: 34 total (21 original + 13 REFAC additions)
- Mapped to phases: 34
- Unmapped: 0 ✓

**Per-phase coverage:**
- Phase 23 (Tech-Debt Sweep): 4 requirements (CLEAN-01..04)
- Phase 24 (Cold-Start Execution): 4 requirements (EXEC-01..04)
- Phase 25 (Reproducibility Regression): 4 requirements (REPRO-01..04)
- Phase 26 (Manuscript Finalization): 5 requirements (MANU-01..05)
- Phase 27 (Milestone Closure): 4 requirements (CLOSE-01..04)
- Phase 28 (Repo Consolidation & Paper Scaffolding): 13 requirements (REFAC-01..13) — executes before Phase 24 per Option A-modified sequencing

---

## Scope Decisions (from /gsd:new-milestone questioning on 2026-04-19)

1. **Endpoint = Manuscript-ready** — v5.0 closes only when `paper.qmd` auto-patched + forest plots/tables exist + limitations rewritten + `quarto render` passes. See MANU-01..MANU-05.
2. **Orchestrator = Phase 21 only** — Phase 14 (K-refit + M4 GPU) deferred to v5.1. The manuscript's primary Bayesian narrative is independent of MLE K-refit. See Deferred section.
3. **Dead-code sweep = all 4 items** — Legacy qlearning import, legacy M2 K-bounds [1,7], `scripts/16b_bayesian_regression.py`, full load-side validation audit. See CLEAN-01..CLEAN-04.
4. **Cross-verify = prior-predictive + recovery regression** — Seeded regression test against v4.0 baseline artifacts. See REPRO-01..REPRO-02. Dropped: posterior-mean replication twice-run protocol, stacking-weight stability twice-run protocol, PXP sanity check (all add cluster cost without proportional scientific value for a v4.0-code-identical refit).

---

*Requirements defined: 2026-04-19*
*Last updated: 2026-04-19 — initial definition*
