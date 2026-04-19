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

- [ ] **CLEAN-01**: Legacy qlearning hierarchical import path removed — if `scripts/fitting/legacy/qlearning_hierarchical_model.py` (or similar) exists, it is deleted; grep `from scripts.fitting.legacy` across `scripts/` returns zero live imports; pytest passes with the removal
- [ ] **CLEAN-02**: Legacy M2 K-bounds [1,7] branch removed from `scripts/fitting/mle_utils.py`; only Collins [2,6] path remains; `parameterization_version` column vocabulary no longer accepts "legacy" value; affected tests updated
- [ ] **CLEAN-03**: `scripts/16b_bayesian_regression.py` deleted entirely (superseded by Phase 18 schema-parity pipeline + Phase 21 hierarchical refits); `docs/MODEL_REFERENCE.md` cross-reference updated; cluster SLURM files referencing 16b removed or updated
- [ ] **CLEAN-04**: Full load-side validation audit — every `az.from_netcdf` and `xr.open_dataset` call in `scripts/{15,16,17,18,21_*}.py` and `validation/*.py` uses `config.load_fits_with_validation` wrapper; enforced via `scripts/fitting/tests/test_load_side_validation.py` grep invariant; no silent NetCDF corruption pathway remains

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
| CLEAN-01 | Phase 23 | Pending |
| CLEAN-02 | Phase 23 | Pending |
| CLEAN-03 | Phase 23 | Pending |
| CLEAN-04 | Phase 23 | Pending |
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

**Coverage:**
- v5.0 requirements: 21 total
- Mapped to phases: 21
- Unmapped: 0 ✓

**Per-phase coverage:**
- Phase 23 (Tech-Debt Sweep): 4 requirements (CLEAN-01..04)
- Phase 24 (Cold-Start Execution): 4 requirements (EXEC-01..04)
- Phase 25 (Reproducibility Regression): 4 requirements (REPRO-01..04)
- Phase 26 (Manuscript Finalization): 5 requirements (MANU-01..05)
- Phase 27 (Milestone Closure): 4 requirements (CLOSE-01..04)

---

## Scope Decisions (from /gsd:new-milestone questioning on 2026-04-19)

1. **Endpoint = Manuscript-ready** — v5.0 closes only when `paper.qmd` auto-patched + forest plots/tables exist + limitations rewritten + `quarto render` passes. See MANU-01..MANU-05.
2. **Orchestrator = Phase 21 only** — Phase 14 (K-refit + M4 GPU) deferred to v5.1. The manuscript's primary Bayesian narrative is independent of MLE K-refit. See Deferred section.
3. **Dead-code sweep = all 4 items** — Legacy qlearning import, legacy M2 K-bounds [1,7], `scripts/16b_bayesian_regression.py`, full load-side validation audit. See CLEAN-01..CLEAN-04.
4. **Cross-verify = prior-predictive + recovery regression** — Seeded regression test against v4.0 baseline artifacts. See REPRO-01..REPRO-02. Dropped: posterior-mean replication twice-run protocol, stacking-weight stability twice-run protocol, PXP sanity check (all add cluster cost without proportional scientific value for a v4.0-code-identical refit).

---

*Requirements defined: 2026-04-19*
*Last updated: 2026-04-19 — initial definition*
