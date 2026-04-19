# Requirements: RLWM Trauma Analysis v4.0

**Defined:** 2026-04-11
**Milestone:** v4.0 — Hierarchical Bayesian Pipeline & LBA Acceleration
**Core Value:** Correctly dissociate perseverative responding from learning-rate effects (alpha-) to accurately identify whether post-reversal failures reflect motor perseveration or outcome insensitivity in trauma populations.

**Research inputs:** `.planning/research/SUMMARY.md` and the 4 underlying STACK/FEATURES/ARCHITECTURE/PITFALLS files.

**User scope decisions (2026-04-11):**
- **M4 hierarchical LBA IS in scope** (despite research recommendation to descope). Phase 17 is committed. Accept ~150-200 GPU-hour total budget and Pareto-k fallback.
- **P0 broken-import bug acknowledged.** Phase 13 Task 1.
- **Deferred defaults (override in-phase if needed):** IES-R subscales use IES-R total + Gram-Schmidt residualized subscales. PyMC dropped entirely; NumPyro-only backend.

---

## v1 Requirements (v4.0 Milestone)

Requirements map to phases 13-18, continuing from the v3.0 milestone which ended at Phase 12.

### INFRA — Infrastructure Repair & Scaffolding

- [x] **INFRA-01**: Fix P0 broken import in `scripts/fitting/fit_bayesian.py:43` by resurrecting `scripts/fitting/numpyro_models.py` at the canonical path (moved out of `legacy/`)
- [x] **INFRA-02**: Pin dependency versions in `pyproject.toml` and `environment_gpu.yml` (`numpyro==0.20.1`, `arviz==0.23.4`, add `netcdf4`, remove `pymc`)
- [x] **INFRA-03**: Create `scripts/fitting/bayesian_diagnostics.py` with `compute_pointwise_log_lik()` helper that vmaps over (chains × samples × participants) using per-trial log-prob versions of existing likelihoods — required so `az.waic`/`az.loo` work with `numpyro.factor`-based models
- [x] **INFRA-04**: Create `scripts/fitting/bayesian_summary_writer.py` that converts ArviZ `InferenceData` to schema-parity CSV (identical column names to MLE fits, plus `_hdi_low`, `_hdi_high`, `_sd` extras) at `output/bayesian/<model>_individual_fits.csv`
- [x] **INFRA-05**: Build non-centered parameterization helper module (`scripts/fitting/numpyro_helpers.py`) with unit tests recovering known parameters from simulated data for every bounded transform (logit for [0,1], sigmoid with rescaling for K, softplus+sigmoid for stick-breaking)
- [x] **INFRA-06**: Add `parameterization_version` column convention to all fit output CSVs; downstream scripts validate on load and fail-loud on mismatch
- [x] **INFRA-07**: Drop PyMC backend entirely from `scripts/16b_bayesian_regression.py`; retain NumPyro-only path as deprecated fast-preview
- [x] **INFRA-08**: Configure `JAX_COMPILATION_CACHE_DIR` on cluster scratch in `cluster/13_bayesian_gpu.slurm`; add compile-time benchmark CI test (< 60s gate)

### K — Collins K Parameterization Research & Refit

- [x] **K-01**: Research how Collins 2012/2014/2018 parameterize WM capacity K in RLWM (free continuous vs bounded by max set-size vs fixed to set-size). Produce `docs/K_PARAMETERIZATION.md` with recommended convention
- [ ] **K-02**: Implement constrained K bounds in `mle_utils.py` for all 7 models based on K-01 findings (options include tighter continuous range, per-participant max_set_size bound, or fixed K = max set-size seen)
- [ ] **K-03**: Refit all 7 models via MLE with constrained K on the cluster; verify K recovery improves (target r ≥ 0.50 minimum, up from current r=0.21)

### GPU — LBA GPU Batching (goal #1 from milestone spec)

- [ ] **GPU-01**: Implement `fit_all_gpu_m4` function in `fit_mle.py` with float64 vmap for M4 synthetic (recovery) path; matches existing `fit_all_gpu` pattern for choice-only models but with lazy float64 enable
- [ ] **GPU-02**: Extend `fit_all_gpu_m4` to handle real-data M4 fitting across N=154 participants
- [ ] **GPU-03**: Verify GPU speedup: M4 parameter recovery wall time reduced from ~48h to < 12h on A100

### HIER — Hierarchical Model Implementations

- [ ] **HIER-01**: Implement M3 hierarchical NumPyro model with `vmap`-over-participants likelihood, non-centered parameterization on all bounded parameters, wrapped via `numpyro.factor`
- [ ] **HIER-02**: Port M1 (Q-learning) from legacy to new canonical `numpyro_models.py` with non-centered pattern (legacy uses centered for some params — rewrite)
- [ ] **HIER-03**: Port M2 (WM-RL) from legacy to new canonical `numpyro_models.py` with vectorized per-participant likelihood (legacy uses Python for-loop — rewrite)
- [ ] **HIER-04**: Implement M5 (WM-RL + phi_rl) hierarchical NumPyro model
- [ ] **HIER-05**: Implement M6a (WM-RL + kappa_s) hierarchical NumPyro model with per-stimulus carry structure
- [ ] **HIER-06**: Implement M6b (WM-RL + dual stick-breaking) hierarchical NumPyro model with unconstrained-space offset for `kappa_total`/`kappa_share` (non-centered on the unconstrained scale, decoded to constrained inside the likelihood)
- [ ] **HIER-07**: Convergence gate on all models: `R-hat ≤ 1.01`, `ESS_bulk ≥ 400`, zero divergences; auto-bump `target_accept_prob` from 0.8 → 0.95 → 0.99 on divergence failures; script refuses to write outputs if gate fails
- [ ] **HIER-08**: Shrinkage diagnostic report after every fit (`1 - var_post_individual / var_post_group`); warn if ratio < 0.3, refuse to treat parameter as "identified" if warning fires
- [ ] **HIER-09**: Posterior predictive check infrastructure stratified by trauma group; reproduce v3.0 M3 learning curves as validation
- [ ] **HIER-10**: Parametric dispatch smoke test — `pytest` fixture that loops over `ALL_MODELS` and smoke-fits each (5 subjects, 200 samples) in < 60s per model; prevents v3.0-style "forgot to add model to dispatch" bugs

### L2 — Level-2 Trauma Regression (goal #3 from milestone spec)

- [ ] **L2-01**: Joint Level-2 regression on single covariate (LEC-total → `kappa`) in M3 as proof-of-concept; reproduce v3.0's surviving-FDR finding under the new hierarchical framework
- [ ] **L2-02**: IES-R subscale correlation audit on real N=154 survey data (condition number, VIF, correlation heatmap) before locking orthogonalization
- [ ] **L2-03**: Orthogonalized subscale parameterization: IES-R total as one predictor + Gram-Schmidt residualized subscales (intrusion/avoidance/hyperarousal) as uncorrelated-by-construction additional predictors
- [ ] **L2-04**: LEC-5 subcategory Level-2 predictors (physical assault, sexual violence, accident, etc. per LEC-5 structure)
- [ ] **L2-05**: Full subscale Level-2 fit on winning model (currently M6b) with complete predictor set (~48 coefficients: 8 params × ~6 predictors)
- [ ] **L2-06**: Permutation null test for Level-2 effects — shuffle trauma labels, refit, verify no spurious surviving effects; catches shrinkage-masked artifacts
- [ ] **L2-07**: Forest plots of Level-2 effects via new `scripts/18_bayesian_level2_effects.py`
- [ ] **L2-08** *(P2, optional)*: Regularized horseshoe prior on M6b Level-2 coefficient family as alternative to `Normal(0, 0.5)` baseline — enable only if baseline has convergence issues or reviewers request aggressive multiplicity handling

### M4H — Hierarchical M4 LBA (goal #1 + user override of research recommendation)

- [ ] **M4H-01**: Implement M4 hierarchical NumPyro model with float64 process isolation (runs in separate Python process from choice-only models; `jax.config.update('jax_enable_x64', True)` + `numpyro.enable_x64()` BEFORE any JAX import)
- [ ] **M4H-02**: Non-centered `log(b - A)` parameterization for LBA threshold offset (avoids boundary funnel pathology)
- [ ] **M4H-03**: `chain_method='vectorized'` (vmap on single A100) with `target_accept_prob=0.95` and reduced sampling budget (`num_warmup=1000`, `num_samples=1500`) to fit 48h SLURM cap
- [ ] **M4H-04**: Checkpoint-and-resume via `mcmc.post_warmup_state`; integration test that kills a run and verifies resume works
- [ ] **M4H-05**: Pareto-k gating — immediately call `az.loo(idata, pointwise=True)` after M4 fit; if > 5% of trials have `k > 0.7`, fall back to choice-only marginal log-likelihood (marginalize over RT) for M4 vs choice-only model comparison
- [ ] **M4H-06**: Dedicated `cluster/13_bayesian_m4_gpu.slurm` with `--time=48:00:00`, `--mem=96G`, `--gres=gpu:a100:1` (NOT V100 — OOM risk for float64 × 154 subjects)

### CMP — Bayesian Model Comparison

- [x] **CMP-01**: Bayesian model comparison via `arviz.compare` with stacking weights across all 6 choice-only models (M1, M2, M3, M5, M6a, M6b)
- [x] **CMP-02**: M4 separate comparison track using choice-only marginal log-likelihood; NOT included in the unified choice-only `az.compare` table
- [x] **CMP-03**: WAIC + LOO reported as primary Bayesian criteria; AIC/BIC retained as backward-compat secondary metrics
- [x] **CMP-04**: Extend `scripts/14_compare_models.py` with `--bayesian-comparison` mode; keep MLE AIC/BIC mode as default for backward compat

### MIG — Pipeline Migration to Source-Flag Pattern

- [x] **MIG-01**: Extend `scripts/15_analyze_mle_by_trauma.py` with `--source mle|bayesian` flag; path resolution changes only, analysis logic unchanged (enabled by schema-parity CSV)
- [x] **MIG-02**: Extend `scripts/16_regress_parameters_on_scales.py` with `--source` flag
- [x] **MIG-03**: Extend `scripts/17_analyze_winner_heterogeneity.py` with `--source` flag
- [x] **MIG-04**: Freeze `scripts/16b_bayesian_regression.py` as deprecated (add deprecation docstring referencing the L2 hierarchical pipeline as replacement; keep runnable as fast-preview)
- [x] **MIG-05**: MLE-vs-Bayesian reliability scatterplots per parameter (new plot in scripts/17 or dedicated helper)

### DOC — Documentation & Manuscript

- [x] **DOC-01**: Update `docs/MODEL_REFERENCE.md` with a dedicated hierarchical Bayesian section (non-centered parameterization, Level-2 regression structure, NumPyro factor pattern, WAIC/LOO workflow)
- [x] **DOC-02**: Manuscript methods section — NumPyro/vmap/`numpyro.factor`/joint Level-2 regression/WAIC-LOO pipeline; replace MLE-centric methods narrative
- [x] **DOC-03**: Manuscript results section — posterior Level-2 effect forest plots replacing MLE post-hoc regression p-values; stacking-weight model comparison table
- [x] **DOC-04**: Manuscript limitations — Pareto-k fallback for M4, residual K recovery caveats if Phase 14 K refit doesn't fully resolve identifiability, M6b shrinkage diagnostics

### PSCAN — Associative Scan Likelihood Parallelization

- [x] **PSCAN-01**: Literature review document `docs/PARALLEL_SCAN_LIKELIHOOD.md` covering AR(1) linear recurrence formulation for Q-learning, PaMoRL PETE algorithm, Mamba/S4 parallel scan, parallel Kalman smoother lineage, and explicit RLWM linear vs non-linear component decomposition
- [x] **PSCAN-02**: `associative_scan_q_update()` function in `jax_likelihoods.py` computing all T Q-values in O(log T) via `jax.lax.associative_scan`; unit test verifying agreement with sequential `lax.scan` to < 1e-5 relative error on 1000-trial synthetic sequences
- [x] **PSCAN-03**: `associative_scan_wm_update()` function handling WM forgetting recurrence and hard overwrite via parallel scan; unit test verifying agreement with sequential implementation
- [x] **PSCAN-04**: `wmrl_m3_multiblock_likelihood_stacked_pscan()` and all 5 other choice-only model pscan variants; total log-likelihood agrees with sequential implementations to < 1e-4 relative error on all 154 real participants
- [x] **PSCAN-05**: GPU benchmark — M3 hierarchical fit with associative scan likelihood on A100 (4 chains, 1000 warmup, 2000 samples, N=154); wall clock, peak VRAM, and divergence count logged to `output/bayesian/pscan_benchmark.json`
- [x] **PSCAN-06**: A/B comparison — posterior parameter means from associative-scan GPU fit agree with sequential CPU fit to within MCMC noise (< 5% relative error on group-level means); WAIC/LOO agree to < 1.0

### DEER — Non-Linear Parallelization Research (Phase 20)

- [x] **DEER-01**: Research document `docs/DEER_NONLINEAR_PARALLELIZATION.md`
  (or equivalent section in `docs/PARALLEL_SCAN_LIKELIHOOD.md`) covering DEER
  fixed-point iteration with worked RLWM mixing non-linearity example, convergence
  analysis (is softmax-mixing a contraction mapping?), comparison with Picard/Newton
  alternatives, Unifying Framework (TMLR 2025) taxonomy placement, and explicit
  go/no-go recommendation with justification.
- [x] **DEER-02**: Go-path implementation `wmrl_m3_multiblock_likelihood_stacked_deer()`
  fully parallelizing all components; numerical agreement with sequential
  implementation to < 1e-3 relative error; benchmark shows additional speedup
  over Phase 19 scan+sequential hybrid. Or no-go documentation (Phase 20 outcome
  was no-go per 20-VERIFICATION.md; DEER-02 is satisfied by the documented
  no-go decision).
- [x] **DEER-03**: Empirical go/no-go evidence: either benchmark data showing
  DEER speedup OR convergence/speedup failure evidence showing why Phase 19
  hybrid remains the final GPU path.
- [x] **DEER-04**: Project-utils guide `JAX_GPU_BAYESIAN_FITTING.md` updated
  with DEER findings + reproducible benchmark script at
  `validation/benchmark_parallel_scan.py`.

### BMS — Bayesian Model Selection Pipeline (Phase 21, Baribault & Collins 2023 / Hess 2025 anchored)

- [x] **BMS-01**: Prior-predictive gate (Phase 21 step 21.1) — orchestrator
  `scripts/21_run_prior_predictive.py` + `cluster/21_1_prior_predictive.slurm`
  simulates choices from priors alone for all 6 choice-only models; accuracy
  curves cover plausible [0.4, 0.9] asymptote range; exit 1 on FAIL blocks
  downstream MCMC.
- [x] **BMS-02**: RFX-BMS + PXP module `scripts/fitting/bms.py` — per Rigoux
  et al. (2014) / Stephan et al. (2009); unit tests at
  `scripts/fitting/tests/test_bms.py`; usable from step 21.5 orchestrator.
- [x] **BMS-03**: Bayesian parameter recovery (step 21.2) —
  `scripts/21_run_bayesian_recovery.py` single-subject + aggregate CLI modes;
  SLURM array `cluster/21_2_recovery.slurm` (1-50) + aggregator
  `cluster/21_2_recovery_aggregate.slurm`; kappa-family pass criterion
  r >= 0.80 AND 95% HDI coverage >= 0.90.
- [x] **BMS-04**: Baseline hierarchical fit runner (step 21.3) —
  `scripts/21_fit_baseline.py` delegates to fit_bayesian.main() with forced
  `--output-subdir 21_baseline`; SLURM `cluster/21_3_fit_baseline.slurm`;
  load-bearing exit-1 shim converts silent gate-fail into SLURM failure.
- [x] **BMS-05**: Baseline convergence + PPC audit (step 21.4) —
  `scripts/21_baseline_audit.py` applies Baribault & Collins (2023) four-criterion
  gate (R-hat <= 1.05, ESS_bulk >= 400, divergences == 0, BFMI >= 0.2); exit 1
  when n_passing < 2 blocks step 21.5; SLURM `cluster/21_4_baseline_audit.slurm`.
- [x] **BMS-06**: PSIS-LOO + stacking + RFX-BMS winner gate (step 21.5) —
  `scripts/21_compute_loo_stacking.py` primary stacking weights (Yao 2018)
  + secondary rfx_bms (Rigoux 2014); soft Pareto-k gate; three-tier winner
  verdict (DOMINANT_SINGLE / TOP_TWO / INCONCLUSIVE_MULTIPLE); tri-state exit
  code; SLURM `cluster/21_5_loo_stacking_bms.slurm`.
- [x] **BMS-07**: Winner L2 refit orchestrator (step 21.6) —
  `scripts/21_fit_with_l2.py` three-branch dispatch (copy for M1/M2;
  2-cov for M3/M5/M6a; subscale for M6b); post-fit beta-site verification;
  SLURM `cluster/21_6_fit_with_l2.slurm` + dispatcher `cluster/21_dispatch_l2_winners.sh`
  + wrapper `cluster/21_6_dispatch_l2.slurm` (--time=14:00:00).
- [x] **BMS-08**: Scale-fit audit (step 21.7) — `scripts/21_scale_audit.py`
  pattern-match beta-site enumeration handles all three L2 tiers; longest-
  prefix-first covariate/target parser; FDR-BH per-winner; unified exit-0 for
  PROCEED_TO_AVERAGING and NULL_RESULT; SLURM `cluster/21_7_scale_audit.slurm`.
- [x] **BMS-09**: Stacking-weighted model averaging (step 21.8) —
  `scripts/21_model_averaging.py` canonical-key matching with _total suffix
  stripping (unifies 2-cov and subscale betas); three short-circuit paths
  (NULL_RESULT, single-winner, multi-winner averaging); optional M6b-subscale
  arm via launch_subscale.flag marker; SLURM `cluster/21_8_model_averaging.slurm`.
- [x] **BMS-10**: Manuscript tables + master orchestrator (step 21.9) —
  `scripts/21_manuscript_tables.py` three table generators with shared
  TableArtefact dataclass; Figure 1 forest plot delegates to Phase 18 script;
  paper.qmd patcher inserts Methods subsection; master orchestrator
  `cluster/21_submit_pipeline.sh` (+x) chains all 9 steps via afterok exclusively
  with local pre-flight pytest gate; SLURM `cluster/21_9_manuscript_tables.slurm`.

---

## v2 Requirements (Deferred to Future Milestones)

### v5.0 candidates

- **ARVIZ1-01**: ArviZ 1.0 migration (`InferenceData` → `xarray.DataTree`)
- **WORKFLOW-01**: Simulation-based calibration (SBC) as standard pre-fit validation
- **MCMC-01**: Switch to `blackjax` for potential sampling speedups
- **HIER-LBA**: Full PMwG-equivalent hierarchical LBA if reviewers demand it (requires building from scratch in JAX)
- **HORSESHOE-01**: Regularized horseshoe as default on all Level-2 families (currently P2 optional in L2-08)

### Science candidates (post-v4.0)

- **M7**: Model with separate WM decay (`phi_WM`) and RL decay (`phi_RL`) parameters
- **M8-ASYMBIAS**: Model with asymmetric negative feedback neglect in RL only (`eta_RL` parameter neglecting reward=0 feedback). Senta, Bishop, Collins (2025) PLOS Comp Biol winning mechanism. Their `_asymbias` variant captured variance that κ perseveration did not — our M3/M6a/M6b may be partially compensating for this missing mechanism.
- **M9-SPLIT-RHO**: Model with split WM confidence (`rho_low` when ns < K, `rho_high` when ns > K). Senta, Bishop, Collins (2025) winning model explicitly conditions ρ on whether WM capacity is exceeded. Current M2-M6b have a single ρ that cannot distinguish these two regimes.
- **DRIFT-01**: Drift-diffusion model (DDM) alternative to LBA for 2-choice blocks
- **NEURAL-01**: EEG integration with Level-1 neural predictors

## Out of Scope

| Feature | Reason |
|---|---|
| Centered parameterization | Pitfall 1 — funnel pathology guaranteed at N=154 with 8 parameters |
| LKJ correlation prior on group parameters | Too expensive at N=154; marginal benefit over independent HalfNormals |
| Discrete K prior | Breaks HMC (no gradient); TruncatedNormal continuous bound is the correct pattern |
| AIC/BIC as primary criterion | Undermines the milestone's rationale (hierarchical = posterior inference replacing point estimates) |
| Post-hoc MLE-parameter regression as primary inference | Replaced by joint Level-2 regression; `16b` kept as fast-preview only |
| Reimplementing JAX likelihoods inside NumPyro | Wrap existing `*_multiblock_likelihood_stacked` via `numpyro.factor`; don't duplicate |
| Saving MCMC samples to CSV | Use NetCDF via `InferenceData.to_netcdf()` |
| SVI as primary fit method | Variational approximation is poor for skewed posteriors; NUTS is the gold standard |
| Running fits locally on laptop | M6b + M4 hierarchical require GPU; infrastructure only, user runs on Monash M3 |
| ArviZ 1.0 migration | Breaking change; deferred to v5.0 |
| Full PMwG-equivalent hierarchical LBA in Python | No JAX precedent; 4-6 weeks with no reference; out of v4.0 scope even though hierarchical M4 IS in scope (M4H phase uses simpler pattern) |
| Single composite trauma score as primary predictor | Hides subscale-specific effects; subscales are the scientific point of the milestone |
| Cross-model hierarchical comparison in one unified `az.compare` table including M4 | Mathematically invalid (joint vs choice-only observable); M4 in separate track via choice-only marginal |

## Traceability

Each v4.0 requirement maps to exactly one phase. Phases 13-18 continue from v3.0 (which ended at Phase 12).

| Requirement | Phase | Status |
|---|---|---|
| INFRA-01 | Phase 13 | Pending |
| INFRA-02 | Phase 13 | Pending |
| INFRA-03 | Phase 13 | Pending |
| INFRA-04 | Phase 13 | Pending |
| INFRA-05 | Phase 13 | Pending |
| INFRA-06 | Phase 13 | Pending |
| INFRA-07 | Phase 13 | Pending |
| INFRA-08 | Phase 13 | Pending |
| K-01 | Phase 13 | Pending |
| K-02 | Phase 14 | Pending |
| K-03 | Phase 14 | Pending |
| GPU-01 | Phase 14 | Pending |
| GPU-02 | Phase 14 | Pending |
| GPU-03 | Phase 14 | Pending |
| HIER-01 | Phase 15 | Pending |
| HIER-07 | Phase 15 | Pending |
| HIER-08 | Phase 15 | Pending |
| HIER-09 | Phase 15 | Pending |
| HIER-10 | Phase 15 | Pending |
| L2-01 | Phase 15 | Pending |
| HIER-02 | Phase 16 | Complete |
| HIER-03 | Phase 16 | Complete |
| HIER-04 | Phase 16 | Complete |
| HIER-05 | Phase 16 | Complete |
| HIER-06 | Phase 16 | Complete |
| L2-02 | Phase 16 | Complete |
| L2-03 | Phase 16 | Complete |
| L2-04 | Phase 16 | Complete (4 predictors — LEC-5 subcats unavailable) |
| L2-05 | Phase 16 | Complete (32 betas — 8 params x 4 covariates) |
| L2-06 | Phase 16 | Complete (infrastructure — cluster execution pending) |
| L2-07 | Phase 16 | Complete |
| L2-08 | Phase 16 | Complete (deferred — baseline not yet tested) |
| M4H-01 | Phase 17 | Complete |
| M4H-02 | Phase 17 | Complete |
| M4H-03 | Phase 17 | Complete |
| M4H-04 | Phase 17 | Complete |
| M4H-05 | Phase 17 | Complete |
| M4H-06 | Phase 17 | Complete |
| CMP-01 | Phase 18 | Complete |
| CMP-02 | Phase 18 | Complete |
| CMP-03 | Phase 18 | Complete |
| CMP-04 | Phase 18 | Complete |
| MIG-01 | Phase 18 | Complete |
| MIG-02 | Phase 18 | Complete |
| MIG-03 | Phase 18 | Complete |
| MIG-04 | Phase 18 | Complete |
| MIG-05 | Phase 18 | Complete |
| DOC-01 | Phase 18 | Complete |
| DOC-02 | Phase 18 | Complete |
| DOC-03 | Phase 18 | Complete |
| DOC-04 | Phase 18 | Complete |
| PSCAN-01 | Phase 19 | Complete |
| PSCAN-02 | Phase 19 | Complete |
| PSCAN-03 | Phase 19 | Complete |
| PSCAN-04 | Phase 19 | Complete |
| PSCAN-05 | Phase 19 | Complete |
| PSCAN-06 | Phase 19 | Complete |

**Coverage summary:**

| Phase | Requirements | Count |
|---|---|---|
| Phase 13 — Infrastructure Repair & Hierarchical Scaffolding | INFRA-01..08, K-01 | 9 |
| Phase 14 — Collins K Refit + GPU LBA Batching | K-02, K-03, GPU-01..03 | 5 |
| Phase 15 — M3 Hierarchical POC with Level-2 Regression | HIER-01, HIER-07..10, L2-01 | 6 |
| Phase 16 — Choice-Only Family Extension + Subscale L2 | HIER-02..06, L2-02..08 | 12 |
| Phase 17 — M4 Hierarchical LBA | M4H-01..06 | 6 |
| Phase 18 — Integration, Comparison, and Manuscript | CMP-01..04, MIG-01..05, DOC-01..04 | 13 |
| Phase 19 — Associative Scan Likelihood Parallelization | PSCAN-01..06 | 6 |
| **Total** | | **57** |

- v1 requirements: 57 unique REQ-IDs (INFRA 8 + K 3 + GPU 3 + HIER 10 + L2 8 + M4H 6 + CMP 4 + MIG 5 + DOC 4 + PSCAN 6). L2-08 is the P2-optional member of the 8-requirement L2 family.
- Mapped to phases: 57
- Unmapped: 0 ✓
- Each requirement appears in exactly one phase (no duplicates verified).

---

*Requirements defined: 2026-04-11*
*Roadmap created: 2026-04-11*
*Source research: .planning/research/SUMMARY.md*
