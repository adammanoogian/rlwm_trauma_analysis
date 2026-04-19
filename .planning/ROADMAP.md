# Roadmap: RLWM Trauma Analysis

## Milestones

- ✅ **v1.0 M3 Infrastructure** - Phases 1-3 (shipped 2026-01-30)
- ✅ **v2.0 Post-Fitting Validation & Publication Readiness** - Phases 4-7 (shipped 2026-02-06, Phases 6-7 deferred)
- ✅ **v3.0 Model Extensions (M4-M6)** - Phases 8-12 (shipped 2026-04-03)
- ✅ **v4.0 Hierarchical Bayesian Pipeline & LBA Acceleration** - Phases 13-22 (shipped 2026-04-19; archive: [milestones/v4.0-ROADMAP.md](milestones/v4.0-ROADMAP.md))
- 🔄 **v5.0 Empirical Artifacts & Manuscript Finalization** - Phases 23-27 (started 2026-04-19)

## Phases

<details>
<summary>✅ v1.0 M3 Infrastructure (Phases 1-3) - SHIPPED 2026-01-30</summary>

### Phase 1: Perseveration Extension
**Goal**: Extend WM-RL model with perseveration parameter κ
**Plans**: 2 plans

Plans:
- [x] 01-01: Core M3 implementation
- [x] 01-02: Backward compatibility validation

### Phase 2: MLE Infrastructure
**Goal**: Complete MLE fitting infrastructure for M3
**Plans**: 2 plans

Plans:
- [x] 02-01: MLE parameter bounds and transforms
- [x] 02-02: CLI integration and testing

### Phase 3: Model Comparison
**Goal**: N-model comparison with Akaike weights
**Plans**: 2 plans

Plans:
- [x] 03-01: Comparison framework
- [x] 03-02: Output formatting and validation

</details>

<details>
<summary>✅ v2.0 Post-Fitting Validation (Phases 4-5) - SHIPPED 2026-02-06 (Phases 6-7 deferred)</summary>

### Phase 4: Regression Visualization
**Goal**: Enhanced visualization and organization for continuous regression analysis
**Plans**: 2 plans

Plans:
- [x] 04-01-PLAN.md — Shared plotting utility + Script 15
- [x] 04-02-PLAN.md — Script 16 + model subdirectories

### Phase 5: Parameter Recovery & Posterior Predictive Checks
**Goal**: Complete parameter recovery pipeline and posterior predictive checks
**Plans**: 5 plans

Plans:
- [x] 05-01-PLAN.md — Core recovery pipeline
- [x] 05-02-PLAN.md — CLI, output, and visualization
- [x] 05-03-PLAN.md — Script 11 wrapper + end-to-end verification
- [x] 05-04-PLAN.md — PPC mode + behavioral comparison
- [x] 05-05-PLAN.md — Model recovery evaluation + Script 09 orchestrator

</details>

<details>
<summary>✅ v3.0 Model Extensions M4-M6 (Phases 8-12) - SHIPPED 2026-04-03</summary>

### Phase 8: M5 RL Forgetting
**Goal**: RL forgetting model with Q-value decay (phi_rl)
**Plans**: 2 plans — [x] 08-01, [x] 08-02

### Phase 9: M6a Stimulus-Specific Perseveration
**Goal**: Per-stimulus perseveration tracking (kappa_s)
**Plans**: 2 plans — [x] 09-01, [x] 09-02

### Phase 10: M6b Dual Perseveration
**Goal**: Global + stimulus-specific kernels with stick-breaking constraint
**Plans**: 2 plans — [x] 10-01, [x] 10-02

### Phase 11: M4 LBA Joint Choice+RT
**Goal**: Joint choice+RT fitting via Linear Ballistic Accumulator
**Plans**: 3 plans — [x] 11-01, [x] 11-02, [x] 11-03

### Phase 12: Cross-Model Integration
**Goal**: Cross-model recovery, downstream script integration, documentation
**Plans**: 3 plans — [x] 12-01, [x] 12-02, [x] 12-03 (gap closure)

</details>

<details>
<summary>✅ v4.0 Hierarchical Bayesian Pipeline & LBA Acceleration (Phases 13-22) — SHIPPED 2026-04-19</summary>

**Milestone Goal:** Move the inference pipeline from MLE point estimates to full hierarchical Bayesian posteriors with trauma subscales as Level-2 predictors, principled Bayesian model comparison (PSIS-LOO + stacking + RFX-BMS/PXP), and GPU-accelerated LBA sampling for M4. Replaces post-hoc FDR correction (48-test M6b family) with a single joint posterior and replaces r < 0.80 MLE recovery with shrinkage-regularized individual-level posteriors. Phases 19-20 extend with GPU-accelerated likelihood via associative scan and DEER non-linear parallelization research. Phase 21 (added 2026-04-12) replaces MLE-preselected-winner workflow with 9-step principled Bayesian selection pipeline anchored to Baribault & Collins 2023 + Hess 2025. Phase 22 (added 2026-04-19) closes verification/traceability/reproducibility debt.

**Full milestone archive:** [milestones/v4.0-ROADMAP.md](milestones/v4.0-ROADMAP.md)
**Requirements archive:** [milestones/v4.0-REQUIREMENTS.md](milestones/v4.0-REQUIREMENTS.md)
**Milestone audit:** [milestones/v4.0-MILESTONE-AUDIT.md](milestones/v4.0-MILESTONE-AUDIT.md)
**Git range:** `81c3570` → `71e063d` (172 commits, 9 days)

**Execution Order:** Phase 13 → Phase 14 → Phase 15 → Phase 16 → Phase 17 → Phase 18 → Phase 19 → Phase 20 → Phase 21 → Phase 22

#### Phase 13: Infrastructure Repair & Hierarchical Scaffolding
**Goal**: Fix the P0 broken-import bug, resurrect `numpyro_models.py` at its canonical path, and stand up the non-centered parameterization helpers, pointwise log-lik helper, and schema-parity CSV writer that every downstream phase depends on. Lock dependency pins and PyMC-drop decision. Deliver Collins K parameterization research as prerequisite to Phase 14 K refit.
**Depends on**: Phase 12 (v3.0 shipped)
**Requirements**: INFRA-01, INFRA-02, INFRA-03, INFRA-04, INFRA-05, INFRA-06, INFRA-07, INFRA-08, K-01
**Success Criteria** (what must be TRUE):
  1. `from scripts.fitting.numpyro_models import ...` succeeds on a fresh checkout without relying on stale `__pycache__`; `python scripts/13_fit_bayesian.py --model qlearning --help` prints help text without ImportError.
  2. Unit tests in `scripts/fitting/tests/test_numpyro_helpers.py` recover known parameters (rel error < 5%) from simulated data for every bounded transform (logit for [0,1], sigmoid with rescaling for K, softplus-sigmoid for stick-breaking).
  3. `compute_pointwise_log_lik()` produces log-lik array of shape (chains, samples, participants, trials) that feeds cleanly into `az.waic(idata)` and `az.loo(idata)` without the "log_likelihood group missing" warning.
  4. `bayesian_summary_writer.py` emits a CSV with column names identical to the existing MLE `qlearning_individual_fits.csv` plus `_hdi_low`/`_hdi_high`/`_sd` suffixes and a `parameterization_version` column; `pytest` test compares schema against a checked-in reference.
  5. Compile-time CI gate passes: first JAX compile of the M3 hierarchical model completes in < 60 seconds on cluster scratch with `JAX_COMPILATION_CACHE_DIR` set.
  6. `docs/K_PARAMETERIZATION.md` exists with explicit recommendation (free continuous vs bounded-by-max-set-size vs fixed K = max set-size) cited from Collins 2012/2014/2018.
**Plans**: 5 plans

Plans:
- [x] 13-01-PLAN.md — P0 import fix, dependency pins, PyMC removal
- [x] 13-02-PLAN.md — Collins K parameterization documentation
- [x] 13-03-PLAN.md — Pointwise log-likelihood refactor (jax_likelihoods.py)
- [x] 13-04-PLAN.md — Non-centered parameterization helpers + parameterization_version
- [x] 13-05-PLAN.md — Bayesian diagnostics, summary writer, cluster SLURM

#### Phase 14: Collins K Refit + GPU LBA Batching
**Goal**: Apply the Collins K convention from Phase 13 to all 7 models' MLE fits and verify K recovery improves above r >= 0.50. In parallel, deliver the GPU-batched LBA path (`fit_all_gpu_m4`) so that M4 parameter recovery becomes tractable (< 12h on A100) — this is a prerequisite for any M4 hierarchical work in Phase 17.
**Depends on**: Phase 13 (K-01 research + INFRA scaffolding)
**Requirements**: K-02, K-03, GPU-01, GPU-02, GPU-03
**Success Criteria** (what must be TRUE):
  1. Constrained-K MLE refit completes on the cluster for all 7 models (M1, M2, M3, M5, M6a, M6b, M4); new `*_individual_fits.csv` files exist under `output/mle/` with the `parameterization_version` column set to "collins_k_v1".
  2. Parameter recovery run with the new K bounds produces `K` (or `capacity`) recovery correlation r >= 0.50 on N=50 synthetic data (up from r=0.21 baseline in quick-005) for at least M3 and M6b; result documented in `output/recovery/collins_k_refit_summary.md`.
  3. `fit_all_gpu_m4(...)` callable from `fit_mle.py` successfully fits a 5-participant M4 synthetic recovery batch in float64 without NaN or dtype errors, and produces parameter estimates within 10% of the ground truth on at least 3/5 participants.
  4. Real-data M4 fit across all N=154 participants completes in < 12 hours wall-clock on an A100 node (compared to the ~48h CPU baseline); runtime logged in `output/mle/wmrl_m4_gpu_runtime.txt`.
  5. AIC/BIC tables regenerated with constrained-K refits; comparison with pre-refit tables shows no catastrophic model ranking flips (M6b remains rank 1 on choice-only AIC; if the ranking flips, a documented scientific decision is made in the SUMMARY).
**Plans**: 3 plans

Plans:
- [ ] 14-01-PLAN.md (Wave 1) — K bounds [1,7]→[2,6] in mle_utils.py + parameterization_version stamp in fit_mle.py
- [ ] 14-02-PLAN.md (Wave 2) — fit_all_gpu_m4 wrapper + M4 synthetic smoke test
- [ ] 14-03-PLAN.md (Wave 3) — SLURM/comparison compatibility check + cluster execution commands

#### Phase 15: M3 Hierarchical Proof-of-Concept with Level-2 Regression
**Goal**: Validate the entire hierarchical NumPyro stack end-to-end on M3 alone with LEC-total → kappa as the single Level-2 covariate. This phase is the gate: if the convergence/shrinkage/schema-parity pipeline works on M3 and reproduces v3.0's surviving-FDR finding (phi �— IES-R and kappa �— LEC-5 from quick-006), it green-lights the mechanical extension to the rest of the model family in Phase 16.
**Depends on**: Phase 13 (all INFRA scaffolding), Phase 14 (Collins K bounds — M3 uses the new K parameterization for the hierarchical fit)
**Requirements**: HIER-01, HIER-07, HIER-08, HIER-09, HIER-10, L2-01
**Success Criteria** (what must be TRUE):
  1. M3 hierarchical fit on the real N=154 dataset passes the full convergence gate: `R-hat <= 1.01` on every group parameter, `ESS_bulk >= 400` on every group parameter, zero divergences; `target_accept_prob` auto-bump from 0.8 to 0.95 to 0.99 is exercised and logged.
  2. Shrinkage diagnostic `1 - var_post_individual / var_post_group` computed for every M3 parameter; `kappa` exceeds 0.3 (identified), and parameters below 0.3 are explicitly flagged in `output/bayesian/wmrl_m3_shrinkage_report.md` so downstream inference treats them as descriptive only.
  3. Level-2 regression on LEC-total reproduces the v3.0 quick-006 finding: the posterior 95% credible interval on the `beta_lec_kappa` coefficient excludes zero in the same direction as the MLE FDR-survivor (`kappa x LEC-5 Total events`) from script 16.
  4. Group-stratified posterior predictive check reproduces the v3.0 M3 behavioral learning curves (accuracy by block �— trauma group) within the 95% PPC envelope for at least 18/21 main-task blocks.
  5. `az.waic(m3_idata)` and `az.loo(m3_idata)` both return without a Pareto-k > 0.7 warning on M3; `output/bayesian/wmrl_m3_individual_fits.csv` loads in `scripts/15_analyze_mle_by_trauma.py --source bayesian --model wmrl_m3` without schema errors.
  6. Parametric dispatch smoke test (`pytest -k "test_smoke_dispatch"`) runs M3 with 5 subjects + 200 samples and completes in < 60 seconds.
**Plans**: 3 plans

Plans:
- [ ] 15-01-PLAN.md (Wave 1) — M3 hierarchical model + stacked data prep + test_compile_gate fix + smoke test
- [ ] 15-02-PLAN.md (Wave 2) — Convergence auto-bump + fit_bayesian.py CLI + shrinkage + WAIC/LOO padding
- [ ] 15-03-PLAN.md (Wave 3) — PPC infrastructure + SLURM script + end-to-end validation checkpoint

#### Phase 16: Choice-Only Family Extension + Subscale Level-2
**Goal**: Mechanically extend the Phase 15 M3 template to the rest of the choice-only family (M1, M2, M5, M6a, M6b) and lock the full subscale Level-2 parameterization (IES-R total + Gram-Schmidt residualized intrusion/avoidance/hyperarousal + LEC-5 subcategories) on the winning model M6b. Includes the collinearity audit (Pitfall 3) and permutation null test (Pitfall 2) that guard the central scientific advance of v4.0. This phase must precede Phase 17 because M6b's non-centered dual stick-breaking parameterization informs M4's analogous threshold-offset decisions.
**Depends on**: Phase 15 (M3 template validated)
**Requirements**: HIER-02, HIER-03, HIER-04, HIER-05, HIER-06, L2-02, L2-03, L2-04, L2-05, L2-06, L2-07, L2-08
**Success Criteria** (what must be TRUE):
  1. All 5 new choice-only hierarchical models (M1, M2, M5, M6a, M6b) pass the HIER-07 convergence gate on real N=154 data: `R-hat <= 1.01`, `ESS_bulk >= 400`, zero divergences; outputs written to `output/bayesian/{model}_individual_fits.csv` and `{model}_posterior.nc`.
  2. IES-R subscale audit in `output/bayesian/level2/ies_r_collinearity_audit.md` reports the condition number of `[intrusion, avoidance, hyperarousal]` design submatrix on the real N=154 survey (condition number < 30 after Gram-Schmidt residualization against IES-R total); the orthogonalized subscale regressor set is checked into `scripts/fitting/level2_design.py`.
  3. Full M6b subscale Level-2 fit completes with ~48 non-zero coefficients (8 parameters x 6 predictors = LEC total + 3 residualized IES-R subscales + LEC-5 physical/sexual/accident subcategories) within a single SLURM job; convergence gate passes.
  4. Permutation null test: 50 random shuffles of the trauma label rows refit under M3 + LEC-total produce zero "surviving" Level-2 effects (posterior credible interval excluding zero on fewer than 5% of shuffles, matching the nominal alpha=0.05 false-positive rate).
  5. `scripts/18_bayesian_level2_effects.py` produces forest plots for the M6b Level-2 regression posterior; output PNG checked into `output/bayesian/figures/m6b_forest_lec5.png` and `m6b_forest_iesr_residuals.png`.
  6. `scripts/14_compare_models.py --bayesian-comparison` runs `az.compare([M1, M2, M3, M5, M6a, M6b], ic='loo', method='stacking')` and produces a stacking-weight table; M6b stacking weight >= 0.5 or the table is flagged as inconclusive in the output markdown.
  7. L2-08 horseshoe prior is either enabled (if the baseline M6b Level-2 fit had convergence issues) or explicitly deferred with justification in the phase SUMMARY.
**Plans**: 7 plans

Plans:
- [x] 16-01-PLAN.md (Wave 1) — Collinearity audit + level2_design.py + LEC-5 data gap resolution
- [x] 16-02-PLAN.md (Wave 1) — Port M1 (Q-learning) + M2 (WM-RL) hierarchical models
- [x] 16-03-PLAN.md (Wave 1) — Port M5, M6a, M6b hierarchical models
- [x] 16-04-PLAN.md (Wave 2) — Refactor fit_bayesian.py dispatch + smoke tests + SLURM scripts
- [x] 16-05-PLAN.md (Wave 3) — M6b full subscale Level-2 model (32 coefficients)
- [x] 16-06-PLAN.md (Wave 3) — Permutation null test infrastructure + SLURM array job
- [x] 16-07-PLAN.md (Wave 4) — Forest plots + Bayesian model comparison + L2-08 decision

#### Phase 17: M4 Hierarchical LBA
**Goal**: Deliver hierarchical M4 (joint choice+RT via LBA) under NumPyro NUTS with float64 process isolation, non-centered `log(b - A)` reparameterization, checkpoint-and-resume for the 48h SLURM wall, and Pareto-k gating for the downstream comparison. User has explicitly kept this phase in scope despite the research recommendation to descope: accept ~150-200 GPU-hour total budget, ~24-48h single-fit wall clock, and the near-certain Pareto-k fallback for M4 vs choice-only comparison. This phase is sequenced after Phase 16 so that M6b's non-centered dual-parameter pattern serves as the reference template for M4's threshold-offset parameterization.
**Depends on**: Phase 14 (GPU LBA batching, float64 infrastructure), Phase 16 (M6b non-centered pattern as template)
**Requirements**: M4H-01, M4H-02, M4H-03, M4H-04, M4H-05, M4H-06
**Success Criteria** (what must be TRUE):
  1. M4 hierarchical fit launches in its own Python process (separate SLURM job) with `jax.config.update('jax_enable_x64', True)` and `numpyro.enable_x64()` executed before any JAX-backed import; integration test confirms `jnp.zeros(1).dtype == jnp.float64` inside the model context.
  2. Non-centered `log(b - A)` parameterization: an integration test generates synthetic LBA data at a known ground-truth `(A, b)`, fits the hierarchical model at N=10 with 500 samples, and recovers both the group-level mean of `log_delta` and individual `A_i` values within 15% relative error (proxy for absence of boundary funnel pathology).
  3. `chain_method='vectorized'` with `num_warmup=1000`, `num_samples=1500`, `target_accept_prob=0.95` fits N=154 real-data M4 within the 48h SLURM cap on an A100; wall clock and number of divergences logged to `output/bayesian/wmrl_m4_run_metadata.json`.
  4. Checkpoint-and-resume integration test: a 200-sample M4 run is killed mid-sampling via `os.kill`, restarted from `mcmc.post_warmup_state`, and the resumed run produces posterior draws whose group-parameter posterior means agree with a non-interrupted reference run to within 10% relative error.
  5. `az.loo(m4_idata, pointwise=True)` is called immediately after the fit; Pareto-k diagnostics are inspected and one of two branches executes: (a) if fewer than 5% of trials have `k > 0.7`, the M4 fit is added to a separate choice+RT comparison track; (b) otherwise, the pipeline falls back to a choice-only marginal log-likelihood for M4 and records the fallback in `output/bayesian/wmrl_m4_pareto_k_report.md`.
  6. `cluster/13_bayesian_m4_gpu.slurm` exists with `--time=48:00:00`, `--mem=96G`, `--gres=gpu:a100:1` and has been dry-run submitted (without actual compute) to verify SLURM directive parsing.
**Plans**: 3 plans

Plans:
- [x] 17-01-PLAN.md (Wave 1) — M4 data prep (RT stacking) + hierarchical model function + unit tests
- [x] 17-02-PLAN.md (Wave 2) — Self-contained fitting script with float64 isolation, checkpoint-resume, Pareto-k gating
- [x] 17-03-PLAN.md (Wave 3) — Integration tests (float64, log(b-A), checkpoint) + SLURM script

#### Phase 18: Integration, Comparison, and Manuscript
**Goal**: Wire the Bayesian fits from Phases 15-17 into the existing downstream analysis pipeline via the schema-parity `--source mle|bayesian` flag (minimal code change by design), freeze `16b_bayesian_regression.py` as deprecated, produce MLE-vs-Bayesian reliability scatterplots, and rewrite the manuscript methods/results/limitations sections around the joint hierarchical narrative (replacing the v3.0 FDR-corrected post-hoc regression).
**Depends on**: Phase 15, Phase 16, Phase 17 (all hierarchical fits complete)
**Requirements**: CMP-01, CMP-02, CMP-03, CMP-04, MIG-01, MIG-02, MIG-03, MIG-04, MIG-05, DOC-01, DOC-02, DOC-03, DOC-04
**Success Criteria** (what must be TRUE):
  1. `scripts/15_analyze_mle_by_trauma.py --source bayesian --model all` runs successfully against `output/bayesian/*_individual_fits.csv` and produces the same analysis artifacts (group-comparison tables, forest plots) as the MLE path; diff of script 15 output schema between `--source mle` and `--source bayesian` is empty except for the added `_hdi_low`/`_hdi_high`/`_sd` columns.
  2. `scripts/16_regress_parameters_on_scales.py --source bayesian` and `scripts/17_analyze_winner_heterogeneity.py --source bayesian` run on every model fit in Phases 15-16 without touching analysis logic.
  3. `scripts/16b_bayesian_regression.py` carries a deprecation docstring in its module header referencing the Level-2 hierarchical pipeline as the replacement, but remains runnable as the fast-preview supplementary path (no PyMC imports; NumPyro-only).
  4. MLE-vs-Bayesian reliability scatterplots (one per parameter x model cell) saved to `output/bayesian/figures/mle_vs_bayes/`; each plot shows posterior mean vs MLE point estimate with 45-degree reference line and highlights shrinkage direction for the winning model M6b.
  5. `scripts/14_compare_models.py --bayesian-comparison` produces a final stacking-weight table covering all 6 choice-only hierarchical models; M4 appears in a separate section with its Pareto-k-gated fallback metric (choice-only marginal) and an explicit note that it is not commensurable with the `az.compare` choice-only table.
  6. `docs/MODEL_REFERENCE.md` has a new "Hierarchical Bayesian Pipeline" section covering non-centered parameterization, Level-2 regression structure, the `numpyro.factor` + post-hoc `compute_pointwise_log_lik` pattern, and the WAIC/LOO workflow; `docs/K_PARAMETERIZATION.md` is cross-referenced.
  7. Manuscript `paper.qmd` methods section replaces the MLE-centric narrative with the hierarchical Bayesian methods (NumPyro, vmap, `numpyro.factor`, joint Level-2 regression, WAIC/LOO); results section presents forest plots and stacking weights in place of FDR-corrected post-hoc regression p-values; limitations section documents the Pareto-k fallback for M4, residual K identifiability caveats if Phase 14 K refit did not fully resolve the issue, and M6b shrinkage diagnostics.
**Plans**: 5 plans

Plans:
- [x] 18-01-PLAN.md (Wave 1) — Add --source mle|bayesian flag to scripts 15, 16, 17
- [x] 18-02-PLAN.md (Wave 1) — Extend script 14 Bayesian comparison: CSV output, M4 track, WAIC
- [x] 18-03-PLAN.md (Wave 1) — Deprecate script 16b + MLE-vs-Bayesian reliability scatterplots
- [x] 18-04-PLAN.md (Wave 1) — MODEL_REFERENCE.md Hierarchical Bayesian Pipeline section
- [x] 18-05-PLAN.md (Wave 2) — Manuscript methods/results/limitations revision

#### Phase 19: Associative Scan Likelihood Parallelization
**Goal**: Replace the O(T) sequential `lax.fori_loop`/`lax.scan` in RLWM likelihood evaluation with O(log T) `jax.lax.associative_scan` for the linear-recurrence components (Q-value updates and WM forgetting), enabling GPU-accelerated MCMC. Benchmark against the CPU baseline established in Phases 15-16 to quantify actual speedup. This phase includes deep research into the associative scan / parallel prefix sum literature (PaMoRL NeurIPS 2024, S4/Mamba, parallel Kalman smoothers) and a rigorous numerical validation against the existing sequential implementation.
**Depends on**: Phase 16 (CPU baseline timing from all 6 choice-only models), Phase 18 (manuscript complete — this is optimization, not science)
**Requirements**: PSCAN-01, PSCAN-02, PSCAN-03, PSCAN-04, PSCAN-05, PSCAN-06
**Success Criteria** (what must be TRUE):
  1. Literature review document `docs/PARALLEL_SCAN_LIKELIHOOD.md` covers: (a) AR(1) linear recurrence formulation for Q-learning updates with proof that observed-data likelihood evaluation is a linear recurrence, (b) PaMoRL PETE algorithm for TD-λ eligibility traces, (c) Mamba/S4 parallel scan for data-dependent decay, (d) parallel Kalman smoother lineage (Särkkä & García-Fernández 2021), (e) explicit statement of which RLWM components are linear-recurrence (Q-update, WM-decay) vs non-linear (WM-Q mixing, softmax).
  2. `associative_scan_q_update()` function in `jax_likelihoods.py` computes all T Q-values in O(log T) parallel time for a single (stimulus, action) stream; unit test verifies element-wise agreement with sequential `lax.scan` to float32 tolerance (< 1e-5 relative error) on 1000-trial synthetic sequences.
  3. `associative_scan_wm_update()` function handles WM forgetting recurrence (`WM_t = φ·WM_{t-1} + (1-φ)/nA`) and hard overwrite (`WM(s,a) ← r`) via the same parallel scan; unit test verifies agreement with sequential implementation.
  4. `wmrl_m3_multiblock_likelihood_stacked_pscan()` replaces the inner `lax.fori_loop` with associative scan for Q and WM, but keeps the non-linear mixing (`w·WM + (1-w)·Q`) and softmax as a sequential post-scan pass; total log-likelihood agrees with existing `wmrl_m3_multiblock_likelihood_stacked()` to < 1e-4 relative error on all 154 real participants.
  5. GPU benchmark: M3 hierarchical fit with associative scan likelihood on A100 (4 chains, 1000 warmup, 2000 samples, N=154) completes in < X hours (where X is determined by the Phase 16 CPU baseline ÷ target speedup); wall clock, peak VRAM, and divergence count logged to `output/bayesian/pscan_benchmark.json`.
  6. A/B comparison: posterior parameter means from associative-scan GPU fit agree with sequential CPU fit (Phase 15 M3 results) to within MCMC noise (< 5% relative error on group-level means); WAIC/LOO agree to < 1.0.
**Plans**: 3 plans

Plans:
- [x] 19-01-PLAN.md (Wave 1) — Literature doc + affine_scan helper + associative_scan_q_update + associative_scan_wm_update + unit tests
- [x] 19-02-PLAN.md (Wave 2) — Pscan block/multiblock likelihoods for all 6 models + real-data agreement validation
- [x] 19-03-PLAN.md (Wave 3) — --use-pscan CLI flag + micro-benchmark + SLURM GPU script + A/B comparison protocol

#### Phase 20: DEER Non-Linear Parallelization (Research)
**Goal**: Investigate DEER-style (ICLR 2024) fixed-point iteration approach to parallelize the remaining non-linear components of the RLWM likelihood: the WM-Q mixing weight computation (`w = ρ/(ρ + set_size)`) → combined policy (`w·WM + (1-w)·Q`) → softmax → log-probability. This is a research phase: the outcome may be "DEER doesn't provide sufficient speedup for T≈600 sequences" or "DEER works and delivers Nx additional speedup." If successful, the fully-parallel likelihood makes GPU-accelerated MCMC viable for ALL models without a sequential post-scan pass. Includes deep-dive into the DEER paper, the Unifying Framework (TMLR 2025), and Newton-iteration convergence guarantees for the RLWM non-linearity.
**Depends on**: Phase 19 (associative scan baseline + benchmark infrastructure)
**Requirements**: DEER-01, DEER-02, DEER-03, DEER-04
**Success Criteria** (what must be TRUE):
  1. Research document `docs/DEER_NONLINEAR_PARALLELIZATION.md` covers: (a) DEER fixed-point iteration algorithm with worked example on RLWM mixing non-linearity, (b) convergence analysis — is the RLWM softmax-mixing a contraction mapping? Under what parameter regimes?, (c) comparison with alternative approaches (Picard iteration, Newton-Raphson, direct linearization), (d) the Unifying Framework (TMLR 2025) perspective on where RLWM fits in the taxonomy, (e) explicit go/no-go recommendation with justification.
  2. If go: `wmrl_m3_multiblock_likelihood_stacked_deer()` fully parallelizes all components; benchmark shows additional speedup over Phase 19's scan+sequential hybrid; numerical agreement with sequential implementation to < 1e-3 relative error.
  3. If no-go: Document why (convergence failure, insufficient speedup for T≈600, numerical instability) with empirical evidence. Phase 19's hybrid approach becomes the final GPU path.
  4. Regardless of outcome: project-utils guide `JAX_GPU_BAYESIAN_FITTING.md` updated with findings, and a reproducible benchmark script exists in `validation/benchmark_parallel_scan.py`.
**Plans**: 3 plans

Plans:
- [x] 20-01-PLAN.md (Wave 1) — DEER research document + perseveration precomputation functions + unit tests
- [x] 20-02-PLAN.md (Wave 2) — Vectorize Phase 2 in all 12 pscan likelihood variants + agreement tests
- [x] 20-03-PLAN.md (Wave 3) — Update PARALLEL_SCAN_LIKELIHOOD.md + extend benchmark script + JAX GPU fitting guide

#### Phase 21: Principled Bayesian Model Selection Pipeline

**Goal:** Replace the MLE-preselected "M6b is the winner" workflow with a fully Bayesian, step-by-step pipeline for model selection and trauma-parameter inference. Each step gets a dedicated cluster submission so the pipeline is re-runnable afresh from step 0. Anchored to Baribault & Collins (2023, *Psychological Methods*, DOI 10.1037/met0000554) and Hess et al. (2025, *Computational Psychiatry* 9(1):76–99, DOI 10.5334/cpsy.116). AIC/BIC are deprecated for hierarchical-Bayesian model selection in this pipeline; **PSIS-LOO + stacking weights** (Yao, Vehtari, Simpson, Gelman 2018) are primary; **RFX-BMS with PXP** (Stephan et al. 2009; Rigoux et al. 2014) is secondary.

**Depends on:** Phase 20 (fully-batched vmap rollout complete; DEER research concluded no-go)

**Motivation.** User flagged three issues with the existing v4.0 plan: (1) MLE AIC was pre-selecting M6b as the winner and all downstream Bayesian inference anchored to that, which is circular; (2) the earlier step-4 PPC comparison to MLE baseline is MLE-anchored and should be dropped as a *selection* criterion (retained only as a sanity check); (3) scale-effect inference and choice-fit comparison were entangled — they are different scientific questions and need a principled linear workflow that separates them.

**Pipeline structure (linear; each step = separate cluster submission):**

1. **21.1 — Prior predictive checks** (all 6 choice-only models). Simulate choices from priors alone; verify simulated accuracy curves cover plausible behavior (~0.4–0.9 asymptotic, not stuck at 0.33 or 1.0). Per Baribault & Collins (2023) gate 2 and Hess et al. (2025) stage 3. New orchestrator `scripts/21_run_prior_predictive.py`; SLURM `cluster/21_prior_predictive.slurm`. Outputs: `output/bayesian/prior_predictive/{model}_prior_sim.nc`.
2. **21.2 — Bayesian parameter recovery** (all 6 models). Simulate N=50 synthetic datasets per model with true params drawn from priors; fit each synthetic with the Bayesian pipeline; assess Pearson *r* of posterior mean vs. true AND 95% HDI coverage calibration (~95% expected). Per Baribault & Collins (2023) gate 3 and Hess et al. (2025) stage 4. Reuses `scripts/11_run_model_recovery.py` structure with Bayesian fitter. SLURM `cluster/21_bayesian_recovery.slurm`. Outputs: `output/bayesian/recovery/{model}_recovery.csv`.
3. **21.3 — Fit all 6 models hierarchically, NO L2 scales (baseline).** 4 chains × 1000 warmup × 2000 samples, `max_tree_depth=10`. All models share common hyperpriors (`PARAM_PRIOR_DEFAULTS`); no trauma covariates. Convergence gate: R-hat < 1.05, ESS_bulk > 400, zero divergences (auto-bump 0.80 → 0.95 → 0.99). Extends existing `cluster/13_bayesian_multigpu.slurm` to iterate all 6 models. Outputs: `output/bayesian/baseline/{model}_posterior.nc`.
4. **21.4 — Baseline convergence & fit-quality audit.** Per-model report: max R-hat, min ESS_bulk, divergences, BFMI, tree-depth saturation. Posterior predictive checks (simulate from posterior draws, compare to observed behavior). Any model failing the gate is excluded from further steps. `scripts/21_baseline_audit.py`; SLURM `cluster/21_baseline_audit.slurm`. Outputs: `output/bayesian/baseline/convergence_report.md`.
5. **21.5 — PSIS-LOO + stacking weights on baseline models (choice-fit ranking).** Compute pointwise log-likelihoods per model (existing `compute_pointwise_log_lik`); `az.loo` per model with Pareto-k diagnostic (< 0.7 for > 99% of observations); `az.compare(ic='loo', method='stacking')` for stacking weights. Secondary: RFX-BMS with PXP (new ~50 LOC `scripts/fitting/bms.py` per Rigoux et al. 2014). Winner(s) = any model within 2×SE(Δelpd) of top. `scripts/21_compute_loo_stacking.py`; SLURM `cluster/21_model_comparison.slurm`.
6. **21.6 — Fit winner(s) hierarchically WITH L2 scales (confirmatory).** Only models identified in 21.5 as winners/co-winners get re-fit with the 4-covariate L2 design matrix (`lec_total`, `iesr_total`, `iesr_intr_resid`, `iesr_avd_resid`). Same MCMC budget as 21.3 for comparability. Hypothesis tests: 95% HDI exclusion of zero for `beta_lec_kappa` / `beta_lec_kappa_total` / `beta_lec_kappa_share` / `beta_lec_kappa_s`. SLURM `cluster/21_winner_scaled.slurm`. Outputs: `output/bayesian/scaled/{winner}_posterior.nc`.
7. **21.7 — Scale-fit audit (gate between 21.6 and 21.8).** Does adding scales degrade fit (divergences, ESS, R-hat)? Do group-level posteriors shift implausibly? Can we recover β coefficients (simulate with known β, fit, check Pearson *r*)? Per Baribault & Collins (2023) troubleshooting protocol. `scripts/21_scale_audit.py`; SLURM `cluster/21_scale_audit.slurm`. Outputs: `output/bayesian/scaled/audit_report.md`.
8. **21.8 — Model-averaged scale effects** (if > 1 winner). Stacking-weighted posterior mixtures for each scale-parameter pair; flag where single-model and averaged inference disagree. Optionally: M6b-subscale exploratory arm (32 βs; horseshoe prior or Bonferroni/FDR multiplicity correction). `scripts/21_model_averaging.py`; SLURM `cluster/21_scale_averaging.slurm`. Outputs: `output/bayesian/scaled/averaged_scale_effects.csv`.
9. **21.9 — Final manuscript tables and figures.** Headline: LOO ranking + stacking weights + PXP. Confirmatory: scale-effect HDIs in winner(s), model-averaged if applicable. Exploratory: M6b-subscale β grid with multiplicity correction. Update `paper.qmd` Methods and Results with explicit workflow description and citations to the two anchor papers.

**Reusable infrastructure (~70% already built):** `scripts/simulations/unified_simulator.py`, `scripts/simulations/generate_data.py`, `scripts/fitting/jax_likelihoods.py` (fully-batched vmap), `scripts/fitting/numpyro_models.py` (all 6 hierarchical models), `scripts/fitting/fit_bayesian.py` (Bayesian fitter with auto-bump), `scripts/fitting/bayesian_diagnostics.py::compute_pointwise_log_lik`, `scripts/11_run_model_recovery.py` (structure reused, fitter swapped), `cluster/13_bayesian_multigpu.slurm` (base template). `validation/compare_posterior_to_mle.py` retained as a **sanity check only**, not as a selection criterion.

**New components to build (~30%):** `scripts/21_run_prior_predictive.py`, `scripts/21_run_bayesian_recovery.py`, `scripts/21_baseline_audit.py`, `scripts/21_compute_loo_stacking.py`, `scripts/21_scale_audit.py`, `scripts/21_model_averaging.py`; one SLURM script per step under `cluster/21_*.slurm`; `scripts/fitting/bms.py` for RFX-BMS + PXP.

**Success Criteria (what must be TRUE):**
  1. Pipeline reproducible from a cold start by running `bash cluster/21_submit_pipeline.sh` and waiting through the cluster queue; each step has its own SLURM submission and a STATE.md-visible output before the next step runs.
  2. Convergence gate (Baribault & Collins 2023: R-hat ≤ 1.05, ESS_bulk ≥ 400, 0 divergences after auto-bump) refuses to proceed past step 21.4 unless all retained models pass.
  3. PSIS-LOO ranks computed with Pareto-k < 0.7 on > 99% of observations; stacking weights sum to 1.0 and the winner's ∆elpd vs. runner-up is reported with SE.
  4. Scale-effect HDIs reported only within models that passed the 21.4 gate AND were identified as winners in 21.5.
  5. Parameter recovery (21.2) meets Pearson *r* ≥ 0.8 for identifiable parameters (κ, κ_total, κ_share) and is explicitly flagged as "descriptive only" for parameters known to be non-identified (α, φ, ρ, K).
  6. Manuscript Methods section (21.9) references both anchor papers (Baribault & Collins 2023; Hess et al. 2025) and describes the pipeline in enough detail that a reader can replicate step-by-step.
  7. `/gsd:audit-milestone` verification passes at milestone close.

**Plans:** 10 plans

Plans:
- [ ] 21-01-PLAN.md (Wave 1) — Prior predictive checks (21.1) + SLURM
- [ ] 21-02-PLAN.md (Wave 1) — scripts/fitting/bms.py (RFX-BMS + PXP) + unit tests
- [ ] 21-03-PLAN.md (Wave 2) — Bayesian parameter recovery (21.2) + SLURM array
- [ ] 21-04-PLAN.md (Wave 2) — Baseline hierarchical fits no L2 (21.3) + save_results output_subdir patch
- [ ] 21-05-PLAN.md (Wave 3) — Convergence + PPC audit (21.4) + pipeline-block exit
- [ ] 21-06-PLAN.md (Wave 4) — PSIS-LOO + stacking + RFX-BMS/PXP (21.5) with user checkpoint
- [ ] 21-07-PLAN.md (Wave 5) — Winner refit with L2 scales (21.6)
- [ ] 21-08-PLAN.md (Wave 6) — Scale-fit audit with FDR-BH gate (21.7)
- [ ] 21-09-PLAN.md (Wave 7) — Model-averaged scale effects + optional M6b-subscale arm (21.8)
- [ ] 21-10-PLAN.md (Wave 8) — Manuscript tables + paper.qmd edits + master orchestrator cluster/21_submit_pipeline.sh (21.9)

#### Phase 22: Milestone v4.0 Closure — Verification, Traceability, Reproducibility

**Goal:** Close all documentation, verification, and traceability debt identified by `.planning/v4.0-MILESTONE-AUDIT.md` so v4.0 can be sealed via `/gsd:complete-milestone`. No new functional code. Every cluster-pending deliverable must be framed as "run `bash cluster/21_submit_pipeline.sh` cold-start" — no piecemeal resubmits, no one-off checks. Every verification statement must cite an automated, reproducible check (pytest, grep invariant, script with deterministic output) — never a manual inspection anecdote.

**Depends on:** Phase 21 (capstone shipped, 11/11 plans, master orchestrator live)

**Motivation.** The audit found that v4.0 is code-satisfied at 57/57 requirements but held back from formal close by three debt categories: (a) three phases (14, 15, 21) have no VERIFICATION.md; (b) DEER-01..04 and the entire Phase 21 BMS workflow are uncataloged in REQUIREMENTS.md; (c) stale progress markers in STATE.md / ROADMAP.md / PROJECT.md lag behind what shipped. The 2026-04-18 shell crash also left the Phase-21-completion STATE.md narrative uncommitted even though every 21-10 code commit landed. User extras (2026-04-19) further require: thesis doc moved to .gitignore, cluster rerun framing standardized on the cold-start pipeline entry point, and all closure checks be reproducible (no one-offs).

**Requirements (new, to be added in plan 22-03 and then back-filed here):**
CLOSE-01 (status-doc refresh), CLOSE-02 (VERIFICATION.md backfill for 14/15/21), CLOSE-03 (traceability extension with DEER + BMS), CLOSE-04 (reproducibility guard: pytest + grep-invariant tests that fail CI if closure claims drift).

**Success Criteria (what must be TRUE):**
  1. `git diff .planning/STATE.md` is empty after plan 22-01; the committed STATE.md narrative matches working-tree state (Phase 21 COMPLETE 11/11, Phase 22 active).
  2. ROADMAP.md Progress Table shows Phases 14, 15, 20, 21 with their true completion dates and cluster-pending status where applicable; no row still reads "Not started" for phases that shipped code.
  3. PROJECT.md "Active (v4.0)" requirements list is empty of shipped items; all shipped items migrated to "Validated" with v4 suffix.
  4. `.planning/phases/14-*/14-VERIFICATION.md`, `.planning/phases/15-*/15-VERIFICATION.md`, and `.planning/phases/21-*/21-VERIFICATION.md` exist, each with YAML frontmatter status + score and an Observable Truths table. 14 documents its cluster-pending items as `deferred_to_execution: bash cluster/21_submit_pipeline.sh`. 15 either has `15-03-SUMMARY.md` (documenting absorption into Phases 16/18/21) OR `15-03-PLAN.md` is removed with the absorption note preserved in 15-VERIFICATION.md.
  5. REQUIREMENTS.md traceability table grows from 57 to ≥ 71 rows: adds DEER-01..04 (Phase 20) and BMS-01..BMS-10 (Phase 21 nine-step pipeline + bms.py module). Every new ID has a phase mapping and a one-line satisfaction evidence. `grep -c "^| DEER-\|^| BMS-" .planning/REQUIREMENTS.md` returns ≥ 14.
  6. `Burrows_J_GDPA_Thesis.*` is gitignored (line present in `.gitignore`) and no longer appears in `git status --untracked-files=normal` output.
  7. **Reproducibility guard:** a new `scripts/closure/check_milestone_state.py` (or `validation/check_v4_closure.py`) exists and can be invoked via `python -m scripts.closure.check_milestone_state --milestone v4.0`, returning exit 0 when all closure invariants hold and exit 1 with a structured diff otherwise. Invariants must include: STATE.md cleanliness, ROADMAP progress-table consistency with on-disk SUMMARY.md counts, presence of the three new VERIFICATION.md files, REQUIREMENTS.md row count, thesis-gitignore guard, and an assertion that every cluster-pending deliverable in the audit references `cluster/21_submit_pipeline.sh` as its rerun entry point. The script is deterministic — running it twice with no intervening edits produces byte-identical stdout.
  8. **No one-off checks anywhere in Phase 22 outputs:** every verification statement in 14/15/21-VERIFICATION.md cites either (a) a pytest test that exercises the claim, (b) a grep invariant checked into the repo, or (c) the closure-state script in SC#7. No plain-English "I confirmed by reading the file" evidence lines.
  9. **Cluster-freshness framing:** every cluster-pending artifact referenced by closure docs is expressed as "produced by a cold-start run of `bash cluster/21_submit_pipeline.sh`" — never as "run sbatch cluster/12_mle_gpu.slurm separately" or equivalent piecemeal wording. The closure state script asserts this via a grep over `.planning/phases/14-*/14-VERIFICATION.md` + the audit's cluster_execution_pending block.
  10. `/gsd:audit-milestone` re-run after Phase 22 produces status `passed` with no critical gaps and an empty `tech_debt` list, or clearly documents any residual deferrals as scope-moved-to-v5.0.

**Plans:** 4 plans (see placeholder titles; planner will produce detailed PLAN.md files)

Plans:
- [ ] 22-01-PLAN.md (Wave 1) — Status-doc refresh (commit STATE.md update; refresh ROADMAP progress table; move PROJECT.md Active→Validated)
- [ ] 22-02-PLAN.md (Wave 1, parallel with 22-01) — Backfill 14/15/21 VERIFICATION.md with cluster-freshness framing + close 15-03 gap + gitignore thesis doc
- [ ] 22-03-PLAN.md (Wave 2, after 22-01 & 22-02) — Extend REQUIREMENTS.md with DEER + BMS requirement families; update traceability table
- [x] 22-04-PLAN.md (Wave 3, after 22-02 & 22-03) — Reproducibility guard script + pytest + grep invariants (enforces SC#7 + SC#8 + SC#9)

</details>

<details open>
<summary>🔄 v5.0 Empirical Artifacts & Manuscript Finalization (Phases 23-27) — STARTED 2026-04-19</summary>

**Milestone Goal:** Execute the Phase 21 Bayesian selection pipeline cold-start on the cluster to produce the full empirical artifact set (LOO/stacking/BMS/forest plots/winner-beta HDIs), sweep residual v4.0 tech debt (legacy qlearning import, legacy M2 K-bounds [1,7], `scripts/16b_bayesian_regression.py`, full load-side validation audit), cross-verify reproducibility against the v4.0 baseline via a seeded regression test (prior-predictive + Bayesian recovery only; stacking-weight / PXP stability checks descoped to save cluster cost), and finalize `paper.qmd` so `quarto render` compiles `paper.pdf` with real winner names, real Pareto-k-informed limitations, and real Level-2 effect estimates. Phase 14 cluster-execution items (K-02, K-03, GPU-01..03) explicitly deferred to v5.1 per /gsd:new-milestone scope decision.

**Scope locked via /gsd:new-milestone questioning (2026-04-19):**
- Endpoint: Manuscript-ready (paper.qmd auto-patched + forest plots/tables + limitations rewrite + `quarto render` passes)
- Orchestrator: Phase 21 only; Phase 14 deferred to v5.1
- Dead-code sweep: all 4 items (legacy qlearning, legacy M2 K-bounds, 16b, load-side validation)
- Cross-verify: prior-predictive + Bayesian recovery regression vs. v4.0 baseline only

**Execution Order:** Phase 23 (cleanup, pre-flight) → Phase 24 (cold-start cluster run — the main event) → Phase 25 (reproducibility regression against v4.0) → Phase 26 (manuscript finalization) → Phase 27 (milestone closure).

#### Phase 23: Tech-Debt Sweep & Pre-Flight Cleanup

**Goal:** Remove residual v4.0 tech debt before the Phase 24 cold-start run so the empirical artifacts are produced against a clean codebase. Four atomic code removals + one audit: (a) delete legacy qlearning hierarchical import path; (b) remove legacy M2 K-bounds [1,7] branch from `mle_utils.py`; (c) delete deprecated `scripts/16b_bayesian_regression.py`; (d) wire `config.load_fits_with_validation` into every downstream NetCDF consumer in scripts 15/16/17/18/21_*. Each removal lands as its own commit with pytest passing. No new scientific functionality — strictly cleanup.

**Depends on:** Phase 22 (v4.0 shipped; v4.0 closure guard green)

**Requirements:** CLEAN-01, CLEAN-02, CLEAN-03, CLEAN-04

**Success Criteria (what must be TRUE):**
  1. `grep -rn "from scripts.fitting.legacy" scripts/` returns zero live imports; any legacy qlearning file has been deleted (or the directory is empty); `pytest scripts/fitting/tests/ -v` passes.
  2. `grep -n "1, 7\|\[1,7\]\|K_BOUNDS_LEGACY" scripts/fitting/mle_utils.py` returns no matches; only Collins [2,6] K-bounds path remains; `parameterization_version` vocabulary no longer accepts "legacy"; affected unit tests updated and passing.
  3. `scripts/16b_bayesian_regression.py` deleted; `find . -name "16b*"` returns nothing; `docs/MODEL_REFERENCE.md` cross-reference updated to remove 16b mention; any cluster SLURM referencing 16b is deleted or updated.
  4. Every `az.from_netcdf` and `xr.open_dataset` call in `scripts/{15,16,17,18}.py`, `scripts/21_*.py`, and `validation/*.py` uses `config.load_fits_with_validation` wrapper; enforced via new `scripts/fitting/tests/test_load_side_validation.py` grep invariant test that fails CI if a bare NetCDF load is reintroduced.
  5. `python validation/check_v4_closure.py --milestone v4.0` still exits 0 on Phase-23-complete HEAD — cleanup did not break any v4.0 closure invariant.

**Plans:** 4 plans (draft; planner will finalize)

Plans:
- [ ] 23-01-PLAN.md (Wave 1) — Delete legacy qlearning import path + grep invariant test
- [ ] 23-02-PLAN.md (Wave 1, parallel with 23-01) — Remove legacy M2 K-bounds [1,7] branch from mle_utils.py + vocabulary update
- [ ] 23-03-PLAN.md (Wave 1, parallel with 23-01/23-02) — Delete scripts/16b_bayesian_regression.py + doc cross-reference update
- [ ] 23-04-PLAN.md (Wave 2, after 23-01/02/03) — Load-side validation audit: wire config.load_fits_with_validation into every downstream NetCDF consumer + grep invariant test

#### Phase 24: Cold-Start Pipeline Execution

**Goal:** Run `bash cluster/21_submit_pipeline.sh` end-to-end from a clean checkout — the main scientific event of v5.0. The 9-step `afterok`-chained pipeline produces prior-predictive → Bayesian recovery → baseline hierarchical fits (6 models) → convergence audit → PSIS-LOO + stacking + RFX-BMS → winner L2 refit → scale audit → model averaging → manuscript tables. Wall-clock expectation: ~50-96 GPU-hours total. Phase output is the full empirical artifact set that Phase 25/26 verify and Phase 26 manuscript-assembles.

**Depends on:** Phase 23 (clean codebase — no tech-debt pollution in cold-start outputs)

**Requirements:** EXEC-01, EXEC-02, EXEC-03, EXEC-04

**Success Criteria (what must be TRUE):**
  1. `bash cluster/21_submit_pipeline.sh` invoked from project root with a clean working tree completes the 9-step afterok chain with zero SLURM job failures; pre-flight pytest gate (`test_numpyro_models_2cov.py`) passes before any `sbatch` call.
  2. Every expected artifact exists: `output/bayesian/21_prior_predictive/{model}_prior_sim.nc` × 6, `21_recovery/{model}_recovery.csv` × 6, `21_baseline/{model}_posterior.nc` × 6 + `convergence_table.csv` + `convergence_report.md`, `21_l2/{winner}_posterior.nc` + `{winner}_beta_hdi_table.csv` for each winner, `21_l2/scale_audit_report.md` + `averaged_scale_effects.csv`, `output/bayesian/manuscript/{loo_stacking,rfx_bms,winner_betas}.{csv,md,tex}`, winner-specific forest plot PNGs.
  3. `output/bayesian/21_execution_log.md` logs SLURM JobID, wall-clock, CPU-hours, GPU-hours, max memory per step; total GPU-hours documented.
  4. `convergence_table.csv` shows ≥ 2 models meeting Baribault & Collins (2023) gate (R-hat ≤ 1.05 AND ESS_bulk ≥ 400 AND divergences = 0 AND BFMI ≥ 0.2); step 21.5 winner determination is `DOMINANT_SINGLE` or `TOP_TWO` (not `FORCED`, not `INCONCLUSIVE_MULTIPLE`).

**Plans:** 2 plans (draft; planner will finalize)

Plans:
- [ ] 24-01-PLAN.md (Wave 1) — Pre-flight checks (pytest gate, clean working tree, SLURM queue healthy) + cold-start submission of bash cluster/21_submit_pipeline.sh + per-step artifact monitoring
- [ ] 24-02-PLAN.md (Wave 2, after pipeline completes) — Post-run artifact audit + `output/bayesian/21_execution_log.md` writeup + convergence-gate verification + winner-determination recording

#### Phase 25: Reproducibility Regression & Closure-Guard Extension

**Goal:** Verify the Phase 24 cold-start outputs reproduce v4.0 baseline artifacts to within seeded-MCMC tolerance, catching any silent regression from the Phase 23 tech-debt sweep. Two regression tests: (a) seeded prior-predictive output byte-identical vs. v4.0 when same seed; within 3 SE when different seed; (b) Bayesian recovery Pearson r ≥ 0.80 for identifiable parameters matches v4.0 baseline; 95% HDI coverage within 5 pp of v4.0. Extends `check_v4_closure.py` with a new `check_v5_closure.py` enforcing v5.0 invariants (phase VERIFICATION files, REQUIREMENTS row count, v5.0 MILESTONES entry, manuscript artifacts exist).

**Depends on:** Phase 24 (needs Phase 21 artifacts for comparison)

**Requirements:** REPRO-01, REPRO-02, REPRO-03, REPRO-04

**Success Criteria (what must be TRUE):**
  1. `scripts/fitting/tests/test_v5_reproducibility.py` exists and passes; seeded prior-predictive regression against v4.0 baseline NetCDF produces byte-identical samples under same seed, posterior means within 3 SE under different seed.
  2. Bayesian recovery regression: Pearson r for kappa / kappa_total / kappa_share ≥ 0.80 matches v4.0 thresholds; 95% HDI coverage within 5 pp of v4.0 baseline; regression failure triggers exit 1 in pytest.
  3. `python validation/check_v4_closure.py --milestone v4.0` exits 0 on Phase-25 HEAD — no v4.0 invariants broken.
  4. `validation/check_v5_closure.py` exists with ≥ 10 invariants (phase VERIFICATION.md files for 23-27, REQUIREMENTS.md row count grows from 71 to ≥ 92, v5.0 entry in MILESTONES.md, EXEC artifacts exist on disk, `paper.pdf` exists); deterministic (byte-identical double-run diff empty); pytest `test_v5_closure.py` passes 3/3 (happy path, determinism, rejects-wrong-milestone).

**Plans:** 2 plans (draft; planner will finalize)

Plans:
- [ ] 25-01-PLAN.md (Wave 1) — test_v5_reproducibility.py with prior-predictive + Bayesian recovery regressions against v4.0 baseline
- [ ] 25-02-PLAN.md (Wave 2, after 25-01) — validation/check_v5_closure.py + scripts/fitting/tests/test_v5_closure.py (3/3 pytest pattern)

#### Phase 26: Manuscript Finalization

**Goal:** Finalize `paper.qmd` with real Phase 24 empirical data — winner names from `loo_stacking_results.csv`, real Pareto-k percentages in limitations, forest plots from real Phase 21.5 winners, and `quarto render` producing `paper.pdf` + `paper.html` without errors. Phase 21's plan 21-10 already auto-patches the Methods section with Quarto `{python}` inline refs; this phase verifies that patch landed with real data, rewrites the limitations section (replacing projected-from-research/PITFALLS.md text with actual Pareto-k diagnostics), regenerates forest plots for the true winner set, and runs `quarto render`.

**Depends on:** Phase 24 (needs real winners and Pareto-k from cold-start) + Phase 25 (trust artifacts are correct)

**Requirements:** MANU-01, MANU-02, MANU-03, MANU-04, MANU-05

**Success Criteria (what must be TRUE):**
  1. `paper.qmd` Methods `### Bayesian Model Selection Pipeline {#sec-bayesian-selection}` cites real stacking weights via `{python} winner_display` inline ref; `grep -n "M6b received the highest" paper.qmd` returns zero matches (placeholder replaced).
  2. Forest plots generated via `scripts/18_bayesian_level2_effects.py` for every model in the cold-start winner set; PNGs in `output/bayesian/figures/`; referenced in `paper.qmd` Results via `@fig-forest-{winner}` cross-refs that all resolve in Quarto render.
  3. Manuscript table artefacts `loo_stacking.{csv,md,tex}`, `rfx_bms.{csv,md,tex}`, `winner_betas.{csv,md,tex}` exist under `output/bayesian/manuscript/`; Quarto cross-refs `@tbl-loo-stacking`, `@tbl-rfx-bms`, `@tbl-winner-betas` resolve in render.
  4. Limitations section references real `pct_high_pareto_k` values from `loo_stacking_results.csv`; projected-from-PITFALLS.md placeholder text removed.
  5. `quarto render paper.qmd` exits 0 from project root with no errors; `_output/paper.pdf` and `_output/paper.html` exist and have non-zero size; `validation/check_v5_closure.py` verifies this via subprocess wrapper.

**Plans:** 3 plans (draft; planner will finalize)

Plans:
- [ ] 26-01-PLAN.md (Wave 1) — paper.qmd Methods + Results verification: winner names, Quarto inline refs, stacking table cross-refs all resolve with real data
- [ ] 26-02-PLAN.md (Wave 1, parallel with 26-01) — Forest plots for every winner + limitations rewrite with real Pareto-k diagnostics
- [ ] 26-03-PLAN.md (Wave 2, after 26-01 & 26-02) — `quarto render paper.qmd` + paper.pdf/paper.html existence check + any last-mile Quarto fixes

#### Phase 27: Milestone v5.0 Closure

**Goal:** Archive v5.0, update MILESTONES.md, extend the reproducibility guard, run `/gsd:audit-milestone`, and close. Mirrors the Phase 22 closure pattern from v4.0 — no new functional code, strictly documentation + verification artifacts.

**Depends on:** Phase 23, 24, 25, 26 (all shipped)

**Requirements:** CLOSE-01, CLOSE-02, CLOSE-03, CLOSE-04

**Success Criteria (what must be TRUE):**
  1. `.planning/MILESTONES.md` has a new "v5.0 Empirical Artifacts & Manuscript Finalization" entry matching v4.0 format (ship date, phase range 23-27, plan count, git range, accomplishments summary).
  2. Archives at `.planning/milestones/v5.0-ROADMAP.md`, `v5.0-REQUIREMENTS.md`, `v5.0-MILESTONE-AUDIT.md`; top-level ROADMAP.md v5.0 section collapsed into `<details>` block matching v4.0 pattern.
  3. `/gsd:audit-milestone` re-run on ship-commit produces `status: passed`, no critical gaps, empty `tech_debt` list (or explicit v5.1 deferral references to Phase 14 items, ArviZ 1.0, SBC workflow).
  4. `validation/check_v5_closure.py` exits 0 on ship-commit; byte-identical double-run diff confirms determinism; `scripts/fitting/tests/test_v5_closure.py` passes 3/3.

**Plans:** 3 plans (draft; planner will finalize)

Plans:
- [ ] 27-01-PLAN.md (Wave 1) — Status-doc refresh (STATE.md, PROJECT.md Active→Validated, MILESTONES.md v5.0 entry)
- [ ] 27-02-PLAN.md (Wave 2, after 27-01) — Archive v5.0-ROADMAP.md + v5.0-REQUIREMENTS.md + v5.0-MILESTONE-AUDIT.md; collapse top-level ROADMAP.md v5.0 section
- [ ] 27-03-PLAN.md (Wave 3, after 27-02) — `/gsd:audit-milestone` run + final closure-guard verification + ship-commit

</details>

## Progress

**Execution Order:**
Phases execute in numeric order: 13 → 14 → 15 → 16 → 17 → 18 → 19 → 20 → 21 → 22 → 23 → 24 → 25 → 26 → 27

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 1. Perseveration Extension | v1.0 | 2/2 | Complete | 2026-01-30 |
| 2. MLE Infrastructure | v1.0 | 2/2 | Complete | 2026-01-30 |
| 3. Model Comparison | v1.0 | 2/2 | Complete | 2026-01-30 |
| 4. Regression Visualization | v2.0 | 2/2 | Complete | 2026-02-06 |
| 5. Parameter Recovery & PPC | v2.0 | 5/5 | Complete | 2026-02-06 |
| 6. Cluster Monitoring | v2.0 | 0/TBD | Deferred | - |
| 7. Publication Polish | v2.0 | 0/TBD | Deferred | - |
| 8. M5 RL Forgetting | v3.0 | 2/2 | Complete (recovery pending) | 2026-04-02 |
| 9. M6a Stimulus-Specific Perseveration | v3.0 | 2/2 | Complete | 2026-04-02 |
| 10. M6b Dual Perseveration | v3.0 | 2/2 | Complete (recovery pending) | 2026-04-03 |
| 11. M4 LBA Joint Choice+RT | v3.0 | 3/3 | Complete (recovery pending) | 2026-04-03 |
| 12. Cross-Model Integration | v3.0 | 3/3 | Complete | 2026-04-03 |
| 13. Infrastructure Repair & Hierarchical Scaffolding | v4.0 | 5/5 | Complete | 2026-04-12 |
| 14. Collins K Refit + GPU LBA Batching | v4.0 | 3/3 | Complete (cluster refit pending) | 2026-04-12 |
| 15. M3 Hierarchical POC with Level-2 Regression | v4.0 | 2/3 | Complete (15-03 absorbed — see 15-VERIFICATION.md) | 2026-04-12 |
| 16. Choice-Only Family Extension + Subscale L2 | v4.0 | 7/7 | Complete | 2026-04-13 |
| 17. M4 Hierarchical LBA | v4.0 | 3/3 | Complete | 2026-04-13 |
| 18. Integration, Comparison, and Manuscript | v4.0 | 5/5 | Complete | 2026-04-13 |
| 19. Associative Scan Likelihood Parallelization | v4.0 | 3/3 | Complete | 2026-04-14 |
| 20. DEER Non-Linear Parallelization (Research) | v4.0 | 3/3 | Complete | 2026-04-14 |
| 21. Principled Bayesian Model Selection Pipeline | v4.0 | 11/11 | Complete | 2026-04-18 |
| 22. Milestone v4.0 Closure | v4.0 | 4/4 | Complete | 2026-04-19 |
| 23. Tech-Debt Sweep & Pre-Flight Cleanup | v5.0 | 0/4 | Not started | — |
| 24. Cold-Start Pipeline Execution | v5.0 | 0/2 | Not started | — |
| 25. Reproducibility Regression & Closure-Guard Extension | v5.0 | 0/2 | Not started | — |
| 26. Manuscript Finalization | v5.0 | 0/3 | Not started | — |
| 27. Milestone v5.0 Closure | v5.0 | 0/3 | Not started | — |
