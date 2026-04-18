# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-11)

**Core value:** Correctly dissociate perseverative responding from learning-rate effects (alpha-) to accurately identify whether post-reversal failures reflect motor perseveration or outcome insensitivity
**Current focus:** v4.0 — Hierarchical Bayesian Pipeline & LBA Acceleration (Phase 20 in progress, 2/3 plans done). **Phase 21 added 2026-04-18** — Principled Bayesian Model Selection Pipeline (linear 9-step workflow, PSIS-LOO + stacking + RFX-BMS/PXP, replaces MLE-preselected winner approach).

## Current Position

Milestone: v4.0 Hierarchical Bayesian Pipeline & LBA Acceleration
Phase: 21 of 21 (Principled Bayesian Model Selection Pipeline) — In progress
Plan: 5 of 11 complete (21-01, 21-02, 21-03, 21-04, 21-11 done 2026-04-18); Phase 20 also still has plan 20-03 outstanding
Status: Step 21.3 baseline hierarchical fit runner live — `scripts/21_fit_baseline.py` delegates to `fit_bayesian.main()` with `--output-subdir 21_baseline` forced via sys.argv rewrite; post-fit `expected.exists()` shim exits 1 on convergence-gate failure so SLURM `--dependency=afterok` chains block instead of silently skipping. Patches `save_results` + `run_posterior_predictive_check` + `write_bayesian_summary` with `output_subdir`/`ppc_output_dir` kwargs (Approach A, backward-compat preserved when None). `cluster/21_3_fit_baseline.slurm` (10h/48G/4-CPU) submittable 6 times in parallel. Prior: Step 21.2 recovery runner live, M3/M5/M6a accept optional `covariate_iesr`, RFX-BMS infrastructure from 21-02 ready, Step 21.1 prior-predictive gate runner live.
Last activity: 2026-04-18 — Plan 21-04 complete: `scripts/21_fit_baseline.py` (225 lines, thin wrapper + load-bearing convergence-gate-surface shim), `cluster/21_3_fit_baseline.slurm` (159 lines). Patched 3 modules with `output_subdir` plumbing (fit_bayesian `save_results` + main()`--output-subdir` CLI; bayesian_diagnostics `run_posterior_predictive_check` +ppc_output_dir; bayesian_summary_writer `write_bayesian_summary` +output_subdir). Task 1 commit `1b39606`, Task 2 commit `8d27407`. Executor hit user-level rate limit after Task 1 landed + Task 2 files on disk but before commit/SUMMARY; orchestrator verified on-disk files matched plan spec, committed Task 2, wrote SUMMARY.md. Prior: Plan 21-03 complete: `scripts/21_run_bayesian_recovery.py` (956 lines, two CLI modes single-subject + aggregate), `scripts/fitting/tests/test_bayesian_recovery.py` (352 lines, 7 tests), `cluster/21_2_recovery.slurm` (129 lines, array 1-50 no autopush), `cluster/21_2_recovery_aggregate.slurm` (116 lines, aggregate + autopush, master orchestrator chains via afterok). Task 1 commit `05f1b79`, Task 2 commit `205f480`. Auto-fixed 1 bug in aggregate mode (`_safe_pearson_r` zero-variance check used `np.std == 0`, replaced with `np.ptp == 0` to avoid floating-point drift at 1e-17). MCMC smoke test fits qlearning on 3 blocks × 20 trials in 16 s on CPU. Prior: Plan 21-11 complete: `scripts/fitting/numpyro_models.py` (wmrl_m3/m5/m6a_hierarchical_model extended with `covariate_iesr` kwarg + guard + `beta_iesr_{target}` Normal(0,1) site + probit shift sum), `scripts/fitting/level2_design.py` (+ `build_level2_design_matrix_2cov` + `COVARIATE_NAMES_2COV`), `scripts/fitting/tests/test_numpyro_models_2cov.py` (635 lines, 4 test gates × 10 parametrizations). Task 1 commit `7503316`, Task 2 commit `49a84ff`. Plan 21-01: `scripts/21_run_prior_predictive.py` (648 lines) + `cluster/21_1_prior_predictive.slurm` (129 lines); commits 6a0fdad, 8fa1ebd. Plan 21-02: `scripts/fitting/bms.py` (335 lines) + `scripts/fitting/tests/test_bms.py` (158 lines); commits 0cef164, 10ca205, 9c0f5b1.

### Phase 21 Decisions (2026-04-18)

- **Plan 21-04 (Step 21.3 baseline hierarchical fit runner):** `output_subdir` plumbed through 3 modules — `save_results` (fit_bayesian.py:577), `write_bayesian_summary` (bayesian_summary_writer.py:155), `run_posterior_predictive_check` via new `ppc_output_dir` kwarg (bayesian_diagnostics.py:645, Approach A). CLI flag `--output-subdir` added to `fit_bayesian.main()` (default None preserves Phase 16 backward-compat). `scripts/21_fit_baseline.py` is a thin wrapper: rewrites `sys.argv` to force `--output-subdir 21_baseline` and calls `fit_main()`, then runs a load-bearing `expected.exists()` check on `output/bayesian/21_baseline/{model}_posterior.nc`. **Load-bearing exit-1 shim**: without it, `save_results` returning None on convergence-gate failure leaves `fit_main()` at exit 0, which lets SLURM `--dependency=afterok` chains silently skip the failed model and let downstream steps "complete" on missing inputs. The shim converts silent gate-fail into a real SLURM failure so Phase 21 SC #2 ("all 6 models converge or are explicitly dropped with documented reason") actually holds. `cluster/21_3_fit_baseline.slurm` is 10h/48G/4-CPU/comp partition, per-MODEL JAX cache, NUMPYRO_HOST_DEVICE_COUNT=4, JAX_PLATFORMS=cpu; submit 6 times in parallel (no cross-model conflicts under `21_baseline/` subdir).
- **Plan 21-03 (Step 21.2 Bayesian parameter recovery):** Two CLI modes (single-subject + aggregate) in one script — the single-subject mode is dispatched by a SLURM array (1-50) for embarrassingly-parallel per-subject fits (~30 min wall-clock per model, vs. ~21h sequential). Aggregate mode runs once post-array. Scope decision locked: **baseline (no-covariate) inference path only**; the 2-cov L2 hook from plan 21-11 is gated via the `test_recovery_2cov_m3` pytest rather than tripling cluster cost (~150 additional MCMC jobs for a confirmatory test). Pass criterion (kappa-family only): `r >= 0.80 AND 95% HDI coverage >= 0.90`; all other parameters labelled `descriptive only` per quick-005 MLE recovery findings (r = 0.21 to 0.77 floor). Prior sampling uses `sigma_scale=0.2` (matches `HalfNormal(0.2)` on `sigma_pr`) and `phi_approx` transform to mirror `sample_bounded_param` exactly. RNG: base seed folded with subject_idx via `jax.random.fold_in`; separate `sim_seed` and `mcmc_seed` within each task. **Autopush gated**: only the aggregator script sources `cluster/autopush.sh`; the 50-task array does NOT (would race 50 concurrent `git push` calls). Master orchestrator (plan 21-10) chains the two scripts via `--dependency=afterok:$ARRAY_JOBID`. Auto-fix during execution: `_safe_pearson_r` zero-variance check switched from `np.std == 0` to `np.ptp == 0` to avoid floating-point drift at 1e-17 on constant posterior-mean vectors.
- **Plan 21-11 (2-cov L2 hook for M3/M5/M6a):** Approach A additive kwarg `covariate_iesr: jnp.ndarray | None = None` chosen over Approach B generic-matrix replacement — keeps blast radius to the three model functions + `level2_design.py` + test file; zero changes to `fit_bayesian._fit_stacked_model` or any other caller. `beta_iesr_{target}` uses same `Normal(0, 1)` prior as `beta_lec_{target}` for symmetry. Guard raises `ValueError` if `covariate_iesr` passed without `covariate_lec` (prevents silent LEC drop). 2-cov design does NOT residualize — raw z-scored totals only (LEC/IES-R r ~ 0.3 in N=138 is well below collinearity threshold; orthogonalization only principled when design cond# > 30, as in Phase 16's 4-cov subscale case). Phase 21 Option C pattern locked: **code all model variants upfront; pipeline dispatches the one the winner needs** — M3/M5/M6a 2-cov path now ready; M6b already has 4-cov subscale; M1/M2 bypass L2 entirely.
- **Plan 21-11 recovery test fallback (locked):** Unified simulator `simulate_agent_fixed` does not expose `kappa` as an Agent constructor parameter, so the recovery test uses a manual NumPy forward-sim of the M3 block (mirrors `wmrl_m3_block_likelihood` semantics: hybrid WM/RL + epsilon + probability-mixing perseveration against one-hot choice kernel of `last_action`). Pre-approved as fallback in plan 21-11 Task 2 action text; documented in test module docstring.
- **Plan 21-01 (prior-predictive gate):** Use NumPyro `Predictive` (not manual sampling) so the full hierarchical structure (`mu_pr`, `sigma_pr`, `z`, transformed `theta`) is honored. One `(draw, participant)` pair per draw (not cartesian 500 × 138) to keep wall-clock under 2 min per model. Reuse real participant trial templates (stimuli, set_sizes, block structure) per Baribault. Exit 1 on FAIL so the SLURM pipeline short-circuits before MCMC.
- **Plan 21-01 simulator quirk:** `NUM_STIMULI=7` in the NumPy simulator accommodates the 1-indexed stimulus values in `task_trials_long.csv` — the JAX likelihoods silently clip out-of-bounds reads (stimulus=6 aliases to row 5), but plain NumPy needs the explicit row. Tracked as a known follow-up for when `prepare_stacked_participant_data` ever explicitly zero-indexes stimuli.

### v4.0 Decisions (pre-refit, 2026-04-17)

- **Canonical analysis cohort (locked):** `config.get_analysis_cohort()` implements the intersection of three gates: (a) task completeness ≥ `MIN_TRIALS_THRESHOLD=400` trials and `MIN_BLOCKS=8` main-task blocks; (b) performance check — overall accuracy ≥ `MIN_OVERALL_ACCURACY=0.40` AND mean accuracy over last `LATE_BLOCK_N=5` main-task blocks ≥ `MIN_LATE_BLOCK_ACCURACY=0.50`; (c) scale completeness — non-NA for every column in `REQUIRED_SURVEY_COLUMNS` (lec_total, ies_total, ies_intrusion, ies_avoidance, ies_hyperarousal). Current cohort size: **N=138** (vs N=154 task-only, N=160 scale-only). 16 participants dropped by late-block learning check.
- **Principled priors (locked):** All `mu_prior_loc` in `PARAM_PRIOR_DEFAULTS` now 0.0 (prior mean = Φ(0) = 0.5 on bounded scale; 95% prior interval ~[0.02, 0.98] given `sigma_pr ~ HalfNormal(0.2)`). Motivation: previous MLE-calibrated values (e.g. kappa=-2.0 implying prior mean 0.023) were empirical-Bayes informative and conservative for L2 null-testing. Legacy values retained in `_PRIOR_LEGACY_MLE_CALIBRATED` dict for sensitivity analyses. See `docs/SCALES_AND_FITTING_AUDIT.md` §3.
- **Cohort filter wired into `fit_bayesian.load_and_prepare_data` (locked):** `use_cohort=True` default. `--no-cohort` CLI flag available for diagnostic/legacy runs. Printed cohort breakdown at load time.
- **Stick-breaking rationale + identifiability paragraph added to paper.qmd (locked):** Methods §Model Fitting now explicitly argues for stick-breaking vs. inequality-constrained joint κ/κ_s sampling and names the three identifiability safeguards (uninformative κ_share prior, auto-bump, recovery validation).
- **Posterior-vs-MLE figure referenced in paper.qmd (locked):** `@fig-posterior-vs-mle` points to `figures/m6b_posterior_vs_mle.png`, to be generated from the output of `validation/compare_posterior_to_mle.py` on pipeline rerun.
- **Scale distributions figure referenced (locked):** `@fig-scale-distributions` points to `figures/scale_distributions.png`, produced by `scripts/analysis/trauma_scale_distributions.py` on pipeline rerun.
- **Cluster/GPU lessons doc (locked):** `docs/CLUSTER_GPU_LESSONS.md` captures the full v4.0 operational writeup — fully-batched vmap pattern with code examples, chain_method decision tree, pscan tradeoffs (CPU 3.7× slower, qlearning 0.28×), JAX/NumPyro pitfalls with context, SLURM templates (CPU-4-chain, single-GPU, multi-GPU), JIT-optimal coding patterns, debugging checklist.  Cross-referenced from `docs/README.md`.

## Hierarchical Bayesian Architecture (current)

### Single-stage joint inference (NOT two-stage PEB)

- All models fit as **single-stage hierarchical Bayesian** via NumPyro NUTS. Individual parameters, group priors, and Level-2 regression coefficients are sampled **simultaneously** in one MCMC run. This is more principled than two-stage PEB (SPM-style) because (a) uncertainty propagates correctly — no plug-in bias from using posterior means as point estimates in a second regression stage, (b) shrinkage happens automatically via partial pooling, (c) L2 coefficients are tempered by L1 uncertainty.
- **Trauma parameters ARE fit together with model parameters** — they are not a separate post-hoc step.
- The refactored fully-batched likelihood emits a single ``numpyro.factor("obs", per_participant_ll.sum())``; the prior 154-per-participant factor-site version was also joint, just slower.

### Level-2 regression — coverage table

| Model | L2 design | Beta sites | Hooked into |
|---|---|---|---|
| M1 qlearning | none (L2 not supported) | 0 | `covariate_lec=None` guard raises |
| M2 wmrl | none (L2 not supported) | 0 | `covariate_lec=None` guard raises |
| M3 wmrl_m3 | LEC total → kappa **[+ optional IES-R total, Phase 21]** | 1 (`beta_lec_kappa`) **or 2 (`beta_lec_kappa` + `beta_iesr_kappa`)** | `wmrl_m3_hierarchical_model` |
| M5 wmrl_m5 | LEC total → kappa **[+ optional IES-R total, Phase 21]** | 1 (`beta_lec_kappa`) **or 2 (`beta_lec_kappa` + `beta_iesr_kappa`)** | `wmrl_m5_hierarchical_model` |
| M6a wmrl_m6a | LEC total → kappa_s **[+ optional IES-R total, Phase 21]** | 1 (`beta_lec_kappa_s`) **or 2 (`beta_lec_kappa_s` + `beta_iesr_kappa_s`)** | `wmrl_m6a_hierarchical_model` |
| M6b wmrl_m6b | LEC total → kappa_total + kappa_share | 2 (`beta_lec_kappa_total`, `beta_lec_kappa_share`) | `wmrl_m6b_hierarchical_model` |
| M6b-subscale | 4 covariates × 8 params (full design) | **32** (`beta_{cov}_{param}`) | `wmrl_m6b_hierarchical_model_subscale` |

**Plan 21-11 Option C note:** M3/M5/M6a now accept an optional second covariate `covariate_iesr: jnp.ndarray | None = None` alongside `covariate_lec`. When both are provided, `beta_iesr_{target}` is sampled with `Normal(0, 1)` prior and both shifts are summed on the probit scale. Guard raises `ValueError` if `covariate_iesr` is passed without `covariate_lec`. Design-matrix builder: `build_level2_design_matrix_2cov(metrics, participant_ids)` returns `(N, 2)` z-scored `[lec_total, iesr_total]`. Test coverage: `scripts/fitting/tests/test_numpyro_models_2cov.py` (9 fast tests + 1 slow recovery smoke test).

### Subscale design matrix (M6b-subscale only)

Four predictors, condition number 11.3 (well under 30 target):
1. `lec_total` — LEC-5 total events (z-scored)
2. `iesr_total` — IES-R total score (z-scored)
3. `iesr_intr_resid` — IES-R intrusion, Gram-Schmidt residualized vs. total (z-scored)
4. `iesr_avd_resid` — IES-R avoidance, Gram-Schmidt residualized vs. total (z-scored)

Hyperarousal residual excluded (exact linear dependence: the three residuals sum to zero). See `scripts/fitting/level2_design.py`.

### What PEB-style two-stage would add (not currently implemented)

- A separate second-stage regression taking the L1 posterior as input and regressing against additional covariates that were not in the MCMC design matrix. Useful for exploratory analyses *after* the main fit completes.
- Functionally already achievable by running `18_bayesian_level2_effects.py` on the posterior NetCDF with extracted per-participant samples, though this is a descriptive/visualization script, not a re-fit.
- Not strictly necessary for v4.0 Phase 15/16 validation gate (kappa × LEC-5 signal) because LEC is already in the joint design.

## Issue 1 Fully-Batched Rollout (2026-04-17, post-Phase-20)

### What changed

- 6 new `*_fully_batched_likelihood` functions in `scripts/fitting/jax_likelihoods.py` (M3 added earlier in commit `aeaa208`; qlearning, wmrl, M5, M6a, M6b added in `c54ee6c`).
- All 6 `*_hierarchical_model*` bodies in `scripts/fitting/numpyro_models.py` refactored from Python for-loop emitting `obs_p{pid}` factors to single `numpyro.factor("obs", per_participant_ll.sum())` via nested `jax.vmap` (outer=participants, inner=blocks).
- `scripts/fitting/fit_bayesian.py`: `_FULLY_BATCHED_MODELS` tuple extended from M3-only to all 6 choice-only models; `stacked_arrays` pre-computed once per MCMC call.
- New `cluster/13_bayesian_multigpu.slurm` — requests `--gres=gpu:4` so `_select_chain_method` returns `"parallel"` (true `jax.pmap` across GPUs) instead of `"vectorized"` (time-multiplexed on 1 device).

### Validation status

| Test | Scope | Status |
|---|---|---|
| Agreement on synthetic (N=5 × 3 draws) | 6 models | PASS, max rel err 2.06e-07 |
| Agreement on real N=154 using MLE params | 6 models × 154 ppts | PASS, max rel err **8.84e-07** |
| End-to-end MCMC dispatch (local CPU) | 4 refactored models | PASS, zero `obs_p{pid}` sites remaining |
| Cluster smoke test (fb_smoke job 54882294) | all 6 models, N=154, 5 warmup + 5 samples | PASS, 109–132 s each |
| Per-iter GPU speedup (qlearning) | before vs after | ~6 min/iter → ~0.5 s/iter (~700×) |

### Why the speedup is larger than hypothesized

Original expectation: 36× from collapsing 154 dispatches to 1. Actual: ~700× per iter on qlearning. The extra gain comes from XLA fusing the entire likelihood computation into a single kernel once the Python for-loop is out of the way — before, each factor was compiled as an independent sub-graph; now there's one fused graph that the optimizer can reorder and vectorize across all 154 participants simultaneously. This also explains why prior runs reported "GPU peak memory: 0.1 MB" — the GPU was idle waiting for Python dispatches 99% of the time.

## Production Refit Readiness

### Diagnostics preserved (unchanged by refactor)

- **NumPyro**: R-hat, ESS bulk, ESS tail, divergences, tree-depth saturation, E-BFMI (via `mcmc.summary_frame()`).
- **ArviZ**: WAIC, LOO, `az.compare()` with stacking weights.
- **`run_inference_with_bump` auto-bump**: retries at `target_accept_prob` 0.80 → 0.95 → 0.99 if divergences persist. Convergence gate: `num_divergences == 0`.
- **Pointwise log-lik** (`compute_pointwise_log_lik`, `bayesian_diagnostics.py:291`): reads `mcmc.get_samples()` by parameter name, not factor site names — transparent to the `obs_p{pid}` → `obs` change.
- **Hessian diagnostics are MLE-only** (`fit_mle.py:1545–1631`). Bayesian uses posterior-based uncertainty. Neither path affected by the refactor.

### Baseline for comparison

- **No cached hierarchical Bayesian posteriors exist** (`output/bayesian/` contains only pscan benchmarks and the IES-R collinearity audit). The upcoming refit is the **first** production hierarchical Bayesian run to complete.
- Sanity baselines: MLE individual fits in `output/mle/{model}_individual_fits.csv` with `parameterization_version=v4.0-K[2,6]-phiapprox-stickbreaking` (quick-006).
- Expected: posterior means near MLE point estimates, with hierarchical shrinkage pulling extreme values toward group means.

### v4.0 Phase 15 POC validation gate (pending)

The original motivation for hierarchical inference: reproduce the MLE-derived kappa × LEC-5 signal under partial-pooling. From quick-006 Task 4:
- M3: 3 FDR-BH survivors (phi × IES-R Hyperarousal, kappa × LEC-5 Total, phi × IES-R Total)
- M6b: `kappa_total × LEC-5` strongest uncorrected (p=0.0028), does not survive FDR-BH
- The kappa × LEC-5 pattern across M3 and M6b is the most scientifically credible because kappa is the recoverable parameter in both

**Pass criterion**: `beta_lec_kappa` (M3) or `beta_lec_kappa_total` (M6b) 95% HDI excludes zero under hierarchical inference.

### Outstanding validation items (to run after refit)

1. **Convergence diagnostics per model**: R-hat < 1.05, ESS bulk > 400, divergences == 0 (auto-bump will retry up to target_accept=0.99).
2. **Posterior vs MLE agreement spot-check**: per-participant posterior means within 2 SD of MLE point estimates (expect shrinkage on poorly-identified params — alpha_pos/alpha_neg/phi/rho/capacity per recovery results).
3. **L2 coefficient HDI check**: 95% HDI for `beta_lec_kappa` / `beta_lec_kappa_total` — does it exclude zero?
4. **Model comparison**: WAIC/LOO via `az.compare(ic='loo', method='stacking')` — does M6b remain the winning model under hierarchical shrinkage? (MLE AIC says it does; Bayesian stacking weights may disagree.)
5. **Subscale signal discovery**: M6b-subscale has 32 beta sites; which (covariate × param) combinations have HDI excluding zero after Bonferroni or FDR?
6. **Parameter recovery under hierarchy**: re-run recovery on M6b with hierarchical MCMC (not MLE) to check if kappa_total/kappa_share recovery (currently r=0.997/0.931 under MLE) holds. Base RLWM params are unrecoverable under MLE (r=0.21–0.77); hierarchical shrinkage may help but cannot manufacture identifiability.

### Known limitations (carry through to results)

- **Capacity (K) identifiability is structural** (quick-005): K posterior will be prior-dominated for most participants. Reported only as descriptive.
- **M6b stick-breaking geometry**: kappa_share posterior can be bimodal when kappa_total is near zero (share becomes non-identifiable). Handled by auto-bump to `target_accept=0.99` if divergences appear.
- **IES-R hyperarousal subscale unavailable for L2** (linear dependence). Committed decision, see 16-01.
- **Number of participants: N=154 for task-data completeness, N=160 for surveys**; hierarchical fit uses N=154 (sorted overlap).

Progress: [████████░░] ~72% (29/~40 plans across Phases 13-20 + Phase 21)

### v4.0 Phase Structure

| Phase | Goal | Requirements | Count |
|---|---|---|---|
| 13 | Infrastructure Repair & Hierarchical Scaffolding (fix P0 import, numpyro scaffolding, Collins K research) | INFRA-01..08, K-01 | 9 |
| 14 | Collins K Refit + GPU LBA Batching (K-02/03 refit + fit_all_gpu_m4) | K-02..03, GPU-01..03 | 5 |
| 15 | M3 Hierarchical POC with Level-2 Regression (validation gate) | HIER-01, HIER-07..10, L2-01 | 6 |
| 16 | Choice-Only Family Extension + Subscale L2 (M1/M2/M5/M6a/M6b + full subscale Level-2) | HIER-02..06, L2-02..08 | 12 |
| 17 | M4 Hierarchical LBA (user-committed despite research recommendation to descope) | M4H-01..06 | 6 |
| 18 | Integration, Comparison, and Manuscript (schema-parity flag flip, WAIC/LOO, paper revision) | CMP-01..04, MIG-01..05, DOC-01..04 | 13 |

**Coverage:** 51/51 requirements mapped (100%), zero orphans.

### Post-Refit Reality (N=154, quick-006)

- Winning model flipped from M5 to **M6b** (dual perseveration with stick-breaking kappa_share).
  - Aggregate AIC: M6b 143324.93 < M5 143897.82 < M6a 144771.59 < M3 144865.92 < M2 147328.17 < M1 152143.11
  - Aggregate BIC agrees: M6b is also rank 1 on BIC; AIC and BIC orderings are identical.
  - Akaike weight of M6b is effectively 1.0.
- Per-participant AIC winners (N=154): M6b 55 (35.7%), M5 41 (26.6%), M6a 38 (24.7%), M3 15 (9.7%), M2 3 (1.9%), M1 2 (1.3%).
- M6b parameter recovery (quick-005 outputs, N=50 synthetic):
  - kappa_total r=0.9971 PASS, kappa_share r=0.9311 PASS
  - alpha_pos r=0.598 FAIL, alpha_neg r=0.516 FAIL, phi r=0.442 FAIL, rho r=0.629 FAIL, capacity r=0.213 FAIL (worst), epsilon r=0.772 FAIL (close)
- Practical implication: trust kappa-level inferences; treat base RLWM parameters as individual-level descriptors only, not identified traits. **This is the core motivation for v4.0 hierarchical shrinkage.**
- Trauma-parameter regressions (quick-006 Task 4, all 7 models, within-model FDR-BH + Bonferroni):
  - Only M3 produces FDR-BH survivors (3 of 42 tests): phi x IES-R Hyperarousal, kappa x LEC-5 Total events, phi x IES-R Total.
  - M6b: 7 uncorrected hits, 0 FDR-BH, 0 Bonferroni. Strongest M6b hit is kappa_total x LEC-5 (p=0.0028 uncorrected, p_fdr=0.135).
  - The kappa x LEC-5 pattern across M3 and M6b is the most scientifically credible signal because kappa is the recoverable parameter in both. **v4.0 Phase 15 must reproduce this under hierarchical inference as the POC validation gate.**

## Performance Metrics

**v1 Milestone:**
- Total plans completed: 6
- Average duration: 25 min
- Total execution time: 2.5 hours

**v2 Milestone:**
- Total plans completed: 7
- Average duration: 20 min
- Total execution time: 140 min

**v3 Milestone:**
- Total plans completed: 6 (incl. 1 gap closure)
- Average duration: 20 min
- Total execution time: 117 min

**v4 Milestone:**
- Total plans completed: 3
- Average duration: ~11 min (13-01, 13-03), ~? min (13-02)
- Total execution time: ongoing

## Accumulated Context

### v4.0 Decisions (21-11 completed 2026-04-18)

- **Phase 21 Option C infrastructure for M3/M5/M6a 2-cov L2 locked:** Approach A additive kwarg `covariate_iesr: jnp.ndarray | None = None` added to `wmrl_m3_hierarchical_model`, `wmrl_m5_hierarchical_model`, `wmrl_m6a_hierarchical_model` alongside existing `covariate_lec`. Approach B (generic `covariate_matrix` + `names` kwargs) rejected — would have required rewrites of `fit_bayesian._fit_stacked_model`, `21_run_prior_predictive.py`, `21_run_bayesian_recovery.py`, `21_fit_baseline.py` and any notebook callers. Approach A preserves zero changes outside the three model functions and `level2_design.py`.
- **`beta_iesr_{target}` prior (locked):** `Normal(0, 1)` — identical to `beta_lec_{target}` for prior symmetry across covariates. No directional bias injected through the prior. Target parameter: `kappa` for M3/M5, `kappa_s` for M6a.
- **Guard semantics (locked):** `covariate_iesr is not None and covariate_lec is None` raises `ValueError("covariate_iesr provided without covariate_lec. …")` with explicit message listing all three valid configurations (both None, both provided, LEC-only). Placed at function top before any `numpyro.sample` calls.
- **2-cov design does NOT residualize (locked):** `build_level2_design_matrix_2cov` returns raw z-scored `[less_total_events, ies_total]`. LEC/IES-R Pearson r is ~0.3 in the canonical N=138 cohort — well below the multicollinearity threshold. Orthogonalization is only principled for the 4-cov subscale design where condition number matters (Phase 16 logic).
- **Recovery test uses manual NumPy forward-sim (locked):** The unified simulator framework (`scripts/simulations/unified_simulator.py`) does not expose `kappa` as an Agent constructor parameter or support per-participant kappa shifts. Test uses `_simulate_m3_block_numpy` that mirrors `wmrl_m3_block_likelihood` semantics exactly (hybrid WM/RL + epsilon + probability-mixing perseveration against one-hot choice kernel of `last_action`). Pre-approved as fallback in plan 21-11 Task 2 action text.
- **Backward-compat preserved:** Phase 16 callers passing only `covariate_lec=<v>` (including `fit_bayesian._fit_stacked_model`, `21_run_bayesian_recovery.py`, all existing tests) are 100% unchanged. `beta_iesr_*` sample sites only appear in the trace when `covariate_iesr` is explicitly passed.

### v4.0 Decisions (21-02 completed 2026-04-18)

- **RFX-BMS + PXP module locked (`scripts/fitting/bms.py`):** Public signature `rfx_bms(log_evidence, alpha0=1.0, max_iter=1000, tol=1e-4, n_xp_samples=1_000_000, seed=42) -> {alpha, r, xp, bor, pxp}`. Strict `(n_subjects, n_models)` 2-D input contract (ValueError on ndim!=2 or non-finite). Faithful port of mfit/bms.m with Dirichlet-categorical VB E-step + 1 M Monte-Carlo XP sampling.
- **Null-model free energy formulation (locked, deviates from plan 21-02 spec):** The plan described F0 as a Dirichlet-posterior VB free energy with `alpha_null = alpha0 + n/K` per component. Under uniform log-evidence the heterogeneous posterior converges to the same concentrations, pinning BOR at 0.5 and losing the "null supported" signal. Replaced with Rigoux 2014 eq. A1 fixed-r form: `F0 = Σ_n logsumexp(log_evidence[n]) - N*log(K)`, which is the data log-likelihood under a delta-posterior at `r = 1/K` (no Dirichlet KL because r is not inferred). Canonical Phase 21 formulation — documented in `_bor` docstring. Produces BOR ~ 1 on uniform data (null supported), BOR ~ 0 on dominant/heterogeneous data (null rejected), matching published behaviour.
- **Dirichlet KL in `_vb_free_energy` now uses standard closed form:** `KL(Dir(α) || Dir(α0)) = gammaln(sum α) - Σ gammaln(α) - gammaln(sum α0) + Σ gammaln(α0) + Σ(α-α0)(ψ(α) - ψ(sum α))`. Earlier version had inverted signs on the `gammaln` terms; test-driven discovery.
- **PXP formula verified numerically to 1e-12:** `pxp = (1 - bor) * xp + bor / K` (Rigoux 2014). XP and PXP algebraically sum to 1 within 1e-10 on every call.
- **No import-time side effects:** `python -c "import scripts.fitting.bms"` silent. Module ready to be imported by Plan 21-05 (LOO+stacking+BMS orchestrator).

### v4.0 Decisions (16-01 completed 2026-04-13)

- **4-predictor L2 design matrix locked:** `lec_total`, `iesr_total`, `iesr_intr_resid`, `iesr_avd_resid`. The hyperarousal residual is excluded because `ies_total = ies_intrusion + ies_avoidance + ies_hyperarousal` exactly in the dataset; all three subscale residuals sum to zero (rank-deficient 3-residual matrix, condition number ~2.4e15). Only 2 subscale residuals are linearly independent.
- **Full design condition number 11.3 (PASS):** The 4-column design matrix [lec_total, iesr_total, intr_resid, avd_resid] has condition number 11.3, well below the target of 30.
- **LEC-5 subcategory data gap confirmed (L2-04 closed):** Only `less_total_events` and `less_personal_events` exist in the pipeline. No physical/sexual/accident taxonomy is defined anywhere in the codebase. `include_lec_subcategories=True` raises `ValueError`.
- **`build_level2_design_matrix` in `scripts/fitting/level2_design.py` is the single source of truth:** All Phase 16 hierarchical models must call this function. `COVARIATE_NAMES` list is authoritative for `beta_*` site naming.
- **N=160 complete-data participants (not 154):** 160 participants have complete IES-R + LEC-5 data; 154 was the MLE fit count (task-data completeness criterion). Level-2 regression uses N=160.

### v4.0 Decisions (16-03 completed 2026-04-13)

- **M5 kappa stays manually sampled (locked):** phi_rl is added to the `sample_bounded_param` for-loop (7 params via loop), but kappa remains outside the loop to preserve the L2 regression hook pattern from M3. HIER-04 complete.
- **M6a kappa_s sites named distinctly (locked):** `kappa_s_mu_pr`, `kappa_s_sigma_pr`, `kappa_s_z`, `kappa_s` — mirrors M3's kappa naming but with `_s` suffix. Same bounds/priors as kappa (`mu_prior_loc=-2.0`). HIER-05 complete.
- **M6b per-participant decode pattern (locked):** `kappa_total_i = sampled["kappa_total"][idx]`, `kappa_share_i = sampled["kappa_share"][idx]`, then `kappa = kappa_total_i * kappa_share_i`, `kappa_s = kappa_total_i * (1.0 - kappa_share_i)`. Decoded values passed to likelihood. Decoding inside the for-loop (not in likelihood). HIER-06 complete.
- **M6b dual L2 regression (locked):** Two independent beta coefficients: `beta_lec_kappa_total` and `beta_lec_kappa_share`, each shifting their respective unconstrained parameter when `covariate_lec is not None`.
- **kappa_share prior (locked):** `mu_prior_loc=0.0` from `PARAM_PRIOR_DEFAULTS["kappa_share"]`; group-mean share near 0.5 a priori on the probit scale (no a priori preference for global vs. stimulus-specific split).
- **phi_rl prior (locked):** `mu_prior_loc=-0.8` from `PARAM_PRIOR_DEFAULTS["phi_rl"]`; same as phi (both are forgetting rates, symmetric justification).

### v4.0 Decisions (16-05 completed 2026-04-12)

- **32 beta sites (not 40) for subscale model (locked):** `wmrl_m6b_hierarchical_model_subscale` has 8 params x 4 covariates = 32 beta sites. Plan said "~40" (5-covariate assumption); actual is 32 per the 16-01 hyperarousal exclusion.
- **beta site naming: `beta_{cov_name}_{param_name}` (locked):** Outer loop over `param_names`, inner loop over `covariate_names`. Example: `beta_lec_total_kappa_total`, `beta_iesr_intr_resid_alpha_pos`.
- **All 8 M6b params bypass `sample_bounded_param` in subscale variant (locked):** Manual `mu_pr/sigma_pr/z` pattern used for all 8 to enable uniform multi-covariate L2 shift application. `sample_bounded_param` is only bypassed in `wmrl_m6b_hierarchical_model_subscale`; other models unchanged.
- **`subscale=True` guard raises `ValueError` (locked):** `--subscale` with model != wmrl_m6b raises `ValueError` (not `NotImplementedError`); the model exists but the subscale variant is M6b-specific.
- **SLURM subscale: 12h/48G (locked):** `cluster/13_bayesian_m6b_subscale.slurm` uses `--time=12:00:00` and `--mem=48G` (vs 8h/32G for standard M6b).
- **beta_* HDI print expanded (locked):** `_fit_stacked_model` now prints all `beta_`-prefixed sites in sorted order (not just `beta_lec_*`), supporting the 32-site subscale output.

### v4.0 Decisions (20-02 completed 2026-04-14)

- **Vectorized Phase 2 in-place modification (locked):** Modified existing pscan functions rather than creating new variants. The pscan path is already the "parallel" path; vectorization completes it.
- **Precompute helpers called inside block functions (locked):** `precompute_last_action_global` and `precompute_last_actions_per_stimulus` called inside each `*_block_likelihood_pscan()`, not hoisted to multiblock level. Keeps block functions self-contained and correct when called standalone.
- **jax.vmap for batched softmax/epsilon (locked):** `jax.vmap(softmax_policy, in_axes=(0, None))` rather than reimplementing a batched version. Clean, reuses existing scalar functions.
- **axis=-1 renormalization for WM-RL models (locked):** `base_probs / jnp.sum(base_probs, axis=-1, keepdims=True)` for batched (T, A) arrays. M1 does not need renormalization (no WM-Q mixing).

### v4.0 Decisions (20-01 completed 2026-04-14)

- **DEER NO-GO (locked):** DEER/quasi-DEER/ELK not applicable to RLWM likelihood. The perseveration carry (`last_action`) is updated with observed actions (data), not model-computed actions. In likelihood evaluation, this is a phantom dependency: `last_action[t] = actions[t-1]` is precomputable from the data array.
- **Vectorized policy GO (locked):** Replace Phase 2 sequential `lax.scan` with vectorized array ops. Precompute `last_action` arrays once before MCMC (parameter-independent). All T trial log-probs computed simultaneously.
- **precompute_last_action_global signature (locked):** `(actions, mask) -> (T,) int32`. Uses `lax.scan` to correctly propagate through masked trials. Located after `associative_scan_wm_update` in jax_likelihoods.py.
- **precompute_last_actions_per_stimulus signature (locked):** `(stimuli, actions, mask, num_stimuli=6) -> (T,) int32`. Uses `lax.scan` with carry shape `(num_stimuli,)`. Returns per-trial last_action for that trial's stimulus.
- **Contraction mapping question moot (informational):** In simulation context, perseveration is NOT a contraction (discrete state, non-differentiable argmax). In likelihood evaluation context, there is no recurrence to analyze.

### v4.0 Decisions (19-03 completed 2026-04-14)

- **use_pscan kwarg propagation via model_args (locked):** `use_pscan` passed as part of the `model_args` dict that gets unpacked as `**model_args` in `mcmc.run()`. NumPyro passes all `run()` kwargs to the model function. No changes to `run_inference_with_bump` or NUTS kernel required.
- **NetCDF pscan suffix (locked):** When `use_pscan=True`, output NetCDF is `{model}_posterior_pscan.nc` (not separate directory). Enables A/B comparison with `{model}_posterior.nc` in same directory.
- **No outer jax.jit in benchmark (locked):** Likelihood functions use Python `if return_pointwise:` branching which is not JIT-compatible. Benchmark calls functions directly; internal `lax.fori_loop`/`lax.scan` handles JIT compilation.
- **CPU pscan baseline (informational):** Pscan ~3.7x slower on CPU for M3 (825ms vs 222ms, 17 blocks x 100 trials). Expected: associative scan has O(2T) work vs O(T) but O(log T) depth. GPU required for speedup.
- **PSCAN-06 criteria (locked):** Group-mean posterior relative error < 5%, WAIC/LOO absolute difference < 1.0. Automated in cluster/13_bayesian_pscan.slurm Stage 3.

### v4.0 Decisions (19-02 completed 2026-04-14)

- **M5 composed affine operator (locked):** For active (s,a) at trial t: `a = (1-alpha)*(1-phi_rl)`, `b = (1-alpha)*phi_rl*Q0 + alpha*r`. For inactive: `a = 1-phi_rl`, `b = phi_rl*Q0`. Fuses phi_rl decay and delta-rule update in a single scan pass.
- **Q_decayed_for_policy post-scan recovery (locked):** `Q_decayed_for_policy = (1-phi_rl)*Q_for_policy + phi_rl*Q0`. Same pattern as wm_for_policy recovery in 19-01 (apply decay to carry-in array). Required for M5 pscan correctness.
- **Agreement threshold (locked):** < 1e-4 relative error for multiblock NLL agreement between pscan and sequential. Q-learning N=154 achieves max rel_err 6.46e-07 (well under threshold).
- **Two-phase pscan architecture (locked):** Phase 1 parallel O(log T) for Q/WM trajectories; Phase 2 sequential O(T) for policy computation (softmax, epsilon, perseveration carry). All 6 models follow this pattern.

### v4.0 Decisions (19-01 completed 2026-04-14)

- **Single-pass WM scan (locked):** `associative_scan_wm_update` uses one scan combining decay+overwrite. `wm_for_policy[t] = (1-phi)*carry_in[t] + phi*wm_init` recovered as post-scan vectorized step (no second scan).
- **Padding trials decay (not identity) (locked):** Confirmed from wmrl_m3_block_likelihood: decay `(1-phi)*WM + phi*baseline` applies on ALL trials. Only the overwrite is masked. Parallel scan base coefficients are decay everywhere; only active+valid positions override to reset `(a=0, b=r)`.
- **wm_after_update[t] = WM after overwrite at trial t (locked):** `= WM_all[t]` directly (no prepend/drop). The prepend/drop indexing applies to `wm_carry_in` (used internally to compute wm_for_policy), not to wm_after_update.
- **Tolerance thresholds (locked):** < 1e-5 relative error for typical alpha (<=0.5), < 1e-3 for extreme alpha (~0.95). Documented in PARALLEL_SCAN_LIKELIHOOD.md Section 5.
- **Alpha approximation scope (locked):** Reward-based approximation affects Q-update scan only (r==1 → alpha_pos vs delta-sign rule). WM overwrite has no approximation (exact reset). WM decay is exact (constant coefficients).

### v4.0 Decisions (18-05 completed 2026-04-13)

- **Methods dual-approach layout (locked):** MLE paragraph first (point estimates, AIC/BIC), hierarchical Bayesian appended after. Two clearly separated approaches. MLE not removed.
- **Level-2 probit regression in Methods (locked):** Design matrix: `lec_total`, `iesr_total`, `iesr_intr_resid`, `iesr_avd_resid`. Hyperarousal excluded (exact linear dependence). beta priors N(0,1) on probit scale. HDI-excludes-zero criterion.
- **Bayesian section renamed (locked):** `### Bayesian Multivariate Regression` -> `### Hierarchical Level-2 Trauma Associations {#sec-bayesian-regression}`. Anchor preserved for cross-references.
- **Stacking weights path (locked):** `output/bayesian/level2/stacking_weights.csv` (matches CMP-04 layout from 18-02).
- **Level-2 forest plot candidate paths (locked):** `output/bayesian/figures/m6b_forest_lec5.png` then `output/bayesian/level2/wmrl_m6b_forest.png`. First found wins.
- **Limitations subsection location (locked):** `### Limitations {#sec-limitations}` inside `## Discussion`, before `## Conclusion`. Covers 4 topics: Pareto-k M4, K identifiability, M6b shrinkage, IES-R orthogonalization.
- **paper.qmd is source of truth (locked):** All edits target paper.qmd; paper.tex is generated by `quarto render`. DOC-02/03/04 complete.

### v4.0 Decisions (18-02 completed 2026-04-13)

- **CMP-04 artifact layout (locked):** `--bayesian-comparison` writes 4 files to `output/bayesian/level2/`: `stacking_weights.md`, `stacking_weights.csv`, `waic_summary.csv`, `m4_comparison.csv`. Every Markdown output has a CSV companion.
- **M4 Pareto-k threshold 5% (locked):** M4 separate track gates LOO reliability on fraction of Pareto-k > 0.7 exceeding 5%. Expected to trigger in production due to joint LBA likelihood.
- **WAIC is secondary metric (locked):** LOO via `az.compare(ic='loo', method='stacking')` remains primary. WAIC computed per model via `az.waic()` loop. `waic_summary.csv` written only if results non-empty.
- **M4 never in compare_dict (locked):** `BAYESIAN_NETCDF_MAP` contains only M1-M6b. M4 separate track loads `output/bayesian/wmrl_m4_posterior.nc` independently inside `run_bayesian_comparison()`.

### v4.0 Decisions (17-03 completed 2026-04-13)

- **Relaxed structural checks for M4 integration tests (locked):** `test_log_delta_recovery` uses delta>0, A>0, b>A checks (not 15% relative error). The 15% criterion applies to the full N=154 cluster fit; at N=10 with short chains the posterior is too diffuse for point recovery.
- **test_checkpoint_resume tests the API directly (locked):** Warmup-pickle-load-resume cycle tested without process-kill simulation. Matches the exact code path in 13_fit_bayesian_m4.py.
- **SLURM GPU env priority reversed vs CPU models (locked):** 13_bayesian_m4_gpu.slurm activates rlwm_gpu first, ds_env as fallback. CPU models prefer ds_env first.
- **Separate JAX cache dir jax_cache_m4_bayesian (locked):** Prevents float64 M4 JIT traces from colliding with float32 choice-only traces in jax_cache_bayesian.
- **NUMPYRO_HOST_DEVICE_COUNT not set in GPU SLURM script (locked):** GPU chain parallelism uses chain_method='vectorized'; the CPU device count env var is irrelevant on GPU and was intentionally excluded.

### v4.0 Decisions (17-02 completed 2026-04-13)

- **Participant-level log-lik for M4 LOO (locked):** `wmrl_m4_multiblock_likelihood_stacked` returns scalar NLL per participant; no `return_pointwise` path in LBA likelihood. `compute_m4_pointwise_loglik` in `13_fit_bayesian_m4.py` computes shape `(chains, samples, n_participants)` via `jax.vmap`. Pareto-k > 0.7 fallback is near-certain in production (expected behavior).
- **3D InferenceData for M4 (locked):** `_build_inference_data_m4` uses `dims=["participant"]` (not `["participant", "trial"]`). Does NOT reuse `build_inference_data_with_loglik` from bayesian_diagnostics.py (which hardcodes 4D shape).
- **Convergence gate writes run metadata on failure (locked):** `_write_run_metadata` called before early-return to ensure `wmrl_m4_run_metadata.json` exists for debugging regardless of convergence.
- **chain_method='vectorized' confirmed for M4 (locked):** Same as choice-only models. `parallel` requires process forking; unavailable on Windows/SLURM.

### v4.0 Decisions (17-01 completed 2026-04-13)

- **Lazy import for lba_likelihood in M4 hierarchical model (locked):** `wmrl_m4_multiblock_likelihood_stacked` and `preprocess_rt_block` imported inside function bodies only. Module-level import of lba_likelihood in numpyro_models.py would activate float64 globally for all choice-only model imports.
- **delta = b - A parameterization (M4H-02, locked):** `delta` sampled log-normal; `b = A + delta` decoded inside participant for-loop. Guarantees b > A without inequality constraints. `b=b_i` (not delta_i) passed to `wmrl_m4_multiblock_likelihood_stacked`.
- **RT padding value = 0.5s (locked):** Padding positions in rts_stacked set to 0.5s to prevent t_star = rt - t0 <= 0 in LBA PDF evaluation. Masked-out trials use safe non-zero RT.
- **No epsilon in M4 hierarchical model (locked):** 10 parameters total (6 RLWM + 4 LBA). Epsilon is a softmax noise term; M4 uses LBA decision dynamics directly.

### v4.0 Decisions (16-07 completed 2026-04-13)

- **L2-08 horseshoe DEFERRED (locked):** Normal(0,1) priors on beta coefficients have not been tested on real data (cluster job not yet run). Horseshoe will be evaluated after cluster job completes by inspecting max_rhat for beta_ sites, posterior SD vs prior scale, and number of 95% HDI exculsions across the 32 sites.
- **18_bayesian_level2_effects.py created (L2-07):** Forest plots grouped by covariate (LEC-5, IES-R, all). `discover_beta_vars()` finds beta_ sites dynamically at runtime — handles any posterior structure. Graceful skip if NetCDF missing.
- **--bayesian-comparison flag in script 14 (SC-6):** `az.compare(ic='loo', method='stacking')` over all 6 choice-only model posteriors. Writes `output/bayesian/level2/stacking_weights.md` with verdict (M6b weight >= 0.5 threshold).

### v4.0 Decisions (16-06 completed 2026-04-13)

- **Permutation shuffles covariate labels only (locked):** `np.random.default_rng(shuffle_idx).permutation(covariate_lec)` randomizes participant-level LEC alignment, not within-participant data. JSON-only output per shuffle (no CSV/NetCDF). Aggregation script provides PASS/FAIL at 5% FPR.
- **Reduced MCMC budget for permutation (locked):** SLURM uses warmup=500/samples=1000 (half of standard). Sufficient for HDI-excludes-zero check; does not need convergence-grade posterior.
- **parser.error() guard at validation time (locked):** `--permutation-shuffle` raises error immediately before data loading if model != wmrl_m3.

### v4.0 Decisions (16-04 completed 2026-04-13)

- **_L2_LEC_SUPPORTED frozenset (locked):** `{wmrl_m3, wmrl_m5, wmrl_m6a, wmrl_m6b}` pass covariate_lec to model; M1/M2 receive None. Guard in `_fit_stacked_model`.
- **log_likelihood group overwrite (locked):** `build_inference_data_with_loglik` deletes pre-existing `log_likelihood` before `add_groups`. `az.from_numpyro` auto-creates per-participant scalar log-probs from `numpyro.factor` sites (wrong shape for WAIC/LOO). Fix committed in `6f99419`.
- **STACKED_MODEL_DISPATCH dispatch pattern (locked):** All 6 choice-only models route through this dict; `save_results` uses `model in STACKED_MODEL_DISPATCH` as the single predicate for convergence gate path.
- **SLURM time limits (locked):** M1=4h, M2=6h, M3=6h, M5=6h, M6a=6h, M6b=8h. M6b extra time for stick-breaking compile overhead.

### v4.0 Decisions (16-02 completed 2026-04-12)

- **qlearning_hierarchical_model_stacked signature (locked):** No `set_sizes_stacked` passed to likelihood; `covariate_lec=None` guard raises `NotImplementedError` if non-None (no natural L2 target for M1). HIER-02 complete.
- **wmrl_hierarchical_model_stacked signature (locked):** `set_sizes_stacked=pdata["set_sizes_stacked"]` always passed to likelihood; `covariate_lec=None` guard raises `NotImplementedError` if non-None (no perseveration parameter as L2 target for M2). HIER-03 complete.
- **Stacked-format pattern established for M1/M2:** `sorted(participant_data_stacked.keys())` for participant ordering; `sample_bounded_param` loop over param list from `PARAM_PRIOR_DEFAULTS`; `numpyro.factor` per participant. All Phase 16-17 stacked models follow this template.

### v4.0 Decisions (15-02 completed 2026-04-12)

- **run_inference_with_bump pattern (locked):** Python retry loop over `target_accept_probs=(0.80, 0.95, 0.99)`; reads `mcmc.get_extra_fields()["diverging"].sum()`; returns immediately on zero divergences; falls through to last run when exhausted. All Phase 16-17 hierarchical models should use this instead of `run_inference`.
- **Convergence gate location in save_results():** Gate lives inside `save_results()` (not `fit_model()`), so the MCMC result is always returned to the caller. Early return from `save_results()` without writing any files is the HIER-07 enforcement point.
- **filter_padding_from_loglik returns NumPy array:** Converts JAX DeviceArray to `np.array(float32)` before masking. ArviZ `add_groups()` requires NumPy-backed arrays; keeping as JAX array causes implicit conversion errors.
- **LEC covariate column name confirmed:** `less_total_events` (not `lec_total_events`) in `output/summary_participant_metrics.csv`. Z-scored in `fit_model()` before passing to model. Falls back to `covariate_lec=None` with warning if CSV or column missing.
- **Shrinkage formula:** `1 - var_indiv / (var_group_mean + 1e-10)` where `var_indiv = np.var(flat)` (all draws AND participants), `var_group_mean = np.var(flat.mean(axis=1))`. Threshold 0.3 for "identified" flag.

### v4.0 Decisions (15-01 completed 2026-04-12)

- **Compile-gate relaxed to 120s for CPU:** Warm M3 (7 params, 5 ppts, 300 iterations) takes 65-80s on CPU; 60s is a GPU cluster target. Use 120s gate for local CI; expect < 60s on cluster. Same relaxation anticipated for M6b in Phase 16.
- **Cold/warm protocol for M3 smoke test:** `test_smoke_dispatch` primes JIT with a cold 5/5-sample run, then times the full 100/200-sample warm run. Mirrors `test_compile_gate.py` pattern.
- **kappa_z uses `.expand([n_participants])` not `numpyro.plate`:** Avoids plate name conflicts; matches `sample_bounded_param` internal pattern. All future M3-family models should use the same approach.
- **prepare_stacked_participant_data sorts participants:** `sorted(data_df[participant_col].unique())` ensures downstream covariate arrays (e.g., `covariate_lec`) align with `sorted(result.keys())`. This is a correctness constraint.

### v4.0 Decisions (14-02 completed 2026-04-12)

- **fit_all_gpu_m4 float64 ordering (locked):** `jax.config.update("jax_enable_x64", True)` must be the FIRST statement in `fit_all_gpu_m4` — cannot be toggled after first JAX array is materialised.
- **lba_likelihood lazy import in wrapper:** Mirrors `main()` pattern (line 2830); importing `wmrl_m4_multiblock_likelihood_stacked` reinforces float64 globally.
- **Smoke test @pytest.mark.slow:** `test_fit_all_gpu_m4_smoke` with `n_starts=5` validates JIT compilation + finite NLL without targeting parameter recovery.
- **GPU-01 requirement complete:** `fit_all_gpu_m4` callable exists in `fit_mle.py` with correct float64 initialization sequence and delegation to `fit_all_gpu`.

### v4.0 Decisions (14-01 completed 2026-04-12)

- **K bounds [2,6] enforced in mle_utils.py:** All 6 WM capacity BOUNDS dicts updated to `(2.0, 6.0)`. K-02 requirement complete.
- **Version stamp pattern (locked):** `fits_df['parameterization_version'] = EXPECTED_PARAMETERIZATION[args.model]` in `fit_mle.py main()` before `to_csv`. Enables `load_fits_with_validation()` downstream rejection of stale fits.
- **Pre-existing test failure:** `test_compile_gate.py::test_compile_gate` fails (JAX scan shape in fixture). Confirmed pre-existing before this plan. Phase 15 must resolve.

### v4.0 Decisions (13-05 completed 2026-04-12)

- **compute_pointwise_log_lik dispatch pattern:** `model_name` string dispatches to correct stacked likelihood fn; `jax.vmap` over (chains, samples_per_chain); returns shape `(chains, samples, participants, n_blocks*max_trials)`. Padded trials inherit log_prob=0.0 from mask.
- **M6b decoding in diagnostics:** `kappa_total*kappa_share -> kappa`, `kappa_total*(1-kappa_share) -> kappa_s` decoded inside `_build_per_participant_fn`, NOT in the likelihood fn itself.
- **Schema-parity column order (locked):** `participant_id -> params -> nll/aic/bic/aicc/pseudo_r2 -> {param}_hdi_low/high/sd -> max_rhat/min_ess_bulk/num_divergences -> n_trials/converged/at_bounds -> parameterization_version`. No `grad_norm`, `hessian_*`, `_se`, `_ci_*`, `high_correlations`.
- **converged gate (locked):** `max_rhat < 1.01 AND min_ess_bulk > 400 AND num_divergences == 0` (strict inequalities).
- **SLURM Bayesian script:** `time=06:00:00`, `mem=32G`, `NUMPYRO_HOST_DEVICE_COUNT=1`; JAX cache block verbatim from 12_mle_gpu.slurm.

### v4.0 Decisions (13-01 completed 2026-04-12)

- **P0 import fixed:** `scripts/fitting/numpyro_models.py` now exists at canonical path. `from scripts.fitting.numpyro_models import ...` resolves without error.
- **PyMC fully removed (INFRA-07 executed):** `pyproject.toml`, `pytest.ini`, `environment_gpu.yml`, `16b_bayesian_regression.py`, and `validation/test_pymc_integration.py` all updated/deleted.
- **Deps pinned:** `numpyro==0.20.1`, `arviz==0.23.4`, `netcdf4` added to all dep specs.

### v4.0 Decisions (13-04 completed 2026-04-12)

- **phi_approx = jax.scipy.stats.norm.cdf (LOCKED):** Not expit, not polynomial approximation. Named function in numpyro_helpers.py for grep-ability.
- **hBayesDM non-centered convention established:** `mu_pr ~ Normal(0,1)`, `sigma_pr ~ HalfNormal(0.2)`, `theta = lower + (upper-lower)*Phi_approx(mu_pr + sigma_pr*z)`. Single implementation in `sample_bounded_param`, used by all Phase 15-17 hierarchical models.
- **PARAM_PRIOR_DEFAULTS locked:** 11 parameters with shifted mu_prior_loc priors. LBA params (v_scale, A, delta, t0) excluded — handled separately in M4 hierarchical model.
- **parameterization_version validation gate active:** `load_fits_with_validation()` in config.py raises ValueError with expected vs actual on mismatch. Phase 14 MLE refit must stamp `"v4.0-K[2,6]-phiapprox"` on output CSVs.
- **float32 phi_approx saturates at ~±6 sigma:** Tests use ±5 (not ±10) to stay non-degenerate.

### v4.0 Decisions (13-03 completed 2026-04-12)

- **return_pointwise pattern:** `*_block_likelihood` functions expose per-trial log-probs via `*, return_pointwise: bool = False`. Plain Python `if` branch (not `jax.lax.cond`) - JIT specializes on static bool via `static_argnames`.
- **Stacked wrapper pointwise uses lax.scan** over block indices (not lax.fori_loop) because scan accumulates outputs. Flat reshape `all_block_probs.reshape(-1)` yields `(n_blocks * MAX_TRIALS_PER_BLOCK,)` for arviz.
- **Padding positions have log_prob=0.0** in pointwise output. Plan 05 `bayesian_diagnostics.py` must filter mask==0 positions before `az.waic()` / `az.loo()` to avoid inflating WAIC effective parameter count.

### v4.0 Decisions (13-02 completed 2026-04-12)

- **K bounds [2, 6] confirmed (K-01):** Lower bound = 2 matching Senta, Bishop, Collins (2025) PLOS Comp Biol 21(9):e1012872 AND structural identifiability (K<2 confounded with rho at ns=2). Upper bound = 6 (task max ns, K>6 non-identified). parameterization_version = "v4.0-K[2,6]-phiapprox". Phase 14 MLE refit must adopt same bounds. Reference: `docs/K_PARAMETERIZATION.md`.
- **Non-centered K transform established:** `K_i = 2.0 + 4.0 * Phi_approx(mu_K_pr + sigma_K_pr * z_K_i)` where `Phi_approx = jax.scipy.stats.norm.cdf`. Group priors: `mu_K_pr ~ Normal(0,1)`, `sigma_K_pr ~ HalfNormal(0.2)`.

### v4.0 Decisions (set at milestone definition 2026-04-11)

- **M4 hierarchical IS in scope** despite research recommendation to descope. Phase 17 committed. Accept ~150-200 GPU-hour total budget, Pareto-k fallback for M4-vs-choice-only comparison.
- **PyMC dropped entirely** from `16b_bayesian_regression.py`; NumPyro-only backend for v4.0.
- **IES-R subscale orthogonalization:** IES-R total + Gram-Schmidt residualized subscales as default; horseshoe prior (L2-08) is P2 optional upgrade.
- **Schema-parity CSV pattern** is the migration cornerstone — downstream scripts 15/16/17 get a single `--source mle|bayesian` flag with no logic rewrite.
- **P0 broken import** (`fit_bayesian.py:43` imports from `scripts.fitting.numpyro_models` but file is in `legacy/`) is Phase 13 Task 1.
- **Compile-time gate:** < 60s for M3 hierarchical; may need relaxation for M6b (unconstrained stick-breaking compiles slower, no benchmark yet).
- **Phase ordering:** P13 → P14 → P15 → P16 → P17 → P18. P15 (M3 POC) is the validation gate before P16 mechanical extension. P17 depends on both P14 (GPU LBA) and P16 (M6b non-centered pattern as template).

### v3.0 Model Decisions (retained for reference)

- Build order: M5 → M6a → M6b → M4 (complexity-ordered; M5 validated pipeline integration pattern)
- M4 gets separate comparison track in `compare_mle_models.py` (joint likelihood incommensurable with choice-only AIC) — **same constraint carries forward into v4.0 Phase 18**
- Parameter recovery r >= 0.80 is a hard gate per model — **replaced in v4.0 by hierarchical shrinkage diagnostic `1 - var_post_individual / var_post_group >= 0.3` plus convergence gate**
- MODEL_REGISTRY in config.py is single source of truth for pipeline scripts
- CHOICE_ONLY_MODELS = ['qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b'] — M4 excluded from cross-model AIC comparison
- M6b: stick-breaking decode in objective functions only (not in transform): `kappa = kappa_total * kappa_share`; `kappa_s = kappa_total * (1 - kappa_share)`. **v4.0 HIER-06 uses non-centered offset on the unconstrained scale and decodes inside the likelihood.**

### Pending Todos

- **Re-fit all 7 models on cluster** (3 bugs fixed: argmin NaN, stimulus sampling, reward mapping). See `.planning/todos/pending/2026-04-07-refit-all-models-on-cluster.md`
- Run parameter recovery for all models after re-fit (50 subj / 3 datasets / 20 starts) — superseded by Phase 14 Collins K refit
- Run full cross-model recovery: `python scripts/11_run_model_recovery.py --mode cross-model --model all --n-subjects 50 --n-datasets 3 --n-starts 20 --n-jobs 8`

### Roadmap Evolution

- **Phases 19-20 added (2026-04-12):** GPU-accelerated likelihood via associative scan (Phase 19) and DEER non-linear parallelization research (Phase 20). Motivated by analysis of why naive vmap-over-participants was 7-13x slower (memory-bandwidth bottleneck, not compute-bound). The real opportunity is parallelizing the TIME dimension via O(log T) associative scan, not the participant dimension. Phases 15-16 must log CPU wall-clock timing as baseline for Phase 19 benchmarking. Key references: PaMoRL (NeurIPS 2024), DEER (ICLR 2024), S4/Mamba, Unifying Framework (TMLR 2025).
- **CPU confirmed correct for Phases 15-18:** RLWM Q-value/WM updates have arithmetic intensity ~0.3 FLOP/byte (GPU needs >50 to saturate). CPU L1 cache (~1ns) beats GPU global memory (~200-400 cycles) for 18-float Q-tables. Associative scan changes the algorithm, not the hardware access pattern.
- **Phase 21 added (2026-04-18):** Principled Bayesian Model Selection Pipeline. Replaces the MLE-preselected "M6b is the winner" flow with a linear 9-step Bayesian workflow: prior predictive (21.1) → Bayesian parameter recovery (21.2) → hierarchical fit no-scales baseline (21.3) → convergence+PPC audit (21.4) → PSIS-LOO + stacking weights + RFX-BMS/PXP ranking (21.5) → winner(s) refit with L2 scales (21.6) → scale-fit audit (21.7) → model-averaged scale effects and exploratory M6b-subscale arm (21.8) → manuscript tables (21.9). Each step = separate SLURM submission, re-runnable from cold start. Anchored to Baribault & Collins (2023, *Psychological Methods*, DOI 10.1037/met0000554) and Hess et al. (2025, *Computational Psychiatry* 9(1):76–99, DOI 10.5334/cpsy.116). AIC/BIC deprecated for hierarchical-Bayesian comparison. ~70% of existing infrastructure (simulator, generate_data, jax_likelihoods, numpyro_models, fit_bayesian, compute_pointwise_log_lik, recovery structure) is directly reusable; ~30% new (orchestrators `scripts/21_*.py`, SLURM scripts `cluster/21_*.slurm`, RFX-BMS implementation in `scripts/fitting/bms.py`). `validation/compare_posterior_to_mle.py` retained as sanity check only, not selection criterion. Motivated by user flagging that (a) MLE AIC pre-selection of M6b is circular for the subsequent Bayesian test of κ×LEC, (b) step-4 PPC anchored to MLE is MLE-anchored, (c) choice-fit comparison and scale-effect inference were entangled and need separation.

### Blockers/Concerns

- **Compile-time gate on M6b**: constrained `kappa_total`/`kappa_share` under non-centered hierarchical sampling may compile slower than the 60s target. Phase 13 may need to relax the gate specifically for M6b. (From research: PITFALLS.md confidence MEDIUM on this point.)
- **Hierarchical LBA has no NumPyro/JAX precedent.** Phase 17 is effectively a research project nested in the milestone — if the non-centered `log(b - A)` + `post_warmup_state` resume pattern fails at scale, Phase 17 falls back to reporting M4 at MLE only.
- **Pareto-k > 0.7 is near-certain for LBA under NUTS.** Phase 17 MUST include the choice-only-marginal fallback path; M4 cannot sit inside a unified `az.compare` table regardless of how it's fit.
- **IES-R subscale correlations in N=154 not yet audited** — Phase 16 begins with a collinearity audit; if condition number after orthogonalization remains > 30, the orthogonalization strategy must be revisited.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 001 | Setup Quarto scientific manuscript for RLWM trauma analysis | 2026-04-05 | 18637da | [001-setup-quarto-manuscript](./quick/001-setup-quarto-manuscript/) |
| 002 | Pipeline fixes, convergence assessment, recovery config, MODEL_REGISTRY | 2026-04-07 | 3095b92 | [002-pipeline-fixes-convergence-recovery-config](./quick/002-pipeline-fixes-convergence-recovery-config/) |
| 003 | Softcode manuscript: winning model, group names, n_starts from data files | 2026-04-07 | d7ea897 | [003-quarto-softcoded-winning-model](./quick/003-quarto-softcoded-winning-model/) |
| 004 | Pipeline sync: survey data fix (scripts 15/16), uncorrected p-values in manuscript, Bayesian MODEL_REGISTRY | 2026-04-07 | 4df1340 | [004-pipeline-sync-uncorrected-peb-config](./quick/004-pipeline-sync-uncorrected-peb-config/) |
| 005 | Re-run pipeline (N=154), model overview + distribution figures in manuscript | 2026-04-08 | 6b045a4 | [005-rerun-pipeline-analyses-update-quarto-manuscript](./quick/005-rerun-pipeline-analyses-update-quarto-manuscript/) |
| 006 | Post-refit verification: M6b winner, BIC + winner heterogeneity + FDR/Bonferroni + manuscript revision | 2026-04-10 | a01febd | [006-post-refit-verification-recovery-manuscript](./quick/006-post-refit-verification-recovery-manuscript/) |
| 007 | GPU MCMC speedup: chain_method=vectorized on GPU + GPU config logging + vmap-over-participants refactor for wmrl_m3 (POC) | 2026-04-16 | 1ac963c | [007-gpu-mcmc-speedup-vmap-chain-method](./quick/007-gpu-mcmc-speedup-vmap-chain-method/) |
| 008 | Cleanup pipeline + docs per 2026-04-17 audit: prune 7 orphaned scripts, move 3 superseded docs to legacy/, scaffold docs/04_methods + 04_results, fix scale_distributions.png path | 2026-04-17 | 2c4f12a | [008-cleanup-pipeline-and-docs](./quick/008-cleanup-pipeline-and-docs/) |
| 009 | Integrate Burrows thesis content into paper.qmd: line-cited catalog (INTEGRATION_MAP.md), 28 reusable passages (~1,850 words), 9 divergence flags, no auto-edits applied | 2026-04-17 | 8ccdc4b | [009-integrate-burrows-thesis-into-paper](./quick/009-integrate-burrows-thesis-into-paper/) |

## Session Continuity

Last session: 2026-04-18T14:56Z
Stopped at: Plan 21-03 complete — Step 21.2 Bayesian parameter recovery runner (`scripts/21_run_bayesian_recovery.py` 956 lines) with two CLI modes (single-subject + aggregate) + SLURM array (1-50) + aggregator + 7 smoke tests (21 s). Baseline scope only (2-cov hook gated via pytest). Kappa-family pass criterion r >= 0.80 AND coverage >= 0.90. Commits 05f1b79, 205f480. Ready for cluster submission via master orchestrator (plan 21-10) or next Phase 21 plan.
Resume file: None
