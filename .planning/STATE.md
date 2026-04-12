# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-11)

**Core value:** Correctly dissociate perseverative responding from learning-rate effects (alpha-) to accurately identify whether post-reversal failures reflect motor perseveration or outcome insensitivity
**Current focus:** v4.0 — Hierarchical Bayesian Pipeline & LBA Acceleration (Phase 13 complete, Phase 14 next)

## Current Position

Milestone: v4.0 Hierarchical Bayesian Pipeline & LBA Acceleration
Phase: 14 of 18 (Collins K Refit + GPU LBA Batching) — in progress
Plan: 1 of 3 complete (14-01 done; 14-02 Wave 2; 14-03 Wave 3)
Status: In progress — 14-01 complete
Last activity: 2026-04-12 — Completed 14-01-PLAN.md (K bounds + version stamp)

Progress: [██░░░░░░░░] ~20% (6/~30 plans across Phases 13-18)

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

## Session Continuity

Last session: 2026-04-12
Stopped at: Completed 14-01-PLAN.md — K bounds [2,6] in all 6 WM BOUNDS dicts, parameterization_version stamp in fit_mle.py
Resume file: None
