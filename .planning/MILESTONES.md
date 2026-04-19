# Project Milestones: RLWM Trauma Analysis

## v4.0 Hierarchical Bayesian Pipeline & LBA Acceleration (Shipped: 2026-04-19)

**Delivered:** Full hierarchical Bayesian inference pipeline with trauma subscales as Level-2 predictors, principled Bayesian model comparison (PSIS-LOO + stacking + RFX-BMS/PXP), GPU-accelerated LBA sampling for M4, and a 9-step cluster-reproducible model selection workflow anchored to Baribault & Collins (2023) and Hess et al. (2025). Replaces post-hoc FDR correction and r < 0.80 MLE recovery with shrinkage-regularized joint posteriors.

**Phases completed:** 13-22 (47 plans total across 10 phases, including Phase 22 closure)

**Key accomplishments:**

- Hierarchical Bayesian models for all 7 variants (M1-M6b, M4) in `numpyro_models.py` with non-centered parameterization and fully-batched `jax.vmap` (post-Phase-20): single `numpyro.factor("obs", per_participant_ll.sum())` per model; ~700× per-iteration speedup on qlearning
- Collins K parameterization research + constrained-K refit (bounds [2, 6] with `phi_approx` transform) replacing the earlier [1, 7] range that produced r = 0.21 capacity recovery
- 32-coefficient M6b subscale Level-2 regression (8 parameters × 4 predictors: LEC total + IES-R total + Gram-Schmidt residualized IES-R intrusion/avoidance); condition number 11.3
- M4 hierarchical LBA with float64 process isolation, non-centered `log(b − A)`, checkpoint-and-resume, and Pareto-k gating fallback to choice-only marginal log-likelihood
- 9-step principled Bayesian model selection pipeline: prior-predictive gate → recovery → baseline fits → convergence audit → PSIS-LOO + stacking + RFX-BMS → L2 refit → scale audit → averaging → manuscript tables. Master orchestrator `cluster/21_submit_pipeline.sh` chains all steps via `afterok` exclusively with a local pre-flight `pytest` gate
- Associative scan likelihood parallelization (`associative_scan_q_update`, `associative_scan_wm_update`) for O(log T) linear-recurrence components; DEER non-linear parallelization researched and concluded no-go with empirical evidence
- Manuscript `paper.qmd` rewritten with hierarchical Bayesian methods/results/limitations; cites Baribault & Collins 2023, Hess 2025, Yao 2018, Stephan 2009 / Rigoux 2014
- Reproducibility guard `validation/check_v4_closure.py` enforcing 8 closure invariants deterministically + pytest regression `scripts/fitting/tests/test_v4_closure.py` (3/3 PASS on completion commit)

**Stats:**

- 706 files changed, +945,115 / −19,716 lines (inflated by fitted-model NetCDFs and figures; actual code ≈ 62,000 LOC across scripts/, validation/, cluster/, docs/)
- 10 phases, 47 plans, ~150 tasks (each committed atomically)
- 9 days from start to ship (2026-04-10 → 2026-04-19)
- 172 commits in milestone range

**Git range:** `81c3570` → `71e063d`

**Cluster-execution pending (code-complete; not scope-dropped):** K-03 / GPU-03 (via `bash cluster/12_submit_all_gpu.sh`), PSCAN GPU A/B benchmark, BMS full 9-step cold-start run (via `bash cluster/21_submit_pipeline.sh`).

**What's next:** v5.0 — run cluster-deferred items, address v4.0 tech debt (legacy M2 K-bounds cleanup, load-side validation in scripts 15/16), consider v2 requirement candidates (ARVIZ1-01, WORKFLOW-01 SBC, HIER-LBA PMwG, horseshoe default). See `.planning/PROJECT.md` for the full v5.0 candidate list.

---

## v3.0 Model Extensions M4-M6 (Shipped: 2026-04-03)

**Delivered:** Extended the model hierarchy from 3 to 7 models — RL forgetting (M5), stimulus-specific perseveration (M6a), dual perseveration (M6b), and joint choice+RT via Linear Ballistic Accumulator (M4) — all integrated into the MLE fitting, comparison, and trauma-analysis pipeline.

**Phases completed:** 8-12 (12 plans total, including 1 gap closure)

**Key accomplishments:**

- M5 RL Forgetting: Q-value decay parameter phi_rl; confirmed new winning model (dAIC=435.6 over M3)
- M6a/M6b Perseveration variants: per-stimulus and dual choice kernels with stick-breaking constraint (kappa + kappa_s <= 1)
- M4 RLWM-LBA: joint choice+RT fitting via Linear Ballistic Accumulator (Brown & Heathcote, 2008) with float64 density functions
- Cross-model recovery: confusion matrix validation infrastructure for 6 choice-only models
- MODEL_REFERENCE.md expanded from 2 to 7 models with complete mathematical specifications
- Gap closure: fixed synthetic generation bugs (M6a/M6b unreachable branches, M3/M5 formula mismatch with likelihood)

**Stats:**

- 11 files modified, +5,961 / -915 lines
- 14,899 LOC across 10 core fitting/analysis files
- 5 phases, 12 plans, 23 commits
- 2 days (2026-04-02 → 2026-04-03)

**Git range:** `b26280c` → `c8360d2`

**What's next:** Run full parameter recovery on cluster (N=50) for M5, M6a, M6b, M4. Cross-model recovery validation. Publication preparation.

---

## v1 M3 Infrastructure (Shipped: 2026-01-30)

**Delivered:** Complete MLE fitting infrastructure for WM-RL M3 model with perseveration parameter (κ), enabling dissociation of outcome-insensitive action repetition from learning-rate effects.

**Phases completed:** 1-3 (6 plans total)

**Key accomplishments:**

- Implemented JAX likelihood functions (`wmrl_m3_block_likelihood()`, `wmrl_m3_multiblock_likelihood()`) with κ·Rep(a) perseveration term
- Extended WMRLHybridAgent with optional kappa parameter, maintaining exact M2 backward compatibility
- Added complete MLE infrastructure (WMRL_M3_BOUNDS, WMRL_M3_PARAMS, `--model wmrl_m3` CLI)
- Created 24+ backward compatibility tests validating M3(κ=0) ≡ M2 to rtol=1e-5
- Fixed critical backward compatibility bug: M3 now branches on κ=0 for M2 probability mixing
- Extended compare_mle_models.py for N-model comparison with Akaike weights

**Stats:**

- 26 files created/modified
- 4,368 lines in key source files
- 3 phases, 6 plans, ~15 tasks
- 2 days from start to ship (2026-01-29 → 2026-01-30)

**Git range:** `d4647a9` → `98a13b2`

**What's next:** Run M3 fits on cluster data, perform M1/M2/M3 model comparison, analyze κ parameter in trauma populations.

---

## v2 Post-Fitting Validation (Archived: 2026-04-02)

**Delivered:** Parameter recovery pipeline (Script 11) and posterior predictive checks (Script 09) with Senta et al. (2025) r >= 0.80 criterion. Regression visualization enhancements for Script 16.

**Phases completed:** 4-5 (7 plans total)

**Key accomplishments:**

- Regression visualization with trauma-group coloring and model-specific outputs
- Parameter recovery pipeline with JAX-based synthetic data generation
- Posterior predictive check framework comparing synthetic vs. observed learning curves
- Model recovery validation (AIC-based winner determination)
- Multi-model support with `--model all` batch mode

**Phases deferred:** 6 (Cluster Monitoring), 7 (Publication Polish) — dropped from scope.

**What's next:** v3 Model Extensions (M4-M6)

---
