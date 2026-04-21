# Project Research Summary

**Project:** RLWM Trauma Analysis
**Milestone:** v4.0 — Hierarchical Bayesian Pipeline & LBA Acceleration
**Domain:** Computational psychiatry / hierarchical Bayesian fitting of RL+WM models with clinical Level-2 covariates, extending an existing JAX/NumPyro/GPU pipeline
**Researched:** 2026-04-11
**Confidence:** MEDIUM-HIGH (choice-only hierarchical extension is well-trodden ground; M4 hierarchical LBA is genuinely novel and high-risk)

---

## Executive Summary

v4.0 extends the mature v3.0 MLE pipeline to full hierarchical Bayesian inference across the RLWM model family, with trauma-scale subscales injected as Level-2 predictors *jointly* with individual-level parameters. The infrastructure path is clear: NumPyro 0.20.1 + ArviZ 0.23.4 (pinned — no 1.0 DataTree migration in this milestone), non-centered parameterization for every bounded parameter via the hBayesDM pattern, a single `numpyro_models.py` holding all 7 models, `vmap`-over-participants likelihoods wrapped in `numpyro.factor`, post-hoc pointwise log-likelihood computation for WAIC/LOO, and schema-parity CSV outputs that let downstream scripts 15/16/17 work unchanged via a single `--source mle|bayesian` flag. GPU dispatch reuses the existing `cluster/12_mle_gpu.slurm` template with one SLURM job per model, pinned to A100 (40 GB VRAM) for float64 LBA and hierarchical memory footprint.

**The single load-bearing open question is M4 (LBA) scope.** The milestone brief locked in "all 7 models including M4 hierarchical". The STACK research says this is *technically feasible* (A100 + float64 before first JAX import + `chain_method="vectorized"` + `post_warmup_state` checkpointing + non-centered `log(b - A)` parameterization) but flags compute budget as underestimated: the realistic envelope is 70–110 GPU hours per run with M4 alone consuming 24–48 hours, potentially exceeding the 48h SLURM wall-clock limit. The FEATURES research (AF-3) recommends the *opposite* — keep M4 at MLE, report it as a separate joint choice+RT track, and present the primary paper narrative around hierarchical Bayesian results for the choice-only family (M1, M2, M3, M5, M6a, M6b). The rationale: (a) there is no published NumPyro/JAX hierarchical LBA implementation as of 2026; (b) the Gunawan et al. PMwG sampler is R-only; (c) building it from scratch is a 4–6 week side-quest with no precedent to validate against; (d) the PITFALLS research shows LBA under NUTS will almost certainly hit Pareto-k > 0.7 on LOO anyway (Pitfall 5), forcing a fallback to choice-only-marginal comparison — which means M4 can't actually sit inside a unified `az.compare()` table no matter how you fit it.

**Recommendation: descope hierarchical M4.** Keep M4 at MLE (as v3.0 does), make v4.0 about "all choice-only models hierarchical with joint Level-2 trauma regression", and preserve M4 as a parallel joint choice+RT track in the paper. This keeps the milestone scoped to ~50 GPU hours, lets the team ship a cleaner scientific story ("hierarchical Bayesian RLWM with trauma covariates"), and sidesteps the Pareto-k comparison pathology and the 48h SLURM wall-clock risk. However, this is a **user-level scope decision**, not a research conclusion, and must be made before the roadmap is written — see "Open Decisions" below.

**P0 infrastructure bug:** `scripts/fitting/fit_bayesian.py:43` imports from `scripts.fitting.numpyro_models` but the file lives at `scripts/fitting/legacy/numpyro_models.py`. A stale `__pycache__` hides the breakage on developer laptops that previously ran the script; a fresh checkout fails. **The very first task of the very first phase must resurrect this import path.**

---

## Key Findings

### Recommended Stack (detail: STACK.md)

**No JAX-side additions.** Pin `numpyro==0.20.1` and `arviz==0.23.4` (exact). Explicitly *do not* migrate to ArviZ 1.0 — the `InferenceData → xarray.DataTree` breaking change is a v5.0 problem. Add `netcdf4` to the GPU env for `InferenceData.to_netcdf()`. Drop PyMC from the GPU env entirely — see Pitfall 7 / Open Decision 3 below.

**Core constraints:**
- **`numpyro.factor` + post-hoc pointwise log-lik helper is mandatory.** Out-of-the-box `arviz.from_numpyro` does not populate the `log_likelihood` group for factor-based models, so `az.waic`/`az.loo` fail silently. A new `scripts/fitting/bayesian_diagnostics.py` module must implement `compute_pointwise_log_lik()` that vmaps over (chains, samples, participants) calling per-trial log-prob versions of the existing likelihood functions. This is the single largest implementation risk in the stack.
- **Float64 is a process-wide flag.** `jax.config.update("jax_enable_x64", True)` + `numpyro.enable_x64()` must run before any JAX import. Mandates separate SLURM jobs for M4 vs choice-only — same constraint already in force for MLE.
- **`chain_method="vectorized"`** (vmap on a single GPU) is the documented multi-chain pattern for Monash M3. `parallel` requires multiple GPUs per job and isn't available on the standard partition.
- **LBA under NUTS needs `target_accept_prob=0.95`** (not default 0.8) and **non-centered `log(b - A)`** reparameterization to avoid boundary-driven funnel pathologies.
- **Compute budget:** 70–110 GPU-hours per full run across 7 models is more realistic than the 50–96h milestone estimate. Budget for ≥ 1 rerun (150–200 GPU-hours total). M4 alone is 24–48h and may exceed the 48h SLURM cap — mitigate with `num_warmup=1000/num_samples=1500` for M4 specifically, or checkpoint-and-resume via `mcmc.post_warmup_state`.

### Expected Features (detail: FEATURES.md)

**Must have (table stakes — 13 features, all P1):**
- Two-level hierarchy (group `mu`, `sigma` → individual) for every fitted model (TS-1)
- Non-centered parameterization on every Level-2 latent (TS-2) — centered breaks under funnel geometry
- `phi_approx`/inverse-probit for [0,1] parameters (TS-3)
- Truncated-normal bounded continuous K (TS-4) — never discrete under HMC
- Weakly informative Beta priors on group means (TS-5) + HalfNormal(0.2–0.3) on group sigmas (TS-6)
- Convergence gate: R-hat ≤ 1.01, ESS_bulk ≥ 400, zero divergences — scripts refuse to write outputs on failure (TS-7)
- **WAIC + PSIS-LOO** replacing AIC/BIC as primary comparison criterion (TS-8)
- **Joint Level-2 trauma regression** — `mu_param = beta_0 + beta_trauma * trauma_z` inside the fit, NOT post-hoc (TS-9). This is the scientific motivation for the milestone.
- Z-scored trauma covariates before fitting (TS-10), per-participant log-likelihood storage (TS-11), posterior predictive checks (TS-12), reproducible seeds + NetCDF persistence (TS-13)

**Differentiators (should have — 10 features):**
- Hierarchical M3/M5/M6a/M6b with joint Level-2 regression (D-1) — genuinely novel; Wisco et al. 2025 published MLE-only
- **Subscale-level Level-2 regressors** (IES-R intrusion/avoidance/hyperarousal, LEC-5 subcategories) instead of totals (D-2) — the central scientific advance
- Regularized horseshoe shrinkage on the 48-coefficient M6b regression family (D-3) — P2, optional in iteration 1
- Stacking weights from `az.compare(method='stacking')` (D-4), Pareto-k flagging of influential participants (D-5), group-stratified PPCs (D-6)
- MLE-vs-Bayesian side-by-side reliability plots (D-8), GPU acceleration via existing SLURM infra (D-9), predictive-overlay paper figures (D-10)

**Defer / explicitly exclude (11 anti-features):**
- **AF-3: Hierarchical M4 LBA — recommended EXCLUDED.** No published NumPyro/JAX precedent; PMwG is R-only; 4–6 weeks side-quest; Pareto-k will likely make LOO comparison invalid. See "Open Decisions" — this contradicts the milestone's locked scope.
- AF-1 centered parameterization; AF-2 LKJ group correlation (too expensive at N=154); AF-4 discrete K prior (kills HMC); AF-5 AIC/BIC as primary criterion; AF-6 post-hoc regression as primary inference; AF-7 10k+ chains; AF-8 single composite trauma score; AF-9 reimplementing likelihoods inside NumPyro; AF-10 save samples to CSV; AF-11 SVI as primary fit

### Architecture Approach (detail: ARCHITECTURE.md)

**Critical pre-existing findings** the roadmap must address before any extension work:

1. **The current Bayesian path is structurally broken.** `scripts/fitting/fit_bayesian.py:43` imports from `scripts.fitting.numpyro_models`, but the file lives at `scripts/fitting/legacy/numpyro_models.py`. **Treat as P0 — first task of first phase.**
2. **Float64 is process-global.** M4 LBA and choice-only fits *must* run in separate Python processes (separate SLURM jobs).
3. **Downstream scripts 15/16/17 and 16b are coupled to a flat CSV schema** via `MODEL_REGISTRY[model]['csv_filename']`. Migration strategy: emit a Bayesian summary CSV with *identical* schema (posterior means in the parameter columns, plus extra `_hdi_low`, `_hdi_high`, `_sd` columns), written to `output/bayesian/`. Downstream scripts get a single `--source mle|bayesian` flag. **No rewrites of 15/16/17 logic required.**

**Target architecture — major components:**
1. **`scripts/fitting/numpyro_models.py`** (RESURRECTED + EXTENDED) — single file, all 7 hierarchical models, replaces the broken legacy import. Uses `jax.vmap` over participants (not a Python for-loop — legacy violates this), wraps existing `*_multiblock_likelihood_stacked` functions from `jax_likelihoods.py` via `numpyro.factor`. Level-2 covariates enter as an additive offset on the unconstrained linear predictor.
2. **`scripts/fitting/bayesian_summary_writer.py`** (NEW) — converts ArviZ `InferenceData` to MLE-schema-compatible flat CSV.
3. **`scripts/fitting/bayesian_diagnostics.py`** (NEW) — post-hoc pointwise log-likelihood helper for WAIC/LOO.
4. **`scripts/fitting/fit_bayesian.py`** (EXTENDED) — accepts all 7 models, adds `--level2-covariates` flag, emits both NetCDF and schema-parity CSV.
5. **`scripts/13_fit_bayesian.py`** (EXTENDED CLI), **`scripts/14_compare_models.py`** (extended with `--bayesian-comparison` mode running `az.compare(idata_dict, ic='loo', method='stacking')`).
6. **`scripts/15/16/17`** (MINIMAL change — add `--source mle|bayesian` flag, path resolution changes, logic unchanged).
7. **`scripts/18_bayesian_level2_effects.py`** (NEW) — forest plots of trauma-parameter effects from the full posterior.
8. **`cluster/13_bayesian_gpu.slurm`** (NEW) — mirrors `12_mle_gpu.slurm` parallel-dispatch pattern; one SLURM job per model; M4 variant with `--time=48:00:00`, `--mem=96G`, `--gres=gpu:a100:1`.

**Key patterns:** schema-parity dual outputs (migration cornerstone), process isolation for float64 LBA, `vmap` for likelihood NOT for chains, Level-2 covariates as additive offset on the unconstrained linear predictor, fast-preview MLE + slow-truth Bayesian dual tier.

### Critical Pitfalls (detail: PITFALLS.md — 10 total)

1. **Centered parameterization on any Level-2 latent** → funnel, divergences, R-hat blowup on `sigma_*`. Prevention: non-centered template on every bounded parameter; unit-test parameter recovery on simulated data.
2. **Hierarchical shrinkage masks non-identifiability.** MLE recovery already fails (`r < 0.80`) for base RLWM params. Prevention: posterior shrinkage diagnostic `1 - var_post_individual / var_post_group`, permutation null for Level-2 effects.
3. **Multicollinear IES-R subscales** (r ≈ 0.6–0.8) give arbitrary slope assignments. Prevention: check design-matrix condition number, orthogonalize against IES-R total (Gram-Schmidt).
4. **Python-loop likelihoods inside NumPyro models** (the legacy pattern) → 10–30 min compile times. Prevention: vmap-over-participants, `numpyro.plate` + single `numpyro.factor` call.
5. **LBA under NUTS hits Pareto-k > 0.7 for LOO.** Prevention: immediate Pareto-k inspection; fall-back to choice-only marginal log-likelihood for M4 vs choice-only comparison. **This is exactly why FEATURES AF-3 recommends excluding hierarchical M4.**
6. **K parameterization change silently invalidates pre-existing MLE fits.** Prevention: `parameterization_version` column in every fit output CSV.
7. **PyMC vs NumPyro prior mismatch** in `16b_bayesian_regression.py` — same `HalfNormal(sigma=...)` means different things. **Recommended: drop PyMC entirely.**
8. **Long GPU runs crash with no checkpoint.** Prevention: `chain_method='sequential'`, `mcmc.post_warmup_state`, integration test with intentional kill.
9. **ArviZ migration breaks downstream consumers** expecting flat CSV. Prevention: schema-parity pattern.
10. **Label-switching across chains for Level-2 slopes** when RLWM has weak RL↔WM identifiability. Prevention: strongly informative priors, chains initialized from same MLE point estimate, rank plots mandatory.

---

## Implications for Roadmap

**Suggested phases: 5 (descoped) or 6 (full scope) — depends on Open Decision 1.**

### Phase 13 — Infrastructure Repair & Hierarchical Scaffolding
**Rationale:** ARCHITECTURE Finding 1 is a P0 bug — the broken import must be fixed before anything else. Parameterization conventions (K bounds, PyMC drop) must be locked here because downstream phases depend on them.
**Delivers:** Resurrected `numpyro_models.py`; `bayesian_diagnostics.py` with `compute_pointwise_log_lik()`; `bayesian_summary_writer.py` (schema-parity CSV); non-centered parameterization helper module with unit tests; JAX compilation cache; `parameterization_version` convention; PyMC drop decision; pinned deps.
**Addresses:** TS-1..6, TS-11, TS-13, D-9; avoids Pitfalls 1, 4, 6, 7.

### Phase 14 — M3 Hierarchical Proof-of-Concept with Level-2 Regression
**Rationale:** Validate the entire stack on M3 + LEC-total → kappa (v3.0's surviving-FDR finding, built-in sanity check).
**Delivers:** M3 hierarchical model; joint Level-2 regression; convergence gate; WAIC+LOO pipeline; PPC reproducing v3.0 M3 curves; single SLURM GPU run; schema-parity CSV verified against scripts 15/16/17.
**Addresses:** TS-7, TS-8, TS-9, TS-10, TS-12, D-8; avoids Pitfalls 2, 3, 10.

### Phase 15 — Choice-Only Family Hierarchical Extension (M1, M2, M5, M6a, M6b)
**Rationale:** Mechanical extension of M3 template in dependency order: M1/M2 (port from legacy) → M5 (adds `phi_rl`) → M6a (per-stimulus state) → M6b (constrained dual perseveration).
**Delivers:** All 6 choice-only models hierarchically fit with Level-2 regression; `14_compare_models.py --bayesian-comparison`; per-model diagnostic CSVs; dispatch sweep smoke test.
**Addresses:** TS-1..13 for all choice-only, D-1, D-4, D-5; avoids Pitfalls 2, 4, 10.

### Phase 16 — Subscale Level-2 Regression & Multiplicity Handling
**Rationale:** Central scientific advance of v4.0. Collinearity (Pitfall 3) requires orthogonalization to be locked before fitting.
**Delivers:** Collinearity audit (condition number, VIF); orthogonalized subscale parameterization (IES-R total + Gram-Schmidt residuals recommended); hierarchical M6b fit with full subscale regressor set; optional regularized horseshoe prior; posterior forest plots; permutation null test.
**Addresses:** D-2, D-3, D-6; avoids Pitfalls 2, 3, 10.

### Phase 17 — M4 Hierarchical LBA *[CONDITIONAL — depends on Open Decision 1]*
**Rationale:** Exists only if Open Decision 1 keeps hierarchical M4 in scope.
**Delivers:** Float64 process-isolated NumPyro M4 LBA model; non-centered `log(b - A)`; `chain_method='vectorized'`; `target_accept_prob=0.95`; reduced `num_warmup=1000/num_samples=1500`; checkpoint-and-resume via `mcmc.post_warmup_state`; Pareto-k gating with choice-only marginal fallback; separate A100 SLURM job (48h).
**Risks:** 24–48h wall clock, Pareto-k > 0.7 almost certain, 4–6 weeks implementation with no precedent.

### Phase 18 — Integration, Comparison, and Paper Artifacts
**Rationale:** Schema-parity pattern means downstream migration is mostly a flag flip.
**Delivers:** Scripts 15/16/17 rerun with `--source bayesian`; `16b_bayesian_regression.py` frozen with deprecation comment; stacking-weight model comparison; MLE-vs-Bayesian reliability scatterplots; predictive-overlay paper figures; `docs/MODEL_REFERENCE.md` updated; manuscript methods/results revision.

### Phase Ordering Rationale

- **P13 must precede P14** — broken-import fix + non-centered helper + backend decision are preconditions.
- **P14 isolates M3 as proof-of-concept** — reproduces known v3.0 result while exercising new stack.
- **P15 dependency order** (M1/M2 → M5 → M6a → M6b) matches model complexity and dispatch-risk.
- **P16 is separated from P15** — orthogonalization should be decided on winning-model data.
- **P17 is conditional and last-among-fits** — M4 is highest-risk and most likely to force fallback.
- **P18 migrates downstream last** — schema-parity lets downstream scripts wait until CSVs are written.

### Research Flags

Phases likely needing `/gsd:research-phase` during planning:
- **Phase 14** — `compute_pointwise_log_lik` implementation. No canonical NumPyro example for vmap over (chains × samples × participants) with `log_density` primitive.
- **Phase 16** — Subscale orthogonalization strategy (residualization vs horseshoe vs PCA) and regularized horseshoe specification for RL contexts.
- **Phase 17** (if in scope) — Hierarchical LBA under NUTS is novel in Python; non-centered `log(b - A)` pattern has no direct NumPyro reference; `post_warmup_state` resume pattern is forum-only.

**Standard patterns (skip research-phase):** Phase 13, 15, 18.

---

## Open Decisions Requiring User Input

### Decision 1 — Is hierarchical M4 (LBA) in scope for v4.0? **[BLOCKS ROADMAP]**

**Current state:** Milestone brief locks in "all 7 models including M4 hierarchical". FEATURES AF-3 strongly recommends descoping. STACK says technically feasible but compute-budget underestimated. PITFALLS shows LOO comparison will likely fail regardless.

**Option (a) — Keep M4 hierarchical in scope:**
- Accept 4–6 weeks extra implementation time with no published NumPyro precedent
- Accept 24–48h SLURM runs per M4 fit; budget for timeout and checkpoint-resume
- Compute budget doubles: 150–200 GPU-hours total
- Accept Pareto-k fallback: M4 vs choice-only comparison must use choice-only marginal
- Does not produce a unified `az.compare` table
- Enables Phase 17 (6 phases total)

**Option (b) — Descope M4 to MLE-only [RECOMMENDED]:**
- Keep v3.0 approach for M4: MLE with joint choice+RT track, reported separately
- v4.0 becomes "hierarchical Bayesian for all choice-only models with joint Level-2 trauma regression"
- Cleaner scientific narrative
- Total GPU budget: ~50 hours per run, ~100 hours with 1 rerun
- Skips Phase 17 (5 phases total: P13, P14, P15, P16, P18)
- If reviewers demand hierarchical LBA: "PMwG sampler in R `pmwg`; out of scope for JAX pipeline"

**Research recommendation:** Option (b). The Pareto-k pathology means even successful M4 hierarchical fitting cannot produce a scientifically valid single comparison table, so the upside of Option (a) is mostly "symmetry" — not a scientific gain. The scientific advance in v4.0 is the joint Level-2 regression on the WM-RL perseveration family, and that's unaffected by the M4 decision.

**User must decide before roadmap creation.**

### Decision 2 — Subscale orthogonalization strategy (can defer to Phase 13)

- (a) IES-R total + Gram-Schmidt residualized subscales **[RECOMMENDED baseline]**
- (b) PCA component 1 + orthogonal residuals
- (c) Raw subscales + regularized horseshoe prior (P2 upgrade path)

### Decision 3 — Single-backend or dual-backend Bayesian pipeline (can defer to Phase 13)

- (a) Drop PyMC entirely **[RECOMMENDED]** — v4.0 marks `16b` as deprecated anyway
- (b) Keep PyMC for `16b` only with wrapper layer
- (c) Keep dual-backend as-is (NOT recommended; Pitfall 7 silent bug)

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | NumPyro 0.20.1 + ArviZ 0.23.4 pins verified; working reference in `legacy/numpyro_models.py`; MEDIUM for LBA-under-NUTS at scale |
| Features | HIGH for choice-only, LOW-MEDIUM for hierarchical LBA (no precedent) |
| Architecture | HIGH (direct codebase inspection with file paths and line numbers) |
| Pitfalls | HIGH for NumPyro patterns; MEDIUM for hierarchical LBA and WAIC across heterogeneous observables |

**Overall confidence:** HIGH for choice-only hierarchical path; MEDIUM for full 7-model scope including M4. The M4 path is the single confidence-lowering factor.

### Gaps to Address

- **K parameterization conventions** — blocked on parallel Collins research thread. Phase 13 must reference before Phase 14 begins.
- **Hierarchical LBA precedent** — no public NumPyro/JAX hierarchical LBA exists. If Option 1(a) selected, Phase 17 is effectively a research project nested in the milestone.
- **IES-R subscale correlations in N=154 sample** — actual correlations not audited. Phase 13 or 16 must start with collinearity audit on real survey data.
- **NumPyro post-warmup checkpoint-resume pattern** — forum-documented only. Phase 17 must validate on short fit before full M4 run.
- **Compile-time blowup on M6b** — constrained `kappa_total`/`kappa_share` under non-centered hierarchical sampling may compile more slowly. No benchmark exists; the < 60s gate may need relaxation for M6b.

---

## Sources

### Primary (HIGH confidence)
- `.planning/research/STACK.md` — NumPyro/ArviZ/JAX version pins, `numpyro.factor` + pointwise log-lik pattern, LBA-under-NUTS constraints, SLURM dispatch, wall-time budget
- `.planning/research/FEATURES.md` — hBayesDM table-stakes (TS-1..13), differentiators (D-1..10), anti-features (AF-1..11 including load-bearing AF-3 M4 exclusion)
- `.planning/research/ARCHITECTURE.md` — directly-verified codebase findings: P0 broken import (`fit_bayesian.py:43`), float64 process-global flag, schema-parity migration pattern
- `.planning/research/PITFALLS.md` — 10 critical pitfalls with prevention strategies and pitfall-to-phase mapping
- Codebase inspection: `scripts/fitting/legacy/numpyro_models.py`, `scripts/fitting/fit_bayesian.py`, `scripts/fitting/lba_likelihood.py`, `scripts/fitting/mle_utils.py`, `cluster/12_mle_gpu.slurm`, `config.py`, `scripts/15/16/16b/17`

### Secondary (MEDIUM confidence)
- Collins (2018), Collins & Frank (2012), Master et al. (2020), Pedersen & Frank (2020), Sullivan-Toole et al. (2022), Wisco et al. PLoS Comp Bio (2025), Senta et al. (2025)
- Vehtari, Gelman & Gabry (2017) — PSIS-LOO; Vehtari et al. (2021) — rank-normalized R-hat; Betancourt (2018) — HMC; Gelman et al. (2020) — Bayesian workflow
- NumPyro/ArviZ documentation: `numpyro.infer.MCMC`, `arviz.loo`, `arviz.compare`, `arviz.plot_khat`
- hBayesDM: Ahn et al. (2017)

### Tertiary (LOW confidence — needs implementation-time validation)
- Gunawan et al. (2020) PMwG sampler — R-only, cited as precedent but not Python reference
- Brown & Heathcote (2008) + Annis et al. (2017) LBA Bayesian fitting — hierarchical LBA rare; PSIS-LOO + LBA failure modes documented in rtdists community
- NumPyro `post_warmup_state` checkpoint-resume — forum threads only
- Regularized horseshoe prior specification for RL contexts — Piironen & Vehtari (2017) is the reference, but no canonical NumPyro RL example

---

*Research completed: 2026-04-11*
*Ready for roadmap: BLOCKED on Open Decision 1 (M4 hierarchical scope). Open Decisions 2 and 3 can be deferred to Phase 13 / Phase 16.*
