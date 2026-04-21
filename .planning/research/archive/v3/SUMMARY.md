# Project Research Summary

**Project:** RLWM Trauma Analysis — v3 Model Extensions (M4-M6)
**Domain:** Computational cognitive modeling — JAX MLE pipeline extension
**Researched:** 2026-04-02
**Confidence:** HIGH (stack verified against live codebase; architecture from direct inspection; features from primary literature; pitfalls from established patterns)

---

## Executive Summary

This milestone adds three model extensions to an existing JAX-based MLE fitting pipeline that already fits M1-M3 (Q-learning, WM-RL, WM-RL+perseveration). The pipeline architecture is well-understood: all new models integrate into `jax_likelihoods.py` (likelihood math), `mle_utils.py` (parameter bounds), `fit_mle.py` (objective closures + GPU dispatch), and downstream scripts 14-16. No new libraries are required. The entire implementation is additive — existing M1-M3 functionality is untouched.

The recommended build order is M5 first (RL forgetting, simplest carry structure), then M6a (stimulus-specific perseveration, carry structure change), then M6b (M5+M6a combined), and M4 last (LBA joint RT+choice, highest complexity). This order validates the full pipeline integration pattern with low-risk models before tackling the LBA density implementation. M5 and M6 are P1 thesis-essential; M4 is P2 (add only after RT data availability is confirmed and M5/M6 are validated).

The three hardest technical problems are: (1) LBA numerical stability — NaN propagation from inadmissible parameter combinations requires float64, log-space CDF, safe `jnp.where` masking, and a `b > A` constraint by construction; (2) M5 decay order — Q-values must be decayed before the update for each trial, not after (copying the existing Q-learning step function and adding decay as a postfix produces scientifically incorrect behavior); (3) M4 model comparison — M4 cannot be AIC-compared against M1-M3 because it optimizes a joint choice+RT likelihood while M1-M3 optimize choice-only. All three risks are preventable with upfront design decisions.

---

## Key Findings

### Recommended Stack

The existing JAX 0.9.0 environment satisfies every requirement for M4-M6 with no new installs. The LBA density requires only `jax.scipy.stats.norm.pdf` and `jax.scipy.stats.norm.cdf`, both already available. The one risk is `jaxopt` deprecation (final version 0.8.5, April 2025): the library still works correctly and should not be migrated during this milestone. Float64 must remain enabled globally (`jax_enable_x64 = True`) — already done for existing models, must be verified for the M4 code path.

**Core technologies:**
- JAX 0.9.0: JIT-compiled likelihoods and autodiff for MLE — no changes, already satisfies M4-M6 math
- `jax.scipy.stats.norm`: PDF/CDF for LBA density — already bundled, no install needed
- `jax.lax.scan`: Sequential trial accumulation — same pattern as M1-M3, different carry shape for M6
- `jaxopt.ScipyBoundedMinimize` 0.8.5: L-BFGS-B with bounds — deprecated but functional; do not migrate
- SciPy 1.16.3 + NumPy 2.3.5: Data prep outside JIT — unchanged

**Critical version requirement:** LBA CDF arguments must use `norm.cdf(-z)` not `1 - norm.cdf(z)` and not naive `norm.sf(z)` — verify against JAX's internal implementation before each JAX upgrade. This is a correctness requirement, not just performance.

### Expected Features

**Must have (P1 — thesis defense essential):**
- M5 `phi_RL` likelihood — RL forgetting with correct decay-before-update order; `phi_RL = 0` must recover M3 exactly
- M6a stimulus-specific perseveration likelihood — per-stimulus `last_actions` array in carry (shape `num_stimuli`), not global scalar
- M6b dual perseveration — M5 + M6a combined with stick-breaking reparameterization to enforce `kappa + kappa_s <= 1`
- Parameter bounds + recovery for M5, M6a, M6b — recovery `r >= 0.80` is a hard gate before real-data fitting
- Model comparison M1-M6 (choice-only models on same AIC/BIC scale)
- Trauma analysis (scripts 15/16) extended to new parameters (`phi_RL`, `kappa_s`)

**Should have (P2 — add if RT data confirmed and M5/M6 validated):**
- M4 LBA joint RT+choice likelihood — `lba_log_density` helper + M4 block/multiblock/stacked variants
- M4 RT preprocessing — `preprocess_rt()` with 100ms minimum and 3000ms maximum cutoffs, per-participant `min_rt_clean`
- M4 parameter bounds + recovery — required before any real-data M4 fitting
- M4 separate comparison track in `compare_mle_models.py` — flagged as `observable_type: choice_rt`, excluded from M1-M3 AIC table

**Defer (P3 — post-defense):**
- Bayesian MCMC for M4 — MLE is sufficient for thesis scope; Bayesian triples compute time
- CV-based unified M4 vs M1-M3 comparison — only if committee requires it
- M7+ models (separate WM/RL decay rates, non-linear capacity functions)

**Explicit anti-features (do not build):**
- Free `t0` for M4 — McDougle & Collins (2021) show `t0`/`b` correlation is r = -0.995 when both free; fix `t0 = 150ms`
- Free `s` (within-trial noise) for M4 — fix `s = 0.1` per McDougle & Collins (2021)
- Epsilon retained in M4 — LBA start-point variability `A` already models undirected noise; combining creates redundancy
- Unified supermodel function — JAX scan carry shape must be statically known; M6 needs `(num_stimuli,)` array that cannot be switched from a scalar at runtime

### Architecture Approach

All new code lives inside existing files — no new modules. The three-function pattern `_block_likelihood` / `_multiblock_likelihood` / `_multiblock_likelihood_stacked` is mandatory for every model: the stacked variant is what GPU vmap uses; omitting it causes 10-50x slowdown. M5/M6 are additive extensions to M3's carry structure; M4 is an orthogonal extension that adds RT arrays to `prepare_participant_data` and a new `lba_log_density` helper. Masked padding (`valid` mask applied to every carry update, not just the log-likelihood accumulator) is a critical correctness requirement that must be applied to M5's decay step on padding trials.

**Major components and their changes:**
1. `jax_likelihoods.py` — add `lba_log_density`, M4/M5/M6a/M6b block+multiblock+stacked functions
2. `mle_utils.py` — add `*_BOUNDS` dicts, `*_PARAMS` lists, and transform functions for each new model
3. `fit_mle.py` — add objective closures, GPU objectives, `warmup_jax_compilation` branches, RT extraction in `prepare_participant_data`, argparse extensions
4. `compare_mle_models.py` — pass new model fits; flag M4 as `joint_likelihood` and exclude from M1-M3 AIC table
5. `model_recovery.py` — extend `get_param_names` and `sample_parameters` bounds dispatch
6. `scripts/15` and `scripts/16` — import `*_PARAMS` from `mle_utils.py` (stop hardcoding), extend model dispatch

**No changes needed:**
- `scripts/03_create_task_trials_csv.py` — `rt` column already present in output CSVs (confirmed by direct inspection)
- `scripts/12_fit_mle.py` — CLI entry-point already delegates fully to `fit_mle.py`

### Critical Pitfalls

1. **LBA NaN propagation from inadmissible parameter combinations** — LBA density is undefined when `t - t0 <= 0` or `A >= b`; both produce NaN that propagates silently through scan. Prevention: (a) clamp `t - t0` to minimum `1e-6`, (b) parameterize `A = sigmoid(a_raw) * b * 0.99` to enforce `A < b` by construction, (c) use double `jnp.where` safe-dummy pattern for conditionally undefined branches, (d) use `logcdf`/`logsf` not `log(cdf)` for tail stability. Write boundary-case unit tests before fitting real data.

2. **Invalid AIC comparison: M4 vs M1-M3** — M4 NLL is over `log p(choice, RT)` while M1-M3 NLL is over `log p(choice)` only. M4 will always rank first — this says nothing about cognitive model quality. Prevention: add `observable_type` field to result dicts; `compare_mle_models.py` must refuse to rank M4 alongside M1-M3 in the same AIC table. This design decision must be made before M4 is implemented, not after.

3. **M5 decay applied after update instead of before** — copying M3's Q-learning step function and appending decay at the end inverts the intended order, confounds `phi_RL` with `alpha_pos`, and produces scientifically invalid fits. Prevention: write the M5 step function from scratch with explicit ordering comment `# ORDER MATTERS: decay THEN update`. Unit test: verify non-presented stimulus Q-values decay while presented stimulus reflects the update.

4. **M6 using global `last_action` scalar instead of per-stimulus array** — copy-paste of M3 carry passes a scalar `last_action`; M6 needs `last_actions[num_stimuli]` initialized to `-1`. Wrong implementation produces near-zero `kappa_s` estimates for all participants. Prevention: define the M6 carry structure explicitly before writing any step function code.

5. **M6b kappa + kappa_s constraint unenforced** — naive bounds `[0,1]` on both allow `kappa=0.9, kappa_s=0.9` (sum > 1), producing negative action probabilities and NaN log-likelihood. Prevention: use stick-breaking reparameterization: `kappa = sigmoid(k1)`, `kappa_s = sigmoid(k2) * (1 - kappa)`. Assert `kappa + kappa_s <= 1.0` for 1000 random unconstrained draws before fitting.

---

## Implications for Roadmap

Research points clearly to a 5-phase structure that mirrors the build order recommendation from ARCHITECTURE.md and the priority tiers from FEATURES.md.

### Phase 1: M5 — RL Forgetting

**Rationale:** Smallest incremental change from M3 (one new parameter, same carry shape). Validates the full pipeline integration pattern end-to-end before any structural changes. The highest-risk correctness issue (decay order) is addressable with a simple unit test.

**Delivers:** M5 likelihood, bounds, objectives, GPU path, model comparison, parameter recovery, and trauma analysis integration. A complete vertical slice of the new-model pipeline.

**Implements from FEATURES.md:** `phi_RL` parameter, `phi_RL = 0` boundary condition test, M5 vs M3 comparison, Hessian SE diagnostics for `phi_RL`, Scripts 15/16 extension.

**Critical pitfall to prevent:** Decay-after-update bug (Pitfall 4). Write unit test before any real-data fitting. Run parameter recovery; `phi_RL` identifiability against `alpha_pos` is the gating question — if `r < 0.80`, reconsider M5 scope before adding M6.

**Research flag:** Standard patterns — no additional research needed. M5 is a direct application of existing Collins (2018) + Senta (2025) formulation.

---

### Phase 2: M6a — Stimulus-Specific Perseveration

**Rationale:** Introduces the novel carry structure change (scalar → per-stimulus array) in isolation before combining with M5. Isolating the architectural change makes debugging cleaner.

**Delivers:** M6a likelihood, bounds, objectives, GPU path, model comparison (M1-M6a), parameter recovery, and trauma analysis integration.

**Implements from FEATURES.md:** Per-stimulus `last_actions` carry array, `kappa_s = 0` boundary condition (reduces to M1/M2 without perseveration), M6a vs M3 AIC comparison.

**Critical pitfall to prevent:** Global vs per-stimulus last action (Pitfall 5). Define carry `(Q, WM, last_actions_per_stim)` before writing the step function. Unit test: alternate-stimulus sequence verifying correct per-stimulus tracking.

**Research flag:** Standard patterns — well-defined from M3 architecture. The only novelty is the array indexing in JAX scan carry; this is a solved pattern in JAX.

---

### Phase 3: M6b — Dual Perseveration

**Rationale:** Mechanical composition of M5 and M6a once both are independently validated. The only new engineering concern is the `kappa + kappa_s <= 1` constraint, which should be designed before any code is written.

**Delivers:** M6b likelihood with stick-breaking reparameterization, bounds, objectives, full model comparison table M1-M6b (choice-only), trauma analysis for both `kappa_s` and `phi_RL`.

**Implements from FEATURES.md:** Dual perseveration with constraint, M6b vs M6a vs M3 comparison to determine which perseveration structure is favored by data.

**Critical pitfall to prevent:** Unconstrained dual kappa (Pitfall 7). Design the stick-breaking parameterization first. Also: M6b vs WM confound (Pitfall 9) — run model recovery and require BIC delta > 6 for M6b over M3 before reporting as a finding.

**Research flag:** Standard patterns — stick-breaking in JAX is a single-line implementation following existing sigmoid transforms.

---

### Phase 4: M4 — LBA Joint Choice+RT

**Rationale:** Highest complexity, highest risk. Benefits from pipeline patterns being validated by Phases 1-3. Also depends on RT data availability confirmation (confirmed — `rt` column exists in CSV, no data pipeline changes needed) and RT preprocessing not yet implemented.

**Delivers:** `lba_log_density` helper, M4 block/multiblock/stacked likelihood, RT preprocessing (`preprocess_rt()` with 100ms/3000ms cutoffs), bounds and objectives, parameter recovery, separate M4 comparison track in `compare_mle_models.py`.

**Implements from FEATURES.md:** LBA PDF/CDF in pure JAX (no external library), entropy-weighted drift rates (`v_i = v_scale * pi_i / H_prior`), RT outlier handling, M4-only comparison track, trauma analysis for `v_scale`.

**Critical pitfalls to prevent:** LBA NaN propagation (Pitfall 1) — write boundary-case unit tests before fitting; invalid AIC comparison M4 vs M1-M3 (Pitfall 2) — design `observable_type` gating before any M4 code; `t0` violating per-participant min RT (Pitfall 3) — implement `preprocess_rt()` before the likelihood function.

**Research flag:** Needs deeper research during planning. The LBA density formula details (Navarro & Fuss 2009 vs Brown & Heathcote 2008 parameterization) were noted as LOW confidence in ARCHITECTURE.md. During Phase 4 planning: verify the exact density formula against the published paper and confirm the `H_prior` entropy computation from McDougle & Collins (2021) before implementing.

---

### Phase 5: Integration and Validation

**Rationale:** After all four models are implemented individually, a consolidation phase runs full-pipeline smoke tests, cleans up any hardcoded parameter lists in scripts 15/16, and produces the thesis comparison tables.

**Delivers:** Complete M1-M6 model comparison table (choice-only), separate M4 comparison track, trauma analysis outputs for all new parameters, documentation update.

**Critical pitfall to prevent:** Scripts 15/16 not updated (Pitfall 8) — smoke test `python scripts/15_analyze_mle_by_trauma.py --model all` as the acceptance criterion.

**Research flag:** Standard patterns — no research needed. Pure integration and verification work.

---

### Phase Ordering Rationale

- **M5 before M6:** M5 uses the same carry shape as M3 (scalar `last_action`). Learning the pipeline integration pattern on M5 makes M6's structural change (array carry) easier to introduce cleanly.
- **M6a before M6b:** M6b is M5 + M6a. Composing two known-working pieces is safer than implementing the composition before validating the components.
- **M4 last:** RT data exists (confirmed, no blocker), but M4 adds the most new mathematics (`lba_log_density`), a preprocessing step, and a methodologically distinct comparison track. All three risks can be addressed more confidently after the pipeline is proven for M5/M6.
- **Parameter recovery as intra-phase gates:** Recovery (`r >= 0.80`) is not a separate phase — it is a gate within each phase. If M5 recovery fails, M6 does not begin. This is a firm scientific requirement from Senta et al. (2025) and Palminteri et al. (2017).
- **M4 RT data:** The `rt` column already exists in `task_trials_long.csv` — no data pipeline work needed. This removes the main uncertainty flagged in FEATURES.md.

### Research Flags

Phases needing deeper research during planning:
- **Phase 4 (M4 LBA):** LBA density formula details need verification against Navarro & Fuss (2009) or Brown & Heathcote (2008) during planning. ARCHITECTURE.md rated implementation confidence LOW. Confirm: exact formula for 3-accumulator racing LBA, the `H_prior` entropy computation from McDougle & Collins (2021) Supplementary, and whether `b` should be fixed or free (FEATURES.md suggests free, at 8 total parameters for M4).

Phases with standard patterns (no research needed):
- **Phase 1 (M5):** Directly specified in Collins (2018) + Senta (2025). Carry structure is identical to M3.
- **Phase 2 (M6a):** Per-stimulus array in JAX scan is a well-documented pattern.
- **Phase 3 (M6b):** Stick-breaking parameterization is a one-liner using existing sigmoid transforms.
- **Phase 5 (Integration):** Pure engineering; no domain research needed.

---

## Confidence Assessment

| Area | Confidence | Notes |
|------|------------|-------|
| Stack | HIGH | All technologies verified against live environment; JAX primitives for LBA confirmed via source inspection |
| Features | HIGH (M4/M5), MEDIUM (M6) | M4/M5 specified in primary literature (McDougle & Collins 2021, Collins 2018). M6 inferred from M3 structure + perseveration literature |
| Architecture | HIGH | All integration points confirmed by direct codebase inspection. RT column presence in CSV confirmed. |
| Pitfalls | HIGH | LBA numerical pitfalls from well-established RT modeling; code-level pitfalls from direct pattern inspection in existing likelihood functions |

**Overall confidence:** HIGH

### Gaps to Address

- **LBA density formula:** ARCHITECTURE.md rates this LOW confidence. During Phase 4 planning, verify the exact 3-accumulator density formula against Brown & Heathcote (2008) Table 1 and the McDougle & Collins (2021) Supplementary. Confirm whether the project uses Navarro & Fuss (2009) or the original Brown & Heathcote form.
- **M5 phi_RL identifiability:** Research flags `phi_RL` / `alpha_pos` collinearity as a genuine risk at extreme learning rates. This cannot be resolved before implementation — it is an empirical question answered by running parameter recovery in Phase 1. If `r < 0.80` for `phi_RL`, the response is either to narrow the prior range or to report M5 as exploratory only.
- **M6 kappa_s / WM confound:** Research (Pitfall 9) identifies that WM and stimulus-specific perseveration are mechanistically similar at the behavioral level. Model recovery (fitting M6 data with M3 and vice versa) is the diagnostic. Plan for this in Phase 2/3 validation, not as an afterthought.
- **M4 `b` free vs fixed:** FEATURES.md suggests M4 has 7 or 8 parameters depending on whether `b` is free. McDougle & Collins (2021) fixed `t0` but left `b` free. Resolve before writing bounds dict.
- **jaxopt migration timeline:** `jaxopt` is deprecated. It works for this milestone, but the next major milestone should plan a ~20-line migration to a manual `scipy.optimize.minimize` wrapper with JAX value-and-gradient. Flag in `pyproject.toml` as a known technical debt item.

---

## Sources

### Primary (HIGH confidence)
- Brown & Heathcote (2008), *Cognitive Psychology* 57(3): 153-178 — LBA density formula, parameter conventions
- McDougle & Collins (2021), *Psychonomic Bulletin & Review* 28: 65-84 — M4 LBA design, entropy-weighted drift rates, `s=0.1` and `t0` identifiability constraints
- Collins & Frank (2018), *PNAS* — `phi_RL` RL forgetting parameter specification
- Senta et al. (2025), *PLOS Computational Biology* — WM-RL model formulation, `phi_RL` and `kappa` structures, `r >= 0.80` recovery criterion
- Direct codebase inspection (`jax_likelihoods.py`, `mle_utils.py`, `fit_mle.py`, `compare_mle_models.py`, `model_recovery.py`, scripts 15/16) — all architecture integration points confirmed
- `output/task_trials_long.csv` column header inspection — `rt` column confirmed present, no data pipeline changes needed

### Secondary (MEDIUM confidence)
- JAX 0.9.0 source `jax/_src/scipy/stats/norm.py` — `sf` implementation verified to use `cdf(-x)` form
- JAX GitHub issue #17199 — `norm.sf` precision bug confirmed fixed
- JAX GitHub discussions #6778, #5039 — `jnp.where` NaN gradient pattern
- `jaxopt` PyPI page — deprecation confirmed, 0.8.5 final release April 2025
- Palminteri et al. (2017) — `r >= 0.80` parameter recovery standard
- Katahira (2018), *Journal of Mathematical Psychology* — stimulus-specific vs global perseveration

### Tertiary (LOW confidence)
- Navarro & Fuss (2009) — alternative LBA density formulation; needs verification vs Brown & Heathcote during Phase 4
- `optimistix` migration path for jaxopt — bounded minimize support limited; ecosystem in flux

---

*Research completed: 2026-04-02*
*Ready for roadmap: yes*
