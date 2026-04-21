# Phase 13 Research — Infrastructure Repair & Hierarchical Scaffolding

**Phase:** 13 (v4.0 milestone)
**Researched:** 2026-04-11 (updated with Senta 2025 + Ehrlich 2025 findings)
**Confidence:** HIGH — K bounds now anchored to Senta, Bishop, Collins 2025 PLOS Comp Biol (the project's reference paper) with full citation verified

## UPDATE (2026-04-11): Senta 2025 + Ehrlich 2025 Re-Audit

After the initial research draft, a targeted search for recent Collins-lab 2025 papers surfaced two critical references that materially affect Phase 13's K parameterization recommendation:

### Senta, Bishop, Collins (2025) — the project's reference paper

**Full citation:** Senta JD, Bishop SJ, Collins AGE (2025). *Dual process impairments in reinforcement learning and working memory systems underlie learning deficits in physiological anxiety*. **PLoS Computational Biology** 21(9): e1012872. DOI: [10.1371/journal.pcbi.1012872](https://doi.org/10.1371/journal.pcbi.1012872). Received Feb 11 2025; Accepted Sep 2 2025; Published Sep 26 2025. Data: https://osf.io/w8ch2/

**Key findings that change Phase 13:**

1. **K parameterization (page 20, verbatim):** "the working memory capacity parameter K, which was constrained to the continuous interval **[2, 6]**". This is the Collins-lab-canonical bound as of 2025 — tighter than Collins 2014's discrete {0..6} and tighter than the initial recommendation in this research (which was [1, 6] for v3.0 schema parity). **The initial K lower=1 rationale is now weaker**; Senta 2025 supersedes McDougle & Collins 2021 as the primary reference for the formula and bounds.

2. **Fitting method:** pure per-participant MLE via MATLAB `fmincon` with 20 random starts. **NOT hierarchical Bayesian.** The project's v4.0 hierarchical Bayesian extension is a legitimate novel contribution over the reference paper — the planner can cite this as scientific justification in the manuscript methods.

3. **Winning model (Model #5, "RLWM_asymbias_2r") has 9 parameters:** α_RL, β_test, ε, **ρ_low, ρ_high (split WM confidence by ns vs K)**, K, φ, η_WM (negative feedback neglect in WM only), i (RL-WM information sharing). **The split-ρ mechanism is NOT in our project's M1-M6b model zoo.** This is a Phase 17+ extension opportunity, flagged but out of Phase 13 scope.

4. **Perseveration κ EXPLICITLY rejected from winning model** (page 20-21): "addition of a perseveration choice kernel (Model #6) did not improve model fit, and did not result in significant perseveration or significant changes in main parameters". Senta's winning mechanism is *asymmetric negative feedback neglect* in RL only (the "_asymbias" variant), NOT perseveration. **This reframes the project's scientific arc:** our M3/M6a/M6b may be partially compensating for a missing "asymbias"-like mechanism. Does not change Phase 13 scope (infrastructure only), but flag for the manuscript limitations section and Phase 17+ model zoo extensions.

5. **Task structure comparison:** Senta 2025 uses **deterministic stimulus-action mapping, NO reversals, 13 learning blocks, set sizes {2, 3, 4, 5, 6}**. Our project task has reversals and set sizes {2, 3, 5, 6}. **Tasks are not directly comparable** — the project cannot replicate Senta 2025's fits on their data nor vice versa without task re-engineering. Use as a parameter-bounds reference only.

6. **Clinical finding:** Senta found physiological anxiety (MASQ-AA) correlated with α_RL ↓ AND φ_WM ↑ (both FWE-corrected p=0.040). K itself was NOT reported as an anxiety correlate. This is a "dual process" finding — trauma/anxiety affects BOTH systems, not one. **Implication for our project:** the hierarchical L2 regression in Phase 16 should explicitly test both α_RL and φ_WM (not just κ and K) against IES-R / LEC-5 subscales. Our current plan already does this.

7. **Model comparison (page 22, verbatim):** "Previous research has shown that Bayesian model selection criteria such as the Bayesian Information Criteria (BIC) tend to over-penalize models in the RLWM class [Collins & Frank 2018]. To confirm this in the current data and support our use of AIC as a measure of model fit, we performed a parallel model recovery analysis for the selected RLWM models using BIC. The confusion matrix for this analysis... confirms that data generated from more complex underlying processes tends to be (incorrectly) best-fit by simpler models when BIC is used." **Phase 13 implication:** the schema-parity CSV should still carry a `bic` column for v3.0 MLE back-compat, but the primary Bayesian comparison criterion in Phase 18 must be WAIC/LOO (already planned), with explicit citation of Senta 2025 p.22 for the BIC rejection rationale.

### Ehrlich, Yoo, Collins (2025) — Phase-17+ reference only

**Full citation:** Ehrlich DB, Yoo AH, Collins AGE (2025). *Strategic Control of Working Memory Operations in Dynamic Reward-based Learning*. PsyArXiv preprint, posted May 23 2025. ID: xkzvq. https://osf.io/preprints/psyarxiv/xkzvq_v1. NOT peer-reviewed. Code: https://github.com/dbehrlich/WMCO_analysis

**Why this does NOT change Phase 13:** Ehrlich introduces the **WMCO (Working Memory Content Orchestration)** framework — a discrete-slot process-level model fit via **sequential Monte Carlo approximate Bayesian computation (SMC-ABC)**. This is a likelihood-free simulation-based inference paradigm, fundamentally incompatible with our NUTS/NumPyro gradient-based pipeline. K is treated as a discrete integer slot count, not continuous. **Cannot directly apply to Phase 13.**

**Why this matters for future work:** Ehrlich's task is the closest published match to our reversal-learning paradigm (it has rapid within-block reversals after 2-4 correct responses). Their best-fit K is 4-6 with substantial heterogeneity, and they explicitly show K interacts with "strategic control" parameters (Overwrite_Rew, Forget_Rew) in ways that can make high K *worse* for performance. **Phase 17+ implication:** if our project wants to publish a full-pipeline comparison with the most recent RLWM literature, SMC-ABC (e.g., via `sbi` Python package) is a candidate methodology. Out of v4.0 scope.

### REVISED K Parameterization Recommendation

**Original recommendation:** K ∈ [1, 6] continuous via `1.0 + 5.0 * Phi_approx(mu + sigma * z)`. Rationale: schema parity with v3.0 MLE.

**Revised recommendation (HIGH confidence):** **K ∈ [2, 6] continuous via `2.0 + 4.0 * Phi_approx(mu + sigma * z)`**. Rationale: match Senta, Bishop, Collins (2025) PLOS Comp Biol exactly, since this is the project's reference paper and the canonical Collins-lab 2025 convention. `parameterization_version` becomes `"v4.0-K[2,6]-phiapprox"`.

**Implication for v3.0 schema parity:** breaks — v3.0 MLE uses K ∈ [1, 7]. Phase 14's constrained-K MLE refit (K-02, K-03) should also adopt K ∈ [2, 6] so the MLE and Bayesian pipelines share the same convention. This adds a scope line to Phase 14: MLE refit uses Senta 2025 bounds, not "Collins K research" freely interpreted. Update `K_PARAMETERIZATION.md` accordingly.

**Alternative (Option B, NOT recommended):** keep K ∈ [1, 6] and document the extension below Senta's lower bound as an intentional "sensitivity check" — tests whether any participants are best-fit at K < 2 (indicating effectively-no-WM behavior). This is a scientifically defensible fallback position if the user does not want to break v3.0 schema parity cleanly.

### Revised `docs/K_PARAMETERIZATION.md` draft

The markdown document should now cite **Senta, Bishop, Collins (2025) PLOS Comp Biol 21(9):e1012872** as the primary authority with page 20 eq. direct quote. Collins 2014 and McDougle & Collins 2021 remain as secondary references (historical and continuous-precedent respectively). Document structure:

1. TL;DR — "K ∈ [2, 6] continuous via non-centered Phi_approx, matching Senta, Bishop, Collins (2025) Model #5"
2. Formula: `w = ρ · min(1, K/ns)`, citing Senta eq. 5
3. Non-centered transform: `K_i = 2.0 + 4.0 * Phi_approx(mu_K_pr + sigma_K_pr * z_K_i)` with standard group priors
4. Historical: Collins 2014 (discrete {0..6}), McDougle 2021 (continuous [2, 5]), Senta 2025 (continuous [2, 6])
5. Rationale for not going lower than 2: Senta convention + ns=2 saturation; scientific interpretation of K<2 as "effectively no WM"
6. Rationale for not going higher than 6: task max ns=6, K>6 is structurally indistinguishable

### Bayesian CSV schema — one addition

Add explicit note to INFRA-04 implementation: **BIC is still reported in the CSV for v3.0 MLE back-compat**, but the `converged` flag and primary model comparison logic use WAIC/LOO (Phase 18, CMP-03). Cite Senta 2025 p.22 in the `bayesian_summary_writer.py` docstring.

### Updated "Open Questions — Planner-Level Decisions"

**Question 2 (K lower bound)** is now **answered with HIGH confidence**: lower bound = 2, match Senta 2025. This was a MEDIUM-confidence call before the 2025 papers surfaced; it is now HIGH.

**New Question 8 (out of Phase 13 scope but must be flagged for ROADMAP):** should the v4.0 model zoo be extended to include a "Senta asymbias" variant (asymmetric negative feedback neglect in RL only, with η_RL parameter) and/or a "split-ρ" variant (ρ_low, ρ_high by ns vs K)? These are the winning-model mechanisms in Senta 2025 that are NOT in our current M1-M6b zoo. **Recommendation: defer to v4.0 Phase 17 research (after hierarchical framework is validated on M1-M6b) or v5.0.** Add to `.planning/REQUIREMENTS.md` v2 (future) requirements list.

### Updated Sources

- [Senta, Bishop, Collins 2025 — PLOS Comp Biol 21(9):e1012872](https://doi.org/10.1371/journal.pcbi.1012872) — **HIGH confidence, primary reference paper**
- [Senta 2025 — bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2025.02.14.638024v1)
- [Senta 2025 data (OSF)](https://osf.io/w8ch2/)
- [Ehrlich, Yoo, Collins 2025 — PsyArXiv xkzvq](https://osf.io/preprints/psyarxiv/xkzvq_v1) — MEDIUM confidence (preprint, not peer-reviewed)

---

## ORIGINAL RESEARCH (pre-Senta 2025 audit, retained for reference)

---

## Executive Summary — The Must-Know Facts

1. **Collins K is continuous in [2, 5] in Collins-lab modern work, NOT discrete.** Collins 2014 (J Neurosci) fit K iteratively over discrete {0..6} via fmincon, but McDougle & Collins 2021 (Psychon Bull Rev) — the most recent Collins-lab reference — uses **continuous C in [2, 5]** with MLE via fmincon. Project's current MLE bounds `[1.0, 7.0]` are wider than Collins-lab convention. The WM weighting formula is uniformly `w = rho * min(1, K/ns)` across all Collins papers. **Recommendation: for Bayesian hierarchical fitting, use continuous K in [1, 6] — rationale in the Collins K section.**

2. **The pointwise log-lik refactor is a one-line change per likelihood function.** `q_learning_block_likelihood` at `jax_likelihoods.py:474` already calls `lax.scan` which returns `(carry, log_probs)` — but the function discards `log_probs` and returns only the scalar `log_lik_total`. **Option A** (refactor existing functions to return `(total, per_trial_vector)`) is dramatically cleaner than post-hoc duplicate functions. All 7 `*_block_likelihood` functions follow the same `lax.scan` idiom and the fix is structurally identical for each.

3. **hBayesDM non-centered convention is `Phi_approx(mu_pr + sigma * z) * upper_bound`, NOT the existing legacy pattern `expit(logit(mu) + sigma * z)`.** The legacy `numpyro_models.py` (lines 144–158) uses a non-standard pattern where the group mean is sampled on the constrained scale as `Beta(3,2)` and then transformed back via `logit` — this is a *centered*-on-the-group-mean parameterization that works numerically but departs from the canonical hBayesDM template. **The new helper module must use the hBayesDM convention**: `mu_pr ~ Normal(0, 1)`, `sigma_pr ~ HalfNormal(0.2)`, `theta_unc = mu_pr + sigma_pr * z`, `theta = Phi_approx(theta_unc) * upper_bound`.

4. **Dependency pins are verified available**: `numpyro==0.20.1` (PyPI March 25, 2025; requires Python ≥3.11, JAX ≥0.7) and `arviz==0.23.4` (PyPI Feb 4, 2026) — both confirmed. Project's JAX 0.9.0 + Python 3.11 satisfy constraints. `arviz==0.23.4` is the deliberate pre-1.0 freeze — ArviZ 1.0.0 shipped March 2, 2026 and would break the `InferenceData` API. `netcdf4` is NOT a hard dependency of ArviZ; required only for `InferenceData.to_netcdf()` and must be added explicitly.

5. **`cluster/12_mle_gpu.slurm` already has the JAX compilation cache pattern working** (lines 106–118): env var `JAX_COMPILATION_CACHE_DIR` set to `/scratch/${_PROJECT}/${USER}/.jax_cache_gpu`, `JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0`, fallback to `fc37` if `$PROJECT` is unset, `mkdir -p` with error fallback. **The new `cluster/13_bayesian_gpu.slurm` should be a fork of `12_mle_gpu.slurm`** with this cache block preserved verbatim, not reinvented.

6. **PyMC removal blast radius is minimal**: exactly 4 non-legacy touchpoints: `pyproject.toml` (two places), `scripts/16b_bayesian_regression.py` (the dual-backend detection block at lines 60–83 + `_run_pymc` at 241), `validation/test_pymc_integration.py`, and `pytest.ini` + `pyproject.toml` `requires_pymc` marker definitions. No `environment.yml` / `environment_gpu.yml` PyMC references exist today (already PyMC-clean on the conda side).

---

## Collins K Parameterization — Concrete Recommendation

**Verified Collins-lab K conventions across papers:**

| Paper | K type | Bounds | Fitting | Set sizes (ns) | Notes |
|-------|--------|--------|---------|----------------|-------|
| Collins & Frank 2012 (Eur J Neurosci) | Discrete | {0..6} | MLE fmincon, iterated over K | 2, 3, 4, 5, 6 | Founding paper |
| Collins, Brown, Gold, Waltz, Frank 2014 (J Neurosci) | **Discrete** | **{0,1,2,3,4,5,6}** | "performed iteratively for capacities n = {0..6}" then fmincon with 50 random starts | 2, 3, 4, 5, 6 | `w = ρ × min(1, K/nS)`. Patients median K=2, controls median K=3. |
| McDougle & Collins 2021 (Psychon Bull Rev) | **Continuous** | **[2, 5]** | MLE via fmincon, 40 iterations | Not RLWM task (instrumental) | `w = ρ * min(1, C/nS)`. Uniform convention `α=[0,1]`, `C=[2,5]` |
| Master et al. 2020 (Dev Cogn Neurosci) | Continuous (inferred) | Unknown | Hierarchical Bayesian | 2, 3 | Full methods PDF inaccessible |

**Key structural facts:**
- The WM weight formula is **uniformly** `w = rho * min(1, K/ns)` across every Collins-lab RLWM paper. `ns` is the current-block set size.
- This project's task uses set sizes **{2, 3, 5, 6}** (verified). Max observed ns = 6, so K values above 6 are meaningless (K/ns clamps to 1 for any ns ≤ K).
- Collins 2014 recovered mostly K ∈ [2, 4] with fmincon. Current MLE `capacity_ci_upper` values in `wmrl_individual_fits.csv` cluster near 3.5 — consistent with Collins 2014 recovery.

**Recommendation for v4.0 Bayesian fitting:**

```
K ∈ [1, 6] continuous.
Transform: K = 1 + 5 * Phi_approx(mu_K_pr + sigma_K_pr * z_K)
Group priors: mu_K_pr ~ Normal(0, 1), sigma_K_pr ~ HalfNormal(0.2)
Individual prior: z_K ~ Normal(0, 1)  [standard non-centered]
```

**Rationale (four reasons):**

1. **Discrete K is incompatible with NUTS** — HMC requires continuous, differentiable log-posteriors. Collins 2014's discrete fmincon-iteration approach would require refitting across 7 K values per chain per sample, which is not possible in a single NumPyro model.
2. **Lower bound 1, not 2** — Collins-lab modern continuous work uses K ∈ [2, 5], but the project's existing MLE pipeline uses K ∈ [1, 7] and produces stable fits. Keeping lower=1 preserves schema parity with v3.0 MLE artifacts. K=1 is documented as "effectively no WM" corner of the posterior.
3. **Upper bound 6, not 7** — the task uses max ns=6, so K=7 is structurally indistinguishable from K=6 (K/ns saturates at K=ns). Tightening removes a non-identifiable edge. **This IS a breaking change from v3.0 MLE** — must bump `parameterization_version` and re-run MLE with new bounds for schema-parity comparison.
4. **`Phi_approx` over sigmoid** — hBayesDM uses `Phi_approx` for historical Stan performance reasons. In NumPyro/JAX, `jax.scipy.stats.norm.cdf` is equally fast as `sigmoid`, but using the standard normal CDF keeps the prior-implied shape consistent with hBayesDM RLWM literature (prior on K centered at [1,6] midpoint = 3.5, matching Collins 2014 control median).

**Parameter recovery correlation for K**: NOT reported in any accessible Collins paper. Phase 13/14 must establish this de novo.

---

## Non-Centered Parameterization — Exact Transforms for 8 Parameters

Using hBayesDM convention throughout: `theta_unc = mu_pr + sigma_pr * z`, then apply link.

| Parameter | Models | Native scale | Group priors | Individual transform | Rationale |
|-----------|--------|--------------|-------------------|--------------------|-----------|
| **alpha_pos** | All | [0, 1] | `mu ~ Normal(0, 1)`, `sigma ~ HalfNormal(0.2)` | `Phi_approx(mu + sigma * z)` | hBayesDM learning-rate convention |
| **alpha_neg** | All | [0, 1] | Same | `Phi_approx(mu + sigma * z)` | Same as alpha_pos |
| **epsilon** | All except M4 | [0, 1] | `mu ~ Normal(-2.5, 1)` (≈0.006 via Phi_approx), `sigma ~ HalfNormal(0.2)` | `Phi_approx(mu + sigma * z)` | Lower prior mean matches legacy `Beta(1,19)` intent |
| **phi** | M2-M6b | [0, 1] | `mu ~ Normal(-0.8, 1)` (≈0.21 via Phi_approx, matches `Beta(2,8)` mean), `sigma ~ HalfNormal(0.2)` | `Phi_approx(mu + sigma * z)` | WM decay rate |
| **rho** | M2-M6b | [0, 1] | `mu ~ Normal(0.8, 1)` (≈0.79, matches `Beta(5,2)` mean), `sigma ~ HalfNormal(0.2)` | `Phi_approx(mu + sigma * z)` | WM reliance |
| **capacity (K)** | M2-M6b | [1, 6] | `mu ~ Normal(0, 1)`, `sigma ~ HalfNormal(0.2)` | **`1.0 + 5.0 * Phi_approx(mu + sigma * z)`** | See Collins K section. `+5.0 *` scales [0,1] to 5 units; `+1.0` shifts to [1, 6]. |
| **kappa** (M3) | M3 | [0, 1] | `mu ~ Normal(-2, 1)` (≈0.02), `sigma ~ HalfNormal(0.2)` | `Phi_approx(mu + sigma * z)` | Perseveration; prior peaked at ~0 matches MLE recovery |
| **kappa_s** (M6a) | M6a | [0, 1] | Same as kappa | `Phi_approx(mu + sigma * z)` | Stimulus-specific perseveration |
| **phi_rl** (M5) | M5 | [0, 1] | `mu ~ Normal(-0.8, 1)`, `sigma ~ HalfNormal(0.2)` | `Phi_approx(mu + sigma * z)` | RL forgetting; same prior family as phi |
| **kappa_total** (M6b) | M6b | [0, 1] | `mu ~ Normal(-2, 1)`, `sigma ~ HalfNormal(0.2)` | `Phi_approx(mu + sigma * z)` | Stick-breaking budget |
| **kappa_share** (M6b) | M6b | [0, 1] | `mu ~ Normal(0, 1)` (uniform-ish), `sigma ~ HalfNormal(0.2)` | `Phi_approx(mu + sigma * z)` | Stick-breaking split — decode inside likelihood |

**Critical stick-breaking note for M6b:** the current MLE CSV stores `kappa_total` and `kappa_share` as raw [0,1] variables and decodes inside the objective function (`mle_utils.py:370-378`: `kappa = kappa_total * kappa_share; kappa_s = kappa_total * (1 - kappa_share)`). The Bayesian model should **replicate this pattern exactly** — sample two independent non-centered bounded variables, decode via `numpyro.deterministic` inside the likelihood. Do NOT attempt a Dirichlet prior on the simplex directly — the unconstrained stick-breaking parameterization is strictly better under NUTS.

**M4 LBA parameters** (out of Phase 13 scope; documented for Phase 17 planner):
- `v_scale`: [0.1, 20.0], scaled `exp(mu_pr + sigma * z)` or direct `log(v_scale)`
- `A`: [0.001, 2.0], Phi_approx scaled
- `delta`: [0.001, 2.0], **sample `log(delta)` directly** — b-A funnel pathology
- `t0`: [0.05, 0.3], Phi_approx scaled
- (no epsilon — M4 has no random-action noise)

**Rationale for Phi_approx over expit:** hBayesDM established `Phi_approx` as the de facto RLWM convention. In NumPyro, use `jax.scipy.stats.norm.cdf` directly. Matches Stan-literature prior shapes from Ahn/Haines/Zhang (2017). **Do NOT use the legacy pattern** `expit(logit(mu_constrained) + sigma * z)` — it conflates constrained and unconstrained scales.

---

## `compute_pointwise_log_lik()` Implementation — Decision

**Decision: Option A** — refactor the existing `*_block_likelihood` functions to return `(total_log_lik, per_trial_log_probs)` instead of just `total_log_lik`. Pointwise log-lik is computed post-sampling via `jax.vmap` over (chains, samples).

**Evidence this is the cleanest path (from `jax_likelihoods.py:474`):**

```python
# EXISTING CODE — line 474:
(Q_final, log_lik_total), log_probs = lax.scan(step_fn, init_carry, scan_inputs)
return log_lik_total  # <- log_probs is computed but DISCARDED
```

`lax.scan`'s second return (`log_probs`) is already the per-trial log-probability vector of shape `(n_trials,)`. The fix is one line per block function:

```python
# NEW CODE:
(Q_final, log_lik_total), log_probs = lax.scan(step_fn, init_carry, scan_inputs)
return log_lik_total, log_probs  # now returns tuple
```

**Why Option A beats B and C:**
- **Option A (refactor existing)**: One-line change per block function (7 total). Gradient-compatible. Zero code duplication. Preserves JIT cache keys.
- **Option B (parallel `*_pointwise_likelihood`)**: 7+ duplicate functions. Maintenance burden. Rejected.
- **Option C (post-hoc re-run via vmap)**: Requires second JAX compile. Slower. Rejected.

**`compute_pointwise_log_lik()` implementation sketch** (for `scripts/fitting/bayesian_diagnostics.py`):

```python
def compute_pointwise_log_lik_qlearning(
    mcmc: MCMC,
    participant_data_stacked: dict,
    *,
    num_stimuli: int = 6,
    num_actions: int = 3,
    q_init: float = 0.5,
) -> jnp.ndarray:
    """Returns shape (chains, samples, participants, n_blocks * max_trials)."""
    samples = mcmc.get_samples(group_by_chain=True)
    alpha_pos_all = samples["alpha_pos"]
    alpha_neg_all = samples["alpha_neg"]
    epsilon_all = samples["epsilon"]

    def _per_participant(alpha_pos_i, alpha_neg_i, epsilon_i, pdata):
        _, pointwise = q_learning_multiblock_likelihood_stacked(
            **pdata, alpha_pos=alpha_pos_i, alpha_neg=alpha_neg_i, epsilon=epsilon_i,
            num_stimuli=num_stimuli, num_actions=num_actions, q_init=q_init,
        )
        return pointwise.reshape(-1)

    vmapped = jax.jit(
        jax.vmap(jax.vmap(_per_participant, in_axes=(0, 0, 0, None)),
                 in_axes=(0, 0, 0, None))
    )

    out = [vmapped(alpha_pos_all[..., i], alpha_neg_all[..., i], epsilon_all[..., i],
                   participant_data_stacked[pid])
           for i, pid in enumerate(participant_data_stacked)]
    return jnp.stack(out, axis=2)
```

**ArviZ integration:**

```python
def build_inference_data_with_loglik(mcmc, pointwise_log_lik):
    idata = az.from_numpyro(mcmc)
    idata.add_groups(
        log_likelihood={"obs": pointwise_log_lik},
        coords={"participant": list(range(pointwise_log_lik.shape[2])),
                "trial": list(range(pointwise_log_lik.shape[3]))},
        dims={"obs": ["participant", "trial"]},
    )
    return idata
```

Then `az.waic(idata)` and `az.loo(idata)` work natively.

**Padding masks:** refactored `*_block_likelihood` already returns `log_prob_masked = log_prob * valid` — padded trials carry zero log-prob. Downstream WAIC/LOO computations naturally ignore zero-contribution trials. No extra masking needed.

---

## JAX Compilation Cache Setup for Monash M3

**Status:** The existing `cluster/12_mle_gpu.slurm:106-118` pattern is **already correct** for v4.0. Copy verbatim into `cluster/13_bayesian_gpu.slurm` with NO changes.

**Exact env var pattern:**

```bash
# JAX Compilation Cache (from cluster/12_mle_gpu.slurm)
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"
```

**Compile-time CI test for INFRA-08 < 60s gate:**

The < 60s gate should be measured on a **warm** cache (second invocation). Cold compile of Q-learning hierarchical on A100 is 30-90s; warm is 5-15s + 20-30s setup. Test pattern:

```python
def test_compile_gate():
    # Warm cache with trivial sample
    model_args = _make_minimal_synthetic_data(n_ppts=2, n_blocks=2, n_trials=20)
    mcmc = MCMC(NUTS(qlearning_hierarchical_model), num_warmup=5, num_samples=5,
                num_chains=1, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), **model_args)  # cold: slow

    t0 = time.monotonic()
    mcmc2 = MCMC(NUTS(qlearning_hierarchical_model), num_warmup=5, num_samples=5,
                 num_chains=1, progress_bar=False)
    mcmc2.run(jax.random.PRNGKey(1), **model_args)
    elapsed = time.monotonic() - t0
    assert elapsed < 60.0
```

---

## Dependency Pin Verification

| Package | Target Pin | Availability | Python/JAX compat | Notes |
|---------|-----------|--------------|-------------------|-------|
| `numpyro` | `==0.20.1` | **VERIFIED** on PyPI (2025-03-25) | Requires Python ≥3.11, JAX ≥0.7. Project has 3.11 + JAX 0.9.0 → **OK** | Maintenance patch over 0.20.0 |
| `arviz` | `==0.23.4` | **VERIFIED** on PyPI (2026-02-04) | Requires Python ≥3.10 → **OK** | Deliberate pre-1.0 freeze; ArviZ 1.0.0 (2026-03-02) breaks `InferenceData` |
| `netcdf4` | No version pin needed | Always available on conda-forge | HDF5 binary | Required for `InferenceData.to_netcdf()`; add explicitly |
| `pymc` | **REMOVE** | n/a | n/a | Only used by `scripts/16b_bayesian_regression.py` fallback |
| `pytensor` | **REMOVE** (PyMC's backend) | n/a | n/a | Only pulled in by PyMC |

**Cascade check:** PyMC is imported only in `scripts/16b_bayesian_regression.py` and `validation/test_pymc_integration.py`. Safe to remove.

**`pyproject.toml` updated `[bayesian]` extra:**

```toml
bayesian = [
    "numpyro==0.20.1",
    "arviz==0.23.4",
    "netcdf4",
]
# NOTE: pymc removed v4.0 (INFRA-07). 16b uses NumPyro-only.
```

**`environment_gpu.yml`** add to pip section:
```yaml
    - numpyro==0.20.1
    - arviz==0.23.4
```

Add `netcdf4` to conda deps (binary package).

---

## Schema-Parity CSV Exact Schema

**Existing MLE CSV schemas (verified via `head -2`):**

| Model | Columns |
|-------|------|
| **qlearning** (M1) | `participant_id, alpha_pos, alpha_neg, epsilon, nll, aic, bic, aicc, pseudo_r2, grad_norm, hessian_condition, hessian_invertible, <param>_se × 3, <param>_ci_lower × 3, <param>_ci_upper × 3, n_trials, converged, n_successful_starts, n_near_best, at_bounds, high_correlations` |
| **wmrl** (M2) | Same + `phi, rho, capacity` and their `_se`/`_ci_*` columns |
| **wmrl_m3** | wmrl + `kappa` with its `_se`/`_ci_*` |
| **wmrl_m5** | wmrl_m3 + `phi_rl` |
| **wmrl_m6a** | wmrl_m3 with `kappa_s` replacing `kappa` |
| **wmrl_m6b** | wmrl_m3 + `kappa_share`, with `kappa_total` replacing `kappa` |
| **wmrl_m4** | `alpha_pos, alpha_neg, phi, rho, capacity, kappa, v_scale, A, delta, t0` + basic outcome columns — **NO** Hessian/SE/CI columns |

**Bayesian schema-parity CSV:**

```
# Identical to MLE schema columns (same order)
participant_id, <param_1>, ..., <param_k>, nll, aic, bic, aicc, pseudo_r2

# Bayesian replacements (NO grad_norm, hessian_*, _se, _ci_*):
<param_1>_hdi_low, <param_1>_hdi_high, <param_1>_sd,
...
<param_k>_hdi_low, <param_k>_hdi_high, <param_k>_sd,

# Convergence diagnostics (per-fit summary):
max_rhat, min_ess_bulk, num_divergences,

# Standard outcomes
n_trials, converged, at_bounds,  # converged = max_rhat < 1.01 AND min_ess_bulk > 400 AND num_divergences == 0

# Parameterization metadata
parameterization_version,  # e.g., "v4.0-K[1,6]-phiapprox"
```

**Key design decisions:**
1. Use **posterior MEAN** for `<param>` columns (not median) — matches MLE point-estimate semantics, avoids surprising downstream consumers
2. Use **95% HDI** for `_hdi_low`/`_hdi_high` — matches `az.summary` default
3. Use **posterior STD** for `_sd` (not "SE" — frequentist term)
4. Include `parameterization_version` as first-class column (redundant per-row but trivial load-time validation)
5. Per-participant-fit `max_rhat`, `min_ess_bulk`, `num_divergences` (per-parameter is overkill; available in NetCDF idata)
6. **Drop `high_correlations`** (Hessian-based, not computable from MCMC). Downstream scripts don't consume it.

**Reference CSV for pytest**: synthesize one row with placeholder values from the MLE schema into `scripts/fitting/tests/fixtures/qlearning_bayesian_reference.csv`. Assert column order + names match; values come from actual fit.

**`parameterization_version` validation helper** in `config.py` or new `scripts/fitting/parameterization_registry.py`:

```python
EXPECTED_PARAMETERIZATION = {
    "qlearning": "v4.0-phiapprox",
    "wmrl": "v4.0-K[1,6]-phiapprox",
    "wmrl_m3": "v4.0-K[1,6]-phiapprox",
    "wmrl_m5": "v4.0-K[1,6]-phiapprox",
    "wmrl_m6a": "v4.0-K[1,6]-phiapprox",
    "wmrl_m6b": "v4.0-K[1,6]-phiapprox-stickbreaking",
    "wmrl_m4": "v4.0-K[1,6]-phiapprox-lba",
}

def load_fits_with_validation(path: Path, model: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "parameterization_version" not in df.columns:
        raise ValueError(f"{path} lacks parameterization_version — v3.0 legacy fit")
    expected = EXPECTED_PARAMETERIZATION[model]
    actual = df["parameterization_version"].unique()
    if len(actual) != 1 or actual[0] != expected:
        raise ValueError(f"{path} has {actual}, expected {expected}")
    return df
```

---

## PyMC Removal Checklist

| File | Change |
|------|--------|
| `pyproject.toml` | Remove `"pymc>=5.0"` from `[bayesian]` extra; remove `"pymc.*"` from mypy overrides (line 109); remove `requires_pymc` marker (line 130-133) |
| `pytest.ini` | Remove `requires_pymc` marker definition (line 12) |
| `scripts/16b_bayesian_regression.py` | **Rewrite** backend detection block (lines 59-83): delete PyMC fallback, assert NumPyro available, fail fast. Delete `_run_pymc` function at line 241. Remove PyMC references from docstring and error messages. |
| `validation/test_pymc_integration.py` | **Delete entire file** |
| `docs/03_methods_reference/MODEL_REFERENCE.md` | Remove PyMC references; add NumPyro hierarchical Bayesian section |
| `environment.yml` | No changes (PyMC not currently listed) |
| `environment_gpu.yml` | No changes (PyMC not currently listed) |
| `scripts/fitting/legacy/pymc*.py` | **Leave in place** (already archived) |

**Critical ordering:** PyMC removal must happen AFTER `16b_bayesian_regression.py` is confirmed to work with NumPyro only. Make NumPyro a hard requirement first, remove PyMC fallback in the same commit.

**`arviz` is still needed** after PyMC drop — used by NumPyro InferenceData conversion.

---

## Open Questions — Planner-Level Decisions

1. **Collins K recovery validation** — No published r-value. Is a K recovery validation task in Phase 13 scope (prerequisite to Phase 14 K refit) or deferred? **Recommendation: defer to Phase 14** alongside the K refit itself.

2. **K lower bound: 1 or 2?** Recommended **1** for schema parity with v3.0. Alternative (K ∈ [2, 6]) is Collins-literature-aligned but breaks parity. **Recommendation: 1, with K < 1.5 flagged as "effectively no WM" in docs.**

3. **Continuous vs rounded K in posterior reporting.** **Recommendation: continuous in CSV, with sidecar `K_rounded` column** for Collins-style integer comparisons.

4. **INFRA-08 compile gate — cold or warm?** **Recommendation: warm** (second invocation only). Mark first invocation as "expected slow".

5. **Pointwise log-lik refactor signature** — breaks MLE callers if `return scalar` → `return (scalar, vector)`. **Recommendation: keyword-only flag** `return_pointwise: bool = False`, default preserves current behavior, Bayesian path opts in.

6. **`numpyro_helpers.py` vs `bayesian_diagnostics.py` module split.** **Recommendation: keep separate** per ROADMAP. Zero cross-dependency.

7. **M4 pointwise log-lik in Phase 13 or defer?** M4 is in `lba_likelihood.py` not `jax_likelihoods.py`. **Recommendation: defer to Phase 17** since M4 hierarchical is Phase 17.

---

## Key Files Referenced

- `scripts/fitting/legacy/numpyro_models.py` — legacy file to resurrect; INFRA-01
- `scripts/fitting/fit_bayesian.py:43` — broken import; INFRA-01
- `scripts/fitting/jax_likelihoods.py:440-476` — `lax.scan` returns `log_probs` that gets discarded (Option A target)
- `scripts/fitting/mle_utils.py:30-106` (bounds) and `:370-378` (M6b stick-breaking decode)
- `scripts/fitting/lba_likelihood.py` — M4 per-trial refactor target (deferred to Phase 17)
- `scripts/16b_bayesian_regression.py:59-83` (PyMC fallback) and `:241` (`_run_pymc`)
- `cluster/12_mle_gpu.slurm:106-118` — JAX cache pattern to reuse verbatim in `cluster/13_bayesian_gpu.slurm`
- `pyproject.toml:43-47` (`[bayesian]` extra) and `:109` (mypy PyMC entry)
- `environment_gpu.yml` — add `netcdf4`, `numpyro==0.20.1`, `arviz==0.23.4`
- `output/mle/*_individual_fits.csv` — reference MLE CSV schemas
- `.planning/research/STACK.md:87-170` — architectural decision on pointwise log-lik
- `.planning/research/PITFALLS.md:11-47, 134-147` — non-centered and K pitfalls

---

## Confidence Assessment

| Area | Level | Reason |
|------|-------|--------|
| Collins K (discrete vs continuous, bounds, formula) | HIGH | Verified from Collins 2014 PMC4188972 and McDougle & Collins 2021 PMC7854965 via WebFetch |
| Non-centered parameterization (hBayesDM) | HIGH | Verified from hBayesDM `ra_prospect.stan` source |
| Pointwise log-lik refactor approach (Option A) | HIGH | Direct inspection of `jax_likelihoods.py:474` — `log_probs` already computed, just discarded |
| JAX compilation cache env var setup | HIGH | Direct inspection of working `cluster/12_mle_gpu.slurm` |
| `numpyro==0.20.1` pin availability | HIGH | Verified on PyPI + GitHub releases |
| `arviz==0.23.4` pin availability | HIGH | Verified on PyPI; ArviZ 1.0.0 confirmed exists |
| `netcdf4` dependency status | MEDIUM | ArviZ docs 403'd; standard guidance |
| Schema-parity CSV design | HIGH | Direct inspection of all 7 MLE CSVs |
| PyMC removal blast radius | HIGH | Direct grep verification |
| Collins K parameter recovery correlation | LOW | Not reported in accessible papers — must be established empirically |
| K lower bound 1 vs 2 decision | MEDIUM | Schema parity vs Collins convention tradeoff |
| Compile-time gate cold vs warm | MEDIUM | Depends on cluster hardware |

---

## Sources

- [Collins, Brown, Gold, Waltz, Frank 2014 — PMC4188972](https://pmc.ncbi.nlm.nih.gov/articles/PMC4188972/) — HIGH confidence, discrete K ∈ {0..6}
- [McDougle & Collins 2021 — PMC7854965](https://pmc.ncbi.nlm.nih.gov/articles/PMC7854965/) — HIGH confidence, continuous C ∈ [2, 5]
- [Collins & Frank 2012 — CCN Berkeley PDF](https://ccn.studentorg.berkeley.edu/pdfs/papers/CollinsFrank_WMRL.pdf) — MEDIUM confidence, founding paper
- [hBayesDM ra_prospect.stan (GitHub)](https://github.com/CCS-Lab/hBayesDM/blob/develop/commons/stan_files/ra_prospect.stan) — HIGH confidence, canonical Phi_approx pattern
- [Ahn, Haines, Zhang 2017 — PMC5869013](https://pmc.ncbi.nlm.nih.gov/articles/PMC5869013/) — HIGH confidence, hBayesDM package paper
- [NumPyro 0.20.1 on PyPI](https://pypi.org/project/numpyro/0.20.1/) — verified
- [ArviZ 0.23.4 on PyPI](https://pypi.org/project/arviz/0.23.4/) — verified

---

*Research completed: 2026-04-11*
*Ready for planning: yes, with 7 planner-level decisions documented above*
