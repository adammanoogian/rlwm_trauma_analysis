# docs/04_methods/ — Methods Documentation Index

Methods notes for published analyses AND supplementary/validation analyses
that do not appear in manuscript/paper.qmd. Each entry points to the
producing script and a short method writeup. Populate entries as new
methods are added or old ones documented.

## Published-in-paper methods

| Topic | Producing script | Method doc |
|---|---|---|
| Task structure and environment | src/rlwm/envs/rlwm_env.py | ../03_methods_reference/TASK_AND_ENVIRONMENT.md |
| Model mathematics | scripts/fitting/jax_likelihoods.py | ../03_methods_reference/MODEL_REFERENCE.md |
| Hierarchical Bayesian architecture | scripts/fitting/numpyro_models.py | [#hierarchical-bayesian-architecture](#hierarchical-bayesian-architecture) |
| Scale orthogonalization (IES-R) | scripts/fitting/level2_design.py | [#scales-orthogonalization-and-audit](#scales-orthogonalization-and-audit) |

## Supplementary / validation methods

| Topic | Producing script | Method doc |
|---|---|---|
| Posterior predictive checks | scripts/05_post_fitting_checks/03_run_posterior_ppc.py | _TODO_ |
| Synthetic-data generation | scripts/03_model_prefitting/01_generate_synthetic_data.py | _TODO_ |
| Parameter sweep | scripts/03_model_prefitting/02_run_parameter_sweep.py | _TODO_ |
| Parameter recovery | scripts/03_model_prefitting/03_run_model_recovery.py | _TODO_ |
| Prior predictive check gate | scripts/03_model_prefitting/04_run_prior_predictive.py | _TODO_ |
| Bayesian parameter recovery | scripts/03_model_prefitting/05_run_bayesian_recovery.py | _TODO_ |
| Posterior-vs-MLE sanity check | validation/compare_posterior_to_mle.py | _TODO_ |

Entries marked _TODO_ are scaffolding. Add short method writeups here
as results are produced or as reviewers ask for them.

---

## Hierarchical Bayesian Architecture

*Merged from docs/HIERARCHICAL_BAYESIAN.md on 2026-04-22 (Phase 29 Plan 02).*

Architecture, Level-2 trauma regression, and validation checklist for the
joint hierarchical inference used by `scripts/04_model_fitting/b_bayesian/fit_bayesian.py`.

This is the canonical reference.  See `.planning/STATE.md` for the current
execution status and `docs/03_methods_reference/MODEL_REFERENCE.md` for the
likelihood math.

---

### 1. Architecture: single-stage joint hierarchical, NOT two-stage PEB

All parameters — L1 individual parameters, L2 group priors, L3 trauma
regression coefficients — are sampled **simultaneously** in one NumPyro
NUTS run.  This is strictly more principled than two-stage PEB (SPM/DCM
style) because:

1. **Uncertainty propagates correctly end-to-end.**  Plug-in two-stage
   approaches use posterior means as point estimates in the group
   regression, which ignores L1 posterior width and inflates Type I error.
2. **Shrinkage happens automatically** via partial pooling — individuals
   with less data or higher noise are pulled harder toward the group mean.
3. **Regression coefficients are tempered** by L1 uncertainty: a trauma
   coefficient cannot be "confident" if the underlying parameter is poorly
   identified in individuals.

Trauma parameters (`beta_lec_kappa`, `beta_lec_kappa_s`,
`beta_lec_kappa_total`, `beta_lec_kappa_share`) are **fit jointly with the
model parameters**, not as a post-hoc step.

#### What a PEB-style two-stage would add

A second-stage regression against covariates NOT in the MCMC design matrix.
Already partially achievable by running
`scripts/06_fit_analyses/bayesian_level2_effects.py` on the posterior NetCDF (descriptive
output, not re-fit).  Not necessary for v4.0 validation because LEC is
already in the joint design.

---

### 2. Hierarchical structure (hBayesDM non-centered convention)

Following Ahn, Haines, Zhang (2017), each bounded parameter uses:

```
mu_pr       ~ Normal(mu_prior_loc, 1.0)          (unbounded group mean)
sigma_pr    ~ HalfNormal(0.2)                    (group variability)
z_i         ~ Normal(0, 1)                       (non-centered offset)
theta_unc_i = mu_pr + sigma_pr * z_i + beta_lec * lec_i  # L2 shift, optional
theta_i     = lower + (upper - lower) * Phi(theta_unc_i)
```

`Phi` is `jax.scipy.stats.norm.cdf` (the `phi_approx` name is conventional).

#### Prior defaults

From `src/rlwm/fitting/numpyro_helpers.py::PARAM_PRIOR_DEFAULTS`:

| Parameter | `lower` | `upper` | `mu_prior_loc` | Implied prior mean on bounded scale |
|---|---|---|---|---|
| `alpha_pos` | 0.0 | 1.0 | 0.0 | 0.50 |
| `alpha_neg` | 0.0 | 1.0 | 0.0 | 0.50 |
| `epsilon` | 0.0 | 1.0 | −2.5 | 0.006 |
| `phi` | 0.0 | 1.0 | −0.8 | 0.21 |
| `rho` | 0.0 | 1.0 | 0.8 | 0.79 |
| `capacity` | 2.0 | 6.0 | 0.0 | 4.0 |
| `kappa`, `kappa_s`, `kappa_total` | 0.0 | 1.0 | −2.0 | 0.023 |
| `kappa_share` | 0.0 | 1.0 | 0.0 | 0.50 |
| `phi_rl` | 0.0 | 1.0 | −0.8 | 0.21 |

Low prior means on `epsilon` and `kappa`-family parameters are calibrated
from MLE fits (quick-006) — most participants have epsilon near 0.02 and
kappa near 0.05.  A `mu_prior_loc` of −2.0 biases the prior toward small
perseveration; see the [Scales Orthogonalization and Audit](#scales-orthogonalization-and-audit)
section for discussion of whether this is conservative for L2 null-testing.

---

### 3. Level-2 coverage table

| Model | L2 design | Beta sites | Hook |
|---|---|---|---|
| M1 `qlearning` | none | 0 | `covariate_lec=None` required |
| M2 `wmrl` | none | 0 | `covariate_lec=None` required |
| M3 `wmrl_m3` | LEC total → κ | `beta_lec_kappa` (1) | probit-scale shift on `kappa_unc` |
| M5 `wmrl_m5` | LEC total → κ | `beta_lec_kappa` (1) | probit-scale shift on `kappa_unc` |
| M6a `wmrl_m6a` | LEC total → κₛ | `beta_lec_kappa_s` (1) | probit-scale shift on `kappa_s_unc` |
| M6b `wmrl_m6b` | LEC total → κ_total, κ_share | `beta_lec_kappa_total`, `beta_lec_kappa_share` (2) | independent shifts on both stick-breaking parameters |
| **M6b-subscale** | 4 covariates × 8 params | **32** (`beta_{cov}_{param}`) | `wmrl_m6b_hierarchical_model_subscale` only |

#### Subscale design (M6b-subscale)

4 covariates with condition number 11.3 (target < 30):
1. `lec_total` — LEC-5 total-events count, z-scored
2. `iesr_total` — IES-R total score, z-scored
3. `iesr_intr_resid` — IES-R intrusion, Gram–Schmidt residualized vs total
4. `iesr_avd_resid` — IES-R avoidance, Gram–Schmidt residualized vs total

IES-R hyperarousal residual is excluded because
`intr_resid + avd_resid + hyp_resid ≡ 0` (rank-deficient 3-residual
submatrix).  Canonical source: `scripts/fitting/level2_design.py`.

---

### 4. Validation checklist (run after each production fit)

#### 4.1 Convergence (automatic gates)

| Check | Target | Where |
|---|---|---|
| Divergences | 0 at some `target_accept_prob ∈ {0.80, 0.95, 0.99}` | `run_inference_with_bump`, `numpyro_models.py:657` |
| R-hat (all sites) | < 1.05 | `output/bayesian/{model}_convergence_summary.csv` |
| ESS bulk (group-level μ_pr) | > 400 | `mcmc.summary_frame()` via ArviZ |
| ESS tail (group-level μ_pr) | > 400 | `mcmc.summary_frame()` via ArviZ |
| Tree-depth saturation | < 10% of iterations | `extra["tree_depth"]` |
| E-BFMI per chain | > 0.3 | ArviZ `az.bfmi()` |

#### 4.2 Posterior-vs-MLE spot check (one-off validation)

Run `python validation/compare_posterior_to_mle.py --model {model}` after
a production fit completes.  Inspect the printed summary:

| Parameter class | Expected pattern |
|---|---|
| Highly-identified (κ_total, κ_share on M6b; κ on M3) | Posterior mean within ~1 MLE SE; >80% of participants inside 2 MLE SE |
| Poorly-identified (α_pos, α_neg, φ, ρ, K) | Posterior means shrink toward group mean; deviations of 2+ MLE SE are common and expected (partial pooling) |
| ε | Usually shrinks strongly, prior is informative |

**Red flags**: a highly-identified parameter drifting far from MLE, or
posterior mean systematically biased in one direction across participants.

#### 4.3 L2 coefficient HDI check (primary science output)

For the kappa-family coefficients, extract 95% HDI from the posterior:

```python
import arviz as az
idata = az.from_netcdf("output/bayesian/wmrl_m6b_posterior.nc")
az.hdi(idata, var_names=["beta_lec_kappa_total", "beta_lec_kappa_share"],
       hdi_prob=0.95)
```

- **Strong signal**: 95% HDI excludes zero
- **Ambiguous**: HDI straddles zero but 90% excludes
- **Null**: HDI firmly straddles zero

Compare to the MLE-based quick-006 Task 4 pattern (uncorrected
`kappa_total × LEC-5` p=0.0028, FDR ns).  Hierarchical shrinkage
typically **widens** HDI vs the MLE-based p-value — a straddling HDI is
not a bug, it is the correct accounting for L1 uncertainty.

#### 4.4 Permutation null test

`cluster/13_bayesian_permutation.slurm` runs M3 with shuffled LEC labels
(100 shuffles).  Pass = fewer than 5% of shuffles produce HDI excluding
zero.  Reduced MCMC budget (`warmup=500, samples=1000`).  ~15 min per
shuffle.

#### 4.5 Model comparison

`python scripts/06_fit_analyses/compare_models.py --bayesian-comparison` runs
`az.compare(ic='loo', method='stacking')` across the 6 choice-only
posteriors.

- **MLE expectation**: M6b Akaike weight ≈ 1.0
- **Bayesian**: stacking weights may redistribute because LOO is more
  conservative than AIC.  M6b staying above 0.5 is the threshold for
  retaining it as the reported winning model

#### 4.6 Subscale signal discovery (M6b-subscale only)

32 beta sites → multiple-testing burden.  Check each:

- **Bonferroni-corrected HDI** at 95%/32 ≈ 99.85% HDI
- **FDR-adjusted ranks** of absolute posterior means

The current prior is `Normal(0, 1)`, weakly informative.  Plan L2-08 (horseshoe
prior) is deferred pending this production run — will be revisited if the
number of HDI-excluding-zero sites is unexpectedly high, which would
indicate the N(0,1) prior isn't providing enough shrinkage.

---

### 5. Known limitations (carry into results writeup)

1. **Capacity K identifiability is structural** — K posterior will be
   prior-dominated for most participants (quick-005).  Report as
   descriptive only, never as an individual-differences measure.
2. **M6b stick-breaking geometry**: `kappa_share` posterior can be
   bimodal when `kappa_total ≈ 0` (share is non-identifiable in that
   regime).  Auto-bump to `target_accept=0.99` resolves most cases; worst
   case needs `dense_mass=True`.
3. **IES-R hyperarousal unavailable** for L2 regression (exact linear
   dependence with intrusion and avoidance residuals).
4. **Sample: N=154 for hierarchical fit**, not N=160.  The 6 extra
   survey-only participants lack task data.
5. **No prior hierarchical Bayesian posterior exists** for direct
   comparison — the upcoming production refit is the first.  MLE fits in
   `output/mle/` are the baseline.

---

### 6. Production refit sequence

#### Spot check first (M6b, winning model)

```bash
git pull origin main
sbatch cluster/13_bayesian_choice_only.slurm  # pass --export=MODEL=wmrl_m6b
# Autopush returns output/bayesian/wmrl_m6b_* on completion
python validation/compare_posterior_to_mle.py --model wmrl_m6b
```

#### If M6b passes, submit the other 5 in parallel

```bash
for m in wmrl qlearning wmrl_m3 wmrl_m5 wmrl_m6a; do
    sbatch --export=ALL,MODEL=$m cluster/13_bayesian_choice_only.slurm
done
sbatch cluster/13_bayesian_choice_only.slurm  # M6b subscale variant if needed
```

#### Optional: multi-GPU pmap for wall-time reduction

`cluster/13_bayesian_multigpu.slurm` allocates `--gres=gpu:4` and lets
NumPyro's `chain_method="parallel"` pmap chains across devices.  Use when
the CPU production path is still too slow after Issue 1 fix — unlikely
at 1 s/iter per chain.

---

### 7. References

- **Senta, Bishop, Collins (2025)**: Primary reference for RLWM model
  structure, fixed β=50, and K ∈ [2, 6] bounds.
- **Ahn, Haines, Zhang (2017)**: hBayesDM non-centered parameterization
  convention and `Phi_approx` transform.
- **Weathers et al. (2013)**: LEC-5 trauma exposure checklist.
- **Weiss & Marmar (1997)** / **Creamer et al. (2003)**: IES-R symptom
  scoring and clinical cutoffs.
- **Collins & Frank (2012)**: RLWM framework origin (unbounded integer K,
  estimated β — both relaxed in our implementation per Senta 2025).

*Historical source: see [legacy/HIERARCHICAL_BAYESIAN.md](../legacy/HIERARCHICAL_BAYESIAN.md) for the original standalone version.*

---

## Scales Orthogonalization and Audit

*Merged from docs/SCALES_AND_FITTING_AUDIT.md on 2026-04-22 (Phase 29 Plan 02).*

Covers two closely-linked questions for v4.0 Phase 15/16:

1. **How should the LEC-5 and IES-R scales be used** given their sample
   distributions, correlation structure, and prior literature practice?
2. **Does our fitting procedure match the literature** where conventions
   exist?  Any non-standard choices flagged with rationale.

The empirical numbers in this doc are computed from
`output/summary_participant_metrics.csv` (N=166 for LEC-5, N=160 for
IES-R complete-data participants).

---

### 1. Scale distributions (this dataset)

#### LEC-5 (Life Events Checklist for DSM-5)

Scoring function: `scripts/utils/scoring_functions.py::score_less()`.
`less_total_events` sums per-item "any exposure" binary indicators across
15 LEC-5 event categories; `less_personal_events` sums "happened to me"
indicators only.

| Column | N | Mean | SD | Range | Median | Skew |
|---|---:|---:|---:|---:|---:|---:|
| `less_total_events` | 166 | 9.75 | 3.67 | [0, 15] | 10.0 | −0.50 |
| `less_personal_events` | 166 | 3.92 | 2.48 | [0, 11] | 3.0 | +0.49 |

Shapiro-Wilk W=0.957, p=1e-4 for `less_total_events` — non-normal, with
ceiling compression (mean 9.75/15 = 65%, left-skewed).

**Interpretation**: The `any_exposure` definition includes "witnessed" and
"learned about" — most respondents encounter at least 10 of the 15 event
categories by that broad criterion.  For narrower "directly experienced"
exposure, use `less_personal_events` instead (mean ≈ 4).

#### IES-R (Impact of Event Scale — Revised)

Scoring function: `scripts/utils/scoring_functions.py::score_ies_r()`.
Subscale item assignments follow Weiss & Marmar (1997).

| Column | N | Mean | SD | Range | Median | Skew |
|---|---:|---:|---:|---:|---:|---:|
| `ies_total` | 160 | 31.71 | 19.83 | [0, 76] | 31.5 | +0.21 |
| `ies_intrusion` | 160 | 10.65 | 6.89 | [0, 27] | 10.0 | +0.26 |
| `ies_avoidance` | 160 | 12.74 | 7.86 | [0, 29] | 12.0 | +0.13 |
| `ies_hyperarousal` | 160 | 8.32 | 6.77 | [0, 25] | 8.0 | +0.48 |

Shapiro-Wilk W=0.972, p=0.002 for `ies_total` — mild non-normality, much
closer to Gaussian than LEC-5.

**Clinical cutoffs** (Creamer, Bell, Failla 2003): 24+ mild concern, 33+
probable PTSD, 37+ severe.  In this sample, **36% score ≥ 33**,
indicating substantial symptom burden — consistent with a community sample
that self-selected into a trauma-exposure study.

#### Pairwise Pearson correlations (N=160)

```
                    LEC_tot  LEC_pers  IES_tot  IES_int  IES_avd  IES_hyp
LEC_total_events    1.000    0.409     0.132    0.154    0.077    0.140
LEC_personal_events 0.409    1.000     0.204    0.233    0.172    0.161
IES_total           0.132    0.204     1.000    0.926    0.914    0.924
IES_intrusion       0.154    0.233     0.926    1.000    0.753    0.821
IES_avoidance       0.077    0.172     0.914    0.753    1.000    0.750
IES_hyperarousal    0.140    0.161     0.924    0.821    0.750    1.000
```

**Key observations:**

- **LEC × IES = 0.13** (weak).  Exposure and current-symptom load measure
  distinct constructs — justifies including both in the L2 design.
- **IES subscales × IES total = 0.91–0.93** (strong).  Direct inclusion
  of both total and subscales causes severe multicollinearity; fixed by
  Gram-Schmidt residualization.
- **LEC subscales are independent enough** (r=0.41) to be distinguishable
  but overlap enough that we use only one (`less_total_events`) in L2.

---

### 2. Our scale-usage choices — audit

#### Standardization: z-score both covariates before MCMC

`scripts/04_model_fitting/b_bayesian/fit_bayesian.py::_load_lec_covariate` and
`scripts/fitting/level2_design.py::build_level2_design_matrix` both z-score
via `(x − mean) / std`.

- **Literature practice**: z-scoring is standard when predictors enter a
  hierarchical regression on a latent probit scale — ensures
  `beta_coef ~ Normal(0, 1)` priors are weakly informative.
- **Our implementation**: ✓ conforms.
- **Alternative considered**: log-transform for left-skewed LEC.  Not
  adopted because the mild skew (|S| < 0.5) is well within the range
  where z-score pre-processing is adequate, and log-transform would
  obscure the count interpretation of the scale.

#### IES-R subscale residualization (Gram–Schmidt)

`scripts/fitting/level2_design.py` centers each subscale then projects out
the total-score component:

```
residual_i = subscale_i_centered
           − (subscale_i_centered · total_centered)
             / (total_centered · total_centered)
             · total_centered
```

- **Literature practice**: two common approaches for handling IES-R
  multicollinearity:
  - **Use total only** — standard in clinical psychology papers (loses
    subscale differentiation)
  - **Residualize subscales on total** — Berkman & Falk (2013) approach,
    now standard in neuroimaging/cogneuro when the researcher wants to
    separate "overall distress" from "subscale-specific" variance
- **Our implementation**: ✓ residualize on total.  This is the more
  informative choice for trauma-parameter research where intrusion vs
  avoidance may have different neural substrates.

#### Hyperarousal exclusion

`ies_intr_resid + ies_avd_resid + ies_hyp_resid ≡ 0` because
`ies_total = ies_intrusion + ies_avoidance + ies_hyperarousal` exactly.
Only two of the three residuals carry independent information.

- **Literature practice**: when three subscales sum to the total by
  construction, all published practice drops one.  Intrusion and
  avoidance are retained because they map to clinically-distinct symptom
  clusters (re-experiencing vs effortful avoidance).  Hyperarousal's
  unique variance is recovered via `−(intr_resid + avd_resid)`.
- **Our implementation**: ✓ drop hyperarousal residual.

#### LEC-5 predictor choice: `less_total_events` (not `less_personal_events`)

- **Rationale**: broader exposure measure (includes witnessed and
  learned-about, per LEC-5 scoring conventions).  Prior literature on
  trauma-symptom regressions typically uses broader exposure scores
  because symptom load correlates with both direct and indirect
  exposure.
- **Alternative to consider in sensitivity analyses**: `less_personal_events`
  (direct exposure only, mean ≈ 4).  May reveal whether kappa × exposure
  effect is driven by direct vs vicarious exposure.  Not in current
  design but easy to swap — one line in `_load_lec_covariate`.

---

### 3. Fitting-procedure audit vs literature

#### Model structure — matches Senta et al. (2025)

| Choice | Senta et al. (2025) | Our implementation | Match |
|---|---|---|---|
| Inverse temperature β | Fixed at 50 during learning | Fixed at 50 (`jax_likelihoods.FIXED_BETA=50`) | ✓ |
| Capacity bounds | K ∈ [2, 6] | K ∈ [2, 6] | ✓ |
| WM update rule | Immediate overwrite WM(s,a) ← r | Immediate overwrite | ✓ |
| WM baseline | 1/nA = 0.333 | 1/nA | ✓ |
| Epsilon noise mixing | p_noisy = ε/nA + (1−ε)p | Same | ✓ |
| Asymmetric learning rates | α_pos, α_neg | Same | ✓ |
| Testing phase | None | None | ✓ |
| γ (discount) | 0 (no bootstrapping) | 0 | ✓ |

#### Hierarchical structure — matches hBayesDM (Ahn, Haines, Zhang 2017)

| Choice | hBayesDM convention | Our implementation | Match |
|---|---|---|---|
| Parameterization | Non-centered (mu_pr + sigma_pr × z) | Same | ✓ |
| Unbounded-to-bounded link | Phi_approx = standard normal CDF | Same (`jax.scipy.stats.norm.cdf`) | ✓ |
| Group mean prior | Normal(mu_prior_loc, 1.0) | Same | ✓ |
| Group sigma prior | HalfNormal(σ) | HalfNormal(0.2) | ✓ (slightly tighter than hBayesDM default 1.0) |
| Individual offset prior | z_i ~ Normal(0, 1) | Same | ✓ |

#### Level-2 regression — standard practice

| Choice | Standard practice | Our implementation | Match |
|---|---|---|---|
| Shift scale | Latent probit (unbounded) scale | `theta_unc_i += beta × x_i` | ✓ |
| Coefficient prior | Normal(0, 1) weakly informative | Same | ✓ |
| Covariate centering | Z-score before MCMC | Same | ✓ |
| Multi-covariate design | Condition number < 30 | 11.3 | ✓ |

#### Non-standard choices (flagged for discussion)

##### Informative-vs-principled priors on κ-family parameters

**History (pre-v4.0-refit, quick-006 era):**
`PARAM_PRIOR_DEFAULTS["kappa"]["mu_prior_loc"] = −2.0`, giving a prior
mean on the bounded scale of `Phi(−2.0) ≈ 0.023`.  Calibrated from
MLE point estimates in quick-006 showing most participants had κ in
[0.02, 0.20].

**Decision (2026-04-17, locked):** All `mu_prior_loc` values in
`PARAM_PRIOR_DEFAULTS` switched to `0.0`, corresponding to a prior
mean of `Phi(0) = 0.5` on the bounded scale with 95% prior interval
≈ [0.02, 0.98] given `sigma_pr ~ HalfNormal(0.2)`.  Rationale:

- MLE-calibrated priors are a mild form of empirical Bayes that
  creates a subtle circularity — the MLE fit already shrinks
  parameters toward their modal values for under-identified
  participants, and we were then telling the hierarchical model "these
  parameters are usually small" on the basis of those MLE results.
- The informative prior toward zero perseveration was **conservative
  for L2 null testing**: if trauma is associated with *increased*
  perseveration, the prior pulling κ down reduces the chance of the
  HDI excluding zero.  This was acceptable for exploratory quick-006
  work but not for the v4.0 primary analysis.
- Principled priors make L2 HDIs interpretable as "data-driven"
  rather than "prior-shifted" conclusions.

**Implementation:** Legacy MLE-calibrated values are retained in
`src/rlwm/fitting/numpyro_helpers.py::_PRIOR_LEGACY_MLE_CALIBRATED`
for sensitivity runs.  To reproduce a pre-refit fit, pass those
values explicitly to `sample_bounded_param`.

**Sensitivity analysis check** (run after primary v4.0 refit):
1. Refit M6b once with `mu_prior_loc=0.0` (primary) and once with
   `mu_prior_loc=-2.0` for κ-family (legacy).
2. Compare 95% HDI on `beta_lec_kappa_total` and
   `beta_lec_kappa_share` across the two runs.
3. If HDI boundaries shift by more than ~20% between the two,
   interpret the primary result cautiously — the prior is driving
   inference meaningfully.  If they agree, the principled prior is
   adequate.

##### Stick-breaking for M6b κ_total / κ_share

Decoded as `κ = κ_total × κ_share` and `κ_s = κ_total × (1 − κ_share)`.
Not in Senta et al.; **originated in this codebase**.

- **Rationale**: enforces `κ + κ_s ≤ 1` cleanly without inequality
  constraints; both components sampled independently on [0, 1] via
  probit transform.
- **Literature parallel**: standard stick-breaking in mixture models
  (Dirichlet process priors).  Novel application to perseveration
  weights.
- **Identifiability caveat**: when `κ_total ≈ 0`, `κ_share` is
  unidentifiable.  Addressed by auto-bump to `target_accept=0.99` and
  by the weakly-informative `κ_share_mu_pr_loc=0.0` prior (group mean
  near 0.5 a priori).  See quick-005 recovery: `κ_total` r=0.997 PASS,
  `κ_share` r=0.931 PASS under MLE.

##### M4 joint LBA (choice + RT)

Orthogonal to the v4.0 hierarchical choice-only rollout.  Uses a separate
pipeline (`13_bayesian_m4_gpu.slurm`, GPU float64) and is not directly
comparable to M1-M6b on AIC/LOO because the likelihood domain differs
(choice + RT vs choice-only).

#### Frequentist-vs-Bayesian β treatment

- Collins & Frank (2012), classic RLWM literature: β is estimated per
  participant.
- Senta et al. (2025): β fixed at 50 during learning (estimated only in
  testing phase, which this task doesn't have).
- Our implementation: β fixed at 50, consistent with Senta.  Trade-off:
  we lose the β-as-signal interpretation (higher β = more
  deterministic policy) but gain identifiability for the WM parameters.

---

### 4. Summary: what to tell the reviewer

1. **Joint hierarchical MCMC, not two-stage PEB** — all individual
   parameters, group priors, and trauma regression coefficients sampled
   simultaneously.  Standard hBayesDM non-centered convention (Ahn et
   al. 2017).
2. **Scale usage is principled**: LEC-5 and IES-R both z-scored; IES-R
   subscale multicollinearity handled via Gram–Schmidt residualization
   with one subscale dropped (mathematically forced).  Condition number
   11.3.
3. **Model structure matches Senta et al. (2025)** across all listed
   design choices (fixed β=50, K ∈ [2, 6], immediate WM overwrite,
   epsilon noise mixing, asymmetric α).
4. **Two non-standard choices**:
   - Informative prior on κ-family (`mu_prior_loc=−2.0`) — conservative,
     data-calibrated from MLE
   - Stick-breaking for M6b κ_total / κ_share — novel but principled
5. **Limitations preserved** from v4.0 Phase 15 planning: K
   identifiability structural, sample N=154 hierarchical / N=160 survey,
   hyperarousal residual excluded by math.

---

### 5. Open questions for v4.0 Phase 15 POC validation

1. Does the MLE-derived `kappa × LEC-5` uncorrected signal (p=0.0028)
   survive hierarchical shrinkage as a 95%-HDI-excludes-zero
   `beta_lec_kappa_total`?
2. Does the informative κ prior (`mu_prior_loc=−2.0`) shift HDI enough
   to warrant a sensitivity analysis with uninformative prior?
3. Do the 32 subscale beta sites in M6b-subscale need horseshoe shrinkage
   (L2-08 deferred), or is `Normal(0,1)` sufficient?

The first production refit will answer (1).  (2) and (3) can be
addressed with targeted re-runs after the primary fits complete.

*Historical source: see [legacy/SCALES_AND_FITTING_AUDIT.md](../legacy/SCALES_AND_FITTING_AUDIT.md) for the original standalone version.*
