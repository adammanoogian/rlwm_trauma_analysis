# Hierarchical Bayesian Pipeline (v4.0)

Architecture, Level-2 trauma regression, and validation checklist for the
joint hierarchical inference used by `scripts/fitting/fit_bayesian.py`.

This is the canonical reference.  See `.planning/STATE.md` for the current
execution status and `docs/03_methods_reference/MODEL_REFERENCE.md` for the
likelihood math.

---

## 1. Architecture: single-stage joint hierarchical, NOT two-stage PEB

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

### What a PEB-style two-stage would add

A second-stage regression against covariates NOT in the MCMC design matrix.
Already partially achievable by running
`scripts/post_mle/18_bayesian_level2_effects.py` on the posterior NetCDF (descriptive
output, not re-fit).  Not necessary for v4.0 validation because LEC is
already in the joint design.

---

## 2. Hierarchical structure (hBayesDM non-centered convention)

Following Ahn, Haines, Zhang (2017), each bounded parameter uses:

```
mu_pr       ~ Normal(mu_prior_loc, 1.0)          (unbounded group mean)
sigma_pr    ~ HalfNormal(0.2)                    (group variability)
z_i         ~ Normal(0, 1)                       (non-centered offset)
theta_unc_i = mu_pr + sigma_pr * z_i + beta_lec * lec_i  # L2 shift, optional
theta_i     = lower + (upper - lower) * Phi(theta_unc_i)
```

`Phi` is `jax.scipy.stats.norm.cdf` (the `phi_approx` name is conventional).

### Prior defaults

From `scripts/fitting/numpyro_helpers.py::PARAM_PRIOR_DEFAULTS`:

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
perseveration; see the fitting audit (`SCALES_AND_FITTING_AUDIT.md`) for
discussion of whether this is conservative for L2 null-testing.

---

## 3. Level-2 coverage table

| Model | L2 design | Beta sites | Hook |
|---|---|---|---|
| M1 `qlearning` | none | 0 | `covariate_lec=None` required |
| M2 `wmrl` | none | 0 | `covariate_lec=None` required |
| M3 `wmrl_m3` | LEC total → κ | `beta_lec_kappa` (1) | probit-scale shift on `kappa_unc` |
| M5 `wmrl_m5` | LEC total → κ | `beta_lec_kappa` (1) | probit-scale shift on `kappa_unc` |
| M6a `wmrl_m6a` | LEC total → κₛ | `beta_lec_kappa_s` (1) | probit-scale shift on `kappa_s_unc` |
| M6b `wmrl_m6b` | LEC total → κ_total, κ_share | `beta_lec_kappa_total`, `beta_lec_kappa_share` (2) | independent shifts on both stick-breaking parameters |
| **M6b-subscale** | 4 covariates × 8 params | **32** (`beta_{cov}_{param}`) | `wmrl_m6b_hierarchical_model_subscale` only |

### Subscale design (M6b-subscale)

4 covariates with condition number 11.3 (target < 30):
1. `lec_total` — LEC-5 total-events count, z-scored
2. `iesr_total` — IES-R total score, z-scored
3. `iesr_intr_resid` — IES-R intrusion, Gram–Schmidt residualized vs total
4. `iesr_avd_resid` — IES-R avoidance, Gram–Schmidt residualized vs total

IES-R hyperarousal residual is excluded because
`intr_resid + avd_resid + hyp_resid ≡ 0` (rank-deficient 3-residual
submatrix).  Canonical source: `scripts/fitting/level2_design.py`.

---

## 4. Validation checklist (run after each production fit)

### 4.1 Convergence (automatic gates)

| Check | Target | Where |
|---|---|---|
| Divergences | 0 at some `target_accept_prob ∈ {0.80, 0.95, 0.99}` | `run_inference_with_bump`, `numpyro_models.py:657` |
| R-hat (all sites) | < 1.05 | `output/bayesian/{model}_convergence_summary.csv` |
| ESS bulk (group-level μ_pr) | > 400 | `mcmc.summary_frame()` via ArviZ |
| ESS tail (group-level μ_pr) | > 400 | `mcmc.summary_frame()` via ArviZ |
| Tree-depth saturation | < 10% of iterations | `extra["tree_depth"]` |
| E-BFMI per chain | > 0.3 | ArviZ `az.bfmi()` |

### 4.2 Posterior-vs-MLE spot check (one-off validation)

Run `python validation/compare_posterior_to_mle.py --model {model}` after
a production fit completes.  Inspect the printed summary:

| Parameter class | Expected pattern |
|---|---|
| Highly-identified (κ_total, κ_share on M6b; κ on M3) | Posterior mean within ~1 MLE SE; >80% of participants inside 2 MLE SE |
| Poorly-identified (α_pos, α_neg, φ, ρ, K) | Posterior means shrink toward group mean; deviations of 2+ MLE SE are common and expected (partial pooling) |
| ε | Usually shrinks strongly, prior is informative |

**Red flags**: a highly-identified parameter drifting far from MLE, or
posterior mean systematically biased in one direction across participants.

### 4.3 L2 coefficient HDI check (primary science output)

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

### 4.4 Permutation null test

`cluster/13_bayesian_permutation.slurm` runs M3 with shuffled LEC labels
(100 shuffles).  Pass = fewer than 5% of shuffles produce HDI excluding
zero.  Reduced MCMC budget (`warmup=500, samples=1000`).  ~15 min per
shuffle.

### 4.5 Model comparison

`python scripts/14_compare_models.py --bayesian-comparison` runs
`az.compare(ic='loo', method='stacking')` across the 6 choice-only
posteriors.

- **MLE expectation**: M6b Akaike weight ≈ 1.0
- **Bayesian**: stacking weights may redistribute because LOO is more
  conservative than AIC.  M6b staying above 0.5 is the threshold for
  retaining it as the reported winning model

### 4.6 Subscale signal discovery (M6b-subscale only)

32 beta sites → multiple-testing burden.  Check each:

- **Bonferroni-corrected HDI** at 95%/32 ≈ 99.85% HDI
- **FDR-adjusted ranks** of absolute posterior means

The current prior is `Normal(0, 1)`, weakly informative.  Plan L2-08 (horseshoe
prior) is deferred pending this production run — will be revisited if the
number of HDI-excluding-zero sites is unexpectedly high, which would
indicate the N(0,1) prior isn't providing enough shrinkage.

---

## 5. Known limitations (carry into results writeup)

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

## 6. Production refit sequence

### Spot check first (M6b, winning model)

```bash
git pull origin main
sbatch cluster/13_bayesian_m6b.slurm
# Autopush returns output/bayesian/wmrl_m6b_* on completion
python validation/compare_posterior_to_mle.py --model wmrl_m6b
```

### If M6b passes, submit the other 5 in parallel

```bash
for m in m1 m2 m3 m5 m6a; do
    sbatch cluster/13_bayesian_${m}.slurm
done
sbatch cluster/13_bayesian_m6b_subscale.slurm  # 32-beta subscale design
```

### Optional: multi-GPU pmap for wall-time reduction

`cluster/13_bayesian_multigpu.slurm` allocates `--gres=gpu:4` and lets
NumPyro's `chain_method="parallel"` pmap chains across devices.  Use when
the CPU production path is still too slow after Issue 1 fix — unlikely
at 1 s/iter per chain.

---

## 7. References

- **Senta, Bishop, Collins (2025)**: Primary reference for RLWM model
  structure, fixed β=50, and K ∈ [2, 6] bounds.
- **Ahn, Haines, Zhang (2017)**: hBayesDM non-centered parameterization
  convention and `Phi_approx` transform.
- **Weathers et al. (2013)**: LEC-5 trauma exposure checklist.
- **Weiss & Marmar (1997)** / **Creamer et al. (2003)**: IES-R symptom
  scoring and clinical cutoffs.
- **Collins & Frank (2012)**: RLWM framework origin (unbounded integer K,
  estimated β — both relaxed in our implementation per Senta 2025).
