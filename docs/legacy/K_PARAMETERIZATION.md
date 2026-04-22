# K Parameterization Reference

**Status:** v4.0 canonical reference (supersedes any K conventions used in v1.0–v3.0)
**Requirement:** K-01

---

## TL;DR

K is the working-memory capacity parameter, constrained to the continuous interval **[2, 6]**
via a non-centered normal CDF (Phi_approx) transform.  This matches the convention of the
project's reference paper: Senta, Bishop, Collins (2025) PLOS Comp Biol 21(9):e1012872, p. 20.
The canonical version string for all v4.0 fits using this convention is `"v4.0-K[2,6]-phiapprox"`.

---

## The WM Weight Formula

All Collins-lab RLWM papers use the same weight formula (Senta 2025 eq. 5; Collins 2014 eq. 1):

```
w(s, a) = rho * min(1, K / ns)
```

where `ns` is the number of stimulus-action pairs in the current block (the "set size"),
`rho` is the WM reliance parameter, and `K` is the capacity.

**Interpretation of K as a crossover point:**

- When `ns <= K`: `min(1, K/ns) = 1`; WM operates at full reliance (`rho`).
- When `ns > K`: `min(1, K/ns) = K/ns < 1`; WM contribution scales down proportionally.

K therefore marks the set-size threshold at which the WM system begins to
be capacity-limited.  A participant with K=3 shows full WM contribution in
2- and 3-item blocks but degraded WM in 5- and 6-item blocks.

This project's task uses set sizes `{2, 3, 5, 6}`.

---

## Non-Centered Hierarchical Transform

The non-centered parameterization follows the hBayesDM convention
(Ahn, Haines, Zhang 2017; Senta 2025 uses per-participant MLE, but the
transform structure below matches the hBayesDM template for NUTS compatibility):

```
# Group-level priors (unconstrained)
mu_K_pr    ~ Normal(0, 1)
sigma_K_pr ~ HalfNormal(0.2)

# Individual-level offsets
z_K_i      ~ Normal(0, 1)         # one per participant

# Individual capacity (constrained to [2, 6])
K_i = 2.0 + 4.0 * Phi_approx(mu_K_pr + sigma_K_pr * z_K_i)
```

`Phi_approx` is the standard normal CDF.  In NumPyro/JAX use:

```python
import jax.scipy.stats as jss
K_i = 2.0 + 4.0 * jss.norm.cdf(mu_K_pr + sigma_K_pr * z_K_i)
```

**Why Phi_approx (not sigmoid):**
The standard normal CDF gives a prior on the unconstrained scale that is Normal(0,1),
matching the group-mean prior exactly.  The sigmoid would give a logistic-shaped prior,
which is slightly heavier-tailed.  hBayesDM uses Phi_approx for this reason throughout
its Stan RLWM models, and the NumPyro port follows suit.

**Prior implied on K_i:**
`mu_K_pr = 0, sigma_K_pr = 0` gives `K_i = 2.0 + 4.0 * 0.5 = 4.0` (midpoint of [2, 6]).
The HalfNormal(0.2) prior on `sigma_K_pr` strongly regularizes individual variation,
consistent with a population where most participants are near K=3–5.

---

## `parameterization_version`

All Bayesian output CSVs written by v4.0 must include a `parameterization_version`
column with the value:

```
v4.0-K[2,6]-phiapprox
```

Downstream scripts (`15_analyze_mle_by_trauma.py`, `16_regress_parameters_on_scales.py`)
validate this column on load and raise a `ValueError` if the string does not match,
preventing accidental mixing of v3.0 MLE fits (which use K in [1, 7]) with v4.0 fits.

---

## Historical Collins-Lab K Conventions

| Paper | Year | K Type | Bounds | Fitting Method | Set Sizes | Notes |
|-------|------|--------|--------|----------------|-----------|-------|
| Collins & Frank (Eur J Neurosci 35:1024) | 2012 | Discrete integer | {0, 1, 2, 3, 4, 5, 6} | MLE fmincon, iterated over all K values | 2, 3, 4, 5, 6 | Founding RLWM paper; established `w = rho * min(1, K/ns)` |
| Collins, Brown, Gold, Waltz, Frank (J Neurosci 34:13747) | 2014 | Discrete integer | {0, 1, 2, 3, 4, 5, 6} | Iterated fmincon with 50 random starts per K | 2, 3, 4, 5, 6 | PMC4188972; patients median K=2, controls median K=3 |
| McDougle & Collins (Psychon Bull Rev 28:1205) | 2021 | Continuous | [2, 5] | MLE fmincon, 40 iterations | Instrumental (not RLWM) | PMC7854965; first Collins-lab continuous K; used symbol C |
| Senta, Bishop, Collins (PLOS Comp Biol 21:e1012872) | 2025 | Continuous | **[2, 6]** | MLE fmincon, 20 random starts | 2, 3, 4, 5, 6 | **Project reference paper**; K constrained to [2,6] per p. 20 |

---

## Why Lower Bound = 2

**1. Senta 2025 convention.**
The project's reference paper explicitly constrains K to [2, 6] (Senta 2025, p. 20).
Matching this bound aligns the project with the most recent Collins-lab canonical standard.

**2. Scientific interpretation of K < 2.**
The smallest set size in this task is `ns = 2`.  At K=2, WM weight in a 2-item block is:

```
rho * min(1, 2/2) = rho    # full WM reliance at ns=2
```

At K=1 (below the lower bound), WM weight in the same 2-item block is:

```
rho * min(1, 1/2) = 0.5 * rho    # half WM reliance at ns=2
```

This 0.5 scaling factor is fully confounded with `rho` itself — any reduction in K below 2
is geometrically absorbed by `rho`, making K < 2 non-identifiable.  The lower bound of 2
is therefore not merely a convention but a structural identifiability requirement given this
task's minimum set size.

**3. Breaking change acknowledgment.**
The v3.0 MLE pipeline used K in [1, 7].  This IS a breaking change.
Phase 14 (requirements K-02, K-03) refits all models with the new [2, 6] bounds so that
the v4.0 MLE and Bayesian pipelines share the same convention.  The
`parameterization_version` column enforces this at runtime.

---

## Why Upper Bound = 6

The task's maximum set size is `ns = 6`.  For any K > 6:

```
min(1, K/ns) = min(1, K/6) = 1    for all ns in {2, 3, 5, 6}
```

K above 6 is therefore structurally indistinguishable from K = 6 — it adds no degrees of
freedom to the likelihood.  Capping at 6 removes this non-identified region of parameter
space and keeps the support finite.  Senta 2025 uses the same upper bound for the same reason.

---

## BIC Rejection Rationale

Senta 2025 (p. 22, verbatim): "Previous research has shown that Bayesian model selection
criteria such as the Bayesian Information Criteria (BIC) tend to over-penalize models in the
RLWM class [Collins & Frank 2018].  To confirm this in the current data and support our use
of AIC as a measure of model fit, we performed a parallel model recovery analysis for the
selected RLWM models using BIC.  The confusion matrix for this analysis... confirms that
data generated from more complex underlying processes tends to be (incorrectly) best-fit by
simpler models when BIC is used."

**v4.0 policy:** BIC is retained in all output CSVs for v3.0 MLE back-compatibility.
It is NOT used as a model-selection criterion.  The primary comparison criterion in Phase 18
(requirement CMP-03) is WAIC/LOO from posterior predictive evaluation.

---

## References

Senta JD, Bishop SJ, Collins AGE (2025).
Dual process impairments in reinforcement learning and working memory systems underlie
learning deficits in physiological anxiety.
*PLoS Computational Biology* 21(9): e1012872.
DOI: 10.1371/journal.pcbi.1012872. Data: https://osf.io/w8ch2/

Collins AGE, Brown JK, Gold JM, Waltz JA, Frank MJ (2014).
Working memory contributions to reinforcement learning impairments in schizophrenia.
*Journal of Neuroscience* 34(41): 13747–56.
PMC4188972.

McDougle SD, Collins AGE (2021).
Modeling the influence of working memory, reinforcement learning, and action uncertainty
on reaction time and choice during instrumental learning.
*Psychonomic Bulletin & Review* 28(4): 1205–18.
PMC7854965.

Collins AGE, Frank MJ (2012).
How much of reinforcement learning is working memory, not reinforcement learning?
A behavioral, computational, and neuroimaging analysis.
*European Journal of Neuroscience* 35(7): 1024–35.
