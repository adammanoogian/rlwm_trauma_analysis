# Phase 33 — Context: Drop α₋ from RL slow module

**Date opened:** 2026-05-08
**Origin:** 2026-05-07/08 conversation — α₋ identifiability investigation triggered
by Phase 32-05 hierarchical Bayesian fit results.

---

## Empirical trigger

Phase 32-05 produced the first full-walltime convex M6b posterior
(`models/bayesian/32_full_convex/wmrl_m6b_posterior.nc`). Its hierarchical
identifiability report (ICC = `var_between / (var_within + var_between)`) showed:

```
alpha_pos    0.6743  identified
alpha_neg    0.0001  WARNING: poorly identified
phi          0.7680  identified
rho          0.6241  identified
capacity     0.5988  identified
kappa_total  0.8678  identified
kappa_share  0.6900  identified
epsilon      0.4533  identified
```

7/8 identified. α₋ is structurally unidentifiable in our hierarchical fit.

The 33-bench GPU rerun (`models/bayesian/33_bench_gpu_m6b/`) reproduced the
same numbers exactly (4 chains × 2000 draws on 4× L40S in 36.7 min wall, 0
divergences, max R̂ = 1.0000, min ESS_bulk = 2302, min BFMI = 0.717). This is
not an MCMC convergence problem — it's a structural unidentifiability.

The same pattern holds across the entire WM-RL family
(`models/bayesian/32_smoke_convex/`):

| Model | α_pos ICC | α_neg ICC |
|---|---|---|
| qlearning | 0.7292 | 0.7085 |
| wmrl_m3   | 0.6765 | 0.0006 |
| wmrl_m5   | 0.6000 | 0.0010 |
| wmrl_m6a  | 0.6589 | 0.0007 |
| wmrl_m6b  | 0.6487 | 0.0007 |

α₋ is well-identified in qlearning (no WM) and structurally collapsed in every
WM-RL variant. The signature is a WM-RL-family feature, not a model-specific bug.

---

## Diagnostic: why α₋ collapses (mechanism)

Three aligned signatures in the M6b convex posterior decompose the failure:

1. **Group mean pinned near zero**: `alpha_neg_mu_pr` posterior mean = -4.37 in
   logit space → ≈ 0.013 in (0,1) space. α_pos by contrast: -1.97 → ≈ 0.122.
   The data say negative-PE learning is ~10× weaker than positive-PE learning
   in the typical participant.
2. **Group SD pinned near zero**: `alpha_neg_sigma_pr` posterior mean = 0.124,
   2.5% quantile = 0.005. The hierarchical prior has learned that there is no
   meaningful between-participant variation in α₋.
3. **Per-participant means collapsed**: range across 158 participants is
   [0.0000, 0.0000] in (0,1) space. Even the latent z-scores
   (`alpha_neg_z`) have ICC = 0.0001, meaning standardized values aren't
   shaped by data either.

This is the textbook signature of an unidentified parameter — within-
participant variance (1.0 in z-space) is huge compared to between-participant
variance (7.7e-5). The likelihood is informative enough to constrain α₋ to
"doesn't matter" but not enough to constrain individual α₋ values.

---

## Literature alignment

Three independent lines of evidence say α₋-on-RL-module is the wrong design
for this task family:

### Senta, Rmus, Hartley & Collins (2025) — our reference paper

Senta 2025 (PLOS Comp Bio 21:e1012872) — the paper whose M2 we forked — does
**not** use α₋. Their canonical winning model uses single α plus a separate
"negative-feedback neglect" parameter η ∈ [0, 1] that multiplicatively
reduces *both* RL and WM updates following negative outcomes:

> "parameter η ∈ [0,1] is introduced which reduces the learning rate for
> both RL and WM modules following negative feedback"
> "RL learning rate α applied when reward = 1 (RL learning rate = 0 when
> reward = 0 in winning model variant)"

### Sugawara & Katahira (2021) — asymmetric α as perseveration artifact

Sugawara & Katahira (Sci Rep 11:3574) showed analytically and empirically that
observed α₊ > α₋ asymmetry in instrumental learning tasks **is largely a
statistical artifact of unmodeled choice perseverance**:

> "this asymmetry can be observed as a statistical bias if the fitted model
> ignores choice autocorrelation (perseverance), which is independent of
> the outcomes."

Our M3+ models *do* include perseveration (κ), so the residual asymmetry
signal that α₋ would otherwise capture is even smaller.

### Collins (2024) — "RL or not RL? Parsing the processes that support
human reward-based learning"

Collins (2024, preprint) reanalyzed 7 RLWM datasets (6 deterministic + 1
probabilistic, hundreds of participants total). Three findings directly bear:

1. **α₋ = 0 is empirically required**:
   > "even with an RL negative learning rate α₋ = 0, RLWM models could not
   > capture the pattern, because WM contributes to the choices even in
   > high set sizes"
2. **The slow module isn't RL**:
   > "best fitting model across 6 data sets... was a model with fixed
   > r₀ = 1, such that receiving incorrect feedback led to the same positive
   > prediction error as correct feedback would. Negative learning rates
   > still included a bias term shared across both modules."
3. **The new winning architecture is WMH (WM + H-agent)**: a Hebbian /
   habit-like slow process that updates association strengths irrespective
   of outcomes (subjective outcome SR(0) = r₀, fixed at 1). Exceedance
   probability > 0.93 in all 6 datasets.

### Why our M6b κ_total finding is consistent with Collins 2024

The H-agent in Collins (2024) is mathematically equivalent to a stimulus-
dependent choice perseveration kernel (Toyama, Katahira & Kunisato 2023).
Our M6b's κ_total parameter captures exactly this kind of stimulus-action
selection bias. The trauma effect we observe — `beta_lec_kappa_total` mean
+0.076, 95% CrI [+0.008, +0.144], excludes zero — is the *correct* place
for an individual-difference signal to land in this task class, given the
H-agent interpretation of the slow module.

Conversely, the per-participant α₋ ~ LEC-5 Spearman correlations reported in
the current manuscript (paper.tex:1389-1393, r_s = -0.18 to -0.30 for
M2, M4, M5) are best understood as Sugawara-style noise correlations on a
parameter that has no identifiable individual signal — and hierarchical
shrinkage correctly nulls them.

---

## Decision matrix considered

| Option | Architecture | Senta alignment | Collins 2024 alignment | Cost | Manuscript impact |
|---|---|---|---|---|---|
| **A: Drop α₋ → single α** | M2-M6b with single learning rate | Partial (Senta has η too) | Partial (Collins says α₋ = 0 specifically) | Low — refit 6 models, ~40 min on 4× L40S | Lose paper.tex:1391 α₋~LEC result; κ_total finding becomes headline; reframe slow module per Collins 2024 |
| **B: Drop α₋, add η** | Senta's exact parameterization | Exact | Partial | Medium — implement η across RL and WM updates, refit | Same headline, slightly different framing; matches Senta literally |
| **D: Replace RL with H-agent (WMH)** | Collins 2024 winning architecture | Divergent | Exact | High — new likelihood code, new recovery, new fit; 2-4 weeks | Major rewrite; would lose comparability with Senta unless we also add Senta's η model; separate paper |

**Phase 33 = Option A.** Phase 34 (provisional) = Option D as follow-up.

Option B was rejected for this manuscript because (a) Collins 2024 says even
α₋ = 0 isn't sufficient — the slow module structurally doesn't track outcomes;
(b) η + α duplicates Sugawara's perseveration-confound argument when we
already have κ; (c) net free-parameter count unchanged so identifiability
gain is minimal vs Option A. The right venue for η is Phase 34's broader
exploration of slow-module reformulations.

---

## Manuscript appendix scope (required deliverable)

A new appendix section in `manuscript/paper.qmd`, "Why α₋ is not free in
our models," documenting:

1. The hierarchical Bayesian identifiability gate result (ICC table from
   Phase 32-05 / 33-bench).
2. The mechanism diagnostic (group mean, group SD, per-pp range, z-score ICC).
3. The literature alignment (Senta 2025, Sugawara 2021, Collins 2024).
4. A pointer to Phase 34 as a deferred follow-up that tests the H-agent
   parameterization head-to-head against single-α RL.

This appendix is part of the Phase 33-01 deliverable, not a separate plan.

---

## Out of scope (for Phase 33)

- Implementing the H-agent (`H_{t+1} = H_t + α_H · (SR(r) - H_t)` with
  `SR(0) = r₀`). Belongs to Phase 34.
- Adding η (Senta's neg-feedback neglect). Considered and rejected;
  belongs to Phase 34 if revisited.
- Permutation null retest (`cluster/13_bayesian_permutation.slurm`) — needs
  rerun after refit but is a separate ~25h job, scheduled outside the
  Phase 33-01 plan.
- Changes to the WM module update rule. WM stays one-shot overwrite.
- Changes to κ, ε, φ, ρ, K parameterization. All preserved.

---

## References

- Senta, I., Rmus, M., Hartley, C. A., & Collins, A. G. E. (2025).
  Working memory and reinforcement learning interactions in human
  decision-making. *PLOS Computational Biology*, 21(9), e1012872.
- Sugawara, M., & Katahira, K. (2021). Dissociation between asymmetric
  value updating and perseverance in human reinforcement learning.
  *Scientific Reports*, 11, 3574.
- Collins, A. G. E. (2024). RL or not RL? Parsing the processes that
  support human reward-based learning. (Preprint;
  `Downloads/RLorNotRL_Collins2024.pdf` for project copy.)
- Toyama, A., Katahira, K., & Kunisato, Y. (2023). Examinations of
  biases by model misspecification and parameter reliability of
  reinforcement learning models. *Computational Brain & Behavior*, 6(4),
  651-670.
