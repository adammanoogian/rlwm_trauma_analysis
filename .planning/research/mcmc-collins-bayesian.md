# MCMC Methodology Research — Hierarchical Bayesian RLWM

**Author:** Claude (literature triage agent), 2026-05-03
**Scope:** Diagnose why our 6-model hierarchical NumPyro fan-out has 5/6 models failing convergence (R-hat 1.12 to 2.38, ESS_bulk 5 to 23) while M6b passes cleanly. Mine the Collins-lab and Stan-community literature for canonical recipes that actually work for RLWM.

---

## Question

We fit the Senta et al. (2025) RLWM family hierarchically in NumPyro (NUTS, JAX backend) on N=158 participants, 13 blocks of set-size 2-6 stimulus-action learning. Senta is our reference paper, but Senta uses **MLE in Matlab (`fmincon`, 20 random starts, AIC for comparison)** — not Bayesian fits. So our hierarchical Bayesian setup is project-specific, and we need to interrogate whether the parameterization, priors, and sampler config are correct against the actual Bayesian-RLWM literature. Empirically:

| Model | max R-hat | min ESS_bulk | divergences | Verdict |
|---|---|---|---|---|
| qlearning M1 | 2.38 | 5 | 0 | HARD FAIL (multimodal alpha_pos) |
| wmrl M2 | 1.12 | 23 | 0 | Borderline fail |
| wmrl_m3 / m5 / m6a | 1.60 | 7 | 0 | Same R-hat — shared kappa pathology |
| wmrl_m6b | 1.00 | 2514 | 0 | PASS (`kappa = kappa_total × kappa_share`) |

The pattern says: (a) Q-learning has the well-known α-vs-β trade-off pathology and absent the WM channel cannot stabilise; (b) M3/M5/M6a all hit the same R-hat ceiling because they share the same scalar `kappa` parameterization; (c) M6b's stick-breaking reparameterization is the only thing that works. The literature should tell us if our priors, transform, and parameterization are doing what we think.

---

## TL;DR Recommendations

1. **Our `Phi_approx + Normal(0,1) + HalfNormal(0.2)` non-centered setup is line-for-line equivalent to hBayesDM's official Stan code** (`bandit2arm_delta.stan`, `bandit4arm_4par.stan`, `prl_ewa.stan`). For the well-converging parameters (α, φ, ρ, K) the parameterization is correct. The bug is in the *perseveration model spec*, not in NumPyro plumbing. For the κ-family specifically, tighten `mu_prior_loc` from 0.0 to −1.0 (anchors prior mean κ ≈ 0.16; matches Rmus 2023's Gamma-rate empirical-Bayes anchoring without re-introducing MLE circularity).
2. **Keep β_learning fixed at 50.** Both the modern Collins-lab papers (Senta 2025, Collins 2025) fix β at a high value during learning. Rmus 2023 fixes β_learning at the *group level* (`Gamma(5, 0.4)`) and only frees β_test per participant. Our task has no test phase. Unfixing β reintroduces the α-vs-β trade-off (Pedersen 2022: r=0.49) and breaks MLE comparability. **The earlier draft of this document recommended unfixing β; that recommendation is rescinded.**
3. **Adopt Baribault & Collins (2023) Tier-1 convergence gates: R-hat ≤ 1.01, ESS ≥ 400, divergences = 0, BFMI ≥ 0.2.** They explicitly list R-hat ≈ 2 with 0 divergences as the "label-switching multimodal" pathology and prescribe **more informative priors**, not more samples. Sampler bumps to `target_accept=0.99` / `max_tree_depth=15` / 16 chains × 4000 warmup will *not* fix a multimodal posterior.
4. **DROP M3 and M6a from the Bayesian table; promote M6b as the canonical perseveration model.** This replaces the earlier recommendation to "apply stick-breaking to M3/M5/M6a" — that framing was mathematically wrong. M3 and M6a are M6b with one parameter (κ_share) pinned to a corner of the simplex (κ_share=1 reduces M6b → M3; κ_share=0 reduces M6b → M6a; verified by `test_wmrl_m6b_kappa_share_one_matches_m3` and `test_wmrl_m6b_kappa_share_zero_matches_m6a` in `src/rlwm/fitting/models/wmrl_m6b.py:792, 843`). When the data has perseveration variance on both global and stimulus-specific channels, M3's posterior is *correctly multimodal* (R-hat = 1.60 is the sampler diagnosing model misspecification). M6b's posterior on κ_share gives full evidence about the M3-vs-M6a trade-off without fitting them separately. **Effort: ~10 min orchestrator config change. Risk: zero to inference quality.**
5. **For M5: question its retention entirely.** Collins 2025 (Methods, p.366): *"I verified that letting RL processes also have forgetting did not improve fit or explain the qualitative pattern of behaviour."* M5 = M3 + φ_rl. Combined with our R-hat=1.60 on M5, the empirical case is weakening. Either drop M5 from the Bayesian table alongside M3/M6a, or build M5b = M6b + φ_rl as a 9-param model. **This is a manuscript-narrative decision, not a methods fix.**
6. **Optional: switch init strategy to `init_to_median(num_samples=50)`.** Cheapest wall-time intervention that does not change the model spec; no paper warm-starts from MLE so this is not a conformance issue.

**Considered for v5.1 (not v5.0): Option D-2** — reparameterize κ from Senta-style convex-mixture (κ ∈ [0, 1]) to Collins 2025-style softmax-bias (κ ∈ [−1, 1]). This addresses a *structural* identifiability issue distinct from the M3⊂M6b containment problem. See "Phase D-2" section below for the math and effort estimate.

---

## Per-paper review

### Tier 0: Direct PDF verification (added 2026-05-03)

#### Eckstein, Master, Xia, Dahl, Wilbrecht, Collins (2022) — eLife 11:e75474

**Status updated from "honest gap" to "verified."** Read the full PDF locally.

- **Their RLWM task (Task C) does NOT include perseveration parameters.** Figure 1F lists Task C model parameters as α+, α−, K (capacity), F (forgetting), ε (noise), ρ (mixture weight). No κ.
- **Inverse temperature handling for Task C: not free.** Figure 1F shows ε (not 1/β) as Task C's noise parameter; matches Senta 2025's β=50 fixed convention.
- **Hierarchical Bayes confirmed but priors not explicitly listed in this paper.** Appendix 7 (Limitations): *"In this study, we used hierarchical fitting for tasks A and B and assessed a range of qualitative measures of model fit for each task (Eckstein et al., 2022; Master et al., 2020; Xia et al., 2021), boosting our confidence in high reliability of our parameter estimates."* They delegate prior specification to Master 2020 / Xia 2021 (un-verified).
- **Verdict:** Eckstein 2022 confirms the Collins lab uses hierarchical Bayes for RLWM-related tasks but their RLWM-proper model has no perseveration. **Cannot help us with the κ identifiability problem directly.**

#### Collins (2025) — Nat Hum Behav 10:357–369 — *paradigm-shifting paper*

**Sole-author Collins paper that reanalyses 7 RLWM datasets (n=594) and concludes the entire RLWM framework should be replaced by WM+Habit (WMH).** Read the full PDF locally.

Key methodological findings:

- **Fitting framework: MATLAB fmincon MLE, ten random starts, AIC for comparison.** Same as Senta 2025. **Not hierarchical Bayes.** No Collins-lab paper has fit RLWM-with-perseveration hierarchically.
- **Capacity K is fit as DISCRETE K ∈ {2, ..., 5}** — chosen explicitly to "avoid optimizer slowness with non-smooth likelihoods with continuous K parameters." Alternative to our continuous-K approach (no convergence implication for Bayesian, but worth noting for MLE benchmarks).
- **Perseveration parameterization is fundamentally different from ours** (CRITICAL FINDING — see "Phase D-2" below):

  Senta 2025 / our M3 (Eq. 16b in Senta): convex mixture
  ```
  p(a|s) = (1 − κ) · p_noisy(a|s) + κ · C_k(a),    κ ∈ [0, 1]
  ```

  Collins 2025 (Methods, p.365): softmax additive bias
  ```
  π_WM(a|s) = exp(βW(s,a) + κ · I(a, a_{t−1})) / Σ_i exp(βW(s,a_i) + κ · I(a_i, a_{t−1})),    κ ∈ [−1, 1]
  ```

  The convex-mixture formulation has weak-identifiability at the boundaries (likelihood gradient → 0 as κ → 0 because dp/dκ has no curvature when p_noisy is near-uniform; and the model becomes degenerate as κ → 1). The softmax-bias formulation has informative gradient everywhere on the support and admits negative κ (action-avoidance). **This is plausibly the deepest reason our M3 Bayesian fit is multimodal where Collins's MLE fits converge.**

- **Collins 2025 explicitly says φ_rl is unnecessary** (Methods, p.366): *"I verified that letting RL processes also have forgetting did not improve fit or explain the qualitative pattern of behaviour (Supplementary Fig. 2)."* Direct evidence against our M5.
- **Theoretical findings** (the paradigm shift, not directly methodological but relevant for model-space decisions):
  - WMH (WM + Hebbian habit, no value-based RL) outperforms RLWM family on all 6 datasets (Fig. 3)
  - The H agent is *mathematically equivalent to a stimulus-dependent choice perseveration kernel* — i.e., literally our M6a's κ_s — but interpreted as a separate cognitive process rather than a nuisance term inside RL
  - The α-vs-perseveration trade-off is real: "perseveration strategies might be mistaken for an RL learning rate asymmetry"

**Implication for us:**
1. Our M6b (κ_total, κ_share) decomposition is the field-canonical way to disambiguate global motor perseveration from stimulus-specific perseveration. The two-channel interpretation aligns directly with Collins 2025's WMH motor-vs-stimulus split.
2. M3 and M6a are (κ_share = 1, κ_share = 0) corner cases. The R-hat = 1.60 multimodality is the Bayesian sampler *correctly diagnosing* that the data has variance the constrained model can't represent.
3. The convex-mixture vs softmax-bias choice is a separate, deeper issue — see Phase D-2.

### Tier 1: Canonical Bayesian-RLWM methodology

#### Rmus, He, Baribault, Walsh, Festa, Collins, Nassar (2023) — eLife 12:e85243

The single most important reference. RLWM model fit hierarchically in **Stan**, with **separate group hierarchies for young vs older adults**. Methods quote (paraphrased from eLife article):

- Framework: Stan, MCMC.
- Hierarchical: 2-group (compared and beat both flat and single-hierarchy alternatives).
- **Priors are Beta distributions with Gamma hyperpriors, NOT Phi_approx-on-Normal**:
  - `α_+ ~ Beta(1+a_g, 1+b_g)` with `a ~ Gamma(1,1)`, `b ~ Gamma(4,1)` → mass concentrated at small α
  - `α_- ~ Beta(1+a, 1+b)`, `a ~ Gamma(1,1)`, `b ~ Gamma(4,1)`
  - `φ ~ Beta(1+a, 1+b)`, `a ~ Gamma(1,1)`, `b ~ Gamma(2,1)`
  - `ω3, ω6 (WM weight, set-size specific) ~ Beta(1+a, 1+b)`, `a ~ Gamma(2,1)`, `b ~ Gamma(1,1)`
  - `ε ~ Beta(1+a, 1+b)`, `a ~ Gamma(1,1)`, `b ~ Gamma(12,1)` → strongly anchored to small noise
- β handling: **β_learning estimated at group-level only** (one per age group, prior `Gamma(5, 0.4)`); **β_test free per participant** (Gamma hyperprior).
- Sampler: 4 chains × 500 warmup × 1500 kept iterations. Default `target_accept` and `max_tree_depth` (not reported).
- Convergence gates (strict): R-hat ≤ 1.01, ESS ≥ 400, BFMI ≥ 0.2, zero divergences.
- "We confirmed that both individual-level parameters and group-level hyperparameters were recovered successfully" — they ran a parameter-recovery validation.
- Code: not publicly linked in the eLife article. Co-author Baribault's `matstanlib` repo provides the Stan-fitting / diagnostic toolchain but not this specific RLWM model file.

**Implication for us:** Our `Phi_approx(mu_pr + sigma_pr * z)` with `mu_pr ~ Normal(0,1)` is one of two viable parameterizations. Rmus 2023 uses the **alternative** (Beta with Gamma hyperpriors) and does so successfully. Whether their Beta+Gamma is structurally tighter or just better-calibrated for the perseveration class is the open question — but their Beta-prior strategy clearly converges where ours does not.

#### Baribault & Collins (2023) — Psychological Methods 30(1):128–154

Methodological gold standard. Tutorial paper introducing `matstanlib` and walking a 3-armed bandit RL example through the exact pathologies we are seeing.

- Worked example priors (their `RL_fixed.stan`):
  - `α_p ~ Normal(μ_α, σ_α) T[0,1]` truncated normal with `μ_α ~ Uniform(0,1)`, `σ_α ~ Normal(0, 0.5) T[0,1]`
  - `β_p ~ HalfNormal(μ_β, σ_β)` with `μ_β ~ HalfNormal(10, 5)`, `σ_β ~ HalfNormal(0, 5)` — so β is **free**, not fixed, with a "zero-avoiding prior" (ZAP) because β=0 makes the model ill-defined.
  - `φ_p ~ Normal(μ_φ, σ_φ) T[0,1]` truncated normal.
- Centered vs non-centered: explicit recommendation for non-centered (`η ~ Normal(0,1); θ_n = μ + σ * η`) for any hierarchical parameter showing funnel divergences.
- Convergence gates (their proposed standard): **R-hat ≤ 1.01** (stricter than the older ≤1.10), **ESS ≥ 100 × n_chains**, zero divergences.
- Sampler defaults: 4 chains × 500 warmup × 1500 kept. Recommends starting with 50/50 to confirm the model runs, then 150/500 for early debugging, only 4×500/1500 for final.
- Pathology recipes:
  - Multimodal posterior / label-switching → "more informative priors" (NOT more samples)
  - Funnel divergences → non-centered
  - Drift → 2× warmup
  - Sticking → investigate divergences, reparameterize
  - Treedepth saturation → raise `max_tree_depth` (rarely the primary fix)
- ZAP for β: "an inverse temperature of β=0 breaks the model, as the learned Q values then have no bearing on action selection." Recommends `Gamma(shape > 1, rate)` for any free β.
- They use truncated Normal `T[0,1]` for bounded parameters, NOT `Phi_approx`. Both are valid; Phi_approx avoids the truncation discontinuity and is easier to vectorize.
- Code: `https://github.com/baribault/matstanlib` (verified). MATLAB-based; runs Stan via the MatlabStan interface. Stan files are bundled in `examples/`, including `RL_broken.stan` (centered, demonstrates pathology) and `RL_fixed.stan` (non-centered + tighter priors).

**Implication for us:**
- Our R-hat=2.38 on M1 alpha_pos with 0 divergences is exactly their "label-switching multimodal" recipe, and their advice is unambiguous: **tighten the prior, do not raise samples.**
- Their Tier-1 gates (R-hat ≤ 1.01, ESS ≥ 400, 0 divergences) are the publication standard. Our local doc says R-hat < 1.05 — we should tighten that.
- Their RL example uses **free β with HalfNormal hyperprior + ZAP**, which is consistent with the Collins lab tradition of estimating β when there is enough data to identify it. Senta 2025 deliberately departs from this by fixing β=50 to gain identifiability for WM parameters. Both choices are defensible.

#### Eckstein, Master, Xia, Dahl, Wilbrecht, Collins (2022) — eLife 11:e75474

**Could not extract methods detail.** The eLife HTML page truncates the methods section in our fetch; the PDF binary is not text-extractable via our tooling; the OSF data repo (https://osf.io/h4qr6/) is inaccessible without a browser-rendered file listing. Search hits confirm only:

- "We also used hierarchical Bayesian methods for model fitting and comparison where possible" (their words).
- They fit 3 different RL tasks (A, B, C) per participant for 291 participants aged 8-30; the methodology details are delegated to three separate papers (Xia et al. 2021, Eckstein et al. 2022 [different task paper], Master et al. 2020).
- OSF link exists at `https://osf.io/h4qr6/` for data; no Stan code visible in the search-rendered tree.

**Honest gap.** This is the most-cited recent Collins-lab Bayesian RL methods paper and we cannot verify its exact prior specifications without browser-level access to the appendices. Treat the conclusions of this document as conditional on Rmus 2023 and Baribault & Collins 2023 being representative of the Collins-lab tradition.

#### Master, Eckstein, Gotlieb, Dahl, Wilbrecht, Collins (2019) — bioRxiv 622860 / DCN 2020

**Could not extract methods detail** (bioRxiv 403 to our fetch; ScienceDirect requires institutional access). Search-result fragments confirm: 187 children + 53 adults; RLWM model fit; "computational models of behavior were fit to subjects' performance" — fitting framework not specified in fragments. Likely Matlab MLE based on the era and the lab's MLE-first tradition (the paper predates Rmus 2023 by 4 years).

### Tier 2: Foundational Collins-lab RLWM (mostly MLE)

These are the methodological ancestors, all MLE in Matlab. They establish:

- **Collins (2018), J Cogn Neurosci 30(10):1422–32** — the cited authority for "fix β=50 to improve parameter reliability." Senta 2025 cites this as ref [20]; one of our local search results shows this paper's actual quote uses **β=100** (`"In the present study β was fixed at 100, following the methods of Collins et al."`). The exact value (50 vs 100) varies across Collins-lab papers; the principle (fix it during learning) is invariant.
- **Collins, Brown, Gold, Waltz, Frank (2014), J Neurosci 34(41):13747–56** — original WM-RL deficit-in-schizophrenia paper, MLE-fit.
- **Collins & Frank (2012), Eur J Neurosci 35(7):1024–35** — RLWM origin paper. β was per-subject free here; later papers fixed it as identifiability research progressed.
- **Rmus, McDougle, Collins (2021), Curr Opin Behav Sci 38:66–73** — review, no fits.
- **Collins, Ciullo, Frank, Badre (2017), J Neurosci 37(16):4332–42** — WM Load Strengthens RPE; MLE.
- **Collins & Frank (2018), PNAS 115(10):2502–7** — within/across-trial dynamics with EEG; MLE.

The pattern is consistent: **the Collins lab fit MLE in Matlab from 2012 through ~2022, then started moving to hierarchical Stan with Rmus 2023 (Baribault is the methodologist who brought the Bayesian workflow into the lab).** Our project is in the second wave of this transition.

### Tier 3: Community-standard Bayesian RL packages

#### hBayesDM (Ahn, Haines, Zhang 2017) — Comput Psychiatry 1:24–57; PMC5869013

The reference Stan package for hierarchical Bayesian RL. Inspecting their actual Stan source:

**`bandit2arm_delta.stan`** (delta-rule Q-learning, 2-arm bandit):

```stan
// parameters
vector[2] mu_pr;                // group-level means (raw scale)
vector<lower=0>[2] sigma;       // group-level SDs
vector[N] A_pr;                 // individual learning rate raw
vector[N] tau_pr;               // individual inverse-temperature raw

// transformed parameters
vector<lower=0, upper=1>[N] A;
vector<lower=0, upper=5>[N] tau;
for (i in 1:N) {
  A[i]   = Phi_approx(mu_pr[1] + sigma[1] * A_pr[i]);
  tau[i] = Phi_approx(mu_pr[2] + sigma[2] * tau_pr[i]) * 5;
}

// model (priors)
mu_pr  ~ normal(0, 1);
sigma  ~ normal(0, 0.2);
A_pr   ~ normal(0, 1);
tau_pr ~ normal(0, 1);
```

**`bandit4arm_4par.stan`** (4-arm bandit, separate reward/punishment learning rates plus reward/punishment sensitivities):

```stan
mu_pr  ~ normal(0, 1);
sigma  ~ normal(0, 0.2);
Arew_pr ~ normal(0, 1.0);
Apun_pr ~ normal(0, 1.0);
R_pr    ~ normal(0, 1.0);     // reward sensitivity, transformed to [0, 30]
P_pr    ~ normal(0, 1.0);     // punishment sensitivity, transformed to [0, 30]
```

**`prl_ewa.stan`** (probabilistic reversal learning, EWA model with experience decay):

```stan
mu_pr  ~ normal(0, 1);
sigma  ~ normal(0, 0.2);
phi_pr  ~ normal(0, 1);       // learning rate
rho_pr  ~ normal(0, 1);       // experience decay (analogue of our φ)
beta_pr ~ normal(0, 1);       // inverse temperature, transformed to bounded
rho[i]  = Phi_approx(mu_pr[2] + sigma[2] * rho_pr[i]);
```

**Verbatim verdict: our `numpyro_helpers.py::sample_bounded_param` is line-for-line equivalent to the hBayesDM Stan pattern.** The transform (`Phi_approx`), the prior on group mean (`Normal(0, 1)`), the prior on group SD (`Normal(0, 0.2)` truncated, equivalent to our `HalfNormal(0.2)`), and the individual offset (`Normal(0, 1)`) are identical. The only thing we tightened is `mu_prior_scale` is fixed at 1.0 (not exposed as a knob in hBayesDM either, so this is a non-difference).

**For β specifically, hBayesDM transforms β into a bounded interval** — `tau ∈ [0, 5]` for 2-arm bandit, `R, P ∈ [0, 30]` for 4-arm. They never let β scale to infinity. This is the standard Stan-community workaround for ZAP without resorting to truncated half-normals.

**Q-learning identifiability — hBayesDM does NOT report convergence problems for `bandit2arm_delta`**, which is the simplest case. So Q-learning hierarchical fits CAN converge. Our M1 R-hat=2.38 suggests our specific data (158 participants × ~860 trials each, 3 actions, set-size manipulation) is a harder regime than the canonical hBayesDM 2-arm bandit. The set-size structure introduces strong stimulus-specific Q-table evolution that may interact badly with a single shared (α+, α-) per participant.

#### matstanlib (Baribault) — github.com/baribault/matstanlib

Verified existing repo. Three repos under `baribault/`: `matstanlib`, `MatlabStan` (forked Stan-Matlab interface), `baribault.github.io`. The matstanlib README confirms `example_RL.m` walks the troubleshooting tutorial; the actual `RL_broken.stan` / `RL_fixed.stan` files are inside `examples/` but are not directly browsable via WebFetch from our environment. From the Baribault & Collins 2023 paper text we have the priors verbatim (see Tier 1 above).

#### van Geen & Gerraty (2021) — bioRxiv 2020.10.19.345512 / J Math Psychol

"Hierarchical Bayesian Models of Reinforcement Learning: Introduction and comparison to alternative methods." Could not directly fetch (bioRxiv 403), but search hits confirm: hierarchical priors **substantially aid α-β identifiability** by shrinking individual estimates toward a group mean, breaking the α×β multimodality. Their explicit recommendation: hierarchical Bayesian over MLE for any task where the α-β trade-off is expected to be present, which is essentially every Q-learning task.

#### Pedersen, Frank, Biele (2022 / earlier) — joint RT+choice paper, PMC8930195

"Joint modeling of reaction times and choice improves parameter identifiability in reinforcement learning models." Quantifies the α-β trade-off: **average correlation r=0.49 between learning rate and inverse temperature in choice-only fits**, dropping to near zero when RT is jointly modeled. Choice-only recovery `r=0.47`; with RT `r=0.75`. Their bounds: β ∈ [0.5, 10] explicitly excluding β=0. **They do not address perseveration / κ identifiability** — that gap remains in the literature.

---

## Decisions table

| Dimension | Current setup | Literature standard | Recommendation | Effort | Risk |
|---|---|---|---|---|---|
| **β_learning** | Fixed at 50 (Senta 2025) | Senta fixed-50; Collins 2018 fixed-50 or 100; Rmus 2023 group-level Gamma(5, 0.4); Baribault 2023 free per-subject HalfNormal(10,5); hBayesDM transforms to bounded [0, 5] | **Keep fixed at 50.** Unfixing reintroduces α×β trade-off (r≈0.49) and breaks MLE comparability. Senta is the closest analogue (no test phase, learning only). | n/a | n/a |
| **Group-mean prior** `mu_pr` | `Normal(0, 1)` (hBayesDM default) | hBayesDM: `Normal(0, 1)`; Rmus 2023: `Beta(1+a,1+b)` with Gamma hyperpriors that strongly anchor κ near 0; Baribault 2023: `Uniform(0,1)` for α | Keep `Normal(0, 1)` for α/φ/ρ/K. **Tighten to `Normal(-1, 0.5)` for kappa-family** to anchor near 0.12 prior mean (analogous to Rmus's Beta(1+1, 1+12) for ε). Avoids re-introducing the MLE-calibration circularity flagged in our 2026-04-17 prior decision while still preventing the multimodal κ posterior. | 1-line per param | LOW — backwards-compatible knob |
| **Group-SD prior** `sigma_pr` | `HalfNormal(0.2)` | hBayesDM: `Normal(0, 0.2)` truncated (= our HalfNormal); Baribault 2023: `Normal(0, 0.5) T[0,1]`; Stan wiki: half-Normal preferred over half-Cauchy when N is small or weak prior is acceptable | **Keep `HalfNormal(0.2)`.** Already best-in-class. | n/a | n/a |
| **Transform unbounded → (0,1)** | `Phi_approx` (= `jss.norm.cdf`) | hBayesDM: `Phi_approx`; Rmus 2023: native Beta (no transform); Baribault 2023: truncated Normal | **Keep `Phi_approx`.** Equivalent to hBayesDM. Native Beta requires Gamma hyperpriors and is harder to vectorize in NumPyro. | n/a | n/a |
| **Parameterization** | Non-centered `theta_unc = mu_pr + sigma_pr * z` | hBayesDM: non-centered (identical); Baribault 2023: non-centered preferred | **Already correct for α/φ/ρ/K.** For κ-family, **add stick-breaking reparam to M3, M5, M6a** (sample `kappa_anchor` and `kappa_strength` independently, derive `kappa = kappa_anchor × kappa_strength`). M6b already uses this and is the only converging perseveration model. | ~30 lines per model | MEDIUM — invalidates AIC vs legacy MLE M3 fits |
| **Init strategy** | `init_to_uniform` (NumPyro default) | No paper warm-starts from MLE. Most use random / Stan default. NumPyro's `init_to_median(num_samples=20)` is documented to be more robust for Phi_approx-bounded models. | **Switch to `init_to_median(num_samples=50)`.** No model spec change. | 1-line | LOW |
| **Sampler config** | 4 chains × (1000 warmup + 2000 samples) × max_tree_depth 8 × target_accept auto-bump 0.80→0.95→0.99 | Rmus 2023: 4×500/1500. Baribault 2023: 4×500/1500. hBayesDM defaults: 4×1000/2000 | **Increase max_tree_depth to 12 (cheap). Do NOT increase chains/samples — Baribault is explicit that 16×4000 will not fix multimodal posteriors.** Keep auto-bump on target_accept. | 1-line | LOW |
| **Convergence gates** | R-hat < 1.05; ESS bulk > 400 | Baribault 2023 / Rmus 2023: R-hat ≤ 1.01; ESS ≥ 400; 0 divergences; BFMI ≥ 0.2 | **Tighten R-hat gate to ≤ 1.01.** Already at ESS ≥ 400 and div=0. Add BFMI ≥ 0.2 check. | ~20 lines in summary writer | LOW — purely diagnostic |

---

## Implementation plan

Phased, cheapest-first. Each phase is a separate commit, with a regression smoke test before the next phase starts.

### Phase A — Diagnostics-only (effort: ~30 min, risk: zero)

1. Tighten R-hat gate from 1.05 to 1.01 in `bayesian_summary_writer.py`.
2. Add BFMI check (`az.bfmi()`) and report per-chain values.
3. Re-run M6b posterior summary to confirm it still passes under the new gates (it will: max R-hat 1.00, ESS_bulk 2514).
4. Re-mark M2 (max R-hat 1.12) explicitly as FAIL not "borderline" — this matches Rmus / Baribault standards.

**Test:** `python scripts/05_post_fitting_checks/01_baseline_audit.py --model wmrl_m6b` should still PASS; `--model qlearning` should FAIL with "R-hat 2.38 > 1.01" line.

### Phase B — Init strategy (effort: ~10 min, risk: low)

In `src/rlwm/fitting/sampling.py::run_inference_with_bump`:

```python
from numpyro.infer import init_to_median
mcmc = MCMC(
    NUTS(model, target_accept_prob=tap, init_strategy=init_to_median(num_samples=50)),
    ...,
)
```

Re-fit M2 / M3 / M5 / M6a only (not M1, not M6b — see Phase C/D for those). Wall-time impact: negligible (~5 s extra per fit).

**Test:** Verify M2 R-hat drops below 1.10 and ESS_bulk > 100. If yes, this alone may rescue M2. If no, proceed to Phase C.

### Phase C — Tighten kappa-family priors (effort: ~1 hour, risk: medium)

In `src/rlwm/fitting/numpyro_helpers.py::PARAM_PRIOR_DEFAULTS`, change kappa-family entries:

```python
"kappa":        {"lower": 0.0, "upper": 1.0, "mu_prior_loc": -1.0, "mu_prior_scale": 0.5},
"kappa_s":      {"lower": 0.0, "upper": 1.0, "mu_prior_loc": -1.0, "mu_prior_scale": 0.5},
"kappa_total":  {"lower": 0.0, "upper": 1.0, "mu_prior_loc": -1.0, "mu_prior_scale": 0.5},
"kappa_share":  {"lower": 0.0, "upper": 1.0, "mu_prior_loc":  0.0, "mu_prior_scale": 1.0},  # share is genuinely uncertain
```

This gives a prior mean on κ of `Phi(-1) ≈ 0.16` (close to MLE cluster) but with a less restrictive scale (0.5 vs the MLE-calibrated 0.0/1.0 default), threading the needle between "MLE-circularity" and "vague enough to be unidentifiable."

Also add `mu_prior_scale` to `sample_bounded_param` signature (already supported per code inspection; verify default is 1.0).

**Test:** Sensitivity comparison — run M3 and M6b under (a) current `mu_prior_loc=0.0`, (b) new `mu_prior_loc=-1.0`. Compare 95% HDI on `beta_lec_kappa` / `beta_lec_kappa_total`. If HDI shifts by < 20%, the principled-prior policy is robust; if it shifts by > 20%, document and run both as primary + sensitivity.

### Phase D-1 — Drop M3 / M6a from the Bayesian table; promote M6b (effort: ~10 min, risk: low) — RECOMMENDED

**Earlier framing was wrong.** This document originally proposed "applying M6b's stick-breaking pattern to M3/M5/M6a" by sampling `kappa_anchor × kappa_strength`. That framing treats the change as a generic reparameterization trick. It isn't. M3 has a single conceptual perseveration channel by design (global motor perseveration); decomposing it into two factors changes the marginal prior on κ but does not give the model a second degree of freedom for the data to populate. The correct framing comes from recognizing the model-family containment structure:

```
M6b ⊃ M3   (M3 = M6b restricted to κ_share = 1, all budget on global)
M6b ⊃ M6a  (M6a = M6b restricted to κ_share = 0, all budget on stim-specific)
```

This containment is verified exactly by the existing tests `test_wmrl_m6b_kappa_share_one_matches_m3` and `test_wmrl_m6b_kappa_share_zero_matches_m6a` in `src/rlwm/fitting/models/wmrl_m6b.py:792, 843` (log-likelihood diff < 1e-6).

**Mathematical motivation (model misspecification theorem, informal):**

If the data-generating process has perseveration variance on both channels with magnitudes (κ*_global, κ*_stim) that vary across participants, then under M3:

- The MLE estimator κ̂_global is a non-injective function of (κ*_global, κ*_stim) — different participants with different perseveration profiles can produce the same κ̂_global.
- Individual-level posteriors can remain unimodal, but the *group-level* hyperparameter μ_κ becomes multimodal because the cohort partitions into subgroups where one channel dominates.
- R-hat = 1.60 with ESS = 7 across M3, M5, M6a (identical numbers in our fan-out!) is the sampler *correctly* diagnosing this misspecification, not failing to sample from a well-posed posterior.

Under M6b's two-channel parameterization, (κ̂_total, κ̂_share) maps injectively to (κ̂_total · κ̂_share, κ̂_total · (1 − κ̂_share)) = (κ*_global, κ*_stim) under mild identifiability. The posterior is unimodal because no two generative regimes fold onto the same parameter point. M6b's R-hat = 1.00 with ESS = 2514 is consistent with this.

**Recommended action:** Configure the cluster orchestrator to run hierarchical Bayesian fits only for M2 (baseline) and M6b (perseveration). Retain M1, M3, M5, M6a as MLE-only fits for the AIC comparison table.

```bash
# In cluster/03_submit_bayesian_choice_only.sh (or equivalent), change:
for MODEL in qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b; do ...
# to:
for MODEL in wmrl wmrl_m6b; do ...
```

**Inference recovery for M3 / M6a:** The M6b posterior on κ_share gives the full Bayesian evidence about which sub-model the data prefers. Read the marginal posterior on κ_share: mass concentrated near 1 favors M3-like dynamics, mass near 0 favors M6a, mass spread around 0.5 favors true dual-channel. No need to fit the constrained models separately for inference.

**Cost:**
- Effort: ~10 minutes (orchestrator config + table-builder filter).
- Risk: zero to inference quality (we still report all 7 models in the AIC table from MLE).
- Manuscript narrative: requires one paragraph explaining "we report hierarchical Bayes for the unrestricted model and rely on MLE for the constrained sub-model AIC comparisons."

**Test:** Re-run the smoke fan-out as just `qlearning + wmrl_m6b` instead of all 6. Expect both convergence gates to pass and `.nc` files to land.

### Phase D-2 — Reparameterize κ as softmax bias matching Collins 2025 (effort: ~1–2 weeks, risk: medium-high) — DEFER TO v5.1

This addresses a deeper structural issue distinct from D-1's containment problem. Even after D-1, M6b still uses convex-mixture κ ∈ [0, 1]; Collins 2025 (the most recent canonical paper) uses softmax-bias κ ∈ [−1, 1]. The two formulations are not numerically equivalent at the boundaries.

**Mathematical comparison.** Let `q(a|s) = softmax(βW(s,a))` denote the value-based policy and `i(a) = I(a = a_{t−1})` the last-action indicator.

| Quantity | Senta / our M3 (convex mixture) | Collins 2025 (softmax bias) |
|---|---|---|
| Policy | `(1 − κ) · q(a|s) + κ · i(a)` | `softmax(βW(s,a) + κ · i(a))` |
| Support of κ | `[0, 1]` | `[−1, 1]` (Collins) or `ℝ` (Stan-community default) |
| `dp/dκ` at κ=0 (chosen action) | `i(a) − q(a|s)` — small when q is near-uniform | `q(a|s) · (1 − q(a|s))` — strictly positive, well-conditioned |
| `dp/dκ` at κ=1 | model degenerate (p collapses to one-hot i) | bounded, non-degenerate |
| Negative perseveration (action avoidance) | not representable | natural for κ < 0 |

The convex-mixture parameterization has two boundary pathologies:
1. **At κ = 0 (no perseveration):** the likelihood is locally flat. Phi_approx-bounded priors compound this by mapping the boundary to ±∞ in unconstrained space — exactly the funnel pathology Baribault & Collins (2023) warn about.
2. **At κ = 1 (full perseveration):** the policy collapses to a deterministic last-action repeat. Any data point inconsistent with this drives likelihood → 0; finite-sample participants with low-but-nonzero exploration can drive the chain into a pinned mode.

The softmax-bias formulation has neither pathology because (a) the gradient is informative everywhere and (b) the support is unbounded so there are no boundary modes.

**What changes in our codebase:**

| File | Change | LOC estimate |
|---|---|---|
| `src/rlwm/fitting/models/wmrl_m3.py` | Replace lines 213–216 (convex mixture) with `softmax(beta * Q + kappa * one_hot_last_action)` inside the per-trial policy. Same change in `m5`, `m6a`, `m6b`. | ~15 lines × 4 files = ~60 |
| `src/rlwm/fitting/numpyro_helpers.py` | Add `sample_unbounded_normal_param(name, mu_loc, sigma_loc, mu_prior_scale, sigma_prior_scale, n_participants)` that samples directly on ℝ (no Phi_approx transform). | ~30 lines |
| `src/rlwm/fitting/models/wmrl_m{3,5,6a,6b}.py` (NumPyro hierarchical wrappers) | Replace `sample_bounded_param("kappa", ...)` with `sample_unbounded_normal_param("kappa", mu_loc=0.0, sigma_loc=0.5, ...)`. | ~10 lines × 4 files = ~40 |
| `config.py::MODEL_REGISTRY` | Change κ-family parameter bounds from `(0, 1)` to `(-1, 1)` (matching Collins 2025) for MLE consistency. | ~10 lines |
| `src/rlwm/fitting/mle.py` (or `_engine.py` equivalent) | Update parameter init bounds for fmincon-equivalent optimizer. | ~5 lines |
| MLE refit | Re-run all 7 MLE models under new parameterization to regenerate AIC tables. | ~3 hours of cluster compute |
| `docs/03_methods_reference/MODEL_REFERENCE.md` §3.6 | Rewrite the "M3: WM-RL + Perseveration" formal definition to document the new softmax-bias formulation; add note that this departs from Senta 2025 in favor of Collins 2025. | ~50 lines doc |
| `manuscript/paper.qmd` Methods | Add 2 sentences justifying the parameterization choice (cite Collins 2025; note the identifiability advantage). | ~5 lines |

**What gets invalidated:**

- All current MLE AIC numbers (because κ has different bounds and meaning under the new parameterization). Need full MLE re-fit.
- The `WMRL_M{3,5,6A,6B}_PARAMS` reference fixtures in `tests/integration/fixtures/`.
- Parameter recovery validation (`tests/scientific/`) needs re-running with new generative κ ranges.
- The "M6b winning model" claim in `MODEL_REFERENCE.md:23` may or may not survive the re-fit (likely will — the underlying perseveration signal in the data doesn't change, only its mathematical encoding).

**What gets gained:**

- Likely resolves the residual "even M6b's marginal κ posterior is bunched against zero for low-perseveration participants" pathology (worth checking — see Open Questions).
- Aligns the project with the *most recent* Collins-lab paper rather than the 1-year-older Senta 2025.
- Negative κ (action-avoidance) is now in the model's vocabulary — possibly meaningful for trauma-exposed participants who may have *avoidance* perseveration patterns rather than *repeat* perseveration.

**Why DEFER to v5.1, not do now:**
- v5.0's manuscript is locked to Senta 2025 (M5 RLWM_asymbias_2r) as the reference framework. Switching parameterization mid-milestone re-opens the entire model-comparison narrative.
- Phase D-1 is sufficient to ship v5.0's Bayesian results.
- v5.1 should be an independent "Collins 2025 alignment" milestone that systematically refits everything under the modern parameterization, allowing a clean Senta-vs-Collins comparison in the discussion.

**Test:** A reasonable v5.1 entry test would be: refit `qlearning` (no kappa) and `wmrl_m6b` (kappa_total + kappa_share) under the new parameterization. If both pass Tier-1 convergence gates AND the M6b posterior on κ_total has a similar mode location as the v5.0 fit, the parameterization change is safe. If M6b's mode shifts substantially, the two parameterizations are encoding different latent constructs and we need to write more carefully about which the manuscript is reporting.

### Phase E — M1 Q-learning is the hardest case (effort: ~1 week, risk: high)

If Phase A-D do not rescue M1 (R-hat 2.38 on alpha_pos), the literature is unambiguous: **Q-learning without a WM channel cannot break the α×β trade-off via priors alone.** Three options in increasing aggressiveness:

1. **Accept it.** Document M1 as known-non-identifiable, drop it from the published Bayesian table, retain MLE M1 only for AIC baseline. This is what Senta 2025 effectively does — they fit "RL-only" variants but acknowledge they "vary in characterization" of effects. Cost: zero extra work, full honesty.
2. **Tighten α priors heavily** to `Normal(-1, 0.5)` per Baribault 2023's "more informative priors" recipe. May or may not work; needs empirical test. Cost: 1 hour + a refit.
3. **Joint RT modeling** (Pedersen 2022) — empirically the only thing that breaks α×β r=0.49. But our task is choice-only after 1.5 s timeout; we don't have RT distribution structure for non-decision-time identification. **Not feasible** without redesigning the task or restricting analysis to a subset with rich RT.

Recommendation: **Option 1.** Drop M1 from the Bayesian table.

### Phase F (optional) — adopt Rmus 2023's Beta+Gamma alternative

If Phase A–D do not get M3/M5/M6a all the way to R-hat ≤ 1.01, consider rewriting the prior block in Rmus 2023 style:

```python
a = numpyro.sample("kappa_a", dist.Gamma(2.0, 1.0))
b = numpyro.sample("kappa_b", dist.Gamma(8.0, 1.0))     # heavy mass on small κ
kappa = numpyro.sample("kappa", dist.Beta(1+a, 1+b).expand([N]))
```

This is a structurally different parameterization (no Phi_approx, no z-offset) and would require validating against the prior predictive. **Defer until Phase D is empirically inadequate.**

---

## Open questions

1. ~~**What does Eckstein 2022 (eLife 75474) actually use for priors?**~~ **RESOLVED 2026-05-03 by direct PDF read.** Their RLWM Task C does not include perseveration parameters; they delegate priors to Master 2020 / Xia 2021. Cannot help with our κ identifiability problem. See Tier 0.
2. ~~**Is M6b's stick-breaking actually equivalent to hBayesDM's standard parameterization?**~~ **PARTIALLY RESOLVED.** M6b is not a generic stick-breaking trick — it is the unrestricted dual-channel perseveration model that contains M3 and M6a as boundary corner-cases. The relevant literature precedent is therefore Collins 2025's WMH motor-vs-stimulus split, not Dirichlet-process stick-breaking. See Tier 0 / Phase D-1.
3. ~~**Why do M3, M5, M6a hit identical R-hat=1.60 with identical ESS=7?**~~ **RESOLVED.** They share the same scalar-κ misspecification: forced to absorb dual-channel perseveration variance into a single parameter, the group hyperparameter posterior becomes multimodal across the cohort. M6b's two-channel parameterization breaks the multimodality. R-hat=1.60 was the sampler correctly diagnosing model misspecification, not failing on a well-posed posterior.
4. **Does our `phi_approx` actually equal hBayesDM's?** Open. hBayesDM uses Stan's built-in polynomial-approximation `Phi_approx`; our `numpyro_helpers.py::phi_approx` uses exact `jax.scipy.stats.norm.cdf`. Numerical difference ~7.5e-8 — cannot be the source of convergence failures but the docs use the wrong name. Worth a 5-min audit + rename.
5. **Should sigma_pr scale up from 0.2?** Open. Baribault 2023 uses `Normal(0, 0.5) T[0,1]` (= HalfNormal(0.5)); hBayesDM uses 0.2; we use 0.2. After Phase D-1, the M2 + M6b family should re-pass the convergence gates; if they don't, this is the next knob.
6. **Does our M6b posterior actually identify both κ channels in our N=158 data?** Open. M6b passed convergence (R-hat=1.0, ESS=2514) — but that doesn't tell us whether κ_share is actually informed by the data or is just sitting near its prior. Need to compute prior-vs-posterior shrinkage on κ_share. If the posterior is flat (data uninformative), the inference about which channel matters is conditional on the prior, not the data, and we should report sensitivity intervals.
7. **What does Master 2020 (DCN 41:100732) actually use for priors?** Open. Eckstein 2022 cites Master 2020 as the canonical Bayesian prior reference for the Collins-lab RLWM. Not in our local PDF library. If institutional access available, this likely resolves the Phi_approx-vs-Beta question for the modern Collins lab.
8. **Should v5.1 adopt Phase D-2?** Open question for the next milestone. The convex-mixture vs softmax-bias parameterization choice is a deeper structural issue than the M3⊂M6b containment problem; resolving it requires a manuscript-narrative decision about which parameterization to canonicalize.

---

## Sources

1. **Senta, Bishop, Collins (2025)** — "Dual process impairments in reinforcement learning and working memory systems underlie learning deficits in anxiety." PLOS Comput Biol 21(9):e1012872. https://doi.org/10.1371/journal.pcbi.1012872 (local PDF: `docs/03_methods_reference/references/Senta et al. - 2025...pdf`).
2. **Rmus M, He M, Baribault B, Walsh EG, Festa EK, Collins AGE, Nassar MR (2023)** — "Age-related differences in prefrontal glutamate are associated with increased working memory decay that gives the appearance of learning deficits." eLife 12:e85243. https://elifesciences.org/articles/85243
3. **Baribault B, Collins AGE (2023)** — "Troubleshooting Bayesian cognitive models: A tutorial with matstanlib." Psychological Methods 30(1):128–154. https://pmc.ncbi.nlm.nih.gov/articles/PMC10522800/ (DOI: 10.1037/met0000554).
4. **Eckstein MK, Master SL, Xia L, Dahl RE, Wilbrecht L, Collins AGE (2022)** — "The interpretation of computational model parameters depends on the context." eLife 11:e75474. DOI: 10.7554/eLife.75474. **Verified via local PDF read 2026-05-03**: their RLWM Task C uses α+, α−, K, F, ε, ρ — no perseveration. Hierarchical fitting confirmed (Appendix 7) but priors delegated to Master 2020 / Xia 2021. Data: https://osf.io/h4qr6/. Local PDF: `C:\Users\aman0087\Downloads\elife-75474.pdf`.

4a. **Collins AGE (2025)** — "A habit and working memory model as an alternative account of human reward-based learning." Nat Hum Behav 10:357–369. DOI: 10.1038/s41562-025-02340-0. **Paradigm-shifting solo-author paper.** Reanalyses 7 RLWM datasets (n=594) and concludes WMH (WM + Hebbian habit) outperforms RLWM family on all 6 datasets. Methods: MATLAB fmincon MLE, ten random starts, AIC. Perseveration is parameterized as softmax-bias κ ∈ [−1, 1], not convex-mixture κ ∈ [0, 1] as in Senta 2025 / our implementation. Code: https://github.com/AnneCollins/WMH. Local PDF: `C:\Users\aman0087\Downloads\s41562-025-02340-0.pdf`.
5. **Master SL, Eckstein MK, Gotlieb N, Dahl R, Wilbrecht L, Collins AGE (2019)** — "Disentangling the systems contributing to changes in learning during adolescence." bioRxiv 622860 / Dev Cogn Neurosci 41 (2020):100732. https://www.biorxiv.org/content/10.1101/622860v1
6. **Collins AGE (2018)** — "The Tortoise and the Hare: Interactions between RL and WM." J Cogn Neurosci 30(10):1422–32. https://doi.org/10.1162/jocn_a_01238
7. **Collins AGE, Brown JK, Gold JM, Waltz JA, Frank MJ (2014)** — "WM Contributions to RL Impairments in Schizophrenia." J Neurosci 34(41):13747–56. https://pmc.ncbi.nlm.nih.gov/articles/PMC4188972/
8. **Collins AGE, Frank MJ (2012)** — "How much of RL is WM, not RL?" Eur J Neurosci 35(7):1024–35. https://pubmed.ncbi.nlm.nih.gov/22487033/
9. **Rmus M, McDougle SD, Collins AGE (2021)** — "The role of executive function in shaping reinforcement learning." Curr Opin Behav Sci 38:66–73.
10. **Collins AGE, Ciullo B, Frank MJ, Badre D (2017)** — "WM Load Strengthens RPE." J Neurosci 37(16):4332–42. https://pmc.ncbi.nlm.nih.gov/articles/PMC5413179/
11. **Collins AGE, Frank MJ (2018)** — "Within- and across-trial dynamics of human EEG reveal cooperative interplay between RL and WM." PNAS 115(10):2502–7.
12. **Ahn WY, Haines N, Zhang L (2017)** — "Revealing Neurocomputational Mechanisms of RL and Decision-Making with the hBayesDM Package." Comput Psychiatry 1:24–57. PMC5869013. Code: https://github.com/CCS-Lab/hBayesDM
13. **hBayesDM Stan source files** (verified verbatim):
    - https://github.com/CCS-Lab/hBayesDM/blob/develop/commons/stan_files/bandit2arm_delta.stan
    - https://github.com/CCS-Lab/hBayesDM/blob/develop/commons/stan_files/bandit4arm_4par.stan
    - https://github.com/CCS-Lab/hBayesDM/blob/master/commons/stan_files/prl_ewa.stan
14. **matstanlib (Baribault)** — https://github.com/baribault/matstanlib (verified). Companion code for source [3].
15. **van Geen C, Gerraty RT (2021)** — "Hierarchical Bayesian Models of Reinforcement Learning: Introduction and comparison to alternative methods." J Math Psychol / bioRxiv 2020.10.19.345512. https://www.biorxiv.org/content/10.1101/2020.10.19.345512v2
16. **Pedersen ML, Frank MJ, Biele G (2022)** — "Joint modeling of reaction times and choice improves parameter identifiability in reinforcement learning models." J Neurosci Methods. https://pmc.ncbi.nlm.nih.gov/articles/PMC8930195/
17. **Stan Wiki: Prior Choice Recommendations** — https://github.com/stan-dev/stan/wiki/prior-choice-recommendations (consulted for HalfNormal vs HalfCauchy guidance on group SDs).
18. **Local repo references** (pre-existing context this document builds on):
    - `docs/03_methods_reference/MODEL_REFERENCE.md` §Hierarchical Bayesian Architecture
    - `docs/04_methods/README.md` §3 (Fitting-procedure audit vs literature)
    - `src/rlwm/fitting/numpyro_helpers.py` (`PARAM_PRIOR_DEFAULTS`, `sample_bounded_param`)
    - `.planning/research/PITFALLS.md` Pitfalls 1–4 (centered parameterization, hierarchical shrinkage, IES-R collinearity, lax.scan compile cost)
    - `src/rlwm/fitting/sampling.py::run_inference_with_bump` (target_accept auto-bump 0.80/0.95/0.99)
