# Feature Research

**Domain:** Computational cognitive modeling — RLWM model extensions (M4-M6)
**Researched:** 2026-04-02
**Confidence:** HIGH for M4/M5 (primary literature verified); MEDIUM for M6 (inferred from M3 structure + perseveration literature)

---

## Context: What Already Exists

The pipeline already implements M1-M3 with the following infrastructure:

- JAX likelihood functions in `scripts/fitting/jax_likelihoods.py` (JIT-compiled, masked-padding architecture)
- MLE fitting with 20 restarts and L-BFGS-B in `scripts/fitting/fit_mle.py`
- Parameter bounds registry in `scripts/fitting/mle_utils.py` (BOUNDS dicts, PARAMS lists)
- Parameter recovery pipeline in `scripts/fitting/model_recovery.py` (r >= 0.80 criterion)
- Model comparison with AIC/BIC and Akaike weights in `scripts/fitting/compare_mle_models.py`
- Trauma analysis scripts (15, 16) that accept `--model all`

New models must slot into this infrastructure without disrupting existing functionality.

---

## Feature Landscape

### Table Stakes (Must Have for Scientific Validity)

These are not optional. Missing any of these means a model cannot be published or interpreted.

| Feature | Why Expected | Complexity | Notes |
|---------|--------------|------------|-------|
| **M4: LBA likelihood function** — joint P(choice=i, RT=t) | LBA without RT data defeats its purpose; choice-only LBA collapses to softmax | HIGH | Requires closed-form Brown & Heathcote (2008) PDF/CDF. Three competing accumulators. Must be JAX-differentiable for L-BFGS-B. |
| **M4: RT data loading and preprocessing** | The pipeline currently loads choices only; RT column must be available per trial | MEDIUM | `task_trials_long.csv` may already have RT; needs verification. Preprocessing: RT in seconds, subtract t0 before LBA density |
| **M4: Drop epsilon, add v_scale, A, b, t0** | LBA subsumes undirected noise via start-point variability A; adding epsilon would double-count noise | MEDIUM | Fixed: s=0.1 (McDougle & Collins, 2021). Fixed: t0=150ms recommended for identifiability (same paper). Free: v_scale, A, b |
| **M4: Policy-to-drift-rate mapping** | The WM-RL hybrid policy must feed into LBA accumulators; the mapping IS the model's claim | HIGH | Following McDougle & Collins (2021): v_i = v_scale * (pi_i / H_prior). Entropy term H_prior creates conflict-based RT slowing. |
| **M5: phi_RL parameter in likelihood** | RL forgetting is the core scientific claim of M5; absent = it's just M3 with a broken parameter | MEDIUM | Q decays toward Q0=1/nA each trial before updating. Order: decay WM → decay Q → compute policy → observe → update WM → update Q |
| **M5: Correct decay update ordering** | Order matters; decaying after updating is mathematically wrong and violates the model spec | LOW | Must match CLAUDE.md update sequence. WM decay already in M3; RL decay is the addition. |
| **M6a: Per-stimulus last_action tracking** | Stimulus-specific perseveration requires a `last_actions` array of shape (num_stimuli,) not a scalar | MEDIUM | Replaces scalar `last_action` with `last_actions[stimulus_idx]`. JAX requires careful indexing for functional updates. |
| **M6b: Dual perseveration with constraint** | Constraint kappa + kappa_s <= 1 is a scientific prior (total perseveration probability must sum to <= 1) | MEDIUM | Constraint can be enforced via bounded optimization (upper bound on each is 1.0 but sum checked) OR reparameterize. |
| **Parameter recovery for each new model** | Without recovery r >= 0.80, parameters cannot be interpreted as measuring what they claim | HIGH | Must run Script 11 equivalent for M4, M5, M6a, M6b before fitting real data. This is the gating criterion. |
| **Model comparison including M1-M6** | Comparing M4/M5/M6 only against each other, not against M1-M3, would be scientifically incomplete | MEDIUM | Script 14 must accept M4/M5/M6 fits and compute AIC/BIC against all models. AIC differs from BIC by k; M4 is not directly comparable to M1-M3 if RT data is added (different data). |
| **Parameter bounds registration** | New models need bounds in `mle_utils.py`; without this, restarts cannot be sampled | LOW | Add `M4_BOUNDS`, `M5_BOUNDS`, `M6A_BOUNDS`, `M6B_BOUNDS` and corresponding PARAMS lists. |

---

### Differentiators (Things That Make This Implementation Particularly Good)

| Feature | Value Proposition | Complexity | Notes |
|---------|-------------------|------------|-------|
| **M4: Entropy-weighted drift rates (piH model)** | The plain pi model (no entropy) fails to capture RT variation across set sizes; entropy is what explains slower RTs on harder trials | MEDIUM | H_prior computed as Shannon entropy of average policy across stimuli in current block. This is McDougle & Collins' key contribution. Without entropy the model loses most of its RT predictive power. |
| **M4: Validate RT data availability before fitting** | Participants without RT data (e.g., timeout trials) should be handled gracefully, not silently dropped | LOW | Log warning if RT is NaN or >3000ms; exclude those trials from likelihood but keep choice. |
| **M5: phi_RL == 0 reduces exactly to M3** | Mathematical equivalence as a sanity check; if the model cannot recover this boundary it has implementation bugs | LOW | Test: fit synthetic M3 data with M5; recovered phi_RL should be near 0. |
| **M6a vs M6b model comparison** | Comparing global-only (M3), stimulus-only (M6a), and dual (M6b) perseveration identifies which type of response stickiness dominates in the data | LOW | The M6 variants compete directly; AIC comparison will tell which perseveration structure is favored. |
| **Constraint enforcement for M6b** | Naive bounds (kappa: [0,1], kappa_s: [0,1]) allow kappa=0.9, kappa_s=0.9, which is not scientifically valid | MEDIUM | Options: (1) inequality constraint in L-BFGS-B (not directly supported), (2) reparameterize: kappa = p * total, kappa_s = (1-p) * total where total in [0,1] and p in [0,1]. Option 2 is simpler with JAX. |
| **Hessian-based standard errors for new params** | `mle_utils.py` already computes Hessian SEs; extending to M4/M5/M6 gives confidence intervals on new params | LOW | Inherited from existing infrastructure; just needs to run on new parameter vectors. |
| **Trauma correlation analysis for new params** | v_scale (M4) and phi_RL (M5) may correlate with trauma scores; this is the scientific payoff | LOW | Scripts 15 and 16 already accept `--model all`; just need the new model names registered. |

---

### Anti-Features (Explicitly Do NOT Build)

| Anti-Feature | Why Requested | Why Problematic | Alternative |
|--------------|---------------|-----------------|-------------|
| **M4: Free non-decision time t0** | "More parameters = better fit" intuition | McDougle & Collins (2021) explicitly showed that free t0 traded off with threshold b, reducing parameter recoverability. t0 correlation with b is r = -0.995 when free. The identifiability cost exceeds the fit benefit. | Fix t0 = 150ms. If reviewers require it, report sensitivity analysis with t0 fixed at 100ms and 200ms. |
| **M4: Free within-trial noise s** | Same "more parameters" intuition | The LBA requires s fixed for identifiability; it is not separately identified from the threshold scale. Fixing s=1 (standard LBA) or s=0.1 (McDougle & Collins scaling) is required. | Fix s = 0.1 following McDougle & Collins (2021). |
| **M4: Epsilon noise retained alongside LBA** | Maintaining backward compatibility | Epsilon models undirected noise. In LBA, start-point variability A already models this. Combining them creates redundancy: two parameters doing the same job, neither recoverable. | Drop epsilon. Add A (start-point range) as the noise/lapse mechanism. |
| **M4: Bayesian MCMC fitting (script 13)** | More principled inference | M4 adds 3 LBA parameters (v_scale, A, b) but not a complex new architecture. The RT distribution makes MLE well-constrained. Bayesian fitting triples compute time. | Implement MLE only; Bayesian can be added post-defense if reviewers require it. |
| **M6b: Unconstrained dual kappa parameters** | Simplicity — just give both kappa and kappa_s bounds [0,1] | kappa + kappa_s > 1 is mathematically possible under naive bounds but scientifically invalid (total perseveration probability > 1 means the model predicts responses exceeding 100%). | Reparameterize: total_kappa = kappa_total in [0,1]; split_kappa = p in [0,1]; kappa = p * kappa_total; kappa_s = (1-p) * kappa_total. |
| **Backward compatibility breaks in compare_mle_models.py** | "Refactor everything while we're in here" | M1-M3 comparison results are already computed and stored. Changing the comparison script API breaks reproducibility of existing comparisons. | Extend compare_mle_models.py with new `--m4/--m5/--m6a/--m6b` flags; keep existing flag interface unchanged. |
| **M4 comparison against M1-M3 on same AIC scale** | "Unified model comparison across all 6 models" | M4 uses RT+choice data; M1-M3 use choice data only. AIC computed on different data is not comparable. Comparing them implies equivalent information content. | Report M4 separately: (a) choice-only fit (comparable to M1-M3), (b) joint RT+choice fit (M4 advantage). Use cross-validated log-likelihood if direct comparison is needed. |
| **New standalone simulation scripts (09/10 variants)** | "Complete coverage" | Scripts 09 and 10 already exist for M1-M3. Duplicating them for M4-M6 creates maintenance burden. | Add model cases to existing `09_generate_synthetic_data.py` and `10_run_parameter_sweep.py` via new `--model m4/m5/m6a/m6b` flags. |

---

## Feature Dependencies

```
[M4 likelihood function]
    └──requires──> [RT data column in task_trials_long.csv]
    └──requires──> [LBA PDF/CDF implementation (JAX-differentiable)]
    └──requires──> [v_scale, A, b parameter bounds in mle_utils.py]
    └──requires──> [Policy-to-drift mapping (pi_i / H_prior)]

[M5 likelihood function]
    └──requires──> [phi_RL bounds in mle_utils.py]
    └──extends──> [M3 likelihood] (M3 carry state + Q-decay before update)

[M6a likelihood function]
    └──extends──> [M3 likelihood] (scalar last_action → array last_actions[num_stimuli])

[M6b likelihood function]
    └──extends──> [M6a likelihood] (add global kappa alongside stimulus-specific kappa_s)
    └──requires──> [M6b reparameterization OR constraint in optimizer]

[Parameter recovery for M4/M5/M6a/M6b]
    └──requires──> [Respective likelihood functions]
    └──requires──> [Respective bounds in mle_utils.py]
    └──blocks──> [Real data fitting] (cannot fit real data until recovery passes)

[Model comparison M1-M6]
    └──requires──> [All individual fits CSVs]
    └──requires──> [compare_mle_models.py extended with new model flags]
    └──NOTE──> [M4 requires separate comparison track (RT data)]

[Trauma analysis for new params]
    └──requires──> [Individual fits CSVs for M4/M5/M6]
    └──requires──> [Scripts 15+16 recognize new model names]
    └──requires──> [Parameter recovery passed]
```

### Dependency Notes

- **M5 requires M3 carry state extension:** M5 adds a `Q_decay_baseline` constant and a `phi_rl` closure variable to the scan step. The M3 carry is `(Q, WM, WM_0, log_lik, last_action)`; M5 is the same — phi_rl acts in the step function, not the carry.
- **M6a conflicts with M3 state structure:** M3 uses a scalar `last_action` (int32); M6a needs `last_actions` array shape (num_stimuli,). These cannot share a carry tuple type in JAX without separate function definitions.
- **M6b extends M6a:** M6b's global kappa is the same as M3's kappa; M6b's kappa_s is M6a's kappa. M6b = M6a carry + extra kappa parameter. If M6a is implemented correctly, M6b is a small extension.
- **RT data dependency is a blocker for M4:** If RT column is absent or unreliable in the CSV, M4 cannot be implemented. This must be verified before any M4 coding begins.

---

## MVP Definition

### Launch With (v1 — thesis defense ready)

These features constitute the minimum needed for M4-M6 to appear in a thesis:

- [ ] **M5 likelihood (phi_RL)** — lowest risk, highest scientific value. phi_RL is the clearest extension to M3 with direct theoretical justification from Collins (2018). Recovery is straightforward.
- [ ] **M6a likelihood (stimulus-specific kappa)** — moderate risk, directly contrasts with M3's global kappa. Recovery expected to be clean.
- [ ] **M6b likelihood (dual perseveration)** — moderate risk; requires M6a working first. The constraint reparameterization is the main engineering challenge.
- [ ] **Parameter bounds for M5, M6a, M6b** — required for any fitting
- [ ] **Parameter recovery for M5, M6a, M6b** — gating criterion before real fits
- [ ] **Model comparison extended to M1-M6 (choice-only models)** — enables the thesis comparison table
- [ ] **Trauma analysis for phi_RL, kappa_s** — the scientific payoff

### Add After Validation (v1.x — if time permits)

- [ ] **M4 likelihood (LBA joint RT+choice)** — add after RT data availability is confirmed. This is the highest-complexity feature and the one most likely to encounter data quality issues (RT outliers, timeouts). Add only after M5/M6 are validated.
- [ ] **M4 parameter recovery** — requires M4 likelihood working first
- [ ] **M4 separate model comparison track** — only meaningful after recovery passes
- [ ] **M4 trauma analysis for v_scale** — only if recovery is clean

### Future Consideration (v2+ — post-defense)

- [ ] **Bayesian MCMC for M4 (numpyro)** — if reviewers require full posteriors for LBA parameters
- [ ] **M4 comparison against M1-M3 via cross-validation** — only if unified comparison is required and CV infrastructure is added
- [ ] **M7+ models** — e.g., decay in both Q and WM with separate rates, or non-linear WM capacity functions

---

## Feature Prioritization Matrix

| Feature | Scientific Value | Implementation Cost | Priority |
|---------|-----------------|---------------------|----------|
| M5 phi_RL likelihood | HIGH — tests whether RL forgetting parallels WM forgetting in trauma | MEDIUM | P1 |
| M5 parameter recovery | HIGH — gating criterion | MEDIUM | P1 |
| M6a stimulus-specific kappa likelihood | HIGH — disambiguates perseveration type | MEDIUM | P1 |
| M6a parameter recovery | HIGH — gating criterion | MEDIUM | P1 |
| M6b dual perseveration | MEDIUM — exploratory; may add complexity without gain | MEDIUM | P1 |
| M6b constraint reparameterization | MEDIUM — required for scientific validity | MEDIUM | P1 |
| Model comparison M1-M6 (choice-only) | HIGH — required for thesis comparison table | LOW | P1 |
| Trauma analysis for new params | HIGH — core thesis claim | LOW | P1 |
| M4 LBA likelihood | MEDIUM — adds RT, but RT not primary outcome | HIGH | P2 |
| M4 parameter recovery | MEDIUM — required if M4 is included | HIGH | P2 |
| M4 RT data verification | HIGH (blocker) — M4 cannot proceed without it | LOW | P2 |
| M4 separate comparison track | MEDIUM | LOW | P2 |
| Bayesian MCMC for M4 | LOW — overkill for thesis scope | VERY HIGH | P3 |
| CV-based unified comparison | LOW — not required if models are on separate data tracks | HIGH | P3 |

**Priority key:**
- P1: Must have for thesis defense
- P2: Should have; include if M4 RT data is available and recovery is clean
- P3: Defer post-defense

---

## Scientific Validity Requirements (Cross-Cutting)

These are not features but constraints that apply to all new models:

**Parameter recovery criterion:** r >= 0.80 (Pearson) for all free parameters, following Senta et al. (2025) and Palminteri et al. (2017). Recovery must use N=100 synthetic participants with parameters sampled uniformly from bounds (matching `model_recovery.py` procedure).

**Nested model boundary conditions:**
- M5 with phi_RL=0 must produce identical likelihood to M3 (same inputs). Test this explicitly.
- M6a with kappa_s=0 must produce identical likelihood to M1/M2 without perseveration (or M3 with global kappa=0).
- M6b with kappa_s=0 reduces to M3; M6b with kappa=0 reduces to M6a. Both must be verified.

**M4 data comparability note:** M4 uses a different data source (choice + RT) than M1-M3 (choice only). AIC/BIC values are NOT directly comparable between M4 and M1-M3. This distinction must be explicit in reporting. The recommended approach is to report M4 vs M3 on hold-out cross-validated log-likelihood, or report M4 advantage as "improvement in individual parameter recovery" not model comparison AIC.

**Parameter count for AIC penalty:**
- M4: 7 params (alpha_pos, alpha_neg, phi, rho, K, v_scale, A) — b can be fixed at A + some constant, OR b is free (8 params). t0 fixed at 150ms, s fixed at 0.1, epsilon dropped.
- M5: 8 params (M3's 7 + phi_rl)
- M6a: 7 params (M3's 7, same count — kappa_s replaces global kappa)
- M6b: 8 params (M3's 7 + kappa_s, with reparameterization giving kappa_total + p = 2 params replacing kappa alone)

---

## Sources

- Brown & Heathcote (2008), "The simplest complete model of choice response time: linear ballistic accumulation." *Cognitive Psychology* 57(3): 153–178. [PubMed](https://pubmed.ncbi.nlm.nih.gov/18243170/)
- McDougle & Collins (2021), "Modeling the influence of working memory, reinforcement, and action uncertainty on reaction time and choice during instrumental learning." *Psychonomic Bulletin & Review* 28: 65–84. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC7854965/) — PRIMARY SOURCE for M4 design.
- Collins & Frank (2018), "Within and across-trial dynamics of human EEG reveal cooperative interactions between reinforcement learning and working memory." *PNAS.* — source for phi_RL (RL forgetting) parameter.
- Senta et al. (2025), "Dual process impairments in reinforcement learning and working memory systems underlie learning deficits in physiological anxiety." *PLOS Computational Biology.* [PLOS](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012872) — confirms phi_RL and kappa model structures.
- Katahira (2018), "How hierarchical models improve point estimates of model parameters at the individual level." *Journal of Mathematical Psychology.* — source for stimulus-specific vs. global perseveration distinction.
- Urai et al. (2022), "Joint modeling of reaction times and choice improves parameter identifiability in reinforcement learning models." *Journal of Neuroscience Methods.* [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8930195/)
- SequentialSamplingModels.jl LBA documentation: [link](https://itsdfish.github.io/SequentialSamplingModels.jl/dev/lba/) — parameter naming conventions.

---

*Feature research for: RLWM model extensions M4-M6 (trauma analysis pipeline)*
*Researched: 2026-04-02*
