# Phase 32 Research — Pointer + Phase-Specific Synthesis

**Status:** Research complete (deep research conducted 2026-05-03 prior to phase creation).

## Canonical research document

The exhaustive 18-source literature review for this phase lives at:

→ **`.planning/research/mcmc-collins-bayesian.md`** (448 lines)

That document was produced before this phase was added to the roadmap, in response to the Path A smoke-test failure (qlearning R-hat = 2.38 on cluster job 55117907). It is the authoritative methodological reference for Phase 32. **Do not duplicate its content here** — read it directly for Per-paper reviews, Decisions table, and full Implementation plan with Phases A–F.

## Phase 32 mapping (research → plan)

| Research-doc Phase | Phase 32 Plan | Status |
|---|---|---|
| Phase A — Diagnostic gate hardening | 32-01 | New |
| Phase B — Init strategy | 32-02 | Bundled with C |
| Phase C — Tighten κ-family priors | 32-02 | Bundled with B |
| Phase D-1 — Drop M3 / M6a from Bayesian | 32-03 | New (extended to also drop M5 per user decision) |
| Phase D-2 — Softmax-bias κ with CLI revert | 32-04 | **Promoted from "v5.1 candidate" to v5.0 ship per user decision 2026-05-03** |
| (Smoke test gates between Phases) | 32-05 | New |
| (Pipeline rerun) | 32-06 | New |

## Key findings driving the phase

1. **Our Phi_approx + Normal(0,1) + HalfNormal(0.2) non-centered setup is line-for-line equivalent to hBayesDM's official Stan code** (`bandit2arm_delta.stan`, `bandit4arm_4par.stan`, `prl_ewa.stan`). For α / φ / ρ / K the parameterization is correct. **The bug is in the perseveration spec, not in NumPyro plumbing.** No general-purpose framework refactor needed.

2. **Collins 2025 (Nat Hum Behav 10:357–369, sole author) is a paradigm-shifting paper** that reanalyses 7 RLWM datasets (n=594) and shows WMH (WM + Hebbian habit, no model-free RL) outperforms RLWM family on all 6 published datasets. Most importantly for us: she uses a *softmax-bias* perseveration κ ∈ [−1, 1], not Senta 2025's *convex-mixture* κ ∈ [0, 1]. The two formulations differ in boundary geometry — convex mixture has a degenerate boundary at κ = 1 and a flat-likelihood boundary at κ = 0; softmax bias has neither. **This is plausibly the deepest reason our M3 Bayesian fit is multimodal where Collins's MLE fits converge.**

3. **M3, M5, M6a are statistically misspecified** relative to the data. They are M6b restricted to corners of the κ_share simplex. R-hat = 1.60 with ESS = 7 across all three (identical numbers!) is the sampler *correctly* diagnosing the misspecification — not a sampling bug. The existing tests `test_wmrl_m6b_kappa_share_one_matches_m3` and `test_wmrl_m6b_kappa_share_zero_matches_m6a` (in `src/rlwm/fitting/models/wmrl_m6b.py:792, 843`) verify the containment exactly (log-likelihood diff < 1e-6).

4. **β = 50 fixed during learning is correct** — both Senta 2025 and Collins 2025 fix β at a high value during learning. Rmus 2023 fixes β_learning at the group level. Our task has no test phase. Unfixing β reintroduces the α-vs-β trade-off (Pedersen 2022: r = 0.49) and breaks MLE comparability.

5. **Convergence gate must tighten to Baribault & Collins (2023) Tier-1 standard:** R-hat ≤ 1.01, ESS ≥ 400, divergences = 0, BFMI ≥ 0.2. Our current ≤ 1.05 is too loose; under it, M2's R-hat = 1.12 is mislabeled as "borderline."

6. **Eckstein 2022 (eLife 11:e75474) is a dead end for our κ problem** — their Task C RLWM model has no perseveration parameters at all (verified by direct PDF read 2026-05-03). They use hierarchical Bayes ("for tasks A and B" per Appendix 7) but priors are delegated to Master 2020 / Xia 2021.

## User decisions captured 2026-05-03

- **Q1 (data-saving fixes):** "wherever we got stalled" — interpreted as: fold any pending writer/schema fixes from Phase 24 stall (canary stage) into Phase 32. Specifically: convergence-table schema needs the new BFMI + per-chain ESS fields from 32-01.
- **Q2 (M5 disposition):** Drop M5 from Bayesian table for now. Do NOT build M5b = M6b + φ_rl. Rationale: Collins 2025 explicitly says φ_rl is unnecessary (Methods, p.366: "I verified that letting RL processes also have forgetting did not improve fit").
- **Q3 (test fixtures):** "cleanup tests as needed" — keep convex-mode tests, add softmax-mode tests, parameterize where simple.
- **CLI revert:** User explicitly asked for `--kappa-parameterization {softmax,convex}` flag with default `softmax` so we can revert if softmax has its own problems.

## Anchor files for the planner

| Concern | File | Notes |
|---|---|---|
| Convergence gate logic | `src/rlwm/fitting/bayesian.py` (around `save_results` → convergence gate) + `scripts/fitting/bayesian_summary_writer.py` | Recent fix `9a62baa` (May 3) — R-hat / ESS / divergences gate |
| NumPyro sampler | `src/rlwm/fitting/sampling.py::run_inference_with_bump` | `target_accept` auto-bump 0.80 → 0.95 → 0.99 |
| Hierarchical helpers | `src/rlwm/fitting/numpyro_helpers.py` (`PARAM_PRIOR_DEFAULTS`, `sample_bounded_param`, `phi_approx`) | Add `sample_unbounded_normal_param` here for softmax κ |
| JAX likelihoods (per model) | `src/rlwm/fitting/models/{wmrl_m3,wmrl_m5,wmrl_m6a,wmrl_m6b}.py` | Convex-mixture κ logic in each `_block_likelihood`. Add softmax-bias variant guarded by a `parameterization: str = "softmax"` kwarg. |
| Hierarchical wrappers | Same files, `*_hierarchical_model` functions | Switch `sample_bounded_param("kappa", ...)` → `sample_unbounded_normal_param("kappa", ..., mu_loc=0.0, sigma_loc=0.5)` when softmax mode. |
| MODEL_REGISTRY | `config.py` | κ bounds depend on parameterization mode. May need to be parameterized. |
| MLE engine | `src/rlwm/fitting/mle.py` | CLI flag plumbing. Per-model parameter bounds depend on κ mode. |
| Bayesian engine | `src/rlwm/fitting/bayesian.py` | CLI flag plumbing. Run-metadata schema needs `kappa_parameterization` field. |
| CLI entries | `scripts/04_model_fitting/a_mle/fit_mle.py`, `scripts/04_model_fitting/b_bayesian/fit_baseline.py`, `fit_bayesian.py` | Argparse additions. |
| Cluster orchestrator | `cluster/04b_bayesian_gpu.slurm` (or equivalent), `cluster/submit_all.sh` | Pass `KAPPA_MODE=softmax` env var; drop M3/M5/M6a from Bayesian model list. |
| Tests | `tests/integration/test_v4_closure.py`, `tests/scientific/test_*.py`, `src/rlwm/fitting/models/wmrl_m6b.py:792, 843` | Cleanup + new softmax tests. v4 closure guards must still pass. |
| Docs | `docs/03_methods_reference/MODEL_REFERENCE.md` §3.6 | Document both κ formulations + Senta vs Collins citation. |
| Manuscript | `manuscript/paper.qmd` Methods section | 2-paragraph addition citing Collins 2025 + parameterization rationale. |

## Effort estimate (from research doc Phase D-2 table)

~280 LOC of code changes across 8–10 files + ~3 hours cluster compute for MLE re-fit + ~50 LOC manuscript edits. Total dev time ~1–2 weeks if done carefully. Wave 1 (32-01..03) is parallelizable and ~50 LOC total.
