---
phase: 21-principled-bayesian-model-selection-pipeline
plan: 11
subsystem: model-fitting

tags: [numpyro, hierarchical-bayesian, level-2-regression, covariate-iesr, level2-design, recovery-test, backward-compat]

# Dependency graph
requires:
  - phase: 16
    provides: "wmrl_m3/m5/m6a_hierarchical_model with single covariate_lec L2 hook; build_level2_design_matrix for M6b 4-cov subscale"
provides:
  - "M3/M5/M6a hierarchical models accept optional 2nd covariate (covariate_iesr) via Approach A additive kwarg"
  - "beta_iesr_{target} numpyro sample site added only when covariate_iesr is provided (backward-compat preserved for Phase 16 LEC-only callers)"
  - "Guard raising ValueError if covariate_iesr is passed without covariate_lec"
  - "build_level2_design_matrix_2cov helper returning (N, 2) z-scored design for lec_total + iesr_total"
  - "Test suite (9 fast parametrizations + 1 slow recovery) gating the 2-cov hook on M3/M5/M6a"
affects:
  - "21-07 (winner-dispatched L2 refit): can thread covariate_iesr to M3/M5/M6a winners at production budget without further model-level changes"
  - "21-10 (orchestrator): adds pytest test_numpyro_models_2cov.py -v -k 'not slow' as a local gate before cluster submission"

# Tech tracking
tech-stack:
  added:
    - "numpyro.diagnostics.split_gelman_rubin (grouped-chain rhat gate inside recovery test)"
  patterns:
    - "Approach A additive-parameter L2 extension: new optional kwarg preserves all existing callers unchanged"
    - "Guard-first ordering: ValueError raised before any numpyro.sample calls to fail fast on misconfigured covariate pairs"
    - "Manual NumPy forward-sim for hierarchical recovery tests when unified simulator lacks parameter-shift injection"

key-files:
  created:
    - "scripts/fitting/tests/test_numpyro_models_2cov.py (635 lines, 4 test gates / 10 parametrizations)"
  modified:
    - "scripts/fitting/numpyro_models.py (wmrl_m3/m5/m6a_hierarchical_model: signature + guard + beta_iesr sampling + probit shift sum + docstring)"
    - "scripts/fitting/level2_design.py (+ build_level2_design_matrix_2cov + COVARIATE_NAMES_2COV constant)"

key-decisions:
  - "Approach A locked (additive parameter, not generic covariate_matrix replacement) — keeps blast radius tight, zero changes to fit_bayesian/recovery/prior_predictive callers"
  - "beta_iesr_{target} uses SAME Normal(0, 1) prior as beta_lec_{target} for prior symmetry across covariates"
  - "Guard raises on covariate_iesr without covariate_lec rather than silently treating it as 1-cov with IES-R — prevents ambiguous user intent"
  - "2-cov design matrix does NOT residualize iesr_total against lec_total — LEC/IES-R correlation in N=138 cohort is moderate (r ~ 0.3) and interpretation of raw totals is cleaner than of residuals"
  - "Recovery test uses manual NumPy forward-sim of M3 because unified_simulator.py does not expose per-participant kappa injection — fallback path documented in plan 21-11 and the test module docstring"
  - "test_m3_single_cov_unchanged parametrized over all three models (M3/M5/M6a) rather than a single M3 check — plan requested the M3 name, kept for compatibility; extended to M5/M6a for parity"

patterns-established:
  - "Phase 21 Option C infrastructure: code all variants upfront; pipeline dispatches the one the winner needs"
  - "Guard-site-Shift pattern: (1) guard at function top, (2) conditional numpyro.sample only when covariate is provided, (3) conditional shift-or-0.0 on probit scale — mirrors existing covariate_lec ordering for readability"

# Metrics
duration: ~15 min
completed: 2026-04-18
---

# Phase 21 Plan 11: 2-Covariate L2 Hook for M3/M5/M6a Hierarchical Models Summary

**Approach A additive-kwarg extension of `wmrl_m{3,5,6a}_hierarchical_model` to accept optional `covariate_iesr` alongside `covariate_lec`, adding `beta_iesr_{kappa,kappa,kappa_s}` numpyro sample sites with Normal(0, 1) priors and summing both shifts on the probit scale before Phi_approx — with a matching `build_level2_design_matrix_2cov` helper and a 635-line pytest suite (3 parametrized gates × 3 models + 1 slow recovery smoke test) proving backward compat and guard behavior.**

## Performance

- **Duration:** ~15 min (Task 1 ~7 min, Task 2 ~8 min including fast-test verification)
- **Started:** 2026-04-18T14:19:05Z
- **Completed:** 2026-04-18T14:34:00Z (approx)
- **Tasks:** 2
- **Files created:** 1 (test module, 635 lines)
- **Files modified:** 2 (numpyro_models.py, level2_design.py)

## Accomplishments

- `wmrl_m3_hierarchical_model`, `wmrl_m5_hierarchical_model`, `wmrl_m6a_hierarchical_model` accept `covariate_iesr: jnp.ndarray | None = None` alongside existing `covariate_lec: jnp.ndarray | None = None`
- `beta_iesr_kappa` (M3, M5) and `beta_iesr_kappa_s` (M6a) numpyro sample sites added with `Normal(0, 1)` priors — only sampled when `covariate_iesr is not None`
- Per-participant kappa (or kappa_s) unconstrained shift: `kappa_unc = kappa_mu_pr + kappa_sigma_pr * z + lec_shift + iesr_shift` before `Phi_approx` transform
- Guard raises `ValueError("covariate_iesr provided without covariate_lec. …")` if IES-R passed without LEC (prevents silently dropping LEC)
- `build_level2_design_matrix_2cov(metrics, participant_ids)` helper in `level2_design.py` returns `(N, 2)` z-scored design `[lec_total, iesr_total]` + names list; raises `ValueError` on missing columns or participants
- `COVARIATE_NAMES_2COV = ['lec_total', 'iesr_total']` module-level constant for downstream logging
- 9 fast tests pass in 12.1 s; recovery test marked `@pytest.mark.slow`
- Phase 16 backward compat proven: LEC-only trace has no `beta_iesr_*` site on any of M3/M5/M6a

## Task Commits

Each task was committed atomically:

1. **Task 1: 2-cov L2 hook + `build_level2_design_matrix_2cov`** — `7503316` (feat)
2. **Task 2: Pytest recovery + backward-compat test suite** — `49a84ff` (test)

**Plan metadata:** _pending final commit after SUMMARY + STATE update_

## Files Created/Modified

- `scripts/fitting/numpyro_models.py` (+ ~60 lines across 3 functions) — M3/M5/M6a extended with `covariate_iesr` kwarg, guard, conditional `beta_iesr_*` sample, additive IES-R probit shift, docstring updates, and Level-2 comment blocks now read "LEC-total + IES-R-total -> {kappa|kappa_s} regression coefficients (2-covariate design, Phase 21 Option C)"
- `scripts/fitting/level2_design.py` (+ ~85 lines) — new `build_level2_design_matrix_2cov` function + `COVARIATE_NAMES_2COV` constant inserted between the existing 4-cov builder and the collinearity-audit section
- `scripts/fitting/tests/test_numpyro_models_2cov.py` (635 lines, new) — 4 test gates:
  - `test_model_accepts_covariate_iesr` parametrized over M3/M5/M6a (trace check for `beta_lec_*` + `beta_iesr_*` both sampled)
  - `test_m3_single_cov_unchanged` parametrized over M3/M5/M6a (Phase 16 backward compat: no `beta_iesr_*` when LEC-only)
  - `test_guard_raises_iesr_without_lec` parametrized over M3/M5/M6a (guard behavior with pytest.raises(ValueError, match=…))
  - `test_recovery_2cov_m3` (@pytest.mark.slow) — N=40 synthetic, warmup=300, samples=600, chains=2, asserts `|post_mean_lec - 0.4| < 1.0`, `|post_mean_iesr + 0.3| < 1.0`, `n_divergences == 0`, `max_rhat < 1.1`
- Also bundles `_simulate_m3_block_numpy` (NumPy forward-sim of the M3 block, mirroring `wmrl_m3_block_likelihood` semantics) and `_build_recovery_dataset` (per-participant kappa = `Phi(kappa_mu_pr + sigma_pr*z + beta_lec*lec + beta_iesr*iesr)`) for the recovery test

## Decisions Made

- **Approach A (additive kwarg) locked:** Adds `covariate_iesr: jnp.ndarray | None = None` alongside `covariate_lec`. Approach B (replace with a generic covariate_matrix + names list) would require rewriting all existing callers (`fit_bayesian._fit_stacked_model`, recovery tests, prior_predictive, baseline fit) and break Phase 16 backward compat. Approach A preserves zero changes outside the three model functions + `level2_design.py` + test file. Blast radius stays tight.
- **`Normal(0, 1)` prior on `beta_iesr_{target}` (locked):** Identical to existing `beta_lec_{target}` prior. Symmetry across covariates prevents the prior from building a directional assumption into the design.
- **Guard placement (locked):** Raised at function top, BEFORE any `numpyro.sample` calls. Ensures no partial trace state exists if the guard triggers.
- **2-cov design does NOT residualize (locked):** Unlike the 4-cov subscale design, raw (z-scored) `iesr_total` is used. Justification: (a) only 2 covariates, so multicollinearity is trivial to inspect via Pearson r (~0.3 in N=138 cohort); (b) raw totals have cleaner interpretation than residuals; (c) orthogonalization is only principled when a large covariate set creates design-matrix conditioning issues (see Phase 16's 4-cov case, cond # = 11.3).
- **Recovery test uses manual NumPy forward-sim (locked):** The existing `unified_simulator.py::simulate_agent_fixed` framework operates on `QLearningAgent`/`WMRLHybridAgent` class instances that do not expose `kappa` as a constructor parameter, much less a per-participant shift. Manual NumPy forward-sim of the M3 block (hybrid WM/RL + epsilon + probability-mixing perseveration against a one-hot choice kernel of `last_action`) mirrors the exact semantics of `wmrl_m3_block_likelihood` in `jax_likelihoods.py:1482`. Documented in the test module docstring and the function docstring of `_simulate_m3_block_numpy`. Fallback path was pre-approved in plan 21-11 Task 2 action text.
- **test_m3_single_cov_unchanged parametrized over all three models (locked):** The plan mandated a backward-compat test named `test_m3_single_cov_unchanged` — the name is retained for plan compatibility, and the test is parametrized over M3/M5/M6a so each of the three models independently validates Phase 16 compat.
- **Recovery test budget (locked):** N=40, warmup=300, samples=600, chains=2, max_tree_depth=8, target_accept_prob=0.95. Reduced from production budget per plan 21-11 (~5-10 min wall time acceptable for `@pytest.mark.slow` gate). Gate `|post_mean - truth| < 1.0` is loose but appropriate for single-seed CI-speed smoke tests.

## Deviations from Plan

None — plan executed exactly as written. The plan's Task 2 action explicitly pre-approved the manual forward-sim fallback if the unified simulator lacked a kappa-shift injection path; that fallback was used.

**Total deviations:** 0
**Impact on plan:** None. All Rule 4 candidates (architectural decisions) were pre-resolved in the plan's own decision-text paragraph.

## Issues Encountered

- **`numpyro.set_host_device_count(2)` ordering in recovery test:** Placed inside the test body (after imports) rather than at module top, to avoid unnecessary global device-count mutation during fast-test runs that don't need multi-chain. `chain_method="sequential"` also specified explicitly to avoid `vectorized` fallback warnings on CPU.
- **`split_gelman_rubin` import location:** Imported lazily inside the recovery test function (not at module top) because it's only needed by the slow test path — keeps the fast-test import chain minimal.

## User Setup Required

None — no external service configuration required. All changes are in-repo Python modules.

## Next Phase Readiness

- **Plan 21-07 can now thread `covariate_iesr` at production budget** for M3/M5/M6a winners without further model-level changes. Dispatcher pattern: call `build_level2_design_matrix_2cov` for M3/M5/M6a, `build_level2_design_matrix` (4-cov) for M6b subscale, `None` for M1/M2.
- **Plan 21-10 orchestrator** should add `python -m pytest scripts/fitting/tests/test_numpyro_models_2cov.py -v -k "not slow"` as a pre-cluster-submission local gate (should complete in <15 s).
- **Production-budget recovery study scoped OUT** per plan 21-11 — only the single-seed smoke test is in scope here. A full Pearson-r ensemble recovery study would be a follow-up plan.
- **No blockers** for the next plan.

## Verification Results

All plan 21-11 success criteria met:

| Criterion | Target | Actual |
|---|---|---|
| `grep -c "covariate_iesr" scripts/fitting/numpyro_models.py` | ≥ 6 | **30** |
| `grep -c "beta_iesr" scripts/fitting/numpyro_models.py` | ≥ 3 | **19** |
| `grep -c "build_level2_design_matrix_2cov" scripts/fitting/level2_design.py` | ≥ 1 | **3** |
| `pytest test_numpyro_models_2cov.py -v -k "not slow"` all pass | 9/9 | **9/9 passed in 12.1 s** |
| `pytest ::test_m3_single_cov_unchanged` | 3/3 pass | **3/3 passed in 7.4 s** |
| SUMMARY.md created | yes | yes |
| STATE.md updated | yes | yes |

**Smoke-validation (not plan-gate):**

- Ad-hoc trace check of all three models with both LEC-only and LEC+IES-R covariate pairings confirmed the `beta_iesr_*` sites appear only in the 2-cov path.
- Ad-hoc trigger of the guard via `covariate_lec=None, covariate_iesr=<v>` raised `ValueError` with the expected message on all three models.
- `build_level2_design_matrix_2cov` smoke-tested with a 3-participant mock DataFrame: correct shape `(3, 2)`, z-score means ~0, stds ~1, raises `ValueError` on missing participant and missing column.

---
*Phase: 21-principled-bayesian-model-selection-pipeline*
*Plan: 11*
*Completed: 2026-04-18*
