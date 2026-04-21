---
phase: 21-principled-bayesian-model-selection-pipeline
plan: 02
subsystem: model-selection

tags: [rfx-bms, pxp, bor, variational-bayes, dirichlet, stephan-2009, rigoux-2014, mfit-port]

# Dependency graph
requires: []
provides:
  - "scripts/fitting/bms.py — standalone RFX-BMS + PXP module"
  - "Public rfx_bms(log_evidence, alpha0=1.0, max_iter=1000, tol=1e-4, n_xp_samples=1_000_000, seed=42) -> {alpha, r, xp, bor, pxp}"
  - "Unit tests covering uniform/dominant/heterogeneous regimes + input validation + PXP identity"
affects:
  - "21-05: LOO+stacking+BMS orchestrator imports rfx_bms"
  - "Phase 17: model-comparison reporting (optional XP/PXP augmentation)"

# Tech tracking
tech-stack:
  added:
    - "scipy.special.psi (digamma) — variational expected log-frequency"
    - "scipy.special.gammaln — Dirichlet normalising constant"
    - "scipy.special.logsumexp — numerically stable evidence marginalisation"
  patterns:
    - "Pure-NumPy port of MATLAB reference (mfit/bms.m) with strict 2-D input contract"
    - "No import-time side effects in infrastructure modules"
    - "Variational KL expressed in closed form to avoid Monte-Carlo noise in BOR"

key-files:
  created:
    - "scripts/fitting/bms.py (335 lines)"
    - "scripts/fitting/tests/test_bms.py (158 lines, 5 tests)"
  modified: []

key-decisions:
  - "Null-model free energy uses FIXED uniform r = 1/K (not Dirichlet posterior alpha_null) — matches Rigoux 2014 eq. A1 under the null"
  - "Monte-Carlo exceedance-probability sampling with 1,000,000 draws (~8 ms wall time) rather than closed-form for K > 2"
  - "Module strictly validates log_evidence.ndim == 2 and np.isfinite — raises ValueError with diagnostic message"

patterns-established:
  - "Three-layer BMS API: _vb_dirichlet_update (E-step loop), _vb_free_energy (ELBO), _exceedance_probability (MC sampler), _bor (null-vs-alt ratio), rfx_bms (public assembler)"
  - "Tests assert XP and PXP sum to 1.0 within 1e-10 as an algebraic invariant on every call"

# Metrics
duration: 10min
completed: 2026-04-18
---

# Phase 21 Plan 02: RFX-BMS + PXP Infrastructure Summary

**NumPy port of mfit/bms.m providing random-effects Bayesian Model Selection (Stephan 2009) with Bayesian Omnibus Risk and protected exceedance probability (Rigoux 2014), ready for import by the Phase 21.5 LOO+stacking+BMS orchestrator.**

## Performance

- **Duration:** 10 min
- **Started:** 2026-04-18T14:18:02Z
- **Completed:** 2026-04-18T14:28:18Z
- **Tasks:** 2
- **Files modified:** 2 (both new)

## Accomplishments

- `rfx_bms(log_evidence, ...)` public API returning `{alpha, r, xp, bor, pxp}` dict
- VB E-step converges in ≤1000 iterations on all tested regimes (uniform, dominant, heterogeneous)
- Exceedance probability estimator with 1 M Dirichlet samples — XP sums to 1.0 to 1e-10
- BOR computed via null-vs-alternative free-energy ratio using closed-form Dirichlet KL
- PXP = (1 - BOR) * XP + BOR/K formula verified to 1e-12 numerical precision against direct reconstruction
- 5 unit tests, all passing (1.89 s wall time)

## Task Commits

Each task was committed atomically:

1. **Task 1: rfx_bms() + helpers** — `0cef164` (feat)
2. **Task 1 follow-up: KL + null-model free-energy bug fix** — `10ca205` (fix, auto-deviation)
3. **Task 2: Unit tests (5 tests)** — `9c0f5b1` (feat)

**Plan metadata:** _pending final commit after SUMMARY + STATE update_

## Files Created/Modified

- `scripts/fitting/bms.py` — 335-line standalone module. Public `rfx_bms` + 4 private helpers (`_vb_dirichlet_update`, `_vb_free_energy`, `_exceedance_probability`, `_bor`). Imports only NumPy and scipy.special.
- `scripts/fitting/tests/test_bms.py` — 158-line test module with 5 pytest tests.

## Decisions Made

- **Null-model free energy formulation (locked, deviates from plan spec):** The plan described F0 as a Dirichlet-posterior free energy with `alpha_null = alpha0 + n/K` per component. This yields F1 == F0 exactly for uniform log-evidence (BOR = 0.5), which contradicts the published "null strongly supported" behaviour in Rigoux 2014. We instead use `F0 = sum_n logsumexp(log_evidence[n]) - N * log(K)`, which is the log-likelihood of the data under a fixed uniform categorical (r = 1/K delta posterior, no KL term). This recovers BOR ~ 1 on uniform data as expected. See `_bor` docstring for full rationale. **Locked as the canonical Phase 21 implementation.**
- **Dirichlet-sampling XP (not closed-form):** For K ≤ 2 a closed-form Beta-CDF exists, but for K ≥ 3 sampling is the standard approach. We use `n_xp_samples = 1_000_000` by default — gives standard error `~ 5e-4` on a probability near 0.25 (sufficient for the 0.02 atol in tests).
- **Strict 2-D input contract:** `log_evidence.ndim != 2` raises ValueError with diagnostic shape/ndim. No silent reshape — the caller is responsible for matching participant-summed LOO log-likelihoods to the `(n_subjects, n_models)` layout.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Inverted signs on Dirichlet KL `gammaln` terms**
- **Found during:** Task 2 (dominant / heterogeneous tests reported BOR = 1.0 when BOR should approach 0)
- **Issue:** The initial `_vb_free_energy` body subtracted `gammaln(sum(alpha))` and added `gammaln(sum(alpha0))`, and added `Σ gammaln(alpha)` / subtracted `Σ gammaln(alpha0)`. Standard KL(Dir(α) || Dir(α0)) has the opposite signs. Result: the ELBO increased when the Dirichlet posterior moved **away** from the prior rather than toward higher data likelihood, inverting BOR in every non-uniform regime.
- **Fix:** Rewrote the KL as the documented closed form `gammaln(sum α) - Σ gammaln(α) - gammaln(sum α0) + Σ gammaln(α0) + Σ(α-α0)(ψ(α) - ψ(sum α))` and subtract it from the expected log-likelihood term.
- **Files modified:** `scripts/fitting/bms.py` (_vb_free_energy docstring + body)
- **Verification:** All three non-trivial BOR tests (uniform, dominant, heterogeneous) now match Rigoux 2014's qualitative predictions.
- **Committed in:** `10ca205`

**2. [Rule 1 - Bug] Null-model free-energy construction gave BOR = 0.5 on uniform data**
- **Found during:** Task 2 (uniform test reported BOR = 0.49 when plan expected BOR > 0.7)
- **Issue:** Per the plan, F0 was evaluated at a Dirichlet posterior `alpha_null = alpha0 + n_subjects / K` using the same `_vb_free_energy` as F1. But in the uniform-log-evidence regime the heterogeneous-model posterior ALSO converges to `alpha = alpha0 + n/K` (because every participant has uniform responsibilities), so F1 == F0 by construction and BOR pins at 0.5 regardless of data. This loses the "null strongly supported" signal the BOR is designed to detect.
- **Fix:** Replaced F0 with the fixed-r form `F0 = sum_n logsumexp(lme_n) - N*log(K)`, which evaluates the data likelihood under a delta-posterior at `r = 1/K` (no KL penalty because r is fixed, not inferred). This matches Rigoux 2014 eq. A1 verbatim.
- **Files modified:** `scripts/fitting/bms.py` (_bor docstring + body; removed the now-unused `alpha0_vec.sum() + n_subjects/n_models` construction)
- **Verification:** Uniform -> BOR = 0.98 (null supported), dominant -> BOR < 1e-20, heterogeneous -> BOR < 1e-8 — all match Rigoux 2014.
- **Committed in:** `10ca205`

---

**Total deviations:** 2 auto-fixed (both Rule 1 - Bug)
**Impact on plan:** Both fixes essential for correctness; the alternative would have produced BORs disagreeing with the reference mfit/bms.m behaviour. The plan's F0 spec has been updated via the `_bor` docstring (cross-referenced to Rigoux 2014 eq. A1) so downstream plans can reuse the corrected formulation. No scope creep.

## Issues Encountered

None beyond the two bugs documented above. The VB E-step, XP sampler, input validation, and PXP formula all worked first-attempt.

## User Setup Required

None — module is pure NumPy + scipy.special; both are already in the project environment.

## Next Phase Readiness

- **Ready for import by Phase 21.5:** `from scripts.fitting.bms import rfx_bms` succeeds with no side effects (verified).
- **Downstream contract (for 21-05 orchestrator):** Caller must pre-assemble `log_evidence` of shape `(n_participants, n_models)` from `arviz.loo(idata, pointwise=True)` by summing `loo_i` over trials per participant. The module will raise ValueError on any shape mismatch or non-finite entry, so LOO-failure rows must be dropped upstream.
- **No downstream blockers** — VB iteration has hard cap of 1000 with 1e-4 tolerance (converges in <50 iters on all tested regimes).

---
*Phase: 21-principled-bayesian-model-selection-pipeline*
*Completed: 2026-04-18*
