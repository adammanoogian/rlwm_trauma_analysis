---
phase: 13-infrastructure-repair-hierarchical-scaffolding
plan: 02
subsystem: docs
tags: [K-parameterization, collins-lab, numpyro, hierarchical, wm-rl, phi-approx]

# Dependency graph
requires:
  - phase: 13-infrastructure-repair-hierarchical-scaffolding
    provides: 13-RESEARCH.md with Senta 2025 K bounds (HIGH confidence)
provides:
  - "docs/K_PARAMETERIZATION.md: standalone reference for K in [2, 6] convention"
  - "Non-centered Phi_approx transform formula for hierarchical K"
  - "Historical Collins-lab K table (2012, 2014, 2021, 2025)"
  - "parameterization_version v4.0-K[2,6]-phiapprox string"
  - "BIC rejection rationale citing Senta 2025 p. 22"
affects:
  - 13-03 (INFRA numpyro scaffolding - uses K transform)
  - 14 (K refit - adopts [2, 6] bounds for MLE refit)
  - 15 (M3 hierarchical POC - uses transform formula directly)
  - 16 (choice-only family extension - same transform template)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "K non-centered transform: 2.0 + 4.0 * Phi_approx(mu_K_pr + sigma_K_pr * z_K_i)"
    - "hBayesDM Phi_approx convention for all bounded [0,1] and scaled parameters"

key-files:
  created:
    - docs/K_PARAMETERIZATION.md
  modified: []

key-decisions:
  - "K lower bound = 2, not 1 (matches Senta 2025; K<2 non-identifiable due to ns=2 min set size)"
  - "Upper bound = 6 (task max ns=6; K>6 structurally indistinguishable)"
  - "parameterization_version string enforces runtime separation of v3.0 MLE and v4.0 Bayesian fits"
  - "BIC retained in CSVs for back-compat but WAIC/LOO is the v4.0 primary criterion"

patterns-established:
  - "K transform pattern: lower + range * Phi_approx(mu_pr + sigma_pr * z) — replicated for all bounded parameters"
  - "Phi_approx = jax.scipy.stats.norm.cdf (not sigmoid) matching hBayesDM Stan convention"

# Metrics
duration: 2min
completed: 2026-04-12
---

# Phase 13 Plan 02: K Parameterization Reference Summary

**K bounded to [2, 6] continuous via non-centered Phi_approx transform, matching Senta, Bishop, Collins (2025) PLOS Comp Biol 21(9):e1012872; standalone reference for all v4.0 hierarchical models**

## Performance

- **Duration:** 2 min
- **Started:** 2026-04-12T08:44:41Z
- **Completed:** 2026-04-12T08:46:38Z
- **Tasks:** 1
- **Files modified:** 1

## Accomplishments

- Wrote `docs/K_PARAMETERIZATION.md` as a self-contained K parameterization reference
- Documented the non-centered transform `K_i = 2.0 + 4.0 * Phi_approx(mu_K_pr + sigma_K_pr * z_K_i)` with NumPyro/JAX code
- Provided scientific rationale for lower bound = 2 (identifiability given ns=2 minimum set size, not just convention)
- Compiled historical Collins-lab K table covering 2012, 2014, 2021, 2025

## Task Commits

1. **Task 1: Write docs/K_PARAMETERIZATION.md** - `c21497a` (docs)

**Plan metadata:** (follows in final commit)

## Files Created/Modified

- `docs/K_PARAMETERIZATION.md` - K ∈ [2, 6] parameterization reference with transform formula, historical table, BIC rationale, and full citations

## Decisions Made

- **K bounds match Senta 2025 exactly** — Senta, Bishop, Collins (2025) is the project's reference paper; using the same [2, 6] bounds removes any need to justify deviation in the manuscript methods.
- **Lower bound = 2 is an identifiability requirement, not just convention** — K < 2 is geometrically confounded with rho at ns=2 (the task minimum set size). This is documented explicitly in the reference.
- **parameterization_version column** enforces runtime separation between v3.0 MLE fits (K in [1, 7]) and v4.0 fits (K in [2, 6]).

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `docs/K_PARAMETERIZATION.md` is complete and cited from `.planning/phases/13-infrastructure-repair-hierarchical-scaffolding/13-RESEARCH.md`.
- Phase 13, Plan 03 (numpyro scaffolding) can proceed — it will use the transform formula documented here.
- Phase 14 (K refit) has the bound change documented and cited — the MLE refit must adopt [2, 6] per the `parameterization_version` contract.
- No blockers or concerns for downstream plans.

---
*Phase: 13-infrastructure-repair-hierarchical-scaffolding*
*Completed: 2026-04-12*
