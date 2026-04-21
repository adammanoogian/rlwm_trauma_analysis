---
phase: 16-choice-only-family-extension-subscale-level-2
plan: 01
subsystem: bayesian-regression
tags: [level2, design-matrix, collinearity, gram-schmidt, ies-r, lec-5, orthogonalization, numpy]

# Dependency graph
requires:
  - phase: 15-m3-hierarchical-poc
    provides: wmrl_m3_hierarchical_model, run_inference_with_bump, prepare_stacked_participant_data
provides:
  - scripts/fitting/level2_design.py with build_level2_design_matrix (4-predictor design)
  - output/bayesian/level2/ies_r_collinearity_audit.md with PASS verdict (cond=11.3)
  - Confirmed: LEC-5 subcategories unavailable; 4-predictor design is final
  - Confirmed: ies_total = ies_intrusion + ies_avoidance + ies_hyperarousal (exact sum)
affects:
  - 16-02 through 16-07 (all Level-2 hierarchical model plans must use build_level2_design_matrix)
  - numpyro_models.py (L2 regression covariate set is now 4, not 5 or 6)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "4-predictor L2 design: lec_total, iesr_total, iesr_intr_resid, iesr_avd_resid"
    - "Gram-Schmidt residualization on mean-centred vectors before z-scoring"
    - "Hyperarousal residual dropped due to exact linear dependence (ies_total = sum of subscales)"

key-files:
  created:
    - scripts/fitting/level2_design.py
    - output/bayesian/level2/ies_r_collinearity_audit.md
  modified: []

key-decisions:
  - "4-predictor design locked: lec_total, iesr_total, iesr_intr_resid, iesr_avd_resid"
  - "Hyperarousal residual omitted: ies_total = ies_intrusion + ies_avoidance + ies_hyperarousal exactly; hyp_resid = -(intr_resid + avd_resid)"
  - "LEC-5 subcategory columns unavailable: only less_total_events and less_personal_events exist in data pipeline"
  - "Collinearity audit uses full 4-column design matrix condition number (not 3-residual sub-matrix)"
  - "COVARIATE_NAMES list in level2_design.py is the single source of truth for beta site naming in NumPyro models"

patterns-established:
  - "build_level2_design_matrix(metrics_df, participant_ids) -> (X, covariate_names): single call site for all L2 regression models"
  - "run_collinearity_audit + write_collinearity_report: audit-then-report pattern for design diagnostics"
  - "include_lec_subcategories=True raises ValueError: makes data gap explicit at call sites"

# Metrics
duration: 11min
completed: 2026-04-13
---

# Phase 16 Plan 01: Level-2 Design Matrix and Collinearity Audit Summary

**4-predictor Gram-Schmidt design matrix (lec_total, iesr_total, iesr_intr_resid, iesr_avd_resid) with condition number 11.3 (PASS); IES-R subscale sum constraint and LEC-5 subcategory gap both confirmed and documented**

## Performance

- **Duration:** 11 min
- **Started:** 2026-04-13T08:02:24Z
- **Completed:** 2026-04-13T08:14:05Z
- **Tasks:** 2
- **Files modified:** 2

## Accomplishments

- Created `scripts/fitting/level2_design.py` as the single source of truth for the Level-2 design matrix; `build_level2_design_matrix()` and `run_collinearity_audit()` are fully callable
- Discovered and corrected a mathematical necessity: `ies_total = ies_intrusion + ies_avoidance + ies_hyperarousal` exactly in this dataset, making the 3-subscale residual sub-matrix rank-deficient; the design uses 4 predictors (not 5)
- Generated `output/bayesian/level2/ies_r_collinearity_audit.md` with PASS verdict (full design condition number 11.3, well below target of 30)
- Confirmed LEC-5 physical/sexual/accident subcategory data is permanently unavailable in the current pipeline; 4-predictor design is the correct final design

## Task Commits

Each task was committed atomically:

1. **Task 1: LEC-5 data gap audit + level2_design.py** - `0c2a9d7` (feat)
2. **Task 2: Run collinearity audit and write report** - `95dde47` (feat)

**Plan metadata:** (docs commit follows)

## Files Created/Modified

- `scripts/fitting/level2_design.py` - Level-2 design matrix builder with `build_level2_design_matrix`, `run_collinearity_audit`, `write_collinearity_report`, and `__main__` block that generates the audit report
- `output/bayesian/level2/ies_r_collinearity_audit.md` - Collinearity audit report with raw correlations, condition numbers, PASS verdict, and both ROADMAP deviations documented

## Decisions Made

- **4-predictor design locked (not 5 or 6):** `lec_total`, `iesr_total`, `iesr_intr_resid`, `iesr_avd_resid`. Hyperarousal residual is omitted because `ies_total = ies_intrusion + ies_avoidance + ies_hyperarousal` exactly in the dataset; all three residuals sum to zero. Only 2 subscale residuals are linearly independent.
- **Intrusion + avoidance retained (not intrusion + hyperarousal or avoidance + hyperarousal):** These map to theoretically distinct trauma symptom clusters — re-experiencing (intrusion) vs. effortful avoidance (avoidance). Hyperarousal's unique variance is recoverable from the other two.
- **Condition number target applied to full 4-column design (not 3-residual sub-matrix):** The 3-residual sub-matrix is always rank-deficient in this dataset; the operationally relevant diagnostic is the full design including lec_total and iesr_total.
- **COVARIATE_NAMES constant in level2_design.py is the authoritative list:** All downstream NumPyro models must use this list for naming `beta_*` sites. The 4-predictor design implies each hierarchical model has 4 Level-2 regression coefficients per parameter with L2 regression applied.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Corrected design from 5 predictors to 4 due to IES-R subscale linear dependence**

- **Found during:** Task 2 (run collinearity audit)
- **Issue:** The plan specified 5 predictors including all 3 IES-R subscale residuals. After running the audit, `ies_total = ies_intrusion + ies_avoidance + ies_hyperarousal` exactly in the N=160 participant dataset. This makes all 3 subscale residuals linearly dependent (they sum to zero), producing a rank-deficient matrix with condition number ~2.4e15. Including `iesr_hyp_resid` in the design would make the Level-2 regression numerically unstable and the MCMC sampler would encounter identifiability issues.
- **Fix:** Dropped `iesr_hyp_resid` from `COVARIATE_NAMES` and `build_level2_design_matrix()`. Updated `run_collinearity_audit()` to report the full 4-column design condition number (11.3) as the operationally relevant diagnostic. Updated audit report and module docstring to explain the mathematical constraint.
- **Files modified:** `scripts/fitting/level2_design.py`
- **Verification:** Full design condition number = 11.3 (PASS), matrix rank = 4, no NaN values, column means ~0 and stds ~1.
- **Committed in:** `95dde47` (Task 2 commit)

---

**Documented but not auto-fixed (pre-existing data gap):**

**2. LEC-5 physical/sexual/accident subcategory columns unavailable**

- **Status:** Pre-existing data gap; documented as ROADMAP deviation (L2-04 resolution)
- **Finding:** Only `less_total_events` and `less_personal_events` exist in the pipeline. Raw item-level columns (`s1_item01`–`s1_item15`) are not mapped to a physical/sexual/accident taxonomy anywhere in the codebase.
- **Resolution:** The 4-predictor design does not include LEC-5 subcategories. `include_lec_subcategories=True` raises `ValueError` to make this explicit at call sites.
- **Documented in:** Module docstring, `build_level2_design_matrix` docstring, `ies_r_collinearity_audit.md`

---

**Total deviations:** 1 auto-fixed (Rule 1 - mathematical correctness), 1 pre-existing data gap documented
**Impact on plan:** The 4-predictor design is mathematically correct and well-conditioned. The subscale count reduction (5→4) propagates to all downstream Phase 16 plans: each hierarchical model will have 4 L2 regression coefficients per parameter (not 5 or 6). No plan architecture changes required.

## Issues Encountered

- Initial collinearity audit output showed residualized condition number ~2.4e15 (FAIL). Root cause: exact summation constraint in the IES-R scoring formula. Fixed by dropping the linearly dependent hyperarousal residual.
- Plan verification check expected `(154, 5)` shape; actual result is `(160, 4)`. The `154` was the N from MLE fits (only participants with sufficient task data); `160` is the correct count of participants with complete IES-R + LEC-5 data. The function correctly raises `RuntimeError` for participants with missing data.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `build_level2_design_matrix(metrics_df, participant_ids)` is ready for use in Plans 16-02 through 16-07 (hierarchical model extensions)
- The 4-predictor covariate set is confirmed: `COVARIATE_NAMES = ["lec_total", "iesr_total", "iesr_intr_resid", "iesr_avd_resid"]`
- Each hierarchical model in Phase 16 will have 4 `beta_*` Level-2 regression sites per parameter with L2 regression
- The hyperarousal residual exclusion is final and documented; no further audit needed

---
*Phase: 16-choice-only-family-extension-subscale-level-2*
*Completed: 2026-04-13*
