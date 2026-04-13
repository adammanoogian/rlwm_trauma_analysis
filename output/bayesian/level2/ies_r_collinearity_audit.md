# IES-R Subscale Collinearity Audit (L2-02)

**N participants (complete IES-R + LEC data):** 160

---

## Raw Subscale Correlations

Pearson *r* between IES-R subscales (before residualization):

| | Intrusion | Avoidance | Hyperarousal |
|---|---|---|---|
| **Intrusion** | 1.000 | 0.753 | 0.821 |
| **Avoidance** | 0.753 | 1.000 | 0.750 |
| **Hyperarousal** | 0.821 | 0.750 | 1.000 |

---

## IES-R Subscale Sum Structure

**Subscales sum exactly to IES-R total:** `True`

Because `ies_total = ies_intrusion + ies_avoidance + ies_hyperarousal` exactly in this dataset, after Gram-Schmidt residualization all three subscale residuals satisfy `intr_resid + avd_resid + hyp_resid = 0`. The three-residual sub-matrix is rank-2 with condition number ~2.4e15.

---

## Condition Numbers

- **Raw subscale matrix** [intrusion, avoidance, hyperarousal]: **7.31**
- **3-residual matrix** [intr_resid, avd_resid, hyp_resid]: **~2.4e15** (rank-deficient; hyperarousal residual = -(intr_resid + avd_resid))
- **Full 4-column design** [lec_total, iesr_total, intr_resid, avd_resid]: **11.32**
- **Target:** < 30

---

## Contextual Correlations

- **LEC-5 total events vs. IES-R total:** *r* = 0.132

---

## ROADMAP Deviation 1: 5 Predictors Reduced to 4 Due to Linear Dependence

The Phase 16 specification implied 5 predictors: lec_total + iesr_total + 3 subscale residuals. Because the three subscale residuals are linearly dependent (their sum is exactly zero), only 2 can enter the design. The hyperarousal residual is dropped; intrusion and avoidance are retained as they map to distinct theoretical symptom clusters (re-experiencing vs. effortful avoidance).

**Final predictor set (4 predictors):**

1. `lec_total` — LEC-5 total events (`less_total_events`)
2. `iesr_total` — IES-R total score (`ies_total`)
3. `iesr_intr_resid` — IES-R intrusion, residualized vs. IES-R total
4. `iesr_avd_resid` — IES-R avoidance, residualized vs. IES-R total

---

## ROADMAP Deviation 2: LEC-5 Subcategory Columns Unavailable

The Phase 16 ROADMAP specified 6+ predictors including LEC-5 physical/sexual/accident subcategories. These columns are **not available** in the current data pipeline:

- `output/summary_participant_metrics.csv`: only `less_total_events` and `less_personal_events`.
- `scripts/utils/scoring_functions.py::score_less()`: computes only totals, no subcategory taxonomy.

---

## Verdict

**PASS** — Full 4-column design matrix condition number 11.32 is below the target of 30.

The Gram-Schmidt residualization produces a well-conditioned design matrix.
The 4-predictor design is approved for Level-2 regression.
