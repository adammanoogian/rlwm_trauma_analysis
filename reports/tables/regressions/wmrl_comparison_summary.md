# WM-RL Regression Comparison: Full Sample vs Epsilon Cutoff

## Summary

| Condition | N | Exclusion Criteria |
|-----------|---|-------------------|
| No cutoff | 49 | None |
| Epsilon < 0.50 | 43 | 6 participants with epsilon > 0.50 |

Excluded participants (high random responding):
- anon_27556: epsilon=0.704 (also only 119 trials)
- 8932: epsilon=0.653
- anon_12026: epsilon=0.642
- anon_95434: epsilon=0.573
- anon_55472: epsilon=0.538
- anon_94162: epsilon=0.520

## Key Significant Findings

### alpha_neg ~ lec_personal_events (Negative Learning Rate × Personal Trauma Exposure)

| Condition | β | SE | t | p | r | R² |
|-----------|---|----|----|---|---|---|
| No cutoff (N=49) | -0.035 | 0.016 | -2.27 | **0.028*** | -0.315 | 0.099 |
| Epsilon < 0.50 (N=43) | -0.022 | 0.018 | -1.25 | 0.220 | -0.191 | 0.037 |

**Interpretation**: The significant negative association between negative learning rate and personal trauma exposure weakens substantially when excluding high-epsilon participants. This suggests the original effect may have been partially driven by participants with high random responding.

### rho ~ lec_personal_events (WM Weight × Personal Trauma Exposure)

| Condition | β | SE | t | p | r | R² |
|-----------|---|----|----|---|---|---|
| No cutoff (N=49) | 0.043 | 0.017 | 2.60 | **0.012*** | 0.355 | 0.126 |
| Epsilon < 0.50 (N=43) | 0.043 | 0.019 | 2.30 | **0.027*** | 0.338 | 0.114 |

**Interpretation**: The positive association between WM weight (rho) and personal trauma exposure remains significant even after excluding high-epsilon participants. This is the most robust finding.

### Marginally Significant Effects

| Parameter | Predictor | Full (p) | Filtered (p) | Direction |
|-----------|-----------|----------|--------------|-----------|
| alpha_pos | lec_personal | 0.061 | 0.164 | Weakened |
| phi | lec_personal | 0.066 | 0.113 | Weakened |
| wm_capacity | ies_avoidance | 0.131 | 0.072 | Strengthened |

## Conclusions

1. **Most robust finding**: Higher personal trauma exposure predicts higher WM weight (rho), regardless of performance filtering.

2. **Sensitive to outliers**: The negative association between alpha_neg and personal trauma was driven partly by high-epsilon participants.

3. **Recommendation**: Report both analyses. The epsilon-filtered sample provides more reliable parameter estimates, while noting that sample size is reduced.

---
Generated: 2026-01-27
