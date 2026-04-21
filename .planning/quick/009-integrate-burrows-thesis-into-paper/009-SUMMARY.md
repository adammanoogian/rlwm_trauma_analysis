---
plan: quick-009
subsystem: manuscript
tags: [thesis-integration, paper.qmd, content-catalog, reusable-prose]
key-files:
  created:
    - .planning/quick/009-integrate-burrows-thesis-into-paper/INTEGRATION_MAP.md
  modified: []
decisions:
  - "No edits applied to paper.qmd (Task 2): all candidates carry structural/contextual risk exceeding the interpretive-safety gate"
metrics:
  duration: ~30 min
  completed: "2026-04-17"
---

# Quick Task 009: Integrate Burrows Thesis Content into paper.qmd — Summary

**One-liner:** Produced a line-cited catalog of ~1,850 reusable words from Burrows GDPA thesis (N=47, M3-winner) against paper.qmd (v4.0, N=138, M6b-winner); no auto-edits applied; 9 divergence flags raised.

---

## Reusable Passages Catalogued (by section)

| Section | verbatim | factual-edit | flag-for-review | do-not-integrate | ~Words reusable |
|---|---|---|---|---|---|
| Introduction | 5 | 0 | 2 | 3 | ~700 |
| Methods | 15 | 7 | 0 | 3 | ~1,150 |
| Results | 0 | 0 | 1 | many | ~0 |
| Discussion | 1 | 0 | 7 | 0 | ~60 |
| **Total** | **21** | **7** | **10** | **>6** | **~1,850** |

---

## Task 2: Edits Applied to paper.qmd

**No edits were applied.**

Every candidate passage that passed the "zero interpretive content" and "fills a genuine gap" gates (principally: IES-R description L81, recruitment/ethics text L71–L73, group classification criteria L85–L87, task trial structure L91, reversal mechanics L105) nonetheless failed Gate 4: each requires structural editorial decisions about insertion point and surrounding context that cannot be reduced to a simple verbatim insertion. In particular:

- **IES-R description (L81)**: The passage naturally introduces the IES-R ≥24 cutoff used for group classification, but paper.qmd's current Participants section introduces groups before describing the IES-R instrument. Inserting without structural reordering would place the cutoff rationale after the groups are already defined.
- **Recruitment/ethics text (L71–L73)**: Verbatim-safe, but the existing Participants section already flows from cohort definition → scale names → group classification. Inserting recruitment text requires a paragraph-order decision (before or after cohort criteria).
- **Task description (L91–L105)**: Richer than paper.qmd's current sketch, but the 12-block vs. up to 21-block discrepancy and the need to integrate with existing task prose makes this a structural replacement decision, not an insertion.

The user's stated preference is "zero edits > risky edits." All candidates routed to Section 7 as "user to transplant manually using this map."

---

## Top 5–10 Highest-Value Reusable Passages

Ranked by immediate transplant value to paper.qmd (difficulty-adjusted):

1. **IES-R scale description — L81 (~150 words, verbatim)**
   Target: `sec-participants` (~L204). Currently paper.qmd says "Impact of Event Scale--Revised (IES-R; @weiss2007impact)" with no instrument description. The thesis paragraph describes the 22-item structure, 3 subscales, 5-point Likert rating, score range 0–88, ≥24 cutoff, internal consistency α≈.95, and Creamer et al. (2003) validity reference. This is the single highest-yield insertion.

2. **Recruitment and data anonymisation — L71 (~120 words, verbatim)**
   Target: `sec-participants` (~L186). Currently paper.qmd has no recruitment description. Thesis text covers SONA platform + social media, unique identifier system, no PII storage, data quality checks. Pure procedural; risk-free.

3. **Ethics approval statement — L73 (~70 words, verbatim)**
   Target: `sec-participants`. Currently absent from paper.qmd. Thesis supplies the exact ethics approval ID (ETH23-8006 -- G--35--2025), the governing codes (Australian Code for the Responsible Conduct of Research, National Statement on Ethical Conduct in Human Research), and consent/eligibility criteria.

4. **Task trial structure — L91 (~120 words, verbatim)**
   Target: `sec-task` (~L230). Thesis provides the complete trial-by-trial timing (500ms fixation, 2000ms response window, 500ms feedback) currently absent from paper.qmd. Only substitution needed: remove "As depicted in Figure 1" cross-reference or add a figure.

5. **Reversal mechanics — L105 (~80 words, verbatim)**
   Target: `sec-task` (~L236). Thesis describes the reversal threshold (12–18 consecutive correct responses), once-per-stimulus constraint, and counter-reset logic. paper.qmd states "rare reversals occurring after 12–18 consecutive correct responses" (L236) but does not describe the counter reset or the once-per-stimulus constraint. Direct transplant of the detail sentences.

6. **WM paragraph — L43 (~110 words, verbatim)**
   Target: `sec-intro` expansion (~L165). Thesis provides the WM background paragraph (capacity-limited system, set-size effects, Bays & Husain 2008, Collins & Frank 2012, trauma-WM disruption via HPA axis and cortisol). The current Introduction mentions WM only briefly. Strong background expansion.

7. **RL paragraph — L44 (~100 words, verbatim)**
   Target: `sec-intro` expansion (~L165). Thesis RL paragraph (prediction error signals, dopaminergic frontostriatal circuits, Collins et al. 2014, trauma-altered RL: heightened negative sensitivity, Pechtel & Pizzagalli 2011, Hiser et al. 2023). Adds mechanistic depth to the current intro.

8. **WM-RL interaction paragraph — L45 (~110 words, verbatim)**
   Target: `sec-intro` expansion. Thesis paragraph on the complementarity of WM and RL in dynamic learning contexts and the set-size-dependent system transition (Collins & Frank 2012, Collins 2018, Collins et al. 2017). Adds the "why both systems matter" framing that the current intro lacks.

9. **Online replication observation — L575 last two sentences (~60 words, verbatim)**
   Target: `sec-discussion` after set-size-effects confirmation. "Notably, these canonical set-size effects were observed in an online testing environment, whereas most prior RLWM studies have been conducted under controlled in-person laboratory conditions. This provides further support for the validity of the task implementation and suggests that the core WM and WM dynamics of the paradigm can be reliably captured in an online format." Strengthens the methods-validity discussion.

10. **DSM-5 trauma definition — L39 (~130 words, verbatim)**
    Target: `sec-intro` (~L152). The current intro opens with "Trauma exposure produces lasting changes in learning…" without defining trauma. The thesis first two sentences provide the DSM-5 definition and historical clinical context. Adds definitional grounding.

---

## Divergence Flags — User Action Items

The following thesis claims are in direct conflict with v4.0 and require user attention when manually reviewing the map.

| # | Thesis lines | Issue | User action |
|---|---|---|---|
| 1 | L31, L65, L67, L248 | N=47 analysed (N=45 modelled) throughout | Any transplanted passage with sample size needs N=138/154 substitution |
| 2 | L31, L362, L386, L514 | "M3 was the winning/best-fit model" | Do not transplant any Results or Discussion that names M3 as winner |
| 3 | L137, L363, L374–L379 | Only M1, M2, M3 fit | Thesis model description is a strict subset; paper.qmd coverage is already superior |
| 4 | L127–L131 | Statistical design: 2×3 mixed ANOVA, 3-group × load | v4.0 uses continuous regression + hierarchical Bayesian L2; do not transplant |
| 5 | L137 | "No hierarchical or group-level pooling applied" | v4.0 has full hierarchical Bayesian NUTS; do not transplant this characterisation |
| 6 | L63 | Power analysis for 2×3 ANOVA design, N=284 target | v4.0 has no equivalent power analysis; user may want to write a new one |
| 7 | L81 | IES-R ≥24 cutoff applied, but framed as 3-group classification | paper.qmd does not state the ≥24 cutoff explicitly (genuine gap); user should add it to sec-participants with 2-group framing |
| 8 | L119 | "All included participants completed at least 16 experimental blocks" | v4.0 uses ≥8-block threshold; do not transplant; sec-exclusions already correct |
| 9 | L568 | kappa × LEC-5 null in N=47 thesis (max |ρ|=0.20, p>.18) | v4.0: M3 kappa × LEC-5 p=0.0019 (FDR-BH survivor); M6b kappa_total × LEC-5 p=0.0028 uncorrected. Critical sign reversal — do not transplant any thesis kappa-trauma text |

---

## Surprising Divergences Noted During Classification

1. **Kappa × LEC-5 sign flip.** The thesis (L568) reports kappa is uncorrelated with trauma in N=47 (maximum |ρ|=0.20, p>.18 for all measures). v4.0 finds a statistically significant positive association (M3: FDR-BH survivor; M6b: p=0.0028 uncorrected). This is the opposite conclusion from the thesis. The N=138 vs N=47 difference likely explains it — the effect was underpowered in the thesis. This is the most important divergence for the manuscript narrative.

2. **IES-R ≥24 cutoff is absent from paper.qmd.** The thesis explicitly states the IES-R ≥24 classification threshold (L81, L85). The current paper.qmd Participants section (L206–L208) defines the two groups by name but never states the ≥24 decision rule. This is a gap the user should fill regardless of whether any thesis text is transplanted.

3. **LESS vs LEC-5 naming.** The thesis uses "LESS" (Life Experiences and Stressors Scale) as the instrument name throughout, noting it is a modified version of the LEC-5. paper.qmd uses "LEC-5" as the label. The actual questionnaire administered was the LESS adaptation. The user should decide whether to acknowledge the LESS adaptation or treat it as LEC-5 throughout (the pipeline column name is `less_total_events`).

4. **12 blocks vs up to 21 blocks.** The thesis describes 12 experimental blocks (L103); paper.qmd describes up to 21 main-task blocks (L238). This reflects a change between the thesis data collection period (shorter task) and the full v4.0 dataset. This discrepancy should be noted in the Methods if both cohorts are discussed.

---

## Handoff Note to User

**Next step:** Open `INTEGRATION_MAP.md` (at `.planning/quick/009-integrate-burrows-thesis-into-paper/INTEGRATION_MAP.md`) and use it as a navigation guide to transplant catalogued passages.

**Recommended order for manual transplantation:**

1. Add IES-R description (L81) to sec-participants (~L204) — most impactful single insertion; fills a genuine gap
2. Add IES-R ≥24 classification cutoff sentence to sec-participants (~L206) — not in the map as verbatim (needs 2-group framing) but the rule is clear from L85
3. Add recruitment + ethics text (L71–L73) to sec-participants (~L186) — verbatim block, placement before cohort criteria description
4. Expand Introduction with WM (L43), RL (L44), WM-RL interaction (L45), DSM-5 definition (L39), and epidemiology (L40) paragraphs — large verbatim additions
5. Strengthen sec-task with trial timing details (L91) and reversal mechanics (L105)
6. Consider adding the 2-sentence online-replication observation (L575) to sec-discussion

**Do not touch (v4.0 incompatible):** All Results statistics, all Discussion paragraphs naming M3 as winning model, all trauma-kappa null findings, all 3-group ANOVA framing, all power analysis text.
