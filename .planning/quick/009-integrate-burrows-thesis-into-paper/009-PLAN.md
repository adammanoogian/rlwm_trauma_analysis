---
phase: 009-integrate-burrows-thesis-into-paper
plan: 009
type: execute
wave: 1
depends_on: []
files_modified:
  - .planning/quick/009-integrate-burrows-thesis-into-paper/INTEGRATION_MAP.md
  - manuscript/paper.qmd  # only if Task 2 applies zero-interpretation edits
autonomous: true

must_haves:
  truths:
    - "User can audit every reusable thesis passage by exact line number without rereading the thesis"
    - "User can see which sections of paper.qmd are candidate insertion targets for each reusable passage"
    - "User is explicitly warned about places where thesis claims contradict current v4.0 analyses"
    - "No interpretive content from the thesis is rewritten, paraphrased, or auto-edited into paper.qmd"
    - "Any edit applied to paper.qmd is purely factual/descriptive and is logged in INTEGRATION_MAP.md"
  artifacts:
    - path: ".planning/quick/009-integrate-burrows-thesis-into-paper/INTEGRATION_MAP.md"
      provides: "Classified, line-cited catalog of reusable thesis content with divergence flags"
      contains: "Introduction, Methods, Results, Discussion, Divergence Flags, Summary Statistics sections"
  key_links:
    - from: "INTEGRATION_MAP.md rows"
      to: "Burrows_J_GDPA_Thesis.md line numbers"
      via: "explicit line range citations (e.g., L123-L145)"
      pattern: "L\\d+(-L\\d+)?"
    - from: "INTEGRATION_MAP.md rows"
      to: "manuscript/paper.qmd anchors"
      via: "section anchor + approximate line number (e.g., sec-methods ~L186)"
      pattern: "sec-\\w+"
---

<objective>
Integrate reusable intro/methods/results-scaffolding/discussion content from Jasmine Burrows's GDPA thesis into `manuscript/paper.qmd` by producing a line-cited classification catalog (INTEGRATION_MAP.md) and, only if demonstrably safe, applying zero-interpretation factual edits to paper.qmd.

Purpose: The thesis contains polished, citable prose (task description, DSM-5 trauma framing, scale descriptions, recruitment text) that the current paper.qmd can reuse verbatim or with minimal factual edits. The user will manually rework all interpretive passages; the executor's job is ONLY to identify and catalog safe reusable material.

Output: One MANDATORY catalog file (INTEGRATION_MAP.md) and optionally a small set of low-risk factual insertions into paper.qmd.
</objective>

<execution_context>
@C:\Users\aman0087\.claude\get-shit-done\workflows\execute-plan.md
</execution_context>

<context>
@.planning/STATE.md
@CLAUDE.md
@Burrows_J_GDPA_Thesis.md
@manuscript/paper.qmd
@docs/SCALES_AND_FITTING_AUDIT.md
</context>

<critical_rules_for_executor>

**Hard rules from the user — non-negotiable:**

1. **DO NOT change any interpretation in the intro or discussion.** The user will manually revise all interpretive/theoretical claims. Your job is ONLY to identify and lift sentences/paragraphs that are safe to reuse verbatim.
2. **Reuse, don't rewrite.** Where a thesis sentence or paragraph is factually correct for the current analysis (task description, RLWM theoretical framing, scale descriptions), reuse it as-is. Never paraphrase.
3. **Flag divergences, don't fix them.** Where thesis content conflicts with current v4.0 analyses (sample size, winning model, statistical approach), flag for the user — do not auto-update interpretation.

**Critical v4.0 divergence facts you MUST apply when classifying:**

| Dimension | Thesis (Burrows) | Current paper.qmd (v4.0) |
|---|---|---|
| Sample size | N=47 analysed (N=45 modelled) | N=138 canonical cohort (N=154 task-only) |
| Winning model | M3 (WM-RL+kappa) | M6b (dual perseveration, stick-breaking) |
| Models fit | M1, M2, M3 | M1, M2, M3, M5, M6a, M6b, M4 (LBA) |
| Trauma analysis | 3-group ANOVA (No Trauma / No Impact / Ongoing Impact) | Continuous LEC-5 + IES-R regression, hierarchical Bayesian L2 |
| Fitting method | MLE only | MLE + hierarchical Bayesian NumPyro NUTS |
| Trauma scale labels | LESS (adapted LEC-5) | lec_total + IES-R subscales |
| Group structure | 3 groups planned, 2 achieved | Continuous/regression-based; 2-group stratification retained |

**Reusability classification labels (use exactly these):**
- `verbatim` — Safe to copy without changes. Pure descriptive/background content.
- `factual-edit` — Reusable with simple factual substitutions (e.g., N=47 → N=138). No interpretation change.
- `flag-for-review` — Contains content that conflicts with v4.0 OR requires user judgment. Do NOT auto-edit.
- `do-not-integrate` — Outdated results or modelling text that contradicts v4.0. Note but do not transplant.

</critical_rules_for_executor>

<tasks>

<task type="auto">
  <name>Task 1: Produce INTEGRATION_MAP.md catalog of reusable thesis content</name>
  <files>.planning/quick/009-integrate-burrows-thesis-into-paper/INTEGRATION_MAP.md</files>
  <action>
Create the INTEGRATION_MAP.md catalog with the following procedure.

**Step 1 — Read both documents in full.**
- Read `Burrows_J_GDPA_Thesis.md` (1161 lines). Section breaks are bold text markers like `**Methods**`, `**Results**`, `**Discussion**`, not `#` headings. Build a mental map of where each section starts/ends by line number.
- Read `manuscript/paper.qmd`. Confirm section anchors:
  - `## Introduction {#sec-intro}` ~L150
  - `## Methods {#sec-methods}` ~L184
  - `### Participants {#sec-participants}` ~L186
  - `### Task {#sec-task}` ~L228
  - `### Computational Models {#sec-models}` ~L240
  - `### Model Fitting {#sec-fitting}` ~L496
  - `### Statistical Analysis {#sec-stats}` ~L588
  - `## Results {#sec-results}` ~L612
  - `## Discussion {#sec-discussion}` ~L1120

**Step 2 — Classify thesis paragraphs section by section.**
Walk through the thesis from top to bottom. For each paragraph (or cohesive multi-paragraph block), decide its classification per the `critical_rules_for_executor` labels. A paragraph is a catalog-worthy row if it is `verbatim`, `factual-edit`, or is interpretive-but-reusable-with-flag (`flag-for-review` — only if there's real lift potential).

Skip paragraphs that are `do-not-integrate` (purely outdated results/statistics) — note their existence in the Divergence Flags section instead.

**Step 3 — Write the catalog.** Structure exactly as below. Every row MUST include: thesis line range (e.g., `L123-L145`), classification label, target paper.qmd section anchor + approximate line, and a short excerpt (first sentence OR a ≤25-word excerpt) so the user can recognize the passage.

Use this exact file structure:

```markdown
# INTEGRATION_MAP.md — Burrows Thesis → paper.qmd

Generated: 2026-04-17
Thesis source: `Burrows_J_GDPA_Thesis.md` (1161 lines)
Target: `manuscript/paper.qmd` (v4.0)

## Classification Legend
- **verbatim** — safe to copy without changes
- **factual-edit** — reusable with simple factual substitutions only (e.g., sample size)
- **flag-for-review** — requires user judgment before transplant
- **do-not-integrate** — contradicts v4.0, logged in Divergence Flags only

---

## 1. Introduction — Reusable Passages

| Thesis lines | Classification | Target in paper.qmd | First sentence / excerpt | Notes |
|---|---|---|---|---|
| L45-L62 | verbatim | sec-intro (~L150) | "Post-traumatic stress disorder (PTSD) is…" | DSM-5 definition paragraph |
| … | … | … | … | … |

## 2. Methods — Reusable Passages

| Thesis lines | Classification | Target in paper.qmd | First sentence / excerpt | Notes |
|---|---|---|---|---|
| … | … | … | … | … |

**Methods subsection coverage checklist:**
- [ ] Participants / recruitment / ethics
- [ ] Power analysis
- [ ] Task description (procedure, stimuli, reward schedule)
- [ ] LEC-5 / LESS scale description
- [ ] IES-R scale description
- [ ] Other scales (PHQ-9, GAD-7, etc. — if present)
- [ ] Computational modelling framing (note: thesis only covers M1/M2/M3)
- [ ] Statistical analysis scaffolding

## 3. Results — Reusable Scaffolding

Likely minimal. Only include procedural/scaffolding sentences that describe the *analysis pipeline*, NOT specific statistics. Example: "Behavioural performance was assessed using a mixed-design ANOVA…" ← reusable. "F(2,44) = 3.21, p = .048…" ← do-not-integrate.

| Thesis lines | Classification | Target in paper.qmd | First sentence / excerpt | Notes |
|---|---|---|---|---|
| … | … | … | … | … |

## 4. Discussion — Reusable Passages

Focus on literature-grounded theoretical points that do NOT depend on this study's specific results. Anything that interprets results of this analysis goes to Divergence Flags, not here.

| Thesis lines | Classification | Target in paper.qmd | First sentence / excerpt | Notes |
|---|---|---|---|---|
| … | … | … | … | … |

## 5. Divergence Flags

Explicit list of places where thesis content contradicts or pre-dates current v4.0 analyses. Each flag is a user action item for manual revision — do NOT auto-edit these.

| Thesis lines | Thesis claim | v4.0 reality | Action for user |
|---|---|---|---|
| L??? | "M3 was the winning model" | M6b (dual perseveration) is winning model | Rewrite conclusion paragraph |
| L??? | "N=47 participants" | N=138 canonical cohort | Factual substitution throughout |
| L??? | "3-group ANOVA by trauma status" | Continuous LEC-5 + IES-R regression + hierarchical Bayesian L2 | Replace analytical framing |
| L??? | "MLE fitting only" | MLE + hierarchical Bayesian NumPyro | Expand methods; add Bayesian results |
| L??? | "Models M1, M2, M3" | M1, M2, M3, M5, M6a, M6b, M4 (LBA) | Expand model family description |
| … | … | … | … |

## 6. Summary Statistics

- Introduction: N paragraphs reusable (verbatim / factual-edit / flag): X / Y / Z, total ~W words
- Methods: X / Y / Z, total ~W words
- Results: X / Y / Z, total ~W words
- Discussion: X / Y / Z, total ~W words
- Divergence flags raised: N
- **Total reusable word count:** ~W words

## 7. Edits Applied to paper.qmd

{If Task 2 did not apply any edits, write: "No edits were applied — all candidate insertions were deemed to carry interpretive risk or were already present in paper.qmd. User to transplant manually using this map."}

{Otherwise list each edit:}

| paper.qmd line | Change | Thesis source lines | Rationale |
|---|---|---|---|
| L??? | Inserted recruitment/ethics paragraph | L??? | Pure factual/descriptive; N updated from 47 → 138 |
| … | … | … | … |
```

**Step 4 — Coverage and rigor requirements.**
- Every row MUST cite exact thesis line numbers. No handwaving like "the paragraph near the top."
- Every row MUST cite a paper.qmd section anchor (e.g., `sec-methods`) and approximate line.
- The Methods section table should be the longest — the thesis Methods is the highest-yield source.
- If you cannot find reusable content in a section, say so explicitly: "No reusable scaffolding identified in thesis Results beyond the sentences already captured in section 3 above."
- Cross-check the "Methods subsection coverage checklist" — every `[ ]` should be filled in (either with a row reference or "not present in thesis").
- The Divergence Flags table MUST include at minimum the five items in the Critical v4.0 divergence facts table above, with exact thesis line numbers located.

**Step 5 — Populate summary statistics.**
Count paragraphs and approximate word counts for each section. This gives the user a sense of integration scope before they sit down to transplant.
  </action>
  <verify>
Run these checks:
1. `ls -la .planning/quick/009-integrate-burrows-thesis-into-paper/INTEGRATION_MAP.md` — file exists, non-empty.
2. Grep for line citations: `grep -cE "L[0-9]+(-L[0-9]+)?" .planning/quick/009-integrate-burrows-thesis-into-paper/INTEGRATION_MAP.md` — expect ≥20 citations.
3. Grep for paper.qmd anchors: `grep -c "sec-" .planning/quick/009-integrate-burrows-thesis-into-paper/INTEGRATION_MAP.md` — expect ≥10 references.
4. Confirm all seven sections are present: `grep -cE "^## [1-7]\." .planning/quick/009-integrate-burrows-thesis-into-paper/INTEGRATION_MAP.md` — expect 7.
5. Confirm all five critical v4.0 divergences appear in the Divergence Flags table (winning model M3→M6b, sample size, model family, trauma analysis approach, fitting method).
6. Open the file and spot-check three rows by actually reading those thesis line ranges — the excerpts must match.
  </verify>
  <done>
INTEGRATION_MAP.md exists at the target path with all seven numbered sections populated. Every catalog row cites exact thesis line numbers and a paper.qmd section anchor. The Methods coverage checklist is fully resolved. The Divergence Flags section captures all five critical v4.0 discrepancies with exact thesis line numbers. The Summary Statistics section provides paragraph/word counts per section. Section 7 is either populated with logged edits or contains the explicit "no edits applied" statement.
  </done>
</task>

<task type="auto">
  <name>Task 2: Apply zero-interpretation factual edits to paper.qmd (conditional)</name>
  <files>manuscript/paper.qmd</files>
  <action>
**This task is CONDITIONAL. Only perform edits that meet ALL of the following gates:**

1. **Zero interpretive content.** The transplanted text must be purely descriptive/factual (task mechanics, recruitment procedure, scale instrument description, ethics statement). NO claims about what results mean, what was hypothesised, or what the findings imply.
2. **Already classified `verbatim` or `factual-edit` in INTEGRATION_MAP.md Task 1.** No edits based on ad-hoc judgment — only items from the catalog.
3. **Fills a genuine gap in paper.qmd.** Read the target paper.qmd section first. If the current paper already covers the content adequately, skip the edit (note this in Section 7 of INTEGRATION_MAP.md as "already present, no insertion needed").
4. **Factual edits are only the minimal substitution.** If reusing a paragraph with N=47 → N=138 substitution, change ONLY the numeric facts listed in the v4.0 divergence table. Do not touch surrounding sentences.

**Safe candidate categories (all must additionally pass the four gates above):**
- Participants subsection: recruitment procedure, ethics approval language, demographic collection instruments — with N updated.
- Task subsection: procedural description (trial structure, stimuli, reward schedule). Only transplant if the current paper.qmd Task section is thin.
- Scale descriptions: LEC-5 (LESS) instrument description, IES-R instrument description, scoring procedure — paragraph-level descriptions of the instruments themselves, NOT how they're used in analysis.

**Unsafe categories — DO NOT edit (route to user instead):**
- Anything in Introduction (framing is interpretive).
- Anything in Discussion (pure interpretation).
- Anything in Results (outdated statistics; v4.0 analyses differ).
- Computational modelling / model-fitting prose (thesis covers M1/M2/M3 only; paper.qmd has M1..M6b + M4).
- Statistical analysis prose (thesis uses 3-group ANOVA + MLE; paper.qmd uses continuous regression + hierarchical Bayesian).

**Procedure for each candidate edit:**
1. Read the exact target section in paper.qmd (±30 lines around the anchor).
2. Determine if the current content is absent/thin enough that insertion adds value.
3. If yes: use `Edit` tool to insert the thesis paragraph verbatim (applying only the factual substitutions listed in the v4.0 divergence table).
4. Log the edit in INTEGRATION_MAP.md Section 7 using the `Edit` tool on the map file:
   - paper.qmd line inserted at
   - Thesis source line range
   - Any factual substitutions made (e.g., "N=47 → N=138")
   - Rationale (why this gate-passed)

**Acceptable outcome — no edits applied.** If none of the candidates pass all four gates, that is a valid and preferred outcome. Update INTEGRATION_MAP.md Section 7 to read:

```
No edits were applied. All candidate insertions either carry interpretive risk, contain v4.0-discrepant facts beyond simple substitution, or are already adequately covered in paper.qmd. User to transplant manually using this map.
```

**Do NOT edit paper.qmd if:**
- You are uncertain whether a passage counts as interpretive.
- The thesis paragraph contains any claim about results or hypotheses.
- The factual substitution would require rewording surrounding sentences.
- The current paper.qmd section already has equivalent content.

When in doubt, skip the edit. The user's explicit preference is "risky edits skipped > auto-edits with interpretive creep."
  </action>
  <verify>
1. If edits were applied: `git diff manuscript/paper.qmd` — inspect every hunk. Each change must be a pure insertion (or minimal factual substitution) with no reworded sentences.
2. `grep -A 2 "^## 7\." .planning/quick/009-integrate-burrows-thesis-into-paper/INTEGRATION_MAP.md` — Section 7 is populated (either with edit log or with the explicit "no edits applied" paragraph).
3. Quarto sanity check (do NOT render, just syntax): `python -c "import pathlib; content = pathlib.Path('manuscript/paper.qmd').read_text(encoding='utf-8'); assert '## Introduction' in content and '## Methods' in content and '## Results' in content and '## Discussion' in content, 'Section anchors broken'"` — all four top-level section headers still present.
4. If edits were applied to paper.qmd: `git diff --stat manuscript/paper.qmd` — line count delta should be modest (typically <100 lines added, ideally <50).
  </verify>
  <done>
Either (a) paper.qmd has received zero-interpretation factual insertions, each logged in INTEGRATION_MAP.md Section 7 with thesis source lines + substitutions noted, and git diff shows pure insertions with no interpretive rewording; or (b) no edits were applied and Section 7 contains the explicit no-edits statement. In either case, paper.qmd's top-level section anchors (Introduction, Methods, Results, Discussion) remain intact.
  </done>
</task>

</tasks>

<verification>
Overall phase verification:
1. INTEGRATION_MAP.md exists, is complete (7 sections), and every row has line citations + paper.qmd anchors.
2. Divergence Flags section explicitly covers all five critical v4.0 discrepancies (sample size, winning model, model family, trauma analysis approach, fitting method).
3. No interpretive text from the thesis has been transplanted into paper.qmd Introduction, Discussion, or Results sections.
4. If paper.qmd was edited: `git diff` is inspectable and every hunk is a pure insertion or minimal factual substitution.
5. Section 7 of INTEGRATION_MAP.md accurately reflects what was (or wasn't) edited.
</verification>

<success_criteria>
- User opens INTEGRATION_MAP.md and can navigate to any reusable passage in the thesis by line number without guessing.
- User can see at a glance which paper.qmd section each passage belongs in.
- User has a complete list of divergence flags to manually address.
- User can trust that no interpretive content has been auto-edited into paper.qmd.
- Total executor wall time: ~30-45 minutes (one thesis read-through + one paper.qmd cross-reference + catalog writing).
</success_criteria>

<output>
After completion, create `.planning/quick/009-integrate-burrows-thesis-into-paper/009-SUMMARY.md` documenting:
- Count of reusable passages catalogued per section
- Whether Task 2 edits were applied (yes/no) and what was inserted if yes
- Any divergence flags that surprised you during classification
- Handoff note to user: "Next step is manual review of INTEGRATION_MAP.md and transplantation of catalogued passages."
</output>
