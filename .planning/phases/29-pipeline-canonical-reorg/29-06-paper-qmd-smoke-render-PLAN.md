---
phase: 29-pipeline-canonical-reorg
plan: 06
type: execute
wave: 4
depends_on: [29-01, 29-02, 29-04b]
files_modified:
  - manuscript/paper.qmd                     (inline-cell python imports + script-path prose references)
  - manuscript/paper.tex                     (line 244 path reference updated per 29-04)
  - manuscript/_output/paper.pdf             (regenerated via quarto render smoke check; not committed if gitignored)
autonomous: true

must_haves:
  truths:
    - "paper.qmd contains zero references to old paths (scripts/{data_processing,behavioral,simulations_recovery,post_mle,bayesian_pipeline}/ or scripts/{12,13,14}_*.py top-level)"  # SC#8, SC#10
    - "paper.qmd line 166 caption reference to docs/SCALES_AND_FITTING_AUDIT.md rewritten to docs/04_methods/README.md#scales-orthogonalization-and-audit (absorbed from 29-02)"
    - "quarto render manuscript/paper.qmd exits 0 (graceful-fallback cells absorb missing artifacts)"  # SC#8
    - "Every inline {python} code cell import in paper.qmd resolves from repo root (PYTHONPATH=.) given the new structure"
  artifacts:
    - path: "manuscript/paper.qmd"
      provides: "Bayesian-first paper with canonical script-path references"
      min_lines: 500
    - path: "manuscript/_output/paper.pdf"
      provides: "Smoke-rendered paper PDF proving all path refs resolve (may include placeholders for missing data)"
  key_links:
    - from: "manuscript/paper.qmd"
      to: "scripts/06_fit_analyses/compare_models.py"
      via: "prose and potentially inline cell"
      pattern: "scripts/06_fit_analyses/compare_models\\.py"
---

<objective>
Update every `scripts/` path reference in `manuscript/paper.qmd` and `manuscript/paper.tex` to the canonical 01–06 layout, and confirm `quarto render` still produces a PDF without path-not-found errors. Phase 28 introduced graceful-fallback `{python}` cells that absorb missing data artifacts — this plan ensures those cells still resolve their Python imports after the reorg.

Purpose: Phase 26 (manuscript finalization) is blocked until paper.qmd renders cleanly on the new tree. This plan unblocks that. Pre-conversation grep found 2 live hits in paper.qmd (lines 630, 650: `scripts/14_compare_models.py`) plus a likely long-line hit on line 171, plus paper.tex line 244.

Output: paper.qmd + paper.tex with canonical paths + successful quarto render.
</objective>

<execution_context>
@C:\Users\aman0087\.claude\get-shit-done\workflows\execute-plan.md
@C:\Users\aman0087\.claude\get-shit-done\templates\summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/phases/29-pipeline-canonical-reorg/29-CONTEXT.md
@.planning/phases/29-pipeline-canonical-reorg/29-01-SUMMARY.md
@manuscript/paper.qmd
@manuscript/paper.tex
</context>

<tasks>

<task type="auto">
  <name>Task 1: Grep + rewrite script-path references in paper.qmd and paper.tex</name>
  <files>
    - manuscript/paper.qmd
    - manuscript/paper.tex
  </files>
  <action>
    1. Grep paper.qmd for every `scripts/` string:
       ```
       grep -n "scripts/" manuscript/paper.qmd
       ```
       Expected at minimum: lines 630 (`scripts/14_compare_models.py`), 650 (`scripts/14_compare_models.py`), 171 (long-line hit with multiple refs), potentially others from Phase 28's Bayesian-first cells.
    2. Grep paper.qmd for `from scripts.` or `import scripts.` (inline `{python}` cells):
       ```
       grep -n "from scripts\.\|import scripts\." manuscript/paper.qmd
       ```
    3. For each hit, map to new canonical path using the table from 29-01:
       - `scripts/12_fit_mle.py` → `scripts/04_model_fitting/a_mle/12_fit_mle.py`
       - `scripts/13_fit_bayesian.py` → `scripts/04_model_fitting/b_bayesian/13_fit_bayesian.py`
       - `scripts/14_compare_models.py` → `scripts/06_fit_analyses/compare_models.py`
       - `scripts/bayesian_pipeline/21_*.py` → per 29-01 destinations
       - `scripts/data_processing/NN_*.py` → `scripts/01_data_preprocessing/`
       - `scripts/behavioral/NN_*.py` → `scripts/02_behav_analyses/`
       - `scripts/simulations_recovery/NN_*.py` → `scripts/03_model_prefitting/`
       - `scripts/post_mle/NN_*.py` → `scripts/06_fit_analyses/` (with renamed-without-number targets)
       - `scripts/analysis/trauma_scale_distributions.py` → `scripts/legacy/analysis/trauma_scale_distributions.py` (per 29-04) OR canonical salvage location if 29-04 salvaged it into `02_behav_analyses/`
    4. In paper.tex line 244, update the escaped-underscore path `scripts/analysis/trauma\_scale\_distributions.py` to match the 29-04 decision.
    5. Verify inline `{python}` cells that import from `scripts.*` modules resolve under the new paths:
       - If a cell does `from scripts.bayesian_pipeline.21_manuscript_tables import <func>` (impossible — `21_` can't start a Python module name), rewrite to `from scripts.06_fit_analyses.manuscript_tables import <func>` (still impossible if the leading `06_` blocks it). More likely: paper.qmd uses subprocess-style invocation (`subprocess.run(["python", "scripts/..."]`) for loading, which is path-based not import-based. In that case the rewrites in step 3 already handle it. Confirm by reading each `{python}` cell that touches scripts/ content.
    6. One known inline pattern from Phase 28 (graceful fallback): `Path("output/bayesian/manuscript/...").exists()` checks — those DON'T reference `scripts/` paths, so they're unchanged.
    7. **Absorbed from Plan 29-02** — rewrite paper.qmd line 166 caption cross-reference. Line 166 currently reads (in prose): `Scale distributions and pairwise correlations are summarized in @fig-scale-distributions and documented in \`docs/SCALES_AND_FITTING_AUDIT.md\`.` Rewrite the backtick-quoted path to `\`docs/04_methods/README.md#scales-orthogonalization-and-audit\``. This was originally 29-02's responsibility but was moved here to keep paper.qmd edits consolidated in a single plan (avoids Wave 1 parallel-write race between 29-01 and 29-02). Confirm via: `grep -n "SCALES_AND_FITTING_AUDIT\|scales-orthogonalization-and-audit" manuscript/paper.qmd` — expect the pre-edit match on line 166 and, post-edit, a match only on the new anchor form.
  </action>
  <verify>
    - `grep -n "scripts/data_processing\|scripts/behavioral\|scripts/simulations_recovery\|scripts/post_mle\|scripts/bayesian_pipeline\|scripts/12_fit_mle\|scripts/13_fit_bayesian\|scripts/14_compare_models\|scripts/fitting/fit_mle\|scripts/fitting/fit_bayesian" manuscript/paper.qmd manuscript/paper.tex` returns ZERO matches
    - `grep -n "scripts/" manuscript/paper.qmd | grep -v "scripts/0[1-6]_\|scripts/utils/\|scripts/legacy/\|scripts/fitting/"` returns only expected hits (fitting library remnants or comments)
    - `grep -n "docs/SCALES_AND_FITTING_AUDIT\.md" manuscript/paper.qmd` returns ZERO matches (line 166 absorbed-from-29-02 edit applied)
    - `grep -n "scales-orthogonalization-and-audit" manuscript/paper.qmd` returns at least 1 match (the rewritten caption)
  </verify>
  <done>Every paper.qmd and paper.tex script-path reference points at canonical 01–06 layout; zero stale old-path refs.</done>
</task>

<task type="auto">
  <name>Task 2: Smoke-render quarto + capture evidence + commit</name>
  <files>
    - manuscript/_output/paper.pdf (generated; may or may not be committed depending on gitignore)
  </files>
  <action>
    1. Run `quarto render manuscript/paper.qmd` from the repo root. Expect exit 0.
    2. Capture the tail of quarto's stdout (last ~40 lines) for the SUMMARY file. Specifically, confirm:
       - Zero `FileNotFoundError` exceptions for script paths
       - Zero `ModuleNotFoundError` in inline python cells
       - Warnings about missing data artifacts (output/bayesian/...) are ACCEPTABLE — graceful-fallback cells handle those. Path-not-found errors on `scripts/` paths are NOT acceptable.
    3. If exit ≠ 0 due to path-not-found error on `scripts/`: identify the unresolved path, fix it, re-render. Iterate until exit 0.
    4. If exit ≠ 0 due to missing data artifacts: confirm the graceful-fallback cell is present and actually catches the error; if not, the Phase 28 fallback cell missed a code path — add a minimal try/except to the cell. These are NOT blocking for this plan's success criterion (SC#8 says "without path-not-found errors"), but document any that need Phase 26 follow-up.
    5. Commit:
       ```
       docs(29-06): update paper.qmd + paper.tex script paths to canonical 01–06 layout
       
       - Rewrote N path references (line numbers preserved where possible)
       - quarto render manuscript/paper.qmd exits 0
       - Graceful-fallback cells still catch missing data artifacts (Phase 28 pattern preserved)
       - Generated manuscript/_output/paper.pdf (gitignored — smoke evidence only)
       ```
  </action>
  <verify>
    - `quarto render manuscript/paper.qmd` exits 0 (capture output)
    - `test -f manuscript/_output/paper.pdf` (or whatever output path is configured)
    - No `FileNotFoundError` or `ModuleNotFoundError` in render stderr related to `scripts/` paths
    - `git log -1 --stat` shows the commit touching paper.qmd / paper.tex
  </verify>
  <done>quarto render succeeds; paper.pdf regenerated; commit landed; Phase 26 unblocked for script-path concerns.</done>
</task>

</tasks>

<verification>
```bash
# Paths updated
grep -n "scripts/data_processing\|scripts/behavioral\|scripts/simulations_recovery\|scripts/post_mle\|scripts/bayesian_pipeline\|scripts/12_fit_mle\|scripts/13_fit_bayesian\|scripts/14_compare_models" manuscript/paper.qmd manuscript/paper.tex \
  || echo "OK: zero stale paths"

# Quarto smoke
quarto render manuscript/paper.qmd
test -f manuscript/_output/paper.pdf
```
</verification>

<success_criteria>
1. `manuscript/paper.qmd` and `manuscript/paper.tex` contain zero references to Phase-28 grouping paths (SC#8, SC#10).
2. `quarto render manuscript/paper.qmd` exits 0 (SC#8).
3. Generated PDF exists at `manuscript/_output/paper.pdf` (or configured output path).
4. Any missing data artifacts produce placeholder cells (Phase 28 pattern preserved), not render failures.
</success_criteria>

<output>
After completion, create `.planning/phases/29-pipeline-canonical-reorg/29-06-SUMMARY.md` with:
- Each script-path reference rewritten (line-level table)
- `quarto render` stdout tail (last 20–40 lines)
- Any cells flagged for Phase 26 follow-up (missing graceful fallback)
- Commit SHA
</output>
