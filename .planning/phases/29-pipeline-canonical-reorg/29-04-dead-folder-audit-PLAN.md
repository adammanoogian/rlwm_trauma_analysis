---
phase: 29-pipeline-canonical-reorg
plan: 04
type: execute
wave: 2
depends_on: [29-01]
files_modified:
  - scripts/analysis/                    (deleted or moved to scripts/legacy/analysis/)
  - scripts/results/                     (deleted or moved to scripts/legacy/results/)
  - scripts/simulations/                 (partially deleted; test-relevant bits preserved OR moved to scripts/legacy/simulations/)
  - scripts/statistical_analyses/        (deleted or moved to scripts/legacy/statistical_analyses/)
  - scripts/visualization/               (partially deleted; paper-referenced figure scripts salvaged; rest moved to scripts/legacy/visualization/)
  - scripts/legacy/                      (new directory — holds files with unclear live dependencies or retained-for-historical-reference)
  - scripts/legacy/README.md             (new — audit record: what was salvaged vs. dropped per folder)
  - validation/test_unified_simulator.py (import updated if simulator files moved to legacy/ OR if extracted to utils/)
  - tests/test_wmrl_exploration.py       (import updated similarly)
  - scripts/03_model_prefitting/09_generate_synthetic_data.py  (import updated if scripts/simulations/generate_data.py moves)
  - scripts/03_model_prefitting/10_run_parameter_sweep.py      (import updated if scripts/simulations/parameter_sweep.py moves)
  - manuscript/paper.tex                 (line 244 reference updated: `scripts/analysis/trauma_scale_distributions.py` path)
autonomous: true

must_haves:
  truths:
    - "scripts/ top level after this plan contains ONLY: 01_data_preprocessing/, 02_behav_analyses/, 03_model_prefitting/, 04_model_fitting/, 05_post_fitting_checks/, 06_fit_analyses/, utils/, fitting/, optionally legacy/ and _maintenance/ (SC#1, SC#3)"
    - "Every file that had a LIVE dependency in the active pipeline (validation/, tests/, cluster/, manuscript/) was either salvaged into canonical location OR retained at scripts/legacy/<folder>/ with importer updated"
    - "Every genuinely-dead file is out of the active tree (either deleted or under legacy/)"
    - "scripts/legacy/README.md records every per-folder decision with evidence (grep hit counts)"
  artifacts:
    - path: "scripts/legacy/README.md"
      provides: "per-folder audit record — what was salvaged vs. dropped"
      min_lines: 40
      contains: "scripts/analysis/"
    - path: "scripts/legacy/simulations/unified_simulator.py"
      provides: "preserved simulator — tests/ and validation/ still import it (unless extracted into utils/)"
      note: "only exists if unified_simulator is kept in legacy rather than fully absorbed into utils/"
  key_links:
    - from: "validation/test_unified_simulator.py"
      to: "scripts/legacy/simulations/unified_simulator.py (or extracted to utils)"
      via: "updated import path"
      pattern: "from scripts\\.(legacy\\.)?simulations\\.unified_simulator"
---

<objective>
Audit the five sibling folders Phase 28 left untouched (`scripts/analysis/`, `scripts/results/`, `scripts/simulations/`, `scripts/statistical_analyses/`, `scripts/visualization/`) and clean them out of the active tree. Per-folder protocol: (1) grep the entire repo for live references EXCLUDING `.planning/`; (2) if zero live hits → delete the folder (git history preserves content); (3) if hits exist → salvage the used files into canonical locations (01–06 stage dirs or `utils/`) OR move the folder to `scripts/legacy/` and update importers. Outcome: `scripts/` top level matches the canonical target listed in SC#1.

Purpose: Purge legacy-cluster cruft left over from pre-Phase-28 layout so future maintainers don't have to distinguish dead vs. live folders at a glance.

Output: `scripts/legacy/README.md` audit record + cleaned `scripts/` top level.
</objective>

<execution_context>
@C:\Users\aman0087\.claude\get-shit-done\workflows\execute-plan.md
@C:\Users\aman0087\.claude\get-shit-done\templates\summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/phases/29-pipeline-canonical-reorg/29-CONTEXT.md
@.planning/phases/29-pipeline-canonical-reorg/29-01-SUMMARY.md
@.planning/phases/29-pipeline-canonical-reorg/29-03-SUMMARY.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Per-folder live-reference audit (grep protocol) + write scripts/legacy/README.md draft</name>
  <files>
    - scripts/legacy/README.md (new)
  </files>
  <action>
    1. `mkdir -p scripts/legacy`
    2. Run the audit protocol for each of the 5 folders. For each file inside each folder, produce a live-reference count using:
       ```
       FOLDER=analysis  # iterate over {analysis, results, simulations, statistical_analyses, visualization}
       for f in scripts/$FOLDER/*.py; do
         name=$(basename "$f" .py)
         count=$(grep -rn \
           "from scripts\.$FOLDER\.$name\|import scripts\.$FOLDER\.$name\|scripts/$FOLDER/$name\.py" \
           . --include="*.py" --include="*.sh" --include="*.slurm" --include="*.md" --include="*.qmd" --include="*.tex" \
           --exclude-dir=.planning --exclude-dir=.git --exclude-dir=scripts/$FOLDER \
           | grep -v "^scripts/legacy/" \
           | wc -l)
         echo "$FOLDER/$name.py: $count live references"
       done
       ```
       (Exclusion list: `.planning/` historical docs; the folder itself; any already-moved `scripts/legacy/` content.)
    3. Tabulate the evidence per folder. For each folder categorize files into:
       - **SALVAGE**: referenced by active pipeline (validation/, tests/, cluster/, manuscript/) → file needs a new home in canonical 01–06 layout or `utils/`
       - **LEGACY-ARCHIVE**: referenced only by `.planning/` historical docs or by other files in the same dead folder → move the folder to `scripts/legacy/<folder>/`
       - **DELETE**: zero references anywhere → git rm (history preserved)
    4. Write `scripts/legacy/README.md` with one H2 section per folder:
       ```markdown
       # scripts/legacy — Archive Audit (Phase 29-04)
       
       Legacy/dead code moved out of the active pipeline during Phase 29 canonical reorganization.
       Contents listed here with the evidence that informed each decision.
       
       ## analysis/
       Audit date: 2026-04-XX. Live-reference count per file:
       | File | Live refs | Decision |
       |------|-----------|----------|
       | trauma_scale_distributions.py | N (referenced by manuscript/paper.tex line 244) | SALVAGE → 02_behav_analyses/ (or kept as-is if manuscript reference is purely textual) |
       | analyze_base_models.py | 0 | DELETE |
       | ... | ... | ... |
       
       ## results/
       ...
       ```
       Reference the pre-gathered evidence in 29-CONTEXT.md §2 (the per-folder known-reference notes).
    5. Commit this audit file FIRST as a snapshot before any deletions land:
       ```
       docs(29-04): scripts/legacy/README.md — per-folder dead-code audit evidence
       ```
       This gives the next tasks a stable reference and makes the subsequent destructive commits auditable.
  </action>
  <verify>
    - `test -f scripts/legacy/README.md`
    - `grep -c "^## " scripts/legacy/README.md` returns exactly 5 (one section per audited folder)
    - `grep -c "Decision" scripts/legacy/README.md` ≥ 1 per section (decision column populated)
    - `git log -1 --stat` shows the docs commit
  </verify>
  <done>Evidence-based audit table committed; every file in every dead-candidate folder has a documented decision.</done>
</task>

<task type="auto">
  <name>Task 2: Execute decisions — salvage LIVE files + move LEGACY-ARCHIVE folders + delete DELETE files</name>
  <files>
    - scripts/analysis/ (partial or full move to legacy/)
    - scripts/results/ (move to legacy/ or delete)
    - scripts/simulations/ (partial move — unified_simulator likely stays under legacy/ because validation/tests still import it)
    - scripts/statistical_analyses/ (move to legacy/ or delete)
    - scripts/visualization/ (partial — salvage any figure scripts referenced by manuscript/paper.qmd figure-generation pipeline)
    - scripts/03_model_prefitting/09_generate_synthetic_data.py (import updated)
    - scripts/03_model_prefitting/10_run_parameter_sweep.py (import updated)
    - validation/test_unified_simulator.py (import updated if simulator path changes)
    - tests/test_wmrl_exploration.py (import updated similarly)
    - manuscript/paper.tex (line 244 updated if `scripts/analysis/trauma_scale_distributions.py` moves)
  </files>
  <action>
    1. For each folder, execute the decision recorded in 29-04-SUMMARY/legacy/README.md:
       
       **`scripts/analysis/` (9 files):**
       - Per 29-CONTEXT.md: mostly dead, EXCEPT `trauma_scale_distributions.py` is referenced by `manuscript/paper.tex` line 244 (figure-caption attribution — "Figure produced by scripts/analysis/trauma_scale_distributions.py").
       - The sibling `scripts/analysis/plotting_utils.py` duplicates `scripts/utils/plotting.py` (handled by 29-03 — this is the deduplication follow-up).
       - Default action: `git mv scripts/analysis scripts/legacy/analysis` (whole-folder archive). Then update `manuscript/paper.tex` line 244 from `scripts/analysis/trauma_scale_distributions.py` → `scripts/legacy/analysis/trauma_scale_distributions.py` (OR rewrite the prose to reference the new canonical path if you choose to salvage just that one file into `02_behav_analyses/` — judgment call based on what the audit found).
       
       **`scripts/results/` (5 files):**
       - Per 29-CONTEXT.md: no external grep hits — fully dead.
       - Action: `git mv scripts/results scripts/legacy/results` (archive-friendly; avoids destructive `git rm`).
       
       **`scripts/simulations/` (5 files; known NOT fully dead):**
       - Files: `generate_data.py`, `parameter_sweep.py`, `unified_simulator.py`, `visualize_parameter_sweeps.py`, `README.md`.
       - Live referrers: `validation/test_unified_simulator.py` (line 24), `tests/test_wmrl_exploration.py` (line 16), `tests/legacy/examples/*.py` (several), `scripts/03_model_prefitting/09_generate_synthetic_data.py` (line 48), `scripts/03_model_prefitting/10_run_parameter_sweep.py` (line 57).
       - Options:
         - (A) `git mv scripts/simulations scripts/legacy/simulations` and rewrite all 5+ importers to `from scripts.legacy.simulations.unified_simulator import ...`. Cheapest option.
         - (B) Extract `unified_simulator.py` into canonical location — but WHERE? It's a Gym-based single-agent simulator, NOT the JAX-posterior-sample simulator (`scripts/utils/ppc.py`). They're different abstractions. Moving it to `utils/` would conflate the two. Could go to `scripts/utils/simulators.py`? — but only if ≥ 2 live callers, and currently the callers are ALL tests + 2 stage-03 wrappers, so it's closer to a test fixture than a shared helper.
         - (C) Extract just `unified_simulator.py` into `src/rlwm/simulators/` (sibling to `src/rlwm/envs/`, `src/rlwm/models/`), since it's a pure library module that tests import.
       - Recommended: Option (A) — move whole folder to `scripts/legacy/simulations/`, update 5+ importers. Smallest code churn, easiest to revert. If 29-07 closure guard starts flagging `scripts/legacy/simulations/` as "active pipeline in disguise," Phase 30 (or an ad-hoc plan) can extract it properly.
       - Update `scripts/03_model_prefitting/09_generate_synthetic_data.py` line 48: `from scripts.simulations.generate_data import main` → `from scripts.legacy.simulations.generate_data import main`.
       - Update `scripts/03_model_prefitting/10_run_parameter_sweep.py` line 57: `from scripts.simulations.parameter_sweep import main` → `from scripts.legacy.simulations.parameter_sweep import main`.
       - Update `validation/test_unified_simulator.py` line 24.
       - Update `tests/test_wmrl_exploration.py` line 16.
       - `tests/legacy/examples/` files are already legacy; rewrite their imports too (even though they're in a legacy dir) to keep them collectable without ImportError.
       
       **`scripts/statistical_analyses/` (1 file — `analyze_feedback_perseveration.py`):**
       - Per 29-CONTEXT.md: no external hits — fully dead.
       - Action: `git mv scripts/statistical_analyses scripts/legacy/statistical_analyses`.
       
       **`scripts/visualization/` (11 files):**
       - Per 29-CONTEXT.md: referenced only by `docs/02_pipeline_guide/PLOTTING_REFERENCE.md` and `.planning/` (historical). User flagged "SALVAGE what's used for paper figure generation."
       - Check `manuscript/paper.qmd` for any `{python}` or shell cell that invokes a file from `scripts/visualization/`:
         - `grep -n "scripts/visualization\|create_publication_figures\|create_supplementary\|create_modeling_figures\|plot_wmrl_forest\|plot_model_comparison\|plot_posterior_diagnostics\|plot_group_parameters\|quick_arviz_plots\|create_parameter_behavior_heatmap" manuscript/paper.qmd`
         - If any hit → SALVAGE that file into `scripts/06_fit_analyses/` (or wherever the figure target is most naturally produced). Update paper.qmd accordingly.
         - Expected: paper.qmd most likely uses pre-rendered PNGs that live in `manuscript/figures/` or `figures/`, NOT runtime Python calls into `scripts/visualization/`. In that case, all 11 files go to `scripts/legacy/visualization/`.
       - Default action: `git mv scripts/visualization scripts/legacy/visualization` unless specific files are salvaged.
       - Update `docs/02_pipeline_guide/PLOTTING_REFERENCE.md` references to point at new legacy paths OR annotate as historical.
    
    2. After all moves, grep for any remaining references to the old top-level folders:
       ```
       grep -rn \
         "scripts/analysis\|scripts/results\|scripts/simulations\|scripts/statistical_analyses\|scripts/visualization\|from scripts\.analysis\|from scripts\.results\|from scripts\.simulations\|from scripts\.statistical_analyses\|from scripts\.visualization" \
         scripts/ tests/ validation/ cluster/ manuscript/ docs/ src/ \
         --include="*.py" --include="*.sh" --include="*.slurm" --include="*.md" --include="*.qmd" --include="*.tex" \
         --exclude-dir=.planning --exclude-dir=scripts/legacy
       ```
       Expected output: ZERO. Any hit means an importer was missed — fix it.
    
    3. Atomic commit:
       ```
       refactor(29-04): dead-folder audit — {analysis,results,simulations,statistical_analyses,visualization} → scripts/legacy/
       
       Per per-file evidence recorded in scripts/legacy/README.md:
         - scripts/analysis/ (9 files) → scripts/legacy/analysis/. Decision: whole-folder archive; trauma_scale_distributions.py referenced only as figure-caption attribution in paper.tex (rewritten to legacy path).
         - scripts/results/ (5 files) → scripts/legacy/results/. Zero live refs.
         - scripts/simulations/ (5 files) → scripts/legacy/simulations/. Retained under legacy/ because validation/test_unified_simulator.py and tests/test_wmrl_exploration.py import from it; 5+ importers rewritten to from scripts.legacy.simulations.*.
         - scripts/statistical_analyses/ (1 file) → scripts/legacy/statistical_analyses/. Zero live refs.
         - scripts/visualization/ (11 files) → scripts/legacy/visualization/. No paper.qmd runtime references; salvages = 0 (pre-rendered PNGs used); docs/02_pipeline_guide/PLOTTING_REFERENCE.md annotated as historical.
       
       v4 closure still PASSES.
       ```
    
    4. Re-run `pytest scripts/fitting/tests/ tests/ validation/ --collect-only` and confirm no new ImportErrors.
  </action>
  <verify>
    - `test ! -d scripts/analysis && test ! -d scripts/results && test ! -d scripts/simulations && test ! -d scripts/statistical_analyses && test ! -d scripts/visualization`
    - `test -d scripts/legacy/analysis && test -d scripts/legacy/results && test -d scripts/legacy/simulations && test -d scripts/legacy/statistical_analyses && test -d scripts/legacy/visualization` (unless some were deleted outright — in which case the README.md documents which ones)
    - `grep -rn "scripts/analysis\|scripts/results\|scripts/simulations\|scripts/statistical_analyses\|scripts/visualization\|from scripts\.analysis\|from scripts\.results\|from scripts\.simulations\|from scripts\.statistical_analyses\|from scripts\.visualization" scripts/ tests/ validation/ cluster/ manuscript/ docs/ src/ --include="*.py" --include="*.sh" --include="*.slurm" --include="*.md" --include="*.qmd" --include="*.tex" --exclude-dir=.planning --exclude-dir=scripts/legacy` returns ZERO
    - `pytest scripts/fitting/tests/test_v4_closure.py -v` PASSES 3/3
    - `pytest validation/test_unified_simulator.py --collect-only` completes without ImportError (even if the test itself is deselected)
  </verify>
  <done>Five legacy folders gone from scripts/ top level; contents either archived under legacy/ with importers updated, or deleted; audit record in README.md matches on-disk reality; v4 closure still green.</done>
</task>

</tasks>

<verification>
```bash
# scripts/ top level canonical
ls -d scripts/*/ | sort | tee /tmp/scripts_top.txt
# expected: scripts/01_data_preprocessing/ scripts/02_behav_analyses/ scripts/03_model_prefitting/ scripts/04_model_fitting/ scripts/05_post_fitting_checks/ scripts/06_fit_analyses/ scripts/fitting/ scripts/legacy/ scripts/utils/
# (plus optionally scripts/_maintenance/ if 29-03 moved one-offs there)

# No dead folders remain at top level
for d in analysis results simulations statistical_analyses visualization; do
  test ! -d scripts/$d || { echo "STILL AT TOP: $d"; exit 1; }
done

# Audit record exists
test -f scripts/legacy/README.md

# Zero stale importers
grep -rn "from scripts\.analysis\|from scripts\.results\|from scripts\.simulations\|from scripts\.statistical_analyses\|from scripts\.visualization" \
  scripts/ tests/ validation/ cluster/ manuscript/ docs/ src/ \
  --include="*.py" --include="*.sh" --include="*.slurm" --include="*.md" --include="*.qmd" --include="*.tex" \
  --exclude-dir=.planning --exclude-dir=scripts/legacy \
  || echo "OK: zero stale imports"

# Tests still collect
pytest scripts/fitting/tests/ tests/ validation/ --collect-only
```
</verification>

<success_criteria>
1. `scripts/` top level contains ONLY: `01_*`, `02_*`, `03_*`, `04_*`, `05_*`, `06_*`, `utils/`, `fitting/` (library remnant), `legacy/`, and optionally `_maintenance/` (SC#1, SC#3).
2. `scripts/legacy/README.md` documents every dead-candidate file's fate with grep evidence (SC#3).
3. All previously-live importers (validation/, tests/, stage-03 wrappers, manuscript/) rewritten to new paths.
4. `grep -rn "from scripts\.(analysis|results|simulations|statistical_analyses|visualization)\." scripts/ tests/ validation/ cluster/ manuscript/ docs/ src/ --exclude-dir=scripts/legacy --exclude-dir=.planning` returns ZERO.
5. `pytest scripts/fitting/tests/test_v4_closure.py` passes 3/3 (SC#9).
6. `pytest --collect-only` succeeds across full test tree with no new ImportError.
</success_criteria>

<output>
After completion, create `.planning/phases/29-pipeline-canonical-reorg/29-04-SUMMARY.md` with:
- For each folder: decision (archive-to-legacy / partial-salvage / full-delete), file count, evidence snippet
- List of importers rewritten
- Commit SHA(s)
- Remaining `scripts/` top-level structure (`ls -d scripts/*/`)
</output>
