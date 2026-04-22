---
phase: 29-pipeline-canonical-reorg
plan: 04b
type: execute
wave: 3
depends_on: [29-03, 29-04]
files_modified:
  # ─── 02_behav_analyses: 05-08 → 01-04 ─────────────────────────────────────
  - scripts/02_behav_analyses/01_summarize_behavioral_data.py          (git mv from 05_*)
  - scripts/02_behav_analyses/02_visualize_task_performance.py         (git mv from 06_*)
  - scripts/02_behav_analyses/03_analyze_trauma_groups.py              (git mv from 07_*)
  - scripts/02_behav_analyses/04_run_statistical_analyses.py           (git mv from 08_*)
  # ─── 03_model_prefitting: 09-13 (minus deleted 09_run_ppc) → 01-05 ────────
  - scripts/03_model_prefitting/01_generate_synthetic_data.py          (git mv from 09_*)
  - scripts/03_model_prefitting/02_run_parameter_sweep.py              (git mv from 10_*)
  - scripts/03_model_prefitting/03_run_model_recovery.py               (git mv from 11_*)
  - scripts/03_model_prefitting/04_run_prior_predictive.py             (git mv from 12_*)
  - scripts/03_model_prefitting/05_run_bayesian_recovery.py            (git mv from 13_*)
  # ─── 04_model_fitting: drop stale globals; resolve entry/library collisions ─
  - scripts/04_model_fitting/a_mle/fit_mle.py                          (git mv from 12_fit_mle.py; thin CLI)
  - scripts/04_model_fitting/a_mle/_engine.py                          (git mv from fit_mle.py; library engine, underscore-private)
  - scripts/04_model_fitting/b_bayesian/fit_bayesian.py                (git mv from 13_fit_bayesian.py; thin CLI for single-model ad-hoc)
  - scripts/04_model_fitting/b_bayesian/fit_baseline.py                (git mv from 21_fit_baseline.py; pipeline entry — hierarchical baseline MCMC)
  - scripts/04_model_fitting/b_bayesian/_engine.py                     (git mv from fit_bayesian.py; library engine — note: OLD fit_bayesian.py is the 43KB library, not CLI)
  - scripts/04_model_fitting/c_level2/fit_with_l2.py                   (git mv from 21_fit_with_l2.py)
  # ─── 05_post_fitting_checks: descriptive → 01-03 ──────────────────────────
  - scripts/05_post_fitting_checks/01_baseline_audit.py                (git mv from baseline_audit.py)
  - scripts/05_post_fitting_checks/02_scale_audit.py                   (git mv from scale_audit.py)
  - scripts/05_post_fitting_checks/03_run_posterior_ppc.py             (git mv from run_posterior_ppc.py IF 29-03 created it; skip if not yet present)
  # ─── 06_fit_analyses: descriptive → 01-08 in paper-read order ─────────────
  - scripts/06_fit_analyses/01_compare_models.py                       (git mv from compare_models.py)
  - scripts/06_fit_analyses/02_compute_loo_stacking.py                 (git mv from compute_loo_stacking.py)
  - scripts/06_fit_analyses/03_model_averaging.py                      (git mv from model_averaging.py)
  - scripts/06_fit_analyses/04_analyze_mle_by_trauma.py                (git mv from analyze_mle_by_trauma.py)
  - scripts/06_fit_analyses/05_regress_parameters_on_scales.py         (git mv from regress_parameters_on_scales.py)
  - scripts/06_fit_analyses/06_analyze_winner_heterogeneity.py         (git mv from analyze_winner_heterogeneity.py)
  - scripts/06_fit_analyses/07_bayesian_level2_effects.py              (git mv from bayesian_level2_effects.py)
  - scripts/06_fit_analyses/08_manuscript_tables.py                    (git mv from manuscript_tables.py)
  # ─── Importer / reference updates across the repo ─────────────────────────
  - scripts/**/__init__.py                                             (updated re-exports if any)
  - scripts/04_model_fitting/a_mle/__init__.py                         (updated — now exports from _engine not fit_mle; notes renaming)
  - scripts/04_model_fitting/b_bayesian/__init__.py                    (updated — same pattern)
  - scripts/**/*.py                                                    (importer updates: sed-style find/replace for all renamed module paths)
  - src/rlwm/**                                                        (reference updates in docstrings / error messages)
  - tests/**                                                           (test-file path updates)
  - validation/**                                                      (closure-guard path updates)
  - scripts/fitting/tests/test_load_side_validation.py                 (re-map entries for renumbered files)
  - config.py                                                          (docstring :mod: refs if any name changed)
  - CLAUDE.md                                                          (Quick Reference path examples updated)
  - README.md                                                          (pipeline block updated)
  - docs/02_pipeline_guide/ANALYSIS_PIPELINE.md                        (pipeline steps updated)
  - manuscript/paper.qmd                                               (inline {python} cell imports — subject to 29-06 ownership of paper.qmd; coordinate)
autonomous: true

must_haves:
  truths:
    - "Every script filename inside scripts/0N_*/ uses either (a) a local 01-N intra-stage prefix that resets per stage, OR (b) no numeric prefix if the folder has parallel-alternative scripts with no execution order."  # SC#1 refinement
    - "No script filename contains a carry-over global number from pre-29-01 layout (i.e., grep 'scripts/0[1-6]_.*/[12][0-9]_' returns zero matches across all stage folders)."
    - "04_model_fitting/a_mle/ contains exactly fit_mle.py (CLI entry) and _engine.py (private library); no name collision."
    - "04_model_fitting/b_bayesian/ contains exactly fit_bayesian.py (ad-hoc CLI), fit_baseline.py (pipeline entry), and _engine.py (private library); no name collision."
    - "04_model_fitting/c_level2/ contains exactly fit_with_l2.py."
    - "Every importer in src/, scripts/, tests/, validation/, manuscript/ references the new filenames; grep for old paths (e.g., 'scripts/02_behav_analyses/05_summarize', 'scripts/04_model_fitting/a_mle/12_fit_mle', 'scripts/06_fit_analyses/compare_models') returns zero matches."
    - "validation/check_v4_closure.py exits 0 with new paths."  # SC#9
    - "pytest scripts/fitting/tests/ tests/ validation/ passes clean."  # SC#11
  artifacts:
    - path: "scripts/04_model_fitting/a_mle/fit_mle.py"
      provides: "canonical MLE CLI entry script — renamed from 12_fit_mle.py"
    - path: "scripts/04_model_fitting/a_mle/_engine.py"
      provides: "MLE library engine — renamed from fit_mle.py; underscore-private convention"
    - path: "scripts/04_model_fitting/b_bayesian/fit_baseline.py"
      provides: "Bayesian pipeline entry — renamed from 21_fit_baseline.py; used by cluster/04b_bayesian_*.slurm"
    - path: "scripts/06_fit_analyses/01_compare_models.py"
      provides: "first-step analysis in paper-read order — renamed from compare_models.py (was 14_compare_models.py)"
    - path: "scripts/06_fit_analyses/08_manuscript_tables.py"
      provides: "final-step manuscript rendering — renamed from manuscript_tables.py (was 21_9_manuscript_tables.py)"
  key_links:
    - from: "cluster/04a_mle_cpu.slurm"
      to: "scripts/04_model_fitting/a_mle/fit_mle.py"
      via: "python invocation in SLURM body"
      pattern: "04_model_fitting/a_mle/fit_mle\\.py"
    - from: "cluster/04b_bayesian_cpu.slurm"
      to: "scripts/04_model_fitting/b_bayesian/fit_baseline.py"
      via: "python invocation in SLURM body"
      pattern: "04_model_fitting/b_bayesian/fit_baseline\\.py"
    - from: "manuscript/paper.qmd"
      to: "scripts/06_fit_analyses/08_manuscript_tables.py"
      via: "Quarto {python} inline cell"
      pattern: "06_fit_analyses/08_manuscript_tables\\.py"
---

<objective>
Apply intra-stage numbering **Scheme D** across all six stage folders after Wave 2 completes. Scheme D rules:

1. **Stage folders (01–06) keep numeric prefixes** — they encode paper IMRaD order and are load-bearing.
2. **Intra-stage numbers reset per stage (start at 01 in each).** No carry-over global numbers (12_, 21_, etc.) survive inside stage folders.
3. **Use intra-stage numbers ONLY when execution order is load-bearing within the stage.** Strict-order stages (01, 02, 03) and paper-read-order stages (05, 06) get numbers; parallel-alternative folders (04/a, 04/b, 04/c) do NOT.
4. **In 04_model_fitting/{a,b,c}/, CLI entry scripts use canonical descriptive names** (`fit_mle.py`, `fit_baseline.py`, `fit_bayesian.py`, `fit_with_l2.py`); library/engine code uses underscore-private convention (`_engine.py`) to avoid collision with entry scripts when global-number prefixes drop.
5. **Model fanout is via CLI flag**, NEVER via per-model scripts. The 7 models (M1/M2/M3/M5/M6a/M6b/M4) are dispatched through `--model <name>` inside single entry scripts — this pattern already exists; do NOT create `fit_m1.py`, `fit_m2.py`, etc.

This plan is purely renumbering + rename + importer updates. Zero functional code changes. Every move is `git mv` to preserve history.

Runs in Wave 3 AFTER Wave 2 (29-03 utils consolidation + 29-04 dead-folder audit) because:
- 29-03 converts `scripts/03_model_prefitting/09_run_ppc.py` into `scripts/utils/ppc.py`, resolving the duplicate-`09_` name collision so renumbering sees a clean 5-file set in stage 03.
- 29-03 creates `scripts/05_post_fitting_checks/run_posterior_ppc.py`, which this plan then renumbers to `03_run_posterior_ppc.py`.
- 29-04 moves dead folders to `scripts/legacy/` — renumbering only touches surviving canonical files.

Runs BEFORE Wave 4 (29-05 cluster SLURM consolidation, 29-06 paper.qmd smoke render) so those plans target the final-final filenames, eliminating a second path-update pass.

Output: clean stage-local numbering everywhere; zero stale global prefixes; all importers updated; v4 closure guard still green.
</objective>

<execution_context>
@C:\Users\aman0087\.claude\get-shit-done\workflows\execute-plan.md
@C:\Users\aman0087\.claude\get-shit-done\templates\summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/phases/29-pipeline-canonical-reorg/29-CONTEXT.md
@.planning/phases/29-pipeline-canonical-reorg/29-01-SUMMARY.md
@.planning/phases/29-pipeline-canonical-reorg/29-03-SUMMARY.md
@.planning/phases/29-pipeline-canonical-reorg/29-04-SUMMARY.md
</context>

<tasks>

<task type="auto" id="1">
  <name>Task 1: Rename files in 02_behav_analyses (05-08 → 01-04)</name>
  <files>
    - scripts/02_behav_analyses/05_summarize_behavioral_data.py → 01_summarize_behavioral_data.py
    - scripts/02_behav_analyses/06_visualize_task_performance.py → 02_visualize_task_performance.py
    - scripts/02_behav_analyses/07_analyze_trauma_groups.py → 03_analyze_trauma_groups.py
    - scripts/02_behav_analyses/08_run_statistical_analyses.py → 04_run_statistical_analyses.py
  </files>
  <action>
    1. `git mv scripts/02_behav_analyses/05_summarize_behavioral_data.py scripts/02_behav_analyses/01_summarize_behavioral_data.py`
    2. Repeat for 06/07/08 → 02/03/04.
    3. Grep for old paths in repo: `grep -rn "02_behav_analyses/0[5-8]_\|behavioral/0[5-8]_" src/ scripts/ tests/ validation/ manuscript/ docs/ cluster/ CLAUDE.md README.md config.py` — update every hit.
    4. Also grep for the Python `Path` construction form: `grep -rn '"02_behav_analyses".*"0[5-8]_\|"0[5-8]_summarize\|"0[5-8]_visualize\|"0[5-8]_analyze_trauma\|"0[5-8]_run_statistical"' --include="*.py" .` — update.
    5. Verify: `python -c "import ast; ast.parse(open('scripts/02_behav_analyses/01_summarize_behavioral_data.py').read())"` for each renamed file (syntax still valid).
  </action>
  <verify>
    - `ls scripts/02_behav_analyses/` shows exactly 01_..., 02_..., 03_..., 04_... (no 05-08).
    - `grep -rn "02_behav_analyses/0[5-8]_" . --include="*.py" --include="*.md" --include="*.slurm" --include="*.sh"` returns zero matches.
    - `pytest tests/ -q` passes (does not touch behav scripts directly but confirms no side effects).
  </verify>
  <done>Atomic commit: `refactor(29-04b): rename 02_behav_analyses files 05-08 → 01-04 + update importers`</done>
</task>

<task type="auto" id="2">
  <name>Task 2: Rename files in 03_model_prefitting (09-13 → 01-05)</name>
  <files>
    - scripts/03_model_prefitting/09_generate_synthetic_data.py → 01_generate_synthetic_data.py
    - scripts/03_model_prefitting/10_run_parameter_sweep.py → 02_run_parameter_sweep.py
    - scripts/03_model_prefitting/11_run_model_recovery.py → 03_run_model_recovery.py
    - scripts/03_model_prefitting/12_run_prior_predictive.py → 04_run_prior_predictive.py
    - scripts/03_model_prefitting/13_run_bayesian_recovery.py → 05_run_bayesian_recovery.py
  </files>
  <action>
    1. Pre-flight check: `ls scripts/03_model_prefitting/09_run_ppc.py` — if this file still exists, ABORT and return checkpoint:human-needed (29-03 was supposed to delete it during utils/ppc.py extraction). Expect this file to be GONE by the time 29-04b runs.
    2. `git mv` each file in the list.
    3. Grep repo for old paths (forward-slash + Python-Path forms); update every hit in src/, scripts/, tests/, validation/, manuscript/, docs/, cluster/, CLAUDE.md, README.md, config.py.
    4. Special attention to `scripts/fitting/tests/test_load_side_validation.py` — if _ENUMERATED_FILES still has `"03_model_prefitting" / "09_generate_synthetic_data.py"` etc., update to `"01_generate_synthetic_data.py"` etc.
    5. Verify ast.parse on each renamed file.
  </action>
  <verify>
    - `ls scripts/03_model_prefitting/` shows exactly 01_..., 02_..., 03_..., 04_..., 05_... (no 09-13, no duplicate 09).
    - `grep -rn "03_model_prefitting/\(09\|1[0-3]\)_" . --include="*.py" --include="*.md" --include="*.slurm" --include="*.sh"` returns zero.
    - `pytest scripts/fitting/tests/test_load_side_validation.py -v` passes (if its enumeration was updated by 29-01/29-04, this plan's renumbering further updates it).
  </verify>
  <done>Atomic commit: `refactor(29-04b): rename 03_model_prefitting files 09-13 → 01-05 + update importers`</done>
</task>

<task type="auto" id="3">
  <name>Task 3: Resolve 04_model_fitting/{a_mle,b_bayesian,c_level2}/ collisions + drop global prefixes</name>
  <files>
    - scripts/04_model_fitting/a_mle/fit_mle.py → _engine.py           (OLD library, 130 KB / 3,157 lines → renamed to private)
    - scripts/04_model_fitting/a_mle/12_fit_mle.py → fit_mle.py        (NEW: thin CLI takes the canonical name)
    - scripts/04_model_fitting/b_bayesian/fit_bayesian.py → _engine.py  (OLD library, 43 KB / 1,173 lines → renamed to private)
    - scripts/04_model_fitting/b_bayesian/13_fit_bayesian.py → fit_bayesian.py  (NEW: thin CLI takes the canonical name)
    - scripts/04_model_fitting/b_bayesian/21_fit_baseline.py → fit_baseline.py  (drop 21_ global prefix; pipeline entry)
    - scripts/04_model_fitting/c_level2/21_fit_with_l2.py → fit_with_l2.py      (drop 21_ global prefix; no collision)
  </files>
  <action>
    1. **Two-step rename for a_mle/ to avoid collision:**
       - Step A: `git mv scripts/04_model_fitting/a_mle/fit_mle.py scripts/04_model_fitting/a_mle/_engine.py` (renames library out of the way first).
       - Step B: `git mv scripts/04_model_fitting/a_mle/12_fit_mle.py scripts/04_model_fitting/a_mle/fit_mle.py` (CLI takes canonical name).
    2. **Same two-step for b_bayesian/:**
       - Step A: `git mv scripts/04_model_fitting/b_bayesian/fit_bayesian.py scripts/04_model_fitting/b_bayesian/_engine.py`.
       - Step B: `git mv scripts/04_model_fitting/b_bayesian/13_fit_bayesian.py scripts/04_model_fitting/b_bayesian/fit_bayesian.py`.
    3. **Single rename for fit_baseline and fit_with_l2:**
       - `git mv scripts/04_model_fitting/b_bayesian/21_fit_baseline.py scripts/04_model_fitting/b_bayesian/fit_baseline.py`.
       - `git mv scripts/04_model_fitting/c_level2/21_fit_with_l2.py scripts/04_model_fitting/c_level2/fit_with_l2.py`.
    4. **Update the new CLI entry scripts** (`fit_mle.py`, `fit_bayesian.py`) so their internal imports now reference `from ._engine import main` (or whatever name was exported) instead of `from fit_mle import main`. Relative imports required due to digit-prefix `04_model_fitting/` package.
    5. **Update __init__.py** in a_mle/ and b_bayesian/ — export the CLI `main` function from `_engine` so external callers still work. Add a comment explaining the underscore-private convention.
    6. **Grep + update all importers:**
       - `grep -rn "04_model_fitting/a_mle/12_fit_mle\|04_model_fitting/a_mle/fit_mle\.py" . --include="*.py" --include="*.md" --include="*.slurm" --include="*.sh"` — DANGER: both old entry (`12_fit_mle.py`) AND old library (`fit_mle.py`) are referenced; disambiguate per-hit:
         - If the caller invoked it as a CLI (e.g., `python scripts/04_model_fitting/a_mle/12_fit_mle.py`) → update to `fit_mle.py`.
         - If the caller imported it as a library (e.g., `from scripts.fitting.fit_mle import compute_X`) → update to `_engine import compute_X`.
       - Repeat disambiguation for b_bayesian/.
       - Single-step updates for `21_fit_baseline.py` → `fit_baseline.py` and `21_fit_with_l2.py` → `fit_with_l2.py` (no ambiguity).
    7. **Python Path form:** `grep -rn '"12_fit_mle"\|"13_fit_bayesian"\|"21_fit_baseline"\|"21_fit_with_l2"' . --include="*.py"` — update test_load_side_validation.py enumeration.
    8. **Run the entry scripts in smoke mode to validate imports still resolve:**
       - `python scripts/04_model_fitting/a_mle/fit_mle.py --help` — must exit 0, print usage.
       - `python scripts/04_model_fitting/b_bayesian/fit_bayesian.py --help` — must exit 0.
       - `python scripts/04_model_fitting/b_bayesian/fit_baseline.py --help` — must exit 0.
       - `python scripts/04_model_fitting/c_level2/fit_with_l2.py --help` — must exit 0.
  </action>
  <verify>
    - `ls scripts/04_model_fitting/a_mle/` shows exactly `_engine.py`, `fit_mle.py`, `__init__.py` (no `12_fit_mle.py`).
    - `ls scripts/04_model_fitting/b_bayesian/` shows exactly `_engine.py`, `fit_bayesian.py`, `fit_baseline.py`, `__init__.py`.
    - `ls scripts/04_model_fitting/c_level2/` shows exactly `fit_with_l2.py`, `__init__.py`.
    - `grep -rn "04_model_fitting/[abc]_[a-z_]*/[12][0-9]_" . --include="*.py" --include="*.md" --include="*.slurm"` returns zero.
    - All four `--help` smoke tests above exit 0.
    - `python -c "from scripts.fitting import bms; print(bms.__name__)"` still works if bms is still there (sanity check — bms was library code that was NOT moved to a_mle/b_bayesian).
  </verify>
  <done>Atomic commit: `refactor(29-04b): rename 04_model_fitting entry scripts + resolve library/CLI collisions via _engine.py`</done>
</task>

<task type="auto" id="4">
  <name>Task 4: Number 05_post_fitting_checks (descriptive → 01-03)</name>
  <files>
    - scripts/05_post_fitting_checks/baseline_audit.py → 01_baseline_audit.py
    - scripts/05_post_fitting_checks/scale_audit.py → 02_scale_audit.py
    - scripts/05_post_fitting_checks/run_posterior_ppc.py → 03_run_posterior_ppc.py  (only if 29-03 created this file; check first)
  </files>
  <action>
    1. Pre-flight check: `ls scripts/05_post_fitting_checks/run_posterior_ppc.py` — if exists (created by 29-03), include in rename; if NOT exists, skip that rename silently and flag in SUMMARY.md as "29-03 did not create run_posterior_ppc.py; deferred to follow-up."
    2. `git mv` each file.
    3. Grep repo for old paths (forward-slash + Python-Path forms); update hits.
    4. Verify ast.parse.
  </action>
  <verify>
    - `ls scripts/05_post_fitting_checks/` shows exactly `01_baseline_audit.py`, `02_scale_audit.py`, and (if present) `03_run_posterior_ppc.py` plus `__init__.py`.
    - `grep -rn "05_post_fitting_checks/\(baseline_audit\|scale_audit\|run_posterior_ppc\)\.py" . --include="*.py" --include="*.md" --include="*.slurm" --include="*.sh"` returns zero.
  </verify>
  <done>Atomic commit: `refactor(29-04b): number 05_post_fitting_checks files 01-03`</done>
</task>

<task type="auto" id="5">
  <name>Task 5: Number 06_fit_analyses (descriptive → 01-08 in paper-read order)</name>
  <files>
    # Paper-read order — first comparison, then Bayesian selection, then averaging, then per-participant analyses, then final tables
    - scripts/06_fit_analyses/compare_models.py → 01_compare_models.py
    - scripts/06_fit_analyses/compute_loo_stacking.py → 02_compute_loo_stacking.py
    - scripts/06_fit_analyses/model_averaging.py → 03_model_averaging.py
    - scripts/06_fit_analyses/analyze_mle_by_trauma.py → 04_analyze_mle_by_trauma.py
    - scripts/06_fit_analyses/regress_parameters_on_scales.py → 05_regress_parameters_on_scales.py
    - scripts/06_fit_analyses/analyze_winner_heterogeneity.py → 06_analyze_winner_heterogeneity.py
    - scripts/06_fit_analyses/bayesian_level2_effects.py → 07_bayesian_level2_effects.py
    - scripts/06_fit_analyses/manuscript_tables.py → 08_manuscript_tables.py
  </files>
  <action>
    1. `git mv` each file.
    2. Grep + update all importers and references (especially cluster/*.slurm — 29-05 runs NEXT and depends on these being final).
    3. Grep for cross-references BETWEEN 06 scripts (e.g., `manuscript_tables.py` imports helpers from `compute_loo_stacking.py`); update relative imports.
    4. Check manuscript/paper.qmd Quarto {python} cells — grep for `06_fit_analyses/` references.
    5. Verify ast.parse on each renamed file.
  </action>
  <verify>
    - `ls scripts/06_fit_analyses/` shows exactly `01_compare_models.py` through `08_manuscript_tables.py` plus `__init__.py`.
    - `grep -rn "06_fit_analyses/\(compare_models\|compute_loo_stacking\|model_averaging\|analyze_mle_by_trauma\|regress_parameters_on_scales\|analyze_winner_heterogeneity\|bayesian_level2_effects\|manuscript_tables\)\.py" . --include="*.py" --include="*.md" --include="*.slurm"` returns zero.
    - `python scripts/06_fit_analyses/01_compare_models.py --help` exits 0 (smoke).
    - `python scripts/06_fit_analyses/08_manuscript_tables.py --help` exits 0 (smoke).
  </verify>
  <done>Atomic commit: `refactor(29-04b): number 06_fit_analyses files 01-08 in paper-read order`</done>
</task>

<task type="auto" id="6">
  <name>Task 6: Full repo importer sweep + closure guard pass</name>
  <files>
    - (any files with stale references that Tasks 1-5 missed)
    - validation/check_v4_closure.py                   (if it hardcodes script paths)
    - scripts/fitting/tests/test_load_side_validation.py  (re-verify enumeration after all renames)
    - CLAUDE.md                                         (Quick Reference block — all example paths updated)
    - README.md                                         (Pipeline block)
    - docs/02_pipeline_guide/ANALYSIS_PIPELINE.md       (stage documentation)
  </files>
  <action>
    1. Comprehensive grep for ALL old names across the repo:
       ```
       grep -rn -E '(02_behav_analyses/0[5-8]_|03_model_prefitting/(09|1[0-3])_|04_model_fitting/a_mle/12_|04_model_fitting/b_bayesian/(13_|21_)|04_model_fitting/c_level2/21_|05_post_fitting_checks/(baseline_audit|scale_audit|run_posterior_ppc)\.py|06_fit_analyses/(compare_models|compute_loo_stacking|model_averaging|analyze_mle_by_trauma|regress_parameters_on_scales|analyze_winner_heterogeneity|bayesian_level2_effects|manuscript_tables)\.py)' . --include="*.py" --include="*.md" --include="*.slurm" --include="*.sh" --include="*.qmd" --include="*.tex"
       ```
    2. For every hit NOT in `.planning/` (historical) and NOT in `docs/legacy/` (archived), update the reference.
    3. `.planning/ROADMAP.md` and `.planning/REQUIREMENTS.md` — SKIP (historical documentation of phase state).
    4. `docs/legacy/*` — SKIP (archived docs; preserve historical text).
    5. `manuscript/paper.qmd` — this file is ALSO owned by Plan 29-06. Coordinate: update paper.qmd script-path references HERE (since we know the final names) and note in SUMMARY.md that 29-06 can skip script-path updates (since this plan already did them).
    6. Run `validation/check_v4_closure.py` — must exit 0.
    7. Run `pytest scripts/fitting/tests/ tests/ validation/ -v` — must pass.
    8. Update CLAUDE.md Quick Reference and README.md Pipeline block with final example paths.
  </action>
  <verify>
    - The comprehensive grep in step 1 returns zero matches outside `.planning/` and `docs/legacy/`.
    - `validation/check_v4_closure.py` exits 0.
    - `pytest scripts/fitting/tests/ tests/ validation/ -q` passes clean.
    - `grep -rn "python scripts/" CLAUDE.md README.md docs/02_pipeline_guide/ANALYSIS_PIPELINE.md` shows ALL example paths are in new canonical form.
  </verify>
  <done>Atomic commit: `fix(29-04b): full repo importer + doc sweep for renumbered stage files`</done>
</task>

<task type="auto" id="7">
  <name>Task 7: Write SUMMARY.md + plan metadata commit</name>
  <files>
    - .planning/phases/29-pipeline-canonical-reorg/29-04b-SUMMARY.md         (new)
    - .planning/STATE.md                                                      (position update)
    - .planning/phases/29-pipeline-canonical-reorg/29-04b-intra-stage-renumbering-PLAN.md  (plan metadata commit)
  </files>
  <action>
    1. Write SUMMARY.md following the template at `C:\Users\aman0087\.claude\get-shit-done\templates\summary.md`. Sections:
       - Deliverables (files renamed, counts per stage)
       - Decisions (underscore-private convention for _engine.py; paper.qmd script refs updated here vs. deferred to 29-06)
       - Deviations (if any tasks hit checkpoints or unexpected conditions)
       - Verification (paste outputs of final grep + v4 closure + pytest)
    2. Update STATE.md: "Phase 29 Plan 04b complete — Scheme D intra-stage renumbering applied; ready for Wave 4 (29-05 SLURM consolidation, 29-06 paper.qmd render)."
    3. Plan metadata commit: `git add .planning/phases/29-pipeline-canonical-reorg/29-04b-intra-stage-renumbering-PLAN.md .planning/phases/29-pipeline-canonical-reorg/29-04b-SUMMARY.md .planning/STATE.md && git commit -m "docs(29-04b): complete intra-stage-renumbering plan"`
  </action>
  <verify>
    - `cat .planning/phases/29-pipeline-canonical-reorg/29-04b-SUMMARY.md` shows valid SUMMARY structure.
    - `git log --oneline -8` shows 7 commits for this plan (6 task commits + 1 metadata).
    - Working tree clean: `git status --porcelain` returns empty.
  </verify>
  <done>Plan complete.</done>
</task>

</tasks>

<success_criteria>

- [ ] All 6 stage folders (01-06) use consistent Scheme D numbering: either local 01-N intra-stage prefix (for strict/loose-order stages) OR no numeric prefix (for parallel-alternative subfolders).
- [ ] Zero carry-over global prefixes (12_, 13_, 14_, 15_, 16_, 17_, 18_, 21_) survive inside any stage folder.
- [ ] `04_model_fitting/a_mle/` and `04_model_fitting/b_bayesian/` each have exactly one `_engine.py` (library) + canonical descriptive CLI entries (`fit_mle.py`; `fit_bayesian.py`, `fit_baseline.py`).
- [ ] All four --help smoke tests in Task 3 pass.
- [ ] `scripts/06_fit_analyses/` files are numbered 01-08 in paper-read order (compare → stacking → averaging → MLE trauma → scale regression → heterogeneity → L2 effects → tables).
- [ ] `validation/check_v4_closure.py` exits 0.
- [ ] `pytest scripts/fitting/tests/ tests/ validation/ -q` passes.
- [ ] No stale path references in src/, scripts/, tests/, validation/, cluster/, CLAUDE.md, README.md, docs/ (excluding docs/legacy/), manuscript/paper.qmd.
- [ ] SUMMARY.md created at `.planning/phases/29-pipeline-canonical-reorg/29-04b-SUMMARY.md`.
- [ ] STATE.md updated with plan-04b completion.
- [ ] Plan metadata commit: `docs(29-04b): complete intra-stage-renumbering plan`.

</success_criteria>

<notes_for_downstream_plans>

**For Plan 29-05 (cluster SLURM consolidation, Wave 4):**
- All script paths are now canonical. Write `cluster/*.slurm` bodies with the final names: `python scripts/04_model_fitting/a_mle/fit_mle.py --model ...`, `python scripts/06_fit_analyses/08_manuscript_tables.py`, etc.
- Do NOT reference any `12_`, `13_`, `14_`, `21_` filenames.

**For Plan 29-06 (paper.qmd smoke render, Wave 4):**
- Task 6 of this plan updates `manuscript/paper.qmd` script-path references AS PART OF THE FULL SWEEP.
- 29-06 can SKIP the "update script-path references" step and go straight to quarto render + paper.qmd line-166 caption edit absorbed from 29-02.

**For Plan 29-07 (closure guard, Wave 5):**
- Add a test assertion: for every file in `scripts/0N_*/`, filename must match either `^\d{2}_[a-z_]+\.py$` (numbered) OR `^[a-z_]+\.py$` (unnumbered) — catches accidental regression to global numbering.
- Add a test assertion: no file in `scripts/04_model_fitting/[abc]_*/` has a numeric prefix (enforces Scheme D rule 3).

**Scheme D reference** (pin this for future contributors in CLAUDE.md):

> ```
> scripts/<stage>/<file>
>   where <stage> is "0N_<descriptive>" (N = 1..6)
>   where <file> is either:
>     (a) "0M_<descriptive>.py" (M = 1..N_in_stage) — use in strict-order or
>         paper-read-order stages (01, 02, 03, 05, 06)
>     (b) "<descriptive>.py" — use in parallel-alternative subfolders
>         (04_model_fitting/{a_mle,b_bayesian,c_level2}/)
> 
> Library/engine code that collides with CLI entry names uses an underscore
> prefix: _engine.py, _helpers.py. Callers import via __init__.py re-exports.
> 
> Model fanout (M1/M2/M3/M5/M6a/M6b/M4) is via CLI flag --model <name>,
> NEVER via per-model script files.
> ```

</notes_for_downstream_plans>
