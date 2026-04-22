---
phase: 29-pipeline-canonical-reorg
plan: 03
type: execute
wave: 2
depends_on: [29-01]
files_modified:
  - scripts/utils/ppc.py                         (new — canonical simulator extracted from 03_model_prefitting/09_run_ppc.py + 03_model_prefitting/12_run_prior_predictive.py)
  - scripts/utils/plotting.py                    (renamed from scripts/utils/plotting_utils.py, deduplicated with scripts/analysis/plotting_utils.py)
  - scripts/utils/stats.py                       (renamed from scripts/utils/statistical_tests.py)
  - scripts/utils/scoring.py                     (renamed from scripts/utils/scoring_functions.py)
  - scripts/utils/data_cleaning.py               (unchanged, confirmed canonical)
  - scripts/utils/__init__.py                    (new — empty, formalizes utils as package)
  - scripts/utils/plotting_utils.py              (deleted)
  - scripts/utils/statistical_tests.py           (deleted)
  - scripts/utils/scoring_functions.py           (deleted)
  - scripts/utils/remap_mle_ids.py               (evaluated; stays or moves to scripts/_maintenance/)
  - scripts/utils/sync_experiment_data.py        (evaluated; stays or moves to scripts/_maintenance/)
  - scripts/utils/update_participant_mapping.py  (evaluated; stays or moves to scripts/_maintenance/)
  - scripts/05_post_fitting_checks/run_posterior_ppc.py  (new — thin orchestrator imports scripts.utils.ppc)
  - scripts/03_model_prefitting/09_run_ppc.py    (rewritten as thin orchestrator using scripts.utils.ppc)
  - scripts/03_model_prefitting/12_run_prior_predictive.py  (rewritten as thin orchestrator)
  - scripts/**/*.py                              (importer updates: plotting_utils → plotting, statistical_tests → stats, scoring_functions → scoring)
autonomous: true

must_haves:
  truths:
    - "Simulator logic lives ONCE — in scripts/utils/ppc.py — and is imported by stage 03 prior PPC, stage 03 generate-synthetic-data, stage 05 posterior PPC, and any other PPC-like workflows (SC#4)"
    - "plotting/stats/scoring helpers have single canonical names (plotting.py, stats.py, scoring.py) and exactly one definition each"
    - "Dead duplicate at scripts/analysis/plotting_utils.py identified; deduplication plan documented (resolution in 29-04 dead-folder audit)"
    - "Every stage folder that needs a shared helper imports it from scripts.utils (no cross-stage imports like `from scripts.03_model_prefitting.X import ...`)"
  artifacts:
    - path: "scripts/utils/ppc.py"
      provides: "canonical single-source PPC simulator (simulate_from_samples, run_prior_ppc, run_posterior_ppc)"
      min_lines: 100
      contains: "def run_"
    - path: "scripts/utils/plotting.py"
      provides: "canonical plot helpers (setup_plot_style, save_figure, get_color_palette, ...)"
    - path: "scripts/utils/stats.py"
      provides: "statistical-test helpers (formerly statistical_tests.py)"
    - path: "scripts/utils/scoring.py"
      provides: "scoring helpers (formerly scoring_functions.py)"
    - path: "scripts/utils/__init__.py"
      provides: "package marker so `from scripts.utils.X import Y` resolves cleanly"
    - path: "scripts/05_post_fitting_checks/run_posterior_ppc.py"
      provides: "thin posterior-PPC orchestrator (imports scripts.utils.ppc)"
  key_links:
    - from: "scripts/03_model_prefitting/09_run_ppc.py"
      to: "scripts/utils/ppc.py"
      via: "from scripts.utils.ppc import simulate_from_samples"
      pattern: "from scripts\\.utils\\.ppc import"
    - from: "scripts/03_model_prefitting/12_run_prior_predictive.py"
      to: "scripts/utils/ppc.py"
      via: "from scripts.utils.ppc import run_prior_ppc"
      pattern: "from scripts\\.utils\\.ppc import"
    - from: "scripts/05_post_fitting_checks/run_posterior_ppc.py"
      to: "scripts/utils/ppc.py"
      via: "from scripts.utils.ppc import run_posterior_ppc"
      pattern: "from scripts\\.utils\\.ppc import"
---

<objective>
With the canonical 01–06 stage layout in place (post 29-01), consolidate shared helpers into `scripts/utils/` and make the simulator a SINGLE source of truth. Extract PPC logic from the current `scripts/03_model_prefitting/09_run_ppc.py` + `12_run_prior_predictive.py` into `scripts/utils/ppc.py`, rename `plotting_utils.py`/`statistical_tests.py`/`scoring_functions.py` to the shorter canonical names (`plotting.py`/`stats.py`/`scoring.py`), deduplicate against any sibling copies (notably `scripts/analysis/plotting_utils.py`), and create a stage 05 `run_posterior_ppc.py` thin orchestrator that imports the utils simulator — mirroring the stage 03 prior-PPC orchestrator.

Purpose: Enforce the phase's core design principle — any function used by ≥ 2 stages lives in `utils/`, never duplicated across stage folders.

Output: canonical `scripts/utils/` package; `run_ppc` simulator callable from both prior (stage 03) and posterior (stage 05) workflows; all importers updated.
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
</context>

<tasks>

<task type="auto">
  <name>Task 1: Extract PPC simulator into scripts/utils/ppc.py (single-source)</name>
  <files>
    - scripts/utils/ppc.py (new)
    - scripts/utils/__init__.py (new — empty or module docstring)
    - scripts/03_model_prefitting/09_run_ppc.py (rewritten as thin orchestrator)
    - scripts/03_model_prefitting/12_run_prior_predictive.py (rewritten as thin orchestrator)
    - scripts/05_post_fitting_checks/run_posterior_ppc.py (new thin orchestrator)
  </files>
  <action>
    1. Read `scripts/03_model_prefitting/09_run_ppc.py` and `scripts/03_model_prefitting/12_run_prior_predictive.py` in full. Identify the simulator functions (likely `simulate_from_samples`, `run_prior_ppc_for_model`, or similarly-named helpers that JAX-simulate choices given parameter samples + environment configuration).
    2. Create `scripts/utils/__init__.py` as an empty file (turns utils/ into a formal package). If it already exists and is non-empty, leave it alone.
    3. Create `scripts/utils/ppc.py` exporting three functions (names adjusted to whatever already exists — use the same names as the longer existing file to minimize downstream API churn):
       - `simulate_from_samples(model_name, posterior_or_prior_samples, env_config, seed) -> xarray.Dataset` — the core JAX simulator
       - `run_prior_ppc(model_name, n_draws, output_dir, seed) -> pathlib.Path` — convenience wrapper for stage 03
       - `run_posterior_ppc(model_name, posterior_netcdf_path, output_dir, seed) -> pathlib.Path` — convenience wrapper for stage 05
       Copy the implementation verbatim from whichever existing file has the canonical version. Prefer `09_run_ppc.py`'s implementation if it's more featureful; otherwise `12_run_prior_predictive.py`'s.
    4. Add NumPy-style docstrings (per CLAUDE.md global conventions).
    5. Add `from __future__ import annotations` at the top.
    6. Rewrite `scripts/03_model_prefitting/09_run_ppc.py` as a thin `argparse`-driven CLI that calls `scripts.utils.ppc.run_posterior_ppc()` (since `09_run_ppc` is the posterior workflow in the original pipeline) or `run_prior_ppc()` depending on its original purpose — read the original docstring to confirm.
    7. Rewrite `scripts/03_model_prefitting/12_run_prior_predictive.py` as a thin CLI calling `scripts.utils.ppc.run_prior_ppc()`.
    8. CREATE `scripts/05_post_fitting_checks/run_posterior_ppc.py` as a NEW thin CLI calling `scripts.utils.ppc.run_posterior_ppc()`. This is the stage 05 counterpart to stage 03's prior-PPC orchestrator — explicitly requested by 29-CONTEXT.md target shape. ~40 lines of argparse + call + done.
    9. Grep for any OTHER file that defines simulator logic inline (duplication):
       - `grep -rn "def simulate_from_samples\|def run_ppc\|def run_prior_ppc\|def run_posterior_ppc\|def simulate_agent" scripts/ --include="*.py"`
       - For each hit outside `scripts/utils/ppc.py` and the thin orchestrators, determine if it's a duplicate that should import from utils. KNOWN: `scripts/simulations/unified_simulator.py` has `simulate_agent_fixed` and `simulate_agent_sampled` — those are the lower-level Gym-environment simulators used by tests/validation. Do NOT dedupe against them; they're a separate layer (single-agent trajectory simulation, not posterior-draw simulation). Document the distinction in the `scripts/utils/ppc.py` module docstring: "For single-agent trajectory simulation given fixed parameters, see `scripts/simulations/unified_simulator.py` (consumed by tests)."
    10. Update internal imports: stage 03 and stage 05 orchestrators now all `from scripts.utils.ppc import ...`.
  </action>
  <verify>
    - `test -f scripts/utils/ppc.py && test -f scripts/utils/__init__.py`
    - `grep -c "^def " scripts/utils/ppc.py` returns ≥ 2 (at least simulate_from_samples + run_prior_ppc; ideally 3)
    - `test -f scripts/05_post_fitting_checks/run_posterior_ppc.py`
    - `grep -n "from scripts.utils.ppc" scripts/03_model_prefitting/09_run_ppc.py scripts/03_model_prefitting/12_run_prior_predictive.py scripts/05_post_fitting_checks/run_posterior_ppc.py` shows 3 matches (one per file)
    - `grep -rn "def run_ppc\|def run_prior_ppc\|def run_posterior_ppc\|def simulate_from_samples" scripts/ --include="*.py" | grep -v "scripts/utils/ppc.py"` returns ZERO (definitions live only in utils/)
    - `python -c "from scripts.utils.ppc import run_prior_ppc, run_posterior_ppc; print('imports ok')"` prints `imports ok` (smoke test; may require PYTHONPATH=. depending on CWD)
  </verify>
  <done>Simulator lives in one file; stage 03 and stage 05 orchestrators are ≤ 80 lines each and delegate to utils/; posterior-PPC orchestrator new-created for stage 05.</done>
</task>

<task type="auto">
  <name>Task 2: Rename utils helpers to canonical short names + update all importers</name>
  <files>
    - scripts/utils/plotting.py              (git mv from plotting_utils.py)
    - scripts/utils/stats.py                 (git mv from statistical_tests.py)
    - scripts/utils/scoring.py               (git mv from scoring_functions.py)
    - scripts/utils/plotting_utils.py        (deleted)
    - scripts/utils/statistical_tests.py     (deleted)
    - scripts/utils/scoring_functions.py     (deleted)
    - scripts/**/*.py                        (importer updates)
  </files>
  <action>
    1. `git mv scripts/utils/plotting_utils.py scripts/utils/plotting.py`
    2. `git mv scripts/utils/statistical_tests.py scripts/utils/stats.py`
    3. `git mv scripts/utils/scoring_functions.py scripts/utils/scoring.py`
    4. Grep for all importers (ACROSS entire repo including src/ but NOT .planning/):
       ```
       grep -rn "from scripts.utils.plotting_utils\|from scripts.utils.statistical_tests\|from scripts.utils.scoring_functions\|import scripts.utils.plotting_utils\|import scripts.utils.statistical_tests\|import scripts.utils.scoring_functions" . --include="*.py" --exclude-dir=.planning --exclude-dir=.git
       ```
    5. Rewrite each hit:
       - `from scripts.utils.plotting_utils` → `from scripts.utils.plotting`
       - `from scripts.utils.statistical_tests` → `from scripts.utils.stats`
       - `from scripts.utils.scoring_functions` → `from scripts.utils.scoring`
    6. Likely referrers (not exhaustive): `scripts/02_behav_analyses/08_run_statistical_analyses.py` (statistical_tests), `scripts/02_behav_analyses/06_visualize_task_performance.py` (plotting_utils), potentially `scripts/06_fit_analyses/*.py` files.
    7. The SIBLING duplicate at `scripts/analysis/plotting_utils.py` is handled in 29-04 (dead-folder audit). This plan does NOT touch it — but DO verify that no currently-live code imports from `scripts.analysis.plotting_utils`:
       - `grep -rn "from scripts.analysis.plotting_utils\|import scripts.analysis.plotting_utils" . --include="*.py" --exclude-dir=.planning`
       - If any live importer exists (not just tests/legacy/), add a note to 29-04 that the canonical version is now `scripts/utils/plotting.py` and the dead-folder audit must rewrite that importer before deleting `scripts/analysis/`.
       - Expected: only `tests/legacy/examples/explore_prior_parameter_space.py` references `scripts.analysis.plotting_utils` — it's legacy and handled separately.
  </action>
  <verify>
    - `test -f scripts/utils/plotting.py && test -f scripts/utils/stats.py && test -f scripts/utils/scoring.py`
    - `test ! -f scripts/utils/plotting_utils.py && test ! -f scripts/utils/statistical_tests.py && test ! -f scripts/utils/scoring_functions.py`
    - `grep -rn "from scripts.utils.plotting_utils\|from scripts.utils.statistical_tests\|from scripts.utils.scoring_functions\|import scripts.utils.plotting_utils\|import scripts.utils.statistical_tests\|import scripts.utils.scoring_functions" scripts/ tests/ validation/ cluster/ manuscript/ docs/ src/ --exclude-dir=.planning` returns ZERO
    - `git log --follow --oneline scripts/utils/plotting.py | head -3` shows the plotting_utils.py history
    - `python -c "from scripts.utils import plotting, stats, scoring; print('ok')"` prints `ok`
  </verify>
  <done>Three canonical helpers renamed; zero stale `plotting_utils`/`statistical_tests`/`scoring_functions` imports; history preserved.</done>
</task>

<task type="auto">
  <name>Task 3: Evaluate one-off maintenance scripts + commit 29-03 atomically</name>
  <files>
    - scripts/utils/remap_mle_ids.py (decision + possible move)
    - scripts/utils/sync_experiment_data.py (decision + possible move)
    - scripts/utils/update_participant_mapping.py (decision + possible move)
    - scripts/_maintenance/ (possibly new directory)
  </files>
  <action>
    1. Read the top 30 lines of each of the three files to determine their nature:
       - `scripts/utils/remap_mle_ids.py`
       - `scripts/utils/sync_experiment_data.py`
       - `scripts/utils/update_participant_mapping.py`
    2. Apply decision rule (per 29-CONTEXT.md §Utils consolidation): a file belongs in `scripts/utils/` if-and-only-if it provides helpers imported by ≥ 2 stage folders. A one-off maintenance CLI script (used by a human for data-curation tasks, never imported) belongs in `scripts/_maintenance/`.
    3. Quick grep: `grep -rn "from scripts.utils.remap_mle_ids\|from scripts.utils.sync_experiment_data\|from scripts.utils.update_participant_mapping\|import scripts.utils.remap_mle_ids\|import scripts.utils.sync_experiment_data\|import scripts.utils.update_participant_mapping" . --include="*.py" --exclude-dir=.planning`
    4. Decision matrix:
       - If ZERO importers AND the file contains CLI `main()` only → move to `scripts/_maintenance/`
       - If any live importer exists → stay in `scripts/utils/`
    5. Execute moves only if clear. If ambiguous on any one file, LEAVE IT IN utils/ and document the deferral in the summary (do not block on this decision).
    6. Commit 29-03 atomically:
       ```
       refactor(29-03): utils consolidation — canonical ppc/plotting/stats/scoring single-sources

       - Extract simulator logic into scripts/utils/ppc.py (single-source; used by 03 prior PPC + 05 posterior PPC + 03 synthetic-data generation)
       - Rename scripts/utils/{plotting_utils,statistical_tests,scoring_functions}.py to {plotting,stats,scoring}.py via git mv
       - Create scripts/utils/__init__.py (empty package marker)
       - Create scripts/05_post_fitting_checks/run_posterior_ppc.py as thin orchestrator (mirror of 03 prior-PPC)
       - Rewrite scripts/03_model_prefitting/{09_run_ppc,12_run_prior_predictive}.py as thin orchestrators
       - Update N importers across scripts/, tests/, validation/
       - One-off maintenance scripts {remap_mle_ids, sync_experiment_data, update_participant_mapping} (handled/deferred per task 3 grep results)
       ```
    7. Verify: `pytest scripts/fitting/tests/ tests/ validation/` passes clean (or at least no NEW failures vs. pre-plan baseline).
  </action>
  <verify>
    - Decision documented in SUMMARY (either moved or kept) for each of the 3 maintenance files
    - `git log -1 --stat` shows the 29-03 commit
    - `pytest scripts/fitting/tests/test_v4_closure.py -v` PASSES 3/3
    - `pytest scripts/fitting/tests/ --collect-only` shows no new collection errors
  </verify>
  <done>29-03 commit landed; utils canonical; maintenance scripts decisions documented; test suite collection clean.</done>
</task>

</tasks>

<verification>
```bash
# Canonical single-source simulator
test -f scripts/utils/ppc.py
grep -rn "def simulate_from_samples\|def run_prior_ppc\|def run_posterior_ppc" scripts/ --include="*.py" | grep -v "scripts/utils/ppc.py" || echo "OK: simulator defined only in utils"

# Canonical names
for f in plotting stats scoring; do test -f scripts/utils/$f.py; done
for f in plotting_utils statistical_tests scoring_functions; do test ! -f scripts/utils/$f.py; done

# No stale importers (zero matches)
grep -rn "from scripts.utils.plotting_utils\|from scripts.utils.statistical_tests\|from scripts.utils.scoring_functions" \
  scripts/ tests/ validation/ cluster/ manuscript/ docs/ src/ --exclude-dir=.planning \
  || echo "OK: zero stale imports"

# Stage 05 posterior-PPC orchestrator exists
test -f scripts/05_post_fitting_checks/run_posterior_ppc.py
grep -n "from scripts.utils.ppc" scripts/05_post_fitting_checks/run_posterior_ppc.py

# Package imports resolve
python -c "from scripts.utils import ppc, plotting, stats, scoring; print('ok')"

# v4 closure
pytest scripts/fitting/tests/test_v4_closure.py -v
```
</verification>

<success_criteria>
1. `scripts/utils/ppc.py` is the ONLY file in `scripts/` defining `simulate_from_samples`/`run_prior_ppc`/`run_posterior_ppc` (SC#4).
2. `scripts/utils/{plotting,stats,scoring}.py` replace the verbose-suffix names; git history preserved.
3. Stage 03 PPC orchestrators (`09_run_ppc.py`, `12_run_prior_predictive.py`) import from `scripts.utils.ppc`, not each other.
4. Stage 05 has a new `run_posterior_ppc.py` orchestrator that also imports from `scripts.utils.ppc`.
5. Every stage file that uses plotting/stats/scoring imports from the canonical short names.
6. Decision made on `remap_mle_ids.py` / `sync_experiment_data.py` / `update_participant_mapping.py` (stay or move to `_maintenance/`).
7. `pytest scripts/fitting/tests/test_v4_closure.py` passes 3/3.
</success_criteria>

<output>
After completion, create `.planning/phases/29-pipeline-canonical-reorg/29-03-SUMMARY.md` with:
- Functions extracted into `scripts/utils/ppc.py` (names + line counts)
- Renames: plotting_utils→plotting, statistical_tests→stats, scoring_functions→scoring (list of importers rewritten per renamed module)
- Decisions made on the 3 maintenance scripts
- Commit SHA
- Any deferred deduplication work (e.g., sibling `scripts/analysis/plotting_utils.py` flagged for 29-04)
</output>
