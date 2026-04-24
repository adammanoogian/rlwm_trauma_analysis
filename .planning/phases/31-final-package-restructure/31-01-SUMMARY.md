---
phase: 31-final-package-restructure
plan: 01
subsystem: infra
tags: [config, ccds, pytest, gitignore, scaffolding, path-constants]

# Dependency graph
requires:
  - phase: 29-pipeline-canonical-reorg
    provides: Scheme D scripts/ layout (final, untouched by this plan)
provides:
  - "config.py CCDS Path constants (8 core + 13 convenience) for Waves B/C/D consumption"
  - "DataParams primary names flipped to INTERIM_DIR + PROCESSED_DIR (31-02 physical moves will populate)"
  - "15 empty CCDS-aligned directories with .gitkeep scaffolding"
  - ".gitignore patterns covering data/raw, data/interim, models/bayesian/*.nc, logs/"
  - "pytest.ini + pyproject.toml testpaths collapsed to single tests/ root + scientific marker registered"
affects: [31-02, 31-03, 31-04, 31-05, 31-06]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "CCDS (Cookiecutter Data Science) data/{raw,interim,processed,external} tiering"
    - "Config-as-single-source-of-truth for Path constants (forbids scatter of hardcoded paths)"
    - "Legacy alias pattern during multi-wave refactor (old + new coexist, removal in Wave E)"
    - ".gitkeep + ! negation pattern to preserve empty-dir scaffolding across fresh clones"

key-files:
  created:
    - data/raw/.gitkeep
    - data/interim/.gitkeep
    - data/processed/.gitkeep
    - data/external/.gitkeep
    - models/bayesian/.gitkeep
    - models/mle/.gitkeep
    - models/ppc/.gitkeep
    - models/recovery/.gitkeep
    - models/parameter_exploration/.gitkeep
    - reports/figures/.gitkeep
    - reports/tables/.gitkeep
    - tests/unit/.gitkeep
    - tests/integration/.gitkeep
    - tests/scientific/.gitkeep
    - logs/.gitkeep
  modified:
    - config.py
    - .gitignore
    - pytest.ini
    - pyproject.toml

key-decisions:
  - "All config.py writes for Waves 1-2 consolidated in this plan (31-02/31-03 must not write config.py to avoid parallel write race)"
  - "Legacy OUTPUT_DIR + FIGURES_DIR retained as transitional aliases; removed in Wave E (plan 31-05)"
  - "pytest.ini addopts synced to pyproject.toml [tool.pytest.ini_options] to avoid drift confusion"
  - "Testpaths collapsed before Wave D moves files — accepts ~207 test reduction as visible failure mode (preferable to silent duplicate collection)"

patterns-established:
  - "Phase 31 CCDS block appended at end of .gitignore (labeled) so Wave E can find it for clean removal"
  - "Scaffolding mkdir list in config.py as single authority for which dirs must exist"

# Metrics
duration: 10min
completed: 2026-04-24
---

# Phase 31 Plan 01: CCDS Foundation Summary

**Foundation wave: config.py exposes 8 core CCDS Path constants + 13 convenience constants + DataParams flipped to interim/processed tiers + 15 empty scaffolding dirs + pytest single-root collapse — all downstream waves now read from these constants.**

## Performance

- **Duration:** 10 min
- **Started:** 2026-04-24T06:24:28Z
- **Completed:** 2026-04-24T06:34:26Z
- **Tasks:** 3 / 3
- **Files modified:** 4
- **Files created:** 15 (.gitkeep scaffolding)

## Accomplishments

- **config.py updated** with 8 core CCDS Path constants (DATA_RAW_DIR, INTERIM_DIR, PROCESSED_DIR, DATA_EXTERNAL_DIR, MODELS_DIR, MODELS_{BAYESIAN,MLE,PPC,RECOVERY,PARAMETER_EXPLORATION}_DIR, REPORTS_DIR, REPORTS_{FIGURES,TABLES}_DIR, LOGS_DIR), 13 convenience constants for plan 31-03 (MODELS_BAYESIAN_{BASELINE,L2,LEVEL2,MANUSCRIPT,PRIOR_PREDICTIVE,RECOVERY} + REPORTS_TABLES_{DESCRIPTIVES,MODEL_COMPARISON,BEHAVIORAL,REGRESSIONS,TRAUMA_GROUPS} + REPORTS_FIGURES_{BAYESIAN,MODEL_COMPARISON}), and 2 legacy aliases (OUTPUT_DIR, FIGURES_DIR).
- **DataParams primary names flipped** to new tiers — TASK_TRIALS_LONG now resolves to `data/processed/task_trials_long.csv`, COLLATED_DATA + PARSED_* now resolve under `data/interim/`. Constants are lazily consumed so they are safe to land ahead of 31-02's physical file moves.
- **get_excluded_participants + get_analysis_cohort** default `data_path` arguments updated from `OUTPUT_DIR / 'task_trials_long.csv'` to `PROCESSED_DIR / 'task_trials_long.csv'` (the two functions that had inline default paths).
- **15 scaffolding directories** created on disk with `.gitkeep` files so git tracks the empty structure across fresh clones.
- **.gitignore** got a clearly labeled "Phase 31 CCDS layout" block covering raw-data PII, interim parsed CSVs, large binary posteriors (`models/bayesian/**/*.nc`), MLE checkpoint CSVs, and wholesale `logs/*` — all with `!<dir>/.gitkeep` negations preserving the scaffolding.
- **pytest.ini + pyproject.toml** both collapsed to `testpaths = ["tests"]` (single root) and both registered the new `scientific` marker. Wave D (plan 31-04) will relocate `scripts/fitting/tests/` + `validation/` into `tests/{unit,integration,scientific}/`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Update config.py with CCDS Path constants** — `80b3823` (feat)
2. **Task 2: Create .gitkeep scaffolding and update .gitignore** — `080e206` (chore)
3. **Task 3: Update pytest.ini testpaths and register scientific marker** — `d17afb4` (chore)

## Files Created/Modified

### Modified

- `config.py` — Added CCDS Path constants block (DATA_RAW_DIR, INTERIM_DIR, PROCESSED_DIR, DATA_EXTERNAL_DIR, MODELS_DIR + 5 subdirs, REPORTS_DIR + 2 subdirs, LOGS_DIR), 13 convenience constants, kept OUTPUT_DIR+FIGURES_DIR as legacy aliases, flipped DataParams primary names to INTERIM_DIR/PROCESSED_DIR, updated default args in 2 helper functions, expanded mkdir loop to 12 dirs.
- `.gitignore` — Appended "Phase 31 CCDS layout" block with 20 lines of new patterns + 8 .gitkeep negations.
- `pytest.ini` — Collapsed testpaths from 3 roots (`tests scripts/fitting/tests validation`) to 1 (`tests`), added `scientific` marker registration.
- `pyproject.toml` — `[tool.pytest.ini_options]` testpaths collapsed to `["tests"]` to match pytest.ini, added scientific marker.

### Created (15 .gitkeep files)

- `data/{raw,interim,processed,external}/.gitkeep`
- `models/{bayesian,mle,ppc,recovery,parameter_exploration}/.gitkeep`
- `reports/{figures,tables}/.gitkeep`
- `tests/{unit,integration,scientific}/.gitkeep`
- `logs/.gitkeep`

### Verification Evidence

- `python config.py` exits 0 and prints full config summary
- `DataParams.TASK_TRIALS_LONG` resolves to `C:\Users\aman0087\Documents\Github\rlwm_trauma_analysis\data\processed\task_trials_long.csv`
- `DataParams.COLLATED_DATA` resolves to `...\data\interim\collated_participant_data.csv`
- Import smoke test: 16 core constants (8 new + 4 model subdirs + 2 report subdirs + 2 legacy) all import; 13 convenience constants all import
- `git check-ignore` confirms: raw participant CSVs, `models/bayesian/**/*.nc`, `logs/*.out` all ignored; `.gitkeep` files NOT ignored (negation pattern works)
- `grep -c "testpaths = tests" pytest.ini` = 2 (one data line + one comment line referencing legacy value)
- `pytest --collect-only` collects 69 tests from `tests/` (down from 276 under old 3-root config — expected until Wave D moves files)

## Decisions Made

- **All config.py writes consolidated in plan 31-01.** The plan frontmatter flagged this explicitly: plans 31-02 and 31-03 run in parallel in Wave 2, and parallel yolo-mode writes to `config.py` would race. So DataParams primary-name flip (needed by 31-02) and 13 convenience constants (needed by 31-03) were landed HERE in 31-01 even though they describe paths not yet physically populated. The constants are lazily resolved at call time, so landing them ahead of physical file moves is safe.
- **Legacy OUTPUT_DIR + FIGURES_DIR kept as aliases** pointing at `PROJECT_ROOT / 'output'` and `PROJECT_ROOT / 'figures'` respectively. Waves B/C will update call sites to use the new CCDS constants; plan 31-05 (Wave E) will delete the legacy aliases once zero callers remain.
- **pytest testpaths collapse accepted as visible regression.** Running `pytest` from the repo root now collects 69 tests instead of 276. This is intentional — the plan explicitly framed it: "the failure mode is 'tests no longer run' (visible), not 'tests silently duplicate' (invisible)." Wave D (plan 31-04) will move the missing tests under `tests/{unit,integration,scientific}/`.
- **pyproject.toml `[tool.pytest.ini_options]` also updated** even though pytest.ini takes precedence — synced to avoid silent drift if a tool reads pyproject without pytest.ini.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Synced pyproject.toml testpaths to match pytest.ini**

- **Found during:** Task 3 verification
- **Issue:** `pyproject.toml` `[tool.pytest.ini_options]` had `testpaths = ["tests", "validation", "scripts/fitting/tests"]` which would drift out of sync with the new single-root pytest.ini. The plan's Task 3 action said "if pyproject.toml has a conflicting testpaths, update or remove it" — so I updated it.
- **Fix:** Changed pyproject.toml testpaths to `["tests"]`, also added `scientific` marker, added a comment pointing to plan 31-04 (Wave D).
- **Files modified:** pyproject.toml
- **Verification:** `grep -n "testpaths" pyproject.toml` → `128:testpaths = ["tests"]` (matches pytest.ini).
- **Committed in:** d17afb4 (Task 3 commit)

---

**Total deviations:** 1 auto-fixed (1 Rule 3 — blocking sync, pre-emptive per plan's explicit guidance).
**Impact on plan:** Minimal. The plan explicitly mentioned this case in Task 3's action text ("keeping both in sync avoids confusion"). No scope creep.

### Observations (not deviations)

- **Pre-existing `scripts.legacy` ImportError in `tests/test_wmrl_exploration.py`.** Before my changes, pytest collection with the old 3-root testpaths reported `ERROR collecting tests/test_wmrl_exploration.py ... ModuleNotFoundError: No module named 'scripts.legacy'` AND `ERROR collecting validation/test_unified_simulator.py` with the same cause (276 tests collected with 2 errors). After Task 3, only `tests/test_wmrl_exploration.py` remains (validation/ no longer in testpaths). This is a **Phase 29 remnant** — `scripts/legacy/` directory was archived but appears to lack `__init__.py`. Explicitly left unfixed in plan 31-01 because:
  1. It's out of scope for the foundation wave (no script moves).
  2. Fixing it might conflict with Phase 29's archival design decisions.
  3. The plan accepted reduced test count as a Wave D fix-up.
- **One `.gitkeep` visible in `git status --porcelain`.** After initial creation, `git status --porcelain` shows `?? logs/.gitkeep` but the other 14 `.gitkeep` files are rolled up under their parent directory entries (`?? data/`, `?? models/`, `?? reports/`, `?? tests/integration/`, etc.). This is standard git behavior for untracked files inside untracked directories — once `git add` was run for Task 2, all 15 .gitkeep files staged correctly.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required for a foundation-scaffolding plan.

## Next Phase Readiness

### Unblocked for Wave 2 (parallel fan-out)

- **Plan 31-02** can safely start: reads `DATA_RAW_DIR`, `INTERIM_DIR`, `PROCESSED_DIR` constants (now defined), writes to `data/raw/`, `data/interim/`, `data/processed/` (now scaffolded with `.gitkeep`). DataParams primary names already flipped — 31-02 just needs to physically `git mv` files into place.
- **Plan 31-03** can safely start: reads `MODELS_DIR`, `REPORTS_DIR`, all 13 convenience constants, and writes to `models/{bayesian,mle,ppc,recovery,parameter_exploration}/` + `reports/{figures,tables}/` (now scaffolded). Zero overlap with 31-02's file list → no git-index race.
- **config.py contract:** neither 31-02 nor 31-03 will touch config.py. All constants they need are already landed.

### Wave 3 (plan 31-04 — test consolidation)

- `tests/{unit,integration,scientific}/` directories exist and are tracked via .gitkeep. Wave D can `git mv` test files directly into these pre-created tiers without directory-creation races.
- `scientific` marker registered in both pytest.ini and pyproject.toml — Wave D's scientific-tier tests can use `@pytest.mark.scientific` immediately.
- Pre-existing `scripts.legacy` ImportError in `tests/test_wmrl_exploration.py` will need resolution during Wave D (either fix the import path or move the test).

### Wave E (plan 31-05 — legacy cleanup)

- Transitional aliases flagged for removal:
  - `config.py`: `OUTPUT_DIR`, `FIGURES_DIR`
  - `.gitignore`: pre-Phase-31 patterns (`output/regressions/**/*.png`, `output/*.nc`, `data/rlwm_trauma_PARTICIPANT_SESSION_*.csv`, `data/backup_*.csv`, `data/participant_id_mapping.json`, `output/parsed_*.csv`, `cluster/logs/`, `output/mle/*_checkpoint.csv`, `output/_tmp_param_sweep*/`, `output/wmrl_m*_*.{csv,json}`, `output/v1/`, `output/bayesian/23.1_smoke/`)
  - `config.py` mkdir loop entries: `OUTPUT_VERSION_DIR`, `FIGURES_VERSION_DIR`
  - `config.py` DataParams simulated-data paths under `OUTPUT_VERSION_DIR`

### Blockers / Concerns

- **None for the Wave 2 fan-out.** Both 31-02 and 31-03 are cleared to run in parallel.
- **Pre-existing `scripts.legacy` ImportError** will propagate into Wave D if unaddressed; flagged here so plan 31-04 picks it up (either add `scripts/legacy/__init__.py` or retire `tests/test_wmrl_exploration.py`).

---
*Phase: 31-final-package-restructure*
*Completed: 2026-04-24*
