---
phase: 24-cold-start-pipeline-execution
plan: 00
subsystem: infra
tags: [path-patch, ccds-canonical, phase31, preflight-gate, wave0, doc-drift]

# Dependency graph
requires:
  - phase: 23-tech-debt-sweep-pre-flight-cleanup
    provides: clean codebase + load-side validation wrapper (CLEAN-01..04 complete)
  - phase: 31-final-package-restructure
    provides: CCDS-canonical top-level layout (models/ reports/ tests/ data/ logs/)
provides:
  - doc-drift-patched ROADMAP.md (Phase 23.1 SC#7, Phase 24 SC#3, Plan 24-02 entry, Phase 26 SC#2)
  - doc-drift-patched REQUIREMENTS.md (EXEC-02, EXEC-03)
  - pinned CCDS audit path constants in tests/scientific/check_phase24_artifacts.py
  - pre-flight pytest gate green on local Windows dev (JAX 0.9.0 / NumPyro 0.19.0)
  - Wave 1 handoff: CCDS path contract enforced at doc layer
affects:
  - 24-01-PLAN.md (inherits the CCDS path contract + green pre-flight gate; must re-run gate on Monash M3 before sbatch)
  - 24-02-PLAN.md (inherits tests/scientific/check_phase24_artifacts.py PATH_CONSTANTS; must fill in check functions without introducing pre-Phase-31 fallback paths)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "PATH_CONSTANTS-pinned stub: module-level Path constants establish Wave 2 import surface before implementation; prevents path drift across plan boundaries"
    - "Phase 31 SC#8 guard: new .py files in tests/ must not contain 'output/bayesian' string literal (even in docstrings)"

key-files:
  created:
    - tests/scientific/check_phase24_artifacts.py
  modified:
    - .planning/ROADMAP.md
    - .planning/REQUIREMENTS.md

key-decisions:
  - "Stub placed at tests/scientific/check_phase24_artifacts.py (not validation/ — Phase 31 plan 31-04 consolidated validation/ into tests/scientific/; CCDS-canonical location)"
  - "REQUIREMENTS.md line 62 (REFAC-08 completed historical item) left as-is — it is a record of Phase 28 scaffolding work, not a live requirement row; modifying it would falsify history"
  - "Local pre-flight gate (Windows) substitutes for Monash M3 gate; Wave 1 MUST re-run on M3 before any sbatch call"
  - "logs/24-00_grep_audit.log and logs/24-00_preflight_pytest.log are gitignored by project convention (logs/* in .gitignore); audit results documented in SUMMARY instead"

patterns-established:
  - "Wave 0 pre-flight: doc-drift patch before cold-start ensures audit script cannot drift to legacy paths"
  - "Stub-first: pin PATH_CONSTANTS in Wave 0, implement check functions in Wave 2 — eliminates drift window between pipeline run and audit"

# Metrics
duration: 9min
completed: 2026-04-24
---

# Phase 24 Plan 00: Wave 0 CCDS Path Reconciliation Summary

**Wave 0 reconciled ROADMAP / REQUIREMENTS doc-drift to Phase 31 CCDS-canonical paths, pinned `tests/scientific/check_phase24_artifacts.py` PATH_CONSTANTS, and confirmed both pre-flight pytest gates green locally (9/9 numpyro_models_2cov + 56/56 phase29_structure); cold-start baseline clean for Wave 1 canary submission.**

## Performance

- **Duration:** 9 min
- **Started:** 2026-04-24T19:37:14Z
- **Completed:** 2026-04-24T19:45:58Z
- **Tasks:** 5 (Tasks 1–4 executed; Task 5 = this SUMMARY)
- **Files modified:** 3 (ROADMAP.md, REQUIREMENTS.md, tests/scientific/check_phase24_artifacts.py)

## Accomplishments

- Code layer (cluster/ + scripts/) confirmed already CCDS-canonical per Phase 31 closure — zero legacy-path matches in 3 grep sections
- Doc layer (ROADMAP + REQUIREMENTS) patched in 7 surgical edits: Phase 23.1 Goal + SC#7, Phase 24 SC#3, Plan 24-02 entry, Phase 26 SC#2, EXEC-02, EXEC-03
- `tests/scientific/check_phase24_artifacts.py` stub created with 10 pinned CCDS path constants; Wave 2 cannot drift to pre-Phase-31 layout
- Pre-flight gate green: 9/9 `test_numpyro_models_2cov.py` + 56/56 `test_v5_phase29_structure.py` (local Windows, JAX 0.9.0 / NumPyro 0.19.0)

## Task Commits

Each task was committed atomically:

1. **Task 1: Grep audit** — logs/24-00_grep_audit.log written to disk (gitignored by project convention); no git commit (nothing to stage in git-tracked files)
2. **Task 2: ROADMAP + REQUIREMENTS patches** — `363b224` (docs)
3. **Task 3: Stub check_phase24_artifacts.py** — `0f10412` (feat)
4. **Task 3+4 fix: docstring SC#8 guard fix + pre-flight gate** — `b55dce8` (fix)
5. **Task 5: SUMMARY** — (plan metadata commit below)

**Plan metadata:** pending (docs(24-00): complete wave 0 path reconciliation)

## Files Created/Modified

- `tests/scientific/check_phase24_artifacts.py` — CCDS path constants stub; Wave 2 import surface; NotImplementedError on invocation
- `.planning/ROADMAP.md` — 5 surgical edits (Phase 23.1 Goal + SC#7, Phase 24 SC#3, Plan 24-02 entry, Phase 26 SC#2)
- `.planning/REQUIREMENTS.md` — 2 surgical edits (EXEC-02 full artifact list, EXEC-03 execution log path)

## Grep Audit Finding

**Code layer (cluster/ + scripts/) is already CCDS-canonical per Phase 31 closure.**

Grep audit results from `logs/24-00_grep_audit.log` (file on disk, gitignored):

- `cluster/*.slurm + cluster/*.sh`: zero matches for `output/bayesian/(21_|manuscript|figures)` — CCDS-canonical
- `scripts/**/*.py`: zero matches for `output/bayesian/(21_|manuscript)` — CCDS-canonical
- `scripts/ + cluster/`: zero matches for `figures/21_bayesian` — CCDS-canonical

**Doc-layer drift hits enumerated by grep (the lines Wave 0 patched):**

ROADMAP.md:
- Line 372: Phase 23.1 Goal — `output/bayesian/21_*/` in cold-start endpoint description
- Line 385: Phase 23.1 SC#7 — `output/bayesian/21_*/` in output path contract
- Line 413: Phase 24 SC#3 — `output/bayesian/21_execution_log.md`
- Line 420: Plan 24-02 roadmap entry — `output/bayesian/21_execution_log.md`
- Line 452: Phase 26 SC#2 — `output/bayesian/figures/` in forest plot PNGs description

REQUIREMENTS.md:
- Line 18: EXEC-02 — multiple `output/bayesian/` artifact paths throughout
- Line 19: EXEC-03 — `output/bayesian/21_execution_log.md`

**Remaining grep hit (not patched — intentional):**
REQUIREMENTS.md line 62: REFAC-08 (completed `[x]`). This is a historical record of Phase 28 scaffolding work (`output/bayesian/21_*` dirs pre-created as `.gitkeep` scaffolding). Modifying it would falsify history. It is outside the scope of Wave 0's live-requirement patch target.

## Files Patched: Before → After

**ROADMAP.md — Phase 23.1 Goal (line 372):**
- Before: `...produces all expected artifacts in \`output/bayesian/21_*/\` produced by GPU MCMC`
- After: `...produces all expected artifacts in \`models/bayesian/21_*/\` produced by GPU MCMC`

**ROADMAP.md — Phase 23.1 SC#7 (line 385):**
- Before: `All GPU runs write to \`output/bayesian/21_*/\` (Phase 21 contract...`
- After: `All GPU runs write to \`models/bayesian/21_*/\` (Phase 31 CCDS-canonical contract...`

**ROADMAP.md — Phase 24 SC#3 (line 413):**
- Before: `\`output/bayesian/21_execution_log.md\` logs SLURM JobID...`
- After: `\`models/bayesian/21_execution_log.md\` logs SLURM JobID...`

**ROADMAP.md — Plan 24-02 entry (line 420):**
- Before: `...+ \`output/bayesian/21_execution_log.md\` from sacct...`
- After: `...+ \`models/bayesian/21_execution_log.md\` from sacct...`

**ROADMAP.md — Phase 26 SC#2 (line 452):**
- Before: `...PNGs in \`output/bayesian/figures/\`; referenced in \`paper.qmd\`...`
- After: `...PNGs in \`reports/figures/bayesian/21_bayesian/\`; referenced in \`paper.qmd\`...`
  (Also updated script reference: `scripts/18_bayesian_level2_effects.py` → `scripts/06_fit_analyses/07_bayesian_level2_effects.py`)

**REQUIREMENTS.md — EXEC-02 (line 18):**
- Before: `output/bayesian/21_prior_predictive/{model}_prior_sim.nc` (and all other `output/bayesian/` artifact paths), `output/bayesian/manuscript/{loo_stacking,rfx_bms,winner_betas}.{csv,md,tex}`
- After: `models/bayesian/21_prior_predictive/{model}_prior_sim.nc` (and `models/bayesian/` prefixes throughout), `reports/tables/model_comparison/{loo_stacking,rfx_bms,winner_betas}.{csv,md,tex}`, forest plot PNGs in `reports/figures/bayesian/21_bayesian/`

**REQUIREMENTS.md — EXEC-03 (line 19):**
- Before: `...logged per step in \`output/bayesian/21_execution_log.md\`...`
- After: `...logged per step in \`models/bayesian/21_execution_log.md\`...`

## PATH_CONSTANTS Pin

The following module-level constants are pinned in `tests/scientific/check_phase24_artifacts.py`. Wave 2 MUST re-use these constants — no string literals in check functions:

```python
MODELS_DIR: Path = Path("models/bayesian")
PRIOR_PREDICTIVE_DIR: Path = MODELS_DIR / "21_prior_predictive"
RECOVERY_DIR: Path = MODELS_DIR / "21_recovery"
BASELINE_DIR: Path = MODELS_DIR / "21_baseline"
L2_DIR: Path = MODELS_DIR / "21_l2"

TABLES_DIR: Path = Path("reports/tables/model_comparison")
FIGURES_DIR: Path = Path("reports/figures/bayesian/21_bayesian")

EXEC_LOG: Path = MODELS_DIR / "21_execution_log.md"
WINNERS_TXT: Path = BASELINE_DIR / "winners.txt"
CONVERGENCE_TABLE: Path = BASELINE_DIR / "convergence_table.csv"
WINNER_REPORT: Path = BASELINE_DIR / "winner_report.md"

BASELINE_MODELS: tuple[str, ...] = (
    "qlearning", "wmrl", "wmrl_m3", "wmrl_m5", "wmrl_m6a", "wmrl_m6b",
)
```

## Pre-flight Gate Result

**Gate 1: `tests/integration/test_numpyro_models_2cov.py -k "not slow"`**
- Result: **9 passed, 1 deselected in 6.09s**
- Tests: 3 acceptance × M3/M5/M6a + 3 backward-compat × M3/M5/M6a + 3 guard × M3/M5/M6a

**Gate 2: `tests/integration/test_v5_phase29_structure.py`**
- Result: **56 passed in 1.12s**
- Notable: One test would have failed (`test_phase31_no_legacy_output_paths[output/bayesian]`) before the docstring fix in the stub — auto-fixed per Rule 1

**Environment:** Windows 11 local dev, JAX 0.9.0, NumPyro 0.19.0
**M3 re-run required:** Wave 1 (24-01-PLAN.md) MUST re-run both gates on the Monash M3 login node (in `ds_env` or `rlwm_gpu` conda env) before any `sbatch` call. This is the CONTEXT.md §Submission strategy gate.

Note: Test file location changed. Original plan spec cited `scripts/fitting/tests/test_numpyro_models_2cov.py`. Phase 31 plan 31-04 moved it to `tests/integration/test_numpyro_models_2cov.py`. Wave 1 must use the CCDS-canonical path.

## Decisions Made

1. **Stub at `tests/scientific/`**: Phase 31 plan 31-04 consolidated `validation/` → `tests/scientific/`. The plan spec said `validation/check_phase24_artifacts.py`, but that directory no longer exists. Placed stub at `tests/scientific/check_phase24_artifacts.py` (CCDS-canonical location, mirrors `check_v4_closure.py` which is also in `tests/scientific/`).

2. **REQUIREMENTS.md line 62 left as-is**: REFAC-08 is a completed item (`[x]`) describing historical Phase 28 work. It references `output/bayesian/` dirs that REFAC-08 pre-created as scaffolding (and Phase 31 subsequently removed). Patching historical records falsifies history; the line is not a live requirement for Phase 24.

3. **Local pre-flight gate accepted**: M3 login node access is not available from this Windows dev box. Local gate passes with same JAX/NumPyro stack. Wave 1 owns the M3 re-run obligation.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Stub docstring triggered Phase 31 SC#8 legacy-path guard**

- **Found during:** Task 4 (pre-flight gate run)
- **Issue:** `tests/scientific/check_phase24_artifacts.py` docstring contained the string `output/bayesian` in the phrase "cannot drift back to the legacy `output/bayesian/...` layout"; this triggered `test_phase31_no_legacy_output_paths[output/bayesian]` in the Phase 31 SC#8 structure guard (which scans all `.py` files in `tests/`)
- **Fix:** Replaced "legacy `output/bayesian/...` layout" with "pre-Phase-31 layout" in docstring; removed `output/bayesian` literal string entirely from stub
- **Files modified:** `tests/scientific/check_phase24_artifacts.py`
- **Verification:** `test_phase31_no_legacy_output_paths[output/bayesian]` now passes; 56/56 structure guard green
- **Committed in:** `b55dce8` (fix(24-00) commit)

**2. [Rule 3 - Blocking] Stub path changed from `validation/` to `tests/scientific/`**

- **Found during:** Task 3 (creating stub)
- **Issue:** Plan spec said `validation/check_phase24_artifacts.py` but `validation/` directory does not exist (Phase 31 plan 31-04 consolidated it into `tests/scientific/`)
- **Fix:** Created stub at `tests/scientific/check_phase24_artifacts.py` (CCDS-canonical location)
- **Files modified:** `tests/scientific/check_phase24_artifacts.py` (created)
- **Verification:** Import works; PATH_CONSTANTS correct; NotImplementedError fires; structure guard passes
- **Committed in:** `0f10412` (feat(24-00) commit)

**3. [Rule 3 - Blocking] Pre-flight gate test path changed from `scripts/fitting/tests/` to `tests/integration/`**

- **Found during:** Task 4 (running pre-flight gate)
- **Issue:** Plan spec cited `scripts/fitting/tests/test_numpyro_models_2cov.py` but that path does not exist (Phase 31 plan 31-04 moved it to `tests/integration/`)
- **Fix:** Ran gate at `tests/integration/test_numpyro_models_2cov.py`; documented in SUMMARY and Wave 1 handoff
- **Verification:** 9 passed, 1 deselected in 6.09s
- **No code committed:** Path correction documented only

---

**Total deviations:** 3 (1 auto-fix Rule 1, 2 auto-fix Rule 3)
**Impact on plan:** All deviations arise from Phase 31 CCDS consolidation that moved files before Wave 0 was planned. No scope creep; plan deliverables fully met at CCDS-canonical paths.

## Issues Encountered

- `logs/24-00_grep_audit.log` and `logs/24-00_preflight_pytest.log` are gitignored (`logs/*` in `.gitignore` per Phase 31 plan 31-05 convention). Files written to disk for local reference but not tracked in git. Audit results captured in this SUMMARY instead.

## Note on CONTEXT.md Sequencing

CONTEXT.md Area 4 declared "Wave 0 precedes cold-start." This plan satisfies that. Wave 1 canary submission proceeds only AFTER this SUMMARY is committed. Per CONTEXT.md §Submission strategy: the M3 login-node pre-flight gate re-run is Wave 1's responsibility, not Wave 0's.

## Handoff to Wave 1 (24-01-PLAN.md)

Wave 1 inherits:

1. **Green pre-flight gate**: `tests/integration/test_numpyro_models_2cov.py -k "not slow"` — 9/9 green locally; re-run on M3 login node (in `ds_env` or `rlwm_gpu`) before any `sbatch` call. Note: test is now at `tests/integration/` not `scripts/fitting/tests/`.

2. **CCDS-canonical path contract enforced at the doc layer**: ROADMAP Phase 23.1 SC#7, Phase 24 SC#3, Plan 24-02 entry, Phase 26 SC#2, REQUIREMENTS EXEC-02, EXEC-03 all cite `models/bayesian/` + `reports/tables/model_comparison/` + `reports/figures/bayesian/21_bayesian/`. No ambiguity about where artifacts should land.

3. **Pinned audit-script expectations**: `tests/scientific/check_phase24_artifacts.py` PATH_CONSTANTS are the single-source-of-truth for Wave 2. Wave 1's artifact-existence assertions in the canary run should reference these same constants.

4. **Phase 31 SC#8 guard active**: New `.py` files added to `tests/` must not contain the string `output/bayesian` (or other Phase 31 legacy path patterns) — even in docstrings or comments.

Wave 2 (24-02-PLAN.md) fills in the check functions in `tests/scientific/check_phase24_artifacts.py` after the cold-start pipeline terminates.

## Next Phase Readiness

**Ready for Wave 1:**
- Doc layer aligned with CCDS code layer
- Pre-flight gates green locally
- PATH_CONSTANTS pinned to prevent drift

**Blockers / concerns:**
- Wave 1 MUST re-run `python -m pytest tests/integration/test_numpyro_models_2cov.py -v -k "not slow" --tb=short` on Monash M3 login node before sbatch
- Wave 1 should verify `tests/integration/test_v5_phase29_structure.py` still passes 56/56 on M3 after any environment setup
- REQUIREMENTS.md line 62 (REFAC-08 historical) still has legacy path strings — this is intentional (completed historical record); Wave 2 audit script must not interpret this line as a live expectation

---
*Phase: 24-cold-start-pipeline-execution*
*Completed: 2026-04-24*
