---
phase: 23
plan: "01"
subsystem: tech-debt
tags: [legacy-deletion, guard-test, CLEAN-01, pytest, scripts.fitting.legacy]
dependency-graph:
  requires: []
  provides:
    - "scripts/fitting/legacy/ deleted from repo tree"
    - "scripts/fitting/tests/test_no_legacy_imports.py — CLEAN-01 CI guard"
  affects:
    - "Phase 24 cold-start cluster run (no latent reintroduction surface)"
tech-stack:
  added: []
  patterns:
    - "grep-invariant pytest guard with self-path exclusion"
key-files:
  created:
    - scripts/fitting/tests/test_no_legacy_imports.py
  modified: []
  deleted:
    - scripts/fitting/legacy/fit_to_data.py
    - scripts/fitting/legacy/numpyro_models.py
    - scripts/fitting/legacy/pymc_models.py
    - scripts/fitting/legacy/pymc_models_functional.py
decisions:
  - "Guard test excludes its own path (_SELF_PATH) from the import scan to prevent false-positive violations from docstrings that document the forbidden pattern."
metrics:
  duration: "~24 minutes"
  completed: "2026-04-19"
---

# Phase 23 Plan 01: Delete scripts/fitting/legacy/ — CLEAN-01 Summary

**One-liner:** Deleted 4 quarantined v3.0-era pymc/numpyro/fit_to_data modules (~2,073 lines) and installed pytest guard `test_no_legacy_imports.py` that CI-fails on reintroduction.

## Requirements Closed

- **CLEAN-01**: `scripts/fitting/legacy/` deleted; zero live imports; pytest guard installed.

## Deliverables

| Artifact | Description |
|---|---|
| `scripts/fitting/tests/test_no_legacy_imports.py` | Pytest guard: asserts directory non-existence + zero live legacy imports in `scripts/` |
| `scripts/fitting/legacy/` (deleted) | 4 v3.0-era modules removed (~2,073 lines across fit_to_data.py, numpyro_models.py, pymc_models.py, pymc_models_functional.py) |

## Lines Deleted

| File | Lines |
|---|---|
| `fit_to_data.py` | 443 |
| `numpyro_models.py` | 741 |
| `pymc_models.py` | 508 |
| `pymc_models_functional.py` | 381 |
| **Total** | **2,073** |

## Commits

| Task | Commit | Description |
|---|---|---|
| Task 1 | `3b391c2` | `chore(tech-debt): add guard test for scripts.fitting.legacy imports` |
| Task 2 | `15456e0` | `chore(tech-debt): delete scripts/fitting/legacy/ directory` |

Note: the legacy directory itself was physically removed in commit `f0d4e60` (CLEAN-03 guard, earlier in the same planning session). Task 2 commit finalises the guard test with the self-path exclusion fix and confirms all verification criteria.

## Verification Results

```
grep -rn "from scripts.fitting.legacy" scripts/   # zero matches outside guard file
ls scripts/fitting/legacy/ 2>/dev/null             # dir does not exist (exit 2)
python -m pytest scripts/fitting/tests/test_no_legacy_imports.py -v  # 1 passed
python -m pytest scripts/fitting/tests/ (20-test subset) -v           # 20 passed
python validation/check_v4_closure.py --milestone v4.0               # EXIT 0 (5/5)
```

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Guard test docstrings triggered false-positive import scan violations**

- **Found during:** Task 2 (first pytest run after directory deletion)
- **Issue:** The test module docstring at lines 5, 12, 13, 86 contains the literal string `from scripts.fitting.legacy` as documentation. The import scanner matched these lines, causing the guard to fail against its own file.
- **Fix:** Added `_SELF_PATH = Path(__file__).resolve()` constant and `if py_file.resolve() == _SELF_PATH: continue` skip in the scan loop to unconditionally exclude the guard file from scanning.
- **Files modified:** `scripts/fitting/tests/test_no_legacy_imports.py`
- **Commit:** `15456e0`

## Next Phase Readiness

- Phase 24 cold-start cluster run (`bash cluster/21_submit_pipeline.sh`) has zero latent reintroduction surface from v3.0-era legacy code.
- No blockers from this plan.
