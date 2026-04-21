---
wave: 2
depends_on: [28-01]
files_modified:
  - scripts/behavioral/05_summarize_behavioral_data.py  (git mv from scripts/05_summarize_behavioral_data.py)
  - scripts/behavioral/06_visualize_task_performance.py  (git mv)
  - scripts/behavioral/07_analyze_trauma_groups.py  (git mv)
  - scripts/behavioral/08_run_statistical_analyses.py  (git mv)
  - scripts/behavioral/__init__.py  (new, empty)
autonomous: true
---

# 28-03 Group Behavioral-Analysis Scripts (05–08)

## Goal

Move the four behavioral-analysis pipeline scripts into a `scripts/behavioral/` subdirectory via `git mv`, keeping numeric prefixes and content unchanged.

## Must Haves

- [ ] `scripts/behavioral/` directory exists with `__init__.py` (empty or docstring-only).
- [ ] All four files moved via `git mv`:
  - `scripts/behavioral/05_summarize_behavioral_data.py`
  - `scripts/behavioral/06_visualize_task_performance.py`
  - `scripts/behavioral/07_analyze_trauma_groups.py`
  - `scripts/behavioral/08_run_statistical_analyses.py`
- [ ] Original `scripts/0{5,6,7,8}_*.py` paths no longer exist.
- [ ] `scripts/behavioral/08_run_statistical_analyses.py` `from scripts.utils.statistical_tests import ...` still resolves correctly from the new location (pytest `pythonpath = .` means this should work, but verify).
- [ ] `pytest scripts/fitting/tests/test_v4_closure.py` still passes 3/3.
- [ ] Atomic commit: `refactor(28-03): group behavioral scripts 05-08 under scripts/behavioral/`.

## Tasks

<tasks>
  <task id="1">
    <title>Pre-flight: grep for refs to old paths</title>
    <detail>Run `grep -rn "scripts/05_summarize_behavioral_data\|scripts/06_visualize_task_performance\|scripts/07_analyze_trauma_groups\|scripts/08_run_statistical_analyses" . --include="*.py" --include="*.sh" --include="*.slurm"`. Flag any code-level matches for update.</detail>
  </task>

  <task id="2">
    <title>Create destination directory + __init__.py</title>
    <detail>`mkdir -p scripts/behavioral/` and create empty `scripts/behavioral/__init__.py`.</detail>
  </task>

  <task id="3">
    <title>git mv the four scripts</title>
    <detail>
      - `git mv scripts/05_summarize_behavioral_data.py scripts/behavioral/05_summarize_behavioral_data.py`
      - `git mv scripts/06_visualize_task_performance.py scripts/behavioral/06_visualize_task_performance.py`
      - `git mv scripts/07_analyze_trauma_groups.py scripts/behavioral/07_analyze_trauma_groups.py`
      - `git mv scripts/08_run_statistical_analyses.py scripts/behavioral/08_run_statistical_analyses.py`</detail>
  </task>

  <task id="4">
    <title>Verify scripts.utils.statistical_tests still imports from new location</title>
    <detail>Per 28-RESEARCH.md §"Import dependency snapshot", `08_run_statistical_analyses.py` imports `from scripts.utils.statistical_tests import ...`. Because `pythonpath = .` is set in `pytest.ini`, this should still resolve. Verify by running `python scripts/behavioral/08_run_statistical_analyses.py --help` (or a minimal smoke invocation).</detail>
  </task>

  <task id="5">
    <title>Update any in-code references found in task 1</title>
    <detail>Edit any surfaced matches. Leave docs refs for plan 28-11.</detail>
  </task>

  <task id="6">
    <title>Smoke-test each script's --help</title>
    <detail>Run `python scripts/behavioral/0{5,6,7,8}_*.py --help` to confirm no import errors.</detail>
  </task>

  <task id="7">
    <title>Atomic commit</title>
    <detail>`refactor(28-03): group behavioral scripts 05-08 under scripts/behavioral/`.</detail>
  </task>
</tasks>

## Verification

```bash
test -f scripts/behavioral/__init__.py
test -f scripts/behavioral/05_summarize_behavioral_data.py
test -f scripts/behavioral/06_visualize_task_performance.py
test -f scripts/behavioral/07_analyze_trauma_groups.py
test -f scripts/behavioral/08_run_statistical_analyses.py

test ! -f scripts/05_summarize_behavioral_data.py
test ! -f scripts/06_visualize_task_performance.py
test ! -f scripts/07_analyze_trauma_groups.py
test ! -f scripts/08_run_statistical_analyses.py

git log --follow --oneline scripts/behavioral/08_run_statistical_analyses.py | head -3
python scripts/behavioral/08_run_statistical_analyses.py --help
pytest scripts/fitting/tests/test_v4_closure.py -v
```

## Requirement IDs

Closes: **REFAC-04**.
