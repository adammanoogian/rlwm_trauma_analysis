---
wave: 2
depends_on: [28-01]
files_modified:
  - scripts/data_processing/01_parse_raw_data.py  (git mv from scripts/01_parse_raw_data.py)
  - scripts/data_processing/02_create_collated_csv.py  (git mv)
  - scripts/data_processing/03_create_task_trials_csv.py  (git mv)
  - scripts/data_processing/04_create_summary_csv.py  (git mv)
  - scripts/data_processing/__init__.py  (new, empty)
autonomous: true
---

# 28-02 Group Data-Processing Scripts (01–04)

## Goal

Move the four data-processing pipeline scripts into a `scripts/data_processing/` subdirectory via `git mv` (preserving history), keeping their numeric prefixes for ordering, without changing any code inside them.

## Must Haves

- [ ] `scripts/data_processing/` directory exists with `__init__.py` (empty or docstring-only).
- [ ] All four files moved via `git mv` (history preserved):
  - `scripts/data_processing/01_parse_raw_data.py`
  - `scripts/data_processing/02_create_collated_csv.py`
  - `scripts/data_processing/03_create_task_trials_csv.py`
  - `scripts/data_processing/04_create_summary_csv.py`
- [ ] Original `scripts/0{1,2,3,4}_*.py` files no longer exist at the old paths.
- [ ] Any internal cross-imports between the four scripts (if present) still resolve — verified by running each with `--help` or by grep-sweeping `from scripts.01_` / `from scripts.02_` style refs (unlikely to exist because numeric-prefixed modules aren't importable as Python identifiers; these are CLI scripts).
- [ ] No SLURM files reference these scripts (verified by grep on `cluster/`).
- [ ] `pytest scripts/fitting/tests/test_v4_closure.py` still passes 3/3.
- [ ] Atomic commit: `refactor(28-02): group data-processing scripts 01-04 under scripts/data_processing/`.

## Tasks

<tasks>
  <task id="1">
    <title>Pre-flight: confirm no cluster or cross-script references</title>
    <detail>Run grep across the repo for any reference to the old paths:
      - `grep -rn "scripts/01_parse_raw_data" . --include="*.sh" --include="*.slurm" --include="*.py" --include="*.md"`
      - Repeat for `02_create_collated_csv`, `03_create_task_trials_csv`, `04_create_summary_csv`.
      If any reference appears in code (not docs), add it to the update list. Documentation references are handled by plan 28-11 (REFAC-12) so leave for now.</detail>
  </task>

  <task id="2">
    <title>Create destination directory</title>
    <detail>`mkdir -p scripts/data_processing/` then create `scripts/data_processing/__init__.py` as an empty file. Commit together with the moves (tasks 3–4).</detail>
  </task>

  <task id="3">
    <title>Move the four scripts via git mv</title>
    <detail>
      - `git mv scripts/01_parse_raw_data.py scripts/data_processing/01_parse_raw_data.py`
      - `git mv scripts/02_create_collated_csv.py scripts/data_processing/02_create_collated_csv.py`
      - `git mv scripts/03_create_task_trials_csv.py scripts/data_processing/03_create_task_trials_csv.py`
      - `git mv scripts/04_create_summary_csv.py scripts/data_processing/04_create_summary_csv.py`</detail>
  </task>

  <task id="4">
    <title>Update any in-code references found in task 1</title>
    <detail>If task 1 surfaced any code-level refs (e.g., subprocess calls, imports), update them to the new paths. Skip documentation refs — plan 28-11 handles those.</detail>
  </task>

  <task id="5">
    <title>Smoke-test each script's --help</title>
    <detail>Run `python scripts/data_processing/01_parse_raw_data.py --help` (and similarly for 02–04) to confirm no import errors from the new location. These scripts use relative imports sparingly; a --help exit-0 is sufficient.</detail>
  </task>

  <task id="6">
    <title>Atomic commit</title>
    <detail>`refactor(28-02): group data-processing scripts 01-04 under scripts/data_processing/`. Body notes git-mv history preservation and zero content changes.</detail>
  </task>
</tasks>

## Verification

```bash
# New layout exists
test -f scripts/data_processing/__init__.py
test -f scripts/data_processing/01_parse_raw_data.py
test -f scripts/data_processing/02_create_collated_csv.py
test -f scripts/data_processing/03_create_task_trials_csv.py
test -f scripts/data_processing/04_create_summary_csv.py

# Old paths gone
test ! -f scripts/01_parse_raw_data.py
test ! -f scripts/02_create_collated_csv.py
test ! -f scripts/03_create_task_trials_csv.py
test ! -f scripts/04_create_summary_csv.py

# History preserved
git log --follow --oneline scripts/data_processing/01_parse_raw_data.py | head -3

# Smoke
python scripts/data_processing/01_parse_raw_data.py --help
python scripts/data_processing/04_create_summary_csv.py --help

# Closure invariants
pytest scripts/fitting/tests/test_v4_closure.py -v
```

## Requirement IDs

Closes: **REFAC-03**.
