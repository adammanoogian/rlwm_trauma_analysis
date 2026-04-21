---
wave: 2
depends_on: [28-01]
files_modified:
  - scripts/post_mle/15_analyze_mle_by_trauma.py  (git mv)
  - scripts/post_mle/16_regress_parameters_on_scales.py  (git mv)
  - scripts/post_mle/17_analyze_winner_heterogeneity.py  (git mv)
  - scripts/post_mle/18_bayesian_level2_effects.py  (git mv)
  - scripts/post_mle/__init__.py  (new, empty)
  - scripts/21_manuscript_tables.py  (subprocess path update — atomic coupling with plan 28-06 IF 28-06 runs in same wave; otherwise update here)
autonomous: true
---

# 28-05 Group Post-MLE / Boundary Scripts (15–18)

## Goal

Move the four post-MLE / MLE-Bayesian-boundary scripts (15, 16, 17, 18) into `scripts/post_mle/` via `git mv`, and update the `21_manuscript_tables.py` subprocess call at line 746 that invokes `scripts/18_bayesian_level2_effects.py` so the rendering-backend reference resolves to the new path.

**Load-bearing note:** `18_bayesian_level2_effects.py` is treated as a rendering library by `21_manuscript_tables.py` via `subprocess.run(["python", "scripts/18_bayesian_level2_effects.py", ...])` (per 28-RESEARCH.md §Q2). The subprocess path MUST be updated in the same commit as the move, otherwise `21_manuscript_tables.py` will fail at runtime.

## Must Haves

- [ ] `scripts/post_mle/` directory exists with `__init__.py`.
- [ ] All four files moved via `git mv`:
  - `scripts/post_mle/15_analyze_mle_by_trauma.py`
  - `scripts/post_mle/16_regress_parameters_on_scales.py`
  - `scripts/post_mle/17_analyze_winner_heterogeneity.py`
  - `scripts/post_mle/18_bayesian_level2_effects.py`
- [ ] Original `scripts/1{5,6,7,8}_*.py` paths no longer exist.
- [ ] `scripts/21_manuscript_tables.py` line ~746 subprocess call updated from `scripts/18_bayesian_level2_effects.py` to `scripts/post_mle/18_bayesian_level2_effects.py`.
- [ ] No other subprocess/CLI caller of `18_bayesian_level2_effects.py` is orphaned (verified by grep).
- [ ] `pytest scripts/fitting/tests/test_v4_closure.py` still passes 3/3.
- [ ] Atomic commit: `refactor(28-05): group post-MLE scripts 15-18 under scripts/post_mle/ + update manuscript_tables subprocess path`.

## Tasks

<tasks>
  <task id="1">
    <title>Pre-flight: inventory all subprocess / CLI references to moved scripts</title>
    <detail>Run:
      - `grep -rn "scripts/15_analyze_mle_by_trauma\|scripts/16_regress_parameters_on_scales\|scripts/17_analyze_winner_heterogeneity\|scripts/18_bayesian_level2_effects" . --include="*.py" --include="*.sh" --include="*.slurm" --include="*.md"`
      Expected match: `scripts/21_manuscript_tables.py` line 746 subprocess call to 18. Any other code-level match must be updated in this plan. Documentation matches defer to plan 28-11.</detail>
  </task>

  <task id="2">
    <title>Create destination directory + __init__.py</title>
    <detail>`mkdir -p scripts/post_mle/` with empty `__init__.py`.</detail>
  </task>

  <task id="3">
    <title>git mv the four scripts</title>
    <detail>
      - `git mv scripts/15_analyze_mle_by_trauma.py scripts/post_mle/15_analyze_mle_by_trauma.py`
      - `git mv scripts/16_regress_parameters_on_scales.py scripts/post_mle/16_regress_parameters_on_scales.py`
      - `git mv scripts/17_analyze_winner_heterogeneity.py scripts/post_mle/17_analyze_winner_heterogeneity.py`
      - `git mv scripts/18_bayesian_level2_effects.py scripts/post_mle/18_bayesian_level2_effects.py`</detail>
  </task>

  <task id="4">
    <title>Update scripts/21_manuscript_tables.py subprocess call</title>
    <detail>Open `scripts/21_manuscript_tables.py` and find the `subprocess.run([..., "scripts/18_bayesian_level2_effects.py", ...])` call (around line 746 per 28-RESEARCH.md §Q2). Change the path argument to `scripts/post_mle/18_bayesian_level2_effects.py`. Do NOT modify any other arguments.
      NOTE: Plan 28-06 will later move 21_*.py files to `scripts/bayesian_pipeline/`. When that runs, the file will be at `scripts/bayesian_pipeline/21_manuscript_tables.py` but its internal subprocess path (now pointing to `scripts/post_mle/18_bayesian_level2_effects.py`) is unaffected because it uses an absolute-from-repo-root path. Confirm by re-reading the line after edit.</detail>
  </task>

  <task id="5">
    <title>Update any other in-code references found in task 1</title>
    <detail>Edit every surfaced match to use the new `scripts/post_mle/` path.</detail>
  </task>

  <task id="6">
    <title>Smoke-test each script's --help</title>
    <detail>Run `python scripts/post_mle/1{5,6,7,8}_*.py --help` to confirm no import errors from the new location.</detail>
  </task>

  <task id="7">
    <title>Atomic commit</title>
    <detail>`refactor(28-05): group post-MLE scripts 15-18 under scripts/post_mle/ + update manuscript_tables subprocess path`. Body notes that 18 is load-bearing as a rendering backend (not a standalone pipeline step).</detail>
  </task>
</tasks>

## Verification

```bash
test -f scripts/post_mle/__init__.py
test -f scripts/post_mle/15_analyze_mle_by_trauma.py
test -f scripts/post_mle/16_regress_parameters_on_scales.py
test -f scripts/post_mle/17_analyze_winner_heterogeneity.py
test -f scripts/post_mle/18_bayesian_level2_effects.py

test ! -f scripts/15_analyze_mle_by_trauma.py
test ! -f scripts/16_regress_parameters_on_scales.py
test ! -f scripts/17_analyze_winner_heterogeneity.py
test ! -f scripts/18_bayesian_level2_effects.py

# Subprocess ref updated
grep -n "post_mle/18_bayesian_level2_effects" scripts/21_manuscript_tables.py

# No stale refs
! grep -rn "scripts/18_bayesian_level2_effects" . --include="*.py" --include="*.sh" --include="*.slurm" | grep -v post_mle

# Smoke
python scripts/post_mle/15_analyze_mle_by_trauma.py --help
python scripts/post_mle/18_bayesian_level2_effects.py --help

pytest scripts/fitting/tests/test_v4_closure.py -v
```

## Requirement IDs

Closes: **REFAC-06**.
