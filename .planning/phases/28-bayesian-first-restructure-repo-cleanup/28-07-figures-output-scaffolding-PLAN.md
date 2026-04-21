---
wave: 2
depends_on: [28-01]
files_modified:
  - output/legacy/v1/  (git mv from output/v1/)
  - output/legacy/_tmp_param_sweep/  (git mv from output/_tmp_param_sweep/)
  - output/legacy/_tmp_param_sweep_wmrl/  (git mv from output/_tmp_param_sweep_wmrl/)
  - output/legacy/modelling_base_models/  (git mv from output/modelling_base_models/)
  - output/legacy/base_model_analysis/  (git mv from output/base_model_analysis/)
  - figures/legacy/v1/  (git mv from figures/v1/)
  - figures/legacy/feedback_learning/  (git mv from figures/feedback_learning/)
  - figures/21_bayesian/.gitkeep  (new)
  - output/bayesian/21_baseline/.gitkeep  (new)
  - output/bayesian/21_l2/.gitkeep  (new — may already exist)
  - output/bayesian/21_recovery/.gitkeep  (new)
  - output/bayesian/21_prior_predictive/.gitkeep  (new)
  - output/bayesian/manuscript/.gitkeep  (new)
  - .gitignore  (if new paths need exclusions)
autonomous: true
---

# 28-07 Figures + Output Scaffolding (Phase 24-ready, paper.qmd-preserving)

## Goal

Scaffold `figures/` and `output/` for the Bayesian-first pipeline without breaking any existing `paper.qmd` relative-path references. This means: (a) move clearly-legacy subdirectories into `output/legacy/` and `figures/legacy/` via `git mv`, (b) pre-create the empty-but-canonical output paths that Phase 24's `cluster/21_submit_pipeline.sh` will write to (`.gitkeep` files so the dirs are tracked), and (c) leave in place every subdirectory that `paper.qmd` currently reads from.

**Non-goal:** No reorganization of `output/mle/`, `output/model_comparison/`, `output/trauma_groups/`, or `output/bayesian/level2/` — these are load-bearing paths referenced by `paper.qmd` (per 28-RESEARCH.md §Q6 and this planner's direct grep). Touching them would force an atomic paper.qmd path-update which is explicitly deferred to 28-10's graceful-fallback pattern.

## Must Haves

- [ ] `output/legacy/` directory exists and contains the five moved legacy subdirectories:
  - `output/legacy/v1/`
  - `output/legacy/_tmp_param_sweep/`
  - `output/legacy/_tmp_param_sweep_wmrl/`
  - `output/legacy/modelling_base_models/`
  - `output/legacy/base_model_analysis/`
- [ ] `figures/legacy/` directory exists and contains:
  - `figures/legacy/v1/` (currently empty in original location, but preserve the path)
  - `figures/legacy/feedback_learning/`
- [ ] Pre-created empty scaffold dirs with `.gitkeep` for Phase 24's outputs:
  - `figures/21_bayesian/.gitkeep`
  - `output/bayesian/21_baseline/.gitkeep`
  - `output/bayesian/21_l2/.gitkeep` (may already be an empty tracked dir — confirm)
  - `output/bayesian/21_recovery/.gitkeep`
  - `output/bayesian/21_prior_predictive/.gitkeep`
  - `output/bayesian/manuscript/.gitkeep`
- [ ] **Unchanged** (load-bearing for paper.qmd): `output/mle/`, `output/model_comparison/`, `output/trauma_groups/`, `output/bayesian/level2/`, `output/regressions/`, `output/ppc/`, `output/recovery/`, `output/behavioral_summary/`, `output/descriptives/`, `output/statistical_analyses/`, `output/supplementary_materials/`, `output/trauma_scale_analysis/`, `output/results_text/`, `output/parameter_exploration/`, `output/model_performance/`, and `figures/ppc/`, `figures/recovery/`, `figures/regressions/`, `figures/mle_trauma_analysis/`, `figures/model_comparison/`, `figures/trauma_groups/`, `figures/behavioral_analysis/`, `figures/behavioral_summary/`, `figures/trauma_scale_analysis/`.
- [ ] Top-level CSVs under `output/*.csv` NOT moved (15 files; paper.qmd may reference some; out of scope for this plan).
- [ ] `grep -n "\.\./output/" manuscript/paper.qmd` returns the SAME paths as before this plan (no paper.qmd changes required).
- [ ] `quarto render manuscript/paper.qmd` still runs to completion after this plan (verified manually; plan 28-10 handles Bayesian-first reordering separately).
- [ ] `pytest scripts/fitting/tests/test_v4_closure.py` still passes 3/3.
- [ ] Atomic commit: `refactor(28-07): move legacy output/figures dirs to legacy/; scaffold Phase 24 output paths with .gitkeep`.

## Tasks

<tasks>
  <task id="1">
    <title>Capture the paper.qmd path set as a baseline invariant</title>
    <detail>Before any moves, run `grep -n "\.\./output/\|\.\./figures/" manuscript/paper.qmd > /tmp/paper_paths_before.txt`. Keep this file as a scratch baseline — after the plan, repeat the grep and `diff` against the baseline to confirm zero path references changed.</detail>
  </task>

  <task id="2">
    <title>Create output/legacy/ and figures/legacy/ containers</title>
    <detail>`mkdir -p output/legacy/ figures/legacy/`. No `.gitkeep` needed (the moved subdirs below will populate them).</detail>
  </task>

  <task id="3">
    <title>git mv legacy output/ subdirs</title>
    <detail>
      - `git mv output/v1 output/legacy/v1`
      - `git mv output/_tmp_param_sweep output/legacy/_tmp_param_sweep`
      - `git mv output/_tmp_param_sweep_wmrl output/legacy/_tmp_param_sweep_wmrl`
      - `git mv output/modelling_base_models output/legacy/modelling_base_models`
      - `git mv output/base_model_analysis output/legacy/base_model_analysis`
      Per 28-RESEARCH.md §Q5, these are pre-refactor artifacts. If any of these dirs is referenced by paper.qmd (checked in task 1), STOP and flag.</detail>
  </task>

  <task id="4">
    <title>git mv legacy figures/ subdirs</title>
    <detail>
      - `git mv figures/v1 figures/legacy/v1` (the v1 dir is empty per 28-RESEARCH.md; git mv still works if it contains any tracked files; `rmdir` if completely untracked).
      - `git mv figures/feedback_learning figures/legacy/feedback_learning`</detail>
  </task>

  <task id="5">
    <title>Pre-create Phase 24 output scaffolding with .gitkeep</title>
    <detail>Create the six destination directories and add a `.gitkeep` file in each:
      - `figures/21_bayesian/.gitkeep`
      - `output/bayesian/21_baseline/.gitkeep`
      - `output/bayesian/21_l2/.gitkeep` (may already exist — confirm)
      - `output/bayesian/21_recovery/.gitkeep`
      - `output/bayesian/21_prior_predictive/.gitkeep`
      - `output/bayesian/manuscript/.gitkeep`
      These paths are referenced by Phase 24's 21_*.py scripts and by plan 28-10's graceful-fallback Quarto cells. Pre-creating them means Phase 24 can write without mkdir, and plan 28-10 can test for file-existence cleanly.</detail>
  </task>

  <task id="6">
    <title>Verify paper.qmd path references are unchanged</title>
    <detail>Run `grep -n "\.\./output/\|\.\./figures/" manuscript/paper.qmd > /tmp/paper_paths_after.txt` and `diff /tmp/paper_paths_before.txt /tmp/paper_paths_after.txt`. Diff must be empty. If not, some path unexpectedly moved — stop and investigate.</detail>
  </task>

  <task id="7">
    <title>Quarto render smoke test</title>
    <detail>If the user has Quarto installed locally, run `quarto render manuscript/paper.qmd` from repo root and confirm exit 0. If Quarto isn't locally available, document in the commit body that render verification deferred to plan 28-10.</detail>
  </task>

  <task id="8">
    <title>Atomic commit</title>
    <detail>`refactor(28-07): move legacy output/figures dirs to legacy/; scaffold Phase 24 output paths with .gitkeep`. Body lists the 7 moved subdirs and the 6 new .gitkeep scaffolds.</detail>
  </task>
</tasks>

## Verification

```bash
# Legacy moves
test -d output/legacy/v1
test -d output/legacy/_tmp_param_sweep
test -d output/legacy/_tmp_param_sweep_wmrl
test -d output/legacy/modelling_base_models
test -d output/legacy/base_model_analysis
test -d figures/legacy/v1
test -d figures/legacy/feedback_learning

test ! -d output/v1
test ! -d output/_tmp_param_sweep
test ! -d output/modelling_base_models
test ! -d output/base_model_analysis
test ! -d figures/feedback_learning

# Scaffold files exist
test -f figures/21_bayesian/.gitkeep
test -f output/bayesian/21_baseline/.gitkeep
test -f output/bayesian/21_recovery/.gitkeep
test -f output/bayesian/21_prior_predictive/.gitkeep
test -f output/bayesian/manuscript/.gitkeep

# Load-bearing paths unchanged
test -d output/mle
test -d output/model_comparison
test -d output/trauma_groups
test -d output/bayesian/level2

# paper.qmd path invariant — diff must be empty
grep -n "\\.\\./output/\\|\\.\\./figures/" manuscript/paper.qmd > /tmp/after.txt
diff /tmp/before.txt /tmp/after.txt  # (baseline captured in task 1)

pytest scripts/fitting/tests/test_v4_closure.py -v
```

## Requirement IDs

Closes: **REFAC-08**.
