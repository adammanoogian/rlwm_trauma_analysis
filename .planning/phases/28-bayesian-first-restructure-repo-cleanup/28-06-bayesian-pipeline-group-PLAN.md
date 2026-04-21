---
wave: 3
depends_on: [28-01, 28-05]
files_modified:
  - scripts/bayesian_pipeline/21_run_prior_predictive.py  (git mv)
  - scripts/bayesian_pipeline/21_run_bayesian_recovery.py  (git mv)
  - scripts/bayesian_pipeline/21_fit_baseline.py  (git mv)
  - scripts/bayesian_pipeline/21_baseline_audit.py  (git mv)
  - scripts/bayesian_pipeline/21_compute_loo_stacking.py  (git mv)
  - scripts/bayesian_pipeline/21_fit_with_l2.py  (git mv)
  - scripts/bayesian_pipeline/21_scale_audit.py  (git mv)
  - scripts/bayesian_pipeline/21_model_averaging.py  (git mv)
  - scripts/bayesian_pipeline/21_manuscript_tables.py  (git mv)
  - scripts/bayesian_pipeline/__init__.py  (new, empty)
  - cluster/21_1_prior_predictive.slurm
  - cluster/21_2_recovery.slurm
  - cluster/21_2_recovery_aggregate.slurm
  - cluster/21_3_fit_baseline.slurm
  - cluster/21_4_baseline_audit.slurm
  - cluster/21_5_loo_stacking_bms.slurm
  - cluster/21_6_dispatch_l2.slurm
  - cluster/21_6_fit_with_l2.slurm
  - cluster/21_7_scale_audit.slurm
  - cluster/21_8_model_averaging.slurm
  - cluster/21_9_manuscript_tables.slurm
autonomous: true
---

# 28-06 Group Phase 21 Bayesian Pipeline Scripts

## Goal

Move all nine `21_*.py` Bayesian pipeline scripts into `scripts/bayesian_pipeline/` via `git mv`, and update the 11 SLURM job files under `cluster/21_*.slurm` that invoke them so the `python scripts/21_*.py` calls resolve to the new paths. The `cluster/21_submit_pipeline.sh` orchestrator is NOT modified — it submits SLURM files, not Python scripts directly (per 28-RESEARCH.md §"Cluster orchestrator map").

Depends on 28-05 because `21_manuscript_tables.py` was edited there (subprocess path to `scripts/post_mle/18_bayesian_level2_effects.py`); moving it here must preserve that edit.

## Must Haves

- [ ] `scripts/bayesian_pipeline/` directory exists with `__init__.py`.
- [ ] All nine `21_*.py` files moved via `git mv`:
  - `21_run_prior_predictive.py`, `21_run_bayesian_recovery.py`, `21_fit_baseline.py`, `21_baseline_audit.py`, `21_compute_loo_stacking.py`, `21_fit_with_l2.py`, `21_scale_audit.py`, `21_model_averaging.py`, `21_manuscript_tables.py`.
- [ ] Original `scripts/21_*.py` paths no longer exist.
- [ ] All 11 `cluster/21_*.slurm` job files updated to call `python scripts/bayesian_pipeline/21_*.py` instead of `python scripts/21_*.py`. (Count = 11 because step 21.2 has two SLURM files and step 21.6 has two SLURM files.)
- [ ] Stale `13_bayesian_m6b.slurm` pattern references in 4 SLURM files' comment lines (`cluster/21_1_prior_predictive.slurm:36`, `cluster/21_2_recovery.slurm:46`, `cluster/21_2_recovery_aggregate.slurm:41`, `cluster/21_3_fit_baseline.slurm:54`) rewritten to reference the parameterized consolidated Bayesian template, so Plan 28-08's grep invariant does not fire on them as false positives. Verify: `grep -rn "13_bayesian_m6b" cluster/21_*.slurm` returns zero matches.
- [ ] `cluster/21_submit_pipeline.sh` unchanged (it `sbatch`es SLURM files; the afterok chain and model-loop structure is preserved).
- [ ] `scripts/bayesian_pipeline/21_manuscript_tables.py` still contains the subprocess reference to `scripts/post_mle/18_bayesian_level2_effects.py` from plan 28-05 (verify post-move).
- [ ] `pytest scripts/fitting/tests/test_v4_closure.py` still passes 3/3.
- [ ] Atomic commit: `refactor(28-06): group Phase 21 Bayesian pipeline scripts under scripts/bayesian_pipeline/ + update 11 SLURM paths`.

## Tasks

<tasks>
  <task id="1">
    <title>Pre-flight: inventory exact SLURM invocations</title>
    <detail>For each `cluster/21_*.slurm` file, grep for `python scripts/21_` and record the exact line(s) to be updated. Per 28-RESEARCH.md §"Cluster orchestrator map", the mapping is:
      - `21_1_prior_predictive.slurm` → `21_run_prior_predictive.py`
      - `21_2_recovery.slurm` + `21_2_recovery_aggregate.slurm` → `21_run_bayesian_recovery.py`
      - `21_3_fit_baseline.slurm` → `21_fit_baseline.py`
      - `21_4_baseline_audit.slurm` → `21_baseline_audit.py`
      - `21_5_loo_stacking_bms.slurm` → `21_compute_loo_stacking.py`
      - `21_6_dispatch_l2.slurm` + `21_6_fit_with_l2.slurm` → `21_fit_with_l2.py`
      - `21_7_scale_audit.slurm` → `21_scale_audit.py`
      - `21_8_model_averaging.slurm` → `21_model_averaging.py`
      - `21_9_manuscript_tables.slurm` → `21_manuscript_tables.py`</detail>
  </task>

  <task id="2">
    <title>Create destination + __init__.py</title>
    <detail>`mkdir -p scripts/bayesian_pipeline/` with empty `__init__.py`.</detail>
  </task>

  <task id="3">
    <title>git mv the nine 21_*.py scripts</title>
    <detail>
      - `git mv scripts/21_run_prior_predictive.py scripts/bayesian_pipeline/21_run_prior_predictive.py`
      - `git mv scripts/21_run_bayesian_recovery.py scripts/bayesian_pipeline/21_run_bayesian_recovery.py`
      - `git mv scripts/21_fit_baseline.py scripts/bayesian_pipeline/21_fit_baseline.py`
      - `git mv scripts/21_baseline_audit.py scripts/bayesian_pipeline/21_baseline_audit.py`
      - `git mv scripts/21_compute_loo_stacking.py scripts/bayesian_pipeline/21_compute_loo_stacking.py`
      - `git mv scripts/21_fit_with_l2.py scripts/bayesian_pipeline/21_fit_with_l2.py`
      - `git mv scripts/21_scale_audit.py scripts/bayesian_pipeline/21_scale_audit.py`
      - `git mv scripts/21_model_averaging.py scripts/bayesian_pipeline/21_model_averaging.py`
      - `git mv scripts/21_manuscript_tables.py scripts/bayesian_pipeline/21_manuscript_tables.py`</detail>
  </task>

  <task id="4">
    <title>Update 11 SLURM files' python invocations</title>
    <detail>For each SLURM file in the table from task 1, replace `python scripts/21_*.py` with `python scripts/bayesian_pipeline/21_*.py`. Preserve all other flags and SBATCH directives. Use `Edit` with the exact old string to avoid accidental broader changes.</detail>
  </task>

  <task id="4b">
    <title>Clean stale `13_bayesian_m6b.slurm` comment references in 4 SLURM files</title>
    <detail>During the SLURM edit pass, also update the "matches ...13_bayesian_m6b.slurm pattern" comment lines so Plan 28-08's grep invariant `grep -rn "13_bayesian_m[1-6]" cluster/` does not hit them as false positives (the 6 per-model templates are deleted in Plan 28-08; these comment lines become stale references). Rewrite each to mention the parameterized consolidated template instead:
      - `cluster/21_1_prior_predictive.slurm` line 36: `# --- Environment Setup (matches 13_bayesian_m6b.slurm pattern) ---` → `# --- Environment Setup (matches Bayesian choice-only template pattern) ---`
      - `cluster/21_2_recovery.slurm` line 46: same substitution
      - `cluster/21_2_recovery_aggregate.slurm` line 41: same substitution
      - `cluster/21_3_fit_baseline.slurm` line 54: `# --- Environment Setup (matches cluster/13_bayesian_m6b.slurm pattern) ---` → `# --- Environment Setup (matches Bayesian choice-only template pattern) ---`
      These are purely comment-line edits — no semantic change. Verify afterward with `grep -rn "13_bayesian_m6b" cluster/21_*.slurm` → zero matches.</detail>
  </task>

  <task id="5">
    <title>Verify 21_submit_pipeline.sh is untouched</title>
    <detail>Re-read `cluster/21_submit_pipeline.sh` and confirm it only references `cluster/21_*.slurm` (SLURM files), never `scripts/21_*.py` directly. If any direct script ref exists (unlikely), update it to the new path and note in commit body.</detail>
  </task>

  <task id="6">
    <title>Verify 21_manuscript_tables.py subprocess edit survived git mv</title>
    <detail>`grep -n "post_mle/18_bayesian_level2_effects" scripts/bayesian_pipeline/21_manuscript_tables.py` — must return a match (the edit from plan 28-05 was preserved).</detail>
  </task>

  <task id="7">
    <title>Smoke-test each script's --help</title>
    <detail>Run `--help` for each of the nine moved scripts to confirm no import errors from the new location (they import from `scripts.fitting.*` and `rlwm.fitting.*` — both should resolve).</detail>
  </task>

  <task id="8">
    <title>Atomic commit</title>
    <detail>`refactor(28-06): group Phase 21 Bayesian pipeline scripts under scripts/bayesian_pipeline/ + update 11 SLURM paths`. Body lists the 9 moved scripts and the 11 SLURM file updates.</detail>
  </task>
</tasks>

## Verification

```bash
# New layout
test -f scripts/bayesian_pipeline/__init__.py
for f in 21_run_prior_predictive 21_run_bayesian_recovery 21_fit_baseline 21_baseline_audit 21_compute_loo_stacking 21_fit_with_l2 21_scale_audit 21_model_averaging 21_manuscript_tables; do
  test -f scripts/bayesian_pipeline/${f}.py
  test ! -f scripts/${f}.py
done

# All 11 SLURM files updated
grep -c "scripts/bayesian_pipeline/21_" cluster/21_1_prior_predictive.slurm \
  cluster/21_2_recovery.slurm cluster/21_2_recovery_aggregate.slurm \
  cluster/21_3_fit_baseline.slurm cluster/21_4_baseline_audit.slurm \
  cluster/21_5_loo_stacking_bms.slurm cluster/21_6_dispatch_l2.slurm \
  cluster/21_6_fit_with_l2.slurm cluster/21_7_scale_audit.slurm \
  cluster/21_8_model_averaging.slurm cluster/21_9_manuscript_tables.slurm

# No stale refs in cluster/
! grep -rn "python scripts/21_" cluster/ --include="*.slurm" --include="*.sh" | grep -v bayesian_pipeline

# Subprocess edit from 28-05 survived
grep -n "post_mle/18_bayesian_level2_effects" scripts/bayesian_pipeline/21_manuscript_tables.py

# Smoke
python scripts/bayesian_pipeline/21_compute_loo_stacking.py --help
python scripts/bayesian_pipeline/21_manuscript_tables.py --help

pytest scripts/fitting/tests/test_v4_closure.py -v
```

## Requirement IDs

Closes: **REFAC-07**.
