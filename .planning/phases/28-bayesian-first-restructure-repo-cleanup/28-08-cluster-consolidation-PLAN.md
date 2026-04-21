---
wave: 4
depends_on: [28-06]
files_modified:
  - cluster/13_bayesian_choice_only.slurm  (new, parameterized)
  - cluster/13_bayesian_m1.slurm  (deleted)
  - cluster/13_bayesian_m2.slurm  (deleted)
  - cluster/13_bayesian_m3.slurm  (deleted)
  - cluster/13_bayesian_m5.slurm  (deleted)
  - cluster/13_bayesian_m6a.slurm  (deleted)
  - cluster/13_bayesian_m6b.slurm  (deleted)
  - cluster/21_submit_pipeline.sh  (update baseline-fit submission to use parameterized template)
  - cluster/21_3_fit_baseline.slurm  (potentially — check if it reuses these templates)
autonomous: true
---

# 28-08 Cluster SLURM Consolidation: 6 per-model → 1 parameterized template

## Goal

Collapse the six structurally-identical per-model Bayesian SLURM templates (`13_bayesian_m{1,2,3,5,6a,6b}.slurm`) into one parameterized `cluster/13_bayesian_choice_only.slurm` dispatched via `--export=MODEL=<name>,TIME=<HH:MM:SS>`, preserving M6b's 36h wall-time. Delete the six old templates outright. Update any orchestrator that currently submits them. Retain all 7 specialized templates (subscale, GPU, multigpu, permutation, pscan, pscan_smoke, fullybatched_smoke) unchanged.

## Must Haves

- [ ] `cluster/13_bayesian_choice_only.slurm` exists, is based on `cluster/13_bayesian_m3.slurm` as the canonical source, and uses env variables:
  - `MODEL="${MODEL:-wmrl_m3}"` for the `--model <MODEL>` CLI arg
  - `TIME="${TIME:-24:00:00}"` with SBATCH time directive either (a) sourced via `#SBATCH --time=$TIME` which does NOT work for SBATCH directives (SBATCH lines are parsed before shell expansion) → use approach (b): `--time` passed via `--export=ALL` AND the submitter specifies `--time=...` on the `sbatch` command line, OR (c) use a small bash dispatcher wrapper that constructs the sbatch command.
  - Recommended approach: keep the SBATCH directive as `#SBATCH --time=24:00:00` (default for all models except M6b), and document in the SLURM comment block that M6b must be submitted with an explicit `sbatch --time=36:00:00 --export=MODEL=wmrl_m6b cluster/13_bayesian_choice_only.slurm` override. This sidesteps the SBATCH-vs-shell-parse ordering problem cleanly.
  - `--job-name=bayesian_${MODEL}` using runtime shell interpolation (this DOES work in `#SBATCH` via the `--job-name=${MODEL}` pattern when submitted with `--export=ALL,MODEL=...` — verify with M3 test submit).
- [ ] The six old per-model templates deleted via `git rm`:
  - `cluster/13_bayesian_m1.slurm`, `13_bayesian_m2.slurm`, `13_bayesian_m3.slurm`, `13_bayesian_m5.slurm`, `13_bayesian_m6a.slurm`, `13_bayesian_m6b.slurm`.
- [ ] All 7 specialized templates retained unchanged:
  - `cluster/13_bayesian_m6b_subscale.slurm`, `13_bayesian_gpu.slurm`, `13_bayesian_multigpu.slurm`, `13_bayesian_permutation.slurm`, `13_bayesian_pscan.slurm`, `13_bayesian_pscan_smoke.slurm`, `13_bayesian_fullybatched_smoke.slurm`.
- [ ] `cluster/21_submit_pipeline.sh` step 21.3 baseline-fit loop updated — if it currently loops over per-model SLURM files, rewrite to submit the parameterized template with `--export=ALL,MODEL=$m[,TIME=36:00:00 for M6b]`. Check `cluster/21_3_fit_baseline.slurm` — per 28-RESEARCH.md it's already parameterized via `--export=MODEL=...`, so it may not need changes. Verify and update only if necessary.
- [ ] Grep invariant: `grep -rn "13_bayesian_m[1-6]" cluster/ --include="*.sh" --include="*.slurm" | grep -v m6b_subscale | grep -v "^[[:space:]]*#"` returns zero matches. Three filters stack:
  - `[1-6]` character class matches m1/m2/m3/m5/m6a/m6b — all six deleted templates.
  - `grep -v m6b_subscale` excludes the retained specialized template `13_bayesian_m6b_subscale.slurm`.
  - `grep -v "^[[:space:]]*#"` excludes any shell-comment lines (defense-in-depth; Plan 28-06 task 4b already rewrites the 4 known stale `# ...13_bayesian_m6b.slurm pattern` comments in `cluster/21_{1_prior_predictive,2_recovery,2_recovery_aggregate,3_fit_baseline}.slurm` to reference the parameterized template, so the comment-strip filter should be redundant — but kept to catch any future commented refs).
- [ ] Net template count in `cluster/` drops from 13 Bayesian-related templates (6 per-model + 7 specialized) to 8 (1 consolidated + 7 specialized).
- [ ] `pytest scripts/fitting/tests/test_v4_closure.py` still passes 3/3.
- [ ] Atomic commit: `refactor(28-08): consolidate 6 per-model Bayesian SLURM templates into cluster/13_bayesian_choice_only.slurm`.

## Tasks

<tasks>
  <task id="1">
    <title>Read all six per-model templates and confirm diff from 28-RESEARCH.md</title>
    <detail>Read `cluster/13_bayesian_m{1,2,3,5,6a,6b}.slurm`. Per 28-RESEARCH.md §Q3 Table, the only differences are `--job-name`, `--time` (M6b is 36h, others 24h), and the `--model <name>` argument. Any additional diffs found here must be captured in the parameterized template or justified.</detail>
  </task>

  <task id="2">
    <title>Create cluster/13_bayesian_choice_only.slurm from m3 as source</title>
    <detail>`cp cluster/13_bayesian_m3.slurm cluster/13_bayesian_choice_only.slurm` (use `Write` tool, not bash `cp`, so git sees it as new). Then edit:
      - `#SBATCH --job-name=bayesian_${MODEL:-m3}` (shell-expanded at submit time — verify SLURM supports this; if not, fall back to a static `--job-name=bayesian_choice_only` and let SLURM's MODEL env var be the discriminator in logs).
      - Keep `#SBATCH --time=24:00:00` as the default. Add a prominent comment block:
        ```
        # M6b requires 36h wall-time due to stick-breaking decode in the 8-parameter model.
        # Submit M6b with an explicit time override:
        #   sbatch --time=36:00:00 --export=ALL,MODEL=wmrl_m6b cluster/13_bayesian_choice_only.slurm
        ```
      - At the top of the shell body, add `MODEL="${MODEL:-wmrl_m3}"` and log it to stdout.
      - Replace the hard-coded `--model wmrl_m3` CLI arg (or whichever model m3's template has) with `--model "$MODEL"`.</detail>
  </task>

  <task id="3">
    <title>Delete the six old per-model templates</title>
    <detail>
      - `git rm cluster/13_bayesian_m1.slurm`
      - `git rm cluster/13_bayesian_m2.slurm`
      - `git rm cluster/13_bayesian_m3.slurm`
      - `git rm cluster/13_bayesian_m5.slurm`
      - `git rm cluster/13_bayesian_m6a.slurm`
      - `git rm cluster/13_bayesian_m6b.slurm`</detail>
  </task>

  <task id="4">
    <title>Audit 21_submit_pipeline.sh and 21_3_fit_baseline.slurm</title>
    <detail>Read both files. Per 28-RESEARCH.md §"Cluster orchestrator map", step 21.3 submits `cluster/21_3_fit_baseline.slurm` with `--export=ALL,MODEL=$m` (already parameterized). Confirm this. If `21_submit_pipeline.sh` or any other orchestrator shell script directly submits the deleted per-model templates, rewrite those `sbatch` lines to use `cluster/13_bayesian_choice_only.slurm` with `--export=ALL,MODEL=<name>` (and `TIME=36:00:00` for M6b).
      Also grep `cluster/*.sh` for `13_bayesian_m[1-6]` — `cluster/submit_full_pipeline.sh` and `cluster/autopush.sh` may reference the deleted files.</detail>
  </task>

  <task id="5">
    <title>Update shell orchestrators that reference deleted templates</title>
    <detail>For each `.sh` file in `cluster/` that grep surfaces as containing `13_bayesian_m{1,2,3,5,6a,6b}.slurm`, rewrite the sbatch invocation to use the consolidated template. Example transformation:
      - Before: `sbatch cluster/13_bayesian_m6b.slurm`
      - After: `sbatch --time=36:00:00 --export=ALL,MODEL=wmrl_m6b cluster/13_bayesian_choice_only.slurm`
      Preserve all `afterok`, `--dependency`, and other flags.</detail>
  </task>

  <task id="6">
    <title>Syntax-check the new SLURM file</title>
    <detail>Run `bash -n cluster/13_bayesian_choice_only.slurm` to check for shell syntax errors (SLURM templates are bash scripts with SBATCH directive comments). Confirm exit 0. Do NOT actually submit to a cluster.</detail>
  </task>

  <task id="7">
    <title>Atomic commit</title>
    <detail>`refactor(28-08): consolidate 6 per-model Bayesian SLURM templates into cluster/13_bayesian_choice_only.slurm`. Body notes M6b TIME override convention, the 6 deletions, and that 7 specialized templates are retained.</detail>
  </task>
</tasks>

## Verification

```bash
# Consolidated template exists
test -f cluster/13_bayesian_choice_only.slurm

# Old per-model templates gone
test ! -f cluster/13_bayesian_m1.slurm
test ! -f cluster/13_bayesian_m2.slurm
test ! -f cluster/13_bayesian_m3.slurm
test ! -f cluster/13_bayesian_m5.slurm
test ! -f cluster/13_bayesian_m6a.slurm
test ! -f cluster/13_bayesian_m6b.slurm

# Specialized templates retained
test -f cluster/13_bayesian_m6b_subscale.slurm
test -f cluster/13_bayesian_gpu.slurm
test -f cluster/13_bayesian_multigpu.slurm
test -f cluster/13_bayesian_permutation.slurm
test -f cluster/13_bayesian_pscan.slurm
test -f cluster/13_bayesian_pscan_smoke.slurm
test -f cluster/13_bayesian_fullybatched_smoke.slurm

# No stale refs (except to retained m6b_subscale); strip comment lines to avoid false positives
grep -rn "13_bayesian_m[1-6]" cluster/ --include="*.sh" --include="*.slurm" \
  | grep -v m6b_subscale \
  | grep -v "^[[:space:]]*[^:]*:[[:space:]]*#"
# Expect: zero output

# Bash syntax check
bash -n cluster/13_bayesian_choice_only.slurm

# MODEL env var present
grep -n "MODEL" cluster/13_bayesian_choice_only.slurm

pytest scripts/fitting/tests/test_v4_closure.py -v
```

## Requirement IDs

Closes: **REFAC-09**.
