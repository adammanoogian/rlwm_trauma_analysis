---
phase: 29-pipeline-canonical-reorg
plan: 05
type: execute
wave: 4
depends_on: [29-01, 29-03, 29-04b]
files_modified:
  - cluster/01_data_processing.slurm                      (new stage-numbered entry point)
  - cluster/02_behav_analyses.slurm                       (new)
  - cluster/03_prefitting_cpu.slurm                       (new; consolidates 09_ppc_gpu + 11_recovery_gpu + 21_1_prior_predictive + 21_2_recovery via --export=STEP=...)
  - cluster/03_prefitting_gpu.slurm                       (new; GPU counterpart)
  - cluster/04a_mle_cpu.slurm                             (renamed from 12_mle.slurm; paths updated)
  - cluster/04a_mle_gpu.slurm                             (renamed from 12_mle_gpu.slurm; paths updated)
  - cluster/04b_bayesian_cpu.slurm                        (renamed from 13_bayesian_choice_only.slurm)
  - cluster/04b_bayesian_gpu.slurm                        (renamed from 13_bayesian_gpu.slurm; M4 LBA target)
  - cluster/04c_level2.slurm                              (renamed from 21_6_fit_with_l2.slurm; new paths)
  - cluster/05_post_checks.slurm                          (new; consolidates 21_4_baseline_audit + 21_7_scale_audit + (new) posterior-PPC)
  - cluster/06_fit_analyses.slurm                         (new; consolidates 14_analysis + 21_5_loo_stacking_bms + 21_8_model_averaging + 21_9_manuscript_tables)
  - cluster/submit_all.sh                                 (new; --afterok chain across 01..06)
  - cluster/21_submit_pipeline.sh                         (updated: points at new cluster entry names OR deleted in favor of submit_all.sh)
  - cluster/12_mle.slurm                                  (deleted — moved to 04a_mle_cpu)
  - cluster/12_mle_gpu.slurm                              (deleted — moved to 04a_mle_gpu)
  - cluster/12_mle_single.slurm                           (deleted — subsumed by 04a)
  - cluster/12_submit_all.sh                              (deleted — superseded by submit_all.sh)
  - cluster/12_submit_all_gpu.sh                          (deleted — superseded by submit_all.sh)
  - cluster/13_bayesian_choice_only.slurm                 (deleted — moved to 04b_bayesian_cpu)
  - cluster/13_bayesian_gpu.slurm                         (deleted — moved to 04b_bayesian_gpu)
  - cluster/13_bayesian_m4_gpu.slurm                      (consolidated into 04b_bayesian_gpu via MODEL=wmrl_m4)
  - cluster/13_bayesian_m6b_subscale.slurm                (consolidated into 04b_bayesian_cpu via MODEL=wmrl_m6b_subscale, TIME=36:00:00)
  - cluster/13_bayesian_multigpu.slurm                    (retained — specialized; internal paths updated only)
  - cluster/13_bayesian_permutation.slurm                 (retained — specialized; paths updated)
  - cluster/13_bayesian_pscan.slurm                       (retained — specialized; paths updated)
  - cluster/13_bayesian_pscan_smoke.slurm                 (retained — specialized; paths updated)
  - cluster/13_bayesian_fullybatched_smoke.slurm          (retained — specialized; paths updated)
  - cluster/13_full_pipeline.slurm                        (retained — paths updated to new stage folders)
  - cluster/14_analysis.slurm                             (deleted — moved to 06_fit_analyses)
  - cluster/09_ppc_gpu.slurm                              (deleted — consolidated into 03_prefitting_gpu)
  - cluster/11_recovery_gpu.slurm                         (deleted — consolidated into 03_prefitting_gpu or stays as specialized)
  - cluster/21_1_prior_predictive.slurm                   (deleted — consolidated into 03_prefitting_cpu)
  - cluster/21_2_recovery.slurm                           (deleted — consolidated into 03_prefitting_cpu)
  - cluster/21_2_recovery_aggregate.slurm                 (deleted or absorbed — aggregation step)
  - cluster/21_3_fit_baseline.slurm                       (deleted — moved to 04b_bayesian_cpu)
  - cluster/21_4_baseline_audit.slurm                     (deleted — moved to 05_post_checks)
  - cluster/21_5_loo_stacking_bms.slurm                   (deleted — moved to 06_fit_analyses)
  - cluster/21_6_fit_with_l2.slurm                        (deleted — moved to 04c_level2)
  - cluster/21_6_dispatch_l2.slurm                        (deleted or preserved per evidence)
  - cluster/21_7_scale_audit.slurm                        (deleted — moved to 05_post_checks)
  - cluster/21_8_model_averaging.slurm                    (deleted — moved to 06_fit_analyses)
  - cluster/21_9_manuscript_tables.slurm                  (deleted — moved to 06_fit_analyses)
autonomous: true

must_haves:
  truths:
    - "Every cluster/*.slurm invokes a python script using the new canonical 01–06 path (or scripts/utils/ for internal helpers)"  # SC#7
    - "No cluster/*.slurm references old paths: scripts/bayesian_pipeline, scripts/post_mle, scripts/data_processing, scripts/behavioral, scripts/simulations_recovery, scripts/12_fit_mle, scripts/13_fit_bayesian, scripts/14_compare_models, scripts/fitting/fit_{mle,bayesian}"  # SC#7, SC#10
    - "Stage-numbered entry points exist (cluster/{01..06}*.slurm) alongside any retained specialized SLURMs"  # SC#7
    - "cluster/submit_all.sh chains the 6 stage SLURMs via --afterok and passes a --dry-run mode for path validation"  # SC#7
    - "bash -n cluster/*.slurm exits 0 for every SLURM file (syntax clean)"  # SC#7
    - "All referenced python scripts resolve on disk (ls-check confirms)"
  artifacts:
    - path: "cluster/submit_all.sh"
      provides: "master pipeline orchestrator — submits 01 → 02 → 03 → 04 → 05 → 06 via --afterok, supports --dry-run for path validation"
      min_lines: 40
      contains: "sbatch"
    - path: "cluster/04b_bayesian_cpu.slurm"
      provides: "consolidated CPU Bayesian SLURM — MODEL= and TIME= parameterized (pattern inherited from Phase 28 13_bayesian_choice_only)"
      contains: "--export=MODEL"
    - path: "cluster/05_post_checks.slurm"
      provides: "convergence + scale audit + posterior-PPC orchestration in one SLURM"
  key_links:
    - from: "cluster/submit_all.sh"
      to: "cluster/{01..06}*.slurm"
      via: "sbatch --afterok chain"
      pattern: "afterok"
---

<objective>
Update every cluster SLURM script to use new canonical script paths, consolidate per-stage SLURMs following the Phase 28 `13_bayesian_choice_only.slurm` parameterization pattern, and provide stage-numbered entry points (`cluster/01*..06*.slurm`) plus a master `submit_all.sh` chain. Critical for Phase 24 cold-start: when cluster jobs run, their python invocations MUST resolve.

Purpose: Prevent Phase 24 cold-start failure-at-submission-time due to stale paths. Stabilize the submission surface area (fewer per-model SLURM templates; one parameterized template per stage + kept specialized variants).

Output: canonical cluster/ directory with 6 stage SLURMs + specialized retained SLURMs + submit_all.sh master.
</objective>

<execution_context>
@C:\Users\aman0087\.claude\get-shit-done\workflows\execute-plan.md
@C:\Users\aman0087\.claude\get-shit-done\templates\summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/phases/29-pipeline-canonical-reorg/29-CONTEXT.md
@.planning/phases/29-pipeline-canonical-reorg/29-01-SUMMARY.md
@.planning/phases/29-pipeline-canonical-reorg/29-03-SUMMARY.md
@.planning/phases/28-bayesian-first-restructure-repo-cleanup/28-08-cluster-consolidation-SUMMARY.md
@cluster/13_bayesian_choice_only.slurm
@cluster/21_submit_pipeline.sh
</context>

<tasks>

<task type="auto">
  <name>Task 1: Mechanical path-update sweep across all existing cluster/*.slurm + *.sh</name>
  <files>
    - All cluster/*.slurm files (internal `python scripts/...` invocations updated)
    - cluster/21_submit_pipeline.sh (updated)
    - cluster/12_submit_all.sh, 12_submit_all_gpu.sh (updated or deleted per Task 2)
    - cluster/submit_full_pipeline.sh (updated)
    - cluster/autopush.sh (scanned for path refs)
  </files>
  <action>
    1. Grep all cluster files for old paths:
       ```
       grep -rn "scripts/data_processing\|scripts/behavioral\|scripts/simulations_recovery\|scripts/post_mle\|scripts/bayesian_pipeline\|scripts/12_fit_mle\|scripts/13_fit_bayesian\|scripts/14_compare_models\|scripts/fitting/fit_mle\|scripts/fitting/fit_bayesian\|scripts/analysis\|scripts/simulations\|scripts/visualization" cluster/
       ```
    2. For each hit, rewrite the path to the canonical 01–06 location:
       - `scripts/data_processing/NN_*.py` → `scripts/01_data_preprocessing/NN_*.py`
       - `scripts/behavioral/NN_*.py` → `scripts/02_behav_analyses/NN_*.py`
       - `scripts/simulations_recovery/NN_*.py` → `scripts/03_model_prefitting/NN_*.py`
       - `scripts/bayesian_pipeline/21_run_prior_predictive.py` → `scripts/03_model_prefitting/12_run_prior_predictive.py`
       - `scripts/bayesian_pipeline/21_run_bayesian_recovery.py` → `scripts/03_model_prefitting/13_run_bayesian_recovery.py`
       - `scripts/12_fit_mle.py` → `scripts/04_model_fitting/a_mle/12_fit_mle.py`
       - `scripts/13_fit_bayesian.py` → `scripts/04_model_fitting/b_bayesian/13_fit_bayesian.py`
       - `scripts/bayesian_pipeline/21_fit_baseline.py` → `scripts/04_model_fitting/b_bayesian/21_fit_baseline.py`
       - `scripts/bayesian_pipeline/21_fit_with_l2.py` → `scripts/04_model_fitting/c_level2/21_fit_with_l2.py`
       - `scripts/bayesian_pipeline/21_baseline_audit.py` → `scripts/05_post_fitting_checks/baseline_audit.py`
       - `scripts/bayesian_pipeline/21_scale_audit.py` → `scripts/05_post_fitting_checks/scale_audit.py`
       - `scripts/14_compare_models.py` → `scripts/06_fit_analyses/compare_models.py`
       - `scripts/bayesian_pipeline/21_compute_loo_stacking.py` → `scripts/06_fit_analyses/compute_loo_stacking.py`
       - `scripts/bayesian_pipeline/21_model_averaging.py` → `scripts/06_fit_analyses/model_averaging.py`
       - `scripts/bayesian_pipeline/21_manuscript_tables.py` → `scripts/06_fit_analyses/manuscript_tables.py`
       - `scripts/post_mle/NN_*.py` → `scripts/06_fit_analyses/*.py` (use the renamed target names from 29-01 Task 4)
    3. Every retained specialized SLURM gets its internal paths updated but keeps its filename for now (e.g., `13_bayesian_pscan.slurm`, `13_bayesian_multigpu.slurm`, `13_full_pipeline.slurm`).
    4. DO NOT DELETE any SLURM files in this task — that happens in Task 2 where we rename into new stage-numbered names.
    5. After all edits, verify every referenced script path exists on disk:
       ```
       grep -h "^python scripts/" cluster/*.slurm | awk '{print $2}' | sort -u | while read p; do
         test -f "$p" || echo "MISSING: $p"
       done
       ```
       Expected: no "MISSING" output.
  </action>
  <verify>
    - `grep -rn "scripts/data_processing\|scripts/behavioral\|scripts/simulations_recovery\|scripts/post_mle\|scripts/bayesian_pipeline\|scripts/12_fit_mle\|scripts/13_fit_bayesian\|scripts/14_compare_models\|scripts/fitting/fit_mle\|scripts/fitting/fit_bayesian" cluster/` returns ZERO
    - Every `python scripts/NN_*/*.py` path in every cluster/*.slurm resolves on disk
    - `bash -n cluster/*.slurm` exits 0 for every SLURM (syntax check)
  </verify>
  <done>All existing cluster SLURMs point at canonical paths; zero stale references; syntax check clean.</done>
</task>

<task type="auto">
  <name>Task 2: Create stage-numbered entry SLURMs + submit_all.sh + rename/consolidate; delete superseded templates</name>
  <files>
    - cluster/01_data_processing.slurm                      (new)
    - cluster/02_behav_analyses.slurm                       (new)
    - cluster/03_prefitting_cpu.slurm                       (new; parameterized STEP=prior_predictive|bayesian_recovery|ppc|synthetic|parameter_sweep|model_recovery)
    - cluster/03_prefitting_gpu.slurm                       (new; GPU counterpart)
    - cluster/04a_mle_cpu.slurm                             (git mv from cluster/12_mle.slurm)
    - cluster/04a_mle_gpu.slurm                             (git mv from cluster/12_mle_gpu.slurm)
    - cluster/04b_bayesian_cpu.slurm                        (git mv from cluster/13_bayesian_choice_only.slurm)
    - cluster/04b_bayesian_gpu.slurm                        (git mv from cluster/13_bayesian_gpu.slurm; M4 LBA MODEL=wmrl_m4 default)
    - cluster/04c_level2.slurm                              (git mv from cluster/21_6_fit_with_l2.slurm)
    - cluster/05_post_checks.slurm                          (new; consolidates 21_4_baseline_audit + 21_7_scale_audit + posterior-PPC; STEP= parameterized)
    - cluster/06_fit_analyses.slurm                         (new; consolidates 21_5_loo_stacking_bms + 21_8_model_averaging + 21_9_manuscript_tables + 14_analysis; STEP= parameterized)
    - cluster/submit_all.sh                                 (new; --afterok chain; supports --dry-run)
    - cluster/21_submit_pipeline.sh                         (shim delegating to submit_all.sh OR deleted with redirect note in cluster/README.md)
    - Deletions: cluster/{09_ppc_gpu, 11_recovery_gpu, 12_mle, 12_mle_gpu, 12_mle_single, 13_bayesian_choice_only, 13_bayesian_gpu, 14_analysis, 21_1_prior_predictive, 21_2_recovery, 21_2_recovery_aggregate, 21_3_fit_baseline, 21_4_baseline_audit, 21_5_loo_stacking_bms, 21_6_dispatch_l2, 21_6_fit_with_l2, 21_7_scale_audit, 21_8_model_averaging, 21_9_manuscript_tables}.slurm + scripts {12_submit_all.sh, 12_submit_all_gpu.sh, 21_dispatch_l2_winners.sh} (either deleted or preserved as historical shims)
  </files>
  <action>
    1. Study the existing `cluster/13_bayesian_choice_only.slurm` pattern (`--export=MODEL=<name>,TIME=<HH:MM:SS>` with `TIME=${TIME:-24:00:00}` default) from Phase 28. Replicate it across new consolidated SLURMs.
    2. Create `cluster/01_data_processing.slurm`:
       - SLURM header: 1 node, 4 CPUs, 16GB RAM, 2h walltime (inherit from `cluster/13_full_pipeline.slurm` preamble).
       - Body: serialize `python scripts/01_data_preprocessing/0{1..4}_*.py` calls (4 steps).
    3. Create `cluster/02_behav_analyses.slurm`:
       - Similar; body invokes `python scripts/02_behav_analyses/0{5..8}_*.py` serially.
       - This replicates the logic in `cluster/13_full_pipeline.slurm` lines 120–129 (which currently invokes behavioral scripts).
    4. Create `cluster/03_prefitting_cpu.slurm` parameterized via `--export=STEP=prior_predictive|bayesian_recovery|ppc|synthetic|parameter_sweep|model_recovery`:
       ```bash
       case "${STEP}" in
         prior_predictive) CMD="python scripts/03_model_prefitting/12_run_prior_predictive.py --model ${MODEL} ..." ;;
         bayesian_recovery) CMD="python scripts/03_model_prefitting/13_run_bayesian_recovery.py --model ${MODEL} ..." ;;
         ppc) CMD="python scripts/03_model_prefitting/09_run_ppc.py --model ${MODEL} ..." ;;
         synthetic) CMD="python scripts/03_model_prefitting/09_generate_synthetic_data.py ..." ;;
         parameter_sweep) CMD="python scripts/03_model_prefitting/10_run_parameter_sweep.py ..." ;;
         model_recovery) CMD="python scripts/03_model_prefitting/11_run_model_recovery.py --model ${MODEL} ..." ;;
         *) echo "ERROR: unknown STEP='${STEP}', expected one of prior_predictive|bayesian_recovery|ppc|synthetic|parameter_sweep|model_recovery" >&2; exit 2 ;;
       esac
       eval "${CMD}"
       ```
       Preserve original --job-name/-o/-e lines from whichever SLURM is being consolidated.
    5. Create `cluster/03_prefitting_gpu.slurm` as GPU counterpart of 03_prefitting_cpu.slurm (mainly for `ppc` STEP).
    6. Rename MLE entry SLURMs:
       - `git mv cluster/12_mle.slurm cluster/04a_mle_cpu.slurm`
       - `git mv cluster/12_mle_gpu.slurm cluster/04a_mle_gpu.slurm`
       - Delete `cluster/12_mle_single.slurm` (superseded by parameterized 04a).
       - Inside each, update python invocation to `scripts/04_model_fitting/a_mle/12_fit_mle.py`.
    7. Rename Bayesian entry SLURMs:
       - `git mv cluster/13_bayesian_choice_only.slurm cluster/04b_bayesian_cpu.slurm`
       - `git mv cluster/13_bayesian_gpu.slurm cluster/04b_bayesian_gpu.slurm`
       - Attempt to consolidate `13_bayesian_m4_gpu.slurm` into `04b_bayesian_gpu.slurm` by making MODEL= parameterized (the M4 LBA path is basically `--export=MODEL=wmrl_m4` with longer TIME). Only delete `13_bayesian_m4_gpu.slurm` if the resulting `04b_bayesian_gpu.slurm` handles both choice-only GPU and M4 LBA cases cleanly; otherwise KEEP `13_bayesian_m4_gpu.slurm` as a retained specialized template and just update its internal path.
       - Attempt to fold `13_bayesian_m6b_subscale.slurm` into `04b_bayesian_cpu.slurm` by making the MODEL=wmrl_m6b_subscale path a recognized case. If the subscale variant uses a truly different config, KEEP as specialized.
       - Inside each, update python invocations to `scripts/04_model_fitting/b_bayesian/{fit_bayesian,21_fit_baseline}.py`.
    8. Rename Level-2 SLURM:
       - `git mv cluster/21_6_fit_with_l2.slurm cluster/04c_level2.slurm`
       - Update internal path to `scripts/04_model_fitting/c_level2/21_fit_with_l2.py`.
    9. Create `cluster/05_post_checks.slurm` consolidating baseline_audit + scale_audit + posterior-PPC:
       - Parameterized STEP=baseline_audit|scale_audit|posterior_ppc.
       - Delete `21_4_baseline_audit.slurm` and `21_7_scale_audit.slurm`.
    10. Create `cluster/06_fit_analyses.slurm` consolidating LOO-stacking + model-averaging + manuscript-tables + compare_models:
       - Parameterized STEP=compare_models|loo_stacking|model_averaging|manuscript_tables|analyze_mle_by_trauma|regress_parameters_on_scales|analyze_winner_heterogeneity|bayesian_level2_effects.
       - Delete `14_analysis.slurm`, `21_5_loo_stacking_bms.slurm`, `21_8_model_averaging.slurm`, `21_9_manuscript_tables.slurm`.
    11. Create `cluster/submit_all.sh`:
       ```bash
       #!/usr/bin/env bash
       # Master orchestrator for the full RLWM pipeline (Phase 29 canonical).
       # Chains 01 -> 02 -> 03 -> 04 -> 05 -> 06 via --afterok.
       # Usage: bash cluster/submit_all.sh [--dry-run] [--from-stage N]
       set -euo pipefail
       DRY_RUN=""
       FROM_STAGE=1
       while [[ $# -gt 0 ]]; do
         case "$1" in
           --dry-run) DRY_RUN=1; shift ;;
           --from-stage) FROM_STAGE="$2"; shift 2 ;;
           *) echo "Unknown arg: $1"; exit 2 ;;
         esac
       done
       
       submit() {
         local script="$1"; shift
         if [[ -n "${DRY_RUN}" ]]; then
           echo "DRY: sbatch $@ $script — verifying path..."
           bash -n "$script"
           grep -E "^python scripts/" "$script" | awk '{print $2}' | while read p; do
             test -f "$p" || { echo "MISSING PYTHON TARGET: $p"; exit 1; }
           done
           echo "DRY: ok"
           echo "FAKEJOBID-$(basename "$script" .slurm)"
         else
           sbatch --parsable "$@" "$script"
         fi
       }
       
       # Stage 01
       J01=$(submit cluster/01_data_processing.slurm)
       # Stage 02
       J02=$(submit --dependency=afterok:$J01 cluster/02_behav_analyses.slurm)
       # Stage 03 prefitting: fan out prior PPC + recovery per model (simplified — dispatch loop per MODEL list)
       # ... (iterate over MODELS=wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b wmrl_m4, submit 03_prefitting_gpu/_cpu with STEP=prior_predictive/bayesian_recovery)
       # Stage 04
       # ... similar fan-out over MODELS for 04a_mle / 04b_bayesian
       # Stage 05
       # Stage 06
       ```
       The fan-out shape already exists in `cluster/21_submit_pipeline.sh` — STUDY IT FIRST and preserve the afterok dependency graph.
    12. Update `cluster/21_submit_pipeline.sh` to either:
       - Delegate to `cluster/submit_all.sh` (one-line shim), OR
       - Delete it and add a note in `cluster/README.md` that `submit_all.sh` supersedes it.
       Recommended: delegation shim, since Phase 24 / memory still references `bash cluster/21_submit_pipeline.sh` as the canonical entry.
    13. Run `cluster/submit_all.sh --dry-run` — expect all paths resolve, all SLURMs pass `bash -n`, exit 0.
    14. Update `cluster/README.md` with the new canonical entry-point list and the `--dry-run` semantics.
  </action>
  <verify>
    - `test -f cluster/submit_all.sh && test -x cluster/submit_all.sh` (or at least `bash cluster/submit_all.sh --dry-run` runs)
    - `for f in 01_data_processing 02_behav_analyses 03_prefitting_cpu 03_prefitting_gpu 04a_mle_cpu 04a_mle_gpu 04b_bayesian_cpu 04b_bayesian_gpu 04c_level2 05_post_checks 06_fit_analyses; do test -f cluster/$f.slurm || { echo "MISSING: $f.slurm"; exit 1; }; done`
    - `bash -n cluster/*.slurm` exits 0 for every SLURM
    - `bash cluster/submit_all.sh --dry-run` exits 0
    - `grep -rn "scripts/data_processing\|scripts/behavioral\|scripts/simulations_recovery\|scripts/post_mle\|scripts/bayesian_pipeline\|scripts/12_fit_mle\|scripts/13_fit_bayesian\|scripts/14_compare_models\|scripts/fitting/fit_mle\|scripts/fitting/fit_bayesian" cluster/` returns ZERO
    - Every referenced python script path in every cluster SLURM exists on disk (verified by submit_all.sh --dry-run)
  </verify>
  <done>Six stage SLURMs + specialized variants + submit_all.sh exist; dry-run validates all paths; superseded per-model templates consolidated/deleted; 21_submit_pipeline.sh delegates to submit_all.sh.</done>
</task>

</tasks>

<verification>
```bash
# Stage SLURMs present
for f in 01_data_processing.slurm 02_behav_analyses.slurm 03_prefitting_cpu.slurm 03_prefitting_gpu.slurm \
         04a_mle_cpu.slurm 04a_mle_gpu.slurm 04b_bayesian_cpu.slurm 04b_bayesian_gpu.slurm 04c_level2.slurm \
         05_post_checks.slurm 06_fit_analyses.slurm submit_all.sh; do
  test -f cluster/$f || { echo "MISSING: cluster/$f"; exit 1; }
done

# Syntax clean
bash -n cluster/*.slurm
bash -n cluster/*.sh

# Dry-run smoke
bash cluster/submit_all.sh --dry-run

# No stale path refs
grep -rn "scripts/data_processing\|scripts/behavioral\|scripts/simulations_recovery\|scripts/post_mle\|scripts/bayesian_pipeline\|scripts/12_fit_mle\|scripts/13_fit_bayesian\|scripts/14_compare_models\|scripts/fitting/fit_mle\|scripts/fitting/fit_bayesian" cluster/ \
  || echo "OK: zero stale cluster paths"

# v4 closure
pytest scripts/fitting/tests/test_v4_closure.py -v
```
</verification>

<success_criteria>
1. Six stage-numbered SLURMs exist: `cluster/0{1..6}*.slurm` (some have both CPU + GPU variants).
2. `cluster/04a/04b/04c` sub-letters captured (SC#2 parallel-alt paths for MLE/Bayesian/L2).
3. `cluster/submit_all.sh` chains all six via `--afterok` and supports `--dry-run` path validation (SC#7).
4. Zero stale script paths in any cluster SLURM (SC#7, SC#10).
5. All python paths referenced in cluster SLURMs exist on disk (dry-run passes).
6. `bash -n cluster/*.slurm` exits 0 for every SLURM (syntax clean).
7. `21_submit_pipeline.sh` either deleted or delegated to `submit_all.sh` (memory/docs still reference it).
</success_criteria>

<output>
After completion, create `.planning/phases/29-pipeline-canonical-reorg/29-05-SUMMARY.md` with:
- New stage SLURMs created with a 1-line description each
- Old SLURMs deleted / renamed / retained (table)
- `submit_all.sh --dry-run` output captured (full stdout)
- Any specialized SLURM that couldn't be consolidated + reason
- Commit SHA
</output>
