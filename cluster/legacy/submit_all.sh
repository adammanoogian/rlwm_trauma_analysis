#!/usr/bin/env bash
# =============================================================================
# Master Pipeline Orchestrator — cluster/submit_all.sh
# =============================================================================
# Chains the 6 stage-numbered entry SLURMs (01..06) via sbatch --afterok.
# Replaces the per-step fan-out that lived in cluster/21_submit_pipeline.sh
# (kept as a shim that delegates here).
#
# Pipeline chain (post-29-05 canonical):
#   Stage 01  — cluster/01_data_processing.slurm     (jsPsych parse + collate + CSV)
#   Stage 02  — cluster/02_behav_analyses.slurm      (behavioural summaries/plots)
#   Stage 03  — cluster/03_prefitting_cpu.slurm      (fan-out per MODEL and STEP)
#   Stage 04  — cluster/04a_mle_cpu.slurm / 04b_bayesian_cpu.slurm
#                                                    (fan-out per MODEL)
#   Stage 05  — cluster/05_post_checks.slurm         (baseline_audit -> scale_audit)
#   Stage 06  — cluster/06_fit_analyses.slurm        (compare -> loo -> average -> tables)
#
# Usage:
#   bash cluster/submit_all.sh                          # full chain, real submission
#                                                      #   stage 01 auto-skips when data/processed/ is populated
#                                                      #   (the OSF-published cohort takes precedence over data/raw/);
#                                                      #   raw is parsed only when processed/ is absent (local cold-start).
#   bash cluster/submit_all.sh --dry-run                # path-check only (no sbatch)
#   bash cluster/submit_all.sh --from-stage 2          # cluster cold-start entry that bypasses the stage 01 SLURM cost
#                                                      #   entirely; equivalent to relying on the auto-skip above.
#   bash cluster/submit_all.sh --from-stage 4           # start mid-pipeline
#   bash cluster/submit_all.sh --models "wmrl_m3 wmrl_m5"  # subset of choice-only models (MLE fan-out)
#   bash cluster/submit_all.sh --bayes-models "qlearning wmrl wmrl_m6b wmrl_m3"
#                                                      # override Bayesian fan-out (default: qlearning wmrl wmrl_m6b
#                                                      #   per Phase 32-03 / config.BAYESIAN_FANOUT_MODELS;
#                                                      #   M3/M5/M6a are M6b corner cases or Collins 2025 ruled out
#                                                      #   so they are dropped from default Bayesian — fit them
#                                                      #   standalone as sensitivity analyses)
#   bash cluster/submit_all.sh --kappa-parameterization convex
#                                                      # Phase 32-04: select kappa parameterization. 'softmax' (default,
#                                                      #   Collins 2025) uses additive bias kappa in [-1, 1].
#                                                      #   'convex' (legacy revert path, Senta 2025) uses mixture
#                                                      #   kappa in [0, 1] — reproduces v5.0 pre-Phase-32 fits
#                                                      #   bit-equivalently. Threads through to stage-04b SLURM
#                                                      #   workers via the KAPPA_MODE env var.
#   bash cluster/submit_all.sh --preflight              # prepend SLURM-automated 2-cov L2 hook gate
#                                                      #   (cluster/00_preflight.slurm runs on a compute node
#                                                      #    with rlwm_gpu activated, then chains stage 01 via afterok)
#
# --dry-run semantics:
#   - For each stage SLURM: runs `bash -n` syntax check
#   - Extracts every `python scripts/...py` invocation and verifies the file exists
#   - Emits a stub FAKEJOBID for each would-be submission
#   - Exits 0 iff every path resolves; exits 1 on any MISSING python target
#
# This is the CANONICAL master entry point per plan 29-05. `cluster/submit_all.sh`
# should replace `cluster/21_submit_pipeline.sh` as the documented entry for
# Phase 24 cold-start.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

DRY_RUN=""
FROM_STAGE=1
DO_PREFLIGHT=""
AUTO_PUSH=true
NOTIFY_EMAIL="${NOTIFY_EMAIL:-}"
GIT_REMOTE="${GIT_REMOTE:-origin}"
MODELS="qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b"
# Phase 32-03: Narrowed Bayesian fan-out (see config.BAYESIAN_FANOUT_MODELS).
# MLE keeps all 7 choice-only models; Bayesian drops M3, M5, M6a because:
# - M3 = M6b at kappa_share = 1.0 (corner of simplex)
# - M6a = M6b at kappa_share = 0.0 (other corner)
# - M5's phi_rl is unnecessary per Collins 2025 Methods p.366
# Override via: --bayes-models "qlearning wmrl wmrl_m6b wmrl_m3"
BAYES_MODELS="${BAYES_MODELS:-qlearning wmrl wmrl_m6b}"
# Phase 32-04: Perseveration kappa parameterization. 'softmax' = Collins
# 2025 additive bias kappa in [-1, 1] (Phase 32-04 default). 'convex' =
# Senta 2025 mixture kappa in [0, 1] (legacy revert path; reproduces v5.0
# pre-Phase-32 fits bit-equivalently).
# Override via: --kappa-parameterization convex
KAPPA_MODE="${KAPPA_MODE:-softmax}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --from-stage) FROM_STAGE="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    --bayes-models) BAYES_MODELS="$2"; shift 2 ;;
    --kappa-parameterization) KAPPA_MODE="$2"; shift 2 ;;
    --preflight) DO_PREFLIGHT=1; shift ;;
    --no-auto-push) AUTO_PUSH=false; shift ;;
    --notify-email) NOTIFY_EMAIL="$2"; shift 2 ;;
    --git-remote) GIT_REMOTE="$2"; shift 2 ;;
    -h|--help)
      grep "^#" "$0" | head -60
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Validate KAPPA_MODE
case "$KAPPA_MODE" in
  softmax|convex) ;;
  *) echo "ERROR: --kappa-parameterization must be 'softmax' or 'convex', got '$KAPPA_MODE'" >&2; exit 2 ;;
esac

echo "============================================================"
echo "[submit_all.sh] $(date)"
echo "  mode:           ${DRY_RUN:+dry-run}${DRY_RUN:-real}"
echo "  stages:         ${FROM_STAGE}..6"
echo "  models (MLE):   ${MODELS}"
echo "  models (Bayes): ${BAYES_MODELS}"
echo "  kappa param:    ${KAPPA_MODE}"
echo "============================================================"

# =============================================================================
# CPU vs GPU dispatch resolution (Phase 23.1)
# =============================================================================
# Default: dispatch the GPU variants of the 3 NUTS-fitting stage scripts:
#   Stage 03 prefitting:    03_prefitting_gpu.slurm   (single-GPU)
#   Stage 04b Bayesian fit: 04b_bayesian_gpu.slurm    (Template C — 4 GPUs + pmap)
#   Stage 04c L2 refit:     04c_level2_gpu.slurm      (Template C — 4 GPUs + pmap)
#
# Escape hatch: `USE_CPU=1 bash cluster/submit_all.sh` reverts those 3 stages
# to their CPU siblings. Stages 01, 02, 05, 06 are CPU-only regardless
# (no NUTS MCMC — pure pandas / ArviZ / NumPy / SciPy / matplotlib).
#
# Per CLUSTER_GPU_LESSONS.md §6 + the M6b proof at job 54894258, the GPU path
# delivers ~3-4x wall-clock speedup at production scale. Phase 24 cold-start
# runs the default (GPU) path.
if [[ "${USE_CPU:-0}" == "1" ]]; then
    echo "[$(date)] USE_CPU=1 detected — dispatching CPU SLURM variants"
    PREFIT_SCRIPT="cluster/03_prefitting_cpu.slurm"
    BAYES_SCRIPT="cluster/04b_bayesian_cpu.slurm"
    L2_SCRIPT="cluster/04c_level2.slurm"
    DISPATCH_MODE="CPU (USE_CPU=1)"
else
    echo "[$(date)] Default GPU dispatch — Phase 23.1 multi-GPU pipeline"
    PREFIT_SCRIPT="cluster/03_prefitting_gpu.slurm"
    BAYES_SCRIPT="cluster/04b_bayesian_gpu.slurm"
    L2_SCRIPT="cluster/04c_level2_gpu.slurm"
    DISPATCH_MODE="GPU (Phase 23.1 default)"
fi
echo "  03 prefitting:  $PREFIT_SCRIPT"
echo "  04b Bayesian:   $BAYES_SCRIPT"
echo "  04c L2 refit:   $L2_SCRIPT"
echo ""

# ---------------------------------------------------------------------------
# Strip Windows CRLF from sibling SLURM/sh files (sbatch rejects \r)
# ---------------------------------------------------------------------------
sed -i 's/\r$//' cluster/*.slurm cluster/*.sh 2>/dev/null || true

# ---------------------------------------------------------------------------
# Helper: submit or dry-check a SLURM script
# ---------------------------------------------------------------------------
DRY_FAKE_JID=1000
# submit: emits ONLY the JID on stdout (so command substitution captures it cleanly).
# All human-readable log output goes to stderr.
submit() {
  local script="$1"; shift
  if [[ ! -f "$script" ]]; then
    echo "ERROR: $script not found" >&2
    return 1
  fi

  if [[ -n "${DRY_RUN}" ]]; then
    # Syntax check
    if ! bash -n "$script"; then
      echo "DRY FAIL: bash -n $script" >&2
      return 1
    fi
    # Verify every `python scripts/...py` invocation resolves on disk.
    # Only inspect lines that actually invoke python (skip comments that
    # merely mention a path as documentation).
    local missing=0
    while IFS= read -r pypath; do
      [[ -z "$pypath" ]] && continue
      if [[ ! -f "$pypath" ]]; then
        echo "DRY MISSING: $script -> $pypath" >&2
        missing=1
      fi
    done < <(grep -E '^[[:space:]]*(python|CMD=.*python|"python"|srun .*python)' "$script" \
             | grep -oE 'scripts/[^[:space:]"'"'"']+\.py' \
             | sort -u)
    if [[ $missing -ne 0 ]]; then
      return 1
    fi
    DRY_FAKE_JID=$((DRY_FAKE_JID + 1))
    local tag
    tag=$(basename "$script" .slurm)
    echo "DRY ok: sbatch $* $script (FAKEJID=$DRY_FAKE_JID tag=$tag)" >&2
    echo "$DRY_FAKE_JID"
  else
    # Validate JID is numeric. Empty / non-numeric stdout from sbatch
    # (transient scheduler hiccup, partition reject without non-zero exit,
    # malformed --parsable response) would silently propagate as
    # `--dependency=afterok:` (empty value) downstream, which SLURM marks
    # `DependencyNeverSatisfied` immediately. Abort the orchestrator here
    # rather than orphan the chain.
    local jid
    jid=$(sbatch --parsable "$@" "$script") || {
      echo "ERROR: sbatch failed for $script (args: $*)" >&2
      return 1
    }
    if [[ ! "$jid" =~ ^[0-9]+$ ]]; then
      echo "ERROR: sbatch returned non-numeric JID '$jid' for $script (args: $*)" >&2
      echo "       Refusing to chain afterok on a bogus JID — orchestrator aborted." >&2
      return 1
    fi
    echo "$jid"
  fi
}

# ---------------------------------------------------------------------------
# Stage 00 — Preflight (SLURM-automated env + 2-cov L2 hook gate)
# ---------------------------------------------------------------------------
# When --preflight is set, submit cluster/00_preflight.slurm as the chain
# root and wire stage 01's --dependency=afterok onto its JID. The preflight
# job activates rlwm_gpu, smoke-tests imports, probes JAX devices, and runs
# tests/integration/test_numpyro_models_2cov.py on a compute node — which
# is the right place for it (the login node lacks pytest and conda).
JPRE=""
if [[ -n "$DO_PREFLIGHT" ]]; then
  if [[ "$FROM_STAGE" -gt 1 ]]; then
    echo "WARNING: --preflight ignored with --from-stage > 1 (preflight wires only to stage 01)"
  else
    echo ""
    echo "[00] Submitting cluster/00_preflight.slurm (env + 2-cov L2 hook gate)"
    JPRE=$(submit cluster/00_preflight.slurm)
    echo "[00] Job: $JPRE"
  fi
fi

# ---------------------------------------------------------------------------
# Stage 01 — Data preprocessing
# ---------------------------------------------------------------------------
J01=""
if [[ "$FROM_STAGE" -le 1 ]]; then
  echo ""
  echo "[01] Submitting cluster/01_data_processing.slurm"
  DEP_STAGE01=()
  [[ -n "$JPRE" ]] && DEP_STAGE01=(--dependency=afterok:"$JPRE")
  J01=$(submit cluster/01_data_processing.slurm "${DEP_STAGE01[@]}")
  echo "[01] Job: $J01"
fi

# ---------------------------------------------------------------------------
# Stage 02 — Behavioural analyses
# ---------------------------------------------------------------------------
J02=""
if [[ "$FROM_STAGE" -le 2 ]]; then
  echo ""
  echo "[02] Submitting cluster/02_behav_analyses.slurm"
  DEP=()
  [[ -n "$J01" ]] && DEP=(--dependency=afterok:"$J01")
  J02=$(submit cluster/02_behav_analyses.slurm "${DEP[@]}")
  echo "[02] Job: $J02"
fi

# ---------------------------------------------------------------------------
# Stage 03 — Prefitting (prior-predictive + Bayesian recovery per model)
# ---------------------------------------------------------------------------
declare -A PRIOR_JOBS
declare -A REC_JOBS
J03_ALL=()
if [[ "$FROM_STAGE" -le 3 ]]; then
  echo ""
  echo "[03] Prefitting (fan-out per model, CPU)"
  DEP_STAGE03=()
  [[ -n "$J02" ]] && DEP_STAGE03=(--dependency=afterok:"$J02")
  for m in $MODELS; do
    PRIOR_JOBS[$m]=$(submit "$PREFIT_SCRIPT" "${DEP_STAGE03[@]}" \
                      --export=ALL,STEP=prior_predictive,MODEL="$m")
    echo "  [03.prior_predictive] $m -> ${PRIOR_JOBS[$m]}"
    J03_ALL+=("${PRIOR_JOBS[$m]}")
    REC_JOBS[$m]=$(submit "$PREFIT_SCRIPT" \
                    --dependency=afterok:"${PRIOR_JOBS[$m]}" \
                    --export=ALL,STEP=bayesian_recovery,MODEL="$m")
    echo "  [03.bayesian_recovery] $m -> ${REC_JOBS[$m]}"
    J03_ALL+=("${REC_JOBS[$m]}")
  done
fi

# ---------------------------------------------------------------------------
# Stage 04 — Model fitting (CPU Bayesian fan-out; M4 LBA uses GPU)
# ---------------------------------------------------------------------------
declare -A BAYES_JOBS
J04_ALL=()
if [[ "$FROM_STAGE" -le 4 ]]; then
  echo ""
  echo "[04b] Bayesian baseline (fan-out per Bayesian model, narrowed per Phase 32-03)"
  # Phase 32-03: iterate over BAYES_MODELS (default: qlearning wmrl wmrl_m6b),
  # NOT MODELS. M3, M5, M6a are dropped from default Bayesian (M6b corner
  # cases / Collins 2025 ruled out phi_rl). MLE fan-out and stage 03
  # prefitting still iterate over MODELS so all 7 models stay in the AIC
  # comparison table and prior-predictive sweeps.
  for m in $BAYES_MODELS; do
    DEP=()
    if [[ -n "${REC_JOBS[$m]:-}" ]]; then
      DEP=(--dependency=afterok:"${REC_JOBS[$m]}")
    fi
    # M6b needs 36h walltime
    TIME_OVERRIDE=()
    [[ "$m" == "wmrl_m6b" ]] && TIME_OVERRIDE=(--time=36:00:00)
    # Phase 32-04: thread KAPPA_MODE through to the SLURM as an env var.
    BAYES_JOBS[$m]=$(submit "$BAYES_SCRIPT" "${DEP[@]}" "${TIME_OVERRIDE[@]}" \
                      --export=ALL,MODEL="$m",KAPPA_MODE="$KAPPA_MODE")
    echo "  [04b] $m -> ${BAYES_JOBS[$m]} (kappa=${KAPPA_MODE})"
    J04_ALL+=("${BAYES_JOBS[$m]}")
  done
fi

# Build colon-separated afterok dependency for stage 05
BAYES_DEP=""
if [[ ${#J04_ALL[@]} -gt 0 ]]; then
  BAYES_DEP="afterok:$(IFS=:; echo "${J04_ALL[*]}")"
fi

# ---------------------------------------------------------------------------
# Stage 04c — L2 winner refit (runs after 04b; gated on winners.txt existing)
# ---------------------------------------------------------------------------
# The L2 refit is a fan-out per winner (M1/M2 copy-through, M3/M5/M6a 2-cov,
# M6b subscale). `cluster/21_6_dispatch_l2.slurm` wraps the dispatcher
# `cluster/21_dispatch_l2_winners.sh` which reads
# models/bayesian/21_baseline/winners.txt and submits one $L2_SCRIPT per
# winner via `sbatch --wait`. The --wait pattern + &+wait in the dispatcher
# ensures the SLURM job stays alive until every L2 fit completes.
#
# We pass L2_FIT_SCRIPT through so the dispatcher routes to $L2_SCRIPT
# (the GPU variant by default; CPU when USE_CPU=1).
#
# Note: winners.txt is produced by step loo_stacking in stage 06. In the
# Phase-29 orchestrator flow, stage 06 loo_stacking runs AFTER stage 05 and
# before the remaining compare/averaging/tables steps. This means stage 04c
# must run AFTER a partial stage 06 (loo_stacking specifically) produces
# winners.txt. To preserve chain integrity, the master orchestrator emits
# ONE `21_6_dispatch_l2.slurm` submission with a dependency on the LOO
# step inside stage 06, and downstream averaging/tables depend on the
# dispatcher. The chain is built at stage-06 time (see Stage 06 block).
L2_DISPATCH_JID=""

# ---------------------------------------------------------------------------
# Stage 05.1 — baseline_audit (post-fitting checks on stage 04b posteriors)
# ---------------------------------------------------------------------------
# Note: scale_audit (stage 05.2) is NOT submitted here. It reads winners.txt
# (produced by stage 06 loo_stacking) AND models/bayesian/21_l2/<winner>_
# posterior.nc (produced by stage 04c L2 dispatch). Both producers run inside
# the stage-06 loop below, so scale_audit is submitted there with a dependency
# on L2_DISPATCH and chained ahead of model_averaging.
J05_BASELINE=""
J05_SCALE=""
if [[ "$FROM_STAGE" -le 5 ]]; then
  echo ""
  echo "[05.1] baseline_audit (post-fitting baseline checks)"
  DEP=()
  [[ -n "$BAYES_DEP" ]] && DEP=(--dependency="$BAYES_DEP")
  J05_BASELINE=$(submit cluster/05_post_checks.slurm "${DEP[@]}" \
                  --export=ALL,STEP=baseline_audit)
  echo "  [05.baseline_audit] -> $J05_BASELINE"
fi

# ---------------------------------------------------------------------------
# Stage 06 — Fit analyses (compare -> loo -> L2 -> scale_audit -> average -> tables)
# ---------------------------------------------------------------------------
J06_ALL=()
if [[ "$FROM_STAGE" -le 6 ]]; then
  echo ""
  echo "[06] Fit analyses (compare_models -> loo_stacking -> L2 -> scale_audit -> model_averaging -> manuscript_tables)"
  DEP=()
  if [[ -n "$J05_BASELINE" ]]; then
    DEP=(--dependency=afterok:"$J05_BASELINE")
  elif [[ -n "$BAYES_DEP" ]]; then
    DEP=(--dependency="$BAYES_DEP")
  fi

  PREV=""
  for step in compare_models loo_stacking model_averaging manuscript_tables; do
    LOCAL_DEP=("${DEP[@]}")
    if [[ -n "$PREV" ]]; then
      LOCAL_DEP=(--dependency=afterok:"$PREV")
    fi
    JID=$(submit cluster/06_fit_analyses.slurm "${LOCAL_DEP[@]}" \
           --export=ALL,STEP="$step")
    echo "  [06.$step] -> $JID"
    J06_ALL+=("$JID")
    PREV="$JID"

    # After loo_stacking, insert (a) L2 winner dispatcher and (b) scale_audit.
    # L2 dispatcher reads winners.txt from loo_stacking, submits one
    # $L2_SCRIPT per winner via sbatch --wait inside
    # cluster/21_dispatch_l2_winners.sh. scale_audit reads winners.txt AND
    # models/bayesian/21_l2/<winner>_posterior.nc (the L2 dispatcher's
    # output), so it MUST run AFTER L2_DISPATCH — wiring it here keeps the
    # afterok chain monotone.
    if [[ "$step" == "loo_stacking" ]]; then
      L2_DISPATCH_JID=$(submit cluster/21_6_dispatch_l2.slurm \
        --dependency=afterok:"$PREV" \
        --export=ALL,L2_FIT_SCRIPT="$L2_SCRIPT")
      echo "  [04c.l2_dispatch] -> $L2_DISPATCH_JID (L2_FIT_SCRIPT=$L2_SCRIPT)"
      J06_ALL+=("$L2_DISPATCH_JID")
      PREV="$L2_DISPATCH_JID"

      J05_SCALE=$(submit cluster/05_post_checks.slurm \
                    --dependency=afterok:"$PREV" \
                    --export=ALL,STEP=scale_audit)
      echo "  [05.scale_audit] -> $J05_SCALE (after L2_DISPATCH)"
      PREV="$J05_SCALE"   # model_averaging + manuscript_tables wait on scale_audit
    fi
  done
fi

# ---------------------------------------------------------------------------
# Auto-push: dependency-chained push job (replaces the per-SLURM autopush
# trap — see commit removing cluster/autopush.sh on 2026-05-05). One push
# at the end avoids the fan-out race where N concurrent fitting jobs sharing
# the NFS-mounted working tree all hit `git push` simultaneously, causing
# auto-stash + rebase conflicts that lose data ("one job saved" /
# "adding non autopushed" recovery commits in recent history).
#
# afterany (not afterok) so the push fires even when fitting crashes — we
# want logs back. Skip with --no-auto-push.
# ---------------------------------------------------------------------------
PUSH_JID=""
if [[ "$AUTO_PUSH" == "true" && -z "$DRY_RUN" ]]; then
  ALL_JOBIDS=()
  [[ -n "$JPRE"          ]] && ALL_JOBIDS+=("$JPRE")
  [[ -n "$J01"           ]] && ALL_JOBIDS+=("$J01")
  [[ -n "$J02"           ]] && ALL_JOBIDS+=("$J02")
  ALL_JOBIDS+=("${J03_ALL[@]}")
  ALL_JOBIDS+=("${J04_ALL[@]}")
  [[ -n "$L2_DISPATCH_JID" ]] && ALL_JOBIDS+=("$L2_DISPATCH_JID")
  [[ -n "$J05_BASELINE"  ]] && ALL_JOBIDS+=("$J05_BASELINE")
  [[ -n "$J05_SCALE"     ]] && ALL_JOBIDS+=("$J05_SCALE")
  ALL_JOBIDS+=("${J06_ALL[@]}")

  if [[ ${#ALL_JOBIDS[@]} -gt 0 ]]; then
    echo ""
    echo "--- Auto-push: dependency-chained results push ---"
    PUSH_DEP=$(IFS=:; echo "${ALL_JOBIDS[*]}")
    PUSH_PARENTS=$(IFS=' '; echo "${ALL_JOBIDS[*]}")

    PUSH_MAIL_FLAGS=()
    if [[ -n "$NOTIFY_EMAIL" ]]; then
      PUSH_MAIL_FLAGS=(--mail-type=END,FAIL --mail-user="$NOTIFY_EMAIL")
    fi

    PUSH_JID=$(sbatch --parsable \
        --dependency=afterany:${PUSH_DEP} \
        "${PUSH_MAIL_FLAGS[@]}" \
        --export="ALL,PARENT_JOBS=${PUSH_PARENTS},NOTIFY_EMAIL=${NOTIFY_EMAIL},GIT_REMOTE=${GIT_REMOTE}" \
        cluster/99_push_results.slurm)
    echo "  push job: $PUSH_JID (afterany on ${#ALL_JOBIDS[@]} parent jobs)"
    [[ -n "$NOTIFY_EMAIL" ]] && echo "  email notifications: $NOTIFY_EMAIL"
  fi
elif [[ "$AUTO_PUSH" != "true" ]]; then
  echo ""
  echo "--- Auto-push: SKIPPED (--no-auto-push) ---"
  echo "  Manual: sbatch --dependency=afterany:<JID1>:<JID2>:... cluster/99_push_results.slurm"
fi

echo ""
echo "============================================================"
echo "[submit_all.sh] done — $(date)"
echo "============================================================"
echo "Mode:     $DISPATCH_MODE"
echo "Stage 00 preflight: ${JPRE:-<not requested>}"
echo "Stage 01: $J01"
echo "Stage 02: $J02"
echo "Stage 03: ${J03_ALL[*]}"
echo "Stage 04: ${J04_ALL[*]}"
echo "Stage 04c L2 dispatch: ${L2_DISPATCH_JID:-<not dispatched; no winners.txt>}"
echo "Stage 05: $J05_BASELINE $J05_SCALE"
echo "Stage 06: ${J06_ALL[*]}"
echo "Auto-push:  ${PUSH_JID:-<skipped>}"
echo "============================================================"
if [[ -n "$DRY_RUN" ]]; then
  echo "DRY-RUN: every stage SLURM passed bash -n and every python target resolved on disk."
fi
