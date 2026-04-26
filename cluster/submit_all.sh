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
#                                                      #   stage 01 auto-skips if data/raw/ empty AND data/processed/ tracked
#   bash cluster/submit_all.sh --dry-run                # path-check only (no sbatch)
#   bash cluster/submit_all.sh --from-stage 2          # canonical cluster cold-start entry
#                                                      #   (use after git pull delivers processed/ CSVs; bypasses stage 01 SLURM cost)
#   bash cluster/submit_all.sh --from-stage 4           # start mid-pipeline
#   bash cluster/submit_all.sh --models "wmrl_m3 wmrl_m5"  # subset of choice-only models
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
MODELS="qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --from-stage) FROM_STAGE="$2"; shift 2 ;;
    --models) MODELS="$2"; shift 2 ;;
    -h|--help)
      grep "^#" "$0" | head -50
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

echo "============================================================"
echo "[submit_all.sh] $(date)"
echo "  mode:    ${DRY_RUN:+dry-run}${DRY_RUN:-real}"
echo "  stages:  ${FROM_STAGE}..6"
echo "  models:  ${MODELS}"
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
    sbatch --parsable "$@" "$script"
  fi
}

# ---------------------------------------------------------------------------
# Stage 01 — Data preprocessing
# ---------------------------------------------------------------------------
J01=""
if [[ "$FROM_STAGE" -le 1 ]]; then
  echo ""
  echo "[01] Submitting cluster/01_data_processing.slurm"
  J01=$(submit cluster/01_data_processing.slurm)
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
  echo "[04b] Bayesian baseline (fan-out per choice-only model, CPU)"
  for m in $MODELS; do
    DEP=()
    if [[ -n "${REC_JOBS[$m]:-}" ]]; then
      DEP=(--dependency=afterok:"${REC_JOBS[$m]}")
    fi
    # M6b needs 36h walltime
    TIME_OVERRIDE=()
    [[ "$m" == "wmrl_m6b" ]] && TIME_OVERRIDE=(--time=36:00:00)
    BAYES_JOBS[$m]=$(submit "$BAYES_SCRIPT" "${DEP[@]}" "${TIME_OVERRIDE[@]}" \
                      --export=ALL,MODEL="$m")
    echo "  [04b] $m -> ${BAYES_JOBS[$m]}"
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
# Stage 05 — Post-fitting checks (baseline_audit -> scale_audit)
# ---------------------------------------------------------------------------
J05_BASELINE=""
J05_SCALE=""
if [[ "$FROM_STAGE" -le 5 ]]; then
  echo ""
  echo "[05] Post-fitting checks (baseline_audit -> scale_audit)"
  DEP=()
  [[ -n "$BAYES_DEP" ]] && DEP=(--dependency="$BAYES_DEP")
  J05_BASELINE=$(submit cluster/05_post_checks.slurm "${DEP[@]}" \
                  --export=ALL,STEP=baseline_audit)
  echo "  [05.baseline_audit] -> $J05_BASELINE"
  J05_SCALE=$(submit cluster/05_post_checks.slurm \
                --dependency=afterok:"$J05_BASELINE" \
                --export=ALL,STEP=scale_audit)
  echo "  [05.scale_audit] -> $J05_SCALE"
fi

# ---------------------------------------------------------------------------
# Stage 06 — Fit analyses (compare -> loo -> average -> tables)
# ---------------------------------------------------------------------------
J06_ALL=()
if [[ "$FROM_STAGE" -le 6 ]]; then
  echo ""
  echo "[06] Fit analyses (compare_models -> loo_stacking -> model_averaging -> manuscript_tables)"
  DEP=()
  if [[ -n "$J05_SCALE" ]]; then
    DEP=(--dependency=afterok:"$J05_SCALE")
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

    # After loo_stacking, insert L2 winner dispatcher (reads winners.txt
    # produced by loo_stacking, submits one $L2_SCRIPT per winner via
    # sbatch --wait inside cluster/21_dispatch_l2_winners.sh).
    if [[ "$step" == "loo_stacking" ]]; then
      L2_DISPATCH_JID=$(submit cluster/21_6_dispatch_l2.slurm \
        --dependency=afterok:"$PREV" \
        --export=ALL,L2_FIT_SCRIPT="$L2_SCRIPT")
      echo "  [04c.l2_dispatch] -> $L2_DISPATCH_JID (L2_FIT_SCRIPT=$L2_SCRIPT)"
      J06_ALL+=("$L2_DISPATCH_JID")
      PREV="$L2_DISPATCH_JID"   # model_averaging + manuscript_tables wait on L2 dispatcher
    fi
  done
fi

echo ""
echo "============================================================"
echo "[submit_all.sh] done — $(date)"
echo "============================================================"
echo "Mode:     $DISPATCH_MODE"
echo "Stage 01: $J01"
echo "Stage 02: $J02"
echo "Stage 03: ${J03_ALL[*]}"
echo "Stage 04: ${J04_ALL[*]}"
echo "Stage 04c L2 dispatch: ${L2_DISPATCH_JID:-<not dispatched; no winners.txt>}"
echo "Stage 05: $J05_BASELINE $J05_SCALE"
echo "Stage 06: ${J06_ALL[*]}"
echo "============================================================"
if [[ -n "$DRY_RUN" ]]; then
  echo "DRY-RUN: every stage SLURM passed bash -n and every python target resolved on disk."
fi
