#!/usr/bin/env bash
# =============================================================================
# Canary Submitter — cluster/03_submit_canary.sh
# =============================================================================
# Stage-03 fan-out submission used to validate the cluster -> repo pipeline
# BEFORE burning a full afterok chain via cluster/submit_all.sh.
#
# Implements Phase 24 Wave 1 Task 2 (24-01-PLAN.md) as a versioned, reusable
# script. Future cold-starts after major refactors should reuse this rather
# than re-typing the inline operator loop documented in the plan.
#
# WHAT IT DOES
#   - Fans out N copies of cluster/03_prefitting_{gpu,cpu}.slurm, one per model,
#     with NO --dependency=afterok between them and NO downstream stages.
#   - Captures: timestamp + HEAD + host -> metadata.txt
#               JobIDs (pipe-delimited model|JID rows) -> jobids.txt
#               sbatch stdout tee -> submission.log
#               post-submit squeue dump -> queue_snapshot.txt
#   - Optionally commits + pushes those four files so the submission state
#     lands on `main` before any per-job autopush hook fires.
#
# WHY A CANARY (not the full chain)
#   - Validates 4 invariants in ~30-45 min wall-clock:
#       (1) login-node sbatch path is healthy
#       (2) Python driver writes to CCDS-canonical
#           models/bayesian/21_prior_predictive/ (NOT legacy output/bayesian/)
#       (3) cluster/autopush.sh hook commits + pushes results
#       (4) ArviZ-schema correctness of written .nc files
#   - On failure: 6 isolated jobs to diagnose, NOT a 30-JobID chain orphaned
#     in the queue.
#
# PRE-FLIGHT CONTRACT
#   cluster/00_preflight.slurm must have produced a recent PASS log at
#   models/bayesian/21_preflight_<JID>.log. The script greps for the literal
#   "PREFLIGHT: PASS" footer. Pass --no-preflight-check to bypass (e.g., when
#   re-submitting after a known-good gate older than the lookup window).
#
# OUTPUT ARTIFACTS (CCDS-canonical; per-job autopush picks these up)
#   models/bayesian/${TAG}_metadata.txt        -- timestamp, HEAD SHA, host
#   models/bayesian/${TAG}_jobids.txt          -- pipe-delimited STEP|MODEL|JID
#   models/bayesian/${TAG}_submission.log      -- sbatch stdout tee
#   models/bayesian/${TAG}_queue_snapshot.txt  -- post-submit squeue dump
#
# REUSE POLICY
#   Default TAG=21_canary anchors artifacts to Phase-21 namespace (matches
#   24-01-PLAN.md Task 3 acceptance gate). For a NEW canary on a different
#   namespace, override `--tag <name>`. The acceptance gate in Task 3 reads
#   these filenames; if you change TAG, update Task 3 accordingly.
#
# USAGE
#   bash cluster/03_submit_canary.sh                    # default: 6 models, GPU, prior_predictive
#   bash cluster/03_submit_canary.sh --cpu              # CPU stage-03 SLURM
#   bash cluster/03_submit_canary.sh --models "wmrl_m3 wmrl_m5"
#   bash cluster/03_submit_canary.sh --step bayesian_recovery --tag 21_canary_recovery
#   bash cluster/03_submit_canary.sh --dry-run          # syntax + path resolution only
#   bash cluster/03_submit_canary.sh --no-preflight-check
#   bash cluster/03_submit_canary.sh --no-push          # skip submission-time git push
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
MODELS_DEFAULT="qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b"
MODELS="$MODELS_DEFAULT"
STEP="prior_predictive"
TAG="21_canary"
USE_CPU=0
DRY_RUN=0
SKIP_PREFLIGHT=0
DO_PUSH=1
PREFLIGHT_GLOB="models/bayesian/21_preflight_*.log"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    --tag) TAG="$2"; shift 2 ;;
    --cpu) USE_CPU=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-preflight-check) SKIP_PREFLIGHT=1; shift ;;
    --no-push) DO_PUSH=0; shift ;;
    -h|--help)
      grep "^#" "$0" | head -80
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# CRLF strip (Windows dev box -> Linux cluster); mirrors submit_all.sh
# ---------------------------------------------------------------------------
sed -i 's/\r$//' cluster/*.slurm cluster/*.sh 2>/dev/null || true

# ---------------------------------------------------------------------------
# GPU vs CPU dispatch (Phase 23.1 default = GPU)
# ---------------------------------------------------------------------------
if [[ "$USE_CPU" == "1" ]]; then
  CANARY_SLURM="cluster/03_prefitting_cpu.slurm"
  DISPATCH="CPU (--cpu)"
else
  CANARY_SLURM="cluster/03_prefitting_gpu.slurm"
  DISPATCH="GPU (Phase 23.1 default)"
fi

if [[ ! -f "$CANARY_SLURM" ]]; then
  echo "ERROR: $CANARY_SLURM not found" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Pre-flight contract: most recent 21_preflight_*.log must end with PASS
# ---------------------------------------------------------------------------
if [[ "$SKIP_PREFLIGHT" != "1" ]]; then
  # shellcheck disable=SC2086
  PREFLIGHT_LOG=$(ls -t $PREFLIGHT_GLOB 2>/dev/null | head -1 || true)
  if [[ -z "$PREFLIGHT_LOG" || ! -f "$PREFLIGHT_LOG" ]]; then
    echo "ERROR: no pre-flight log matching $PREFLIGHT_GLOB found." >&2
    echo "       Run 'sbatch cluster/00_preflight.slurm' first, or pass --no-preflight-check." >&2
    exit 1
  fi
  if ! grep -q "^PREFLIGHT: PASS$" "$PREFLIGHT_LOG"; then
    echo "ERROR: most recent pre-flight log does NOT contain 'PREFLIGHT: PASS': $PREFLIGHT_LOG" >&2
    echo "       Re-run cluster/00_preflight.slurm or pass --no-preflight-check." >&2
    exit 1
  fi
  echo "[$(date)] Pre-flight gate OK -> $PREFLIGHT_LOG"
fi

# ---------------------------------------------------------------------------
# Metadata + artifact paths
# ---------------------------------------------------------------------------
mkdir -p models/bayesian
CANARY_TS=$(date -u +%Y%m%dT%H%M%SZ)
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || echo unknown)
HOSTNAME_=$(hostname)

META="models/bayesian/${TAG}_metadata.txt"
JIDS="models/bayesian/${TAG}_jobids.txt"
SUBMIT_LOG="models/bayesian/${TAG}_submission.log"
QUEUE_SNAP="models/bayesian/${TAG}_queue_snapshot.txt"

{
  echo "Canary timestamp (UTC): $CANARY_TS"
  echo "HEAD commit at submission: $HEAD_SHA"
  echo "Hostname: $HOSTNAME_"
  echo "Dispatch: $DISPATCH"
  echo "SLURM script: $CANARY_SLURM"
  echo "STEP: $STEP"
  echo "Models: $MODELS"
  echo "Tag: $TAG"
} > "$META"

echo "============================================================"
echo "[03_submit_canary.sh] $(date)"
if [[ "$DRY_RUN" == "1" ]]; then
  echo "  mode:     dry-run"
else
  echo "  mode:     real"
fi
echo "  dispatch: $DISPATCH"
echo "  slurm:    $CANARY_SLURM"
echo "  step:     $STEP"
echo "  tag:      $TAG"
echo "  models:   $MODELS"
echo "  ts:       $CANARY_TS"
echo "  head:     $HEAD_SHA"
echo "============================================================"

# Reset jobids + submission log
: > "$JIDS"
: > "$SUBMIT_LOG"

# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------
# DRY_FAKE_JID is incremented in the parent shell (NOT inside a $(...) subshell)
# so each model gets a distinct FAKEJID — this is the bug that bites
# cluster/submit_all.sh:106 silently (every fake job is JID=1001 there).
DRY_FAKE_JID=2000

# Single up-front bash -n on the canary SLURM (cheaper than per-model)
if [[ "$DRY_RUN" == "1" ]]; then
  if ! bash -n "$CANARY_SLURM"; then
    echo "DRY FAIL: bash -n $CANARY_SLURM" >&2
    exit 1
  fi
fi

for m in $MODELS; do
  if [[ "$DRY_RUN" == "1" ]]; then
    DRY_FAKE_JID=$((DRY_FAKE_JID + 1))
    JID="$DRY_FAKE_JID"
    echo "DRY ok: sbatch --export=ALL,STEP=$STEP,MODEL=$m $CANARY_SLURM (FAKEJID=$JID)" >&2
  else
    JID=$(sbatch --parsable --export=ALL,STEP="$STEP",MODEL="$m" "$CANARY_SLURM")
  fi
  echo "[canary.${STEP}] $m -> $JID" | tee -a "$SUBMIT_LOG"
  echo "${STEP}|${m}|${JID}" >> "$JIDS"
done

# ---------------------------------------------------------------------------
# Queue snapshot (skipped under --dry-run or absent squeue)
# ---------------------------------------------------------------------------
if [[ "$DRY_RUN" != "1" ]] && command -v squeue >/dev/null 2>&1; then
  squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %.10l %R" --sort=i \
    | tee "$QUEUE_SNAP"
else
  echo "(dry-run or squeue unavailable on this host)" > "$QUEUE_SNAP"
fi

# ---------------------------------------------------------------------------
# Submission-time autopush (so the dev box sees what was submitted BEFORE
# any per-job autopush completes ~15-30 min later).
# ---------------------------------------------------------------------------
if [[ "$DO_PUSH" == "1" && "$DRY_RUN" != "1" ]]; then
  echo ""
  echo "[$(date)] Committing + pushing submission artifacts..."
  git add "$META" "$JIDS" "$SUBMIT_LOG" "$QUEUE_SNAP" 2>/dev/null || true
  if git diff --cached --quiet 2>/dev/null; then
    echo "  nothing staged (artifacts unchanged) — skipping push"
  else
    git commit -m "chore(canary): submit ${STEP} fan-out (${TAG}, ${CANARY_TS})" \
      && git push \
      || echo "  WARNING: commit/push failed (artifacts still on disk; per-job autopush will retry)"
  fi
fi

echo ""
echo "============================================================"
echo "[03_submit_canary.sh] done — $(date)"
echo "============================================================"
echo "  metadata:        $META"
echo "  jobids:          $JIDS"
echo "  submission log:  $SUBMIT_LOG"
echo "  queue snapshot:  $QUEUE_SNAP"
echo ""
echo "Monitor:"
echo "  watch -n 30 \"squeue -u \$USER; ls -la models/bayesian/21_prior_predictive/ 2>/dev/null | head -20\""
echo ""
echo "Canary terminus = squeue empty AND sacct shows all jobs at COMPLETED/FAILED."
echo "Then run the four-criteria acceptance gate (Task 3 of"
echo ".planning/phases/24-cold-start-pipeline-execution/24-01-PLAN.md)."
echo "============================================================"
