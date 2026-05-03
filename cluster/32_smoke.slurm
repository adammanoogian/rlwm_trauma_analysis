#!/bin/bash
# =============================================================================
# SLURM Batch Script: Phase 32-05 Cluster Smoke (Dual-Mode Kappa)
# =============================================================================
# Smoke-fits ONE model under ONE kappa parameterization on a 30-participant
# subsample. Used as the go/no-go gate for Plan 32-06 phase closure: 4
# Bayesian (qlearning x {softmax,convex} + wmrl_m6b x {softmax,convex}) +
# 2 MLE (qlearning x softmax + wmrl_m6b x softmax) submitted in fan-out by
# the user via .planning/phases/32-mcmc-methodology-update/32-05-jobs-submitted.txt.
#
# Why a separate template (instead of reusing 04a_mle_gpu.slurm /
# 04b_bayesian_gpu.slurm):
# - Sub-sampled data path differs (data/processed/task_trials_long_30p_smoke.csv)
# - Per-mode output subdir differs (32_smoke_softmax / 32_smoke_convex) so
#   the regression baseline at models/bayesian/21_baseline/ is NOT
#   overwritten by the smoke-cohort posteriors.
# - Walltime tighter (4h not 6h, smaller cohort fits faster).
# - Stage selector (STAGE=bayesian|mle) keeps both backends behind one
#   sbatch entry-point so the manifest reads cleanly.
#
# GPU caveat: this template always requests 1 GPU (--gres=gpu:1). qlearning
# Bayesian runs CPU-fast and does not need the GPU; the wasted allocation
# is acceptable for a 6-job smoke. If queue pressure makes that tradeoff
# painful, submit qlearning under CPU partition manually with:
#     sbatch --partition=comp --gres= ... cluster/32_smoke.slurm
#
# Usage (one job per (model, mode, stage) tuple):
#   sbatch --export=ALL,MODEL=qlearning,KAPPA_MODE=softmax,STAGE=bayesian \
#          cluster/32_smoke.slurm
#   sbatch --export=ALL,MODEL=wmrl_m6b,KAPPA_MODE=convex,STAGE=bayesian \
#          cluster/32_smoke.slurm
#   sbatch --export=ALL,MODEL=qlearning,KAPPA_MODE=softmax,STAGE=mle \
#          cluster/32_smoke.slurm
#
# Outputs:
#   - models/bayesian/32_smoke_${KAPPA_MODE}/${MODEL}_posterior.nc
#   - models/bayesian/32_smoke_${KAPPA_MODE}/${MODEL}_individual_fits.csv
#   - models/mle/32_smoke_${KAPPA_MODE}/${MODEL}_individual_fits.csv  (STAGE=mle)
#   - logs/cluster/32_smoke/32_smoke_${SLURM_JOB_ID}.{out,err}
# =============================================================================

#SBATCH --job-name=rlwm-32-smoke
#SBATCH --output=logs/cluster/32_smoke/32_smoke_%j.out
#SBATCH --error=logs/cluster/32_smoke/32_smoke_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -e

# EXIT trap fires on success OR failure so partial logs land in git and
# the autopush hook persists artefacts even on bash-level errors. Mirrors
# the cluster/04b_bayesian_gpu.slurm convention.
_ON_EXIT() {
    local _exit=$?
    [ -f cluster/autopush.sh ] && source cluster/autopush.sh || true
    exit "$_exit"
}
trap _ON_EXIT EXIT

# =============================================================================
# Configuration — override via --export when submitting
# =============================================================================
# MODEL has NO default — submission MUST specify it via --export.
if [ -z "${MODEL:-}" ]; then
    echo "ERROR: MODEL is not set. Submit with --export=ALL,MODEL=<name>,..."
    echo "  Valid: qlearning | wmrl_m6b (smoke set per Plan 32-05)"
    exit 2
fi
KAPPA_MODE="${KAPPA_MODE:-softmax}"
STAGE="${STAGE:-bayesian}"
SUBSAMPLE_N="${SUBSAMPLE_N:-30}"
OUT_SUBDIR="${OUT_SUBDIR:-32_smoke_${KAPPA_MODE}}"

# Validate KAPPA_MODE early — fit_baseline.py / fit_mle.py would fail
# anyway, but a top-line message saves a 4h queue-wait on a typo.
case "$KAPPA_MODE" in
    softmax|convex) ;;
    *)
        echo "ERROR: KAPPA_MODE='$KAPPA_MODE' invalid. Use 'softmax' or 'convex'."
        exit 2
        ;;
esac
case "$STAGE" in
    bayesian|mle) ;;
    *)
        echo "ERROR: STAGE='$STAGE' invalid. Use 'bayesian' or 'mle'."
        exit 2
        ;;
esac

# =============================================================================
# Environment Setup (M3 specific)
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

mkdir -p logs/cluster/32_smoke

echo "============================================================"
echo "Phase 32-05 — Cluster Smoke Fit"
echo "============================================================"
echo "Job ID:         ${SLURM_JOB_ID:-local}"
echo "Node:           ${SLURMD_NODENAME:-$(hostname)}"
echo "Project:        $PROJECT_ROOT"
echo "MODEL:          $MODEL"
echo "KAPPA_MODE:     $KAPPA_MODE"
echo "STAGE:          $STAGE"
echo "SUBSAMPLE_N:    $SUBSAMPLE_N"
echo "OUT_SUBDIR:     $OUT_SUBDIR"
echo "Start time:     $(date)"
echo "============================================================"

# Activate rlwm_gpu by name first (uses .condarc), then explicit /scratch
# paths. ds_env (CPU-only) fallback is acceptable for qlearning Bayesian
# but not for wmrl_m6b on the smoke; let the underlying Python detect
# missing GPU and emit a clear warning rather than silently degrading.
if conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu by name"
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/fc37/$USER/"
elif conda activate /scratch/${PROJECT:-fc37}/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/${PROJECT:-fc37}/$USER/"
else
    echo "ERROR: rlwm_gpu conda env activation failed."
    echo "       Create with: bash cluster/00_setup_env_gpu.sh"
    exit 1
fi

# Re-register editable rlwm install — egg-link goes stale if the repo
# moves. Idempotent; mirrors 04b_bayesian_gpu.slurm.
pip install -e . --quiet --no-deps 2>/dev/null || \
    echo "WARNING: pip install -e . --no-deps failed; 'import rlwm' may fail downstream"

# =============================================================================
# Subsample data for smoke speed (head N participants)
# =============================================================================
SMOKE_DATA="data/processed/task_trials_long_${SUBSAMPLE_N}p_smoke.csv"
if [ ! -f "$SMOKE_DATA" ]; then
    echo "[$(date)] Building smoke subsample at $SMOKE_DATA"
    python - <<PYEOF
import pandas as pd

src = "data/processed/task_trials_long.csv"
out = "${SMOKE_DATA}"
n = ${SUBSAMPLE_N}

df = pd.read_csv(src)
# Use sona_id (canonical participant identifier per CLAUDE.md). Fall back
# to participant_id if a future refactor renames the column.
pid_col = "sona_id" if "sona_id" in df.columns else "participant_id"
pids = df[pid_col].drop_duplicates().head(n).tolist()
sub = df[df[pid_col].isin(pids)].copy()
sub.to_csv(out, index=False)
print(f"Wrote {out} with {len(pids)} participants ({len(sub)} trials).")
PYEOF
else
    echo "[$(date)] Smoke subsample already exists: $SMOKE_DATA"
fi

# =============================================================================
# JAX cache + env exports (per-mode + per-stage to avoid cross-contamination
# between softmax / convex compiled bytecodes)
# =============================================================================
export PYTHONUNBUFFERED=1
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_32_smoke_${STAGE}_${MODEL}_${KAPPA_MODE}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" || {
    echo "WARNING: cache dir create failed"
    unset JAX_COMPILATION_CACHE_DIR
}
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo ""

# =============================================================================
# Dispatch to fit_baseline.py (Bayesian) or fit_mle.py (MLE)
# =============================================================================
if [ "$STAGE" = "bayesian" ]; then
    mkdir -p "models/bayesian/${OUT_SUBDIR}"
    echo "[$(date)] Starting Bayesian smoke fit: MODEL=$MODEL KAPPA_MODE=$KAPPA_MODE"
    python scripts/04_model_fitting/b_bayesian/fit_baseline.py \
        --model "$MODEL" \
        --kappa-parameterization "$KAPPA_MODE" \
        --data "$SMOKE_DATA" \
        --output-subdir "$OUT_SUBDIR"
    EXIT_CODE=$?
elif [ "$STAGE" = "mle" ]; then
    mkdir -p "models/mle/${OUT_SUBDIR}"
    echo "[$(date)] Starting MLE smoke fit: MODEL=$MODEL KAPPA_MODE=$KAPPA_MODE"
    python scripts/04_model_fitting/a_mle/fit_mle.py \
        --model "$MODEL" \
        --kappa-parameterization "$KAPPA_MODE" \
        --data "$SMOKE_DATA" \
        --output-subdir "$OUT_SUBDIR"
    EXIT_CODE=$?
fi

echo ""
echo "[$(date)] Smoke fit finished (exit code: $EXIT_CODE)"

# =============================================================================
# Post-fit artefact audit
# =============================================================================
if [ "$STAGE" = "bayesian" ]; then
    EXPECTED_NC="models/bayesian/${OUT_SUBDIR}/${MODEL}_posterior.nc"
    EXPECTED_CSV="models/bayesian/${OUT_SUBDIR}/${MODEL}_individual_fits.csv"
elif [ "$STAGE" = "mle" ]; then
    EXPECTED_NC=""
    EXPECTED_CSV="models/mle/${OUT_SUBDIR}/${MODEL}_individual_fits.csv"
fi

MISSING=0
if [ -n "$EXPECTED_NC" ]; then
    if [ -f "$EXPECTED_NC" ]; then
        echo "  [OK] $EXPECTED_NC ($(du -h "$EXPECTED_NC" | cut -f1))"
    else
        echo "  [MISSING] $EXPECTED_NC"
        MISSING=$((MISSING + 1))
    fi
fi
if [ -f "$EXPECTED_CSV" ]; then
    echo "  [OK] $EXPECTED_CSV ($(du -h "$EXPECTED_CSV" | cut -f1))"
else
    echo "  [MISSING] $EXPECTED_CSV"
    MISSING=$((MISSING + 1))
fi

if [ $EXIT_CODE -eq 0 ] && [ $MISSING -gt 0 ]; then
    echo "*** ERROR: smoke fit returned exit=0 but $MISSING expected artefact(s) MISSING ***"
    EXIT_CODE=1
fi

echo ""
echo "============================================================"
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"
echo "============================================================"

exit $EXIT_CODE
