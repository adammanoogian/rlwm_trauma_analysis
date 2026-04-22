#!/bin/bash
# =============================================================================
# SLURM Batch Script: Stage 05 — Post-Fitting Checks (consolidated)
# =============================================================================
# Parameterized STEP dispatcher for the three 05_post_fitting_checks/ scripts:
#   STEP=baseline_audit   -> 01_baseline_audit.py      (R-hat/ESS/BFMI gate)
#   STEP=scale_audit      -> 02_scale_audit.py         (L2 beta HDI + FDR-BH)
#   STEP=posterior_ppc    -> 03_run_posterior_ppc.py   (posterior predictive checks)
#
# Consolidates:
#   - cluster/21_4_baseline_audit.slurm -> STEP=baseline_audit
#   - cluster/21_7_scale_audit.slurm    -> STEP=scale_audit
#   - cluster/09_ppc_gpu.slurm          -> STEP=posterior_ppc (CPU-only path; for
#                                          GPU-accelerated PPC see
#                                          cluster/03_prefitting_gpu.slurm or
#                                          invoke with USE_GPU=1 to auto-append
#                                          --use-gpu)
#
# Pure ArviZ / pandas / statsmodels (no MCMC, no JAX compilation). CPU-only
# except when STEP=posterior_ppc with USE_GPU=1.
#
# Usage:
#   sbatch --export=ALL,STEP=baseline_audit cluster/05_post_checks.slurm
#   sbatch --export=ALL,STEP=scale_audit cluster/05_post_checks.slurm
#   sbatch --export=ALL,STEP=posterior_ppc,MODEL=wmrl_m3 cluster/05_post_checks.slurm
# =============================================================================

#SBATCH --job-name=p29_05_post_checks
#SBATCH --output=cluster/logs/05_post_checks_%j.out
#SBATCH --error=cluster/logs/05_post_checks_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=comp

set -euo pipefail

STEP="${STEP:-}"
MODEL="${MODEL:-wmrl_m3}"
USE_GPU="${USE_GPU:-0}"

# Baseline audit thresholds
RHAT="${RHAT:-1.05}"
ESS="${ESS:-400}"
BFMI="${BFMI:-0.2}"

# Scale audit thresholds + paths
HDI_PROB="${HDI_PROB:-0.95}"
FDR_ALPHA="${FDR_ALPHA:-0.05}"
ESS_DROP_THRESHOLD="${ESS_DROP_THRESHOLD:-0.5}"
L2_DIR="${L2_DIR:-output/bayesian/21_l2/}"
BASELINE_DIR="${BASELINE_DIR:-output/bayesian/21_baseline/}"
WINNERS_FILE="${WINNERS_FILE:-output/bayesian/21_baseline/winners.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-}"

if [ -z "$STEP" ]; then
    echo "ERROR: STEP env var is required. Valid values:" >&2
    echo "       baseline_audit | scale_audit | posterior_ppc" >&2
    exit 2
fi

module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Stage 05 — Post-Fitting Checks: STEP=$STEP"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       ${SLURMD_NODENAME:-$(hostname)}"
echo "Project:    $PROJECT_ROOT"
echo "Start time: $(date)"
echo "============================================================"

if conda activate ds_env 2>/dev/null; then
    echo "Activated ds_env by name"
elif conda activate /scratch/fc37/$USER/conda/envs/ds_env 2>/dev/null; then
    echo "Activated ds_env from /scratch/fc37/$USER/"
elif conda activate /scratch/${PROJECT:-fc37}/$USER/conda/envs/ds_env 2>/dev/null; then
    echo "Activated ds_env from /scratch/${PROJECT:-fc37}/$USER/"
elif conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu (fallback)"
else
    echo "ERROR: Failed to activate conda environment (tried ds_env, rlwm_gpu)"
    exit 1
fi

mkdir -p cluster/logs

export PYTHONUNBUFFERED=1

case "${STEP}" in
  baseline_audit)
    OUTPUT_DIR="${OUTPUT_DIR:-output/bayesian/21_baseline/}"
    mkdir -p "$OUTPUT_DIR"
    CMD=(python scripts/05_post_fitting_checks/01_baseline_audit.py
         --baseline-dir output/bayesian/21_baseline/
         --output-dir "$OUTPUT_DIR"
         --rhat-threshold "$RHAT"
         --ess-threshold "$ESS"
         --bfmi-threshold "$BFMI")
    ;;
  scale_audit)
    OUTPUT_DIR="${OUTPUT_DIR:-output/bayesian/21_l2/}"
    mkdir -p "$OUTPUT_DIR"
    CMD=(python scripts/05_post_fitting_checks/02_scale_audit.py
         --l2-dir "$L2_DIR"
         --baseline-dir "$BASELINE_DIR"
         --winners-file "$WINNERS_FILE"
         --hdi-prob "$HDI_PROB"
         --fdr-alpha "$FDR_ALPHA"
         --ess-drop-threshold "$ESS_DROP_THRESHOLD"
         --output-dir "$OUTPUT_DIR")
    ;;
  posterior_ppc)
    CMD=(python scripts/05_post_fitting_checks/03_run_posterior_ppc.py
         --model "$MODEL")
    if [ "$USE_GPU" = "1" ]; then
        CMD+=(--use-gpu)
    fi
    ;;
  *)
    echo "ERROR: unknown STEP='${STEP}'" >&2
    echo "       expected one of: baseline_audit | scale_audit | posterior_ppc" >&2
    exit 2
    ;;
esac

echo ""
echo "[$(date)] Running: ${CMD[*]}"
"${CMD[@]}"
EXIT_CODE=$?

echo ""
echo "[$(date)] Stage 05 STEP=$STEP finished (exit code: $EXIT_CODE)"

if [[ -f cluster/autopush.sh ]]; then
    source cluster/autopush.sh
fi

exit $EXIT_CODE
