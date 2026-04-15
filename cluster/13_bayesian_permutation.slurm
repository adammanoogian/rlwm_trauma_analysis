#!/bin/bash
# =============================================================================
# SLURM Array Job: Permutation Null Test for M3 LEC Covariate (L2-06)
# =============================================================================
# Runs 50 parallel permutation shuffles of the LEC covariate under M3
# hierarchical NUTS.  Each shuffle uses reduced MCMC budget (500 warmup,
# 1000 samples) and saves a compact JSON summary.
#
# Usage:
#   sbatch cluster/13_bayesian_permutation.slurm
#
# After completion, aggregate results with:
#   python scripts/fitting/aggregate_permutation_results.py
#
# Outputs per shuffle:
#   output/bayesian/permutation/shuffle_{idx}_results.json
#
# Aggregated summary:
#   output/bayesian/permutation/permutation_summary.md
# =============================================================================

#SBATCH --job-name=perm_null
#SBATCH --output=logs/perm_null_%A_%a.out
#SBATCH --error=logs/perm_null_%A_%a.err
#SBATCH --array=0-49
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

# --- Environment Setup (matches 12_mle_gpu.slurm pattern) ---
module load miniforge3

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Permutation Null Test — Shuffle ${SLURM_ARRAY_TASK_ID}"
echo "============================================================"
echo "Array job ID: ${SLURM_ARRAY_JOB_ID:-local}"
echo "Task ID:      ${SLURM_ARRAY_TASK_ID:-0}"
echo "Node:         ${SLURMD_NODENAME:-$(hostname)}"
echo "Project:      $PROJECT_ROOT"
echo "Start time:   $(date)"
echo "============================================================"

# Activate conda environment
# Try ds_env first (CPU Bayesian), then rlwm_gpu as fallback
if conda activate ds_env 2>/dev/null; then
    echo "Activated ds_env by name"
elif conda activate /scratch/fc37/$USER/conda/envs/ds_env 2>/dev/null; then
    echo "Activated ds_env from /scratch/fc37/$USER/"
elif conda activate /scratch/${PROJECT:-fc37}/$USER/conda/envs/ds_env 2>/dev/null; then
    echo "Activated ds_env from /scratch/$PROJECT/$USER/"
elif conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu (fallback)"
else
    echo "ERROR: Failed to activate conda environment (tried ds_env, rlwm_gpu)"
    exit 1
fi

# Verify jax is importable
python -c "import jax; print(f'JAX version: {jax.__version__}, platforms: {jax.devices()}')" || {
    echo "ERROR: JAX import failed after conda activation"
    exit 1
}
echo ""

# --- JAX compilation cache ---
export JAX_PLATFORMS=cpu
export NUMPYRO_HOST_DEVICE_COUNT=4
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_bayesian"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"

# --- Create output directories ---
mkdir -p logs
mkdir -p output/bayesian/permutation

# --- Run permutation shuffle ---
echo ""
echo "[$(date)] Starting permutation shuffle ${SLURM_ARRAY_TASK_ID}"

python scripts/fitting/fit_bayesian.py \
    --model wmrl_m3 \
    --data output/task_trials_long.csv \
    --permutation-shuffle "$SLURM_ARRAY_TASK_ID" \
    --chains 4 \
    --warmup 500 \
    --samples 1000 \
    --seed 42

EXIT_CODE=$?

echo ""
echo "[$(date)] Shuffle ${SLURM_ARRAY_TASK_ID} finished (exit code: $EXIT_CODE)"

if [ $EXIT_CODE -eq 0 ]; then
    OUT_FILE="output/bayesian/permutation/shuffle_${SLURM_ARRAY_TASK_ID}_results.json"
    if [ -f "$OUT_FILE" ]; then
        echo "  Output: $OUT_FILE ($(du -h "$OUT_FILE" | cut -f1))"
    else
        echo "  WARNING: Expected output $OUT_FILE not found"
    fi
else
    echo "ERROR: fit_bayesian.py exited with code $EXIT_CODE"
    exit $EXIT_CODE
fi

# Auto-push logs and results when job completes
source cluster/autopush.sh
