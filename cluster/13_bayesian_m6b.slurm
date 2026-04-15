#!/bin/bash
# =============================================================================
# SLURM Batch Script: M6b Hierarchical Bayesian Fit (Monash M3)
# =============================================================================
# Runs M6b (WM-RL+dual perseveration) hierarchical NumPyro fit with Level-2
# LEC regression on kappa_total and kappa_share.
# CPU-only (no GPU needed for choice-only models).
#
# M6b is the winning model (v4.0 post-refit, N=154). Extended wall-clock time
# allocated to account for stick-breaking decode in 8-parameter model.
#
# Usage:
#   sbatch cluster/13_bayesian_m6b.slurm
#
# Outputs saved to:
#   - output/bayesian/wmrl_m6b_individual_fits.csv
#   - output/bayesian/wmrl_m6b_posterior.nc
#   - output/bayesian/wmrl_m6b_shrinkage_report.md
#   - output/bayesian/wmrl_m6b_ppc_results.csv
# =============================================================================

#SBATCH --job-name=bayesian_m6b
#SBATCH --output=logs/bayesian_m6b_%j.out
#SBATCH --error=logs/bayesian_m6b_%j.err
#SBATCH --time=36:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

# --- Environment Setup (matches 12_mle_gpu.slurm pattern) ---
module load miniforge3

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "M6b (WM-RL+dual perseveration) Hierarchical Bayesian Fit"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Project: $PROJECT_ROOT"
echo "Start time: $(date)"
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
mkdir -p output/bayesian

# --- Run M6b hierarchical fit ---
echo ""
echo "[$(date)] Starting M6b (WM-RL+dual) hierarchical Bayesian fit"

python scripts/fitting/fit_bayesian.py \
    --model wmrl_m6b \
    --data output/task_trials_long.csv \
    --chains 4 \
    --warmup 1000 \
    --samples 2000 \
    --seed 42 \
    --max-tree-depth 8

EXIT_CODE=$?

echo ""
echo "[$(date)] M6b hierarchical fit finished (exit code: $EXIT_CODE)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Expected outputs:"
    for f in output/bayesian/wmrl_m6b_individual_fits.csv \
             output/bayesian/wmrl_m6b_posterior.nc \
             output/bayesian/wmrl_m6b_shrinkage_report.md \
             output/bayesian/wmrl_m6b_ppc_results.csv; do
        if [ -f "$f" ]; then
            echo "  ✓ $f ($(du -h "$f" | cut -f1))"
        else
            echo "  ✗ $f (MISSING — convergence gate may have blocked writes)"
        fi
    done
else
    echo "ERROR: fit_bayesian.py exited with code $EXIT_CODE"
fi

# Auto-push logs and results when job completes
source cluster/autopush.sh
