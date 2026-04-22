#!/bin/bash
# =============================================================================
# SLURM Batch Script: Choice-Only Hierarchical Bayesian Fit (Monash M3)
# =============================================================================
# Parameterized template for all 6 choice-only models (M1/M2/M3/M5/M6a/M6b).
# Replaces the 6 per-model templates (13_bayesian_m{1,2,3,5,6a,6b}.slurm).
# CPU-only (no GPU needed for choice-only models).
#
# Usage:
#   sbatch --export=ALL,MODEL=wmrl_m3 cluster/13_bayesian_choice_only.slurm
#   sbatch --export=ALL,MODEL=qlearning cluster/13_bayesian_choice_only.slurm
#   sbatch --export=ALL,MODEL=wmrl cluster/13_bayesian_choice_only.slurm
#   sbatch --export=ALL,MODEL=wmrl_m5 cluster/13_bayesian_choice_only.slurm
#   sbatch --export=ALL,MODEL=wmrl_m6a cluster/13_bayesian_choice_only.slurm
#
# M6b requires 36h wall-time due to stick-breaking decode in the 8-parameter
# model. Submit M6b with an explicit time override:
#   sbatch --time=36:00:00 --export=ALL,MODEL=wmrl_m6b \
#     cluster/13_bayesian_choice_only.slurm
#
# Outputs saved to:
#   - output/bayesian/${MODEL}_individual_fits.csv
#   - output/bayesian/${MODEL}_posterior.nc
#   - output/bayesian/${MODEL}_shrinkage_report.md
#   - output/bayesian/${MODEL}_ppc_results.csv
# =============================================================================

#SBATCH --job-name=bayesian_choice_only
#SBATCH --output=logs/bayesian_choice_only_%j.out
#SBATCH --error=logs/bayesian_choice_only_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

# --- Model selection (override via --export=ALL,MODEL=<name>) ---
MODEL="${MODEL:-wmrl_m3}"

# --- Environment Setup (matches 12_mle_gpu.slurm pattern) ---
module load miniforge3

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Choice-Only Hierarchical Bayesian Fit"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Project: $PROJECT_ROOT"
echo "Model: $MODEL"
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

# --- Run hierarchical fit ---
echo ""
echo "[$(date)] Starting ${MODEL} hierarchical Bayesian fit"

python scripts/04_model_fitting/b_bayesian/fit_bayesian.py \
    --model "$MODEL" \
    --data output/task_trials_long.csv \
    --chains 4 \
    --warmup 1000 \
    --samples 2000 \
    --seed 42 \
    --max-tree-depth 8

EXIT_CODE=$?

echo ""
echo "[$(date)] ${MODEL} hierarchical fit finished (exit code: $EXIT_CODE)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Expected outputs:"
    for f in "output/bayesian/${MODEL}_individual_fits.csv" \
             "output/bayesian/${MODEL}_posterior.nc" \
             "output/bayesian/${MODEL}_shrinkage_report.md" \
             "output/bayesian/${MODEL}_ppc_results.csv"; do
        if [ -f "$f" ]; then
            echo "  [OK] $f ($(du -h "$f" | cut -f1))"
        else
            echo "  [MISSING] $f (convergence gate may have blocked writes)"
        fi
    done
else
    echo "ERROR: fit_bayesian.py exited with code $EXIT_CODE"
fi

# Auto-push logs and results when job completes
source cluster/autopush.sh
