#!/bin/bash
# =============================================================================
# SLURM Batch Script: Stage 04b — Choice-Only Hierarchical Bayesian Fit (CPU)
# =============================================================================
# Parameterized template for all 6 choice-only models (M1/M2/M3/M5/M6a/M6b).
# Replaces the 6 per-model templates (13_bayesian_m{1,2,3,5,6a,6b}.slurm) and
# the 13_bayesian_m6b_subscale.slurm variant (via --export=SUBSCALE=1).
# CPU-only (no GPU needed for choice-only models).
#
# Usage:
#   sbatch --export=ALL,MODEL=wmrl_m3 cluster/04b_bayesian_cpu.slurm
#   sbatch --export=ALL,MODEL=qlearning cluster/04b_bayesian_cpu.slurm
#   sbatch --export=ALL,MODEL=wmrl cluster/04b_bayesian_cpu.slurm
#   sbatch --export=ALL,MODEL=wmrl_m5 cluster/04b_bayesian_cpu.slurm
#   sbatch --export=ALL,MODEL=wmrl_m6a cluster/04b_bayesian_cpu.slurm
#
# M6b requires 36h wall-time due to stick-breaking decode in the 8-parameter
# model. Submit M6b with an explicit time override:
#   sbatch --time=36:00:00 --export=ALL,MODEL=wmrl_m6b \
#     cluster/04b_bayesian_cpu.slurm
#
# M6b subscale variant (4 covariates x 8 params = 32 beta sites). Needs 12h+
# walltime and requires output/summary_participant_metrics.csv to supply the
# IES-R + LEC columns.
#   sbatch --time=12:00:00 --mem=48G \
#     --export=ALL,MODEL=wmrl_m6b,SUBSCALE=1 cluster/04b_bayesian_cpu.slurm
#
# Outputs saved to:
#   - output/bayesian/${MODEL}_individual_fits.csv
#   - output/bayesian/${MODEL}_posterior.nc
#   - output/bayesian/${MODEL}_shrinkage_report.md
#   - output/bayesian/${MODEL}_ppc_results.csv
# =============================================================================

#SBATCH --job-name=bayesian_cpu
#SBATCH --output=logs/bayesian_cpu_%j.out
#SBATCH --error=logs/bayesian_cpu_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

# --- Model selection + MCMC knobs (override via --export=ALL,KEY=VAL,...) ---
MODEL="${MODEL:-wmrl_m3}"
CHAINS="${CHAINS:-4}"
WARMUP="${WARMUP:-1000}"
SAMPLES="${SAMPLES:-2000}"
SEED="${SEED:-42}"
MAX_TREE_DEPTH="${MAX_TREE_DEPTH:-8}"
SUBSCALE="${SUBSCALE:-0}"  # 1 enables --subscale flag (M6b subscale variant)

# --- Environment Setup ---
module load miniforge3

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Stage 04b — Choice-Only Hierarchical Bayesian Fit (CPU)"
echo "============================================================"
echo "Job ID:         ${SLURM_JOB_ID:-local}"
echo "Node:           ${SLURMD_NODENAME:-$(hostname)}"
echo "Project:        $PROJECT_ROOT"
echo "Model:          $MODEL"
echo "Chains:         $CHAINS"
echo "Warmup:         $WARMUP"
echo "Samples:        $SAMPLES"
echo "Seed:           $SEED"
echo "Max tree depth: $MAX_TREE_DEPTH"
echo "Subscale:       $SUBSCALE"
echo "Start time:     $(date)"
echo "============================================================"

# Activate conda environment
# Try ds_env first (CPU Bayesian), then rlwm_gpu as fallback
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

# --- Verify required input files ---
if [ "$SUBSCALE" = "1" ]; then
    echo ""
    echo "Checking subscale-required input files..."
    for f in output/task_trials_long.csv output/summary_participant_metrics.csv; do
        if [ -f "$f" ]; then
            echo "  OK: $f ($(du -h "$f" | cut -f1))"
        else
            echo "  ERROR: $f is MISSING — required for subscale L2 regression"
            exit 1
        fi
    done
fi

# --- Build fit_bayesian.py command ---
CMD=(python scripts/04_model_fitting/b_bayesian/fit_bayesian.py
     --model "$MODEL"
     --data output/task_trials_long.csv
     --chains "$CHAINS"
     --warmup "$WARMUP"
     --samples "$SAMPLES"
     --seed "$SEED"
     --max-tree-depth "$MAX_TREE_DEPTH")

if [ "$SUBSCALE" = "1" ]; then
    CMD+=(--subscale)
fi

# --- Run hierarchical fit ---
echo ""
echo "[$(date)] Starting ${MODEL} hierarchical Bayesian fit${SUBSCALE:+ (subscale=$SUBSCALE)}"
echo "  Command: ${CMD[*]}"

"${CMD[@]}"

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
