#!/bin/bash
# =============================================================================
# SLURM Batch Script: Step 21.1 Prior-Predictive Gate (Monash M3)
# =============================================================================
# Runs the Baribault & Collins (2023) prior-predictive check for a single
# choice-only model, dispatched via the MODEL env var.  CPU-only (no GPU
# needed — the Predictive call is trivial and the simulator is pure NumPy).
#
# Usage:
#   sbatch --export=ALL,MODEL=wmrl_m3 cluster/21_1_prior_predictive.slurm
#   sbatch --export=ALL,MODEL=wmrl_m6b,NUM_DRAWS=1000 cluster/21_1_prior_predictive.slurm
#
# The 9-step principled pipeline submits this script 6 times (once per
# choice-only model).  Exit code 0 on PASS, 1 on FAIL so dependency chains
# can short-circuit.
#
# Outputs saved to:
#   - output/bayesian/21_prior_predictive/{MODEL}_prior_sim.nc
#   - output/bayesian/21_prior_predictive/{MODEL}_prior_accuracy.csv
#   - output/bayesian/21_prior_predictive/{MODEL}_gate.md
# =============================================================================

#SBATCH --job-name=bayes_21_1_prior_pred
#SBATCH --output=logs/bayesian_21_1_prior_%j.out
#SBATCH --error=logs/bayesian_21_1_prior_%j.err
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

# --- Dispatch knobs (overridable via --export=ALL,MODEL=...) ---
MODEL="${MODEL:-wmrl_m3}"
NUM_DRAWS="${NUM_DRAWS:-500}"
SEED="${SEED:-42}"

# --- Environment Setup (matches Bayesian choice-only template pattern) ---
module load miniforge3

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Step 21.1 Prior-Predictive Gate: MODEL=${MODEL}"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Project: $PROJECT_ROOT"
echo "MODEL: $MODEL"
echo "NUM_DRAWS: $NUM_DRAWS"
echo "SEED: $SEED"
echo "Start time: $(date)"
echo "============================================================"

# Activate conda environment — ds_env first, with /scratch ladder fallback
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

# --- JAX compilation cache (per-MODEL to avoid cache collisions) ---
export JAX_PLATFORMS=cpu
export NUMPYRO_HOST_DEVICE_COUNT=1
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_21_prior_${MODEL}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"

# --- Create output directories ---
mkdir -p logs
mkdir -p output/bayesian/21_prior_predictive

# --- Run prior predictive gate ---
echo ""
echo "[$(date)] Starting step 21.1 prior-predictive gate for MODEL=${MODEL}"

python scripts/03_model_prefitting/12_run_prior_predictive.py \
    --model "$MODEL" \
    --data output/task_trials_long.csv \
    --num-draws "$NUM_DRAWS" \
    --seed "$SEED" \
    --output-dir output/bayesian/21_prior_predictive/

EXIT_CODE=$?

echo ""
echo "[$(date)] Prior-predictive gate finished (exit code: $EXIT_CODE)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Gate verdict: PASS"
    echo "Expected outputs:"
    for f in output/bayesian/21_prior_predictive/${MODEL}_prior_sim.nc \
             output/bayesian/21_prior_predictive/${MODEL}_prior_accuracy.csv \
             output/bayesian/21_prior_predictive/${MODEL}_gate.md; do
        if [ -f "$f" ]; then
            echo "  ✓ $f ($(du -h "$f" | cut -f1))"
        else
            echo "  ✗ $f (MISSING)"
        fi
    done
elif [ $EXIT_CODE -eq 1 ]; then
    echo "Gate verdict: FAIL — see output/bayesian/21_prior_predictive/${MODEL}_gate.md"
else
    echo "ERROR: scripts/21_run_prior_predictive.py exited with code $EXIT_CODE"
fi

# Auto-push logs and results when job completes
source cluster/autopush.sh

exit $EXIT_CODE
