#!/bin/bash
# =============================================================================
# SLURM Batch Script: Phase 21 Step 21.3 — Baseline Hierarchical Bayesian Fit
# =============================================================================
# Fits ONE of the 6 choice-only models (qlearning, wmrl, wmrl_m3, wmrl_m5,
# wmrl_m6a, wmrl_m6b) hierarchically with covariate_lec=None (NO L2 scales).
# Establishes the anti-circular baseline for the Wave-5 PSIS-LOO + stacking
# + RFX-BMS/PXP ranking (step 21.5). Every candidate model starts as equal
# — no MLE-preselected winner.
#
# Delegates to scripts/21_fit_baseline.py which forces
# --output-subdir 21_baseline on fit_bayesian.py, so all artefacts land under
# output/bayesian/21_baseline/ and Phase 16 posteriors at
# output/bayesian/{model}_posterior.nc are NEVER overwritten.
#
# Usage (single model, default wmrl_m6b):
#   sbatch cluster/21_3_fit_baseline.slurm
#
# Usage (specific model):
#   sbatch --export=MODEL=qlearning cluster/21_3_fit_baseline.slurm
#
# Usage (all 6 models in parallel, no cross-model file conflicts):
#   for m in qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b; do
#       sbatch --export=MODEL=$m cluster/21_3_fit_baseline.slurm
#   done
#
# Outputs saved to:
#   - output/bayesian/21_baseline/{MODEL}_individual_fits.csv
#   - output/bayesian/21_baseline/{MODEL}_posterior.nc
#   - output/bayesian/21_baseline/{MODEL}_shrinkage_report.md
#   - output/bayesian/21_baseline/{MODEL}_ppc_results.csv
#
# Time ceiling: 10h absorbs compile time + auto-bump retries. M1 (qlearning,
# 3 params) ~2h; M2-M5 ~4-6h; M6b (stick-breaking, 8 params) ~6-8h on CPU
# with fully-batched vmap.
# =============================================================================

#SBATCH --job-name=bayes_21_3_baseline
#SBATCH --output=logs/bayesian_21_3_%j.out
#SBATCH --error=logs/bayesian_21_3_%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

# --- Baseline fit parameters (override via --export=KEY=VAL,...) ---
MODEL="${MODEL:-wmrl_m6b}"
WARMUP="${WARMUP:-1000}"
SAMPLES="${SAMPLES:-2000}"
CHAINS="${CHAINS:-4}"
SEED="${SEED:-42}"
MAX_TREE_DEPTH="${MAX_TREE_DEPTH:-10}"

# --- Environment Setup (matches Bayesian choice-only template pattern) ---
module load miniforge3

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Phase 21 Step 21.3 — Baseline Hierarchical Fit (NO L2)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Project: $PROJECT_ROOT"
echo "Model: $MODEL"
echo "Warmup: $WARMUP"
echo "Samples: $SAMPLES"
echo "Chains: $CHAINS"
echo "Seed: $SEED"
echo "Max tree depth: $MAX_TREE_DEPTH"
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

# --- JAX compilation cache (unique per model so parallel runs don't collide) ---
export JAX_PLATFORMS=cpu
export NUMPYRO_HOST_DEVICE_COUNT=4
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_21_baseline_${MODEL}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"

# --- Create output directories ---
mkdir -p output/bayesian/21_baseline
mkdir -p logs

# --- Run Step 21.3 baseline fit via 21_fit_baseline.py orchestrator ---
# Orchestrator forces --output-subdir 21_baseline on fit_bayesian.py so all
# artefacts are routed to output/bayesian/21_baseline/, and adds a
# load-bearing post-fit expected.exists() check that exits 1 on convergence
# gate failure (see scripts/21_fit_baseline.py docstring).
echo ""
echo "[$(date)] Starting Step 21.3 baseline hierarchical Bayesian fit for $MODEL"

python scripts/04_model_fitting/b_bayesian/21_fit_baseline.py \
    --model "$MODEL" \
    --data output/task_trials_long.csv \
    --chains "$CHAINS" \
    --warmup "$WARMUP" \
    --samples "$SAMPLES" \
    --seed "$SEED" \
    --max-tree-depth "$MAX_TREE_DEPTH"

EXIT_CODE=$?

echo ""
echo "[$(date)] Step 21.3 baseline fit for $MODEL finished (exit code: $EXIT_CODE)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Expected outputs:"
    for f in output/bayesian/21_baseline/${MODEL}_individual_fits.csv \
             output/bayesian/21_baseline/${MODEL}_posterior.nc \
             output/bayesian/21_baseline/${MODEL}_shrinkage_report.md \
             output/bayesian/21_baseline/${MODEL}_ppc_results.csv; do
        if [ -f "$f" ]; then
            echo "  [OK] $f ($(du -h "$f" | cut -f1))"
        else
            echo "  [MISSING] $f (convergence gate may have blocked writes — but script should have exit 1)"
        fi
    done
else
    echo "ERROR: scripts/21_fit_baseline.py exited with code $EXIT_CODE"
    echo "Either convergence gate failed (R-hat, ESS, divergences) or the"
    echo "data/model/argparse layer rejected the inputs. Check log for the"
    echo "CONVERGENCE GATE FAIL message or Python traceback."
fi

# Auto-push logs and results when job completes
source cluster/autopush.sh

exit $EXIT_CODE
