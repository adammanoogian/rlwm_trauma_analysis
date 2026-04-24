#!/bin/bash
# =============================================================================
# SLURM Batch Script: Multi-GPU Hierarchical Bayesian Fit (chains via pmap)
# =============================================================================
# Production variant of 13_bayesian_*.slurm that allocates one GPU per chain
# and lets NumPyro's chain_method="parallel" pmap the chains across devices.
#
# After the Issue 1 fully-batched rollout (commit c54ee6c), the
# participant-level for-loop bottleneck is gone and single-GPU MCMC runs at
# ~1s/iter on N=154 (vs ~6 min/iter before).  At that per-iter speed,
# "vectorized" chains (4 chains time-multiplexed on 1 GPU) already complete
# in ~1-2h at production budget.  This multi-GPU script gives true parallel
# chain execution (chain_method="parallel") when wall time needs to drop
# further, e.g. for 8 chain-ensemble sensitivity runs.
#
# _select_chain_method (numpyro_models.py:67) auto-selects "parallel" when
#   jax.local_device_count() >= num_chains
# else falls back to "vectorized".  So all we do here is request N GPUs
# via SLURM and set MODEL/CHAINS.
#
# Usage:
#   # Default: 4 chains across 4 GPUs
#   sbatch --export=MODEL=wmrl_m6b cluster/13_bayesian_multigpu.slurm
#
#   # Custom: 8 chains across 8 GPUs, needs --gres=gpu:8 override
#   sbatch --gres=gpu:8 --export=MODEL=wmrl_m6b,CHAINS=8 \
#       cluster/13_bayesian_multigpu.slurm
# =============================================================================

#SBATCH --job-name=bayes_mgpu
#SBATCH --output=logs/bayesian_mgpu_%j.out
#SBATCH --error=logs/bayesian_mgpu_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4

# =============================================================================
# Configuration — override via --export when submitting
# =============================================================================
MODEL="${MODEL:-wmrl_m6b}"
CHAINS="${CHAINS:-4}"
WARMUP="${WARMUP:-1000}"
SAMPLES="${SAMPLES:-2000}"
MAX_TREE_DEPTH="${MAX_TREE_DEPTH:-10}"
SEED="${SEED:-42}"

_VALID_MODELS=("qlearning" "wmrl" "wmrl_m3" "wmrl_m5" "wmrl_m6a" "wmrl_m6b")
_MODEL_OK=false
for m in "${_VALID_MODELS[@]}"; do
    [[ "$MODEL" == "$m" ]] && _MODEL_OK=true
done
if ! $_MODEL_OK; then
    echo "ERROR: MODEL='$MODEL' not in supported list: ${_VALID_MODELS[*]}"
    echo "       (M6b subscale and M4 LBA use dedicated SLURM scripts)"
    exit 1
fi

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Multi-GPU Bayesian Fit — $MODEL (chains via pmap)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Project: $PROJECT_ROOT"
echo "Model: $MODEL"
echo "Chains: $CHAINS  Warmup: $WARMUP  Samples: $SAMPLES"
echo "Max tree depth: $MAX_TREE_DEPTH  Seed: $SEED"
echo "Start time: $(date)"
echo "============================================================"

if conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu by name"
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/fc37/$USER/"
elif conda activate ds_env 2>/dev/null; then
    echo "Activated ds_env (fallback, no GPU — pmap will not work)"
else
    echo "ERROR: Failed to activate conda environment"
    exit 1
fi

# Verify GPU allocation matches chain count (otherwise chain_method falls
# back to "vectorized" which defeats the point of this script).
python -c "
import jax
import sys
devices = jax.devices()
n_gpu = sum(1 for d in devices if d.platform == 'gpu')
print(f'JAX {jax.__version__} devices: {devices}')
print(f'GPU count: {n_gpu}')
if n_gpu < $CHAINS:
    print(f'WARNING: {n_gpu} GPU(s) < {$CHAINS} chains.  chain_method will be')
    print(f'         vectorized, not parallel.  Consider --gres=gpu:$CHAINS.')
" || {
    echo "ERROR: JAX import failed"
    exit 1
}
echo ""

# JAX compilation cache — separate per model to avoid cross-contamination
export NUMPYRO_HOST_DEVICE_COUNT="$CHAINS"
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_mgpu_${MODEL}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || unset JAX_COMPILATION_CACHE_DIR
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo ""

export PYTHONUNBUFFERED=1

mkdir -p logs models/bayesian

# =============================================================================
# Run the fit
# =============================================================================
echo ""
echo "[$(date)] Starting multi-GPU Bayesian fit for $MODEL"

python scripts/04_model_fitting/b_bayesian/fit_bayesian.py \
    --model "$MODEL" \
    --data data/processed/task_trials_long.csv \
    --chains "$CHAINS" \
    --warmup "$WARMUP" \
    --samples "$SAMPLES" \
    --max-tree-depth "$MAX_TREE_DEPTH" \
    --seed "$SEED"

EXIT_CODE=$?
echo ""
echo "[$(date)] Fit finished (exit code: $EXIT_CODE)"

# Auto-push logs and results when job completes
if [[ -f cluster/autopush.sh ]]; then
    source cluster/autopush.sh
fi

exit $EXIT_CODE
