#!/bin/bash
# =============================================================================
# SLURM Batch Script: Fully-Batched MCMC Smoke Test (shortest variant)
# =============================================================================
# Minimal validation that the Issue 1 fix (fully-batched vmap likelihood across
# all 6 choice-only models) unblocks sequential MCMC on real N=154 data.
#
# The prior smoke test (logs/pscan_smoke_54845743) timed out after 6h at
# iter 65/100 of qlearning warmup (~6 min per NUTS iteration).  That was
# caused by 154 separate numpyro.factor("obs_p{pid}") sites dispatching
# 154 GPU kernels per leapfrog step.  Commit c54ee6c collapsed these to
# one numpyro.factor("obs", ...) call per model via nested vmap.
#
# This smoke test does ONLY Stage 2 (sequential MCMC) — no micro-benchmark
# (already validated numerically), no pscan stage (orthogonal concern).
# Minimal budget per model: 1 chain × 5 warmup × 5 samples, max_tree_depth=6.
#
# Pass criterion: every model should finish inside ~5 minutes.  If any
# model is still taking > 2 min per NUTS iteration, the refactor did not
# deliver the expected dispatch-overhead reduction.
#
# Expected wall time with fix: ~15-30 min total (cold JIT + 6 × ~2 min runs).
# Expected wall time if fix broken: will time out, matching prior behavior.
#
# Usage:
#   sbatch cluster/13_bayesian_fullybatched_smoke.slurm
# =============================================================================

#SBATCH --job-name=fb_smoke
#SBATCH --output=logs/fb_smoke_%j.out
#SBATCH --error=logs/fb_smoke_%j.err
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# =============================================================================
# Configuration — SHORTEST viable MCMC budget
# =============================================================================
CHAINS=1
WARMUP=5
SAMPLES=5
MAX_TREE_DEPTH=6
MODELS="qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b"

# Per-model hard timeout (seconds).  If a single model exceeds this, the
# `timeout` wrapper kills it and we move on — prevents one slow model from
# eating the whole wall budget.
PER_MODEL_TIMEOUT=600

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Fully-Batched MCMC Smoke Test"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Project: $PROJECT_ROOT"
echo "Chains: $CHAINS | Warmup: $WARMUP | Samples: $SAMPLES"
echo "Max tree depth: $MAX_TREE_DEPTH"
echo "Per-model timeout: ${PER_MODEL_TIMEOUT}s"
echo "Models: $MODELS"
echo "Start time: $(date)"
echo "============================================================"

if conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu by name"
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/fc37/$USER/"
elif conda activate ds_env 2>/dev/null; then
    echo "Activated ds_env (fallback, no GPU)"
else
    echo "ERROR: Failed to activate conda environment"
    exit 1
fi

python -c "
import jax
devices = jax.devices()
print(f'JAX version: {jax.__version__}, platforms: {devices}')
gpu_devices = [d for d in devices if d.platform == 'gpu']
if gpu_devices:
    print(f'GPU(s) available: {gpu_devices}')
else:
    print('WARNING: No GPU detected')
" || {
    echo "ERROR: JAX import failed"
    exit 1
}
echo ""

# JAX compilation cache
export NUMPYRO_HOST_DEVICE_COUNT=1
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_fb_smoke"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || unset JAX_COMPILATION_CACHE_DIR
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo ""

export PYTHONUNBUFFERED=1
DATA_FILE="output/task_trials_long.csv"
if [[ ! -f "$DATA_FILE" ]]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

declare -A RESULTS
OVERALL_PASS=true

# =============================================================================
# Sequential MCMC smoke — all 6 models, minimal budget
# =============================================================================
for MODEL in $MODELS; do
    echo ""
    echo "------------------------------------------------------------"
    echo "  $MODEL (timeout ${PER_MODEL_TIMEOUT}s)"
    echo "  Started: $(date)"
    echo "------------------------------------------------------------"

    MODEL_START=$(date +%s)

    timeout "$PER_MODEL_TIMEOUT" python scripts/04_model_fitting/b_bayesian/fit_bayesian.py \
        --model "$MODEL" \
        --data "$DATA_FILE" \
        --output output/fb_smoke \
        --chains "$CHAINS" \
        --warmup "$WARMUP" \
        --samples "$SAMPLES" \
        --max-tree-depth "$MAX_TREE_DEPTH" \
        --seed 42

    MODEL_EXIT=$?
    MODEL_END=$(date +%s)
    MODEL_WALL=$((MODEL_END - MODEL_START))

    if [[ $MODEL_EXIT -eq 0 ]]; then
        RESULTS[$MODEL]="PASS (${MODEL_WALL}s)"
    elif [[ $MODEL_EXIT -eq 124 ]]; then
        # 124 = `timeout` expired
        RESULTS[$MODEL]="TIMEOUT (${MODEL_WALL}s, exceeded ${PER_MODEL_TIMEOUT}s)"
        OVERALL_PASS=false
    elif [[ $MODEL_EXIT -eq 137 || $MODEL_EXIT -eq 139 ]]; then
        RESULTS[$MODEL]="KILLED (exit=$MODEL_EXIT, ${MODEL_WALL}s)"
        OVERALL_PASS=false
        echo "  $MODEL: KILLED (exit=$MODEL_EXIT) — aborting remaining models"
        break
    else
        RESULTS[$MODEL]="EXIT=$MODEL_EXIT (${MODEL_WALL}s)"
        # Exit codes other than 0/124/137/139 may still be acceptable (e.g.
        # convergence gate fails with 5-warmup budget).  Only flag as fail
        # if crashed quickly (< 30s = import/compile error).
        if [[ $MODEL_WALL -lt 30 ]]; then
            OVERALL_PASS=false
        fi
    fi
    echo "  $MODEL: exit=$MODEL_EXIT, wall=${MODEL_WALL}s"
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "FULLY-BATCHED SMOKE TEST SUMMARY"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "End time: $(date)"
echo ""
printf "%-12s %s\n" "Model" "Result"
printf "%-12s %s\n" "-------" "------"
for MODEL in $MODELS; do
    printf "%-12s %s\n" "$MODEL" "${RESULTS[$MODEL]}"
done
echo ""
if $OVERALL_PASS; then
    echo "OVERALL: PASS (all models completed inside per-model timeout)"
    echo ""
    echo "If wall times are < 60s per model after JIT warmup, the Issue 1"
    echo "fix has delivered the expected dispatch-overhead reduction."
else
    echo "OVERALL: FAIL (one or more models timed out or crashed)"
fi
echo "============================================================"

# Auto-push logs and results when job completes
if [[ -f cluster/autopush.sh ]]; then
    source cluster/autopush.sh
fi
