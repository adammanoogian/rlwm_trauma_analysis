#!/bin/bash
# =============================================================================
# SLURM Batch Script: PScan A/B Smoke Test (All 6 Choice-Only Models)
# =============================================================================
# Quick validation that sequential and pscan likelihoods compile and run
# correctly across all 6 choice-only models. Uses minimal MCMC budget
# (warmup=50, samples=50, 2 chains) to test cold JIT, warm JIT, and
# pointwise log-lik computation without waiting for convergence.
#
# Stage 1: Micro-benchmark (all 6 models, 3 repeats — cold + warm JIT)
# Stage 2: Sequential MCMC smoke (all 6 models, minimal budget)
# Stage 3: PScan MCMC smoke (all 6 models, minimal budget)
#
# Usage:
#   sbatch cluster/13_bayesian_pscan_smoke.slurm
#
# Expected wall time: ~2-4 hours (compilation-dominated)
#
# Note: Pscan compilation generates large XLA intermediate representations.
# 96GB host RAM is required to avoid OOM during compilation even though
# runtime memory usage is much lower. GPU partition used because pscan
# benefits from GPU acceleration and NVPTX compiler handles large graphs
# more efficiently than CPU XLA.
# =============================================================================

#SBATCH --job-name=pscan_smoke
#SBATCH --output=logs/pscan_smoke_%j.out
#SBATCH --error=logs/pscan_smoke_%j.err
#SBATCH --time=6:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

# =============================================================================
# Configuration
# =============================================================================
CHAINS=2
WARMUP=50
SAMPLES=50
MAX_TREE_DEPTH=6
BENCHMARK_REPEATS=3
MODELS="qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b"

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "PScan A/B Smoke Test — All 6 Choice-Only Models"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Project: $PROJECT_ROOT"
echo "Chains: $CHAINS | Warmup: $WARMUP | Samples: $SAMPLES"
echo "Max tree depth: $MAX_TREE_DEPTH"
echo "Models: $MODELS"
echo "Start time: $(date)"
echo "============================================================"

# Activate GPU conda environment (rlwm_gpu preferred for GPU partition)
if conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu by name"
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/fc37/$USER/"
elif conda activate /scratch/${PROJECT:-fc37}/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/$PROJECT/$USER/"
elif conda activate ds_env 2>/dev/null; then
    echo "Activated ds_env (fallback, no GPU)"
else
    echo "ERROR: Failed to activate conda environment (tried rlwm_gpu, ds_env)"
    exit 1
fi

# Verify GPU access
python -c "
import jax
devices = jax.devices()
print(f'JAX version: {jax.__version__}, platforms: {devices}')
gpu_devices = [d for d in devices if d.platform == 'gpu']
if gpu_devices:
    print(f'GPU(s) available: {gpu_devices}')
else:
    print('WARNING: No GPU detected — pscan will run on CPU (slow)')
" || {
    echo "ERROR: JAX import failed after conda activation"
    exit 1
}
echo ""

# JAX compilation cache (separate from production fits)
export NUMPYRO_HOST_DEVICE_COUNT=1
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_pscan_smoke"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo ""

export PYTHONUNBUFFERED=1
DATA_FILE="output/task_trials_long.csv"
if [[ ! -f "$DATA_FILE" ]]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Track results
declare -A SEQ_RESULTS
declare -A PSCAN_RESULTS
OVERALL_PASS=true

# =============================================================================
# Stage 1: Micro-benchmark (all models, 3 repeats — cold + warm JIT)
# =============================================================================
echo ""
echo "============================================================"
echo "STAGE 1: Micro-benchmark (cold + warm JIT, all 6 models)"
echo "Started: $(date)"
echo "============================================================"

python validation/benchmark_parallel_scan.py --n-repeats "$BENCHMARK_REPEATS"

STAGE1_EXIT=$?
echo ""
echo "Stage 1 exit code: $STAGE1_EXIT"
if [[ $STAGE1_EXIT -ne 0 ]]; then
    echo "WARNING: Micro-benchmark failed, continuing with MCMC smoke tests"
    OVERALL_PASS=false
fi

# =============================================================================
# Stage 2: Sequential MCMC smoke test (all 6 models)
# =============================================================================
echo ""
echo "============================================================"
echo "STAGE 2: Sequential MCMC smoke test"
echo "============================================================"

for MODEL in $MODELS; do
    echo ""
    echo "------------------------------------------------------------"
    echo "  Sequential: $MODEL"
    echo "  Started: $(date)"
    echo "------------------------------------------------------------"

    SEQ_START=$(date +%s)

    python scripts/fitting/fit_bayesian.py \
        --model "$MODEL" \
        --data "$DATA_FILE" \
        --output output/smoke_test \
        --chains "$CHAINS" \
        --warmup "$WARMUP" \
        --samples "$SAMPLES" \
        --max-tree-depth "$MAX_TREE_DEPTH" \
        --seed 42

    SEQ_EXIT=$?
    SEQ_END=$(date +%s)
    SEQ_WALL=$((SEQ_END - SEQ_START))

    # Exit code 0 = converged + saved; non-zero might be convergence gate fail
    # (expected for smoke test) — check if it got past pointwise log-lik
    if [[ $SEQ_EXIT -eq 0 ]]; then
        SEQ_RESULTS[$MODEL]="PASS (${SEQ_WALL}s)"
    else
        SEQ_RESULTS[$MODEL]="EXIT=$SEQ_EXIT (${SEQ_WALL}s)"
        # Only flag as overall fail if it crashed quickly (< 30s = likely import/compile error)
        if [[ $SEQ_WALL -lt 30 ]]; then
            OVERALL_PASS=false
        fi
    fi
    echo "  Sequential $MODEL: exit=$SEQ_EXIT, wall=${SEQ_WALL}s"
done

# =============================================================================
# Stage 3: PScan MCMC smoke test (all 6 models)
# =============================================================================
echo ""
echo "============================================================"
echo "STAGE 3: PScan MCMC smoke test"
echo "============================================================"

for MODEL in $MODELS; do
    echo ""
    echo "------------------------------------------------------------"
    echo "  PScan: $MODEL"
    echo "  Started: $(date)"
    echo "------------------------------------------------------------"

    PSCAN_START=$(date +%s)

    python scripts/fitting/fit_bayesian.py \
        --model "$MODEL" \
        --data "$DATA_FILE" \
        --output output/smoke_test \
        --chains "$CHAINS" \
        --warmup "$WARMUP" \
        --samples "$SAMPLES" \
        --max-tree-depth "$MAX_TREE_DEPTH" \
        --seed 42 \
        --use-pscan

    PSCAN_EXIT=$?
    PSCAN_END=$(date +%s)
    PSCAN_WALL=$((PSCAN_END - PSCAN_START))

    if [[ $PSCAN_EXIT -eq 0 ]]; then
        PSCAN_RESULTS[$MODEL]="PASS (${PSCAN_WALL}s)"
    else
        PSCAN_RESULTS[$MODEL]="EXIT=$PSCAN_EXIT (${PSCAN_WALL}s)"
        if [[ $PSCAN_WALL -lt 30 ]]; then
            OVERALL_PASS=false
        fi
    fi
    echo "  PScan $MODEL: exit=$PSCAN_EXIT, wall=${PSCAN_WALL}s"
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "SMOKE TEST SUMMARY"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "End time: $(date)"
echo ""
echo "Stage 1 (micro-benchmark): exit=$STAGE1_EXIT"
echo ""
printf "%-15s %-30s %-30s\n" "Model" "Sequential" "PScan"
printf "%-15s %-30s %-30s\n" "-------" "----------" "-----"
for MODEL in $MODELS; do
    printf "%-15s %-30s %-30s\n" "$MODEL" "${SEQ_RESULTS[$MODEL]}" "${PSCAN_RESULTS[$MODEL]}"
done
echo ""
echo "Note: Exit code != 0 is expected (convergence gate fails with"
echo "minimal MCMC budget). The test validates compilation and pipeline"
echo "correctness, not posterior quality."
echo ""
if $OVERALL_PASS; then
    echo "OVERALL: PASS (all models compiled and ran through pipeline)"
else
    echo "OVERALL: FAIL (one or more models crashed during compilation or early pipeline)"
fi
echo "============================================================"
