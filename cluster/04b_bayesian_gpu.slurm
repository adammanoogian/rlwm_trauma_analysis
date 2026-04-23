#!/bin/bash
# =============================================================================
# SLURM Batch Script: Stage 04b — Choice-Only Hierarchical Bayesian Fit (GPU, Template C)
# =============================================================================
# GPU multi-GPU pmap variant of cluster/04b_bayesian_cpu.slurm.
#
# Template C (4 GPUs + chain_method=parallel via pmap) per
# docs/CLUSTER_GPU_LESSONS.md §6. Auto-selected by
# src/rlwm/fitting/numpyro_models._select_chain_method when
# jax.local_device_count() >= n_chains (we request --gres=gpu:4 and default
# --chains 4 so pmap takes the parallel path instead of the slower
# time-multiplexed vectorized path).
#
# Wall-clock baseline: M6b 4-GPU pmap job 54894258
# (logs/bayesian_mgpu_54894258.out, 2026-04-18) completed in ~3h; other
# choice-only models are proportionally faster.
#
# M6b subscale walltime override via orchestrator:
#   sbatch --time=36:00:00 --export=ALL,MODEL=wmrl_m6b,SUBSCALE=1 \
#     cluster/04b_bayesian_gpu.slurm
#
# CPU sibling cluster/04b_bayesian_cpu.slurm is preserved unchanged and
# remains dispatchable via USE_CPU=1 in the master orchestrator
# (cluster/submit_all.sh).
#
# Usage:
#   sbatch cluster/04b_bayesian_gpu.slurm                                  # Default model (wmrl_m3)
#   sbatch --export=ALL,MODEL=qlearning cluster/04b_bayesian_gpu.slurm     # Q-Learning
#   sbatch --export=ALL,MODEL=wmrl_m6b,SUBSCALE=1 \
#     --time=36:00:00 cluster/04b_bayesian_gpu.slurm                       # M6b subscale
#   sbatch --export=ALL,MODEL=wmrl_m3,LIMIT=5 \
#     cluster/04b_bayesian_gpu.slurm                                       # Quick test
#
# Optional GPU type override (default L40S; pins JIT cache to one arch per
# CLUSTER_GPU_LESSONS.md §5):
#   sbatch --export=GPU_TYPE=a100 cluster/04b_bayesian_gpu.slurm
#
# Outputs saved to:
#   - output/bayesian/${MODEL}_individual_fits.csv
#   - output/bayesian/${MODEL}_posterior.nc
#   - output/bayesian/${MODEL}_shrinkage_report.md
#   - output/bayesian/${MODEL}_ppc_results.csv
# =============================================================================

#SBATCH --job-name=rlwm-bayesian-gpu-mgpu
#SBATCH --output=cluster/logs/04b_bayesian_gpu_%j.out
#SBATCH --error=cluster/logs/04b_bayesian_gpu_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
#SBATCH --constraint=${GPU_TYPE:-l40s}

# =============================================================================
# Configuration — override via --export when submitting
# =============================================================================
MODEL="${MODEL:-wmrl_m3}"
LIMIT="${LIMIT:-}"
CHAINS="${CHAINS:-4}"
WARMUP="${WARMUP:-1000}"
SAMPLES="${SAMPLES:-2000}"
SEED="${SEED:-42}"
MAX_TREE_DEPTH="${MAX_TREE_DEPTH:-8}"
SUBSCALE="${SUBSCALE:-0}"  # 1 enables --subscale flag (M6b subscale variant)

# =============================================================================
# Environment Setup (M3 specific)
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Stage 04b (GPU, Template C) — Hierarchical Bayesian Fit"
echo "============================================================"
echo "Job ID:         ${SLURM_JOB_ID:-local}"
echo "Node:           ${SLURMD_NODENAME:-$(hostname)}"
echo "Partition:      ${SLURM_JOB_PARTITION:-gpu}"
echo "GPU(s):         ${CUDA_VISIBLE_DEVICES:-none}"
echo "Project:        $PROJECT_ROOT"
echo "Model:          $MODEL"
echo "Chains:         $CHAINS"
echo "Warmup:         $WARMUP"
echo "Samples:        $SAMPLES"
echo "Seed:           $SEED"
echo "Max tree depth: $MAX_TREE_DEPTH"
echo "Subscale:       $SUBSCALE"
echo "Limit:          ${LIMIT:-<all participants>}"
echo "Start time:     $(date)"
echo "============================================================"

# Activate rlwm_gpu conda environment. Try by name (uses user's .condarc),
# then explicit /scratch paths. NO ds_env fallback — the whole point of this
# script is Template C pmap on GPU; ds_env would silently degrade the
# chain_method to vectorized and defeat the purpose.
if conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu by name"
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/fc37/$USER/"
elif conda activate /scratch/${PROJECT:-fc37}/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/${PROJECT:-fc37}/$USER/"
else
    echo "ERROR: Failed to activate rlwm_gpu — GPU pipeline cannot run on ds_env (CPU-only)"
    echo "       Create it first with: bash cluster/00_setup_env_gpu.sh"
    echo "       Or use USE_CPU=1 with cluster/submit_all.sh for the CPU path."
    exit 1
fi

# Verify GPU allocation matches chain count. If n_gpu < CHAINS, NumPyro's
# _select_chain_method falls back to 'vectorized' which defeats the entire
# point of Template C.
CHAINS_NUM=$CHAINS
python -c "
import jax
devices = jax.devices()
n_gpu = sum(1 for d in devices if d.platform == 'gpu')
print(f'JAX {jax.__version__} devices: {devices}')
print(f'GPU count: {n_gpu}')
if n_gpu < ${CHAINS_NUM}:
    print(f'WARNING: {n_gpu} GPU(s) < ${CHAINS_NUM} chains. chain_method will degrade to vectorized.')
" || { echo 'ERROR: JAX import failed'; exit 1; }
echo ""

# =============================================================================
# Cache + env exports — encode CLUSTER_GPU_LESSONS.md operational invariants
# =============================================================================
# §2: NUMPYRO_HOST_DEVICE_COUNT must match $CHAINS so pmap sees all devices
# §5: per-model + _gpu-suffixed cache so CPU/GPU + cross-model don't contaminate
# §6: PYTHONUNBUFFERED=1 so logs flush in real time on stalls
export NUMPYRO_HOST_DEVICE_COUNT="$CHAINS"
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_04b_bayesian_gpu_${MODEL}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export PYTHONUNBUFFERED=1
mkdir -p "$JAX_COMPILATION_CACHE_DIR" || {
    echo 'WARNING: cache dir create failed'
    unset JAX_COMPILATION_CACHE_DIR
}
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo ""

# Create output + log directories
mkdir -p cluster/logs
mkdir -p output/bayesian

# =============================================================================
# Verify required input files
# =============================================================================
DATA_FILE="output/task_trials_long.csv"
if [[ ! -f "$DATA_FILE" ]]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    echo "Run the data parsing script first:"
    echo "  python scripts/01_data_preprocessing/01_parse_raw_data.py"
    exit 1
fi

if [ "$SUBSCALE" = "1" ]; then
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

# =============================================================================
# Build fit_bayesian.py command
# =============================================================================
# NO --use-gpu flag: pmap selection is automatic via
# numpyro_models._select_chain_method when n_devices >= n_chains. The
# --use-gpu flag was the single-GPU-mode convention (now obsolete for
# Template C).
CMD=(python scripts/04_model_fitting/b_bayesian/fit_bayesian.py
     --model "$MODEL"
     --data "$DATA_FILE"
     --chains "$CHAINS"
     --warmup "$WARMUP"
     --samples "$SAMPLES"
     --seed "$SEED"
     --max-tree-depth "$MAX_TREE_DEPTH")

if [ "$SUBSCALE" = "1" ]; then
    CMD+=(--subscale)
fi

if [[ -n "$LIMIT" ]]; then
    CMD+=(--limit "$LIMIT")
fi

# =============================================================================
# Run hierarchical fit (GPU Template C)
# =============================================================================
echo "============================================================"
echo "Fitting Bayesian model: $MODEL (Template C, 4 GPUs + pmap)"
echo "Started: $(date)"
echo "============================================================"

# Start background GPU monitoring (log every 5 min during fitting)
if command -v nvidia-smi &>/dev/null; then
    (while true; do
        echo "[GPU $(date +%H:%M:%S)] $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{printf "util=%s%% mem=%s/%sMB temp=%sC", $1, $2, $3, $4}')"
        sleep 300
    done) &
    GPU_MONITOR_PID=$!
fi

echo "[$(date)] Starting ${MODEL} hierarchical Bayesian fit${SUBSCALE:+ (subscale=$SUBSCALE)}"
echo "  Command: ${CMD[*]}"
echo ""

"${CMD[@]}"
EXIT_CODE=$?

# Stop GPU monitoring
[[ -n "$GPU_MONITOR_PID" ]] && kill "$GPU_MONITOR_PID" 2>/dev/null && wait "$GPU_MONITOR_PID" 2>/dev/null
unset GPU_MONITOR_PID

echo ""
echo "[$(date)] ${MODEL} hierarchical fit finished (exit code: $EXIT_CODE)"

# =============================================================================
# Post-fit expected-artifacts audit (mirror of 04b_bayesian_cpu.slurm)
# =============================================================================
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
    echo "*** ERROR: Model $MODEL FAILED with exit code $EXIT_CODE ***"
    echo "*** Check stderr log: cluster/logs/04b_bayesian_gpu_${SLURM_JOB_ID}.err ***"
fi

# =============================================================================
# Job Resource Report
# =============================================================================
echo ""
echo "============================================================"
echo "JOB RESOURCE REPORT"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"

if command -v nvidia-smi &>/dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
        --format=csv,noheader 2>/dev/null | while IFS=',' read -r name total used free util temp; do
        echo "  Model:       $name"
        echo "  VRAM Total:  $total"
        echo "  VRAM Used:   $used"
        echo "  VRAM Free:   $free"
        echo "  Utilization: $util"
        echo "  Temperature: $temp"
    done
fi

echo "SLURM Accounting:"
sacct -j "$SLURM_JOB_ID" \
    --format=JobID%20,Elapsed,MaxRSS,MaxVMSize,ReqMem,AllocCPUS,State%12 \
    --noheader 2>/dev/null | head -5 || echo "  (not yet available)"
echo "============================================================"

# Auto-push logs and results when job completes
source cluster/autopush.sh
