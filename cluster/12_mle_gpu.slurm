#!/bin/bash
# =============================================================================
# SLURM Batch Script: GPU-Accelerated MLE Fitting for RLWM Models (Monash M3)
# =============================================================================
# Runs MLE parameter estimation using GPU acceleration via JAX CUDA.
# Requires the rlwm_gpu conda environment (see environment_gpu.yml).
#
# Usage:
#   sbatch cluster/12_mle_gpu.slurm                                    # All models
#   sbatch --export=MODEL=wmrl_m3 cluster/12_mle_gpu.slurm             # Single model
#   sbatch --export=MODEL=qlearning,LIMIT=3 cluster/12_mle_gpu.slurm   # Quick test (3 participants)
#
# Performance (approximate for 47 participants):
#   - CPU sequential:  ~30 minutes per model
#   - CPU parallel:    ~3 minutes per model (16 cores)
#   - GPU:             ~2-5 minutes per model
#
# M3 GPU Options (request via --gres=gpu:TYPE:N):
#   - V100:  16GB VRAM, good balance (m3g partition)
#   - A100:  40GB VRAM, fastest (gpu partition)
#   - A40:   48GB VRAM, large memory (gpu partition)
#   - T4:    16GB VRAM, basic GPU (gpu/desktop partition)
#
# Outputs saved to:
#   - output/mle/{model}_individual_fits.csv
#   - output/mle/{model}_group_summary.csv
#   - cluster/logs/mle_gpu_{jobid}.out
#
# Reference: https://docs.erc.monash.edu/old-M3/M3/slurm/partitions/
# =============================================================================

#SBATCH --job-name=rlwm_mle_gpu
#SBATCH --output=cluster/logs/mle_gpu_%j.out
#SBATCH --error=cluster/logs/mle_gpu_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

# =============================================================================
# Configuration (set via --export or use defaults)
# =============================================================================
MODELS="${MODEL:-qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b wmrl_m4}"
LIMIT="${LIMIT:-}"

# =============================================================================
# Environment Setup (M3 specific)
# =============================================================================

# Load miniforge3 only - JAX pip packages bundle CUDA libraries
# Do NOT load cuda module - it conflicts with pip-installed CUDA libraries
module load miniforge3

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "RLWM GPU-Accelerated MLE Fitting - M3 SLURM Job"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Partition: ${SLURM_JOB_PARTITION:-gpu}"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-none}"
echo "Project: $PROJECT_ROOT"
echo "Start time: $(date)"
echo "============================================================"

# Activate GPU conda environment
# Try by name first (uses user's .condarc), then fall back to common paths
if conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu by name"
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/fc37/$USER/"
elif conda activate /scratch/$PROJECT/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/$PROJECT/$USER/"
else
    echo "ERROR: Failed to activate 'rlwm_gpu' conda environment"
    echo "Tried: rlwm_gpu (by name), /scratch/fc37/$USER/..., /scratch/$PROJECT/$USER/..."
    echo "Create it first with: bash cluster/00_setup_env_gpu.sh"
    exit 1
fi

# Verify GPU is available to JAX
echo ""
echo "Verifying GPU access..."
python -c "
import jax
devices = jax.devices()
print(f'JAX devices: {devices}')
gpu_devices = [d for d in devices if d.platform == \"gpu\"]
if gpu_devices:
    print(f'GPU(s) available: {gpu_devices}')
else:
    print('WARNING: No GPU detected! JAX will use CPU.')
    print('Check CUDA module and conda environment.')
"
echo ""

# Create logs directory if needed
mkdir -p cluster/logs

# =============================================================================
# JAX Compilation Cache (optional for GPU but helps with restarts)
# =============================================================================
# GPU compilation uses CUDA/PTX which is portable across GPU types, but caching
# still reduces startup time on subsequent runs. Unlike CPU caching, GPU caching
# doesn't have the CPU feature mismatch problem.
# $PROJECT may be empty on some GPU nodes — fall back to fc37
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo ""

# =============================================================================
# Run MLE Fitting (GPU-accelerated)
# =============================================================================

DATA_FILE="output/task_trials_long.csv"

# Check data file exists
if [[ ! -f "$DATA_FILE" ]]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    echo "Run the data parsing script first:"
    echo "  python scripts/01_parse_raw_data.py"
    exit 1
fi

# =============================================================================
# Checkpoint Cleanup (delete stale checkpoints unless opted out)
# =============================================================================
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-}"

if [[ -z "$KEEP_CHECKPOINTS" ]]; then
    echo "Cleaning up stale checkpoints..."
    for model in $MODELS; do
        CKPT="output/mle/${model}_checkpoint.csv"
        if [[ -f "$CKPT" ]]; then
            echo "  Deleted: $CKPT"
            rm -f "$CKPT"
        fi
    done
    echo ""
else
    echo "KEEP_CHECKPOINTS set — preserving existing checkpoints for resume"
    echo ""
fi

# =============================================================================
# Parallel Dispatch: if multiple models, submit separate jobs and exit
# =============================================================================
MODEL_COUNT=$(echo $MODELS | wc -w)

if [[ "$MODEL_COUNT" -gt 1 ]]; then
    echo "Dispatching $MODEL_COUNT models as parallel GPU jobs..."
    echo ""

    JOBIDS=()
    for model in $MODELS; do
        EXPORTS="MODEL=$model"
        [[ -n "$LIMIT" ]] && EXPORTS="$EXPORTS,LIMIT=$LIMIT"
        [[ -n "$KEEP_CHECKPOINTS" ]] && EXPORTS="$EXPORTS,KEEP_CHECKPOINTS=$KEEP_CHECKPOINTS"

        JOBID=$(sbatch --export="$EXPORTS" --job-name="mle_${model}" "$0" | awk '{print $NF}')
        JOBIDS+=("$JOBID")
        echo "  $model -> Job $JOBID"
    done

    echo ""
    echo "All jobs submitted. Monitor with:"
    echo "  squeue -u $USER"
    echo ""
    echo "Tail logs:"
    for i in "${!JOBIDS[@]}"; do
        model=$(echo $MODELS | cut -d' ' -f$((i+1)))
        echo "  tail -f cluster/logs/mle_gpu_${JOBIDS[$i]}.out   # $model"
    done
    exit 0
fi

for model in $MODELS; do
    echo ""
    echo "============================================================"
    echo "Fitting model: $model (GPU-accelerated)"
    echo "Started: $(date)"
    echo "============================================================"

    # Build command with optional --limit
    CMD="python scripts/fitting/fit_mle.py \
        --model $model \
        --data $DATA_FILE \
        --output output/mle/ \
        --n-starts 50 \
        --seed 42 \
        --use-gpu \
        --compute-diagnostics"

    if [[ -n "$LIMIT" ]]; then
        CMD="$CMD --limit $LIMIT"
    fi

    eval $CMD
    EXIT_CODE=$?

    if [[ $EXIT_CODE -ne 0 ]]; then
        echo ""
        echo "*** ERROR: Model $model FAILED with exit code $EXIT_CODE ***"
        echo "*** Check stderr log: cluster/logs/mle_gpu_${SLURM_JOB_ID}.err ***"
        echo ""
    fi

    echo "Completed: $(date) (exit code: $EXIT_CODE)"
done

echo ""
echo "============================================================"
echo "All models complete!"
echo "End time: $(date)"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  output/mle/*_individual_fits.csv"
echo "  output/mle/*_group_summary.csv"
echo "============================================================"

# =============================================================================
# Job Resource Report
# =============================================================================
echo ""
echo "============================================================"
echo "JOB RESOURCE REPORT"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo ""

# GPU stats via nvidia-smi (available on GPU nodes)
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
    echo ""
fi

# SLURM accounting (may not be available until job completes)
echo "SLURM Accounting:"
sacct -j "$SLURM_JOB_ID" \
    --format=JobID%20,Elapsed,MaxRSS,MaxVMSize,ReqMem,AllocCPUS,State%12 \
    --noheader 2>/dev/null | head -5 || echo "  (not yet available)"
echo ""

# jobstats hint
echo "For detailed stats after completion:"
echo "  jobstats ${SLURM_JOB_ID}"
echo "============================================================"
