#!/bin/bash
# =============================================================================
# SLURM Batch Script: GPU-Accelerated Bayesian Fitting for RLWM Models (Monash M3)
# =============================================================================
# Runs hierarchical Bayesian parameter estimation using NumPyro/NUTS with
# GPU acceleration via JAX CUDA.
# Requires the rlwm_gpu conda environment (see environment_gpu.yml).
#
# Usage:
#   sbatch cluster/04b_bayesian_gpu.slurm                                    # Default model (wmrl_m3)
#   sbatch --export=MODEL=qlearning cluster/04b_bayesian_gpu.slurm           # Q-Learning
#   sbatch --export=MODEL=wmrl_m3,LIMIT=5 cluster/04b_bayesian_gpu.slurm    # Quick test (5 participants)
#   sbatch --time=48:00:00 --gres=gpu:a100:1 --mem=96G \
#     --export=MODEL=wmrl_m4 cluster/04b_bayesian_gpu.slurm                  # M4 LBA (GPU required)
#
# Performance (approximate):
#   154 participants, wmrl_m3 (NUTS, 4 chains, 1000 warmup, 2000 samples): ~4-6 hours (A100)
#   154 participants, qlearning (NUTS, 4 chains, 1000 warmup, 2000 samples): ~2-3 hours (A100)
#
# M3 GPU Options (request via --gres=gpu:TYPE:N):
#   - V100:  16GB VRAM, good balance (m3g partition)
#   - A100:  40GB VRAM, fastest (gpu partition)
#   - A40:   48GB VRAM, large memory (gpu partition)
#
# Outputs saved to:
#   - output/bayesian/{model}_individual_fits.csv
#   - output/bayesian/{model}_inference_data.nc (ArviZ NetCDF)
#   - cluster/logs/bayesian_gpu_{jobid}.out
#
# Reference: https://docs.erc.monash.edu/old-M3/M3/slurm/partitions/
# =============================================================================

#SBATCH --job-name=rlwm-bayesian-gpu
#SBATCH --output=cluster/logs/bayesian_gpu_%j.out
#SBATCH --error=cluster/logs/bayesian_gpu_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

# =============================================================================
# Configuration (set via --export or use defaults)
# =============================================================================
MODEL="${MODEL:-wmrl_m3}"
LIMIT="${LIMIT:-}"
CHAINS="${CHAINS:-4}"
WARMUP="${WARMUP:-1000}"
SAMPLES="${SAMPLES:-2000}"

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
echo "RLWM GPU-Accelerated Bayesian Fitting - M3 SLURM Job"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Partition: ${SLURM_JOB_PARTITION:-gpu}"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-none}"
echo "Project: $PROJECT_ROOT"
echo "Model: $MODEL"
echo "Chains: $CHAINS  |  Warmup: $WARMUP  |  Samples: $SAMPLES"
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

# NumPyro host device count — 1 GPU per job
export NUMPYRO_HOST_DEVICE_COUNT=1

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
# Run Bayesian Fitting (GPU-accelerated)
# =============================================================================

DATA_FILE="output/task_trials_long.csv"

# Ensure real-time output for tail -f monitoring
export PYTHONUNBUFFERED=1

# Check data file exists
if [[ ! -f "$DATA_FILE" ]]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    echo "Run the data parsing script first:"
    echo "  python scripts/01_data_preprocessing/01_parse_raw_data.py"
    exit 1
fi

echo ""
echo "============================================================"
echo "Fitting Bayesian model: $MODEL (GPU-accelerated)"
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

# Build command with optional --limit
CMD="python scripts/04_model_fitting/b_bayesian/fit_bayesian.py \
    --model $MODEL \
    --data $DATA_FILE \
    --output output/bayesian/ \
    --chains $CHAINS \
    --warmup $WARMUP \
    --samples $SAMPLES \
    --use-gpu"

if [[ -n "$LIMIT" ]]; then
    CMD="$CMD --limit $LIMIT"
fi

eval $CMD
EXIT_CODE=$?

# Stop GPU monitoring
[[ -n "$GPU_MONITOR_PID" ]] && kill "$GPU_MONITOR_PID" 2>/dev/null && wait "$GPU_MONITOR_PID" 2>/dev/null
unset GPU_MONITOR_PID

if [[ $EXIT_CODE -ne 0 ]]; then
    echo ""
    echo "*** ERROR: Model $MODEL FAILED with exit code $EXIT_CODE ***"
    echo "*** Check stderr log: cluster/logs/bayesian_gpu_${SLURM_JOB_ID}.err ***"
    echo ""
fi

echo "Completed: $(date) (exit code: $EXIT_CODE)"

echo ""
echo "============================================================"
echo "Bayesian fitting complete!"
echo "End time: $(date)"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  output/bayesian/${MODEL}_individual_fits.csv"
echo "  output/bayesian/${MODEL}_inference_data.nc"
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

# Auto-push logs and results when job completes
source cluster/autopush.sh
