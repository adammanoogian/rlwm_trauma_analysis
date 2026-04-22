#!/bin/bash
# =============================================================================
# SLURM Batch Script: GPU-Accelerated Parameter Recovery (Monash M3)
# =============================================================================
# Runs parameter recovery validation using GPU acceleration via JAX CUDA.
# Tests whether MLE fitting can accurately recover known parameter values
# from synthetic data.
#
# Usage:
#   sbatch cluster/11_recovery_gpu.slurm                                         # wmrl_m3 (default)
#   sbatch --export=MODEL=qlearning cluster/11_recovery_gpu.slurm                # Single model
#   sbatch --export=MODEL=wmrl_m3,NSUBJ=100,NDATASETS=10 cluster/11_recovery_gpu.slurm  # Custom settings
#
# Performance (approximate):
#   - CPU sequential:  ~20-30 minutes per dataset
#   - GPU:             ~5-10 minutes per dataset
#
# Pass/Fail Criterion:
#   - Pearson r >= 0.80 per parameter (Senta et al., 2025)
#
# Outputs saved to:
#   - output/recovery/{model}/recovery_results.csv
#   - output/recovery/{model}/recovery_metrics.csv
#   - figures/recovery/{model}/*_recovery.png
#   - cluster/logs/recovery_gpu_{jobid}.out
#
# Reference: https://docs.erc.monash.edu/old-M3/M3/slurm/partitions/
# =============================================================================

#SBATCH --job-name=rlwm_recovery_gpu
#SBATCH --output=cluster/logs/recovery_gpu_%j.out
#SBATCH --error=cluster/logs/recovery_gpu_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

# =============================================================================
# Configuration (set via --export or use defaults)
# =============================================================================
# Recovery budget: 50 subjects x 3 datasets x 20 starts
#   Q-learning:   ~80s/subj x 50 x 3 =  3.3 hours  (well within 12h)
#   WM-RL M3/M5: ~200s/subj x 50 x 3 =  8.3 hours  (fits in 12h)
#   M4 (LBA):   ~1000s/subj x 50 x 3 = 41.7 hours  (needs --time=48:00:00 or reduce NSUBJ=30)
MODEL="${MODEL:-wmrl_m3}"
NSUBJ="${NSUBJ:-50}"
NDATASETS="${NDATASETS:-3}"
NSTARTS="${NSTARTS:-20}"
SEED="${SEED:-42}"

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
echo "RLWM Parameter Recovery (GPU) - M3 SLURM Job"
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
# JAX Compilation Cache
# =============================================================================
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo ""

# Ensure real-time output for tail -f monitoring
export PYTHONUNBUFFERED=1

# =============================================================================
# Run Parameter Recovery (GPU-accelerated)
# =============================================================================

echo "============================================================"
echo "Parameter Recovery Settings"
echo "============================================================"
echo "Model:         $MODEL"
echo "N subjects:    $NSUBJ"
echo "N datasets:    $NDATASETS"
echo "N starts:      $NSTARTS"
echo "Seed:          $SEED"
echo "============================================================"
echo ""

# Start background GPU monitoring (log every 5 min)
if command -v nvidia-smi &>/dev/null; then
    (while true; do
        echo "[GPU $(date +%H:%M:%S)] $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{printf "util=%s%% mem=%s/%sMB temp=%sC", $1, $2, $3, $4}')"
        sleep 300
    done) &
    GPU_MONITOR_PID=$!
fi

python scripts/03_model_prefitting/03_run_model_recovery.py \
    --model $MODEL \
    --n-subjects $NSUBJ \
    --n-datasets $NDATASETS \
    --n-starts $NSTARTS \
    --seed $SEED \
    --use-gpu

EXITCODE=$?

# Stop GPU monitoring
[[ -n "$GPU_MONITOR_PID" ]] && kill "$GPU_MONITOR_PID" 2>/dev/null && wait "$GPU_MONITOR_PID" 2>/dev/null

echo ""
echo "============================================================"
echo "Recovery complete!"
echo "End time: $(date)"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "  output/recovery/$MODEL/recovery_results.csv"
echo "  output/recovery/$MODEL/recovery_metrics.csv"
echo "  figures/recovery/$MODEL/"
echo "============================================================"

exit $EXITCODE
