#!/bin/bash
# =============================================================================
# SLURM Batch Script: GPU-Accelerated MLE Fitting for RLWM Models (Monash M3)
# =============================================================================
# Runs MLE parameter estimation using GPU acceleration via JAX CUDA.
# Requires the rlwm_gpu conda environment (see environment_gpu.yml).
#
# Usage:
#   sbatch cluster/run_mle_gpu.slurm                         # All models
#   sbatch --export=MODEL=wmrl_m3 cluster/run_mle_gpu.slurm  # Single model
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
#SBATCH --time=00:15:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

# =============================================================================
# Configuration (set via --export or use defaults)
# =============================================================================
MODELS="${MODEL:-qlearning wmrl wmrl_m3}"

# =============================================================================
# Environment Setup (M3 specific)
# =============================================================================

# Load miniforge3 and CUDA
module load miniforge3
module load cuda/12.1.1

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
conda activate rlwm_gpu

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

for model in $MODELS; do
    echo ""
    echo "============================================================"
    echo "Fitting model: $model (GPU-accelerated)"
    echo "Started: $(date)"
    echo "============================================================"

    python scripts/fitting/fit_mle.py \
        --model "$model" \
        --data "$DATA_FILE" \
        --output output/mle/ \
        --n-starts 20 \
        --seed 42 \
        --use-gpu

    echo "Completed: $(date)"
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
