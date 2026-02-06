#!/bin/bash
#SBATCH --job-name=ppc_gpu
#SBATCH --output=logs/ppc_gpu_%j.out
#SBATCH --error=logs/ppc_gpu_%j.err
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Posterior Predictive Checks with GPU acceleration
# Usage: sbatch cluster/run_ppc_gpu.slurm
# Or with custom settings:
# sbatch --export=MODEL=wmrl_m3 cluster/run_ppc_gpu.slurm

# Default values (can be overridden via --export)
MODEL=${MODEL:-wmrl_m3}
SKIP_MODEL_RECOVERY=${SKIP_MODEL_RECOVERY:-false}

# Load modules and activate environment
module load cuda/11.8
source activate rlwm_gpu

# Create logs directory if needed
mkdir -p logs

echo "=========================================="
echo "Posterior Predictive Check (GPU)"
echo "=========================================="
echo "Model: $MODEL"
echo "Skip model recovery: $SKIP_MODEL_RECOVERY"
echo "=========================================="

# Build command
CMD="python scripts/09_run_ppc.py --model $MODEL --use-gpu"
if [ "$SKIP_MODEL_RECOVERY" = "true" ]; then
    CMD="$CMD --skip-model-recovery"
fi

# Run PPC
$CMD

echo "=========================================="
echo "PPC complete!"
echo "Results: output/ppc/$MODEL/"
echo "Figures: figures/ppc/$MODEL/"
echo "=========================================="
