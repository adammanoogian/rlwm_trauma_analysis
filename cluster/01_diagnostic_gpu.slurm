#!/bin/bash
# =============================================================================
# SLURM Batch Script: GPU Performance Diagnostic
# =============================================================================
# Quick diagnostic to test JAX/GPU performance before running full fitting.
#
# Usage:
#   sbatch cluster/01_diagnostic_gpu.slurm
#
# =============================================================================

#SBATCH --job-name=rlwm_diag
#SBATCH --output=logs/diagnostic_%j.out
#SBATCH --error=logs/diagnostic_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "RLWM GPU Diagnostic - M3 SLURM Job"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-none}"
echo "Project: $PROJECT_ROOT"
echo "Start time: $(date)"
echo "============================================================"

# Activate GPU conda environment
if conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu"
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/fc37/$USER/"
else
    echo "ERROR: Failed to activate 'rlwm_gpu' conda environment"
    exit 1
fi

# Verify GPU
echo ""
python -c "
import jax
devices = jax.devices()
print(f'JAX devices: {devices}')
gpu_devices = [d for d in devices if d.platform == \"gpu\"]
print(f'GPU available: {len(gpu_devices) > 0}')
"
echo ""

# Create logs directory
mkdir -p logs

# =============================================================================
# Run Diagnostic
# =============================================================================
echo "Running GPU performance diagnostic..."
echo ""

cd scripts/fitting

# Force unbuffered Python output so we see progress in real-time
export PYTHONUNBUFFERED=1

python -u diagnose_gpu.py

echo ""
echo "============================================================"
echo "Diagnostic complete!"
echo "End time: $(date)"
echo "============================================================"
