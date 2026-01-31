#!/bin/bash
# =============================================================================
# GPU Environment Setup Script for Monash M3 Cluster
# =============================================================================
# Creates the rlwm_gpu conda environment with JAX CUDA support.
#
# Usage:
#   # On M3 login node or interactive session:
#   bash cluster/setup_env_gpu.sh
#
# Or manually:
#   module load miniforge3
#   module load cuda/12.1.1
#   mamba env create -f environment_gpu.yml
#   conda activate rlwm_gpu
#
# Note: This script uses mamba for faster dependency resolution.
# If mamba is unavailable, it falls back to conda.
# =============================================================================

set -e  # Exit on error

echo "============================================================"
echo "RLWM GPU Environment Setup (M3 Cluster)"
echo "============================================================"

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "Project: $PROJECT_ROOT"

# Load required modules
echo ""
echo "Loading modules..."
module load miniforge3
module load cuda/12.1.1

# Check CUDA version
echo ""
echo "CUDA version:"
nvcc --version 2>/dev/null || echo "nvcc not found (CUDA module may not be loaded)"

# Check if environment already exists
if conda env list | grep -q "^rlwm_gpu "; then
    echo ""
    echo "Environment 'rlwm_gpu' already exists."
    read -p "Remove and recreate? (y/N): " response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n rlwm_gpu -y
    else
        echo "Keeping existing environment. Exiting."
        exit 0
    fi
fi

# Create environment using mamba (faster) or conda (fallback)
echo ""
echo "Creating rlwm_gpu environment..."
if command -v mamba &> /dev/null; then
    echo "Using mamba for faster installation..."
    mamba env create -f environment_gpu.yml
else
    echo "Mamba not found, using conda (this may take longer)..."
    conda env create -f environment_gpu.yml
fi

# Activate and verify
echo ""
echo "Verifying installation..."
conda activate rlwm_gpu

echo ""
echo "Python version:"
python --version

echo ""
echo "Testing JAX GPU support..."
python -c "
import jax
print(f'JAX version: {jax.__version__}')
devices = jax.devices()
print(f'Available devices: {devices}')
gpu_devices = [d for d in devices if d.platform == 'gpu']
if gpu_devices:
    print(f'SUCCESS: GPU(s) detected: {gpu_devices}')
    # Quick computation test
    import jax.numpy as jnp
    x = jnp.ones((1000, 1000))
    y = jnp.dot(x, x)
    y.block_until_ready()
    print('GPU computation test passed!')
else:
    print('WARNING: No GPU detected. JAX will use CPU.')
    print('This may be normal on a login node. Test on a GPU node.')
"

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "To use the GPU environment:"
echo "  conda activate rlwm_gpu"
echo ""
echo "To run GPU-accelerated MLE fitting:"
echo "  sbatch cluster/run_mle_gpu.slurm"
echo ""
echo "============================================================"
