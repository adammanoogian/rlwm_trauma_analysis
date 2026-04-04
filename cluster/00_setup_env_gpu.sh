#!/bin/bash
# =============================================================================
# GPU Environment Setup Script for Monash M3 Cluster
# =============================================================================
# Creates or updates the rlwm_gpu conda environment with JAX CUDA support.
#
# Usage:
#   bash cluster/00_setup_env_gpu.sh           # Fresh install or interactive update
#   bash cluster/00_setup_env_gpu.sh --update  # Force update existing env (no delete)
#   bash cluster/00_setup_env_gpu.sh --fresh   # Delete and recreate from scratch
#
# Note: CUDA module is NOT required. JAX's pip packages bundle their own CUDA
# runtime libraries (cuSPARSE, cuBLAS, cuDNN, etc.). GPU access works via the
# cluster's NVIDIA kernel driver on GPU nodes.
# =============================================================================

set -e  # Exit on error

# Parse arguments
MODE="interactive"
while [[ $# -gt 0 ]]; do
    case $1 in
        --update|-u)
            MODE="update"
            shift
            ;;
        --fresh|-f)
            MODE="fresh"
            shift
            ;;
        --help|-h)
            echo "Usage: bash cluster/00_setup_env_gpu.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --update, -u    Update existing environment (no delete)"
            echo "  --fresh, -f     Delete and recreate environment"
            echo "  --help, -h      Show this help message"
            echo ""
            echo "Default: Interactive mode (prompts for action if env exists)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "RLWM GPU Environment Setup (M3 Cluster)"
echo "============================================================"

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "Project: $PROJECT_ROOT"
echo "Mode: $MODE"

# Load miniforge3 (DO NOT load cuda module - JAX bundles CUDA libraries)
echo ""
echo "Loading modules..."
module load miniforge3

# Check if environment exists
ENV_EXISTS=false
if conda env list | grep -q "rlwm_gpu"; then
    ENV_EXISTS=true
    echo "Environment 'rlwm_gpu' already exists."
fi

# =============================================================================
# Handle different modes
# =============================================================================

if [[ "$MODE" == "update" ]]; then
    # --update: Force update existing environment
    if [[ "$ENV_EXISTS" == false ]]; then
        echo "ERROR: Environment 'rlwm_gpu' does not exist. Use without --update for fresh install."
        exit 1
    fi

    echo ""
    echo "Updating existing environment..."

    # Activate environment
    conda activate rlwm_gpu

    # Remove conflicting conda CUDA packages if present
    echo "Removing conflicting CUDA packages (if any)..."
    conda remove cuda-nvcc cudatoolkit --force -y 2>/dev/null || true

    # Upgrade Python to 3.11 if needed
    PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "$PYTHON_VERSION" != "3.11" ]]; then
        echo "Upgrading Python $PYTHON_VERSION -> 3.11..."
        conda install python=3.11 -y
    fi

    # Force reinstall JAX with CUDA support
    echo "Installing/upgrading JAX with CUDA 12 support..."
    pip install --upgrade --force-reinstall "jax[cuda12]>=0.5.0" "jaxopt>=0.8.0"

    echo ""
    echo "Update complete!"

elif [[ "$MODE" == "fresh" ]]; then
    # --fresh: Delete and recreate
    if [[ "$ENV_EXISTS" == true ]]; then
        echo "Removing existing environment..."
        conda env remove -n rlwm_gpu -y || {
            echo "WARNING: Could not remove environment. Trying update instead..."
            MODE="update"
            conda activate rlwm_gpu
            conda remove cuda-nvcc cudatoolkit --force -y 2>/dev/null || true
            pip install --upgrade --force-reinstall "jax[cuda12]>=0.5.0" "jaxopt>=0.8.0"
        }
    fi

    if [[ "$MODE" == "fresh" ]]; then
        echo ""
        echo "Creating fresh rlwm_gpu environment..."
        if command -v mamba &> /dev/null; then
            echo "Using mamba for faster installation..."
            mamba env create -f environment_gpu.yml
        else
            echo "Using conda..."
            conda env create -f environment_gpu.yml
        fi
        conda activate rlwm_gpu
    fi

else
    # Interactive mode (default)
    if [[ "$ENV_EXISTS" == true ]]; then
        echo ""
        echo "Options:"
        echo "  [u] Update existing environment (recommended)"
        echo "  [f] Delete and recreate fresh"
        echo "  [q] Quit"
        read -p "Choose action [u/f/q]: " response

        case $response in
            [Uu]*)
                echo ""
                echo "Updating existing environment..."
                conda activate rlwm_gpu
                conda remove cuda-nvcc cudatoolkit --force -y 2>/dev/null || true
                PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
                if [[ "$PYTHON_VERSION" != "3.11" ]]; then
                    echo "Upgrading Python $PYTHON_VERSION -> 3.11..."
                    conda install python=3.11 -y
                fi
                pip install --upgrade --force-reinstall "jax[cuda12]>=0.5.0" "jaxopt>=0.8.0"
                ;;
            [Ff]*)
                echo "Removing existing environment..."
                conda env remove -n rlwm_gpu -y || {
                    echo "WARNING: Could not remove. Falling back to update..."
                    conda activate rlwm_gpu
                    conda remove cuda-nvcc cudatoolkit --force -y 2>/dev/null || true
                    pip install --upgrade --force-reinstall "jax[cuda12]>=0.5.0" "jaxopt>=0.8.0"
                }
                if conda env list | grep -q "rlwm_gpu"; then
                    : # Already handled by fallback
                else
                    echo "Creating fresh environment..."
                    if command -v mamba &> /dev/null; then
                        mamba env create -f environment_gpu.yml
                    else
                        conda env create -f environment_gpu.yml
                    fi
                    conda activate rlwm_gpu
                fi
                ;;
            *)
                echo "Exiting."
                exit 0
                ;;
        esac
    else
        # Environment doesn't exist - create fresh
        echo ""
        echo "Creating rlwm_gpu environment..."
        if command -v mamba &> /dev/null; then
            echo "Using mamba for faster installation..."
            mamba env create -f environment_gpu.yml
        else
            echo "Using conda..."
            conda env create -f environment_gpu.yml
        fi
        conda activate rlwm_gpu
    fi
fi

# =============================================================================
# Verify installation
# =============================================================================

echo ""
echo "============================================================"
echo "Verifying installation..."
echo "============================================================"

echo ""
echo "Python version:"
python --version

echo ""
echo "JAX version:"
python -c "import jax; print(f'JAX {jax.__version__}')"

echo ""
echo "Testing JAX device detection..."
python -c "
import jax
devices = jax.devices()
print(f'Available devices: {devices}')
gpu_devices = [d for d in devices if d.platform == 'gpu']
if gpu_devices:
    print(f'SUCCESS: GPU(s) detected: {gpu_devices}')
    import jax.numpy as jnp
    x = jnp.ones((1000, 1000))
    y = jnp.dot(x, x)
    y.block_until_ready()
    print('GPU computation test passed!')
else:
    print('INFO: No GPU detected (normal on login node).')
    print('Test on a GPU node: srun --partition=gpu --gres=gpu:1 --pty bash')
"

echo ""
echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo ""
echo "To use the GPU environment:"
echo "  conda activate rlwm_gpu"
echo ""
echo "To test on a GPU node:"
echo "  srun --partition=gpu --gres=gpu:1 --time=00:10:00 --pty bash"
echo "  module load miniforge3 && conda activate rlwm_gpu"
echo "  python -c \"import jax; print(jax.devices())\""
echo ""
echo "To run GPU-accelerated MLE fitting:"
echo "  sbatch cluster/12_mle_gpu.slurm"
echo ""
echo "============================================================"
