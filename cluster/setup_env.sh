#!/bin/bash
# =============================================================================
# RLWM Trauma Analysis - M3 Cluster Environment Setup
# =============================================================================
# This script sets up the conda environment on Monash M3 (MASSIVE).
#
# Usage:
#   module load miniforge3
#   ./cluster/setup_env.sh
#
# Prerequisites:
#   - Must be on M3 login node
#   - Run 'module load miniforge3' first
#
# Reference: https://docs.erc.monash.edu/Compute/HPC/M3/Software/Conda/
# =============================================================================

set -e  # Exit on error

echo "============================================================"
echo "RLWM Trauma Analysis - M3 Environment Setup"
echo "============================================================"

# =============================================================================
# Step 1: Check prerequisites
# =============================================================================

# Check if on M3
if [[ ! -d "/scratch" ]]; then
    echo "WARNING: /scratch not found. Are you on M3?"
fi

# Check if miniforge3 is loaded
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found!"
    echo ""
    echo "Please load the miniforge3 module first:"
    echo "  module load miniforge3"
    echo ""
    echo "Then run this script again."
    exit 1
fi

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

# =============================================================================
# Step 2: Configure conda to use scratch (M3 requirement)
# =============================================================================

# Detect HPC project ID from current path or prompt
if [[ -z "$PROJECT" ]]; then
    # Try to detect from common paths
    if [[ "$PROJECT_ROOT" =~ /scratch/([^/]+)/ ]]; then
        PROJECT="${BASH_REMATCH[1]}"
        echo "Detected HPC project: $PROJECT"
    elif [[ "$PROJECT_ROOT" =~ /projects/([^/]+)/ ]]; then
        PROJECT="${BASH_REMATCH[1]}"
        echo "Detected HPC project: $PROJECT"
    else
        echo "Could not auto-detect HPC project ID."
        read -p "Enter your M3 project ID (e.g., nq46): " PROJECT
    fi
fi

if [[ -z "$PROJECT" ]]; then
    echo "ERROR: PROJECT ID is required for M3 scratch storage."
    exit 1
fi

# Set up conda to use scratch storage (avoids home quota issues)
CONDA_HOME="/scratch/$PROJECT/$USER/conda"
echo ""
echo "Configuring conda to use scratch storage..."
echo "  CONDA_HOME: $CONDA_HOME"

# Create conda directories
mkdir -p "$CONDA_HOME/pkgs"
mkdir -p "$CONDA_HOME/envs"

# Configure conda (if not already configured)
if ! grep -q "$CONDA_HOME" ~/.condarc 2>/dev/null; then
    conda config --add pkgs_dirs "$CONDA_HOME/pkgs"
    conda config --add envs_dirs "$CONDA_HOME/envs"
    echo "  Updated ~/.condarc"
else
    echo "  Conda already configured for scratch"
fi

# =============================================================================
# Step 3: Create environment
# =============================================================================

echo ""

# Check if environment already exists
if conda env list | grep -q "^rlwm "; then
    echo "Environment 'rlwm' already exists."
    read -p "Remove and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        mamba env remove -n rlwm -y 2>/dev/null || conda env remove -n rlwm -y
    else
        echo "Keeping existing environment. Updating packages..."
        mamba env update -f environment.yml --prune 2>/dev/null || conda env update -f environment.yml --prune
        echo "Done!"
        exit 0
    fi
fi

echo ""
echo "Creating conda environment 'rlwm'..."
echo "This may take a few minutes..."
echo ""

# Use mamba (faster, available with miniforge3)
mamba env create -f environment.yml

echo ""
echo "============================================================"
echo "SUCCESS! Environment created in scratch."
echo "============================================================"
echo ""
echo "Location: $CONDA_HOME/envs/rlwm"
echo ""

# Verify JAX version (critical for memory usage)
echo "Verifying JAX installation..."
JAX_VERSION=$(conda run -n rlwm python -c "import jax; print(jax.__version__)" 2>/dev/null)
echo "  JAX version: $JAX_VERSION"

if [[ "$JAX_VERSION" == "0.9.0" ]]; then
    echo "  ✓ Correct version installed (0.9.0 from PyPI)"
else
    echo "  ⚠ WARNING: Expected JAX 0.9.0, got $JAX_VERSION"
    echo "    This may cause OOM errors during fitting."
    echo "    Try: conda activate rlwm && pip install --upgrade jax==0.9.0 jaxlib==0.9.0"
fi

echo ""
echo "To activate the environment:"
echo "  conda activate rlwm"
echo ""
echo "To run MLE fitting:"
echo "  sbatch cluster/run_mle.slurm"
echo ""
echo "NOTE: Do NOT run 'conda init' - it breaks Strudel."
echo "============================================================"
