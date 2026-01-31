#!/bin/bash
# =============================================================================
# RLWM Trauma Analysis - Cluster Environment Setup
# =============================================================================
# This script sets up the conda environment on a university cluster.
#
# Usage:
#   ./cluster/setup_env.sh
#
# Prerequisites:
#   - conda or mamba must be available (usually via 'module load anaconda' or similar)
# =============================================================================

set -e  # Exit on error

echo "============================================================"
echo "RLWM Trauma Analysis - Environment Setup"
echo "============================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found!"
    echo "Try loading the anaconda module first:"
    echo "  module load anaconda"
    echo "  # or: module load miniconda"
    exit 1
fi

# Navigate to project root (parent of cluster/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

# Check if environment already exists
if conda env list | grep -q "^rlwm "; then
    echo "Environment 'rlwm' already exists."
    read -p "Remove and recreate? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n rlwm -y
    else
        echo "Keeping existing environment. Updating packages..."
        conda env update -f environment.yml --prune
        echo "Done!"
        exit 0
    fi
fi

# Create environment
echo "Creating conda environment 'rlwm'..."
echo "This may take a few minutes..."
echo ""

# Prefer mamba if available (much faster)
if command -v mamba &> /dev/null; then
    echo "Using mamba (faster)..."
    mamba env create -f environment.yml
else
    echo "Using conda..."
    conda env create -f environment.yml
fi

echo ""
echo "============================================================"
echo "SUCCESS! Environment created."
echo "============================================================"
echo ""
echo "To activate the environment:"
echo "  conda activate rlwm"
echo ""
echo "To run MLE fitting:"
echo "  sbatch cluster/run_mle.slurm"
echo ""
echo "============================================================"
