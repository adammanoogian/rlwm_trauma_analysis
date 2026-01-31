#!/bin/bash
# =============================================================================
# SLURM Batch Script: Single Model MLE Fitting with Timing Test
# =============================================================================
# Runs MLE fitting for a single model. Use --limit for quick timing tests.
#
# Usage:
#   # Quick timing test (3 participants)
#   sbatch --export=MODEL=wmrl_m3,LIMIT=3 cluster/run_mle_single.slurm
#
#   # Full run for single model
#   sbatch --export=MODEL=wmrl_m3 cluster/run_mle_single.slurm
#
#   # Different models: qlearning, wmrl, wmrl_m3
#   sbatch --export=MODEL=qlearning cluster/run_mle_single.slurm
# =============================================================================

#SBATCH --job-name=rlwm_mle_single
#SBATCH --output=cluster/logs/mle_single_%x_%j.out
#SBATCH --error=cluster/logs/mle_single_%x_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=standard

# =============================================================================
# Configuration (set via --export or use defaults)
# =============================================================================
MODEL="${MODEL:-wmrl_m3}"
LIMIT="${LIMIT:-}"

# =============================================================================
# Environment Setup
# =============================================================================
module load anaconda 2>/dev/null || module load miniconda 2>/dev/null || true

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "RLWM MLE Fitting - Single Model"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Model: $MODEL"
echo "Limit: ${LIMIT:-full dataset}"
echo "Start time: $(date)"
echo "============================================================"

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rlwm

mkdir -p cluster/logs

# =============================================================================
# Run MLE Fitting
# =============================================================================
DATA_FILE="output/task_trials_long.csv"

if [[ ! -f "$DATA_FILE" ]]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Build command
CMD="python scripts/fitting/fit_mle.py --model $MODEL --data $DATA_FILE --output output/mle/ --n-starts 20 --seed 42"

if [[ -n "$LIMIT" ]]; then
    CMD="$CMD --limit $LIMIT"
fi

echo "Running: $CMD"
echo ""

$CMD

echo ""
echo "============================================================"
echo "Complete!"
echo "End time: $(date)"
echo "============================================================"
