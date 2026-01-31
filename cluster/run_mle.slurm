#!/bin/bash
# =============================================================================
# SLURM Batch Script: MLE Fitting for RLWM Models (Monash M3)
# =============================================================================
# Runs MLE parameter estimation for all three models (M1, M2, M3)
#
# Usage:
#   sbatch cluster/run_mle.slurm                    # Run all models
#   sbatch --export=MODEL=qlearning cluster/run_mle.slurm  # Single model
#
# Outputs saved to:
#   - output/mle/{model}_individual_fits.csv
#   - output/mle/{model}_group_summary.csv
#   - cluster/logs/mle_{jobid}.out
#
# Reference: https://docs.erc.monash.edu/old-M3/M3/slurm/partitions/
# =============================================================================

#SBATCH --job-name=rlwm_mle
#SBATCH --output=cluster/logs/mle_%j.out
#SBATCH --error=cluster/logs/mle_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

# =============================================================================
# Environment Setup (M3 specific)
# =============================================================================

# Load miniforge3 (M3's conda distribution)
module load miniforge3

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "RLWM MLE Fitting - M3 SLURM Job"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Partition: ${SLURM_JOB_PARTITION:-comp}"
echo "Project: $PROJECT_ROOT"
echo "Start time: $(date)"
echo "============================================================"

# Activate conda environment
# Note: Do NOT use 'conda init' on M3 - it breaks Strudel
conda activate rlwm

# Create logs directory if needed
mkdir -p cluster/logs

# =============================================================================
# Run MLE Fitting
# =============================================================================

# Default: run all models. Override with --export=MODEL=qlearning
MODELS="${MODEL:-qlearning wmrl wmrl_m3}"

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
    echo "Fitting model: $model"
    echo "Started: $(date)"
    echo "============================================================"

    python scripts/fitting/fit_mle.py \
        --model "$model" \
        --data "$DATA_FILE" \
        --output output/mle/ \
        --n-starts 20 \
        --seed 42

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
