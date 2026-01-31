#!/bin/bash
# =============================================================================
# SLURM Batch Script: Parallel MLE Fitting for RLWM Models (Monash M3)
# =============================================================================
# Runs MLE parameter estimation using multiple CPU cores for faster execution.
# Uses joblib for parallel participant fitting.
#
# Usage:
#   sbatch cluster/run_mle_parallel.slurm                    # All models, 8 workers
#   sbatch --export=MODEL=wmrl_m3 cluster/run_mle_parallel.slurm  # Single model
#   sbatch --export=NJOBS=4 cluster/run_mle_parallel.slurm   # Custom worker count
#
# Performance (approximate for 47 participants):
#   - Sequential (1 core):  ~30 minutes per model
#   - 4 workers:            ~8 minutes per model
#   - 8 workers:            ~4 minutes per model (default, memory-safe)
#
# Outputs saved to:
#   - output/mle/{model}_individual_fits.csv
#   - output/mle/{model}_group_summary.csv
#   - cluster/logs/mle_parallel_{jobid}.out
#
# Reference: https://docs.erc.monash.edu/old-M3/M3/slurm/partitions/
# =============================================================================

#SBATCH --job-name=rlwm_mle_parallel
#SBATCH --output=cluster/logs/mle_parallel_%j.out
#SBATCH --error=cluster/logs/mle_parallel_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --partition=comp

# =============================================================================
# Configuration (set via --export or use defaults)
# =============================================================================
# Note: 8 workers is more memory-safe with JAX+joblib. Each JAX worker uses
# ~5-8GB for JIT compilation. With 8 workers, peak usage is ~60-80GB.
MODELS="${MODEL:-qlearning wmrl wmrl_m3}"
NJOBS="${NJOBS:-8}"

# =============================================================================
# Environment Setup (M3 specific)
# =============================================================================

# Load miniforge3 (M3's conda distribution)
module load miniforge3

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "RLWM Parallel MLE Fitting - M3 SLURM Job"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Partition: ${SLURM_JOB_PARTITION:-comp}"
echo "CPUs requested: ${SLURM_CPUS_PER_TASK:-16}"
echo "Parallel workers: $NJOBS"
echo "Project: $PROJECT_ROOT"
echo "Start time: $(date)"
echo "============================================================"

# Activate conda environment
# Note: Do NOT use 'conda init' on M3 - it breaks Strudel
conda activate rlwm

# Create logs directory if needed
mkdir -p cluster/logs

# =============================================================================
# JAX Compilation Cache (prevents redundant JIT compilation across workers)
# =============================================================================
export JAX_COMPILATION_CACHE_DIR="/scratch/fc37/aman0087/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
echo "JAX cache: $JAX_COMPILATION_CACHE_DIR"

# =============================================================================
# Run MLE Fitting (Parallel)
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
    echo "Fitting model: $model (parallel with $NJOBS workers)"
    echo "Started: $(date)"
    echo "============================================================"

    python scripts/fitting/fit_mle.py \
        --model "$model" \
        --data "$DATA_FILE" \
        --output output/mle/ \
        --n-starts 20 \
        --seed 42 \
        --n-jobs "$NJOBS"

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
