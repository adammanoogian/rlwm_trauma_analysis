#!/bin/bash
# =============================================================================
# SLURM Batch Script: MLE Fitting for RLWM Models (Monash M3)
# =============================================================================
# Runs MLE parameter estimation using sequential fitting (memory-safe).
# JAX + joblib parallel fitting is memory-prohibitive due to LLVM compilation
# overhead in each worker process (~15GB per worker).
#
# Usage:
#   sbatch cluster/run_mle.slurm                         # All models, sequential
#   sbatch --export=MODEL=wmrl_m3 cluster/run_mle.slurm  # Single model
#   sbatch --export=NJOBS=4 cluster/run_mle.slurm        # Custom workers (caution)
#
# Performance (47 participants):
#   - Sequential (default):  ~30 minutes per model, ~20-30GB memory
#   - Parallel (8 workers):  NOT RECOMMENDED - causes LLVM OOM crashes
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
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

# =============================================================================
# Configuration (set via --export or use defaults)
# =============================================================================
# Note: Sequential (NJOBS=1) is the safe default. Parallel fitting with JAX
# causes LLVM compilation OOM errors even with 120GB memory.
MODELS="${MODEL:-qlearning wmrl wmrl_m3}"
NJOBS="${NJOBS:-1}"

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
echo "CPUs requested: ${SLURM_CPUS_PER_TASK:-4}"
echo "Memory requested: ${SLURM_MEM_PER_NODE:-32G}"
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
# JAX Compilation Cache (prevents redundant JIT compilation across runs)
# =============================================================================
export JAX_COMPILATION_CACHE_DIR="/scratch/fc37/aman0087/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
echo "JAX cache: $JAX_COMPILATION_CACHE_DIR"

# =============================================================================
# Run MLE Fitting
# =============================================================================

DATA_FILE="output/task_trials_long.csv"

# Check data file exists
if [[ ! -f "$DATA_FILE" ]]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    echo "Run the data parsing script first:"
    echo "  python scripts/01_parse_raw_data.py"
    exit 1
fi

# Track overall success
ALL_SUCCESS=true

for model in $MODELS; do
    echo ""
    echo "============================================================"
    echo "Fitting model: $model (workers: $NJOBS)"
    echo "Started: $(date)"
    echo "============================================================"

    python scripts/fitting/fit_mle.py \
        --model "$model" \
        --data "$DATA_FILE" \
        --output output/mle/ \
        --n-starts 20 \
        --seed 42 \
        --n-jobs "$NJOBS"

    EXIT_CODE=$?
    if [[ $EXIT_CODE -ne 0 ]]; then
        echo "WARNING: Model $model exited with code $EXIT_CODE"
        ALL_SUCCESS=false
    fi

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

# =============================================================================
# Job Resource Utilization Statistics
# =============================================================================
echo ""
echo "============================================================"
echo "JOB RESOURCE UTILIZATION"
echo "============================================================"
# sacct provides detailed resource usage after job completion
# Note: MaxRSS shows peak memory in KB, MaxVMSize shows peak virtual memory
sacct -j $SLURM_JOB_ID --format=JobID,JobName%15,Elapsed,MaxRSS,MaxVMSize,ReqMem,AllocCPUS,TotalCPU,State%10 2>/dev/null || echo "sacct not available (job may still be running)"
echo "============================================================"

# Final status
if [[ "$ALL_SUCCESS" == "true" ]]; then
    echo ""
    echo "FINAL STATUS: ALL MODELS COMPLETED SUCCESSFULLY"
    exit 0
else
    echo ""
    echo "FINAL STATUS: ONE OR MORE MODELS FAILED"
    exit 1
fi
