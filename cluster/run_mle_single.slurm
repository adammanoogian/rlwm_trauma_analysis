#!/bin/bash
# =============================================================================
# SLURM Batch Script: Single Model MLE Fitting with Timing (Monash M3)
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
#
# Reference: https://docs.erc.monash.edu/old-M3/M3/slurm/partitions/
# =============================================================================

#SBATCH --job-name=rlwm_mle_single
#SBATCH --output=cluster/logs/mle_single_%j.out
#SBATCH --error=cluster/logs/mle_single_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=16
#SBATCH --partition=comp

# =============================================================================
# Configuration (set via --export or use defaults)
# =============================================================================
MODEL="${MODEL:-wmrl_m3}"
LIMIT="${LIMIT:-}"

# =============================================================================
# Environment Setup (M3 specific)
# =============================================================================

# Load miniforge3 (M3's conda distribution)
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

# Capture start time for metrics
START_TIME=$(date)

echo "============================================================"
echo "RLWM MLE Fitting - Single Model (M3)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Partition: ${SLURM_JOB_PARTITION:-comp}"
echo "Model: $MODEL"
echo "Limit: ${LIMIT:-full dataset}"
echo "Start time: $(date)"
echo "============================================================"

# Activate conda environment
conda activate /scratch/fc37/aman0087/conda/envs/rlwm || {
    echo "ERROR: Failed to activate 'rlwm' conda environment"
    echo "Create it first with: conda env create -f environment.yml"
    exit 1
}

mkdir -p cluster/logs

# =============================================================================
# Thread Control (match threads to allocated CPUs)
# =============================================================================
# Allow each thread pool to use 4 threads for BLAS/JAX operations.
# With 4 threads × 4 implicit workers = 16 total threads, matching --cpus-per-task=16
#
# This configuration:
#   - Provides multi-threading benefits for JAX compilation warmup
#   - Allows BLAS parallelism for matrix operations
#   - Prevents oversubscription (threads ≤ allocated CPUs)
#
# Reference: https://docs.erc.monash.edu/Compute/HPC/M3/RunningJobsOnM3/EffectiveUsage
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export JAX_PLATFORMS=cpu
echo "Thread settings: OMP/MKL/OPENBLAS/NUMEXPR = 4 threads each"

# =============================================================================
# Environment Validation (fail fast if packages missing)
# =============================================================================
echo ""
echo "Validating environment..."

python -c "
import sys
missing = []
required = ['jax', 'jaxlib', 'scipy', 'pandas', 'numpy', 'joblib']
for pkg in required:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

if missing:
    print(f'ERROR: Missing required packages: {missing}')
    print('Install with: pip install ' + ' '.join(missing))
    sys.exit(1)

# Check project fitting module is importable
sys.path.insert(0, '.')
try:
    from scripts.fitting import fit_mle, jax_likelihoods, mle_utils
except ImportError as e:
    print(f'ERROR: Cannot import fitting module: {e}')
    print('Make sure you are in the project root directory')
    sys.exit(1)

print('Environment OK: all packages available')
" || exit 1

echo ""

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

EXIT_CODE=$?

# =============================================================================
# Job Resource Metrics (automatically captured)
# =============================================================================
echo ""
echo "============================================================"
echo "JOB RESOURCE METRICS"
echo "============================================================"
echo "Command run: sbatch cluster/run_mle_single.slurm"
echo "Model: $MODEL"
echo "Limit: ${LIMIT:-full dataset}"
echo "Thread settings: OMP=$OMP_NUM_THREADS, MKL=$MKL_NUM_THREADS"
echo ""

# Create output directory for metrics if needed
mkdir -p output/mle

# Create metrics file
METRICS_FILE="output/mle/job_metrics_single_${SLURM_JOB_ID}.txt"

{
    echo "========================================"
    echo "SLURM JOB METRICS REPORT (Single Model)"
    echo "========================================"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Job Name: $SLURM_JOB_NAME"
    echo "Node: $SLURMD_NODENAME"
    echo "Partition: $SLURM_JOB_PARTITION"
    echo ""
    echo "REQUESTED RESOURCES:"
    echo "  CPUs: $SLURM_CPUS_PER_TASK"
    echo "  Memory: $SLURM_MEM_PER_NODE"
    echo "  Time Limit: $SLURM_TIMELIMIT"
    echo ""
    echo "CONFIGURATION:"
    echo "  Model: $MODEL"
    echo "  Limit: ${LIMIT:-full dataset}"
    echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
    echo "  MKL_NUM_THREADS: $MKL_NUM_THREADS"
    echo ""
    echo "TIMING:"
    echo "  Start: $START_TIME"
    echo "  End: $(date)"
    echo ""
    echo "ACTUAL RESOURCE USAGE (sacct):"
    sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,MaxVMSize,ReqMem,AllocCPUS,TotalCPU,State%10 2>/dev/null || echo "  (sacct not yet available)"
    echo ""
    echo "========================================"
} | tee "$METRICS_FILE"

echo ""
echo "Metrics saved to: $METRICS_FILE"
echo "============================================================"
echo "Complete!"
echo "End time: $(date)"
echo "============================================================"

exit $EXIT_CODE
