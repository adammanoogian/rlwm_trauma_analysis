#!/bin/bash
# =============================================================================
# SLURM Batch Script: MLE Fitting for RLWM Models (Monash M3)
# =============================================================================
# Runs MLE parameter estimation with parallel fitting (4 workers default).
# JAX compilation is cached to reduce memory overhead on subsequent runs.
#
# Usage:
#   sbatch cluster/12_mle.slurm                         # All models, 4 workers
#   sbatch --export=MODEL=wmrl_m3 cluster/12_mle.slurm  # Single model
#   sbatch --export=NJOBS=8 cluster/12_mle.slurm        # More workers (if memory allows)
#
# Performance (47 participants):
#   - Parallel (4 workers):  ~8-10 minutes per model with Hessian diagnostics
#   - Sequential (NJOBS=1):  ~30 minutes per model (use if OOM issues)
#
# Outputs saved to:
#   - output/mle/{model}_individual_fits.csv
#   - output/mle/{model}_group_summary.csv
#   - output/mle/timing_log.csv (per-participant timing)
#   - output/mle/performance_summary.json (overall performance metrics)
#   - cluster/logs/mle_{jobid}.out
#
# Reference: https://docs.erc.monash.edu/old-M3/M3/slurm/partitions/
# =============================================================================

#SBATCH --job-name=rlwm_mle
#SBATCH --output=cluster/logs/mle_%j.out
#SBATCH --error=cluster/logs/mle_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --partition=comp

# =============================================================================
# Configuration (set via --export or use defaults)
# =============================================================================
# Note: NJOBS=4 is the recommended default with JAX compilation caching.
# Use NJOBS=1 if you experience OOM errors on first run (cache cold).
MODELS="${MODEL:-qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b wmrl_m4}"
NJOBS="${NJOBS:-4}"

# =============================================================================
# Environment Setup (M3 specific)
# =============================================================================

# Load miniforge3 (M3's conda distribution)
module load miniforge3

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

# Capture start time for metrics
START_TIME=$(date)

echo "============================================================"
echo "RLWM MLE Fitting - M3 SLURM Job"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Partition: ${SLURM_JOB_PARTITION:-comp}"
echo "CPUs requested: ${SLURM_CPUS_PER_TASK:-16}"
echo "Memory requested: ${SLURM_MEM_PER_NODE:-64G}"
echo "Parallel workers: $NJOBS"
echo "Project: $PROJECT_ROOT"
echo "Start time: $(date)"
echo "============================================================"

# Activate conda environment
# Note: Do NOT use 'conda init' on M3 - it breaks Strudel
conda activate /scratch/${PROJECT}/${USER}/conda/envs/rlwm || {
    echo "ERROR: Failed to activate 'rlwm' conda environment"
    echo "Create it first with: conda env create -f environment.yml"
    exit 1
}

# Create logs directory if needed
mkdir -p cluster/logs

# =============================================================================
# Thread Control (match NJOBS × threads to allocated CPUs)
# =============================================================================
# Allow each joblib worker to use 4 threads for BLAS/JAX operations.
# With NJOBS=4 workers × 4 threads = 16 total threads, matching --cpus-per-task=16
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

# =============================================================================
# XLA COMPILATION CONTROLS (for parallel worker stability)
# =============================================================================
# Each joblib worker is a separate process, so LLVM memory issues can still occur.
# These flags help prevent memory exhaustion during JIT compilation.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_cpu_parallel_codegen_split_count=1"
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=4

echo "Thread settings: OMP/MKL/OPENBLAS/NUMEXPR = 4 threads each"
echo "Total threads: $NJOBS workers × 4 threads = $(($NJOBS * 4))"
echo "XLA settings: single-threaded LLVM compilation per worker"

# =============================================================================
# Environment Validation (fail fast if packages missing)
# =============================================================================
echo ""
echo "Validating environment..."

# Check Python is available
if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found in environment"
    exit 1
fi

# Check required packages are importable
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
# JAX Compilation Cache (NODE-SPECIFIC to prevent CPU feature mismatch)
# =============================================================================
# IMPORTANT: Cache is node-specific because XLA generates machine-specific code.
# Different M3 nodes have different CPU features (AVX-512, SHA, etc.), so a kernel
# compiled on one node may crash with SIGILL on another. Using ${SLURMD_NODENAME}
# ensures each node maintains its own compatible cache.
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache/${SLURMD_NODENAME:-$(hostname)}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"

# =============================================================================
# Run MLE Fitting
# =============================================================================
# NOTE: If you encounter "Cannot allocate memory" errors from LLVM, use
# 12_mle_single.slurm instead, which uses taskset to limit CPU affinity
# and prevent LLVM from spawning too many compilation threads.
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
        --n-jobs "$NJOBS" \
        --compute-diagnostics

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
# Job Resource Metrics (automatically captured)
# =============================================================================
echo ""
echo "============================================================"
echo "JOB RESOURCE METRICS"
echo "============================================================"
echo "Command run: sbatch cluster/12_mle.slurm"
echo "Models fitted: $MODELS"
echo "Parallel workers: $NJOBS"
echo "Thread settings: OMP=$OMP_NUM_THREADS, MKL=$MKL_NUM_THREADS"
echo ""

# Create output directory for metrics if needed
mkdir -p output/mle

# Create metrics file
METRICS_FILE="output/mle/job_metrics_${SLURM_JOB_ID}.txt"

{
    echo "========================================"
    echo "SLURM JOB METRICS REPORT"
    echo "========================================"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Job Name: $SLURM_JOB_NAME"
    echo "Node: $SLURMD_NODENAME"
    echo "Partition: $SLURM_JOB_PARTITION"
    echo "Submit Dir: $SLURM_SUBMIT_DIR"
    echo ""
    echo "REQUESTED RESOURCES:"
    echo "  CPUs: $SLURM_CPUS_PER_TASK"
    echo "  Memory: $SLURM_MEM_PER_NODE"
    echo "  Time Limit: $SLURM_TIMELIMIT"
    echo ""
    echo "CONFIGURATION:"
    echo "  Models: $MODELS"
    echo "  NJOBS (parallel workers): $NJOBS"
    echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
    echo "  MKL_NUM_THREADS: $MKL_NUM_THREADS"
    echo "  OPENBLAS_NUM_THREADS: $OPENBLAS_NUM_THREADS"
    echo ""
    echo "TIMING:"
    echo "  Start: $START_TIME"
    echo "  End: $(date)"
    echo ""
    echo "ACTUAL RESOURCE USAGE (sacct):"
    sacct -j $SLURM_JOB_ID --format=JobID,Elapsed,MaxRSS,MaxVMSize,ReqMem,AllocCPUS,TotalCPU,CPUTime,State%10 2>/dev/null || echo "  (sacct not yet available - job may still be running)"
    echo ""
    echo "EFFICIENCY ANALYSIS:"
    echo "  Formula: Efficiency = TotalCPU / (Elapsed × AllocCPUS)"
    echo "  Good efficiency: ~75-100%"
    echo "  If >> 100%: oversubscription (too many threads per CPU)"
    echo "  If << 75%: underutilization (increase NJOBS or reduce CPUs)"
    echo ""
    echo "M3 MONITORING COMMANDS (for future reference):"
    echo "  squeue -u \$USER                  # Check job status"
    echo "  sacct -j <JOBID> --format=JobID,Elapsed,MaxRSS,TotalCPU,State"
    echo "  ssh <node> && htop -u \$USER      # Real-time CPU monitoring"
    echo ""
    echo "========================================"
} | tee "$METRICS_FILE"

echo ""
echo "Metrics saved to: $METRICS_FILE"
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
