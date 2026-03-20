#!/bin/bash
# =============================================================================
# SLURM Batch Script: Single Model MLE Fitting with Timing (Monash M3)
# =============================================================================
# Runs MLE fitting for a single model. Use --limit for quick timing tests.
#
# Usage:
#   # Quick timing test (3 participants)
#   sbatch --export=MODEL=wmrl_m3,LIMIT=3 cluster/12_mle_single.slurm
#
#   # Full run for single model
#   sbatch --export=MODEL=wmrl_m3 cluster/12_mle_single.slurm
#
#   # Different models: qlearning, wmrl, wmrl_m3
#   sbatch --export=MODEL=qlearning cluster/12_mle_single.slurm
#
# Reference: https://docs.erc.monash.edu/old-M3/M3/slurm/partitions/
# =============================================================================

#SBATCH --job-name=rlwm_mle_single
#SBATCH --output=cluster/logs/mle_single_%j.out
#SBATCH --error=cluster/logs/mle_single_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
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
conda activate /scratch/${PROJECT}/${USER}/conda/envs/rlwm || {
    echo "ERROR: Failed to activate 'rlwm' conda environment"
    echo "Create it first with: conda env create -f environment.yml"
    exit 1
}

# Store the Python path for srun (srun creates fresh environment without conda activation)
# Using 'which python' to get the path from the activated conda environment
PYTHON_BIN=$(which python)
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "ERROR: Python not found in activated conda environment"
    exit 1
fi
echo "Python binary: $PYTHON_BIN"

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

# =============================================================================
# JAX COMPILATION CACHE (CRITICAL FOR TWO-PHASE APPROACH)
# =============================================================================
# The warmup phase compiles kernels with limited CPUs (2) to prevent LLVM
# memory exhaustion. The main fitting phase reuses these cached kernels.
# Without persistent caching, each phase would recompile from scratch.
#
# IMPORTANT: Cache is NODE-SPECIFIC because XLA generates machine-specific code.
# Different cluster nodes have different CPU features, so sharing a cache
# between nodes causes "CPU feature mismatch" errors.
export JAX_COMPILATION_CACHE_DIR="/scratch/${PROJECT}/${USER}/.jax_cache/${SLURMD_NODENAME:-$(hostname)}"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# =============================================================================
# XLA COMPILATION CONTROLS (CRITICAL FOR MEMORY STABILITY)
# =============================================================================
#
# PROBLEM: LLVM compilation uses parallel threads (default: 32) which can exhaust
# memory during JIT compilation even with 60GB+ free RAM.
#
# SOLUTION: Two flags work together to prevent memory issues:
#
#   --xla_cpu_parallel_codegen_split_count=1
#     Controls LLVM compilation parallelism. Default is 32 threads, which compete
#     for allocations and cause "Cannot allocate memory" errors.
#     Setting to 1 makes compilation single-threaded but reliable.
#
#   --xla_cpu_multi_thread_eigen=false
#     Controls RUNTIME Eigen operations (already set). This affects matrix ops
#     during actual execution, not compilation.
#
# TRADE-OFF: Initial compilation is slower (~120s vs ~60s) but prevents crashes.
# Runtime performance is UNAFFECTED - compiled kernel is cached and reused.
#
# References:
# - https://guides.lw1.at/all-xla-options/
# - https://docs.jax.dev/en/latest/xla_flags.html
# - https://github.com/jax-ml/jax/issues/6790
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false --xla_cpu_parallel_codegen_split_count=1"
export TF_NUM_INTEROP_THREADS=1
export TF_NUM_INTRAOP_THREADS=4

# =============================================================================
# DEBUGGING: JAX Compilation & Memory Traces (DISABLED FOR PRODUCTION)
# =============================================================================
# These flags help isolate where memory issues occur during JIT compilation.
# WARNING: Creates 9.5MB+ log files (74,000+ lines) - use only for one-time debugging!
#
# JAX_LOG_COMPILES: Logs each JIT compilation event
# JAX_EXPLAIN_CACHE_MISSES: Explains why recompilation occurs (shape changes, etc.)
# JAX_LOGGING_LEVEL: Verbose JAX internal logging
#
# References:
# - https://docs.jax.dev/en/latest/config_options.html
# - https://openxla.org/xla/tools
#
# Uncomment for debugging only:
# export JAX_LOG_COMPILES=1
# export JAX_EXPLAIN_CACHE_MISSES=1
# export JAX_LOGGING_LEVEL=INFO

# Uncomment for deep debugging (creates large files):
# export XLA_FLAGS="${XLA_FLAGS} --xla_dump_to=${PROJECT_ROOT}/output/mle/xla_dump_${SLURM_JOB_ID}"
# export XLA_FLAGS="${XLA_FLAGS} --xla_dump_hlo_as_text"

echo "Thread settings: OMP/MKL/OPENBLAS/NUMEXPR = 4 threads each"
echo "XLA settings: single-threaded LLVM compilation + single-threaded Eigen"
echo "JAX cache: $JAX_COMPILATION_CACHE_DIR"
echo "Debug flags: DISABLED (uncomment in script for debugging)"

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
# MEMORY MONITORING FUNCTION (for OOM debugging)
# =============================================================================
# Logs current memory state with timestamp for isolating memory exhaustion.
log_memory() {
    local label="$1"
    local timestamp=$(date '+%H:%M:%S')
    local mem_info=$(free -h 2>/dev/null | grep Mem || echo "N/A")
    echo "[$timestamp] MEMORY ($label): $mem_info"
}

# =============================================================================
# Run MLE Fitting
# =============================================================================
DATA_FILE="output/task_trials_long.csv"

if [[ ! -f "$DATA_FILE" ]]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# Build command
CMD="python scripts/fitting/fit_mle.py --model $MODEL --data $DATA_FILE --output output/mle/ --n-starts 20 --seed 42 --compute-diagnostics"

if [[ -n "$LIMIT" ]]; then
    CMD="$CMD --limit $LIMIT"
fi

echo "Running: $CMD"
echo ""

# Pre-fit memory state
echo "=== PRE-FIT MEMORY STATE ==="
log_memory "PRE-FIT"
free -m 2>/dev/null || echo "(free command not available)"
echo ""

# =============================================================================
# TWO-PHASE EXECUTION (CRITICAL FOR LLVM MEMORY STABILITY)
# =============================================================================
#
# PROBLEM: LLVM's execution engine spawns threads equal to the number of CPUs
# detected via std::thread::hardware_concurrency(). With --cpus-per-task=16,
# it spawns 16 threads that all try to allocate memory simultaneously,
# causing "Cannot allocate memory" errors (seen in job 50989988).
#
# WHY TASKSET DIDN'T WORK: SLURM's cgroups v2 blocks CPU affinity changes
# from within the job (job 50991451: "taskset: Invalid argument").
#
# WHY XLA FLAGS WEREN'T ENOUGH:
# - xla_cpu_parallel_codegen_split_count controls MODULE SPLITTING, not LLVM threads
# - LLVM's internal JIT thread pool is separate and has no CPU-side XLA flag
#
# SOLUTION: Two-phase srun approach
# - Phase 1: Run warmup with srun --cpus-per-task=2 (LLVM sees 2 CPUs, spawns 2 threads)
# - Phase 2: Run main fitting with full CPUs (kernels cached, no LLVM compilation)
#
# References:
# - https://github.com/jax-ml/jax/issues/16272 (LLVM worker thread limit)
# - https://github.com/jax-ml/jax/issues/6790 (srun workaround)
# =============================================================================

# -----------------------------------------------------------------------------
# PHASE 1: JIT WARMUP WITH LIMITED CPUs
# -----------------------------------------------------------------------------
# Run warmup with only 2 CPUs so LLVM spawns only 2 compilation threads.
# This prevents memory exhaustion during JIT compilation.
# Compiled kernels are saved to JAX_COMPILATION_CACHE_DIR.

echo "Phase 1: JIT warmup with limited CPUs (prevents LLVM memory exhaustion)..."
WARMUP_START=$(date +%s)

srun --ntasks=1 --cpus-per-task=2 --exact --export=ALL $PYTHON_BIN scripts/fitting/warmup_jit.py --model $MODEL

WARMUP_EXIT=$?
WARMUP_END=$(date +%s)
echo "Warmup completed in $((WARMUP_END - WARMUP_START)) seconds (exit code: $WARMUP_EXIT)"

if [[ $WARMUP_EXIT -ne 0 ]]; then
    echo "ERROR: Warmup failed with exit code $WARMUP_EXIT"
    exit $WARMUP_EXIT
fi

echo ""

# -----------------------------------------------------------------------------
# PHASE 2: MAIN FITTING WITH FULL CPUs
# -----------------------------------------------------------------------------
# Now run the actual fitting. JAX will load compiled kernels from cache,
# so no LLVM compilation occurs. All 16 CPUs available for optimization.
#
# NOTE: We use srun here to ensure environment variables (JAX_COMPILATION_CACHE_DIR,
# XLA_FLAGS, etc.) are properly propagated. This also ensures that if there's a
# cache miss, LLVM will only see 4 CPUs, preventing memory exhaustion.

echo "Phase 2: Main fitting with full CPU allocation..."

# Build args for srun (need to reconstruct since $CMD includes 'python')
FITTING_ARGS="--model $MODEL --data $DATA_FILE --output output/mle/ --n-starts 20 --seed 42 --compute-diagnostics"
if [[ -n "$LIMIT" ]]; then
    FITTING_ARGS="$FITTING_ARGS --limit $LIMIT"
fi

srun --ntasks=1 --cpus-per-task=4 --exact --export=ALL $PYTHON_BIN scripts/fitting/fit_mle.py $FITTING_ARGS

EXIT_CODE=$?

# Post-fit memory state
echo ""
echo "=== POST-FIT MEMORY STATE ==="
log_memory "POST-FIT"
free -m 2>/dev/null || echo "(free command not available)"

# =============================================================================
# Job Resource Metrics (automatically captured)
# =============================================================================
echo ""
echo "============================================================"
echo "JOB RESOURCE METRICS"
echo "============================================================"
echo "Command run: sbatch cluster/12_mle_single.slurm"
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
