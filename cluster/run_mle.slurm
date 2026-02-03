#!/bin/bash
# =============================================================================
# SLURM Batch Script: MLE Fitting for RLWM Models (Monash M3)
# =============================================================================
# Runs MLE parameter estimation with parallel fitting (4 workers default).
# JAX compilation is cached to reduce memory overhead on subsequent runs.
#
# Usage:
#   sbatch cluster/run_mle.slurm                         # All models, 4 workers
#   sbatch --export=MODEL=wmrl_m3 cluster/run_mle.slurm  # Single model
#   sbatch --export=NJOBS=8 cluster/run_mle.slurm        # More workers (if memory allows)
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
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

# =============================================================================
# Configuration (set via --export or use defaults)
# =============================================================================
# Note: NJOBS=4 is the recommended default with JAX compilation caching.
# Use NJOBS=1 if you experience OOM errors on first run (cache cold).
MODELS="${MODEL:-qlearning wmrl wmrl_m3}"
NJOBS="${NJOBS:-4}"

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
conda activate rlwm || {
    echo "ERROR: Failed to activate 'rlwm' conda environment"
    echo "Create it first with: conda env create -f environment.yml"
    exit 1
}

# Create logs directory if needed
mkdir -p cluster/logs

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
