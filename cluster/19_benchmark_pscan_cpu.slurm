#!/bin/bash
# =============================================================================
# SLURM Batch Script: PScan CPU Benchmark (Monash M3)
# =============================================================================
# Runs the micro-benchmark on a CPU compute node to establish the sequential
# vs pscan baseline timing WITHOUT GPU. This is the control condition for
# comparing against the GPU benchmark (19_benchmark_pscan_gpu.slurm).
#
# Usage:
#   sbatch cluster/19_benchmark_pscan_cpu.slurm                    # All 6 models
#   sbatch --export=MODEL=wmrl_m3 cluster/19_benchmark_pscan_cpu.slurm  # Single model
#   sbatch --export=REPEATS=50 cluster/19_benchmark_pscan_cpu.slurm     # More repeats
#
# Outputs:
#   - output/bayesian/pscan_benchmark.json (overwritten per run)
#   - cluster/logs/pscan_bench_cpu_{jobid}.out
# =============================================================================

#SBATCH --job-name=pscan-bench-cpu
#SBATCH --output=cluster/logs/pscan_bench_cpu_%j.out
#SBATCH --error=cluster/logs/pscan_bench_cpu_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

# =============================================================================
# Configuration
# =============================================================================
MODEL="${MODEL:-all}"
REPEATS="${REPEATS:-20}"

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "RLWM PScan CPU Benchmark"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Partition: ${SLURM_JOB_PARTITION:-comp}"
echo "CPUs: ${SLURM_CPUS_PER_TASK:-4}"
echo "Project: $PROJECT_ROOT"
echo "Model: $MODEL"
echo "Repeats: $REPEATS"
echo "Start time: $(date)"
echo "============================================================"

# Activate conda environment (CPU — ds_env preferred, rlwm_gpu fallback)
if conda activate ds_env 2>/dev/null; then
    echo "Activated ds_env"
elif conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu (fallback)"
elif conda activate /scratch/fc37/$USER/conda/envs/ds_env 2>/dev/null; then
    echo "Activated ds_env from /scratch/fc37/$USER/"
else
    echo "ERROR: Failed to activate conda environment"
    exit 1
fi

# Force CPU backend
export JAX_PLATFORMS=cpu

# JAX compilation cache
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_cpu"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"

# Verify CPU backend
echo ""
python -c "
import jax
print(f'JAX devices: {jax.devices()}')
print(f'JAX default backend: {jax.default_backend()}')
"
echo ""

export PYTHONUNBUFFERED=1
mkdir -p cluster/logs

# =============================================================================
# Run benchmark
# =============================================================================
echo "Running benchmark..."
echo ""

if [[ "$MODEL" == "all" ]]; then
    python validation/benchmark_parallel_scan.py --n-repeats "$REPEATS"
else
    python validation/benchmark_parallel_scan.py --model "$MODEL" --n-repeats "$REPEATS"
fi

BENCH_EXIT=$?

echo ""
echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo "Exit code: $BENCH_EXIT"
echo "End time: $(date)"
echo "Results: output/bayesian/pscan_benchmark.json"
echo "============================================================"

exit $BENCH_EXIT
