#!/bin/bash
# =============================================================================
# SLURM Batch Script: PScan GPU Benchmark (Monash M3)
# =============================================================================
# Runs the micro-benchmark on a GPU node to measure the actual speedup from
# associative scan parallelism on GPU. Compare results against the CPU
# benchmark (19_benchmark_pscan_cpu.slurm) to quantify GPU benefit.
#
# Usage:
#   sbatch cluster/19_benchmark_pscan_gpu.slurm                    # All 6 models
#   sbatch --export=MODEL=wmrl_m3 cluster/19_benchmark_pscan_gpu.slurm  # Single model
#   sbatch --export=REPEATS=50 cluster/19_benchmark_pscan_gpu.slurm     # More repeats
#
# For the full MCMC A/B comparison (sequential vs pscan posterior agreement):
#   sbatch cluster/13_bayesian_pscan.slurm
#
# Outputs:
#   - output/bayesian/pscan_benchmark.json (overwritten per run)
#   - cluster/logs/pscan_bench_gpu_{jobid}.out
# =============================================================================

#SBATCH --job-name=pscan-bench-gpu
#SBATCH --output=cluster/logs/pscan_bench_gpu_%j.out
#SBATCH --error=cluster/logs/pscan_bench_gpu_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

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
echo "RLWM PScan GPU Benchmark"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Partition: ${SLURM_JOB_PARTITION:-gpu}"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-none}"
echo "Project: $PROJECT_ROOT"
echo "Model: $MODEL"
echo "Repeats: $REPEATS"
echo "Start time: $(date)"
echo "============================================================"

# Activate GPU conda environment
if conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu by name"
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/fc37/$USER/"
elif conda activate /scratch/$PROJECT/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/$PROJECT/$USER/"
else
    echo "ERROR: Failed to activate 'rlwm_gpu' conda environment"
    echo "Create it first with: bash cluster/00_setup_env_gpu.sh"
    exit 1
fi

# JAX compilation cache (separate from CPU cache)
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"

# Verify GPU access
echo ""
echo "Verifying GPU access..."
python -c "
import jax
devices = jax.devices()
print(f'JAX devices: {devices}')
print(f'JAX default backend: {jax.default_backend()}')
gpu_devices = [d for d in devices if d.platform == 'gpu']
if gpu_devices:
    print(f'GPU(s) available: {gpu_devices}')
else:
    print('WARNING: No GPU detected! Benchmark will run on CPU.')
"
echo ""

# GPU info
if command -v nvidia-smi &>/dev/null; then
    echo "GPU hardware:"
    nvidia-smi --query-gpu=name,memory.total,driver_version \
        --format=csv,noheader 2>/dev/null
    echo ""
fi

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

# Post-run GPU memory
if command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "GPU memory after benchmark:"
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total \
        --format=csv,noheader 2>/dev/null
fi

echo ""
echo "============================================================"
echo "COMPLETE"
echo "============================================================"
echo "Exit code: $BENCH_EXIT"
echo "End time: $(date)"
echo "Results: output/bayesian/pscan_benchmark.json"
echo ""
echo "Next steps:"
echo "  Compare CPU vs GPU results in pscan_benchmark.json"
echo "  For full MCMC A/B comparison: sbatch cluster/13_bayesian_pscan.slurm"
echo "============================================================"

exit $BENCH_EXIT
