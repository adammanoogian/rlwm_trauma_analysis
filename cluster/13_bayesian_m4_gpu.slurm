#!/bin/bash
# =============================================================================
# SLURM Batch Script: M4 Hierarchical LBA Bayesian Fit (Monash M3 A100 GPU)
# =============================================================================
# Runs M4 (WM-RL+LBA) hierarchical NumPyro fit on a single A100 GPU.
# M4 is the ONLY model in this pipeline that requires GPU for Bayesian fitting.
# All choice-only models (M1/M2/M3/M5/M6a/M6b) use CPU-only MCMC.
#
# Requirements addressed: M4H-01 (float64 isolation), M4H-02 (log(b-A)
# non-centered parameterization), M4H-04 (checkpoint-resume via warmup
# pickle), M4H-05 (Pareto-k gating), M4H-06 (SLURM batch submission).
#
# Wall time: 48h (LBA joint choice+RT likelihood is ~10x slower than
# choice-only models; 4 chains x 1000 warmup x 1500 samples on N=154).
#
# Checkpoint-resume strategy: If the job is killed after warmup completes,
# the warmup state is pickled to output/bayesian/m4_warmup_state.pkl.
# Resubmitting this script will detect the checkpoint and resume directly
# from sampling (no re-warmup needed).
#
# Usage:
#   sbatch cluster/13_bayesian_m4_gpu.slurm
#
#   # Resume after wall-time kill (warmup already saved):
#   sbatch cluster/13_bayesian_m4_gpu.slurm   # same command; auto-detects checkpoint
#
# Outputs saved to:
#   - output/bayesian/wmrl_m4_individual_fits.csv
#   - output/bayesian/wmrl_m4_posterior.nc
#   - output/bayesian/wmrl_m4_shrinkage_report.md
#   - output/bayesian/wmrl_m4_run_metadata.json
#   - output/bayesian/wmrl_m4_pareto_k_report.md
#   - output/bayesian/wmrl_m4_pareto_k_report.json
#   - output/bayesian/m4_warmup_state.pkl  (checkpoint; removed on clean run)
# =============================================================================

#SBATCH --job-name=bayesian_m4_lba
#SBATCH --output=logs/bayesian_m4_%j.out
#SBATCH --error=logs/bayesian_m4_%j.err
#SBATCH --time=48:00:00
#SBATCH --mem=96G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

# --- Environment Setup ---
module load miniforge3

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "M4 (WM-RL+LBA) Hierarchical Bayesian Fit — GPU A100"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Project: $PROJECT_ROOT"
echo "Start time: $(date)"
echo "============================================================"

# Activate conda environment
# Try rlwm_gpu first (GPU-enabled JAX), then ds_env as fallback
# Note: reversed order vs choice-only SLURM scripts (which prefer ds_env)
if conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu (GPU JAX env)"
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/fc37/$USER/"
elif conda activate /scratch/${PROJECT:-fc37}/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/$PROJECT/$USER/"
elif conda activate ds_env 2>/dev/null; then
    echo "Activated ds_env (fallback — GPU support not guaranteed)"
else
    echo "ERROR: Failed to activate conda environment (tried rlwm_gpu, ds_env)"
    exit 1
fi

# --- JAX GPU environment ---
# GPU parallelism via chain_method='vectorized' (NOT NUMPYRO_HOST_DEVICE_COUNT)
# Do NOT set NUMPYRO_HOST_DEVICE_COUNT for GPU runs
export JAX_PLATFORMS=cuda

_PROJECT="${PROJECT:-fc37}"
# Separate JAX cache from choice-only models (float64 vs float32 JIT traces)
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_m4_bayesian"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo ""

# --- GPU + float64 verification (M4H-01) ---
python -c "
import jax
jax.config.update('jax_enable_x64', True)
devices = jax.devices()
print(f'JAX version: {jax.__version__}')
print(f'Devices: {devices}')
print(f'Float64: {jax.numpy.zeros(1).dtype}')
assert any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devices), \
    'No GPU device found! M4 requires GPU.'
assert jax.numpy.zeros(1).dtype == jax.numpy.float64, \
    'Float64 not active!'
print('GPU + float64 verification PASSED')
" || {
    echo "ERROR: GPU or float64 verification failed"
    exit 1
}
echo ""

# --- Create output directories ---
mkdir -p logs
mkdir -p output/bayesian

# --- Run M4 hierarchical LBA fit ---
echo "[$(date)] Starting M4 (WM-RL+LBA) hierarchical Bayesian fit"
echo "  chains=4, warmup=1000, samples=1500, seed=42"
echo "  checkpoint-dir=output/bayesian"
echo ""

python scripts/13_fit_bayesian_m4.py \
    --data output/task_trials_long.csv \
    --chains 4 \
    --warmup 1000 \
    --samples 1500 \
    --seed 42 \
    --checkpoint-dir output/bayesian

EXIT_CODE=$?

echo ""
echo "[$(date)] M4 hierarchical LBA fit finished (exit code: $EXIT_CODE)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Expected outputs:"
    for f in output/bayesian/wmrl_m4_individual_fits.csv \
             output/bayesian/wmrl_m4_posterior.nc \
             output/bayesian/wmrl_m4_shrinkage_report.md \
             output/bayesian/wmrl_m4_run_metadata.json \
             output/bayesian/wmrl_m4_pareto_k_report.md \
             output/bayesian/wmrl_m4_pareto_k_report.json \
             output/bayesian/m4_warmup_state.pkl; do
        if [ -f "$f" ]; then
            echo "  + $f ($(du -h "$f" | cut -f1))"
        else
            echo "  - $f (MISSING)"
        fi
    done
else
    echo "ERROR: 13_fit_bayesian_m4.py exited with code $EXIT_CODE"
    echo "Check: logs/bayesian_m4_${SLURM_JOB_ID}.err"
    echo ""
    echo "If wall-time was exceeded after warmup, resubmit to resume:"
    echo "  sbatch cluster/13_bayesian_m4_gpu.slurm"
fi
