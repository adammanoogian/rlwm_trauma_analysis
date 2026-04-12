#!/bin/bash
#SBATCH --job-name=bayesian_m3
#SBATCH --output=logs/bayesian_m3_%j.out
#SBATCH --error=logs/bayesian_m3_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

# --- JAX compilation cache (prevents re-compilation across runs) ---
export JAX_COMPILATION_CACHE_DIR="/scratch/$USER/jax_cache"
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true"
export JAX_PLATFORMS=cpu

# --- NumPyro device count ---
export NUMPYRO_HOST_DEVICE_COUNT=1

# --- Activate environment ---
module load anaconda
conda activate ds_env

# --- Create output directories ---
mkdir -p logs
mkdir -p output/bayesian

# --- Run M3 hierarchical fit ---
echo "[$(date)] Starting M3 hierarchical Bayesian fit"
echo "JAX_COMPILATION_CACHE_DIR=$JAX_COMPILATION_CACHE_DIR"

python scripts/fitting/fit_bayesian.py \
    --model wmrl_m3 \
    --data output/task_trials_long.csv \
    --chains 4 \
    --warmup 1000 \
    --samples 2000 \
    --seed 42

echo "[$(date)] M3 hierarchical fit complete"
echo "Outputs:"
echo "  - output/bayesian/wmrl_m3_individual_fits.csv"
echo "  - output/bayesian/wmrl_m3_posterior.nc"
echo "  - output/bayesian/wmrl_m3_shrinkage_report.md"
echo "  - output/bayesian/wmrl_m3_ppc_results.csv"
