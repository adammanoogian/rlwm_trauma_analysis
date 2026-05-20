#!/bin/bash
# cluster/_setup.sh -- Sourced (not executed) by every Snakemake rule.
#
# Delegates environment setup to the shared library (cluster/lib/cluster_env.sh)
# and adds Snakemake-specific concerns: thread=1 defaults (Snakemake manages
# parallelism), editable install with flock, and the verify_gpu wrapper.
#
# CUSTOMIZED for rlwm_trauma_analysis:
#   ENV_NAME = rlwm_gpu
#   _PROJECT = fc37
#   PKG_NAME = rlwm

set -euo pipefail

# --- Project configuration ---
_PROJECT="${PROJECT:-fc37}"
ENV_NAME="rlwm_gpu"

# --- Source shared library ---
# cluster_env.sh provides: activate_env, setup_jax_cache, setup_jax_flags,
# verify_gpu, verify_imports, print_job_header, crlf_guard
SETUP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "${SETUP_DIR}/lib/cluster_env.sh" ]]; then
    source "${SETUP_DIR}/lib/cluster_env.sh"
else
    echo "WARNING: cluster/lib/cluster_env.sh not found -- using inline fallbacks"
    # Minimal inline fallback if lib/ not yet copied
    module load miniforge3 2>/dev/null || module load anaconda 2>/dev/null || true
fi

# --- Snakemake thread overrides ---
# Snakemake controls parallelism at the rule level (threads: N in each rule),
# so we pin per-process threads to 1 to prevent thread explosion when multiple
# rules run concurrently. Standalone .slurm scripts use cluster_env.sh's
# setup_jax_flags() with higher per-worker threads instead.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# --- Conda activation ---
activate_env "$ENV_NAME"

# --- Editable install (serialized to prevent race in array jobs) ---
PKG_NAME="rlwm"
LOCK="/tmp/${ENV_NAME}_install.lock"

if ! python -c "import ${PKG_NAME}" 2>/dev/null; then
    (
        flock -w 120 200 || { echo "WARNING: flock timeout, skipping install"; exit 0; }
        pip install -e . --quiet --no-deps 2>/dev/null || true
    ) 200>"$LOCK"
fi
