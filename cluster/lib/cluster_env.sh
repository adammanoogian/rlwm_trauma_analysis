#!/usr/bin/env bash
# =============================================================================
# cluster_env.sh — Shared environment setup for SLURM jobs
# =============================================================================
# Source this at the start of any SLURM script for standardized environment
# activation, JAX configuration, and thread control.
#
# Usage (at top of your .slurm file, after #SBATCH directives):
#   source cluster/lib/cluster_env.sh
#   activate_env "rlwm_gpu"        # conda activation with fallback ladder
#   setup_jax_cache                # per-node JAX compilation cache
#   setup_jax_flags                # XLA thread control (CPU jobs)
#   verify_gpu                     # GPU visibility check (GPU jobs)
#   print_job_header               # standardized job info block
#
# Copied from project_utils/templates/hpc/lib/cluster_env.sh and customized
# for the rlwm_trauma_analysis project.
# =============================================================================

# Project allocation fallback (override via SLURM --export or in your script)
_PROJECT="${PROJECT:-fc37}"

# -- activate_env: conda activation with fallback ladder --------------------
# Tries: name -> /scratch/$PROJECT/$USER/ -> /scratch/<fallback>/$USER/
# Exits with clear error if all fail.
activate_env() {
    local env_name="$1"
    if [[ -z "$env_name" ]]; then
        echo "ERROR: activate_env requires an environment name"
        exit 1
    fi

    # Load conda (miniforge preferred over anaconda for batch shell compatibility)
    if module load miniforge3 2>/dev/null; then
        : # loaded successfully
    elif module load anaconda 2>/dev/null; then
        eval "$(conda shell.bash hook)" 2>/dev/null || true
    fi

    if conda activate "$env_name" 2>/dev/null; then
        echo "Activated: $env_name (by name)"
    elif conda activate "/scratch/${_PROJECT}/${USER}/conda/envs/${env_name}" 2>/dev/null; then
        echo "Activated: $env_name (from /scratch/${_PROJECT}/${USER}/)"
    elif conda activate "/scratch/${PROJECT}/${USER}/conda/envs/${env_name}" 2>/dev/null; then
        echo "Activated: $env_name (from /scratch/${PROJECT}/${USER}/)"
    else
        echo "ERROR: Failed to activate conda environment: $env_name"
        echo "  Tried:"
        echo "    conda activate $env_name"
        echo "    conda activate /scratch/${_PROJECT}/${USER}/conda/envs/${env_name}"
        echo "  Create with: conda env create -f environment.yml"
        exit 1
    fi
}

# -- setup_jax_cache: per-node compilation cache -----------------------------
# GPU cache is portable across same-type GPUs.
# CPU cache is NODE-SPECIFIC (different CPU features -> SIGILL crashes).
setup_jax_cache() {
    local cache_suffix="${1:-}"  # optional suffix, e.g., "gpu" or "cpu"

    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] || [[ -n "$cache_suffix" && "$cache_suffix" == "gpu" ]]; then
        # GPU: portable across same GPU type
        export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu"
    else
        # CPU: node-specific to avoid SIGILL from feature mismatch
        export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache/${SLURMD_NODENAME:-local}"
    fi

    export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0

    mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || {
        echo "WARNING: Could not create JAX cache dir, disabling cache"
        unset JAX_COMPILATION_CACHE_DIR
        return 0
    }
    echo "JAX cache: $JAX_COMPILATION_CACHE_DIR"
}

# -- setup_jax_flags: XLA thread control for CPU jobs ------------------------
# Prevents LLVM memory exhaustion during JIT compilation on CPU.
# Safe to call on GPU jobs (flags are CPU-specific, harmless if GPU is primary).
setup_jax_flags() {
    local njobs="${1:-4}"
    local threads_per="${2:-4}"

    # Safe baseline (all JAX versions)
    export XLA_FLAGS="${XLA_FLAGS:---xla_cpu_multi_thread_eigen=false}"

    # Thread control
    export OMP_NUM_THREADS="$threads_per"
    export MKL_NUM_THREADS="$threads_per"
    export TF_NUM_INTEROP_THREADS=1
    export TF_NUM_INTRAOP_THREADS="$threads_per"
    export OPENBLAS_NUM_THREADS="$threads_per"

    echo "Thread config: NJOBS=$njobs x $threads_per threads/worker = $((njobs * threads_per)) total"
}

# -- verify_gpu: check CUDA is visible (framework-agnostic) ------------------
# Tries JAX first, falls back to PyTorch, falls back to nvidia-smi.
# Works for both JAX-based (NumPyro, active inference) and PyTorch-based
# (DCM, neural net) projects.
verify_gpu() {
    echo ""
    echo "GPU verification:"

    # Guard: skip if no GPU was requested
    if [[ "${SLURM_GPUS_ON_NODE:-0}" -eq 0 ]] && [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "  (no GPU requested -- skipping)"
        return 0
    fi

    python3 -c "
import sys
# Try JAX first (most of our fitting code)
try:
    import jax
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    if gpu_devices:
        for d in gpu_devices:
            print(f'  JAX: {d.platform}:{d.id} -- {d.device_kind}')
        sys.exit(0)
except ImportError:
    pass

# Fall back to PyTorch
try:
    import torch
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / 1e9
            print(f'  PyTorch: cuda:{i} -- {name} ({mem:.1f} GB)')
        sys.exit(0)
except ImportError:
    pass

print('  WARNING: No GPU detected by JAX or PyTorch.')
print('  Check: CUDA_VISIBLE_DEVICES, conda env, jax[cuda] or torch install')
sys.exit(1)
" || {
        # Last resort: nvidia-smi
        if command -v nvidia-smi &>/dev/null; then
            echo "  (Python GPU check failed, but nvidia-smi reports:)"
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | \
                sed 's/^/    /'
        else
            echo "  GPU verification FAILED"
            return 1
        fi
    }
}

# -- verify_imports: smoke-test critical imports -----------------------------
# Pass module names as arguments. Fails fast if any import is missing.
verify_imports() {
    local modules=("$@")
    if [[ ${#modules[@]} -eq 0 ]]; then
        return 0
    fi

    local import_str=""
    for mod in "${modules[@]}"; do
        import_str+="import ${mod}; "
    done
    import_str+="print('imports OK: ${modules[*]}')"

    python3 -c "$import_str" || {
        echo "ERROR: Import check failed. Missing dependency in conda env."
        exit 1
    }
}

# -- print_job_header: standardized job info block ---------------------------
print_job_header() {
    local title="${1:-SLURM Job}"
    echo "================================================================"
    echo "  $title"
    echo "================================================================"
    echo "  Job ID:     ${SLURM_JOB_ID:-local}"
    echo "  Job name:   ${SLURM_JOB_NAME:-unknown}"
    echo "  Node:       ${SLURMD_NODENAME:-$(hostname)}"
    echo "  Partition:  ${SLURM_JOB_PARTITION:-unknown}"
    echo "  CPUs:       ${SLURM_CPUS_ON_NODE:-unknown}"
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "  GPU(s):     $CUDA_VISIBLE_DEVICES"
    fi
    echo "  Commit:     $(git rev-parse --short HEAD 2>/dev/null || echo 'n/a') [$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') dirty]"
    echo "  Start:      $(date)"
    echo "================================================================"
    echo ""
}

# -- crlf_guard: strip CRLF from current script -----------------------------
# Call this if the script might have been checked out on Windows.
# Safe to call unconditionally (no-op on clean files).
crlf_guard() {
    local self="${BASH_SOURCE[1]:-$0}"
    if grep -Pq '\r$' "$self" 2>/dev/null; then
        echo "WARNING: CRLF detected in $self -- fixing in-place"
        sed -i 's/\r$//' "$self"
    fi
}
