#!/bin/bash
# =============================================================================
# SLURM Batch Script: Parallel Scan (PScan) GPU Benchmark for RLWM Models
# =============================================================================
# Two-stage benchmark comparing sequential lax.scan vs O(log T) associative
# scan likelihoods on A100 GPU.
#
# Stage 1: Micro-benchmark (single-participant likelihood wall time)
# Stage 2: Full MCMC A/B comparison (M3 hierarchical, sequential vs pscan)
# Stage 3: Posterior agreement check (PSCAN-06 criterion)
#
# Usage:
#   sbatch cluster/13_bayesian_pscan.slurm
#   sbatch --export=MODEL=wmrl_m5 cluster/13_bayesian_pscan.slurm
#
# Requirements:
#   - rlwm_gpu conda environment with JAX CUDA
#   - output/task_trials_long.csv (trial data)
#   - output/summary_participant_metrics.csv (for L2 regression)
#
# Outputs:
#   - output/bayesian/pscan_benchmark.json (Stage 1)
#   - output/bayesian/{model}_posterior.nc (Stage 2 sequential)
#   - output/bayesian/{model}_posterior_pscan.nc (Stage 2 pscan)
#   - output/bayesian/pscan_ab_comparison.json (Stage 3)
#   - cluster/logs/bayesian_pscan_{jobid}.out
#
# PSCAN-05: GPU benchmark infrastructure
# PSCAN-06: A/B posterior comparison (< 5% relative error on group means,
#           WAIC/LOO agreement < 1.0)
# =============================================================================

#SBATCH --job-name=bayesian_pscan
#SBATCH --output=cluster/logs/bayesian_pscan_%j.out
#SBATCH --error=cluster/logs/bayesian_pscan_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4

# =============================================================================
# Configuration
# =============================================================================
MODEL="${MODEL:-wmrl_m3}"
CHAINS="${CHAINS:-4}"
WARMUP="${WARMUP:-1000}"
SAMPLES="${SAMPLES:-2000}"
BENCHMARK_REPEATS="${BENCHMARK_REPEATS:-50}"

# =============================================================================
# Environment Setup
# =============================================================================

module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "RLWM Parallel Scan GPU Benchmark"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Partition: ${SLURM_JOB_PARTITION:-gpu}"
echo "GPU(s): ${CUDA_VISIBLE_DEVICES:-none}"
echo "Project: $PROJECT_ROOT"
echo "Model: $MODEL"
echo "Chains: $CHAINS  |  Warmup: $WARMUP  |  Samples: $SAMPLES"
echo "Benchmark repeats: $BENCHMARK_REPEATS"
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
    exit 1
fi

export NUMPYRO_HOST_DEVICE_COUNT=1

# JAX compilation cache
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_gpu"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" || {
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
gpu_devices = [d for d in devices if d.platform == 'gpu']
if gpu_devices:
    print(f'GPU(s) available: {gpu_devices}')
else:
    print('WARNING: No GPU detected!')
"
echo ""

mkdir -p cluster/logs
export PYTHONUNBUFFERED=1

DATA_FILE="output/task_trials_long.csv"
if [[ ! -f "$DATA_FILE" ]]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

# =============================================================================
# Stage 1: Micro-benchmark (single-participant likelihood timing)
# =============================================================================
echo ""
echo "============================================================"
echo "STAGE 1: Micro-benchmark (sequential vs pscan likelihood)"
echo "Started: $(date)"
echo "============================================================"

python validation/benchmark_parallel_scan.py \
    --model "$MODEL" \
    --n-repeats "$BENCHMARK_REPEATS"

STAGE1_EXIT=$?
echo "Stage 1 exit code: $STAGE1_EXIT"

# =============================================================================
# Stage 2a: Sequential baseline MCMC
# =============================================================================
echo ""
echo "============================================================"
echo "STAGE 2a: Sequential baseline MCMC ($MODEL)"
echo "Started: $(date)"
echo "============================================================"

# Start GPU monitoring
if command -v nvidia-smi &>/dev/null; then
    (while true; do
        echo "[GPU $(date +%H:%M:%S)] $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{printf "util=%s%% mem=%s/%sMB temp=%sC", $1, $2, $3, $4}')"
        sleep 300
    done) &
    GPU_MONITOR_PID=$!
fi

SEQ_START=$(date +%s)

python scripts/fitting/fit_bayesian.py \
    --model "$MODEL" \
    --data "$DATA_FILE" \
    --output output/ \
    --chains "$CHAINS" \
    --warmup "$WARMUP" \
    --samples "$SAMPLES" \
    --use-gpu

SEQ_EXIT=$?
SEQ_END=$(date +%s)
SEQ_WALL=$((SEQ_END - SEQ_START))

echo "Stage 2a (sequential) exit code: $SEQ_EXIT, wall time: ${SEQ_WALL}s"

# =============================================================================
# Stage 2b: PScan MCMC
# =============================================================================
echo ""
echo "============================================================"
echo "STAGE 2b: PScan MCMC ($MODEL --use-pscan)"
echo "Started: $(date)"
echo "============================================================"

PSCAN_START=$(date +%s)

python scripts/fitting/fit_bayesian.py \
    --model "$MODEL" \
    --data "$DATA_FILE" \
    --output output/ \
    --chains "$CHAINS" \
    --warmup "$WARMUP" \
    --samples "$SAMPLES" \
    --use-gpu \
    --use-pscan

PSCAN_EXIT=$?
PSCAN_END=$(date +%s)
PSCAN_WALL=$((PSCAN_END - PSCAN_START))

echo "Stage 2b (pscan) exit code: $PSCAN_EXIT, wall time: ${PSCAN_WALL}s"

# Stop GPU monitoring
[[ -n "$GPU_MONITOR_PID" ]] && kill "$GPU_MONITOR_PID" 2>/dev/null && wait "$GPU_MONITOR_PID" 2>/dev/null
unset GPU_MONITOR_PID

# =============================================================================
# Stage 3: A/B Posterior Comparison (PSCAN-06)
# =============================================================================
echo ""
echo "============================================================"
echo "STAGE 3: A/B Posterior Comparison (PSCAN-06)"
echo "Started: $(date)"
echo "============================================================"

python -c "
import json
import sys
from pathlib import Path

model = '${MODEL}'
bayesian_dir = Path('output/bayesian')
seq_path = bayesian_dir / f'{model}_posterior.nc'
pscan_path = bayesian_dir / f'{model}_posterior_pscan.nc'

if not seq_path.exists():
    print(f'ERROR: Sequential posterior not found: {seq_path}')
    sys.exit(1)
if not pscan_path.exists():
    print(f'ERROR: PScan posterior not found: {pscan_path}')
    sys.exit(1)

import arviz as az
import numpy as np

idata_seq = az.from_netcdf(str(seq_path))
idata_pscan = az.from_netcdf(str(pscan_path))

# Compare group-level posterior means (all *_mu_pr sites)
results = {'model': model, 'parameters': {}}
all_pass = True

seq_vars = list(idata_seq.posterior.data_vars)
mu_pr_sites = sorted(v for v in seq_vars if v.endswith('_mu_pr'))

print(f'Comparing {len(mu_pr_sites)} group-level parameters:')
print(f'{\"Parameter\":<25} {\"Seq Mean\":>10} {\"PScan Mean\":>10} {\"Rel Error\":>12} {\"Pass\":>6}')
print('-' * 70)

for pname in mu_pr_sites:
    seq_mean = float(idata_seq.posterior[pname].values.mean())
    pscan_mean = float(idata_pscan.posterior[pname].values.mean())
    rel_error = abs(seq_mean - pscan_mean) / (abs(seq_mean) + 1e-10)
    passes = rel_error < 0.05  # PSCAN-06 criterion: < 5% relative error
    if not passes:
        all_pass = False
    results['parameters'][pname] = {
        'seq_mean': round(seq_mean, 6),
        'pscan_mean': round(pscan_mean, 6),
        'rel_error': round(rel_error, 6),
        'passes_5pct': passes,
    }
    status = 'PASS' if passes else 'FAIL'
    print(f'{pname:<25} {seq_mean:>10.4f} {pscan_mean:>10.4f} {rel_error:>12.2e} {status:>6}')

# Compare WAIC
try:
    waic_seq = az.waic(idata_seq)
    waic_pscan = az.waic(idata_pscan)
    waic_diff = abs(waic_seq.waic - waic_pscan.waic)
    waic_passes = waic_diff < 1.0
    results['waic'] = {
        'seq': round(float(waic_seq.waic), 2),
        'pscan': round(float(waic_pscan.waic), 2),
        'abs_diff': round(float(waic_diff), 4),
        'passes_1pt': waic_passes,
    }
    if not waic_passes:
        all_pass = False
    status = 'PASS' if waic_passes else 'FAIL'
    print(f'\nWAIC: seq={waic_seq.waic:.2f}, pscan={waic_pscan.waic:.2f}, diff={waic_diff:.4f} [{status}]')
except Exception as exc:
    print(f'WARNING: WAIC comparison failed: {exc}')
    results['waic'] = {'error': str(exc)}

# Compare LOO
try:
    loo_seq = az.loo(idata_seq, pointwise=True)
    loo_pscan = az.loo(idata_pscan, pointwise=True)
    loo_diff = abs(loo_seq.loo - loo_pscan.loo)
    loo_passes = loo_diff < 1.0
    results['loo'] = {
        'seq': round(float(loo_seq.loo), 2),
        'pscan': round(float(loo_pscan.loo), 2),
        'abs_diff': round(float(loo_diff), 4),
        'passes_1pt': loo_passes,
    }
    if not loo_passes:
        all_pass = False
    status = 'PASS' if loo_passes else 'FAIL'
    print(f'LOO:  seq={loo_seq.loo:.2f}, pscan={loo_pscan.loo:.2f}, diff={loo_diff:.4f} [{status}]')
except Exception as exc:
    print(f'WARNING: LOO comparison failed: {exc}')
    results['loo'] = {'error': str(exc)}

results['overall_pass'] = all_pass
results['seq_wall_seconds'] = ${SEQ_WALL}
results['pscan_wall_seconds'] = ${PSCAN_WALL}
mcmc_speedup = ${SEQ_WALL} / max(${PSCAN_WALL}, 1)
results['mcmc_speedup'] = round(mcmc_speedup, 3)

print(f'\nMCMC wall time: sequential={${SEQ_WALL}}s, pscan={${PSCAN_WALL}}s, speedup={mcmc_speedup:.2f}x')
print(f'\nOVERALL PSCAN-06: {\"PASS\" if all_pass else \"FAIL\"}')

# Save comparison
out_path = bayesian_dir / 'pscan_ab_comparison.json'
with open(out_path, 'w') as fh:
    json.dump(results, fh, indent=2)
print(f'Comparison saved to: {out_path}')
"

STAGE3_EXIT=$?
echo "Stage 3 exit code: $STAGE3_EXIT"

# =============================================================================
# Job Summary
# =============================================================================
echo ""
echo "============================================================"
echo "JOB SUMMARY"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Model: $MODEL"
echo "Stage 1 (micro-benchmark): exit=$STAGE1_EXIT"
echo "Stage 2a (sequential MCMC): exit=$SEQ_EXIT, wall=${SEQ_WALL}s"
echo "Stage 2b (pscan MCMC): exit=$PSCAN_EXIT, wall=${PSCAN_WALL}s"
echo "Stage 3 (A/B comparison): exit=$STAGE3_EXIT"
echo "End time: $(date)"
echo ""
echo "Results:"
echo "  output/bayesian/pscan_benchmark.json"
echo "  output/bayesian/${MODEL}_posterior.nc"
echo "  output/bayesian/${MODEL}_posterior_pscan.nc"
echo "  output/bayesian/pscan_ab_comparison.json"
echo "============================================================"

# GPU info
if command -v nvidia-smi &>/dev/null; then
    echo ""
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free \
        --format=csv,noheader 2>/dev/null
fi
