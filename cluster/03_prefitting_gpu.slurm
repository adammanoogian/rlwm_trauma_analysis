#!/bin/bash
# =============================================================================
# SLURM Batch Script: Stage 03 — Model Prefitting (GPU)
# =============================================================================
# GPU counterpart of cluster/03_prefitting_cpu.slurm. Supports the same STEP
# dispatch surface, but allocates 1 GPU and passes --use-gpu where the
# underlying script accepts it (model_recovery, synthetic large runs).
#
# STEP=synthetic          -> 01_generate_synthetic_data.py
# STEP=parameter_sweep    -> 02_run_parameter_sweep.py
# STEP=model_recovery     -> 03_run_model_recovery.py --use-gpu
# STEP=prior_predictive   -> 04_run_prior_predictive.py       (CPU is fine too; GPU is optional)
# STEP=bayesian_recovery  -> 05_run_bayesian_recovery.py      (CPU is fine; GPU optional for large n_subj)
#
# NOTE: The deleted-in-29-04b scripts/03_model_prefitting/09_run_ppc.py is NOT
# dispatched here. Posterior PPC lives in stage 05 (see cluster/05_post_checks.slurm
# STEP=posterior_ppc for the GPU-accelerated thin orchestrator).
#
# Usage:
#   sbatch --export=ALL,STEP=model_recovery,MODEL=wmrl_m3 cluster/03_prefitting_gpu.slurm
#   sbatch --export=ALL,STEP=synthetic cluster/03_prefitting_gpu.slurm
# =============================================================================

#SBATCH --job-name=p29_03_prefit_gpu
#SBATCH --output=cluster/logs/03_prefitting_gpu_%j.out
#SBATCH --error=cluster/logs/03_prefitting_gpu_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -euo pipefail

STEP="${STEP:-}"
MODEL="${MODEL:-wmrl_m3}"
SEED="${SEED:-42}"
NSUBJ="${NSUBJ:-50}"
NDATASETS="${NDATASETS:-3}"
NSTARTS="${NSTARTS:-20}"

if [ -z "$STEP" ]; then
    echo "ERROR: STEP env var is required. Valid values:" >&2
    echo "       synthetic | parameter_sweep | model_recovery | prior_predictive | bayesian_recovery" >&2
    exit 2
fi

module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Stage 03 (GPU) — Model Prefitting: STEP=$STEP MODEL=$MODEL"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s):     ${CUDA_VISIBLE_DEVICES:-none}"
echo "Project:    $PROJECT_ROOT"
echo "Start time: $(date)"
echo "============================================================"

if conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu by name"
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/fc37/$USER/"
elif conda activate /scratch/${PROJECT:-fc37}/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/${PROJECT:-fc37}/$USER/"
else
    echo "ERROR: Failed to activate 'rlwm_gpu' conda environment"
    exit 1
fi

python -c "
import jax
devices = jax.devices()
print(f'JAX devices: {devices}')
gpu_devices = [d for d in devices if d.platform == 'gpu']
print(f'GPU count: {len(gpu_devices)}')
" || {
    echo "ERROR: JAX import failed"
    exit 1
}

_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_03_prefit_gpu"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" || unset JAX_COMPILATION_CACHE_DIR
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"

mkdir -p cluster/logs output/bayesian/21_prior_predictive output/bayesian/21_recovery output/recovery

export PYTHONUNBUFFERED=1

case "${STEP}" in
  synthetic)
    CMD=(python scripts/03_model_prefitting/01_generate_synthetic_data.py --seed "$SEED")
    ;;
  parameter_sweep)
    CMD=(python scripts/03_model_prefitting/02_run_parameter_sweep.py --seed "$SEED")
    ;;
  model_recovery)
    CMD=(python scripts/03_model_prefitting/03_run_model_recovery.py \
         --model "$MODEL" \
         --n-subjects "$NSUBJ" \
         --n-datasets "$NDATASETS" \
         --n-starts "$NSTARTS" \
         --seed "$SEED" \
         --use-gpu)
    ;;
  prior_predictive)
    CMD=(python scripts/03_model_prefitting/04_run_prior_predictive.py \
         --model "$MODEL" \
         --data output/task_trials_long.csv \
         --num-draws "${NUM_DRAWS:-500}" \
         --seed "$SEED" \
         --output-dir output/bayesian/21_prior_predictive/)
    ;;
  bayesian_recovery)
    CMD=(python scripts/03_model_prefitting/05_run_bayesian_recovery.py \
         --mode "${RECOVERY_MODE:-single-subject}" \
         --model "$MODEL" \
         --subject-idx "${SUBJECT_IDX:-${SLURM_ARRAY_TASK_ID:-1}}" \
         --seed "$SEED" \
         --output-dir output/bayesian/21_recovery/)
    ;;
  *)
    echo "ERROR: unknown STEP='${STEP}'" >&2
    echo "       expected one of: synthetic | parameter_sweep | model_recovery | prior_predictive | bayesian_recovery" >&2
    exit 2
    ;;
esac

echo ""
echo "[$(date)] Running: ${CMD[*]}"
"${CMD[@]}"
EXIT_CODE=$?

echo ""
echo "[$(date)] Stage 03 (GPU) STEP=$STEP MODEL=$MODEL finished (exit code: $EXIT_CODE)"

if [[ -f cluster/autopush.sh ]]; then
    source cluster/autopush.sh
fi

exit $EXIT_CODE
