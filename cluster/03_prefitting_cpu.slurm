#!/bin/bash
# =============================================================================
# SLURM Batch Script: Stage 03 — Model Prefitting (CPU)
# =============================================================================
# Parameterized STEP dispatcher for the 5 scripts under 03_model_prefitting/:
#   STEP=synthetic          -> 01_generate_synthetic_data.py
#   STEP=parameter_sweep    -> 02_run_parameter_sweep.py
#   STEP=model_recovery     -> 03_run_model_recovery.py      (MODEL= required)
#   STEP=prior_predictive   -> 04_run_prior_predictive.py    (MODEL= required)
#   STEP=bayesian_recovery  -> 05_run_bayesian_recovery.py   (MODEL= required)
#
# Replaces the per-step SLURMs (21_1_prior_predictive, 21_2_recovery,
# 21_2_recovery_aggregate) with a single parameterized CPU template following
# the Phase 28 13_bayesian_choice_only.slurm pattern.
#
# NOTE: The deleted-in-29-04b scripts/03_model_prefitting/09_run_ppc.py is NOT
# dispatched here. Posterior PPC (the thin orchestrator) now lives in stage 05
# (scripts/05_post_fitting_checks/03_run_posterior_ppc.py) per Scheme D and is
# reachable via cluster/05_post_checks.slurm STEP=posterior_ppc.
#
# Usage:
#   sbatch --export=ALL,STEP=prior_predictive,MODEL=wmrl_m3 cluster/03_prefitting_cpu.slurm
#   sbatch --export=ALL,STEP=bayesian_recovery,MODEL=wmrl_m5 cluster/03_prefitting_cpu.slurm
#   sbatch --export=ALL,STEP=synthetic cluster/03_prefitting_cpu.slurm
#   sbatch --export=ALL,STEP=parameter_sweep cluster/03_prefitting_cpu.slurm
#   sbatch --export=ALL,STEP=model_recovery,MODEL=wmrl_m3 cluster/03_prefitting_cpu.slurm
# =============================================================================

#SBATCH --job-name=p29_03_prefit_cpu
#SBATCH --output=logs/03_prefitting_cpu_%j.out
#SBATCH --error=logs/03_prefitting_cpu_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

set -euo pipefail

STEP="${STEP:-}"
MODEL="${MODEL:-wmrl_m3}"
SEED="${SEED:-42}"

if [ -z "$STEP" ]; then
    echo "ERROR: STEP env var is required. Valid values:" >&2
    echo "       synthetic | parameter_sweep | model_recovery | prior_predictive | bayesian_recovery" >&2
    exit 2
fi

module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

# Install autopush trap (must come BEFORE any failure-prone command).
# autopush.sh registers an EXIT trap that fires on success or failure,
# so SLURM .err/.out logs land in git regardless of outcome — see
# cluster/autopush.sh for rationale.
[ -f cluster/autopush.sh ] && source cluster/autopush.sh


echo "============================================================"
echo "Stage 03 (CPU) — Model Prefitting: STEP=$STEP MODEL=$MODEL"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       ${SLURMD_NODENAME:-$(hostname)}"
echo "Project:    $PROJECT_ROOT"
echo "Start time: $(date)"
echo "============================================================"

if conda activate ds_env 2>/dev/null; then
    echo "Activated ds_env by name"
elif conda activate /scratch/fc37/$USER/conda/envs/ds_env 2>/dev/null; then
    echo "Activated ds_env from /scratch/fc37/$USER/"
elif conda activate /scratch/${PROJECT:-fc37}/$USER/conda/envs/ds_env 2>/dev/null; then
    echo "Activated ds_env from /scratch/${PROJECT:-fc37}/$USER/"
elif conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu (fallback)"
else
    echo "ERROR: Failed to activate conda environment (tried ds_env, rlwm_gpu)"
    exit 1
fi

# Re-register editable rlwm install (idempotent; protects against stale
# egg-link after folder moves / fresh compute-node caches). Mirrors
# cluster/00_preflight.slurm step [2/5].
pip install -e . --quiet --no-deps 2>/dev/null || \
    echo "WARNING: pip install -e . --no-deps failed; 'import rlwm' may fail downstream"

python -c "import jax; print(f'JAX version: {jax.__version__}, platforms: {jax.devices()}')" || {
    echo "ERROR: JAX import failed after conda activation"
    exit 1
}

export JAX_PLATFORMS=cpu
export NUMPYRO_HOST_DEVICE_COUNT=4
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_03_prefit_${STEP}_${MODEL}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" || unset JAX_COMPILATION_CACHE_DIR
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"

mkdir -p logs models/bayesian/21_prior_predictive models/bayesian/21_recovery models/recovery

export PYTHONUNBUFFERED=1

case "${STEP}" in
  synthetic)
    CMD=(python scripts/03_model_prefitting/01_generate_synthetic_data.py --seed "$SEED")
    ;;
  parameter_sweep)
    CMD=(python scripts/03_model_prefitting/02_run_parameter_sweep.py --seed "$SEED")
    ;;
  model_recovery)
    CMD=(python scripts/03_model_prefitting/03_run_model_recovery.py --model "$MODEL" --seed "$SEED")
    ;;
  prior_predictive)
    CMD=(python scripts/03_model_prefitting/04_run_prior_predictive.py \
         --model "$MODEL" \
         --data data/processed/task_trials_long.csv \
         --num-draws "${NUM_DRAWS:-500}" \
         --seed "$SEED" \
         --output-dir models/bayesian/21_prior_predictive/)
    ;;
  bayesian_recovery)
    CMD=(python scripts/03_model_prefitting/05_run_bayesian_recovery.py \
         --mode "${RECOVERY_MODE:-single-subject}" \
         --model "$MODEL" \
         --subject-idx "${SUBJECT_IDX:-${SLURM_ARRAY_TASK_ID:-1}}" \
         --seed "$SEED" \
         --output-dir models/bayesian/21_recovery/)
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
echo "[$(date)] Stage 03 (CPU) STEP=$STEP MODEL=$MODEL finished (exit code: $EXIT_CODE)"

if [[ -f cluster/autopush.sh ]]; then
    source cluster/autopush.sh
fi

exit $EXIT_CODE
