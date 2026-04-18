#!/bin/bash
# =============================================================================
# SLURM Array Script: Step 21.2 Bayesian Parameter Recovery (Monash M3)
# =============================================================================
# Fits ONE synthetic subject per array task via hierarchical MCMC for the
# requested choice-only model.  With --array=1-50 the 50 synthetic subjects
# run in parallel, finishing the per-model recovery in ~30 min wall clock
# (vs. ~21h sequential).
#
# Usage:
#   sbatch --export=ALL,MODEL=wmrl_m3 cluster/21_2_recovery.slurm
#   sbatch --export=ALL,MODEL=wmrl_m6b,SEED_BASE=2000 cluster/21_2_recovery.slurm
#
# Post-array aggregation MUST run separately (separate SLURM script).  Chain
# via the master pipeline orchestrator (plan 21-10):
#   ARRAY_JOBID=$(sbatch --parsable --export=ALL,MODEL=$MODEL \
#                         cluster/21_2_recovery.slurm)
#   sbatch --dependency=afterok:$ARRAY_JOBID \
#          --export=ALL,MODEL=$MODEL \
#          cluster/21_2_recovery_aggregate.slurm
#
# Outputs saved to:
#   - output/bayesian/21_recovery/{MODEL}_subject_001.json
#   - output/bayesian/21_recovery/{MODEL}_subject_002.json
#   - ... (one JSON per array task)
#
# NOTE: autopush.sh is intentionally NOT sourced here — 50 concurrent pushes
# from the array tasks would race each other and waste bandwidth.  The
# aggregator script sources autopush instead, once the array completes.
# =============================================================================

#SBATCH --job-name=bayes_21_2_recovery
#SBATCH --output=logs/bayesian_21_2_recovery_%A_%a.out
#SBATCH --error=logs/bayesian_21_2_recovery_%A_%a.err
#SBATCH --time=01:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=comp
#SBATCH --array=1-50

# --- Dispatch knobs (overridable via --export=ALL,MODEL=...) ---
MODEL="${MODEL:-wmrl_m3}"
N_SUBJECTS="${N_SUBJECTS:-50}"
SEED_BASE="${SEED_BASE:-1000}"

# --- Environment Setup (matches 13_bayesian_m6b.slurm pattern) ---
module load miniforge3

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Step 21.2 Bayesian Recovery: MODEL=${MODEL}"
echo "Array task: ${SLURM_ARRAY_TASK_ID:-local} / ${SLURM_ARRAY_TASK_COUNT:-?}"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Array ID: ${SLURM_ARRAY_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Project: $PROJECT_ROOT"
echo "MODEL: $MODEL"
echo "N_SUBJECTS: $N_SUBJECTS"
echo "SEED_BASE: $SEED_BASE"
echo "Start time: $(date)"
echo "============================================================"

# Activate conda environment — ds_env first, with /scratch ladder fallback
if conda activate ds_env 2>/dev/null; then
    echo "Activated ds_env by name"
elif conda activate /scratch/fc37/$USER/conda/envs/ds_env 2>/dev/null; then
    echo "Activated ds_env from /scratch/fc37/$USER/"
elif conda activate /scratch/${PROJECT:-fc37}/$USER/conda/envs/ds_env 2>/dev/null; then
    echo "Activated ds_env from /scratch/$PROJECT/$USER/"
elif conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu (fallback)"
else
    echo "ERROR: Failed to activate conda environment (tried ds_env, rlwm_gpu)"
    exit 1
fi

# Verify jax is importable
python -c "import jax; print(f'JAX version: {jax.__version__}, platforms: {jax.devices()}')" || {
    echo "ERROR: JAX import failed after conda activation"
    exit 1
}
echo ""

# --- JAX compilation cache (per-MODEL to avoid cross-model cache pollution;
#     shared across array tasks of the SAME model to warm-start after task 1) ---
export JAX_PLATFORMS=cpu
export NUMPYRO_HOST_DEVICE_COUNT=2
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_21_recovery_${MODEL}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"

# --- Create output directories ---
mkdir -p logs
mkdir -p output/bayesian/21_recovery

# --- Resolve per-array-task subject id + seed ---
SUBJECT_IDX="${SLURM_ARRAY_TASK_ID:-1}"
SUBJECT_SEED=$((SEED_BASE + SUBJECT_IDX))

echo ""
echo "[$(date)] Starting single-subject recovery fit"
echo "  SUBJECT_IDX=$SUBJECT_IDX"
echo "  SUBJECT_SEED=$SUBJECT_SEED"

python scripts/21_run_bayesian_recovery.py \
    --mode single-subject \
    --model "$MODEL" \
    --subject-idx "$SUBJECT_IDX" \
    --seed "$SUBJECT_SEED" \
    --output-dir output/bayesian/21_recovery/

EXIT_CODE=$?

echo ""
echo "[$(date)] Single-subject recovery finished (exit code: $EXIT_CODE)"

if [ $EXIT_CODE -eq 0 ]; then
    JSON_FILE=$(printf "output/bayesian/21_recovery/%s_subject_%03d.json" \
                       "$MODEL" "$SUBJECT_IDX")
    if [ -f "$JSON_FILE" ]; then
        echo "  ✓ $JSON_FILE ($(du -h "$JSON_FILE" | cut -f1))"
    else
        echo "  ✗ $JSON_FILE (MISSING — single-subject runner did not write output)"
        EXIT_CODE=2
    fi
else
    echo "ERROR: single-subject runner exited with code $EXIT_CODE"
fi

# NOTE: autopush.sh is NOT sourced here (avoid 50 concurrent pushes).
#       The aggregator script (21_2_recovery_aggregate.slurm) handles the push.

exit $EXIT_CODE
