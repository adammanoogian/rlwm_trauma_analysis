#!/bin/bash
# =============================================================================
# SLURM Batch Script: Stage 04b — Choice-Only Hierarchical Bayesian Fit (CPU)
# =============================================================================
# Parameterized template for all 6 choice-only models (M1/M2/M3/M5/M6a/M6b).
# Replaces the 6 per-model templates (13_bayesian_m{1,2,3,5,6a,6b}.slurm) and
# the 13_bayesian_m6b_subscale.slurm variant (via --export=SUBSCALE=1).
# CPU-only (no GPU needed for choice-only models).
#
# Usage:
#   sbatch --export=ALL,MODEL=wmrl_m3 cluster/04b_bayesian_cpu.slurm
#   sbatch --export=ALL,MODEL=qlearning cluster/04b_bayesian_cpu.slurm
#   sbatch --export=ALL,MODEL=wmrl cluster/04b_bayesian_cpu.slurm
#   sbatch --export=ALL,MODEL=wmrl_m5 cluster/04b_bayesian_cpu.slurm
#   sbatch --export=ALL,MODEL=wmrl_m6a cluster/04b_bayesian_cpu.slurm
#
# M6b requires 36h wall-time due to stick-breaking decode in the 8-parameter
# model. Submit M6b with an explicit time override:
#   sbatch --time=36:00:00 --export=ALL,MODEL=wmrl_m6b \
#     cluster/04b_bayesian_cpu.slurm
#
# M6b subscale variant (4 covariates x 8 params = 32 beta sites). Needs 12h+
# walltime and requires data/processed/summary_participant_metrics.csv to supply the
# IES-R + LEC columns.
#   sbatch --time=12:00:00 --mem=48G \
#     --export=ALL,MODEL=wmrl_m6b,SUBSCALE=1 cluster/04b_bayesian_cpu.slurm
#
# Outputs saved to:
#   - models/bayesian/${MODEL}_individual_fits.csv
#   - models/bayesian/${MODEL}_posterior.nc
#   - models/bayesian/${MODEL}_shrinkage_report.md
#   - models/bayesian/${MODEL}_ppc_results.csv
# =============================================================================

#SBATCH --job-name=bayesian_cpu
#SBATCH --output=logs/bayesian_cpu_%j.out
#SBATCH --error=logs/bayesian_cpu_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

# --- Model selection + MCMC knobs (override via --export=ALL,KEY=VAL,...) ---
MODEL="${MODEL:-wmrl_m3}"
CHAINS="${CHAINS:-4}"
WARMUP="${WARMUP:-1000}"
SAMPLES="${SAMPLES:-2000}"
SEED="${SEED:-42}"
MAX_TREE_DEPTH="${MAX_TREE_DEPTH:-8}"
SUBSCALE="${SUBSCALE:-0}"  # 1 enables --subscale flag (M6b subscale variant)

# --- Environment Setup ---
module load miniforge3

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

# Install autopush trap (must come BEFORE any failure-prone command).
# autopush.sh registers an EXIT trap that fires on success or failure,
# so SLURM .err/.out logs land in git regardless of outcome — see
# cluster/autopush.sh for rationale.
[ -f cluster/autopush.sh ] && source cluster/autopush.sh


echo "============================================================"
echo "Stage 04b — Choice-Only Hierarchical Bayesian Fit (CPU)"
echo "============================================================"
echo "Job ID:         ${SLURM_JOB_ID:-local}"
echo "Node:           ${SLURMD_NODENAME:-$(hostname)}"
echo "Project:        $PROJECT_ROOT"
echo "Model:          $MODEL"
echo "Chains:         $CHAINS"
echo "Warmup:         $WARMUP"
echo "Samples:        $SAMPLES"
echo "Seed:           $SEED"
echo "Max tree depth: $MAX_TREE_DEPTH"
echo "Subscale:       $SUBSCALE"
echo "Start time:     $(date)"
echo "============================================================"

# Activate conda environment
# Try ds_env first (CPU Bayesian), then rlwm_gpu as fallback
if conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu by name"
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/fc37/$USER/"
elif conda activate /scratch/${PROJECT:-fc37}/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/${PROJECT:-fc37}/$USER/"
else
    echo "ERROR: Failed to activate rlwm_gpu — see cluster/00_setup_env_gpu.sh"
    exit 1
fi

# Re-register editable rlwm install (idempotent; protects against stale
# egg-link after folder moves / fresh compute-node caches). Mirrors
# cluster/00_preflight.slurm step [2/5].
pip install -e . --quiet --no-deps 2>/dev/null || \
    echo "WARNING: pip install -e . --no-deps failed; 'import rlwm' may fail downstream"

# Verify jax is importable
python -c "import jax; print(f'JAX version: {jax.__version__}, platforms: {jax.devices()}')" || {
    echo "ERROR: JAX import failed after conda activation"
    exit 1
}
echo ""

# --- JAX compilation cache ---
export JAX_PLATFORMS=cpu
export NUMPYRO_HOST_DEVICE_COUNT=4
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_bayesian"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"

# --- Create output directories ---
mkdir -p logs
mkdir -p models/bayesian/21_baseline

# Clear any stale per-model artefacts from a previous run (different cohort N,
# aborted fit, etc.). Without this, a fit failure mid-rerun would leave the
# previous run's posterior in place and stage 05 would silently audit it.
echo "[$(date)] Clearing stale ${MODEL} artefacts from models/bayesian/21_baseline/"
for stale in "models/bayesian/21_baseline/${MODEL}_individual_fits.csv" \
             "models/bayesian/21_baseline/${MODEL}_posterior.nc" \
             "models/bayesian/21_baseline/${MODEL}_shrinkage_report.md" \
             "models/bayesian/21_baseline/${MODEL}_ppc_results.csv"; do
    if [[ -f "$stale" ]]; then
        echo "  removing $stale"
        rm -f "$stale"
    fi
done

# --- Verify required input files ---
if [ "$SUBSCALE" = "1" ]; then
    echo ""
    echo "Checking subscale-required input files..."
    for f in data/processed/task_trials_long.csv data/processed/summary_participant_metrics.csv; do
        if [ -f "$f" ]; then
            echo "  OK: $f ($(du -h "$f" | cut -f1))"
        else
            echo "  ERROR: $f is MISSING — required for subscale L2 regression"
            exit 1
        fi
    done
fi

# --- Build fit_baseline.py command (Phase-21 pipeline entry) ---
# IMPORTANT: This invokes `fit_baseline.py` (NOT `fit_bayesian.py`). The
# pipeline entry forces --output-subdir=21_baseline so artefacts land in
# models/bayesian/21_baseline/ where stage 05 baseline_audit reads them,
# AND adds a load-bearing expected.exists() check that converts silent
# convergence-gate failures into a non-zero SLURM exit. fit_bayesian.py is
# the ad-hoc CLI and writes to bare models/bayesian/, which would silently
# orphan the afterok chain.
#
# fit_baseline.py does NOT accept --subscale (subscale runs through
# cluster/04c_level2.slurm, not 04b). If SUBSCALE=1 is set it will be
# logged and ignored here.
CMD=(python scripts/04_model_fitting/b_bayesian/fit_baseline.py
     --model "$MODEL"
     --data data/processed/task_trials_long.csv
     --chains "$CHAINS"
     --warmup "$WARMUP"
     --samples "$SAMPLES"
     --seed "$SEED"
     --max-tree-depth "$MAX_TREE_DEPTH")

if [ "$SUBSCALE" = "1" ]; then
    echo "WARNING: SUBSCALE=1 ignored under fit_baseline.py."
    echo "         Subscale fits run through cluster/04c_level2.slurm."
fi

# --- Run hierarchical fit ---
echo ""
echo "[$(date)] Starting ${MODEL} hierarchical Bayesian fit${SUBSCALE:+ (subscale=$SUBSCALE)}"
echo "  Command: ${CMD[*]}"

"${CMD[@]}"

EXIT_CODE=$?

echo ""
echo "[$(date)] ${MODEL} hierarchical fit finished (exit code: $EXIT_CODE)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Expected outputs (under models/bayesian/21_baseline/):"
    MISSING_ARTIFACTS=0
    for f in "models/bayesian/21_baseline/${MODEL}_individual_fits.csv" \
             "models/bayesian/21_baseline/${MODEL}_posterior.nc" \
             "models/bayesian/21_baseline/${MODEL}_shrinkage_report.md" \
             "models/bayesian/21_baseline/${MODEL}_ppc_results.csv"; do
        if [ -f "$f" ]; then
            echo "  [OK] $f ($(du -h "$f" | cut -f1))"
        else
            echo "  [MISSING] $f"
            MISSING_ARTIFACTS=$((MISSING_ARTIFACTS + 1))
        fi
    done
    if [ $MISSING_ARTIFACTS -gt 0 ]; then
        echo ""
        echo "*** ERROR: ${MODEL} fit returned exit=0 but $MISSING_ARTIFACTS expected artefact(s) MISSING ***"
        echo "*** Treating as failure for afterok chain integrity (stage 05 reads from 21_baseline/). ***"
        EXIT_CODE=1
    fi
else
    echo "ERROR: fit_baseline.py exited with code $EXIT_CODE"
fi

# Auto-push logs and results when job completes (autopush exit code is
# isolated from fit exit code so a transient network failure does not
# mark the SLURM job as failed and orphan the afterok chain).
source cluster/autopush.sh

exit $EXIT_CODE
