#!/bin/bash
# =============================================================================
# SLURM Batch Script: Phase 21 Step 21.6 — Winner Refit WITH L2 Scales
# =============================================================================
# Refits ONE winner (MODEL env var) hierarchically with the appropriate
# Level-2 design. Reads models/bayesian/21_baseline/winners.txt from step
# 21.5 (plan 21-06) to validate that MODEL is in the winner set before
# fitting.
#
# Per user-approved Option C (Phase 21 revision):
#   - M1 (qlearning) / M2 (wmrl) — no L2 target; copy baseline posterior.
#   - M3 (wmrl_m3) / M5 (wmrl_m5) / M6a (wmrl_m6a) — 2-covariate L2 via the
#     `covariate_iesr` kwarg from plan 21-11 (LEC total + IES-R total).
#   - M6b (wmrl_m6b) — full 4-covariate subscale (32 betas).
#
# Delegates to scripts/04_model_fitting/c_level2/fit_with_l2.py which routes outputs through
# save_results(..., output_subdir="21_l2") so Phase 16 posteriors at
# models/bayesian/{model}_posterior.nc and 21.3 baselines at
# models/bayesian/21_baseline/{model}_posterior.nc are NEVER overwritten.
#
# Usage (single model, default wmrl_m6b):
#   sbatch cluster/04c_level2.slurm
#
# Usage (specific model; must be in winners.txt):
#   sbatch --export=MODEL=wmrl_m3 cluster/04c_level2.slurm
#
# Usage (from master pipeline orchestrator, chained to stage 04b afterok):
#   # Loop over winners.txt entries on the submit host
#   for m in $(python -c "import sys; \
#       names={'M1':'qlearning','M2':'wmrl','M3':'wmrl_m3','M5':'wmrl_m5',\
#              'M6a':'wmrl_m6a','M6b':'wmrl_m6b'}; \
#       print(' '.join(names[t.strip()] for t in \
#       open('models/bayesian/21_baseline/winners.txt').read().split(',') \
#       if t.strip()))"); do
#       sbatch --dependency=afterok:$LOO_JOBID \
#              --export=MODEL=$m cluster/04c_level2.slurm
#   done
#
# Outputs saved to:
#   - models/bayesian/21_l2/{MODEL}_individual_fits.csv
#   - models/bayesian/21_l2/{MODEL}_posterior.nc
#   - models/bayesian/21_l2/{MODEL}_shrinkage_report.md
#   - models/bayesian/21_l2/{MODEL}_ppc_results.csv
#
# Time ceiling: 12h absorbs all three tiers:
#   - M6b subscale (32 betas, 8-param stick-breaking): ~8-12h worst case.
#   - M3/M5/M6a 2-cov L2 (2 betas, extra dim near-decoupled): ~45-75 min.
#   - M1/M2 copy-through (shutil.copy2): ~1 second.
# One cap fits all tiers; use overrides only for sensitivity sweeps.
# =============================================================================

#SBATCH --job-name=bayes_21_6_l2
#SBATCH --output=logs/bayesian_21_6_%j.out
#SBATCH --error=logs/bayesian_21_6_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

# --- Step 21.6 fit parameters (override via --export=KEY=VAL,...) ---
MODEL="${MODEL:-wmrl_m6b}"
WARMUP="${WARMUP:-1000}"
SAMPLES="${SAMPLES:-2000}"
CHAINS="${CHAINS:-4}"
SEED="${SEED:-42}"
MAX_TREE_DEPTH="${MAX_TREE_DEPTH:-10}"
WINNERS_FILE="${WINNERS_FILE:-models/bayesian/21_baseline/winners.txt}"

# --- Environment Setup (matches cluster/21_3_fit_baseline.slurm pattern) ---
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
echo "Phase 21 Step 21.6 — Winner Refit WITH L2 Scales"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Project: $PROJECT_ROOT"
echo "Model: $MODEL"
echo "Warmup: $WARMUP"
echo "Samples: $SAMPLES"
echo "Chains: $CHAINS"
echo "Seed: $SEED"
echo "Max tree depth: $MAX_TREE_DEPTH"
echo "Winners file: $WINNERS_FILE"
echo "Start time: $(date)"
echo "============================================================"

# Activate conda environment
# Try ds_env first (CPU Bayesian), then rlwm_gpu as fallback
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

# --- JAX compilation cache (unique per model so parallel runs don't collide) ---
export JAX_PLATFORMS=cpu
export NUMPYRO_HOST_DEVICE_COUNT=4
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_21_l2_${MODEL}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" || {
    echo "WARNING: Could not create JAX cache dir, disabling cache"
    unset JAX_COMPILATION_CACHE_DIR
}
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"

# --- Create output directories ---
mkdir -p models/bayesian/21_l2
mkdir -p logs

# --- Verify required input files ---
echo ""
echo "Checking required input files..."
for f in data/processed/task_trials_long.csv data/processed/summary_participant_metrics.csv "$WINNERS_FILE"; do
    if [ -f "$f" ]; then
        echo "  OK: $f ($(du -h "$f" | cut -f1))"
    else
        echo "  ERROR: $f is MISSING — required for step 21.6"
        exit 1
    fi
done

# --- Run Step 21.6 winner L2 refit via fit_with_l2.py orchestrator ---
# The orchestrator:
#   1. Parses $WINNERS_FILE (comma-separated display names) into internal ids.
#   2. Validates $MODEL is in the winner set (exits 1 otherwise).
#   3. Dispatches to copy / 2-cov / subscale branch based on model family.
#   4. Applies save_results convergence gate inside each fit branch.
#   5. Post-fit expected.exists() check converts silent gate-fail into exit 1.
echo ""
echo "[$(date)] Starting Step 21.6 L2 refit for $MODEL"

python scripts/04_model_fitting/c_level2/fit_with_l2.py \
    --model "$MODEL" \
    --data data/processed/task_trials_long.csv \
    --chains "$CHAINS" \
    --warmup "$WARMUP" \
    --samples "$SAMPLES" \
    --seed "$SEED" \
    --max-tree-depth "$MAX_TREE_DEPTH" \
    --output models/ \
    --winners-file "$WINNERS_FILE"

EXIT_CODE=$?

echo ""
echo "[$(date)] Step 21.6 L2 refit for $MODEL finished (exit code: $EXIT_CODE)"

# Expected-artifacts audit — mirrors cluster/21_3_fit_baseline.slurm.
# For the M1/M2 copy path, only the posterior.nc is produced (no rerun of
# individual_fits/shrinkage_report/ppc csvs); those entries will show
# [MISSING] by design and that is acceptable for the copy tier.
if [ $EXIT_CODE -eq 0 ]; then
    echo "Expected outputs under models/bayesian/21_l2/:"
    for f in models/bayesian/21_l2/${MODEL}_posterior.nc \
             models/bayesian/21_l2/${MODEL}_individual_fits.csv \
             models/bayesian/21_l2/${MODEL}_shrinkage_report.md \
             models/bayesian/21_l2/${MODEL}_ppc_results.csv; do
        if [ -f "$f" ]; then
            echo "  [OK] $f ($(du -h "$f" | cut -f1))"
        else
            echo "  [MISSING] $f"
        fi
    done
else
    echo "ERROR: scripts/04_model_fitting/c_level2/fit_with_l2.py exited with code $EXIT_CODE"
    echo "Either convergence gate failed (R-hat, ESS, divergences),"
    echo "winners.txt validation rejected --model, or the copy-source"
    echo "baseline NetCDF was missing. See stderr above for details."
fi

# Auto-push logs and results when job completes
source cluster/autopush.sh

# Propagate exit code to the master orchestrator's --dependency=afterok
# chain (plan 21-10). Load-bearing — DO NOT replace with `exit 0`.
exit $EXIT_CODE
