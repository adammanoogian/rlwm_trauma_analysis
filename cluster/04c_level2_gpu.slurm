#!/bin/bash
# =============================================================================
# SLURM Batch Script: Stage 04c — Winner Refit WITH L2 Scales (GPU, Template C)
# =============================================================================
# GPU multi-GPU pmap variant of cluster/04c_level2.slurm.
#
# Template C (4 GPUs + chain_method=parallel via pmap) per
# docs/CLUSTER_GPU_LESSONS.md §6. M6b subscale (32 betas, 8-param
# stick-breaking) is the primary motivation — CPU worst-case ~8-12h; GPU
# expected ~3-4h per the M6b proof at job 54894258
# (logs/bayesian_mgpu_54894258.out, 2026-04-18).
#
# CPU sibling cluster/04c_level2.slurm is preserved unchanged and remains
# dispatchable via USE_CPU=1 in the master orchestrator
# (cluster/submit_all.sh); that orchestrator also propagates L2_FIT_SCRIPT
# through cluster/21_dispatch_l2_winners.sh so GPU vs CPU routing happens
# automatically for all winners (M1/M2 copy-through, M3/M5/M6a 2-cov,
# M6b subscale).
#
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
# Delegates to scripts/04_model_fitting/c_level2/fit_with_l2.py which
# routes outputs through save_results(..., output_subdir="21_l2") so
# Phase 16 posteriors at models/bayesian/{model}_posterior.nc and 21.3
# baselines at models/bayesian/21_baseline/{model}_posterior.nc are NEVER
# overwritten.
#
# Usage (single model, default wmrl_m6b):
#   sbatch cluster/04c_level2_gpu.slurm
#
# Usage (specific model; must be in winners.txt):
#   sbatch --export=MODEL=wmrl_m3 cluster/04c_level2_gpu.slurm
#
# Optional GPU type override (default L40S; pins JIT cache to one arch per
# CLUSTER_GPU_LESSONS.md §5):
#   sbatch --export=GPU_TYPE=a100 cluster/04c_level2_gpu.slurm
#
# Outputs saved to:
#   - models/bayesian/21_l2/{MODEL}_individual_fits.csv
#   - models/bayesian/21_l2/{MODEL}_posterior.nc
#   - models/bayesian/21_l2/{MODEL}_shrinkage_report.md
#   - models/bayesian/21_l2/{MODEL}_ppc_results.csv
#
# Time ceiling: 8h absorbs all three tiers on GPU:
#   - M6b subscale (32 betas, 8-param stick-breaking): ~3-4h expected
#     (GPU Template C); worst-case auto-bump fits inside 8h.
#   - M3/M5/M6a 2-cov L2 (2 betas, extra dim near-decoupled): ~15-30 min.
#   - M1/M2 copy-through (shutil.copy2): ~1 second.
# One cap fits all tiers; override for sensitivity sweeps.
# =============================================================================

#SBATCH --job-name=bayes_04c_l2_gpu
#SBATCH --output=logs/bayesian_04c_l2_gpu_%j.out
#SBATCH --error=logs/bayesian_04c_l2_gpu_%j.err
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
# GPU-type selection via CLI only (shell-var expansion does not work in #SBATCH directives).
# Override on command line if needed:  sbatch --constraint=l40s cluster/04c_level2_gpu.slurm

# --- Step 21.6 fit parameters (override via --export=KEY=VAL,...) ---
MODEL="${MODEL:-wmrl_m6b}"
WARMUP="${WARMUP:-1000}"
SAMPLES="${SAMPLES:-2000}"
CHAINS="${CHAINS:-4}"
SEED="${SEED:-42}"
MAX_TREE_DEPTH="${MAX_TREE_DEPTH:-10}"
WINNERS_FILE="${WINNERS_FILE:-models/bayesian/21_baseline/winners.txt}"

# --- Environment Setup ---
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Stage 04c (GPU, Template C) — Winner Refit WITH L2 Scales"
echo "============================================================"
echo "Job ID:         ${SLURM_JOB_ID:-local}"
echo "Node:           ${SLURMD_NODENAME:-$(hostname)}"
echo "Partition:      ${SLURM_JOB_PARTITION:-gpu}"
echo "GPU(s):         ${CUDA_VISIBLE_DEVICES:-none}"
echo "Project:        $PROJECT_ROOT"
echo "Model:          $MODEL"
echo "Warmup:         $WARMUP"
echo "Samples:        $SAMPLES"
echo "Chains:         $CHAINS"
echo "Seed:           $SEED"
echo "Max tree depth: $MAX_TREE_DEPTH"
echo "Winners file:   $WINNERS_FILE"
echo "Start time:     $(date)"
echo "============================================================"

# Activate rlwm_gpu conda environment. NO ds_env fallback — GPU Template C
# requires CUDA; ds_env would silently degrade chain_method to vectorized.
if conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu by name"
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/fc37/$USER/"
elif conda activate /scratch/${PROJECT:-fc37}/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/${PROJECT:-fc37}/$USER/"
else
    echo "ERROR: Failed to activate rlwm_gpu — GPU pipeline cannot run on ds_env (CPU-only)"
    echo "       Create it first with: bash cluster/00_setup_env_gpu.sh"
    echo "       Or use USE_CPU=1 with cluster/submit_all.sh for the CPU path."
    exit 1
fi

# Verify GPU allocation matches chain count. If n_gpu < CHAINS, NumPyro's
# _select_chain_method falls back to 'vectorized' which defeats Template C.
CHAINS_NUM=$CHAINS
python -c "
import jax
devices = jax.devices()
n_gpu = sum(1 for d in devices if d.platform == 'gpu')
print(f'JAX {jax.__version__} devices: {devices}')
print(f'GPU count: {n_gpu}')
if n_gpu < ${CHAINS_NUM}:
    print(f'WARNING: {n_gpu} GPU(s) < ${CHAINS_NUM} chains. chain_method will degrade to vectorized.')
" || { echo 'ERROR: JAX import failed'; exit 1; }
echo ""

# =============================================================================
# Cache + env exports — encode CLUSTER_GPU_LESSONS.md operational invariants
# =============================================================================
export NUMPYRO_HOST_DEVICE_COUNT="$CHAINS"
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_04c_level2_gpu_${MODEL}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export PYTHONUNBUFFERED=1
mkdir -p "$JAX_COMPILATION_CACHE_DIR" || {
    echo 'WARNING: cache dir create failed'
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
echo "[$(date)] Starting Step 21.6 L2 refit for $MODEL (Template C, 4 GPUs + pmap)"

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

# Expected-artifacts audit — mirrors cluster/04c_level2.slurm.
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
# chain. Load-bearing — DO NOT replace with `exit 0`.
exit $EXIT_CODE
