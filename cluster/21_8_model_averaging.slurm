#!/bin/bash
# =============================================================================
# SLURM Batch Script: Phase 21 Step 21.8 — Model Averaging + Subscale Arm
# =============================================================================
# Runs scripts/21_model_averaging.py over the winner L2 posteriors
# produced by step 21.6 (output/bayesian/21_l2/{winner}_posterior.nc),
# using stacking weights from step 21.5
# (output/bayesian/21_baseline/loo_stacking_results.csv) to build a
# Yao et al. (2018) weighted posterior mixture for each canonical
# (covariate_family, target_parameter) key. Handles mixed beta
# cardinality across winners via `_total` suffix normalisation so
# M6b subscale `beta_lec_total_kappa` and M3/M5 `beta_lec_kappa`
# collapse to the same key.
#
# Three short-circuit paths are internal to the Python script:
#   - NULL_RESULT audit header -> writes averaging_skipped.md + exit 0.
#   - Single winner -> writes single_winner_mode.md + exit 0.
#   - Multi-winner -> averaged_scale_effects.csv + averaging_summary.md.
#
# Optional M6b-subscale exploratory arm
# -------------------------------------
# When M6b is in the winner set, the Python script writes a marker
# file `output/bayesian/21_l2/launch_subscale.flag`. This SLURM wrapper
# reads the marker and fires `cluster/13_bayesian_m6b_subscale.slurm`
# FIRE-AND-FORGET (sbatch returns immediately with the JID; this job
# continues through autopush and exits without waiting for the 12h
# subscale fit). The subscale posterior writes to the Phase-16
# canonical path `output/bayesian/wmrl_m6b_subscale_posterior.nc` —
# plan 21-10 reads from there if the NetCDF appeared by manuscript-
# build time. We intentionally do NOT patch the subscale SLURM to
# accept OUTPUT_SUBDIR (option (a) in plan 21-09 Task 2) so the
# Phase-16 canonical path stays stable across downstream consumers.
#
# Set LAUNCH_SUBSCALE=0 in --export to globally disable the arm even
# when M6b is a winner (useful for dry runs that don't want to burn
# a 12h cluster slot).
#
# This step is pure-ArviZ / pandas / numpy (no JAX, no MCMC, no GPU).
# The 1h / 32G / 2-CPU allocation is plenty for reading up to 6
# NetCDFs and building 8k-sample mixtures.
#
# Usage:
#   sbatch cluster/21_8_model_averaging.slurm
#
# Usage (chained from step 21.7 afterok):
#   sbatch --dependency=afterok:$AUDIT2_JID cluster/21_8_model_averaging.slurm
#
# Usage (subscale arm disabled even if M6b wins):
#   sbatch --export=LAUNCH_SUBSCALE=0 cluster/21_8_model_averaging.slurm
#
# Outputs under output/bayesian/21_l2/:
#   - averaged_scale_effects.csv      (multi-winner primary artefact)
#   - averaging_summary.md            (multi-winner narrative)
#   - averaging_skipped.md            (NULL_RESULT short-circuit only)
#   - single_winner_mode.md           (single-winner short-circuit only)
#   - launch_subscale.flag            (cleared by this wrapper after use)
# =============================================================================

#SBATCH --job-name=bayes_21_8_averaging
#SBATCH --output=logs/bayesian_21_8_%j.out
#SBATCH --error=logs/bayesian_21_8_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --partition=comp

# --- Env-var overrides (override via --export=KEY=VAL,...) ---
L2_DIR="${L2_DIR:-output/bayesian/21_l2/}"
STACKING_RESULTS="${STACKING_RESULTS:-output/bayesian/21_baseline/loo_stacking_results.csv}"
WINNERS_FILE="${WINNERS_FILE:-output/bayesian/21_baseline/winners.txt}"
AUDIT_REPORT="${AUDIT_REPORT:-output/bayesian/21_l2/scale_audit_report.md}"
OUTPUT_DIR="${OUTPUT_DIR:-output/bayesian/21_l2/}"
LAUNCH_SUBSCALE="${LAUNCH_SUBSCALE:-1}"

# --- Environment Setup (matches cluster/21_7_scale_audit.slurm pattern) ---
module load miniforge3

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Phase 21 Step 21.8 — Model Averaging + Optional Subscale Arm"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Project: $PROJECT_ROOT"
echo "L2 dir: $L2_DIR"
echo "Stacking results: $STACKING_RESULTS"
echo "Winners file: $WINNERS_FILE"
echo "Audit report: $AUDIT_REPORT"
echo "Output dir: $OUTPUT_DIR"
echo "Launch subscale arm: $LAUNCH_SUBSCALE"
echo "Start time: $(date)"
echo "============================================================"

# Activate conda environment — ds_env first, /scratch ladder fallback.
# No JAX required for this step (pure ArviZ / pandas / numpy).
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

# Verify arviz + pandas importable — guard against env drift.
python -c "import arviz as az; import pandas as pd; import numpy as np; print(f'ArviZ: {az.__version__}, pandas: {pd.__version__}, numpy: {np.__version__}')" || {
    echo "ERROR: ArviZ / pandas / numpy import failed after conda activation"
    exit 1
}
echo ""

# --- Create output directories ---
mkdir -p output/bayesian/21_l2
mkdir -p logs

# --- Run Step 21.8 model averaging ---
echo ""
echo "[$(date)] Starting Step 21.8 model averaging"

python scripts/bayesian_pipeline/21_model_averaging.py \
    --l2-dir "$L2_DIR" \
    --stacking-results "$STACKING_RESULTS" \
    --winners-file "$WINNERS_FILE" \
    --audit-report "$AUDIT_REPORT" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "[$(date)] Step 21.8 averaging finished (exit code: $EXIT_CODE)"

# =============================================================================
# Optional M6b-subscale exploratory arm (FIRE-AND-FORGET)
# =============================================================================
# The Python script writes launch_subscale.flag when M6b is in the
# winner set and --launch-subscale-arm is on (default True). We read
# the flag here and submit cluster/13_bayesian_m6b_subscale.slurm in
# the background — sbatch --parsable returns the JID immediately and
# this wrapper DOES NOT block waiting for the subscale fit (which
# takes ~12h). Plan 21-10 reads the subscale posterior from the
# Phase-16 canonical path
# output/bayesian/wmrl_m6b_subscale_posterior.nc
# IF it appeared by manuscript-build time. The flag is removed after
# submission so re-runs of 21.8 don't double-submit.
# =============================================================================
SUBSCALE_FLAG="${OUTPUT_DIR%/}/launch_subscale.flag"
if [ -f "$SUBSCALE_FLAG" ] && [ "$LAUNCH_SUBSCALE" = "1" ]; then
    echo "[SUBSCALE ARM] Marker file $SUBSCALE_FLAG present; M6b is a winner."
    echo "[SUBSCALE ARM] Submitting cluster/13_bayesian_m6b_subscale.slurm"
    echo "[SUBSCALE ARM] in fire-and-forget mode (main chain does NOT wait)."
    SUBSCALE_JID=$(sbatch --parsable \
        --export=ALL,MODEL=wmrl_m6b,OUTPUT_SUBDIR=21_l2,SUBSCALE_OUTPUT_NAME=m6b_subscale \
        cluster/13_bayesian_m6b_subscale.slurm)
    if [ -n "$SUBSCALE_JID" ]; then
        echo "[SUBSCALE ARM] Submitted as job $SUBSCALE_JID"
        echo "[SUBSCALE ARM] Subscale posterior expected at:"
        echo "[SUBSCALE ARM]   output/bayesian/wmrl_m6b_subscale_posterior.nc"
        echo "[SUBSCALE ARM] (Phase-16 canonical path — plan 21-10 reads this location)"
        rm -f "$SUBSCALE_FLAG"
        echo "[SUBSCALE ARM] Cleared marker $SUBSCALE_FLAG"
    else
        echo "[SUBSCALE ARM] WARNING: sbatch returned empty JID — subscale NOT submitted"
        echo "[SUBSCALE ARM] Marker $SUBSCALE_FLAG retained for manual retry"
    fi
elif [ -f "$SUBSCALE_FLAG" ]; then
    echo "[SUBSCALE ARM] Marker $SUBSCALE_FLAG present but LAUNCH_SUBSCALE=0"
    echo "[SUBSCALE ARM] Subscale arm NOT submitted (user override)."
    echo "[SUBSCALE ARM] To submit manually later:"
    echo "[SUBSCALE ARM]   sbatch cluster/13_bayesian_m6b_subscale.slurm"
else
    echo "[SUBSCALE ARM] No launch_subscale.flag present — M6b not a winner,"
    echo "[SUBSCALE ARM] or user disabled via --no-launch-subscale-arm."
fi

# =============================================================================
# Report expected outputs
# =============================================================================
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Expected outputs under $OUTPUT_DIR:"
    for f in "${OUTPUT_DIR%/}/averaged_scale_effects.csv" \
             "${OUTPUT_DIR%/}/averaging_summary.md" \
             "${OUTPUT_DIR%/}/averaging_skipped.md" \
             "${OUTPUT_DIR%/}/single_winner_mode.md"; do
        if [ -f "$f" ]; then
            echo "  [OK] $f ($(du -h "$f" | cut -f1))"
        fi
    done
else
    # Exit-propagation here is critical: the master pipeline orchestrator
    # (plan 21-10) chains step 21.9 via --dependency=afterok:$AVG_JID,
    # so a non-zero exit blocks the chain. This branch fires ONLY for
    # genuine errors (missing inputs, corrupt NetCDFs, import failures)
    # — NULL_RESULT and single-winner short-circuits exit 0 internally.
    echo "[ERROR] Averaging failed with exit code $EXIT_CODE — genuine error, not null/single-winner short-circuit"
    echo "[ERROR] Typical causes:"
    echo "[ERROR]   1. winners.txt missing or malformed (check step 21.5)"
    echo "[ERROR]   2. {winner}_posterior.nc missing or corrupt (check step 21.6)"
    echo "[ERROR]   3. loo_stacking_results.csv missing (check step 21.5)"
    echo "[ERROR] See script stderr output above for root cause."
fi

# Auto-push logs and results when job completes
source cluster/autopush.sh

# Load-bearing — master orchestrator chains step 21.9 via
# --dependency=afterok. Both NULL_RESULT and single-winner short-
# circuits exit 0 so the dependency naturally advances; exit 1
# blocks only on genuine errors. DO NOT replace with `exit 0`.
exit $EXIT_CODE
