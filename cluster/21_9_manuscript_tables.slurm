#!/bin/bash
# =============================================================================
# SLURM Batch Script: Phase 21 Step 21.9 — Manuscript Tables + Forest Plots
# =============================================================================
# Runs scripts/21_manuscript_tables.py to consolidate Phase 21 pipeline
# outputs (steps 21.1 through 21.8) into publication-ready Tables 1/2/3
# (.csv + .md + .tex) and Figure 1 forest plots.
#
# This is the FINAL step of the Phase 21 pipeline — it depends on step 21.8
# (model averaging) via --dependency=afterok in the master orchestrator
# (cluster/21_submit_pipeline.sh). The script is invoked with
# --no-paper-edit because paper.qmd lives in the repo and is edited locally
# during plan execution (reviewed via git diff), not on the cluster.
#
# Pure-ArviZ / pandas / matplotlib (no JAX, no MCMC, no GPU). The 30 min /
# 16 G / 2-CPU allocation is plenty for reading the upstream artefacts and
# writing tables + forest PNGs.
#
# Inputs:
#   - output/bayesian/21_baseline/loo_stacking_results.csv (step 21.5)
#   - output/bayesian/21_baseline/rfx_bms_pxp.csv          (step 21.5)
#   - output/bayesian/21_baseline/winners.txt              (step 21.5)
#   - output/bayesian/21_l2/scale_audit_report.md          (step 21.7)
#   - output/bayesian/21_l2/{winner}_beta_hdi_table.csv    (step 21.7)
#   - output/bayesian/21_l2/averaged_scale_effects.csv     (step 21.8)
#   - output/bayesian/21_l2/{winner}_posterior.nc          (step 21.6)
#   - output/bayesian/wmrl_m6b_subscale_posterior.nc       (step 21.8 fire-and-forget; OPTIONAL)
#
# Outputs:
#   - output/bayesian/21_tables/table1_loo_stacking.{csv,md,tex}
#   - output/bayesian/21_tables/table2_rfx_bms.{csv,md,tex}
#   - output/bayesian/21_tables/table3_winner_betas.{csv,md,tex}
#   - output/bayesian/21_tables/null_result_summary.md   (NULL_RESULT branch only)
#   - figures/21_bayesian/forest_{winner}.png            (PROCEED branch only)
#
# Usage:
#   sbatch cluster/21_9_manuscript_tables.slurm
#
# Usage (chained from step 21.8 afterok):
#   sbatch --dependency=afterok:$AVG_JID cluster/21_9_manuscript_tables.slurm
# =============================================================================

#SBATCH --job-name=bayes_21_9_tables
#SBATCH --output=logs/bayesian_21_9_%j.out
#SBATCH --error=logs/bayesian_21_9_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=comp

# --- Env-var overrides (override via --export=KEY=VAL,...) ---
BASELINE_DIR="${BASELINE_DIR:-output/bayesian/21_baseline/}"
L2_DIR="${L2_DIR:-output/bayesian/21_l2/}"
FIGURES_DIR="${FIGURES_DIR:-figures/21_bayesian/}"
TABLES_DIR="${TABLES_DIR:-output/bayesian/21_tables/}"
SUBSCALE_NC="${SUBSCALE_NC:-output/bayesian/wmrl_m6b_subscale_posterior.nc}"

# --- Environment Setup (matches cluster/21_8_model_averaging.slurm pattern) ---
module load miniforge3

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Phase 21 Step 21.9 — Manuscript Tables + Forest Plots"
echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          ${SLURMD_NODENAME:-$(hostname)}"
echo "Project:       $PROJECT_ROOT"
echo "Baseline dir:  $BASELINE_DIR"
echo "L2 dir:        $L2_DIR"
echo "Figures dir:   $FIGURES_DIR"
echo "Tables dir:    $TABLES_DIR"
echo "Subscale NC:   $SUBSCALE_NC"
echo "Start time:    $(date)"
echo "============================================================"

# Activate conda environment — ds_env first, /scratch ladder fallback.
# No JAX required for this step (pure ArviZ / pandas / matplotlib).
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

# Verify ArviZ + pandas + matplotlib importable — guard against env drift.
python -c "import arviz as az; import pandas as pd; import matplotlib; print(f'ArviZ: {az.__version__}, pandas: {pd.__version__}, matplotlib: {matplotlib.__version__}')" || {
    echo "ERROR: ArviZ / pandas / matplotlib import failed after conda activation"
    exit 1
}
echo ""

# --- Create output directories ---
mkdir -p figures/21_bayesian
mkdir -p output/bayesian/21_tables
mkdir -p logs

# --- Run Step 21.9 manuscript tables ---
echo ""
echo "[$(date)] Starting Step 21.9 manuscript tables"

python scripts/bayesian_pipeline/21_manuscript_tables.py \
    --baseline-dir "$BASELINE_DIR" \
    --l2-dir "$L2_DIR" \
    --figures-dir "$FIGURES_DIR" \
    --tables-dir "$TABLES_DIR" \
    --subscale-nc "$SUBSCALE_NC" \
    --no-paper-edit

EXIT_CODE=$?

echo ""
echo "[$(date)] Step 21.9 manuscript tables finished (exit code: $EXIT_CODE)"

# =============================================================================
# Report expected outputs
# =============================================================================
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "Expected outputs under $TABLES_DIR:"
    for f in "${TABLES_DIR%/}/table1_loo_stacking.csv" \
             "${TABLES_DIR%/}/table1_loo_stacking.md" \
             "${TABLES_DIR%/}/table1_loo_stacking.tex" \
             "${TABLES_DIR%/}/table2_rfx_bms.csv" \
             "${TABLES_DIR%/}/table2_rfx_bms.md" \
             "${TABLES_DIR%/}/table2_rfx_bms.tex" \
             "${TABLES_DIR%/}/table3_winner_betas.csv" \
             "${TABLES_DIR%/}/table3_winner_betas.md" \
             "${TABLES_DIR%/}/table3_winner_betas.tex" \
             "${TABLES_DIR%/}/null_result_summary.md"; do
        if [ -f "$f" ]; then
            echo "  [OK] $f ($(du -h "$f" | cut -f1))"
        fi
    done
    echo ""
    echo "Expected figures under $FIGURES_DIR:"
    for f in "${FIGURES_DIR%/}"/forest_*.png; do
        if [ -f "$f" ]; then
            echo "  [OK] $f ($(du -h "$f" | cut -f1))"
        fi
    done
else
    # Exit-propagation here is the LAST link in the Phase 21 afterok chain.
    # Step 21.9 is the manuscript-build terminus; if it fails, the master
    # orchestrator (cluster/21_submit_pipeline.sh) reports the failure and
    # the pipeline ends with no manuscript-ready artefacts.
    echo "[ERROR] Manuscript table generation failed with exit code $EXIT_CODE"
    echo "[ERROR] Typical causes:"
    echo "[ERROR]   1. loo_stacking_results.csv missing (check step 21.5)"
    echo "[ERROR]   2. winners.txt missing (check step 21.5)"
    echo "[ERROR]   3. {winner}_beta_hdi_table.csv missing (check step 21.7)"
    echo "[ERROR]   4. Forest-plot subprocess failed (check step 21.6 posteriors)"
    echo "[ERROR] See script stderr output above for root cause."
fi

# Auto-push logs and results when job completes
source cluster/autopush.sh

# Load-bearing — terminus of Phase 21 afterok chain. Returns the script's
# exit code so the master orchestrator can report the final outcome.
exit $EXIT_CODE
