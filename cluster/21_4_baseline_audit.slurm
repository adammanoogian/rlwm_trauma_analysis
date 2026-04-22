#!/bin/bash
# =============================================================================
# SLURM Batch Script: Phase 21 Step 21.4 — Baseline Convergence + PPC Audit
# =============================================================================
# Runs scripts/05_post_fitting_checks/01_baseline_audit.py over all 6 baseline posteriors produced
# by step 21.3 (output/bayesian/21_baseline/{model}_posterior.nc) and writes
# convergence_table.csv + convergence_report.md to the same directory.
#
# Enforces the Baribault & Collins (2023, DOI 10.1037/met0000554) gate as
# a HARD PIPELINE BLOCK between step 21.3 and step 21.5:
#   R-hat <= 1.05 AND ESS_bulk >= 400 AND divergences == 0 AND BFMI >= 0.2
# Fewer than 2 models passing -> exit 1 -> SLURM --dependency=afterok chains
# block step 21.5 (PSIS-LOO + stacking + RFX-BMS).
#
# This is a pure-ArviZ / pandas audit — no JAX, no MCMC, no GPU. The 30 min
# / 8 G / 2-CPU allocation is plenty for reading 6 NetCDFs and computing
# summaries.
#
# Usage:
#   sbatch cluster/21_4_baseline_audit.slurm
#
# Usage (custom thresholds):
#   sbatch --export=RHAT=1.01,ESS=500 cluster/21_4_baseline_audit.slurm
#
# Outputs:
#   - output/bayesian/21_baseline/convergence_table.csv
#   - output/bayesian/21_baseline/convergence_report.md
#
# Exit codes:
#   0 -> >= 2 models passed; step 21.5 may proceed
#   1 -> < 2 models passed; [PIPELINE BLOCK] — fix models and re-run 21.3
# =============================================================================

#SBATCH --job-name=bayes_21_4_audit
#SBATCH --output=logs/bayesian_21_4_audit_%j.out
#SBATCH --error=logs/bayesian_21_4_audit_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --partition=comp

# --- Gate thresholds (override via --export=KEY=VAL,...) ---
RHAT="${RHAT:-1.05}"
ESS="${ESS:-400}"
BFMI="${BFMI:-0.2}"

# --- Environment Setup (matches 21_3_fit_baseline.slurm pattern) ---
module load miniforge3

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Phase 21 Step 21.4 — Baseline Convergence Audit"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Project: $PROJECT_ROOT"
echo "R-hat threshold: <= $RHAT"
echo "ESS threshold: >= $ESS"
echo "BFMI threshold: >= $BFMI"
echo "Start time: $(date)"
echo "============================================================"

# Activate conda environment — ds_env first, /scratch ladder fallback.
# No JAX required for this step (pure ArviZ / pandas), but ds_env already
# has arviz pinned to 0.22+ which is what the audit expects.
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

# Verify arviz is importable
python -c "import arviz as az; print(f'ArviZ version: {az.__version__}')" || {
    echo "ERROR: ArviZ import failed after conda activation"
    exit 1
}
echo ""

# --- Create output directories ---
mkdir -p output/bayesian/21_baseline
mkdir -p logs

# --- Run Step 21.4 convergence audit ---
echo ""
echo "[$(date)] Starting Step 21.4 baseline convergence audit"

python scripts/05_post_fitting_checks/01_baseline_audit.py \
    --baseline-dir output/bayesian/21_baseline/ \
    --output-dir output/bayesian/21_baseline/ \
    --rhat-threshold "$RHAT" \
    --ess-threshold "$ESS" \
    --bfmi-threshold "$BFMI"

EXIT_CODE=$?

echo ""
echo "[$(date)] Step 21.4 audit finished (exit code: $EXIT_CODE)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Audit verdict: PASS (>= 2 models converged, step 21.5 may proceed)"
    echo "Expected outputs:"
    for f in output/bayesian/21_baseline/convergence_table.csv \
             output/bayesian/21_baseline/convergence_report.md; do
        if [ -f "$f" ]; then
            echo "  [OK] $f ($(du -h "$f" | cut -f1))"
        else
            echo "  [MISSING] $f (this should not happen — script crashed pre-write?)"
        fi
    done
else
    # Exit-propagation is critical: the master pipeline orchestrator
    # (plan 21-10) chains step 21.5 via `--dependency=afterok:$AUDIT_JOBID`,
    # so a non-zero exit here naturally blocks the chain.
    echo "[PIPELINE BLOCK] Convergence gate failed — step 21.5 cannot proceed"
    echo "Inspect output/bayesian/21_baseline/convergence_report.md for"
    echo "per-model diagnostics. Typical fixes: re-run 21.3 with increased"
    echo "warmup/samples, reparameterise the failing model, or explicitly"
    echo "drop it from the pipeline with documented rationale."
fi

# Auto-push logs and results when job completes
source cluster/autopush.sh

# Load-bearing — master orchestrator chains step 21.5 via
# --dependency=afterok. Without this propagation, a gate failure would
# silently continue to stacking.
exit $EXIT_CODE
