#!/bin/bash
# =============================================================================
# SLURM Batch Script: Phase 21 Step 21.5 — LOO + stacking + RFX-BMS
# =============================================================================
# Runs scripts/21_compute_loo_stacking.py over the baseline models that
# passed plan 21-05's convergence gate (pipeline_action == "PROCEED_TO_LOO"
# in output/bayesian/21_baseline/convergence_table.csv) and writes
# loo_stacking_results.csv + rfx_bms_pxp.csv + winner_report.md +
# winners.txt to the same directory.
#
# Primary comparison: LOO + stacking weights (Yao et al. 2018, DOI
# 10.1214/17-BA1091). Secondary: RFX-BMS + protected exceedance
# probability (Stephan et al. 2009; Rigoux et al. 2014).
#
# Pipeline position: runs AFTER `cluster/21_4_baseline_audit.slurm` and
# BEFORE `cluster/21_6_winner_l2_refit.slurm` (plan 21-07). Exit codes:
#   0 -> DOMINANT_SINGLE or TOP_TWO or FORCED; step 21.6 auto-proceeds.
#   1 -> fewer than 2 models eligible; [ABORT] pipeline halt.
#   2 -> INCONCLUSIVE_MULTIPLE; pipeline pauses at USER CHECKPOINT in
#        winner_report.md — user reruns with --force-winners and then
#        resubmits step 21.6 manually, OR re-fits problematic models.
#
# Usage:
#   sbatch cluster/21_5_loo_stacking_bms.slurm
#
# Usage (manual resume after INCONCLUSIVE_MULTIPLE):
#   sbatch --export=FORCE_WINNERS=M3,M6b cluster/21_5_loo_stacking_bms.slurm
#
# This step is pure-ArviZ/NumPy/SciPy (no JAX, no MCMC). The 2h/32G/4-CPU
# allocation is generous — LOO pointwise evaluation over 138 participants
# * 21 blocks * 54 trials is a few minutes — but stays within the `comp`
# partition's standard per-job budget to match the rest of Wave 5.
# =============================================================================

#SBATCH --job-name=bayes_21_5_loo_stacking
#SBATCH --output=logs/bayesian_21_5_%j.out
#SBATCH --error=logs/bayesian_21_5_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

# --- Optional --force-winners override (comma-separated display names) ---
FORCE_WINNERS="${FORCE_WINNERS:-}"

# --- Environment Setup (matches 21_4_baseline_audit.slurm pattern) ---
module load miniforge3

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Phase 21 Step 21.5 — LOO + stacking + RFX-BMS"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Project: $PROJECT_ROOT"
if [ -n "$FORCE_WINNERS" ]; then
    echo "FORCE_WINNERS override: $FORCE_WINNERS"
fi
echo "Start time: $(date)"
echo "============================================================"

# Activate conda environment — ds_env first, /scratch ladder fallback.
# No JAX required (pure ArviZ/NumPy/SciPy); ds_env has arviz 0.22+ which
# ships az.compare(method='stacking') with the expected DataFrame schema.
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

# Verify arviz + scipy importable.
python -c "import arviz as az; import scipy; print(f'ArviZ: {az.__version__}, SciPy: {scipy.__version__}')" || {
    echo "ERROR: arviz/scipy import failed after conda activation"
    exit 1
}
echo ""

# --- Create output directories ---
mkdir -p output/bayesian/21_baseline
mkdir -p logs

# --- Run Step 21.5 LOO + stacking + RFX-BMS ---
echo ""
echo "[$(date)] Starting Step 21.5 LOO + stacking + RFX-BMS"

# Assemble optional --force-winners argument.
EXTRA_ARGS=()
if [ -n "$FORCE_WINNERS" ]; then
    EXTRA_ARGS+=(--force-winners "$FORCE_WINNERS")
fi

python scripts/06_fit_analyses/compute_loo_stacking.py \
    --baseline-dir output/bayesian/21_baseline/ \
    --output-dir output/bayesian/21_baseline/ \
    "${EXTRA_ARGS[@]}"

EXIT_CODE=$?

echo ""
echo "[$(date)] Step 21.5 LOO + stacking finished (exit code: $EXIT_CODE)"
echo "LOO+stacking+BMS exit code: $EXIT_CODE"

case $EXIT_CODE in
  0)
    echo "[PIPELINE] Clear winner — step 21.6 will auto-proceed"
    echo "Expected outputs:"
    for f in output/bayesian/21_baseline/loo_stacking_results.csv \
             output/bayesian/21_baseline/rfx_bms_pxp.csv \
             output/bayesian/21_baseline/winner_report.md \
             output/bayesian/21_baseline/winners.txt; do
        if [ -f "$f" ]; then
            echo "  [OK] $f ($(du -h "$f" | cut -f1))"
        else
            echo "  [MISSING] $f (this should not happen — script crashed pre-write?)"
        fi
    done
    if [ -f output/bayesian/21_baseline/winners.txt ]; then
        echo "Winners: $(cat output/bayesian/21_baseline/winners.txt)"
    fi
    ;;
  1)
    echo "[ABORT] Convergence-eligible pool < 2 models — step 21.5 cannot run"
    echo "Inspect output/bayesian/21_baseline/convergence_report.md and"
    echo "fix the failing models (re-run 21.3 with bigger warmup/samples,"
    echo "reparameterise, or explicitly drop with rationale)."
    ;;
  2)
    echo "[CHECKPOINT] Inconclusive winner set — USER ACTION REQUIRED"
    echo "Review output/bayesian/21_baseline/winner_report.md and rerun"
    echo "with --force-winners. For example, on the cluster:"
    echo "  sbatch --export=FORCE_WINNERS=M3,M6b cluster/21_5_loo_stacking_bms.slurm"
    echo "Then manually submit step 21.6 (it will NOT auto-proceed via"
    echo "--dependency=afterok since this job exits non-zero)."
    ;;
  *)
    echo "[ERROR] Unexpected exit code $EXIT_CODE — inspect log output"
    ;;
esac

# Auto-push logs and results when job completes.
source cluster/autopush.sh

# Load-bearing — master orchestrator chains step 21.6 via
# --dependency=afterok. Exit 0 advances; exit 1 aborts the chain; exit 2
# pauses at the USER CHECKPOINT.
exit $EXIT_CODE
