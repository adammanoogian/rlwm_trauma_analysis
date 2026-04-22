#!/bin/bash
# =============================================================================
# SLURM Batch Script: Phase 21 Step 21.7 — Scale-Fit Audit (L2 betas + FDR-BH)
# =============================================================================
# Runs scripts/21_scale_audit.py over the winners' L2-refit posteriors
# produced by step 21.6 (output/bayesian/21_l2/{winner}_posterior.nc) and
# writes scale_audit_report.md + per-winner beta_hdi_table.csv to the
# same directory.
#
# Enforces the scale-fit gate between step 21.6 and step 21.8 via a YAML-
# front-matter `pipeline_action` header in scale_audit_report.md. Unlike
# step 21.4's hard-gate, step 21.7 uses UNIFIED EXIT-0 SEMANTICS
# (plan-checker Issue #4 resolution):
#
#   exit 0 + YAML `pipeline_action: PROCEED_TO_AVERAGING`
#     -> step 21.8 averages normally.
#   exit 0 + YAML `pipeline_action: NULL_RESULT`
#     -> step 21.8 runs, reads the header, writes `averaging_skipped.md`,
#        exits 0, and step 21.9 reports the null. Null result is valid
#        science, NOT an audit error.
#   exit 1
#     -> genuine audit error (missing winners.txt, corrupt NetCDF,
#        statsmodels import failure). The master orchestrator's
#        `--dependency=afterok:$AUDIT2_JID` for 21.8 blocks correctly.
#
# The YAML header is parsed via awk (single pass, no Python) so the
# decision is visible in the SLURM stdout without re-reading any
# NetCDFs. Both soft outcomes MUST exit 0 so --dependency=afterok
# naturally advances to 21.8 — the soft-skip logic is internal to
# 21_model_averaging.py.
#
# This is a pure-ArviZ / pandas / statsmodels audit — no JAX, no MCMC,
# no GPU. The 30 min / 16 G / 2-CPU allocation is plenty for reading
# <= 6 NetCDFs and computing HDI + FDR-BH summaries.
#
# Usage:
#   sbatch cluster/21_7_scale_audit.slurm
#
# Usage (from master orchestrator, chained to step 21.6 afterok):
#   sbatch --dependency=afterok:$L2_JOBID cluster/21_7_scale_audit.slurm
#
# Outputs:
#   - output/bayesian/21_l2/scale_audit_report.md
#   - output/bayesian/21_l2/{winner}_beta_hdi_table.csv (per winner)
# =============================================================================

#SBATCH --job-name=bayes_21_7_scale_audit
#SBATCH --output=logs/bayesian_21_7_%j.out
#SBATCH --error=logs/bayesian_21_7_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=comp

# --- Gate thresholds (override via --export=KEY=VAL,...) ---
HDI_PROB="${HDI_PROB:-0.95}"
FDR_ALPHA="${FDR_ALPHA:-0.05}"
ESS_DROP_THRESHOLD="${ESS_DROP_THRESHOLD:-0.5}"
L2_DIR="${L2_DIR:-output/bayesian/21_l2/}"
BASELINE_DIR="${BASELINE_DIR:-output/bayesian/21_baseline/}"
WINNERS_FILE="${WINNERS_FILE:-output/bayesian/21_baseline/winners.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-output/bayesian/21_l2/}"

# --- Environment Setup (matches cluster/21_4_baseline_audit.slurm pattern) ---
module load miniforge3

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Phase 21 Step 21.7 — Scale-Fit Audit (L2 betas + FDR-BH)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Project: $PROJECT_ROOT"
echo "L2 dir: $L2_DIR"
echo "Baseline dir: $BASELINE_DIR"
echo "Winners file: $WINNERS_FILE"
echo "Output dir: $OUTPUT_DIR"
echo "HDI prob: $HDI_PROB"
echo "FDR-BH alpha: $FDR_ALPHA"
echo "ESS-drop threshold: $ESS_DROP_THRESHOLD"
echo "Start time: $(date)"
echo "============================================================"

# Activate conda environment — ds_env first, /scratch ladder fallback.
# No JAX required for this step (pure ArviZ / pandas / statsmodels), but
# ds_env already has arviz 0.22+ and statsmodels pinned.
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

# Verify arviz + statsmodels importable — guard against env drift before
# spending a SLURM slot on an import failure halfway through processing.
python -c "import arviz as az; import statsmodels; print(f'ArviZ: {az.__version__}, statsmodels: {statsmodels.__version__}')" || {
    echo "ERROR: ArviZ or statsmodels import failed after conda activation"
    exit 1
}
echo ""

# --- Create output directories ---
mkdir -p output/bayesian/21_l2
mkdir -p logs

# --- Run Step 21.7 scale-fit audit ---
echo ""
echo "[$(date)] Starting Step 21.7 scale-fit audit"

python scripts/05_post_fitting_checks/02_scale_audit.py \
    --l2-dir "$L2_DIR" \
    --baseline-dir "$BASELINE_DIR" \
    --winners-file "$WINNERS_FILE" \
    --hdi-prob "$HDI_PROB" \
    --fdr-alpha "$FDR_ALPHA" \
    --ess-drop-threshold "$ESS_DROP_THRESHOLD" \
    --output-dir "$OUTPUT_DIR"

EXIT_CODE=$?

echo ""
echo "[$(date)] Step 21.7 audit finished (exit code: $EXIT_CODE)"

# =============================================================================
# Unified exit-0 semantics (plan-checker Issue #4 resolution):
# Both PROCEED_TO_AVERAGING and NULL_RESULT exit 0. Distinguish via the
# YAML front-matter `pipeline_action` header of scale_audit_report.md.
# awk parses only between the first and second `---` markers (flag
# toggles state, `next` skips the marker lines themselves).
# =============================================================================
if [ $EXIT_CODE -eq 0 ]; then
    PIPELINE_ACTION=$(awk '/^---$/{flag=!flag;next} flag && /^pipeline_action:/{print $2; exit}' \
        "${OUTPUT_DIR%/}/scale_audit_report.md")
    echo "[AUDIT] pipeline_action = $PIPELINE_ACTION"
    case "$PIPELINE_ACTION" in
        PROCEED_TO_AVERAGING)
            echo "[PROCEED] Step 21.8 averaging will run over L2-refit posteriors"
            ;;
        NULL_RESULT)
            echo "[NULL RESULT] Zero beta sites survived FDR-BH — step 21.8"
            echo "[NULL RESULT] will read YAML header, write averaging_skipped.md,"
            echo "[NULL RESULT] and exit 0; step 21.9 reports the null result"
            ;;
        *)
            echo "[WARN] Unrecognized pipeline_action '$PIPELINE_ACTION' —"
            echo "[WARN] check output/bayesian/21_l2/scale_audit_report.md YAML"
            ;;
    esac

    echo ""
    echo "Expected outputs under $OUTPUT_DIR:"
    for f in "${OUTPUT_DIR%/}/scale_audit_report.md"; do
        if [ -f "$f" ]; then
            echo "  [OK] $f ($(du -h "$f" | cut -f1))"
        else
            echo "  [MISSING] $f (this should not happen — script crashed pre-write?)"
        fi
    done
    # Per-winner CSVs are variable in count; list whatever is there.
    ls -la "${OUTPUT_DIR%/}"/*_beta_hdi_table.csv 2>/dev/null | \
        awk '{print "  [OK] "$NF" ("$5" bytes)"}' || \
        echo "  (no per-winner beta_hdi_table.csv files found)"
else
    # Exit-propagation here is critical: the master pipeline orchestrator
    # (plan 21-10) chains step 21.8 via `--dependency=afterok:$AUDIT2_JID`,
    # so a non-zero exit here naturally blocks the chain. This branch
    # fires ONLY for genuine audit errors (not NULL_RESULT, which is 0).
    echo "[ERROR] Audit failed with exit code $EXIT_CODE — genuine audit error, not null result"
    echo "[ERROR] Typical causes:"
    echo "[ERROR]   1. winners.txt missing or malformed (check step 21.5)"
    echo "[ERROR]   2. {winner}_posterior.nc corrupt or load failure"
    echo "[ERROR]   3. statsmodels.stats.multitest import failed (check ds_env)"
    echo "[ERROR] See scale audit stderr output above for root cause."
fi

# Auto-push logs and results when job completes
source cluster/autopush.sh

# Load-bearing — master orchestrator chains step 21.8 via
# --dependency=afterok. Both PROCEED_TO_AVERAGING and NULL_RESULT exit
# 0 so the dependency naturally advances; exit 1 blocks only on
# genuine audit errors. DO NOT replace with `exit 0`.
exit $EXIT_CODE
