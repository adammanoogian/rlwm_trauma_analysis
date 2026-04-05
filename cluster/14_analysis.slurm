#!/bin/bash
# =============================================================================
# SLURM Batch Script: Post-Fitting Analysis (Steps 14-16)
# =============================================================================
# Runs model comparison, trauma analysis, and scale regressions.
# CPU-only, fast (~5 min total). Designed to run after all MLE fits complete.
#
# Usage:
#   sbatch cluster/14_analysis.slurm
#   sbatch --dependency=afterok:12345:12346 cluster/14_analysis.slurm
#
# =============================================================================

#SBATCH --job-name=rlwm_analysis
#SBATCH --output=cluster/logs/analysis_%j.out
#SBATCH --error=cluster/logs/analysis_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

conda activate rlwm_gpu 2>/dev/null || \
conda activate rlwm 2>/dev/null || {
    echo "ERROR: Failed to activate conda environment"
    exit 1
}

export PYTHONUNBUFFERED=1
mkdir -p cluster/logs

echo "============================================================"
echo "Post-Fitting Analysis (Steps 14-16)"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Start time: $(date)"
echo "============================================================"

FAILED=false

run_step() {
    local desc="$1"
    shift
    echo ""
    echo "--- $desc ---"
    echo "  $@"
    "$@"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  WARNING: '$desc' exited with code $rc"
        FAILED=true
    fi
    return $rc
}

run_step "14 — Model comparison" \
    python scripts/14_compare_models.py

run_step "15 — Parameter-trauma analysis" \
    python scripts/15_analyze_mle_by_trauma.py --model all

run_step "16 — Scale regressions" \
    python scripts/16_regress_parameters_on_scales.py --model all

echo ""
echo "============================================================"
echo "Analysis complete! End time: $(date)"
echo "============================================================"
echo "Outputs:"
echo "  output/model_comparison/"
echo "  output/mle_by_trauma/"
echo "  output/regressions/"
echo "============================================================"

if [[ "$FAILED" == "true" ]]; then
    echo "WARNING: One or more steps had errors."
    exit 1
fi
exit 0
