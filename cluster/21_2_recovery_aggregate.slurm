#!/bin/bash
# =============================================================================
# SLURM Script: Step 21.2 Recovery Aggregation (Monash M3)
# =============================================================================
# Post-array aggregation step for the Bayesian parameter recovery workflow.
# Combines all {MODEL}_subject_*.json produced by the 21_2_recovery.slurm
# array into a per-model recovery CSV + summary.md with the final PASS/FAIL
# verdict for the kappa family.
#
# Usage (typically chained from the array job via afterok dependency):
#   ARRAY_JOBID=$(sbatch --parsable --export=ALL,MODEL=$MODEL \
#                         cluster/21_2_recovery.slurm)
#   sbatch --dependency=afterok:$ARRAY_JOBID \
#          --export=ALL,MODEL=$MODEL \
#          cluster/21_2_recovery_aggregate.slurm
#
# Standalone usage (after the array has finished):
#   sbatch --export=ALL,MODEL=wmrl_m3 cluster/21_2_recovery_aggregate.slurm
#
# Outputs saved to:
#   - output/bayesian/21_recovery/{MODEL}_recovery.csv
#   - output/bayesian/21_recovery/{MODEL}_recovery_summary.md
#
# Exit code:
#   0 — all kappa-family parameters PASS (or model has no kappa family)
#   1 — one or more kappa-family parameters FAIL
# =============================================================================

#SBATCH --job-name=bayes_21_2_aggregate
#SBATCH --output=logs/bayesian_21_2_aggregate_%j.out
#SBATCH --error=logs/bayesian_21_2_aggregate_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=comp

# --- Dispatch knobs (overridable via --export=ALL,MODEL=...) ---
MODEL="${MODEL:-wmrl_m3}"
N_SUBJECTS="${N_SUBJECTS:-50}"

# --- Environment Setup (matches Bayesian choice-only template pattern) ---
module load miniforge3

# Navigate to project directory
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Step 21.2 Recovery Aggregation: MODEL=${MODEL}"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Project: $PROJECT_ROOT"
echo "MODEL: $MODEL"
echo "N_SUBJECTS: $N_SUBJECTS"
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

# Verify pandas + arviz are importable (aggregation dependencies)
python -c "import pandas, arviz; print('pandas, arviz OK')" || {
    echo "ERROR: Required imports failed after conda activation"
    exit 1
}
echo ""

# --- Create output directories (idempotent) ---
mkdir -p logs
mkdir -p output/bayesian/21_recovery

# --- Run aggregation ---
echo ""
echo "[$(date)] Running aggregation for MODEL=$MODEL (n_subjects=$N_SUBJECTS)"

python scripts/bayesian_pipeline/21_run_bayesian_recovery.py \
    --mode aggregate \
    --model "$MODEL" \
    --n-subjects "$N_SUBJECTS" \
    --output-dir output/bayesian/21_recovery/

EXIT_CODE=$?

echo ""
echo "[$(date)] Aggregation finished (exit code: $EXIT_CODE)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Recovery verdict: PASS (or model has no kappa family)"
elif [ $EXIT_CODE -eq 1 ]; then
    echo "Recovery verdict: FAIL — see output/bayesian/21_recovery/${MODEL}_recovery_summary.md"
else
    echo "ERROR: aggregate runner exited with unexpected code $EXIT_CODE"
fi

echo ""
echo "Expected outputs:"
for f in output/bayesian/21_recovery/${MODEL}_recovery.csv \
         output/bayesian/21_recovery/${MODEL}_recovery_summary.md; do
    if [ -f "$f" ]; then
        echo "  ✓ $f ($(du -h "$f" | cut -f1))"
    else
        echo "  ✗ $f (MISSING)"
    fi
done

# Auto-push logs and results once the aggregate job completes.
# Safe here because this is a single job, not an array.
source cluster/autopush.sh

exit $EXIT_CODE
