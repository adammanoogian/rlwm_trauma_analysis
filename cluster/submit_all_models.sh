#!/bin/bash
# =============================================================================
# Submit Separate SLURM Jobs for Each MLE Model
# =============================================================================
# Submits qlearning, wmrl, and wmrl_m3 as independent jobs for:
#   - Better fault isolation (one model failing doesn't affect others)
#   - Individual time limits per model
#   - Parallel model execution if resources allow
#
# Usage:
#   bash cluster/submit_all_models.sh
#
# =============================================================================

cd "$(dirname "$0")/.."

echo "============================================================"
echo "Submitting MLE Fitting Jobs"
echo "============================================================"
echo ""

for model in qlearning wmrl wmrl_m3; do
    jobid=$(sbatch --export=ALL,MODEL="$model" --parsable cluster/run_mle_parallel.slurm 2>&1)
    if [[ $? -eq 0 && -n "$jobid" ]]; then
        echo "  $model: Job $jobid submitted"
    else
        echo "  $model: FAILED - $jobid"
    fi
done

echo ""
echo "============================================================"
echo "All jobs submitted!"
echo "============================================================"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f cluster/logs/mle_parallel_*.out"
echo ""
echo "Results will be saved to:"
echo "  output/mle/"
echo "============================================================"
