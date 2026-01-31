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
    jobid=$(sbatch --export=MODEL=$model --parsable cluster/run_mle_parallel.slurm)
    echo "  $model: Job $jobid submitted"
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
