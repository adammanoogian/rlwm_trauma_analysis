#!/bin/bash
# =============================================================================
# Submit Separate GPU SLURM Jobs for Each MLE Model
# =============================================================================
# Submits qlearning, wmrl, and wmrl_m3 as independent GPU jobs for:
#   - Better fault isolation (one model failing doesn't affect others)
#   - Individual time limits per model
#   - Parallel model execution on separate GPU nodes
#
# Usage:
#   bash cluster/submit_all_models_gpu.sh
#
# Requirements:
#   - rlwm_gpu conda environment (see cluster/setup_env_gpu.sh)
#   - GPU partition access on M3
#
# =============================================================================

cd "$(dirname "$0")/.."

echo "============================================================"
echo "Submitting GPU MLE Fitting Jobs"
echo "============================================================"
echo ""

for model in qlearning wmrl wmrl_m3; do
    jobid=$(sbatch --export=ALL,MODEL="$model" --parsable cluster/run_mle_gpu.slurm 2>&1)
    if [[ $? -eq 0 && -n "$jobid" ]]; then
        echo "  $model: Job $jobid submitted (GPU)"
    else
        echo "  $model: FAILED - $jobid"
    fi
done

echo ""
echo "============================================================"
echo "All GPU jobs submitted!"
echo "============================================================"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "View logs:"
echo "  tail -f cluster/logs/mle_gpu_*.out"
echo ""
echo "Results will be saved to:"
echo "  output/mle/"
echo "============================================================"
