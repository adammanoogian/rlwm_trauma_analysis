#!/bin/bash
# =============================================================================
# Submit Separate GPU SLURM Jobs for Each MLE Model
# =============================================================================
# Submits all 7 models as independent GPU jobs for:
#   - Better fault isolation (one model failing doesn't affect others)
#   - Individual time limits per model
#   - Parallel model execution on separate GPU nodes
#   - M4 (RLWM-LBA) enables float64 automatically via fit_mle.py
#
# Usage:
#   bash cluster/12_submit_all_gpu.sh
#
# Requirements:
#   - rlwm_gpu conda environment (see cluster/00_setup_env_gpu.sh)
#   - GPU partition access on M3
#
# =============================================================================

cd "$(dirname "$0")/.."

echo "============================================================"
echo "Submitting GPU MLE Fitting Jobs"
echo "============================================================"
echo ""

for model in qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b wmrl_m4; do
    # M4 (LBA joint choice+RT) needs more walltime: float64 + CDF/PDF density is ~10x
    # slower per trial than choice-only softmax models
    EXTRA_ARGS=""
    if [[ "$model" == "wmrl_m4" ]]; then
        EXTRA_ARGS="--time=24:00:00"
    fi

    jobid=$(sbatch --export=ALL,MODEL="$model" $EXTRA_ARGS --parsable cluster/12_mle_gpu.slurm 2>&1)
    if [[ $? -eq 0 && -n "$jobid" ]]; then
        echo "  $model: Job $jobid submitted (GPU)${EXTRA_ARGS:+ [24h walltime]}"
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
