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
#   bash cluster/12_submit_all_gpu.sh --auto-push    # Push results to git when done
#
# Requirements:
#   - rlwm_gpu conda environment (see cluster/00_setup_env_gpu.sh)
#   - GPU partition access on M3
#
# =============================================================================

cd "$(dirname "$0")/.."

# Parse arguments
AUTO_PUSH=false
for arg in "$@"; do
    case "$arg" in
        --auto-push) AUTO_PUSH=true ;;
    esac
done

echo "============================================================"
echo "Submitting GPU MLE Fitting Jobs"
echo "Auto-push: $AUTO_PUSH"
echo "============================================================"
echo ""

ALL_JOBIDS=()
# Model list — keep in sync with config.py MODEL_REGISTRY
# Choice-only: qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b
# Joint RT:    wmrl_m4
for model in qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b wmrl_m4; do
    # M4 (LBA joint choice+RT) needs more walltime: float64 + CDF/PDF density is ~10x
    # slower per trial than choice-only softmax models
    EXTRA_ARGS=""
    if [[ "$model" == "wmrl_m4" ]]; then
        EXTRA_ARGS="--time=24:00:00"
    fi

    jobid=$(sbatch --export=ALL,MODEL="$model" $EXTRA_ARGS --parsable cluster/12_mle_gpu.slurm 2>&1)
    if [[ $? -eq 0 && -n "$jobid" ]]; then
        ALL_JOBIDS+=("$jobid")
        echo "  $model: Job $jobid submitted (GPU)${EXTRA_ARGS:+ [24h walltime]}"
    else
        echo "  $model: FAILED - $jobid"
    fi
done

# Auto-push: submit a dependent git push job
push_jobid=""
if [[ "$AUTO_PUSH" == "true" && ${#ALL_JOBIDS[@]} -gt 0 ]]; then
    echo ""
    DEPENDENCY=$(IFS=:; echo "${ALL_JOBIDS[*]}")
    push_jobid=$(sbatch \
        --dependency=afterany:${DEPENDENCY} \
        --export=ALL,PARENT_JOBS="${ALL_JOBIDS[*]// /,}" \
        --job-name="push_results" \
        --parsable cluster/99_push_results.slurm 2>&1)

    if [[ $? -eq 0 && -n "$push_jobid" ]]; then
        echo "  push_results: Job $push_jobid (after all fits)"
    else
        echo "  push_results: FAILED - $push_jobid"
    fi
fi

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
if [[ -n "$push_jobid" ]]; then
echo "  tail -f cluster/logs/push_results_${push_jobid}.out"
fi
echo ""
echo "Results will be saved to:"
echo "  output/mle/"
if [[ -n "$push_jobid" ]]; then
echo ""
echo "Results will be auto-pushed to a git branch when fitting completes."
fi
echo "============================================================"
