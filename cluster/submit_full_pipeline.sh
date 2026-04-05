#!/bin/bash
# =============================================================================
# Wave-Based Pipeline Orchestrator
# =============================================================================
# Submits the full analysis pipeline as independent SLURM jobs with
# dependency chains, so each wave runs only after the previous completes.
#
#   Wave 1: MLE fitting (7 parallel GPU jobs, one per model)
#   Wave 2: Recovery + PPC (parallel GPU jobs, each depends on its fit job)
#   Wave 3: Model comparison + trauma analysis (1 CPU job, depends on all fits)
#
# Usage:
#   bash cluster/submit_full_pipeline.sh               # Full pipeline
#   bash cluster/submit_full_pipeline.sh --skip-wave2   # Fitting + analysis only
#   bash cluster/submit_full_pipeline.sh --models "wmrl_m5 wmrl_m6a"  # Subset
#
# All jobs are fault-isolated: if M4 fails, M5 recovery still runs.
# Wave 3 depends on ALL wave 1 jobs via afterany (runs even if some fail).
#
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# =============================================================================
# Parse arguments
# =============================================================================
SKIP_WAVE2=false
MODELS="qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b wmrl_m4"
# Models that support PPC (09_run_ppc.py --model choices)
PPC_MODELS="qlearning wmrl wmrl_m3"
# Models to run recovery for (all supported)
RECOVERY_MODELS="$MODELS"

for arg in "$@"; do
    case "$arg" in
        --skip-wave2) SKIP_WAVE2=true ;;
        --models=*) MODELS="${arg#*=}" ;;
        --models) shift; MODELS="$1" ;;
    esac
done

echo "============================================================"
echo "RLWM Full Pipeline — Wave-Based Orchestrator"
echo "============================================================"
echo "Models:      $MODELS"
echo "Skip wave 2: $SKIP_WAVE2"
echo "Submitted:   $(date)"
echo "============================================================"
echo ""

# =============================================================================
# Wave 1: MLE Fitting (parallel GPU jobs)
# =============================================================================
echo "WAVE 1: MLE Fitting"
echo "------------------------------------------------------------"

declare -A FIT_JOBS  # model -> jobid
ALL_FIT_IDS=()

for model in $MODELS; do
    # M4 needs more walltime
    EXTRA_ARGS=""
    if [[ "$model" == "wmrl_m4" ]]; then
        EXTRA_ARGS="--time=24:00:00"
    fi

    jobid=$(sbatch --export=ALL,MODEL="$model" \
        --job-name="fit_${model}" \
        $EXTRA_ARGS \
        --parsable cluster/12_mle_gpu.slurm 2>&1)

    if [[ $? -eq 0 && -n "$jobid" ]]; then
        FIT_JOBS[$model]=$jobid
        ALL_FIT_IDS+=("$jobid")
        echo "  fit_${model}: Job $jobid${EXTRA_ARGS:+ [24h]}"
    else
        echo "  fit_${model}: FAILED - $jobid"
    fi
done
echo ""

# Build colon-separated dependency string for wave 3
FIT_DEPENDENCY=$(IFS=:; echo "${ALL_FIT_IDS[*]}")

# =============================================================================
# Wave 2: Recovery + PPC (each depends on its model's fit job)
# =============================================================================
if [[ "$SKIP_WAVE2" != "true" ]]; then
    echo "WAVE 2: Recovery + PPC (after each model's fit)"
    echo "------------------------------------------------------------"

    WAVE2_IDS=()

    for model in $MODELS; do
        fit_jobid="${FIT_JOBS[$model]:-}"
        [[ -z "$fit_jobid" ]] && continue

        # Recovery: depends on fit completing (uses same fitting code path)
        rec_jobid=$(sbatch \
            --dependency=afterok:${fit_jobid} \
            --export=ALL,MODEL="$model" \
            --job-name="rec_${model}" \
            --parsable cluster/11_recovery_gpu.slurm 2>&1)

        if [[ $? -eq 0 && -n "$rec_jobid" ]]; then
            WAVE2_IDS+=("$rec_jobid")
            echo "  rec_${model}: Job $rec_jobid (after fit $fit_jobid)"
        else
            echo "  rec_${model}: FAILED - $rec_jobid"
        fi

        # PPC: only for models that 09_run_ppc.py supports
        if echo "$PPC_MODELS" | grep -qw "$model"; then
            ppc_jobid=$(sbatch \
                --dependency=afterok:${fit_jobid} \
                --export=ALL,MODEL="$model" \
                --job-name="ppc_${model}" \
                --parsable cluster/09_ppc_gpu.slurm 2>&1)

            if [[ $? -eq 0 && -n "$ppc_jobid" ]]; then
                WAVE2_IDS+=("$ppc_jobid")
                echo "  ppc_${model}: Job $ppc_jobid (after fit $fit_jobid)"
            else
                echo "  ppc_${model}: FAILED - $ppc_jobid"
            fi
        fi
    done
    echo ""
else
    echo "WAVE 2: Skipped (--skip-wave2)"
    echo ""
fi

# =============================================================================
# Wave 3: Analysis (depends on ALL fitting jobs completing)
# =============================================================================
echo "WAVE 3: Model Comparison + Trauma Analysis"
echo "------------------------------------------------------------"

# Use afterany so analysis runs even if some models fail — it will use
# whatever fits are available
analysis_jobid=$(sbatch \
    --dependency=afterany:${FIT_DEPENDENCY} \
    --job-name="analysis" \
    --parsable cluster/14_analysis.slurm 2>&1)

if [[ $? -eq 0 && -n "$analysis_jobid" ]]; then
    echo "  analysis: Job $analysis_jobid (after all fits)"
else
    echo "  analysis: FAILED - $analysis_jobid"
fi

echo ""
echo "============================================================"
echo "All jobs submitted!"
echo "============================================================"
echo ""
echo "Pipeline dependency chain:"
echo "  Wave 1 (fit):      ${ALL_FIT_IDS[*]}"
if [[ "$SKIP_WAVE2" != "true" && ${#WAVE2_IDS[@]} -gt 0 ]]; then
echo "  Wave 2 (rec+ppc):  ${WAVE2_IDS[*]}"
fi
echo "  Wave 3 (analysis): ${analysis_jobid:-failed}"
echo ""
echo "Monitor all jobs:"
echo "  squeue -u \$USER -o '%.10i %.12j %.8T %.10M %.6D %R'"
echo "  watch -n 30 'squeue -u \$USER'   # auto-refresh every 30s"
echo ""
echo "Tail live logs:"
for model in $MODELS; do
    jid="${FIT_JOBS[$model]:-}"
    [[ -n "$jid" ]] && echo "  tail -f cluster/logs/mle_gpu_${jid}.out   # fit_${model}"
done
[[ -n "${analysis_jobid:-}" ]] && echo "  tail -f cluster/logs/analysis_${analysis_jobid}.out   # analysis"
echo ""
echo "Cancel entire pipeline:"
echo "  scancel ${ALL_FIT_IDS[*]} ${WAVE2_IDS[*]:-} ${analysis_jobid:-}"
echo "============================================================"
