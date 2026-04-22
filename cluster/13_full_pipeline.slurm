#!/bin/bash
# =============================================================================
# SLURM Batch Script: Full Analysis Pipeline (Steps 03-16)
# =============================================================================
# Runs the complete analysis pipeline from processed data through final results.
# Assumes task_trials_long.csv and summary_participant_metrics.csv already exist
# (produced locally by run_data_pipeline.py, steps 01-04).
#
# This script runs sequentially:
#   Phase A (03-08): Data formatting + behavioral analysis  (~5 min)
#   Phase B (12):    MLE fitting (7 models, GPU, 154 subj)   (~40-60 hours)
#   Phase C (14-16): Model comparison + trauma analysis      (~5 min)
#
# Optional validation steps (09-11) are skipped by default.
# Set VALIDATE=true to include them (~30 min extra).
#
# Usage:
#   sbatch cluster/13_full_pipeline.slurm                          # Standard run
#   sbatch --export=VALIDATE=true cluster/13_full_pipeline.slurm   # With validation
#   sbatch --export=SKIP_BEHAVIORAL=true cluster/13_full_pipeline.slurm  # Skip 05-08
#
# Outputs:
#   output/mle/*_individual_fits.csv       — Fitted parameters
#   output/model_comparison/               — AIC/BIC comparison
#   output/mle_by_trauma/                  — Parameter-trauma relationships
#   output/regressions/                    — Scale regression results
#   figures/                               — All generated figures
#   cluster/logs/full_pipeline_*.out       — This job's log
#
# =============================================================================

#SBATCH --job-name=rlwm_full
#SBATCH --output=cluster/logs/full_pipeline_%j.out
#SBATCH --error=cluster/logs/full_pipeline_%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# =============================================================================
# Configuration
# =============================================================================
VALIDATE="${VALIDATE:-false}"
SKIP_BEHAVIORAL="${SKIP_BEHAVIORAL:-false}"
DATA_FILE="output/task_trials_long.csv"

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "RLWM Full Pipeline — M3 SLURM Job"
echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node:        ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s):      ${CUDA_VISIBLE_DEVICES:-none}"
echo "Validate:    $VALIDATE"
echo "Skip behav:  $SKIP_BEHAVIORAL"
echo "Start time:  $(date)"
echo "============================================================"

# Activate GPU conda environment
_PROJECT="${PROJECT:-fc37}"
conda activate rlwm_gpu 2>/dev/null || \
conda activate /scratch/${_PROJECT}/${USER}/conda/envs/rlwm_gpu 2>/dev/null || {
    echo "ERROR: Failed to activate 'rlwm_gpu' conda environment"
    echo "Create it first with: bash cluster/00_setup_env_gpu.sh"
    exit 1
}

mkdir -p cluster/logs

# Verify data
if [[ ! -f "$DATA_FILE" ]]; then
    echo "ERROR: $DATA_FILE not found."
    echo "Run 'python run_data_pipeline.py' locally first, then git push."
    exit 1
fi

export PYTHONUNBUFFERED=1
STEP=0
FAILED=false

run_step() {
    STEP=$((STEP + 1))
    local desc="$1"
    shift
    echo ""
    echo "============================================================"
    echo "[$STEP] $desc"
    echo "  $@"
    echo "  Started: $(date)"
    echo "============================================================"
    "$@"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  WARNING: Step $STEP exited with code $rc"
        FAILED=true
        return $rc
    fi
    echo "  Completed: $(date)"
    return 0
}

# =============================================================================
# Phase A: Behavioral Analysis (Steps 05-08)
# =============================================================================
if [[ "$SKIP_BEHAVIORAL" != "true" ]]; then
    echo ""
    echo "============================================================"
    echo "PHASE A: Behavioral Analysis (Steps 05-08)"
    echo "============================================================"

    run_step "05 — Summarize behavioral data" \
        python scripts/02_behav_analyses/05_summarize_behavioral_data.py

    run_step "06 — Visualize task performance" \
        python scripts/02_behav_analyses/06_visualize_task_performance.py

    run_step "07 — Analyze trauma groups" \
        python scripts/02_behav_analyses/07_analyze_trauma_groups.py

    run_step "08 — Run statistical analyses" \
        python scripts/02_behav_analyses/08_run_statistical_analyses.py
else
    echo ""
    echo "Skipping Phase A (--export=SKIP_BEHAVIORAL=true)"
fi

# =============================================================================
# Phase B: MLE Fitting (Step 12) — GPU accelerated
# =============================================================================
echo ""
echo "============================================================"
echo "PHASE B: MLE Model Fitting (Step 12)"
echo "============================================================"

# Start background GPU monitoring for fitting phase
if command -v nvidia-smi &>/dev/null; then
    (while true; do
        echo "[GPU $(date +%H:%M:%S)] $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{printf "util=%s%% mem=%s/%sMB temp=%sC", $1, $2, $3, $4}')"
        sleep 300
    done) &
    GPU_MONITOR_PID=$!
fi

for model in qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b wmrl_m4; do
    run_step "12 — Fit MLE: $model (GPU)" \
        python scripts/04_model_fitting/a_mle/12_fit_mle.py \
            --model "$model" \
            --data "$DATA_FILE" \
            --output output/mle/ \
            --n-starts 50 \
            --seed 42 \
            --use-gpu \
            --compute-diagnostics
done

# Stop GPU monitoring after fitting phase
[[ -n "$GPU_MONITOR_PID" ]] && kill "$GPU_MONITOR_PID" 2>/dev/null && wait "$GPU_MONITOR_PID" 2>/dev/null
unset GPU_MONITOR_PID

# =============================================================================
# Phase C: Model Comparison + Results Analysis (Steps 14-16)
# =============================================================================
echo ""
echo "============================================================"
echo "PHASE C: Results Analysis (Steps 14-16)"
echo "============================================================"

run_step "14 — Compare models" \
    python scripts/06_fit_analyses/compare_models.py

run_step "15 — Analyze MLE by trauma" \
    python scripts/15_analyze_mle_by_trauma.py --model all

run_step "16 — Regress parameters on scales" \
    python scripts/16_regress_parameters_on_scales.py --model all

# =============================================================================
# Phase D (Optional): Validation (Steps 09, 11)
# =============================================================================
if [[ "$VALIDATE" == "true" ]]; then
    echo ""
    echo "============================================================"
    echo "PHASE D: Validation (Steps 09, 11)"
    echo "============================================================"

    run_step "09 — Posterior predictive checks (wmrl_m3)" \
        python scripts/09_run_ppc.py --model wmrl_m3 --use-gpu

    run_step "11 — Parameter recovery (wmrl_m3)" \
        python scripts/11_run_model_recovery.py --model wmrl_m3 --use-gpu
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"
echo "End time: $(date)"
echo ""
echo "Key outputs:"
echo "  output/mle/*_individual_fits.csv     — Fitted parameters"
echo "  output/mle/*_group_summary.csv       — Group summaries"
echo "  output/model_comparison/             — Model comparison"
echo "  output/mle_by_trauma/                — Parameter-trauma"
echo "  output/regressions/                  — Scale regressions"
echo "  figures/                             — All figures"
echo ""
echo "To download results:"
echo "  git add output/mle/ output/model_comparison/ output/mle_by_trauma/"
echo "  git add output/regressions/*.csv figures/"
echo "  git commit -m 'Add pipeline results from M3'"
echo "  git push"
echo "============================================================"

if [[ "$FAILED" == "true" ]]; then
    echo "WARNING: One or more steps had errors. Check log above."
    exit 1
fi
exit 0
