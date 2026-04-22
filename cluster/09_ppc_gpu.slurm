#!/bin/bash
# =============================================================================
# SLURM Batch Script: Posterior Predictive Checks (GPU)
# =============================================================================
# Runs posterior predictive checks with GPU acceleration.
# Maps to: scripts/05_post_fitting_checks/run_posterior_ppc.py
#
# The stage-03 mirror (scripts/03_model_prefitting/09_run_ppc.py) was deleted
# in plan 29-04b — the thin orchestrator is now authoritative in stage 05,
# which is where posterior diagnostics live per Scheme D.
#
# Usage:
#   sbatch cluster/09_ppc_gpu.slurm
#   sbatch --export=MODEL=wmrl_m3 cluster/09_ppc_gpu.slurm
#
# =============================================================================

#SBATCH --job-name=rlwm_ppc_gpu
#SBATCH --output=cluster/logs/ppc_gpu_%j.out
#SBATCH --error=cluster/logs/ppc_gpu_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Default values (can be overridden via --export)
MODEL=${MODEL:-wmrl_m3}
SKIP_MODEL_RECOVERY=${SKIP_MODEL_RECOVERY:-false}

# Load modules and activate environment
# Note: Do NOT load cuda module — JAX bundles its own CUDA libraries
module load miniforge3
conda activate rlwm_gpu || {
    echo "ERROR: Failed to activate 'rlwm_gpu' conda environment"
    exit 1
}

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
mkdir -p cluster/logs

echo "=========================================="
echo "Posterior Predictive Check (GPU)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Model: $MODEL"
echo "Skip model recovery: $SKIP_MODEL_RECOVERY"
echo "Start time: $(date)"
echo "=========================================="

# Build command
CMD="python scripts/05_post_fitting_checks/run_posterior_ppc.py --model $MODEL --use-gpu"
if [ "$SKIP_MODEL_RECOVERY" = "true" ]; then
    CMD="$CMD --skip-model-recovery"
fi

export PYTHONUNBUFFERED=1

# Start background GPU monitoring (log every 5 min)
if command -v nvidia-smi &>/dev/null; then
    (while true; do
        echo "[GPU $(date +%H:%M:%S)] $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{printf "util=%s%% mem=%s/%sMB temp=%sC", $1, $2, $3, $4}')"
        sleep 300
    done) &
    GPU_MONITOR_PID=$!
fi

$CMD

# Stop GPU monitoring
[[ -n "$GPU_MONITOR_PID" ]] && kill "$GPU_MONITOR_PID" 2>/dev/null && wait "$GPU_MONITOR_PID" 2>/dev/null

echo "=========================================="
echo "PPC complete!"
echo "Results: output/ppc/$MODEL/"
echo "Figures: figures/ppc/$MODEL/"
echo "End time: $(date)"
echo "=========================================="
