#!/bin/bash
# =============================================================================
# SLURM Batch Script: Phase 33 Scientific-Tier Test Gate
# =============================================================================
# Runs the scientific test tier (pytest tests/scientific/) on a CPU node.
# Validates parameter recovery, model consistency, and M4/M3 backward compat
# after the Phase 33 alpha_neg removal.
#
# Scientific tests take > 60 s each and require either GPU (M4 benchmarks)
# or many minutes of CPU.  They are excluded from the local fast/integration
# CI gates and validated exclusively on the cluster via this script.
#
# Usage:
#   sbatch cluster/33_scientific_tier.slurm
#
# Outputs:
#   logs/33-01_scientific_${SLURM_JOB_ID}.out  — full pytest output
#   logs/33-01_scientific_${SLURM_JOB_ID}.err  — stderr
#
# Exit code mirrors pytest exit code (0 = all passed, non-zero = failures).
# =============================================================================

#SBATCH --job-name=33_sci_gate
#SBATCH --output=logs/33-01_scientific_%j.out
#SBATCH --error=logs/33-01_scientific_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:T4:1

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Phase 33 Scientific-Tier Test Gate"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Project: $PROJECT_ROOT"
echo "Commit: $(git rev-parse --short HEAD 2>/dev/null) [$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ') dirty] — $(git log -1 --format=%s 2>/dev/null)"
echo "Start time: $(date)"
echo "============================================================"

if mamba activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu (mamba)"
elif conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu (conda)"
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/fc37/$USER/"
elif conda activate /scratch/${PROJECT:-fc37}/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/${PROJECT:-fc37}/$USER/"
else
    echo "ERROR: Failed to activate rlwm_gpu — see cluster/00_setup_env_gpu.sh"
    exit 1
fi

# Verify GPU available (some scientific tests benchmark GPU vs CPU)
python -c "
import jax
devices = jax.devices()
print(f'JAX {jax.__version__} devices: {devices}')
" || {
    echo "ERROR: JAX import failed"
    exit 1
}
echo ""

export PYTHONUNBUFFERED=1
mkdir -p logs

# =============================================================================
# Run scientific tier
# =============================================================================
echo ""
echo "[$(date)] Starting scientific-tier tests"
echo "Tests: tests/scientific/"
echo ""

python -m pytest tests/scientific/ -v \
    --tb=short \
    -p no:warnings \
    2>&1

EXIT_CODE=$?
echo ""
echo "============================================================"
echo "[$(date)] Scientific-tier gate finished (exit code: $EXIT_CODE)"
if [[ $EXIT_CODE -eq 0 ]]; then
    echo "VERDICT: PASS — all scientific tests passed"
else
    echo "VERDICT: FAIL — see above for failures"
fi
echo "============================================================"

exit $EXIT_CODE
