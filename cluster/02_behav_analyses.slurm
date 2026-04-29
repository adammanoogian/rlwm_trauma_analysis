#!/bin/bash
# =============================================================================
# SLURM Batch Script: Stage 02 — Behavioral Analyses (Monash M3)
# =============================================================================
# Serializes the four 02_behav_analyses/*.py scripts (post-29-04b renumbering):
#   1. 01_summarize_behavioral_data.py
#   2. 02_visualize_task_performance.py
#   3. 03_analyze_trauma_groups.py
#   4. 04_run_statistical_analyses.py
#
# CPU-only (pandas/matplotlib/seaborn). Runs under 10 minutes on N=154.
# Requires data/processed/task_trials_long.csv from stage 01.
#
# Usage:
#   sbatch cluster/02_behav_analyses.slurm
#   sbatch --dependency=afterok:$JID_01 cluster/02_behav_analyses.slurm
# =============================================================================

#SBATCH --job-name=p29_02_behav
#SBATCH --output=logs/02_behav_analyses_%j.out
#SBATCH --error=logs/02_behav_analyses_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

set -euo pipefail

module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

# Install autopush trap (must come BEFORE any failure-prone command).
# autopush.sh registers an EXIT trap that fires on success or failure,
# so SLURM .err/.out logs land in git regardless of outcome — see
# cluster/autopush.sh for rationale.
[ -f cluster/autopush.sh ] && source cluster/autopush.sh


echo "============================================================"
echo "Stage 02 — Behavioral Analyses (01..04)"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       ${SLURMD_NODENAME:-$(hostname)}"
echo "Project:    $PROJECT_ROOT"
echo "Start time: $(date)"
echo "============================================================"

if conda activate ds_env 2>/dev/null; then
    echo "Activated ds_env by name"
elif conda activate /scratch/fc37/$USER/conda/envs/ds_env 2>/dev/null; then
    echo "Activated ds_env from /scratch/fc37/$USER/"
elif conda activate /scratch/${PROJECT:-fc37}/$USER/conda/envs/ds_env 2>/dev/null; then
    echo "Activated ds_env from /scratch/${PROJECT:-fc37}/$USER/"
elif conda activate rlwm 2>/dev/null; then
    echo "Activated rlwm (fallback)"
elif conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu (fallback)"
else
    echo "ERROR: Failed to activate conda environment (tried ds_env, rlwm, rlwm_gpu)"
    exit 1
fi

mkdir -p logs reports/figures reports/tables data/processed

export PYTHONUNBUFFERED=1

# Verify stage-01 output exists
if [[ ! -f data/processed/task_trials_long.csv ]]; then
    echo "ERROR: data/processed/task_trials_long.csv missing — run stage 01 first"
    exit 1
fi

run_step() {
    local desc="$1"
    shift
    echo ""
    echo "--- $desc ---"
    echo "  $*"
    "$@"
}

run_step "01 — Summarize behavioral data" \
    python scripts/02_behav_analyses/01_summarize_behavioral_data.py

run_step "02 — Visualize task performance" \
    python scripts/02_behav_analyses/02_visualize_task_performance.py

run_step "03 — Analyze trauma groups" \
    python scripts/02_behav_analyses/03_analyze_trauma_groups.py

run_step "04 — Run statistical analyses" \
    python scripts/02_behav_analyses/04_run_statistical_analyses.py

echo ""
echo "============================================================"
echo "Stage 02 complete — $(date)"
echo "============================================================"

if [[ -f cluster/autopush.sh ]]; then
    source cluster/autopush.sh
fi
