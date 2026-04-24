#!/bin/bash
# =============================================================================
# SLURM Batch Script: Stage 01 — Data Preprocessing (Monash M3)
# =============================================================================
# Serializes the four 01_data_preprocessing/*.py scripts:
#   1. 01_parse_raw_data.py       — parse raw jsPsych JSON dumps
#   2. 02_create_collated_csv.py  — collate per-participant CSVs
#   3. 03_create_task_trials_csv.py — produce task_trials_long[_all].csv
#   4. 04_create_summary_csv.py   — produce summary_participant_metrics.csv
#
# CPU-only (pure NumPy/pandas). Runs under 5 minutes on N=154.
#
# Usage:
#   sbatch cluster/01_data_processing.slurm
#
# Outputs:
#   - data/processed/task_trials_long.csv
#   - data/processed/task_trials_long_all.csv
#   - data/processed/summary_participant_metrics.csv
# =============================================================================

#SBATCH --job-name=p29_01_data
#SBATCH --output=cluster/logs/01_data_processing_%j.out
#SBATCH --error=cluster/logs/01_data_processing_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

set -euo pipefail

module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Stage 01 — Data Preprocessing (01..04)"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       ${SLURMD_NODENAME:-$(hostname)}"
echo "Project:    $PROJECT_ROOT"
echo "Start time: $(date)"
echo "============================================================"

# Activate conda environment — ds_env first (lightweight CPU), fallback ladder
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

mkdir -p cluster/logs data/processed data/interim

export PYTHONUNBUFFERED=1

run_step() {
    local desc="$1"
    shift
    echo ""
    echo "--- $desc ---"
    echo "  $*"
    "$@"
}

run_step "01 — Parse raw jsPsych data" \
    python scripts/01_data_preprocessing/01_parse_raw_data.py

run_step "02 — Create collated CSV" \
    python scripts/01_data_preprocessing/02_create_collated_csv.py

run_step "03 — Create task trials long CSV" \
    python scripts/01_data_preprocessing/03_create_task_trials_csv.py

run_step "04 — Create summary participant CSV" \
    python scripts/01_data_preprocessing/04_create_summary_csv.py

echo ""
echo "============================================================"
echo "Stage 01 complete — $(date)"
echo "============================================================"
echo "Key outputs:"
echo "  data/processed/task_trials_long.csv"
echo "  data/processed/task_trials_long_all.csv"
echo "  data/processed/summary_participant_metrics.csv"
echo "============================================================"

# Auto-push logs and results when job completes
if [[ -f cluster/autopush.sh ]]; then
    source cluster/autopush.sh
fi
