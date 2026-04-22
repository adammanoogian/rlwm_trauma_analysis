#!/bin/bash
# =============================================================================
# SLURM Batch Script: Stage 04c — L2 Winner Dispatcher (wrapper)
# =============================================================================
# Proper SLURM wrapper around cluster/21_dispatch_l2_winners.sh, which
# reads winners.txt and submits one cluster/04c_level2.slurm per
# winner via `sbatch --wait` (blocking parallel waits via `&` + `wait`).
#
# Why a SLURM wrapper instead of `sbatch --wrap=...`?
#
#   The dispatcher uses `sbatch --wait` internally so that this dispatcher
#   job stays alive until ALL child L2 fits complete. The parent
#   orchestrator (cluster/21_submit_pipeline.sh) then chains step 21.7
#   via --dependency=afterok:$DISPATCH_JID, which only fires after the
#   dispatcher exits cleanly.
#
#   M6b subscale worst case is ~12h. A `--wrap=` job inherits the
#   account default time limit (usually 1-4h) and would be killed by
#   the scheduler long before the L2 fits complete, silently orphaning
#   the entire downstream chain.
#
#   This wrapper declares --time=14:00:00 (12h M6b worst case + 2h
#   slack) and --mem=2G --cpus-per-task=1 because the dispatcher itself
#   only blocks on sbatch --wait — it does no real computation.
#
# Plan-checker Issue #6: this wrapper is the documented fix for the
# "scheduler kills the --wrap-blocked dispatcher" failure mode.
#
# Usage (from the master orchestrator only — do not call directly):
#   sbatch --dependency=afterok:$LOO_JID cluster/21_6_dispatch_l2.slurm
# =============================================================================

#SBATCH --job-name=21_dispatch_l2
#SBATCH --output=logs/21_dispatch_l2_%j.out
#SBATCH --error=logs/21_dispatch_l2_%j.err
#SBATCH --time=14:00:00
#SBATCH --mem=2G
#SBATCH --cpus-per-task=1
#SBATCH --partition=comp

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"

echo "============================================================"
echo "Phase 21 Step 21.6 — L2 Winner Dispatcher"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       ${SLURMD_NODENAME:-$(hostname)}"
echo "Project:    $(pwd)"
echo "Start time: $(date)"
echo "============================================================"
echo ""

bash cluster/21_dispatch_l2_winners.sh
EXIT_CODE=$?

echo ""
echo "[$(date)] Dispatcher exited with code $EXIT_CODE"
exit $EXIT_CODE
