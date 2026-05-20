#!/bin/bash
# =============================================================================
# DEPRECATED SHIM — kept for backward compatibility (Phase 29 Wave 4 / 29-05)
# =============================================================================
# Historical note: this script used to orchestrate the 9-step Phase-21
# Bayesian pipeline via its own sbatch --dependency=afterok chain.
# Phase 29 Wave 4 (plan 29-05) consolidated the 9 per-step SLURMs into 6
# stage-numbered entry points (cluster/01..06_*.slurm) + a new master
# orchestrator `cluster/submit_all.sh` that now owns the afterok chain.
#
# User memory (v4.0 SHIPPED) documented `bash cluster/21_submit_pipeline.sh`
# as the canonical entry point, so this shim preserves that invocation.
# Every new caller should use `cluster/submit_all.sh` directly.
#
# Usage (unchanged — still runs the full chain):
#   bash cluster/21_submit_pipeline.sh           # delegates to submit_all.sh
#   bash cluster/21_submit_pipeline.sh --dry-run # passes through to submit_all.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

# Pre-flight gate is now a SLURM job (cluster/00_preflight.slurm), submitted
# as stage 00 by `cluster/submit_all.sh --preflight` and chained to stage 01
# via afterok. The earlier login-node `pytest` invocation was removed because
# the login node has neither conda nor pytest available; running the same
# 2-cov L2 hook test on a compute node with rlwm_gpu activated is the
# canonical SLURM-automated path (see CLUSTER_GPU_LESSONS.md / user feedback
# memory feedback_cluster_automation.md).
echo "[21_submit_pipeline.sh] Delegating to cluster/submit_all.sh --preflight (post-29-05 canonical master)"
echo ""
exec bash cluster/submit_all.sh --preflight "$@"
