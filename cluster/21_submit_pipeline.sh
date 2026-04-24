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

# Pre-flight gate (preserved from v4.0): the 2-covariate L2 hook for M3/M5/M6a
# is load-bearing for stage 04c. Fail fast LOCALLY before burning cluster cycles.
echo "[$(date)] Pre-flight: verifying 2-covariate L2 hook (plan 21-11)..."
if ! pytest tests/integration/test_numpyro_models_2cov.py -v -k "not slow" --tb=short; then
  echo "[ABORT] plan 21-11 tests failed — 2-covariate L2 hook for M3/M5/M6a is not wired correctly." >&2
  echo "[ABORT] No cluster jobs submitted. Fix src/rlwm/fitting/numpyro_models.py and retry." >&2
  exit 1
fi
echo "[$(date)] Pre-flight OK — 2-covariate L2 hook ready."
echo ""

echo "[21_submit_pipeline.sh] Delegating to cluster/submit_all.sh (post-29-05 canonical master)"
echo ""
exec bash cluster/submit_all.sh "$@"
