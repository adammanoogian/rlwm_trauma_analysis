#!/bin/bash
# =============================================================================
# SLURM Batch Script: Canary Acceptance Gate (productionizes 24-01-PLAN.md Task 3)
# =============================================================================
# Closes the canary lifecycle that begins with cluster/03_submit_canary.sh.
# Runs the four CONTEXT.md §Submission strategy acceptance criteria against
# the canary's autopushed artifacts + per-model Baribault gate readouts,
# then writes models/bayesian/21_canary_acceptance_report.md and autopushes
# it to main.
#
# WHAT THIS REPLACES
#   24-01-PLAN.md Task 3 used to ship four inline-bash blocks the operator
#   had to copy/paste into M3 (sacct check, ArviZ load loop, git log
#   --since, CCDS path negative check). That pattern violates the "fixes at
#   the call level" rule — every future cold-start would re-type the same
#   blocks. This script is the call-level productionization: one sbatch
#   call closes the canary, the verdict report autopushes itself.
#
# DEPENDENCIES (must already be on main when this runs)
#   models/bayesian/21_canary_metadata.txt        -- written by 03_submit_canary.sh
#   models/bayesian/21_canary_jobids.txt          -- written by 03_submit_canary.sh
#   models/bayesian/21_prior_predictive/<m>_prior_sim.nc      -- written by canary jobs
#   models/bayesian/21_prior_predictive/<m>_gate.md           -- written by canary jobs
#
# OUTPUT
#   models/bayesian/21_canary_acceptance_report.md  -- 4-criteria verdict +
#                                                     per-model Baribault
#                                                     (HARD_PASS / SOFT_PASS / FAIL)
#                                                     + overall APPROVED|REJECTED
#   logs/03_check_canary_<JID>.{out,err}            -- SLURM accounting trail
#
# EXIT SEMANTICS (the script's exit code propagates from check_canary.py)
#   0  APPROVED  — operator may run `bash cluster/submit_all.sh --from-stage 4`
#   1  REJECTED  — operator must root-cause + submit a fresh canary
#
# USAGE
#   sbatch cluster/03_check_canary.slurm                 # standalone
#   sbatch --wait cluster/03_check_canary.slurm          # blocking from another orchestrator
# =============================================================================

#SBATCH --job-name=p24_check_canary
#SBATCH --output=logs/03_check_canary_%j.out
#SBATCH --error=logs/03_check_canary_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2

set -euo pipefail

module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"
export PROJECT_ROOT

mkdir -p logs models/bayesian

echo "============================================================"
echo "Phase 24 Canary Acceptance Gate (SLURM-automated)"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       ${SLURMD_NODENAME:-$(hostname)}"
echo "Project:    $PROJECT_ROOT"
echo "Git HEAD:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Start time: $(date)"
echo "============================================================"

# ---------------------------------------------------------------
# Activate rlwm env (ArviZ + numpy + pandas + numpyro)
# Same dispatch ladder used by cluster/03_prefitting_gpu.slurm.
# CPU env (rlwm) is sufficient — no GPU needed for the verifier.
# Falls back to rlwm_gpu if rlwm isn't present.
# ---------------------------------------------------------------
if conda activate rlwm 2>/dev/null; then
    echo "Activated rlwm env"
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm 2>/dev/null; then
    echo "Activated rlwm from /scratch/fc37/$USER/"
elif conda activate /scratch/${PROJECT:-fc37}/$USER/conda/envs/rlwm 2>/dev/null; then
    echo "Activated rlwm from /scratch/${PROJECT:-fc37}/$USER/"
elif conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu env (rlwm CPU env not found; fallback)"
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/fc37/$USER/ (fallback)"
else
    echo "ERROR: failed to activate rlwm or rlwm_gpu conda env"
    exit 1
fi
echo "  python: $(which python)"

# ---------------------------------------------------------------
# Pull any concurrent autopushes so criterion (c) sees the latest commits.
# This matters because canary jobs autopush asynchronously — their commits
# may have landed on origin/main while this gate was queued.
# ---------------------------------------------------------------
echo ""
echo "[$(date)] Pulling latest from origin/main before evaluation..."
git pull --ff-only 2>&1 || echo "  (git pull failed — proceeding with local state)"

# ---------------------------------------------------------------
# Run the acceptance gate
# ---------------------------------------------------------------
echo ""
echo "[$(date)] Running tests/scientific/check_canary.py..."
EXIT_CODE=0
python tests/scientific/check_canary.py || EXIT_CODE=$?

echo ""
echo "[$(date)] check_canary.py exit code: $EXIT_CODE"

# ---------------------------------------------------------------
# Autopush the report (commits 21_canary_acceptance_report.md to main)
# Sourced AFTER the verifier so the verdict file is present to commit.
# ---------------------------------------------------------------
if [[ -f cluster/autopush.sh ]]; then
    source cluster/autopush.sh
fi

echo ""
echo "============================================================"
echo "End time:   $(date)"
echo "Exit code:  $EXIT_CODE"
echo "============================================================"

exit $EXIT_CODE
