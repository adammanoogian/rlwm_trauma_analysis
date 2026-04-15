#!/bin/bash
# =============================================================================
# Auto-push results after SLURM job completes
# =============================================================================
# Source this at the end of any SLURM script to automatically commit and push
# logs and output files when the job finishes.
#
# Usage (at the end of a SLURM script):
#   source cluster/autopush.sh
#
# Requires:
#   - PROJECT_ROOT set to the repo root directory
#   - SLURM_JOB_ID set (automatic in SLURM jobs)
#   - Git SSH key accessible from compute nodes
# =============================================================================

_autopush() {
    local job_id="${SLURM_JOB_ID:-unknown}"
    local job_name="${SLURM_JOB_NAME:-slurm_job}"

    echo ""
    echo "============================================================"
    echo "AUTO-PUSH: Committing and pushing results"
    echo "============================================================"

    cd "${PROJECT_ROOT:-.}" || return 1

    # Stage logs for this job
    git add logs/ output/bayesian/ 2>/dev/null

    # Check if there's anything to commit
    if git diff --cached --quiet 2>/dev/null; then
        echo "  No new files to commit — skipping push"
        return 0
    fi

    git commit -m "${job_name} job ${job_id} results" 2>/dev/null || {
        echo "  WARNING: git commit failed (maybe nothing staged)"
        return 1
    }

    git push 2>&1 || {
        echo "  WARNING: git push failed (network or auth issue)"
        return 1
    }

    echo "  Auto-push complete"
}

_autopush
