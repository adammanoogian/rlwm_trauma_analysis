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
    git add logs/ models/bayesian/ reports/ 2>/dev/null

    # Check if there's anything to commit
    if git diff --cached --quiet 2>/dev/null; then
        echo "  No new files to commit — skipping push"
        return 0
    fi

    git commit -m "${job_name} job ${job_id} results" 2>/dev/null || {
        echo "  WARNING: git commit failed (maybe nothing staged)"
        return 1
    }

    # Pull-rebase-retry loop — handles the fan-out race where N concurrent
    # compute-node autopushes all try to push to origin/main at the same
    # time. Without this, the first push wins and subsequent jobs get a
    # non-fast-forward rejection; the second-to-finish job's autopush then
    # silently absorbs the first's not-yet-pushed commit on its working tree
    # (because all jobs share the NFS-mounted repo), losing the
    # self-titled commit and creating "ghost" autopushes.
    #
    # The loop:
    #   1. Try git push.
    #   2. On failure, git pull --rebase to incorporate any concurrent commits.
    #   3. Retry push (up to 3 attempts).
    #
    # Standard distributed-update pattern; no scheduler coordination needed.
    local pushed=0
    for attempt in 1 2 3; do
        if git push 2>&1; then
            pushed=1
            break
        fi
        echo "  push attempt $attempt failed (likely concurrent autopush race); pulling --rebase and retrying..."
        if ! git pull --rebase 2>&1; then
            echo "  WARNING: git pull --rebase failed — aborting retry loop"
            break
        fi
    done
    if [[ "$pushed" != "1" ]]; then
        echo "  WARNING: git push failed after 3 attempts (network, auth, or persistent rebase conflict)"
        return 1
    fi

    echo "  Auto-push complete"
}

_autopush
