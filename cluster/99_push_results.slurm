#!/bin/bash
# =============================================================================
# SLURM Batch Script: Auto-Push Results to Git
# =============================================================================
# Creates a results branch, commits output files, and pushes to remote.
# Designed to run as a final dependent job after fitting + analysis complete.
#
# Strategy (default — branch push):
#   1. Creates branch: results/slurm-YYYYMMDD-HHMMSS
#   2. Stages result files (outputs, figures — not logs or checkpoints)
#   3. Commits with descriptive message including parent job IDs
#   4. Pushes branch to GIT_REMOTE
#   5. Returns to original branch
#   User merges when ready: git merge origin/results/slurm-...
#
# Strategy (opt-in — push to main):
#   Uses git pull --rebase before committing. Riskier if the user pushed
#   code changes while jobs ran (merge conflicts fail silently).
#
# Usage:
#   # After fitting jobs (dependency chain):
#   sbatch --dependency=afterany:12345:12346 cluster/99_push_results.slurm
#
#   # Custom branch name:
#   sbatch --export=BRANCH_NAME=results/m5-refit cluster/99_push_results.slurm
#
#   # Push directly to main (risky if others pushed while jobs ran):
#   sbatch --export=PUSH_TO_MAIN=true cluster/99_push_results.slurm
#
#   # Override remote (if origin is not your repo):
#   sbatch --export=GIT_REMOTE=upstream cluster/99_push_results.slurm
#
# Prerequisites:
#   - SSH key or credential helper configured for git push on cluster nodes
#   - Git remote configured (defaults to 'origin'; override with GIT_REMOTE)
#
# ⚠️  WINDOWS / CRLF GOTCHA:
#   Git on Windows converts LF → CRLF on checkout even with .gitattributes.
#   sbatch rejects scripts with CRLF line endings (#!/bin/bash\r fails).
#   Fix: run this in your submit script BEFORE any sbatch calls:
#     sed -i 's/\r$//' cluster/*.slurm cluster/*.sh 2>/dev/null || true
#   This is the #1 cause of "invalid argument" or "bad interpreter" errors
#   when submitting from a Windows-cloned repo.
#
# =============================================================================

#SBATCH --job-name=push_results
#SBATCH --output=cluster/logs/push_results_%j.out
#SBATCH --error=cluster/logs/push_results_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH --partition=comp

# =============================================================================
# Configuration
# =============================================================================
PUSH_TO_MAIN="${PUSH_TO_MAIN:-false}"
BRANCH_NAME="${BRANCH_NAME:-results/slurm-$(date +%Y%m%d-%H%M%S)}"
PARENT_JOBS="${PARENT_JOBS:-${SLURM_JOB_DEPENDENCY:-unknown}}"
# Override if 'origin' is not your personal fork (e.g. GIT_REMOTE=upstream)
GIT_REMOTE="${GIT_REMOTE:-origin}"

# =============================================================================
# Environment Setup (minimal — only need git, no conda/python)
# =============================================================================
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

mkdir -p cluster/logs

echo "============================================================"
echo "Auto-Push Results to Git"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Project: $PROJECT_ROOT"
echo "Remote: $GIT_REMOTE"
echo "Strategy: $(if [[ "$PUSH_TO_MAIN" == "true" ]]; then echo 'push to main'; else echo "branch: $BRANCH_NAME"; fi)"
echo "Parent jobs: $PARENT_JOBS"
echo "Start: $(date)"
echo "============================================================"

# =============================================================================
# Verify Git Setup
# =============================================================================
echo ""
echo "Checking git configuration..."

if ! command -v git &>/dev/null; then
    echo "ERROR: git not found on this node"
    exit 1
fi

# Verify we're in a git repo
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo "ERROR: Not inside a git repository: $PROJECT_ROOT"
    exit 1
fi

# Verify remote exists
if ! git remote get-url "$GIT_REMOTE" &>/dev/null; then
    echo "ERROR: No '$GIT_REMOTE' remote configured"
    echo "Available remotes: $(git remote)"
    echo "Override with: sbatch --export=GIT_REMOTE=<name> ..."
    exit 1
fi

# Verify push access (dry-run)
echo "Testing push access to $GIT_REMOTE..."
if ! git ls-remote --exit-code "$GIT_REMOTE" HEAD &>/dev/null 2>&1; then
    echo "ERROR: Cannot reach $GIT_REMOTE remote"
    echo "Check SSH keys or credential helper on cluster nodes"
    echo "Remote URL: $(git remote get-url "$GIT_REMOTE")"
    exit 1
fi
echo "  Remote accessible: $(git remote get-url "$GIT_REMOTE")"

# Show current state
CURRENT_BRANCH=$(git branch --show-current)
echo "  Current branch: $CURRENT_BRANCH"
echo "  HEAD: $(git rev-parse --short HEAD)"
echo ""

# =============================================================================
# Stage Result Files
# =============================================================================
echo "Staging result files..."

STAGED_COUNT=0

# Helper: stage files matching a pattern, count them
stage_files() {
    local pattern="$1"
    local label="$2"
    local count
    # Use git add with glob — suppresses errors for missing paths
    count=$(git add --dry-run -- $pattern 2>/dev/null | wc -l)
    if [[ "$count" -gt 0 ]]; then
        git add -- $pattern 2>/dev/null
        echo "  + $label ($count files)"
        STAGED_COUNT=$((STAGED_COUNT + count))
    fi
}

# Core MLE results (from step 12)
stage_files "output/mle/*_individual_fits.csv" "MLE individual fits"
stage_files "output/mle/*_group_summary.csv" "MLE group summaries"
stage_files "output/mle/*_performance_summary.json" "MLE performance summaries"
stage_files "output/mle/*_timing_log.csv" "MLE timing logs"

# Model comparison (from step 14)
stage_files "output/model_comparison/*" "model comparison"

# Trauma analysis (from step 15)
stage_files "output/mle_by_trauma/*" "trauma analysis"

# Regression results (from step 16 — .gitignore excludes PNGs)
stage_files "output/regressions/*" "regression results"

# Figures (analysis plots)
stage_files "figures/mle_trauma_analysis/*" "trauma analysis figures"
stage_files "figures/recovery/**/*" "recovery figures"
stage_files "figures/regressions/**/*" "regression figures"

echo ""

# Check if there's anything to commit
if [[ "$STAGED_COUNT" -eq 0 ]]; then
    # Double-check with git status
    if git diff --cached --quiet; then
        echo "No new or modified result files to commit."
        echo "All outputs may already be committed, or fitting produced no files."
        echo "STATUS: NOTHING TO PUSH"
        exit 0
    fi
fi

# Show what will be committed
echo "Staged changes:"
git diff --cached --stat
echo ""

# =============================================================================
# Commit and Push
# =============================================================================
if [[ "$PUSH_TO_MAIN" == "true" ]]; then
    # -------------------------------------------------------------------------
    # Strategy: Push directly to main
    # -------------------------------------------------------------------------
    echo "Pulling latest main with rebase..."
    if ! git pull --rebase "$GIT_REMOTE" main; then
        echo ""
        echo "ERROR: git pull --rebase failed (likely merge conflict)"
        echo "Results are staged but NOT pushed. Manual resolution required."
        exit 1
    fi

    git commit -m "results: pipeline outputs (SLURM jobs: ${PARENT_JOBS})"

    echo "Pushing to main on $GIT_REMOTE..."
    if git push "$GIT_REMOTE" main; then
        echo ""
        echo "STATUS: PUSHED TO MAIN"
    else
        echo ""
        echo "ERROR: git push to main failed"
        exit 1
    fi
else
    # -------------------------------------------------------------------------
    # Strategy: Push to results branch (default, safe)
    # -------------------------------------------------------------------------
    echo "Creating results branch: $BRANCH_NAME"
    git checkout -b "$BRANCH_NAME"

    git commit -m "results: SLURM pipeline outputs (jobs: ${PARENT_JOBS})

Branch: $BRANCH_NAME
Node: ${SLURMD_NODENAME:-$(hostname)}
Date: $(date --iso-8601=seconds 2>/dev/null || date +%Y-%m-%dT%H:%M:%S)

Merge with:
  git checkout main && git merge $BRANCH_NAME"

    echo "Pushing branch to $GIT_REMOTE..."
    if git push -u "$GIT_REMOTE" "$BRANCH_NAME"; then
        echo ""
        echo "STATUS: PUSHED TO BRANCH $BRANCH_NAME"
    else
        echo ""
        echo "ERROR: git push failed"
        echo "Results are committed locally on branch $BRANCH_NAME but NOT pushed."
        echo "Manual push: git push -u origin $BRANCH_NAME"
        exit 1
    fi

    # Return to original branch so subsequent jobs (if any) aren't confused
    echo ""
    echo "Returning to branch: $CURRENT_BRANCH"
    git checkout "$CURRENT_BRANCH"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "Git Push Complete!"
echo "============================================================"
echo "End time: $(date)"
echo ""
if [[ "$PUSH_TO_MAIN" == "true" ]]; then
    echo "Pull locally: git pull $GIT_REMOTE main"
else
    echo "Merge locally: git fetch $GIT_REMOTE && git merge $GIT_REMOTE/$BRANCH_NAME"
fi
echo "============================================================"
