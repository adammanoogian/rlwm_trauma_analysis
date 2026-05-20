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
#   # Email notification — two independent mechanisms, use either or both:
#   #   (1) SLURM-native (terse scheduler email, fires even on script crash):
#   sbatch --mail-type=END,FAIL --mail-user=adam.manoogian@monash.edu \
#          --dependency=afterany:JID cluster/99_push_results.slurm
#   #   (2) Custom rich email at exit (branch, commit hash, file counts):
#   sbatch --export=ALL,NOTIFY_EMAIL=adam.manoogian@monash.edu \
#          --dependency=afterany:JID cluster/99_push_results.slurm
#
# History (2026-05-05):
#   The per-job autopush.sh trap (commit e10cd66, deleted 2026-05-05) was
#   replaced by this dependency-chained Pattern A push. The trap raced on
#   the shared NFS-mounted PROJECT_ROOT when fan-out fitting jobs all hit
#   `git push` concurrently — auto-stash + rebase conflicts dropped the
#   files visible in the "one job saved" / "adding non autopushed"
#   recovery commits. One push job at the end + afterany dependency gives
#   the same "logs-back-on-failure" guarantee without the race.
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
#SBATCH --output=logs/push_results_%j.out
#SBATCH --error=logs/push_results_%j.err
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
# Optional: rich custom email at exit (branch, commit hash, file counts).
# Requires `mail`/`mailx`/`sendmail` on the compute node. Empty = disabled.
NOTIFY_EMAIL="${NOTIFY_EMAIL:-}"

# =============================================================================
# Environment Setup (minimal — only need git, no conda/python)
# =============================================================================
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

mkdir -p logs

# =============================================================================
# Email Notification (custom body, fired via trap at exit)
# =============================================================================
PUSH_STATUS="UNKNOWN"
PUSH_SUMMARY=""

_send_notify_email() {
    local exit_code=$?
    [[ -z "$NOTIFY_EMAIL" ]] && return 0
    local mailer=""
    for cand in mail mailx sendmail; do
        if command -v "$cand" &>/dev/null; then mailer="$cand"; break; fi
    done
    if [[ -z "$mailer" ]]; then
        echo "  NOTIFY_EMAIL set but no mail/mailx/sendmail found — skipping custom email"
        return 0
    fi

    local subject="[SLURM ${SLURM_JOB_ID:-local}] push_results: ${PUSH_STATUS}"
    local body
    body=$(cat <<EOF
SLURM push_results job ${SLURM_JOB_ID:-local} finished with status: ${PUSH_STATUS}
Exit code: ${exit_code}
Node:      ${SLURMD_NODENAME:-$(hostname)}
Project:   ${PROJECT_ROOT}
Remote:    ${GIT_REMOTE}
Parent:    ${PARENT_JOBS}
End time:  $(date)

${PUSH_SUMMARY}

Log: logs/push_results_${SLURM_JOB_ID:-local}.out
EOF
)
    case "$mailer" in
        mail|mailx)
            printf '%s\n' "$body" | "$mailer" -s "$subject" "$NOTIFY_EMAIL" \
                && echo "  notify email sent to $NOTIFY_EMAIL" \
                || echo "  WARN: $mailer exit non-zero sending to $NOTIFY_EMAIL"
            ;;
        sendmail)
            {
                printf 'To: %s\nSubject: %s\n\n' "$NOTIFY_EMAIL" "$subject"
                printf '%s\n' "$body"
            } | sendmail -t "$NOTIFY_EMAIL" \
                && echo "  notify email sent to $NOTIFY_EMAIL (sendmail)" \
                || echo "  WARN: sendmail failed"
            ;;
    esac
}
trap _send_notify_email EXIT

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
    PUSH_STATUS="FAILED_PREFLIGHT"
    PUSH_SUMMARY="git not found on compute node"
    exit 1
fi

# Verify we're in a git repo
if ! git rev-parse --is-inside-work-tree &>/dev/null; then
    echo "ERROR: Not inside a git repository: $PROJECT_ROOT"
    PUSH_STATUS="FAILED_PREFLIGHT"
    PUSH_SUMMARY="Not inside a git repository: ${PROJECT_ROOT}"
    exit 1
fi

# Verify remote exists
if ! git remote get-url "$GIT_REMOTE" &>/dev/null; then
    echo "ERROR: No '$GIT_REMOTE' remote configured"
    echo "Available remotes: $(git remote)"
    echo "Override with: sbatch --export=GIT_REMOTE=<name> ..."
    PUSH_STATUS="FAILED_PREFLIGHT"
    PUSH_SUMMARY="No '${GIT_REMOTE}' remote configured. Available: $(git remote | tr '\n' ' ')"
    exit 1
fi

# Verify push access (dry-run)
echo "Testing push access to $GIT_REMOTE..."
if ! git ls-remote --exit-code "$GIT_REMOTE" HEAD &>/dev/null 2>&1; then
    echo "ERROR: Cannot reach $GIT_REMOTE remote"
    echo "Check SSH keys or credential helper on cluster nodes"
    echo "Remote URL: $(git remote get-url "$GIT_REMOTE")"
    PUSH_STATUS="FAILED_PREFLIGHT"
    PUSH_SUMMARY="Cannot reach ${GIT_REMOTE} ($(git remote get-url "$GIT_REMOTE")). Check SSH keys on compute nodes."
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
stage_files "models/mle/*_individual_fits.csv" "MLE individual fits"
stage_files "models/mle/*_group_summary.csv" "MLE group summaries"
stage_files "models/mle/*_performance_summary.json" "MLE performance summaries"
stage_files "models/mle/*_timing_log.csv" "MLE timing logs"

# Model comparison (from step 14)
stage_files "reports/tables/model_comparison/*" "model comparison"

# Trauma analysis (from step 15)
stage_files "reports/tables/mle_by_trauma/*" "trauma analysis"

# Regression results (from step 16 — .gitignore excludes PNGs)
stage_files "reports/tables/regressions/*" "regression results"

# Figures (analysis plots)
stage_files "reports/figures/mle_trauma_analysis/*" "trauma analysis figures"
stage_files "reports/figures/recovery/**/*" "recovery figures"
stage_files "reports/figures/regressions/**/*" "regression figures"

echo ""

# Check if there's anything to commit
if [[ "$STAGED_COUNT" -eq 0 ]]; then
    # Double-check with git status
    if git diff --cached --quiet; then
        echo "No new or modified result files to commit."
        echo "All outputs may already be committed, or fitting produced no files."
        echo "STATUS: NOTHING TO PUSH"
        PUSH_STATUS="NOTHING_TO_PUSH"
        PUSH_SUMMARY="No result files changed since last push."
        exit 0
    fi
fi

# Show what will be committed
echo "Staged changes:"
git diff --cached --stat
echo ""
STAGED_STAT=$(git diff --cached --stat | tail -n 1)

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
        PUSH_STATUS="FAILED_REBASE_CONFLICT"
        PUSH_SUMMARY="Results staged but NOT pushed (rebase on main failed)."
        exit 1
    fi

    git commit -m "results: pipeline outputs (SLURM jobs: ${PARENT_JOBS})"
    COMMIT_HASH=$(git rev-parse --short HEAD)

    echo "Pushing to main on $GIT_REMOTE..."
    if git push "$GIT_REMOTE" main; then
        echo ""
        echo "STATUS: PUSHED TO MAIN"
        PUSH_STATUS="PUSHED_TO_MAIN"
        PUSH_SUMMARY=$(printf 'Commit: %s\nStaged: %s\nPull locally: git pull %s main' \
                       "$COMMIT_HASH" "$STAGED_STAT" "$GIT_REMOTE")
    else
        echo ""
        echo "ERROR: git push to main failed"
        PUSH_STATUS="FAILED_PUSH"
        PUSH_SUMMARY="Commit ${COMMIT_HASH} made locally but push to main failed."
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
    COMMIT_HASH=$(git rev-parse --short HEAD)

    echo "Pushing branch to $GIT_REMOTE..."
    if git push -u "$GIT_REMOTE" "$BRANCH_NAME"; then
        echo ""
        echo "STATUS: PUSHED TO BRANCH $BRANCH_NAME"
        PUSH_STATUS="PUSHED_TO_BRANCH"
        PUSH_SUMMARY=$(printf 'Branch: %s\nCommit: %s\nStaged: %s\nMerge locally: git fetch %s && git merge %s/%s' \
                       "$BRANCH_NAME" "$COMMIT_HASH" "$STAGED_STAT" \
                       "$GIT_REMOTE" "$GIT_REMOTE" "$BRANCH_NAME")
    else
        echo ""
        echo "ERROR: git push failed"
        echo "Results are committed locally on branch $BRANCH_NAME but NOT pushed."
        echo "Manual push: git push -u $GIT_REMOTE $BRANCH_NAME"
        PUSH_STATUS="FAILED_PUSH"
        PUSH_SUMMARY="Branch ${BRANCH_NAME} committed locally as ${COMMIT_HASH} but push failed. Manual: git push -u ${GIT_REMOTE} ${BRANCH_NAME}"
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
