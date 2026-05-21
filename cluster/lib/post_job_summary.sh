#!/usr/bin/env bash
# =============================================================================
# post_job_summary.sh -- Print job metrics after completion
# =============================================================================
# Source at the end of any .slurm script (or call from an EXIT trap) to get
# a compact accounting summary from sacct.
#
# Usage:
#   source cluster/lib/post_job_summary.sh
#
# Safe to source outside SLURM (prints a skip message and returns 0).
# =============================================================================
set -euo pipefail

JOB_ID="${SLURM_JOB_ID:-unknown}"
if [[ "${JOB_ID}" == "unknown" ]]; then
    echo "post_job_summary: Not running inside a SLURM job. Skipping."
    exit 0
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  JOB SUMMARY: ${JOB_ID}"
echo "════════════════════════════════════════════════════════════════"

sacct -j "${JOB_ID}" \
    --format=JobID,JobName%30,Elapsed,MaxRSS,MaxVMSize,ExitCode,State \
    --noheader --parsable2 \
| while IFS='|' read -r jobid name elapsed maxrss maxvm exitcode state; do
    if [[ "${jobid}" == "${JOB_ID}" ]]; then
        echo "  Job Name:    ${name}"
        echo "  Elapsed:     ${elapsed}"
        echo "  Max RSS:     ${maxrss:-N/A}"
        echo "  Max VM Size: ${maxvm:-N/A}"
        echo "  Exit Code:   ${exitcode}"
        echo "  State:       ${state}"
    fi
done

echo "════════════════════════════════════════════════════════════════"
echo ""
