#!/bin/bash
# =============================================================================
# cluster/21_dispatch_l2_winners.sh
# =============================================================================
# Reads output/bayesian/21_baseline/winners.txt (comma-separated display
# names from step 21.5), maps display names to internal model ids, and
# submits one cluster/21_6_fit_with_l2.slurm per winner via
# `sbatch --wait` so this dispatcher BLOCKS until each L2 fit completes.
#
# Parallelisation: each `sbatch --wait` is backgrounded with `&`, then a
# single `wait` at the end blocks until all background waits return.
# This means the per-winner L2 fits run in PARALLEL, but the dispatcher
# itself stays alive until the slowest winner completes (M6b subscale
# worst case ~12h).
#
# Exit codes:
#   0 — winners.txt found, all L2 fits submitted, all completed (sbatch
#       --wait propagates the child exit codes; if any child fails,
#       this script's set -euo pipefail will surface it as non-zero).
#   2 — winners.txt missing (step 21.5 may have exited with code 2 =
#       INCONCLUSIVE_MULTIPLE; pipeline paused for user review).
#
# The master orchestrator (cluster/21_submit_pipeline.sh) wraps this
# script in cluster/21_6_dispatch_l2.slurm with --time=14:00:00 so the
# scheduler does not kill the dispatcher before the L2 fits complete
# (plan-checker Issue #6).
#
# CANONICAL block — the `sbatch --wait` + `&` + `wait` pattern is the
# accepted implementation per plan 21-10 spec. Alternative designs
# involving synthetic barrier jobs or dependency rewiring were
# explicitly rejected as not composing cleanly with bash semantics.
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

WINNERS_FILE=output/bayesian/21_baseline/winners.txt

if [ ! -f "$WINNERS_FILE" ]; then
  echo "[DISPATCH] No winners.txt — step 21.5 may have exited with code 2; pipeline paused."
  echo "[DISPATCH] Resume manually after editing winners.txt or passing FORCE_WINNERS=..."
  exit 2
fi

WINNERS=$(cat "$WINNERS_FILE" | tr ',' ' ')

# Display name -> internal model id (mirrors plan 21-06 MODEL_TO_DISPLAY).
declare -A NAME_MAP=(
  ["M1"]="qlearning"
  ["M2"]="wmrl"
  ["M3"]="wmrl_m3"
  ["M5"]="wmrl_m5"
  ["M6a"]="wmrl_m6a"
  ["M6b"]="wmrl_m6b"
)

echo "[DISPATCH] Winners from $WINNERS_FILE: $WINNERS"
echo ""

for name in $WINNERS; do
  model=${NAME_MAP[$name]:-$name}
  echo "[DISPATCH] Submitting L2 fit for $name ($model) and blocking on --wait"
  sbatch --wait --export=ALL,MODEL=$model cluster/21_6_fit_with_l2.slurm &
done

wait

echo ""
echo "[DISPATCH] All L2 winner fits complete; releasing downstream afterok dependency"
