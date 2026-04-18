#!/bin/bash
# =============================================================================
# Phase 21: Principled Bayesian Model Selection Pipeline — Master Orchestrator
# =============================================================================
# Chains all 9 pipeline steps via sbatch --dependency=afterok.
#
# Steps:
#   21.1 Prior predictive (6 parallel jobs, one per model)
#   21.2 Bayesian parameter recovery (per-model array + aggregate)
#   21.3 Baseline hierarchical fits (6 parallel)
#   21.4 Convergence + PPC audit (1 job, after all 21.3 succeed)
#   21.5 PSIS-LOO + stacking + RFX-BMS (1 job, after 21.4 PASS)
#        Exit-2 from 21.5 = INCONCLUSIVE_MULTIPLE winner -> pause for user
#                            review (winner_report.md + winners.txt manual edit
#                            -> resubmit 21.6+ with FORCE_WINNERS=...)
#   21.6 L2 dispatcher (proper SLURM wrapper, 14h time cap to absorb M6b
#                       subscale 12h worst case via internal sbatch --wait)
#   21.7 Scale-fit audit (after dispatcher; UNIFIED EXIT-0 — both
#                          PROCEED_TO_AVERAGING and NULL_RESULT exit 0)
#   21.8 Stacking-weighted model averaging (afterok 21.7; reads YAML header
#         and self-skips on NULL_RESULT)
#   21.9 Manuscript tables + forest plots (afterok 21.8; capstone)
#
# Pre-flight gate:
#   The 2-covariate L2 hook for M3/M5/M6a (plan 21-11) MUST pass its fast
#   pytest suite locally BEFORE any cluster job is submitted. This is a
#   local gate, not a SLURM job — keeps the cluster from chewing through
#   cycles when the L2 hook is broken.
#
# Notes:
# - All dependencies use afterok exclusively. Plan 21-08 unified the
#   exit-0 protocol so PROCEED_TO_AVERAGING and NULL_RESULT both advance
#   the chain naturally (NULL handled inside 21.8/21.9 logic).
# - The L2 dispatcher (21_6_dispatch_l2.slurm) is a proper SLURM wrapper
#   with --time=14:00:00 because the dispatcher uses `sbatch --wait`
#   internally to block on M6b subscale (12h worst case). A `--wrap=` job
#   inherits short default time limits and would be killed by the
#   scheduler before the L2 fits complete.
#
# Usage:
#   bash cluster/21_submit_pipeline.sh                 # full chain
#
# Resubmission after step 21.5 INCONCLUSIVE_MULTIPLE pause:
#   1. Review output/bayesian/21_baseline/winner_report.md
#   2. Edit output/bayesian/21_baseline/winners.txt (or pass FORCE_WINNERS=...)
#   3. Manually submit step 21.6+ chain with the dispatcher SLURM wrapper.
#
# =============================================================================

set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b"

# Strip Windows CRLF from sibling SLURM/sh files (sbatch rejects \r).
sed -i 's/\r$//' cluster/*.slurm cluster/*.sh 2>/dev/null || true

echo "============================================================"
echo "[$(date)] Phase 21 pipeline orchestrator"
echo "============================================================"
echo "Models: $MODELS"
echo ""

# =============================================================================
# Pre-flight gate: 2-covariate L2 infrastructure (plan 21-11)
# =============================================================================
# The 2-cov L2 hook for M3/M5/M6a is the load-bearing addition that
# unblocks step 21.6's L2 refit pathway. If its fast tests fail, the
# entire pipeline downstream of step 21.5 will fail in 21.6 — so we
# fail fast LOCALLY before burning cluster cycles.
echo "[$(date)] Pre-flight: verifying 2-covariate L2 hook (plan 21-11)..."
if ! pytest scripts/fitting/tests/test_numpyro_models_2cov.py -v -k "not slow" --tb=short; then
  echo "[ABORT] plan 21-11 tests failed — 2-covariate L2 hook for M3/M5/M6a is not wired correctly."
  echo "[ABORT] No cluster jobs submitted. Fix scripts/fitting/numpyro_models.py and retry."
  exit 1
fi
echo "[$(date)] Pre-flight OK — 2-covariate L2 hook ready."
echo ""

# =============================================================================
# Step 21.1 — prior predictive (6 parallel jobs)
# =============================================================================
declare -A PPC
for m in $MODELS; do
  PPC[$m]=$(sbatch --parsable --export=ALL,MODEL=$m cluster/21_1_prior_predictive.slurm)
  echo "  [21.1] $m -> job ${PPC[$m]}"
done
PPC_DEP="afterok:$(IFS=:; echo "${PPC[*]}")"

# =============================================================================
# Step 21.2 — Bayesian parameter recovery (array + aggregate per model)
# =============================================================================
declare -A REC_ARRAY
declare -A REC_AGG
for m in $MODELS; do
  REC_ARRAY[$m]=$(sbatch --parsable --dependency=afterok:${PPC[$m]} \
    --export=ALL,MODEL=$m cluster/21_2_recovery.slurm)
  REC_AGG[$m]=$(sbatch --parsable --dependency=afterok:${REC_ARRAY[$m]} \
    --export=ALL,MODEL=$m cluster/21_2_recovery_aggregate.slurm)
  echo "  [21.2] $m -> array job ${REC_ARRAY[$m]}, aggregate ${REC_AGG[$m]}"
done
REC_DEP="afterok:$(IFS=:; echo "${REC_AGG[*]}")"

# =============================================================================
# Step 21.3 — baseline hierarchical fits (6 parallel)
# =============================================================================
declare -A BASE
for m in $MODELS; do
  BASE[$m]=$(sbatch --parsable --dependency=${REC_DEP} \
    --export=ALL,MODEL=$m cluster/21_3_fit_baseline.slurm)
  echo "  [21.3] $m -> job ${BASE[$m]}"
done
BASE_DEP="afterok:$(IFS=:; echo "${BASE[*]}")"

# =============================================================================
# Step 21.4 — convergence + PPC audit (1 job, after all 21.3 succeed)
# =============================================================================
AUDIT_JID=$(sbatch --parsable --dependency=${BASE_DEP} cluster/21_4_baseline_audit.slurm)
echo "  [21.4] convergence audit -> job $AUDIT_JID"

# =============================================================================
# Step 21.5 — LOO + stacking + BMS (1 job, after audit passes)
# =============================================================================
# Tri-state exit code: 0 advance, 1 abort, 2 pause-at-checkpoint
# (INCONCLUSIVE_MULTIPLE -> user reviews winner_report.md + edits
# winners.txt). afterok blocks 21.6 dispatcher on exit 1 OR 2.
LOO_JID=$(sbatch --parsable --dependency=afterok:$AUDIT_JID cluster/21_5_loo_stacking_bms.slurm)
echo "  [21.5] LOO+stacking+BMS -> job $LOO_JID"

# =============================================================================
# Step 21.6 — L2 dispatcher (proper SLURM wrapper, 14h cap)
# =============================================================================
# The dispatcher reads winners.txt (written by 21.5) and submits one L2
# fit per winner via `sbatch --wait` (cluster/21_dispatch_l2_winners.sh).
# Why a proper SLURM wrapper instead of `sbatch --wrap=`?
#   - The dispatcher BLOCKS until each child L2 fit completes (sbatch
#     --wait inside the dispatcher script).
#   - M6b subscale worst case is ~12h.
#   - A `--wrap=` job inherits short default time limits and would be
#     KILLED by the scheduler before the L2 fits complete, silently
#     orphaning the downstream afterok dependency chain.
# 21_6_dispatch_l2.slurm declares --time=14:00:00 to absorb the 12h
# worst case + 2h slack.
DISPATCH_JID=$(sbatch --parsable --dependency=afterok:$LOO_JID \
    cluster/21_6_dispatch_l2.slurm)
echo "  [21.6] L2 dispatch (reads winners.txt) -> job $DISPATCH_JID"

# =============================================================================
# Step 21.7 — scale audit (depends on dispatcher; dispatcher exits only
# after all child L2 fits complete via internal sbatch --wait)
# =============================================================================
AUDIT2_JID=$(sbatch --parsable --dependency=afterok:$DISPATCH_JID cluster/21_7_scale_audit.slurm)
echo "  [21.7] scale audit -> job $AUDIT2_JID"

# =============================================================================
# Step 21.8 — stacking-weighted model averaging
# =============================================================================
# afterok dependency (plan-checker Issue #4 resolution): plan 21-08
# unified the exit-0 protocol so both PROCEED_TO_AVERAGING and
# NULL_RESULT exit 0 from step 21.7. Exit 1 is reserved for genuine
# audit errors (missing winners.txt, corrupt NetCDF, statsmodels
# import failure). afterok is the correct dependency: it blocks on
# real failures but advances on either valid scientific outcome.
# 21_model_averaging.py reads the YAML pipeline_action header itself
# and soft-skips averaging when pipeline_action == NULL_RESULT.
AVG_JID=$(sbatch --parsable --dependency=afterok:$AUDIT2_JID cluster/21_8_model_averaging.slurm)
echo "  [21.8] averaging -> job $AVG_JID"

# =============================================================================
# Step 21.9 — manuscript tables + forest plots (capstone, afterok 21.8)
# =============================================================================
TABLES_JID=$(sbatch --parsable --dependency=afterok:$AVG_JID cluster/21_9_manuscript_tables.slurm)
echo "  [21.9] tables -> job $TABLES_JID"

echo ""
echo "============================================================"
echo "[$(date)] Phase 21 pipeline submitted"
echo "============================================================"
echo "Job summary:"
echo "  21.1 prior predictive : ${PPC[*]}"
echo "  21.2 recovery agg     : ${REC_AGG[*]}"
echo "  21.3 baseline fits    : ${BASE[*]}"
echo "  21.4 audit            : $AUDIT_JID"
echo "  21.5 LOO+stacking+BMS : $LOO_JID"
echo "  21.6 L2 dispatcher    : $DISPATCH_JID"
echo "  21.7 scale audit      : $AUDIT2_JID"
echo "  21.8 averaging        : $AVG_JID"
echo "  21.9 manuscript tables: $TABLES_JID"
echo ""
echo "Monitor progress:  squeue -u \$USER"
echo "Final outputs:     output/bayesian/21_tables/, figures/21_bayesian/"
echo "============================================================"
