#!/bin/bash
# =============================================================================
# SLURM Batch Script: Stage 06 — Fit Analyses (consolidated)
# =============================================================================
# Parameterized STEP dispatcher for the eight 06_fit_analyses/ scripts:
#   STEP=compare_models                 -> 01_compare_models.py
#   STEP=loo_stacking                   -> 02_compute_loo_stacking.py
#   STEP=model_averaging                -> 03_model_averaging.py
#   STEP=analyze_mle_by_trauma          -> 04_analyze_mle_by_trauma.py --model all
#   STEP=regress_parameters_on_scales   -> 05_regress_parameters_on_scales.py --model all
#   STEP=analyze_winner_heterogeneity   -> 06_analyze_winner_heterogeneity.py
#   STEP=bayesian_level2_effects        -> 07_bayesian_level2_effects.py
#   STEP=manuscript_tables              -> 08_manuscript_tables.py
#
# Consolidates:
#   - cluster/14_analysis.slurm              -> STEP=compare_models | analyze_mle_by_trauma | regress_parameters_on_scales
#   - cluster/21_5_loo_stacking_bms.slurm    -> STEP=loo_stacking
#   - cluster/21_8_model_averaging.slurm     -> STEP=model_averaging
#   - cluster/21_9_manuscript_tables.slurm   -> STEP=manuscript_tables
#
# Pure ArviZ / pandas / statsmodels / matplotlib (no MCMC). CPU-only.
#
# Usage:
#   sbatch --export=ALL,STEP=compare_models cluster/06_fit_analyses.slurm
#   sbatch --export=ALL,STEP=loo_stacking cluster/06_fit_analyses.slurm
#   sbatch --export=ALL,STEP=model_averaging cluster/06_fit_analyses.slurm
#   sbatch --export=ALL,STEP=manuscript_tables cluster/06_fit_analyses.slurm
# =============================================================================

#SBATCH --job-name=p29_06_fit_analyses
#SBATCH --output=cluster/logs/06_fit_analyses_%j.out
#SBATCH --error=cluster/logs/06_fit_analyses_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=comp

set -euo pipefail

STEP="${STEP:-}"

# Scoped overrides per STEP (harmless when unused)
BASELINE_DIR="${BASELINE_DIR:-output/bayesian/21_baseline/}"
L2_DIR="${L2_DIR:-output/bayesian/21_l2/}"
WINNERS_FILE="${WINNERS_FILE:-output/bayesian/21_baseline/winners.txt}"
STACKING_RESULTS="${STACKING_RESULTS:-output/bayesian/21_baseline/loo_stacking_results.csv}"
AUDIT_REPORT="${AUDIT_REPORT:-output/bayesian/21_l2/scale_audit_report.md}"
FIGURES_DIR="${FIGURES_DIR:-figures/21_bayesian/}"
TABLES_DIR="${TABLES_DIR:-output/bayesian/21_tables/}"
SUBSCALE_NC="${SUBSCALE_NC:-output/bayesian/wmrl_m6b_subscale_posterior.nc}"
FORCE_WINNERS="${FORCE_WINNERS:-}"

if [ -z "$STEP" ]; then
    echo "ERROR: STEP env var is required. Valid values:" >&2
    echo "       compare_models | loo_stacking | model_averaging |" >&2
    echo "       analyze_mle_by_trauma | regress_parameters_on_scales |" >&2
    echo "       analyze_winner_heterogeneity | bayesian_level2_effects |" >&2
    echo "       manuscript_tables" >&2
    exit 2
fi

module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

echo "============================================================"
echo "Stage 06 — Fit Analyses: STEP=$STEP"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       ${SLURMD_NODENAME:-$(hostname)}"
echo "Project:    $PROJECT_ROOT"
echo "Start time: $(date)"
echo "============================================================"

if conda activate ds_env 2>/dev/null; then
    echo "Activated ds_env by name"
elif conda activate /scratch/fc37/$USER/conda/envs/ds_env 2>/dev/null; then
    echo "Activated ds_env from /scratch/fc37/$USER/"
elif conda activate /scratch/${PROJECT:-fc37}/$USER/conda/envs/ds_env 2>/dev/null; then
    echo "Activated ds_env from /scratch/${PROJECT:-fc37}/$USER/"
elif conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu (fallback)"
elif conda activate rlwm 2>/dev/null; then
    echo "Activated rlwm (fallback)"
else
    echo "ERROR: Failed to activate conda environment (tried ds_env, rlwm_gpu, rlwm)"
    exit 1
fi

mkdir -p cluster/logs output/model_comparison output/mle_by_trauma output/regressions \
         output/bayesian/21_tables figures/21_bayesian

export PYTHONUNBUFFERED=1

case "${STEP}" in
  compare_models)
    CMD=(python scripts/06_fit_analyses/01_compare_models.py)
    ;;
  loo_stacking)
    CMD=(python scripts/06_fit_analyses/02_compute_loo_stacking.py
         --baseline-dir "$BASELINE_DIR"
         --output-dir "$BASELINE_DIR")
    if [ -n "$FORCE_WINNERS" ]; then
        CMD+=(--force-winners "$FORCE_WINNERS")
    fi
    ;;
  model_averaging)
    CMD=(python scripts/06_fit_analyses/03_model_averaging.py
         --l2-dir "$L2_DIR"
         --stacking-results "$STACKING_RESULTS"
         --winners-file "$WINNERS_FILE"
         --audit-report "$AUDIT_REPORT"
         --output-dir "$L2_DIR")
    ;;
  analyze_mle_by_trauma)
    CMD=(python scripts/06_fit_analyses/04_analyze_mle_by_trauma.py --model all)
    ;;
  regress_parameters_on_scales)
    CMD=(python scripts/06_fit_analyses/05_regress_parameters_on_scales.py --model all)
    ;;
  analyze_winner_heterogeneity)
    CMD=(python scripts/06_fit_analyses/06_analyze_winner_heterogeneity.py)
    ;;
  bayesian_level2_effects)
    CMD=(python scripts/06_fit_analyses/07_bayesian_level2_effects.py)
    ;;
  manuscript_tables)
    CMD=(python scripts/06_fit_analyses/08_manuscript_tables.py
         --baseline-dir "$BASELINE_DIR"
         --l2-dir "$L2_DIR"
         --figures-dir "$FIGURES_DIR"
         --tables-dir "$TABLES_DIR"
         --subscale-nc "$SUBSCALE_NC"
         --no-paper-edit)
    ;;
  *)
    echo "ERROR: unknown STEP='${STEP}'" >&2
    exit 2
    ;;
esac

echo ""
echo "[$(date)] Running: ${CMD[*]}"
"${CMD[@]}"
EXIT_CODE=$?

echo ""
echo "[$(date)] Stage 06 STEP=$STEP finished (exit code: $EXIT_CODE)"

if [[ -f cluster/autopush.sh ]]; then
    source cluster/autopush.sh
fi

exit $EXIT_CODE
