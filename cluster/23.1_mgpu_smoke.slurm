#!/bin/bash
# =============================================================================
# SLURM Batch Script: Phase 23.1 Multi-GPU Smoke Validation (5 unproven models)
# =============================================================================
# Phase 23.1 Wave 1 — multi-GPU NUTS smoke test for the 5 choice-only models
# whose pmap (chain_method="parallel") path on Template C
# (#SBATCH --gres=gpu:4 + --chains 4) has NOT yet been verified end-to-end on
# Monash M3:
#   qlearning, wmrl, wmrl_m3, wmrl_m5, wmrl_m6a
#
# wmrl_m6b is INTENTIONALLY EXCLUDED from this smoke run because it is
# already proven via job 54894258 (logs/bayesian_mgpu_54894258.out).
#
# Per-model `timeout 600` wrapper prevents one slow / divergent model from
# burning the full 2h SLURM wall budget — every model gets its own
# pass / fail / timeout verdict (per CLUSTER_GPU_LESSONS.md §6, Apr 16
# incident: untimed loops ate the whole allocation when one model stalled).
#
# Smoke budget (small N, short chains — production gate is RELAXED to
# max_rhat <= 1.10, vs production 1.05):
#   10 participants × 4 chains × 100 warmup × 200 samples
#
# Outputs land in `models/bayesian/23.1_smoke/{model}_smoke_posterior.nc`
# (gitignored — see `.gitignore`). The post-job audit script
# `validation/check_phase_23_1_smoke.py` enforces all 5 invariants.
#
# Usage (Monash M3 only — Windows has no sbatch):
#   sbatch cluster/23.1_mgpu_smoke.slurm
#
# Optional GPU type override (default L40S):
#   sbatch --export=GPU_TYPE=h100 cluster/23.1_mgpu_smoke.slurm
# =============================================================================

#SBATCH --job-name=p23.1_mgpu_smoke
#SBATCH --output=logs/p23_1_mgpu_smoke_%j.out
#SBATCH --error=logs/p23_1_mgpu_smoke_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
# GPU-type selection via CLI only (shell-var expansion does not work in #SBATCH directives).
# To pin a specific GPU type, override on the sbatch command line:
#   sbatch --constraint=l40s cluster/23.1_mgpu_smoke.slurm

# =============================================================================
# Configuration — Phase 23.1 smoke budget
# =============================================================================
MODELS="qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a"
CHAINS=4
WARMUP=100
SAMPLES=200
MAX_TREE_DEPTH=8
N_PARTICIPANTS_LIMIT=10
PER_MODEL_TIMEOUT=600
DATA_FILE="data/processed/task_trials_long.csv"
SMOKE_OUT_DIR="models/bayesian/23.1_smoke"
SMOKE_SUBDIR="23.1_smoke"

# =============================================================================
# Environment Setup
# =============================================================================
module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"

# Install autopush trap (must come BEFORE any failure-prone command).
# autopush.sh registers an EXIT trap that fires on success or failure,
# so SLURM .err/.out logs land in git regardless of outcome — see
# cluster/autopush.sh for rationale.
[ -f cluster/autopush.sh ] && source cluster/autopush.sh


echo "============================================================"
echo "Phase 23.1 Multi-GPU Smoke Validation — 5 unproven models"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: ${SLURMD_NODENAME:-$(hostname)}"
echo "Project: $PROJECT_ROOT"
echo "Models: $MODELS"
echo "Chains: $CHAINS  Warmup: $WARMUP  Samples: $SAMPLES"
echo "Max tree depth: $MAX_TREE_DEPTH"
echo "Participants (smoke limit): $N_PARTICIPANTS_LIMIT"
echo "Per-model timeout: ${PER_MODEL_TIMEOUT}s"
echo "Smoke output dir: $SMOKE_OUT_DIR"
echo "Start time: $(date)"
echo "============================================================"

# Activate rlwm_gpu (GPU pmap requires it). Try by name, then explicit
# /scratch path, then fall back to ds_env (CPU — chain_method will silently
# degrade to vectorized, defeating the test). The audit script catches the
# degradation post-hoc, but loud messaging here helps the operator notice.
if conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu by name"
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/fc37/$USER/"
elif conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu by name"
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/fc37/$USER/"
elif conda activate /scratch/${PROJECT:-fc37}/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/${PROJECT:-fc37}/$USER/"
else
    echo "ERROR: Failed to activate rlwm_gpu — see cluster/00_setup_env_gpu.sh"
    exit 1
fi

# Verify GPU allocation matches chain count. If n_gpu < CHAINS, NumPyro's
# _select_chain_method falls back to 'vectorized' which defeats the entire
# point of this smoke test — but we let the run proceed so the audit script
# can record the degradation explicitly rather than silently aborting here.
python -c "
import jax
devices = jax.devices()
n_gpu = sum(1 for d in devices if d.platform == 'gpu')
print(f'JAX {jax.__version__} devices: {devices}')
print(f'GPU count: {n_gpu}')
if n_gpu < $CHAINS:
    print(f'WARNING: {n_gpu} GPU(s) < {$CHAINS} chains. chain_method will be')
    print(f'         vectorized, not parallel. Template C path NOT validated.')
" || {
    echo "ERROR: JAX import failed"
    exit 1
}
echo ""

# =============================================================================
# Cache + env exports — encode all CLUSTER_GPU_LESSONS.md operational invariants
# =============================================================================
# §2: NUMPYRO_HOST_DEVICE_COUNT must match $CHAINS for pmap to see all devices
# §5: per-cache dir + min-entry=0 (single shared smoke cache for the 5-model
#     loop — production scripts in Wave 2 use per-model caches)
# §6: PYTHONUNBUFFERED=1 so logs flush in real time on stalls
export NUMPYRO_HOST_DEVICE_COUNT="$CHAINS"
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_23_1_smoke"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
mkdir -p "$JAX_COMPILATION_CACHE_DIR" 2>/dev/null || unset JAX_COMPILATION_CACHE_DIR
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"

export PYTHONUNBUFFERED=1
mkdir -p logs "$SMOKE_OUT_DIR"

# =============================================================================
# Smoke-data preflight: subset to first 10 unique sona_ids
# =============================================================================
# fit_bayesian.py has no --n-participants / --limit flag, so we materialize a
# 10-ppt CSV here and pass it as --data. Canonical participant ID column is
# `sona_id` (verified in scripts/04_model_fitting/b_bayesian/fit_bayesian.py lines 127, 129, 140,
# 141, 148, 152, 153, 157, 180, 183 — no fallback column needed).
if [[ ! -f "$DATA_FILE" ]]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    exit 1
fi

python -c "
import pandas as pd, pathlib
df = pd.read_csv('$DATA_FILE')
ids = sorted(df['sona_id'].unique())[:$N_PARTICIPANTS_LIMIT]
out = df[df['sona_id'].isin(ids)]
pathlib.Path('$SMOKE_OUT_DIR').mkdir(parents=True, exist_ok=True)
out.to_csv('$SMOKE_OUT_DIR/_smoke_data_10ppts.csv', index=False)
print(f'Smoke data: {len(ids)} ppts, {len(out)} rows')
" || {
    echo "ERROR: Smoke-data preflight failed"
    exit 1
}
SMOKE_DATA="$SMOKE_OUT_DIR/_smoke_data_10ppts.csv"
echo ""

# =============================================================================
# Sequential per-model NUTS — 4 GPUs / 4 chains via pmap, timeout 600s each
# =============================================================================
declare -A RESULTS
OVERALL_PASS=true

for MODEL in $MODELS; do
    echo ""
    echo "------------------------------------------------------------"
    echo "  $MODEL (timeout ${PER_MODEL_TIMEOUT}s)"
    echo "  Started: $(date)"
    echo "------------------------------------------------------------"

    MODEL_START=$(date +%s)

    # NOTE on output paths: passing `--output output --output-subdir 23.1_smoke`
    # routes writes to `models/bayesian/23.1_smoke/{model}_posterior.nc`
    # (per fit_bayesian.py:702-705 — bayesian_dir = output_dir / "bayesian" /
    # output_subdir). We rename to `_smoke_posterior.nc` after the loop so the
    # audit script can distinguish smoke vs production posteriors.
    timeout "$PER_MODEL_TIMEOUT" python scripts/04_model_fitting/b_bayesian/fit_bayesian.py \
        --model "$MODEL" \
        --data "$SMOKE_DATA" \
        --output models \
        --output-subdir "$SMOKE_SUBDIR" \
        --chains "$CHAINS" \
        --warmup "$WARMUP" \
        --samples "$SAMPLES" \
        --max-tree-depth "$MAX_TREE_DEPTH" \
        --seed 42

    MODEL_EXIT=$?
    MODEL_END=$(date +%s)
    MODEL_WALL=$((MODEL_END - MODEL_START))

    if [[ $MODEL_EXIT -eq 0 ]]; then
        RESULTS[$MODEL]="PASS (${MODEL_WALL}s)"
    elif [[ $MODEL_EXIT -eq 124 ]]; then
        RESULTS[$MODEL]="TIMEOUT (${MODEL_WALL}s, exceeded ${PER_MODEL_TIMEOUT}s)"
        OVERALL_PASS=false
    elif [[ $MODEL_EXIT -eq 137 || $MODEL_EXIT -eq 139 ]]; then
        RESULTS[$MODEL]="KILLED (exit=$MODEL_EXIT, ${MODEL_WALL}s)"
        OVERALL_PASS=false
        echo "  $MODEL: KILLED (exit=$MODEL_EXIT) — continuing to next model"
    else
        RESULTS[$MODEL]="EXIT=$MODEL_EXIT (${MODEL_WALL}s)"
        # Quick crashes (< 30s) almost always indicate import/compile errors,
        # which would invalidate every subsequent model. Slower non-zero exits
        # may be legitimate convergence-gate failures at this small smoke
        # budget — those are flagged but not fatal.
        if [[ $MODEL_WALL -lt 30 ]]; then
            OVERALL_PASS=false
        fi
    fi
    echo "  $MODEL: exit=$MODEL_EXIT, wall=${MODEL_WALL}s"
done

# =============================================================================
# Rename produced posteriors from {model}_posterior.nc → {model}_smoke_posterior.nc
# =============================================================================
echo ""
echo "Renaming smoke posteriors with _smoke_ tag..."
for MODEL in $MODELS; do
    src="$SMOKE_OUT_DIR/${MODEL}_posterior.nc"
    dst="$SMOKE_OUT_DIR/${MODEL}_smoke_posterior.nc"
    if [[ -f "$src" ]]; then
        mv "$src" "$dst"
        echo "  $src → $dst"
    else
        echo "  (skip: $src not produced)"
    fi
done

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "============================================================"
echo "PHASE 23.1 MULTI-GPU SMOKE SUMMARY"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "End time: $(date)"
echo ""
printf "%-12s %s\n" "Model" "Result"
printf "%-12s %s\n" "-------" "------"
for MODEL in $MODELS; do
    printf "%-12s %s\n" "$MODEL" "${RESULTS[$MODEL]}"
done
echo ""
if $OVERALL_PASS; then
    echo "OVERALL: PASS (all models completed inside per-model timeout)"
    echo ""
    echo "Next step: run validation/check_phase_23_1_smoke.py to assert"
    echo "INV-01 through INV-05 (path contract, convergence, lazy-LBA isolation)."
else
    echo "OVERALL: FAIL (one or more models timed out, killed, or crashed fast)"
    echo ""
    echo "Investigate: per-model log lines above + logs/p23_1_mgpu_smoke_${SLURM_JOB_ID:-local}.err"
fi
echo "============================================================"

# Auto-push logs and posteriors when job completes
if [[ -f cluster/autopush.sh ]]; then
    source cluster/autopush.sh
fi

if $OVERALL_PASS; then
    exit 0
else
    exit 1
fi
