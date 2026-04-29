#!/bin/bash
# =============================================================================
# SLURM Batch Script: Stage 04b — Choice-Only Hierarchical Bayesian Fit (GPU, Template C)
# =============================================================================
# GPU multi-GPU pmap variant of cluster/04b_bayesian_cpu.slurm.
#
# Template C (4 GPUs + chain_method=parallel via pmap) per
# docs/CLUSTER_GPU_LESSONS.md §6. Auto-selected by
# src/rlwm/fitting/numpyro_models._select_chain_method when
# jax.local_device_count() >= n_chains (we request --gres=gpu:4 and default
# --chains 4 so pmap takes the parallel path instead of the slower
# time-multiplexed vectorized path).
#
# Wall-clock baseline: M6b 4-GPU pmap job 54894258
# (logs/bayesian_mgpu_54894258.out, 2026-04-18) completed in ~3h; other
# choice-only models are proportionally faster.
#
# M6b subscale walltime override via orchestrator:
#   sbatch --time=36:00:00 --export=ALL,MODEL=wmrl_m6b,SUBSCALE=1 \
#     cluster/04b_bayesian_gpu.slurm
#
# CPU sibling cluster/04b_bayesian_cpu.slurm is preserved unchanged and
# remains dispatchable via USE_CPU=1 in the master orchestrator
# (cluster/submit_all.sh).
#
# Usage:
#   sbatch cluster/04b_bayesian_gpu.slurm                                  # Default model (wmrl_m3)
#   sbatch --export=ALL,MODEL=qlearning cluster/04b_bayesian_gpu.slurm     # Q-Learning
#   sbatch --export=ALL,MODEL=wmrl_m6b,SUBSCALE=1 \
#     --time=36:00:00 cluster/04b_bayesian_gpu.slurm                       # M6b subscale
#   sbatch --export=ALL,MODEL=wmrl_m3,LIMIT=5 \
#     cluster/04b_bayesian_gpu.slurm                                       # Quick test
#
# Optional GPU type override (default L40S; pins JIT cache to one arch per
# CLUSTER_GPU_LESSONS.md §5):
#   sbatch --export=GPU_TYPE=a100 cluster/04b_bayesian_gpu.slurm
#
# Outputs saved to (Phase-21 pipeline subdir, read by stage 05 baseline_audit):
#   - models/bayesian/21_baseline/${MODEL}_individual_fits.csv
#   - models/bayesian/21_baseline/${MODEL}_posterior.nc
#   - models/bayesian/21_baseline/${MODEL}_shrinkage_report.md
#   - models/bayesian/21_baseline/${MODEL}_ppc_results.csv
# =============================================================================

#SBATCH --job-name=rlwm-bayesian-gpu-mgpu
#SBATCH --output=logs/04b_bayesian_gpu_%j.out
#SBATCH --error=logs/04b_bayesian_gpu_%j.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu
# GPU-type selection via CLI only (shell-var expansion does not work in #SBATCH directives).
# Override on command line if needed:  sbatch --constraint=l40s cluster/04b_bayesian_gpu.slurm

# =============================================================================
# Configuration — override via --export when submitting
# =============================================================================
MODEL="${MODEL:-wmrl_m3}"
LIMIT="${LIMIT:-}"
CHAINS="${CHAINS:-4}"
WARMUP="${WARMUP:-1000}"
SAMPLES="${SAMPLES:-2000}"
SEED="${SEED:-42}"
MAX_TREE_DEPTH="${MAX_TREE_DEPTH:-8}"
SUBSCALE="${SUBSCALE:-0}"  # 1 enables --subscale flag (M6b subscale variant)

# =============================================================================
# Environment Setup (M3 specific)
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
echo "Stage 04b (GPU, Template C) — Hierarchical Bayesian Fit"
echo "============================================================"
echo "Job ID:         ${SLURM_JOB_ID:-local}"
echo "Node:           ${SLURMD_NODENAME:-$(hostname)}"
echo "Partition:      ${SLURM_JOB_PARTITION:-gpu}"
echo "GPU(s):         ${CUDA_VISIBLE_DEVICES:-none}"
echo "Project:        $PROJECT_ROOT"
echo "Model:          $MODEL"
echo "Chains:         $CHAINS"
echo "Warmup:         $WARMUP"
echo "Samples:        $SAMPLES"
echo "Seed:           $SEED"
echo "Max tree depth: $MAX_TREE_DEPTH"
echo "Subscale:       $SUBSCALE"
echo "Limit:          ${LIMIT:-<all participants>}"
echo "Start time:     $(date)"
echo "============================================================"

# Activate rlwm_gpu conda environment. Try by name (uses user's .condarc),
# then explicit /scratch paths. NO ds_env fallback — the whole point of this
# script is Template C pmap on GPU; ds_env would silently degrade the
# chain_method to vectorized and defeat the purpose.
if conda activate rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu by name"
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/fc37/$USER/"
elif conda activate /scratch/${PROJECT:-fc37}/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "Activated rlwm_gpu from /scratch/${PROJECT:-fc37}/$USER/"
else
    echo "ERROR: Failed to activate rlwm_gpu — GPU pipeline cannot run on ds_env (CPU-only)"
    echo "       Create it first with: bash cluster/00_setup_env_gpu.sh"
    echo "       Or use USE_CPU=1 with cluster/submit_all.sh for the CPU path."
    exit 1
fi

# Re-register editable rlwm install. The egg-link in site-packages records an
# absolute path, so it goes stale if the repo was moved (e.g. folder flatten)
# or if this is a fresh compute-node-cache that lost its installed pointer.
# Idempotent (~3-5s on hot install). Mirrors cluster/00_preflight.slurm
# step [2/5]; without this, `import rlwm` raises ModuleNotFoundError on every
# job and silently breaks the afterok chain via DependencyNeverSatisfied
# (cf. JID batch 55046494-55046528 cascade, 2026-04-29).
pip install -e . --quiet --no-deps 2>/dev/null || \
    echo "WARNING: pip install -e . --no-deps failed; 'import rlwm' may fail downstream"

# Verify GPU allocation matches chain count. If n_gpu < CHAINS, NumPyro's
# _select_chain_method falls back to 'vectorized' which defeats the entire
# point of Template C.
CHAINS_NUM=$CHAINS
python -c "
import jax
devices = jax.devices()
n_gpu = sum(1 for d in devices if d.platform == 'gpu')
print(f'JAX {jax.__version__} devices: {devices}')
print(f'GPU count: {n_gpu}')
if n_gpu < ${CHAINS_NUM}:
    print(f'WARNING: {n_gpu} GPU(s) < ${CHAINS_NUM} chains. chain_method will degrade to vectorized.')
" || { echo 'ERROR: JAX import failed'; exit 1; }
echo ""

# =============================================================================
# Cache + env exports — encode CLUSTER_GPU_LESSONS.md operational invariants
# =============================================================================
# §2: NUMPYRO_HOST_DEVICE_COUNT must match $CHAINS so pmap sees all devices
# §5: per-model + _gpu-suffixed cache so CPU/GPU + cross-model don't contaminate
# §6: PYTHONUNBUFFERED=1 so logs flush in real time on stalls
export NUMPYRO_HOST_DEVICE_COUNT="$CHAINS"
_PROJECT="${PROJECT:-fc37}"
export JAX_COMPILATION_CACHE_DIR="/scratch/${_PROJECT}/${USER}/.jax_cache_04b_bayesian_gpu_${MODEL}"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export PYTHONUNBUFFERED=1
mkdir -p "$JAX_COMPILATION_CACHE_DIR" || {
    echo 'WARNING: cache dir create failed'
    unset JAX_COMPILATION_CACHE_DIR
}
echo "JAX cache: ${JAX_COMPILATION_CACHE_DIR:-disabled}"
echo ""

# Create output + log directories
mkdir -p logs
mkdir -p models/bayesian/21_baseline

# Clear any stale per-model artefacts from a previous run (different cohort N,
# aborted fit, etc.). Without this, a fit failure mid-rerun would leave the
# previous run's posterior in place and stage 05 would silently audit it.
echo "[$(date)] Clearing stale ${MODEL} artefacts from models/bayesian/21_baseline/"
for stale in "models/bayesian/21_baseline/${MODEL}_individual_fits.csv" \
             "models/bayesian/21_baseline/${MODEL}_posterior.nc" \
             "models/bayesian/21_baseline/${MODEL}_shrinkage_report.md" \
             "models/bayesian/21_baseline/${MODEL}_ppc_results.csv"; do
    if [[ -f "$stale" ]]; then
        echo "  removing $stale"
        rm -f "$stale"
    fi
done

# =============================================================================
# Verify required input files
# =============================================================================
DATA_FILE="data/processed/task_trials_long.csv"
if [[ ! -f "$DATA_FILE" ]]; then
    echo "ERROR: Data file not found: $DATA_FILE"
    echo "Run the data parsing script first:"
    echo "  python scripts/01_data_preprocessing/01_parse_raw_data.py"
    exit 1
fi

if [ "$SUBSCALE" = "1" ]; then
    echo "Checking subscale-required input files..."
    for f in data/processed/task_trials_long.csv data/processed/summary_participant_metrics.csv; do
        if [ -f "$f" ]; then
            echo "  OK: $f ($(du -h "$f" | cut -f1))"
        else
            echo "  ERROR: $f is MISSING — required for subscale L2 regression"
            exit 1
        fi
    done
fi

# =============================================================================
# Build fit_baseline.py command (Phase-21 pipeline entry)
# =============================================================================
# IMPORTANT: This invokes `fit_baseline.py` (NOT `fit_bayesian.py`). The
# pipeline entry forces --output-subdir=21_baseline so artefacts land in
# models/bayesian/21_baseline/ where stage 05 baseline_audit reads them,
# AND adds a load-bearing expected.exists() check that converts silent
# convergence-gate failures into a non-zero SLURM exit. fit_bayesian.py is
# the ad-hoc CLI and writes to bare models/bayesian/, which would silently
# orphan the afterok chain.
#
# fit_baseline.py does NOT accept --subscale or --limit (subscale runs
# through cluster/04c_level2_gpu.slurm, not 04b). If you set those env
# vars they will be ignored here.
#
# NO --use-gpu flag: pmap selection is automatic via
# numpyro_models._select_chain_method when n_devices >= n_chains.
CMD=(python scripts/04_model_fitting/b_bayesian/fit_baseline.py
     --model "$MODEL"
     --data "$DATA_FILE"
     --chains "$CHAINS"
     --warmup "$WARMUP"
     --samples "$SAMPLES"
     --seed "$SEED"
     --max-tree-depth "$MAX_TREE_DEPTH")

if [ "$SUBSCALE" = "1" ] || [[ -n "$LIMIT" ]]; then
    echo "WARNING: SUBSCALE/LIMIT env vars are ignored under fit_baseline.py."
    echo "         Subscale fits use cluster/04c_level2_gpu.slurm; for ad-hoc"
    echo "         single-model fits with --limit, invoke fit_bayesian.py directly."
fi

# =============================================================================
# Run hierarchical fit (GPU Template C)
# =============================================================================
echo "============================================================"
echo "Fitting Bayesian model: $MODEL (Template C, 4 GPUs + pmap)"
echo "Started: $(date)"
echo "============================================================"

# Start background GPU monitoring (log every 5 min during fitting)
if command -v nvidia-smi &>/dev/null; then
    (while true; do
        echo "[GPU $(date +%H:%M:%S)] $(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits 2>/dev/null | awk -F', ' '{printf "util=%s%% mem=%s/%sMB temp=%sC", $1, $2, $3, $4}')"
        sleep 300
    done) &
    GPU_MONITOR_PID=$!
fi

echo "[$(date)] Starting ${MODEL} hierarchical Bayesian fit${SUBSCALE:+ (subscale=$SUBSCALE)}"
echo "  Command: ${CMD[*]}"
echo ""

"${CMD[@]}"
EXIT_CODE=$?

# Stop GPU monitoring
[[ -n "$GPU_MONITOR_PID" ]] && kill "$GPU_MONITOR_PID" 2>/dev/null && wait "$GPU_MONITOR_PID" 2>/dev/null
unset GPU_MONITOR_PID

echo ""
echo "[$(date)] ${MODEL} hierarchical fit finished (exit code: $EXIT_CODE)"

# =============================================================================
# Post-fit expected-artifacts audit (mirror of 04b_bayesian_cpu.slurm)
# =============================================================================
if [ $EXIT_CODE -eq 0 ]; then
    echo "Expected outputs (under models/bayesian/21_baseline/):"
    MISSING_ARTIFACTS=0
    for f in "models/bayesian/21_baseline/${MODEL}_individual_fits.csv" \
             "models/bayesian/21_baseline/${MODEL}_posterior.nc" \
             "models/bayesian/21_baseline/${MODEL}_shrinkage_report.md" \
             "models/bayesian/21_baseline/${MODEL}_ppc_results.csv"; do
        if [ -f "$f" ]; then
            echo "  [OK] $f ($(du -h "$f" | cut -f1))"
        else
            echo "  [MISSING] $f"
            MISSING_ARTIFACTS=$((MISSING_ARTIFACTS + 1))
        fi
    done
    if [ $MISSING_ARTIFACTS -gt 0 ]; then
        echo ""
        echo "*** ERROR: ${MODEL} fit returned exit=0 but $MISSING_ARTIFACTS expected artefact(s) MISSING ***"
        echo "*** Treating as failure for afterok chain integrity (stage 05 reads from 21_baseline/). ***"
        EXIT_CODE=1
    fi
else
    echo "*** ERROR: Model $MODEL FAILED with exit code $EXIT_CODE ***"
    echo "*** Check stderr log: logs/04b_bayesian_gpu_${SLURM_JOB_ID}.err ***"
fi

# =============================================================================
# Job Resource Report
# =============================================================================
echo ""
echo "============================================================"
echo "JOB RESOURCE REPORT"
echo "============================================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"

if command -v nvidia-smi &>/dev/null; then
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu \
        --format=csv,noheader 2>/dev/null | while IFS=',' read -r name total used free util temp; do
        echo "  Model:       $name"
        echo "  VRAM Total:  $total"
        echo "  VRAM Used:   $used"
        echo "  VRAM Free:   $free"
        echo "  Utilization: $util"
        echo "  Temperature: $temp"
    done
fi

echo "SLURM Accounting:"
sacct -j "$SLURM_JOB_ID" \
    --format=JobID%20,Elapsed,MaxRSS,MaxVMSize,ReqMem,AllocCPUS,State%12 \
    --noheader 2>/dev/null | head -5 || echo "  (not yet available)"
echo "============================================================"

# Auto-push logs and results when job completes (autopush exit code is
# isolated from fit exit code so a transient network failure does not
# mark the SLURM job as failed and orphan the afterok chain).
source cluster/autopush.sh

exit $EXIT_CODE
