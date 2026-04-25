#!/bin/bash
# =============================================================================
# SLURM Batch Script: Phase 24 Wave 1 Pre-flight Gate (compute-node, automated)
# =============================================================================
# Purpose: reproducible cluster pre-flight that runs as a SLURM job instead of
# an interactive login-node command. Any operator on any submit host with
# `sbatch` and `git push` access can run this — no shell-state setup required.
#
# What it verifies on a compute node (NOT login node):
#   1. rlwm_gpu env activates cleanly via the same dispatch logic as
#      cluster/03_prefitting_gpu.slurm
#   2. `pip install -e . --no-deps` succeeds (idempotent on already-installed
#      envs; closes the "fresh env / fresh operator" reproducibility gap)
#   3. JAX devices probe — catches CUDA / jaxlib plugin mismatch on real GPU
#      hardware (the login-node pytest cannot test this — no GPU on login)
#   4. `from rlwm.fitting.core import pad_block_to_max` + per-model imports
#      smoke test
#   5. tests/integration/test_numpyro_models_2cov.py pytest gate (expects
#      9 passed, 1 deselected)
#
# Output:
#   - models/bayesian/21_preflight_<JID>.log  -- single PASS/FAIL artifact
#     (autopushed by cluster/autopush.sh on completion)
#   - logs/00_preflight_<JID>.{out,err}       -- SLURM accounting trail
#
# Pass criteria (operator/orchestrator greps after autopush):
#   - Literal line "PREFLIGHT: PASS" at end of preflight log
#   - SLURM exit 0
#
# Failure recovery (CONTEXT.md Area 2 framing — NOT "retry"):
#   - Diagnose from logs/00_preflight_<JID>.{out,err}
#   - Patch root cause (env, install, CUDA stack, etc.)
#   - Commit + push the fix
#   - Fresh sbatch — same seed; new JobID
#
# Usage:
#   sbatch cluster/00_preflight.slurm
# =============================================================================

#SBATCH --job-name=p24_preflight
#SBATCH --output=logs/00_preflight_%j.out
#SBATCH --error=logs/00_preflight_%j.err
#SBATCH --time=00:15:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

set -euo pipefail

module load miniforge3

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
PROJECT_ROOT="$(pwd)"
export PROJECT_ROOT

# Pre-flight log lives in CCDS path so autopush picks it up
mkdir -p models/bayesian logs
PREFLIGHT_LOG="models/bayesian/21_preflight_${SLURM_JOB_ID:-local}.log"
exec > >(tee "$PREFLIGHT_LOG") 2>&1

echo "============================================================"
echo "Phase 24 Wave 1 Pre-flight Gate (SLURM-automated)"
echo "============================================================"
echo "Job ID:     ${SLURM_JOB_ID:-local}"
echo "Node:       ${SLURMD_NODENAME:-$(hostname)}"
echo "GPU(s):     ${CUDA_VISIBLE_DEVICES:-none}"
echo "Project:    $PROJECT_ROOT"
echo "Git HEAD:   $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Start time: $(date)"
echo "============================================================"

# ---------------------------------------------------------------
# Step 1: activate rlwm_gpu (mirrors cluster/03_prefitting_gpu.slurm dispatch)
# ---------------------------------------------------------------
echo ""
echo "[1/5] Activating rlwm_gpu env..."
if conda activate rlwm_gpu 2>/dev/null; then
    echo "  -> activated by name"
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "  -> activated from /scratch/fc37/$USER/"
elif conda activate /scratch/${PROJECT:-fc37}/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    echo "  -> activated from /scratch/${PROJECT:-fc37}/$USER/"
else
    echo "FAIL [1/5]: could not activate rlwm_gpu via any known path"
    echo "PREFLIGHT: FAIL"
    exit 1
fi
echo "  python: $(which python)"
echo "  pip:    $(which pip)"

# ---------------------------------------------------------------
# Step 2: pip install -e . (idempotent; closes the reproducibility gap)
# ---------------------------------------------------------------
echo ""
echo "[2/5] pip install -e . --no-deps (idempotent registration)..."
if pip install -e . --quiet --no-deps; then
    echo "  -> install OK"
else
    echo "FAIL [2/5]: pip install -e . errored"
    echo "PREFLIGHT: FAIL"
    exit 1
fi

# ---------------------------------------------------------------
# Step 3: JAX devices probe (catches CUDA / jaxlib plugin mismatch)
# ---------------------------------------------------------------
echo ""
echo "[3/5] JAX devices probe..."
python -c "
import sys
import jax
devs = jax.devices()
gpu_devs = [d for d in devs if d.platform == 'gpu']
print(f'  jax.__version__ = {jax.__version__}')
print(f'  jax.devices()   = {devs}')
print(f'  GPU count       = {len(gpu_devs)}')
if not gpu_devs:
    print('FAIL [3/5]: no GPU device visible to JAX (CUDA or jaxlib plugin mismatch?)')
    sys.exit(1)
print('  -> JAX sees GPU')
" || { echo "PREFLIGHT: FAIL"; exit 1; }

# ---------------------------------------------------------------
# Step 4: rlwm import smoke test
# ---------------------------------------------------------------
echo ""
echo "[4/5] rlwm import smoke test..."
python -c "
import importlib
from rlwm.fitting.core import pad_block_to_max  # noqa: F401
print('  -> rlwm.fitting.core OK')
for m in ['qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b']:
    importlib.import_module(f'rlwm.fitting.models.{m}')
    print(f'  -> rlwm.fitting.models.{m} OK')
print('  -> all rlwm imports succeed')
" || { echo "FAIL [4/5]: rlwm import failure"; echo "PREFLIGHT: FAIL"; exit 1; }

# ---------------------------------------------------------------
# Step 5: pytest gate (matches cluster/21_submit_pipeline.sh:26)
# ---------------------------------------------------------------
echo ""
echo "[5/5] pytest gate (test_numpyro_models_2cov.py)..."
pytest tests/integration/test_numpyro_models_2cov.py -v -k "not slow" --tb=short || {
    echo "FAIL [5/5]: pytest gate red"
    echo "PREFLIGHT: FAIL"
    exit 1
}

echo ""
echo "============================================================"
echo "PREFLIGHT: PASS"
echo "End time: $(date)"
echo "============================================================"

# ---------------------------------------------------------------
# Autopush the log so the gate verdict syncs to main without operator action
# ---------------------------------------------------------------
if [[ -f cluster/autopush.sh ]]; then
    source cluster/autopush.sh
fi

exit 0
