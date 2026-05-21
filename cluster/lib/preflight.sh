#!/usr/bin/env bash
# =============================================================================
# preflight.sh -- T0 preflight checks for rlwm_trauma_analysis
# =============================================================================
# Lightweight, fast gate that validates the compute-node environment before
# any real work starts.  Source from any .slurm script AFTER cluster_env.sh:
#
#   source cluster/lib/cluster_env.sh
#   source cluster/lib/preflight.sh   # exits non-zero on failure
#
# Checks:
#   1. Conda environment active and matches expected name (rlwm_gpu)
#   2. Core Python packages importable (numpy, pandas, jax, numpyro)
#   3. Disk space >= 1 GB on working directory
#   4. rlwm package importable (editable install present)
#
# Exit behaviour:
#   - Exits 1 on any FAIL, halting the sourcing script (set -e propagates).
#   - WARNs are non-fatal (e.g. wrong env name but still usable).
# =============================================================================
set -euo pipefail

EXPECTED_ENV="rlwm_gpu"
ERRORS=0

echo ""
echo "──────────────────────────────────────────────────────────────"
echo "  PREFLIGHT CHECKS"
echo "──────────────────────────────────────────────────────────────"

# ── 1. Conda environment check ──────────────────────────────────
if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
    echo "PREFLIGHT FAIL: No conda environment active." >&2
    ERRORS=$((ERRORS + 1))
elif [[ "${CONDA_DEFAULT_ENV}" != "${EXPECTED_ENV}" ]]; then
    echo "PREFLIGHT WARN: Expected ${EXPECTED_ENV}, got ${CONDA_DEFAULT_ENV}" >&2
fi
echo "PREFLIGHT OK: conda env = ${CONDA_DEFAULT_ENV:-none}"

# ── 2. Python importability check ───────────────────────────────
# Core stack: numpy + pandas (data), jax (compute), numpyro (Bayesian fitting)
if ! python -c "import numpy; import pandas; import jax; import numpyro" 2>/dev/null; then
    echo "PREFLIGHT FAIL: Cannot import numpy/pandas/jax/numpyro." >&2
    ERRORS=$((ERRORS + 1))
else
    echo "PREFLIGHT OK: core packages importable (numpy, pandas, jax, numpyro)"
fi

# ── 3. Disk space check ─────────────────────────────────────────
MIN_FREE_KB=1048576  # 1 GB
WORK_DIR="${SLURM_SUBMIT_DIR:-.}"
FREE_KB=$(df -k "${WORK_DIR}" | awk 'NR==2 {print $4}')
if [[ "${FREE_KB}" -lt "${MIN_FREE_KB}" ]]; then
    echo "PREFLIGHT FAIL: Less than 1 GB free on ${WORK_DIR}." >&2
    ERRORS=$((ERRORS + 1))
else
    echo "PREFLIGHT OK: disk space sufficient ($(( FREE_KB / 1024 )) MB free)"
fi

# ── 4. rlwm package importable ──────────────────────────────────
# Catches stale egg-link after repo moves or missing pip install -e .
if ! python -c "from rlwm.fitting.core import pad_block_to_max" 2>/dev/null; then
    echo "PREFLIGHT FAIL: Cannot import rlwm package (run: pip install -e . --no-deps)" >&2
    ERRORS=$((ERRORS + 1))
else
    echo "PREFLIGHT OK: rlwm package importable"
fi

# ── Result ───────────────────────────────────────────────────────
echo "──────────────────────────────────────────────────────────────"
if [[ "${ERRORS}" -gt 0 ]]; then
    echo "PREFLIGHT: ${ERRORS} check(s) failed. Aborting." >&2
    exit 1
fi
echo "PREFLIGHT: All checks passed."
echo ""
