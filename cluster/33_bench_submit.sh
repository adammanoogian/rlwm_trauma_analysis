#!/bin/bash
# Phase 33-bench: GPU vs CPU vs pscan campaign — close v4.0 TBD lines.
#
# One-shot submitter. Activates rlwm_gpu (Python 3.10+) so the editable
# install refresh works on the login node (system python is 3.9, which
# fails pyproject `requires >=3.10`). Runs alongside Phase 32-05 M6b CPU
# fits without resource contention (CPU M6b on `comp`, this campaign on
# `gpu` + one extra `comp` baseline).
#
# Usage (login node, one liner):
#   bash cluster/33_bench_submit.sh
#
# What it submits (4 jobs, all OUT_SUBDIR=33_bench_* so v4.0 21_baseline/
# canonical artefacts stay untouched):
#   1. M3 GPU   — 4 GPUs, pmap, matched MCMC settings
#   2. M3 CPU   — 4 cores, parallel, matched comparator
#   3. M6b GPU  — 4 GPUs, pmap, comparator to your running CPU M6b
#   4. Pscan    — 1 GPU, microbenchmark (Tier A, sequential vs assoc-scan)
#
# Outputs:
#   - logs/cluster/33_bench/JIDS.env   (job IDs for the morning extractor)
#   - logs/04b_bayesian_*.{out,err}    (production fit logs)
#   - logs/pscan_bench_*.{out,err}     (microbenchmark log)
#   - models/bayesian/33_bench_*/      (production fit artefacts)
#   - models/bayesian/pscan_benchmark_gpu.json
#
# After all four complete:
#   bash cluster/33_bench_extract.sh
set -euo pipefail

# Resolve repo root regardless of CWD
cd "$(dirname "$0")/.."

# -----------------------------------------------------------------------------
# Activate rlwm_gpu — needed because login-node `python3` is 3.9 which fails
# the pyproject `requires-python = ">=3.10"` constraint and `pip install -e .`
# blows up. SLURM job nodes activate this themselves; we mirror that here.
# -----------------------------------------------------------------------------
module load miniforge3 2>/dev/null || true

if conda activate rlwm_gpu 2>/dev/null; then
    :
elif conda activate /scratch/fc37/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    :
elif conda activate /scratch/${PROJECT:-fc37}/$USER/conda/envs/rlwm_gpu 2>/dev/null; then
    :
else
    echo "ERROR: cannot activate rlwm_gpu. Submit jobs without the editable refresh:"
    echo "  Skip the pip step manually and re-run, OR run cluster/00_setup_env_gpu.sh first."
    exit 1
fi

PYVER=$(python -c 'import sys; print(".".join(str(v) for v in sys.version_info[:3]))')
echo "[33-bench] Active env python: $PYVER"
case "$PYVER" in
    3.10*|3.11*|3.12*) ;;
    *) echo "ERROR: rlwm_gpu has Python $PYVER (needed >= 3.10). Re-create env."; exit 1 ;;
esac

# -----------------------------------------------------------------------------
# Sync repo + clean line endings + refresh editable install
# -----------------------------------------------------------------------------
echo "[33-bench] git pull..."
git pull --rebase origin main

echo "[33-bench] CRLF strip on cluster/*.slurm..."
sed -i 's/\r$//' cluster/*.slurm

echo "[33-bench] pip install -e . (refresh editable install)..."
pip install -e . --quiet --no-deps 2>/dev/null || \
    echo "WARNING: editable install refresh failed (continuing — may rely on prior install)"

# -----------------------------------------------------------------------------
# Submit the 4 jobs
# Matched MCMC across (1)(2)(3): chains=4, warmup=1000, samples=2000, seed=42, MTD=8
# (those are the SLURM template defaults — we do not override here).
# KAPPA_MODE=convex: matches Phase 32-05 smoke verdict.
# -----------------------------------------------------------------------------
echo ""
echo "[33-bench] Submitting jobs..."
echo ""

JID_GPU_M3=$(sbatch --parsable --time=04:00:00 \
    --export=ALL,MODEL=wmrl_m3,KAPPA_MODE=convex,OUT_SUBDIR=33_bench_gpu_m3 \
    cluster/04b_bayesian_gpu.slurm)

JID_CPU_M3=$(sbatch --parsable --time=08:00:00 \
    --export=ALL,MODEL=wmrl_m3,KAPPA_MODE=convex,OUT_SUBDIR=33_bench_cpu_m3 \
    cluster/04b_bayesian_cpu.slurm)

JID_GPU_M6B=$(sbatch --parsable --time=10:00:00 \
    --export=ALL,MODEL=wmrl_m6b,KAPPA_MODE=convex,OUT_SUBDIR=33_bench_gpu_m6b \
    cluster/04b_bayesian_gpu.slurm)

JID_PSCAN=$(sbatch --parsable --time=01:00:00 \
    --partition=gpu --gres=gpu:1 --mem=32G --cpus-per-task=4 \
    --job-name=pscan-bench \
    --output=logs/pscan_bench_%j.out --error=logs/pscan_bench_%j.err \
    --wrap='module load miniforge3 && conda activate rlwm_gpu && cd "$SLURM_SUBMIT_DIR" && python tests/scientific/benchmark_parallel_scan.py --n-repeats 20')

# -----------------------------------------------------------------------------
# Persist JIDs for the extractor
# -----------------------------------------------------------------------------
mkdir -p logs/cluster/33_bench
cat > logs/cluster/33_bench/JIDS.env <<EOF
# Phase 33-bench job IDs — sourced by cluster/33_bench_extract.sh
JID_GPU_M3=$JID_GPU_M3
JID_CPU_M3=$JID_CPU_M3
JID_GPU_M6B=$JID_GPU_M6B
JID_PSCAN=$JID_PSCAN
SUBMITTED_AT=$(date -Iseconds)
EOF

cat <<EOF

==============================================================
Phase 33-bench: 4 jobs submitted

  $JID_GPU_M3  M3 GPU   (4 GPUs, pmap)        OUT_SUBDIR=33_bench_gpu_m3
  $JID_CPU_M3  M3 CPU   (4 cores, parallel)   OUT_SUBDIR=33_bench_cpu_m3
  $JID_GPU_M6B  M6b GPU  (4 GPUs, pmap)        OUT_SUBDIR=33_bench_gpu_m6b
  $JID_PSCAN  Pscan    (1 GPU, Tier A µ-bench)

JIDs:    logs/cluster/33_bench/JIDS.env
Watch:   squeue -u \$USER -j $JID_GPU_M3,$JID_CPU_M3,$JID_GPU_M6B,$JID_PSCAN
Logs:    tail -f logs/04b_bayesian_*.out logs/pscan_bench_*.out
Extract: bash cluster/33_bench_extract.sh   (when all 4 complete)
==============================================================
EOF
