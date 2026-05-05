#!/bin/bash
# Phase 33-bench: morning extractor — pulls timing, convergence, and
# pscan numbers from the four jobs submitted by 33_bench_submit.sh.
#
# Usage:  bash cluster/33_bench_extract.sh
# Run after `squeue` shows all 4 jobs Complete or Failed.
set -uo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f logs/cluster/33_bench/JIDS.env ]]; then
    echo "ERROR: logs/cluster/33_bench/JIDS.env not found."
    echo "       Did you run cluster/33_bench_submit.sh?"
    exit 1
fi
source logs/cluster/33_bench/JIDS.env

echo "=================================================================="
echo "Phase 33-bench extraction (submitted ${SUBMITTED_AT:-unknown})"
echo "=================================================================="

# ---------------------------------------------------------------- Wall + memory
echo ""
echo "--- Wall-time + memory (sacct) ---"
sacct -j $JID_GPU_M3,$JID_CPU_M3,$JID_GPU_M6B,$JID_PSCAN \
    --format=JobID,JobName%20,Elapsed,State,MaxRSS,AllocCPUs,AllocGRES,Partition -P \
    2>/dev/null || echo "sacct unavailable"

# ---------------------------------------------------------------- Per-iter logs
echo ""
echo "--- Per-iter timing + chain_method (from logs) ---"
for jid in $JID_GPU_M3 $JID_CPU_M3 $JID_GPU_M6B; do
    log=$(ls logs/04b_bayesian_*_${jid}.out 2>/dev/null | head -1)
    if [[ -z "${log:-}" ]]; then
        echo ""
        echo ">>> JID $jid: NO LOG FOUND"
        continue
    fi
    echo ""
    echo ">>> $log"
    grep -E "chain_method|Out subdir|Model:|^\[timing\]|samples/sec|warmup wall|sampling wall|Started:|finished" "$log" 2>/dev/null | head -25
done

# ---------------------------------------------------------------- Convergence
echo ""
echo ""
echo "--- Convergence quality (matched MCMC → matched convergence?) ---"
for sub in 33_bench_gpu_m3 33_bench_cpu_m3 33_bench_gpu_m6b; do
    csv=$(ls models/bayesian/${sub}/wmrl_m*_individual_fits.csv 2>/dev/null | head -1)
    if [[ -z "${csv:-}" ]]; then
        echo ""
        echo ">>> $sub: NO INDIVIDUAL_FITS.csv"
        continue
    fi
    echo ""
    echo ">>> $csv"
    awk -F, '
        NR==1 {
            for (i=1; i<=NF; i++) {
                if ($i=="participant_id") p=i
                if ($i=="max_rhat")     r=i
                if ($i=="min_ess")      e=i
                if ($i=="divergences")  d=i
                if ($i=="bfmi")         b=i
                if ($i=="converged")    c=i
            }
            printf "  %-12s %-7s %-7s %-4s %-7s %s\n", "pid", "rhat", "ess", "div", "bfmi", "conv"
        }
        NR>1 && NR<=9 {
            printf "  %-12s %-7s %-7s %-4s %-7s %s\n", $p, $r, $e, $d, $b, $c
        }
    ' "$csv"
done

# ---------------------------------------------------------------- Shrinkage ICCs
echo ""
echo ""
echo "--- Shrinkage report ICCs (per-parameter identifiability) ---"
for sub in 33_bench_gpu_m3 33_bench_cpu_m3 33_bench_gpu_m6b; do
    md=$(ls models/bayesian/${sub}/wmrl_m*_shrinkage_report.md 2>/dev/null | head -1)
    if [[ -z "${md:-}" ]]; then
        echo ">>> $sub: NO SHRINKAGE REPORT"
        continue
    fi
    echo ""
    echo ">>> $md"
    grep -E "^\| [a-z]" "$md" 2>/dev/null
    grep -E "^\*\*Summary" "$md" 2>/dev/null
done

# ---------------------------------------------------------------- Pscan µ-bench
echo ""
echo ""
echo "--- Pscan microbenchmark (Tier A, sequential vs associative_scan) ---"
if [[ -f models/bayesian/pscan_benchmark_gpu.json ]]; then
    python -c "
import json, sys
with open('models/bayesian/pscan_benchmark_gpu.json') as f:
    data = json.load(f)
print(f\"Backend: {data.get('backend', '?')}, devices: {data.get('devices', '?')}\")
print(f\"{'Model':<12} {'Sequential':>12} {'PScan':>12} {'Speedup':>10}\")
print(f\"{'-'*12:<12} {'-'*12:>12} {'-'*12:>12} {'-'*10:>10}\")
for entry in data.get('results', []):
    seq_ms = entry.get('seq_ms_warm', float('nan'))
    psc_ms = entry.get('pscan_ms_warm', float('nan'))
    sp = seq_ms / psc_ms if psc_ms else float('nan')
    print(f\"{entry.get('model','?'):<12} {seq_ms:>10.2f}ms {psc_ms:>10.2f}ms {sp:>9.2f}x\")
"
else
    echo "JSON not produced — check logs/pscan_bench_${JID_PSCAN}.out"
    log="logs/pscan_bench_${JID_PSCAN}.out"
    [[ -f "$log" ]] && tail -30 "$log"
fi

# ---------------------------------------------------------------- Verdict template
echo ""
echo ""
echo "=================================================================="
echo "Verdict skeleton — fill in from numbers above"
echo "=================================================================="
cat <<'EOF'
Q1: Does GPU beat CPU at matched convergence on M3?
    GPU M3 elapsed: ____   CPU M3 elapsed: ____   ratio: ____
    Both rhat<1.01, ess>=400, div=0, bfmi>=0.2? ____ / ____

Q2: Does multi-GPU pmap scale to M6b under matched settings?
    GPU M6b elapsed: ____   (CPU M6b in-flight under Phase 32-05)

Q3: Does pscan help on GPU per-likelihood-eval?
    Per-model speedups from pscan_benchmark_gpu.json: ____

Q4: chain_method auto-selection working?
    GPU logs should show "chain_method: parallel" (4 GPUs → pmap)
    CPU log should show "chain_method: parallel" (NUMPYRO_HOST_DEVICE_COUNT=4)

Decision rules:
  - GPU >3x faster at matched convergence → flip default to GPU template (Phase 33+)
  - GPU within 30% or loses → keep CPU default, document empirical numbers
  - pscan microbenchmark gives any model >2x on GPU → enable --use-pscan opt-in
EOF

# ---------------------------------------------------------------- Push reminder
echo ""
echo "When ready to share results:"
echo "  git pull --rebase origin main"
echo "  git add models/bayesian/33_bench_*/ models/bayesian/pscan_benchmark_gpu.json \\"
echo "          logs/04b_bayesian_*.{out,err} logs/pscan_bench_*.{out,err} \\"
echo "          logs/cluster/33_bench/JIDS.env"
echo "  git commit -m 'results(33-bench): GPU vs CPU vs pscan benchmark campaign'"
echo "  git push origin main"
