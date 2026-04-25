#!/usr/bin/env bash
# =============================================================================
# Comprehensive Pre-Submit-All Validation — cluster/00_precheck.sh
# =============================================================================
# ONE login-node entry point that runs every check needed before
# `cluster/submit_all.sh` can be fired without worry. Combines:
#
#   L1 Repo + git state         (login-side, ~1s)
#   L2 Filesystem + data        (login-side, ~1s)
#   L3 Code syntax + structure  (login-side, ~10s)
#   L4 Local pytest fast tier   (login-side, ~1-2 min — CI-equivalent)
#   L5 Pipeline dry-run         (login-side, ~5s — every stage SLURM walked)
#   L6 Compute-node gate        (sbatch --wait cluster/00_preflight.slurm,
#                                ~3-10 min queue + 2 min run)
#
# Layers L1-L5 fail fast on cheap checks. L6 only fires after the local layers
# pass, so we never burn cluster cycles on a precheck that local code already
# rejects.
#
# Set -e is intentionally OFF — every layer accumulates its verdict so the
# operator sees ALL failures in the final summary, not just the first.
#
# Usage:
#   bash cluster/00_precheck.sh                  # full precheck (default)
#   bash cluster/00_precheck.sh --skip-cluster   # local layers only (no sbatch)
#   bash cluster/00_precheck.sh --skip-pytest    # skip the local pytest run
#   bash cluster/00_precheck.sh --skip-dryrun    # skip submit_all.sh --dry-run
#   bash cluster/00_precheck.sh --quick          # alias for --skip-cluster --skip-pytest
#
# Exit codes:
#   0  PRECHECK: PASS  (safe to run cluster/submit_all.sh)
#   1  PRECHECK: FAIL  (one or more layers failed — see summary)
#
# Reuse policy:
#   Run this every time you (a) clone fresh, (b) merge upstream, (c) bump a
#   dependency, (d) cold-start after a long pause. The compute-node gate
#   catches env drift the local layers cannot (CUDA / jaxlib plugin mismatch,
#   GPU visibility, scratch path drift).
# =============================================================================

set -uo pipefail
cd "$(dirname "$0")/.."

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
SKIP_CLUSTER=0
SKIP_PYTEST=0
SKIP_DRYRUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-cluster) SKIP_CLUSTER=1; shift ;;
    --skip-pytest)  SKIP_PYTEST=1; shift ;;
    --skip-dryrun)  SKIP_DRYRUN=1; shift ;;
    --quick)        SKIP_CLUSTER=1; SKIP_PYTEST=1; shift ;;
    -h|--help)
      grep "^#" "$0" | head -50
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# Layer-result tracking — each layer calls `mark <key> <PASS|FAIL|SKIP> <msg>`
# ---------------------------------------------------------------------------
LAYER_ORDER=("L1_repo" "L2_filesystem" "L3_code_syntax" "L4_local_pytest" "L5_pipeline_dryrun" "L6_compute_gate")
declare -A LAYER_RESULT
declare -A LAYER_MSG
OVERALL=PASS

mark() {
  local key="$1" result="$2" msg="$3"
  LAYER_RESULT[$key]="$result"
  LAYER_MSG[$key]="$msg"
  if [[ "$result" == "FAIL" ]]; then OVERALL=FAIL; fi
  local symbol="?"
  case "$result" in
    PASS) symbol="OK" ;;
    FAIL) symbol="!!" ;;
    SKIP) symbol="--" ;;
    WARN) symbol=".." ;;
  esac
  printf "[%s] %-22s %s — %s\n" "$symbol" "$key" "$result" "$msg"
}

echo "============================================================"
echo "[00_precheck.sh] $(date)"
echo "  skip-cluster: $SKIP_CLUSTER"
echo "  skip-pytest:  $SKIP_PYTEST"
echo "  skip-dryrun:  $SKIP_DRYRUN"
echo "============================================================"

# =============================================================================
# L1 — Repo + git state
# =============================================================================
echo ""
echo "------------------------------------------------------------"
echo "L1 — Repo + git state"
echo "------------------------------------------------------------"
if [[ ! -d .git ]]; then
  mark "L1_repo" "FAIL" "not in a git repo (no .git/ at $(pwd))"
else
  BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
  HEAD_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
  DIRTY=$(git status --porcelain 2>/dev/null | wc -l | tr -d ' ')
  L1_DETAILS="branch=$BRANCH head=$HEAD_SHA dirty=$DIRTY"
  if [[ "$BRANCH" != "main" ]]; then
    L1_DETAILS="$L1_DETAILS (WARN: not on main)"
  fi
  if [[ "$DIRTY" != "0" ]]; then
    L1_DETAILS="$L1_DETAILS (WARN: working tree dirty — $DIRTY paths)"
  fi
  mark "L1_repo" "PASS" "$L1_DETAILS"
fi

# =============================================================================
# L2 — Filesystem + data
# =============================================================================
echo ""
echo "------------------------------------------------------------"
echo "L2 — Filesystem + data"
echo "------------------------------------------------------------"
L2_FAILS=()

# Ensure required output dirs (idempotent — created if absent)
for d in models/bayesian models/bayesian/21_prior_predictive models/bayesian/21_recovery models/bayesian/21_baseline logs reports/figures reports/tables; do
  if ! mkdir -p "$d" 2>/dev/null; then
    L2_FAILS+=("cannot create $d")
  fi
done

# Required input data (canonical CCDS path)
DATA_FILE="data/processed/task_trials_long.csv"
if [[ ! -f "$DATA_FILE" ]]; then
  L2_FAILS+=("missing $DATA_FILE — run scripts/01_data_preprocessing/03_create_task_trials_csv.py")
else
  ROW_COUNT=$(wc -l < "$DATA_FILE" 2>/dev/null | tr -d ' ')
  HEADER=$(head -1 "$DATA_FILE")
  # Minimum schema check: header should mention these CCDS-canonical columns
  for col in sona_id stimulus key_press correct rt; do
    if ! grep -q "$col" <<< "$HEADER"; then
      L2_FAILS+=("$DATA_FILE missing column '$col' in header")
    fi
  done
  if [[ "$ROW_COUNT" -lt 100 ]]; then
    L2_FAILS+=("$DATA_FILE has only $ROW_COUNT lines (expected >> 100)")
  fi
fi

if [[ ${#L2_FAILS[@]} -eq 0 ]]; then
  mark "L2_filesystem" "PASS" "dirs OK; $DATA_FILE rows=$ROW_COUNT"
else
  mark "L2_filesystem" "FAIL" "${L2_FAILS[*]}"
fi

# =============================================================================
# L3 — Code syntax + structure
# =============================================================================
echo ""
echo "------------------------------------------------------------"
echo "L3 — Code syntax + structure"
echo "------------------------------------------------------------"
L3_FAILS=()

# CRLF strip — idempotent; mirrors submit_all.sh:98
sed -i 's/\r$//' cluster/*.slurm cluster/*.sh 2>/dev/null || true

# bash -n on every cluster/*.slurm and *.sh
for f in cluster/*.slurm cluster/*.sh; do
  [[ -f "$f" ]] || continue
  if ! bash -n "$f" 2>/dev/null; then
    L3_FAILS+=("bash -n $f")
  fi
done

# python compile on scripts/ src/ (catches SyntaxError without importing)
PY_COMPILE_LOG=$(mktemp 2>/dev/null || echo /tmp/precheck_pyc_$$.log)
if command -v python >/dev/null 2>&1; then
  if ! python -m compileall -q scripts/ src/ > "$PY_COMPILE_LOG" 2>&1; then
    L3_FAILS+=("python -m compileall failed — see $PY_COMPILE_LOG")
  fi
else
  L3_FAILS+=("python not on PATH")
fi

if [[ ${#L3_FAILS[@]} -eq 0 ]]; then
  mark "L3_code_syntax" "PASS" "bash -n all cluster/* and python compileall scripts/ src/ clean"
  rm -f "$PY_COMPILE_LOG" 2>/dev/null
else
  mark "L3_code_syntax" "FAIL" "${L3_FAILS[*]}"
fi

# =============================================================================
# L4 — Local pytest fast tier (CI-equivalent per CLAUDE.md)
# =============================================================================
echo ""
echo "------------------------------------------------------------"
echo "L4 — Local pytest fast tier"
echo "------------------------------------------------------------"
if [[ "$SKIP_PYTEST" == "1" ]]; then
  mark "L4_local_pytest" "SKIP" "skipped via --skip-pytest / --quick"
else
  if ! command -v pytest >/dev/null 2>&1; then
    mark "L4_local_pytest" "FAIL" "pytest not on PATH (activate env or pip install -e .)"
  else
    PYTEST_LOG=$(mktemp 2>/dev/null || echo /tmp/precheck_pytest_$$.log)
    # Fast tier per CLAUDE.md: not slow, not scientific
    if pytest tests/ -m "not slow and not scientific" --tb=short -q > "$PYTEST_LOG" 2>&1; then
      LAST_LINE=$(tail -1 "$PYTEST_LOG")
      mark "L4_local_pytest" "PASS" "$LAST_LINE"
      rm -f "$PYTEST_LOG"
    else
      LAST_LINE=$(tail -1 "$PYTEST_LOG")
      mark "L4_local_pytest" "FAIL" "$LAST_LINE (full log: $PYTEST_LOG)"
    fi
  fi
fi

# =============================================================================
# L5 — Pipeline dry-run (submit_all.sh --dry-run)
# =============================================================================
echo ""
echo "------------------------------------------------------------"
echo "L5 — Pipeline dry-run (submit_all.sh --dry-run)"
echo "------------------------------------------------------------"
if [[ "$SKIP_DRYRUN" == "1" ]]; then
  mark "L5_pipeline_dryrun" "SKIP" "skipped via --skip-dryrun"
elif [[ ! -f cluster/submit_all.sh ]]; then
  mark "L5_pipeline_dryrun" "FAIL" "cluster/submit_all.sh not found"
else
  DRYRUN_LOG=$(mktemp 2>/dev/null || echo /tmp/precheck_dryrun_$$.log)
  if bash cluster/submit_all.sh --dry-run > "$DRYRUN_LOG" 2>&1; then
    DRY_OK=$(grep -c "^DRY ok:" "$DRYRUN_LOG" || true)
    mark "L5_pipeline_dryrun" "PASS" "$DRY_OK stage submissions resolved"
    rm -f "$DRYRUN_LOG"
  else
    LAST_LINE=$(tail -3 "$DRYRUN_LOG" | head -1)
    mark "L5_pipeline_dryrun" "FAIL" "$LAST_LINE (full log: $DRYRUN_LOG)"
  fi
fi

# =============================================================================
# L6 — Compute-node gate (sbatch --wait cluster/00_preflight.slurm)
# =============================================================================
echo ""
echo "------------------------------------------------------------"
echo "L6 — Compute-node gate (sbatch --wait cluster/00_preflight.slurm)"
echo "------------------------------------------------------------"
if [[ "$SKIP_CLUSTER" == "1" ]]; then
  mark "L6_compute_gate" "SKIP" "skipped via --skip-cluster / --quick"
elif ! command -v sbatch >/dev/null 2>&1; then
  mark "L6_compute_gate" "SKIP" "sbatch not on PATH (not on a SLURM-cluster login node)"
elif [[ ! -f cluster/00_preflight.slurm ]]; then
  mark "L6_compute_gate" "FAIL" "cluster/00_preflight.slurm not found"
else
  echo "  Submitting cluster/00_preflight.slurm with sbatch --wait..."
  echo "  This blocks until the job completes (queue + ~2 min runtime)."
  PREFLIGHT_JID=""
  PREFLIGHT_EXIT=0
  PREFLIGHT_JID=$(sbatch --parsable --wait cluster/00_preflight.slurm) || PREFLIGHT_EXIT=$?
  if [[ -z "$PREFLIGHT_JID" ]]; then
    mark "L6_compute_gate" "FAIL" "sbatch did not return a JobID (exit=$PREFLIGHT_EXIT)"
  else
    PREFLIGHT_LOG="models/bayesian/21_preflight_${PREFLIGHT_JID}.log"
    if [[ ! -f "$PREFLIGHT_LOG" ]]; then
      mark "L6_compute_gate" "FAIL" "job $PREFLIGHT_JID exit=$PREFLIGHT_EXIT — log not found at $PREFLIGHT_LOG"
    elif ! grep -q "^PREFLIGHT: PASS$" "$PREFLIGHT_LOG"; then
      LAST=$(tail -3 "$PREFLIGHT_LOG" | head -1)
      mark "L6_compute_gate" "FAIL" "job $PREFLIGHT_JID — log lacks PREFLIGHT: PASS ($LAST)"
    else
      mark "L6_compute_gate" "PASS" "job $PREFLIGHT_JID PASS — $PREFLIGHT_LOG"
    fi
  fi
fi

# =============================================================================
# Final summary
# =============================================================================
echo ""
echo "============================================================"
echo "PRECHECK SUMMARY"
echo "============================================================"
for L in "${LAYER_ORDER[@]}"; do
  printf "  %-22s %-6s  %s\n" "$L" "${LAYER_RESULT[$L]:-MISS}" "${LAYER_MSG[$L]:-(no result)}"
done
echo "------------------------------------------------------------"
echo "PRECHECK: $OVERALL"
echo "============================================================"

if [[ "$OVERALL" == "PASS" ]]; then
  echo ""
  echo "Safe to run:  bash cluster/submit_all.sh"
  echo "Or canary:    bash cluster/03_submit_canary.sh"
  exit 0
else
  echo ""
  echo "Do NOT run cluster/submit_all.sh until every FAIL is resolved." >&2
  exit 1
fi
