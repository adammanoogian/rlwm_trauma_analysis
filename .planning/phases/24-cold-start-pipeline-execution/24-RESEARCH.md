# Phase 24: Cold-Start Pipeline Execution — Research

**Researched:** 2026-04-19
**Domain:** SLURM `afterok`-chained execution + post-run audit on Monash M3 cluster
**Confidence:** HIGH (every finding cited from concrete repo files; no claim relies on training data)

> **No CONTEXT.md** for this phase — `--skip-discussion` was passed. Research scope is the operational map of `cluster/21_submit_pipeline.sh`, the artifacts each step produces, the SLURM-accounting surface, the convergence-gate verification path, and the failure modes the planner must encode in 24-01-PLAN.md (pre-flight + submission + monitoring) and 24-02-PLAN.md (post-run audit + execution log + winner determination recording).

## Summary

Phase 24 is a **wiring job**, not a new-implementation job. The 9-step orchestrator (`cluster/21_submit_pipeline.sh`), all 11 SLURM scripts (`cluster/21_*.slurm`), all 9 driver Python scripts (`scripts/21_*.py`), and the local pre-flight pytest target (`scripts/fitting/tests/test_numpyro_models_2cov.py`) already exist and were code-verified at the v4.0 close (see `.planning/phases/21-principled-bayesian-model-selection-pipeline/21-VERIFICATION.md` SC#1: "VERIFIED (code); DEFERRED (execution)"). Phase 23 just landed the clean codebase on which this run executes (4 commits across 23-01..23-04, all CLEAN-01..04 closed). This phase consumes that infrastructure and produces empirical artifacts.

The two-wave structure is forced by the 50-96 GPU-hour wall clock: Wave 1 cannot complete in a single conversation. **Wave 1** = pre-flight checks on a Windows submit host (the dev box), submission of `bash cluster/21_submit_pipeline.sh` from the Monash M3 login node, and a per-step monitoring contract that detects early failure of the chain (SLURM `afterok` propagates failures: a single `EXCLUDED_*` exit from step 21.3/21.4 cancels the rest of the chain via `--dependency=afterok`). **Wave 2** = post-run audit (artifact inventory enumerated below as `must_haves`), `output/bayesian/21_execution_log.md` writeup from `sacct` accounting, and explicit recording of the winner-determination path (`DOMINANT_SINGLE` / `TOP_TWO` / `FORCED` / `INCONCLUSIVE_MULTIPLE`).

**Critical finding the planner must surface:** the ROADMAP/REQUIREMENTS state expected paths `output/bayesian/manuscript/{loo_stacking,rfx_bms,winner_betas}.{csv,md,tex}` and forest plots under `output/bayesian/figures/`. The actual scripts write to `output/bayesian/21_tables/{table1_loo_stacking,table2_rfx_bms,table3_winner_betas}.{csv,md,tex}` and forest PNGs to **two locations** (`figures/21_bayesian/forest_*.png` per the SLURM script line 145 + `output/bayesian/figures/{internal}_forest.png` per the Python script line 764). The audit plan must verify the *actual* paths from the scripts — not the paths in the ROADMAP — and the planner should flag the doc drift for closure (Phase 27) to resolve.

**Primary recommendation:** Follow the existing infrastructure verbatim. Wave 1 = invoke `bash cluster/21_submit_pipeline.sh` on the cluster login node, capture all 11 JobIDs from stdout, and monitor via `squeue -u $USER` + tail of `logs/bayesian_21_*_*.{out,err}`. Wave 2 = enumerate the must-have artifact list below, parse `convergence_table.csv` for the EXEC-04 gate, parse `winners.txt` + `winner_report.md` for the EXEC-04 winner-type assertion, and compose `21_execution_log.md` from `sacct -j $JID --format=JobID,JobName,Elapsed,TotalCPU,MaxRSS,State,ReqTRES,AllocTRES`.

## Standard Stack

The stack for this phase is **the existing Phase 21 infrastructure**. Treat anything not in this table as out-of-scope for Phase 24 — the planner should not introduce new tooling.

### Core orchestration (already present)

| Tool | Path | Purpose |
|---|---|---|
| Master orchestrator | `cluster/21_submit_pipeline.sh` | Pre-flight pytest gate + submits all 11 SLURM jobs with `afterok` deps |
| Per-step SLURM scripts | `cluster/21_{1..9}_*.slurm` (11 files) | One sbatch per step; each handles its own `module load miniforge3` + `conda activate ds_env` + JAX cache dir |
| L2 dispatcher (sub-script) | `cluster/21_dispatch_l2_winners.sh` | Reads `winners.txt`, `sbatch --wait &` per winner, single `wait` at end |
| Auto-push hook | `cluster/autopush.sh` | Sourced at end of each SLURM script — `git add logs/ output/bayesian/`, commit, push |
| Pre-flight pytest target | `scripts/fitting/tests/test_numpyro_models_2cov.py` | 9 fast tests (3 acceptance + 3 backward-compat + 3 guard) — `pytest -k "not slow" --tb=short` |

### Driver Python scripts (already present)

| Step | SLURM | Python driver | Output dir |
|---|---|---|---|
| 21.1 | `21_1_prior_predictive.slurm` | `scripts/21_run_prior_predictive.py` | `output/bayesian/21_prior_predictive/` |
| 21.2 array | `21_2_recovery.slurm` (array=1-50) | `scripts/21_run_bayesian_recovery.py --mode single-subject` | `output/bayesian/21_recovery/` (per-subject JSON) |
| 21.2 agg | `21_2_recovery_aggregate.slurm` | `scripts/21_run_bayesian_recovery.py --mode aggregate` | `output/bayesian/21_recovery/` (CSV + summary.md) |
| 21.3 | `21_3_fit_baseline.slurm` | `scripts/21_fit_baseline.py` | `output/bayesian/21_baseline/` |
| 21.4 | `21_4_baseline_audit.slurm` | `scripts/21_baseline_audit.py` | `output/bayesian/21_baseline/` (table + report) |
| 21.5 | `21_5_loo_stacking_bms.slurm` | `scripts/21_compute_loo_stacking.py` | `output/bayesian/21_baseline/` (winners.txt + reports) |
| 21.6 dispatcher | `21_6_dispatch_l2.slurm` (14 h cap) | `cluster/21_dispatch_l2_winners.sh` | (no direct outputs; submits children) |
| 21.6 child | `21_6_fit_with_l2.slurm` (12 h cap) | `scripts/21_fit_with_l2.py` | `output/bayesian/21_l2/` |
| 21.7 | `21_7_scale_audit.slurm` | `scripts/21_scale_audit.py` | `output/bayesian/21_l2/` |
| 21.8 | `21_8_model_averaging.slurm` | `scripts/21_model_averaging.py` | `output/bayesian/21_l2/` |
| 21.9 | `21_9_manuscript_tables.slurm` | `scripts/21_manuscript_tables.py` | `output/bayesian/21_tables/` + `figures/21_bayesian/` |

### SLURM accounting tools (system-provided on Monash M3)

| Tool | Use | Reference |
|---|---|---|
| `sbatch --parsable` | Capture pure JobID for use in `--dependency=afterok:JID` | Used 11× in `21_submit_pipeline.sh` |
| `squeue -u $USER` | Live state of pending/running jobs in this user's queue | Echoed at end of `21_submit_pipeline.sh` |
| `sacct -j JOBID` | Post-completion accounting (Elapsed, MaxRSS, State, AllocTRES) | Slurm-bundled; available on M3 login node |
| `scancel JOBID` | Manual cancellation of a stuck step | Not auto-wired; planner should reference for risk-mitigation |
| `seff JOBID` (optional) | Per-job efficiency report; equivalent to a wrapped `sacct` query | Available on most M3 builds |

### Alternatives Considered (REJECTED)

| Instead of | Could Use | Why rejected |
|---|---|---|
| `bash cluster/21_submit_pipeline.sh` | Piecemeal `sbatch cluster/21_X.slurm` per step | Phase 22 SC#9 + v4.0 closure guard explicitly bans this — `validation/check_v4_closure.py::check_cluster_freshness_framing` enforces canonical entry. Loses afterok dependency chain → silent orphan jobs. |
| `--dependency=afterany` | Continue chain on failure for diagnostics | Master orchestrator uses `afterok` exclusively (verified: `grep "afterany" cluster/21_submit_pipeline.sh` returns empty per v4.0 audit). On failure we want to STOP, not continue with garbage inputs. |
| Re-implementing audit script | Custom Python audit | `validation/check_v4_closure.py` is the local pattern (deterministic, sentinel-checked, `dataclass CheckResult` shape) — re-use the `_is_archived()` / per-check function pattern instead of inventing one. |

## Architecture Patterns

### Recommended Phase 24 directory structure

```
.planning/phases/24-cold-start-pipeline-execution/
├── 24-RESEARCH.md          # this file
├── 24-01-PLAN.md           # Wave 1: pre-flight + submission + monitoring
├── 24-01-SUMMARY.md        # Wave 1 SUMMARY (after pipeline kicks off)
├── 24-02-PLAN.md           # Wave 2: post-run audit + execution log + verification
├── 24-02-SUMMARY.md        # Wave 2 SUMMARY (after pipeline completes)
└── 24-VERIFICATION.md      # Phase verification (mirrors 23-VERIFICATION.md pattern)
```

### Pattern 1: Pre-flight gate (already implemented at line 71-77 of orchestrator)

**What:** Run `pytest scripts/fitting/tests/test_numpyro_models_2cov.py -v -k "not slow" --tb=short` LOCALLY before any sbatch call. If it fails, abort with "[ABORT] No cluster jobs submitted."

**Where it lives:** `cluster/21_submit_pipeline.sh:71-77` — already wired. Wave 1 plan must NOT re-implement; it must INVOKE.

**Test surface (from `test_numpyro_models_2cov.py` module docstring):**
- 9 parametrized tests in 4 gate groups (3 acceptance × M3/M5/M6a + 3 backward-compat × M3/M5/M6a + 3 guard × M3/M5/M6a — totals 9 fast tests; +1 slow test marked `@pytest.mark.slow` excluded by `-k "not slow"`).
- Tests run NumPyro `handlers.trace(seed(model))` only — no MCMC. Sub-second per test on CPU.
- Pass criteria: trace contains `beta_lec_{target}` and `beta_iesr_{target}` sites for the 2-cov path; LEC-only path produces no `beta_iesr_*` site; `(lec=None, iesr=<v>)` raises `ValueError` with substring `"covariate_iesr provided without covariate_lec"`.

**Pre-flight invocation (Windows-friendly via WSL or Git Bash):**
```bash
pytest scripts/fitting/tests/test_numpyro_models_2cov.py -v -k "not slow" --tb=short
```

### Pattern 2: Submission flow (existing)

**Sequence inside `cluster/21_submit_pipeline.sh`:**

```text
LINE 56  : sed -i 's/\r$//' cluster/*.slurm cluster/*.sh   # CRLF strip (Windows-checkout safety)
LINE 71-77: Pre-flight pytest gate (LOCAL, no sbatch)
LINE 84-87: Step 21.1 — 6× sbatch (one per MODEL); array of JobIDs in PPC[]; PPC_DEP="afterok:JID1:JID2:..."
LINE 95-101: Step 21.2 — 6× recovery.slurm (afterok:PPC[m]) THEN 6× recovery_aggregate.slurm (afterok:REC_ARRAY[m])
LINE 108-112: Step 21.3 — 6× fit_baseline.slurm (afterok:REC_DEP composite)
LINE 118: Step 21.4 — 1× audit.slurm (afterok:BASE_DEP composite)
LINE 127: Step 21.5 — 1× loo_stacking.slurm (afterok:AUDIT_JID)
LINE 144-145: Step 21.6 — 1× dispatcher.slurm (afterok:LOO_JID); dispatcher then sbatch --wait & per winner
LINE 152: Step 21.7 — 1× scale_audit.slurm (afterok:DISPATCH_JID)
LINE 166: Step 21.8 — 1× model_averaging.slurm (afterok:AUDIT2_JID)
LINE 172: Step 21.9 — 1× manuscript_tables.slurm (afterok:AVG_JID)
LINE 180-188: Echo job summary
```

Total job count: **6 + 6 + 6 + 6 + 1 + 1 + 1 + 1 + 1 + 1 = 30 sbatch invocations** (the 21.6 dispatcher itself spawns 1-6 more children at runtime depending on winner count).

**Env vars passed:** Only `MODEL=$m` is exported on the per-model jobs (steps 21.1, 21.2, 21.3). Steps 21.4 onward read fixed paths and are MODEL-agnostic. The 21.6 children inherit `MODEL` from the dispatcher's `sbatch --wait --export=ALL,MODEL=$model` (line 63 of `21_dispatch_l2_winners.sh`).

**CRLF stripping line (LINE 56):** `sed -i 's/\r$//' cluster/*.slurm cluster/*.sh 2>/dev/null || true` — strips Windows `\r` line endings. **Implication for Wave 1:** if the user clones from Windows or the SLURM scripts are edited via a Windows editor, this auto-fix saves the run. The planner should NOT add a separate dos2unix step; it's already wired and idempotent.

### Pattern 3: Per-step monitoring (Wave 1)

**Live monitoring contract (after submission):**

```bash
# 1. Capture full submission stdout to a tee log
bash cluster/21_submit_pipeline.sh 2>&1 | tee output/bayesian/21_submission_$(date +%Y%m%d_%H%M%S).log

# 2. Parse JobIDs from the tee log (orchestrator echoes "  [21.X] $m -> job $JID")
grep -E "\[21\.[0-9]+\].*-> (job|array job)" output/bayesian/21_submission_*.log

# 3. Live queue state (run periodically by hand or via watch)
squeue -u $USER -o "%.10i %.20j %.8T %.10M %.10l %R" --sort=i

# 4. Per-step tail when a job hits RUNNING
tail -f logs/bayesian_21_3_<JID>.out logs/bayesian_21_3_<JID>.err

# 5. Failure detection: any step exits non-zero → afterok cancels downstream jobs
# squeue will show downstream jobs disappearing with state CANCELLED in `sacct -j <JID>`
sacct -j <JID> --format=JobID,JobName,State,ExitCode,Elapsed -n
```

**Log path conventions (verified from each `#SBATCH --output=` line):**

| Step | stdout pattern | stderr pattern |
|---|---|---|
| 21.1 | `logs/bayesian_21_1_prior_%j.out` | `logs/bayesian_21_1_prior_%j.err` |
| 21.2 array | `logs/bayesian_21_2_recovery_%A_%a.out` | `logs/bayesian_21_2_recovery_%A_%a.err` |
| 21.2 agg | `logs/bayesian_21_2_aggregate_%j.out` | `logs/bayesian_21_2_aggregate_%j.err` |
| 21.3 | `logs/bayesian_21_3_%j.out` | `logs/bayesian_21_3_%j.err` |
| 21.4 | `logs/bayesian_21_4_audit_%j.out` | `logs/bayesian_21_4_audit_%j.err` |
| 21.5 | `logs/bayesian_21_5_%j.out` | `logs/bayesian_21_5_%j.err` |
| 21.6 dispatcher | `logs/21_dispatch_l2_%j.out` | `logs/21_dispatch_l2_%j.err` |
| 21.6 child | `logs/bayesian_21_6_%j.out` | `logs/bayesian_21_6_%j.err` |
| 21.7 | `logs/bayesian_21_7_%j.out` | `logs/bayesian_21_7_%j.err` |
| 21.8 | `logs/bayesian_21_8_%j.out` | `logs/bayesian_21_8_%j.err` |
| 21.9 | `logs/bayesian_21_9_%j.out` | `logs/bayesian_21_9_%j.err` |

**Early-failure signals (per-script verbatim from grep of each SLURM):**
- Step 21.1: `"Gate verdict: FAIL"` in stdout → `output/bayesian/21_prior_predictive/{model}_gate.md` has FAIL verdict.
- Step 21.3: `"CONVERGENCE GATE FAIL"` or `"ERROR: scripts/21_fit_baseline.py exited with code"` in stdout.
- Step 21.4: `"[PIPELINE BLOCK] Convergence gate failed"` in stdout → `convergence_report.md` has `n_passing < 2`.
- Step 21.5: `"[CHECKPOINT] Inconclusive winner set"` (exit 2 = pause) or `"[ABORT] Convergence-eligible pool < 2 models"` (exit 1 = abort).
- Step 21.6 child: `"ERROR: scripts/21_fit_with_l2.py exited with code"`.
- Step 21.7: `"[ERROR] Audit failed with exit code"` (NOT `[NULL RESULT]` which is exit 0 by design).

### Pattern 4: Post-run audit (Wave 2) — follow `validation/check_v4_closure.py` shape

**Why this template:** It's already the local convention (`dataclass CheckResult`, deterministic file-system + git invariants, sentinel determinism check). v5.0's `validation/check_v5_closure.py` (Phase 25) extends the same pattern. Phase 24's audit can either (a) be folded into `check_v5_closure.py` (Phase 25 scope) and Phase 24 gets a one-shot Python audit script under `validation/check_phase24_artifacts.py`, OR (b) be inline in `24-02-PLAN.md` task list as a sequence of `Path.exists()` and `pd.read_csv()` checks.

**Recommendation for the planner:** Use option (b) — keep Phase 24's audit as plan-task-level `Bash` invocations (one task per artifact group), defer the reusable closure-guard upgrade to Phase 25. Reason: Phase 24 only audits files; Phase 25 needs to add new checks anyway and will include the artifact-existence check naturally.

### Anti-Patterns to Avoid

- **Anti-pattern: piecemeal sbatch resubmission.** Banned by v4.0 closure-freshness guard. If step 21.5 exits 2 (`INCONCLUSIVE_MULTIPLE`), document the resume path verbatim from `cluster/21_submit_pipeline.sh:43-46` ("Edit `output/bayesian/21_baseline/winners.txt` ... Manually submit step 21.6+ chain with the dispatcher SLURM wrapper") — do NOT auto-retry by submitting individual sbatches from the plan.
- **Anti-pattern: ignoring auto-push side effects.** Every SLURM script sources `cluster/autopush.sh` which does `git add logs/ output/bayesian/`, commit, push. The cold-start run will produce **at least 30 auto-commits on `main`** during execution (one per SLURM job). The planner should expect and document this — and DO NOT manually `git add` the artifacts in Wave 2; they're already committed by autopush.
- **Anti-pattern: running on a dirty working tree.** Pre-flight should `git diff --quiet` before submitting. A dirty tree means autopush will silently include unrelated WIP changes in its `git add output/bayesian/` commits. Note: the *pre-flight pytest gate already passes* on the dirty tree at HEAD — that gate doesn't enforce cleanliness.
- **Anti-pattern: parsing forest plot expectations from ROADMAP.md verbatim.** ROADMAP says `output/bayesian/manuscript/` and `output/bayesian/figures/`. Scripts write to `output/bayesian/21_tables/` and `figures/21_bayesian/` (and `output/bayesian/figures/` for one path). Parse the actual scripts (or this RESEARCH.md must-haves table below), not the ROADMAP.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---|---|---|---|
| Job submission orchestration | Per-step shell loop | `cluster/21_submit_pipeline.sh` | Already implements 11 sbatch calls + afterok deps + dispatcher wrapper; tested at v4.0 close |
| L2 winner dispatch | `for m in winners; do sbatch ...; done` | `cluster/21_dispatch_l2_winners.sh` (CANONICAL block) | Uses `sbatch --wait & ... wait` pattern; alternatives explicitly rejected by plan 21-10 spec |
| Pre-flight pytest gate | Re-running tests as a separate plan task | Just invoke `bash cluster/21_submit_pipeline.sh` (runs gate inline at line 72) | Orchestrator already aborts on test failure with "[ABORT] No cluster jobs submitted" |
| CRLF stripping for Windows checkout | `dos2unix` task | `sed -i 's/\r$//'` line at orchestrator:56 | Already idempotent; runs every invocation |
| Per-job log push | Manual `git add output/bayesian/` after each step | `cluster/autopush.sh` sourced by every SLURM script | Auto-commits on every job completion |
| Convergence gate verification | Re-implementing R-hat/ESS/BFMI math | `pd.read_csv("output/bayesian/21_baseline/convergence_table.csv")` | Step 21.4 already computed it; just count `pipeline_action == "PROCEED_TO_LOO"` rows |
| Winner determination check | Re-running `az.compare()` | `cat output/bayesian/21_baseline/winners.txt` + grep `winner_report.md` for `Winner type:` | Step 21.5 already wrote the verdict; verification is a string match |
| SLURM accounting | Custom timing wrappers | `sacct -j <JID> --format=JobID,JobName,Elapsed,TotalCPU,MaxRSS,State,AllocTRES` | M3 standard; no extra deps |

**Key insight:** Phase 24 is **execution and audit**, not implementation. Every line of new code should be either (a) a `Bash` task that calls existing infrastructure, or (b) a `pd.read_csv()` / `Path.exists()` check on the resulting artifact. The planner should resist any task that touches `scripts/21_*.py` or `cluster/21_*.slurm` — those are frozen at v4.0 close and re-verified at Phase 23 close.

## Common Pitfalls

### Pitfall 1: CRLF on Windows checkout

**What goes wrong:** SLURM rejects scripts with `\r` line endings: `sbatch: error: Batch script contains DOS line breaks (\r\n)`.
**Why it happens:** Windows-checked-out repo has CRLF on `cluster/*.slurm` and `cluster/*.sh`. The user is on a Windows submit host (env shows Windows 11 in the gitStatus banner).
**How to avoid:** Already wired — `cluster/21_submit_pipeline.sh:56` does `sed -i 's/\r$//' cluster/*.slurm cluster/*.sh` before any sbatch. **But** the orchestrator must itself be runnable on the cluster (not Windows) — so the user's Wave 1 task is to `git pull` on the M3 login node and run `bash cluster/21_submit_pipeline.sh` THERE, not from the Windows dev box. Document this explicitly in 24-01-PLAN.md.
**Warning signs:** First sbatch call fails with the DOS line break error, no JobIDs returned, "[ABORT] No cluster jobs submitted" message.

### Pitfall 2: INCONCLUSIVE_MULTIPLE pause at step 21.5

**What goes wrong:** Step 21.5 exits with code 2 (NOT 0, NOT 1). Master orchestrator uses `afterok` so steps 21.6+ are silently CANCELLED by the scheduler. Pipeline halts mid-way.
**Why it happens:** Stacking weights are spread thinly: top model weight < 0.5 AND top-two combined < 0.8. The auto-determination falls through to "user must decide".
**How to avoid:** Cannot avoid without re-running 21.3 with bigger warmup/samples or explicitly forcing the winner set. Document in 24-01-PLAN.md monitoring task: if `winner_report.md` shows `Winner type: INCONCLUSIVE_MULTIPLE`, the operator (a) reads the report, (b) edits `output/bayesian/21_baseline/winners.txt`, and (c) manually submits `sbatch cluster/21_6_dispatch_l2.slurm` (no afterok dep needed; the dispatcher will read the edited winners.txt). Then re-chain 21.7→21.9 manually with afterok. **EXEC-04 forbids `FORCED` outcome**, so this branch is a phase-blocker that requires re-fitting, not just forcing.
**Warning signs:** Step 21.5 SLURM job state = `COMPLETED` but ExitCode = `2:0`; downstream 21.6+ jobs in CANCELLED state in `sacct`.

### Pitfall 3: M6b subscale 12-hour worst case

**What goes wrong:** Step 21.6 child for M6b (8-param stick-breaking, 4 covariates × 8 = 32 betas) takes up to 12 hours. The dispatcher (`21_6_dispatch_l2.slurm`) is wrapped at `--time=14:00:00` to absorb this. If wall clock exceeds 14h (extreme case: M6b non-converges), the dispatcher gets killed by scheduler before children finish, and `afterok` to step 21.7 silently cancels.
**Why it happens:** M6b stick-breaking parameterization has known funnel pathology even with non-centered reparameterization.
**How to avoid:** Document the 14h time cap as immutable. If M6b in winner set AND M6b L2 fit blows past 12h, the operator must monitor and either (a) accept partial pipeline (manually skip 21.7+ and use baseline M6b posterior), or (b) re-launch step 21.3 for M6b with more samples and start over. Wave 1 monitoring should flag M6b L2 fit progress at 8h elapsed time as a heads-up checkpoint.
**Warning signs:** `squeue` shows `21_dispatch_l2` with `TIME=12:00:00+`; `21_6_<JID>.out` for M6b shows last log line older than ~30 min.

### Pitfall 4: GPU contention (NON-issue for choice-only models)

**What goes wrong:** Misconception. There is **no GPU contention** in Phase 24 because all 6 choice-only baseline models run CPU-only. The user's MEMORY.md notes "CPU is correct for choice-only MCMC (vmap 7-13x slower); GPU only for M4 LBA". M4 LBA is **explicitly out of scope** for v5.0 (Phase 14 GPU items deferred to v5.1 per REQUIREMENTS.md).
**How to avoid:** No action needed. Document explicitly in 24-01-PLAN.md "Resources: 6 baseline jobs × `--cpus-per-task=4 --mem=48G --time=10h` on `comp` partition; ZERO GPU jobs in Phase 24 scope" so the planner doesn't waste a verification step on `--gres=gpu` checks.

### Pitfall 5: Auto-push commits during run

**What goes wrong:** Each SLURM job sources `cluster/autopush.sh` which does `git add logs/ output/bayesian/` then commit+push. During the cold-start run, **~30+ commits land on `main`** automatically. If the operator has WIP files staged at submission time, those get swept into the first auto-commit.
**Why it happens:** Designed convenience. The dev box and the M3 cluster share the same `origin` so artifacts flow back automatically.
**How to avoid:** Pre-flight check in 24-01-PLAN.md: `git status --porcelain` returns empty (clean working tree). If non-empty, abort and instruct the user to commit or stash WIP first.
**Warning signs:** Unexpected commits in `git log --oneline main` with messages like `bayes_21_3_baseline job <JID> results`.

### Pitfall 6: Path drift between ROADMAP and scripts

**What goes wrong:** ROADMAP success criterion #2 lists `output/bayesian/manuscript/{loo_stacking,rfx_bms,winner_betas}.{csv,md,tex}` and "winner-specific forest plot PNGs" without dir. Scripts write to `output/bayesian/21_tables/{table1_loo_stacking,table2_rfx_bms,table3_winner_betas}.{csv,md,tex}` (note the `table{1,2,3}_` prefix and `21_tables/` dir not `manuscript/`). Forest plots have **two paths** in the actual code: `figures/21_bayesian/forest_*.png` (per `21_9_manuscript_tables.slurm:145`) and `output/bayesian/figures/{internal}_forest.png` + `m6b_forest_lec5.png` (per `scripts/21_manuscript_tables.py:764-765`).
**Why it happens:** ROADMAP was written before the SLURM script was finalized; the discrepancy was not caught at v4.0 close.
**How to avoid:** Audit Plan (24-02) `must_haves` MUST list the actual paths from the scripts (see Artifact Inventory below). Flag the doc-drift in 24-02-SUMMARY for Phase 27 closure to resolve. Do NOT touch the scripts to match the ROADMAP — Phase 26 (manuscript finalization) and `paper.qmd` reference the actual paths.
**Warning signs:** `Path("output/bayesian/manuscript/loo_stacking.csv").exists()` returns False even though everything ran successfully.

## Artifact Inventory (must_haves for goal-backward verification)

This is the canonical list the planner should encode as `must_haves` in 24-02-PLAN.md. Sources are verified from each driver script and SLURM script.

### Step 21.1 outputs (per model — 6 models, so multiply by 6)

```
output/bayesian/21_prior_predictive/{model}_prior_sim.nc       # NetCDF, ArviZ
output/bayesian/21_prior_predictive/{model}_prior_accuracy.csv # per-draw accuracy
output/bayesian/21_prior_predictive/{model}_gate.md            # PASS/FAIL verdict
```

Where `{model}` ∈ `{qlearning, wmrl, wmrl_m3, wmrl_m5, wmrl_m6a, wmrl_m6b}`. **Total: 18 files.**

### Step 21.2 outputs (per model)

```
output/bayesian/21_recovery/{model}_subject_001.json  ... _050.json  # 50 per model = 300 JSONs
output/bayesian/21_recovery/{model}_recovery.csv                     # post-aggregation
output/bayesian/21_recovery/{model}_recovery_summary.md              # PASS/FAIL on kappa family
```

**Total: 300 JSONs + 6 CSVs + 6 MDs = 312 files.** Audit may skip JSONs (intermediates) and only verify the 12 aggregated outputs.

### Step 21.3 outputs (per model)

```
output/bayesian/21_baseline/{model}_individual_fits.csv
output/bayesian/21_baseline/{model}_posterior.nc        # the load-bearing artifact
output/bayesian/21_baseline/{model}_shrinkage_report.md
output/bayesian/21_baseline/{model}_ppc_results.csv
```

**Total: 24 files (6 models × 4 outputs).**

### Step 21.4 outputs

```
output/bayesian/21_baseline/convergence_table.csv     # the EXEC-04 verification surface
output/bayesian/21_baseline/convergence_report.md
```

**Total: 2 files.** `convergence_table.csv` columns (verified from `scripts/21_baseline_audit.py:371-385`): `model, max_rhat, min_ess_bulk, n_divergences, min_bfmi, ppc_coverage, gate_status, pipeline_action`.

### Step 21.5 outputs

```
output/bayesian/21_baseline/loo_stacking_results.csv  # az.compare DataFrame + pct_high_pareto_k
output/bayesian/21_baseline/rfx_bms_pxp.csv           # alpha, r, xp, bor, pxp, pxp_exceeds_95
output/bayesian/21_baseline/winner_report.md          # human-readable verdict; "Winner type: ..."
output/bayesian/21_baseline/winners.txt               # comma-separated display names: "M3,M6b" or "M6b"
```

**Total: 4 files.** `winner_report.md` contains the line `Winner type: **{type}**` where `{type}` ∈ `{DOMINANT_SINGLE, TOP_TWO, INCONCLUSIVE_MULTIPLE, FORCED}` — this is the EXEC-04 winner-determination assertion surface.

### Step 21.6 outputs (per WINNER, count varies)

```
output/bayesian/21_l2/{winner}_individual_fits.csv     # M1/M2 copy path may NOT produce this
output/bayesian/21_l2/{winner}_posterior.nc            # the load-bearing artifact
output/bayesian/21_l2/{winner}_shrinkage_report.md     # M1/M2 copy path may NOT produce this
output/bayesian/21_l2/{winner}_ppc_results.csv         # M1/M2 copy path may NOT produce this
```

`{winner}` is the internal model id mapped from display name via NAME_MAP in `21_dispatch_l2_winners.sh:48-55` (e.g. M3 → wmrl_m3). **For winners ∈ {M1, M2}: only the `_posterior.nc` is produced** (copy-through path per `21_6_fit_with_l2.slurm:170-173`). For winners ∈ {M3, M5, M6a, M6b}: all 4 files. The audit MUST conditionally check based on winner identity.

### Step 21.7 outputs

```
output/bayesian/21_l2/scale_audit_report.md           # YAML frontmatter pipeline_action key
output/bayesian/21_l2/{winner}_beta_hdi_table.csv     # one per winner
```

`scale_audit_report.md` YAML key `pipeline_action:` is one of `PROCEED_TO_AVERAGING` or `NULL_RESULT`. Both are exit-0 by design (unified-exit-0 protocol).

### Step 21.8 outputs

```
output/bayesian/21_l2/averaged_scale_effects.csv      # multi-winner primary artifact
output/bayesian/21_l2/averaging_summary.md            # multi-winner narrative
output/bayesian/21_l2/averaging_skipped.md            # NULL_RESULT branch (mutually exclusive with above 2)
output/bayesian/21_l2/single_winner_mode.md           # single-winner branch (mutually exclusive)
output/bayesian/21_l2/launch_subscale.flag            # cleared after sub-arm sbatch — should be ABSENT post-run
output/bayesian/wmrl_m6b_subscale_posterior.nc        # OPTIONAL fire-and-forget; only if M6b wins
```

Audit logic: exactly ONE of `{averaged_scale_effects.csv + averaging_summary.md, averaging_skipped.md, single_winner_mode.md}` should exist. `launch_subscale.flag` should NOT exist post-run.

### Step 21.9 outputs

```
output/bayesian/21_tables/table1_loo_stacking.{csv,md,tex}   # 3 files
output/bayesian/21_tables/table2_rfx_bms.{csv,md,tex}        # 3 files
output/bayesian/21_tables/table3_winner_betas.{csv,md,tex}   # 3 files
output/bayesian/21_tables/null_result_summary.md             # NULL_RESULT branch only
figures/21_bayesian/forest_{winner}.png                      # PROCEED branch only, one per winner
output/bayesian/figures/{internal}_forest.png                # also written per scripts/21_manuscript_tables.py:764
output/bayesian/figures/m6b_forest_lec5.png                  # if M6b wins, per :765
```

**Total in PROCEED branch: 9 table files + N forest PNGs (N = winner count).** Note the dual forest path discrepancy.

### Phase 24-specific outputs (NEW — Wave 2 produces these)

```
output/bayesian/21_execution_log.md                          # Wave 2 deliverable; SLURM accounting
output/bayesian/21_submission_<TIMESTAMP>.log                # Wave 1 tee log of orchestrator stdout
.planning/phases/24-cold-start-pipeline-execution/24-01-SUMMARY.md
.planning/phases/24-cold-start-pipeline-execution/24-02-SUMMARY.md
.planning/phases/24-cold-start-pipeline-execution/24-VERIFICATION.md
```

## Code Examples

### EXEC-03 execution-log enumerator (Wave 2 task)

Compose `output/bayesian/21_execution_log.md` from sacct output. Schema:

```bash
# Source: SLURM sacct manual + project autopush convention
# For each captured JobID from the Wave 1 tee log:
sacct -j ${JID} --format=JobID,JobName,Partition,Elapsed,TotalCPU,MaxRSS,State,ExitCode,AllocTRES,ReqMem -P --noheader
```

**Recommended Markdown table structure for `21_execution_log.md`:**

```markdown
# Phase 24 Cold-Start Execution Log

**Submitted:** {timestamp}
**Submission host:** {hostname}
**Cluster:** Monash M3
**Total wall-clock:** {start} → {end} ({elapsed})

## Per-step accounting

| Step | JobID | JobName | Partition | Elapsed | TotalCPU | MaxRSS | State | ExitCode | AllocTRES |
|---|---|---|---|---|---|---|---|---|---|
| 21.1 qlearning | {JID} | bayes_21_1_prior_pred | comp | HH:MM:SS | HH:MM:SS | NNN K | COMPLETED | 0:0 | cpu=4,mem=16G |
| 21.1 wmrl | ... |
| ... |

## Aggregate budget

| Resource | Total used | Notes |
|---|---|---|
| CPU-hours | {sum} | Sum of TotalCPU across all jobs |
| GPU-hours | 0 | Zero by design — all choice-only models CPU-only |
| Wall-clock | {elapsed} | Submission to terminus |
| Peak memory (any single job) | {max(MaxRSS)} | |
```

**Note:** TotalCPU is the right column for "CPU-hours" (sum of per-cpu time across all allocated CPUs). `Elapsed × cpus-per-task` is also valid but TotalCPU is what was actually consumed.

### EXEC-04 convergence-gate verification (Wave 2 task)

```python
# Source: convergence_table.csv schema verified from scripts/21_baseline_audit.py:371-385
import pandas as pd
df = pd.read_csv("output/bayesian/21_baseline/convergence_table.csv")
n_passing = (df["pipeline_action"] == "PROCEED_TO_LOO").sum()
assert n_passing >= 2, (
    f"EXEC-04 FAIL: only {n_passing} models met Baribault & Collins gate. "
    f"Per-model: {df[['model', 'pipeline_action']].to_dict('records')}"
)

# Verify the four threshold columns explicitly
gate_pass = (
    (df["max_rhat"] <= 1.05)
    & (df["min_ess_bulk"] >= 400)
    & (df["n_divergences"] == 0)
    & (df["min_bfmi"] >= 0.2)
)
assert gate_pass.sum() == n_passing, "Gate columns disagree with pipeline_action — investigate"
```

### EXEC-04 winner-determination check

```python
# Source: winner_report.md format verified from scripts/21_compute_loo_stacking.py:459
from pathlib import Path
report = Path("output/bayesian/21_baseline/winner_report.md").read_text()

# Find the "Winner type: **X**" line
import re
m = re.search(r"Winner type:\s*\*\*([A-Z_]+)\*\*", report)
assert m, "winner_report.md missing 'Winner type:' line"
winner_type = m.group(1)

# EXEC-04 forbids FORCED and INCONCLUSIVE_MULTIPLE
assert winner_type in ("DOMINANT_SINGLE", "TOP_TWO"), (
    f"EXEC-04 FAIL: winner_type={winner_type!r}; expected DOMINANT_SINGLE or TOP_TWO. "
    f"FORCED requires user override; INCONCLUSIVE_MULTIPLE means stacking was non-decisive."
)

# Cross-check: winners.txt should have 1 entry for DOMINANT_SINGLE, 2 for TOP_TWO
winners_txt = Path("output/bayesian/21_baseline/winners.txt").read_text().strip()
n_winners = len([w for w in winners_txt.split(",") if w.strip()])
expected = {"DOMINANT_SINGLE": 1, "TOP_TWO": 2}[winner_type]
assert n_winners == expected, f"winners.txt has {n_winners} entries; expected {expected} for {winner_type}"
```

### NetCDF integrity check (Wave 2 task — bonus depth beyond `Path.exists()`)

```python
# Source: config.load_netcdf_with_validation (Phase 23 CLEAN-04 deliverable, config.py:754)
from pathlib import Path
from config import load_netcdf_with_validation, MODEL_REGISTRY

baseline_models = ["qlearning", "wmrl", "wmrl_m3", "wmrl_m5", "wmrl_m6a", "wmrl_m6b"]
for model in baseline_models:
    path = Path(f"output/bayesian/21_baseline/{model}_posterior.nc")
    idata = load_netcdf_with_validation(path, model)  # raises on corruption
    # Sanity: posterior contains the named parameters
    expected_params = MODEL_REGISTRY[model]["params"]
    missing = [p for p in expected_params if p not in idata.posterior.data_vars]
    assert not missing, f"{model}: missing posterior vars {missing}"
```

This catches partial-write NetCDFs that pass `Path.exists()` but fail `az.from_netcdf()`.

## State of the Art

This phase has no "state of the art" surface — it's execution of an already-shipped pipeline. The pipeline itself is anchored to:

| Anchor paper | DOI | Pipeline role |
|---|---|---|
| Baribault & Collins (2023) | [10.1037/met0000554](https://doi.org/10.1037/met0000554) | Convergence gate (R-hat ≤ 1.05, ESS ≥ 400, divergences = 0, BFMI ≥ 0.2) at step 21.4 |
| Hess et al. (2025) | [10.5334/cpsy.116](https://doi.org/10.5334/cpsy.116) | Staged Bayesian workflow: prior predictive → recovery → fits → comparison |
| Yao, Vehtari, Simpson, Gelman (2018) | [10.1214/17-BA1091](https://doi.org/10.1214/17-BA1091) | LOO + stacking as primary ranking (step 21.5) |
| Stephan et al. (2009) + Rigoux et al. (2014) | [10.1016/j.neuroimage.2009.03.025](https://doi.org/10.1016/j.neuroimage.2009.03.025), [10.1016/j.neuroimage.2013.08.065](https://doi.org/10.1016/j.neuroimage.2013.08.065) | RFX-BMS + protected exceedance probability (secondary, step 21.5) |

All four references are already cited in `manuscript/paper.qmd:1033` (verified by Phase 21 VERIFICATION). Phase 24 does not need to update them.

## Open Questions

### 1. Submit host for `bash cluster/21_submit_pipeline.sh`

**What we know:** The orchestrator must be invoked on a host where (a) `pytest` runs the local pre-flight (the gate runs with the dev/cluster Python env, not Windows), (b) `sbatch` is on PATH (Monash M3 login node), and (c) `git` can push (autopush.sh).
**What's unclear:** Whether the user runs the orchestrator from the M3 login node (typical) or from WSL on the Windows dev box (unusual). The first sbatch call in `21_submit_pipeline.sh` will fail on a non-cluster host with `command not found`.
**Recommendation for the planner:** 24-01-PLAN.md task 1 should explicitly require `ssh m3.massive.org.au` (or whatever the M3 login URL is) and `cd ~/Documents/.../rlwm_trauma_analysis && git pull && bash cluster/21_submit_pipeline.sh`. Document the M3 login as the canonical submit host.

### 2. Wall-clock budget — 50-96 GPU-hours estimate

**What we know:** ROADMAP states "~50-96 GPU-hours total". Per-step time caps from SLURM `#SBATCH --time=`:
- 21.1 prior predictive: 1.5 h × 6 models = 9 h max
- 21.2 recovery: 1.5 h × 50 array tasks × 6 models = 450 task-hours (but parallel, so ~1.5 h wall)
- 21.2 aggregate: 0.5 h × 6 = 3 h
- 21.3 baseline: 10 h × 6 (parallel) = ~10 h wall worst case
- 21.4 audit: 0.5 h
- 21.5 LOO+stacking: 2 h
- 21.6 dispatcher: 14 h cap (M6b worst case)
- 21.7 audit: 0.5 h
- 21.8 averaging: 1 h
- 21.9 tables: 0.5 h
**Sum (sequential wall-clock estimate, parallel within steps):** ~10 + 1.5 + 3 + 10 + 0.5 + 2 + 14 + 0.5 + 1 + 0.5 ≈ **43 h wall** under worst-case caps. ROADMAP's "50-96 GPU-hours" likely refers to TotalCPU (cpus × elapsed), not wall — `21_3_fit_baseline.slurm` uses `--cpus-per-task=4`, so 6 jobs × 4 cpus × 10 h = 240 CPU-hours just for step 21.3.
**What's unclear:** Whether "GPU-hours" in the ROADMAP is a misnomer (Phase 24 has zero GPU jobs) and should read "CPU-hours". The planner should clarify in 24-02 SUMMARY: zero GPU-hours, ~250-400 CPU-hours, 24-48h wall-clock typical.
**Recommendation:** 24-01-PLAN.md should set operator expectations: kick off, expect to monitor for 24-48 hours, plan a 2-day window before checking status.

### 3. What does "clean working tree" mean for autopush?

**What we know:** Autopush does `git add logs/ output/bayesian/` — only those two paths. So pre-existing modified files outside those paths (e.g., the WIP files currently shown in `gitStatus`) are NOT swept into the auto-commits.
**What's unclear:** Whether modified files in `logs/` or `output/bayesian/` (not currently the case per the gitStatus banner) would pollute. None visible at HEAD.
**Recommendation:** 24-01 pre-flight should `git status --porcelain logs/ output/bayesian/` and refuse to proceed if either has uncommitted changes. Other paths (cluster/, scripts/, output/regressions/, etc.) can be dirty.

### 4. Should Phase 24 verify CPU-vs-GPU env?

**What we know:** Choice-only models are CPU-only by design (MEMORY.md: "vmap 7-13x slower on GPU"). All Phase 21 SLURM scripts use `--partition=comp` (not `--partition=gpu`) and set `JAX_PLATFORMS=cpu`. So Phase 24 jobs cannot accidentally land on GPU.
**What's unclear:** Whether the operator should also verify M4 LBA jobs are NOT submitted (M4 is out of v5.0 scope). Looking at the orchestrator: `MODELS="qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b"` (line 53) — no `wmrl_m4`. So M4 is structurally excluded from the chain.
**Recommendation:** No verification needed; document that line 53's MODELS list is the source of truth.

## Sources

### Primary (HIGH confidence — all from repo HEAD)

- `cluster/21_submit_pipeline.sh` — Master orchestrator, line numbers cited verbatim
- `cluster/21_1_prior_predictive.slurm` — Step 21.1 SLURM (resource caps, output paths, exit codes)
- `cluster/21_2_recovery.slurm` + `cluster/21_2_recovery_aggregate.slurm` — Step 21.2 array + post-aggregation
- `cluster/21_3_fit_baseline.slurm` — Step 21.3 (baseline fits)
- `cluster/21_4_baseline_audit.slurm` — Step 21.4 (convergence gate; pipeline-block on < 2 models)
- `cluster/21_5_loo_stacking_bms.slurm` — Step 21.5 (tri-state exit codes documented in script header)
- `cluster/21_6_dispatch_l2.slurm` + `cluster/21_dispatch_l2_winners.sh` + `cluster/21_6_fit_with_l2.slurm` — L2 dispatch + child fits
- `cluster/21_7_scale_audit.slurm` — Step 21.7 (unified exit-0 protocol; YAML pipeline_action)
- `cluster/21_8_model_averaging.slurm` — Step 21.8 (subscale fire-and-forget arm)
- `cluster/21_9_manuscript_tables.slurm` — Step 21.9 (table + forest plot writer)
- `cluster/autopush.sh` — Auto-commit hook sourced by every SLURM script
- `scripts/fitting/tests/test_numpyro_models_2cov.py` — Pre-flight pytest target (9 fast + 1 slow tests)
- `scripts/21_baseline_audit.py` — convergence_table.csv schema (lines 371-385) + gate logic
- `scripts/21_compute_loo_stacking.py` — Winner determination tri-state logic (lines 322-359)
- `scripts/21_manuscript_tables.py` — Output table/figure paths (lines 764-765, 990)
- `validation/check_v4_closure.py` — Closure-guard pattern template (dataclass CheckResult, deterministic checks)
- `config.py:754` — `load_netcdf_with_validation` for NetCDF integrity validation
- `.planning/REQUIREMENTS.md` — EXEC-01..04 + scope decisions
- `.planning/ROADMAP.md` — Phase 24 success criteria (lines 370-389)
- `.planning/phases/21-principled-bayesian-model-selection-pipeline/21-VERIFICATION.md` — Phase 21 cluster-pending framing
- `.planning/phases/23-tech-debt-sweep-pre-flight-cleanup/23-VERIFICATION.md` — Local SUMMARY/VERIFICATION conventions
- `.planning/milestones/v4.0-MILESTONE-AUDIT.md` — Cluster-execution-pending framing for Phase 21

### Secondary (MEDIUM confidence)

- Per-script docstrings cited inline above (high credibility within-repo, but not external standards)

### Tertiary (LOW confidence)

- None. This research is entirely repo-grounded; no WebSearch / Context7 / WebFetch was needed because Phase 24 is execution of existing infrastructure, not implementation of a new domain.

## Metadata

**Confidence breakdown:**

- Standard stack (existing infrastructure): **HIGH** — every tool and path verified by direct file reads
- Architecture patterns (orchestrator flow, monitoring, audit): **HIGH** — line-numbered citations from `cluster/21_submit_pipeline.sh` and per-step SLURM scripts
- Pitfalls (CRLF, INCONCLUSIVE_MULTIPLE, M6b 12h, path drift): **HIGH** — pitfalls 1, 2, 3 are documented in script comments; pitfalls 5, 6 are findings from this research
- Open questions: **MEDIUM** — items 1 (submit host) and 3 (clean tree definition) are operational questions the planner should resolve in Wave 1 task descriptions

**Research date:** 2026-04-19
**Valid until:** 30 days (the cluster scripts are frozen at v4.0 close; the only thing that can invalidate this research is a Phase 23-style sweep that touches `cluster/21_*.slurm` or `scripts/21_*.py` — which is out-of-scope for v5.0)

**Open Questions for the planner to resolve in 24-01-PLAN.md (not blockers; just decisions to encode):**

1. Submit host: explicitly call out M3 login node (not Windows dev box).
2. Wall-clock expectations: ~24-48h, not 50-96 (the ROADMAP figure conflates GPU-hours and CPU-hours; v5.0 has zero GPU jobs).
3. Pre-flight `git status --porcelain logs/ output/bayesian/` (not whole repo — autopush only touches those two paths).
4. Resume protocol on `INCONCLUSIVE_MULTIPLE`: document the manual edit-and-resubmit path, but flag that EXEC-04 forbids this outcome — it's a phase blocker requiring re-fit, not a recoverable pause.
