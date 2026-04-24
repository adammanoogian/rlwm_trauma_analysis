# Phase 24: Cold-Start Pipeline Execution - Context

**Gathered:** 2026-04-24
**Status:** Ready for planning (plans 24-01, 24-02 pre-dating this context need revision — see "Implications for existing plans" at bottom)

<domain>
## Phase Boundary

Execute the 9-step Bayesian pipeline end-to-end on Monash M3 (`bash cluster/21_submit_pipeline.sh`, afterok-chained) and audit the resulting empirical artifacts. Deliverables:

- Full artifact set at **CCDS-canonical paths** (`models/bayesian/21_*/` + `reports/tables/model_comparison/` + `reports/figures/`)
- SLURM execution log (`models/bayesian/21_execution_log.md`)
- Winner-determination record (conforming to the pre-declared acceptance policy below)

Downstream: Phase 25 (reproducibility regression against v4.0) and Phase 26 (manuscript finalization) consume these artifacts. What Phase 24 does NOT include: introducing new pipeline steps, re-specifying priors, adding models to the pool, changing the EXEC-04 requirement text (Phase 25 handles any post-hoc EXEC-04 rewrite).

</domain>

<decisions>
## Implementation Decisions

### Submission strategy

- **Canary-first**: Submit ONLY `cluster/21_1_prior_predictive.slurm` as a canary (~15-30 min wall). Do not submit the full 9-step chain until the canary passes all four acceptance criteria below.
- **Canary acceptance gate (ALL four must pass)**:
  1. SLURM exit code 0 AND expected `.nc` files exist for all 6 models at the canary's declared output directory.
  2. Each `.nc` loads via `arviz.from_netcdf()` without error AND has the expected posterior-predictive dim structure (chains × draws × subjects × trials).
  3. `git log` on the artifact directory shows a fresh commit from the SLURM autopush hook (end-to-end cluster → repo flow validated).
  4. Artifacts land at the **Phase 31 CCDS-canonical path** (`models/bayesian/21_prior_predictive/...`), NOT legacy `output/bayesian/...` — validates the Area 4 path-patch.
- **Canary → full chain handoff**: Human-gated. Operator inspects outputs manually, then resumes with `bash cluster/21_submit_pipeline.sh` edited (or `--skip-21-1` flag) to start at step 21.2. No auto-cascade, no semi-auto delay window.
- **Canary failure policy**: Abort + root-cause. Diagnose, patch, re-run pytest gate, fresh canary submission. Do NOT fix-forward past a failing canary; do NOT retry the same canary on the assumption of cluster flakiness.
- **Pre-flight pytest gate**: Runs on the Monash M3 login node only (`pytest scripts/fitting/tests/test_numpyro_models_2cov.py -k "not slow"` — the existing `cluster/21_submit_pipeline.sh` contract). No dev-box pytest gate.

### Failure-recovery policy (no "retry" semantics)

- **"Retry" is rejected as a framing.** Three apparent motivations collapse:
  - **MCMC re-seeding after divergence** = seed-shopping / p-hacking. Banned.
  - **Infra "retry" (timeout, OOM, node eviction)** is really a **resource-allocation adjustment**. Bump `--mem=` or `--time=` in the SLURM file, commit, fresh resubmit (same seed). Not called a "retry" in the audit log.
  - **MCMC diagnostic failure (R-hat > 1.05, divergences > 0)** is NOT a SLURM failure at all — job exits 0. Routed to the convergence-acceptance policy below, not to recovery.
- **SLURM resource spec**: Conservative padding — `--time=` at ~1.5× max observed v4.0 wall-clock per step; `--mem=` at ~1.5× peak MaxRSS. Trade queue time for one-shot success.
- **Mid-chain SLURM failure (timeout/OOM)**: Keep completed-step artifacts. Patch the failing step's SLURM spec in a single atomic commit. Resubmit only the failing step + afterok-chain the remaining downstream steps via `--dependency=afterok:<new_jid>`. Same seed; fresh JobID.
- **Mid-job completion with failing diagnostics** (job exits 0, R-hat/divergences fail): log-and-continue. Do NOT halt the chain. Step 21.5 LOO+stacking will filter via the Area 3 pool-inclusion rule.
- **Wave 2 audit provenance**: Single sacct entry per successful step (JobID, Elapsed, MaxRSS, State, AllocTRES). Resource-adjustment events get one-line operator notes — no "retry chain" table.

### Convergence / winner acceptance (PRE-DECLARED before seeing cold-start data)

Policy locked up front to prevent threshold-shopping post-hoc. Whatever the cold-start produces is the result.

- **If < 2 models meet the Baribault & Collins (2023) gate** (R-hat ≤ 1.05 AND ESS_bulk ≥ 400 AND divergences = 0 AND BFMI ≥ 0.2): **accept and document**. Report actual convergence state in the manuscript. No re-specification of priors (that would be prior-shopping). Phase 25 will rewrite the EXEC-04 text at audit time to match reality — EXEC-04 is an aspiration, not a pass/fail gate.
- **If winner determination returns `INCONCLUSIVE_MULTIPLE`** (3+ models with stacking weight > threshold): **accept as `TOP_K`**. Manuscript reports the full tied set; hierarchical L2 regression runs on each of them (or on model-averaged betas per Yao et al. 2018 BMA). No artificial winner-forcing by tie-break. Phase 25/26 must handle arbitrary winner-set sizes.
- **Non-convergent models (failing Baribault gate) are EXCLUDED from the LOO+stacking pool.** Justification: a non-convergent posterior produces a biased ELPD estimate; Yao et al. (2018) stacking requires valid ELPD. Stack only over the Baribault-passing subset.
- **`FORCED` winner is treated identically to `INCONCLUSIVE_MULTIPLE`** — report all candidates, no single winner. The `FORCED` label in the pipeline's decision code becomes a pure diagnostic with no scientific implication.

### Path-drift reconciliation (Wave 0 pre-phase required)

- **Target layout: Phase 31 CCDS-canonical paths.** Cold-start writes directly to the final locations; no post-run mv/cp.
  - Model artifacts: `models/bayesian/21_prior_predictive/`, `21_recovery/`, `21_baseline/`, `21_l2/`
  - Manuscript tables: `reports/tables/model_comparison/table{1,2,3}_*.{csv,md,tex}`
  - Figures (forest plots etc.): `reports/figures/`
- **Patch scope: ALL 9 steps** (not just manuscript tables). Every output path across 21.1-21.9 migrates to CCDS. Estimated ~20-30 path constants to update across SLURM + Python.
- **Patch happens BEFORE the cold-start** — Wave 0 (or new sub-plan 24-00). Canary 21.1 validates the patched path contract before committing to the full chain.
- **Wave 2 audit script policy**: `validation/check_phase24_artifacts.py` enforces a **single canonical CCDS path**. No fallback to legacy `output/bayesian/...`. If any artifact appears at a legacy path, audit fails. Forces the patch to be complete.
- **Doc drift fixed atomically with code patch**. Same commit (or commit group) that patches SLURM + Python also updates:
  - ROADMAP Phase 24 SC#2 (currently references `output/bayesian/manuscript/...`)
  - `.planning/REQUIREMENTS.md` rows for EXEC-02/03/04 that cite legacy paths
  - `validation/check_phase24_artifacts.py` artifact expectations
- **`cluster/21_submit_pipeline.sh` inspection**: Pre-phase should grep the orchestrator itself for hardcoded legacy paths (e.g., pytest gate target, `winners.txt` location) and patch if present.

### Claude's Discretion

- Specific `--time=` and `--mem=` numbers per step (plug in v4.0 max × 1.5 from sacct history; planner/researcher pull the numbers)
- sacct query format for execution log (JobID, Elapsed, MaxRSS, State, AllocTRES at minimum; additional columns at researcher/planner's choice)
- Exact canary inspection checklist format (operator runbook detail)
- Implementation of the Wave 0 patch — file-by-file sweep vs. grep-and-replace script vs. sed one-liner
- Whether the Wave 0 patch lives in `24-01-PLAN.md` as a new Wave 0 section or becomes a sibling `24-00-PLAN.md` — planner's call based on task atomicity

</decisions>

<specifics>
## Specific Ideas

- **"Accept reality" philosophy** extends across failure-recovery AND convergence acceptance. A failing result is a scientific result, not a problem to rescue by re-running with different seeds / priors / thresholds.
- **Phase 25 reproducibility regression depends on seed discipline.** That's why infra "retries" keep the same seed — so a rerun-after-OOM is byte-identical to the hypothetical un-interrupted run. This is the precise reason the "retry" vocabulary is misleading: a same-seed resubmit after a node eviction is *the same run*, not a *re-run*.
- **Canary 21.1 was chosen (not 21.2) because** its wall-clock is the shortest (~15-30 min vs. 2-3h for 21.2 recovery arrays) — the canary's job is to catch path-drift and env bugs cheaply, not to stress-test every model's fit path. 21.2 would catch more but costs more; the tradeoff favors minimum-exposed-compute for a canary.
- **Phase 31 closure claim** (ROADMAP: "future models/ artifacts will land directly in CCDS layout") is contradicted by the pre-Phase-31 SLURM scripts. The Wave 0 path-patch is what *actually* makes that claim true. Phase 24 completes the Phase 31 migration.

</specifics>

<deferred>
## Deferred Ideas

- **EXEC-04 gate text rewrite** — Phase 25 territory. The current text ("≥ 2 models meeting gate; winner ∈ {DOMINANT_SINGLE, TOP_TWO}") becomes aspirational, not pass/fail. Phase 25's audit will propose new wording that matches whatever the cold-start produced.
- **Forest plot regeneration for multi-winner sets** — Phase 26 territory. Phase 26's plan 26-02 already covers "Forest plots for every winner + limitations rewrite"; this now means "every winner in the potentially-larger-than-2 set."
- **Hierarchical L2 regression on multi-winner set** — Phase 24's 21.6 dispatcher already loops over `winners.txt`. If `winners.txt` contains 3+ winners, the dispatcher naturally runs 3+ L2 refits. No new capability needed; just verify the dispatcher doesn't cap at 2.
- **SBC (simulation-based calibration)** — v5.1 material, not Phase 24.
- **Full-chain submission without canary** — considered and rejected (submission-strategy Area 1).
- **Seed variation for MCMC sensitivity analysis** — explicit seed-shopping; rejected at the policy level.

</deferred>

<implications>
## Implications for Existing Plans

`24-01-PLAN.md` and `24-02-PLAN.md` were drafted 2026-04-19 with `--skip-discussion`, so they predate the decisions above. Revisions needed:

**24-01-PLAN revisions:**
- Add **Wave 0** (path-patch pre-phase) before the current Wave 1. Wave 0 outputs: patched SLURM + Python drivers + ROADMAP + REQUIREMENTS, all CCDS-canonical, pytest gate green.
- Change Wave 1 from "full-chain submission" to **canary 21.1 submission + human-gated handoff**. Add the four-criteria canary acceptance gate as explicit validation steps.
- Add explicit canary-failure-handling section (abort + root-cause, no retry, no fix-forward past canary).
- Operator briefing: document the "no retries" framing — SLURM failures are resource-spec bugs, not retries; diagnostic failures are log-and-continue.

**24-02-PLAN revisions:**
- Audit script (`validation/check_phase24_artifacts.py`) expects CCDS paths only; no legacy fallback.
- Convergence acceptance policy pre-declared in audit code (not a pass/fail gate — a reporting surface). Audit reports: # models passing Baribault, winner-determination label (DOMINANT_SINGLE / TOP_TWO / TOP_K / FORCED-as-INCONCLUSIVE), members of stacking pool (Baribault-passing only).
- Wave 2 `execution_log.md` has single sacct entry per step, no retry-chain table; resource-adjustment events get one-line operator notes.
- Add explicit note that EXEC-04 text may not match reality — Phase 25 will reconcile.

**24-RESEARCH.md implications:**
- Research's "follow existing infrastructure verbatim" recommendation is **superseded** for path outputs (Wave 0 patch required). Rest of research still authoritative.

</implications>

---

*Phase: 24-cold-start-pipeline-execution*
*Context gathered: 2026-04-24*
