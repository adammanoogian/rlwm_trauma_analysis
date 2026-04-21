---
phase: 21-principled-bayesian-model-selection-pipeline
plan: 08
subsystem: bayesian-fitting
tags: [arviz, statsmodels, fdr-bh, hdi, level2-regression, audit, slurm, phase21]

# Dependency graph
requires:
  - phase: 21-principled-bayesian-model-selection-pipeline/21-07
    provides: output/bayesian/21_l2/{winner}_posterior.nc for each L2-refitted winner
  - phase: 21-principled-bayesian-model-selection-pipeline/21-06
    provides: output/bayesian/21_baseline/winners.txt (comma-separated display names)
  - phase: 21-principled-bayesian-model-selection-pipeline/21-05
    provides: output/bayesian/21_baseline/{winner}_posterior.nc for ESS-degradation comparison
  - phase: 21-principled-bayesian-model-selection-pipeline/21-11
    provides: beta_iesr_{target} site naming convention so pattern-match enumeration captures the IES-R family
provides:
  - scripts/21_scale_audit.py (~650 lines) — per-winner beta HDI + FDR-BH + ESS degradation audit
  - cluster/21_7_scale_audit.slurm (~192 lines) — 30min/16G/2-CPU/comp submission template with YAML-header awk parse
  - Unified exit-0 semantics (plan-checker Issue #4) — PROCEED_TO_AVERAGING and NULL_RESULT both exit 0
  - YAML-front-matter pipeline_action header that downstream 21.8 model averaging reads for self-skip
affects:
  - 21-09 model averaging / posterior summarisation (reads pipeline_action from scale_audit_report.md YAML header)
  - 21-10 master pipeline orchestrator (uses --dependency=afterok:$AUDIT2_JID for step 21.8 submission)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Pattern-match beta_* enumeration via `startswith('beta_')` captures all three L2 tiers uniformly (M1/M2: 0, M3/M5/M6a: 2, M6b subscale: 32) without model-family branching"
    - "Longest-prefix-first covariate/target parser for beta site names — `iesr_intr_resid` before `iesr`; `kappa_total` and `kappa_share` before `kappa`"
    - "FDR-BH applied per-winner (not cross-winner) on surrogate posterior-tail-probability p-values — documented as an HDI-agreement diagnostic, not frequentist FDR"
    - "ESS-degradation check via min-ESS ratio on NON-beta (shared) parameters between 21_l2 and 21_baseline fits — apples-to-apples because baseline has no L2 sites"
    - "Unified exit-0 semantics: PROCEED_TO_AVERAGING and NULL_RESULT both exit 0; exit 1 reserved for genuine audit errors; YAML front-matter `pipeline_action` header distinguishes the two soft outcomes for downstream self-skip logic"

key-files:
  created:
    - scripts/21_scale_audit.py
    - cluster/21_7_scale_audit.slurm
    - .planning/phases/21-principled-bayesian-model-selection-pipeline/21-08-SUMMARY.md
  modified:
    - .planning/STATE.md

key-decisions:
  - "Pattern-match beta_* enumeration over a hand-coded per-model whitelist — a single `startswith('beta_')` check correctly handles 0, 2, and 32 sites across the three L2 tiers with no branching; future covariates added downstream won't need code changes here"
  - "FDR-BH on posterior tail probabilities is documented as an HDI-AGREEMENT DIAGNOSTIC, not a frequentist false-discovery rate. Caveat surfaced at the TOP of the report body so reviewers hit it before scanning individual beta rows"
  - "Unified exit-0 semantics (plan-checker Issue #4): NULL_RESULT is valid science, not an audit error. Both PROCEED and NULL exit 0; YAML header distinguishes them. Exit 1 reserved for FileNotFoundError / ValueError / RuntimeError from the core audit loop only"
  - "ESS-degradation check compares L2 vs baseline min-ESS on NON-beta params (shared subset) — baseline has no L2 sites so this is apples-to-apples. Degraded winners get WARNING section + `ess_degraded_models` YAML list entry but still exit 0"
  - "ImportError on statsmodels.stats.multitest is a FATAL audit error (exit 1) with a clear install message, not a soft null. Env drift should fail loudly at script startup rather than halfway through per-winner processing"

patterns-established:
  - "YAML front-matter for machine-readable pipeline actions — awk single-pass parse in SLURM (`/^---$/{flag=!flag;next} flag && /^pipeline_action:/{print $2; exit}`) avoids re-reading NetCDFs or re-running Python for the decision"
  - "Per-winner CSV with metadata trailer row (beta_site='__METADATA__') carrying max_rhat_betas / min_ess_betas / ess_degraded — keeps single-table CSV while surfacing per-model diagnostics without a separate header file"
  - "Beta-site enumeration log `[AUDIT] {winner}: {N} beta sites: [list]` is deliberately verbose so SLURM stdout makes the IES-R family inclusion (plan 21-11's 2-cov hook) observable to reviewers at a glance"

# Metrics
duration: 22min
completed: 2026-04-18
---

# Phase 21 Plan 08: Scale-Fit Audit Summary

**Scale-fit audit between step 21.6 L2 refit and step 21.8 model averaging — per-winner beta HDI + FDR-BH + ESS degradation check with YAML-front-matter pipeline_action header driving unified exit-0 semantics.**

## Performance

- **Duration:** ~22 min
- **Started:** 2026-04-18
- **Completed:** 2026-04-18
- **Tasks:** 2
- **Files created:** 3 (script + SLURM + SUMMARY)

## Accomplishments

- `scripts/21_scale_audit.py` (~650 lines) — loads each winner's `21_l2/{winner}_posterior.nc`, pattern-matches all `beta_*` data vars (captures 0 sites for M1/M2 copy-through, 2 sites for M3/M5/M6a 2-cov path per plan 21-11, 32 sites for M6b subscale), computes posterior mean/SD/HDI/tail_p per site, applies FDR-BH across the full beta-site set per winner, compares L2 vs baseline min-ESS on NON-beta parameters, writes per-winner `{winner}_beta_hdi_table.csv` + aggregated `scale_audit_report.md`.
- `cluster/21_7_scale_audit.slurm` (~192 lines, 30min/16G/2-CPU/comp, no JAX) — ds_env conda ladder, pre-flight arviz/statsmodels import check, 7 env-var overrides (HDI_PROB/FDR_ALPHA/ESS_DROP_THRESHOLD/L2_DIR/BASELINE_DIR/WINNERS_FILE/OUTPUT_DIR), awk YAML-header parse with case-statement routing for PROCEED_TO_AVERAGING / NULL_RESULT / unrecognized, `source cluster/autopush.sh`, `exit $EXIT_CODE` (load-bearing for afterok chain).
- Unified exit-0 semantics verified via synthetic stub subprocesses: PROCEED_TO_AVERAGING exits 0 when at least one beta survives FDR-BH; NULL_RESULT exits 0 when zero betas survive (not exit 1 — null result is valid science per plan-checker Issue #4); exit 1 fires only on `FileNotFoundError` (missing winners.txt, corrupt NetCDF) or `ImportError` (statsmodels missing).

## Task Commits

1. **Task 1: Scale audit script with FDR-BH gate** — `1bea8ca` (feat)
2. **Task 2: SLURM for step 21.7** — `5f846d7` (feat)

## Files Created/Modified

- `scripts/21_scale_audit.py` — 11-column per-beta-site table + metadata trailer; longest-prefix-first covariate/target parser (`_COVARIATE_FAMILIES = (iesr_intr_resid, iesr_avd_resid, iesr, lec)`, `_KNOWN_TARGETS = (alpha_pos, alpha_neg, phi, rho, capacity, epsilon, phi_rl, kappa_total, kappa_share, kappa_s, kappa)`); `@dataclass` records `BetaSite` + `WinnerAudit`; `_audit_one_winner` handles missing-NetCDF soft-skip via `load_warning` rather than exit 1 (downstream orchestrator already blocked via afterok); separate `_min_ess_on_non_beta` for baseline/L2 shared-subset ratio; `_format_yaml_header` + `_format_report_body` split for clean separation of machine-readable and human-readable output.
- `cluster/21_7_scale_audit.slurm` — matches 21_4_baseline_audit.slurm pattern (pure ArviZ, no JAX); 7 env-var overrides; awk parse of YAML front-matter mirrors the plan-08 task spec exactly (`/^---$/{flag=!flag;next} flag && /^pipeline_action:/{print $2; exit}`); per-winner `*_beta_hdi_table.csv` listing via `ls -la` globs (count varies by winner set).

## Decisions Made

- **Pattern-match enumeration over per-model whitelist.** A single `[v for v in idata.posterior.data_vars if v.startswith("beta_")]` correctly handles all three L2 tiers (M1/M2: 0 sites, M3/M5/M6a: 2 sites each, M6b subscale: 32 sites) with no model-family branching. Future covariates added upstream in plan 21-11 or Phase 16 will be picked up automatically. The explicit `[AUDIT] {winner}: {N} beta sites: [list]` log line makes enumeration observable in SLURM stdout so a reviewer can trust at a glance that plan 21-11's IES-R family (2 beta sites on M3/M5/M6a) actually made it into the saved NetCDF.
- **FDR-BH per-winner, not cross-winner.** The full set of beta sites per winner is the unit of multiple-comparisons correction. M6b's 32 sites form one correction group; M3/M5/M6a's 2 sites form one correction group each. Cross-winner FDR would conflate unrelated hypotheses (kappa in M3 vs kappa_share in M6b are not the same effect). Caveat documented in report body: FDR-BH on posterior tail probabilities is an HDI-AGREEMENT DIAGNOSTIC, not frequentist FDR — the posterior tail p is a credibility measure, not a null-hypothesis error rate.
- **Unified exit-0 semantics (plan-checker Issue #4).** PROCEED_TO_AVERAGING and NULL_RESULT both exit 0. Exit 1 is reserved for `FileNotFoundError` (missing winners.txt), `ValueError` (malformed winners.txt or bad beta naming), and `RuntimeError` (corrupt NetCDF, `az.from_netcdf` failure). ImportError on statsmodels is also exit 1 with a clear install message at the top-of-module guard. This lets the master orchestrator use `--dependency=afterok:$AUDIT2_JID` for step 21.8 without NULL_RESULT falsely blocking the chain — 21_model_averaging.py reads the YAML header and self-skips internally.
- **ESS degradation on shared (NON-beta) params only.** Baseline has no L2 sites so the shared subset IS the baseline's full parameter set. Comparing L2 `non_beta min_ess` vs baseline `min_ess` is apples-to-apples. Degraded winners (ratio < 1 - ess_drop_threshold, default 0.5) get a WARNING block + entry in `ess_degraded_models` YAML list, but still exit 0 — degradation is surfaced loudly for user judgement, not auto-kill.
- **Per-winner CSV metadata via trailer row.** Instead of a separate header file or YAML-in-CSV hybrid, the trailer row has `beta_site='__METADATA__'` and squeezes `max_rhat_betas` / `min_ess_betas` / `ess_degraded` into the last two columns as formatted strings. Keeps single-table CSV round-trippable via `pd.read_csv` while still carrying per-winner diagnostics.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] argparse percent-literal escape in --ess-drop-threshold help string**

- **Found during:** Task 1 verification (`python scripts/21_scale_audit.py --help`)
- **Issue:** argparse's `_expand_help` applies printf-style `%`-substitution to help text before rendering. The help string `"default 0.5 = flag if L2 min-ESS drops below 50% of baseline"` contained a literal `%` that argparse interpreted as a format specifier, raising `TypeError: %o format: an integer is required, not dict`. `--help` crashed before printing any arguments.
- **Fix:** Escaped the literal as `50%%` per argparse's documented escaping convention.
- **Files modified:** `scripts/21_scale_audit.py` (one line in the `--ess-drop-threshold` help kwarg).
- **Verification:** `python scripts/21_scale_audit.py --help` now renders all 7 arguments cleanly; the rendered help text shows `50%` correctly (argparse unescapes on output).
- **Commit:** `1bea8ca` (folded into Task 1 commit, pre-initial-push).

---

**Total deviations:** 1 auto-fixed (1 blocking).
**Impact on plan:** Zero scope creep. The `50%%` escape is a standard argparse idiom unrelated to the scientific logic. All success criteria from the plan still hold.

## Issues Encountered

None — all verification passes:

- `python scripts/21_scale_audit.py --help` shows all 7 arguments (l2-dir, baseline-dir, winners-file, hdi-prob, fdr-alpha, ess-drop-threshold, output-dir).
- Statsmodels import guard verified via `python -c "from statsmodels.stats.multitest import multipletests"` OK.
- Dry-run with 2-cov stub (M3 with `beta_lec_kappa` signal + `beta_iesr_kappa` null): pipeline_action=PROCEED_TO_AVERAGING, exit 0, YAML header + per-winner CSV written correctly.
- Dry-run with all-null betas: pipeline_action=NULL_RESULT, exit 0 (not 1).
- Dry-run with missing winners.txt: exit 1, `[FATAL] FileNotFoundError` on stderr.
- M6b subscale stub with 32 beta sites across 4 covariate families: all enumerated correctly, longest-prefix-first parser handles `iesr_intr_resid_kappa_total` and `iesr_avd_resid_kappa_share` without collision.
- M1 copy-through (zero betas): handled gracefully, pipeline_action=NULL_RESULT (nothing to survive FDR), exit 0, per-winner CSV written with just the metadata trailer row.
- SLURM awk parse of YAML front-matter: `awk '/^---$/{flag=!flag;next} flag && /^pipeline_action:/{print $2; exit}' scale_audit_report.md` returns `PROCEED_TO_AVERAGING` correctly.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Step 21.8 (plan 21-09, model averaging / posterior summarisation) can consume `output/bayesian/21_l2/scale_audit_report.md` YAML front-matter to decide whether to run averaging or self-skip. The `pipeline_action` header is parseable via either awk (in SLURM) or PyYAML (in Python).
- Master pipeline orchestrator (plan 21-10) is unblocked for its Wave 7 step: `sbatch --dependency=afterok:$L2_JID cluster/21_7_scale_audit.slurm`. The NULL_RESULT path does NOT block afterok because exit 0 is preserved — the soft-skip is entirely inside `21_model_averaging.py`.
- Per-winner `beta_hdi_table.csv` is the canonical source of per-site statistics for manuscript Table 3 generation in plan 21-10 (covariate_family is already a column so groupby-family aggregates are trivial).

---
*Phase: 21-principled-bayesian-model-selection-pipeline*
*Completed: 2026-04-18*
