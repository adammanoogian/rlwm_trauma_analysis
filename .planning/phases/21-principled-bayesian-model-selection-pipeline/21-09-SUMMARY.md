---
phase: 21-principled-bayesian-model-selection-pipeline
plan: 09
subsystem: bayesian-fitting
tags: [arviz, stacking, yao-2018, model-averaging, canonical-key, subscale-arm, slurm, phase21]

# Dependency graph
requires:
  - phase: 21-principled-bayesian-model-selection-pipeline/21-06
    provides: output/bayesian/21_baseline/loo_stacking_results.csv (stacking weights per display-name model)
  - phase: 21-principled-bayesian-model-selection-pipeline/21-07
    provides: output/bayesian/21_l2/{winner}_posterior.nc per winner (2-cov for M3/M5/M6a, 32-beta subscale for M6b, copy-through for M1/M2)
  - phase: 21-principled-bayesian-model-selection-pipeline/21-08
    provides: output/bayesian/21_l2/scale_audit_report.md (YAML pipeline_action header) + {winner}_beta_hdi_table.csv (single-winner HDI flags for disagreement detection)
  - phase: 16-m6b-subscale
    provides: cluster/13_bayesian_m6b_subscale.slurm + fit_bayesian --subscale path writing wmrl_m6b_subscale_posterior.nc
provides:
  - scripts/21_model_averaging.py (~600 lines) — stacking-weighted mixture averaging across winners with canonical-key matching
  - cluster/21_8_model_averaging.slurm (~217 lines) — 1h/32G/2-CPU/comp submission with fire-and-forget subscale arm chain
  - Three exit-0 short-circuit paths (NULL_RESULT, single-winner, multi-winner) all reporting via --dependency=afterok-compatible exit codes
  - Canonical-key normalisation rule for the 2-cov vs subscale beta-site cardinality mismatch
  - Option (a) subscale-compatibility decision (plan 21-10 reads from Phase-16 canonical path, no patch to subscale SLURM)
affects:
  - 21-10 master pipeline orchestrator (reads averaged_scale_effects.csv for manuscript Table 3; reads wmrl_m6b_subscale_posterior.nc IF it appeared by manuscript-build time)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Canonical-key matching via `_total` suffix stripping — M6b subscale `beta_lec_total_kappa` and M3/M5 2-cov `beta_lec_kappa` collapse to `('lec', 'kappa')` key, enabling cross-winner averaging even when winners have different beta cardinalities (2 vs 32 sites)"
    - "Stacking-weighted subsampling: `n_sub = int(w * TARGET_TOTAL_SAMPLES=8000)` without replacement via seeded `np.random.default_rng(42)` keeps concatenated mixture bounded for stable `az.hdi` while preserving tail-p resolution"
    - "Three exit-0 short-circuit paths (NULL_RESULT / single-winner / multi-winner) mirror step-21.7 unified exit-0 semantics — all are valid scientific outcomes, not errors"
    - "Fire-and-forget subscale-arm chain: Python writes marker file, SLURM reads it and `sbatch --parsable`'s the subscale SLURM, then continues through autopush + exit WITHOUT waiting for the 12h fit — main pipeline chain NEVER blocks on exploratory arm"
    - "Subscale-exclusive families reported from sole winner with `single_source=True` + `source_winner_if_single` column (user-approved Option C: no scientifically defensible way to average what doesn't exist in the 2-cov winners)"
    - "Disagreement detection via canonical-key re-parsing: compare averaged `excludes_zero` vs. plan-08 per-winner `excludes_zero_hdi`; conservative OR rule (any contributing winner disagreeing flags the row for manuscript scrutiny)"
    - "NumPy 2.x `np.bool_` coercion: `isinstance(np.True_, bool)` is False in numpy >= 2.0, so CSV-read booleans require explicit `isinstance(v, (bool, np.bool_))` check to avoid silent True→False inversion of disagreement flags"

key-files:
  created:
    - scripts/21_model_averaging.py
    - cluster/21_8_model_averaging.slurm
    - .planning/phases/21-principled-bayesian-model-selection-pipeline/21-09-SUMMARY.md
  modified:
    - .planning/STATE.md

key-decisions:
  - "Canonical-key matching strips `_total` suffix so M6b subscale betas on `lec_total`/`iesr_total` match the 2-cov 2-cov winners' `lec`/`iesr` betas by key — enables cross-winner mixture concatenation across winners with different beta cardinalities"
  - "TARGET_TOTAL_SAMPLES = 8000 for the concatenated mixture — large enough for stable `az.hdi` and tail-p resolution (distinguishes p=0.01 from p=0.02), bounded enough to fit comfortably in RAM"
  - "Subscale-exclusive keys (M6b's `iesr_intr_resid` / `iesr_avd_resid`) reported from M6b alone with `single_source=True` per user-approved Option C — no scientifically defensible way to average what doesn't exist in the 2-cov winners' posteriors"
  - "Option (a) for subscale-path compatibility: plan 21-10 reads the subscale posterior from Phase-16 canonical `output/bayesian/wmrl_m6b_subscale_posterior.nc` rather than patching the subscale SLURM to accept OUTPUT_SUBDIR. Rationale: keeps the Phase-16 contract stable across downstream consumers; an exploratory arm shouldn't force a file-layout change on a production Phase-16 script"
  - "Fire-and-forget subscale arm: SLURM wrapper reads `launch_subscale.flag` marker, `sbatch --parsable`'s the subscale SLURM, removes the marker to prevent double-submission on re-runs, and continues without waiting for the 12h fit. Main pipeline chain (step 21.9) is NOT blocked on the subscale fit completing"
  - "Three short-circuit paths all exit 0 (mirror plan-checker Issue #4 unified exit-0 semantics from step 21.7): NULL_RESULT audit header → `averaging_skipped.md`, single winner → `single_winner_mode.md`, multi-winner → primary averaging CSV + summary MD. Exit 1 reserved for genuine errors (FileNotFoundError / KeyError / RuntimeError) so `--dependency=afterok:$AVG_JID` chain naturally advances for valid outcomes"
  - "Disagreement detection uses conservative OR rule: if ANY contributing winner's plan-08 `excludes_zero_hdi` differs from the averaged `averaged_excludes_zero`, flag the row. Surfaces keys most likely to shift the manuscript narrative after averaging; does not silently overwrite single-winner inferences"

patterns-established:
  - "Canonical-key matching rule for the 2-cov vs subscale beta-site cardinality mismatch — strip `_total` suffix on `lec_total` / `iesr_total`, keep subscale-only families (`iesr_intr_resid` / `iesr_avd_resid`) full names"
  - "Marker-file subscale arm chain: Python writes `launch_subscale.flag`, SLURM reads + clears it. Separation of concerns keeps the Python script SLURM-agnostic and keeps sbatch lifecycle decisions in the cluster wrapper"
  - "Exit-0 semantics for three outcomes (NULL / single-winner / multi-winner-averaging) — afterok chain advances naturally without forcing the master orchestrator to branch on the averaging result"

# Metrics
duration: 35min
completed: 2026-04-18
---

# Phase 21 Plan 09: Step 21.8 Model Averaging Summary

**Stacking-weighted posterior mixture averaging across winners via Yao et al. (2018) — canonical-key matching handles mixed beta cardinality (2-cov vs 32-beta subscale); fire-and-forget M6b-subscale exploratory arm chained to the main averaging SLURM via a marker-file contract.**

## Performance

- **Duration:** ~35 min
- **Started:** 2026-04-18
- **Completed:** 2026-04-18
- **Tasks:** 2
- **Files created:** 3 (script + SLURM + SUMMARY)

## Accomplishments

- `scripts/21_model_averaging.py` (~600 lines, pure ArviZ + pandas + numpy) — hand-written YAML front-matter parser for the step-21.7 `pipeline_action` header (avoids adding PyYAML to ds_env beyond the already-loaded trio), three short-circuit paths all exit 0 (NULL_RESULT / single-winner / multi-winner), stacking-weighted subsampling `n_sub=int(w*8000)` without replacement via seeded `np.random.default_rng(42)`, canonical-key parser with `_total` suffix stripping (`beta_lec_total_kappa` ↔ `beta_lec_kappa` → `("lec", "kappa")`), subscale-exclusive-family path via `single_source=True` + `source_winner_if_single` column (user-approved Option C), disagreement detection vs plan-08 `{winner}_beta_hdi_table.csv` with `_coerce_bool` helper that handles strings, Python bool, and numpy 2.x `np.bool_` (load-bearing — the initial draft silently inverted every True flag because NumPy 2.x stripped the `bool`/`int` subclass relationship from `np.bool_`), optional M6b-subscale exploratory arm marker write (`launch_subscale.flag`) governed by `--launch-subscale-arm` / `--no-launch-subscale-arm` CLI pair.
- `cluster/21_8_model_averaging.slurm` (~217 lines, 1h/32G/2-CPU/comp, no JAX) — ds_env conda ladder matching `cluster/21_7_scale_audit.slurm`, 6 env-var overrides (L2_DIR / STACKING_RESULTS / WINNERS_FILE / AUDIT_REPORT / OUTPUT_DIR / LAUNCH_SUBSCALE), pre-flight arviz+pandas+numpy import check, fire-and-forget subscale-arm chain (`sbatch --parsable` returns JID immediately; wrapper continues through autopush + exit without waiting for the 12h subscale fit), marker cleanup after submission (prevents double-submission on pipeline re-runs), LAUNCH_SUBSCALE=0 --export hook to disable the arm globally even when M6b is a winner (useful for dry runs). `grep -c "launch_subscale.flag\|SUBSCALE" cluster/21_8_model_averaging.slurm` = 30 matches (threshold was >= 2).
- All four success-criteria-enumerated dry-runs verified end-to-end against stub NetCDFs generated via `arviz.from_dict`: (1) `--help` shows all 7 arguments cleanly; (2) NULL_RESULT audit header short-circuits to `averaging_skipped.md` and exits 0; (3) single-winner `winners.txt='M3'` short-circuits to `single_winner_mode.md` and exits 0; (4) multi-winner M3+M6b with overlapping `beta_lec_kappa` + `beta_lec_total_kappa` both collapsed to `("lec", "kappa")` with `n_winners_contributing=2`, subscale-exclusive `iesr_intr_resid|kappa` flagged `single_source=True, source_winner_if_single=wmrl_m6b, n_winners_contributing=1`, `launch_subscale.flag` written when M6b is in winners AND default `--launch-subscale-arm`. Also verified `--no-launch-subscale-arm` override (flag NOT written) and M3+M5-only winner set (no subscale marker, 2 overlapping keys averaged correctly). Disagreement detection verified via deliberate plan-08 CSV flip (M3 `beta_lec_kappa` excludes_zero=False forced) → `disagreement_flag=True` on averaged `lec|kappa` row.

## Task Commits

1. **Task 1: Stacking-weighted model averaging script** — `1e9b3c4` (feat)
2. **Task 2: SLURM with fire-and-forget subscale arm chain** — `30d53d5` (feat)

_Metadata commit will follow — `docs(21-09): complete model averaging plan`._

## Files Created/Modified

- `scripts/21_model_averaging.py` — three short-circuit exit-0 paths, canonical-key `_total` suffix stripping rule, weighted-sample concatenation mixture with `TARGET_TOTAL_SAMPLES=8000`, `_flag_disagreements` with numpy-bool-aware `_coerce_bool`, marker-file subscale-arm hook (`--launch-subscale-arm` default True). CLI: `--l2-dir / --stacking-results / --winners-file / --audit-report / --output-dir / --launch-subscale-arm / --no-launch-subscale-arm / --rng-seed` (8 args total including `--help`).
- `cluster/21_8_model_averaging.slurm` — 1h/32G/2-CPU/comp allocation, ds_env ladder (identical to 21_7_scale_audit pattern), 6 env-var overrides, fire-and-forget subscale submission with marker cleanup. Env-var contract: `LAUNCH_SUBSCALE=0` disables the arm globally.
- `.planning/STATE.md` — plan count 9→10 of 11 complete, Current Position + Last activity bumped to 21-09, new "Plan 21-09" decision entry prepended to Phase 21 Decisions section.

## Decisions Made

- **Canonical-key matching via `_total` suffix stripping.** `beta_lec_total_kappa` (M6b subscale) and `beta_lec_kappa` (M3/M5 2-cov) collapse to the same canonical key `("lec", "kappa")` via a `_COVARIATE_CANONICAL` mapping dict. Without this, the two posteriors would live in parallel CSV rows with no cross-winner averaging ever triggered. Preserves raw site name for diagnostic traceability (logged in `[AVG]` stdout line). Subscale-only families (`iesr_intr_resid`, `iesr_avd_resid`) keep their full names — they have no 2-cov counterpart, so no suffix to strip. `_COVARIATE_FAMILIES` ordered longest-prefix-first so `iesr_intr_resid` matches before `iesr_total` matches before `iesr` (load-bearing — alphabetising would break the parser).
- **`TARGET_TOTAL_SAMPLES = 8000` for the concatenated mixture.** Large enough that `az.hdi` is well-conditioned and `tail_p` has enough resolution to distinguish p=0.01 from p=0.02 (requires at least ~200 samples on each tail). Bounded enough that a 6-winner mixture fits comfortably in RAM. If a winner's posterior has fewer than `int(w * 8000)` samples, the helper returns the full posterior as a degenerate case rather than crashing.
- **Subscale-exclusive keys reported from single winner (user-approved Option C).** Keys like `("iesr_intr_resid", "kappa")` exist only in M6b's subscale posterior, never in M3/M5/M6a's 2-cov posteriors. Three options were discussed upstream: (A) drop them entirely, (B) average with a zero-mean dummy from the absent winners, (C) report verbatim from M6b with `single_source=True`. Option C was locked as the only scientifically defensible path — A loses real information, B invents values that don't exist in the posterior. The `source_winner_if_single` column surfaces the origin so manuscript tables can disclose "reported from M6b alone".
- **Fire-and-forget subscale-arm marker-file contract.** The Python script writes `launch_subscale.flag` — a marker file, not a sbatch call — when M6b is a winner AND `--launch-subscale-arm` is on (default True). The SLURM wrapper reads the marker, fires `cluster/13_bayesian_m6b_subscale.slurm` via `sbatch --parsable` (returns the subscale JID immediately), removes the marker, and continues through autopush + exit WITHOUT waiting. Rationale: (a) separation of concerns — Python stays SLURM-agnostic, sbatch lifecycle lives in the bash wrapper; (b) marker cleanup prevents double-submission on pipeline re-runs; (c) fire-and-forget means the main chain (step 21.9 via `--dependency=afterok:$AVG_JID`) is NEVER blocked on the 12h subscale fit — plan 21-10 reads the subscale posterior IF it appeared by manuscript-build time.
- **Option (a) for subscale-path compatibility.** Plan 21-09 spec presented two options for reconciling the subscale SLURM's default output path (`output/bayesian/wmrl_m6b_subscale_posterior.nc` from Phase 16) with the 21_l2/ directory target: (a) accept the Phase-16 path and have plan 21-10 read from there; (b) patch the subscale SLURM to accept `OUTPUT_SUBDIR` env var. Locked Option (a) — keeps the Phase-16 contract stable across downstream consumers; an exploratory arm shouldn't force a file-layout change on a production Phase-16 script that already has its own SLURM history and test coverage. Plan 21-10's manuscript build reads from `output/bayesian/wmrl_m6b_subscale_posterior.nc` directly.
- **LAUNCH_SUBSCALE=0 --export override.** Even when M6b is a winner and the marker file is written, setting `LAUNCH_SUBSCALE=0` in the SLURM --export skips the sbatch submission (with a loud `[SUBSCALE ARM] ... NOT submitted (user override)` log line) and retains the marker for manual submission later. Primary use case: dry runs that don't want to burn a 12h cluster slot on a fresh exploratory arm.
- **Disagreement conservative OR rule.** For multi-winner averaged keys, loop through all contributing winners and set `disagreement_flag=True` if ANY winner's plan-08 `excludes_zero_hdi` differs from the averaged `averaged_excludes_zero`. Conservative (flag up, not down) for manuscript scrutiny — these are the rows most likely to shift the narrative after averaging, so surface them loudly rather than silently overwriting single-winner inferences.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — Bug] NumPy 2.x `np.bool_` no longer subclasses Python `bool` or `int`, silently inverting every disagreement flag**

- **Found during:** Task 1 verification with the multi-winner M3+M6b stub (`iesr_intr_resid|kappa` and `lec|kappa` rows were flagged `disagreement_flag=True` despite the plan-08 stub explicitly setting both to `excludes_zero_hdi=True` matching the averaged flag).
- **Issue:** The initial `_coerce_bool` helper handled Python `bool`, Python `int`/`float`, and string cases but fell through to `return False` for `numpy.bool_`. NumPy 2.x (installed numpy 2.3.5) stripped the `bool`/`int` subclass relationship that numpy 1.x maintained: `isinstance(np.True_, bool)` returns `False`, `isinstance(np.True_, (int, float))` returns `False`, and the bare `bool(v)` on the column would have been string-truthy on the object-dtype column anyway (because the plan-08 metadata trailer row forces the entire `excludes_zero_hdi` column to object dtype — a CSV round-trip idiom covered in plan 21-08's SUMMARY). The net effect: every True flag read from the plan-08 CSV got coerced to False, which then differed from the averaged_excludes_zero=True result on all strong-effect rows, flagging them all as false-positive disagreements.
- **Fix:** Extended `_coerce_bool` with explicit `isinstance(value, (bool, np.bool_))` branch (returns `bool(value)`) and widened the numeric branch to `(int, float, np.integer, np.floating)` for forward-compat with numpy scalar arrays. Docstring now calls out the NumPy 2.x subclass behaviour explicitly so future maintainers don't re-introduce the bug.
- **Files modified:** `scripts/21_model_averaging.py` (`_coerce_bool` helper, ~8 lines).
- **Verification:** Re-ran the multi-winner M3+M6b stub — `disagreements: 0` (down from 2 false positives). Verified disagreement detection still triggers correctly by flipping M3's `beta_lec_kappa` plan-08 flag to `excludes_zero_hdi=False` → `disagreement_flag=True` on the averaged `lec|kappa` row (expected, since averaged excludes_zero=True differs from M3's single-winner False).
- **Commit:** `1e9b3c4` (folded into Task 1 commit, pre-initial-push).

---

**Total deviations:** 1 auto-fixed (1 bug).
**Impact on plan:** Zero scope creep. The `np.bool_` coercion fix is a NumPy 2.x compatibility issue unrelated to the scientific logic; without it, every averaged row with an agreeing True flag would have been falsely reported as a disagreement in the manuscript table, potentially misleading reviewers into scrutinising stable effects.

## Issues Encountered

None — all verification passes:

- `python scripts/21_model_averaging.py --help` shows all 7 arguments plus the `--launch-subscale-arm` / `--no-launch-subscale-arm` mutually-exclusive pair.
- NULL_RESULT path: stub audit report with `pipeline_action: NULL_RESULT` → `averaging_skipped.md` written, exit 0.
- Single-winner path: `winners.txt` containing only `M3` → `single_winner_mode.md` written with correct internal-id + display-name reference, exit 0.
- Multi-winner path (M3 2-cov + M6b subscale stub):
  - `beta_lec_kappa` (M3) and `beta_lec_total_kappa` (M6b) correctly collapsed to canonical key `("lec", "kappa")` with `n_winners_contributing=2`, `single_source=False`.
  - `beta_iesr_kappa` (M3) and `beta_iesr_total_kappa` (M6b) correctly collapsed to `("iesr", "kappa")` with `n_winners_contributing=2`.
  - Subscale-exclusive keys `iesr_intr_resid|kappa` and `iesr_avd_resid|*` correctly flagged `single_source=True` + `source_winner_if_single=wmrl_m6b` + `n_winners_contributing=1`.
  - `launch_subscale.flag` written when M6b is in winners AND `--launch-subscale-arm` (default).
  - `--no-launch-subscale-arm` override: flag NOT written.
  - M3+M5-only winner set (no M6b): no flag written, 2 overlapping keys averaged correctly (both `lec|kappa` and `iesr|kappa` n_winners=2).
- Disagreement detection after `_coerce_bool` numpy-bool fix: stub with all matching flags → 0 disagreements; stub with deliberately flipped M3 plan-08 `beta_lec_kappa` excludes_zero=False → 1 disagreement (lec|kappa), detection working as designed.
- SLURM `grep -c "launch_subscale.flag\|SUBSCALE" cluster/21_8_model_averaging.slurm` = 30 (criterion was >= 2).

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Step 21.9 (plan 21-10, master pipeline orchestrator + manuscript table build) can consume `output/bayesian/21_l2/averaged_scale_effects.csv` for its Table 3 generation. Columns are stable and documented in the script docstring + this summary.
- Master orchestrator (plan 21-10) is unblocked for its Wave 8 step: `sbatch --dependency=afterok:$AUDIT2_JID cluster/21_8_model_averaging.slurm`. Both NULL_RESULT and single-winner short-circuits exit 0 so afterok chain naturally advances to step 21.9 — the soft-skip semantics are entirely inside the Python script + marker-file contract.
- M6b-subscale exploratory arm is queued automatically when M6b is a winner, but does NOT block the main chain — plan 21-10's manuscript build reads `output/bayesian/wmrl_m6b_subscale_posterior.nc` IF it appeared by the time the build runs. Fire-and-forget semantics mean a slow 12h subscale fit never holds up the primary averaged-effects report.
- Plan 21-10 remaining items: master pipeline orchestrator (SLURM chain step 21.1 → step 21.9 with afterok dependencies and checkpoint hooks) + manuscript table/figure generation scripts (Tables 2/3 + residualisation sanity figure). Subscale-arm read contract locked per this plan's Option (a) decision.

---
*Phase: 21-principled-bayesian-model-selection-pipeline*
*Completed: 2026-04-18*
