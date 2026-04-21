# Phase 12: Cross-Model Integration - Research Findings

**Researched:** 2026-04-02
**Phase:** Final integration pass (v3.0)
**Question answered:** What is the actual current state of INTG-01 through INTG-05?

---

## Executive Summary

The pipeline wiring for M4-M6 was done incrementally during Phases 8-11. Most of
INTG-01/02/03 is already complete. The real work in Phase 12 is:

1. **INTG-04** (model recovery cross-validation) — no infrastructure exists for it
2. **INTG-05** (documentation) — MODEL_REFERENCE.md is severely out of date
3. **Smoke-testing** that the full `--model all` and auto-detection paths actually work end-to-end with all 7 models present simultaneously

---

## INTG-01: Script 14 — Current State

**Status: COMPLETE with one gap**

Script `14_compare_models.py` fully implements the two-track design:

- `find_mle_files()` (line 542) auto-detects M1, M2, M3, M4, M5, M6a, M6b from
  `output/mle/` with fallback to `output/`
- Explicit `--m1` through `--m6b` and `--m4` flags are all present
- `fits_dict.pop('M4', None)` at line 668 separates M4 before any AIC comparison
- Choice-only comparison runs on `{M1, M2, M3, M5, M6a, M6b}` (whatever was auto-detected)
- M4 gets its own printed section with per-param mean/SEM and a saved CSV
  (`output/model_comparison/m4_separate_track_summary.csv`)

**Gap: stratified comparison** at line 801 runs `stratified_comparison(choice_only_dict, ...)`
— this passes whatever is in `choice_only_dict` directly, which now potentially
includes 6 models (M1-M3, M5, M6a, M6b). The Fisher's exact test logic (lines 392-414)
is hardcoded for a 2x2 table and only handles the top 2 winners. With 6 choice-only
models, the Fisher test section will always reduce to a 2-model winner comparison, which
is correct behavior. No code change required — this is by design.

**Verdict: No changes needed to script 14.** Auto-detection handles all models. The two
comparison tracks are correctly separated.

---

## INTG-02: Script 15 — Current State

**Status: COMPLETE**

`15_analyze_mle_by_trauma.py`:

- `WMRL_M5_PARAMS`, `WMRL_M6A_PARAMS`, `WMRL_M6B_PARAMS`, `WMRL_M4_PARAMS` are all
  defined at lines 102-104 (verified in file)
- `load_data()` (lines 131-265) loads all 7 model files defensively — M5, M6a, M6b, M4
  fall back to `output/` root if not found in `output/mle/`
- `MODEL_CONFIG` is built conditionally: M5/M6a/M6b/M4 are inserted only when their
  DataFrames are non-None (lines 836-843)
- `--model all` expands to `['qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a',
  'wmrl_m6b', 'wmrl_m4']` filtered by `MODEL_CONFIG` membership (lines 846-848)
- `PARAM_NAMES` display dict (lines 113-129) includes `phi_rl`, `kappa_s`,
  `kappa_total`, `kappa_share`, `v_scale`, `A`, `delta`, `t0`

**Verdict: No changes needed. `--model all` correctly skips models whose fit files
don't exist.**

---

## INTG-03: Script 16 — Current State

**Status: COMPLETE**

`16_regress_parameters_on_scales.py`:

- `--model all` expands to all 7 models at line 743
- `load_integrated_data()` renames all new parameter columns (`phi_rl`, `kappa_s`,
  `kappa_total`, `kappa_share`, `v_scale`, `A`, `delta`, `t0`) to `*_mean` suffix
  (lines 164-182)
- Per-model `param_cols` dispatch (lines 792-807) has branches for `wmrl_m5`,
  `wmrl_m6a`, `wmrl_m6b`, `wmrl_m4`

**One potential issue:** The `load_integrated_data()` function takes a single
`params_path` argument and a `model_type` string. For `wmrl_m4`, `model_type` still
falls into the `else` branch (line 152: `else: # wmrl, wmrl_m3, or wmrl_m5`) — this
branch handles all WMRL-family models via column presence checks, so M4's LBA params
(`v_scale`, `A`, `delta`, `t0`) are renamed correctly. The comment on line 152 is
stale (should say "or wmrl_m4, wmrl_m6a, wmrl_m6b") but the logic is correct.

**Verdict: No functional changes needed. The stale comment in `load_integrated_data`
is cosmetic.**

---

## INTG-04: Model Recovery Cross-Validation — Current State

**Status: NOT IMPLEMENTED — significant work required**

The `run_model_recovery_check()` function in `scripts/fitting/model_recovery.py`
(line 1206) is the existing infrastructure for cross-model recovery. However, it only
fits `['qlearning', 'wmrl', 'wmrl_m3']` (line 1253 — hardcoded list of 3 models).

**What INTG-04 requires:**
"Each generating model wins by AIC against competitors" — meaning:
- Generate data from M5 → fit M1-M3+M5+M6a+M6b → M5 should win
- Generate data from M6a → fit M1-M3+M5+M6a+M6b → M6a should win
- Generate data from M6b → fit M1-M3+M5+M6a+M6b → M6b should win
- M4 is choice+RT so it cannot be compared against choice-only models in the
  same AIC table — M4 cross-model recovery is separate (M4 vs M4 only)

**Existing infrastructure gap:** `run_model_recovery_check()` needs to be extended
to include M5, M6a, M6b in the comparison set. There is no script that calls
`run_model_recovery_check()` — it was never wired into script 11 or any pipeline
script. It's a standalone function only.

**Practical approach for INTG-04:**
The most practical implementation is to extend `run_model_recovery_check()` with a
`comparison_models` parameter defaulting to the choice-only set
`['qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b']`, then wire it
into script 11 via a `--mode model-recovery` flag (distinct from the existing
`--mode recovery` parameter-recovery mode in `model_recovery.py`).

Script 11 (`11_run_model_recovery.py`) currently only calls `run_parameter_recovery()`
— it does not call `run_model_recovery_check()` at all.

**M4 cross-model note:** M4 uses joint choice+RT likelihood; its AIC is not comparable
to choice-only models. For M4 model recovery, the only valid check is that M4 vs
(choice-only models fit to the same RT+choice data using choice likelihood only) — but
this requires fitting choice-only models to RT data while ignoring RT, which is unusual.
The simplest valid approach for M4 is to skip the cross-model AIC comparison entirely
and rely on parameter recovery (M4-10) as the identifiability check.

---

## INTG-05: Documentation — Current State

**Status: SIGNIFICANTLY OUT OF DATE**

`docs/03_methods_reference/MODEL_REFERENCE.md` (1002 lines) contains:
- Section 1: Overview table — only M1 (Q-Learning) and M2 (WM-RL) listed
- Section 2: Q-Learning math (complete)
- Section 3: WM-RL Hybrid math (complete) — this is the base for M3-M6
- Section 4: Comparison to Senta et al. (2025) — mentions "Intentional Simplifications"
  including features that were later implemented as M3-M6
- Section 9: Code structure listing — references old file names (`fit_with_jax.py`,
  `fit_both_models.py`) that no longer exist; doesn't mention `lba_likelihood.py`

**What's missing from MODEL_REFERENCE.md:**
| Missing Content | Location Needed |
|----------------|-----------------|
| M3 WM-RL+κ perseveration math | New section 4 or subsection of 3 |
| M5 RL forgetting (phi_rl decay) math | New section |
| M6a stimulus-specific perseveration math | New section |
| M6b dual perseveration + stick-breaking | New section |
| M4 LBA joint choice+RT math | New section (large) |
| Updated model overview table (M1-M6) | Section 1 |
| Updated parameter tables for M3-M6 | Per-section |
| Updated code structure (lba_likelihood.py) | Section 9 |
| Model comparison tracks explanation | New section |

**Section 4.2 ("Intentional Simplifications")** currently lists features as NOT
implemented that were subsequently implemented:
- "Information sharing (i)" → implemented as M3's kappa
- The text needs to be revised to reflect what was actually built

**`CLAUDE.md` quick reference gaps:**
The `CLAUDE.md` quick reference shows `--model wmrl_m3` as the final model in
the pipeline but doesn't document `wmrl_m4`, `wmrl_m5`, `wmrl_m6a`, `wmrl_m6b`.
The "Model Fitting" section needs updating.

---

## Key Gaps Summary

| Requirement | Gap Level | Estimated Effort |
|-------------|-----------|------------------|
| INTG-01: Script 14 two tracks | No gap — complete | 0 |
| INTG-02: Script 15 --model all | No gap — complete | 0 |
| INTG-03: Script 16 --model all | No gap — complete | 0 |
| INTG-04: Cross-model recovery | Significant gap — no infrastructure for M5/M6 recovery check | Medium (1-2 plans) |
| INTG-05: MODEL_REFERENCE.md | Large gap — M3-M6 undocumented | Medium-Large (1 plan) |
| INTG-05: CLAUDE.md quick ref | Small gap | Small |

---

## Phase Structure Implications

Phase 12 needs two work units:

**Plan 01 — Model Recovery Cross-Validation (INTG-04)**
Extend `run_model_recovery_check()` in `model_recovery.py` to accept a configurable
`comparison_models` list (default: `['qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5',
'wmrl_m6a', 'wmrl_m6b']`). Add a `--mode cross-model` option to script 11 that calls
it for each generating model and produces a confusion matrix showing generator vs winner.
M4 is excluded from the cross-model comparison for the AIC reason noted above.

**Plan 02 — Documentation (INTG-05)**
Update `docs/03_methods_reference/MODEL_REFERENCE.md` to add M3-M6 model math.
Update `CLAUDE.md` quick reference section.
The key decisions for the docs plan:
- Merge into the single existing file (per CLAUDE.md doc standards: "merge, don't multiply")
- Model overview table expands to 7 rows (M1-M6, noting M4 is choice+RT only)
- Section 4.2 is revised (not left stale)
- Section 9 code structure updated to mention `lba_likelihood.py`

---

## Specific Technical Notes for Planning

### For INTG-04 implementation

The existing `run_model_recovery_check()` function uses `subprocess` to call
`scripts/12_fit_mle.py` for each model — this is the correct approach. The extension
needed is:
1. Replace the hardcoded `models = ['qlearning', 'wmrl', 'wmrl_m3']` with a parameter
2. For choice-only generators (M5, M6a, M6b), the comparison set is all choice-only
   models, not M4
3. A confusion matrix printout: rows = generative model, columns = winning model

The confuse-matrix structure already exists conceptually in the return value
(`confusion_entry` tuple at line 1309). It just needs to be run for all generators
and aggregated.

### For INTG-05 documentation

M3 math (perseveration/kappa) needs to be documented. Kappa is not mentioned in the
current MODEL_REFERENCE.md at all despite being the winning model (M3). The kappa
mechanism: on each trial, the previous action for the current stimulus generates a
perseveration bonus added to the hybrid policy before softmax, scaled by kappa.

M5 math: Q-value decay before delta-rule update. `Q(s,a) ← (1-phi_rl)*Q(s,a) + phi_rl*Q0`
applied before the standard delta-rule. phi_rl=0 reduces to M3.

M6a math: stimulus-specific perseveration. Per-stimulus last-action carry (integer array,
sentinel -1 for first presentation). kappa_s scales the perseveration bonus independently
per stimulus rather than globally.

M6b math: dual perseveration. Both global kappa (M3-style) and stimulus-specific
kappa_s (M6a-style). Stick-breaking parameterization: kappa_total in [0,1],
kappa_share in [0,1]; decoded as kappa = kappa_total * kappa_share,
kappa_s = kappa_total * (1 - kappa_share). Enforces kappa + kappa_s <= 1.

M4 math (LBA joint choice+RT): drift rates v_i = v_scale * pi_hybrid_i. Linear Ballistic
Accumulator (Brown & Heathcote 2008). b = A + delta (b > A by construction). t0 = non-decision
time. RT = t_winner + t0. Joint log-likelihood = log_choice_lik + log_rt_lik. No epsilon
parameter (RT noise replaces it). Requires float64.

### For verification planning

The observable truth for INTG-04 is: run `11_run_model_recovery.py --mode cross-model`
for each of M5, M6a, M6b and confirm that each generating model wins AIC (> 50% of
datasets for at least N=5 synthetic datasets). This can be verified with a small N
(N=10 subjects, N=3 datasets) as a smoke test in the plan.

The observable truth for INTG-05 is: MODEL_REFERENCE.md has sections for M3, M4, M5,
M6a, M6b with parameter tables matching the constants in `mle_utils.py`.

---

## What Does NOT Need Doing

- Script 14 comparison tracks: already implemented in Phase 11
- Script 15 `--model all`: already includes all 7 models
- Script 16 `--model all`: already includes all 7 models
- Script 11 parameter recovery: already supports all 7 models via `--model all`
- M4 separate AIC track: already enforced in script 14
- Any new fitting infrastructure: all likelihoods are complete

The verification reports from Phases 8-11 confirm this. The phase 12 work is
exclusively the cross-model AIC recovery check and documentation.
