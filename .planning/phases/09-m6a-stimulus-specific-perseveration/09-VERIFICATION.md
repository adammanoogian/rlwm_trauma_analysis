---
phase: 09-m6a-stimulus-specific-perseveration
verified: 2026-04-02T22:12:20Z
status: passed
score: 6/6 must-haves verified
gaps: []
---

# Phase 9: M6a Stimulus-Specific Perseveration Verification Report

**Phase Goal:** Users can fit a perseveration model that tracks last-action per stimulus independently, rather than using a global scalar
**Verified:** 2026-04-02T22:12:20Z
**Status:** passed
**Re-verification:** No - initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Running --model wmrl_m6a completes and writes fit results | VERIFIED | output/mle/wmrl_m6a_individual_fits.csv exists with kappa_s column; 3 converged participants |
| 2 | M6a carry array correctly resets last_actions for all stimuli at block boundaries | VERIFIED | wmrl_m6a_block_likelihood reinitializes last_actions_init on every call; model_recovery.py resets last_actions = {} inside block loop |
| 3 | First presentation of each stimulus uses uniform (1/nA) fallback with no kernel | VERIFIED | Line 2115: use_m2_path = logical_or(kappa_s == 0.0, last_action_s < 0); sentinel -1 triggers M2 path |
| 4 | M6a appears in AIC/BIC comparison table alongside M1-M3 and M5 | VERIFIED | output/model_comparison/comparison_results.csv contains M6a row; auto-detected from output file |
| 5 | M6-01/M6-02: JAX likelihood with per-stimulus int32 array carry | VERIFIED | last_actions_init = jnp.full((num_stimuli,), -1, dtype=jnp.int32); carry is 5-tuple; distinct from M3 scalar |
| 6 | M6-05/M6-06: kappa_s bounds [0,1], transforms, param names, CLI flag | VERIFIED | WMRL_M6A_BOUNDS has kappa_s=(0.0,1.0); both JAX transforms exist; argparse choices include wmrl_m6a |

**Score:** 6/6 truths verified

---

## Required Artifacts

| Artifact | Status | Details |
|----------|--------|---------|
| scripts/fitting/jax_likelihoods.py: wmrl_m6a_block_likelihood | VERIFIED | 3050 lines total; function at line 1998; 183-line implementation with lax.scan; three inline tests |
| scripts/fitting/jax_likelihoods.py: wmrl_m6a_multiblock_likelihood | VERIFIED | Line 2185; fast path (fori_loop) and fallback (Python loop); block reset via per-call initialization |
| scripts/fitting/jax_likelihoods.py: wmrl_m6a_multiblock_likelihood_stacked | VERIFIED | Line 2287; GPU-compatible stacked variant |
| scripts/fitting/mle_utils.py: WMRL_M6A_BOUNDS/PARAMS/transforms | VERIFIED | 1229 lines; bounds at line 71; params at line 92; both JAX transforms at lines 280-315; 15+ dispatch elifs |
| scripts/fitting/fit_mle.py: full pipeline with all dispatch points | VERIFIED | 2579 lines; all four fit_all_gpu dispatch blocks handle wmrl_m6a; Hessian dispatch at line 1172 |
| scripts/fitting/model_recovery.py: per-stimulus dict generation | VERIFIED | 1301 lines; dict reset inside block loop at line 257; per-stimulus kernel at line 298-301 |
| scripts/14_compare_models.py: M6a auto-detection and --m6a arg | VERIFIED | 775 lines; find_mle_files at line 547; argparse at line 580; load block at line 631 |
| scripts/15_analyze_mle_by_trauma.py: kappa_s analysis | VERIFIED | WMRL_M6A_PARAMS defined; defensive load; MODEL_CONFIG entry; empty-DataFrame guards |
| scripts/16_regress_parameters_on_scales.py: kappa_s regression | VERIFIED | kappa_s->kappa_s_mean rename; format_label entry; wmrl_m6a elif branch |
| output/mle/wmrl_m6a_individual_fits.csv | VERIFIED | Exists; kappa_s column confirmed; 3 participants, all converged=True |
| output/model_comparison/comparison_results.csv | VERIFIED | M6a as first row (best AIC/BIC); alongside M5, M3, M2 |

---

## Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| fit_mle.py --model wmrl_m6a | wmrl_m6a_multiblock_likelihood_stacked | _make_jax_objective_wmrl_m6a (line 426) | WIRED |
| wmrl_m6a_block_likelihood | per-stimulus last_actions array | lax.scan carry with jnp.full((num_stimuli,),-1,int32) | WIRED |
| First-presentation | uniform fallback | use_m2_path = last_action_s < 0 gate (line 2115) | WIRED |
| Block boundary reset JAX | fresh last_actions per block | wmrl_m6a_block_likelihood reinitializes carry on each call | WIRED |
| Block boundary reset Python | fresh dict per block | last_actions = {} at line 257, inside block loop starting line 235 | WIRED |
| script 14 | M6a in AIC/BIC table | auto-detect wmrl_m6a_individual_fits.csv or --m6a arg | WIRED |

---

## Requirements Coverage

| Requirement | Status | Notes |
|-------------|--------|-------|
| M6-01: JAX likelihood with stimulus-specific kappa_s | SATISFIED | wmrl_m6a_block_likelihood fully implemented at line 1998 |
| M6-02: Per-stimulus last_action tracking via int32 array in lax.scan carry | SATISFIED | jnp.full((num_stimuli,), -1, dtype=jnp.int32); unconditional .at[stimulus].set() update |
| M6-03: Uniform fallback (1/nA) for first presentation | SATISFIED | use_m2_path = last_action_s < 0 enforces structurally |
| M6-04: last_action per stimulus resets at block boundaries | SATISFIED | JAX: per-call reinitialization; Python recovery: dict reset in block loop |
| M6-05: MLE bounds, logit transform, param names | SATISFIED | WMRL_M6A_BOUNDS kappa_s=(0.0,1.0); both JAX transforms; WMRL_M6A_PARAMS |
| M6-06: CLI --model wmrl_m6a integration | SATISFIED | argparse choices include wmrl_m6a; all four fit_all_gpu dispatch points handled |

---

## Anti-Patterns Found

None. No TODO/FIXME/placeholder patterns found in M6a code paths. No empty returns or stub implementations.

---

## Informational Notes

**Parameter recovery (N=2, quick smoke test):** A quick N=2, 1-dataset, 3-starts recovery run was completed and writes to output/recovery/wmrl_m6a/. The r=1.0 values in recovery_metrics.csv are artifacts of N=2 (any two points produce perfect correlation). The r >= 0.80 gate at N=50 is Phase 10 criterion (M6-11), not Phase 9.

**kappa_s lower bound = 0.0:** The forward transform jax_unbounded_to_bounded(x, lower=0.0, upper=1.0) uses sigmoid scaling and is always valid. The inverse at exact boundary 0.0 would produce -inf but is avoided in recovery via compute_diagnostics=False and in main fitting via LHS sampling.

---

## Gaps Summary

No gaps. All six observable truths pass all three verification levels (exists, substantive, wired). Phase goal achieved.

---

_Verified: 2026-04-02T22:12:20Z_
_Verifier: Claude (gsd-verifier)_
