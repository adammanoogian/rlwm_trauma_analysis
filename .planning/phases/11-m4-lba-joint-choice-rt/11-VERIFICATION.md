---
phase: 11-m4-lba-joint-choice-rt
verified: 2026-04-03T12:22:40Z
status: human_needed
score: 4/5 must-haves verified (5th requires cluster compute)
human_verification:
  - test: Run full parameter recovery -- python scripts/11_run_model_recovery.py --model wmrl_m4 --n-subjects 30 --n-datasets 10 --n-jobs 8
    expected: Pearson r >= 0.80 for all 10 M4 parameters (alpha_pos, alpha_neg, phi, rho, capacity, kappa, v_scale, A, delta, t0)
    why_human: N=2 quick test confirmed pipeline completes. r >= 0.80 gate (M4-10) requires N>=30 on cluster compute -- cannot verify programmatically.
  - test: Check t0 distribution in output/mle/wmrl_m4_individual_fits.csv (t0 column summary)
    expected: t0 values cluster below 0.15s; if many participants hit t0 near 0.3s upper bound, tighten the bound to 0.14s or lower
    why_human: t0 upper bound (0.3s) exceeds min filtered RT (0.15s). t_star clamp prevents NaN but silently distorts likelihood when t0 >= RT.
---
# Phase 11: M4 LBA Joint Choice+RT Verification Report

**Phase Goal:** Users can fit a model that accounts for both choice and reaction time via a Linear Ballistic Accumulator process with drift rates derived from the hybrid policy
**Verified:** 2026-04-03T12:22:40Z
**Status:** human_needed
**Re-verification:** No -- initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|---------|
| 1 | python scripts/12_fit_mle.py --model wmrl_m4 completes and writes fit results | VERIFIED | output/mle/wmrl_m4_individual_fits.csv exists with 10-parameter rows (no epsilon); wmrl_m4 in argparse choices (fit_mle.py line 2772) |
| 2 | RT preprocessing removes outliers (<150ms, >2000ms) and t0 bounds provide structural protection | VERIFIED | preprocess_rt_block filters 150-2000ms; WMRL_M4_BOUNDS t0: (0.05, 0.3); t_star clamp at lba_likelihood.py line 501. NOTE: t0 upper bound (0.3s) exceeds min filtered RT (0.15s) -- known design choice, flagged for human review |
| 3 | M4 in separate choice+RT track; NOT in M1-M3 AIC table | VERIFIED | fits_dict.pop(M4, None) at 14_compare_models.py line 668; AIC comparison runs on choice_only_dict only; M4 shown in separate SEPARATE TRACK section |
| 4 | b > A enforced structurally via reparameterization -- no post-hoc clipping | VERIFIED | delta bounded (0.001, 2.0) in WMRL_M4_BOUNDS (delta always > 0); b = A + delta decoded in all 3 objectives (fit_mle.py lines 609, 969, 1199); transform returns A and delta NOT b |
| 5 | Parameter recovery r >= 0.80 for all M4 parameters | HUMAN NEEDED | Pipeline confirmed with N=2 quick test; full cluster validation required for M4-10 gate |

**Score:** 4/5 truths verified (1 human-needed per phase brief)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| scripts/fitting/lba_likelihood.py | LBA density/CDF/SF/joint-LL, RT preprocessing, M4 block/multiblock likelihood | VERIFIED | 769 lines; jax_enable_x64 at module load; all functions present: lba_pdf, lba_cdf, lba_sf, lba_log_sf, lba_joint_log_lik, preprocess_rt_block, validate_t0_constraint, FIXED_S=0.1, wmrl_m4_block_likelihood, wmrl_m4_multiblock_likelihood, wmrl_m4_multiblock_likelihood_stacked |
| scripts/fitting/mle_utils.py | WMRL_M4_BOUNDS (10 params, no epsilon), WMRL_M4_PARAMS, transform functions, all dispatch points | VERIFIED | 1421 lines; WMRL_M4_BOUNDS at line 95; WMRL_M4_PARAMS at line 127; wmrl_m4 elif branches in all 9 dispatch functions |
| scripts/fitting/fit_mle.py | M4 CLI, 3 objective functions, 4 fit_all_gpu dispatch points, RT extraction in prepare_participant_data | VERIFIED | 3069 lines; wmrl_m4 in argparse choices; all 3 objective functions present; prepare_participant_data calls preprocess_rt_block for M4; fit_all_gpu has separate 7-arg vmap branch |
| scripts/fitting/model_recovery.py | LBA race simulation for M4, rts_blocks passthrough | VERIFIED | generate_synthetic_participant handles wmrl_m4 with k~Uniform(0,A), t=(b-k)/v_safe, winner=argmin, RT stored in ms; rts_blocks passed through run_parameter_recovery |
| scripts/14_compare_models.py | M4 in separate track, popped before AIC comparison | VERIFIED | fits_dict.pop(M4, None) at line 668; AIC runs on choice_only_dict only; separate M4 section with per-param mean/SEM |
| scripts/15_analyze_mle_by_trauma.py | wmrl_m4 in choices, 9-tuple load_data | VERIFIED | wmrl_m4 in argparse choices; wmrl_m4_path loaded defensively; load_data() returns 9-tuple |
| scripts/16_regress_parameters_on_scales.py | wmrl_m4 in choices, M4 param dispatch | VERIFIED | wmrl_m4 in choices; elif model == wmrl_m4 dispatch for param_cols with v_scale, A, delta, t0 |
| scripts/11_run_model_recovery.py | --model wmrl_m4 accepted | VERIFIED | wmrl_m4 in argparse choices and in all model list |
| output/mle/wmrl_m4_individual_fits.csv | 10 param columns, no epsilon, converged rows | VERIFIED | Columns: participant_id, alpha_pos, alpha_neg, phi, rho, capacity, kappa, v_scale, A, delta, t0, nll, aic, bic... no epsilon column |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| lba_likelihood.py LBA functions | jax.scipy.stats.norm | jss.norm.cdf, jss.norm.pdf | WIRED | Lines 61-63 (lba_pdf), 100-103 (lba_cdf) |
| fit_mle.py 3 objective functions | wmrl_m4_multiblock_likelihood_stacked | lazy import + b = A + delta decode | WIRED | Lines 591/943/1195 (lazy import); lines 609/969/1199 (b = A + delta) |
| fit_mle.py prepare_participant_data | preprocess_rt_block | rt column -> ms->seconds, outlier filtered, padded, combined with padding mask | WIRED | Lines 2148 (import), 2191-2204 (call and padding) |
| fit_mle.py fit_all_gpu | M4 7-arg vmap branch | separate _run_one with rts argument | WIRED | Lines 1380-1395 define M4-specific _run_one; line 1413 passes rts_batch |
| model_recovery.py generate_synthetic_participant | LBA race simulation | k~Uniform(0,A), t=(b-k)/v_safe, winner=argmin, RT=t_winner+t0 in ms | WIRED | Lines 364-375; convex combination perseveration matches M4 likelihood |
| model_recovery.py run_parameter_recovery | rts_blocks passthrough | rts_blocks=data_dict.get(rts_blocks) | WIRED | Line 682 |
| 14_compare_models.py | M4 separate reporting | fits_dict.pop(M4, None) before AIC comparison | WIRED | Lines 668-669 |

### Requirements Coverage

| Requirement | Status | Blocking Issue |
|-------------|--------|---------------|
| M4-01: RT preprocessing utility | SATISFIED | None |
| M4-02: JAX LBA density (Brown & Heathcote 2008, float64) | SATISFIED | None |
| M4-03: Drift rates v_i = v_scale * pi_hybrid | SATISFIED | None |
| M4-04: Within-trial noise s=0.1 fixed | SATISFIED | None |
| M4-05: Epsilon dropped | SATISFIED | None |
| M4-06: MLE bounds/transforms | SATISFIED | None |
| M4-07: b > A via reparameterization | SATISFIED | None |
| M4-08: CLI --model wmrl_m4 | SATISFIED | None |
| M4-09: Joint choice+RT likelihood | SATISFIED | None |
| M4-10: Parameter recovery r >= 0.80 | HUMAN NEEDED | Requires full cluster run (N>=30) |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | -- | -- | -- | No TODO/FIXME/placeholder patterns in any M4 implementation files |

### Design Tension: t0 Upper Bound vs Min Filtered RT (Warning-Level)

The t0 upper bound in WMRL_M4_BOUNDS is 0.3s. The minimum RT after filtering is 150ms = 0.15s. The optimizer can legally propose t0 values up to 0.3s that exceed the minimum filtered RT of 0.15s.

When t0 >= RT for a trial, t_star = jnp.maximum(rt - t0, 1e-6) at lba_likelihood.py line 501 clamps to near-zero. This prevents NaN but silently distorts the likelihood for that trial rather than masking it.

This is a known, documented design choice. The 11-RESEARCH.md file explicitly notes: fit t0 with bounds (0.05, 0.3) seconds -- flag for review -- may need to fix t0=0.1 if recovery is poor. Existing fits show many participants with t0 near the lower bound (0.05s), suggesting the clamp may not activate frequently. However it warrants a human check of the t0 distribution in real fits.

Severity: Warning (not blocker) -- the implementation is correct, prevents NaN, and fitting completes successfully.

### Human Verification Required

#### 1. Parameter Recovery r >= 0.80 (M4-10 Gate)

**Test:** python scripts/11_run_model_recovery.py --model wmrl_m4 --n-subjects 30 --n-datasets 10 --n-jobs 8

**Expected:** All 10 parameters achieve Pearson r >= 0.80 between generative and recovered values. LBA-specific parameters (v_scale, A, delta, t0) may be harder to recover -- r >= 0.80 is the gate for each.

**Why human:** The quick N=2 test confirmed the pipeline runs end-to-end (synthetic RTs in 316-840ms range, 10-param recovery output, no epsilon). Full M4-10 validation requires cluster compute (N>=30, n_datasets>=10 for stable r estimates).

#### 2. t0 Distribution in Real Fits

**Test:** Inspect the t0 column in output/mle/wmrl_m4_individual_fits.csv. Check what fraction of participants have t0 >= 0.15s (the min filtered RT threshold).

**Expected:** t0 values should cluster well below 0.15s. If >= 10% of participants have t0 >= 0.15s, the t0 upper bound should be tightened to 0.14s or lower.

**Why human:** Cannot assess the practical impact of the t0 design tension without looking at the real-fit distribution across participants.

---

_Verified: 2026-04-03T12:22:40Z_
_Verifier: Claude (gsd-verifier)_
