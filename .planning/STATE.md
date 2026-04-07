# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-03)

**Core value:** Correctly dissociate perseverative responding from learning-rate effects (alpha-) to accurately identify whether post-reversal failures reflect motor perseveration or outcome insensitivity
**Current focus:** v3.0 shipped. Planning next milestone.

## Current Position

Phase: Not started (next milestone not yet defined)
Plan: Not started
Status: v3.0 milestone complete and archived. Ready to plan next milestone.
Last activity: 2026-04-05 — Logit clamp fix, wave pipeline orchestrator, Quarto manuscript setup

## Performance Metrics

**v1 Milestone:**
- Total plans completed: 6
- Average duration: 25 min
- Total execution time: 2.5 hours

**v2 Milestone:**
- Total plans completed: 7
- Average duration: 20 min
- Total execution time: 140 min

**v3 Milestone:**
- Total plans completed: 6 (incl. 1 gap closure)
- Average duration: 20 min
- Total execution time: 117 min

## Accumulated Context

### Model Naming

- M1: Q-learning (alpha+, alpha-, epsilon) — existing
- M2: WM-RL hybrid (alpha+, alpha-, phi, rho, K, epsilon) — existing
- M3: WM-RL + kappa perseveration — v1 shipped, winning model
- M4: RLWM-LBA joint choice+RT (M3 learning + v_scale, b, A, t0) — v3 Phase 11
- M5: WM-RL + phi_RL RL forgetting (M3 + Q-value decay before update) — v3 Phase 8
- M6a: WM-RL + kappa_s stimulus-specific perseveration (replaces global kappa) — v3 Phase 9
- M6b: WM-RL + kappa + kappa_s dual perseveration (stick-breaking constraint) — v3 Phase 10

### Key Decisions

- v3: Build order M5 → M6a → M6b → M4 (complexity-ordered; M5 validates pipeline integration pattern)
- v3: M4 gets separate comparison track in compare_mle_models.py (joint likelihood incommensurable with choice-only AIC)
- v3: Parameter recovery r >= 0.80 is a hard gate within each phase before proceeding to next phase
- v3: phi_RL decay applied BEFORE delta-rule update (not after) — correctness-critical per Senta et al.
- v3: M6 carry uses per-stimulus last_actions array (shape num_stimuli), not global scalar
- M5: phi_rl decay target Q0=1/nA=0.333 (NOT q_init=0.5) — matches WM baseline convention
- M5: phi_rl placed at param index 6, epsilon at index 7 (WMRL_M5_PARAMS order)
- M5: phi_rl=0 algebraic identity reduces exactly to M3 (no conditional branch needed, 0.00e+00 difference verified)
- M5: Model extension pattern established: copy M3 dispatch blocks, add new param as penultimate before epsilon
- M5: Confirmed new winning model: dAIC=435.6 over M3, dBIC=226.9 (very strong evidence per Burnham & Anderson)
- Pipeline: Script 14 --mle-dir default corrected to output/mle; fallback search in output/ root added
- Pipeline: All downstream scripts (14, 15, 16, model_recovery) follow elif wmrl_m5 extension pattern for M6a/M6b/M4
- M6a: kappa_s (stimulus-specific) replaces global kappa; carry changes from scalar to (num_stimuli,) int32 array
- M6a: last_actions.at[stimulus].set() update unconditional on valid (not gated on use_m2_path)
- M6a: kappa_s lower bound is 0.0 (matching M3 kappa); starting default 0.1 (small positive)
- M6a: Per-stimulus tracking structurally verified vs M3 (NLL diff=0.693147 for 2-stimulus sequence)
- M6a: All four fit_all_gpu dispatch points must be explicit elif -- missing any causes silent data corruption
- M6a: model_recovery.py uses per-stimulus last_actions={} dict (not global scalar); resets at each block; kernel only if stimulus was seen before in block
- M6a: compute_diagnostics=False in run_parameter_recovery() avoids ZeroDivisionError at parameter bounds (logit transform issue when optimizer reaches exact upper bound)
- M6a: n_starts and n_jobs exposed in run_parameter_recovery() and script 11 argparse
- M6b: Stick-breaking decode in objective functions only (not in transform): kappa = kappa_total * kappa_share; kappa_s = kappa_total * (1 - kappa_share)
- M6b: Dual carry in lax.scan: 6-element tuple (Q, WM, WM_0, log_lik, last_action_scalar, last_actions_array)
- M6b: Effective-weight gating (eff_kappa = jnp.where(has_global, kappa, 0.0)) — no use_m2_path branch needed
- M6b: kappa_share=1.0 verified = M3 (diff 0.0e+00); kappa_share=0.0 verified = M6a (diff 0.0e+00)
- M6b: fit_all_participants has SECOND param_cols dispatch (separate from main()) — both must include wmrl_m6b
- M6b: wmrl_m6b_block_likelihood takes decoded kappa/kappa_s (not kappa_total/kappa_share)
- M6b recovery: generate_synthetic_participant uses dual-kernel tracking; BOTH last_action (global) and last_actions dict (per-stimulus) maintained and reset at block boundaries
- M6b recovery: stick-breaking decode in generation (kappa = kappa_total * kappa_share) mirrors objective decode
- M6b downstream: script 15 load_data() now returns 8-tuple; script 14 auto-detects wmrl_m6b_individual_fits.csv via patterns dict
- M4 (11-01): LBA density module isolated in lba_likelihood.py with jax_enable_x64=True; jax_likelihoods.py NOT modified (float32 contamination avoided)
- M4: log-density from lba_joint_log_lik CAN be positive (defective PDF integrates to <1 over time but can exceed 1 at a point); test asserts finiteness + ordering not negativity
- M4: validate_t0_constraint is standalone diagnostic only; structural protection comes from WMRL_M4_BOUNDS t0 upper limit in mle_utils
- M4 (11-02): wmrl_m4_block_likelihood returns POSITIVE NLL (-log_lik_total); objective functions call it with `return nll` NOT `return -log_lik` (differs from choice-only models)
- M4: b > A reparameterization decode (b = A + delta) in ALL three objective functions; NOT in transform functions
- M4: prepare_participant_data valid_rt must be padded to max_trials before multiplying with padding mask (shape compatibility)
- M4: fit_all_gpu Stage 3 requires separate _run_one branch with 7 data args (stimuli, actions, rewards, masks, set_sizes, rts) -- no silent fallthrough to M5 branch
- M4: Float64 enabled lazily via `jax.config.update("jax_enable_x64", True)` in prepare_participant_data and main() only when model==wmrl_m4
- M4 (11-03): LBA race simulation uses numpy rng (not JAX) for start points k~Uniform(0,A); t=(b-k)/max(v,1e-6); winner=argmin(t); RT=t[winner]+t0 in milliseconds
- M4 (11-03): Convex combination perseveration in synthetic generation: (1-kappa)*hybrid + kappa*Ck -- matches M4 likelihood, prevents kappa recovery bias (M3/M5 still use additive renorm)
- M4 (11-03): Script 14 pops M4 from fits_dict BEFORE AIC comparison; reports M4 in separate section; --m4 argparse flag
- M4 (11-03): Script 15 load_data() returns 9-tuple (adds wmrl_m4 at end); Script 16 renames v_scale/A/delta/t0 to v_scale_mean/A_mean/delta_mean/t0_mean
- 12-01: CHOICE_ONLY_MODELS = ['qlearning', 'wmrl', 'wmrl_m3', 'wmrl_m5', 'wmrl_m6a', 'wmrl_m6b'] — M4 excluded from cross-model AIC comparison
- 12-01: run_model_recovery_check --output bug fixed (was passing file path, now correctly passes directory to fit_mle.py)
- 12-01: Cross-model confusion matrix uses plurality criterion (generator must win >50% of datasets)
- 12-01: Smoke test confirmed M5 wins AIC against all 6 choice-only models (N=5, n_datasets=1)
- 12-02: MODEL_REFERENCE.md is single authoritative source for all 7 model mathematics (M1-M6b, M4)
- 12-02: Section 4.2 renamed "Simplifications and Extensions" — perseveration no longer listed as unimplemented
- 12-03 (gap closure): M6a/M6b elif branches were siblings of `if last_action is not None:` — unreachable after trial 1; fixed to independent if/elif chain
- 12-03 (gap closure): M3/M5 changed from additive renormalization to convex combination (1-kappa)*P_noisy + kappa*Ck, matching jax_likelihoods.py
- 12-03 (gap closure): Epsilon applied BEFORE perseveration for ALL non-M4 models; non-M4 action selection no longer re-applies epsilon
- 12-03 (gap closure): Q-learning epsilon moved inline to Q-learning branch (was in now-removed action selection double-application block)
- 12-03 (gap closure): M6a kappa_s recovery verified non-degenerate (range=0.3275, r=1.000, N=2); M6b kappa_total recovery verified (range=0.1546, r=1.000, N=2)

### Pending Todos

- **Re-fit all 7 models on cluster** (3 bugs fixed: argmin NaN, stimulus sampling, reward mapping). See `.planning/todos/pending/2026-04-07-refit-all-models-on-cluster.md`
- Run parameter recovery for all models after re-fit (50 subj / 3 datasets / 20 starts)
- Run full cross-model recovery: `python scripts/11_run_model_recovery.py --mode cross-model --model all --n-subjects 50 --n-datasets 3 --n-starts 20 --n-jobs 8`

### Blockers/Concerns

- Phase 8 (M5): Stimulus sampling bug FIXED (quick-002). Re-run recovery: `python scripts/fitting/model_recovery.py --model wmrl_m5 --n-subjects 50 --n-datasets 3 --n-starts 20 --use-gpu` or `sbatch --export=MODEL=wmrl_m5 cluster/11_recovery_gpu.slurm`
- Phase 9 (M6a): Full parameter recovery not yet run — quick test (N=2) verified functional. Run on cluster: `sbatch --export=MODEL=wmrl_m6a cluster/11_recovery_gpu.slurm`
- Phase 9 (M6b): Full parameter recovery not yet run — quick test (N=2) verified functional. Run on cluster: `sbatch --export=MODEL=wmrl_m6b cluster/11_recovery_gpu.slurm`
- Phase 11 (M4): Full parameter recovery not yet run — quick test (N=2) verified functional. M4 needs ~48h: `sbatch --time=48:00:00 --export=MODEL=wmrl_m4,NSUBJ=30 cluster/11_recovery_gpu.slurm`

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 001 | Setup Quarto scientific manuscript for RLWM trauma analysis | 2026-04-05 | 18637da | [001-setup-quarto-manuscript](./quick/001-setup-quarto-manuscript/) |
| 002 | Pipeline fixes, convergence assessment, recovery config, MODEL_REGISTRY | 2026-04-07 | 3095b92 | [002-pipeline-fixes-convergence-recovery-config](./quick/002-pipeline-fixes-convergence-recovery-config/) |
| 003 | Softcode manuscript: winning model, group names, n_starts from data files | 2026-04-07 | d7ea897 | [003-quarto-softcoded-winning-model](./quick/003-quarto-softcoded-winning-model/) |

### Key Decisions Added (quick-002)

| Decision | Rationale |
|----------|-----------|
| MODEL_REGISTRY in config.py is single source of truth for pipeline scripts | Prevents model list drift; mle_utils.py PARAMS/BOUNDS untouched (JAX inner loop) |
| Stimulus sampled from range(set_size) per block in synthetic generation | Root cause of M5 recovery failure (r=0.03-0.57); Q/WM tables now (6,3) matching likelihood |
| Recovery defaults: n_starts=20, n_datasets=3 (was 50, 10) | Adequate for r-metric; saves 60-70% runtime |
| Wave 3 analysis uses afterok by default (not afterany) | Ensures all 7 models present before comparison; --analysis-after-any flag for partial results |

## Session Continuity

Last session: 2026-04-07
Stopped at: Completed quick task 003 — softcoded manuscript with data-driven winning model, group names, model table.
Resume file: None
