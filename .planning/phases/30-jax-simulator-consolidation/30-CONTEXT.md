---
phase: 30-jax-simulator-consolidation
milestone: v5.1 (proposed) — may run in v5.0 if time permits before Phase 27 closure
created: 2026-04-23
status: proposed
depends_on:
  - 29-08 (vertical-by-model layout provides per-model file home)
  - v5 shim cleanup (commits 5841069 + d20bca6; canonical rlwm.fitting.mle / rlwm.fitting.bayesian homes exist)
blocks:
  - none (architectural improvement; no downstream phase requires this to proceed)
---

# Phase 30 — JAX Simulator Consolidation

## Goal

Add JAX-based simulators as siblings to the existing likelihoods in
`src/rlwm/fitting/models/<m>.py`, refactor `scripts/utils/ppc.py` to
delegate to them, and delete the now-obsolete `src/rlwm/models/` NumPy
Agent classes. End state: every model's math (likelihood + hierarchical
Bayesian + simulation) lives in exactly ONE file per model, matching the
Phase-29-08 vertical-by-model architecture.

## Motivation

Simulator math currently lives in three implementations:

1. **`scripts/utils/ppc.py`** — pure NumPy simulators
   (`_simulate_qlearning`, `_simulate_wmrl_family`). Used by prior- and
   posterior-predictive checks; canonical PPC simulator per Phase 29-03.
2. **`src/rlwm/models/{q_learning,wm_rl_hybrid}.py`** — Gym-stateful
   NumPy Agent classes (`QLearningAgent.step()`). Used by zero
   production scripts; 644 lines of unmaintained cargo (see v5 shim
   cleanup audit, 2026-04-23).
3. **`src/rlwm/fitting/models/<m>.py`** — JAX likelihood functions.
   Does NOT sample actions; takes observed actions and computes log
   P(actions | stimuli, params).

The same per-trial update rule (`Q[s,a] += alpha * (r - Q[s,a])` for
Q-learning; the equivalent wmrl/m3/m5/m6a/m6b updates) is duplicated
across #1 and #2. #3 already exports the primitive (`q_learning_step`
in `qlearning.__all__`) but no simulator builds on it.

Adding JAX simulators as siblings to the likelihoods collapses this
landscape to ONE canonical math home per model. Downstream consequences:

- **PPC speedup (10–100×):** JIT + vmap over posterior draws replaces
  Python for-loops over participants × blocks × trials. Concrete
  example: posterior PPC at N=138 × 2000 draws × 21 blocks × 100 trials
  = 579M simulated trials. NumPy takes ~minutes; JAX vmap completes in
  seconds once the JIT cache is warm.
- **`src/rlwm/models/` becomes deletable** — the Agent classes were
  the pre-v4.0 Gym-interactive exploration API; superseded first by
  `scripts/utils/ppc.py` and definitively by a JAX-simulator home in
  `rlwm.fitting.models/`.
- **Consistency-by-construction:** the simulator and likelihood share
  the same per-trial primitive (`<m>_step`), so any numerical drift
  between "what the model generates" and "what the model scores" is
  structurally impossible.

## Scope

### In-scope

- Add `<m>_block_simulate(stimuli, rng_key, params, correct_map) ->
  (actions, rewards)` to each of the 6 choice-only model files under
  `src/rlwm/fitting/models/<m>.py` (qlearning, wmrl, wmrl_m3, wmrl_m5,
  wmrl_m6a, wmrl_m6b). Pure JAX, JIT-compilable, vmap-compatible.
- Per-model `<m>_multiblock_simulate` + `<m>_multiparticipant_simulate`
  wrappers mirroring the likelihood API (multi-block loop, stacked
  across participants).
- Refactor `scripts/utils/ppc.py` to delegate to JAX simulators:
  replace the private NumPy `_simulate_qlearning` /
  `_simulate_wmrl_family` functions with JAX-dispatching wrappers.
  Public API (`simulate_from_samples`, `run_prior_ppc`,
  `run_posterior_ppc`) preserved.
- Consistency test: pytest gate asserting that sampling from
  `<m>_block_simulate` followed by scoring with
  `<m>_block_likelihood` at the same params produces a log-prob
  consistent with the sampler's per-trial categorical log-prob (within
  floating-point tolerance). Catches any divergence between the two
  paths.
- Delete `src/rlwm/models/` (3 files, 644 lines). Update/delete 5
  active importers:
    - `tests/test_wmrl_exploration.py` — delete (debug script,
      imports from `scripts/legacy/simulations/` — already cargo)
    - `tests/test_rlwm_package.py` — drop `rlwm.models.*` canonical-
      path tests (keep `rlwm.envs.*` tests)
    - `validation/test_model_consistency.py` — delete (tests Agent
      class determinism; no consumer)
    - `validation/test_parameter_recovery.py` — delete (pre-v4.0
      recovery test; superseded by
      `scripts/03_model_prefitting/03_run_model_recovery.py` +
      `05_run_bayesian_recovery.py`)
    - `validation/test_unified_simulator.py` — delete (imports from
      `scripts/legacy/simulations/` — both dependency and target are
      legacy)
- Benchmark: record pre- and post-refactor wall-clock for a
  representative PPC run (prior-predictive at N=138, 500 draws, M6b)
  and document the speedup in 30-VERIFICATION.md.

### M4 LBA status

M4 (RLWM-LBA) is out-of-scope for this phase. The LBA choice+RT
generative model is substantially more complex than the choice-only
softmax + Q-update path and would double the scope. M4 PPC is deferred
to a future phase or left on CPU (current behavior).

### Out-of-scope (deferred)

- Rewriting the Gym environment itself (`rlwm.envs.*`) — the Gym env
  is orthogonal to agent classes and remains as-is.
- `scripts/legacy/simulations/unified_simulator.py` — already in
  legacy/, untouched by this phase.
- Replacing `rlwm.fitting.numpyro_helpers.py` (hBayesDM non-centered
  helpers) with JAX-native equivalents — unrelated to simulators.

## Proposed Plans

**Plan 30-01 — JAX Simulators (per model) [autonomous: true]**
Add `<m>_block_simulate` + `<m>_multiblock_simulate` +
`<m>_multiparticipant_simulate` to the 6 choice-only model files.
Reuses the existing `<m>_step` primitive that each file already
exports. Update `__all__` lists. Signature mirrors the likelihood
(same stimuli/params argument structure + `rng_key` input and
`(actions, rewards)` output instead of log-lik scalar). Estimated
~80 lines per model × 6 = ~480 lines added.

**Plan 30-02 — PPC Delegation Refactor [autonomous: true]**
Refactor `scripts/utils/ppc.py._simulate_{qlearning,wmrl_family}` to
call the new JAX simulators. Preserve the public API
(`simulate_from_samples` dispatch, `run_prior_ppc`,
`run_posterior_ppc`). JIT-compile the per-model simulator; vmap over
participants and over posterior draws. Delete the ~400 lines of
duplicate NumPy simulator logic.

**Plan 30-03 — Simulator↔Likelihood Consistency Test [autonomous: true]**
New pytest file `scripts/fitting/tests/test_simulator_likelihood_consistency.py`.
For each of the 6 models: sample (a, r) from `<m>_block_simulate` at
params θ, RNG seed k; then compute
`<m>_block_likelihood(stimuli, actions_sampled, rewards_sampled, θ)`
and verify the resulting log-lik matches the per-trial categorical
log-probs accumulated during sampling (modulo floating-point
tolerance). This proves the simulator uses the same generative
structure the likelihood assumes. One test per model × 2 seeds = 12
test cases.

**Plan 30-04 — Delete `src/rlwm/models/` [autonomous: false, user-approval gate]**
Remove `src/rlwm/models/` (3 files). Update/delete the 5 active
importers per the In-scope list above. Update
`tests/test_rlwm_package.py` canonical-path tests to drop `rlwm.models`
assertions. Regression-check: `pytest tests/ validation/
scripts/fitting/tests/` runs green. The user-approval gate exists
because this is a breaking API removal — any external code (not in
this repo) that imports `rlwm.models.QLearningAgent` will break.

**Plan 30-05 — PPC Benchmark + Documentation [autonomous: true]**
Run a representative PPC benchmark (prior-predictive N=138, 500
draws, M6b subscale) pre- and post-Plan-30-02 — actually just
post, since pre is git-history. Record wall-clock in
`30-VERIFICATION.md`. Update `docs/03_methods_reference/MODEL_REFERENCE.md`
(or `docs/04_methods/`) to reference the JAX simulator as the
canonical PPC entry point. Note the architectural story: every
model's math (likelihood + hierarchical + simulate) lives in ONE file.

## Success Criteria

1. **SIM-01 — Per-model JAX simulators exist.** Each of the 6
   choice-only model files under `src/rlwm/fitting/models/` exports
   `<m>_block_simulate`, `<m>_multiblock_simulate`,
   `<m>_multiparticipant_simulate` in `__all__`. Smoke test: `from
   rlwm.fitting.models.qlearning import q_learning_block_simulate` and
   calling it with known params reproduces documented reference output
   (fixed RNG seed).
2. **SIM-02 — PPC delegates to JAX.** `scripts/utils/ppc.py` no
   longer contains `_simulate_qlearning` / `_simulate_wmrl_family` as
   standalone implementations; instead it calls
   `rlwm.fitting.models.<m>.<m>_multiparticipant_simulate` after a
   thin shape-adapting wrapper. Grep invariant: `grep -n "def
   _simulate_" scripts/utils/ppc.py` returns zero matches.
3. **SIM-03 — Consistency test passes.** `pytest
   scripts/fitting/tests/test_simulator_likelihood_consistency.py`
   exits 0 with 12/12 PASS. Catches any drift between generative and
   inferential math.
4. **SIM-04 — `rlwm.models/` deleted.** `src/rlwm/models/` does not
   exist; `grep -rn "from rlwm.models\|import rlwm.models" scripts/
   tests/ validation/ src/` returns zero matches (excluding
   `scripts/legacy/`). `tests/test_rlwm_package.py` no longer
   references `rlwm.models`.
5. **SIM-05 — PPC speedup documented.** `30-VERIFICATION.md` contains
   a wall-clock measurement for a representative PPC run (N=138, 500
   draws, M6b) showing the post-refactor runtime. Documented in
   `docs/` alongside the architectural story.
6. **v4.0 closure guard still green.** `python
   validation/check_v4_closure.py --milestone v4.0` exits 0 on Phase
   30 HEAD (architectural refactor, no v4.0 invariants changed).
7. **Full pytest suite green.** `pytest scripts/fitting/tests/ tests/
   validation/` exits 0 with zero new failures vs pre-Phase-30
   baseline.

## Sequencing

Phase 30 is architectural tech debt. It is NOT a prerequisite for
Phase 24 cold-start, Phase 25 reproducibility regression, Phase 26
manuscript finalization, or Phase 27 closure. Three reasonable
sequencings:

- **(A) Execute in v5.0 before Phase 27 closure.** Prereq: Phase 29
  complete (it is). Fits the "final clean form" narrative user
  requested during v5.0 shim cleanup. Adds ~600 lines of code / ~400
  lines deleted; ~1 day of work.
- **(B) Defer to v5.1.** Lets v5.0 close with its original scope
  intact (Empirical Artifacts & Manuscript Finalization); Phase 30
  becomes the opening phase of v5.1 (Architecture & Performance).
- **(C) Hold as tech-debt.** Keep `src/rlwm/models/` as cargo; accept
  that simulator math lives in two places; defer indefinitely.

**Recommended: (B).** v5.0 goal is empirical artifacts for the
manuscript; Phase 30 is pure refactor with no empirical payoff.
Cleaner milestone boundaries. Phase 30 runs between Phase 27 closure
and the next empirical phase in v5.1.

User decides at planning time.

## Risks

- **JAX `jax.random.categorical` + `jax.lax.scan` over sampling:** the
  standard JAX pattern for Markov-chain-like samplers requires
  carrying the RNG key through the scan. This is well-trodden but
  needs care — wrong `jax.random.split` threading can produce
  identical actions across trials.
- **Numerical equivalence with `scripts/utils/ppc.py` NumPy
  baseline:** the JAX simulator will not produce byte-identical
  samples vs. the NumPy simulator at the same seed (different RNG
  implementations). Consistency test (Plan 30-03) checks the
  *distribution*, not individual samples. Plan 30-02's delegation
  switch will measurably change PPC output byte-for-byte; downstream
  consumers (prior-predictive gate reports, posterior-predictive
  figures) need to be re-baselined.
- **`rlwm.envs.*` coupling:** the Agent classes in `rlwm.models/`
  take a Gym env as input. Deleting them removes the Gym-interactive
  workflow. If anyone ever needs to visualize agent behavior in the
  Gym env, they must resurrect from git history OR implement a new
  JAX-simulator-backed Gym agent (small, not in scope here).

## Plans

- [ ] 30-01-PLAN.md (Wave 1) — Add JAX simulators to 6 choice-only
  model files; export in `__all__`; smoke tests for each.
- [ ] 30-02-PLAN.md (Wave 2, after 30-01) — Refactor
  `scripts/utils/ppc.py` to delegate to JAX simulators; preserve
  public API; delete ~400 lines of NumPy duplicates.
- [ ] 30-03-PLAN.md (Wave 2, parallel with 30-02) —
  `test_simulator_likelihood_consistency.py`: 12 test cases asserting
  generative/inferential math alignment.
- [ ] 30-04-PLAN.md (Wave 3, after 30-01..30-03, **user-approval
  gate**) — Delete `src/rlwm/models/` + update/delete 5 callers.
- [ ] 30-05-PLAN.md (Wave 3, parallel with 30-04) — Benchmark +
  docs update; end-of-phase 30-VERIFICATION.md.

## Deliverables

| Artifact | Location | Size |
|---|---|---|
| JAX simulators per model | `src/rlwm/fitting/models/<m>.py` (appended) | +80 lines × 6 models |
| PPC delegation | `scripts/utils/ppc.py` (modified) | -400 lines, +~60 lines |
| Consistency test | `scripts/fitting/tests/test_simulator_likelihood_consistency.py` (new) | ~200 lines |
| Deletion | `src/rlwm/models/` (removed) | -644 lines |
| Benchmark | `.planning/phases/30-.../30-VERIFICATION.md` | new |
| Doc update | `docs/03_methods_reference/MODEL_REFERENCE.md` or `docs/04_methods/` | small |

Net: ~400 lines added (JAX simulators + consistency test + delegation
wrappers), ~1040 lines deleted (NumPy duplicates + rlwm.models/);
repository shrinks by ~640 lines while gaining simulator capabilities.

## Out of scope (future phases)

- M4 LBA JAX simulator (deferred; PPC of M4 stays on CPU for now)
- Gym-env-wrapped JAX agent (if interactive exploration is ever
  needed again)
- Replacing `scripts/utils/ppc.py` entirely with a class-based
  interface (current functional API is fine; no user demand for
  object wrapping)
