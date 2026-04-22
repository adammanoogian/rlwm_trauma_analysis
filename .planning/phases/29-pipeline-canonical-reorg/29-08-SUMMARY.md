---
phase: 29-pipeline-canonical-reorg
plan: 08
subsystem: infra
tags: [refactor, src-layout, jax, numpyro, vertical-by-model, re-export-shim, byte-identical, ast-hash-verification]

# Dependency graph
requires:
  - phase: 29-01-scripts-canonical-reorg
    provides: canonical 6-stage scripts/ layout (unchanged by this plan)
  - phase: 29-07-closure-guard-extension
    provides: pytest closure guard + REFAC-14..20 requirements baseline
provides:
  - src/rlwm/fitting/core.py (681 lines): shared JAX primitives
  - src/rlwm/fitting/models/<m>.py (x7): one file per Senta 2025 model
  - src/rlwm/fitting/sampling.py (777 lines): MCMC orchestration layer
  - jax_likelihoods.py + numpyro_models.py retained as re-export shims
  - byte-identical preservation of every pre-refactor function body (0 hash drift)
affects:
  - Phase 30+ (future model additions: one new models/m7.py file instead of edits across 2 files)
  - v5.0 milestone closure (REFAC-14..20 vertical-layout requirement satisfied)
  - v4.0 closure invariants (preserved via shims — zero regression)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Vertical-by-model layout: core.py (shared) + models/<model>.py (self-contained per model)"
    - "Re-export shim pattern: legacy jax_likelihoods.py + numpyro_models.py retained as wildcard re-exports"
    - "AST-hash byte-identity verification for pure-relocation refactors"
    - "Programmatic source-segment relocation via ast.get_source_segment + AST index"

key-files:
  created:
    - src/rlwm/fitting/core.py
    - src/rlwm/fitting/models/__init__.py
    - src/rlwm/fitting/models/qlearning.py
    - src/rlwm/fitting/models/wmrl.py
    - src/rlwm/fitting/models/wmrl_m3.py
    - src/rlwm/fitting/models/wmrl_m5.py
    - src/rlwm/fitting/models/wmrl_m6a.py
    - src/rlwm/fitting/models/wmrl_m6b.py
    - src/rlwm/fitting/models/wmrl_m4.py
    - src/rlwm/fitting/sampling.py
  modified:
    - src/rlwm/fitting/jax_likelihoods.py (6113 lines -> 129-line re-export shim)
    - src/rlwm/fitting/numpyro_models.py (2722 lines -> 35-line re-export shim)

key-decisions:
  - "Single-commit atomic refactor (vs multi-commit per destination): a pure-relocation refactor can only be validated after ALL destinations land plus shims; splitting would leave intermediate commits with broken imports"
  - "numpyro_helpers.py retained as-is (not absorbed into sampling.py): it is self-contained hBayesDM-style non-centered-parameter helpers (phi_approx, sample_bounded_param, sample_capacity, sample_model_params, PARAM_PRIOR_DEFAULTS); its responsibilities are orthogonal to MCMC orchestration; merging would mix concerns and break the 5 external callers that import from it directly"
  - "Per-model files include their test_* smoke drivers (ad-hoc, not pytest) alongside their math for colocation; jax_likelihoods.py shim preserves the pre-refactor __main__ driver by importing them back"
  - "Wildcard re-export with per-file __all__ list (as opposed to explicit symbol re-exports in the shim): preserves backward compat with minimum boilerplate and fewest edit points for future additions"
  - "Verbatim preservation of inline deferred imports (e.g., `from rlwm.fitting.jax_likelihoods import q_learning_fully_batched_likelihood` inside qlearning_hierarchical_model_stacked): these are load-bearing for circular-avoidance semantics and work correctly via the shim; rewriting them would risk byte-identity drift"

patterns-established:
  - "Vertical-by-model: adding M7 becomes one new file models/m7.py + one line in the shim (was: edits at 12+ distant locations across 2 large files)"
  - "Shared-primitive extraction into core.py with explicit __all__ (15 entries)"
  - "Per-model __all__ curation sized by caller-import scan (14 entries for qlearning, 13 for wmrl, 10 for m3/m5/m6a, 12 for m6b, 2 for m4, 9 for sampling)"

# Metrics
duration: ~2h (execution: read plan, build relocation map, programmatic extraction, fix 3 deviations, verify)
completed: 2026-04-22
---

# Phase 29-08: src/rlwm/fitting/ Vertical-by-Model Refactor Summary

**Split the 6113-line jax_likelihoods.py and 2722-line numpyro_models.py into core.py (681 lines) + 7 self-contained models/<model>.py files + sampling.py (777 lines), preserving byte-identity of every relocated function via ast-hash verification and keeping the old import paths live via thin wildcard re-export shims.**

## Performance

- **Duration:** ~2 hours execution (including deviation fixes and verification)
- **Started:** 2026-04-22
- **Completed:** 2026-04-22
- **Tasks:** 1 of 3 executed (Task 1 user-approval checkpoint resolved "approved" pre-execution; Task 2 "defer" path skipped; Task 3 "execute" path ran)
- **Files created:** 10 (core.py, 8 models/*.py including __init__, sampling.py)
- **Files modified:** 2 (both shim conversions)
- **Lines moved:** 8835 lines of function bodies relocated byte-identically
- **Commits:** 2 (47f8b68 initial refactor + be4594a circular-import fix)

## Accomplishments

- Every Senta et al. (2025) model now has ONE file holding its JAX likelihood variants (sequential + pscan + fully_batched) AND its NumPyro hierarchical wrapper. Adding M7 is now a single-file addition instead of edits scattered across two 6000-line files.
- v4.0 closure guard completely preserved (3/3 pytest pass, 5/5 CLI pass, exit 0) — zero regression to the milestone v4.0 invariants.
- Phase 29 structure guard unchanged (31/31 pass) — the vertical refactor did not touch any of the 6-stage scripts/ structure that 29-07 pinned.
- Top-level tests/ 69/69 pass (1 skipped as before). validation/ 54/54 pass (including the critical test_m3_backward_compat.py suite that exercises the full M3 likelihood through the shim).
- Fitting-layer smoke tests 38/38 pass (v4_closure, load_side_validation, compile_gate, bms, bayesian_summary, numpyro_helpers) + 27/27 pscan core agreements + test_wmrl_model 2/2 + test_prior_predictive 1/1 (gate_helper failure is pre-existing, unrelated to this plan).

## Task Commits

1. **Task 3: Execute the refactor — extract core, split per-model, add re-export shims** — `47f8b68` (refactor)
2. **Task 3 follow-up: fix circular import by moving stacking helpers to core.py** — `be4594a` (fix)
3. **Plan metadata** — [this summary + STATE.md commit, below]

## Files Created/Modified

### Created (canonical homes)

| File | Lines | Role |
|---|---:|---|
| `src/rlwm/fitting/core.py` | 866 | Shared JAX primitives: padding, softmax, epsilon noise, associative scan operators, perseveration precompute, stacking helpers, module constants |
| `src/rlwm/fitting/models/__init__.py` | 15 | Package docstring enumerating the 7 model modules |
| `src/rlwm/fitting/models/qlearning.py` | 1201 | M1 asymmetric Q-learning — 14 symbols in __all__ |
| `src/rlwm/fitting/models/wmrl.py` | 1213 | M2 WM-RL hybrid — 13 symbols |
| `src/rlwm/fitting/models/wmrl_m3.py` | 1146 | M3 = M2 + global perseveration (kappa) — 10 symbols |
| `src/rlwm/fitting/models/wmrl_m5.py` | 1141 | M5 = M3 + RL forgetting (phi_rl), current winning choice-only model — 10 symbols |
| `src/rlwm/fitting/models/wmrl_m6a.py` | 1047 | M6a = M2 + stimulus-specific perseveration (kappa_s) — 10 symbols |
| `src/rlwm/fitting/models/wmrl_m6b.py` | 1371 | M6b = M2 + dual perseveration (kappa_total, kappa_share) + subscale variant — 12 symbols |
| `src/rlwm/fitting/models/wmrl_m4.py` | 339 | M4 RLWM-LBA joint choice+RT (numpyro-only, no JAX likelihood) — 2 symbols |
| `src/rlwm/fitting/sampling.py` | 601 | MCMC orchestration: run_inference, samples_to_arviz, chain-method selector, data prep — 7 symbols |

### Modified (now re-export shims)

| File | Before | After | Role |
|---|---:|---:|---|
| `src/rlwm/fitting/jax_likelihoods.py` | 6113 lines | 129 lines | Wildcard re-exports from core + every models/<m>.py + __main__ smoke-test driver |
| `src/rlwm/fitting/numpyro_models.py` | 2722 lines | 41 lines | Wildcard re-exports from core + sampling + every models/<m>.py |

### Retained as-is

- `src/rlwm/fitting/numpyro_helpers.py` (308 lines) — self-contained hBayesDM-style priors; not part of relocation. 5 external callers import from it directly.

## Before / After Structure

```
BEFORE (pre-29-08):
  src/rlwm/fitting/
  ├── __init__.py           (1 line)
  ├── jax_likelihoods.py    (6113 lines — 69 functions + 6 constants for ALL 7 models interleaved)
  ├── numpyro_models.py     (2722 lines — 20 functions for ALL 7 models interleaved)
  └── numpyro_helpers.py    (308 lines — hBayesDM helpers)

AFTER (post-29-08, including be4594a stacking-fix):
  src/rlwm/fitting/
  ├── __init__.py           (1 line, unchanged)
  ├── core.py               (866 lines, NEW — shared JAX primitives + stacking helpers)
  ├── sampling.py           (601 lines, NEW — MCMC orchestration)
  ├── jax_likelihoods.py    (129 lines, SHIM — wildcard re-exports)
  ├── numpyro_models.py     (41 lines, SHIM — wildcard re-exports)
  ├── numpyro_helpers.py    (308 lines, unchanged)
  └── models/
      ├── __init__.py       (15 lines, NEW — package docstring)
      ├── qlearning.py      (1201 lines, NEW — M1: JAX + numpyro)
      ├── wmrl.py           (1213 lines, NEW — M2: JAX + numpyro)
      ├── wmrl_m3.py        (1146 lines, NEW — M3: JAX + numpyro)
      ├── wmrl_m5.py        (1141 lines, NEW — M5: JAX + numpyro)
      ├── wmrl_m6a.py       (1047 lines, NEW — M6a: JAX + numpyro)
      ├── wmrl_m6b.py       (1371 lines, NEW — M6b: JAX + numpyro + subscale variant)
      └── wmrl_m4.py        (339 lines, NEW — M4: numpyro-only)
```

Total line count: 9144 (before) -> 9418 (after). Net +274 lines accounts for per-file headers + module docstrings + __all__ lists + added imports; the function bodies themselves are byte-identical.

## __all__ Coverage per Destination

| File | __all__ size | Caller-scan verified |
|---|---:|---:|
| core.py | 17 | shared primitives (padding, softmax, scans, perseveration precompute, module constants) + 2 stacking helpers moved in fix commit be4594a |
| models/qlearning.py | 14 | covers prepare_block_data, q_learning_*_likelihood[*], qlearning_hierarchical_model[*_stacked], test_* |
| models/wmrl.py | 13 | covers wmrl_*_likelihood[*] (sequential + pscan + fully_batched), wmrl_hierarchical_model[*_stacked], test_* |
| models/wmrl_m3.py | 10 | covers wmrl_m3_*_likelihood[*], wmrl_m3_hierarchical_model, test_* |
| models/wmrl_m5.py | 10 | covers wmrl_m5_*_likelihood[*], wmrl_m5_hierarchical_model, test_* |
| models/wmrl_m6a.py | 10 | covers wmrl_m6a_*_likelihood[*], wmrl_m6a_hierarchical_model, test_* |
| models/wmrl_m6b.py | 12 | covers wmrl_m6b_*_likelihood[*], wmrl_m6b_hierarchical_model + _subscale, test_* |
| models/wmrl_m4.py | 2 | covers prepare_stacked_participant_data_m4, wmrl_m4_hierarchical_model |
| sampling.py | 7 | covers _select_chain_method, prepare_data_for_numpyro, test_likelihood_compilation, run_inference, run_inference_with_bump, samples_to_arviz, test_model_with_synthetic_data |
| **Total** | **95** | |

External-caller-import scan found 40 symbols imported from `rlwm.fitting.jax_likelihoods` + 17 from `rlwm.fitting.numpyro_models` + 5 from `rlwm.fitting.numpyro_helpers` = 62 unique external-import symbols. All 62 resolve via the shims (verified: `python -c "from rlwm.fitting.jax_likelihoods import *; from rlwm.fitting.numpyro_models import *; print('shims ok')"` prints `shims ok`, and an explicit per-symbol `hasattr` check confirms all expected-by-caller symbols exist on the shim modules).

## Hash-Equality Verification

A pre-refactor SHA-256 baseline was captured for every top-level `def`/`class`/module-level-`Assign` node in the three source files via `ast.get_source_segment(..., padded=False)` + `hashlib.sha256(canon.encode()).hexdigest()[:16]` (canon = `'\n'.join(l.rstrip() for l in seg.splitlines()).strip('\n')`).

Baseline captured 101 top-level symbols:
- `jax_likelihoods.py`: 75 (69 functions + 6 module constants)
- `numpyro_models.py`: 20 (20 functions, 0 module constants)
- `numpyro_helpers.py`: 6 (3 functions + 2 constants + 1 function)

Post-relocation re-hash:
- Every one of the 95 relocated symbols has **identical hash** in its new home (95 relocated + 6 numpyro_helpers retained = 101 total, zero drift).
- Byte-identity proves no accidental edits slipped in during the mechanical relocation.
- Only additions: per-file `CONST:__all__` entries (one per destination file, expected and required).

Evidence artifacts (not committed, in `_tmp_*.{py,json}`):
- `_tmp_ast_baseline.json`: pre-refactor hash manifest
- `_tmp_verify_hashes.py`: post-relocation verifier; output "Drift (same name, different hash): 0"

## Deviations from Plan

Three deviations, all auto-fixed per GSD Rule 1 (auto-fix bugs). All three were imports that my programmatic-relocation script omitted from the per-destination-file header templates — the bodies themselves were byte-identical to pre-refactor, but the new files needed some top-level imports that the old jax_likelihoods.py / numpyro_models.py had at module scope.

### Auto-fixed Issues

**1. [Rule 1 - Bug] Missing `from jax import lax` in 6 per-model files**

- **Found during:** Task 3 (after writing all destination files, ran `test_pscan_likelihoods.py` and got `NameError: name 'lax' is not defined` in every multiblock function's `lax.fori_loop` call)
- **Issue:** My initial header templates for `models/qlearning.py`, `models/wmrl.py`, `models/wmrl_m3.py`, `models/wmrl_m5.py`, `models/wmrl_m6a.py`, `models/wmrl_m6b.py` imported `jax` and `jax.numpy as jnp` but not `lax`. The original `jax_likelihoods.py` had `from jax import lax` at top-level.
- **Fix:** Added `from jax import lax` to all six per-model headers in the relocation script and regenerated. Hash equality re-verified post-fix: zero drift.
- **Files modified:** `src/rlwm/fitting/models/qlearning.py`, `wmrl.py`, `wmrl_m3.py`, `wmrl_m5.py`, `wmrl_m6a.py`, `wmrl_m6b.py` (only the header — imports, not body)
- **Verification:** `test_pscan_likelihoods.py::test_affine_scan_ar1 test_affine_scan_reset test_pscan_agreement_synthetic[qlearning]` all PASS after fix; scope expanded to full `test_pscan_likelihoods.py` minus slow n=154 tests (27/27 PASS).
- **Committed in:** `47f8b68` (part of the atomic refactor commit — fix landed before the commit)

**2. [Rule 1 - Bug] Missing `pandas`, `typing.Any`, `numpyro.infer.MCMC/NUTS` in multiple per-model files**

- **Found during:** Task 3 (after running test_wmrl_model.py + test_m4_hierarchical.py)
- **Issue:** NumPyro hierarchical models + M4 data prep reference `pd.` (pandas.DataFrame preprocessing), `Any` (type hints), and `MCMC`/`NUTS` (inside the hierarchical models). Original `numpyro_models.py` imported all three at top-level.
- **Fix:** Added `import pandas as pd`, `from typing import Any`, `from numpyro.infer import MCMC, NUTS` to `models/qlearning.py` and `models/wmrl.py`; added `import pandas as pd`, `from typing import Any`, `from ..core import MAX_TRIALS_PER_BLOCK, pad_block_to_max` to `models/wmrl_m4.py`.
- **Files modified:** `src/rlwm/fitting/models/qlearning.py`, `wmrl.py`, `wmrl_m4.py` (headers only)
- **Verification:** `test_wmrl_model.py::test_wmrl_model_compilation` + `test_wmrl_prior_ranges` both PASS; `test_compile_gate.py` (end-to-end MCMC-compile smoke via qlearning_hierarchical_model) PASSES.
- **Committed in:** `47f8b68`

**3. [Rule 1 - Bug] `sampling.py` missing `MAX_TRIALS_PER_BLOCK`, `pad_block_to_max`, `prepare_block_data`, `q_learning_multiblock_likelihood`, `pandas`, `typing.Any`**

- **Found during:** Task 3 (after running test_prior_predictive.py, `stack_across_participants` raised `NameError: name 'MAX_TRIALS_PER_BLOCK' is not defined`)
- **Issue:** `sampling.py` contains `stack_across_participants` + `prepare_data_for_numpyro` + `test_likelihood_compilation` + `test_model_with_synthetic_data` which reference `MAX_TRIALS_PER_BLOCK`, `pad_block_to_max`, `prepare_block_data`, `q_learning_multiblock_likelihood`, `pd.DataFrame`, and `Any` type hints. Original `numpyro_models.py` imported all of these at top level.
- **Fix:** Added `from typing import Any`, `import pandas as pd`, `from .core import MAX_TRIALS_PER_BLOCK, pad_block_to_max`, `from .models.qlearning import prepare_block_data, q_learning_multiblock_likelihood` to `sampling.py` header.
- **Files modified:** `src/rlwm/fitting/sampling.py` (header only)
- **Verification:** `test_prior_predictive.py::test_prior_predictive_wmrl_m3_smoke` PASSES; `test_compile_gate.py` (end-to-end MCMC) PASSES.
- **Committed in:** `47f8b68`

---

**4. [Rule 1 - Bug] Circular import: stacking helpers need to be callable inside hierarchical-model bodies**

- **Found during:** Task 3 post-commit verification (ran `test_m3_hierarchical.py::test_smoke_dispatch` and got `NameError: name 'stack_across_participants' is not defined` inside `wmrl_m3_hierarchical_model` body)
- **Issue:** `stack_across_participants` and `prepare_stacked_participant_data` are called directly inside 6 of the 7 model files' hierarchical-model function bodies (verbatim from pre-refactor `numpyro_models.py` where everything lived in one module). Initially I placed them in `sampling.py`, but `sampling.py` already imports from `models.qlearning` (for `prepare_block_data` + `q_learning_multiblock_likelihood`) — so having the per-model files also import from `sampling.py` creates a circular import at module load. Since the function-body calls are byte-locked (can't add deferred-import statements without hash drift), the imports have to succeed at module load.
- **Fix:** Move `stack_across_participants` and `prepare_stacked_participant_data` from `sampling.py` to `core.py`. These are pure data-prep utilities over padded arrays with no MCMC or numpyro orchestration dependencies — they only use `jnp`, `np`, `MAX_TRIALS_PER_BLOCK`, `pad_block_to_max`, which `core.py` already provides. `core.py` __all__ grows from 15 to 17; `sampling.py` __all__ shrinks from 9 to 7. Per-model files now `from ..core import stack_across_participants, prepare_stacked_participant_data`. The `numpyro_models.py` shim adds `from .core import *` so that external callers who import stacking helpers from the legacy path still work.
- **Files modified:** `src/rlwm/fitting/core.py` (+ 2 symbols + body), `src/rlwm/fitting/sampling.py` (- 2 symbols + body), `src/rlwm/fitting/numpyro_models.py` (+ `from .core import *`), 7 of the models/*.py files (+ imports)
- **Verification:** `test_m3_hierarchical.py::test_smoke_dispatch` + `test_smoke_dispatch_with_l2` both PASS (2/2, 154s); the full `test_m3_hierarchical.py` suite (8 tests including 7 `*_fully_batched_matches_sequential`) all PASS; `test_m4_hierarchical.py` all 3 tests PASS; `test_numpyro_models_2cov.py` 9/9 collected tests PASSED before I killed the long-running recovery test (not reached in the suite). Hash-drift re-audit: zero drift.
- **Committed in:** `be4594a` (follow-up fix commit — separate from the initial 47f8b68 to keep the "pure relocation" vs "correctness fix" boundary clear in git history)

---

**Total deviations:** 4 auto-fixed (4 missing-import / structural bugs, all from programmatic-header-template gaps + one circular-import design mistake)
**Impact on plan:** All four fixes were necessary for the refactor to be non-breaking. Zero scope creep. The first three (import additions) restore pre-refactor imports that existed at module top-level in the original source files; the fourth (stacking-helper relocation) is a structural adjustment specific to the vertical-by-model target and could not have been verified until test-time. Hash-equality verification (zero drift on 95 relocated symbols) confirms no accidental edits to function bodies were introduced.

## Issues Encountered

- **Windows subprocess output buffering under pytest `-q`**: When running the full `scripts/fitting/tests/` suite via `python -m pytest ... -q`, pytest's stdout buffering on Windows + Git-Bash redirection caused the output file to remain 0 bytes for the duration of the run. Switching to `-v` (per-test flushing) or running focused subsets resolved this. The pytest runs themselves were not affected — this was purely a terminal-visibility issue during background monitoring. Confirmed via process memory (3.5 GB RSS) that JAX was actively compiling likelihoods. Mitigation: ran multiple small focused subsets (27 pscan core + 54 validation + 69 top-level tests + 38 fitting smoke) totaling 188 passes.
- **Pre-existing test failures (not caused by this refactor, verified via pre-refactor run)**: `test_bayesian_recovery.py` + `test_mle_quick.py` collection errors reference `scripts.fitting.fit_bayesian` / `scripts.fitting.fit_mle` which were deleted in Phase 28; `test_prior_predictive_gate_helper` references `scripts/bayesian_pipeline/21_run_prior_predictive.py` which was moved in Phase 28 to `scripts/03_model_prefitting/12_run_prior_predictive.py`. None of these are related to 29-08 — they are Phase-28 orphans.

## Verification Evidence Summary

| Check | Result | Evidence |
|---|---|---|
| AST hash drift on relocated symbols | 0 / 95 | `_tmp_verify_hashes.py`: "Drift (same name, different hash): 0" |
| Shim smoke: wildcard imports succeed | PASS | `python -c "from rlwm.fitting.jax_likelihoods import *; from rlwm.fitting.numpyro_models import *; print('shims ok')"` prints `shims ok` |
| Shim coverage: every caller-expected symbol accessible | 62/62 | Explicit `hasattr(jax_mod, sym)` + `hasattr(np_mod, sym)` for every symbol in the caller-import scan |
| v4.0 closure pytest | 3/3 PASS | `pytest scripts/fitting/tests/test_v4_closure.py` |
| v4.0 closure CLI | 5/5 PASS, exit 0 | `python validation/check_v4_closure.py` |
| Phase 29 structure closure pytest | 31/31 PASS | `pytest tests/test_v5_phase29_structure.py` |
| Top-level tests/ | 69 passed, 1 skipped | `pytest tests/` |
| validation/ | 54/54 PASS | `pytest validation/` (includes `test_m3_backward_compat.py` — full M3 likelihood path through shim) |
| Fitting smoke (v4 closure, compile_gate, bms, numpyro_helpers) | 38/38 PASS | `pytest scripts/fitting/tests/{test_v4_closure,test_load_side_validation,test_compile_gate,test_bms,test_bayesian_summary,test_numpyro_helpers}.py` |
| pscan likelihoods core (agreement synthetic + real data) | 27/27 PASS | `pytest scripts/fitting/tests/test_pscan_likelihoods.py` minus slow n=154 tests |

## Next Phase Readiness

- **Phase 29 ready for final aggregate verification.** Orchestrator should flip REFAC-14..REFAC-20 status from Pending to Complete in `.planning/REQUIREMENTS.md` and file the 29-aggregate VERIFICATION.md update referencing commit `47f8b68` plus this SUMMARY.
- **v4.0 closure guards remain green.** Zero regression to milestone v4.0.
- **Blockers:** None.
- **Concerns for future phases:**
  - If a future phase renames any of the new canonical file paths (e.g., moves `core.py` into a submodule), the shim files must be updated to match. Current shims use relative imports (`from .core import *`, `from .models.qlearning import *`) so as long as the relative layout is preserved inside `src/rlwm/fitting/`, the shims stay valid.
  - The wildcard re-export shims trigger no ruff warnings because of the `# noqa: F401,F403` comments, but ruff/mypy strict modes may flag them. If that becomes an issue, the shims can be rewritten with explicit symbol lists — straightforward mechanical edit.

---
*Phase: 29-pipeline-canonical-reorg*
*Plan: 08*
*Completed: 2026-04-22*
