---
phase: 29-pipeline-canonical-reorg
plan: 08
type: execute
wave: 6
depends_on: [29-01, 29-07]
optional: true
files_modified:
  - src/rlwm/fitting/core.py                         (new — ~300 lines: padding, softmax, epsilon, scan primitives)
  - src/rlwm/fitting/models/__init__.py              (new)
  - src/rlwm/fitting/models/qlearning.py             (new — M1 likelihood variants + numpyro wrapper)
  - src/rlwm/fitting/models/wmrl.py                  (new — M2)
  - src/rlwm/fitting/models/wmrl_m3.py               (new — M3)
  - src/rlwm/fitting/models/wmrl_m5.py               (new — M5)
  - src/rlwm/fitting/models/wmrl_m6a.py              (new — M6a)
  - src/rlwm/fitting/models/wmrl_m6b.py              (new — M6b + subscale variant)
  - src/rlwm/fitting/models/wmrl_m4.py               (new — M4 LBA)
  - src/rlwm/fitting/sampling.py                     (new — run_inference, samples_to_arviz, chain selector)
  - src/rlwm/fitting/jax_likelihoods.py              (gutted to re-export shims OR deleted depending on decision)
  - src/rlwm/fitting/numpyro_models.py               (gutted to re-export shims OR deleted)
  - src/rlwm/fitting/numpyro_helpers.py              (absorbed into sampling.py OR retained as-is)
  - scripts/fitting/tests/test_*.py                  (import updates if jax_likelihoods/numpyro_models shims are deleted)
  - validation/**/*.py                               (import updates if shims deleted)
  - validation/check_v4_closure.py                   (updated if v4 invariants enumerate specific paths that changed)
autonomous: false

must_haves:
  truths:
    - "User has explicitly APPROVED executing this vertical refactor (checkpoint task 1)"
    - "If executed: Every model's likelihood variants + numpyro wrapper live in ONE file (src/rlwm/fitting/models/<model>.py)"
    - "If executed: v4 closure guard (pytest test_v4_closure.py + validation/check_v4_closure.py) still PASSES"
    - "If executed: Every caller of the old jax_likelihoods / numpyro_models import paths either works via re-export shims OR is updated"
    - "If DEFERRED: A note is added to .planning/MILESTONES.md or the v6.0 requirements draft capturing the rationale"
  artifacts:
    - path: "src/rlwm/fitting/core.py"
      provides: "shared JAX primitives (padding, softmax, epsilon, scan) previously in jax_likelihoods.py"
      min_lines: 100
    - path: "src/rlwm/fitting/models/qlearning.py"
      provides: "M1 Q-learning likelihood variants + numpyro wrapper in one file"
    - path: "src/rlwm/fitting/sampling.py"
      provides: "run_inference + samples_to_arviz + chain-selector utilities"
  key_links:
    - from: "src/rlwm/fitting/models/qlearning.py"
      to: "src/rlwm/fitting/core.py"
      via: "from .core import pad_sequences, softmax, ..."
      pattern: "from \\.core import"
---

<objective>
OPTIONAL vertical-by-model refactor of `src/rlwm/fitting/`. Current state: `jax_likelihoods.py` is 6,113 lines and `numpyro_models.py` is 2,722 lines — both contain content for ALL 7 models interleaved. Adding a new model requires edits to both files in distant locations. Vertical target: `src/rlwm/fitting/core.py` (shared primitives) + `src/rlwm/fitting/models/<model>.py` (everything for one model — likelihood variants + numpyro wrapper) + `src/rlwm/fitting/sampling.py` (chain-selector / run_inference utilities).

**THIS PLAN IS GATED ON USER APPROVAL.** Task 1 is a checkpoint that MUST resume with "approved" before any refactor work proceeds. If user denies, Task 2 writes a deferred-to-v6.0 note and the phase summary records the deferral.

Purpose: Make fitting/ maintainable at scale. Adding M7+ today requires editing two 6000-line and 2700-line files in distant places; after refactor, one new `models/m7.py` file captures everything.

Risk: Four plus hours of mechanical work. v4 closure guards reference specific import paths (`from rlwm.fitting.jax_likelihoods import ...`). A refactor MUST either (a) preserve old paths via re-export shims, OR (b) update every caller + closure-guard invariant file simultaneously.

Output: EITHER (a) refactored src/rlwm/fitting/ with old paths preserved via shims + v4 closure green, OR (b) a deferral note in the phase summary explaining why this plan didn't run.
</objective>

<execution_context>
@C:\Users\aman0087\.claude\get-shit-done\workflows\execute-plan.md
@C:\Users\aman0087\.claude\get-shit-done\templates\summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/phases/29-pipeline-canonical-reorg/29-CONTEXT.md
@.planning/phases/29-pipeline-canonical-reorg/29-07-SUMMARY.md
@src/rlwm/fitting/jax_likelihoods.py
@src/rlwm/fitting/numpyro_models.py
@src/rlwm/fitting/numpyro_helpers.py
</context>

<tasks>

<task type="checkpoint:decision" gate="blocking">
  <name>Task 1: User approval gate</name>
  <decision>Execute the vertical refactor of src/rlwm/fitting/, OR defer to a future milestone (v6.0)?</decision>
  <context>
The current horizontal-by-type split (jax_likelihoods.py = 6,113 lines; numpyro_models.py = 2,722 lines; both span all 7 models) works but is hard to navigate and adds friction to new-model additions. A vertical-by-model split makes each model self-contained.

Trade-offs:
- FOR: Maintainability for future model additions. If v6.0 or later introduces M7/M8/etc., the refactor pays off. Better testability (per-model tests become natural).
- AGAINST: 4+ hours of mechanical work. Risk of silent breakage in v4 closure guards that pin specific import paths. v5.0 may be the last milestone touching model math — in which case the investment is sunk cost. Current code is ugly but not broken.

Recommendation from 29-CONTEXT.md §4: "If v5.0 is the last milestone touching model math, this refactor can defer to v6.0. Don't block earlier plans on this decision." Planner's read: unless the user has a concrete v6.0+ model-addition plan, DEFER is the conservative choice.

Scope if approved: create `src/rlwm/fitting/core.py` + `models/{qlearning,wmrl,wmrl_m3,wmrl_m5,wmrl_m6a,wmrl_m6b,wmrl_m4}.py` + `sampling.py`; split the two big files mechanically; either preserve old import paths via thin re-export shims (safer, leaves dead code) or update every caller (cleaner, higher blast radius).
  </context>
  <options>
    <option id="execute">
      <name>Execute the refactor</name>
      <pros>Vertical layout; M7+ additions become single-file; long-term maintainability</pros>
      <cons>4+ hours; risk of silent import-path breakage; requires re-export shims OR simultaneous caller update; v5.0 may be the last math-touching milestone</cons>
    </option>
    <option id="defer">
      <name>Defer to v6.0</name>
      <pros>Zero risk to v4 closure guards; zero time spent; current layout ugly-but-working</pros>
      <cons>Deferred tech debt; if v6.0+ adds models, pain compounds</cons>
    </option>
  </options>
  <resume-signal>Type "approved" to execute the refactor, or "defer" to record the deferral and skip Task 2.</resume-signal>
</task>

<task type="auto">
  <name>Task 2: (IF user said "defer") Record deferral in phase summary + v6.0 candidate list</name>
  <files>
    - .planning/phases/29-pipeline-canonical-reorg/29-08-SUMMARY.md (new — documents the deferral)
  </files>
  <action>
    Execute this task ONLY if user chose "defer" in Task 1.
    
    1. Write `.planning/phases/29-pipeline-canonical-reorg/29-08-SUMMARY.md`:
    ```markdown
    # 29-08: src/rlwm/fitting/ vertical refactor — DEFERRED
    
    **Decision:** Deferred to v6.0 (or dropped if no new model-math work planned).
    **Date:** {YYYY-MM-DD}
    **User signal:** "defer"
    
    ## Rationale
    
    Current layout:
    - `src/rlwm/fitting/jax_likelihoods.py` — 6,113 lines
    - `src/rlwm/fitting/numpyro_models.py` — 2,722 lines
    - `src/rlwm/fitting/numpyro_helpers.py` — 308 lines
    
    Proposed vertical target: `core.py` + `models/{qlearning,wmrl,wmrl_m3,wmrl_m5,wmrl_m6a,wmrl_m6b,wmrl_m4}.py` + `sampling.py`.
    
    Ugly but working. v5.0 likely the last model-math-touching milestone. Sunk-cost risk outweighs current navigation pain. Re-evaluate when (a) adding a new model (M7+) or (b) expanding likelihood variants.
    
    ## v6.0 candidate
    
    If v6.0 introduces new models, start that milestone with this refactor as Phase 1.
    ```
    
    2. Add the same note to `.planning/MILESTONES.md` under a "v6.0 candidates" or "Deferred items" section, if such a section exists. (If not, skip — the SUMMARY file above is sufficient.)
    
    3. Commit:
       ```
       docs(29-08): defer src/rlwm/fitting/ vertical refactor to v6.0
       
       User signaled "defer" at Task 1 approval gate. Rationale and v6.0 candidacy documented in 29-08-SUMMARY.md.
       Current horizontal layout retained; v4 closure guards untouched.
       ```
    
    4. EXIT this plan. Phase 29 proceeds to closure via 29-07 (already executed upstream).
  </action>
  <verify>
    - If deferred: `test -f .planning/phases/29-pipeline-canonical-reorg/29-08-SUMMARY.md` and it contains "DEFERRED"
    - v4 closure still green (unchanged from 29-07's state)
  </verify>
  <done>Deferral recorded; phase exits cleanly.</done>
</task>

<task type="auto">
  <name>Task 3: (IF user said "approved") Execute the refactor — extract core, split per-model, add re-export shims</name>
  <files>
    - src/rlwm/fitting/core.py (new)
    - src/rlwm/fitting/models/__init__.py (new)
    - src/rlwm/fitting/models/qlearning.py (new)
    - src/rlwm/fitting/models/wmrl.py (new)
    - src/rlwm/fitting/models/wmrl_m3.py (new)
    - src/rlwm/fitting/models/wmrl_m5.py (new)
    - src/rlwm/fitting/models/wmrl_m6a.py (new)
    - src/rlwm/fitting/models/wmrl_m6b.py (new)
    - src/rlwm/fitting/models/wmrl_m4.py (new)
    - src/rlwm/fitting/sampling.py (new)
    - src/rlwm/fitting/jax_likelihoods.py (gutted → re-export shim)
    - src/rlwm/fitting/numpyro_models.py (gutted → re-export shim)
  </files>
  <action>
    Execute this task ONLY if user chose "approved" in Task 1.
    
    1. Read `src/rlwm/fitting/jax_likelihoods.py` in full (6,113 lines). Identify:
       - Top-level helpers (padding, softmax, epsilon-adjust, scan primitives) → these go into `core.py`.
       - Per-model sections (qlearning, wmrl, wmrl_m3, wmrl_m5, wmrl_m6a, wmrl_m6b, wmrl_m4). Each section contains multiple likelihood variants (sequential, pscan, etc.) — they all stay together in that model's file.
       - Inline tests (if any) — stay adjacent to their math (in per-model files).
    2. Read `src/rlwm/fitting/numpyro_models.py` in full (2,722 lines). Identify per-model numpyro wrappers. Each goes into the same `models/<model>.py` file as its corresponding likelihood.
    3. Read `src/rlwm/fitting/numpyro_helpers.py` (308 lines). Absorb into `sampling.py` unless it has distinct responsibilities — in that case retain as-is.
    4. Create `src/rlwm/fitting/core.py` with:
       - `from __future__ import annotations`
       - NumPy-style module docstring
       - The shared JAX primitives, in their original form (no changes to logic).
    5. Create `src/rlwm/fitting/models/__init__.py` — empty or with a module docstring listing the 7 model modules.
    6. For each model (qlearning, wmrl, wmrl_m3, wmrl_m5, wmrl_m6a, wmrl_m6b, wmrl_m4):
       - Create `src/rlwm/fitting/models/<model>.py`
       - `from __future__ import annotations`
       - `from ..core import <primitives it needs>`
       - Paste the model's likelihood variants + numpyro wrapper verbatim.
    7. Create `src/rlwm/fitting/sampling.py` with `run_inference`, `samples_to_arviz`, chain-selector utilities. Absorb `numpyro_helpers.py` contents if that's the decision; else leave helpers where they are.
    8. CHOOSE re-export strategy — RECOMMENDED: thin shim preserves v4 closure invariants:
       - Replace `src/rlwm/fitting/jax_likelihoods.py` content with ONE LINE per-symbol re-exports:
         ```python
         """Legacy import path — canonical home moved in Phase 29-08 to rlwm.fitting.core + rlwm.fitting.models.<model>."""
         from __future__ import annotations
         
         # Shared primitives
         from .core import *  # noqa: F401,F403
         # Per-model exports
         from .models.qlearning import *  # noqa: F401,F403
         from .models.wmrl import *  # noqa: F401,F403
         from .models.wmrl_m3 import *  # noqa: F401,F403
         from .models.wmrl_m5 import *  # noqa: F401,F403
         from .models.wmrl_m6a import *  # noqa: F401,F403
         from .models.wmrl_m6b import *  # noqa: F401,F403
         from .models.wmrl_m4 import *  # noqa: F401,F403
         ```
         Every consumer that does `from rlwm.fitting.jax_likelihoods import <symbol>` still works unchanged. Trade-off: wildcard imports; each model file should define `__all__` to control what re-exports.
       - Same treatment for `src/rlwm/fitting/numpyro_models.py`:
         ```python
         """Legacy import path — canonical home moved in Phase 29-08 to rlwm.fitting.models.<model>."""
         from __future__ import annotations
         from .models.qlearning import *  # noqa: F401,F403
         # ... etc
         ```
    9. Run the full test suite:
       - `pytest scripts/fitting/tests/ tests/ validation/ -v`
       - `validation/check_v4_closure.py --milestone v4.0`
       - Expect zero new failures.
    10. If any test fails due to a missing symbol (e.g., a consumer imports a symbol that the wildcard export didn't include), fix the per-model file's `__all__` list and re-run.
    11. Atomic commit:
        ```
        refactor(29-08): src/rlwm/fitting/ vertical-by-model layout
        
        - core.py: shared JAX primitives (padding, softmax, epsilon, scan)
        - models/{qlearning,wmrl,wmrl_m3,wmrl_m5,wmrl_m6a,wmrl_m6b,wmrl_m4}.py: one file per model (likelihoods + numpyro wrapper)
        - sampling.py: run_inference + samples_to_arviz + chain-selector (absorbs numpyro_helpers)
        - jax_likelihoods.py + numpyro_models.py retained as re-export shims for backward-compat (v4 closure invariants preserved)
        - Full pytest suite passes; v4 closure exits 0
        ```
  </action>
  <verify>
    - `test -f src/rlwm/fitting/core.py && test -d src/rlwm/fitting/models`
    - `for m in qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b wmrl_m4; do test -f src/rlwm/fitting/models/$m.py || { echo "MISSING: $m"; exit 1; }; done`
    - `test -f src/rlwm/fitting/sampling.py`
    - `python -c "from rlwm.fitting.jax_likelihoods import *; from rlwm.fitting.numpyro_models import *; print('shims ok')"` prints `shims ok`
    - `pytest scripts/fitting/tests/test_v4_closure.py -v` PASSES 3/3
    - `pytest scripts/fitting/tests/ tests/ validation/ -v` — zero new failures vs. pre-plan baseline
    - `validation/check_v4_closure.py --milestone v4.0` exits 0
    - `pytest tests/test_v5_phase29_structure.py -v` still passes (no structural regressions from 29-07)
  </verify>
  <done>Vertical-by-model layout in place; shims preserve old paths; full test suite green.</done>
</task>

</tasks>

<verification>
```bash
# If deferred: only 29-08-SUMMARY.md exists with "DEFERRED" content
# If executed:
test -f src/rlwm/fitting/core.py
test -d src/rlwm/fitting/models
ls src/rlwm/fitting/models/
pytest scripts/fitting/tests/test_v4_closure.py -v
python validation/check_v4_closure.py --milestone v4.0
pytest tests/test_v5_phase29_structure.py -v  # structure guard still passes
```
</verification>

<success_criteria>
EITHER:
- (DEFERRED path) `.planning/phases/29-pipeline-canonical-reorg/29-08-SUMMARY.md` exists documenting the deferral; no src/ changes; phase 29 closes via 29-07 artifacts only.
OR:
- (EXECUTED path) `src/rlwm/fitting/` contains `core.py`, `models/{qlearning,wmrl,wmrl_m3,wmrl_m5,wmrl_m6a,wmrl_m6b,wmrl_m4}.py`, `sampling.py`; old top-level files `jax_likelihoods.py` + `numpyro_models.py` retained as re-export shims; v4 closure green; full test suite passes.
</success_criteria>

<output>
After completion, create `.planning/phases/29-pipeline-canonical-reorg/29-08-SUMMARY.md` with:
- If deferred: deferral rationale + v6.0 candidacy note
- If executed: file structure diff (before/after), line-count deltas per file, v4 closure + pytest evidence, commit SHA
</output>
