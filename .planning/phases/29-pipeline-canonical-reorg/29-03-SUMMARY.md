---
phase: 29-pipeline-canonical-reorg
plan: "03"
subsystem: infra
tags: [utils, refactor, ppc, simulator, single-source, canonical-names, git-mv]

# Dependency graph
requires:
  - phase: 29-pipeline-canonical-reorg
    plan: "01"
    provides: "Canonical 01-06 stage layout (01_data_preprocessing/, 02_behav_analyses/, 03_model_prefitting/, 04_model_fitting/, 05_post_fitting_checks/, 06_fit_analyses/, scripts/fitting/, scripts/utils/)"
provides:
  - "scripts/utils/ppc.py: canonical single-source PPC simulator (simulate_from_samples + run_prior_ppc + run_posterior_ppc) used by stage 03 prior PPC, stage 03 posterior PPC, and stage 05 posterior PPC"
  - "scripts/utils/__init__.py: formal package marker with module-map docstring"
  - "Canonical short-named helpers: plotting.py, stats.py, scoring.py (renamed from plotting_utils.py, statistical_tests.py, scoring_functions.py — git history preserved)"
  - "scripts/05_post_fitting_checks/run_posterior_ppc.py: thin stage-05 orchestrator mirroring the stage-03 prior-PPC orchestrator"
  - "scripts/_maintenance/: new home for one-off human-run CLIs (remap_mle_ids, sync_experiment_data, update_participant_mapping); utils/ is now reserved for helpers shared by ≥ 2 stages"
  - "Nine importers rewritten (01_parse_raw_data, 02_create_collated_csv, 04_create_summary_csv, 08_run_statistical_analyses, analyze_mle_by_trauma, regress_parameters_on_scales, scripts/fitting/model_recovery.py, scripts/fitting/level2_design.py docstring)"
affects:
  - "29-04: flagged scripts/analysis/plotting_utils.py sibling — handled by 29-04 dead-folder audit (archived to scripts/legacy/analysis/)"
  - "29-05: cluster SLURM consolidation can assume canonical utils names"
  - "29-06: paper.qmd smoke render — no utils-facing breakage expected"
  - "29-07: closure-guard extension may hash scripts/utils/ppc.py as a locked artifact"
  - "29-08: src/rlwm/fitting vertical refactor — scripts.utils.ppc is a candidate for eventual promotion to rlwm.fitting.ppc if the numpy-simulator layer deserves library-tier status"

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Single-source simulator: simulate_from_samples / run_prior_ppc / run_posterior_ppc defined exactly once (scripts/utils/ppc.py); every PPC-adjacent workflow imports from there — never from sibling stage folders"
    - "Utils membership rule: a file belongs in scripts/utils/ iff ≥ 2 stage folders import it; one-off CLIs go to scripts/_maintenance/"
    - "Canonical short names in utils/: prefer plotting.py over plotting_utils.py, stats.py over statistical_tests.py, scoring.py over scoring_functions.py (the '_utils' / '_functions' / '_tests' suffix is redundant inside a module named 'utils')"
    - "Thin orchestrator pattern: stage CLI scripts ≤ 200 lines, delegate simulation / pipeline logic to scripts.utils.X"
    - "Lazy heavy-import pattern in scripts/utils/ppc.py: arviz / jax / numpyro / scripts.fitting.* imported inside run_prior_ppc() / run_posterior_ppc() body so the module itself stays cheap to import (no JAX startup cost for bare `from scripts.utils.ppc import simulate_from_samples`)"

key-files:
  created:
    - scripts/utils/__init__.py
    - scripts/utils/ppc.py
    - scripts/05_post_fitting_checks/run_posterior_ppc.py
    - scripts/_maintenance/__init__.py
  modified:
    - scripts/03_model_prefitting/09_run_ppc.py (rewritten as thin orchestrator, 204 → 193 lines; delegates to scripts.utils.ppc.run_posterior_ppc)
    - scripts/03_model_prefitting/12_run_prior_predictive.py (rewritten as thin orchestrator, 649 → 109 lines; delegates to scripts.utils.ppc.run_prior_ppc)
    - scripts/fitting/model_recovery.py (3 import rewrites)
    - scripts/fitting/level2_design.py (2 docstring path rewrites)
    - scripts/02_behav_analyses/08_run_statistical_analyses.py (1 import rewrite)
    - scripts/01_data_preprocessing/01_parse_raw_data.py (1 import rewrite)
    - scripts/01_data_preprocessing/02_create_collated_csv.py (1 import rewrite)
    - scripts/01_data_preprocessing/04_create_summary_csv.py (1 import rewrite)
    - scripts/06_fit_analyses/analyze_mle_by_trauma.py (1 import rewrite)
    - scripts/06_fit_analyses/regress_parameters_on_scales.py (1 import rewrite)
  renamed:
    - scripts/utils/plotting_utils.py → scripts/utils/plotting.py (git mv; history preserved — traces back to feat(05-04) add plot_behavioral_comparison)
    - scripts/utils/statistical_tests.py → scripts/utils/stats.py (git mv; history preserved)
    - scripts/utils/scoring_functions.py → scripts/utils/scoring.py (git mv; history preserved)
    - scripts/utils/remap_mle_ids.py → scripts/_maintenance/remap_mle_ids.py (git mv)
    - scripts/utils/sync_experiment_data.py → scripts/_maintenance/sync_experiment_data.py (git mv)
    - scripts/utils/update_participant_mapping.py → scripts/_maintenance/update_participant_mapping.py (git mv)

key-decisions:
  - "Keep prior- and posterior-PPC as TWO functions in scripts/utils/ppc.py, not one monolithic entry point — they consume very different input artifacts (NumPyro Predictive samples vs MLE-fit CSVs) and write different output schemas. A shared simulate_from_samples() kernel dispatches per-trial logic; the two run_*_ppc() wrappers orchestrate load/simulate/write for each workflow."
  - "Lift the prior-PPC implementation from 12_run_prior_predictive.py verbatim (it was the richer, newer Baribault-gate version); the posterior-PPC side is a thin wrapper over scripts.fitting.model_recovery.{run_posterior_predictive_check, run_model_recovery_check} which already existed as the authoritative behavioral-comparison + model-recovery pipeline."
  - "Use lazy imports for JAX / NumPyro / ArviZ / scripts.fitting.* inside the run_*_ppc() bodies so that cheap callers (e.g. unit tests that only need simulate_from_samples) can import scripts.utils.ppc without paying the JAX startup cost."
  - "Do not dedupe against scripts/simulations/unified_simulator.py's simulate_agent_fixed / simulate_agent_sampled — those are a distinct, lower-level Gym-environment layer (single-agent trajectory given fixed params, consumed by tests/validation). The module docstring of scripts/utils/ppc.py explicitly documents this boundary so future readers know where each simulator belongs. (29-04 subsequently archived that file to scripts/legacy/simulations/ but the layering distinction remains valid for any future re-promotion.)"
  - "Move the three maintenance CLIs (remap_mle_ids, sync_experiment_data, update_participant_mapping) out of utils/ into a new scripts/_maintenance/ package. Rule: utils/ is for helpers imported by ≥ 2 stages; these three have zero importers and contain CLI main() only, so they belong with ad-hoc human-run tooling."
  - "Preserve the bare-name sys.path trick in scripts/01_data_preprocessing/{01,02,04}*.py for `from scoring import ...` (rather than forcing them to `from scripts.utils.scoring import ...`) to minimise diff footprint — those three files are Phase-28-era and the sys.path.append(...utils) pattern is entrenched. A future refactor can switch them to canonical imports; not in scope for 29-03."

patterns-established:
  - "Thin-orchestrator + utils-library split: stage scripts do argparse + IO + call utils.X.run_foo(); simulation / comparison / gate logic lives in utils/"
  - "One canonical short name per utility module inside utils/ (plotting, stats, scoring, ppc, data_cleaning) — no '_utils' or '_functions' or '_tests' suffixes"
  - "scripts/_maintenance/ as the home for one-off human-run CLIs (decoupled from the automated pipeline but kept inside scripts/ for discoverability)"

# Metrics
duration: ~35 min (with parallel-agent index-race recovery)
completed: 2026-04-22
---

# Phase 29 Plan 03: Utils Consolidation — Canonical PPC/Plotting/Stats/Scoring Single-Sources

**Extracted PPC simulator into scripts/utils/ppc.py as the single source of truth; renamed plotting_utils/statistical_tests/scoring_functions to canonical short names (plotting/stats/scoring); added stage-05 run_posterior_ppc orchestrator; moved 3 one-off CLIs to scripts/_maintenance/; rewrote 9 importers.**

## Performance

- **Duration:** ~35 min (including recovery from parallel-agent git-index race with 29-04)
- **Started:** 2026-04-22 (Wave 2 of Phase 29)
- **Completed:** 2026-04-22
- **Tasks:** 3 / 3
- **Files modified / created / renamed:** 20 (4 new, 10 modified, 6 renamed)

## Accomplishments

- **Single-source PPC simulator.** `scripts/utils/ppc.py` (935 lines, 11 top-level `def`s) now holds the one authoritative draws-level simulator. It exports `simulate_from_samples`, `run_prior_ppc`, and `run_posterior_ppc`. Stage 03 prior PPC, stage 03 posterior PPC, and stage 05 posterior PPC all import from this one file — zero cross-stage simulator copies remain.
- **Canonical short names.** `plotting_utils.py → plotting.py`, `statistical_tests.py → stats.py`, `scoring_functions.py → scoring.py`. All three `git mv`'d (history preserved — `git log --follow scripts/utils/plotting.py` traces back to `feat(05-04) add plot_behavioral_comparison`).
- **Stage 05 posterior-PPC orchestrator.** New `scripts/05_post_fitting_checks/run_posterior_ppc.py` (134 lines) — thin argparse CLI that delegates to `scripts.utils.ppc.run_posterior_ppc`. Gives stage 05 a first-class PPC entry point alongside `baseline_audit.py` and `scale_audit.py`.
- **Thin stage-03 orchestrators.** `09_run_ppc.py` slimmed from 204 → 193 lines (imports `run_posterior_ppc`); `12_run_prior_predictive.py` slimmed from 649 → 109 lines (imports `run_prior_ppc`). Both are now argparse + dispatch only.
- **Maintenance CLIs relocated.** `scripts/_maintenance/` (new folder) houses `remap_mle_ids.py`, `sync_experiment_data.py`, `update_participant_mapping.py` — all three have zero importers and contain only `def main()` CLIs, so they moved out of utils/ per the "≥ 2 stage importers" rule.
- **Nine importer rewrites.** All live code now uses canonical names:
  - `plotting_utils → plotting`: scripts/fitting/model_recovery.py (×3 call sites), scripts/06_fit_analyses/analyze_mle_by_trauma.py, scripts/06_fit_analyses/regress_parameters_on_scales.py
  - `statistical_tests → stats`: scripts/02_behav_analyses/08_run_statistical_analyses.py
  - `scoring_functions → scoring`: scripts/01_data_preprocessing/01_parse_raw_data.py, scripts/01_data_preprocessing/02_create_collated_csv.py, scripts/01_data_preprocessing/04_create_summary_csv.py (bare-name via sys.path, not fully-qualified — preserved intentionally to minimise diff footprint)
  - `scoring_functions.py → scoring.py` docstring path: scripts/fitting/level2_design.py (×2)

## Functions in scripts/utils/ppc.py

11 top-level definitions:

| Function | Purpose | Lines |
|---|---|---|
| `_softmax(x, beta)` | Numerically stable softmax | 70-85 |
| `_apply_epsilon(probs, epsilon, n_act)` | hBayesDM-style uniform-mixing noise | 88-105 |
| `_simulate_qlearning(...)` | M1 per-trial simulator | 108-150 |
| `_simulate_wmrl_family(...)` | M2/M3/M5/M6a/M6b per-trial simulator with kappa/kappa_s/phi_rl dispatch | 153-279 |
| `simulate_from_samples(model, params, stim, ss, rng)` | Public dispatch → qlearning or wmrl_family | 337-390 |
| `_extract_param_vector(...)` | Pluck `(draw, participant)` param dict (handles M6b kappa-stick-breaking decode) | 393-445 |
| `_unstack_participant_template(stacked)` | Reconstruct per-participant block list from stacked tensors | 453-500 |
| `_evaluate_gate(accuracies)` | Baribault & Collins (2023) 3-part gate | 503-540 |
| `_write_gate_report(...)` | Markdown PASS/FAIL report | 543-610 |
| `run_prior_ppc(model, data_path, num_draws, seed, output_dir)` | **Public: Baribault gate pipeline for stage 03** | 613-788 |
| `run_posterior_ppc(model, fitted_params_path, real_data_path, output_dir, figures_dir, ...)` | **Public: behavioral comparison + model recovery pipeline for stages 03 / 05** | 804-897 |

## Importer rewrites (per renamed module)

### plotting_utils → plotting (5 rewrites in 3 files)

| File | Line | Before → After |
|---|---|---|
| scripts/fitting/model_recovery.py | 53 | `from scripts.utils.plotting_utils import plot_behavioral_comparison` → `from scripts.utils.plotting import plot_behavioral_comparison` |
| scripts/fitting/model_recovery.py | 854 | `from scripts.utils.plotting_utils import plot_scatter_with_annotations` → `from scripts.utils.plotting import plot_scatter_with_annotations` |
| scripts/fitting/model_recovery.py | 959 | `from scripts.utils.plotting_utils import plot_kde_comparison` → `from scripts.utils.plotting import plot_kde_comparison` |
| scripts/06_fit_analyses/analyze_mle_by_trauma.py | 81 | `from utils.plotting_utils import (...)` → `from utils.plotting import (...)` |
| scripts/06_fit_analyses/regress_parameters_on_scales.py | 93 | `from utils.plotting_utils import (...)` → `from utils.plotting import (...)` |

### statistical_tests → stats (1 rewrite in 1 file)

| File | Line | Before → After |
|---|---|---|
| scripts/02_behav_analyses/08_run_statistical_analyses.py | 68 | `from scripts.utils.statistical_tests import (...)` → `from scripts.utils.stats import (...)` |

### scoring_functions → scoring (3 rewrites in 3 files + 2 docstring updates)

| File | Line | Before → After |
|---|---|---|
| scripts/01_data_preprocessing/01_parse_raw_data.py | 41 | `from scoring_functions import score_ies_r, score_less` → `from scoring import score_ies_r, score_less` |
| scripts/01_data_preprocessing/02_create_collated_csv.py | 22 | `from scoring_functions import calculate_all_task_metrics` → `from scoring import calculate_all_task_metrics` |
| scripts/01_data_preprocessing/04_create_summary_csv.py | 32 | `from scoring_functions import calculate_all_task_metrics, score_ies_r, score_less` → `from scoring import calculate_all_task_metrics, score_ies_r, score_less` |
| scripts/fitting/level2_design.py | 12 | docstring path `scripts/utils/scoring_functions.py::score_less()` → `scripts/utils/scoring.py::score_less()` |
| scripts/fitting/level2_design.py | 567 | markdown output `scripts/utils/scoring_functions.py::score_less()` → `scripts/utils/scoring.py::score_less()` |

## Decisions Made on Maintenance Scripts

All three one-off maintenance CLIs in `scripts/utils/` were evaluated per the 29-CONTEXT.md decision rule ("utils/ iff ≥ 2 stage folders import it"):

| File | Importers found | `def main()` present | Nature | Decision |
|---|---|---|---|---|
| `remap_mle_ids.py` | 0 | yes (guarded `if __name__ == "__main__"`) | One-time MLE-ID remap (legacy hash ↔ assigned_id reconciliation) | **Move to `scripts/_maintenance/`** |
| `sync_experiment_data.py` | 0 | yes | Safely copy new participant CSVs from experiment folder into analysis data | **Move to `scripts/_maintenance/`** |
| `update_participant_mapping.py` | 0 | yes | Scan `data/` and refresh `participant_id_mapping.json` | **Move to `scripts/_maintenance/`** |

All three are now at `scripts/_maintenance/`. A new `scripts/_maintenance/__init__.py` documents the folder's purpose. Their internal `project_root = Path(__file__).resolve().parents[2]` is still correct after the move (both old and new locations are 2 levels deep).

## Task Commits

Executed as a single atomic commit per the plan's Task 3 commit guidance:

1. **29-03 (all three tasks)** — `298f82d` (refactor: utils consolidation — canonical ppc/plotting/stats/scoring single-sources)

20 files changed, 946 insertions(+), 1031 deletions(-). Net deletion because `12_run_prior_predictive.py` lost 540 lines that moved into `scripts/utils/ppc.py` (and were re-exported via the thin orchestrator).

## Deviations from Plan

### Coordination: Parallel-Agent Index-Race Recovery

The plan was executed in Wave 2 in parallel with 29-04 (dead-folder audit). During Task 1/2 execution, the sibling 29-04 agent ran `git commit` on its own scope, which created a transient confusing state in which:

- My staged changes appeared in `git status` under 29-04's commit (via a since-amended SHA `f0f02bd`), then subsequently disappeared when 29-04 amended its commit to `e574fed` with only its own scope.
- My working-tree files remained intact throughout; only the index was disturbed.
- Recovery: re-staged all 29-03 files with explicit `git add` / `git rm` per-file and committed cleanly as `298f82d`.

No code or documentation changes were needed as a result of this race; only the staging workflow was reworked. Final commit is clean with atomic scope (20 files, all strictly in 29-03's territory).

### Rule 3 (blocking): none invoked
### Rule 1 (bug fix): none invoked
### Rule 2 (missing critical): none invoked
### Rule 4 (architectural): none invoked — `scripts/_maintenance/` creation is part of the plan (explicitly listed in `<files>` block)

Total deviations from plan content: **zero**. All work matches the plan as written.

## Deferred Deduplication

### `scripts/analysis/plotting_utils.py` — sibling duplicate flagged for 29-04

The file `scripts/analysis/plotting_utils.py` shares the same basename with the pre-rename `scripts/utils/plotting_utils.py` but is NOT a true duplicate:

| Concern | scripts/utils/plotting_utils.py → plotting.py (this plan) | scripts/analysis/plotting_utils.py (29-04 scope) |
|---|---|---|
| Top-level defs | `get_color_palette`, `add_colored_scatter`, `TRAUMA_GROUP_COLORS`, `plot_scatter_with_annotations`, `plot_kde_comparison`, `plot_behavioral_comparison` | `setup_plot_style`, `save_figure`, `aggregate_by_condition`, `plot_line_with_error`, `format_axes`, `get_color_palette` |
| Overlap | `get_color_palette` appears in both, with **different signatures** | |
| Live importers (pre-29-04) | stage 02, 06, scripts/fitting/ (now all rewritten to `scripts.utils.plotting`) | `scripts/simulations/visualize_parameter_sweeps.py`, `tests/legacy/examples/explore_prior_parameter_space.py` — both legacy code |

29-04 handled the deletion by archiving `scripts/analysis/` → `scripts/legacy/analysis/` (the legacy importers were rewritten to `scripts.legacy.simulations.*` / `scripts.legacy.analysis.*` accordingly). The two files now coexist but at clearly separated paths: the LIVE canonical helpers are at `scripts/utils/plotting.py` and the ARCHIVED helpers from the old analysis/ folder are at `scripts/legacy/analysis/plotting_utils.py`. No further dedup work is required — each file serves a clearly distinct caller audience.

### `scripts/01_data_preprocessing/{01,02,04}*.py` — bare-name sys.path imports

These three scripts use `sys.path.append(.../utils)` followed by `from scoring import ...` rather than `from scripts.utils.scoring import ...`. This pattern was preserved intentionally to minimise the 29-03 diff footprint (those files were last touched by Phase 28). A future tech-debt plan can migrate them to canonical fully-qualified imports. Not in scope for 29-03.

## Verification

All plan verification checks pass:

```bash
# Canonical single-source simulator
$ grep -rn "def simulate_from_samples\|def run_prior_ppc\|def run_posterior_ppc" scripts/ --include="*.py" | grep -v "scripts/utils/ppc.py" | grep -v "scripts/legacy/"
(empty — OK)

# Canonical names present, old names absent
$ for f in plotting stats scoring ppc; do test -f scripts/utils/$f.py && echo present; done
present × 4
$ for f in plotting_utils statistical_tests scoring_functions; do test ! -f scripts/utils/$f.py && echo absent; done
absent × 3

# No stale importers in live code
$ grep -rn "from scripts.utils.plotting_utils\|from scripts.utils.statistical_tests\|from scripts.utils.scoring_functions" scripts/ tests/ validation/ cluster/ manuscript/ docs/ src/ --exclude-dir=.planning --exclude-dir=legacy
(zero matches)

# Stage 05 posterior-PPC orchestrator present and wired to utils
$ test -f scripts/05_post_fitting_checks/run_posterior_ppc.py && echo present
present
$ grep -n "from scripts.utils.ppc" scripts/05_post_fitting_checks/run_posterior_ppc.py
30:from scripts.utils.ppc import run_posterior_ppc

# Package imports resolve
$ python -c "from scripts.utils import ppc, plotting, stats, scoring; print('ok')"
ok

# v4 closure guard
$ pytest scripts/fitting/tests/test_v4_closure.py -v
3 passed in 1.87s

# Rename history preserved
$ git log --follow --oneline scripts/utils/plotting.py | head -3
298f82d refactor(29-03): utils consolidation...
3a3a4ed feat(05-04): add plot_behavioral_comparison() to plotting_utils
b693523 feat(05-02): add visualization functions for parameter recovery
```

## Issues Encountered

- **Parallel-agent git index race with 29-04.** Both plans ran in Wave 2 concurrently. 29-04's first commit at SHA `f0f02bd` transiently picked up my staged utils/ changes (appeared in its commit stat). 29-04 subsequently amended to `e574fed` with only its own scope, leaving my work un-committed in the working tree. Resolved by re-staging all 29-03 files fresh and committing as `298f82d`. No data loss; no rework required.

## Next Plan Readiness

- Ready for **29-05** (cluster SLURM consolidation): all utils helpers are at canonical names, so any SLURM scripts that reference utils paths or set `PYTHONPATH` can assume `scripts/utils/{plotting,stats,scoring,ppc,data_cleaning}`.
- Ready for **29-06** (paper.qmd smoke render): no Quarto-facing breakage expected; the manuscript doesn't import from `scripts/utils/` directly.
- Ready for **29-07** (closure-guard extension): `scripts/utils/ppc.py` is a strong candidate for hash-locking in a future closure invariant, because any post-29-03 change to the simulator would need to maintain prior-PPC gate parity with `0cb1e2b` and behavioral comparison parity with existing MLE-PPC artifacts.
- Ready for **29-08** (`src/rlwm/fitting/` vertical refactor): `scripts.utils.ppc.simulate_from_samples` is a candidate for eventual promotion to `rlwm.fitting.ppc` once the numpy-layer / jax-layer distinction stabilises. Not urgent — current location is correct for the canonical-layout milestone.

---
*Phase: 29-pipeline-canonical-reorg*
*Plan: 03*
*Completed: 2026-04-22*
*Commit: 298f82d*
