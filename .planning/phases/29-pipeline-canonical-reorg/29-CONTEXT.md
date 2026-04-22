# Phase 29 Context — Pipeline Canonical Reorganization & Cleanup

**Phase:** 29
**Slug:** 29-pipeline-canonical-reorg
**Milestone:** v5.0 Empirical Artifacts & Manuscript Finalization
**Sequencing:** Runs AFTER Phase 28 (closed 2026-04-22), BEFORE Phase 24 cold-start execution (paper.qmd path references must stabilize first).
**Origin:** 2026-04-22 user discussion — Phase 28 did initial grouping into 5 subdirs under time pressure, but did NOT implement canonical paper-ordered 01–06 stage layout, utils/ consolidation, dead-folder cleanup, docs/ integration, or `src/rlwm/fitting/` vertical refactor. This phase finishes what Phase 28 started.

---

## Scope — Three Coupled Workstreams

### 1) Scripts Reorganization (PRIMARY; execute FIRST per user)

Transition current grouping → canonical paper-directional 01–06 stage layout.

**Current state (post Phase 28):**
```
scripts/
├── 12_fit_mle.py                (top-level entry)
├── 13_fit_bayesian.py           (top-level entry)
├── 14_compare_models.py         (top-level entry)
├── data_processing/             # 01_–04_ numbered
├── behavioral/                  # 05_–08_ numbered
├── simulations_recovery/        # 09_–11_ numbered (09_run_ppc sits here)
├── post_mle/                    # 15_–18_ numbered
├── bayesian_pipeline/           # 21_* (9 steps)
├── fitting/                     # library: fit_mle/fit_bayesian implementations, mle_utils, bms, model_recovery, lba_likelihood, bayesian_diagnostics, etc.
├── analysis/                    # ← legacy cluster: 9 files (analyze_base_models.py, etc.)
├── results/                     # ← legacy cluster: 5 files
├── simulations/                 # ← legacy cluster: 5 files (parameter_sweep.py, generate_data.py, unified_simulator.py, visualize_parameter_sweeps.py)
├── statistical_analyses/        # ← legacy cluster: 1 file
├── utils/                       # utilities: data_cleaning, plotting_utils, scoring_functions, statistical_tests, sync_experiment_data, remap_mle_ids, update_participant_mapping
└── visualization/               # ← legacy cluster: 11 files (create_modeling_figures.py etc.)
```

**Target state (user-specified):**
```
scripts/
├── 01_data_preprocessing/       # ← current data_processing/ (01_parse_raw_data, 02_create_collated_csv, 03_create_task_trials_csv, 04_create_summary_csv)
├── 02_behav_analyses/           # ← current behavioral/ (05_summarize_behavioral_data, 06_visualize_task_performance, 07_analyze_trauma_groups, 08_run_statistical_analyses)
├── 03_model_prefitting/         # Simulation + prior predictive + recovery (before fitting real data)
│   ├── 09_generate_synthetic_data.py    # from simulations_recovery/
│   ├── 10_run_parameter_sweep.py        # from simulations_recovery/
│   ├── 11_run_model_recovery.py         # from simulations_recovery/
│   ├── 12_run_prior_predictive.py       # from bayesian_pipeline/21_run_prior_predictive.py
│   └── 13_run_bayesian_recovery.py      # from bayesian_pipeline/21_run_bayesian_recovery.py
├── 04_model_fitting/
│   ├── a_mle/                           # ← scripts/12_fit_mle.py entry + scripts/fitting/fit_mle.py impl
│   ├── b_bayesian/                      # ← scripts/13_fit_bayesian.py entry + scripts/fitting/fit_bayesian.py impl + bayesian_pipeline/21_fit_baseline.py
│   └── c_level2/                        # ← bayesian_pipeline/21_fit_with_l2.py
├── 05_post_fitting_checks/              # Convergence / validity / diagnostics (NOT substantive science)
│   ├── run_posterior_ppc.py             # orchestrates utils/ppc.py with posterior samples (mirror of 03's prior PPC)
│   ├── baseline_audit.py                # ← bayesian_pipeline/21_baseline_audit.py (Rhat, ESS, divergences)
│   └── scale_audit.py                   # ← bayesian_pipeline/21_scale_audit.py (orthogonalization check)
├── 06_fit_analyses/                     # Substantive results (what the fits say)
│   ├── compare_models.py                # ← scripts/14_compare_models.py
│   ├── compute_loo_stacking.py          # ← bayesian_pipeline/21_compute_loo_stacking.py
│   ├── model_averaging.py               # ← bayesian_pipeline/21_model_averaging.py
│   ├── analyze_mle_by_trauma.py         # ← post_mle/15_analyze_mle_by_trauma.py
│   ├── regress_parameters_on_scales.py  # ← post_mle/16_regress_parameters_on_scales.py
│   ├── analyze_winner_heterogeneity.py  # ← post_mle/17_analyze_winner_heterogeneity.py
│   ├── bayesian_level2_effects.py       # ← post_mle/18_bayesian_level2_effects.py
│   └── manuscript_tables.py             # ← bayesian_pipeline/21_manuscript_tables.py
└── utils/                               # SHARED library (imported, not run directly)
    ├── ppc.py                           # the simulator — used by 03 prior PPC and 05 posterior PPC
    ├── plotting.py                      # consolidated plot helpers (current utils/plotting_utils.py)
    ├── stats.py                         # ← current utils/statistical_tests.py
    ├── scoring.py                       # ← current utils/scoring_functions.py
    └── data_cleaning.py                 # ← current utils/data_cleaning.py
```

**Key principle:** Any code used by TWO OR MORE stages lives in `utils/`. Never duplicate code across stage folders. `run_ppc` is shared by 03 (prior) and 05 (posterior) — the simulator is ONE `utils/ppc.py`, invoked from two thin orchestrator scripts.

**Design intent (from joint discussion):**
- Numbered stages map 1:1 to paper IMRaD sections: 01/02 = Methods/Descriptives, 03 = simulation studies (supplement), 04 = main fits, 05 = convergence/validity (supplement), 06 = Results.
- Letter sub-partition `04/{a,b,c}` captures "same stage, different method" — MLE vs Bayesian are parallel-alternative paths, Level-2 depends on Bayesian baseline.
- Model comparison (14_compare_models), LOO stacking, model averaging go in **06** (results), NOT **05** (checks). 05 is "did it converge?" 06 is "what does it mean?"

---

### 2) Utils Organization & Dead-Folder Audit (execute SECOND per user)

Five current sibling folders that Phase 28 did NOT fold in — must audit each before delete/move:

| Folder | Files | Known external references (grep evidence) |
|--------|-------|---------------------------------------------|
| `scripts/analysis/` | 9 (analyze_base_models.py, restore_and_reconcile_fits.py, compute_derived_behavioral_metrics.py, trauma_scale_distributions.py, preliminary_parameter_behavior.py, preliminary_wmrl_descriptives.py, analysis_modelling_base_models.py, plotting_utils.py, __init__.py) | Referenced only from own files + .planning/ docs — **likely fully dead** |
| `scripts/results/` | 5 (detailed_computational_results.py, format_computational_results.py, format_feedback_perseveration_results.py, summarize_all_results.py, verify_writeup.py) | No external grep hits — **likely fully dead** |
| `scripts/simulations/` | 5 (generate_data.py, parameter_sweep.py, unified_simulator.py, visualize_parameter_sweeps.py, README.md) | Referenced by `validation/test_unified_simulator.py`, `tests/test_wmrl_exploration.py`, legacy examples — **NOT fully dead; requires careful audit** |
| `scripts/statistical_analyses/` | 1 (analyze_feedback_perseveration.py) | No external hits — **likely dead** |
| `scripts/visualization/` | 11 (create_modeling_figures.py, create_publication_figures.py, create_supplementary_materials.py, plot_group_parameters.py, plot_model_comparison.py, plot_posterior_diagnostics.py, plot_wmrl_forest.py, quick_arviz_plots.py, create_modeling_tables.py, create_parameter_behavior_heatmap.py, create_supplementary_table_s3.py) | Referenced only from docs/ (PLOTTING_REFERENCE.md) — **likely dead but high-value figures; SALVAGE what's used** |

**Audit protocol per folder:**
1. For every `.py` in folder, run `grep -rn "from scripts\.<folder>\." src/ scripts/ cluster/ tests/ validation/ manuscript/` — direct imports.
2. Run `grep -rn "scripts/<folder>/<file>" cluster/ *.sh` — shell script invocations.
3. Run `grep -rn "scripts\.<folder>\.<module>" manuscript/paper.qmd` — Quarto executable cells.
4. If ZERO live references in the ACTIVE pipeline (ignore `.planning/` historical docs) → move entire folder to `legacy/` (or delete if git history preserves it).
5. If references exist → extract the still-used file(s) into the canonical new location (01–06 or utils/), update importers, then handle the remainder per rule 4.

**Utils consolidation (PR to finalize):**
- `scripts/utils/plotting_utils.py` + `scripts/analysis/plotting_utils.py` → single `scripts/utils/plotting.py` (deduplicate).
- Any helper imported by ≥ 2 stage folders goes into `scripts/utils/`.
- `scripts/utils/remap_mle_ids.py`, `sync_experiment_data.py`, `update_participant_mapping.py` are one-off maintenance scripts — decide: keep in utils/ OR move to `scripts/_maintenance/` (judgment call; planner may resolve).

---

### 3) `docs/` Spare-File Integration (lowest-risk; can parallelize)

| File | Size | Target destination |
|------|------|---------------------|
| `docs/CLUSTER_GPU_LESSONS.md` | 34 KB | **LEAVE IN PLACE** (per user — cluster tuning still active) |
| `docs/HIERARCHICAL_BAYESIAN.md` | 10 KB | Merge → `docs/04_methods/README.md` (Bayesian fitting section) |
| `docs/K_PARAMETERIZATION.md` | 7.6 KB | Merge → `docs/03_methods_reference/MODEL_REFERENCE.md` (parameter section) |
| `docs/PARALLEL_SCAN_LIKELIHOOD.md` | 18 KB | Keep alongside `03_methods_reference/MODEL_REFERENCE.md` as a technical companion (18 KB is too much for a subsection; 1:1 implementation reference) |
| `docs/SCALES_AND_FITTING_AUDIT.md` | 13 KB | Merge → `docs/04_methods/README.md` (scales orthogonalization subsection) |

**Protocol per merge:**
1. Copy content into target doc at appropriate section boundary.
2. Update any docs/ cross-references to point at new location.
3. Move original file to `docs/legacy/` (preserves history; allows rollback).
4. Grep repo for references to the old filename — update or note "see legacy/" stubs.

---

### 4) `src/rlwm/fitting/` Vertical Refactor (OPTIONAL; judgment call)

**Current:**
- `jax_likelihoods.py` = 6,113 lines (core JAX helpers + ~5 likelihood variants × 7 models + inline tests)
- `numpyro_models.py` = 2,722 lines (per-model numpyro priors + wrapper that calls `*_likelihood` from jax_likelihoods)
- `numpyro_helpers.py` = 308 lines

**Design concern:** Adding a new model requires edits in BOTH files (horizontal split = by type of code, not by model). Works now but navigability suffers at this scale.

**Proposed vertical target:**
```
src/rlwm/fitting/
├── core.py              # ~300 lines: padding, softmax, epsilon, scan primitives
├── models/
│   ├── qlearning.py     # M1: all likelihood variants + numpyro wrapper
│   ├── wmrl.py          # M2
│   ├── wmrl_m3.py       # M3
│   ├── wmrl_m5.py       # M5
│   ├── wmrl_m6a.py      # M6a
│   ├── wmrl_m6b.py      # M6b (+ subscale variant)
│   └── wmrl_m4.py       # M4 LBA
└── sampling.py          # run_inference, samples_to_arviz, chain selector
```

**Risk:** v4.0 closure guards depend on specific import paths (`from rlwm.fitting.jax_likelihoods import ...`). A refactor MUST either (a) preserve old import paths via re-exports, or (b) update closure guards and every caller simultaneously. Option (a) is safer but leaves dead re-export stubs.

**Recommendation to planner:** Include as a final plan gated on user approval. If v5.0 is the last milestone touching model math, this refactor can defer to v6.0; current layout is ugly but not broken. Don't block earlier plans on this decision.

---

## Cluster/SLURM Knock-On Updates (MUST be in this phase — blocking)

Current `cluster/*.slurm` files reference old script paths (e.g., `scripts/bayesian_pipeline/21_fit_baseline.py`, `scripts/12_fit_mle.py`). After reorg, these paths change. Cluster scripts are load-bearing (Phase 24 cold-start will invoke them).

**Target shape (from user spec):**
```
cluster/
├── 01_data_processing.slurm
├── 02_behav_analyses.slurm
├── 03_prefitting_{cpu,gpu}.slurm
├── 04a_mle_{cpu,gpu}.slurm        # --export=MODEL=wmrl_m3,...
├── 04b_bayesian_{cpu,gpu}.slurm   # --export=MODEL=...,TIME=...
├── 04c_level2.slurm
├── 05_post_checks.slurm
├── 06_fit_analyses.slurm
└── submit_all.sh                  # chains via --afterok (extend pattern from 21_submit_pipeline.sh)
```

**Files to update (from current cluster/ listing):**
- 35+ existing SLURM files — many are per-model variants of the same job that must consolidate via `--export=MODEL=...`.
- The `21_*.slurm` series (9 steps) — update internal `python scripts/bayesian_pipeline/21_*.py` → `python scripts/05_post_fitting_checks/*.py` etc.
- `21_submit_pipeline.sh` — update afterok chain to new slurm names.
- `12_mle*.slurm`, `13_bayesian_*.slurm` — consolidate.

---

## Paper.qmd Coupling (MUST be checked)

`manuscript/paper.qmd` has Quarto `{python}` inline cells that reference script modules. Grep earlier showed `paper.qmd` contains references to `scripts/` paths. Any reorg must:
1. Grep `manuscript/paper.qmd` + `manuscript/paper.tex` for `scripts/` path references.
2. Update inline `{python}` cells to new import paths.
3. Verify `quarto render paper.qmd` still succeeds (graceful-fallback cells per Phase 28 should absorb some breakage).

---

## Success Criteria (for planner to refine)

1. `scripts/` top level contains ONLY: `01_data_preprocessing/`, `02_behav_analyses/`, `03_model_prefitting/`, `04_model_fitting/`, `05_post_fitting_checks/`, `06_fit_analyses/`, `utils/`, and a small `fitting/` library remnant for anything too heavy to move (OR `fitting/` is folded into `src/rlwm/fitting/`). No other top-level folders.
2. Dead folders (`analysis/`, `results/`, `simulations/`, `statistical_analyses/`, `visualization/`) are EITHER deleted OR moved to `scripts/legacy/` with an audit record in the phase summary listing what was salvaged vs. dropped.
3. Every function used by ≥ 2 stage folders lives in `scripts/utils/`. `grep -rn "def run_ppc\|def run_posterior_predictive" scripts/` shows the definition lives in `utils/`, not duplicated.
4. Every file in 03 and 05 that invokes PPC imports from `scripts/utils/ppc.py` — NOT from each other.
5. `docs/HIERARCHICAL_BAYESIAN.md`, `K_PARAMETERIZATION.md`, `SCALES_AND_FITTING_AUDIT.md` no longer exist at top level of `docs/` — content is merged into `03_methods_reference/` or `04_methods/`, and original files sit under `docs/legacy/`.
6. `docs/CLUSTER_GPU_LESSONS.md` is untouched (user directive).
7. `docs/PARALLEL_SCAN_LIKELIHOOD.md` either remains at top level or is moved to `03_methods_reference/` alongside `MODEL_REFERENCE.md` (either is acceptable).
8. `cluster/` SLURM files reference new script paths; `bash cluster/submit_all.sh --dry-run` (or equivalent smoke check) passes path validation for every stage.
9. `manuscript/paper.qmd` renders via `quarto render` without path-not-found errors (graceful-fallback cells catch anything missing).
10. All v4.0 closure invariants (`validation/check_v4_closure.py`) still pass after reorg.
11. Every moved Python file has its importers updated — `grep -rn "from scripts.data_processing" src/ scripts/ tests/ validation/ manuscript/` returns ZERO matches post-reorg (all use new paths).
12. `pytest` on `scripts/fitting/tests/` + `tests/` + `validation/` passes clean.
13. Phase closure pytest test (`tests/test_v5_phase29_structure.py` or equivalent) asserts the canonical directory shape exists and dead folders are gone.

---

## Execution Order (user-specified)

1. **Scripts reorganization** — do this FIRST. Big rename wave. Update all importers in one pass.
2. **Utils consolidation + dead folder audit** — AFTER reorg so we can see what's truly shared/dead from the new vantage point.
3. **Cluster SLURM updates** — AFTER scripts are in place (paths must be stable).
4. **Docs merges** — can PARALLELIZE with cluster updates (independent blast radius).
5. **Paper.qmd reference updates + smoke render** — after scripts/cluster stable.
6. **`src/rlwm/fitting/` vertical refactor** — OPTIONAL, judgment call, defer unless clearly worth it.
7. **Closure guard extension** — final plan; adds pytest invariant test for new structure.

---

## Known Risks & Coupling Points

1. **Import path breakage.** Every Python file under moved folders has relative/absolute imports that will break. `grep` sweep + atomic commit per stage.
2. **SLURM path drift.** Cluster scripts are consumed by Phase 24 (cold-start). If paths are wrong, Phase 24 re-run fails on cluster. Must `--dry-run` all SLURMs after reorg.
3. **Paper.qmd inline refs.** Phase 28 introduced Quarto `{python}` cells with graceful fallback — good. But any `from scripts.X import Y` in those cells breaks silently (fallback masks it). Must explicitly grep.
4. **Closure guards.** `validation/check_v4_closure.py` + new Phase 25 guard reference specific paths; update with atomic commit alongside reorg.
5. **`scripts/simulations/` is NOT fully dead** — `validation/test_unified_simulator.py` imports from it. Either extract the test-relevant bits into canonical location, update the test, OR preserve a `scripts/legacy/simulations/` path.
6. **Memory of Phase 28 says "src/rlwm/fitting/ is now authoritative for jax_likelihoods/numpyro_models/numpyro_helpers (narrow migration); environments/ + models/ shims deleted"** — verify this is already done; don't re-do it. The remaining src/ work is ONLY the optional vertical refactor of `src/rlwm/fitting/` itself.
7. **Phase 24 (cold-start) has NOT run yet per ROADMAP** — so no real output artifacts to break. But pipeline paths must be right when it does run.

---

## Tentative Plan Breakdown (planner may refine)

- **29-01** (Wave 1): Scripts reorganization — rename/move 5 sibling groups into 01–06 stage folders; update all internal Python importers; atomic commit per stage. Keeps `scripts/fitting/` untouched for now.
- **29-02** (Wave 1, parallel with 29-01): `docs/` merges — low blast radius, can proceed independently.
- **29-03** (Wave 2, after 29-01): Utils consolidation — move shared helpers (plotting, stats, scoring, ppc) into `scripts/utils/`; deduplicate `scripts/analysis/plotting_utils.py` vs `scripts/utils/plotting_utils.py`; create `scripts/utils/ppc.py` as canonical simulator.
- **29-04** (Wave 2, after 29-01): Dead folder audit — per-folder grep protocol, salvage references, move to `scripts/legacy/` (or delete if fully dead).
- **29-05** (Wave 3, after 29-01 & 29-03): Cluster SLURM consolidation — parameterize per-model slurms via `--export=MODEL=...`, update all internal paths, provide stage-numbered entry points, dry-run verify.
- **29-06** (Wave 3, after 29-01): `manuscript/paper.qmd` + `paper.tex` path updates — grep & fix Quarto inline cell references.
- **29-07** (Wave 4, after all others): Closure guard extension — pytest test for new directory shape + dead folder absence + canonical utils usage.
- **29-08** (OPTIONAL, Wave 5): `src/rlwm/fitting/` vertical refactor — gated on user approval at plan-check time. If approved, split per-model; preserve import paths via re-exports.

---

## Evidence Snapshot (already gathered in prior conversation)

- `src/rlwm/fitting/jax_likelihoods.py` — 6,113 lines; contains JAX helpers + 7 models × ~5 likelihood variants + inline tests.
- `src/rlwm/fitting/numpyro_models.py` — 2,722 lines; per-model numpyro priors + factor wrapper.
- `scripts/fitting/` — 13 files totaling ~20,000 lines (fit_mle.py = 3,157; mle_utils.py = 1,424; model_recovery.py = 1,612; bayesian_diagnostics.py = 905; fit_bayesian.py = 1,173; bms.py = 335; compare_mle_models.py = 454; lba_likelihood.py = 769; level2_design.py = 649; bayesian_summary_writer.py = 382; warmup_jit.py = 284; aggregate_permutation_results.py = 213).
- `cluster/` — 35+ SLURM files; per-model variants of 12_mle, 13_bayesian already partly consolidated in Phase 28 (`13_bayesian_choice_only.slurm` is the pattern).
- `scripts/analysis/` — 9 files, no live imports from rest of active pipeline.
- `scripts/simulations/` — references in `validation/` and `tests/` (NOT fully dead).
- `scripts/visualization/` — 11 files, referenced only in `.planning/` docs.

---

## Downstream Consumers

- **Phase 24** (cold-start pipeline execution) — MUST see correct script paths.
- **Phase 26** (manuscript finalization) — MUST see stable paper.qmd inline refs.
- **Phase 27** (v5.0 closure) — MUST see closure guards passing on new structure.
