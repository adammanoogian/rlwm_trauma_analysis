# scripts/legacy — Archive Audit (Phase 29-04)

Legacy/dead code moved out of the active pipeline during Phase 29 canonical
reorganization (Plan 29-04 — dead-folder audit). Contents listed here with
the grep evidence that informed each decision.

**Audit date:** 2026-04-22
**Executed by:** Plan 29-04 (dead-folder audit)
**Scope:** Five sibling folders Phase 28 left at the top level of `scripts/`:
`analysis/`, `results/`, `simulations/`, `statistical_analyses/`,
`visualization/`.

---

### Audit protocol

For every `.py` file inside each of the five candidate folders, a
live-reference sweep was run against the active tree (excluding
`.planning/`, `.git/`, and the folder itself):

```bash
grep -rn \
  "from scripts\.$FOLDER\.$name\|import scripts\.$FOLDER\.$name\|scripts/$FOLDER/$name\.py" \
  . --include="*.py" --include="*.sh" --include="*.slurm" \
     --include="*.md" --include="*.qmd" --include="*.tex" \
  --exclude-dir=.planning --exclude-dir=.git
```

Decision categories:

- **SALVAGE** — still imported by active pipeline → needs a new canonical
  home under 01–06 or `utils/`.
- **LEGACY-ARCHIVE** — referenced only by same-folder files, `.planning/`
  historical docs, or legacy test examples → move to `scripts/legacy/<folder>/`
  and rewrite the few importers that survive.
- **DELETE** — zero references anywhere (git history preserves content).

Because `git mv` preserves history and costs nothing beyond the move itself,
the default action in this plan was LEGACY-ARCHIVE (whole-folder) rather
than DELETE. This keeps the files auditable via `scripts/legacy/` until a
future milestone decides they are safe to fully remove.

---

## analysis/ — LEGACY-ARCHIVE (whole folder)

**File count:** 9 (8 `.py` + `__init__.py`).

| File | Live refs (active tree) | Decision |
|------|-------------------------|----------|
| `__init__.py` | 0 | LEGACY-ARCHIVE |
| `analysis_modelling_base_models.py` | 1 (self-ref from `restore_and_reconcile_fits.py` print message only) | LEGACY-ARCHIVE |
| `analyze_base_models.py` | 1 (own docstring usage line) | LEGACY-ARCHIVE |
| `compute_derived_behavioral_metrics.py` | 1 (self-ref print message) | LEGACY-ARCHIVE |
| `plotting_utils.py` | 2 (`scripts/simulations/visualize_parameter_sweeps.py:24`; `tests/legacy/examples/explore_prior_parameter_space.py:63`) — both in archived/legacy trees | LEGACY-ARCHIVE |
| `preliminary_parameter_behavior.py` | 0 | LEGACY-ARCHIVE |
| `preliminary_wmrl_descriptives.py` | 0 | LEGACY-ARCHIVE |
| `restore_and_reconcile_fits.py` | 0 | LEGACY-ARCHIVE |
| `trauma_scale_distributions.py` | 2 (`manuscript/paper.tex:244`; `manuscript/paper.qmd:171`) — **textual caption attribution only, not runtime imports** | LEGACY-ARCHIVE + update captions to legacy path |

**Extra note — stale shell-script references:** `run_data_pipeline.sh`
references four files in `scripts/analysis/` that **no longer exist on
disk**: `visualize_human_performance.py`, `visualize_scale_distributions.py`,
`visualize_scale_correlations.py`, `summarize_behavioral_data.py`. These are
pre-existing broken references from an earlier cleanup wave. Since the
script is already broken for those four calls (the files were deleted
before this phase started), moving the remaining `analysis/` folder to
`legacy/` does not worsen the situation. The broken block is left as-is
for a future tech-debt plan; documenting the fact here so it is visible.

**Action taken:** `git mv scripts/analysis scripts/legacy/analysis`. Two
manuscript caption paths in `paper.tex:244` and `paper.qmd:171` rewritten
from `scripts/analysis/trauma_scale_distributions.py` to
`scripts/legacy/analysis/trauma_scale_distributions.py` so the caption
attribution matches on-disk reality. Because the two cross-folder imports
(`visualize_parameter_sweeps.py`, `explore_prior_parameter_space.py`) are
themselves inside folders that are moving to legacy/ this same commit,
their imports are rewritten in-place to `from scripts.legacy.analysis.plotting_utils`.

---

## results/ — LEGACY-ARCHIVE (whole folder)

**File count:** 5 (no `__init__.py`).

| File | Live refs (active tree) | Decision |
|------|-------------------------|----------|
| `detailed_computational_results.py` | 0 | LEGACY-ARCHIVE |
| `format_computational_results.py` | 0 | LEGACY-ARCHIVE |
| `format_feedback_perseveration_results.py` | 0 | LEGACY-ARCHIVE |
| `summarize_all_results.py` | 0 | LEGACY-ARCHIVE |
| `verify_writeup.py` | 0 | LEGACY-ARCHIVE |

Zero external references anywhere outside `.planning/` historical docs.
Could be DELETE per the strict protocol; archiving to legacy/ instead keeps
the files available for a future "what did we try?" audit without cost.

**Action taken:** `git mv scripts/results scripts/legacy/results`. No importer rewrites needed.

---

## simulations/ — LEGACY-ARCHIVE (whole folder; multiple live importers rewritten)

**File count:** 5 (4 `.py` + `README.md` + `__init__.py`).

This folder is the ONLY one with live references from the active pipeline.
Per 29-CONTEXT.md §Audit §5, `unified_simulator.py` is still imported by
validation tests and stage-03 wrappers. The plan evaluated three options
(A = archive-and-rewrite, B = extract to `scripts/utils/`, C = extract to
`src/rlwm/simulators/`) and selected **Option A** (archive whole folder)
because:
1. All live callers are either tests or thin pass-through wrappers — the
   simulator is closer to a test fixture than a shared library helper.
2. The cheaper churn (rename 7 import sites vs. reshape the library layout)
   matches the dead-folder-audit scope better than a semantic relocation.
3. If a future milestone decides the simulator earns canonical status, a
   subsequent plan can `git mv` it out of `legacy/` then.

| File | Live refs (active tree) | Decision |
|------|-------------------------|----------|
| `__init__.py` | 0 | LEGACY-ARCHIVE |
| `README.md` | 0 (doc-internal example imports only) | LEGACY-ARCHIVE |
| `generate_data.py` | 1 (`scripts/03_model_prefitting/09_generate_synthetic_data.py:48`) | LEGACY-ARCHIVE + rewrite importer |
| `parameter_sweep.py` | 1 (`scripts/03_model_prefitting/10_run_parameter_sweep.py:57`) | LEGACY-ARCHIVE + rewrite importer |
| `unified_simulator.py` | 5 (`validation/test_unified_simulator.py:24`, `tests/test_wmrl_exploration.py:16`, `tests/legacy/examples/explore_prior_parameter_space.py:62`, `tests/legacy/examples/example_parameter_sweep.py:22`, `scripts/simulations/{generate_data,parameter_sweep}.py` internal) | LEGACY-ARCHIVE + rewrite importers |
| `visualize_parameter_sweeps.py` | 1 (`tests/legacy/examples/example_visualize_sweeps.py:19`) | LEGACY-ARCHIVE + rewrite importer |

**Action taken:** `git mv scripts/simulations scripts/legacy/simulations`.
Importers rewritten to `from scripts.legacy.simulations.<module>`:

1. `scripts/03_model_prefitting/09_generate_synthetic_data.py` (line 48)
2. `scripts/03_model_prefitting/10_run_parameter_sweep.py` (line 57)
3. `validation/test_unified_simulator.py` (line 24)
4. `tests/test_wmrl_exploration.py` (line 16)
5. `tests/legacy/examples/explore_prior_parameter_space.py` (line 62)
6. `tests/legacy/examples/example_parameter_sweep.py` (line 22)
7. `tests/legacy/examples/example_visualize_sweeps.py` (line 19)

The `from scripts.simulations.unified_simulator import ...` calls inside
`generate_data.py` and `parameter_sweep.py` themselves (i.e., imports
internal to the moved folder) are also rewritten to the new path so the
folder keeps working if ever re-imported.

---

## statistical_analyses/ — LEGACY-ARCHIVE (whole folder)

**File count:** 1.

| File | Live refs (active tree) | Decision |
|------|-------------------------|----------|
| `analyze_feedback_perseveration.py` | 0 | LEGACY-ARCHIVE |

Zero external references. Archiving rather than deleting for auditability.

**Action taken:** `git mv scripts/statistical_analyses scripts/legacy/statistical_analyses`. No importer rewrites.

---

## visualization/ — LEGACY-ARCHIVE (whole folder)

**File count:** 11.

| File | Live refs (active tree) | Decision |
|------|-------------------------|----------|
| `create_modeling_figures.py` | 0 | LEGACY-ARCHIVE |
| `create_modeling_tables.py` | 0 | LEGACY-ARCHIVE |
| `create_parameter_behavior_heatmap.py` | 0 | LEGACY-ARCHIVE |
| `create_publication_figures.py` | 0 | LEGACY-ARCHIVE |
| `create_supplementary_materials.py` | 0 | LEGACY-ARCHIVE |
| `create_supplementary_table_s3.py` | 0 | LEGACY-ARCHIVE |
| `plot_group_parameters.py` | 0 (only docs/02_pipeline_guide/PLOTTING_REFERENCE.md usage examples) | LEGACY-ARCHIVE |
| `plot_model_comparison.py` | 0 (same — docs only) | LEGACY-ARCHIVE |
| `plot_posterior_diagnostics.py` | 0 (own docstring usage line + docs/04_results) | LEGACY-ARCHIVE |
| `plot_wmrl_forest.py` | 0 | LEGACY-ARCHIVE |
| `quick_arviz_plots.py` | 0 (own docstring + docs) | LEGACY-ARCHIVE |

`manuscript/paper.qmd` does NOT invoke any of these at render time —
verified via grep. Paper figure panels are either pre-rendered PNGs under
`figures/` / `manuscript/figures/` or (per Phase 28) generated inline from
graceful-fallback `{python}` cells that depend on NetCDF outputs, not on
these one-off utilities.

**Action taken:** `git mv scripts/visualization scripts/legacy/visualization`.
No importer rewrites (no live Python imports exist). The four
`test_load_side_validation.py` enumeration entries that plan 29-01
commented out with a `TODO(29-04)` marker are restored in this plan to
point at `scripts/legacy/visualization/<file>.py`, and the similar entry
for `scripts/simulations/generate_data.py` is restored to
`scripts/legacy/simulations/generate_data.py`. Docs cross-references
(`docs/02_pipeline_guide/PLOTTING_REFERENCE.md`,
`docs/04_results/README.md`, `docs/01_project_protocol/plotting_config_guide.md`,
`docs/README.md`) are NOT updated in this plan — they are textual guides
describing legacy tooling; a separate docs-maintenance plan can re-annotate
them as historical. Leaving them in place is low-risk because they are
prose rather than executable paths.

---

### Post-move scripts/ top-level contents

After Plan 29-04 lands, `ls -d scripts/*/` yields:

```
scripts/01_data_preprocessing/
scripts/02_behav_analyses/
scripts/03_model_prefitting/
scripts/04_model_fitting/
scripts/05_post_fitting_checks/
scripts/06_fit_analyses/
scripts/fitting/          # library remnant (helpers, tests, bms, mle_utils)
scripts/legacy/           # this archive
scripts/utils/            # shared helpers
```

(plus optionally `scripts/_maintenance/` if plan 29-03 moves one-off
maintenance scripts there; 29-03 runs in parallel with 29-04 and the two
plans do not touch each other's directories.)

---

### Future maintenance

- If a file under `scripts/legacy/` is later found to be imported by an
  active-tree module, the v4 closure guard
  (`validation/check_v4_closure.py`) will flag it. The remediation is
  either (a) move the file out of `legacy/` into its proper canonical
  location, or (b) rewrite the importer to stop using it.
- Phase 30 (or a v6.0 cleanup plan) can decide to fully `git rm -r
  scripts/legacy/` once enough time has passed and no regressions
  surfaced. Git history preserves the content in either case.
