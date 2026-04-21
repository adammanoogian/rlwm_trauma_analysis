# Phase 18: Integration, Comparison, and Manuscript - Research

**Researched:** 2026-04-13
**Domain:** Python pipeline migration (argparse flag flip), ArviZ LOO/stacking comparison, matplotlib reliability scatterplots, Quarto manuscript revision
**Confidence:** HIGH (all claims verified against codebase or ArviZ 0.22.0 live API)

---

## Summary

Phase 18 integrates Phases 15-17 hierarchical Bayesian fits into the existing
downstream pipeline via a `--source mle|bayesian` flag on scripts 15, 16, 17.
The schema-parity CSV produced by `bayesian_summary_writer.py` means the
migration is a path-routing change with no analysis-logic rewrite.

The Bayesian model comparison infrastructure is already partially built:
`scripts/14_compare_models.py` already has `--bayesian-comparison` mode with
`run_bayesian_comparison()`, `_load_bayesian_compare_dict()`, and
`_pareto_k_summary()` implemented and wired to the `main()` entry point.
The NetCDF map `BAYESIAN_NETCDF_MAP` is already defined at module scope.
What is missing is the M4 separate-track section and the output CSV (only a
Markdown file is currently written).

Scripts 15 and 16 have no `--source` flag — data loading is unconditionally
from `output/mle/`. Script 17 similarly hard-codes the MLE directory.
Script 16b has no deprecation notice yet.

The manuscript (`manuscript/paper.qmd`) still describes PyMC-based Bayesian
regression and MLE-centric model fitting; the Methods section references PyMC
explicitly and the Bayesian regression section (`#sec-bayesian-regression`)
needs to be replaced with the Level-2 hierarchical narrative.
`docs/03_methods_reference/MODEL_REFERENCE.md` exists but has no Hierarchical
Bayesian Pipeline section.

**Primary recommendation:** Implement tasks in this order — MIG-01/02/03
(flag plumbing, lowest coupling risk) → MIG-04 (deprecation header) →
MIG-05 (reliability scatterplots) → CMP-01/02/03/04 (extend script 14) →
DOC-01 (MODEL_REFERENCE) → DOC-02/03/04 (manuscript).

---

## Standard Stack

The established libraries for this phase:

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| arviz | 0.22.0 (installed) | LOO, WAIC, az.compare, stacking weights | Already used in script 14; confirmed live API |
| matplotlib | installed | Reliability scatterplots | Already used throughout pipeline |
| pandas | installed | CSV I/O, schema-parity column manipulation | Already used throughout |
| argparse | stdlib | --source flag in scripts 15/16/17 | Already used in all three scripts |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| numpy | installed | 45-degree reference line, HDI spread computation | Scatterplot geometry |
| seaborn | installed | Consistent plot styling | All new figures |
| quarto | installed | Compile paper.qmd to PDF | Manuscript rebuild after edits |

**No new packages needed.** All libraries are already in the environment.

---

## Architecture Patterns

### Pattern 1: --source Flag (MIG-01, MIG-02, MIG-03)

**What:** Add `--source mle|bayesian` to `argparse` in scripts 15, 16, 17.
Path resolution changes only; analysis functions stay untouched.

**When to use:** Wherever data loading resolves `output/mle/<model>_individual_fits.csv`.

**Implementation for script 15:**

```python
# Source: codebase inspection of scripts/15_analyze_mle_by_trauma.py
parser.add_argument(
    '--source',
    type=str,
    default='mle',
    choices=['mle', 'bayesian'],
    help='Fit source: mle (default) or bayesian (reads output/bayesian/*_individual_fits.csv)',
)
```

Then in `load_data()` (or a refactored `_resolve_fits_dir(source)` helper):

```python
def _resolve_fits_dir(source: str, project_root: Path) -> Path:
    if source == 'bayesian':
        return project_root / 'output' / 'bayesian'
    return project_root / 'output' / 'mle'
```

Script 15's `load_data()` hard-codes `OUTPUT_DIR = PROJECT_ROOT / "output" / "mle"`.
The fix is to pass `fits_dir` as a parameter rather than using the module-level constant.

Script 16's `load_integrated_data()` takes `params_path: Path` directly —
the CLI just needs to construct the right path from `--source`.

Script 17's `load_per_participant_aic()` takes `mle_dir: Path` already —
add a `--source` flag and pass the resolved dir.

**Key constraint:** When `--source bayesian`, the Bayesian CSV has extra
columns (`_hdi_low`, `_hdi_high`, `_sd`, `max_rhat`, `min_ess_bulk`,
`num_divergences`) beyond the MLE schema. Analysis logic must not break on
unknown columns — pandas column slicing by name already handles this safely.
The schema-parity core columns (participant_id, param names, nll, aic, bic)
are identical between sources.

**Output path routing:** When `--source bayesian`, write results to
`output/bayesian/analysis/<model>_*` and `figures/bayesian/<model>_*` to
avoid clobbering MLE outputs. This is a new convention for Phase 18.

### Pattern 2: MLE-vs-Bayesian Reliability Scatterplot (MIG-05)

**What:** One scatter plot per (parameter, model) cell. X-axis = MLE point
estimate, Y-axis = Bayesian posterior mean. 45-degree reference line.
Shrinkage direction highlighted for M6b (posterior mean pulled toward group
mean relative to MLE).

**Output path:** `output/bayesian/figures/mle_vs_bayes/{model}_{param}.png`

**Implementation pattern (standalone function):**

```python
def plot_mle_vs_bayes_reliability(
    mle_df: pd.DataFrame,
    bayes_df: pd.DataFrame,
    param: str,
    model_name: str,
    output_dir: Path,
    *,
    highlight_shrinkage: bool = False,
) -> Path:
    """One scatter per parameter. Source: Phase 18 design."""
    fig, ax = plt.subplots(figsize=(4, 4))
    x = mle_df[param].values
    y = bayes_df[param].values  # posterior mean
    ax.scatter(x, y, alpha=0.5, s=20)
    # 45-degree reference
    lim = [min(x.min(), y.min()), max(x.max(), y.max())]
    ax.plot(lim, lim, 'k--', lw=1, alpha=0.6)
    if highlight_shrinkage:
        # draw arrows from MLE to posterior mean for each participant
        for xi, yi in zip(x, y):
            ax.annotate('', xy=(yi, yi), xytext=(xi, xi),
                        arrowprops=dict(arrowstyle='->', color='steelblue', lw=0.5))
    ax.set_xlabel(f'MLE {param}')
    ax.set_ylabel(f'Posterior mean {param}')
    ax.set_title(f'{model_name}: {param}')
    out = output_dir / f'{model_name}_{param}.png'
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    return out
```

This can be a helper in `scripts/fitting/plot_reliability.py` or inline in
a new `scripts/18b_mle_vs_bayes_reliability.py`. The planner should create a
dedicated helper script rather than stuffing this into script 17 (it does not
depend on winner heterogeneity analysis).

### Pattern 3: Bayesian Comparison Extension (CMP-01..04)

**What:** Extend `run_bayesian_comparison()` in `scripts/14_compare_models.py`
to write a CSV stacking-weight table and add an M4 separate-track section.

**Current state:**
- `run_bayesian_comparison()` exists at line 639
- `BAYESIAN_NETCDF_MAP` covers M1-M6b (6 choice-only)
- Writes only `stacking_weights.md` (Markdown)
- No M4 separate track
- No output CSV

**What needs adding:**

1. Write `output/bayesian/level2/stacking_weights.csv` (machine-readable)
   alongside the existing Markdown file.
2. Add M4 separate-track block using M4 posterior at
   `output/bayesian/wmrl_m4_posterior.nc` with Pareto-k gating:

```python
# M4 separate track (choice-only marginal fallback)
M4_NETCDF = "output/bayesian/wmrl_m4_posterior.nc"
m4_path = project_root / M4_NETCDF
if m4_path.exists():
    idata_m4 = az.from_netcdf(str(m4_path))
    if hasattr(idata_m4, "log_likelihood"):
        loo_m4 = az.loo(idata_m4, pointwise=True)
        k_pct_m4 = float(np.mean(loo_m4.pareto_k.values > 0.7) * 100)
        print(f"\nM4 separate track: Pareto-k > 0.7: {k_pct_m4:.1f}%")
        if k_pct_m4 > 10:
            print("  => Using choice-only marginal NLL fallback (Pareto-k unreliable)")
        # Report M4 elpd_loo with warning — do NOT include in az.compare dict
```

3. `az.compare` signature (verified against arviz 0.22.0):
   `az.compare(compare_dict, ic='loo', method='stacking')` — already correct
   in existing code. Returns DataFrame with columns: rank, elpd, pIC,
   elpd_diff, weight, se, dse, warning, scale.

### Pattern 4: Deprecation Header for 16b (MIG-04)

**What:** Add module-level deprecation docstring. The file is already NumPyro-only
(PyMC removed in INFRA-07).

```python
"""
16b: Bayesian Linear Regression (Parameters -> Scale Totals)
=============================================================

.. deprecated::
    This script is superseded by the Level-2 hierarchical pipeline implemented
    in Phase 16 (scripts/13_fit_bayesian.py with --subscale, and
    scripts/18_bayesian_level2_effects.py). The Level-2 approach jointly
    estimates individual parameters and trauma associations in a single
    inference pass, providing proper uncertainty propagation from the
    participant level to the regression level.

    This script is retained as a fast-preview / supplementary path that
    runs a post-hoc NumPyro regression on MLE point estimates. It does NOT
    require MCMC convergence from Phase 15-17 fits and produces approximate
    directional evidence suitable for exploratory analysis.

    Use this script for: quick sanity-check before running the full pipeline.
    Use scripts/18_bayesian_level2_effects.py for: manuscript-quality results.

[existing docstring content continues below]
"""
```

### Pattern 5: MODEL_REFERENCE.md Hierarchical Bayesian Section (DOC-01)

**What:** Append a new section to `docs/03_methods_reference/MODEL_REFERENCE.md`.
Cross-reference `docs/K_PARAMETERIZATION.md`.

**Section structure:**

```markdown
## 8. Hierarchical Bayesian Pipeline (v4.0)

### 8.1 Non-Centered Parameterization
[mu_pr ~ Normal(0,1), sigma_pr ~ HalfNormal(0.2), theta = lower + (upper-lower)*Phi_approx(mu_pr + sigma_pr*z)]

### 8.2 Level-2 Regression Structure
[4-predictor design, beta sites, build_level2_design_matrix]

### 8.3 NumPyro Factor Pattern and Pointwise Log-Likelihood
[numpyro.factor, compute_pointwise_log_lik, bayesian_diagnostics.py]

### 8.4 WAIC/LOO Workflow
[az.loo, az.compare, stacking weights, Pareto-k gating for M4]

### 8.5 Schema-Parity CSV
[bayesian_summary_writer.py, --source flag pattern]
```

### Pattern 6: Manuscript Revision (DOC-02/03/04)

**What:** Edit `manuscript/paper.qmd` in three sections.

**Methods section changes (DOC-02):**
- `#sec-fitting`: Replace L-BFGS-B / MLE / AIC narrative with NumPyro NUTS
  hierarchical inference. Keep MLE as "primary frequentist comparison";
  add Bayesian as primary inference path.
- `#sec-stats`: Replace PyMC reference with NumPyro. Replace
  "Bayesian multivariate regressions" PyMC block with Level-2 hierarchical
  joint regression description.

**Results section changes (DOC-03):**
- `#sec-model-comparison` (line 542): Add stacking-weight table from
  `output/bayesian/level2/stacking_weights.csv`.
- `#sec-bayesian-regression` (line 892): Replace v3.0 FDR post-hoc regression
  narrative with Level-2 posterior forest plots loaded from
  `output/bayesian/level2/`.

**Limitations section:** No existing limitations section — add one before
`#sec-discussion` or as a subsection of Discussion. Content: Pareto-k M4
fallback, residual K identifiability, M6b shrinkage diagnostics.

### Anti-Patterns to Avoid

- **Rewriting analysis logic in scripts 15/16/17:** Only path resolution
  changes. If any function signature change is needed, the smell is wrong.
- **Including M4 in az.compare dict:** M4 joint likelihood is incommensurable
  with choice-only log-likelihoods. M4 gets its own separate print block.
- **Mixing mle and bayesian output directories:** Use distinct output paths
  when `--source bayesian` to avoid overwriting MLE artifacts.
- **Filtering masked padding in LOO:** The `compute_pointwise_log_lik` function
  in `bayesian_diagnostics.py` assigns `log_prob=0.0` to padded positions.
  Padding positions MUST be filtered before calling `az.waic()` / `az.loo()`
  (confirmed in 13-03 decisions). The `build_inference_data_with_loglik()`
  function must handle this; verify before CMP-01 runs.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Stacking weights computation | Custom weight optimizer | `az.compare(method='stacking')` | Dirichlet optimization correctly implemented, handles numerical stability |
| LOO / WAIC calculation | Custom cross-validation | `az.loo()`, `az.waic()` | Pareto smoothing handles importance weights, already in codebase |
| Pareto-k threshold checking | Custom threshold logic | `az.loo(pointwise=True).pareto_k` | Already used in `_pareto_k_summary()` in script 14 |
| Schema-parity column mapping | Custom CSV writer | `bayesian_summary_writer.write_bayesian_summary()` | Already built in INFRA-04, tested, schema locked |
| Posterior mean extraction | Custom arviz.posterior parse | `az.summary(idata, var_names=params)['mean']` | Handles chain collapsing correctly |

**Key insight:** The schema-parity pattern was designed precisely so scripts
15/16/17 require zero analysis-logic changes. Any temptation to rewrite
statistical logic during this phase is scope creep.

---

## Common Pitfalls

### Pitfall 1: Padding Zeros Inflate WAIC/LOO
**What goes wrong:** `az.loo()` or `az.waic()` is called on an InferenceData
that includes `log_prob=0.0` for padded trial positions. Zero log-prob
contributions inflate the effective parameter count (`pIC`) and deflate the
ELPD estimate, making models with more padding look artificially better.
**Why it happens:** The stacked likelihood returns a flat `(n_blocks * MAX_TRIALS)` array with 0.0 at mask==0 positions (13-03 locked decision).
**How to avoid:** `build_inference_data_with_loglik()` must filter mask==0
positions before constructing the log-likelihood group. Verify this is
implemented before running CMP-01. If not, add a mask-aware reshape step.
**Warning signs:** `pIC` values larger than the number of free parameters by
a factor >2; ELPD values unusually high for simple models.

### Pitfall 2: Bayesian Output Overwrites MLE Output
**What goes wrong:** If `--source bayesian` writes to `output/mle/` (the
default for script 15) it silently clobbers the MLE group-comparison CSVs.
**Why it happens:** Scripts 15/16/17 currently have `OUTPUT_DIR = ... / "output" / "mle"` hard-coded at module scope.
**How to avoid:** Route Bayesian outputs to `output/bayesian/analysis/`
whenever `--source bayesian` is active. Add a `_resolve_output_dir(source)`
helper that scripts 15/16/17 all call.

### Pitfall 3: Script 15 `load_data()` is Hard to Parametrize
**What goes wrong:** `load_data()` in script 15 loads all models unconditionally from `OUTPUT_DIR` (module-level constant). Adding `--source` requires passing the directory into `load_data()`.
**Why it happens:** Module-level constant design.
**How to avoid:** Refactor `load_data()` to accept `fits_dir: Path` as the first argument. Keep backward compat by defaulting to `OUTPUT_DIR`. The analysis path `main()` passes the resolved directory.

### Pitfall 4: arviz 0.22.0 `az.compare` Column Names
**What goes wrong:** Code expects `'elpd_loo'` column but gets `'elpd'` (or vice versa) depending on arviz version.
**Why it happens:** Column names changed between arviz versions.
**How to avoid:** arviz 0.22.0 (installed version) uses `'elpd'` as the column name in `az.compare` output. The existing `run_bayesian_comparison()` already uses `comparison['weight']` and `comparison.index` — inspect these carefully when adding M4 track and CSV write.
**Warning signs:** `KeyError: 'elpd_loo'` or `KeyError: 'weight'` at runtime.

### Pitfall 5: M4 NetCDF Path Inconsistency
**What goes wrong:** M4 hierarchical posterior is saved to a different path than the choice-only models because Phase 17 used `13_fit_bayesian_m4.py` (separate script).
**Why it happens:** M4 uses `output/bayesian/wmrl_m4_posterior.nc` from Phase 17's `13_fit_bayesian_m4.py`, which may use a different naming convention.
**How to avoid:** Before implementing CMP-02, verify the exact NetCDF path written by `scripts/13_fit_bayesian_m4.py`. Check `--output-dir` argument default in that script.

### Pitfall 6: Manuscript `#sec-bayesian-regression` references PyMC
**What goes wrong:** The Methods section at line 483 still says "PyMC with weakly informative priors...NUTS sampling (4 chains, 1000 warmup, 2000 draws)". After DOC-02, both places must say NumPyro.
**Why it happens:** PyMC was dropped in INFRA-07 but the manuscript wasn't updated.
**How to avoid:** Search for "PyMC" in paper.qmd (`grep -n "PyMC\|pymc"`) and replace every occurrence. Also update `#sec-stats` statistical analysis description.

---

## Code Examples

### Verified: az.compare with stacking (arviz 0.22.0)

```python
# Source: live arviz 0.22.0 API inspection (2026-04-13)
import arviz as az

# compare_dict: {model_name: InferenceData_with_log_likelihood_group}
comparison = az.compare(compare_dict, ic='loo', method='stacking')
# Returns DataFrame with columns: rank, elpd, pIC, elpd_diff, weight, se, dse, warning, scale
# Index = model names from compare_dict keys
# comparison['weight'] = stacking weight per model
# comparison.index[0] = top model name

# Write to CSV
comparison.to_csv(output_dir / 'stacking_weights.csv')
```

### Verified: az.loo with Pareto-k (already in codebase at line 630)

```python
# Source: scripts/14_compare_models.py line 630 (existing, verified working)
loo_result = az.loo(idata, pointwise=True)
k_vals = loo_result.pareto_k.values         # ndarray of per-obs Pareto-k
pct_high = float(np.mean(k_vals > 0.7) * 100)
```

### Verified: Schema-parity path resolution pattern

```python
# Source: scripts/fitting/bayesian_summary_writer.py (INFRA-04)
# Bayesian CSV location: output/bayesian/{model}_individual_fits.csv
# MLE CSV location:      output/mle/{model}_individual_fits.csv

def _resolve_fits_dir(source: str, project_root: Path) -> Path:
    """Source: Phase 18 design, consistent with bayesian_summary_writer.py."""
    if source == 'bayesian':
        return project_root / 'output' / 'bayesian'
    return project_root / 'output' / 'mle'
```

### Verified: Script 17 main() already takes mle_dir as local variable

```python
# Source: scripts/17_analyze_winner_heterogeneity.py lines 280-283
def main() -> None:
    mle_dir = project_root / "output" / "mle"   # <- only change needed for --source
    output_dir = project_root / "output" / "model_comparison"
    ...
    winners = load_per_participant_aic(mle_dir)  # passes dir, safe to reroute
```

### Verified: Script 16 load_integrated_data takes params_path directly

```python
# Source: scripts/16_regress_parameters_on_scales.py line 118
def load_integrated_data(params_path: Path, model_type: str = 'qlearning', ...) -> pd.DataFrame:
    params_df = pd.read_csv(params_path)   # direct path — CLI constructs it
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| PyMC backend for Bayesian regression | NumPyro-only (INFRA-07) | Phase 13 | `16b` docstring says PyMC dropped but manuscript still references PyMC |
| FDR-corrected post-hoc regression as primary inference | Level-2 hierarchical joint regression | Phase 16 | Results section must shift from p-value tables to forest plots |
| Per-participant MLE point estimates as analysis input | Posterior means with HDI as analysis input | Phase 18 | `--source bayesian` routing |
| M6b as winning model by AIC | M6b confirmed by WAIC/LOO stacking | Phase 18 (pending) | Stacking weight table replaces Akaike weight table |

**Deprecated/outdated:**
- PyMC references in `manuscript/paper.qmd#sec-stats` (line 483): replace with NumPyro Level-2
- "Bayesian Multivariate Regression" section `#sec-bayesian-regression` (line 892): replace with Level-2 posterior forest plot section
- `MODEL_REFERENCE.md` note "Current winning model: M5": update to M6b (already wrong per STATE.md)

---

## Open Questions

1. **M4 NetCDF path from Phase 17**
   - What we know: `scripts/13_fit_bayesian_m4.py` exists; it uses `--output-dir`
   - What's unclear: Does it write `output/bayesian/wmrl_m4_posterior.nc` or
     `output/bayesian/m4_posterior.nc` or something else?
   - Recommendation: Read the `--output-dir` default and `az.to_netcdf()` call
     in `13_fit_bayesian_m4.py` before implementing CMP-02. Do not assume path.

2. **Padding filter in build_inference_data_with_loglik()**
   - What we know: 13-03 decision says padding gets log_prob=0.0; plan says
     "plan 05 bayesian_diagnostics.py must filter mask==0 positions before
     az.waic()/az.loo()"
   - What's unclear: Was this actually implemented in `bayesian_diagnostics.py`
     or is the filtering left to the caller?
   - Recommendation: Read `build_inference_data_with_loglik()` in
     `scripts/fitting/bayesian_diagnostics.py` fully before running CMP-01.
     If padding is not filtered, add `log_lik_obs = log_lik[mask != 0]` before
     passing to InferenceData.

3. **Stacking weights output directory for script 14**
   - What we know: `run_bayesian_comparison()` writes to `level2_dir` which
     is `output_dir / "bayesian" / "level2"` (constructed inside the function)
   - What's unclear: The function takes `output_dir: Path` but internally uses
     `level2_dir = ...`. Inspect lines 639-750 for the exact `level2_dir`
     construction to know where the CSV should be written.
   - Recommendation: Keep the same directory (`output/bayesian/level2/`);
     just add `.to_csv()` alongside the existing `.md` write.

4. **Manuscript Limitations section location**
   - What we know: `paper.qmd` has sections: Intro, Methods, Results,
     Discussion, Conclusion, References, Appendix. No standalone Limitations.
   - What's unclear: Should limitations be a new `## Limitations` section
     before Discussion, or a subsection within Discussion?
   - Recommendation: Add `### Limitations {#sec-limitations}` as the last
     subsection within `## Discussion`, before `## Conclusion`. This is the
     APA/journal convention.

5. **`--source bayesian` for script 15 and column renaming**
   - What we know: Script 15 renames MLE columns with `_mean` suffix internally
     (e.g., `alpha_pos` -> `alpha_pos_mean`) via `load_integrated_data()` in
     script 16, but script 15's `load_data()` does NOT rename — it uses
     raw column names.
   - What's unclear: When Bayesian CSV has extra `_hdi_low/_hdi_high/_sd`
     columns, does any downstream plotting code in script 15 break on
     encountering unknown columns?
   - Recommendation: Test by inspecting which column names script 15
     downstream functions reference. Column-select by explicit list (not
     `df.columns` glob) is safe.

---

## Sources

### Primary (HIGH confidence)
- `scripts/14_compare_models.py` — verified `run_bayesian_comparison()`,
  `BAYESIAN_NETCDF_MAP`, `--bayesian-comparison` arg, `az.compare` call (lines 572-742, 818)
- `scripts/fitting/bayesian_summary_writer.py` — schema-parity column order,
  output path `output/bayesian/{model}_individual_fits.csv` (lines 1-260)
- `scripts/fitting/bayesian_diagnostics.py` — `compute_pointwise_log_lik`,
  parameter dispatch, vmap pattern (lines 1-230)
- `scripts/15_analyze_mle_by_trauma.py` — `load_data()` structure, no `--source` flag (lines 131-200, 803-821)
- `scripts/16_regress_parameters_on_scales.py` — `load_integrated_data()` signature, no `--source` flag (lines 118-180, 678-719)
- `scripts/17_analyze_winner_heterogeneity.py` — `main()` local `mle_dir`, no `--source` flag (lines 279-329)
- `scripts/16b_bayesian_regression.py` — no deprecation docstring yet (lines 1-60)
- `docs/03_methods_reference/MODEL_REFERENCE.md` — no hierarchical Bayesian section, winning model note outdated (lines 1-80)
- `manuscript/paper.qmd` — PyMC reference at line 483, section map lines 143-1161
- `.planning/STATE.md` — locked decisions, schema-parity pattern, M4 separate track constraint
- ArviZ 0.22.0 live API — `az.compare(ic='loo', method='stacking')` returns DataFrame with `rank, elpd, pIC, elpd_diff, weight, se, dse, warning, scale` columns (verified via Python subprocess)

### Tertiary (LOW confidence)
- Phase 17 M4 NetCDF output path: assumed `output/bayesian/wmrl_m4_posterior.nc`
  based on naming convention — not verified by reading `13_fit_bayesian_m4.py`
  source. Flag as open question.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all libraries already installed and used in codebase
- Architecture (flag plumbing): HIGH — codebase inspection confirmed exact structure
- Architecture (az.compare): HIGH — verified against live API
- Pitfalls: HIGH — derived from locked decisions in STATE.md and codebase inspection
- Open questions: honest gaps, not padded

**Research date:** 2026-04-13
**Valid until:** 2026-05-13 (stable domain; arviz API stable across minor versions)
