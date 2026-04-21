# Architecture Research — v4.0 Hierarchical Bayesian + LBA GPU

**Domain:** Computational psychiatry pipeline extension (hierarchical Bayesian inference, GPU-vectorized RT-choice joint modeling, Level-2 covariate regression)
**Researched:** 2026-04-10
**Confidence:** HIGH (sourced from direct codebase inspection of `numpyro_models.py`, `fit_bayesian.py`, `fit_mle.py`, `lba_likelihood.py`, `mle_utils.py`, `16b_bayesian_regression.py`, `14_compare_models.py`, `12_mle_gpu.slurm`, `config.py`, and the downstream consumer scripts 15/16/17)

---

## Critical Pre-existing Findings (Verified by Inspection)

Before recommending any new architecture, three facts about the current state must be on the record because they reshape the v4.0 design:

### Finding 1 — The current Bayesian path is structurally broken

`scripts/fitting/fit_bayesian.py` line 43 imports from `scripts.fitting.numpyro_models`, but `scripts/fitting/numpyro_models.py` does NOT exist at that location. The file lives in `scripts/fitting/legacy/numpyro_models.py`. A stale `__pycache__/numpyro_models.cpython-313.pyc` exists in the non-legacy `__pycache__/` directory (apparent leftover from a prior layout), so the import may resolve via cache on machines that previously ran the script, but a fresh checkout will fail.

**Implication for v4.0:** The first phase ("scaffolding") MUST resurrect or relocate `numpyro_models.py` to its expected import path. This is non-negotiable before any extension work begins. Treat this as a P0 bug, not a P1 enhancement.

### Finding 2 — LBA float64 is a process-wide flag, not module-local

`scripts/fitting/lba_likelihood.py` line 11: `jax.config.update("jax_enable_x64", True)`. This flips a global JAX flag that, once set, affects every JAX array created in the same Python process — including the choice-only float32 likelihoods. The choice-only modules defend themselves by passing explicit `dtype=jnp.float32` everywhere (verified at lines 214, 215, 737, 757, 804 etc. of `jax_likelihoods.py`), so they survive the flag flip, but they lose the throughput benefit of native float32 once LBA has been imported in the same process.

**Implication for v4.0:** The hierarchical M4-LBA fit and the hierarchical M1-M6b choice-only fits MUST run in separate Python processes (separate SLURM jobs). They cannot share a process. This is identical to the constraint already in force for MLE M4 and is the reason `12_mle_gpu.slurm` dispatches one SLURM job per model.

### Finding 3 — Downstream scripts (15/16/17/16b) are coupled to a flat CSV schema

All downstream consumption flows through `MODEL_REGISTRY[model]['csv_filename']` resolving to `output/mle/{model}_individual_fits.csv`. Verified:

- `scripts/17_analyze_winner_heterogeneity.py:114` — `csv_name = MODEL_REGISTRY[model]["csv_filename"]`
- `scripts/16_regress_parameters_on_scales.py:756` — `params_path = Path(f'output/mle/{model}_individual_fits.csv')`
- `scripts/16b_bayesian_regression.py:548` — `params_path = Path(f'output/mle/{MODEL_REGISTRY[args.model]["csv_filename"]}')`
- `scripts/15_analyze_mle_by_trauma.py:148-174` — direct CSV reads per model

The CSV is one row per participant, with columns for each parameter point estimate (`alpha_pos`, `alpha_neg`, `phi`, `rho`, `capacity`, ...) plus diagnostics (`nll`, `aic`, `bic`).

**Implication for v4.0:** A clean migration strategy is to make the hierarchical pipeline emit a CSV with the exact same schema (parameter columns now containing posterior means instead of MLE point estimates), written to a parallel directory `output/bayesian/{model}_individual_summary.csv`. Downstream scripts get a single new flag `--source mle|bayesian` that toggles the directory. No rewrites of 15/16/17 logic required — they keep working on whichever flat CSV they're pointed at. The richer ArviZ NetCDF posteriors (full samples, group-level parameters, Level-2 covariates) live alongside the summary CSV for the few downstream needs that require them (forest plots, group comparisons).

This is the migration pattern the rest of this document assumes.

---

## System Overview

### Current architecture (v3.0, MLE-centric)

```
                    Trial CSV (output/task_trials_long.csv)
                                   |
                                   v
                    +--------------+--------------+
                    |   prepare_block_data        |   <- jax_likelihoods.py
                    |   (jax_likelihoods.py)      |
                    +--------------+--------------+
                                   |
                                   v
       +---------------------------+---------------------------+
       |                                                       |
       v                                                       v
+--------------+                                      +-----------------+
|  MLE path    |                                      | Bayesian path   |
| (12_fit_mle) |                                      | (13_fit_bayes)  |
+------+-------+                                      +--------+--------+
       |                                                       |
       | jaxopt.LBFGS + vmap                                   | NUTS HMC
       | (CPU per-pid OR GPU vmap-all)                         | (M1/M2 only,
       v                                                       |  CURRENTLY BROKEN)
+----------------------------------+                           v
| output/mle/{model}_individual_   |              output/bayesian_fits/{model}.nc
|   fits.csv (1 row/participant)   |              (NetCDF, M1/M2 only, no Level-2)
+----------------+-----------------+
                 |
                 v
   +-------------+--------------+
   | 14_compare_models           |  <- AIC/BIC across MLE fits
   | 15_analyze_mle_by_trauma    |  <- group comparisons on point ests
   | 16_regress_..._on_scales    |  <- OLS regression
   | 16b_bayesian_regression     |  <- post-hoc PyMC/NumPyro on MLE point ests
   | 17_analyze_winner_heterog.  |  <- winner per participant
   +-----------------------------+
```

The MLE path is mature and feeds everything. The Bayesian path is a vestigial branch with broken imports, only two models, and no Level-2 covariates.

### Target architecture (v4.0, dual-output)

```
                    Trial CSV + Survey CSV (lec_total, ies_total, subscales)
                                   |
                                   v
                +------------------+------------------+
                |  prepare_block_data (existing)      |
                |  + prepare_level2_covariates (NEW)  |
                +------------------+------------------+
                                   |
                                   v
       +---------------------------+---------------------------+
       |                                                       |
       v                                                       v
+--------------+                              +----------------------------------+
|  MLE path    |                              |  Hierarchical Bayesian path      |
| (UNCHANGED   |                              |  (NEW: 13_fit_bayesian extended  |
|  except K    |                              |   to all 7 models, Level-2       |
|  bounds)     |                              |   covariates, GPU LBA via vmap)  |
+------+-------+                              +-----+----------------------+-----+
       |                                            |                      |
       | jaxopt.LBFGS + vmap                        | NUTS (choice-only)   | NUTS (LBA)
       | (CPU per-pid OR GPU vmap-all)              | float32 process      | float64 process
       v                                            v                      v
+----------------------------------+   +----------------------------+  +----------------------+
| output/mle/{model}_individual_   |   | output/bayesian/{model}_   |  | output/bayesian/     |
|   fits.csv  (FAST PREVIEW TIER)  |   |   posterior.nc (full draws)|  | wmrl_m4_posterior.nc |
| output/mle/{model}_group_summ.   |   | output/bayesian/{model}_   |  | + summary CSV        |
|   csv                             |   |   individual_summary.csv  |  +----------------------+
+----------------+-----------------+   |   (post-mean per pid;     |
                 |                     |    schema MATCHES MLE CSV) |
                 |                     +-------------+--------------+
                 |                                   |
                 |  --source mle (default)           |  --source bayesian
                 |                                   |
                 +-------------------+---------------+
                                     v
   +---------------------------------+---------------------------------+
   | 14_compare_models  (extended: AIC/BIC for MLE path,                |
   |                     WAIC/LOO via arviz.compare for Bayesian path,  |
   |                     reports both side by side)                     |
   |                                                                    |
   | 15/16/17  (UNCHANGED logic; new --source flag selects directory)   |
   | 16b       (UNCHANGED; eventually deprecated when 15/16/17 pull     |
   |            from Bayesian summaries directly)                       |
   |                                                                    |
   | 18_bayesian_level2_effects  (NEW: forest plots of Level-2 effects  |
   |                              from posterior NetCDF; only consumer  |
   |                              that needs full ArviZ object)         |
   +--------------------------------------------------------------------+
```

The key insight: by emitting a flat per-participant CSV from the Bayesian path with the same schema as MLE, **dual paths become a routing problem, not a rewrite problem.** The full ArviZ NetCDF still exists for the one or two downstream needs (Level-2 effects, posterior predictive checks, model comparison via `arviz.compare`) that genuinely require it.

---

## Component Responsibilities

| Component | Responsibility | Current State | v4.0 Change |
|-----------|----------------|---------------|-------------|
| `scripts/fitting/jax_likelihoods.py` | Per-trial JAX log-likelihoods for M1/M2/M3/M5/M6a/M6b (float32) | 7 models, JIT-compiled, vmap-friendly | UNCHANGED. Already supports all needed models. K bound is in `mle_utils.py`, not here. |
| `scripts/fitting/lba_likelihood.py` | M4 LBA likelihood (float64, process-global) | Working, JIT/vmap-compatible | UNCHANGED. Already vmap-compatible (line 197 uses `jax.vmap` over accumulators). |
| `scripts/fitting/mle_utils.py` | MLE bounds, transforms, PARAMS lists | All 7 models present; capacity bounds `(1.0, 7.0)` for all WM models | MODIFIED. Capacity bounds tightened per Collins research (probably `(2.0, 5.0)` or `(2.0, 6.0)`). Single source of truth for bounds. |
| `scripts/fitting/fit_mle.py` | MLE driver (CPU per-pid, GPU vmap) | Working, NaN-safe, checkpoint-resume | UNCHANGED algorithmically. Picks up new K bounds from `mle_utils.py` automatically. |
| `scripts/fitting/numpyro_models.py` | Hierarchical NumPyro models (M1, M2 only, broken import) | LIVES IN `legacy/`; broken import path | RESURRECTED + EXTENDED. Move to non-legacy location. Add M3, M5, M6a, M6b, M4 hierarchical models. Add Level-2 covariate signature. Replace Python-loop participant likelihood with vmap. |
| `scripts/fitting/fit_bayesian.py` | Bayesian driver | Only accepts `qlearning|wmrl`; broken imports | EXTENDED. Accept all 7 models. Add `--level2-covariates` flag pointing to a CSV with columns `sona_id, lec_intrusion, lec_avoidance, ies_total, ...`. Add `--save-individual-summary-csv` flag (default ON) to emit MLE-compatible flat CSV alongside NetCDF. |
| `scripts/13_fit_bayesian.py` | CLI wrapper | Working | EXTENDED to pass through new args. |
| `scripts/14_compare_models.py` | AIC/BIC comparison + winner selection | Works on MLE CSVs | EXTENDED, NOT REPLACED. Add `--bayesian-comparison` mode that loads NetCDFs and runs `arviz.compare(idatas, ic='loo')`. Default mode (MLE AIC/BIC) unchanged. |
| `scripts/15_analyze_mle_by_trauma.py` | Group comparisons of params by trauma group | Reads MLE CSVs directly | MINIMALLY MODIFIED. Add `--source mle|bayesian` flag. Path resolution becomes `output/{source}/{model}_individual_*.csv`. Logic unchanged. |
| `scripts/16_regress_parameters_on_scales.py` | OLS regression | Reads MLE CSVs | Same `--source` flag treatment. |
| `scripts/16b_bayesian_regression.py` | Post-hoc Bayesian regression on MLE point estimates | Working, dual-backend | DEPRECATED IN PLACE, not deleted. With true hierarchical Bayesian fits including Level-2 covariates, the post-hoc regression becomes redundant — the Level-2 effects are now estimated jointly with individual parameters. Keep `16b` working through v4.0 as a sanity check, mark for removal in v5.0. |
| `scripts/17_analyze_winner_heterogeneity.py` | Winner-per-participant analysis | Reads MLE CSVs via `MODEL_REGISTRY` | Same `--source` flag treatment. |
| `scripts/18_bayesian_level2_effects.py` (NEW) | Forest plots of Level-2 covariate effects | — | NEW. Loads `{model}_posterior.nc`, extracts `beta_lec_*`, `beta_ies_*` parameters, plots posterior densities and HDIs. Only script that genuinely requires the full ArviZ object. |
| `scripts/19_bayesian_model_comparison.py` (NEW, optional) | If 14 gets too crowded | — | OPTIONAL. Could split out the Bayesian comparison from script 14 if 14 becomes unwieldy. Default plan: keep in 14. |
| `cluster/13_bayesian_gpu.slurm` (NEW) | SLURM driver for hierarchical fits | — | NEW. Mirrors `12_mle_gpu.slurm` parallel-dispatch pattern. One SLURM job per model. M4 gets a separate, longer-walltime variant (`13_bayesian_gpu_m4.slurm`) because of float64 + LBA. |
| `config.py MODEL_REGISTRY` | Single source of truth for model metadata | All 7 models, csv_filename | EXTENDED. Add `bayesian_implemented: bool` (already covered by checking `numpyro_models.py` registration), `bayesian_csv_filename` for the new individual summary path, `level2_supported: bool`. |

---

## Recommended Project Structure (v4.0 deltas only)

```
scripts/
  fitting/
    jax_likelihoods.py          # UNCHANGED
    lba_likelihood.py           # UNCHANGED
    mle_utils.py                # MODIFIED: K bounds tightened post-Collins
    fit_mle.py                  # UNCHANGED
    fit_bayesian.py             # EXTENDED: all 7 models, Level-2, dual output
    numpyro_models.py           # RESURRECTED FROM legacy/, EXTENDED to all 7 models
    bayesian_summary_writer.py  # NEW: convert ArviZ idata -> flat CSV
    legacy/
      numpyro_models.py         # KEEP as historical reference
      ...

  13_fit_bayesian.py            # EXTENDED CLI wrapper
  14_compare_models.py          # EXTENDED: --bayesian-comparison mode
  15_analyze_mle_by_trauma.py   # +--source flag
  16_regress_parameters...py    # +--source flag
  16b_bayesian_regression.py    # FROZEN, deprecation comment added
  17_analyze_winner_heterog.py  # +--source flag
  18_bayesian_level2_effects.py # NEW

cluster/
  12_mle.slurm                  # MODIFIED: rerun all models with new K bounds
  12_mle_gpu.slurm              # SAME modification
  13_bayesian_gpu.slurm         # NEW: dispatcher for choice-only Bayesian
  13_bayesian_gpu_m4.slurm      # NEW: longer walltime for LBA
  13_submit_all_bayesian.sh     # NEW: convenience script

output/
  mle/                          # UNCHANGED: continues to receive MLE outputs
    {model}_individual_fits.csv
    {model}_group_summary.csv
  bayesian/                     # NEW DIRECTORY
    {model}_posterior.nc        # Full ArviZ InferenceData (NetCDF)
    {model}_individual_summary.csv  # Posterior means per participant; schema MATCHES MLE CSV
    {model}_group_summary.csv   # Group-level posterior means + HDIs
    {model}_diagnostics.csv     # R-hat, ESS, divergences per chain
```

### Structure Rationale

- **`output/mle/` and `output/bayesian/` are siblings, not nested.** This is what enables the `--source` flag pattern in downstream scripts: a single environment variable or flag flips one segment of the path, nothing else changes.
- **Schema parity is enforced at write time** by `bayesian_summary_writer.py`, NOT by changing the readers. The writer is the only file that knows both about ArviZ structure AND about the MLE CSV column convention.
- **NumPyro models live in one file (`numpyro_models.py`), not per-model files.** The legacy file already shows the pattern works for M1+M2. Splitting into per-model files would require duplicating the priors, the non-centered transformations, and the Level-2 plumbing seven times. A single file with seven model functions plus shared helper functions for priors and Level-2 injection is the right granularity. Comparison: `jax_likelihoods.py` already holds all 7 likelihoods in one file (3500+ lines) without difficulty.

---

## Architectural Patterns

### Pattern 1 — Schema-parity dual outputs (the migration cornerstone)

**What:** Both MLE and Bayesian paths emit a flat per-participant CSV with the **identical column schema** (sona_id + each parameter + nll/aic/bic or their Bayesian equivalents loo/waic/posterior_mean_nll). The full Bayesian information lives in a sidecar NetCDF for the one or two analyses that need it.

**When to use:** When migrating a downstream pipeline from one inference method to another and rewriting downstream consumers is high-cost. (This is exactly v4.0's situation.)

**Trade-offs:**
- (+) Downstream scripts (15, 16, 17) get a single one-line change: replace hardcoded `output/mle/` with `output/{args.source}/`.
- (+) MLE remains a fully-supported "fast preview" tier indefinitely. Researcher can iterate on a model in MLE in 5 minutes, validate with hierarchical in 6 hours.
- (+) The Bayesian individual_summary.csv still loses information (point estimates instead of full posteriors) but this is exactly what most downstream group-comparison and regression analyses need anyway.
- (-) Some analyses that should use the full posterior (e.g., propagating uncertainty into a meta-regression) will silently use point estimates if the researcher uses the wrong source. Mitigation: emit `*_individual_summary.csv` with extra columns `*_hdi_low`, `*_hdi_high`, and `*_posterior_sd` so downstream tools CAN use them.

**Example:**
```python
# scripts/fitting/bayesian_summary_writer.py
def write_individual_summary(idata: az.InferenceData, model: str, output_dir: Path):
    """Convert ArviZ idata to MLE-compatible flat CSV.

    Schema: one row per participant, one column per parameter (posterior mean).
    Extra columns: {param}_hdi_low, {param}_hdi_high, {param}_sd, log_lik_post_mean.
    """
    param_names = MODEL_REGISTRY[model]['params']
    posterior = idata.posterior  # (chain, draw, participant) for each individual param

    rows = []
    for p_idx, sona_id in enumerate(idata.observed_data.attrs.get('participant_ids', [])):
        row = {'sona_id': sona_id}
        for param in param_names:
            samples = posterior[param].values[..., p_idx].flatten()
            row[param] = float(samples.mean())
            row[f'{param}_sd'] = float(samples.std())
            row[f'{param}_hdi_low'], row[f'{param}_hdi_high'] = az.hdi(samples, hdi_prob=0.95)
        rows.append(row)

    df = pd.DataFrame(rows)
    out_path = output_dir / f'{model}_individual_summary.csv'
    df.to_csv(out_path, index=False)
    return out_path
```

### Pattern 2 — Process isolation for float64 LBA

**What:** M4 (LBA) requires `jax.config.update("jax_enable_x64", True)`, which is a process-wide flag. Separate Python processes are mandatory between LBA fits and choice-only fits. Use SLURM job separation, not Python imports.

**When to use:** Whenever a JAX-based pipeline mixes float32 and float64 likelihoods. There is no in-process workaround.

**Trade-offs:**
- (+) Each model fit is a clean, self-contained process. Failures isolate.
- (+) GPU memory is fully released between models.
- (-) JAX recompilation per process. Mitigated by `JAX_COMPILATION_CACHE_DIR` (already set in `12_mle_gpu.slurm`).

**Example:** Mirror the existing `12_mle_gpu.slurm` parallel dispatch pattern (already verified at lines 161-188). One SLURM job per model.

### Pattern 3 — vmap for likelihood, NOT for chains, under NUTS

**What:** Inside a NumPyro model, the participant likelihood is computed via `jax.vmap` over a stacked batch of participant data, NOT via a Python loop. The MCMC chain parallelism is handled separately by NumPyro's `chain_method="parallel"` (which uses `pmap` across XLA devices) — DO NOT vmap chains yourself.

**When to use:** This is the standard NumPyro pattern. The legacy `numpyro_models.py` violates it (line 165: `for i, participant_id in enumerate(participant_ids)` produces a Python-level loop that JAX traces but cannot batch). For 154 participants this becomes a >150x slowdown vs. vmap.

**Trade-offs:**
- (+) Single vmapped likelihood call per gradient evaluation = single XLA-fused kernel = full GPU utilization.
- (+) Per-participant data must be padded to fixed shape (already handled by `pad_blocks_to_max` in `jax_likelihoods.py` lines 86-215). Reuse the existing padding utilities.
- (-) Padding wastes some compute on shorter participants. The padding utilities already handle masking; the wasted compute is bounded by `(MAX_BLOCKS * MAX_TRIALS_PER_BLOCK) - (actual trials)` which is small for this dataset.

**Example:**
```python
# scripts/fitting/numpyro_models.py (NEW, replacing legacy)
def wmrl_m5_hierarchical_model(participant_data_stacked, level2_covariates=None):
    """All participants in one vmapped likelihood call.

    participant_data_stacked is the OUTPUT of pad_blocks_to_max applied
    across all participants — shape (n_participants, MAX_BLOCKS, MAX_TRIALS).
    """
    n_pid = participant_data_stacked['stimuli'].shape[0]

    # === Group-level priors (tightened K per Collins) ===
    mu_capacity = numpyro.sample('mu_capacity',
        dist.TruncatedNormal(3.5, 1.0, low=2.0, high=5.0))   # <- Collins-constrained
    sigma_capacity = numpyro.sample('sigma_capacity', dist.HalfNormal(0.8))
    # ... other group params

    # === Level-2 (trauma) effects on each individual param ===
    if level2_covariates is not None:
        n_cov = level2_covariates.shape[1]
        beta_capacity = numpyro.sample('beta_capacity', dist.Normal(0., 1.).expand([n_cov]))
        beta_alpha_pos = numpyro.sample('beta_alpha_pos', dist.Normal(0., 1.).expand([n_cov]))
        # ... etc
        capacity_offset_lvl2 = level2_covariates @ beta_capacity   # (n_pid,)
    else:
        capacity_offset_lvl2 = jnp.zeros(n_pid)

    # === Individual-level (non-centered) ===
    with numpyro.plate('participants', n_pid):
        z_capacity = numpyro.sample('z_capacity', dist.Normal(0, 1))
        capacity = numpyro.deterministic('capacity',
            jnp.clip(mu_capacity + sigma_capacity * z_capacity + capacity_offset_lvl2,
                     2.0, 5.0))
        # ... other individual params

    # === Vmapped likelihood — KEY CHANGE FROM LEGACY ===
    # Build per-participant param vector, vmap the likelihood
    params_vec = jnp.stack([alpha_pos, alpha_neg, phi, rho, capacity, kappa, phi_rl, epsilon], axis=-1)

    def per_pid_loglik(params, stim, act, rew, mask, ss):
        return wmrl_m5_multiblock_likelihood_stacked(  # already exists in jax_likelihoods.py
            stim, act, rew, mask, ss,
            alpha_pos=params[0], alpha_neg=params[1], phi=params[2], rho=params[3],
            capacity=params[4], kappa=params[5], phi_rl=params[6], epsilon=params[7],
        )

    log_liks = jax.vmap(per_pid_loglik)(
        params_vec,
        participant_data_stacked['stimuli'],
        participant_data_stacked['actions'],
        participant_data_stacked['rewards'],
        participant_data_stacked['masks'],
        participant_data_stacked['set_sizes'],
    )

    numpyro.factor('log_lik', log_liks.sum())
```

The `*_multiblock_likelihood_stacked` versions already exist in `jax_likelihoods.py` (verified at lines 626, 1349, 1557, 1942, 2287, 2644 for all 6 choice-only variants, and `lba_likelihood.py` line 608 for M4). They take pre-padded fixed-shape arrays and are designed exactly for vmap. **The hard work is already done.**

### Pattern 4 — Level-2 covariates as an additive offset on the linear predictor

**What:** Trauma scale subscales are injected as a linear regression on the constrained scale of each individual parameter. For bounded parameters (alpha, phi, etc.), the linear predictor lives in unconstrained space (logit), then is squashed back. For capacity, it lives directly on the bounded scale with a clip.

**When to use:** When you want hierarchical estimation of "trauma -> parameter" effects WITH proper uncertainty propagation and shrinkage, instead of post-hoc regression on point estimates (which is what `16b_bayesian_regression.py` currently does).

**Trade-offs:**
- (+) Single inference pass: posterior captures parameter-trauma joint uncertainty.
- (+) Shrinkage at the individual level reduces noise, improving statistical power for detecting trauma effects.
- (-) Adds n_params * n_covariates new free parameters at the population level. For M5 (8 params) and 4 trauma subscales, that's 32 new betas. This is fine for n=154, but priors must be informative (`Normal(0, 1)`) to prevent overfitting.
- (-) Cannot easily mix-and-match which params depend on which subscales (full crossing). For v4.0, full crossing is acceptable; future versions could add a covariate-mask matrix.

**Example signature change to `prepare_data_for_numpyro`:**
```python
def prepare_data_for_numpyro(
    data_df: pd.DataFrame,
    survey_df: pd.DataFrame | None = None,        # NEW
    level2_columns: list[str] | None = None,      # NEW: e.g. ['lec_intrusion', 'lec_avoidance', 'ies_total']
    standardize_level2: bool = True,              # NEW: z-score covariates
    participant_col: str = 'sona_id',
    ...
) -> tuple[dict, jnp.ndarray | None]:
    """Returns (participant_data_stacked, level2_matrix or None).

    level2_matrix is a (n_participants, n_covariates) jnp.ndarray aligned to the
    participant order in participant_data_stacked. None if no covariates requested.
    """
```

### Pattern 5 — Fast-preview MLE + slow-truth Bayesian

**What:** MLE remains the default fitting method for iteration. Once a model is decided, hierarchical Bayesian is run as the publication-quality fit.

**When to use:** Always, in this project. The MLE fit takes 5 minutes (CPU) or 5 seconds (GPU) per model. The hierarchical fit takes 4-12 hours per model. Researchers should NOT be waiting on hierarchical fits during model development.

**Trade-offs:**
- (+) Fast iteration loop preserved.
- (+) Researcher can validate K bound changes, prior choices, etc., on MLE before paying the hierarchical cost.
- (-) Risk of MLE-Bayesian disagreement that gets ignored. Mitigation: script `14_compare_models.py` should print per-model deltas between MLE point estimate and Bayesian posterior mean as a sanity check.

---

## Data Flow

### MLE path (UNCHANGED)

```
trial_csv -> prepare_block_data -> participant_data dict (per-pid arrays)
                                              |
                                              v
                                  fit_all_participants (CPU joblib)
                                              OR
                                  fit_all_gpu (vmap over starts and pids)
                                              |
                                              v
                          {model}_individual_fits.csv
                          {model}_group_summary.csv
```

### Bayesian path (NEW)

```
trial_csv + survey_csv
        |
        v
prepare_data_for_numpyro (extended)
   - returns: stacked padded participant arrays (jnp)
   - returns: level2_covariates matrix (jnp) or None
        |
        v
{model}_hierarchical_model() <- numpyro_models.py
   - vmapped per-participant likelihood
   - non-centered participant params
   - Level-2 betas at population level
   - Collins-constrained K prior
        |
        v
NUTS (via numpyro.infer.MCMC)
   - chains: 4 (parallel via numpyro.set_host_device_count or pmap on multi-GPU)
   - warmup: 1500 (more than v3.0's 1000 because of Level-2 params)
   - samples: 2000
        |
        v
+--------------------------------+
| ArviZ InferenceData            |
| - posterior (full draws)       |
| - sample_stats (divergences)   |
| - observed_data                |
| - log_likelihood (for LOO)     |
+----------------+---------------+
                 |
                 v
   +-------------+----------------+
   |                              |
   v                              v
{model}_posterior.nc      bayesian_summary_writer
(full NetCDF)                     |
                                  v
                  {model}_individual_summary.csv
                  {model}_group_summary.csv
                  {model}_diagnostics.csv
```

### Downstream consumption (after `--source` flag added)

```
            +------------------+      +------------------+
            | output/mle/{m}_  |      | output/bayesian/ |
            | individual_fits  |      | {m}_individual_  |
            +--------+---------+      | summary.csv      |
                     |                +--------+---------+
                     +-----+----------+
                           |
                  --source mle (default) or bayesian
                           |
                           v
            Same downstream code:
            15_analyze_mle_by_trauma
            16_regress_..._on_scales
            17_analyze_winner_heterog
            14_compare_models  (also reads NetCDFs for WAIC/LOO when --bayesian-comparison)
            18_bayesian_level2_effects (reads NetCDF directly)
```

---

## Build Order (respects dependency chain)

Phase numbering continues the v3.0 sequence (last v3.0 phase was P12; v4.0 starts at P13).

| Phase | Name | Depends on | Outputs | Dependency rationale |
|-------|------|------------|---------|----------------------|
| **P13** | Collins K Research | — | `docs/COLLINS_K_BOUNDS.md` summarizing literature on appropriate K bounds | Must complete BEFORE any refit (MLE or Bayesian) so all downstream fits use the same bound. |
| **P14** | K Bound Refit (MLE) | P13 | New MLE fits for all 7 models with constrained K, regenerated `output/mle/*_individual_fits.csv` | Refit MLE first because it's fast (~30 min total for all 7 models). Validates the K bound doesn't break any model. Surfaces problems cheaply before paying for hierarchical fits. |
| **P15** | Bayesian scaffolding fix | P14 (any K fix to mle_utils.py is shared) | `numpyro_models.py` resurrected to non-legacy path; broken import fixed; existing M1+M2 hierarchical fits reproduce v3.0 (with new K bound) | Pre-requisite for ANY new hierarchical work. Get the broken pipeline back to its v3.0 baseline. |
| **P16** | numpyro_models extension to all 7 models (no Level-2 yet) | P15 | M3, M5, M6a, M6b, M4 hierarchical models added; vmap-stacked likelihoods used (NOT Python loops); recovery test on synthetic data | Must complete before Level-2 because Level-2 requires the model to already exist. M4 added last because of float64 / process isolation. Each model added one at a time, with synthetic-data recovery test before moving to the next. |
| **P17** | Level-2 covariate plumbing | P16 | `prepare_data_for_numpyro` extended; all 7 models accept `level2_covariates` arg; recovery test with simulated covariate effects | Must come after the basic hierarchical models work. Adding Level-2 to a broken model wastes time. |
| **P18** | Real-data hierarchical fits (choice-only models) | P17 + P14 (constrained K) | `output/bayesian/{m}_posterior.nc` + summary CSVs for M1, M2, M3, M5, M6a, M6b | Choice-only models share float32 process. Fit them as parallel SLURM jobs (pattern from `12_mle_gpu.slurm`). |
| **P19** | Real-data hierarchical fit (M4 LBA) | P17, P18 (validates the Level-2 plumbing on choice-only first) | `output/bayesian/wmrl_m4_posterior.nc` + summary CSV | Separate process / SLURM job because of float64. Long walltime. Run AFTER P18 succeeds so any Level-2 bugs are caught on the cheaper fits first. |
| **P20** | Bayesian model comparison | P18, P19 | Extended `14_compare_models.py` with WAIC/LOO via `arviz.compare`; output table mirrors AIC/BIC table | Once posteriors exist, comparison becomes straightforward. Single ArviZ call. |
| **P21** | Downstream `--source` flag rollout | P18 | 15, 16, 17 accept `--source` flag; default still `mle`; CI/test runs both sources to verify schema parity | Pure plumbing — schema parity from P18's writer makes this 1-line changes. |
| **P22** | New Level-2 effects script (18) | P18, P19 | `scripts/18_bayesian_level2_effects.py`; forest plots of Level-2 betas per model; written to `figures/bayesian/level2_effects/{model}/` | Final user-facing deliverable. Requires posteriors to exist. |
| **P23** | Manuscript integration | All above | Updated tables and figures in `manuscript/paper.tex`; final model selection rationale | Always last. |

**Critical path:** P13 -> P14 -> P15 -> P16 -> P17 -> P18 -> P19 -> P20 -> P21/P22 -> P23.

P15 cannot start before the broken Bayesian path is fixed (this is the riskiest phase because it surfaces unknown unknowns from the legacy file).

P19 (M4-LBA hierarchical) is the highest-risk technical phase because: (a) float64 process isolation; (b) LBA likelihood is the hardest to make NUTS-friendly (gradients are subtle); (c) longest walltime. Schedule it with a ~2x time buffer.

---

## Migration Strategy: Dual Outputs, Cutover Deferred

**Recommended approach: Maintain dual MLE + Bayesian paths through v4.0 and v5.0. Do not deprecate MLE.**

Reasoning:

1. **MLE is structurally faster** (5 sec/model on GPU vs. 4-12 hr/model for hierarchical). Researchers will always want the fast tier for iteration. Removing it would punish exploratory analysis.

2. **MLE serves as a sanity check on Bayesian fits.** If MLE point estimates and Bayesian posterior means disagree by more than 1 SD, something is wrong (prior misspecification, divergence, label switching). Keeping both allows automatic disagreement flagging in `14_compare_models.py`.

3. **Existing downstream scripts already work on MLE CSVs.** The schema-parity migration pattern means downstream code paths are NEARLY identical between MLE and Bayesian — `--source mle|bayesian` is a one-line flag, not a fork.

4. **Bayesian fits have fragility risks** (NUTS divergences, slow chains, OOM on certain GPU types). If a hierarchical fit fails the night before a deadline, MLE remains the fallback.

**Cutover plan (deferred to v5.0 or later):**
- v4.0: Default downstream `--source` is `mle`. Bayesian fits available but opt-in.
- v4.x patch: Once Bayesian fits are stable for all 7 models on the production data, flip default to `--source bayesian` for `15`, `16`, `17`. Keep MLE accessible.
- v5.0: Re-evaluate whether `16b_bayesian_regression.py` should be deleted (it's superseded by Level-2 covariates inside the hierarchical fit).
- v6.0+: Re-evaluate deleting the MLE GPU vmap path. Probably never — it's too useful.

---

## Cluster Orchestration Pattern

**Recommendation: One SLURM job per model. NOT one job per model-chain combination. NOT one big job with within-GPU parallelism.**

Reasoning:

1. **Mirrors the existing pattern.** `12_mle_gpu.slurm` lines 161-188 already do parallel dispatch one-job-per-model. This is well-tested on Monash M3. Reuse the pattern verbatim.

2. **NumPyro handles intra-job chain parallelism.** Setting `numpyro.set_host_device_count(args.chains)` and using `chain_method="parallel"` runs all 4 chains within a single GPU job using XLA's `pmap`. On a single-GPU node, the chains time-share but XLA handles overlap. On a multi-GPU node, NumPyro pmaps chains across devices. This is the right level for chain parallelism — Python-level orchestration of one-job-per-chain would require manual NetCDF merging afterward.

3. **Process isolation for M4.** As established, M4 must be a separate process (separate SLURM job) because of `jnp.float64`. The "one job per model" pattern enforces this naturally.

4. **Walltime sizing.** Choice-only models with hierarchical NUTS on n=154 with 4 chains, 1500 warmup, 2000 samples: estimate 4-8 hours per model on A100, longer on V100. M4-LBA: estimate 12-24 hours. Set walltimes accordingly:

| Model | Walltime | Partition | GPU | Memory |
|-------|----------|-----------|-----|--------|
| M1 (qlearning) | 04:00:00 | gpu | 1x A100 or V100 | 32G |
| M2 (wmrl) | 06:00:00 | gpu | 1x A100 or V100 | 32G |
| M3 (wmrl_m3) | 08:00:00 | gpu | 1x A100 | 32G |
| M5 (wmrl_m5) | 08:00:00 | gpu | 1x A100 | 32G |
| M6a (wmrl_m6a) | 08:00:00 | gpu | 1x A100 | 32G |
| M6b (wmrl_m6b) | 10:00:00 | gpu | 1x A100 | 32G |
| **M4 (wmrl_m4 LBA)** | **24:00:00** | **gpu** | **1x A100 (40GB)** | **64G** |

5. **Checkpointing.** NumPyro NUTS does not support mid-chain checkpointing the way the MLE path does (where each completed participant is a checkpoint row). Hierarchical fits are all-or-nothing per chain. Mitigation: save the InferenceData to NetCDF immediately after each chain completes if running chains sequentially. Or accept that a job failure means restarting that model. (For 154 participants and ~6 hour fits, restart cost is acceptable.)

6. **GPU type matters.** A100 (40GB) is strongly preferred for hierarchical fits. V100 (16GB) will OOM on some models with the full padded participant matrix (154 pid × 17 blocks × 100 trials × n_params × 4 chains). Test on V100 first with `--limit 30` participants before queuing the full A100 job.

**Cluster file structure:**
```
cluster/
  13_bayesian_gpu.slurm           # Dispatcher: parses MODEL env var, mirrors 12_mle_gpu.slurm
  13_bayesian_gpu_choice_only.sh  # Convenience: submits M1, M2, M3, M5, M6a, M6b in parallel
  13_bayesian_gpu_m4.slurm        # Separate slurm with longer walltime, 64G mem
```

---

## Anti-Patterns

### Anti-Pattern 1: Per-model NumPyro files

**What people do:** Create `numpyro_qlearning.py`, `numpyro_wmrl.py`, `numpyro_wmrl_m5.py`, etc.

**Why it's wrong:** Forces duplication of (a) the prior helper functions, (b) the non-centered parameterization boilerplate, (c) the Level-2 covariate injection code, (d) the vmap-likelihood call pattern. With 7 models and 8 group-level params each, that's ~50 lines of duplicated boilerplate per file = 350 lines of duplication. Bug fixes must be made 7 times.

**Do this instead:** One `numpyro_models.py` with 7 model functions (`qlearning_hierarchical_model`, `wmrl_hierarchical_model`, ..., `wmrl_m4_hierarchical_model`) and shared helper functions (`_sample_alpha_priors`, `_sample_capacity_prior_collins`, `_inject_level2_offsets`, `_vmapped_likelihood`). Pattern is already validated by `jax_likelihoods.py` which holds 7 model likelihoods in one file without difficulty.

### Anti-Pattern 2: Python loop over participants in numpyro model

**What people do:** Iterate `for pid in participants: numpyro.factor(f'obs_{pid}', loglik(...))` (this is what the legacy `numpyro_models.py` does at line 165).

**Why it's wrong:** JAX traces every iteration as a separate operation. NumPyro then has to grad through all 154 separate factors, producing 154 separate XLA kernels per gradient evaluation. Empirically: ~150x slower than vmap. For a 6-hour fit, this is the difference between 6 hours and 6 weeks.

**Do this instead:** Pre-pad all participant data to fixed shape (use existing `pad_blocks_to_max` from `jax_likelihoods.py`), stack into `(n_pid, MAX_BLOCKS, MAX_TRIALS)` arrays, vmap the likelihood across the participant axis, sum to a single `numpyro.factor`. Fast.

### Anti-Pattern 3: Mixing float32 and float64 likelihoods in one process

**What people do:** Import both `jax_likelihoods` (float32) and `lba_likelihood` (float64) in the same script.

**Why it's wrong:** `jax.config.update("jax_enable_x64", True)` is process-global. Once `lba_likelihood` is imported, every JAX array defaults to float64 unless explicitly typed. The choice-only modules survive (because they cast explicitly) but lose float32 throughput. Worse, partial casting can introduce silent precision mismatches in arithmetic operations that combine arrays of different dtypes.

**Do this instead:** Run M4 in a separate Python process (separate SLURM job). Never import both modules in the same script. Enforce this with an import-time assertion in `lba_likelihood.py`: if `jax_likelihoods` has already been imported, raise.

### Anti-Pattern 4: Replacing 16b instead of deprecating it

**What people do:** Delete `16b_bayesian_regression.py` once Level-2 covariates are inside the hierarchical model.

**Why it's wrong:** `16b` is a useful sanity check. If the hierarchical fit's Level-2 betas disagree wildly with `16b`'s post-hoc regression on MLE point estimates, something is wrong with one of them. Keeping both lets you cross-validate. Also, `16b` produces nicer per-parameter forest plots than the per-model arrangement of the new `18_bayesian_level2_effects.py`.

**Do this instead:** Mark `16b` as DEPRECATED in v4.0 with a comment, but keep it functional. Plan removal for v5.0 or later, after the hierarchical Level-2 effects have been validated against `16b` outputs.

### Anti-Pattern 5: Rewriting downstream scripts (15, 16, 17) for ArviZ

**What people do:** Rewrite `15_analyze_mle_by_trauma.py` to load `output/bayesian/{model}_posterior.nc` and operate on `idata.posterior` arrays directly.

**Why it's wrong:** Triples the script complexity, breaks backward compatibility, requires every downstream consumer to learn ArviZ idioms, and ties downstream code to a specific posterior representation. Also: most group comparisons and regressions only care about parameter point estimates anyway — the full posterior is overkill.

**Do this instead:** Schema-parity dual outputs (Pattern 1). Downstream scripts get a `--source` flag, MLE and Bayesian flat CSVs have identical column schemas, downstream code is unchanged. Only `18_bayesian_level2_effects.py` (the one analysis that genuinely needs the full posterior) reads the NetCDF directly.

---

## Integration Points

### New files

| File | Purpose |
|------|---------|
| `scripts/fitting/numpyro_models.py` | (Resurrected from legacy) All 7 hierarchical NumPyro models, Level-2 covariate support, vmap-stacked likelihoods, Collins K priors |
| `scripts/fitting/bayesian_summary_writer.py` | Convert ArviZ InferenceData to MLE-schema flat CSV |
| `scripts/18_bayesian_level2_effects.py` | Forest plots of Level-2 (trauma) betas from posterior NetCDF |
| `cluster/13_bayesian_gpu.slurm` | SLURM dispatcher for hierarchical fits, mirrors `12_mle_gpu.slurm` |
| `cluster/13_bayesian_gpu_m4.slurm` | Separate SLURM with longer walltime + larger memory for M4-LBA |
| `cluster/13_submit_all_bayesian.sh` | Convenience script for parallel dispatch |
| `output/bayesian/` (new directory) | Hierarchical fit outputs |
| `docs/COLLINS_K_BOUNDS.md` | Literature review supporting the constrained K bounds |

### Modified files

| File | Changes |
|------|---------|
| `scripts/fitting/mle_utils.py` | K bounds tightened in all `WMRL_*_BOUNDS` dicts (lines 43, 53, 64, 76, 87, 100). One change point. Affects MLE only — Bayesian K bounds live in `numpyro_models.py`. |
| `scripts/fitting/fit_bayesian.py` | Accept all 7 models (not just qlearning|wmrl). Add `--level2-covariates` arg. Add `--save-individual-summary-csv` arg (default ON). Remove the `BAYESIAN_IMPLEMENTED = {'qlearning', 'wmrl'}` whitelist. |
| `scripts/13_fit_bayesian.py` | CLI wrapper passes through new args. |
| `scripts/14_compare_models.py` | Add `--bayesian-comparison` mode that loads all `output/bayesian/*_posterior.nc` files and runs `arviz.compare`. AIC/BIC mode unchanged. Add per-model "MLE vs Bayesian point estimate" sanity check table. |
| `scripts/15_analyze_mle_by_trauma.py` | Add `--source mle|bayesian` flag (default `mle`). Path resolution: `output/{source}/{csv_name}`. |
| `scripts/16_regress_parameters_on_scales.py` | Same `--source` flag treatment. |
| `scripts/16b_bayesian_regression.py` | Add deprecation comment at top. No code change. |
| `scripts/17_analyze_winner_heterogeneity.py` | Same `--source` flag treatment. |
| `config.py` | Add `bayesian_implemented: bool` and `level2_supported: bool` to `MODEL_REGISTRY`. Add `output/bayesian/` path constant. |
| `cluster/12_mle.slurm`, `cluster/12_mle_gpu.slurm` | No code change, but trigger reruns of all 7 models after K bounds change in P14. |

### Internal boundaries

| Boundary | Communication | Notes |
|----------|---------------|-------|
| `numpyro_models.py` <-> `jax_likelihoods.py` | Direct function call (`*_multiblock_likelihood_stacked`) | The stacked likelihoods already exist for all 7 models, designed for vmap. This is the cleanest integration point. |
| `numpyro_models.py` <-> `lba_likelihood.py` | Direct function call (`wmrl_m4_multiblock_likelihood_stacked`) | MUST be in a separate process. Hard rule. |
| `fit_bayesian.py` <-> `bayesian_summary_writer.py` | Function call after MCMC completes | Writer takes idata + model name, writes to `output/bayesian/{model}_individual_summary.csv` |
| `15/16/17` <-> `output/{mle|bayesian}/` | File system; flat CSV with shared schema | The schema-parity migration cornerstone. |
| `18_bayesian_level2_effects.py` <-> `output/bayesian/{model}_posterior.nc` | ArviZ NetCDF read | Only consumer that touches the full NetCDF directly. |
| `14_compare_models.py` <-> `output/bayesian/*_posterior.nc` | ArviZ `from_netcdf` + `arviz.compare` | Optional, opt-in via `--bayesian-comparison` flag. |

---

## Scaling Considerations (per fit, not per user)

| Sample size | Choice-only hierarchical fit | M4-LBA hierarchical fit |
|-------------|------------------------------|-------------------------|
| n=50 (pilot) | 30 min - 1 hr on A100 | 2-4 hr on A100 |
| n=154 (current cohort) | 4-8 hr on A100, 8-16 hr on V100 | 12-24 hr on A100, 24-48 hr on V100 |
| n=500 (hypothetical) | 12-24 hr on A100 | 36-72 hr on A100 |

### Scaling priorities

1. **First bottleneck: GPU memory.** With n=154, padded `(154 × 17 × 100)` arrays times n_params times n_chains pushes V100 16GB toward OOM. Mitigation: chain_method="sequential" instead of "parallel" (slower but lower memory), or upgrade to A100 40GB.

2. **Second bottleneck: NUTS divergences from poor priors.** As n grows, posterior gets sharper and the non-centered parameterization may need re-tuning. Symptoms: divergence count > 1% of samples. Mitigation: increase `target_accept_prob` from 0.8 to 0.95, increase warmup, re-check that all parameters use non-centered parameterization (some legacy capacity uses centered + clip — that's OK at n=154 but may fail at larger n).

3. **Third bottleneck: walltime.** Single-job 24h walltime is the practical M3 cluster ceiling. For larger n or more complex models, would need to fit in chunks (e.g., split chains across jobs and merge NetCDFs manually with `xarray.concat`).

---

## Sources

- `scripts/fitting/legacy/numpyro_models.py` (the actual file, despite being in `legacy/` — currently the only NumPyro model definition that exists)
- `scripts/fitting/fit_bayesian.py` lines 43-49 (broken import) and 148-155 (BAYESIAN_IMPLEMENTED whitelist)
- `scripts/fitting/jax_likelihoods.py` lines 76-80 (constants), 86-215 (padding utilities), 626/1349/1557/1942/2287/2644 (`*_multiblock_likelihood_stacked` functions ready for vmap)
- `scripts/fitting/lba_likelihood.py` line 11 (`jax.config.update("jax_enable_x64", True)`), lines 53-56/93-95 (explicit float64 casts), line 197 (vmap over accumulators), line 608 (`wmrl_m4_multiblock_likelihood_stacked`)
- `scripts/fitting/mle_utils.py` lines 43, 53, 64, 76, 87, 100 (capacity bounds in all WMRL_*_BOUNDS dicts — single change point for Collins refit)
- `scripts/fitting/fit_mle.py` lines 1224-1483 (`fit_all_gpu`, nested vmap pattern), 2321-2414 (CPU/GPU dispatch logic)
- `scripts/14_compare_models.py` lines 78-89 (uses `MODEL_REGISTRY['wmrl_m4']['params']`)
- `scripts/15_analyze_mle_by_trauma.py` lines 148-174 (hardcoded MLE CSV reads — these become the `--source` flag injection points)
- `scripts/16_regress_parameters_on_scales.py` line 756 (hardcoded `output/mle/{model}_individual_fits.csv` path)
- `scripts/16b_bayesian_regression.py` lines 58-83 (dual-backend pattern: NumPyro on GPU, PyMC on CPU — proven template for new hierarchical scripts), lines 282-299 (NumPyro NUTS pattern with `set_host_device_count`)
- `scripts/17_analyze_winner_heterogeneity.py` lines 57, 73, 114, 143 (uses `MODEL_REGISTRY[model]['csv_filename']` consistently — clean abstraction to extend with `--source`)
- `cluster/12_mle_gpu.slurm` lines 32-38 (resource declarations), 161-188 (parallel dispatch one-job-per-model pattern), 105-118 (`JAX_COMPILATION_CACHE_DIR`)
- `config.py` lines 170-243 (`MODEL_REGISTRY` single source of truth), 249-276 (`PyMCParams`)

---
*Architecture research for: rlwm_trauma_analysis v4.0 hierarchical Bayesian extension*
*Researched: 2026-04-10*
*Confidence: HIGH (all integration points verified by direct file inspection)*
