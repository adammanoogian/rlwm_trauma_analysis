# Codebase Concerns

**Analysis Date:** 2026-01-28

## Tech Debt

**Parameter Bounds in Fitting (Risk: Moderate Impact)**
- Issue: Q-learning and WM-RL parameter bounds are identical at 0.001-0.999, but capacity parameter for WM-RL should have stricter bounds (1.0-7.0 as integer)
- Files: `scripts/fitting/mle_utils.py` (WMRL_BOUNDS definition)
- Impact: Capacity fitting may converge to non-integer or out-of-bounds values; model interpretation becomes unclear
- Fix approach: Enforce integer constraint on capacity during optimization; add post-fit rounding with validation

**Data Pipeline Dependency on External Sequences**
- Issue: `TaskSequenceLoader` in `environments/task_config.py` expects sequence files from parent rlwm_trauma project; falls back to synthetic generation if missing
- Files: `environments/task_config.py` (lines 47-54), `environments/rlwm_env.py` (initialization)
- Impact: Exact task replication requires external sequence files; code must handle missing files gracefully but this creates fragility
- Fix approach: Add clear error messaging; document sequence file requirements in README; consider bundling essential sequence files

**Missing Data Path Validation**
- Issue: Config paths assume specific directory structure (data/, output/, figures/); no validation that directories exist before operations
- Files: `config.py` (lines 51-53), `scripts/01_parse_raw_data.py` (DATA_DIR, OUTPUT_DIR)
- Impact: Silent failures if directories missing; unclear where to find input data
- Fix approach: Add startup validation that logs missing paths; create directories with sensible defaults; document expected structure

**Return None in Data Cleaning Functions**
- Issue: Several utility functions return None on failure instead of raising exceptions or logging warnings
- Files: `scripts/01_parse_raw_data.py` (line 64), `scripts/utils/data_cleaning.py` (line 310), `scripts/analysis/parse_surveys_for_mle.py` (lines 204, 218, 232, 237, 251)
- Impact: Failures silently propagate; downstream code may crash with unclear origin; hard to debug incomplete data
- Fix approach: Raise informative exceptions or log warnings instead; validate data at entry points

## Known Bugs

**Participant ID Remapping Inconsistency (Risk: High - Data Integrity)**
- Symptoms: Multiple participant ID schemes (sona_id, assigned_id, anonymous fallback); potential for ID mismatches between datasets
- Files: `scripts/01_parse_raw_data.py` (lines 88-100), `data/participant_id_mapping.json`
- Trigger: Running parsing with incomplete or mismatched participant mapping file
- Workaround: Manually verify participant IDs match between task_trials_long.csv and MLE output; cross-reference with mapping JSON
- Impact: Analysis results may incorrectly match parameters to wrong participants, invalidating statistical conclusions

**Set Size Box Space Definition Issue**
- Symptoms: `observation_space` defines set_size as continuous Box(1,) but values are discrete (2, 3, 5, 6)
- Files: `environments/rlwm_env.py` (lines 95-100)
- Trigger: Type validation or type-aware code that expects Box observations to be truly continuous
- Workaround: Use integer casting when accessing set_size from observation
- Impact: Low impact in practice since set_size is numeric, but violates gym space contracts and may cause issues with type-strict RL libraries

**Reversal Counter Reset Logic (Risk: Low - Behavioral)**
- Symptoms: In main_task phase, reversal_points set to np.inf after first reversal (line 378 in rlwm_env.py) preventing additional reversals on same stimulus per block
- Files: `environments/rlwm_env.py` (lines 356-381)
- Trigger: Tasks with multiple reversals per stimulus in main phase
- Workaround: Design tasks to expect at most 1 reversal per stimulus per block
- Impact: Matches experimental design intent but could cause surprises if task structure changes

## Security Considerations

**Unvalidated Data Paths (Risk: Low)**
- Risk: File paths from DataParams config could allow directory traversal if combined with user input
- Files: `config.py` (DataParams paths), `scripts/01_parse_raw_data.py` (file loading)
- Current mitigation: Paths are hardcoded in config; no user input in path construction
- Recommendations: Continue avoiding string concatenation with user input; use Path.resolve() to normalize; add assertions that output goes to expected directories

**Participant Survey Data Privacy (Risk: Medium - If Shared)**
- Risk: Parsed survey data contains trauma scores and participant IDs; inadvertent sharing could compromise privacy
- Files: `output/parsed_survey2.csv`, `output/mle/participant_surveys.csv`
- Current mitigation: Files in .gitignore; not tracked in version control
- Recommendations: Add clear data governance documentation; verify all PHI/sensitive files excluded from git; consider de-identification by default

## Performance Bottlenecks

**JAX Compilation Overhead (Risk: Development Experience)**
- Problem: First call to JAX-compiled likelihood functions incurs JIT compilation overhead (seconds per fit)
- Files: `scripts/fitting/jax_likelihoods.py`, `scripts/fitting/fit_mle.py` (objective functions)
- Cause: jax.jit() applied to multiblock_likelihood; compiles independently for each data shape
- Improvement path: Cache compiled functions with functools.lru_cache; pre-compile during module load; document expected warmup time

**Memory Usage During MLE Fitting (Risk: Scales with Participants)**
- Problem: fit_mle.py loads all participant data simultaneously; 20 random starts × N_participants optimizations in memory
- Files: `scripts/fitting/fit_mle.py` (lines 320-400, fit all participants loop)
- Cause: No parallel processing or streaming; all optimization results held in memory
- Improvement path: Add batch processing with generator; implement checkpoint-restart for large N; add memory monitoring

**Q-Table Memory for Large Stimulus Sets (Risk: Low - Task Limited)**
- Problem: Q-table allocated as (MAX_STIMULI, NUM_ACTIONS) = (6, 3) = 18 floats; scales quadratically with future expansion
- Files: `models/q_learning.py` (line 102), `models/wm_rl_hybrid.py` (lines 132-136)
- Cause: Dense matrix allocation even for small set sizes
- Improvement path: Only allocate Q[current_set_size, :] per block; resize on reset; negligible impact for current scales

**Pandas Operations in Data Pipeline (Risk: Medium - Scales Quadratically)**
- Problem: Data aggregation uses groupby().apply() and concatenation patterns that can be slow for large CSV files
- Files: `scripts/01_parse_raw_data.py`, `scripts/02_create_collated_csv.py`, `scripts/03_create_task_trials_csv.py`
- Cause: Multiple reads/writes of same data; repeated filtering and reshaping
- Improvement path: Vectorize groupby operations; use built-in aggregation methods; consider polars for larger future datasets

## Fragile Areas

**Experiment Participant Exclusion Logic (Files: Fragile)**
- Files: `config.py` (EXCLUDED_PARTICIPANTS hardcoded list, lines 27-44)
- Why fragile: Exclusion criteria hardcoded by participant ID; adding new exclusions requires config edit; no versioning of exclusion decisions
- Safe modification: Create exclusion criteria as predicates (min_trials, min_accuracy); compute dynamically from data; document rationale in EXCLUSION_REASONS
- Test coverage: Needs validation that excluded participants properly removed from all downstream analyses

**Model Parameter Constraints (Implementation Risk)**
- Files: `scripts/fitting/mle_utils.py` (bounded_to_unbounded, unbounded_to_bounded functions)
- Why fragile: Logit transform can numerically fail at boundaries (logit(0.001) or logit(0.999) approaches infinity); no error handling
- Safe modification: Add epsilon buffer (0.001 → 1e-6 or 0.999 - 1e-6); test corner cases; validate that unconstrained space covers parameter ranges
- Test coverage: Need unit tests for boundary values

**Reversal Point Initialization (Environment State)**
- Files: `environments/rlwm_env.py` (lines 330-354, _initialize_reversals method)
- Why fragile: Reversal criteria depend on TaskParams.RARE_REVERSALS flag; different phase types have hardcoded reversal values; coupling between phases
- Safe modification: Parameterize reversal settings at environment creation; create reversalConfig class; validate that reversal_points initialized before use
- Test coverage: Test all phase types; verify reversals occur at expected times; validate state doesn't leak between resets

## Scaling Limits

**Participant Sample Size (Current: ~60, Observed Limit: ~500 manageable)**
- Current capacity: MLE fitting pipeline processes ~54 participants in ~1 hour on single CPU
- Limit: At ~500+ participants, JAX compilation and optimization time becomes bottleneck; memory usage ~2GB per full run
- Scaling path: Implement parallel optimization (joblib, ray); batch JAX compilations; add incremental fitting with checkpoint-restart; consider GPU acceleration for likelihood evaluation

**Block Structure Scaling (Current: 23 blocks, Design Limit: ~50)**
- Current capacity: Environment handles 23 blocks with 30-90 trials per block efficiently
- Limit: Beyond 50 blocks, data structures (episode_stimuli, episode_correct history lists) cause memory pressure
- Scaling path: Use numpy arrays with pre-allocation; implement sliding window history; batch history writes to disk

**Trial-Level Data Storage (Current: ~60k trials, Limit: ~10M trials)**
- Current capacity: task_trials_long.csv loads into pandas comfortably (~100MB disk, ~200MB RAM)
- Limit: At 10M+ trials (1000+ participants × 23 blocks × 400 trials), CSV becomes unwieldy
- Scaling path: Switch to parquet or hdf5; implement lazy loading; chunk by participant; use dask for parallel processing

## Dependencies at Risk

**NumPy/JAX Version Mismatch (Risk: Low - Pinned)**
- Risk: JAX likelihood functions use jnp operations that changed in JAX 0.4+; requires numpy>=1.24
- Impact: Code fails silently with wrong type conversions if older JAX installed
- Migration plan: Document exact JAX version requirements; add version checks at runtime; consider static type checking with mypy

**PyMC Ecosystem Transition (Risk: Medium - Under Development)**
- Risk: Code includes both PyMC3 (fitting/pymc_models.py) and newer backends; PyMC3 going into maintenance mode
- Impact: Support and security updates may lag; new features only in PyMC v5+
- Migration plan: Phase out PyMC3 code; consolidate on numpyro/JAX; deprecation period of 2-3 releases

**Gymnasium vs Gym (Risk: Low - Already Transitioned)**
- Risk: Code uses gymnasium (gym successor) but older gym may still be installed in some environments
- Impact: Import errors if only gym installed; environ check in rlwm_env.py (line 14)
- Migration plan: Explicitly require gymnasium in setup.py/requirements.txt; add clear error message if gym detected

## Missing Critical Features

**Validation/Testing Framework (Risk: High - Code Quality)**
- Problem: No comprehensive test suite for core models (q_learning.py, wm_rl_hybrid.py); existing tests scattered in legacy/
- Blocks: Cannot confidently refactor model code; hard to verify model matches Senta et al. (2025) specifications
- Impact: Risk of subtle model bugs remaining undetected; prevents future optimization
- Recommendation: Implement unit tests for all model methods; add model recovery validation; create integration tests for fitting pipeline

**Likelihood Function Validation Against Reference (Risk: Medium - Model Correctness)**
- Problem: JAX likelihoods not independently verified against analytical solutions or reference implementations
- Blocks: Uncertainty if likelihoods exactly match paper specifications; parameter recovery tests are only indirect validation
- Impact: Parameter estimates may be biased if likelihood wrong
- Recommendation: Add test_jax_likelihoods.py with known parameter cases; compare against R/Stan reference; add gradient checks with finite differences

**Data Quality Report (Risk: Medium - Transparency)**
- Problem: No automated data quality checks; parsing produces no summary of data issues (missing values, outliers, invalid ranges)
- Blocks: Cannot assess data reliability; hard to document limitations
- Impact: Downstream analyses may be compromised by bad data; hard to publish methodology
- Recommendation: Add data_quality.py that generates HTML report with: trial counts per participant, accuracy distributions, RT distributions, missing values, exclusion summaries

## Test Coverage Gaps

**Environment Reset and State Isolation (Untested Area)**
- What's not tested: Whether reversal_points properly reset between episodes; whether stimulus sampling is truly random; whether block transitions work correctly
- Files: `environments/rlwm_env.py` (reset, step, _initialize_reversals methods)
- Risk: State leakage between episodes could cause silent test failures in learning curves
- Priority: High (foundational for all downstream models)
- Fix: Create test_rlwm_env.py with: 100 resets checking state independence, 1000 steps verifying stimulus distribution, block boundary validation

**Asymmetric Learning Rates (Untested Area)**
- What's not tested: Whether Q-learning correctly applies alpha_pos vs alpha_neg; whether prediction error sign correctly determined; whether learning curves differ between models
- Files: `models/q_learning.py` (update method, lines 188-242)
- Risk: Parameter recovery fitting could converge to wrong values if learning logic broken
- Priority: High (critical model component)
- Fix: Create test_q_learning.py with: synthetic data with known alpha_pos>alpha_neg and verify recovery; unit test PE sign detection

**WM-RL Hybrid Weight Computation (Untested Area)**
- What's not tested: Whether omega = ρ * min(1, K/N_s) correctly implemented; whether WM decay applies before/after update correctly; whether normalization of hybrid probabilities maintains sum=1
- Files: `models/wm_rl_hybrid.py` (get_adaptive_weight, get_hybrid_probs, update methods)
- Risk: Hybrid weight calculation errors could invalidate WM-RL fits; decay timing wrong could bias parameter recovery
- Priority: High (WM-specific logic)
- Fix: Create test_wm_rl_hybrid.py with: parameter sweep validating omega bounds; verify WM decay followed by reward overwrite; test probability normalization across conditions

**Parameter Bounds and Constrained Optimization (Untested Area)**
- What's not tested: Whether logit/inverse-logit transforms correctly map bounded ↔ unbounded spaces; whether optimization respects bounds; corner cases (p=0.001, p=0.999)
- Files: `scripts/fitting/mle_utils.py` (bounded_to_unbounded, unbounded_to_bounded functions)
- Risk: Parameters may converge outside valid ranges; logit approaching infinity could cause NaN
- Priority: Medium (affects fitting stability)
- Fix: Create test_mle_utils.py with: roundtrip tests for all parameter ranges; verify transforms are monotonic; test corner values

**MLE Convergence Validation (Untested Area)**
- What's not tested: Whether multiple random starts actually improve solution quality; whether best solution selection (by NLL) is stable; whether convergence flags properly indicate success
- Files: `scripts/fitting/fit_mle.py` (fit_participant function, lines 175-260)
- Risk: Reporting "converged" solution that's actually local minimum; unclear if fitting actually worked
- Priority: High (end-to-end fitting quality)
- Fix: Add validation that best_result.success==True; implement heuristic that if <5/20 starts converge, flag as unreliable; store all start solutions for inspection

**Block-Aware Data Preparation (Untested Area)**
- What's not tested: Whether prepare_block_data() correctly splits multi-block participant data; whether Q-values reset between blocks; whether likelihoods computed per-block correctly
- Files: `scripts/fitting/jax_likelihoods.py` (prepare_block_data, multiblock_likelihood functions)
- Risk: If block boundaries mishandled, likelihood biased; parameter estimates systematically wrong
- Priority: High (fundamental to fitting)
- Fix: Create test_jax_likelihoods.py with: synthetic multi-block data with known parameters; verify likelihood increases monotonically with more data; test with varying block counts

---

*Concerns audit: 2026-01-28*
