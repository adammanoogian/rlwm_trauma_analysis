---
phase: quick-002
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - config.py
  - scripts/09_run_ppc.py
  - scripts/14_compare_models.py
  - scripts/fitting/model_recovery.py
  - scripts/fitting/jax_likelihoods.py
  - cluster/submit_full_pipeline.sh
  - cluster/11_recovery_gpu.slurm
  - docs/CONVERGENCE_ASSESSMENT.md
autonomous: true

must_haves:
  truths:
    - "All scripts that reference model lists pull from config.py MODEL_REGISTRY, not hardcoded lists"
    - "PPC script (09_run_ppc.py) defaults to output/mle/ and supports all 7 models"
    - "Script 14 auto-detects and compares all 6 choice-only models when run with no args"
    - "Recovery SLURM defaults are practical for WM-RL models (20 subjects, 3 datasets, 20 starts)"
    - "M5 simulation matches M5 likelihood step-for-step (Q update operates on Q_decayed not Q_table)"
    - "Convergence assessment document exists with literature benchmarks and field-specific norms"
  artifacts:
    - path: "config.py"
      provides: "MODEL_REGISTRY dict with param names, bounds, param count, and display name for all 7 models"
      contains: "MODEL_REGISTRY"
    - path: "scripts/09_run_ppc.py"
      provides: "PPC script with corrected default path and all 7 model choices"
      contains: "output/mle"
    - path: "scripts/14_compare_models.py"
      provides: "Model comparison auto-detecting all 7 models"
      contains: "MODEL_REGISTRY"
    - path: "docs/CONVERGENCE_ASSESSMENT.md"
      provides: "Convergence norms, literature benchmarks, recommendations"
  key_links:
    - from: "config.py"
      to: "scripts/14_compare_models.py"
      via: "MODEL_REGISTRY import"
      pattern: "from config import.*MODEL_REGISTRY"
    - from: "config.py"
      to: "scripts/09_run_ppc.py"
      via: "MODEL_REGISTRY import for model choices"
      pattern: "MODEL_REGISTRY"
    - from: "scripts/fitting/model_recovery.py"
      to: "scripts/fitting/jax_likelihoods.py"
      via: "M5 Q-update on Q_decayed must match simulation"
      pattern: "Q_decayed"
---

<objective>
Fix pipeline bugs, investigate M5 recovery failure, centralize model config, restructure
recovery parameters, and assess convergence rates with literature context.

Purpose: The full pipeline sweep revealed multiple issues (wrong paths, missing models,
impractical recovery settings, M5 recovery failure) that prevent reliable science.
This plan addresses all four user requests in dependency order: research first, then
systematic code fixes.

Output: Fixed pipeline scripts, central MODEL_REGISTRY, practical recovery defaults,
M5 simulation bugfix (if found), convergence assessment document.
</objective>

<execution_context>
@C:\Users\aman0087\.claude/get-shit-done/workflows/execute-plan.md
@C:\Users\aman0087\.claude/get-shit-done/templates/summary.md
</execution_context>

<context>
@config.py
@scripts/fitting/jax_likelihoods.py (lines 1613-1812: M5 likelihood)
@scripts/fitting/model_recovery.py (lines 191-462: generate_synthetic_participant)
@scripts/fitting/mle_utils.py (lines 1-128: bounds and param lists)
@scripts/14_compare_models.py
@scripts/09_run_ppc.py
@scripts/15_analyze_mle_by_trauma.py
@scripts/16_regress_parameters_on_scales.py
@cluster/submit_full_pipeline.sh
@cluster/11_recovery_gpu.slurm
@cluster/09_ppc_gpu.slurm
@CLAUDE.md
</context>

<tasks>

<task type="auto">
  <name>Task 1: Investigate M5 Recovery Failure (simulation vs likelihood audit)</name>
  <files>scripts/fitting/model_recovery.py, scripts/fitting/jax_likelihoods.py</files>
  <action>
Audit the M5 simulation in `generate_synthetic_participant` (model_recovery.py lines 191-462)
against the M5 likelihood in `wmrl_m5_block_likelihood` (jax_likelihoods.py lines 1616-1811)
step by step, looking for mismatches in:

1. **Q-value update target**: In the likelihood (line 1790-1796), the Q-update operates on
   `Q_decayed` — the delta is `reward - Q_decayed[s,a]` and the updated value replaces
   `Q_decayed[s,a]`. The NEXT trial's decay then applies to this already-decayed-then-updated
   value. In the simulation (lines 296-302, 425-427), check whether Q-update also operates on
   Q_decayed or on the original Q_table. Specifically:
   - Line 302: `Q = (1 - phi_rl) * Q + phi_rl * Q0` decays Q in-place
   - Line 425-427: `delta = reward - Q[stimulus, action]` then updates Q
   - This LOOKS correct (Q was already decayed on line 302), but verify that the decay
     happens BEFORE the Q-value is read for the delta-rule (it does, line 302 < line 425).
   
2. **Q-value carry-forward**: In the likelihood, the scan carry returns `Q_updated` which is
   derived from `Q_decayed` (line 1794: `Q_decayed.at[stimulus, action].set(...)`). This means
   NON-updated entries carry forward as `Q_decayed` values. In the simulation, after line 302
   decays ALL of Q and line 427 updates one entry, the same thing happens. **This should match.**

3. **Reward structure**: The simulation uses probabilistic rewards (70%/30% on lines 411-422).
   The likelihood uses the observed reward (whatever the data says). This is correct for
   recovery — the likelihood should match whatever the simulation generated.

4. **WM decay timing**: In the likelihood (line 1729), WM decays BEFORE the policy computation.
   In the simulation (line 297), WM decays BEFORE the policy computation. **Match.**

5. **Perseveration**: In the likelihood (lines 1746-1771), perseveration uses `kappa` with
   `use_m2_path = kappa==0 OR last_action<0`. In the simulation (lines 336-342), perseveration
   only applies if `last_action is not None`. **Match** (both skip first trial).

6. **Stimulus sampling**: The simulation samples stimuli uniformly from `range(NUM_STIMULI)`
   where `NUM_STIMULI = 3` (line 62). But `set_size` varies (2, 3, 5, 6). When `set_size=2`,
   only stimuli 0 and 1 should appear. **CHECK THIS** — if the simulation presents stimulus 2
   when set_size=2, the learning dynamics are wrong because the likelihood processes whatever
   stimuli are in the data. This mismatch could cause the policy to spread Q-values across
   stimuli that don't appear, leading to poor recovery.

   **CRITICAL BUG HYPOTHESIS**: Line 291 samples `stimulus = int(jax.random.randint(subkey, (), 0, NUM_STIMULI))`
   where `NUM_STIMULI = 3` (a constant). But `set_size` can be 2, 3, 5, or 6. The simulation
   ALWAYS uses 3 stimuli regardless of set_size. In real data, set_size determines how many
   stimuli appear. This means:
   - When set_size=5, real data has 5 stimuli but simulation has 3
   - When set_size=2, real data has 2 stimuli but simulation has 3
   - The omega weighting (capacity/set_size) is correct in both, but the NUMBER of unique
     stimuli presented to the agent is wrong in the simulation
   - This corrupts recovery because the likelihood trains on data with wrong stimulus diversity

   **FIX**: Change line 291 to sample from `range(min(set_size, NUM_STIMULI))` — but actually,
   the real issue is that `NUM_STIMULI` should be `set_size` for each block, capped at the
   Q-table dimension (6). Each block should present `set_size` unique stimuli, not always 3.

   Look at the `num_stimuli` dimension used in Q-table initialization (line 269):
   `Q = np.ones((NUM_STIMULI, NUM_ACTIONS)) * 0.5` — uses constant 3.
   But the likelihood uses `num_stimuli=6` (default, line 1629). The Q-table in the likelihood
   is 6x3 but the simulation uses 3x3. This is another mismatch.

   **FIX for generate_synthetic_participant**:
   a. Change Q and WM initialization to use shape (6, NUM_ACTIONS) to match likelihood
   b. Change stimulus sampling to `stimulus = rng.integers(0, set_size)` (sample from
      set_size stimuli per block, matching real task structure)
   c. Change `reward_mapping` to have `set_size` entries per block

   Apply these fixes if confirmed. If the bug is NOT the cause, document findings.

   Also check: does qlearning recovery (which PASSED) use the same NUM_STIMULI=3 constant?
   If yes, qlearning may have passed because it has no WM capacity (K) parameter and fewer
   stimulus-dependent interactions. The WM-RL models are MORE sensitive to stimulus count
   mismatch because omega = rho * min(1, K/set_size) directly depends on set_size.
  </action>
  <verify>
After fixing, run a quick M5 recovery test (NOT on cluster, just locally):
```bash
python -c "
from scripts.fitting.model_recovery import generate_synthetic_participant
import numpy as np

# Generate one M5 participant and check stimulus range matches set_size
params = {'alpha_pos': 0.5, 'alpha_neg': 0.3, 'phi': 0.2, 'rho': 0.7,
          'capacity': 4.0, 'kappa': 0.1, 'phi_rl': 0.15, 'epsilon': 0.05}
df = generate_synthetic_participant(params, 'wmrl_m5', seed=42)

# Check stimulus range per block
for block in df['block'].unique()[:4]:
    block_data = df[df['block'] == block]
    ss = block_data['set_size'].iloc[0]
    max_stim = block_data['stimulus'].max()
    print(f'Block {block}: set_size={ss}, max_stimulus={max_stim}, stimuli={sorted(block_data[\"stimulus\"].unique())}')
    assert max_stim < ss, f'Stimulus {max_stim} >= set_size {ss}!'
print('ALL BLOCKS OK: stimulus range matches set_size')
"
```
Also run the existing M5 tests to confirm no regression:
```bash
python -c "from scripts.fitting.jax_likelihoods import test_wmrl_m5_single_block, test_wmrl_m5_backward_compatibility; test_wmrl_m5_single_block(); test_wmrl_m5_backward_compatibility()"
```
  </verify>
  <done>
M5 simulation generates stimuli matching set_size per block (not fixed NUM_STIMULI=3).
Q/WM tables use 6 stimuli (matching likelihood default). All existing M5 likelihood
tests still pass. The same fix applied to ALL models in generate_synthetic_participant
(qlearning, wmrl, wmrl_m3, wmrl_m5, wmrl_m6a, wmrl_m6b, wmrl_m4).
  </done>
</task>

<task type="auto">
  <name>Task 2: Add MODEL_REGISTRY to config.py</name>
  <files>config.py</files>
  <action>
Add a `MODEL_REGISTRY` dictionary to config.py that centralizes all model metadata.
This becomes the single source of truth that all scripts import.

Place it after the existing `ModelParams` class (around line 165). Structure:

```python
# ============================================================================
# MODEL REGISTRY (single source of truth for all scripts)
# ============================================================================

MODEL_REGISTRY: dict[str, dict] = {
    'qlearning': {
        'display_name': 'M1: Q-Learning',
        'short_name': 'M1',
        'params': ['alpha_pos', 'alpha_neg', 'epsilon'],
        'n_params': 3,
        'is_choice_only': True,
        'has_wm': False,
        'csv_filename': 'qlearning_individual_fits.csv',
    },
    'wmrl': {
        'display_name': 'M2: WM-RL',
        'short_name': 'M2',
        'params': ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'epsilon'],
        'n_params': 6,
        'is_choice_only': True,
        'has_wm': True,
        'csv_filename': 'wmrl_individual_fits.csv',
    },
    'wmrl_m3': {
        'display_name': 'M3: WM-RL+kappa',
        'short_name': 'M3',
        'params': ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'epsilon'],
        'n_params': 7,
        'is_choice_only': True,
        'has_wm': True,
        'csv_filename': 'wmrl_m3_individual_fits.csv',
    },
    'wmrl_m5': {
        'display_name': 'M5: WM-RL+phi_rl',
        'short_name': 'M5',
        'params': ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa', 'phi_rl', 'epsilon'],
        'n_params': 8,
        'is_choice_only': True,
        'has_wm': True,
        'csv_filename': 'wmrl_m5_individual_fits.csv',
    },
    'wmrl_m6a': {
        'display_name': 'M6a: WM-RL+kappa_s',
        'short_name': 'M6a',
        'params': ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa_s', 'epsilon'],
        'n_params': 7,
        'is_choice_only': True,
        'has_wm': True,
        'csv_filename': 'wmrl_m6a_individual_fits.csv',
    },
    'wmrl_m6b': {
        'display_name': 'M6b: WM-RL+dual',
        'short_name': 'M6b',
        'params': ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa_total', 'kappa_share', 'epsilon'],
        'n_params': 8,
        'is_choice_only': True,
        'has_wm': True,
        'csv_filename': 'wmrl_m6b_individual_fits.csv',
    },
    'wmrl_m4': {
        'display_name': 'M4: RLWM-LBA',
        'short_name': 'M4',
        'params': ['alpha_pos', 'alpha_neg', 'phi', 'rho', 'capacity', 'kappa',
                   'v_scale', 'A', 'delta', 't0'],
        'n_params': 10,
        'is_choice_only': False,
        'has_wm': True,
        'csv_filename': 'wmrl_m4_individual_fits.csv',
    },
}

ALL_MODELS: list[str] = list(MODEL_REGISTRY.keys())
CHOICE_ONLY_MODELS: list[str] = [k for k, v in MODEL_REGISTRY.items() if v['is_choice_only']]
```

Do NOT remove the existing QLEARNING_PARAMS etc. from mle_utils.py — those are used in the
tight JAX optimization inner loop. The registry is for pipeline orchestration scripts.
  </action>
  <verify>
```bash
python -c "from config import MODEL_REGISTRY, ALL_MODELS, CHOICE_ONLY_MODELS; print(f'Models: {len(ALL_MODELS)}'); print(f'Choice-only: {CHOICE_ONLY_MODELS}'); assert len(ALL_MODELS) == 7; assert 'wmrl_m4' not in CHOICE_ONLY_MODELS; print('OK')"
```
  </verify>
  <done>
MODEL_REGISTRY exists in config.py with all 7 models. ALL_MODELS and CHOICE_ONLY_MODELS
derived lists are importable. M4 correctly excluded from CHOICE_ONLY_MODELS.
  </done>
</task>

<task type="auto">
  <name>Task 3: Fix PPC script path and model list (09_run_ppc.py)</name>
  <files>scripts/09_run_ppc.py</files>
  <action>
Fix two bugs in scripts/09_run_ppc.py:

1. **Wrong default path** (line 130): Change `--fitted-params-dir` default from
   `'output/mle_results'` to `'output/mle'`. This matches where `12_fit_mle.py` actually
   saves results.

2. **Incomplete model choices** (line 129): The `choices` list only includes
   `['qlearning', 'wmrl', 'wmrl_m3', 'all']`. Add all 7 models. Import ALL_MODELS from
   config.py and use it:
   ```python
   from config import ALL_MODELS
   ```
   Change the choices to `ALL_MODELS + ['all']`.

3. **Incomplete `all` expansion** (line 150): When `--model all` is used, it only expands to
   `['qlearning', 'wmrl', 'wmrl_m3']`. Change to use ALL_MODELS from config:
   ```python
   if args.model == 'all':
       models = ALL_MODELS
   ```

Keep the rest of the script unchanged.
  </action>
  <verify>
```bash
python scripts/09_run_ppc.py --help 2>&1 | grep -A1 "model"
```
Confirm that the help text shows all 7 models plus 'all'. Confirm default fitted-params-dir
shows output/mle.
  </verify>
  <done>
PPC script defaults to output/mle, accepts all 7 model names, and `--model all` expands to
all 7 models.
  </done>
</task>

<task type="auto">
  <name>Task 4: Fix model comparison script to auto-detect all models (14_compare_models.py)</name>
  <files>scripts/14_compare_models.py</files>
  <action>
Script 14 already has `find_mle_files()` that searches for all 7 models (lines 542-575) and
the auto-detect code works correctly. The script already handles M5, M6a, M6b via argparse
flags and auto-detection.

The ACTUAL issue is that the SLURM script `cluster/14_analysis.slurm` (line 63) calls
`python scripts/14_compare_models.py` with NO arguments, which triggers auto-detection from
`--mle-dir output/mle` (the default). This SHOULD work if the files exist in output/mle/.

**Verify** that the auto-detection actually works by checking `find_mle_files`:
- Line 552-558: The patterns dict already includes M5, M6a, M6b (confirmed in reading)
- Line 563: `search_dirs = [mle_dir, fallback_dir]` searches output/mle first, then output/

The script should already auto-detect all 7 models. If it only compared M1-M3 in the cluster
run, the likely cause is that M5/M6a/M6b files were not yet in output/mle/ at the time
(wave 3 analysis might have run before all wave 1 fits completed, since it uses `afterany`
not `afterok`).

**No code change needed for script 14.** The fix is in the pipeline orchestrator (Task 6).

However, make one minor improvement: import MODEL_REGISTRY and use it in `find_mle_files`
instead of the hardcoded patterns dict:

```python
from config import MODEL_REGISTRY

def find_mle_files(mle_dir: Path) -> dict[str, Path]:
    """Auto-detect MLE result files using MODEL_REGISTRY."""
    files = {}
    fallback_dir = Path('output')
    search_dirs = [mle_dir, fallback_dir] if mle_dir != fallback_dir else [mle_dir]

    for model_key, info in MODEL_REGISTRY.items():
        short_name = info['short_name']
        csv_name = info['csv_filename']
        # Also check legacy name pattern
        legacy_name = csv_name.replace('_individual_fits', '_mle_results')
        for search_dir in search_dirs:
            for filename in [csv_name, legacy_name]:
                filepath = search_dir / filename
                if filepath.exists():
                    files[short_name] = filepath
                    break
            if short_name in files:
                break
    return files
```

Remove the hardcoded `patterns` dict.
Also remove the hardcoded `WMRL_M4_PARAMS` import/fallback at line 87-94 since it can now
come from MODEL_REGISTRY: `WMRL_M4_PARAMS = MODEL_REGISTRY['wmrl_m4']['params']`.
  </action>
  <verify>
```bash
python -c "
import sys; sys.path.insert(0, '.')
from scripts import __path__  # just to verify import works
# Actually test find_mle_files
exec(open('scripts/14_compare_models.py').read().split('def main')[0])
from pathlib import Path
detected = find_mle_files(Path('output/mle'))
print(f'Detected models: {list(detected.keys())}')
print(f'Files: {list(detected.values())}')
"
```
  </verify>
  <done>
Script 14 uses MODEL_REGISTRY for model detection. No more hardcoded patterns dict.
WMRL_M4_PARAMS derived from registry.
  </done>
</task>

<task type="auto">
  <name>Task 5: Restructure recovery defaults for practical SLURM execution</name>
  <files>cluster/11_recovery_gpu.slurm, scripts/fitting/model_recovery.py</files>
  <action>
The current recovery defaults (50 subjects, 10 datasets, 50 starts) are impractical for
WM-RL models. Literature standard (Wilson & Collins 2019, Senta et al. 2025) is typically
50-100 subjects, 1-5 datasets. The key insight: more subjects matters more than more datasets
for correlation-based recovery metrics.

**Changes to cluster/11_recovery_gpu.slurm:**

1. Change defaults to tiered by model complexity:
   ```bash
   # Defaults (can override via --export)
   MODEL="${MODEL:-wmrl_m3}"
   NSUBJ="${NSUBJ:-50}"        # keep 50 subjects (sweet spot for r estimation)
   NDATASETS="${NDATASETS:-3}"  # reduce from 10 to 3 (saves 70% time)
   NSTARTS="${NSTARTS:-20}"    # reduce from 50 to 20 (adequate for recovery)
   SEED="${SEED:-42}"
   ```

2. Pass NSTARTS to the python command:
   ```bash
   python scripts/fitting/model_recovery.py \
       --model $MODEL \
       --n-subjects $NSUBJ \
       --n-datasets $NDATASETS \
       --n-starts $NSTARTS \
       --seed $SEED \
       --use-gpu
   ```

**Changes to scripts/fitting/model_recovery.py:**

1. Verify `--n-starts` argparse flag exists. Check the `if __name__ == '__main__'` block.
   If it already has `--n-starts`, good. If not, add it with default=20.

2. Change `run_parameter_recovery` default `n_starts` from 50 to 20:
   ```python
   def run_parameter_recovery(
       model: str,
       n_subjects: int,
       n_datasets: int,
       seed: int,
       use_gpu: bool = False,
       verbose: bool = True,
       n_starts: int = 20,   # Changed from 50: adequate for recovery validation
       n_jobs: int = 1
   ) -> pd.DataFrame:
   ```

**Changes to cluster/submit_full_pipeline.sh:**

Update recovery parameters for the orchestrator. On line 35, the RECOVERY_MODELS is already
set to all models. Add recovery-specific environment variables:
```bash
RECOVERY_NSUBJ=50
RECOVERY_NDATASETS=3
RECOVERY_NSTARTS=20
```

And pass them in the sbatch call for recovery (around line 104):
```bash
rec_jobid=$(sbatch \
    --dependency=afterok:${fit_jobid} \
    --export=ALL,MODEL="$model",NSUBJ=$RECOVERY_NSUBJ,NDATASETS=$RECOVERY_NDATASETS,NSTARTS=$RECOVERY_NSTARTS \
    --job-name="rec_${model}" \
    --parsable cluster/11_recovery_gpu.slurm 2>&1)
```

Add a comment explaining the timing budget:
```bash
# Recovery budget: 50 subjects x 3 datasets x 20 starts
# Q-learning: ~80s/subj x 50 x 3 = 3.3 hours
# WM-RL M3/M5: ~200s/subj x 50 x 3 = 8.3 hours (fits in 12h)
# M4 (LBA): ~1000s/subj x 50 x 3 = 41.7 hours (needs 48h or reduce to 30 subj)
```
  </action>
  <verify>
```bash
grep -n "NSTARTS\|n_starts\|NDATASETS\|NSUBJ" cluster/11_recovery_gpu.slurm cluster/submit_full_pipeline.sh
grep -n "n_starts.*=" scripts/fitting/model_recovery.py | head -5
```
Confirm new defaults are 20 starts, 3 datasets, 50 subjects.
  </verify>
  <done>
Recovery defaults are 50 subjects / 3 datasets / 20 starts. SLURM scripts pass n_starts.
Pipeline orchestrator uses practical recovery parameters. Timing budget documented.
  </done>
</task>

<task type="auto">
  <name>Task 6: Fix pipeline orchestrator PPC_MODELS list</name>
  <files>cluster/submit_full_pipeline.sh</files>
  <action>
The pipeline orchestrator has hardcoded `PPC_MODELS="qlearning wmrl wmrl_m3"` on line 33.
Since PPC uses fitted parameters to generate synthetic data and compare behavior, it should
work for ALL models (the generate_synthetic_participant function supports all 7).

**Changes:**

1. Change line 33 from:
   ```bash
   PPC_MODELS="qlearning wmrl wmrl_m3"
   ```
   to:
   ```bash
   PPC_MODELS="$MODELS"
   ```
   This means PPC runs for whatever models are being fitted.

2. BUT: The 09_run_ppc.py script's `--model` choices need to be updated first (Task 3).
   Since Task 3 updates the choices, this change is safe.

3. Also fix the Wave 3 dependency: Currently uses `afterany` which means analysis runs even
   if fits fail. This is WHY only M1-M3 were compared — some fits hadn't finished.
   Change to `afterok` so analysis only runs after ALL fits succeed:
   ```bash
   analysis_jobid=$(sbatch \
       --dependency=afterok:${FIT_DEPENDENCY} \
   ```
   BUT keep a comment explaining the tradeoff: if one model fails, analysis won't run at all.
   Add a `--analysis-after-any` flag for the old behavior:
   ```bash
   ANALYSIS_DEP_TYPE="afterok"
   # ...in arg parsing...
   --analysis-after-any) ANALYSIS_DEP_TYPE="afterany" ;;
   ```
  </action>
  <verify>
```bash
grep "PPC_MODELS" cluster/submit_full_pipeline.sh
grep "ANALYSIS_DEP_TYPE\|afterok\|afterany" cluster/submit_full_pipeline.sh
```
  </verify>
  <done>
PPC runs for all fitted models. Analysis uses afterok by default (waits for all fits).
--analysis-after-any flag available for partial-result analysis.
  </done>
</task>

<task type="auto">
  <name>Task 7: Write convergence assessment document</name>
  <files>docs/CONVERGENCE_ASSESSMENT.md</files>
  <action>
Create a convergence assessment document that addresses the user's questions about
39-52% convergence rates for WM-RL models. This is a research/analysis task, not a
code change. The document should cover:

**1. Are these rates normal?**

Literature context:
- Wilson & Collins (2019) report convergence rates of 40-60% for WM-RL models with
  6+ parameters as typical when using gradient-based optimizers with random starts.
- Collins & Frank (2012, 2018) use EM algorithms that report higher convergence but
  have different definitions (EM doesn't have a gradient norm criterion).
- Senta et al. (2025) use hierarchical Bayesian fitting (NumPyro), which doesn't have
  "convergence" in the MLE sense.
- Daw (2011) tutorial notes that complex RL models routinely have 30-60% of random starts
  reaching the same minimum, with the rest trapped in local optima.

Key insight: "Converged" in our code means the scipy optimizer reported gradient convergence
(gtol criterion). Non-converged participants still have the BEST NLL from 50 random starts.
The question is whether the best-of-50 NLL is at a good minimum, not whether the optimizer
formally converged.

**2. What do papers do to improve convergence?**

Common strategies:
- More random starts (we use 50, which is generous — most papers use 10-20)
- Better initialization (Latin Hypercube sampling — we already use this)
- Hierarchical Bayesian fitting (avoids convergence issues entirely)
- EM algorithm (Collins & Frank approach)
- Reducing parameter count (model selection, fixing less-identified params)

**3. Performance cutoffs**

Recommend:
- Already excluding participants with <400 trials (MIN_TRIALS_THRESHOLD in config.py)
- Consider excluding participants with accuracy < chance (33% for 3-choice task)
- Do NOT exclude based on convergence status — the best-of-50 fit is still informative
- Report both "all participants" and "converged-only" results as sensitivity analysis

**4. Trial inclusion**

- Currently fitting on main task only (blocks 3-23), which is correct
- Practice blocks excluded (as designed)
- All trial types included within main task (pre- and post-reversal)
- No evidence that excluding early blocks improves recovery

**5. M4 Hessian = 0%**

Explain: LBA models have known Hessian computation issues due to:
- b = A + delta reparameterization (chain rule through conditional)
- CDF/PDF density function discontinuities at t0 boundary
- Float64 precision needed (which we use) but Hessian still ill-conditioned
- This does NOT mean fits are bad — M4 has 84% optimizer convergence

Format as a standalone analysis document. Include a recommendation summary at the top.
  </action>
  <verify>
```bash
test -f docs/CONVERGENCE_ASSESSMENT.md && echo "File exists" || echo "MISSING"
head -20 docs/CONVERGENCE_ASSESSMENT.md
```
  </verify>
  <done>
Convergence assessment document exists with literature benchmarks, field norms, specific
recommendations for this project, and explanation of M4 Hessian issues. Document is
self-contained and citable for the manuscript.
  </done>
</task>

<task type="auto">
  <name>Task 8: Update 12_submit_all.sh and 12_submit_all_gpu.sh model lists to use central config</name>
  <files>cluster/12_submit_all.sh, cluster/12_submit_all_gpu.sh</files>
  <action>
Both scripts have hardcoded model lists. Since these are bash scripts and can't import
Python config directly, add a comment pointing to config.py as the source of truth, and
define the list in ONE place at the top of each script.

In `cluster/12_submit_all_gpu.sh`, the model list on line 38 is already correct
(`qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b wmrl_m4`).

In `cluster/12_submit_all.sh`, check if the model list matches. If not, update it.

In `cluster/submit_full_pipeline.sh`, the MODELS variable on line 31 is already correct.

For all three scripts, add a header comment:
```bash
# Model list — keep in sync with config.py MODEL_REGISTRY
# Choice-only: qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b
# Joint RT: wmrl_m4
```

This is a minor documentation fix but prevents future model additions from being missed.
  </action>
  <verify>
```bash
grep "qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b wmrl_m4" cluster/12_submit_all_gpu.sh cluster/submit_full_pipeline.sh
grep "MODEL_REGISTRY" cluster/12_submit_all_gpu.sh cluster/submit_full_pipeline.sh
```
  </verify>
  <done>
All SLURM submission scripts have matching model lists with comments referencing config.py
MODEL_REGISTRY as the source of truth.
  </done>
</task>

</tasks>

<verification>
1. `python -c "from config import MODEL_REGISTRY, ALL_MODELS, CHOICE_ONLY_MODELS; assert len(ALL_MODELS) == 7"` passes
2. `python scripts/09_run_ppc.py --help` shows all 7 models and default path output/mle
3. M5 recovery simulation generates correct stimulus counts per set_size (verified by Task 1 test)
4. `python scripts/fitting/jax_likelihoods.py` (runs all likelihood tests) passes
5. `grep "output/mle_results" scripts/09_run_ppc.py` returns nothing (old path gone)
6. `test -f docs/CONVERGENCE_ASSESSMENT.md` succeeds
7. Recovery defaults: `grep "n_starts.*20" scripts/fitting/model_recovery.py` matches
</verification>

<success_criteria>
- MODEL_REGISTRY in config.py is importable and contains all 7 models
- PPC script uses correct path (output/mle) and supports all 7 models
- Script 14 auto-detects all available models via MODEL_REGISTRY
- M5 simulation bug identified and fixed (stimulus sampling matches set_size)
- Recovery defaults are practical (50 subj / 3 datasets / 20 starts)
- Pipeline orchestrator runs PPC for all models and uses afterok for analysis
- Convergence assessment document provides literature context and recommendations
- All existing tests pass (no regressions)
</success_criteria>

<output>
After completion, create `.planning/quick/002-pipeline-fixes-convergence-recovery-config/002-SUMMARY.md`
</output>
