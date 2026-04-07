---
created: 2026-04-07T14:30
title: Re-fit all models on cluster + recovery + PPC
area: cluster
files:
  - scripts/fitting/fit_mle.py:1444
  - scripts/fitting/model_recovery.py:288
  - cluster/submit_full_pipeline.sh
---

## Problem

Three bugs were fixed that invalidate all prior cluster results:

1. **argmin NaN propagation** (`fit_mle.py:1444`): `jnp.argmin` picked NaN as the
   "best" NLL, discarding 40-51% of WM-RL participants who had 49/50 valid starts.
   M6b had only 75/154 valid participants. Fixed: NaN replaced with inf before argmin.

2. **Stimulus sampling** (`model_recovery.py:288`): Synthetic data always used 3
   stimuli regardless of block set_size (2/3/5/6). All prior recovery results are
   invalid. Fixed: `rng.integers(0, set_size)` per block.

3. **Reward mapping** (`model_recovery.py:288,467`): `rng.permutation(set_size)`
   produced unreachable actions (3,4,5) for set_size 5/6. Fixed:
   `rng.integers(0, NUM_ACTIONS, size=set_size)`.

Additionally, `fit_mle.py` now saves `{model}_all_start_nlls.csv` with all 50 NLLs
per participant for multi-start diagnostics.

## Solution

Run full pipeline on cluster with all fixes:

```bash
# Full pipeline: fit all 7 models + recovery + PPC + analysis
bash cluster/submit_full_pipeline.sh

# Or if only re-fitting (skip recovery/PPC for speed):
bash cluster/submit_full_pipeline.sh --skip-wave2
```

After re-fit completes, re-run local analyses:
```bash
python scripts/14_compare_models.py
python scripts/15_analyze_mle_by_trauma.py --model all
python scripts/16_regress_parameters_on_scales.py --model all
```

Expected improvements:
- All 7 models should have ~154 valid participants (not 75-97)
- Aggregate AIC comparison valid on matched N
- Recovery results valid for all models (M5 should now pass)
- All-start NLLs saved for diagnostic analysis
