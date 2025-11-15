# RLWM Task Parameters

Summary of task parameters from actual experimental data and configuration.

## Task Structure (from actual data)

### Block Organization
- **Practice blocks**: 2 blocks (blocks 1-2)
  - Block 1: Static practice (no reversals)
  - Block 2: Dynamic practice (with reversals)
- **Main task blocks**: 21 blocks (blocks 3-23)
- **Total blocks**: 23

### Trials per Block

**From actual experimental data:**
- **Mean**: 58 trials per block
- **Median**: 45 trials per block
- **Range**: 30-90 trials per block
- **Distribution**:
  - 30 trials: ~29% of blocks
  - 45 trials: ~24% of blocks
  - 75 trials: ~29% of blocks
  - 90 trials: ~18% of blocks

**For simulations:**
- **Default**: 45 trials (median, good balance)
- **Quick tests**: 30 trials (minimum)
- **Full envelope**: 100 trials (covers max + buffer)

### Set Sizes

**From actual experimental data:**
- **Set sizes used**: [2, 3, 5, 6]
- **Excluded**: 4 (per experimental design)

**Working memory load classification:**
- **Low load**: Set size ≤ 3 (sizes 2, 3)
- **High load**: Set size ≥ 4 (sizes 5, 6)

## Reversal Parameters

**From config.py:**
- **Reversal criterion**: 12-18 consecutive correct responses
  - `REVERSAL_MIN = 12`
  - `REVERSAL_MAX = 18`
- **Reversal type**: Rare, late reversals
- **Max reversals**: 1 per stimulus per block

**Practice block (dynamic):**
- **Criterion**: 5 consecutive correct
- **Required**: 2 reversals detected

## Stimulus and Response

**Stimuli:**
- **Maximum unique stimuli**: 6
- **Active per block**: Depends on set size (2, 3, 5, or 6)

**Responses:**
- **Actions**: 3 (mapped to J, K, L keys → 0, 1, 2)
- **Feedback**: Binary (correct=1, incorrect=0)
- **Timeout**: 2000ms maximum response time

## Timing (reference only)

*Note: Timing is not simulated in models, but included for reference*

- **Fixation**: 500ms
- **Trial duration**: 2000ms (maximum)
- **Feedback**: 500ms
- **Total trial duration**: ~3000ms

## Reward Structure

- **Correct response**: +1 point
- **Incorrect response**: 0 points
- **No negative rewards**

## Usage in Simulations

### Default Configuration

```python
from config import TaskParams

# Create environment with actual task parameters
env = create_rlwm_env(
    set_size=3,  # or any from [2, 3, 5, 6]
    phase_type='main_task',
    max_trials_per_block=TaskParams.TRIALS_PER_BLOCK_MEDIAN,  # 45
    seed=42
)
```

### Parameter Exploration

```bash
# Use actual task parameters (recommended)
python scripts/simulations/explore_prior_parameter_space.py \
    --model both \
    --n-samples 200 \
    --set-sizes 2 3 5 6 \  # All set sizes from actual task
    --num-trials 45 \        # Median from actual task
    --n-jobs -1
```

### Quick Test

```bash
# Faster test with minimal trials
python scripts/simulations/explore_prior_parameter_space.py \
    --model qlearning \
    --n-samples 20 \
    --set-sizes 2 3 \        # Subset of set sizes
    --num-trials 30 \        # Minimum from actual task
    --n-jobs -1
```

## Summary Table

| Parameter | Value | Source |
|-----------|-------|--------|
| **Block Organization** |
| Total blocks | 23 | Actual data |
| Practice blocks | 2 (blocks 1-2) | Actual data |
| Main task blocks | 21 (blocks 3-23) | Actual data |
| **Trials per Block** |
| Mean | 58 trials | Actual data |
| Median | 45 trials | Actual data |
| Range | 30-90 trials | Actual data |
| Simulation default | 45 trials | Median |
| **Set Sizes** |
| Available set sizes | [2, 3, 5, 6] | Actual data |
| Excluded | [4] | Actual data |
| Low load | ≤ 3 (sizes 2, 3) | Config |
| High load | ≥ 4 (sizes 5, 6) | Config |
| **Reversals** |
| Criterion range | 12-18 consecutive | Config |
| Type | Rare, late | Config |
| Max per stimulus | 1 per block | Config |
| **Responses** |
| Number of actions | 3 (J, K, L) | Config |
| Reward correct | +1 | Config |
| Reward incorrect | 0 | Config |
| Response timeout | 2000ms | Config |

## References

- **Configuration**: `config.py` → `TaskParams` class
- **Actual data**: `output/task_trials_long.csv`
- **Environment**: `environments/rlwm_env.py`

## Notes

1. **Why 45 trials?**
   - Median of actual experimental data (30-90 range)
   - Good balance: representative but efficient for simulations
   - Allows ~2-4 reversals per stimulus (at 12-18 consecutive correct)

2. **Why all 4 set sizes?**
   - Matches actual experimental design
   - Tests both low load (2, 3) and high load (5, 6) conditions
   - Critical for WM-RL model which adapts based on set size

3. **Reversal frequency:**
   - With 45 trials and 12-18 criterion, expect ~2-3 reversals per block
   - Matches "rare, late reversals" design
   - Sufficient for RL agents to learn stimulus-action mappings

4. **For faster testing:**
   - Use 30 trials (minimum observed)
   - Use subset of set sizes (e.g., [2, 5] for low/high contrast)
   - Reduce n_samples (e.g., 20 instead of 200)
