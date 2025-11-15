# Prior-Based Parameter Space Exploration

Explore how different parameter combinations (sampled from prior distributions) affect model behavior.

## Features

- ✅ **Parallelized**: Use all CPUs with `--n-jobs -1`
- ✅ **Progress bars**: Real-time progress tracking with tqdm
- ✅ **Timers**: Automatic timing of simulations
- ✅ **Comprehensive heatmaps**: Visualize all parameter interactions
- ✅ **Prior sampling**: Uses same priors as PyMC fitting

## Installation

```bash
# Install tqdm for progress bars
pip install tqdm

# Or install all dev requirements
pip install -r requirements-dev.txt
```

## Quick Start

```bash
# Quick test (serial, ~2-3 minutes)
python scripts/simulations/explore_prior_parameter_space.py --model qlearning --n-samples 50 --num-trials 30 --num-reps 2

# Full exploration with parallelization (recommended, ~10-15 minutes with 8 CPUs)
python scripts/simulations/explore_prior_parameter_space.py --model qlearning --n-samples 200 --n-jobs -1

# Both models in parallel (~20-30 minutes with 8 CPUs)
python scripts/simulations/explore_prior_parameter_space.py --model both --n-samples 200 --n-jobs -1
```

## Command-Line Arguments

```bash
python scripts/simulations/explore_prior_parameter_space.py --help
```

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | both | Model to explore: `qlearning`, `wmrl`, or `both` |
| `--n-samples` | int | 200 | Number of parameter sets to sample from priors |
| `--set-sizes` | int+ | [3, 5] | Task set sizes to test |
| `--num-trials` | int | 50 | Trials per simulation |
| `--num-reps` | int | 3 | Repetitions per condition (for averaging) |
| `--seed` | int | 42 | Random seed for reproducibility |
| `--n-jobs` | int | 1 | Parallel jobs: `1` = serial, `-1` = all CPUs |

## Example Output

```
================================================================================
PRIOR-BASED PARAMETER SPACE EXPLORATION
================================================================================

Configuration:
  Model: qlearning
  Prior samples: 200
  Set sizes: [3, 5]
  Trials per simulation: 50
  Repetitions: 3
  Parallel jobs: -1 (all CPUs)
  Random seed: 42

--------------------------------------------------------------------------------
EXPLORING QLEARNING PARAMETER SPACE
--------------------------------------------------------------------------------

Sampling 200 parameter sets from prior distributions...
  Sampled: alpha_pos, alpha_neg, beta

Prior sample statistics:
       alpha_pos  alpha_neg      beta
count     200.000    200.000   200.000
mean        0.501      0.498     1.987
std         0.287      0.289     1.412
min         0.012      0.008     0.112
25%         0.264      0.247     1.021
50%         0.502      0.501     1.765
75%         0.738      0.749     2.654
max         0.988      0.992     7.234

Running simulations for 200 parameter sets...
  Using 8 parallel job(s)
  Starting parallel simulations...
  Simulating: 100%|███████████████████████████| 200/200 [02:34<00:00, 1.29param_set/s]
  Complete! Elapsed time: 2.58 minutes (154.7 seconds)
  Average time per parameter set: 0.77 seconds

Saved results: output/parameter_exploration/qlearning_prior_exploration_n200_seed42.csv

Creating visualizations...
  1. Pairwise parameter heatmaps...
     ✓ Saved: figures/parameter_exploration/qlearning_prior_parameter_heatmaps.png
  2. Marginal parameter effects...
     ✓ Saved: figures/parameter_exploration/qlearning_prior_marginal_effects.png

Performance summary:
  Mean accuracy: 0.652 ± 0.124
  Max accuracy: 0.887
  Min accuracy: 0.312

Best parameter combination found:
  alpha_pos = 0.456
  alpha_neg = 0.123
  beta = 3.214
  → Accuracy = 0.887

================================================================================
EXPLORATION COMPLETE!
================================================================================

Results saved to: output/parameter_exploration
Figures saved to: figures/parameter_exploration/

Total runtime: 3.12 minutes (187.4 seconds)
```

## Outputs

### Data Files

**Location**: `output/parameter_exploration/`

- `{model}_prior_exploration_n{N}_seed{S}.csv` - Full results table

**Columns**:
- Parameter values: `alpha_pos`, `alpha_neg`, `beta`, etc.
- Metrics: `accuracy_mean`, `accuracy_std`
- Metadata: `set_size`, `num_trials`, `num_reps`

### Visualizations

**Location**: `figures/parameter_exploration/`

1. **`{model}_prior_parameter_heatmaps.png`**
   - Multi-panel heatmap showing all pairwise parameter interactions
   - **Q-learning**: 3 heatmaps (alpha+×alpha-, alpha+×beta, alpha-×beta)
   - **WM-RL**: 8 heatmaps (capacity×rho, capacity×phi, rho×phi, etc.)
   - Color scale: Green = high accuracy, Red = low accuracy

2. **`{model}_prior_marginal_effects.png`**
   - Individual parameter effects on accuracy
   - Shows how each parameter affects performance when averaged across other parameters
   - Includes confidence bands (±1 SD)

## Prior Distributions

The script samples from the same prior distributions used in PyMC model fitting:

### Q-Learning
- `alpha_pos ~ Beta(2, 2)` - Positive prediction error learning rate
- `alpha_neg ~ Beta(2, 2)` - Negative prediction error learning rate
- `beta ~ Gamma(2, 1)` - Inverse temperature (exploration vs exploitation)

### WM-RL (Working Memory + RL)
- All Q-learning parameters +
- `beta_wm ~ Gamma(2, 1)` - WM-specific inverse temperature
- `capacity ~ DiscreteUniform(2, 6)` - Working memory capacity
- `phi ~ Beta(2, 2)` - WM decay rate
- `rho ~ Beta(2, 2)` - Base WM reliance

## Performance Tips

### Speed Optimization

1. **Use parallelization** for large n-samples:
   ```bash
   --n-jobs -1  # Use all CPUs (6-8x speedup)
   ```

2. **Reduce trials for exploration**:
   ```bash
   --num-trials 30 --num-reps 2  # Faster, still informative
   ```

3. **Test with small sample first**:
   ```bash
   --n-samples 50  # Quick test before full run
   ```

### Estimated Runtimes

| n-samples | num-trials | n-jobs | Model | Estimated Time |
|-----------|------------|--------|-------|----------------|
| 50 | 30 | 1 | qlearning | ~5 min |
| 50 | 30 | -1 (8 CPUs) | qlearning | ~1 min |
| 200 | 50 | 1 | qlearning | ~25 min |
| 200 | 50 | -1 (8 CPUs) | qlearning | ~3-4 min |
| 200 | 50 | -1 (8 CPUs) | wmrl | ~8-10 min |
| 200 | 50 | -1 (8 CPUs) | both | ~12-15 min |

*Times vary based on CPU speed and number of cores*

## Interpreting Results

### Heatmaps

- **Green regions**: Parameter combinations that produce high accuracy
- **Red regions**: Poor-performing parameter combinations
- **Large green areas**: Model is robust to parameter variation
- **Narrow peaks**: Model is sensitive to specific parameter values

### Key Questions to Answer

1. **Which parameters matter most?**
   - Look at marginal effects plots
   - Parameters with steep slopes = high impact

2. **Are there parameter interactions?**
   - Look at heatmaps for non-diagonal patterns
   - Example: Does optimal alpha_pos depend on beta?

3. **What's the optimal parameter range?**
   - Identify green regions in heatmaps
   - Use for setting informed priors in model fitting

4. **How robust is the model?**
   - Large green regions = robust
   - Narrow peaks = sensitive (harder to fit reliably)

## Next Steps

After running parameter exploration:

1. **Identify promising parameter ranges** from heatmaps
2. **Use insights to set informed priors** for Bayesian fitting
3. **Compare Q-learning vs WM-RL** parameter sensitivity
4. **Run full model fitting** on behavioral data:
   ```bash
   python scripts/fitting/fit_to_data.py --model both
   ```

## Troubleshooting

### "tqdm not installed"
```bash
pip install tqdm
```

### Slow performance
- Use `--n-jobs -1` for parallelization
- Reduce `--n-samples` for testing
- Reduce `--num-trials` and `--num-reps`

### Out of memory
- Reduce `--n-samples`
- Reduce `--n-jobs` (use fewer CPUs)
- Close other applications

### Multiprocessing errors on Windows
- The script uses `if __name__ == "__main__":` guard (required for Windows)
- Make sure you're running the script directly, not importing it

## Related Scripts

- `parameter_sweep.py` - Grid-based parameter sweeps (deterministic)
- `example_parameter_sweep.py` - Quick demo of parameter sweeps
- `visualize_parameter_sweeps.py` - Visualization functions
- `explore_prior_parameter_space.py` - **This script** (prior-based sampling)

## Citation

If you use this for parameter exploration in publications, cite the prior distributions and mention the sampling approach.
