# Model Performance Visualization

Visualize winning model performance with trial-by-trial accuracy plots and performance breakdowns.

## Overview

This module provides tools to:
1. **Generate predictions** from fitted models on behavioral data
2. **Visualize learning curves** showing accuracy over trials
3. **Analyze performance by trial position** (early/late, pre/post reversal)
4. **Compare performance across set sizes**

## Files

- `visualize_model_performance.py` - Plotting functions
- `plot_winning_model.py` - Main script to generate predictions and plots
- `test_performance_plots.py` - Test script with mock data

## Quick Start

### 1. Plot Winning Model (Auto-Selected)

Automatically selects the best-fitting model based on BIC:

```bash
python scripts/analysis/plot_winning_model.py
```

This will:
- Load the model with lowest BIC from `output/fitting/`
- Generate trial-by-trial predictions on behavioral data
- Create 4 visualizations (see below)
- Save predictions to `output/model_performance/`

### 2. Plot Specific Model

```bash
# Plot Q-learning model
python scripts/analysis/plot_winning_model.py --model qlearning

# Plot WM-RL model
python scripts/analysis/plot_winning_model.py --model wmrl
```

### 3. Custom Settings

```bash
python scripts/analysis/plot_winning_model.py \
  --model auto \
  --criterion AIC \
  --threshold 5 \
  --data data/processed/task_trials_long.csv \
  --posterior-dir output/fitting
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | str | `data/processed/task_trials_long.csv` | Path to behavioral data |
| `--posterior-dir` | str | `output/fitting` | Directory with fitted model posteriors |
| `--model` | str | `auto` | Model to plot: `auto`, `qlearning`, or `wmrl` |
| `--criterion` | str | `BIC` | Criterion for auto-selection: `BIC`, `AIC`, `WAIC`, `LOO` |
| `--threshold` | int | `4` | Trial threshold for early/late classification |
| `--seed` | int | `42` | Random seed for reproducibility |

## Outputs

### Predictions Data

**Location**: `output/model_performance/{model}_predictions.csv`

**Columns**:
- `subject_id`, `block`, `trial` - Identifiers
- `trial_num` - Trial number within block (1-indexed)
- `set_size` - Set size condition (2, 3, 5, or 6)
- `stimulus` - Presented stimulus
- `actual_response` - Participant's response
- `actual_correct` - Whether participant was correct
- `model_choice` - Model's predicted choice
- `model_correct` - Whether model prediction was correct (0/1)
- `trials_since_reversal` - Trials since reversal occurred
- `is_post_reversal` - Whether trial is after reversal (boolean)

### Visualizations

**Location**: `figures/model_performance/`

#### 1. Combined Performance Analysis
**File**: `{model}_performance_analysis.png`

Two-panel figure:
- **Left panel**: Learning curve (accuracy over trials since block start)
- **Right panel**: Performance by trial position (bar chart)

#### 2. Learning Curve (Since Block Start)
**File**: `{model}_learning_curve_since_start.png`

Line graph showing accuracy progression from trial 1 onward:
- **Y-axis**: Accuracy (%)
- **X-axis**: Trial number since block start
- **Lines**: One line per set size (color-coded)
- **Bands**: ±1 SEM confidence bands
- **Horizontal line**: 50% chance level

**What to look for**:
- Does accuracy improve over trials? (learning)
- Do different set sizes show different learning rates?
- Is there a plateau or continuing improvement?

#### 3. Learning Curve (Since Reversal)
**File**: `{model}_learning_curve_since_reversal.png`

Line graph showing accuracy after contingency reversal:
- **Y-axis**: Accuracy (%)
- **X-axis**: Trials since reversal
- **Lines**: One line per set size (color-coded)
- **Bands**: ±1 SEM confidence bands

**What to look for**:
- Does accuracy drop immediately after reversal?
- How quickly does the model re-learn?
- Does set size affect reversal learning speed?

#### 4. Performance by Trial Position
**File**: `{model}_performance_by_position.png`

Bar chart showing accuracy in different trial contexts:

**Trial Positions** (default threshold = 4 trials):
1. **Early Block** (< 4 trials since start)
2. **Late Block** (≥ 4 trials since start, before reversal)
3. **Early Post-Reversal** (< 4 trials since reversal)
4. **Late Post-Reversal** (≥ 4 trials since reversal)

**Bars**: Grouped by set size within each position
- Error bars: ±1 SEM
- Value labels on bars show exact accuracy

**What to look for**:
- **Early vs Late Block**: Is performance better after initial learning?
- **Reversal effect**: Does accuracy drop in "Early Post-Reversal"?
- **Recovery**: Does accuracy recover in "Late Post-Reversal"?
- **Set size interactions**: Do effects differ by working memory load?

## Interpreting Results

### Expected Patterns

**Q-Learning Model**:
- Gradual learning over trials (exponential approach to asymptote)
- Performance drop immediately after reversal
- Recovery depends on learning rates (α+, α-)
- Set size may affect noise but not learning mechanism
- Higher β (inverse temperature) → steeper learning curves

**WM-RL Hybrid Model**:
- Faster initial learning (especially for low set sizes within capacity)
- Performance advantage for set sizes ≤ capacity (K)
- Potential "catastrophic forgetting" after reversal if WM is overwritten
- Recovery may be faster for low set sizes (WM can re-learn quickly)
- Set size interactions more pronounced than Q-learning

### Key Comparisons

1. **Learning Speed**:
   - Slope of learning curve in "Early Block"
   - Steeper slope = faster learning

2. **Asymptotic Performance**:
   - Accuracy in "Late Block"
   - Ceiling performance level

3. **Reversal Adaptation**:
   - Accuracy drop from "Late Block" to "Early Post-Reversal"
   - Larger drop = more disruption from reversal

4. **Recovery Speed**:
   - Slope from "Early Post-Reversal" to "Late Post-Reversal"
   - Steeper slope = faster re-learning

5. **Working Memory Effects**:
   - Performance difference between set sizes
   - Diverging lines in learning curves = set size-dependent learning

## Example Workflow

```bash
# 1. Fit models to data (if not already done)
python scripts/fitting/fit_to_data.py --model both

# 2. Compare models
python scripts/analysis/run_model_comparison.py

# 3. Plot winning model performance
python scripts/analysis/plot_winning_model.py --model auto --criterion BIC

# 4. Or plot specific model
python scripts/analysis/plot_winning_model.py --model wmrl
```

## Testing Without Fitted Models

Use the test script with simulated data:

```bash
python scripts/analysis/test_performance_plots.py
```

This generates mock predictions with realistic patterns:
- Learning over trials
- Reversal effects
- Set size effects

Useful for:
- Verifying plotting functions work
- Understanding expected output format
- Designing custom analyses

## Using Plotting Functions Directly

```python
from scripts.analysis.visualize_model_performance import (
    plot_learning_curves,
    plot_performance_by_trial_position,
    plot_combined_performance_analysis
)
import pandas as pd

# Load predictions
predictions_df = pd.read_csv('output/model_performance/qlearning_predictions.csv')

# Create learning curve
plot_learning_curves(
    predictions_df,
    trial_type='since_start',
    save_path='my_learning_curve.png',
    title='My Custom Title'
)

# Create performance breakdown
plot_performance_by_trial_position(
    predictions_df,
    n_trials_threshold=5,  # Custom threshold
    save_path='my_performance_breakdown.png'
)

# Combined analysis
plot_combined_performance_analysis(
    predictions_df,
    n_trials_threshold=4,
    save_dir=Path('my_figures'),
    model_name='My Model'
)
```

## Data Requirements

The prediction DataFrame must have these columns:
- `trial_num` - Trial number within block (1-indexed)
- `set_size` - Set size condition
- `correct` or `model_correct` - Binary accuracy (0/1)
- `trials_since_reversal` - Trials since reversal
- `is_post_reversal` - Boolean indicating post-reversal trials

Optional columns for additional context:
- `subject_id`, `block`, `trial` - Identifiers
- `stimulus`, `response` - Trial details

## Customization

### Change Trial Threshold

Default is 4 trials for early/late classification. Adjust with `--threshold`:

```bash
# Use 3 trials as threshold
python scripts/analysis/plot_winning_model.py --threshold 3

# Use 5 trials as threshold
python scripts/analysis/plot_winning_model.py --threshold 5
```

### Custom Data Source

```bash
python scripts/analysis/plot_winning_model.py \
  --data my_custom_data.csv \
  --posterior-dir my_custom_posteriors/
```

### Select Model by Different Criterion

```bash
# Use AIC instead of BIC
python scripts/analysis/plot_winning_model.py --criterion AIC

# Use WAIC (Bayesian)
python scripts/analysis/plot_winning_model.py --criterion WAIC

# Use LOO (Leave-One-Out)
python scripts/analysis/plot_winning_model.py --criterion LOO
```

## Troubleshooting

### "Fitted model not found"

Make sure you've run model fitting first:
```bash
python scripts/fitting/fit_to_data.py --model both
```

### "Model comparison not found"

The script will default to Q-learning if comparison file doesn't exist. To generate comparison:
```bash
python scripts/analysis/run_model_comparison.py
```

### Empty or Missing Data

Check that behavioral data file exists and has required columns:
- `subject_id`, `block`, `trial`
- `set_size`, `stimulus`, `response`, `correct`

### Plots Don't Match Expectations

1. Verify model parameters are reasonable (check `{model}_predictions.csv`)
2. Check overall accuracy is above chance
3. Verify reversal detection is working (check `is_post_reversal` column)
4. Try different threshold values if early/late split seems off

## Related Scripts

- `fit_to_data.py` - Fit models to behavioral data
- `run_model_comparison.py` - Compare models with information criteria
- `explore_prior_parameter_space.py` - Parameter space exploration
- `visualize_parameter_sweeps.py` - Parameter sweep visualizations

## Citation

When using these visualizations in publications, consider reporting:
1. Model selection criterion and value (e.g., "BIC = 1234.5")
2. Trial threshold for early/late classification
3. Number of trials/blocks/subjects analyzed
4. Overall accuracy and performance by condition
