# Research: Parameter Recovery for RLWM Models

**Phase:** 05-parameter-recovery
**Researched:** 2026-02-06
**Confidence:** HIGH

---

## Executive Summary

Parameter recovery is a fundamental validation method in computational cognitive modeling that tests whether a model fitting procedure can accurately recover known parameter values from synthetic data. Following Senta et al. (2025) methodology, this phase implements a parameter recovery pipeline to validate MLE fitting quality using the criterion r >= 0.80 for correlation between true and recovered parameters.

**Key Findings:**
1. Parameter recovery with r > 0.8 is the standard criterion in computational modeling
2. Recovery quality improves with trial count: ~0.8 for 100 trials, ~0.93 for 300 trials
3. Standard metrics: Pearson r, RMSE, and bias (mean difference)
4. Visual validation requires scatter plots (true vs recovered) and distribution overlays (KDE)
5. Synthetic data must match real data trial structure exactly for valid comparison

**Implementation Approach:**
- Sample true parameters uniformly from MLE bounds
- Generate synthetic datasets matching real task structure (21 blocks, reversals)
- Fit via MLE using existing fit_participant_mle() function
- Compute recovery metrics per parameter
- Visualize with scatter plots + regression lines and KDE overlays
- Report pass/fail against r >= 0.80 criterion

---

## 1. Parameter Recovery Methodology

### 1.1 Definition and Purpose

**Parameter recovery** assesses whether a model fitting procedure can accurately estimate the true parameters that generated synthetic data. This validates:
- Correct model implementation
- Sufficient data quantity for parameter estimation
- Parameter identifiability (parameters are not redundant)
- Optimization procedure reliability

**Source:** [Parameter and Model Recovery of Reinforcement Learning Models](https://link.springer.com/article/10.1007/s42113-022-00139-0)

### 1.2 Standard Procedure

Following best practices in computational cognitive modeling:

1. **Sample True Parameters:** Draw parameter values from plausible ranges (uniform sampling from MLE bounds)
2. **Generate Synthetic Data:** Simulate behavior from the model using true parameters
3. **Fit Model:** Apply MLE fitting procedure to recover parameters
4. **Compare True vs Recovered:** Compute correlation, RMSE, bias
5. **Evaluate Criterion:** Check if Pearson r >= 0.80 for all parameters

**Source:** [Ten simple rules for the computational modeling of behavioral data](https://pmc.ncbi.nlm.nih.gov/articles/PMC6879303/)

### 1.3 The r >= 0.80 Criterion

**Standard Threshold:** Parameter recovery with Pearson correlation r > 0.8 is widely accepted as indicating adequate parameter identifiability and fitting quality.

**Empirical Evidence from RL Models:**
- r ~ 0.8 with 100 trials
- r ~ 0.93 with 300 trials
- This project: 400-1000 trials per participant (expect r > 0.8)

**Interpretation:**
- r >= 0.80: Parameters are recoverable, model is properly identified
- r < 0.80: Insufficient data, parameter confounding, or optimization issues
- Context-dependent: With large N or strong effects, r > 0.8 may be sufficient; with small N or weak effects, higher r may be needed

**Source:** [Parameter and Model Recovery for Restless Bandit Problems](https://www.biorxiv.org/content/10.1101/2021.10.27.466089v1)

### 1.4 Senta et al. (2025) Approach

Jennifer D. Senta, Sonia J. Bishop, and Anne G.E. Collins verified parameter identifiability in their RLWM model by:
1. Simulating data with fixed parameter values
2. Assessing accuracy of recovered parameters
3. Performing model recovery to confirm the generative model best fits its own data

**Source:** [Dual process impairments in RL and WM systems](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012872)

---

## 2. Recovery Metrics

### 2.1 Pearson Correlation (r)

**Primary Metric:** Correlation between true and recovered parameters.

**Interpretation:**
- r = 1.0: Perfect recovery
- r >= 0.80: Adequate recovery (standard threshold)
- r < 0.80: Poor recovery, investigate causes

**Formula:**
```
r = Σ[(x_i - x̄)(y_i - ȳ)] / √[Σ(x_i - x̄)² × Σ(y_i - ȳ)²]
```

Where x = true parameters, y = recovered parameters.

**Implementation:** `scipy.stats.pearsonr(true_params, recovered_params)`

### 2.2 Root Mean Square Error (RMSE)

**Measures:** Average magnitude of recovery error.

**Formula:**
```
RMSE = √[Σ(true_i - recovered_i)² / N]
```

**Interpretation:**
- Lower RMSE = better recovery
- Units match parameter scale
- Sensitive to large errors (squared differences)

**Use Case:** Identifies parameters with systematic large errors even if correlation is high.

**Source:** [RMSE is not enough: Guidelines to robust data-model comparisons](https://www.sciencedirect.com/science/article/pii/S1364682621000857)

### 2.3 Bias (Mean Difference)

**Measures:** Systematic over- or under-estimation.

**Formula:**
```
Bias = mean(recovered - true)
```

**Interpretation:**
- Bias = 0: Unbiased recovery
- Bias > 0: Systematic overestimation
- Bias < 0: Systematic underestimation

**Use Case:** Detects optimizer bias (e.g., parameters drifting toward bounds).

**Source:** [A Strategy for Using Bias and RMSE as Outcomes in Monte Carlo Studies](https://digitalcommons.wayne.edu/cgi/viewcontent.cgi?article=2938&context=jmasm)

### 2.4 Complementary Metrics

These three metrics complement each other:
- **Pearson r:** Overall linear relationship
- **RMSE:** Magnitude of errors
- **Bias:** Direction of systematic errors

**Example Interpretation:**
- High r + Low RMSE + Low Bias: Excellent recovery
- High r + High RMSE: Good correlation but large variability
- High r + High Bias: Systematic offset (slope ≠ 1)
- Low r + Low RMSE: Poor correlation despite small errors (non-linear relationship)

---

## 3. Synthetic Data Generation

### 3.1 Principle: Match Real Data Structure

**Critical Requirement:** Synthetic data must match real task structure exactly:
- Same number of trials per block (real data: 30-90 trials, mean=58, median=45)
- Same block structure (21 main task blocks)
- Same reversal schedule (reversals at 12-18 consecutive correct)
- Same stimulus set sizes (2, 3, 5, 6)
- Same action space (3 responses)

**Rationale:** Ensures recovered parameters are comparable to real fitted parameters. Mismatched structure introduces artifacts.

**Source:** [An Introduction to Good Practices in Cognitive Modeling](https://scite.ai/reports/an-introduction-to-good-practices-2X63Z1)

### 3.2 Parameter Sampling Strategy

**Uniform Sampling from MLE Bounds:**

For Q-learning (M1):
- alpha_pos: (0.001, 0.999)
- alpha_neg: (0.001, 0.999)
- epsilon: (0.001, 0.999)

For WM-RL (M2):
- alpha_pos, alpha_neg, phi, rho: (0.001, 0.999)
- capacity: (1.0, 7.0)
- epsilon: (0.001, 0.999)

For WM-RL M3 (with perseveration):
- Same as M2 plus kappa: (0.0, 1.0)

**Sampling Method:** Use numpy uniform sampling within bounds to ensure even coverage of parameter space.

**Avoid:** Sampling from narrow distributions (e.g., normal around defaults) as this doesn't test full parameter space.

### 3.3 Behavioral Simulation

**Workflow per synthetic participant:**
1. Sample true parameters uniformly from bounds
2. Initialize model state (Q-tables, WM matrices)
3. For each block:
   - Reset state at block boundary
   - For each trial:
     - Present stimulus
     - Compute action probabilities (softmax with beta=50)
     - Apply epsilon noise
     - Sample action stochastically
     - Generate reward (from task environment or simplified reward function)
     - Update model state (Q-learning and/or WM updates)
4. Store trial-level data in DataFrame matching task_trials_long.csv structure

**Source:** Existing `scripts/simulations/generate_synthetic_data.py` provides template.

### 3.4 Data Format Requirements

**Output DataFrame must include:**
- `sona_id`: Participant ID (synthetic IDs: 90001, 90002, ...)
- `block`: Block number (3-23 for main task)
- `trial_in_block`: Trial index within block
- `stimulus`: Stimulus index (0-5)
- `key_press`: Action index (0-2)
- `reward`: Reward (0 or 1)
- `set_size`: Set size for this block (2, 3, 5, or 6)

**Critical:** Must match exact column names and data types from real data for compatibility with fit_participant_mle().

---

## 4. Visualization Best Practices

### 4.1 Scatter Plots: True vs Recovered

**Purpose:** Primary visualization showing parameter recovery quality.

**Design:**
- X-axis: True parameter values
- Y-axis: Recovered parameter values
- Identity line: y = x (perfect recovery)
- Regression line: Fitted line showing actual relationship
- Annotations: Display r, RMSE, bias on plot

**Implementation:** Use matplotlib or seaborn.

**Example (seaborn):**
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.regplot(x=true_params, y=recovered_params,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red'})
plt.plot([min_val, max_val], [min_val, max_val],
         'k--', label='Perfect recovery')
plt.xlabel('True parameter')
plt.ylabel('Recovered parameter')
plt.title(f'{param_name}: r={r:.3f}, RMSE={rmse:.3f}, Bias={bias:.3f}')
```

**Source:** [seaborn.regplot documentation](https://seaborn.pydata.org/generated/seaborn.regplot.html)

### 4.2 Distribution Overlays (KDE)

**Purpose:** Compare distribution of recovered parameters to real fitted parameters as sanity check.

**Design:**
- Overlapping KDE plots
- Real fitted parameters (from actual data)
- Recovered parameters (from synthetic data)
- Visual check: Distributions should overlap if simulation is realistic

**Implementation:**
```python
import seaborn as sns

sns.kdeplot(real_fitted_params, label='Real data fits', fill=True, alpha=0.3)
sns.kdeplot(recovered_params, label='Recovered from synthetic', fill=True, alpha=0.3)
plt.xlabel(f'{param_name}')
plt.ylabel('Density')
plt.legend()
plt.title(f'{param_name} Distribution Comparison')
```

**Interpretation:**
- Overlap: Synthetic data produces realistic parameter distributions
- No overlap: Check simulation procedure or parameter sampling

**Source:** [Visualizing statistical relationships - seaborn](https://seaborn.pydata.org/tutorial/relational.html)

### 4.3 Additional Diagnostic Plots

**AIC/BIC Comparison:**
- Compare distribution of AIC/BIC from real fits vs synthetic fits
- Should have similar ranges if synthetic data is realistic
- Boxplots or histograms

**Convergence Diagnostics:**
- Plot convergence rate (% successful optimizations)
- Plot n_near_best distribution
- Identifies if synthetic data is too easy/hard to fit

---

## 5. Implementation Strategy

### 5.1 Script Organization

**Primary Script:** `scripts/fitting/model_recovery.py`

**CLI Interface:**
```bash
python scripts/fitting/model_recovery.py --model wmrl_m3 --n-subjects 50 --n-datasets 10 --use-gpu
```

**Arguments:**
- `--model`: Model to test (qlearning, wmrl, wmrl_m3)
- `--n-subjects`: Number of synthetic participants per dataset
- `--n-datasets`: Number of independent datasets to generate
- `--use-gpu`: Enable GPU acceleration (optional)
- `--output`: Output directory (default: output/recovery/{model}/)
- `--seed`: Random seed for reproducibility

**Wrapper Script:** `scripts/11_run_model_recovery.py`
- Imports model_recovery module
- Runs recovery for all models
- Evaluates r >= 0.80 criterion
- Prints PASS/FAIL for each model

### 5.2 Recovery Loop Workflow

```python
def run_parameter_recovery(model, n_subjects, n_datasets, seed):
    results = []

    for dataset_idx in tqdm(range(n_datasets), desc="Datasets"):
        # 1. Sample true parameters for this dataset
        true_params_list = sample_parameters_uniformly(model, n_subjects, seed + dataset_idx)

        # 2. Generate synthetic data
        synthetic_data = generate_synthetic_data(model, true_params_list)

        # 3. Fit model to synthetic data (sequential, use existing fit_participant_mle)
        for subj_idx in tqdm(range(n_subjects), desc=f"Dataset {dataset_idx+1}", leave=False):
            true_params = true_params_list[subj_idx]
            subj_data = synthetic_data[synthetic_data['sona_id'] == 90000 + subj_idx]

            # Prepare data in block format
            data_dict = prepare_participant_data(subj_data, 90000 + subj_idx, model)

            # Fit via MLE (reuse existing function)
            fit_result = fit_participant_mle(**data_dict, model=model, n_starts=50)

            # Store true vs recovered
            results.append({
                'dataset': dataset_idx,
                'subject': subj_idx,
                **{f'true_{p}': true_params[p] for p in param_names},
                **{f'recovered_{p}': fit_result[p] for p in param_names},
                'nll': fit_result['nll'],
                'converged': fit_result['converged']
            })

    return pd.DataFrame(results)
```

### 5.3 GPU vs CPU Considerations

**Decision (from 05-CONTEXT.md):** Sequential GPU fitting preferred over CPU parallel.

**Rationale:**
- CPU parallel with joblib causes LLVM compilation issues under memory pressure
- GPU sequential is faster than CPU parallel for WMRL models
- Avoids compilation overhead from multiple worker processes

**Implementation:**
- Pass `--use-gpu` flag to enable JAX GPU backend
- Use sequential fitting loop (no joblib)
- Add tqdm progress bar for user feedback

### 5.4 Output Organization

**Data Outputs:** `output/recovery/{model}/`
- `recovery_results.csv`: Wide-format table with all true/recovered values
  - Columns: dataset, subject, true_alpha_pos, recovered_alpha_pos, ..., nll, converged
- `recovery_metrics.csv`: Summary metrics per parameter
  - Columns: parameter, pearson_r, rmse, bias, pass_fail

**Figure Outputs:** `figures/recovery/{model}/`
- `{param_name}_recovery.png`: Scatter plot (true vs recovered) with annotations
- `{param_name}_distribution.png`: KDE overlay (real vs recovered)
- `aic_bic_comparison.png`: Boxplots comparing fit quality

---

## 6. Validation and Reporting

### 6.1 Pass/Fail Criteria

**Per Parameter:**
- PASS if Pearson r >= 0.80
- FAIL if Pearson r < 0.80

**Overall Model:**
- PASS if ALL parameters meet r >= 0.80
- FAIL if ANY parameter fails

**Example Output:**
```
Parameter Recovery Results (M3: WM-RL + perseveration)
========================================================
alpha_pos:  r=0.89, RMSE=0.042, Bias=-0.003  [PASS]
alpha_neg:  r=0.85, RMSE=0.051, Bias=0.007   [PASS]
phi:        r=0.82, RMSE=0.061, Bias=-0.012  [PASS]
rho:        r=0.87, RMSE=0.048, Bias=0.004   [PASS]
capacity:   r=0.91, RMSE=0.287, Bias=0.021   [PASS]
kappa:      r=0.78, RMSE=0.089, Bias=-0.015  [FAIL]
epsilon:    r=0.84, RMSE=0.039, Bias=0.008   [PASS]
--------------------------------------------------------
Overall: FAIL (1/7 parameters below threshold)
```

### 6.2 Recommended Sample Sizes

**Based on Literature:**
- Minimum n_subjects = 30 for stable correlation estimates
- Recommended n_subjects = 50-100 to match real study power
- n_datasets = 10 provides distribution of recovery metrics

**For This Project:**
- Real sample size: ~80 participants (after exclusions)
- Recommended: `--n-subjects 50` (feasible, matches study scale)
- Multiple datasets: `--n-datasets 10` (computational cost vs precision trade-off)

### 6.3 Troubleshooting Poor Recovery

**If r < 0.80 for some parameters:**

**Check 1: Convergence Rate**
- If convergence < 80%, optimization is failing
- Solution: Increase n_starts, adjust bounds, check initial values

**Check 2: Synthetic Data Quality**
- Plot behavior (accuracy, switch rates)
- Compare to real data distributions
- Ensure realistic behavior emerges

**Check 3: Parameter Confounding**
- Check parameter correlations in real fits
- High correlation (r > 0.9) suggests non-identifiability
- May need to fix one parameter or simplify model

**Check 4: Sample Size**
- Try increasing n_subjects
- Check if r improves with more data

**Source:** [The interpretation of computational model parameters depends on the context](https://pmc.ncbi.nlm.nih.gov/articles/PMC9635876/)

---

## 7. Python Libraries and Tools

### 7.1 Core Libraries (Already in Use)

**JAX:** For model simulation (likelihoods are already JAX-based)
- Use existing `jax_likelihoods.py` functions
- Supports GPU acceleration

**NumPy:** For parameter sampling and metric computation
- `np.random.uniform()` for parameter sampling
- `np.corrcoef()` or `scipy.stats.pearsonr()` for correlation

**Pandas:** For data handling
- Synthetic data stored in DataFrames
- Matches existing task_trials_long.csv format

**SciPy:** For statistics and metrics
- `scipy.stats.pearsonr()`: Pearson correlation + p-value
- `scipy.stats.spearmanr()`: Rank correlation (optional)

**Matplotlib/Seaborn:** For visualization
- `sns.regplot()`: Scatter + regression line
- `sns.kdeplot()`: Distribution overlays
- `plt.text()`: Annotations (r, RMSE, bias)

**tqdm:** For progress bars
- `tqdm(range(n_datasets))`: Dataset loop
- Nested tqdm for subject-level fitting

### 7.2 Recommended Code Structure

**Imports:**
```python
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pathlib import Path

# Existing project modules
from scripts.fitting.fit_mle import fit_participant_mle, prepare_participant_data
from scripts.fitting.jax_likelihoods import (
    q_learning_multiblock_likelihood,
    wmrl_multiblock_likelihood,
    wmrl_m3_multiblock_likelihood
)
from scripts.fitting.mle_utils import QLEARNING_BOUNDS, WMRL_BOUNDS, WMRL_M3_BOUNDS
```

**Utility Functions:**
```python
def sample_parameters_uniformly(model, n_subjects, seed):
    """Sample true parameters uniformly from bounds."""
    rng = np.random.default_rng(seed)
    bounds = get_bounds(model)
    params_list = []
    for _ in range(n_subjects):
        params = {p: rng.uniform(low, high) for p, (low, high) in bounds.items()}
        params_list.append(params)
    return params_list

def compute_recovery_metrics(true, recovered):
    """Compute Pearson r, RMSE, bias."""
    r, p = pearsonr(true, recovered)
    rmse = np.sqrt(np.mean((true - recovered)**2))
    bias = np.mean(recovered - true)
    return {'r': r, 'p': p, 'rmse': rmse, 'bias': bias}
```

### 7.3 No Additional Dependencies Required

All necessary libraries are already in the project environment:
- JAX (for simulation)
- NumPy, Pandas (data handling)
- SciPy (statistics)
- Matplotlib, Seaborn (visualization)
- tqdm (progress bars)

**No new installations needed.**

---

## 8. Confidence Assessment

| Area | Confidence | Rationale |
|------|------------|-----------|
| **Methodology** | HIGH | Well-established in computational modeling literature, Senta et al. followed standard procedures |
| **Metrics** | HIGH | Pearson r, RMSE, bias are standard and well-documented |
| **Criterion** | HIGH | r >= 0.80 is widely used threshold, empirically validated for RL models |
| **Implementation** | HIGH | All required infrastructure exists (fitting, likelihoods, data format) |
| **Visualization** | HIGH | Standard scatter + KDE plots, well-supported by matplotlib/seaborn |
| **Libraries** | HIGH | All libraries already in use, no new dependencies |

**Overall Confidence: HIGH**

All components are well-established, documented, and align with existing codebase infrastructure.

---

## 9. Key Recommendations for Planning

### 9.1 Must-Have Features

1. **Synthetic Data Generator**
   - Match real task structure exactly (21 blocks, reversals at 12-18 correct)
   - Sample parameters uniformly from MLE bounds
   - Output DataFrame matching task_trials_long.csv format

2. **Recovery Loop**
   - Import fit_participant_mle() directly (no subprocess)
   - Sequential fitting with tqdm progress
   - Store true vs recovered parameters

3. **Metrics Computation**
   - Pearson r, RMSE, bias per parameter
   - Pass/fail evaluation (r >= 0.80)

4. **Visualization**
   - Scatter plots with identity line, regression line, annotations
   - KDE overlays comparing real vs recovered distributions
   - AIC/BIC sanity check plots

5. **Reporting**
   - Print summary table with metrics and pass/fail
   - Save recovery_results.csv and recovery_metrics.csv
   - Save all plots to figures/recovery/{model}/

### 9.2 Nice-to-Have Enhancements

1. **Multiple Recovery Runs**
   - Run recovery with different n_subjects (30, 50, 100)
   - Plot r vs n_subjects to show data requirement

2. **Robustness Checks**
   - Test recovery with different random seeds
   - Compute confidence intervals on r via bootstrapping

3. **Cross-Model Comparison**
   - Run recovery for all three models (M1, M2, M3)
   - Compare which parameters recover best/worst

### 9.3 Potential Pitfalls

1. **Optimizer Convergence Issues**
   - Synthetic data may be easier/harder to fit than real data
   - Monitor convergence rates
   - May need to adjust n_starts or optimization settings

2. **Parameter Sampling Artifacts**
   - Uniform sampling may create unrealistic behavior patterns
   - Check behavioral metrics (accuracy, switch rates) match real data

3. **Computational Cost**
   - With n_subjects=50, n_datasets=10, n_starts=50: 25,000 optimizations
   - Estimate runtime and use GPU if available
   - Consider starting with smaller n_datasets for testing

4. **Interpretation Challenges**
   - High r doesn't guarantee unbiased recovery (check bias)
   - Low r may be acceptable if RMSE is small (practical vs statistical)
   - Parameter confounding may limit recovery (check correlations)

---

## 10. Sources

**Parameter Recovery Methodology:**
- [Parameter and Model Recovery of Reinforcement Learning Models for Restless Bandit Problems](https://link.springer.com/article/10.1007/s42113-022-00139-0)
- [Ten simple rules for the computational modeling of behavioral data](https://pmc.ncbi.nlm.nih.gov/articles/PMC6879303/)
- [An Introduction to Good Practices in Cognitive Modeling](https://scite.ai/reports/an-introduction-to-good-practices-2X63Z1)

**Senta et al. (2025):**
- [Dual process impairments in reinforcement learning and working memory systems underlie learning deficits in physiological anxiety](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012872)

**Recovery Metrics:**
- [RMSE is not enough: Guidelines to robust data-model comparisons](https://www.sciencedirect.com/science/article/pii/S1364682621000857)
- [A Strategy for Using Bias and RMSE as Outcomes in Monte Carlo Studies](https://digitalcommons.wayne.edu/cgi/viewcontent.cgi?article=2938&context=jmasm)

**Computational Psychiatry Context:**
- [Computational Neuroscience Approach to Psychiatry: A Review on Theory-driven Approaches](https://pmc.ncbi.nlm.nih.gov/articles/PMC8813324/)
- [The interpretation of computational model parameters depends on the context](https://pmc.ncbi.nlm.nih.gov/articles/PMC9635876/)

**Visualization:**
- [seaborn.regplot documentation](https://seaborn.pydata.org/generated/seaborn.regplot.html)
- [Visualizing statistical relationships - seaborn](https://seaborn.pydata.org/tutorial/relational.html)

**Synthetic Data Generation:**
- [A Reinforcement Learning Approach to Synthetic Data Generation](https://arxiv.org/html/2512.21395v1)

---

## RESEARCH COMPLETE

**Status:** Ready for planning
**Next Steps:** Create detailed implementation plan (RECV-01 through RECV-06)
**Estimated Effort:** Medium (3-5 implementation sessions)
- Session 1: Synthetic data generator + parameter sampling
- Session 2: Recovery loop + metrics computation
- Session 3: Visualization + reporting
- Session 4: Script 11 wrapper + testing
- Session 5: Documentation + verification

**Key Decisions Made:**
- Use r >= 0.80 as pass/fail criterion (standard in field)
- Metrics: Pearson r, RMSE, bias (comprehensive set)
- Sequential GPU fitting (avoid CPU parallel LLVM issues)
- Import fit_participant_mle() directly (not subprocess)
- Match real task structure exactly for synthetic data
- Default n_subjects=50 (matches study scale)
- KDE overlays for sanity checking simulation realism
