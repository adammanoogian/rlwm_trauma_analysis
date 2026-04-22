"""
Example: Model Comparison with Information Criteria

Demonstrates how to compare Q-learning and WM-RL models using multiple
information criteria (BIC, AIC, WAIC, LOO) and how to interpret results.

This is an educational script that explains what each criterion means and
how to use them to select the best model for your data.

Usage:
    python tests/examples/example_model_comparison.py
"""

import pandas as pd
from pathlib import Path
import sys

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

print("=" * 80)
print("MODEL COMPARISON TUTORIAL")
print("=" * 80)
print()
print("This script demonstrates how to compare computational models using")
print("information criteria: BIC, AIC, WAIC, and LOO.")
print()

# ============================================================================
# BACKGROUND: WHAT ARE INFORMATION CRITERIA?
# ============================================================================

print("-" * 80)
print("BACKGROUND: INFORMATION CRITERIA")
print("-" * 80)
print()

print("Information criteria balance model fit (how well the model explains data)")
print("with model complexity (how many parameters it has).")
print()

print("We use FOUR criteria in this project:")
print()

print("1. AIC (Akaike Information Criterion)")
print("   Formula: AIC = 2k - 2·log(L)")
print("   - k = number of parameters")
print("   - L = likelihood at maximum")
print("   - Interpretation: LOWER is better")
print("   - Penalizes complexity moderately")
print()

print("2. BIC (Bayesian Information Criterion)")
print("   Formula: BIC = k·log(n) - 2·log(L)")
print("   - n = number of observations")
print("   - Interpretation: LOWER is better")
print("   - Penalizes complexity more strongly than AIC")
print("   - Preferred for large datasets")
print()

print("3. WAIC (Widely Applicable Information Criterion)")
print("   - Fully Bayesian criterion")
print("   - Uses entire posterior distribution (not just MAP)")
print("   - Computed via ArviZ from MCMC traces")
print("   - Interpretation: LOWER is better")
print("   - More robust than AIC/BIC for hierarchical models")
print()

print("4. LOO (Leave-One-Out Cross-Validation)")
print("   - Estimates out-of-sample prediction accuracy")
print("   - Uses PSIS-LOO (Pareto Smoothed Importance Sampling)")
print("   - Computed via ArviZ from MCMC traces")
print("   - Interpretation: LOWER is better")
print("   - Gold standard for model comparison")
print()

# ============================================================================
# EVIDENCE STRENGTH (Kass & Raftery, 1995)
# ============================================================================

print("-" * 80)
print("INTERPRETING DIFFERENCES (Δ)")
print("-" * 80)
print()

print("When comparing two models, we calculate Δ = IC_worse - IC_better")
print("The size of Δ tells us how strong the evidence is:")
print()

print("  Δ < 2   : Weak evidence (models are similar)")
print("  2-6     : Positive evidence (one model is likely better)")
print("  6-10    : Strong evidence (one model is clearly better)")
print("  Δ > 10  : Very strong evidence (overwhelming support)")
print()

# ============================================================================
# EXAMPLE WORKFLOW
# ============================================================================

print("-" * 80)
print("EXAMPLE WORKFLOW")
print("-" * 80)
print()

print("To run a full model comparison, you would:")
print()
print("STEP 1: Fit both models to behavioral data")
print("  $ python scripts/04_model_fitting/a_mle/12_fit_mle.py --model qlearning")
print("  $ python scripts/04_model_fitting/a_mle/12_fit_mle.py --model wmrl")
print()

print("STEP 2: Run model comparison script")
print("  $ python scripts/06_fit_analyses/compare_models.py")
print()

print("STEP 3: Review outputs")
print("  - CSV table: output/model_comparison/comparison_all_criteria.csv")
print("  - Plots: figures/model_comparison/")
print()

# ============================================================================
# SIMULATED EXAMPLE
# ============================================================================

print("-" * 80)
print("SIMULATED EXAMPLE: INTERPRETING RESULTS")
print("-" * 80)
print()

print("Let's walk through a simulated comparison to show you how to interpret")
print("the output. Here's what you might see:")
print()

# Create example results
example_results = pd.DataFrame({
    'model': ['Q-Learning', 'WM-RL'],
    'n_params': [39, 98],  # Example: 13 participants × (3 or 7) + group-level
    'log_likelihood': [-1234.5, -1189.2],
    'AIC': [2547.0, 2574.4],
    'BIC': [2631.8, 2759.2],
    'WAIC': [2545.2, 2568.9],
    'LOO': [2546.1, 2570.3],
    'AIC_rank': [1, 2],
    'BIC_rank': [1, 2],
    'WAIC_rank': [1, 2],
    'LOO_rank': [1, 2],
    'AIC_weight': [0.89, 0.11],
    'BIC_weight': [0.998, 0.002]
})

print("COMPARISON TABLE:")
print("=" * 80)
print(example_results[['model', 'n_params', 'AIC', 'BIC', 'WAIC', 'LOO']].to_string(index=False))
print()

print("RANKINGS (1 = best):")
print(example_results[['model', 'AIC_rank', 'BIC_rank', 'WAIC_rank', 'LOO_rank']].to_string(index=False))
print()

print("MODEL WEIGHTS (probability that model is best):")
print(example_results[['model', 'AIC_weight', 'BIC_weight']].to_string(index=False))
print()

# ============================================================================
# INTERPRETATION
# ============================================================================

print("-" * 80)
print("INTERPRETATION OF EXAMPLE")
print("-" * 80)
print()

# Calculate deltas
delta_aic = example_results.loc[1, 'AIC'] - example_results.loc[0, 'AIC']
delta_bic = example_results.loc[1, 'BIC'] - example_results.loc[0, 'BIC']
delta_waic = example_results.loc[1, 'WAIC'] - example_results.loc[0, 'WAIC']
delta_loo = example_results.loc[1, 'LOO'] - example_results.loc[0, 'LOO']

print("DELTAS (how much worse is WM-RL?):")
print(f"  ΔAIC = {delta_aic:.1f}")
print(f"  ΔBIC = {delta_bic:.1f}")
print(f"  ΔWAIC = {delta_waic:.1f}")
print(f"  ΔLOO = {delta_loo:.1f}")
print()

print("EVIDENCE STRENGTH:")
print()

# AIC interpretation
print(f"AIC (ΔAIC = {delta_aic:.1f}):")
if delta_aic < 2:
    strength = "weak"
elif delta_aic < 6:
    strength = "positive"
elif delta_aic < 10:
    strength = "strong"
else:
    strength = "very strong"
print(f"  → {strength.upper()} evidence for Q-Learning")
print(f"  → Q-Learning has {example_results.loc[0, 'AIC_weight']*100:.1f}% probability of being best model")
print()

# BIC interpretation
print(f"BIC (ΔBIC = {delta_bic:.1f}):")
if delta_bic < 2:
    strength = "weak"
elif delta_bic < 6:
    strength = "positive"
elif delta_bic < 10:
    strength = "strong"
else:
    strength = "very strong"
print(f"  → {strength.upper()} evidence for Q-Learning")
print(f"  → BIC penalizes WM-RL heavily for having {example_results.loc[1, 'n_params']} parameters")
print(f"  → Q-Learning has {example_results.loc[0, 'BIC_weight']*100:.1f}% probability of being best model")
print()

# WAIC interpretation
print(f"WAIC (ΔWAIC = {delta_waic:.1f}):")
if delta_waic < 2:
    strength = "weak"
elif delta_waic < 6:
    strength = "positive"
elif delta_waic < 10:
    strength = "strong"
else:
    strength = "very strong"
print(f"  → {strength.upper()} evidence for Q-Learning")
print(f"  → Fully Bayesian criterion using entire posterior")
print()

# LOO interpretation
print(f"LOO (ΔLOO = {delta_loo:.1f}):")
if delta_loo < 2:
    strength = "weak"
elif delta_loo < 6:
    strength = "positive"
elif delta_loo < 10:
    strength = "strong"
else:
    strength = "very strong"
print(f"  → {strength.upper()} evidence for Q-Learning")
print(f"  → Gold standard: estimates out-of-sample prediction")
print()

# ============================================================================
# DECISION MAKING
# ============================================================================

print("-" * 80)
print("DECISION MAKING GUIDE")
print("-" * 80)
print()

print("Which criterion should you trust most?")
print()

print("1. If all criteria agree:")
print("   → STRONG CONCLUSION: One model is clearly better")
print("   → Trust the consensus")
print()

print("2. If BIC favors simpler model but WAIC/LOO are close:")
print("   → BIC is penalizing complexity heavily")
print("   → Check if added complexity is justified")
print("   → Look at WAIC/LOO (more robust for hierarchical models)")
print()

print("3. If AIC/BIC disagree with WAIC/LOO:")
print("   → Point estimate (MAP) may be misleading")
print("   → Trust WAIC/LOO (they use full posterior)")
print()

print("4. General recommendation:")
print("   → PRIMARY: Use LOO (most robust)")
print("   → SECONDARY: Check WAIC for confirmation")
print("   → CONTEXT: Consider BIC for parsimony")
print("   → REPORT: Show all four for transparency")
print()

# ============================================================================
# IN THIS EXAMPLE
# ============================================================================

print("-" * 80)
print("CONCLUSION FOR THIS EXAMPLE")
print("-" * 80)
print()

print("Based on our simulated results:")
print()
print("✓ ALL FOUR criteria favor Q-Learning")
print(f"✓ Evidence ranges from {strength} (LOO) to very strong (BIC)")
print("✓ Q-Learning achieves better fit with fewer parameters")
print()
print("INTERPRETATION:")
print("  The simpler Q-Learning model is sufficient to explain the data.")
print("  The additional complexity of WM-RL (working memory system) does not")
print("  provide enough improvement in fit to justify the extra parameters.")
print()
print("SCIENTIFIC IMPLICATION:")
print("  Participants may be relying primarily on reinforcement learning (RL)")
print("  rather than working memory (WM) in this task.")
print()

# ============================================================================
# VISUALIZATION GUIDE
# ============================================================================

print("-" * 80)
print("UNDERSTANDING THE VISUALIZATIONS")
print("-" * 80)
print()

print("The 14_compare_models.py script generates two plots:")
print()

print("1. information_criteria_comparison.png (4-panel plot)")
print("   - Each panel shows one criterion (AIC, BIC, WAIC, LOO)")
print("   - Lower bars = better model")
print("   - Look for: Do all panels agree? Or is there disagreement?")
print()

print("2. model_weights.png (2-panel plot)")
print("   - Left: AIC weights (probability model is best)")
print("   - Right: BIC weights")
print("   - Higher bars = more evidence for that model")
print("   - Look for: Does one model have >90% weight? (strong evidence)")
print()

# ============================================================================
# COMMON PITFALLS
# ============================================================================

print("-" * 80)
print("COMMON PITFALLS TO AVOID")
print("-" * 80)
print()

print("❌ DON'T: Only use BIC because it's most popular")
print("✓ DO: Report all four criteria for transparency")
print()

print("❌ DON'T: Ignore small differences (Δ < 2) as 'significant'")
print("✓ DO: Recognize when models are practically equivalent")
print()

print("❌ DON'T: Trust point estimates (MAP) for complex hierarchical models")
print("✓ DO: Use WAIC/LOO which account for full posterior uncertainty")
print()

print("❌ DON'T: Automatically prefer simpler models")
print("✓ DO: Let the data decide via information criteria")
print()

print("❌ DON'T: Forget that these are relative comparisons")
print("✓ DO: Remember both models could be bad (absolute fit matters too!)")
print()

# ============================================================================
# NEXT STEPS
# ============================================================================

print("-" * 80)
print("NEXT STEPS")
print("-" * 80)
print()

print("After model comparison, you should:")
print()

print("1. VALIDATE: Check absolute fit (posterior predictive checks)")
print("   - Does the winning model actually fit the data well?")
print("   - Or is it just 'least bad' of two poor models?")
print()

print("2. INTERPRET: Extract parameter estimates from winning model")
print("   - What do the fitted parameters tell us about behavior?")
print("   - Are there individual differences?")
print()

print("3. VISUALIZE: Create publication-ready figures")
print("   - Model predictions vs actual behavior")
print("   - Parameter distributions and correlations")
print()

print("4. REPORT: Write up results following best practices")
print("   - Report all four criteria (transparency)")
print("   - Show evidence strength (Δ values)")
print("   - Discuss model interpretation (not just 'which won')")
print()

# ============================================================================
# REFERENCES
# ============================================================================

print("-" * 80)
print("RECOMMENDED READING")
print("-" * 80)
print()

print("Model comparison theory:")
print("  - Kass & Raftery (1995) - Bayes factors")
print("  - Burnham & Anderson (2002) - Model Selection and Multimodel Inference")
print("  - Vehtari et al. (2017) - Practical Bayesian model evaluation using LOO")
print()

print("Computational modeling in psychology:")
print("  - Wilson & Collins (2019) - Ten simple rules for computational modeling")
print("  - Palminteri et al. (2017) - The importance of falsification")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("TUTORIAL COMPLETE!")
print("=" * 80)
print()

print("KEY TAKEAWAYS:")
print()
print("1. Use MULTIPLE criteria (not just one)")
print("2. Interpret Δ values, not just which model 'wins'")
print("3. Trust LOO/WAIC over BIC/AIC for hierarchical models")
print("4. Report all results transparently")
print("5. Consider absolute fit, not just relative comparison")
print()

print("To run actual model comparison on your data:")
print("  $ python scripts/06_fit_analyses/compare_models.py")
print()
