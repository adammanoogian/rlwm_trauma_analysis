"""
Example: Visualize Parameter Sweep Results

Demonstrates how to load and visualize parameter sweep results,
including individual model plots and comparative visualizations.

Usage:
    python tests/examples/example_visualize_sweeps.py
"""

import pandas as pd
from pathlib import Path
import sys

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.simulations.visualize_parameter_sweeps import (
    plot_qlearning_sweep,
    plot_wmrl_sweep,
    plot_comparative_accuracy_by_setsize,
    plot_comparative_heatmaps
)

print("=" * 80)
print("PARAMETER SWEEP VISUALIZATION EXAMPLE")
print("=" * 80)
print()
print("This script demonstrates how to visualize parameter sweep results.")
print("It will load the most recent sweep results and generate all plots.")
print()

# ============================================================================
# STEP 1: LOCATE PARAMETER SWEEP RESULTS
# ============================================================================

print("-" * 80)
print("STEP 1: LOCATING PARAMETER SWEEP RESULTS")
print("-" * 80)
print()

sweep_dir = project_root / 'output' / 'parameter_sweeps'

if not sweep_dir.exists():
    print(f"ERROR: Parameter sweep directory not found: {sweep_dir}")
    print()
    print("To generate sweep results, run one of:")
    print("  python tests/examples/example_parameter_sweep.py")
    print("  python scripts/10_run_parameter_sweep.py")
    print()
    sys.exit(1)

# Find most recent sweep files
q_files = list(sweep_dir.glob('*qlearning*.csv'))
wmrl_files = list(sweep_dir.glob('*wmrl*.csv'))

if not q_files or not wmrl_files:
    print(f"ERROR: No sweep results found in {sweep_dir}")
    print()
    print("Expected files:")
    print("  - *qlearning*.csv")
    print("  - *wmrl*.csv")
    print()
    print("Run a parameter sweep first!")
    sys.exit(1)

# Get most recent files (sorted by name, which includes timestamp)
q_file = sorted(q_files)[-1]
wmrl_file = sorted(wmrl_files)[-1]

print(f"Found Q-learning results: {q_file.name}")
print(f"Found WM-RL results: {wmrl_file.name}")
print()

# ============================================================================
# STEP 2: LOAD THE DATA
# ============================================================================

print("-" * 80)
print("STEP 2: LOADING DATA")
print("-" * 80)
print()

print(f"Loading {q_file.name}...")
qlearning_df = pd.read_csv(q_file)
print(f"  Loaded {len(qlearning_df)} Q-learning parameter combinations")

print(f"Loading {wmrl_file.name}...")
wmrl_df = pd.read_csv(wmrl_file)
print(f"  Loaded {len(wmrl_df)} WM-RL parameter combinations")
print()

# Show data structure
print("Q-learning data columns:", list(qlearning_df.columns))
print("WM-RL data columns:", list(wmrl_df.columns))
print()

# ============================================================================
# STEP 3: VISUALIZE Q-LEARNING RESULTS
# ============================================================================

print("-" * 80)
print("STEP 3: VISUALIZING Q-LEARNING PARAMETER SWEEP")
print("-" * 80)
print()

print("Creating Q-learning visualizations...")
print("  - Alpha+ (positive PE learning rate) effects")
print("  - Alpha- (negative PE learning rate) effects")
print("  - Beta (inverse temperature) effects")
print("  - Alpha+ × Beta heatmap")
print()

plot_qlearning_sweep(qlearning_df, save_dir=sweep_dir, show=False)
print("✓ Saved: figures/parameter_sweeps/qlearning_individual.png")
print()

# Show some insights
print("Q-Learning Insights:")
print()

# Find best overall combination
best_idx = qlearning_df['accuracy_mean'].idxmax()
best_row = qlearning_df.loc[best_idx]
print(f"Best overall performance:")
print(f"  Alpha+ = {best_row['alpha_pos']:.2f}")
print(f"  Alpha- = {best_row['alpha_neg']:.2f}")
print(f"  Beta = {best_row['beta']:.1f}")
print(f"  Set Size = {best_row['set_size']}")
print(f"  Accuracy = {best_row['accuracy_mean']:.3f} ± {best_row['accuracy_std']:.3f}")
print()

# Best by set size
set_sizes = sorted(qlearning_df['set_size'].unique())
print("Best performance by set size:")
for ss in set_sizes:
    ss_data = qlearning_df[qlearning_df['set_size'] == ss]
    best_ss_idx = ss_data['accuracy_mean'].idxmax()
    best_ss = ss_data.loc[best_ss_idx]
    print(f"  Set Size {ss}: α+={best_ss['alpha_pos']:.2f}, α-={best_ss['alpha_neg']:.2f}, "
          f"β={best_ss['beta']:.1f} → {best_ss['accuracy_mean']:.3f}")
print()

# ============================================================================
# STEP 4: VISUALIZE WM-RL RESULTS
# ============================================================================

print("-" * 80)
print("STEP 4: VISUALIZING WM-RL PARAMETER SWEEP")
print("-" * 80)
print()

print("Creating WM-RL visualizations...")
print("  - Capacity (K) effects")
print("  - Rho (base WM reliance) effects")
print("  - Phi (WM decay) effects")
print("  - Capacity × Rho heatmap")
print()

plot_wmrl_sweep(wmrl_df, save_dir=sweep_dir, show=False)
print("✓ Saved: figures/parameter_sweeps/wmrl_individual.png")
print()

# Show some insights
print("WM-RL Insights:")
print()

# Find best overall combination
best_idx = wmrl_df['accuracy_mean'].idxmax()
best_row = wmrl_df.loc[best_idx]
print(f"Best overall performance:")
print(f"  Capacity (K) = {best_row['capacity']}")
print(f"  Rho (WM reliance) = {best_row['rho']:.2f}")
if 'phi' in wmrl_df.columns:
    print(f"  Phi (decay) = {best_row['phi']:.2f}")
print(f"  Set Size = {best_row['set_size']}")
print(f"  Accuracy = {best_row['accuracy_mean']:.3f} ± {best_row['accuracy_std']:.3f}")
print()

# Best by set size
print("Best performance by set size:")
for ss in set_sizes:
    ss_data = wmrl_df[wmrl_df['set_size'] == ss]
    best_ss_idx = ss_data['accuracy_mean'].idxmax()
    best_ss = ss_data.loc[best_ss_idx]
    phi_str = f", φ={best_ss['phi']:.2f}" if 'phi' in wmrl_df.columns else ""
    print(f"  Set Size {ss}: K={best_ss['capacity']}, ρ={best_ss['rho']:.2f}{phi_str} "
          f"→ {best_ss['accuracy_mean']:.3f}")
print()

# ============================================================================
# STEP 5: COMPARATIVE VISUALIZATIONS
# ============================================================================

print("-" * 80)
print("STEP 5: CREATING COMPARATIVE VISUALIZATIONS")
print("-" * 80)
print()

print("Creating model comparison plots...")
print()

# Side-by-side accuracy comparison
print("1. Comparative accuracy by set size...")
plot_comparative_accuracy_by_setsize(qlearning_df, wmrl_df, save_dir=sweep_dir, show=False)
print("   ✓ Saved: figures/parameter_sweeps/comparative_accuracy.png")
print()

# Side-by-side heatmaps
print("2. Comparative parameter space heatmaps...")
plot_comparative_heatmaps(qlearning_df, wmrl_df, save_dir=sweep_dir, show=False)
print("   ✓ Saved: figures/parameter_sweeps/comparative_heatmaps.png")
print()

# ============================================================================
# STEP 6: SUMMARY AND INTERPRETATION
# ============================================================================

print("-" * 80)
print("STEP 6: SUMMARY AND INTERPRETATION GUIDE")
print("-" * 80)
print()

print("INTERPRETING THE RESULTS:")
print()

print("1. Individual Model Plots (4 subplots each):")
print("   - Top-left: Main parameter effect averaged across others")
print("   - Top-right: Second parameter effect")
print("   - Bottom-left: Third parameter effect")
print("   - Bottom-right: Heatmap showing interaction of two key parameters")
print()

print("2. Comparative Accuracy Plot:")
print("   - Bar chart comparing best Q-learning vs best WM-RL for each set size")
print("   - Shows which model architecture performs better overall")
print("   - Look for: Does one model dominate? Or do they perform similarly?")
print()

print("3. Comparative Heatmaps:")
print("   - Left: Q-learning parameter space (Alpha+ × Beta)")
print("   - Right: WM-RL parameter space (Capacity × Rho)")
print("   - Look for: Robustness (large green regions) vs sensitivity (narrow peaks)")
print()

print("KEY QUESTIONS TO ASK:")
print()

print("Q: Which model performs better overall?")
for ss in set_sizes:
    q_best = qlearning_df[qlearning_df['set_size'] == ss]['accuracy_mean'].max()
    wmrl_best = wmrl_df[wmrl_df['set_size'] == ss]['accuracy_mean'].max()
    winner = "WM-RL" if wmrl_best > q_best else "Q-Learning"
    diff = abs(wmrl_best - q_best)
    print(f"  Set Size {ss}: {winner} wins by {diff:.3f} ({q_best:.3f} vs {wmrl_best:.3f})")
print()

print("Q: How does set size affect performance?")
for model_name, df in [("Q-Learning", qlearning_df), ("WM-RL", wmrl_df)]:
    print(f"  {model_name}:")
    for ss in set_sizes:
        ss_mean = df[df['set_size'] == ss]['accuracy_mean'].mean()
        print(f"    Set Size {ss}: {ss_mean:.3f} (average across all parameters)")
print()

print("Q: Are the models robust to parameter changes?")
print("  Check the heatmaps:")
print("  - Large green regions = robust (good performance across parameters)")
print("  - Narrow peaks = sensitive (performance drops quickly with bad parameters)")
print()

print("=" * 80)
print("VISUALIZATION COMPLETE!")
print("=" * 80)
print()
print(f"All figures saved to: figures/parameter_sweeps/")
print()
print("Generated files:")
print("  1. qlearning_individual.png - Q-learning parameter effects")
print("  2. wmrl_individual.png - WM-RL parameter effects")
print("  3. comparative_accuracy.png - Model comparison by set size")
print("  4. comparative_heatmaps.png - Parameter space comparison")
print()
print("Next steps:")
print("  - Review the figures to understand model behavior")
print("  - Identify optimal parameter ranges for each model")
print("  - Use insights to guide model fitting to behavioral data")
print("  - Run model comparison (scripts/06_fit_analyses/compare_models.py)")
print()
