"""
Create Supplementary Table S3: Feedback Sensitivity and Perseveration Statistics.

This script generates a formatted supplementary table showing descriptive statistics
and trauma correlations for behavioral indices of feedback sensitivity and perseveration.

Author: Analysis Pipeline
Date: February 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ============================================================================
# Configuration
# ============================================================================

# Input files
DESCRIPTIVES_FILE = project_root / "output" / "statistical_analyses" / "feedback_perseveration_descriptives.csv"
CORRELATIONS_FILE = project_root / "output" / "statistical_analyses" / "feedback_perseveration_trauma_correlations.csv"

# Output directory
OUTPUT_DIR = project_root / "output" / "supplementary_materials"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Main Function
# ============================================================================

def main():
    print("=" * 80)
    print("CREATING SUPPLEMENTARY TABLE S3")
    print("Feedback Sensitivity and Perseveration Statistics")
    print("=" * 80)
    print()
    
    # ========================================================================
    # Load Data
    # ========================================================================
    
    print("Loading descriptive statistics...")
    df_desc = pd.read_csv(DESCRIPTIVES_FILE)
    print(f"  Loaded descriptives for {len(df_desc)} metrics")
    
    print("\nLoading correlation results...")
    df_corr = pd.read_csv(CORRELATIONS_FILE)
    print(f"  Loaded {len(df_corr)} correlation tests")
    
    # ========================================================================
    # Build Table
    # ========================================================================
    
    print("\nBuilding supplementary table...")
    
    table_rows = []
    
    for _, desc_row in df_desc.iterrows():
        metric_name = desc_row['Metric']
        metric_key = metric_name.lower().replace(' ', '_')
        
        # Get correlations for this metric
        metric_corrs = df_corr[df_corr['Metric'] == metric_key].copy()
        
        # Descriptive statistics
        mean_sd = f"{desc_row['Mean']:.3f} ({desc_row['SD']:.3f})"
        
        # LESS Total
        less_corr = metric_corrs[metric_corrs['Trauma_Variable'] == 'less_total_events']
        if len(less_corr) > 0:
            less_beta = less_corr['Beta_standardized'].values[0]
            less_p = less_corr['p_value'].values[0]
            less_str = f"β = {less_beta:.3f}, p = {less_p:.3f}"
        else:
            less_str = "—"
        
        # IES-R Total
        ies_total_corr = metric_corrs[metric_corrs['Trauma_Variable'] == 'ies_total']
        if len(ies_total_corr) > 0:
            ies_beta = ies_total_corr['Beta_standardized'].values[0]
            ies_p = ies_total_corr['p_value'].values[0]
            ies_total_str = f"β = {ies_beta:.3f}, p = {ies_p:.3f}"
        else:
            ies_total_str = "—"
        
        # Strongest IES-R subscale
        subscale_corrs = metric_corrs[
            metric_corrs['Trauma_Variable'].isin([
                'ies_intrusion', 'ies_avoidance', 'ies_hyperarousal'
            ])
        ].copy()
        
        if len(subscale_corrs) > 0:
            # Find strongest by absolute rho
            subscale_corrs['abs_rho'] = subscale_corrs['Spearman_rho'].abs()
            strongest_idx = subscale_corrs['abs_rho'].idxmax()
            strongest = subscale_corrs.loc[strongest_idx]
            
            subscale_name = strongest['Trauma_Variable'].replace('ies_', '').capitalize()
            subscale_rho = strongest['Spearman_rho']
            subscale_p = strongest['p_value']
            
            strongest_str = f"{subscale_name} (ρ = {subscale_rho:.3f}, p = {subscale_p:.3f})"
        else:
            strongest_str = "—"
        
        # Add row
        table_rows.append({
            'Metric': metric_name,
            'Mean (SD)': mean_sd,
            'LESS Total': less_str,
            'IES-R Total': ies_total_str,
            'Strongest IES-R Subscale': strongest_str
        })
    
    df_table = pd.DataFrame(table_rows)
    
    # ========================================================================
    # Save Table
    # ========================================================================
    
    output_file = OUTPUT_DIR / "supplementary_table_s3_feedback_perseveration.csv"
    df_table.to_csv(output_file, index=False)
    
    print(f"\nSupplementary Table S3 saved to:")
    print(f"  {output_file}")
    
    # ========================================================================
    # Display Table
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("SUPPLEMENTARY TABLE S3")
    print("=" * 80)
    print()
    print("Descriptive statistics and correlations between trauma measures and")
    print("behavioural indices of feedback sensitivity and perseveration (N = 48)")
    print()
    
    # Print formatted table
    print(df_table.to_string(index=False))
    
    print("\n" + "=" * 80)
    print("Note. LESS = Life Events Scale for Students; IES-R = Impact of Event")
    print("Scale-Revised. Feedback sensitivity reflects win-stay/lose-shift")
    print("strategy use. Perseveration index reflects choice repetition after")
    print("errors. β = standardized regression coefficient from linear regression;")
    print("ρ = Spearman rank correlation coefficient. No correlations were")
    print("statistically significant (all ps > .14).")
    print("=" * 80)
    
    print("\n✓ Table creation complete")


if __name__ == "__main__":
    main()
