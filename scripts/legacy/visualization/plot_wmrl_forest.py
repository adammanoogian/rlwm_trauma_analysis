"""
Create forest plot for WM-RL parameters by trauma group.

Plots individual participant parameters colored by trauma group.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from plotting_config import PlotConfig

# Trauma group colors
TRAUMA_COLORS = {
    'No Trauma': '#2ecc71',
    'Trauma - No Ongoing Impact': '#f39c12',
    'Trauma - Ongoing Impact': '#e74c3c'
}

# Participant to trauma group mapping
PARTICIPANT_GROUPS = {
    10001: 'No Trauma',
    10002: 'Trauma - No Ongoing Impact',
    9187: 'Trauma - Ongoing Impact'
}

# WM-RL posterior summary (from terminal output)
POSTERIOR_DATA = {
    'mu_alpha_neg': {'mean': 0.00, 'std': 0.00},
    'mu_alpha_pos': {'mean': 0.31, 'std': 0.11},
    'mu_beta': {'mean': 2.20, 'std': 1.26},
    'mu_beta_wm': {'mean': 4.16, 'std': 2.03},
    'mu_capacity': {'mean': 2.13, 'std': 0.70},
    'mu_phi': {'mean': 0.16, 'std': 0.08},
    'mu_rho': {'mean': 0.68, 'std': 0.10},
    'sigma_alpha_neg': {'mean': 0.26, 'std': 0.19},
    'sigma_alpha_pos': {'mean': 0.58, 'std': 0.22},
    'sigma_beta': {'mean': 1.73, 'std': 0.90},
    'sigma_beta_wm': {'mean': 2.90, 'std': 1.23},
    'sigma_capacity': {'mean': 1.14, 'std': 0.59},
    'sigma_phi': {'mean': 0.26, 'std': 0.20},
    'sigma_rho': {'mean': 0.37, 'std': 0.24},
    'z_alpha_neg': [
        {'mean': -0.41, 'std': 0.96, 'pid': 9187},   # Index 0
        {'mean': -0.02, 'std': 0.99, 'pid': 10001},  # Index 1
        {'mean': -0.08, 'std': 1.00, 'pid': 10002}   # Index 2
    ],
    'z_alpha_pos': [
        {'mean': -1.60, 'std': 0.86, 'pid': 9187},   # Index 0
        {'mean': -0.25, 'std': 1.02, 'pid': 10001},  # Index 1
        {'mean': 1.03, 'std': 0.85, 'pid': 10002}    # Index 2
    ],
    'z_beta': [
        {'mean': 0.76, 'std': 0.51, 'pid': 9187},    # Index 0
        {'mean': -1.13, 'std': 0.72, 'pid': 10001},  # Index 1
        {'mean': 0.53, 'std': 0.44, 'pid': 10002}    # Index 2
    ],
    'z_beta_wm': [
        {'mean': 1.04, 'std': 0.66, 'pid': 9187},    # Index 0
        {'mean': np.nan, 'std': np.nan, 'pid': 10001},  # Index 1 - Missing from output
        {'mean': np.nan, 'std': np.nan, 'pid': 10002}   # Index 2 - Missing from output
    ],
    'z_rho': [
        {'mean': -0.92, 'std': 1.11, 'pid': 9187},   # Index 0
        {'mean': 0.11, 'std': 0.97, 'pid': 10001},   # Index 1
        {'mean': 0.60, 'std': 0.99, 'pid': 10002}    # Index 2
    ]
}


def compute_participant_parameters():
    """Compute individual participant parameters from hierarchical structure."""
    participants = []
    
    for i in range(3):
        pid = POSTERIOR_DATA['z_alpha_neg'][i]['pid']
        group = PARTICIPANT_GROUPS[pid]
        
        # Compute actual parameters from z-scores: param = mu + sigma * z
        alpha_neg = (POSTERIOR_DATA['mu_alpha_neg']['mean'] + 
                     POSTERIOR_DATA['sigma_alpha_neg']['mean'] * POSTERIOR_DATA['z_alpha_neg'][i]['mean'])
        alpha_pos = (POSTERIOR_DATA['mu_alpha_pos']['mean'] + 
                     POSTERIOR_DATA['sigma_alpha_pos']['mean'] * POSTERIOR_DATA['z_alpha_pos'][i]['mean'])
        beta = (POSTERIOR_DATA['mu_beta']['mean'] + 
                POSTERIOR_DATA['sigma_beta']['mean'] * POSTERIOR_DATA['z_beta'][i]['mean'])
        
        if not np.isnan(POSTERIOR_DATA['z_beta_wm'][i]['mean']):
            beta_wm = (POSTERIOR_DATA['mu_beta_wm']['mean'] + 
                      POSTERIOR_DATA['sigma_beta_wm']['mean'] * POSTERIOR_DATA['z_beta_wm'][i]['mean'])
        else:
            beta_wm = POSTERIOR_DATA['mu_beta_wm']['mean']
            
        rho = (POSTERIOR_DATA['mu_rho']['mean'] + 
               POSTERIOR_DATA['sigma_rho']['mean'] * POSTERIOR_DATA['z_rho'][i]['mean'])
        
        participants.append({
            'participant_id': pid,
            'trauma_group': group,
            'alpha_neg': alpha_neg,
            'alpha_pos': alpha_pos,
            'beta': beta,
            'beta_wm': beta_wm,
            'rho': rho,
            'capacity': POSTERIOR_DATA['mu_capacity']['mean'],  # Using group mean (no z provided)
            'phi': POSTERIOR_DATA['mu_phi']['mean']  # Using group mean (no z provided)
        })
    
    return pd.DataFrame(participants)


def compute_participant_parameters_with_uncertainty():
    """Compute individual participant parameters with uncertainty (mean ± std)."""
    participants = []
    
    for i in range(3):
        pid = POSTERIOR_DATA['z_alpha_neg'][i]['pid']
        group = PARTICIPANT_GROUPS[pid]
        
        # For hierarchical models: param = mu + sigma * z
        # Uncertainty propagates: std_param ≈ sigma * std_z
        
        alpha_neg_mean = (POSTERIOR_DATA['mu_alpha_neg']['mean'] + 
                         POSTERIOR_DATA['sigma_alpha_neg']['mean'] * POSTERIOR_DATA['z_alpha_neg'][i]['mean'])
        alpha_neg_std = POSTERIOR_DATA['sigma_alpha_neg']['mean'] * POSTERIOR_DATA['z_alpha_neg'][i]['std']
        
        alpha_pos_mean = (POSTERIOR_DATA['mu_alpha_pos']['mean'] + 
                         POSTERIOR_DATA['sigma_alpha_pos']['mean'] * POSTERIOR_DATA['z_alpha_pos'][i]['mean'])
        alpha_pos_std = POSTERIOR_DATA['sigma_alpha_pos']['mean'] * POSTERIOR_DATA['z_alpha_pos'][i]['std']
        
        beta_mean = (POSTERIOR_DATA['mu_beta']['mean'] + 
                    POSTERIOR_DATA['sigma_beta']['mean'] * POSTERIOR_DATA['z_beta'][i]['mean'])
        beta_std = POSTERIOR_DATA['sigma_beta']['mean'] * POSTERIOR_DATA['z_beta'][i]['std']
        
        if not np.isnan(POSTERIOR_DATA['z_beta_wm'][i]['mean']):
            beta_wm_mean = (POSTERIOR_DATA['mu_beta_wm']['mean'] + 
                           POSTERIOR_DATA['sigma_beta_wm']['mean'] * POSTERIOR_DATA['z_beta_wm'][i]['mean'])
            beta_wm_std = POSTERIOR_DATA['sigma_beta_wm']['mean'] * POSTERIOR_DATA['z_beta_wm'][i]['std']
        else:
            beta_wm_mean = POSTERIOR_DATA['mu_beta_wm']['mean']
            beta_wm_std = POSTERIOR_DATA['sigma_beta_wm']['mean']
            
        rho_mean = (POSTERIOR_DATA['mu_rho']['mean'] + 
                   POSTERIOR_DATA['sigma_rho']['mean'] * POSTERIOR_DATA['z_rho'][i]['mean'])
        rho_std = POSTERIOR_DATA['sigma_rho']['mean'] * POSTERIOR_DATA['z_rho'][i]['std']
        
        participants.append({
            'participant_id': pid,
            'trauma_group': group,
            'alpha_neg_mean': alpha_neg_mean,
            'alpha_neg_std': alpha_neg_std,
            'alpha_pos_mean': alpha_pos_mean,
            'alpha_pos_std': alpha_pos_std,
            'beta_mean': beta_mean,
            'beta_std': beta_std,
            'beta_wm_mean': beta_wm_mean,
            'beta_wm_std': beta_wm_std,
            'rho_mean': rho_mean,
            'rho_std': rho_std,
            'capacity_mean': POSTERIOR_DATA['mu_capacity']['mean'],
            'capacity_std': POSTERIOR_DATA['sigma_capacity']['mean'],
            'phi_mean': POSTERIOR_DATA['mu_phi']['mean'],
            'phi_std': POSTERIOR_DATA['sigma_phi']['mean']
        })
    
    return pd.DataFrame(participants)


def plot_wmrl_forest():
    """Create forest plot for WM-RL parameters with uncertainty intervals."""
    
    # Get participant parameters with uncertainty
    df = compute_participant_parameters_with_uncertainty()
    
    # Parameters to plot
    params = [
        ('alpha_pos', 'α+ (Pos. Learning Rate)'),
        ('alpha_neg', 'α- (Neg. Learning Rate)'),
        ('beta', 'β (RL Temperature)'),
        ('beta_wm', 'β_WM (WM Temperature)'),
        ('rho', 'ρ (WM Decay)'),
        ('capacity', 'K (Capacity)'),
        ('phi', 'φ (Stickiness)')
    ]
    
    n_params = len(params)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_positions = np.arange(n_params)
    
    for i, (param_key, param_label) in enumerate(params):
        y = y_positions[i]
        
        # Plot each participant with error bars
        for idx, row in df.iterrows():
            color = TRAUMA_COLORS[row['trauma_group']]
            mean = row[f'{param_key}_mean']
            std = row[f'{param_key}_std']
            
            y_offset = y + (idx - 1) * 0.15
            
            # Plot 95% credible interval (±1.96 std)
            ax.errorbar(mean, y_offset, xerr=1.96*std, 
                       fmt='o', color=color, markersize=8, alpha=0.8,
                       capsize=4, capthick=2, elinewidth=2,
                       label=row['trauma_group'] if i == 0 else '')
        
        # Plot group mean with uncertainty
        if param_key == 'alpha_pos':
            group_mean = POSTERIOR_DATA['mu_alpha_pos']['mean']
            group_std = POSTERIOR_DATA['mu_alpha_pos']['std']
        elif param_key == 'alpha_neg':
            group_mean = POSTERIOR_DATA['mu_alpha_neg']['mean']
            group_std = POSTERIOR_DATA['mu_alpha_neg']['std']
        elif param_key in ['beta', 'beta_wm', 'rho', 'capacity', 'phi']:
            group_mean = POSTERIOR_DATA[f'mu_{param_key}']['mean']
            group_std = POSTERIOR_DATA[f'mu_{param_key}']['std']
        else:
            group_mean = 0
            group_std = 0
        
        ax.errorbar(group_mean, y, xerr=1.96*group_std,
                   fmt='s', color='black', markersize=8, alpha=0.7,
                   markerfacecolor='none', markeredgewidth=2,
                   capsize=4, capthick=2, elinewidth=2)
    
    # Formatting
    ax.set_yticks(y_positions)
    ax.set_yticklabels([label for _, label in params], fontsize=PlotConfig.AXIS_LABEL_SIZE)
    ax.set_xlabel('Parameter Value', fontsize=PlotConfig.AXIS_LABEL_SIZE, fontweight='bold')
    ax.tick_params(axis='x', labelsize=PlotConfig.TICK_LABEL_SIZE)
    ax.grid(True, axis='x', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend1 = ax.legend(by_label.values(), by_label.keys(), 
                       loc='upper right', fontsize=PlotConfig.LEGEND_SIZE, 
                       title='Trauma Group', title_fontsize=PlotConfig.LEGEND_SIZE)
    
    # Add second legend for group mean
    from matplotlib.lines import Line2D
    group_mean_line = Line2D([0], [0], marker='s', color='w', 
                            markerfacecolor='none', markeredgecolor='black',
                            markeredgewidth=2, markersize=8, label='Group Mean')
    ax.legend(handles=[group_mean_line], loc='lower right', 
             fontsize=PlotConfig.LEGEND_SIZE)
    ax.add_artist(legend1)
    
    ax.set_title('WM-RL Model Parameters by Trauma Group (N=3)',
                fontsize=PlotConfig.TITLE_SIZE, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save
    output_path = Path('output/v1/wmrl_forest_by_trauma_group.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == '__main__':
    plot_wmrl_forest()
