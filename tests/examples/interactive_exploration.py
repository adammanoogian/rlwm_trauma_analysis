"""
Interactive Parameter Space Exploration

Jupyter notebook-friendly tools for exploring model behavior interactively.

Usage in Jupyter:
    from tests.examples.interactive_exploration import explore_qlearning_interactive
    explore_qlearning_interactive()

    # Or for WM-RL:
    from tests.examples.interactive_exploration import explore_wmrl_interactive
    explore_wmrl_interactive()
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from environments.rlwm_env import create_rlwm_env
from models.q_learning import create_q_learning_agent, simulate_agent_on_env
from models.wm_rl_hybrid import create_wm_rl_agent, simulate_wm_rl_on_env

try:
    from ipywidgets import interact, FloatSlider, IntSlider, fixed, Dropdown
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False
    print("Warning: ipywidgets not available. Install with: pip install ipywidgets")


def explore_qlearning_interactive():
    """
    Interactive widget to explore Q-learning parameters.

    Creates interactive sliders for alpha, beta, set_size, and num_trials.
    Updates plots in real-time as parameters change.
    """
    if not WIDGETS_AVAILABLE:
        print("ERROR: ipywidgets not installed. Install with: pip install ipywidgets")
        return

    def run_and_plot(alpha=0.3, beta=3.0, set_size=3, num_trials=100):
        """Run simulation and plot results."""
        # Run simulation
        env = create_rlwm_env(set_size=set_size, seed=42)
        agent = create_q_learning_agent(alpha=alpha, beta=beta, seed=42)

        results = simulate_agent_on_env(agent, env, num_trials=num_trials, log_history=False)

        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        # 1. Learning curve
        ax = axes[0]
        window = min(10, len(results['correct']) // 2)
        correct = results['correct']
        if len(correct) >= window:
            ma = np.convolve(correct, np.ones(window)/window, mode='valid')
            ax.plot(ma, linewidth=2, color='steelblue')
        ax.axhline(1/3, color='red', linestyle='--', alpha=0.5, label='Chance (33%)')
        ax.set_xlabel('Trial', fontsize=11)
        ax.set_ylabel(f'Accuracy (MA-{window})', fontsize=11)
        ax.set_title(f'Learning Curve\nα={alpha}, β={beta}, set_size={set_size}',
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])

        # 2. Q-values heatmap
        ax = axes[1]
        q_table = agent.get_q_table()
        im = ax.imshow(q_table[:set_size, :], aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax.set_xlabel('Action', fontsize=11)
        ax.set_ylabel('Stimulus', fontsize=11)
        ax.set_title('Final Q-values', fontsize=12, fontweight='bold')
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(['J (0)', 'K (1)', 'L (2)'])
        ax.set_yticks(range(set_size))
        plt.colorbar(im, ax=ax, label='Q-value')

        # 3. Performance summary
        ax = axes[2]
        ax.axis('off')
        summary_text = f"""
        PERFORMANCE SUMMARY
        {'─' * 30}

        Final Accuracy:     {results['accuracy']:.1%}
        Total Reward:       {results['total_reward']:.0f} / {num_trials}
        Trials Completed:   {results['num_trials']}

        PARAMETERS
        {'─' * 30}

        Learning Rate (α):  {alpha:.2f}
        Inverse Temp (β):   {beta:.1f}
        Set Size:           {set_size}

        INTERPRETATION
        {'─' * 30}
        """

        # Add interpretation
        if results['accuracy'] > 0.8:
            interpretation = "✓ Strong learning!"
        elif results['accuracy'] > 0.5:
            interpretation = "○ Moderate learning"
        else:
            interpretation = "✗ Poor learning"

        if alpha < 0.2:
            alpha_interp = "  Low α → slow learning"
        elif alpha > 0.6:
            alpha_interp = "  High α → fast but volatile"
        else:
            alpha_interp = "  Medium α → balanced"

        if beta < 1.5:
            beta_interp = "  Low β → more exploration"
        elif beta > 5:
            beta_interp = "  High β → strong exploitation"
        else:
            beta_interp = "  Medium β → balanced"

        summary_text += f"\n{interpretation}\n{alpha_interp}\n{beta_interp}"

        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top', family='monospace')

        plt.tight_layout()
        plt.show()

    # Create interactive widget
    interact(
        run_and_plot,
        alpha=FloatSlider(min=0.01, max=0.99, step=0.05, value=0.3,
                         description='α (Learning Rate)', style={'description_width': 'initial'}),
        beta=FloatSlider(min=0.1, max=20, step=0.5, value=3.0,
                        description='β (Inverse Temp)', style={'description_width': 'initial'}),
        set_size=IntSlider(min=2, max=6, step=1, value=3,
                          description='Set Size', style={'description_width': 'initial'}),
        num_trials=IntSlider(min=20, max=200, step=10, value=100,
                            description='Trials', style={'description_width': 'initial'})
    )


def explore_wmrl_interactive():
    """
    Interactive widget to explore WM-RL hybrid parameters.

    Creates interactive sliders for alpha, beta, capacity, w_wm, and task parameters.
    """
    if not WIDGETS_AVAILABLE:
        print("ERROR: ipywidgets not installed. Install with: pip install ipywidgets")
        return

    def run_and_plot(alpha=0.2, beta=3.0, capacity=4, w_wm=0.6, set_size=5, num_trials=100):
        """Run WM-RL simulation and plot results."""
        # Run simulation
        env = create_rlwm_env(set_size=set_size, seed=42)
        agent = create_wm_rl_agent(
            alpha=alpha, beta=beta, capacity=capacity,
            lambda_decay=0.1, w_wm=w_wm, seed=42
        )

        results = simulate_wm_rl_on_env(agent, env, num_trials=num_trials, log_history=False)

        # Create plots
        fig = plt.figure(figsize=(16, 8))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 1. Learning curve
        ax = fig.add_subplot(gs[0, :2])
        window = min(10, len(results['correct']) // 2)
        correct = results['correct']
        if len(correct) >= window:
            ma = np.convolve(correct, np.ones(window)/window, mode='valid')
            ax.plot(ma, linewidth=2, color='steelblue', label='Accuracy')
        ax.axhline(1/3, color='red', linestyle='--', alpha=0.5, label='Chance')
        ax.set_xlabel('Trial', fontsize=11)
        ax.set_ylabel(f'Accuracy (MA-{window})', fontsize=11)
        ax.set_title(f'Learning Curve (WM-RL Hybrid)\nα={alpha}, β={beta}, capacity={capacity}, w_wm={w_wm}',
                     fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])

        # 2. Q-values
        ax = fig.add_subplot(gs[1, 0])
        q_table = agent.get_q_table()
        im = ax.imshow(q_table[:set_size, :], aspect='auto', cmap='viridis', vmin=0, vmax=1)
        ax.set_xlabel('Action', fontsize=10)
        ax.set_ylabel('Stimulus', fontsize=10)
        ax.set_title('Q-values (RL)', fontsize=11, fontweight='bold')
        ax.set_xticks([0, 1, 2])
        plt.colorbar(im, ax=ax, label='Q-value')

        # 3. WM buffer contents
        ax = fig.add_subplot(gs[1, 1])
        wm_contents = agent.get_wm_contents()
        if wm_contents:
            stims = [m['stimulus'] for m in wm_contents]
            actions = [m['action'] for m in wm_contents]
            rewards = [m['reward'] for m in wm_contents]
            strengths = [m['strength'] for m in wm_contents]

            y_pos = np.arange(len(wm_contents))
            colors = ['green' if r > 0 else 'red' for r in rewards]

            ax.barh(y_pos, strengths, color=colors, alpha=0.6)
            ax.set_yticks(y_pos)
            ax.set_yticklabels([f"S{s}→A{a}" for s, a in zip(stims, actions)])
            ax.set_xlabel('Memory Strength', fontsize=10)
            ax.set_title(f'WM Buffer ({len(wm_contents)}/{capacity})', fontsize=11, fontweight='bold')
            ax.set_xlim([0, 1])
        else:
            ax.text(0.5, 0.5, 'WM Buffer Empty', ha='center', va='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.axis('off')

        # 4. Performance summary
        ax = fig.add_subplot(gs[0, 2])
        ax.axis('off')

        wm_retrieval_rate = results.get('wm_retrieval_rate', 0)

        summary_text = f"""
        PERFORMANCE
        {'─' * 20}

        Accuracy:   {results['accuracy']:.1%}
        Total Reward: {results['total_reward']:.0f}
        WM Retrieval: {wm_retrieval_rate:.1%}

        PARAMETERS
        {'─' * 20}

        α (Learning): {alpha:.2f}
        β (Inv Temp): {beta:.1f}
        Capacity:     {capacity}
        w_wm:         {w_wm:.2f}

        INTERPRETATION
        {'─' * 20}
        """

        if set_size > capacity:
            interp = f"Set size ({set_size}) >\nCapacity ({capacity})\n→ RL must help!"
        elif set_size <= capacity:
            interp = f"Set size ≤ capacity\n→ WM can handle all"
        else:
            interp = ""

        if w_wm > 0.7:
            strategy = "Relies heavily on WM"
        elif w_wm < 0.3:
            strategy = "Relies heavily on RL"
        else:
            strategy = "Balanced WM + RL"

        summary_text += f"\n{interp}\n\n{strategy}"

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontsize=9, verticalalignment='top', family='monospace')

        # 5. WM retrieval over time
        ax = fig.add_subplot(gs[1, 2])
        if 'wm_retrieved' in results:
            wm_retrieved = results['wm_retrieved']
            window = min(10, len(wm_retrieved) // 2)
            if len(wm_retrieved) >= window:
                ma = np.convolve(wm_retrieved, np.ones(window)/window, mode='valid')
                ax.plot(ma, linewidth=2, color='purple')
            ax.set_xlabel('Trial', fontsize=10)
            ax.set_ylabel(f'WM Retrieval (MA-{window})', fontsize=10)
            ax.set_title('WM Usage', fontsize=11, fontweight='bold')
            ax.set_ylim([0, 1])
            ax.grid(alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'WM retrieval data\nnot available',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

        plt.show()

    # Create interactive widget
    interact(
        run_and_plot,
        alpha=FloatSlider(min=0.01, max=0.99, step=0.05, value=0.2,
                         description='α (Learning Rate)', style={'description_width': 'initial'}),
        beta=FloatSlider(min=0.1, max=20, step=0.5, value=3.0,
                        description='β (Inverse Temp)', style={'description_width': 'initial'}),
        capacity=IntSlider(min=1, max=7, step=1, value=4,
                          description='WM Capacity', style={'description_width': 'initial'}),
        w_wm=FloatSlider(min=0.0, max=1.0, step=0.1, value=0.6,
                        description='w_wm (WM Weight)', style={'description_width': 'initial'}),
        set_size=IntSlider(min=2, max=6, step=1, value=5,
                          description='Set Size', style={'description_width': 'initial'}),
        num_trials=IntSlider(min=20, max=200, step=10, value=100,
                            description='Trials', style={'description_width': 'initial'})
    )


def compare_models_interactive():
    """
    Interactive comparison between Q-learning and WM-RL.

    Shows side-by-side performance for the same task configuration.
    """
    if not WIDGETS_AVAILABLE:
        print("ERROR: ipywidgets not installed")
        return

    def run_comparison(set_size=4, num_trials=100, alpha=0.3, beta=3.0, capacity=4, w_wm=0.6):
        """Compare Q-learning vs WM-RL."""
        # Q-learning
        env_q = create_rlwm_env(set_size=set_size, seed=42)
        agent_q = create_q_learning_agent(alpha=alpha, beta=beta, seed=42)
        results_q = simulate_agent_on_env(agent_q, env_q, num_trials=num_trials, log_history=False)

        # WM-RL
        env_wmrl = create_rlwm_env(set_size=set_size, seed=42)
        agent_wmrl = create_wm_rl_agent(
            alpha=alpha, beta=beta, capacity=capacity,
            lambda_decay=0.1, w_wm=w_wm, seed=42
        )
        results_wmrl = simulate_wm_rl_on_env(agent_wmrl, env_wmrl, num_trials=num_trials, log_history=False)

        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Learning curves
        ax = axes[0]
        window = min(10, num_trials // 5)

        correct_q = results_q['correct']
        correct_wmrl = results_wmrl['correct']

        if len(correct_q) >= window:
            ma_q = np.convolve(correct_q, np.ones(window)/window, mode='valid')
            ma_wmrl = np.convolve(correct_wmrl, np.ones(window)/window, mode='valid')

            ax.plot(ma_q, linewidth=2, label='Q-learning', color='steelblue')
            ax.plot(ma_wmrl, linewidth=2, label='WM-RL', color='purple')

        ax.axhline(1/3, color='red', linestyle='--', alpha=0.5, label='Chance')
        ax.set_xlabel('Trial', fontsize=12)
        ax.set_ylabel(f'Accuracy (MA-{window})', fontsize=12)
        ax.set_title(f'Model Comparison (set_size={set_size})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        ax.set_ylim([0, 1])

        # Performance bars
        ax = axes[1]
        models = ['Q-learning', 'WM-RL']
        accuracies = [results_q['accuracy'], results_wmrl['accuracy']]
        colors = ['steelblue', 'purple']

        bars = ax.bar(models, accuracies, color=colors, alpha=0.7)
        ax.axhline(1/3, color='red', linestyle='--', alpha=0.5, label='Chance')
        ax.set_ylabel('Final Accuracy', fontsize=12)
        ax.set_title('Model Performance', fontsize=14, fontweight='bold')
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)

        # Add values on bars
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{acc:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        plt.tight_layout()
        plt.show()

        print(f"\nPerformance Comparison:")
        print(f"  Q-learning:  {results_q['accuracy']:.1%}")
        print(f"  WM-RL:       {results_wmrl['accuracy']:.1%}")
        print(f"  Difference:  {(results_wmrl['accuracy'] - results_q['accuracy']):.1%}")

    # Create widget
    interact(
        run_comparison,
        set_size=IntSlider(min=2, max=6, step=1, value=4,
                          description='Set Size', style={'description_width': 'initial'}),
        num_trials=IntSlider(min=50, max=200, step=10, value=100,
                            description='Trials', style={'description_width': 'initial'}),
        alpha=FloatSlider(min=0.01, max=0.99, step=0.05, value=0.3,
                         description='α (both models)', style={'description_width': 'initial'}),
        beta=FloatSlider(min=0.1, max=20, step=0.5, value=3.0,
                        description='β (both models)', style={'description_width': 'initial'}),
        capacity=IntSlider(min=1, max=7, step=1, value=4,
                          description='WM Capacity', style={'description_width': 'initial'}),
        w_wm=FloatSlider(min=0.0, max=1.0, step=0.1, value=0.6,
                        description='w_wm', style={'description_width': 'initial'})
    )


if __name__ == "__main__":
    print("Interactive exploration tools for RLWM models")
    print("\nAvailable functions:")
    print("  - explore_qlearning_interactive()")
    print("  - explore_wmrl_interactive()")
    print("  - compare_models_interactive()")
    print("\nUse these in a Jupyter notebook for interactive parameter exploration!")
