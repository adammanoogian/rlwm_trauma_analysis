"""
Generate Synthetic Data from RL Models

This script generates simulated behavioral data using fitted model posteriors
or specified parameter values. Useful for:
1. Posterior predictive checks
2. Power analyses
3. Parameter recovery studies
4. Model validation

Workflow:
1. Load fitted posteriors (or use default parameters)
2. Create RLWM environment with task structure
3. Run agents (Q-learning or WM-RL) for multiple participants
4. Save simulated data in same format as human data
5. Compare distributions to human data

Usage:
    python simulations/generate_data.py --model qlearning --n-participants 50
    python simulations/generate_data.py --model wmrl --posteriors output/v1/qlearning_posterior.nc
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
from tqdm import tqdm
from typing import Dict, List, Optional

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config import TaskParams, ModelParams, OUTPUT_VERSION_DIR, AnalysisParams, load_netcdf_with_validation

from environments.rlwm_env import create_rlwm_env
from environments.task_config import TaskConfigGenerator
from models.q_learning import QLearningAgent
from models.wm_rl_hybrid import WMRLHybridAgent
from scripts.simulations.unified_simulator import simulate_agent_fixed, simulate_agent_sampled

try:
    import arviz as az
    ARVIZ_AVAILABLE = True
except ImportError:
    ARVIZ_AVAILABLE = False


def sample_parameters_from_posterior(
    posterior_path: Path,
    n_participants: int,
    model_type: str = 'qlearning'
) -> List[Dict[str, float]]:
    """
    Sample parameters from fitted posterior distributions.

    Parameters
    ----------
    posterior_path : Path
        Path to saved InferenceData (.nc file)
    n_participants : int
        Number of participant parameter sets to sample
    model_type : str
        'qlearning' or 'wmrl'

    Returns
    -------
    list
        List of parameter dictionaries
    """
    if not ARVIZ_AVAILABLE:
        raise ImportError("arviz required. Install with: pip install arviz")

    print(f"Loading posterior from: {posterior_path}")
    trace = load_netcdf_with_validation(posterior_path, model_type)

    # Get posterior samples
    posterior = trace.posterior

    # Sample random participants from posterior
    n_chains, n_draws, n_fitted_participants = posterior['alpha'].shape

    param_list = []

    for _ in range(n_participants):
        # Random chain and draw
        chain = np.random.randint(0, n_chains)
        draw = np.random.randint(0, n_draws)
        participant = np.random.randint(0, n_fitted_participants)

        if model_type == 'qlearning':
            params = {
                'alpha': float(posterior['alpha'][chain, draw, participant]),
                'beta': float(posterior['beta'][chain, draw, participant]),
                'gamma': 0.0,  # Usually fixed at 0
            }
        elif model_type == 'wmrl':
            params = {
                'alpha': float(posterior['alpha'][chain, draw, participant]),
                'beta': float(posterior['beta'][chain, draw, participant]),
                'gamma': 0.0,
                'capacity': int(np.round(posterior['capacity'][chain, draw, participant])),
                'lambda_decay': float(posterior['lambda_decay'][chain, draw, participant]),
                'w_wm': float(posterior['w_wm'][chain, draw, participant]),
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        param_list.append(params)

    return param_list


def sample_parameters_from_defaults(
    n_participants: int,
    model_type: str = 'qlearning',
    add_noise: bool = True
) -> List[Dict[str, float]]:
    """
    Sample parameters from default distributions.

    Parameters
    ----------
    n_participants : int
        Number of participants
    model_type : str
        'qlearning' or 'wmrl'
    add_noise : bool
        Whether to add inter-individual variability

    Returns
    -------
    list
        List of parameter dictionaries
    """
    param_list = []

    for _ in range(n_participants):
        if model_type == 'qlearning':
            if add_noise:
                alpha = np.clip(np.random.normal(0.3, 0.15), 0.01, 0.99)
                beta = np.clip(np.random.gamma(2, 1), 0.1, 10)
            else:
                alpha = ModelParams.ALPHA_DEFAULT
                beta = ModelParams.BETA_DEFAULT

            params = {
                'alpha': alpha,
                'beta': beta,
                'gamma': 0.0,
            }

        elif model_type == 'wmrl':
            if add_noise:
                alpha = np.clip(np.random.normal(0.25, 0.12), 0.01, 0.99)
                beta = np.clip(np.random.gamma(2, 1), 0.1, 10)
                capacity = int(np.clip(np.random.normal(4, 1.5), 1, 7))
                lambda_decay = np.clip(np.random.beta(2, 8), 0.0, 1.0)
                w_wm = np.clip(np.random.beta(3, 3), 0.0, 1.0)
            else:
                alpha = ModelParams.ALPHA_DEFAULT
                beta = ModelParams.BETA_DEFAULT
                capacity = ModelParams.WM_CAPACITY_DEFAULT
                lambda_decay = ModelParams.LAMBDA_DECAY_DEFAULT
                w_wm = ModelParams.W_WM_DEFAULT

            params = {
                'alpha': alpha,
                'beta': beta,
                'gamma': 0.0,
                'capacity': capacity,
                'lambda_decay': lambda_decay,
                'w_wm': w_wm,
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        param_list.append(params)

    return param_list


def simulate_participant(
    participant_id: str,
    params: Dict[str, float],
    model_type: str,
    set_sizes: List[int],
    trials_per_block: int = 100,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Simulate one participant across multiple blocks.

    Parameters
    ----------
    participant_id : str
        Participant ID
    params : dict
        Model parameters
    model_type : str
        'qlearning' or 'wmrl'
    set_sizes : list
        Set size for each block
    trials_per_block : int
        Trials per block
    seed : int, optional
        Random seed

    Returns
    -------
    pd.DataFrame
        Simulated trial data
    """
    all_trials = []

    for block_idx, set_size in enumerate(set_sizes):
        block_num = block_idx + 3  # Start from block 3 (main task)

        # Create environment
        env = create_rlwm_env(
            set_size=set_size,
            phase_type='main_task',
            max_trials_per_block=trials_per_block,
            seed=seed
        )

        # Select agent class and run simulation using unified simulator
        if model_type == 'qlearning':
            agent_class = QLearningAgent
            # Ensure required params are present
            sim_params = {
                'num_stimuli': TaskParams.MAX_STIMULI,
                'num_actions': TaskParams.NUM_ACTIONS,
                'alpha': params['alpha'],
                'beta': params['beta'],
                'gamma': params.get('gamma', 0.0),
                'q_init': ModelParams.Q_INIT_VALUE
            }
        elif model_type == 'wmrl':
            agent_class = WMRLHybridAgent
            sim_params = {
                'num_stimuli': TaskParams.MAX_STIMULI,
                'num_actions': TaskParams.NUM_ACTIONS,
                'alpha': params['alpha'],
                'beta': params['beta'],
                'capacity': params['capacity'],
                'lambda_decay': params['lambda_decay'],
                'w_wm': params['w_wm'],
                'gamma': params.get('gamma', 0.0),
                'q_init': ModelParams.Q_INIT_VALUE
            }
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Use unified simulator
        result = simulate_agent_fixed(
            agent_class=agent_class,
            params=sim_params,
            env=env,
            num_trials=trials_per_block,
            seed=seed
        )

        # Create trial dataframe
        for trial_idx in range(len(result.stimuli)):
            trial_data = {
                'participant_id': participant_id,
                'block': block_num,
                'trial': trial_idx + 1,
                'set_size': set_size,
                'load_condition': 'low' if set_size <= 3 else 'high',
                'stimulus': result.stimuli[trial_idx] + 1,  # Convert to 1-indexed
                'key_press': result.actions[trial_idx],
                'correct': result.correct[trial_idx],
                'reward': result.rewards[trial_idx],
            }

            # Add parameters (for reference)
            trial_data.update({f'param_{k}': v for k, v in params.items()})

            all_trials.append(trial_data)

    return pd.DataFrame(all_trials)


def generate_dataset(
    n_participants: int,
    model_type: str,
    param_list: List[Dict[str, float]],
    num_blocks: int = 21,
    trials_per_block: int = 100,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate full dataset for multiple participants.

    Parameters
    ----------
    n_participants : int
        Number of participants
    model_type : str
        'qlearning' or 'wmrl'
    param_list : list
        List of parameter dicts
    num_blocks : int
        Number of blocks per participant
    trials_per_block : int
        Trials per block
    seed : int, optional
        Random seed

    Returns
    -------
    pd.DataFrame
        Full simulated dataset
    """
    # Generate block sequences
    generator = TaskConfigGenerator(seed=seed)
    block_sequences = [
        generator.generate_block_sequence(num_blocks, seed=seed + i if seed else None)
        for i in range(n_participants)
    ]

    # Simulate each participant
    all_data = []

    print(f"\nSimulating {n_participants} participants...")
    for i in tqdm(range(n_participants)):
        participant_id = f'SIM_{i+1:04d}'
        params = param_list[i]
        set_sizes = block_sequences[i]

        participant_data = simulate_participant(
            participant_id=participant_id,
            params=params,
            model_type=model_type,
            set_sizes=set_sizes,
            trials_per_block=trials_per_block,
            seed=seed + i if seed else None
        )

        all_data.append(participant_data)

    return pd.concat(all_data, ignore_index=True)


def main():
    """Main simulation workflow."""
    parser = argparse.ArgumentParser(description='Generate synthetic behavioral data')
    parser.add_argument('--model', type=str, default='qlearning',
                        choices=['qlearning', 'wmrl'],
                        help='Model type')
    parser.add_argument('--n-participants', type=int, default=50,
                        help='Number of simulated participants')
    parser.add_argument('--num-blocks', type=int, default=21,
                        help='Number of blocks per participant')
    parser.add_argument('--trials-per-block', type=int, default=100,
                        help='Trials per block')
    parser.add_argument('--posteriors', type=str, default=None,
                        help='Path to fitted posterior .nc file (optional)')
    parser.add_argument('--add-noise', action='store_true',
                        help='Add inter-individual variability to default params')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_VERSION_DIR),
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=AnalysisParams.RANDOM_SEED,
                        help='Random seed')

    args = parser.parse_args()

    print("=" * 80)
    print("SYNTHETIC DATA GENERATION FOR RLWM TASK")
    print("=" * 80)
    print(f"\nSettings:")
    print(f"  Model: {args.model}")
    print(f"  Participants: {args.n_participants}")
    print(f"  Blocks: {args.num_blocks}")
    print(f"  Trials/block: {args.trials_per_block}")
    print(f"  Posteriors: {args.posteriors or 'None (using defaults)'}")
    print(f"  Seed: {args.seed}")

    # Sample parameters
    if args.posteriors:
        param_list = sample_parameters_from_posterior(
            Path(args.posteriors),
            args.n_participants,
            args.model
        )
        print(f"\nSampled {len(param_list)} parameter sets from posterior")
    else:
        param_list = sample_parameters_from_defaults(
            args.n_participants,
            args.model,
            add_noise=args.add_noise
        )
        print(f"\nGenerated {len(param_list)} parameter sets from defaults")

    # Show example parameters
    print("\nExample parameters:")
    for key, val in param_list[0].items():
        print(f"  {key}: {val:.3f}" if isinstance(val, float) else f"  {key}: {val}")

    # Generate data
    data = generate_dataset(
        n_participants=args.n_participants,
        model_type=args.model,
        param_list=param_list,
        num_blocks=args.num_blocks,
        trials_per_block=args.trials_per_block,
        seed=args.seed
    )

    print(f"\nGenerated {len(data)} trials")

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f'simulated_data_{args.model}_{timestamp}.csv'

    data.to_csv(output_file, index=False)
    print(f"\nSaved simulated data to: {output_file}")

    # Summary statistics
    print("\nSummary statistics:")
    print(f"  Overall accuracy: {data['correct'].mean():.3f}")
    print(f"  Accuracy by set size:")
    for set_size, group in data.groupby('set_size'):
        print(f"    Set size {set_size}: {group['correct'].mean():.3f}")

    print("\n" + "=" * 80)
    print("SIMULATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
