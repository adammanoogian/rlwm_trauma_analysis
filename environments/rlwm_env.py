"""
RLWM Environment: Reinforcement Learning + Working Memory Task

Trial-based gym environment for the RLWM task. Each env.step() represents
one complete trial: stimulus presentation → response → feedback.

Based on the jsPsych implementation with:
- Variable set sizes (2, 3, 5, 6 stimuli)
- 3-choice responses (J/K/L → actions 0/1/2)
- Rare, late reversals (12-18 consecutive correct)
- Binary rewards (+1 correct, 0 incorrect)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Tuple, Optional, Any
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import TaskParams


class RLWMEnv(gym.Env):
    """
    Reinforcement Learning + Working Memory Environment.

    This is a trial-based environment where each step represents a complete trial.
    The agent must learn stimulus-response mappings that occasionally reverse.

    Observation Space
    -----------------
    Dict with keys:
        - 'stimulus': Discrete(6) - which stimulus (0-5)
        - 'set_size': Box(1,) - current set size (2, 3, 5, or 6)
        - 'block': Discrete(24) - current block (0-23)
        - 'phase': Discrete(3) - phase type (0=practice_static, 1=practice_dynamic, 2=main)

    Action Space
    ------------
    Discrete(3) - which key to press (0=J, 1=K, 2=L)

    Rewards
    -------
    +1 for correct response, 0 for incorrect

    Episode Termination
    -------------------
    Episode ends when max_trials_per_block is reached or when truncated.

    Parameters
    ----------
    set_size : int, optional
        Fixed set size for all blocks. If None, will vary according to task design.
    block_sequence : list, optional
        Sequence of set sizes per block. If None, uses default task structure.
    max_trials_per_block : int, optional
        Maximum trials per block. Default is 100.
    phase_type : str, optional
        Task phase: 'practice_static', 'practice_dynamic', or 'main_task'.
        Default is 'main_task'.
    seed : int, optional
        Random seed for reproducibility.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        set_size: Optional[int] = None,
        block_sequence: Optional[list] = None,
        max_trials_per_block: int = 100,
        phase_type: str = 'main_task',
        seed: Optional[int] = None,
    ):
        super().__init__()

        # Set random seed
        self.seed_value = seed
        self.rng = np.random.RandomState(seed)

        # Task parameters
        self.set_size = set_size
        self.block_sequence = block_sequence
        self.max_trials_per_block = max_trials_per_block
        self.phase_type = phase_type

        # Define observation space
        self.observation_space = spaces.Dict({
            'stimulus': spaces.Discrete(TaskParams.MAX_STIMULI),
            'set_size': spaces.Box(
                low=min(TaskParams.SET_SIZES),
                high=max(TaskParams.SET_SIZES),
                shape=(1,),
                dtype=np.int32
            ),
            'block': spaces.Discrete(TaskParams.TOTAL_BLOCKS),
            'phase': spaces.Discrete(3),  # 0=practice_static, 1=practice_dynamic, 2=main
        })

        # Define action space
        self.action_space = spaces.Discrete(TaskParams.NUM_ACTIONS)

        # Initialize state variables
        self.current_block = 0
        self.trial_in_block = 0
        self.current_stimulus = 0
        self.current_set_size = 2

        # Stimulus-response mappings (correct_responses[stimulus] = correct_action)
        self.correct_responses = np.zeros(TaskParams.MAX_STIMULI, dtype=np.int32)

        # Reversal tracking
        self.reversal_points = np.full(TaskParams.MAX_STIMULI, np.inf)
        self.correct_counters = np.zeros(TaskParams.MAX_STIMULI, dtype=np.int32)
        self.reversals_done = np.zeros(TaskParams.MAX_STIMULI, dtype=np.int32)

        # Trial history
        self.trial_count = 0
        self.episode_rewards = []
        self.episode_stimuli = []
        self.episode_actions = []
        self.episode_correct = []

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict, Dict]:
        """
        Reset the environment for a new block.

        Parameters
        ----------
        seed : int, optional
            Random seed
        options : dict, optional
            Additional options (can specify 'block', 'set_size', 'phase_type')

        Returns
        -------
        observation : dict
            Initial observation
        info : dict
            Additional information
        """
        # Handle seed
        if seed is not None:
            self.seed_value = seed
            self.rng = np.random.RandomState(seed)

        # Handle options
        if options:
            if 'block' in options:
                self.current_block = options['block']
            if 'set_size' in options:
                self.current_set_size = options['set_size']
            if 'phase_type' in options:
                self.phase_type = options['phase_type']

        # Initialize block
        self.trial_in_block = 0
        self.trial_count = 0

        # Determine set size for this block
        if self.set_size is not None:
            # Fixed set size mode
            self.current_set_size = self.set_size
        elif self.block_sequence is not None and len(self.block_sequence) > self.current_block:
            # Use provided sequence
            self.current_set_size = self.block_sequence[self.current_block]
        else:
            # Random set size from allowed values
            self.current_set_size = self.rng.choice(TaskParams.SET_SIZES)

        # Initialize correct responses randomly for active stimuli
        for stim in range(self.current_set_size):
            self.correct_responses[stim] = self.rng.choice(TaskParams.ACTIONS)

        # Initialize reversal points
        self._initialize_reversals()

        # Reset counters
        self.correct_counters.fill(0)
        self.reversals_done.fill(0)

        # Reset history
        self.episode_rewards = []
        self.episode_stimuli = []
        self.episode_actions = []
        self.episode_correct = []

        # Sample first stimulus
        self.current_stimulus = self._sample_stimulus()

        # Create observation
        obs = self._get_observation()

        # Info
        info = {
            'block': self.current_block,
            'trial': self.trial_in_block,
            'set_size': self.current_set_size,
            'phase_type': self.phase_type,
            'correct_response': self.correct_responses[self.current_stimulus],
        }

        return obs, info

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one trial: agent responds to stimulus, receives feedback.

        Parameters
        ----------
        action : int
            Action taken (0, 1, or 2)

        Returns
        -------
        observation : dict
            Next observation
        reward : float
            Reward received (+1 or 0)
        terminated : bool
            Whether episode ended (always False in this environment)
        truncated : bool
            Whether episode was truncated (max trials reached)
        info : dict
            Additional information
        """
        # Check if action is correct
        correct_action = self.correct_responses[self.current_stimulus]
        is_correct = (action == correct_action)

        # Compute reward
        reward = TaskParams.REWARD_CORRECT if is_correct else TaskParams.REWARD_INCORRECT

        # Update counters
        if is_correct:
            self.correct_counters[self.current_stimulus] += 1
        else:
            self.correct_counters[self.current_stimulus] = 0

        # Check for reversal
        reversal_occurred = False
        if (self.correct_counters[self.current_stimulus] >= self.reversal_points[self.current_stimulus] and
            self.reversals_done[self.current_stimulus] < TaskParams.MAX_REVERSALS_PER_STIM):
            reversal_occurred = True
            self._execute_reversal(self.current_stimulus)

        # Store history
        self.episode_rewards.append(reward)
        self.episode_stimuli.append(self.current_stimulus)
        self.episode_actions.append(action)
        self.episode_correct.append(is_correct)

        # Increment trial counters
        self.trial_in_block += 1
        self.trial_count += 1

        # Check if block is complete
        truncated = self.trial_in_block >= self.max_trials_per_block
        terminated = False  # Episodes don't naturally terminate

        # Sample next stimulus
        if not truncated:
            self.current_stimulus = self._sample_stimulus()

        # Create next observation
        obs = self._get_observation()

        # Info
        info = {
            'block': self.current_block,
            'trial': self.trial_in_block,
            'set_size': self.current_set_size,
            'phase_type': self.phase_type,
            'stimulus': self.episode_stimuli[-1],
            'action': action,
            'correct_response': correct_action,
            'is_correct': is_correct,
            'reversal_occurred': reversal_occurred,
            'correct_counter': self.correct_counters[self.episode_stimuli[-1]],
            'accuracy': np.mean(self.episode_correct),
            'total_reward': sum(self.episode_rewards),
        }

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> Dict:
        """
        Create observation dictionary.

        Returns
        -------
        dict
            Observation with stimulus, set_size, block, phase
        """
        # Map phase_type to integer
        phase_map = {
            'practice_static': 0,
            'practice_dynamic': 1,
            'main_task': 2,
        }
        phase_int = phase_map.get(self.phase_type, 2)

        return {
            'stimulus': self.current_stimulus,
            'set_size': np.array([self.current_set_size], dtype=np.int32),
            'block': self.current_block,
            'phase': phase_int,
        }

    def _sample_stimulus(self) -> int:
        """
        Sample a stimulus from the current set.

        Returns
        -------
        int
            Stimulus index (0 to set_size-1)
        """
        return self.rng.choice(self.current_set_size)

    def _initialize_reversals(self):
        """Initialize reversal points for each stimulus based on phase type."""
        if self.phase_type == 'practice_static':
            # No reversals in practice_static
            self.reversal_points.fill(np.inf)

        elif self.phase_type == 'practice_dynamic':
            # Reversals occur after fixed number of correct responses
            for stim in range(self.current_set_size):
                self.reversal_points[stim] = TaskParams.PRACTICE_DYNAMIC_REVERSAL_CRITERION

        elif self.phase_type == 'main_task':
            # Rare, late reversals
            if TaskParams.RARE_REVERSALS:
                for stim in range(self.current_set_size):
                    self.reversal_points[stim] = self.rng.randint(
                        TaskParams.REVERSAL_MIN,
                        TaskParams.REVERSAL_MAX + 1
                    )
            else:
                # No reversals
                self.reversal_points.fill(np.inf)
        else:
            # Default: no reversals
            self.reversal_points.fill(np.inf)

    def _execute_reversal(self, stimulus: int):
        """
        Execute a reversal for the given stimulus.

        Parameters
        ----------
        stimulus : int
            Stimulus index to reverse
        """
        # Reset counter
        self.correct_counters[stimulus] = 0

        # Increment reversals done
        self.reversals_done[stimulus] += 1

        # Choose new correct response (different from current)
        current_response = self.correct_responses[stimulus]
        possible_responses = [a for a in TaskParams.ACTIONS if a != current_response]
        self.correct_responses[stimulus] = self.rng.choice(possible_responses)

        # Block future reversals for this stimulus (in main task)
        if self.phase_type == 'main_task':
            self.reversal_points[stimulus] = np.inf
        elif self.phase_type == 'practice_dynamic':
            # In practice_dynamic, set new reversal point
            self.reversal_points[stimulus] = TaskParams.PRACTICE_DYNAMIC_REVERSAL_CRITERION

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Compute performance metrics for the current episode.

        Returns
        -------
        dict
            Performance metrics including accuracy, total reward, etc.
        """
        if len(self.episode_correct) == 0:
            return {
                'accuracy': 0.0,
                'total_reward': 0.0,
                'num_trials': 0,
                'num_reversals': 0,
            }

        return {
            'accuracy': np.mean(self.episode_correct),
            'total_reward': sum(self.episode_rewards),
            'num_trials': len(self.episode_correct),
            'num_reversals': int(np.sum(self.reversals_done)),
        }

    def seed(self, seed: int):
        """
        Set the random seed.

        Parameters
        ----------
        seed : int
            Random seed value
        """
        self.seed_value = seed
        self.rng = np.random.RandomState(seed)

    def render(self):
        """Render the environment (not implemented)."""
        pass

    def close(self):
        """Close the environment (nothing to close)."""
        pass


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def create_rlwm_env(
    set_size: Optional[int] = None,
    phase_type: str = 'main_task',
    seed: Optional[int] = None,
    **kwargs
) -> RLWMEnv:
    """
    Factory function to create RLWM environment with common configurations.

    Parameters
    ----------
    set_size : int, optional
        Fixed set size. If None, varies across blocks.
    phase_type : str
        Task phase: 'practice_static', 'practice_dynamic', or 'main_task'
    seed : int, optional
        Random seed
    **kwargs
        Additional arguments passed to RLWMEnv

    Returns
    -------
    RLWMEnv
        Configured environment
    """
    return RLWMEnv(
        set_size=set_size,
        phase_type=phase_type,
        seed=seed,
        **kwargs
    )


def test_environment():
    """Test the RLWM environment with random actions."""
    print("Testing RLWM Environment")
    print("=" * 80)

    # Create environment
    env = create_rlwm_env(set_size=3, phase_type='main_task', seed=42)

    # Reset
    obs, info = env.reset()
    print(f"\nInitial observation: {obs}")
    print(f"Initial info: {info}")

    # Run 20 trials
    print("\nRunning 20 trials with random actions...")
    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Trial {i+1}: stim={info['stimulus']}, "
              f"action={action}, correct={info['correct_response']}, "
              f"reward={reward:.0f}, correct={info['is_correct']}, "
              f"acc={info['accuracy']:.2f}")

        if info['reversal_occurred']:
            print(f"  → REVERSAL occurred for stimulus {info['stimulus']}!")

        if terminated or truncated:
            print(f"\nEpisode ended. Metrics: {env.get_performance_metrics()}")
            break

    print("\n" + "=" * 80)
    print("Test complete!")


if __name__ == "__main__":
    test_environment()
