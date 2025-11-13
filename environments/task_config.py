"""
Task Configuration Loader for RLWM Environment

Loads sequence files from the jsPsych implementation to ensure exact
replication of the experimental design.

Sequence files contain:
- Row 1: allStims - stimulus IDs for each trial
- Row 2: corKey - initial correct responses
- Row 3: setSizes - set size per trial
- Row 4: allBlocks - block numbers
- Row 5: imgFolders - image folder IDs
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import TaskParams


class TaskSequenceLoader:
    """
    Loads and manages task sequences from jsPsych CSV files.

    The sequence files (sequence0.csv through sequence19.csv) contain
    the exact trial-by-trial structure of the task, including stimulus
    order, initial correct responses, set sizes, and block assignments.
    """

    def __init__(self, sequence_dir: Optional[Path] = None):
        """
        Initialize the sequence loader.

        Parameters
        ----------
        sequence_dir : Path, optional
            Directory containing sequence CSV files.
            If None, looks for sequences in the rlwm_trauma project.
        """
        if sequence_dir is None:
            # Try to find rlwm_trauma directory
            rlwm_trauma_dir = project_root.parent / 'rlwm_trauma' / 'js'
            if rlwm_trauma_dir.exists():
                sequence_dir = rlwm_trauma_dir
            else:
                # Fall back to relative path
                sequence_dir = project_root / 'sequences'

        self.sequence_dir = Path(sequence_dir)
        self.sequences = {}  # Cache loaded sequences

    def load_sequence(self, sequence_id: int) -> Dict[str, np.ndarray]:
        """
        Load a specific sequence file.

        Parameters
        ----------
        sequence_id : int
            Sequence number (0-19)

        Returns
        -------
        dict
            Dictionary with keys:
            - 'stims': stimulus IDs per trial
            - 'correct_keys': initial correct responses per trial (0-2)
            - 'set_sizes': set sizes per trial
            - 'blocks': block numbers per trial
            - 'img_folders': image folder IDs per trial
        """
        if sequence_id in self.sequences:
            return self.sequences[sequence_id]

        sequence_file = self.sequence_dir / f'sequence{sequence_id}.csv'

        if not sequence_file.exists():
            raise FileNotFoundError(f"Sequence file not found: {sequence_file}")

        # Load CSV (no header, 5 rows)
        df = pd.read_csv(sequence_file, header=None)

        # Parse rows
        stims = df.iloc[0].values.astype(int) - 1  # Convert to 0-indexed
        correct_keys = df.iloc[1].values.astype(int)  # 999 for practice, 0-2 for main
        set_sizes = df.iloc[2].values.astype(int)
        blocks = df.iloc[3].values.astype(int)
        img_folders = df.iloc[4].values.astype(int)

        # Convert correct_keys to 0-indexed (999 stays as-is for practice)
        correct_keys = np.where(correct_keys == 999, 999, correct_keys)

        sequence = {
            'stims': stims,
            'correct_keys': correct_keys,
            'set_sizes': set_sizes,
            'blocks': blocks,
            'img_folders': img_folders,
        }

        # Cache
        self.sequences[sequence_id] = sequence

        return sequence

    def get_block_trials(
        self,
        sequence_id: int,
        block: int
    ) -> Dict[str, np.ndarray]:
        """
        Get all trials for a specific block from a sequence.

        Parameters
        ----------
        sequence_id : int
            Sequence number (0-19)
        block : int
            Block number (1-23)

        Returns
        -------
        dict
            Dictionary with trial data for the block
        """
        sequence = self.load_sequence(sequence_id)

        # Filter by block
        block_mask = sequence['blocks'] == block

        return {
            'stims': sequence['stims'][block_mask],
            'correct_keys': sequence['correct_keys'][block_mask],
            'set_sizes': sequence['set_sizes'][block_mask],
            'blocks': sequence['blocks'][block_mask],
            'img_folders': sequence['img_folders'][block_mask],
        }

    def get_block_config(
        self,
        sequence_id: int,
        block: int
    ) -> Dict[str, any]:
        """
        Get configuration for a specific block.

        Parameters
        ----------
        sequence_id : int
            Sequence number (0-19)
        block : int
            Block number (1-23)

        Returns
        -------
        dict
            Block configuration including:
            - 'set_size': set size for this block
            - 'num_trials': number of trials
            - 'stimuli': unique stimulus IDs in block
            - 'initial_correct_responses': dict mapping stimulus to initial correct response
        """
        block_data = self.get_block_trials(sequence_id, block)

        set_size = int(block_data['set_sizes'][0])  # Should be constant within block
        num_trials = len(block_data['stims'])
        stimuli = np.unique(block_data['stims'])

        # Get initial correct responses (first occurrence of each stimulus)
        initial_correct = {}
        for stim in stimuli:
            first_idx = np.where(block_data['stims'] == stim)[0][0]
            correct_key = block_data['correct_keys'][first_idx]
            if correct_key != 999:  # Skip practice blocks with 999
                initial_correct[int(stim)] = int(correct_key)

        return {
            'set_size': set_size,
            'num_trials': num_trials,
            'stimuli': stimuli.tolist(),
            'initial_correct_responses': initial_correct,
        }

    def create_block_sequence(
        self,
        sequence_id: int,
        start_block: int = 3,
        end_block: int = 23
    ) -> List[int]:
        """
        Create a list of set sizes for each block in the main task.

        Parameters
        ----------
        sequence_id : int
            Sequence number (0-19)
        start_block : int
            First block to include (default: 3, first main task block)
        end_block : int
            Last block to include (default: 23)

        Returns
        -------
        list
            List of set sizes, one per block
        """
        sequence = self.load_sequence(sequence_id)

        set_sizes = []
        for block in range(start_block, end_block + 1):
            block_mask = sequence['blocks'] == block
            if np.any(block_mask):
                set_size = sequence['set_sizes'][block_mask][0]
                set_sizes.append(int(set_size))

        return set_sizes


class TaskConfigGenerator:
    """
    Generates task configurations for simulation without requiring sequence files.

    This is useful for creating synthetic task structures with controlled parameters.
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the configuration generator.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility
        """
        self.rng = np.random.RandomState(seed)

    def generate_block_config(
        self,
        set_size: int,
        num_trials: int = 100,
        reversal_prob: float = 0.3,
        seed: Optional[int] = None
    ) -> Dict[str, any]:
        """
        Generate a synthetic block configuration.

        Parameters
        ----------
        set_size : int
            Number of unique stimuli (2, 3, 5, or 6)
        num_trials : int
            Number of trials in the block
        reversal_prob : float
            Probability that each stimulus will have a reversal
        seed : int, optional
            Random seed

        Returns
        -------
        dict
            Block configuration with:
            - 'set_size': set size
            - 'num_trials': number of trials
            - 'stimuli': list of stimulus IDs
            - 'initial_correct_responses': dict mapping stimulus to initial correct response
            - 'trial_stimuli': array of stimulus IDs for each trial
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        # Create stimulus list
        stimuli = list(range(set_size))

        # Assign random initial correct responses
        initial_correct = {
            stim: self.rng.choice(TaskParams.ACTIONS)
            for stim in stimuli
        }

        # Generate trial sequence (sample stimuli with replacement)
        trial_stimuli = self.rng.choice(stimuli, size=num_trials)

        return {
            'set_size': set_size,
            'num_trials': num_trials,
            'stimuli': stimuli,
            'initial_correct_responses': initial_correct,
            'trial_stimuli': trial_stimuli,
        }

    def generate_block_sequence(
        self,
        num_blocks: int = 21,
        set_size_distribution: Optional[List[int]] = None,
        seed: Optional[int] = None
    ) -> List[int]:
        """
        Generate a sequence of set sizes for multiple blocks.

        Parameters
        ----------
        num_blocks : int
            Number of blocks
        set_size_distribution : list, optional
            List of set sizes to sample from. If None, uses TaskParams.SET_SIZES.
        seed : int, optional
            Random seed

        Returns
        -------
        list
            List of set sizes, one per block
        """
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        if set_size_distribution is None:
            set_size_distribution = TaskParams.SET_SIZES

        # Sample set sizes
        set_sizes = self.rng.choice(set_size_distribution, size=num_blocks).tolist()

        return set_sizes


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_task_sequence(
    sequence_id: int,
    sequence_dir: Optional[Path] = None
) -> TaskSequenceLoader:
    """
    Load a task sequence from file.

    Parameters
    ----------
    sequence_id : int
        Sequence number (0-19)
    sequence_dir : Path, optional
        Directory containing sequence files

    Returns
    -------
    TaskSequenceLoader
        Loaded sequence loader
    """
    loader = TaskSequenceLoader(sequence_dir)
    loader.load_sequence(sequence_id)
    return loader


def generate_synthetic_config(
    num_blocks: int = 21,
    seed: Optional[int] = None
) -> Tuple[List[int], TaskConfigGenerator]:
    """
    Generate a synthetic task configuration.

    Parameters
    ----------
    num_blocks : int
        Number of blocks
    seed : int, optional
        Random seed

    Returns
    -------
    set_sizes : list
        List of set sizes per block
    generator : TaskConfigGenerator
        Generator object for creating block configs
    """
    generator = TaskConfigGenerator(seed)
    set_sizes = generator.generate_block_sequence(num_blocks, seed=seed)
    return set_sizes, generator


def test_sequence_loader():
    """Test loading sequence files."""
    print("Testing TaskSequenceLoader")
    print("=" * 80)

    # Try to load sequence 0
    try:
        loader = TaskSequenceLoader()
        sequence = loader.load_sequence(0)

        print(f"\nLoaded sequence 0:")
        print(f"  Total trials: {len(sequence['stims'])}")
        print(f"  Unique blocks: {np.unique(sequence['blocks'])}")
        print(f"  Set sizes: {np.unique(sequence['set_sizes'])}")

        # Get block 3 config (first main task block)
        block_config = loader.get_block_config(0, 3)
        print(f"\nBlock 3 configuration:")
        print(f"  Set size: {block_config['set_size']}")
        print(f"  Number of trials: {block_config['num_trials']}")
        print(f"  Stimuli: {block_config['stimuli']}")
        print(f"  Initial correct responses: {block_config['initial_correct_responses']}")

    except FileNotFoundError as e:
        print(f"\nSequence files not found: {e}")
        print("This is expected if rlwm_trauma repository is not in the parent directory.")

    # Test synthetic generation
    print("\n" + "=" * 80)
    print("Testing TaskConfigGenerator (synthetic)")
    print("=" * 80)

    generator = TaskConfigGenerator(seed=42)
    block_config = generator.generate_block_config(set_size=3, num_trials=50)

    print(f"\nGenerated block configuration:")
    print(f"  Set size: {block_config['set_size']}")
    print(f"  Number of trials: {block_config['num_trials']}")
    print(f"  Stimuli: {block_config['stimuli']}")
    print(f"  Initial correct responses: {block_config['initial_correct_responses']}")
    print(f"  Trial stimuli (first 10): {block_config['trial_stimuli'][:10]}")

    # Generate block sequence
    set_sizes = generator.generate_block_sequence(num_blocks=21, seed=42)
    print(f"\nGenerated block sequence (21 blocks):")
    print(f"  Set sizes: {set_sizes}")
    print(f"  Set size distribution: {dict(zip(*np.unique(set_sizes, return_counts=True)))}")

    print("\n" + "=" * 80)
    print("Test complete!")


if __name__ == "__main__":
    test_sequence_loader()
