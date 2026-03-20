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

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from rlwm.config import TaskParams


class TaskSequenceLoader:
    """
    Loads and manages task sequences from jsPsych CSV files.

    The sequence files (sequence0.csv through sequence19.csv) contain
    the exact trial-by-trial structure of the task.
    """

    def __init__(self, sequence_dir: Path | None = None):
        if sequence_dir is None:
            # Default: look for sequences relative to this file
            pkg_root = Path(__file__).parent.parent.parent.parent
            rlwm_trauma_dir = pkg_root.parent / "rlwm_trauma" / "js"
            if rlwm_trauma_dir.exists():
                sequence_dir = rlwm_trauma_dir
            else:
                sequence_dir = pkg_root / "sequences"

        self.sequence_dir = Path(sequence_dir)
        self.sequences: dict[int, dict[str, np.ndarray]] = {}

    def load_sequence(self, sequence_id: int) -> dict[str, np.ndarray]:
        if sequence_id in self.sequences:
            return self.sequences[sequence_id]

        sequence_file = self.sequence_dir / f"sequence{sequence_id}.csv"
        if not sequence_file.exists():
            raise FileNotFoundError(f"Sequence file not found: {sequence_file}")

        df = pd.read_csv(sequence_file, header=None)

        stims = df.iloc[0].values.astype(int) - 1
        correct_keys = df.iloc[1].values.astype(int)
        set_sizes = df.iloc[2].values.astype(int)
        blocks = df.iloc[3].values.astype(int)
        img_folders = df.iloc[4].values.astype(int)

        correct_keys = np.where(correct_keys == 999, 999, correct_keys)

        sequence = {
            "stims": stims,
            "correct_keys": correct_keys,
            "set_sizes": set_sizes,
            "blocks": blocks,
            "img_folders": img_folders,
        }

        self.sequences[sequence_id] = sequence
        return sequence

    def get_block_trials(
        self, sequence_id: int, block: int
    ) -> dict[str, np.ndarray]:
        sequence = self.load_sequence(sequence_id)
        block_mask = sequence["blocks"] == block
        return {
            "stims": sequence["stims"][block_mask],
            "correct_keys": sequence["correct_keys"][block_mask],
            "set_sizes": sequence["set_sizes"][block_mask],
            "blocks": sequence["blocks"][block_mask],
            "img_folders": sequence["img_folders"][block_mask],
        }

    def get_block_config(self, sequence_id: int, block: int) -> dict:
        block_data = self.get_block_trials(sequence_id, block)
        set_size = int(block_data["set_sizes"][0])
        num_trials = len(block_data["stims"])
        stimuli = np.unique(block_data["stims"])

        initial_correct = {}
        for stim in stimuli:
            first_idx = np.where(block_data["stims"] == stim)[0][0]
            correct_key = block_data["correct_keys"][first_idx]
            if correct_key != 999:
                initial_correct[int(stim)] = int(correct_key)

        return {
            "set_size": set_size,
            "num_trials": num_trials,
            "stimuli": stimuli.tolist(),
            "initial_correct_responses": initial_correct,
        }

    def create_block_sequence(
        self,
        sequence_id: int,
        start_block: int = 3,
        end_block: int = 23,
    ) -> list[int]:
        sequence = self.load_sequence(sequence_id)
        set_sizes = []
        for block in range(start_block, end_block + 1):
            block_mask = sequence["blocks"] == block
            if np.any(block_mask):
                set_size = sequence["set_sizes"][block_mask][0]
                set_sizes.append(int(set_size))
        return set_sizes


class TaskConfigGenerator:
    """Generates task configurations for simulation without sequence files."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.RandomState(seed)

    def generate_block_config(
        self,
        set_size: int,
        num_trials: int = 100,
        reversal_prob: float = 0.3,
        seed: int | None = None,
    ) -> dict:
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        stimuli = list(range(set_size))
        initial_correct = {
            stim: self.rng.choice(TaskParams.ACTIONS) for stim in stimuli
        }
        trial_stimuli = self.rng.choice(stimuli, size=num_trials)

        return {
            "set_size": set_size,
            "num_trials": num_trials,
            "stimuli": stimuli,
            "initial_correct_responses": initial_correct,
            "trial_stimuli": trial_stimuli,
        }

    def generate_block_sequence(
        self,
        num_blocks: int = 21,
        set_size_distribution: list[int] | None = None,
        seed: int | None = None,
    ) -> list[int]:
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        if set_size_distribution is None:
            set_size_distribution = TaskParams.SET_SIZES
        return self.rng.choice(set_size_distribution, size=num_blocks).tolist()


def load_task_sequence(
    sequence_id: int,
    sequence_dir: Path | None = None,
) -> TaskSequenceLoader:
    loader = TaskSequenceLoader(sequence_dir)
    loader.load_sequence(sequence_id)
    return loader


def generate_synthetic_config(
    num_blocks: int = 21,
    seed: int | None = None,
) -> tuple[list[int], TaskConfigGenerator]:
    generator = TaskConfigGenerator(seed)
    set_sizes = generator.generate_block_sequence(num_blocks, seed=seed)
    return set_sizes, generator
