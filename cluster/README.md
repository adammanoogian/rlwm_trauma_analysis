# Cluster Setup for RLWM Trauma Analysis

This directory contains scripts for running the MLE fitting pipeline on a university SLURM cluster.

## Quick Start

```bash
# 1. Clone the repo on the cluster
git clone https://github.com/YOUR_USERNAME/rlwm_trauma_analysis.git
cd rlwm_trauma_analysis

# 2. Load conda module (cluster-specific)
module load anaconda  # or: module load miniconda

# 3. Set up the environment
chmod +x cluster/setup_env.sh
./cluster/setup_env.sh

# 4. Run MLE fitting (all models)
sbatch cluster/run_mle.slurm
```

## Scripts

| Script | Description |
|--------|-------------|
| `setup_env.sh` | Creates the conda environment from `environment.yml` |
| `run_mle.slurm` | SLURM job to fit all 3 models (M1, M2, M3) |
| `run_mle_single.slurm` | SLURM job for single model with timing options |

## Usage Examples

### Full MLE Fitting (All Models)

```bash
sbatch cluster/run_mle.slurm
```

This runs Q-learning (M1), WM-RL (M2), and WM-RL with perseveration (M3) sequentially.

### Single Model

```bash
# Q-learning only
sbatch --export=MODEL=qlearning cluster/run_mle.slurm

# WM-RL M3 only
sbatch --export=MODEL=wmrl_m3 cluster/run_mle.slurm
```

### Timing Test (Quick)

Test on a few participants first to estimate total runtime:

```bash
# Test with 3 participants (separates JIT compilation from steady-state)
sbatch --export=MODEL=wmrl_m3,LIMIT=3 cluster/run_mle_single.slurm
```

The output will show:
- JIT compilation time (1st participant)
- Steady-state time per participant
- Extrapolated total time for full dataset

## Output Files

Results are saved to `output/mle/`:

```
output/mle/
├── qlearning_individual_fits.csv    # Per-participant parameters
├── qlearning_group_summary.csv      # Group statistics
├── wmrl_individual_fits.csv
├── wmrl_group_summary.csv
├── wmrl_m3_individual_fits.csv
└── wmrl_m3_group_summary.csv
```

Job logs are saved to `cluster/logs/`.

## Resource Requirements

Default SLURM settings:
- **Time**: 4 hours (all models) / 2 hours (single model)
- **Memory**: 8 GB
- **CPUs**: 4

Adjust in the SLURM script headers if needed:
```bash
#SBATCH --time=06:00:00    # Increase time
#SBATCH --mem=16G          # Increase memory
```

## Troubleshooting

### "conda not found"

Load the appropriate module for your cluster:
```bash
module load anaconda3    # or
module load miniconda    # or
module load conda
```

### Data file not found

Ensure the data has been parsed:
```bash
conda activate rlwm
python scripts/01_parse_raw_data.py
```

### Job times out

Increase the time limit or use `--limit` to test subset first:
```bash
#SBATCH --time=08:00:00
```
