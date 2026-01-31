# Cluster Setup for RLWM Trauma Analysis (Monash M3)

This directory contains scripts for running the MLE fitting pipeline on Monash M3 (MASSIVE).

## Quick Start

```bash
# 1. SSH to M3
ssh <username>@m3.massive.org.au

# 2. Clone the repo (use scratch for storage)
cd /scratch/<project>/$USER
git clone https://github.com/adammanoogian/rlwm_trauma_analysis.git
cd rlwm_trauma_analysis

# 3. Load miniforge3 and set up environment
module load miniforge3
chmod +x cluster/setup_env.sh
./cluster/setup_env.sh

# 4. Run MLE fitting
sbatch cluster/run_mle.slurm
```

## M3-Specific Configuration

| Setting | Value | Notes |
|---------|-------|-------|
| Module | `miniforge3` | NOT anaconda |
| Partition | `comp` | Default compute partition |
| Conda location | `/scratch/$PROJECT/$USER/conda` | Avoids home quota issues |
| Environment | `mamba` | Faster than conda |

**Important**: Do NOT run `conda init` on M3 - it breaks Strudel desktop.

## Scripts

| Script | Description |
|--------|-------------|
| `setup_env.sh` | One-time conda environment setup (configures scratch storage) |
| `run_mle.slurm` | SLURM job to fit all 3 models (M1, M2, M3) |
| `run_mle_single.slurm` | SLURM job for single model with `--limit` option |

## Usage Examples

### Full MLE Fitting (All Models)

```bash
sbatch cluster/run_mle.slurm
```

Runs Q-learning (M1), WM-RL (M2), and WM-RL with perseveration (M3) sequentially.
Expected time: ~2-4 hours total.

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

# Check output
cat cluster/logs/mle_single_<jobid>.out
```

The output will show:
- JIT compilation time (1st participant)
- Steady-state time per participant
- Extrapolated total time for all 47 participants

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

## Resource Settings

Default SLURM configuration:

```bash
#SBATCH --time=04:00:00      # 4 hours (all models) / 2 hours (single)
#SBATCH --mem=8G             # 8 GB RAM
#SBATCH --cpus-per-task=4    # 4 CPU cores
#SBATCH --partition=comp     # Standard compute partition
```

### Adjusting Resources

For longer jobs or more memory:
```bash
sbatch --time=08:00:00 --mem=16G cluster/run_mle.slurm
```

### M3 Partition Options

| Partition | Nodes | Cores | Memory/Node | Use Case |
|-----------|-------|-------|-------------|----------|
| comp | 79 | 1864 | Up to 1.5TB | Default, general purpose |
| m3i | 45 | 810 | 181 GB | Standard compute |
| m3j | 11 | 198 | 373 GB | High memory |
| m3m | 1 | 18 | 948 GB | Very high memory |
| short | 2 | 36 | 181 GB | Quick tests |

## Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# View job output in real-time
tail -f cluster/logs/mle_<jobid>.out

# Cancel a job
scancel <jobid>
```

## Troubleshooting

### "conda: command not found"

Make sure miniforge3 is loaded:
```bash
module load miniforge3
```

### "Environment 'rlwm' not found"

Run the setup script:
```bash
module load miniforge3
./cluster/setup_env.sh
```

### Data file not found

Ensure the data has been parsed:
```bash
conda activate rlwm
python scripts/01_parse_raw_data.py
```

### Job times out

Increase the time limit:
```bash
sbatch --time=08:00:00 cluster/run_mle.slurm
```

Or test with fewer participants first:
```bash
sbatch --export=MODEL=wmrl_m3,LIMIT=5 cluster/run_mle_single.slurm
```

## References

- [M3 Documentation](https://docs.erc.monash.edu/Compute/HPC/M3/)
- [Conda on M3](https://docs.erc.monash.edu/Compute/HPC/M3/Software/Conda/)
- [M3 Partitions](https://docs.erc.monash.edu/old-M3/M3/slurm/partitions/)
