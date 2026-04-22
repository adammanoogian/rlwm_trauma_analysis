# M3 Cluster Setup & MLE Fitting Guide

Complete step-by-step guide for running the RLWM project on Monash M3 cluster.

## Cluster Script Index

Scripts are numbered to match the pipeline scripts they invoke:

```
cluster/
├── 00_setup_env.sh          # CPU conda environment setup
├── 00_setup_env_gpu.sh      # GPU conda environment setup (JAX CUDA)
├── 01_diagnostic_gpu.slurm  # GPU/JAX validation (run first)
├── 09_ppc_gpu.slurm         # Posterior predictive checks  → scripts/05_post_fitting_checks/03_run_posterior_ppc.py
├── 11_recovery_gpu.slurm    # Parameter recovery           → scripts/03_model_prefitting/03_run_model_recovery.py
├── 12_mle.slurm             # MLE fitting (CPU parallel)   → scripts/04_model_fitting/a_mle/fit_mle.py
├── 12_mle_gpu.slurm         # MLE fitting (GPU)            → scripts/04_model_fitting/a_mle/fit_mle.py
├── 12_mle_single.slurm      # MLE fitting (single model)   → scripts/04_model_fitting/a_mle/fit_mle.py
├── 12_submit_all.sh         # Submit all CPU MLE jobs
├── 12_submit_all_gpu.sh     # Submit all GPU MLE jobs
├── 13_full_pipeline.slurm   # FULL PIPELINE: steps 05-16 (GPU)
├── logs/                    # SLURM output (gitignored)
└── README.md                # This file
```

---

## Step 1: Initial Login & Project Setup

### 1.1 SSH into M3
```bash
ssh YOUR_USERNAME@m3.massive.org.au
```

### 1.2 Configure Conda for Scratch Storage (CRITICAL - Do Once)

**M3 Rule: Store conda envs in `/scratch/`, NOT home directory!**

```bash
mkdir -p ~/.conda

cat > ~/.condarc << 'EOF'
pkgs_dirs:
  - /scratch/$PROJECT/$USER/conda/pkgs
envs_dirs:
  - /scratch/$PROJECT/$USER/conda/envs
EOF

mkdir -p /scratch/$PROJECT/$USER/conda/pkgs
mkdir -p /scratch/$PROJECT/$USER/conda/envs
```

### 1.3 Verify You Haven't Run `conda init`

**M3 Rule: NEVER run `conda init` - it breaks Strudel!**

```bash
grep -A5 ">>> conda initialize >>>" ~/.bashrc
# If output exists, remove the block from ~/.bashrc
```

---

## Step 2: Clone/Upload the Project

### 2.1 Clone from GitHub (Recommended for Code)

```bash
cd /projects/$PROJECT/$USER
git clone https://github.com/YOUR_USERNAME/rlwm_trauma_analysis.git
cd rlwm_trauma_analysis
```

### 2.2 Set Up Git on M3 (First Time Only)

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@monash.edu"

# (Optional) SSH key for GitHub
ssh-keygen -t ed25519 -C "your.email@monash.edu"
cat ~/.ssh/id_ed25519.pub
# Add to GitHub: Settings > SSH Keys > New SSH Key
```

### 2.3 Sync Changes with GitHub

```bash
cd /projects/$PROJECT/$USER/rlwm_trauma_analysis
git pull origin main

# Push results back
git add output/mle/*.csv
git commit -m "Add MLE fitting results from M3"
git push origin main
```

### 2.4 Transfer Large Data Files via SCP

```bash
# LOCAL → M3 (upload raw data)
scp output/task_trials_long.csv YOUR_USERNAME@m3.massive.org.au:/projects/$PROJECT/$USER/rlwm_trauma_analysis/output/

# M3 → LOCAL (download results)
scp -r YOUR_USERNAME@m3.massive.org.au:/projects/$PROJECT/$USER/rlwm_trauma_analysis/output/mle/ ./results/
```

| Content | Sync Method | Reason |
|---------|-------------|--------|
| Python scripts, configs | **GitHub** | Version control |
| SLURM scripts | **GitHub** | Part of codebase |
| Raw data (`*.csv` > 50MB) | **SCP** | Too large for GitHub |
| Results (`output/mle/`) | **GitHub** or **SCP** | Small CSVs can go to GitHub |
| Participant data (sensitive) | **SCP only** | May not belong in repo |

---

## Step 3: Create the Conda Environment

### 3.1 Load Miniforge (M3's Conda)
```bash
module load miniforge3
```

### 3.2 Create CPU Environment
```bash
cd /projects/$PROJECT/$USER/rlwm_trauma_analysis
mamba env create -f environment.yml

conda env list
# Should show: /scratch/$PROJECT/$USER/conda/envs/rlwm
```

### 3.3 (Optional) Create GPU Environment
```bash
# Note: Do NOT load CUDA module — JAX pip packages bundle CUDA libraries
mamba env create -f environment_gpu.yml

conda env list
# Should show: /scratch/$PROJECT/$USER/conda/envs/rlwm_gpu
```

---

## Step 4: Check Available Resources

```bash
show_cluster            # Overall cluster utilization
sinfo -p comp           # CPU partition (up to 96 CPUs/node)
sinfo -p gpu            # GPU partition (A100, T4, A40)
show_budget             # Your project allocations
```

| Partition | Max CPUs/node | Max Memory | GPUs | Use For |
|-----------|---------------|------------|------|---------|
| `comp` | 96 | 1532 GB | None | CPU parallel fitting |
| `gpu` | varies | varies | A100, T4, A40 | GPU fitting |
| `m3g` | varies | varies | V100 (56 total) | GPU fitting |
| `short` | 18 | 181 GB | None | Quick tests (<30 min) |

---

## Step 5: Run a Quick Test (Interactive)

```bash
# Request 4 CPUs for 30 minutes
srun --partition=comp --cpus-per-task=4 --mem=8G --time=00:30:00 --pty bash

# Activate and test
module load miniforge3
conda activate rlwm
cd /projects/$PROJECT/$USER/rlwm_trauma_analysis

python scripts/04_model_fitting/a_mle/fit_mle.py \
    --model qlearning \
    --data output/task_trials_long.csv \
    --limit 2 \
    --n-jobs 4 \
    --n-starts 5

exit
```

---

## Step 6: Submit Batch Jobs

### 6.1 Create Logs Directory
```bash
mkdir -p cluster/logs
```

### 6.2 Full Pipeline (Steps 05-16, Recommended)

Run behavioral analysis, MLE fitting, and results analysis in one job:

```bash
cd /projects/$PROJECT/$USER/rlwm_trauma_analysis
sbatch cluster/13_full_pipeline.slurm                          # Standard
sbatch --export=VALIDATE=true cluster/13_full_pipeline.slurm   # + validation (09, 11)
sbatch --export=SKIP_BEHAVIORAL=true cluster/13_full_pipeline.slurm  # Skip 05-08
squeue -u $USER
```

### 6.3 Submit All 3 Models as Parallel GPU Jobs

```bash
bash cluster/12_submit_all_gpu.sh
squeue -u $USER
```

### 6.3 Submit Single GPU Job

GPU fitting avoids LLVM/CPU issues and is comparable speed to parallel CPU.

```bash
sbatch cluster/12_mle_gpu.slurm                                    # All 3 models
sbatch --export=MODEL=wmrl_m3 cluster/12_mle_gpu.slurm             # Single model
tail -f cluster/logs/mle_gpu_*.out
```

### 6.4 Submit CPU Parallel Job (Alternative)

```bash
sbatch cluster/12_mle.slurm
tail -f cluster/logs/mle_*.out
```

### 6.5 Submit Single Model (For Testing)
```bash
sbatch --export=MODEL=wmrl_m3,LIMIT=3 cluster/12_mle_single.slurm  # Quick timing test
sbatch --export=MODEL=wmrl_m3 cluster/12_mle_single.slurm          # Full dataset
```

### 6.6 Parameter Recovery
```bash
sbatch cluster/11_recovery_gpu.slurm
sbatch --export=MODEL=qlearning cluster/11_recovery_gpu.slurm
```

### 6.7 Posterior Predictive Checks
```bash
sbatch cluster/09_ppc_gpu.slurm
sbatch --export=MODEL=wmrl_m3 cluster/09_ppc_gpu.slurm
```

---

## Step 7: Monitor & Retrieve Results

### 7.1 Check Job Status
```bash
squeue -u $USER
scontrol show job JOBID
scancel JOBID
```

### 7.2 View Output Logs
```bash
ls -la cluster/logs/
tail -f cluster/logs/mle_gpu_*.out
```

### 7.3 Real-Time Monitoring

```bash
watch -n 5 'squeue -u $USER'
squeue -u $USER -o "%.8i %.9P %.20j %.8u %.2t %.10M %.6D %R"
sacct -j <JOBID> --format=JobID,Elapsed,MaxRSS,MaxVMSize,TotalCPU,State
```

### 7.4 Check Results
```bash
ls -la output/mle/
cat output/mle/qlearning_group_summary.csv
```

### 7.5 Download Results
```bash
# From LOCAL machine:
scp -r YOUR_USERNAME@m3.massive.org.au:/projects/$PROJECT/$USER/rlwm_trauma_analysis/output/mle/ ./results/
```

---

## Quick Reference

```bash
# Load environment (every session)
module load miniforge3
conda activate rlwm

# Submit jobs
sbatch cluster/13_full_pipeline.slurm     # FULL PIPELINE: steps 05-16 (recommended)
bash cluster/12_submit_all_gpu.sh         # All 3 models (GPU, fitting only)
sbatch cluster/12_mle.slurm               # All 3 models (CPU parallel)
sbatch cluster/12_mle_gpu.slurm           # All 3 models (GPU, single job)
sbatch cluster/11_recovery_gpu.slurm      # Parameter recovery
sbatch cluster/09_ppc_gpu.slurm           # Posterior predictive checks

# Monitor
squeue -u $USER
tail -f cluster/logs/mle_gpu_*.out

# Cancel
scancel JOBID
scancel -u $USER
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `conda: command not found` | Run `module load miniforge3` |
| `CondaEnvironmentNotFound` | Check `conda env list`, ensure env is in scratch |
| Job stuck in `PENDING` | Run `squeue -u $USER` to see reason; try smaller resource request |
| GPU not detected | Ensure using `rlwm_gpu` env on a GPU node. Do NOT load cuda module |
| `SIGILL` or CPU feature mismatch | Clear stale cache: `rm -rf /scratch/$PROJECT/$USER/.jax_cache` and retry |
| `Cannot allocate memory` (LLVM) | Use GPU fitting, or use `12_mle_single.slurm` with two-phase approach |

---

## Available Scripts

| Script | Description | Resources | Time |
|--------|-------------|-----------|------|
| `13_full_pipeline.slurm` | **Full pipeline**: steps 05-16 | 4 CPU + GPU | ~25min |
| `12_submit_all_gpu.sh` | 3 independent GPU MLE jobs | 4 CPU + GPU each | ~5min/model |
| `12_mle.slurm` | CPU parallel fitting (all models) | 16 CPU | 30min |
| `12_mle_gpu.slurm` | GPU fitting (single job, all models) | 4 CPU + GPU | 15min |
| `12_mle_single.slurm` | Single model (timing tests) | 16 CPU | 2h |
| `11_recovery_gpu.slurm` | Parameter recovery validation | 4 CPU + GPU | 30min |
| `09_ppc_gpu.slurm` | Posterior predictive checks | 4 CPU + GPU | 8h |
| `01_diagnostic_gpu.slurm` | GPU/JAX validation | 4 CPU + GPU | 30min |

## Expected Runtimes (47 Participants, All 3 Models)

| Configuration | Time | Script |
|---------------|------|--------|
| Sequential (1 CPU) | ~90 min | `12_mle.slurm` (NJOBS=1) |
| Parallel (16 CPUs) | ~10 min | `12_mle.slurm` |
| GPU (1x A100) | ~5-10 min | `12_mle_gpu.slurm` |

---

## References

- [M3 Documentation](https://docs.erc.monash.edu/Compute/HPC/M3/)
- [Conda on M3](https://docs.erc.monash.edu/Compute/HPC/M3/Software/Conda/)
- [M3 Partitions](https://docs.erc.monash.edu/old-M3/M3/slurm/partitions/)
