# M3 Cluster Setup & MLE Fitting Guide

Complete step-by-step guide for running the RLWM project on Monash M3 cluster.

---

## Step 1: Initial Login & Project Setup

### 1.1 SSH into M3
```bash
ssh YOUR_USERNAME@m3.massive.org.au
```

### 1.2 Configure Conda for Scratch Storage (CRITICAL - Do Once)

**⚠️ M3 Rule: Store conda envs in `/scratch/`, NOT home directory!**

```bash
# Create conda config directory
mkdir -p ~/.conda

# Create/edit conda config
cat > ~/.condarc << 'EOF'
pkgs_dirs:
  - /scratch/$PROJECT/$USER/conda/pkgs
envs_dirs:
  - /scratch/$PROJECT/$USER/conda/envs
EOF

# Create the directories (replace $PROJECT with your actual project code)
mkdir -p /scratch/$PROJECT/$USER/conda/pkgs
mkdir -p /scratch/$PROJECT/$USER/conda/envs
```

### 1.3 Verify You Haven't Run `conda init`

**⚠️ M3 Rule: NEVER run `conda init` - it breaks Strudel!**

```bash
# Check if conda init block exists in your .bashrc
grep -A5 ">>> conda initialize >>>" ~/.bashrc

# If you see output, remove it:
nano ~/.bashrc
# Delete everything between ">>> conda initialize >>>" and "<<< conda initialize <<<"
```

---

## Step 2: Clone/Upload the Project

### 2.1 Clone from GitHub (Recommended for Code)

GitHub is the preferred method for syncing code between local and cluster.

```bash
# Navigate to your project space
cd /projects/$PROJECT/$USER

# Clone the repo (use HTTPS or SSH depending on your setup)
git clone https://github.com/YOUR_USERNAME/rlwm_trauma_analysis.git
# OR with SSH key:
git clone git@github.com:YOUR_USERNAME/rlwm_trauma_analysis.git

cd rlwm_trauma_analysis
```

### 2.2 Set Up Git on M3 (First Time Only)

```bash
# Configure git identity
git config --global user.name "Your Name"
git config --global user.email "your.email@monash.edu"

# (Optional) Set up SSH key for GitHub
ssh-keygen -t ed25519 -C "your.email@monash.edu"
cat ~/.ssh/id_ed25519.pub
# Copy output and add to GitHub: Settings > SSH Keys > New SSH Key
```

### 2.3 Sync Changes with GitHub

```bash
# Pull latest changes from GitHub (do this before each session)
cd /projects/$PROJECT/$USER/rlwm_trauma_analysis
git pull origin main

# Push results/changes back to GitHub
git add output/mle/*.csv
git commit -m "Add MLE fitting results from M3"
git push origin main
```

### 2.4 Transfer Large Data Files via SCP

**⚠️ GitHub has a 100MB file limit.** Large data files should be transferred via SCP.

```bash
# FROM LOCAL → M3 (upload raw data)
scp output/task_trials_long.csv YOUR_USERNAME@m3.massive.org.au:/projects/$PROJECT/$USER/rlwm_trauma_analysis/output/

# Upload entire data directory
scp -r data/ YOUR_USERNAME@m3.massive.org.au:/projects/$PROJECT/$USER/rlwm_trauma_analysis/

# FROM M3 → LOCAL (download results)
scp YOUR_USERNAME@m3.massive.org.au:/projects/$PROJECT/$USER/rlwm_trauma_analysis/output/mle/*.csv ./results/

# Download recursively
scp -r YOUR_USERNAME@m3.massive.org.au:/projects/$PROJECT/$USER/rlwm_trauma_analysis/output/ ./local_output/
```

### 2.5 Recommended Workflow: Code vs Data

| Content | Sync Method | Reason |
|---------|-------------|--------|
| Python scripts, configs | **GitHub** | Version control, easy sync |
| `environment.yml`, SLURM scripts | **GitHub** | Part of codebase |
| Raw data (`*.csv` > 50MB) | **SCP** | Too large for GitHub |
| Results (`output/mle/`) | **GitHub** or **SCP** | Small CSVs can go to GitHub |
| Participant data (sensitive) | **SCP only** | May not belong in repo |

### 2.6 Add Large Files to .gitignore

Ensure large data files aren't accidentally committed:

```bash
# Check current .gitignore
cat .gitignore

# Add patterns for large files if needed
echo "data/raw/*.csv" >> .gitignore
echo "*.pkl" >> .gitignore
echo "*.npy" >> .gitignore
```

---

## Step 3: Create the Conda Environment

### 3.1 Load Miniforge (M3's Conda)
```bash
module load miniforge3
```

### 3.2 Create CPU Environment (For Parallel Fitting)
```bash
cd /projects/$PROJECT/$USER/rlwm_trauma_analysis

# Use mamba (faster than conda)
mamba env create -f environment.yml

# Verify it was created in scratch
conda env list
# Should show: /scratch/$PROJECT/$USER/conda/envs/rlwm
```

### 3.3 (Optional) Create GPU Environment
```bash
# Load CUDA first
module load cuda/12.2.0

# Create GPU environment
mamba env create -f environment_gpu.yml

# Verify
conda env list
# Should show: /scratch/$PROJECT/$USER/conda/envs/rlwm_gpu
```

---

## Step 4: Check Available Resources

### 4.1 See Cluster Status
```bash
# Overall cluster utilization
show_cluster

# Check specific partition availability
sinfo -p comp      # CPU partition (up to 96 CPUs/node)
sinfo -p gpu       # GPU partition (A100, T4, A40)
sinfo -p m3g       # V100 GPUs
```

### 4.2 Check Your Quota/Allocations
```bash
# See your project allocations
show_budget

# Check disk usage
lfs quota -h /projects/$PROJECT
lfs quota -h /scratch/$PROJECT
```

### 4.3 M3 Resource Summary

| Partition | Max CPUs/node | Max Memory | GPUs | Use For |
|-----------|---------------|------------|------|---------|
| `comp` | 96 | 1532 GB | None | CPU parallel fitting |
| `gpu` | varies | varies | A100, T4, A40 | GPU fitting |
| `m3g` | varies | varies | V100 (56 total) | GPU fitting |
| `short` | 18 | 181 GB | None | Quick tests (<30 min) |

---

## Step 5: Run a Quick Test (Interactive)

### 5.1 Start Interactive Session
```bash
# Request 4 CPUs for 30 minutes (for testing)
srun --partition=comp --cpus-per-task=4 --mem=8G --time=00:30:00 --pty bash
```

### 5.2 Activate Environment & Test
```bash
# Load module and activate env
module load miniforge3
conda activate rlwm

# Navigate to project
cd /projects/$PROJECT/$USER/rlwm_trauma_analysis

# Quick test: fit 2 participants with 4 cores
python scripts/fitting/fit_mle.py \
    --model qlearning \
    --data output/task_trials_long.csv \
    --limit 2 \
    --n-jobs 4 \
    --n-starts 5

# Exit interactive session when done
exit
```

---

## Step 6: Submit Batch Jobs

### 6.1 Create Logs Directory
```bash
mkdir -p cluster/logs
```

### 6.2 Submit GPU Job (Recommended for Reliability)

**⚠️ GPU fitting is now the recommended approach** because it avoids LLVM/CPU issues:
- No LLVM thread spawning memory exhaustion
- No CPU feature mismatch between heterogeneous cluster nodes
- Comparable speed to parallel CPU (2-5 minutes per model)

```bash
cd /projects/$PROJECT/$USER/rlwm_trauma_analysis

# Submit GPU-accelerated fitting (all 3 models)
sbatch cluster/run_mle_gpu.slurm

# For specific model:
sbatch --export=MODEL=wmrl_m3 cluster/run_mle_gpu.slurm

# Check job status
squeue -u $USER

# Watch the output in real-time
tail -f cluster/logs/mle_gpu_*.out
```

### 6.3 Submit CPU Parallel Job (Alternative)

Use CPU fitting if GPU queue is unavailable. Note: M3 has heterogeneous nodes with
different CPU features, which can cause "CPU feature mismatch" errors if the JAX
cache is shared between nodes. The scripts now use node-specific caching to avoid this.

```bash
# Submit parallel fitting (16 cores, all 3 models)
sbatch cluster/run_mle.slurm

# Check job status
squeue -u $USER

# Watch the output in real-time
tail -f cluster/logs/mle_*.out
```

### 6.4 Submit Single Model (For Testing)
```bash
# Quick timing test (3 participants only)
sbatch --export=MODEL=wmrl_m3,LIMIT=3 cluster/run_mle_single.slurm

# Single model, full dataset
sbatch --export=MODEL=wmrl_m3 cluster/run_mle_parallel.slurm
```

---

## Step 7: Monitor & Retrieve Results

### 7.1 Check Job Status
```bash
# Your jobs
squeue -u $USER

# Detailed job info
scontrol show job JOBID

# Cancel a job if needed
scancel JOBID
```

### 7.2 View Output Logs
```bash
# List log files
ls -la cluster/logs/

# View latest output
tail -100 cluster/logs/mle_parallel_*.out

# Real-time monitoring
tail -f cluster/logs/mle_parallel_*.out
```

### 7.3 Check Results
```bash
# Results are saved to output/mle/
ls -la output/mle/

# View summary
cat output/mle/qlearning_group_summary.csv
cat output/mle/wmrl_group_summary.csv
cat output/mle/wmrl_m3_group_summary.csv
```

### 7.4 Download Results to Local Machine
```bash
# From your LOCAL machine:
scp -r YOUR_USERNAME@m3.massive.org.au:/projects/$PROJECT/$USER/rlwm_trauma_analysis/output/mle/ ./results/
```

---

## Quick Reference: Common Commands

```bash
# Load environment (do this every session)
module load miniforge3
conda activate rlwm

# Check cluster
show_cluster
sinfo -p comp
sinfo -p gpu

# Submit jobs
sbatch cluster/run_mle_parallel.slurm          # CPU parallel (recommended)
sbatch cluster/run_mle_gpu.slurm               # GPU (if env set up)

# Monitor
squeue -u $USER
tail -f cluster/logs/mle_parallel_*.out

# Cancel
scancel JOBID
scancel -u $USER  # Cancel ALL your jobs
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `conda: command not found` | Run `module load miniforge3` |
| `CondaEnvironmentNotFound` | Check `conda env list`, ensure env is in scratch |
| Job stuck in `PENDING` | Run `squeue -u $USER` to see reason; try smaller resource request |
| `ModuleNotFoundError: joblib` | Env may need updating: `mamba env update -f environment.yml` |
| GPU not detected | Ensure `module load cuda/12.2.0` and using `rlwm_gpu` env |
| `SIGILL` or CPU feature mismatch | Clear stale cache: `rm -rf /scratch/$PROJECT/$USER/.jax_cache` and retry |
| `Cannot allocate memory` (LLVM) | Use GPU fitting, or use `run_mle_single.slurm` with its two-phase approach |

---

## Available SLURM Scripts

| Script | Description | Cores | Time |
|--------|-------------|-------|------|
| `run_mle.slurm` | Sequential fitting (all models) | 4 | 4h |
| `run_mle_parallel.slurm` | Parallel fitting (all models) | 16 | 30min |
| `run_mle_gpu.slurm` | GPU-accelerated fitting | 4 + GPU | 15min |
| `run_mle_single.slurm` | Single model (timing tests) | 4 | 2h |

---

## Expected Runtimes (47 Participants, All 3 Models)

| Configuration | Time | SLURM Script |
|---------------|------|--------------|
| Sequential (1 CPU) | ~90 min | `run_mle.slurm` |
| Parallel (16 CPUs) | ~10 min | `run_mle_parallel.slurm` |
| GPU (1x A100) | ~5-10 min | `run_mle_gpu.slurm` |

---

## References

- [M3 Documentation](https://docs.erc.monash.edu/Compute/HPC/M3/)
- [Conda on M3](https://docs.erc.monash.edu/Compute/HPC/M3/Software/Conda/)
- [M3 Partitions](https://docs.erc.monash.edu/old-M3/M3/slurm/partitions/)
