# M3 Cluster Setup & Pipeline Guide

Complete step-by-step guide for running the RLWM project on the Monash M3 cluster.

---

## Canonical SLURM Layout (post-29-05)

Six stage-numbered entry SLURMs + one master orchestrator cover the full pipeline.
Every SLURM invokes python scripts via the post-29-04b canonical stage-numbered
paths (`scripts/0{1..6}_*/*.py`).

```
cluster/
├── submit_all.sh                     # MASTER: chains 01..06 via afterok (canonical entry)
├── 21_submit_pipeline.sh             # Shim delegating to submit_all.sh (back-compat)
│
├── 01_data_processing.slurm          # Stage 01: parse + collate + trials CSV + summary CSV
├── 02_behav_analyses.slurm           # Stage 02: summary/visualise/trauma groups/statistics
├── 03_prefitting_cpu.slurm           # Stage 03 CPU: STEP=synthetic|parameter_sweep|model_recovery|prior_predictive|bayesian_recovery
├── 03_prefitting_gpu.slurm           # Stage 03 GPU: same STEP dispatch, GPU-accelerated
├── 04a_mle_cpu.slurm                 # Stage 04a: MLE fitting (CPU parallel, NJOBS knob)
├── 04a_mle_gpu.slurm                 # Stage 04a: MLE fitting (GPU)
├── 04b_bayesian_cpu.slurm            # Stage 04b: hierarchical Bayesian (CPU; all 6 choice-only + subscale)
├── 04b_bayesian_gpu.slurm            # Stage 04b: hierarchical Bayesian (GPU; M4 LBA target)
├── 04c_level2.slurm                  # Stage 04c: winner refit WITH L2 scales
├── 05_post_checks.slurm              # Stage 05: STEP=baseline_audit|scale_audit|posterior_ppc
├── 06_fit_analyses.slurm             # Stage 06: STEP=compare_models|loo_stacking|model_averaging|manuscript_tables|...
│
├── 13_bayesian_multigpu.slurm        # [RETAINED-SPECIALIZED] multi-GPU chains (pmap across 4 GPUs)
├── 13_bayesian_permutation.slurm     # [RETAINED-SPECIALIZED] permutation null test (array 0-49)
├── 23.1_mgpu_smoke.slurm             # [RETAINED-SPECIALIZED] Phase 23.1 multi-GPU smoke
├── 21_6_dispatch_l2.slurm            # [RETAINED] L2 winner dispatcher wrapper (sbatch --wait blocker)
├── 21_dispatch_l2_winners.sh         # [RETAINED] L2 winner fan-out shell driver
├── 01_diagnostic_gpu.slurm           # [RETAINED] GPU/JAX smoke diagnostic
├── 99_push_results.slurm             # [RETAINED] auto-push results to git
│
├── 00_setup_env.sh                   # CPU conda environment setup
├── 00_setup_env_gpu.sh               # GPU conda environment setup (JAX CUDA)
├── autopush.sh                       # Auto-push helper (sourced by SLURMs)
├── legacy/                           # Archived shipped-milestone SLURMs (pscan/fullybatched benchmarks, wave-based orchestrator)
├── logs/                             # SLURM output (gitignored)
└── README.md                         # this file
```

### What changed in 29-05

- **6 stage entry SLURMs** (`01..06_*.slurm`) replace the per-step templates.
- **`cluster/submit_all.sh`** is the new canonical master orchestrator (afterok chain).
- **`cluster/21_submit_pipeline.sh`** still works — it's now a one-line shim that delegates to `submit_all.sh`.
- Every python invocation in every cluster SLURM resolves on disk (post-29-04b paths).

### Deleted in 29-05 (consolidated)

| Deleted | Replacement |
|---|---|
| `09_ppc_gpu.slurm` | `05_post_checks.slurm` STEP=posterior_ppc (USE_GPU=1) |
| `11_recovery_gpu.slurm` | `03_prefitting_gpu.slurm` STEP=model_recovery |
| `12_mle.slurm` | **renamed** `04a_mle_cpu.slurm` |
| `12_mle_gpu.slurm` | **renamed** `04a_mle_gpu.slurm` |
| `12_mle_single.slurm` | `04a_mle_cpu.slurm` with `--export=NJOBS=1` |
| `12_submit_all.sh`, `12_submit_all_gpu.sh` | `submit_all.sh` |
| `13_bayesian_choice_only.slurm` | **renamed** `04b_bayesian_cpu.slurm` |
| `13_bayesian_gpu.slurm` | **renamed** `04b_bayesian_gpu.slurm` |
| `13_bayesian_m4_gpu.slurm` | `04b_bayesian_gpu.slurm` with `MODEL=wmrl_m4,TIME=48:00:00` |
| `13_bayesian_m6b_subscale.slurm` | `04b_bayesian_cpu.slurm` with `MODEL=wmrl_m6b,SUBSCALE=1,TIME=12:00:00` |
| `14_analysis.slurm` | `06_fit_analyses.slurm` STEP=compare_models \| analyze_mle_by_trauma \| regress_parameters_on_scales |
| `21_1_prior_predictive.slurm` | `03_prefitting_cpu.slurm` STEP=prior_predictive |
| `21_2_recovery.slurm` | `03_prefitting_cpu.slurm` STEP=bayesian_recovery (array support via SLURM_ARRAY_TASK_ID) |
| `21_2_recovery_aggregate.slurm` | `03_prefitting_cpu.slurm` STEP=bayesian_recovery RECOVERY_MODE=aggregate |
| `21_3_fit_baseline.slurm` | `04b_bayesian_cpu.slurm` (baseline output subdir is handled by fit_bayesian internally) |
| `21_4_baseline_audit.slurm` | `05_post_checks.slurm` STEP=baseline_audit |
| `21_5_loo_stacking_bms.slurm` | `06_fit_analyses.slurm` STEP=loo_stacking |
| `21_6_fit_with_l2.slurm` | **renamed** `04c_level2.slurm` |
| `21_7_scale_audit.slurm` | `05_post_checks.slurm` STEP=scale_audit |
| `21_8_model_averaging.slurm` | `06_fit_analyses.slurm` STEP=model_averaging |
| `21_9_manuscript_tables.slurm` | `06_fit_analyses.slurm` STEP=manuscript_tables |

### Retained-specialized SLURMs (kept verbatim because they have a dedicated workflow)

| SLURM | Why kept |
|---|---|
| `13_bayesian_multigpu.slurm` | Multi-GPU chain pmap (not just model selection — different chain_method) |
| `13_bayesian_permutation.slurm` | Permutation-null SLURM array (50 shuffles); orthogonal dispatch surface |
| `23.1_mgpu_smoke.slurm` | Phase 23.1 multi-GPU validation (per-model 10-minute smoke) |
| `21_6_dispatch_l2.slurm` + `21_dispatch_l2_winners.sh` | L2 winner fan-out (sbatch --wait blocker, referenced from winners.txt) |
| `01_diagnostic_gpu.slurm` | GPU/JAX readiness check (run-first template for new users) |
| `99_push_results.slurm` | Auto-push results to git (dependency-chained from fitting SLURMs) |

### Archived to `cluster/legacy/` (2026-04-23, post-Phase-29)

Seven shipped-milestone SLURMs moved to `cluster/legacy/` — all referenced milestones are closed (Phases 19 + 20 shipped in v4.0) and none are invoked by `submit_all.sh`. Resurrect via `git mv cluster/legacy/<file> cluster/` if a rerun is needed.

| Archived | Original purpose |
|---|---|
| `13_full_pipeline.slurm` | Single-job all-in-one quick-smoke (superseded by `submit_all.sh`) |
| `13_bayesian_pscan.slurm`, `13_bayesian_pscan_smoke.slurm` | Parallel-scan A/B benchmark (Phase 19 shipped) |
| `13_bayesian_fullybatched_smoke.slurm` | Fully-batched vmap smoke (Phase 20 shipped) |
| `19_benchmark_pscan_{cpu,gpu}.slurm` | pscan micro-benchmark wrappers (Phase 19 shipped) |
| `submit_full_pipeline.sh` | Wave-based pipeline orchestrator (alt to `submit_all.sh` — redundant entry point) |

---

## Canonical full-pipeline invocation

```bash
# Full chain (01 -> 02 -> 03 -> 04 -> 05 -> 06) via --afterok
bash cluster/submit_all.sh

# Dry-run: verify every SLURM passes bash -n and every python path resolves
bash cluster/submit_all.sh --dry-run

# Restart mid-pipeline
bash cluster/submit_all.sh --from-stage 5

# Subset of models
bash cluster/submit_all.sh --models "wmrl_m3 wmrl_m5"

# Back-compat shim (preserves v4.0 user-memory invocation)
bash cluster/21_submit_pipeline.sh           # -> delegates to submit_all.sh
```

---

## Step 1: Initial Login & Project Setup

### 1.1 SSH into M3
```bash
ssh YOUR_USERNAME@m3.massive.org.au
```

### 1.2 Configure Conda for Scratch Storage (do this once)

**M3 rule: Store conda envs in `/scratch/`, NOT home directory.**

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

### 1.3 Verify you haven't run `conda init`

**M3 rule: never run `conda init` — it breaks Strudel.**

```bash
grep -A5 ">>> conda initialize >>>" ~/.bashrc
# If output exists, remove the block from ~/.bashrc
```

---

## Step 2: Clone the Project

```bash
cd /projects/$PROJECT/$USER
git clone https://github.com/YOUR_USERNAME/rlwm_trauma_analysis.git
cd rlwm_trauma_analysis
pip install -e .       # required: src/rlwm/ is the authoritative import package
```

---

## Step 3: Create conda environments

```bash
module load miniforge3
mamba env create -f environment.yml          # CPU
mamba env create -f environment_gpu.yml      # GPU (optional, required for M4 LBA)
```

---

## Step 4: Submit the pipeline

### Option A — full chain

```bash
mkdir -p cluster/logs
bash cluster/submit_all.sh
squeue -u $USER
```

### Option B — individual stages

```bash
# Stage 01 (data processing)
sbatch cluster/01_data_processing.slurm

# Stage 02 (behavioural analyses)
sbatch cluster/02_behav_analyses.slurm

# Stage 03 (prefitting: fan-out per model)
for m in qlearning wmrl wmrl_m3 wmrl_m5 wmrl_m6a wmrl_m6b; do
  sbatch --export=ALL,STEP=prior_predictive,MODEL=$m cluster/03_prefitting_cpu.slurm
done

# Stage 04a MLE (GPU)
sbatch cluster/04a_mle_gpu.slurm                                    # all models
sbatch --export=MODEL=wmrl_m3 cluster/04a_mle_gpu.slurm             # single model

# Stage 04b Bayesian (CPU for choice-only)
sbatch --export=ALL,MODEL=wmrl_m3 cluster/04b_bayesian_cpu.slurm
sbatch --time=36:00:00 --export=ALL,MODEL=wmrl_m6b cluster/04b_bayesian_cpu.slurm
sbatch --time=12:00:00 --mem=48G \
  --export=ALL,MODEL=wmrl_m6b,SUBSCALE=1 cluster/04b_bayesian_cpu.slurm

# Stage 04b Bayesian (GPU — M4 LBA only)
sbatch --time=48:00:00 --gres=gpu:a100:1 --mem=96G \
  --export=ALL,MODEL=wmrl_m4 cluster/04b_bayesian_gpu.slurm

# Stage 04c Level-2 refit
sbatch --export=MODEL=wmrl_m3 cluster/04c_level2.slurm

# Stage 05 post-fitting checks
sbatch --export=ALL,STEP=baseline_audit cluster/05_post_checks.slurm
sbatch --export=ALL,STEP=scale_audit cluster/05_post_checks.slurm

# Stage 06 fit analyses
sbatch --export=ALL,STEP=compare_models cluster/06_fit_analyses.slurm
sbatch --export=ALL,STEP=loo_stacking cluster/06_fit_analyses.slurm
sbatch --export=ALL,STEP=manuscript_tables cluster/06_fit_analyses.slurm
```

---

## Step 5: Monitor & retrieve results

```bash
squeue -u $USER
sacct -j <JOBID> --format=JobID,Elapsed,MaxRSS,MaxVMSize,TotalCPU,State
tail -f cluster/logs/*_%j.out

scancel JOBID
scancel -u $USER
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `conda: command not found` | Run `module load miniforge3` |
| `CondaEnvironmentNotFound` | Check `conda env list`, ensure env is in scratch |
| Job stuck in `PENDING` | Run `squeue -u $USER` to see reason; try smaller resource request |
| GPU not detected | Ensure using `rlwm_gpu` env on a GPU node; do NOT load cuda module |
| `SIGILL` / CPU feature mismatch | Clear stale cache: `rm -rf /scratch/$PROJECT/$USER/.jax_cache` |
| `Cannot allocate memory` (LLVM) | Use `--export=NJOBS=1` on `04a_mle_cpu.slurm` or switch to `04a_mle_gpu.slurm` |

---

## References

- [M3 Documentation](https://docs.erc.monash.edu/Compute/HPC/M3/)
- [Conda on M3](https://docs.erc.monash.edu/Compute/HPC/M3/Software/Conda/)
- [M3 Partitions](https://docs.erc.monash.edu/old-M3/M3/slurm/partitions/)
- `.planning/phases/29-pipeline-canonical-reorg/29-05-SUMMARY.md` for the consolidation audit trail
