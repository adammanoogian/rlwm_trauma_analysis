# cluster/rules/pipeline.smk -- Core RLWM analysis pipeline rules.
#
# Six-stage pipeline matching the numbered scripts/ layout:
#   Stage 1: Data preprocessing (01_data_preprocessing/)
#   Stage 2: Behavioral analyses (02_behav_analyses/)
#   Stage 3: Model prefitting (03_model_prefitting/)
#   Stage 4: Model fitting -- MLE + Bayesian (04_model_fitting/)
#   Stage 5: Post-fitting checks (05_post_fitting_checks/)
#   Stage 6: Fit analyses (06_fit_analyses/)
#
# Conventions:
#   - Output sentinel flags to cluster/results/ for Snakemake tracking
#   - Source cluster/_setup.sh at the start of every shell block
#   - Use print_job_header for standardized logging
#   - Mark aggregation/local rules as localrules
#   - All resource specs via lookup_resource() from job_manifest.yaml

localrules: pipeline_done


# --- Wildcard constraints ---
wildcard_constraints:
    model="|".join(config["choice_models"]),


# =============================================================================
# Stage 1: Data Preprocessing
# =============================================================================
# Serializes 01_parse_raw_data -> 02_create_collated_csv ->
# 03_create_task_trials_csv -> 04_create_summary_csv.
# CPU-only (pure pandas/numpy). Auto-skips when data/processed/ is populated
# (published-cohort path per the OSF deposit semantics).

rule data_processing:
    output:
        flag="cluster/results/data_processing.done",
    resources:
        slurm_partition=lambda wc, attempt: lookup_resource("data_processing", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("data_processing", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("data_processing", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("data_processing", "runtime"),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Stage 1: Data Preprocessing"
        export PYTHONUNBUFFERED=1

        mkdir -p data/processed data/interim

        # Published-cohort auto-skip: if processed CSVs exist, skip parsing.
        if [[ -f data/processed/task_trials_long.csv \
           && -f data/processed/task_trials_long_all.csv \
           && -f data/processed/summary_participant_metrics.csv ]]; then
            echo "data/processed/ populated -- skipping stage 01 (published-cohort path)"
            touch {output.flag}
            exit 0
        fi

        python scripts/01_data_preprocessing/01_parse_raw_data.py
        python scripts/01_data_preprocessing/02_create_collated_csv.py
        python scripts/01_data_preprocessing/03_create_task_trials_csv.py
        python scripts/01_data_preprocessing/04_create_summary_csv.py

        touch {output.flag}
        """


# =============================================================================
# Stage 2: Behavioral Analyses
# =============================================================================
# Serializes 01_summarize -> 02_visualize -> 03_trauma_groups ->
# 04_statistical_analyses.
# CPU-only (pandas/matplotlib/seaborn).

rule behav_analyses:
    input:
        "cluster/results/data_processing.done",
    output:
        flag="cluster/results/behav_analyses.done",
    resources:
        slurm_partition=lambda wc, attempt: lookup_resource("behav_analyses", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("behav_analyses", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("behav_analyses", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("behav_analyses", "runtime"),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Stage 2: Behavioral Analyses"
        export PYTHONUNBUFFERED=1

        mkdir -p reports/figures reports/tables

        python scripts/02_behav_analyses/01_summarize_behavioral_data.py
        python scripts/02_behav_analyses/02_visualize_task_performance.py
        python scripts/02_behav_analyses/03_analyze_trauma_groups.py
        python scripts/02_behav_analyses/04_run_statistical_analyses.py

        touch {output.flag}
        """


# =============================================================================
# Stage 3: Model Prefitting (NOT model-specific, runs once)
# =============================================================================
# Serializes 01_generate_synthetic_data -> 02_run_parameter_sweep ->
# 03_run_model_recovery -> 04_run_prior_predictive -> 05_run_bayesian_recovery.
# Prior predictive and Bayesian recovery are the Baribault & Collins 2023 gate.

rule prefitting:
    input:
        "cluster/results/data_processing.done",
    output:
        flag="cluster/results/prefitting.done",
    resources:
        slurm_partition=lambda wc, attempt: lookup_resource("prefitting", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("prefitting", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("prefitting", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("prefitting", "runtime"),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Stage 3: Model Prefitting"
        export PYTHONUNBUFFERED=1

        mkdir -p models/recovery models/parameter_exploration

        python scripts/03_model_prefitting/01_generate_synthetic_data.py
        python scripts/03_model_prefitting/02_run_parameter_sweep.py
        python scripts/03_model_prefitting/03_run_model_recovery.py
        python scripts/03_model_prefitting/04_run_prior_predictive.py
        python scripts/03_model_prefitting/05_run_bayesian_recovery.py

        touch {output.flag}
        """


# =============================================================================
# Stage 4a: MLE Fitting (per-model wildcard, GPU)
# =============================================================================
# One SLURM job per choice-only model. GPU-accelerated JAX fitting.

rule mle_fit:
    input:
        "cluster/results/data_processing.done",
    output:
        flag="cluster/results/mle/{model}.done",
    resources:
        slurm_partition=lambda wc, attempt: lookup_resource("mle_fit", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("mle_fit", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("mle_fit", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("mle_fit", "runtime"),
        gres=lambda wc, attempt: (
            f"gpu:{lookup_resource('mle_fit', 'gpus')}"
            if lookup_resource("mle_fit", "gpus") > 0
            else ""
        ),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Stage 4a: MLE Fit -- {wildcards.model}"
        setup_jax_cache gpu
        verify_gpu
        export PYTHONUNBUFFERED=1

        mkdir -p models/mle

        python scripts/04_model_fitting/a_mle/fit_mle.py \
            --model {wildcards.model} \
            --data {config[data_file]} \
            --use-gpu

        touch {output.flag}
        """


# =============================================================================
# Stage 4b: Bayesian Fitting (per-model wildcard, 1-GPU vectorized)
# =============================================================================
# One SLURM job per choice-only model. 1 GPU with chain_method="vectorized"
# (vmap, 4 chains on 1 device). QOS normal caps at 4 GPUs/user, so 1 GPU/job
# lets 4 models run concurrently (~18h total vs ~42h serialized at 4 GPU/job).
# sampling.py:_select_chain_method auto-selects vectorized when devices < chains.
# --allow-gate-failure writes diagnostics even when convergence gate fails.

rule bayesian_fit:
    input:
        "cluster/results/data_processing.done",
    output:
        flag="cluster/results/bayesian/{model}.done",
    resources:
        slurm_partition=lambda wc, attempt: lookup_resource("bayesian_fit", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("bayesian_fit", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("bayesian_fit", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("bayesian_fit", "runtime"),
        gres=lambda wc, attempt: (
            f"gpu:{lookup_resource('bayesian_fit', 'gpus')}"
            if lookup_resource("bayesian_fit", "gpus") > 0
            else ""
        ),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Stage 4b: Bayesian Fit -- {wildcards.model}"
        setup_jax_cache gpu
        verify_gpu
        export PYTHONUNBUFFERED=1
        export NUMPYRO_HOST_DEVICE_COUNT={config[mcmc_chains]}

        mkdir -p models/bayesian

        EXCLUDE_FLAG=""
        if [ -n "{config[exclude_participants]}" ]; then
            EXCLUDE_FLAG="--exclude-participants {config[exclude_participants]}"
        fi

        python scripts/04_model_fitting/b_bayesian/fit_bayesian.py \
            --model {wildcards.model} \
            --data {config[data_file]} \
            --chains {config[mcmc_chains]} \
            --warmup {config[mcmc_warmup]} \
            --samples {config[mcmc_samples]} \
            --output-subdir {config[bayesian_subdir]} \
            --allow-gate-failure \
            $EXCLUDE_FLAG

        touch {output.flag}
        """


# =============================================================================
# Stage 5: Post-Fitting Checks
# =============================================================================
# Baseline audit: R-hat/ESS/BFMI gate on all Bayesian posteriors.
# Runs after ALL Bayesian fits complete.

rule baseline_audit:
    input:
        expand(
            "cluster/results/bayesian/{model}.done",
            model=config["choice_models"],
        ),
    output:
        flag="cluster/results/baseline_audit.done",
    resources:
        slurm_partition=lambda wc, attempt: lookup_resource("baseline_audit", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("baseline_audit", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("baseline_audit", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("baseline_audit", "runtime"),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Stage 5: Baseline Audit"
        export PYTHONUNBUFFERED=1

        python scripts/05_post_fitting_checks/01_baseline_audit.py \
            --baseline-dir models/bayesian/{config[bayesian_subdir]}/ \
            --output-dir models/bayesian/{config[bayesian_subdir]}/

        touch {output.flag}
        """


# =============================================================================
# Stage 6: Fit Analyses
# =============================================================================

# --- 6.1: MLE Model Comparison ---
# Compare choice-only models by AIC/BIC. Runs after all MLE fits complete.
rule mle_compare:
    input:
        expand(
            "cluster/results/mle/{model}.done",
            model=config["choice_models"],
        ),
    output:
        flag="cluster/results/mle_compare.done",
    resources:
        slurm_partition=lambda wc, attempt: lookup_resource("mle_compare", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("mle_compare", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("mle_compare", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("mle_compare", "runtime"),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Stage 6.1: MLE Model Comparison"
        export PYTHONUNBUFFERED=1

        mkdir -p reports/tables/model_comparison

        python scripts/06_fit_analyses/01_compare_models.py

        touch {output.flag}
        """


# --- 6.2: Bayesian Model Selection (RFX-BMS) ---
# Random-effects BMS with PXP (Stephan 2009; Rigoux 2014). The script also
# computes PSIS-LOO internally but those results are NOT used — Pareto-k
# diagnostics showed 58-60% of observations > 0.7, making importance
# sampling unreliable. Only the RFX-BMS output (rfx_bms_pxp.csv) is
# consumed downstream. Memory reduced from 192 GB (LOO-era) to 32 GB.
rule model_selection:
    input:
        "cluster/results/baseline_audit.done",
    output:
        flag="cluster/results/model_selection.done",
    resources:
        slurm_partition=lambda wc, attempt: lookup_resource("model_selection", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("model_selection", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("model_selection", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("model_selection", "runtime"),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Stage 6.2: Bayesian Model Selection (RFX-BMS)"
        export PYTHONUNBUFFERED=1

        python scripts/06_fit_analyses/02_compute_loo_stacking.py \
            --baseline-dir models/bayesian/{config[bayesian_subdir]}/ \
            --output-dir models/bayesian/{config[bayesian_subdir]}/

        touch {output.flag}
        """


# --- 6.3: Manuscript Tables ---
# Final tables + figures for the manuscript. Depends on both LOO stacking
# (Bayesian track) and MLE comparison (frequentist track).
rule manuscript_tables:
    input:
        "cluster/results/model_selection.done",
        "cluster/results/mle_compare.done",
    output:
        flag="cluster/results/manuscript_tables.done",
    resources:
        slurm_partition=lambda wc, attempt: lookup_resource("manuscript_tables", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("manuscript_tables", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("manuscript_tables", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("manuscript_tables", "runtime"),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Stage 6.3: Manuscript Tables"
        export PYTHONUNBUFFERED=1

        mkdir -p reports/tables/model_comparison reports/figures/bayesian/21_bayesian

        python scripts/06_fit_analyses/08_manuscript_tables.py \
            --baseline-dir models/bayesian/{config[bayesian_subdir]}/ \
            --l2-dir models/bayesian/{config[l2_subdir]}/ \
            --figures-dir reports/figures/bayesian/ \
            --tables-dir reports/tables/model_comparison/ \
            --no-paper-edit

        touch {output.flag}
        """


# =============================================================================
# Pipeline completion sentinel
# =============================================================================

rule pipeline_done:
    """Aggregate sentinel -- marks full pipeline completion."""
    input:
        "cluster/results/manuscript_tables.done",
        "cluster/results/behav_analyses.done",
        "cluster/results/prefitting.done",
    output:
        "cluster/results/pipeline_done.flag",
    shell:
        "touch {output}"
