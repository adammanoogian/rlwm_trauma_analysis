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
# Stage 4a-M4: MLE Fitting for M4 LBA (separate track, GPU, float64)
# =============================================================================
# M4 uses a joint choice+RT LBA likelihood. Its AIC is NOT comparable to
# choice-only models. Runs independently from the choice-only wildcard.

rule mle_fit_m4:
    input:
        "cluster/results/data_processing.done",
    output:
        flag="cluster/results/mle/wmrl_m4.done",
    resources:
        slurm_partition=lambda wc, attempt: lookup_resource("mle_fit_m4", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("mle_fit_m4", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("mle_fit_m4", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("mle_fit_m4", "runtime"),
        gres=lambda wc, attempt: (
            f"gpu:{lookup_resource('mle_fit_m4', 'gpus')}"
            if lookup_resource("mle_fit_m4", "gpus") > 0
            else ""
        ),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Stage 4a-M4: MLE Fit -- wmrl_m4 (LBA separate track)"
        export PYTHONUNBUFFERED=1

        mkdir -p models/mle

        python scripts/04_model_fitting/a_mle/fit_mle.py \
            --model wmrl_m4 \
            --data {config[data_file]} \
            --n-jobs {resources.cpus_per_task}

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
# Stage 5b: Posterior Predictive Checks (MLE track, per-model)
# =============================================================================
# Generates synthetic data from fitted params and compares to real data.
# Runs after MLE fits complete. CPU-parallel (no GPU).

rule ppc:
    input:
        expand(
            "cluster/results/mle/{model}.done",
            model=config["choice_models"],
        ),
    output:
        flag="cluster/results/ppc.done",
    resources:
        slurm_partition=lambda wc, attempt: lookup_resource("ppc", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("ppc", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("ppc", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("ppc", "runtime"),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Stage 5b: Posterior Predictive Checks"
        export PYTHONUNBUFFERED=1

        mkdir -p models/ppc reports/figures/ppc

        python scripts/05_post_fitting_checks/03_run_posterior_ppc.py \
            --model all --skip-model-recovery --n-jobs {resources.cpus_per_task}

        touch {output.flag}
        """


# =============================================================================
# Stage 5c: Level-2 Winner Refit (Bayesian track)
# =============================================================================
# Refits the BMS winner(s) with Level-2 trauma scale predictors.
# Reads winners.txt from model_selection step.

rule level2_refit:
    input:
        "cluster/results/model_selection.done",
    output:
        flag="cluster/results/level2_refit.done",
    resources:
        slurm_partition=lambda wc, attempt: lookup_resource("level2_refit", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("level2_refit", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("level2_refit", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("level2_refit", "runtime"),
        gres=lambda wc, attempt: (
            f"gpu:{lookup_resource('level2_refit', 'gpus')}"
            if lookup_resource("level2_refit", "gpus") > 0
            else ""
        ),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Stage 5c: Level-2 Winner Refit"
        setup_jax_cache gpu
        verify_gpu
        export PYTHONUNBUFFERED=1

        WINNERS=$(cat models/bayesian/{config[bayesian_subdir]}/winners.txt)
        for WINNER_DISPLAY in $(echo "$WINNERS" | tr ',' ' '); do
            # Map display name (M6b) to internal key (wmrl_m6b)
            WINNER_KEY=$(python -c "
from config import MODEL_REGISTRY
for k,v in MODEL_REGISTRY.items():
    if v['short_name'] == '$WINNER_DISPLAY':
        print(k); break
")
            echo "Refitting winner: $WINNER_DISPLAY ($WINNER_KEY)"
            python scripts/04_model_fitting/c_level2/fit_with_l2.py \
                --model "$WINNER_KEY" \
                --data {config[data_file]} \
                --chains {config[mcmc_chains]} \
                --warmup {config[mcmc_warmup]} \
                --samples {config[mcmc_samples]} \
                --l2-subdir {config[l2_subdir]} \
                --baseline-subdir {config[bayesian_subdir]}
        done

        touch {output.flag}
        """


# =============================================================================
# Stage 5d: Scale Audit (Bayesian track)
# =============================================================================
# Validates L2 refit posteriors: beta HDIs, ESS degradation vs baseline.

rule scale_audit:
    input:
        "cluster/results/level2_refit.done",
    output:
        flag="cluster/results/scale_audit.done",
    resources:
        slurm_partition=lambda wc, attempt: lookup_resource("scale_audit", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("scale_audit", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("scale_audit", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("scale_audit", "runtime"),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Stage 5d: Scale Audit"
        export PYTHONUNBUFFERED=1

        python scripts/05_post_fitting_checks/02_scale_audit.py \
            --l2-dir models/bayesian/{config[l2_subdir]}/ \
            --baseline-dir models/bayesian/{config[bayesian_subdir]}/

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


# --- 6.3: MLE Trauma Analysis ---
# Group comparisons + correlations between model parameters and trauma measures.
rule mle_trauma:
    input:
        "cluster/results/mle_compare.done",
    output:
        flag="cluster/results/mle_trauma.done",
    resources:
        slurm_partition=lambda wc, attempt: lookup_resource("mle_trauma", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("mle_trauma", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("mle_trauma", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("mle_trauma", "runtime"),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Stage 6.3: MLE Trauma Analysis"
        export PYTHONUNBUFFERED=1

        python scripts/06_fit_analyses/04_analyze_mle_by_trauma.py --model all

        touch {output.flag}
        """


# --- 6.4: MLE Regression on Scales ---
# Univariate + multiple regressions of model parameters on trauma scales.
rule mle_regression:
    input:
        "cluster/results/mle_trauma.done",
    output:
        flag="cluster/results/mle_regression.done",
    resources:
        slurm_partition=lambda wc, attempt: lookup_resource("mle_regression", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("mle_regression", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("mle_regression", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("mle_regression", "runtime"),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Stage 6.4: MLE Regression on Scales"
        export PYTHONUNBUFFERED=1

        python scripts/06_fit_analyses/05_regress_parameters_on_scales.py --model all

        touch {output.flag}
        """


# --- 6.5: Winner Heterogeneity ---
# Per-participant winner analysis using M6b as reference frame.
rule winner_heterogeneity:
    input:
        "cluster/results/mle_compare.done",
    output:
        flag="cluster/results/winner_heterogeneity.done",
    resources:
        slurm_partition=lambda wc, attempt: lookup_resource("winner_heterogeneity", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("winner_heterogeneity", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("winner_heterogeneity", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("winner_heterogeneity", "runtime"),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Stage 6.5: Winner Heterogeneity"
        export PYTHONUNBUFFERED=1

        python scripts/06_fit_analyses/06_analyze_winner_heterogeneity.py

        touch {output.flag}
        """


# --- 6.6: Cross-Model Significance Heatmap ---
# Parameter x trauma association matrix across M3, M5, M6a, M6b.
rule cross_model_heatmap:
    input:
        "cluster/results/mle_regression.done",
    output:
        flag="cluster/results/cross_model_heatmap.done",
    resources:
        slurm_partition=lambda wc, attempt: lookup_resource("cross_model_heatmap", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("cross_model_heatmap", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("cross_model_heatmap", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("cross_model_heatmap", "runtime"),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Stage 6.6: Cross-Model Significance Heatmap"
        export PYTHONUNBUFFERED=1

        python scripts/06_fit_analyses/09_plot_cross_model_significance.py

        touch {output.flag}
        """


# --- 6.7: Manuscript Tables ---
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


# --- 6.8: Model Averaging ---
# Stacking-weighted averaging of L2 beta posteriors across winners.
rule model_averaging:
    input:
        "cluster/results/scale_audit.done",
    output:
        flag="cluster/results/model_averaging.done",
    resources:
        slurm_partition=lambda wc, attempt: lookup_resource("model_averaging", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("model_averaging", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("model_averaging", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("model_averaging", "runtime"),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Stage 6.8: Model Averaging"
        export PYTHONUNBUFFERED=1

        python scripts/06_fit_analyses/03_model_averaging.py \
            --l2-dir models/bayesian/{config[l2_subdir]}/ \
            --stacking-results models/bayesian/{config[bayesian_subdir]}/loo_stacking_results.csv

        touch {output.flag}
        """


# --- 6.9: Bayesian Level-2 Effects ---
# Forest plots for L2 beta coefficients from winner's L2-refit posterior.
rule level2_effects:
    input:
        "cluster/results/scale_audit.done",
    output:
        flag="cluster/results/level2_effects.done",
    resources:
        slurm_partition=lambda wc, attempt: lookup_resource("level2_effects", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("level2_effects", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("level2_effects", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("level2_effects", "runtime"),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Stage 6.9: Bayesian Level-2 Effects"
        export PYTHONUNBUFFERED=1

        mkdir -p reports/figures/bayesian

        WINNERS=$(cat models/bayesian/{config[bayesian_subdir]}/winners.txt)
        for WINNER_DISPLAY in $(echo "$WINNERS" | tr ',' ' '); do
            WINNER_KEY=$(python -c "
from config import MODEL_REGISTRY
for k,v in MODEL_REGISTRY.items():
    if v['short_name'] == '$WINNER_DISPLAY':
        print(k); break
")
            python scripts/06_fit_analyses/07_bayesian_level2_effects.py \
                --model "$WINNER_KEY" \
                --posterior-path models/bayesian/{config[l2_subdir]}/"$WINNER_KEY"_posterior.nc
        done

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
        "cluster/results/ppc.done",
        "cluster/results/mle_trauma.done",
        "cluster/results/mle_regression.done",
        "cluster/results/winner_heterogeneity.done",
        "cluster/results/cross_model_heatmap.done",
        "cluster/results/model_averaging.done",
        "cluster/results/level2_effects.done",
    output:
        "cluster/results/pipeline_done.flag",
    shell:
        "touch {output}"
