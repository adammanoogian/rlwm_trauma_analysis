# cluster/rules/validate.smk -- Scientific validation rules.
#
# These rules submit pytest to the cluster for tests that need
# GPU or long runtime. Integrated into the DAG so they can depend
# on data prep and optionally gate pipeline execution.
#
# Test code lives in tests/ with markers:
#   @pytest.mark.slow       -- long-running (>3 min)
#   @pytest.mark.scientific -- scientific validation (parameter recovery, etc.)
#
# Run validation independently:
#   snakemake -s cluster/Snakefile --profile cluster/snakemake-profile validate

localrules: validate


rule validate_scientific:
    """Run scientific validation tests (parameter recovery, model consistency) on GPU."""
    output:
        flag="cluster/results/validate_scientific_done.flag",
        report="cluster/results/validate_scientific_report.xml",
    resources:
        partition=lambda wc, attempt: lookup_resource("validate_scientific", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("validate_scientific", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("validate_scientific", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("validate_scientific", "runtime"),
        slurm_extra=lambda wc, attempt: (
            f"'--gres=gpu:{lookup_resource('validate_scientific', 'gpus')}'"
            if lookup_resource("validate_scientific", "gpus") > 0
            else "''"
        ),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Validate: Scientific Tier"
        setup_jax_cache gpu
        verify_gpu
        export PYTHONUNBUFFERED=1

        python -m pytest tests/scientific/ \
            -v --tb=short --no-header \
            --junitxml={output.report} \
        && touch {output.flag}
        """


rule validate_integration:
    """Run integration tests (fitting smoke, structure guard) on CPU."""
    output:
        flag="cluster/results/validate_integration_done.flag",
        report="cluster/results/validate_integration_report.xml",
    resources:
        partition=lambda wc, attempt: lookup_resource("validate_integration", "partition"),
        mem_mb=lambda wc, attempt: lookup_resource("validate_integration", "mem_mb"),
        cpus_per_task=lambda wc, attempt: lookup_resource("validate_integration", "cpus_per_task"),
        runtime=lambda wc, attempt: lookup_resource("validate_integration", "runtime"),
    shell:
        """
        source cluster/_setup.sh
        print_job_header "Validate: Integration Tier"
        export PYTHONUNBUFFERED=1

        python -m pytest tests/integration/ \
            -v --tb=short --no-header \
            --junitxml={output.report} \
        && touch {output.flag}
        """


rule validate:
    """Aggregate -- all validation passes."""
    input:
        "cluster/results/validate_scientific_done.flag",
        "cluster/results/validate_integration_done.flag",
    output:
        "cluster/results/validate_done.flag",
    shell:
        "touch {output}"
