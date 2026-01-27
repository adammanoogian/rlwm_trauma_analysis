# Participant Exclusions Applied

## Summary
- **Total participants in raw data:** 54
- **Excluded participants:** 6
- **Final sample for all analyses:** 48

## Exclusion Criteria

### Data Quality (4 participants)
Participants excluded for insufficient trial completion:

1. **ID 10012:** 87 trials (expected 807-1077)
   - Completed <10% of experiment
   
2. **ID 10040:** 12 trials, 0% accuracy
   - Abandoned experiment immediately after starting
   
3. **ID 10045:** 39 trials, 23% accuracy
   - Did not complete experiment
   
4. **ID 10049:** 18 trials, 28% accuracy
   - Did not complete experiment

### Duplicate Participants (2 participants)
Participants excluded as duplicate sessions:

5. **ID 10044:** Duplicate of 10043
   - Same participant started experiment twice on 2025-11-24
   - 10043 started at 16:31:17, 10044 at 16:31:24 (7 seconds apart)
   - Keeping 10043 (first session)
   
6. **ID 10073:** Duplicate of 10072
   - Same participant started experiment twice on 2025-12-18
   - 10072 started at 20:03:58, 10073 at 20:04:04 (6 seconds apart)
   - Keeping 10072 (first session)

## Implementation

Exclusions are applied in:

### Configuration
- `config.py`: Central list of excluded participant IDs
  ```python
  EXCLUDED_PARTICIPANTS = [10012, 10040, 10045, 10049, 10044, 10073]
  ```

### Behavioral Analysis Pipeline
- `scripts/run_statistical_analyses.py`: ANOVA and regression analyses
- `scripts/generate_descriptive_tables.py`: Demographic and performance tables
- All statistical outputs reflect N=48

### Computational Modeling Pipeline
- `scripts/fitting/fit_mle.py`: Maximum likelihood parameter estimation
- `scripts/fitting/fit_with_jax.py`: Bayesian parameter estimation  
- `scripts/analysis/regress_parameters_on_scales.py`: Parameter-trauma correlations
- All model fitting uses same N=48 participants

## Manuscript Correspondence

This N=48 matches the sample size reported in the manuscript methods section.

After exclusions, trauma groups are:
- **Trauma - No Ongoing Impact (IES-R <24):** n = TBD
- **Trauma - Ongoing Impact (IES-R ≥24):** n = TBD

(Run pipeline to get final group sizes)

## Next Steps

Run the complete pipeline to regenerate all outputs with N=48:
```bash
conda activate ds_env
python run_data_pipeline.py --no-sync
```

This will update:
- Table 1 (Demographics) with N=48
- Table 2 (Trauma Scores) with N=48  
- Table 3 (Performance by Load×Group) with N=48
- ANOVA results with df=(1,46) for between-subjects effects
- All figures and supplementary tables
