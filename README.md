# RLWM Trauma Analysis - Data Processing Pipeline

This repository contains a data processing pipeline for analyzing jsPsych experimental data from a Reinforcement Learning Working Memory (RLWM) task combined with trauma assessment surveys.

## Overview

The pipeline processes raw jsPsych output and creates three analysis-ready datasets:
1. **Collated Dataset** - Wide-format with all individual responses
2. **Task Trials Dataset** - Long-format trial-by-trial task data
3. **Summary Dataset** - Derived metrics and scale scores

## Requirements

- Python 3.10+
- Conda or Miniconda
- Required packages: pandas>=2.0.0, numpy>=2.0.0

## Environment Setup

### Option 1: Use the existing ds_env environment
If you already have the `ds_env` environment:
```bash
conda activate ds_env
```

### Option 2: Create environment from environment.yml (Recommended)
Recreate the exact environment with all dependencies:
```bash
conda env create -f environment.yml
conda activate ds_env
```

### Option 3: Create minimal environment with pip
Install only core dependencies:
```bash
conda create -n rlwm_analysis python=3.10
conda activate rlwm_analysis
pip install -r requirements.txt
```

### Option 4: Update existing environment
```bash
conda activate ds_env
conda env update -f environment.yml
```

## Pipeline Structure

```
rlwm_trauma_analysis/
├── example_dataset_pilot.csv        # Raw jsPsych data (input)
├── README.md                        # Documentation
├── environment.yml                  # Full conda environment specification
├── requirements.txt                 # Core Python dependencies
├── scripts/
│   ├── 01_parse_raw_data.py        # Extract and parse raw data
│   ├── 02_create_collated_csv.py   # Create collated dataset
│   ├── 03_create_task_trials_csv.py # Create trial-level dataset
│   ├── 04_create_summary_csv.py    # Create summary metrics dataset
│   └── utils/
│       ├── data_cleaning.py        # Data parsing functions
│       └── scoring_functions.py    # Scoring and metrics functions
└── output/
    ├── parsed_demographics.csv      # Intermediate file
    ├── parsed_survey1.csv           # Intermediate file
    ├── parsed_survey2.csv           # Intermediate file
    ├── parsed_task_trials.csv       # Intermediate file
    ├── collated_participant_data.csv   # FINAL OUTPUT 1
    ├── task_trials_long.csv            # FINAL OUTPUT 2
    └── summary_participant_metrics.csv # FINAL OUTPUT 3
```

## Usage

Run the scripts sequentially:

```bash
# Step 1: Parse raw data
python scripts/01_parse_raw_data.py

# Step 2: Create collated dataset
python scripts/02_create_collated_csv.py

# Step 3: Create task trials dataset
python scripts/03_create_task_trials_csv.py

# Step 4: Create summary metrics dataset
python scripts/04_create_summary_csv.py
```

Or run all steps at once:
```bash
python scripts/01_parse_raw_data.py && python scripts/02_create_collated_csv.py && python scripts/03_create_task_trials_csv.py && python scripts/04_create_summary_csv.py
```

## Data Sources

### Demographics (8 variables)
- age_years, country, primary_language
- gender, education, relationship_status, living_arrangement, screen_time

### Survey 1: Life Events Checklist - DSM-5 (LEC-5)
- **15 items** assessing trauma exposure
- **Multi-select responses**: personal, witnessed, learned, job-related, unsure, not applicable
- **Output format**: 30 binary columns (any_exposure + personal for each item)

**Derived scores:**
- `lec_total_events`: Count of unique traumatic events experienced
- `lec_personal_events`: Count of events experienced personally
- `lec_sum_exposures`: Total exposure count across all types

### Survey 2: Impact of Event Scale - Revised (IES-R)
- **22 items** measuring PTSD symptomatology
- **Response scale**: 0 (Not at all) to 4 (Extremely)
- **Timeframe**: Past 7 days

**Subscales:**
- **Intrusion** (7 items): Items 1, 3, 6, 9, 14, 16, 20
- **Avoidance** (8 items): Items 5, 7, 8, 11, 12, 13, 17, 22
- **Hyperarousal** (7 items): Items 2, 4, 10, 15, 18, 19, 21

**Derived scores:**
- `ies_intrusion`, `ies_avoidance`, `ies_hyperarousal`, `ies_total`

### Task: Reinforcement Learning Working Memory (RLWM)
- **Design**: Probabilistic reversal learning with working memory manipulation
- **Blocks**: Main task blocks 3-23 (practice blocks 1-2 excluded)
- **Set sizes**: 2, 3, 5, 6 stimuli
- **Load conditions**: Low (set size ≤ 3), High (set size > 3)
- **Reversals**: Rare reversals after 12-18 consecutive correct responses

**Trial structure:**
- Present stimulus image
- Response: J, K, or L key
- Feedback: +1 (correct) or 0 (incorrect)
- 2000ms timeout, 500ms feedback, 500ms fixation

## Output Datasets

### 1. Collated Participant Data (collated_participant_data.csv)
**Format**: One row per participant
**Columns**: 95 total
- Participant ID (1)
- Demographics (8)
- Survey 1 responses (30 binary: any_exposure + personal per item)
- Survey 2 responses (22 numeric: 0-4 scale)
- Task performance metrics (34)

### 2. Task Trials Long (task_trials_long.csv)
**Format**: One row per trial
**Columns**: 20 total
- sona_id, trial_in_experiment, block, trial_in_block
- set_size, load_condition, stimulus
- key_press, key_answer, correct
- rt, rt_category, timeout
- time_elapsed, trial_index, phase_type
- set, reversal_crit, counter

**Use cases:**
- Trial-level analysis
- Learning curves
- Response time distributions
- Error analysis

### 3. Summary Participant Metrics (summary_participant_metrics.csv)
**Format**: One row per participant
**Columns**: 50 total
- Participant ID (1)
- Demographics (8)
- LEC-5 summary scores (3)
- IES-R summary scores (4)
- Task metrics (34)

**Task metrics include:**
- **Overall**: accuracy, mean RT, median RT, completion rate
- **By load**: accuracy & RT for low/high load
- **By set size**: accuracy & RT for set sizes 2, 3, 5, 6
- **By time period**: early/middle/late block accuracy & RT
- **Learning**: learning slope, improvement from early to late
- **Reversals**: number detected, performance drop, adaptation rate

## Data Quality

From pilot data (1 participant, 990 trials):
- Overall accuracy: 81.7%
- Timeout rate: 0.0%
- Mean RT: 612ms
- Low load accuracy: 84.8%
- High load accuracy: 80.2%
- Learning slope: 0.0087 (positive improvement)
- Reversals detected: 111

## Key Features

1. **Robust parsing**: Handles multi-select survey responses and JSON-encoded data
2. **Comprehensive metrics**: 34 task performance measures per participant
3. **Flexible output**: Both wide (collated) and long (trial-level) formats
4. **Validated scoring**: Standard LEC-5 and IES-R subscale calculations
5. **Advanced task metrics**: Learning curves, reversal adaptation, set-size effects

## Data Dictionary

### LEC-5 Item Labels
1. Natural disaster
2. Transportation accident
3. Significant incident resulting in injury or harm
4. Physically aggressive or threatening interaction
5. Exposure to weapon-related violence or threats
6. Exposure to war/combat environments
7. Forced confinement or loss of personal freedom
8. Serious illness or injury with potential risk to life
9. Severe human suffering
10. Sudden/unexpected death
11. Bullying, harassment, or intimidation
12. Displacement or homelessness
13. Sudden/distressing change in family structure
14. Prolonged emotional mistreatment/invalidation/neglect
15. Any other very stressful event or experience

### IES-R Scoring
- **Clinical cutoff**: Total score ≥ 33 suggests probable PTSD
- **Subscale interpretation**:
  - Intrusion: Re-experiencing symptoms
  - Avoidance: Avoidance behaviors and emotional numbing
  - Hyperarousal: Heightened arousal and reactivity

## Notes

- Main task includes blocks 3+ only (blocks 1-2 are practice)
- Set size 4 excluded per experimental design (EXCLUDE_SET_SIZES)
- Task sections in jsPsych output have section=NaN (identified by block column)
- Reversal detection based on counter reset (> 3 step decrease)
- Missing demographic data normal for pilot testing

## Citation

If you use this pipeline, please cite the original scales:
- **LEC-5**: Weathers, F.W., et al. (2013). The Life Events Checklist for DSM-5 (LEC-5)
- **IES-R**: Weiss, D.S., & Marmar, C.R. (1997). The Impact of Event Scale - Revised

## Environment Details

The `ds_env` conda environment includes:

**Core packages:**
- Python 3.10.18
- pandas 2.3.1
- numpy 2.2.6

**Additional analysis tools:**
- matplotlib 3.10.3
- seaborn 0.13.2
- scipy 1.15.2
- scikit-learn 1.7.1
- statsmodels 0.14.5

**Development tools:**
- JupyterLab 4.4.5
- IPython 8.37.0

Full environment specifications are available in `environment.yml`.

## Contact

For questions about the pipeline or data analysis, please open an issue in this repository.
