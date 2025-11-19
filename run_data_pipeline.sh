#!/bin/bash
# Complete Data Processing and Analysis Pipeline
# Runs all data parsing, cleaning, and behavioral analysis (excludes model fitting/simulation)
# By default, syncs data from experiment folder before processing

set -e  # Exit on any error

echo "=========================================="
echo "RLWM Trauma Analysis - Data Pipeline"
echo "=========================================="
echo ""

echo "[1/11] Syncing data from experiment folder..."
python sync_experiment_data.py

echo "[2/11] Updating participant ID mapping..."
python scripts/update_participant_mapping.py

echo "[3/11] Parsing raw jsPsych data..."
python scripts/01_parse_raw_data.py

echo "[4/11] Creating collated participant data..."
python scripts/02_create_collated_csv.py

echo "[5/11] Creating task trials CSV..."
python scripts/03_create_task_trials_csv.py

echo "[6/11] Creating summary CSV..."
python scripts/04_create_summary_csv.py

echo "[7/11] Parsing all participants (including partial data)..."
python scripts/parse_all_participants.py

echo "[8/11] Generating human performance visualizations..."
python scripts/analysis/visualize_human_performance.py --data output/task_trials_long_all_participants.csv

echo "[9/11] Generating scale distributions..."
python scripts/analysis/visualize_scale_distributions.py

echo "[10/11] Generating scale correlations..."
python scripts/analysis/visualize_scale_correlations.py

echo "[11/11] Creating summary report..."
python scripts/analysis/summarize_behavioral_data.py

echo ""
echo "=========================================="
echo "Pipeline Complete!"
echo "=========================================="
echo "Outputs generated in:"
echo "  - output/*.csv"
echo "  - figures/behavioral_summary/*.png"
echo "  - output/behavioral_summary/data_summary_report.txt"
