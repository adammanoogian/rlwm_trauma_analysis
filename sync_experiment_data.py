"""
Sync Experiment Data to Analysis Project

Safely copies new participant CSV files from the experiment data folder
to the analysis project data folder without modifying source files.

Usage:
    python sync_experiment_data.py
"""

import shutil
from pathlib import Path
from datetime import datetime
import sys

# Configuration
EXPERIMENT_DATA_DIR = Path('../rlwm_trauma/data')  # Experiment folder
ANALYSIS_DATA_DIR = Path('data')  # Analysis project folder
FILE_PATTERN = 'rlwm_trauma_PARTICIPANT_SESSION_*.csv'
LOG_FILE = ANALYSIS_DATA_DIR / 'sync_log.txt'

def log_message(message, print_to_console=True):
    """Write message to log file and optionally print to console."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_entry = f"[{timestamp}] {message}"

    if print_to_console:
        print(message)

    # Append to log file
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')

def validate_experiment_folder():
    """Check if experiment data folder exists and is accessible."""
    if not EXPERIMENT_DATA_DIR.exists():
        print(f"ERROR: Experiment data folder not found: {EXPERIMENT_DATA_DIR.absolute()}")
        print(f"\nExpected location: {EXPERIMENT_DATA_DIR.absolute()}")
        print("\nPlease ensure the experiment folder exists or update EXPERIMENT_DATA_DIR in this script.")
        return False

    if not EXPERIMENT_DATA_DIR.is_dir():
        print(f"ERROR: Path exists but is not a directory: {EXPERIMENT_DATA_DIR.absolute()}")
        return False

    return True

def sync_data():
    """Sync new data files from experiment folder to analysis folder."""

    print("=" * 80)
    print("SYNCING EXPERIMENT DATA TO ANALYSIS PROJECT")
    print("=" * 80)
    print()

    # Validate folders
    print(f"Source (experiment): {EXPERIMENT_DATA_DIR.absolute()}")
    print(f"Target (analysis):   {ANALYSIS_DATA_DIR.absolute()}")
    print()

    if not validate_experiment_folder():
        log_message("SYNC FAILED: Experiment folder not found", print_to_console=False)
        return False

    # Ensure analysis data folder exists
    ANALYSIS_DATA_DIR.mkdir(exist_ok=True)

    # Find all matching CSV files in experiment folder
    source_files = sorted(EXPERIMENT_DATA_DIR.glob(FILE_PATTERN))

    if not source_files:
        print(f"No files matching pattern '{FILE_PATTERN}' found in experiment folder.")
        log_message(f"SYNC COMPLETE: No files found matching {FILE_PATTERN}")
        return True

    print(f"Found {len(source_files)} files in experiment folder")
    print()

    # Track sync statistics
    new_files = []
    existing_files = []
    updated_files = []
    errors = []

    # Copy new files
    for source_file in source_files:
        target_file = ANALYSIS_DATA_DIR / source_file.name

        try:
            if target_file.exists():
                # Check if source is newer than target
                source_mtime = source_file.stat().st_mtime
                target_mtime = target_file.stat().st_mtime

                if source_mtime > target_mtime:
                    # Source is newer, update the file
                    shutil.copy2(source_file, target_file)
                    updated_files.append(source_file.name)
                    print(f"  UPDATED: {source_file.name}")
                else:
                    # Target is up to date
                    existing_files.append(source_file.name)
            else:
                # New file, copy it
                shutil.copy2(source_file, target_file)
                new_files.append(source_file.name)
                print(f"  NEW: {source_file.name}")

        except Exception as e:
            errors.append((source_file.name, str(e)))
            print(f"  ERROR: {source_file.name} - {e}")

    # Summary
    print()
    print("=" * 80)
    print("SYNC SUMMARY")
    print("=" * 80)
    print(f"Total files in experiment folder: {len(source_files)}")
    print(f"New files copied: {len(new_files)}")
    print(f"Existing files updated: {len(updated_files)}")
    print(f"Existing files (unchanged): {len(existing_files)}")
    print(f"Errors: {len(errors)}")

    # Log summary
    log_message(f"SYNC COMPLETE: {len(new_files)} new, {len(updated_files)} updated, {len(existing_files)} unchanged, {len(errors)} errors", print_to_console=False)

    if new_files:
        log_message(f"New files: {', '.join(new_files)}", print_to_console=False)

    if updated_files:
        log_message(f"Updated files: {', '.join(updated_files)}", print_to_console=False)

    if errors:
        print("\nErrors encountered:")
        for filename, error in errors:
            print(f"  {filename}: {error}")
            log_message(f"ERROR: {filename} - {error}", print_to_console=False)

    print()
    print(f"Sync log: {LOG_FILE}")
    print("=" * 80)

    return len(errors) == 0

def main():
    """Main entry point."""
    success = sync_data()

    if not success:
        sys.exit(1)

    print("\nNext steps:")
    print("  1. Run: python scripts/update_participant_mapping.py")
    print("  2. Run: python run_data_pipeline.py --no-sync")
    print()

if __name__ == '__main__':
    main()
