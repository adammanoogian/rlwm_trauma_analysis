"""
Monitor WM-RL fitting progress and write status to log file.
Run with: python scripts/monitor_wmrl.py [check_name]
"""
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def get_process_status(pid=196612):
    """Check if process is still running and get CPU time."""
    try:
        result = subprocess.run(
            ['powershell', '-Command',
             f"Get-Process -Id {pid} -ErrorAction SilentlyContinue | Select-Object Id,TotalProcessorTime,WorkingSet64 | ConvertTo-Json"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0 and result.stdout.strip():
            import json
            data = json.loads(result.stdout)
            return {
                'running': True,
                'cpu_time': data.get('TotalProcessorTime', 'unknown'),
                'memory_gb': data.get('WorkingSet64', 0) / (1024**3)
            }
    except Exception as e:
        pass
    return {'running': False, 'cpu_time': None, 'memory_gb': 0}

def check_log_completion():
    """Check if the WM-RL fitting log shows completion."""
    log_path = Path('output/mle_wmrl_fitting_log.txt')
    if log_path.exists():
        content = log_path.read_text()
        if 'FITTING COMPLETE' in content or 'Fitting complete' in content:
            return True, content
    return False, None

def check_output_files():
    """Check if output files exist."""
    wmrl_fits = Path('output/mle/wmrl_individual_fits.csv')
    wmrl_summary = Path('output/mle/wmrl_group_summary.csv')
    return wmrl_fits.exists(), wmrl_summary.exists()

def main():
    check_name = sys.argv[1] if len(sys.argv) > 1 else "manual"

    monitor_log = Path('output/wmrl_monitor_log.txt')

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Gather status
    process = get_process_status()
    completed, log_content = check_log_completion()
    fits_exist, summary_exists = check_output_files()

    # Build status report
    lines = [
        f"\n{'='*60}",
        f"[{timestamp}] WM-RL MONITOR CHECK: {check_name}",
        f"{'='*60}",
    ]

    if completed:
        lines.append("STATUS: [COMPLETED]")
        lines.append(f"  - wmrl_individual_fits.csv exists: {fits_exist}")
        lines.append(f"  - wmrl_group_summary.csv exists: {summary_exists}")
        if log_content:
            # Extract summary from log
            if 'Group Summary' in log_content:
                summary_start = log_content.find('=== WM-RL Group Summary ===')
                if summary_start != -1:
                    lines.append("\nGROUP SUMMARY:")
                    lines.append(log_content[summary_start:summary_start+500])
    elif process['running']:
        lines.append("STATUS: Still running")
        lines.append(f"  - CPU Time: {process['cpu_time']}")
        lines.append(f"  - Memory: {process['memory_gb']:.2f} GB")
        lines.append(f"  - Output files ready: fits={fits_exist}, summary={summary_exists}")
    else:
        lines.append("STATUS: Process NOT running")
        lines.append(f"  - Output files exist: fits={fits_exist}, summary={summary_exists}")
        if fits_exist:
            lines.append("  - Process may have completed!")
        else:
            lines.append("  - WARNING: Process stopped but no output files found!")

    lines.append(f"{'='*60}\n")

    report = '\n'.join(lines)

    # Write to monitor log (UTF-8 to avoid encoding issues)
    with open(monitor_log, 'a', encoding='utf-8') as f:
        f.write(report)

    # Also print to stdout
    print(report)

if __name__ == '__main__':
    main()
