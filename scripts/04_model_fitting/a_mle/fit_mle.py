#!/usr/bin/env python
"""MLE fitting entry-point (stage 04/a) — thin CLI.

Maximum Likelihood Estimation fitting for computational models. This script
is a thin argparse wrapper; the fitting library lives at canonical path
``rlwm.fitting.mle`` (moved from ``_engine.py`` in v5.0 shim cleanup).

Models available
----------------
- ``qlearning``: Q-learning (M1), 3 params (alpha+, alpha-, epsilon)
- ``wmrl``: WM-RL (M2), 6 params
- ``wmrl_m3``: WM-RL + perseveration (M3), 7 params
- ``wmrl_m5``: WM-RL + RL forgetting (M5), 8 params
- ``wmrl_m6a``: WM-RL + stimulus-specific perseveration (M6a), 7 params
- ``wmrl_m6b``: WM-RL + dual perseveration (M6b), 8 params
- ``wmrl_m4``: RLWM-LBA joint choice+RT (M4), 10 params (separate track)

Usage
-----
>>> python scripts/04_model_fitting/a_mle/fit_mle.py --model qlearning
>>> python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m3 --n-jobs 16
>>> python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m3 --use-gpu

Cluster execution
-----------------
>>> sbatch cluster/04a_mle_cpu.slurm
>>> sbatch --export=MODEL=wmrl_m3 cluster/04a_mle_gpu.slurm
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap project root onto sys.path so `import config`, `import rlwm`,
# etc. resolve regardless of the caller's CWD.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from rlwm.fitting.mle import main


if __name__ == "__main__":
    main()
