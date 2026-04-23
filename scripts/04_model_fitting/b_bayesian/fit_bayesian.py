#!/usr/bin/env python
"""Bayesian hierarchical fitting entry-point (stage 04/b) — thin CLI.

Bayesian hierarchical model fitting using NumPyro NUTS. This script is a
thin argparse wrapper; the fitting library lives at canonical path
``rlwm.fitting.bayesian`` (moved from ``_engine.py`` in v5.0 shim cleanup).

For the principled v4.0 Bayesian pipeline, prefer ``fit_baseline.py``
(pipeline entry, forces ``--output-subdir=21_baseline`` and guards the
convergence gate). Use this script for ad-hoc single-model fits that do
not need to land in the 21_baseline/ subdir.

Models available
----------------
- ``qlearning``: Q-learning (M1)
- ``wmrl``: WM-RL (M2)
- ``wmrl_m3``: WM-RL + perseveration (M3)
- ``wmrl_m5``: WM-RL + RL forgetting (M5)
- ``wmrl_m6a``: WM-RL + stimulus-specific perseveration (M6a)
- ``wmrl_m6b``: WM-RL + dual perseveration (M6b)
- ``wmrl_m4``: RLWM-LBA joint choice+RT (M4) — separate GPU pipeline

Usage
-----
>>> python scripts/04_model_fitting/b_bayesian/fit_bayesian.py --model qlearning
>>> python scripts/04_model_fitting/b_bayesian/fit_bayesian.py --model wmrl_m6b \\
...     --chains 4 --warmup 1000 --samples 2000
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

from rlwm.fitting.bayesian import main


if __name__ == "__main__":
    main()
