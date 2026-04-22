#!/usr/bin/env python
"""Bayesian hierarchical fitting entry-point (stage 04/b) — thin CLI.

Bayesian hierarchical model fitting using NumPyro NUTS. This script is a
thin argparse wrapper; the fitting library lives alongside it as
``_engine.py`` (underscore-private per Scheme D — plan 29-04b renamed the
1,173-line library out of the way so this file could take the canonical
``fit_bayesian.py`` name).

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

Key features
------------
- Full posterior distributions (not just point estimates)
- Hierarchical structure: individual + group-level parameters
- NUTS sampler for efficient posterior exploration
- ArviZ integration for diagnostics and visualization

Inputs / outputs
----------------
- In:  ``output/task_trials_long.csv`` (behavioral data)
- Out: ``output/bayesian_fits/<model>_trace.nc`` (ArviZ InferenceData)
- Out: ``output/bayesian_fits/<model>_summary.csv`` (parameter summary)
- Out: ``figures/bayesian_fits/<model>_posterior.png``

Usage
-----
>>> python scripts/04_model_fitting/b_bayesian/fit_bayesian.py --model qlearning
>>> python scripts/04_model_fitting/b_bayesian/fit_bayesian.py --model wmrl_m6b \\
...     --chains 4 --warmup 1000 --samples 2000

Next steps
----------
- Run ``scripts/06_fit_analyses/01_compare_models.py --use-waic`` for
  Bayesian model comparison
- Examine posterior distributions for parameter uncertainty
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# Bootstrap project root onto sys.path so `import config`, `import rlwm`,
# etc. resolve regardless of the caller's CWD.
_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import the `main` entry point from the co-located library engine.  The
# parent package `scripts.04_model_fitting.b_bayesian` cannot be imported
# via the standard dotted form because Python dotted names cannot start
# with a digit (`04_model_fitting` is illegal).  Relative imports
# `from ._engine` also fail when this file is invoked as a script (no
# known parent package).  Workaround: load the engine module directly by
# absolute path.
_ENGINE_PATH = _THIS_FILE.with_name("_engine.py")
_spec = importlib.util.spec_from_file_location(
    "_bayesian_engine", str(_ENGINE_PATH)
)
assert _spec is not None and _spec.loader is not None, (
    f"Could not create import spec for Bayesian engine at {_ENGINE_PATH}"
)
_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_engine)
main = _engine.main


if __name__ == "__main__":
    main()
