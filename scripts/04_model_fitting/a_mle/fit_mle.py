#!/usr/bin/env python
"""MLE fitting entry-point (stage 04/a) — thin CLI.

Maximum Likelihood Estimation fitting for computational models. This script
is a thin argparse wrapper; the fitting library lives alongside it as
``_engine.py`` (underscore-private per Scheme D — plan 29-04b renamed the
3,157-line library out of the way so this file could take the canonical
``fit_mle.py`` name).

Models available
----------------
- ``qlearning``: Q-learning (M1), 3 params (α₊, α₋, ε)
- ``wmrl``: WM-RL (M2), 6 params (α₊, α₋, φ, ρ, K, ε)
- ``wmrl_m3``: WM-RL + perseveration (M3), 7 params
- ``wmrl_m5``: WM-RL + RL forgetting (M5), 8 params
- ``wmrl_m6a``: WM-RL + stimulus-specific perseveration (M6a), 7 params
- ``wmrl_m6b``: WM-RL + dual perseveration (M6b), 8 params
- ``wmrl_m4``: RLWM-LBA joint choice+RT (M4), 10 params (separate track)

Key features
------------
- JAX-accelerated likelihood computation
- Multi-start optimization via ``jaxopt.LBFGS``
- Parallel fitting across participants (``--n-jobs``)
- Optional GPU acceleration (``--use-gpu``)
- AIC/BIC computation for model comparison

Inputs / outputs
----------------
- In:  ``output/task_trials_long.csv`` (behavioral data)
- Out: ``output/mle/<model>_mle_results.csv`` (fitted parameters)
- Out: ``output/mle/<model>_model_fit.json`` (fit metadata)

Usage
-----
>>> python scripts/04_model_fitting/a_mle/fit_mle.py --model qlearning
>>> python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m3 --n-jobs 16
>>> python scripts/04_model_fitting/a_mle/fit_mle.py --model wmrl_m3 --use-gpu

Cluster execution
-----------------
>>> sbatch cluster/12_mle.slurm                            # CPU parallel
>>> sbatch --export=MODEL=wmrl_m3 cluster/12_mle_gpu.slurm  # GPU

Next steps
----------
- Run ``scripts/06_fit_analyses/01_compare_models.py`` for model selection
- Run ``scripts/06_fit_analyses/04_analyze_mle_by_trauma.py`` for
  parameter-trauma relationships
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
# parent package `scripts.04_model_fitting.a_mle` cannot be imported via
# the standard dotted form because Python dotted names cannot start with a
# digit (`04_model_fitting` is illegal).  Relative imports `from ._engine`
# also fail when this file is invoked as a script (no known parent
# package).  Workaround: load the engine module directly by absolute path.
_ENGINE_PATH = _THIS_FILE.with_name("_engine.py")
_spec = importlib.util.spec_from_file_location(
    "_mle_engine", str(_ENGINE_PATH)
)
assert _spec is not None and _spec.loader is not None, (
    f"Could not create import spec for MLE engine at {_ENGINE_PATH}"
)
_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_engine)
main = _engine.main


if __name__ == "__main__":
    main()
