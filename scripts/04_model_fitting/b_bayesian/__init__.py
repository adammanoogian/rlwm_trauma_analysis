# Stage folder name starts with a digit so standard dotted-path imports do
# NOT work (`scripts.04_model_fitting.b_bayesian` is illegal in Python —
# dotted names cannot start with a digit). External callers that need to
# reuse library functions should load `_engine.py` via
# `importlib.util.spec_from_file_location`, following the pattern in
# `fit_bayesian.py` and `fit_baseline.py` (the two CLI wrappers in this
# folder).
#
# Layout (Scheme D, plan 29-04b):
#   - `fit_bayesian.py`  — ad-hoc CLI entry (~85-line argparse wrapper)
#   - `fit_baseline.py`  — Phase 21 pipeline entry (forces
#                           --output-subdir=21_baseline, guards convergence)
#   - `_engine.py`       — library implementation (1,173 lines; underscore-
#                           private so it does not collide with the CLI name).
"""Bayesian fitting entry-points, implementation, and baseline fit (04/b-bayesian)."""
