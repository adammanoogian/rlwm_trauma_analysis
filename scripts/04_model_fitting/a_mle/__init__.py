# Stage folder name starts with a digit so standard dotted-path imports do
# NOT work (`scripts.04_model_fitting.a_mle` is illegal in Python — dotted
# names cannot start with a digit). External callers that need to reuse
# library functions should load `_engine.py` via
# `importlib.util.spec_from_file_location`, following the pattern in
# `fit_mle.py` (the thin CLI in this folder).
#
# Layout (Scheme D, plan 29-04b):
#   - `fit_mle.py`  — canonical CLI entry (74-line argparse wrapper)
#   - `_engine.py`  — library implementation (3,157 lines; underscore-
#                     private so it does not collide with the CLI name).
"""MLE fitting entry-point and implementation (04/a-mle)."""
