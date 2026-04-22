"""Shared utilities for the RLWM pipeline.

Any function used by two or more stage folders lives here as a single
authoritative source of truth.  Stage scripts import via
``from scripts.utils.X import Y`` — never via cross-stage imports.

Modules
-------
data_cleaning
    jsPsych raw-data parsers (surveys, demographics, task trials).
scoring
    Survey and task scoring (LESS, IES-R, task metrics).
stats
    Statistical-test helpers (assumption checks, mixed ANOVAs, regressions).
plotting
    Shared plot helpers (color palettes, annotated scatters, KDEs).
ppc
    Single-source posterior / prior predictive-check simulators used by
    stage 03 (prior PPC) and stage 05 (posterior PPC).
"""
