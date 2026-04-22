"""One-off human-run maintenance CLIs (not importable shared utilities).

Moved here from ``scripts/utils/`` during Phase 29 canonical-layout reorg
(plan 29-03): a file belongs in ``scripts/utils/`` if-and-only-if it
provides helpers imported by two or more stage folders. The scripts in
this package are pure CLIs with no importers; they live here so
``scripts/utils/`` is reserved for true shared libraries.

Scripts
-------
remap_mle_ids
    One-time remapping of legacy MLE participant IDs to canonical
    assigned_ids.
sync_experiment_data
    Safely copy new participant CSVs from the experiment folder into the
    analysis data directory.
update_participant_mapping
    Scan ``data/`` and refresh ``participant_id_mapping.json`` with new
    anonymous IDs for unseen participants.

These scripts are invoked ad-hoc by humans when a data-curation task
arises; they are not part of the automated pipeline.
"""
