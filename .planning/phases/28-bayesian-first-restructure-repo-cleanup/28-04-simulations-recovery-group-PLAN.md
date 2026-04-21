---
wave: 2
depends_on: [28-01]
files_modified:
  - scripts/simulations_recovery/09_generate_synthetic_data.py  (git mv)
  - scripts/simulations_recovery/09_run_ppc.py  (git mv)
  - scripts/simulations_recovery/10_run_parameter_sweep.py  (git mv)
  - scripts/simulations_recovery/11_run_model_recovery.py  (git mv)
  - scripts/simulations_recovery/__init__.py  (new, empty)
  - cluster/09_ppc_gpu.slurm
  - cluster/11_recovery_gpu.slurm
autonomous: true
---

# 28-04 Group Simulations & Recovery Scripts (09–11)

## Goal

Move the four simulations/recovery scripts into `scripts/simulations_recovery/` via `git mv`, update the two SLURM files that invoke them, and verify the `scripts.fitting.model_recovery` imports still resolve from the new locations.

## Must Haves

- [ ] `scripts/simulations_recovery/` directory exists with `__init__.py`.
- [ ] All four files moved via `git mv`:
  - `scripts/simulations_recovery/09_generate_synthetic_data.py`
  - `scripts/simulations_recovery/09_run_ppc.py`
  - `scripts/simulations_recovery/10_run_parameter_sweep.py`
  - `scripts/simulations_recovery/11_run_model_recovery.py`
- [ ] Original `scripts/09_*.py`, `scripts/10_*.py`, `scripts/11_*.py` paths no longer exist.
- [ ] `cluster/09_ppc_gpu.slurm` updated to call `python scripts/simulations_recovery/09_run_ppc.py` (or whichever 09 script it invokes; verify by reading the SLURM file).
- [ ] `cluster/11_recovery_gpu.slurm` updated to call `python scripts/simulations_recovery/11_run_model_recovery.py`.
- [ ] `scripts/simulations_recovery/09_run_ppc.py` `from scripts.fitting.model_recovery import ...` still resolves (assuming REFAC-02 kept model_recovery.py in scripts/fitting/ per planning_context §1).
- [ ] `scripts/simulations_recovery/11_run_model_recovery.py` same verification.
- [ ] `pytest scripts/fitting/tests/test_v4_closure.py` still passes 3/3.
- [ ] Atomic commit: `refactor(28-04): group simulations/recovery scripts 09-11 under scripts/simulations_recovery/`.

## Tasks

<tasks>
  <task id="1">
    <title>Pre-flight: verify exact SLURM invocations</title>
    <detail>Read `cluster/09_ppc_gpu.slurm` and `cluster/11_recovery_gpu.slurm` to confirm which 09/11 scripts they call. The `09_` prefix has TWO scripts (`09_generate_synthetic_data.py` and `09_run_ppc.py`); verify which the SLURM invokes. Grep `cluster/` for any other refs to 09/10/11 scripts.</detail>
  </task>

  <task id="2">
    <title>Create destination directory + __init__.py</title>
    <detail>`mkdir -p scripts/simulations_recovery/` with empty `__init__.py`.</detail>
  </task>

  <task id="3">
    <title>git mv the four scripts</title>
    <detail>
      - `git mv scripts/09_generate_synthetic_data.py scripts/simulations_recovery/09_generate_synthetic_data.py`
      - `git mv scripts/09_run_ppc.py scripts/simulations_recovery/09_run_ppc.py`
      - `git mv scripts/10_run_parameter_sweep.py scripts/simulations_recovery/10_run_parameter_sweep.py`
      - `git mv scripts/11_run_model_recovery.py scripts/simulations_recovery/11_run_model_recovery.py`</detail>
  </task>

  <task id="4">
    <title>Update cluster SLURM files</title>
    <detail>Edit:
      - `cluster/09_ppc_gpu.slurm` — change `python scripts/09_*.py` to `python scripts/simulations_recovery/09_*.py` (use the exact filename the SLURM invokes, per task 1).
      - `cluster/11_recovery_gpu.slurm` — same pattern.
      Do NOT change any other SLURM fields (wall time, memory, GPU requests, env setup).</detail>
  </task>

  <task id="5">
    <title>Smoke-test</title>
    <detail>Run `python scripts/simulations_recovery/09_run_ppc.py --help` and `python scripts/simulations_recovery/11_run_model_recovery.py --help` to confirm the `scripts.fitting.model_recovery` import still resolves after the move.</detail>
  </task>

  <task id="6">
    <title>Atomic commit</title>
    <detail>`refactor(28-04): group simulations/recovery scripts 09-11 under scripts/simulations_recovery/`. Note the SLURM updates in commit body.</detail>
  </task>
</tasks>

## Verification

```bash
test -f scripts/simulations_recovery/__init__.py
test -f scripts/simulations_recovery/09_generate_synthetic_data.py
test -f scripts/simulations_recovery/09_run_ppc.py
test -f scripts/simulations_recovery/10_run_parameter_sweep.py
test -f scripts/simulations_recovery/11_run_model_recovery.py

test ! -f scripts/09_generate_synthetic_data.py
test ! -f scripts/09_run_ppc.py
test ! -f scripts/10_run_parameter_sweep.py
test ! -f scripts/11_run_model_recovery.py

# SLURM updates present
grep -n "simulations_recovery" cluster/09_ppc_gpu.slurm cluster/11_recovery_gpu.slurm

# No stale SLURM refs
! grep -rn "scripts/09_run_ppc\|scripts/11_run_model_recovery" cluster/ --include="*.slurm" --include="*.sh" | grep -v simulations_recovery

# Smoke
python scripts/simulations_recovery/09_run_ppc.py --help
python scripts/simulations_recovery/11_run_model_recovery.py --help

pytest scripts/fitting/tests/test_v4_closure.py -v
```

## Requirement IDs

Closes: **REFAC-05**.
