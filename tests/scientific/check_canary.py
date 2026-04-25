"""Canary acceptance gate — productionizes Task 3 of 24-01-PLAN.md.

Reads the canary submission artifacts written by ``cluster/03_submit_canary.sh``
(``21_canary_jobids.txt``, ``21_canary_metadata.txt``) and evaluates the four
CONTEXT.md §Submission strategy acceptance criteria plus per-model Baribault
gate verdicts (HARD_PASS / SOFT_PASS / FAIL).

Writes ``models/bayesian/21_canary_acceptance_report.md`` with PASS/FAIL per
criterion + per-model row + overall APPROVED/REJECTED verdict.

Invoked from ``cluster/03_check_canary.slurm``; can also be run interactively
on the M3 login node from the project root::

    python tests/scientific/check_canary.py

Exit codes
----------
0
    APPROVED — canary acceptance gate green; operator can proceed to
    ``bash cluster/submit_all.sh --from-stage 4``.
1
    REJECTED — one or more hard criteria failed; operator must root-cause
    before submitting a fresh canary.

Notes
-----
SOFT_PASS verdicts (encoded in scripts/utils/ppc.py:_classify_gate_verdict)
are treated as advisory — they do NOT block APPROVED. Only FAIL verdicts
do. This matches the Phase 24 Wave 1 soft-pass policy decision.

Path-constants source: tests/scientific/check_phase24_artifacts.py
(Wave 0 pin). Canary artifact filenames must match the TAG=21_canary
default in cluster/03_submit_canary.sh.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path so the tests.scientific.* import resolves
# regardless of the caller's CWD (matches the pattern in
# scripts/03_model_prefitting/04_run_prior_predictive.py).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from tests.scientific.check_phase24_artifacts import (  # noqa: E402
    BASELINE_MODELS,
    MODELS_DIR,
    PRIOR_PREDICTIVE_DIR,
)

# Negative-check legacy path (Wave 0 §Path-drift acceptance).
LEGACY_BAYESIAN_DIR: Path = Path("output/bayesian")

# Canary artifact filenames — must match TAG=21_canary in
# cluster/03_submit_canary.sh.
CANARY_TAG: str = "21_canary"
JOBIDS_FILE: Path = MODELS_DIR / f"{CANARY_TAG}_jobids.txt"
METADATA_FILE: Path = MODELS_DIR / f"{CANARY_TAG}_metadata.txt"
REPORT_FILE: Path = MODELS_DIR / f"{CANARY_TAG}_acceptance_report.md"


@dataclass(frozen=True)
class CriterionResult:
    """Outcome of one acceptance-gate criterion."""

    name: str
    verdict: str  # PASS | FAIL
    detail: str


@dataclass(frozen=True)
class ModelGateResult:
    """Per-model Baribault gate readout (parsed from ``{model}_gate.md``)."""

    model: str
    jid: str
    verdict: str  # HARD_PASS | SOFT_PASS | FAIL | UNKNOWN
    median: float | None
    detail: str


def _read_jobids() -> dict[str, str]:
    """Parse ``21_canary_jobids.txt`` into ``{model: jid}``."""
    if not JOBIDS_FILE.exists():
        return {}
    out: dict[str, str] = {}
    for line in JOBIDS_FILE.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("|")
        if len(parts) != 3:
            continue
        _, model, jid = parts
        out[model] = jid
    return out


def _read_metadata() -> dict[str, str]:
    """Parse ``21_canary_metadata.txt`` into a dict."""
    if not METADATA_FILE.exists():
        return {}
    out: dict[str, str] = {}
    for line in METADATA_FILE.read_text().splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            out[key.strip()] = value.strip()
    return out


def _check_a_sacct_and_nc(jobids: dict[str, str]) -> CriterionResult:
    """Criterion (a): SLURM exit 0 for all jobs AND ``.nc`` files exist."""
    name = "(a) SLURM exit 0 + 6 .nc files"
    if not jobids:
        return CriterionResult(name, "FAIL", "no jobids file")

    # SLURM completion state via sacct — skipped on dev box (no sacct).
    sacct_detail = ""
    if shutil.which("sacct"):
        jid_list = ",".join(jobids.values())
        try:
            proc = subprocess.run(
                [
                    "sacct",
                    "-j",
                    jid_list,
                    "--format=JobID,JobName,State,ExitCode,Elapsed",
                    "-P",
                    "--noheader",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            rows = [
                line.split("|") for line in proc.stdout.splitlines() if line.strip()
            ]
            parents = [r for r in rows if "." not in r[0]]
            fails = [r for r in parents if r[2] != "COMPLETED" or r[3] != "0:0"]
            if fails:
                return CriterionResult(
                    name,
                    "FAIL",
                    f"{len(fails)} parent jobs not COMPLETED 0:0: "
                    f"{[r[:4] for r in fails[:3]]}",
                )
            sacct_detail = f"{len(parents)} jobs COMPLETED 0:0"
        except (subprocess.TimeoutExpired, OSError) as exc:
            return CriterionResult(name, "FAIL", f"sacct error: {exc!r}")
    else:
        sacct_detail = "(sacct unavailable — SLURM exit-code check skipped)"

    missing = [
        m
        for m in BASELINE_MODELS
        if not (PRIOR_PREDICTIVE_DIR / f"{m}_prior_sim.nc").exists()
    ]
    if missing:
        return CriterionResult(
            name, "FAIL", f"missing prior_sim.nc for {missing}"
        )

    return CriterionResult(
        name,
        "PASS",
        f"{sacct_detail}; all {len(BASELINE_MODELS)} prior_sim.nc files present",
    )


def _check_b_arviz_load() -> CriterionResult:
    """Criterion (b): ArviZ loads each ``.nc`` file with at least one group."""
    name = "(b) ArviZ load + dim check"
    try:
        import arviz as az
    except ImportError:
        return CriterionResult(name, "FAIL", "arviz not installed in current env")

    failures: list[str] = []
    summary: list[str] = []
    for m in BASELINE_MODELS:
        path = PRIOR_PREDICTIVE_DIR / f"{m}_prior_sim.nc"
        if not path.exists():
            failures.append(f"{m}: missing")
            continue
        try:
            idata = az.from_netcdf(str(path))
            groups = list(idata.groups())
            if not groups:
                failures.append(f"{m}: no ArviZ groups")
            else:
                summary.append(f"{m}({len(groups)})")
        except Exception as exc:  # noqa: BLE001 — surface every load error
            failures.append(f"{m}: {exc!r}")

    if failures:
        return CriterionResult(name, "FAIL", "; ".join(failures[:5]))
    return CriterionResult(
        name,
        "PASS",
        f"all {len(BASELINE_MODELS)} loaded — {', '.join(summary)}",
    )


def _check_c_autopush_commit(metadata: dict[str, str]) -> CriterionResult:
    """Criterion (c): ≥ 1 commit on main touches the canary dir post-submit."""
    name = "(c) Autopush commit on main"
    ts = metadata.get("Canary timestamp (UTC)")
    if not ts:
        return CriterionResult(name, "FAIL", "no canary timestamp in metadata")

    # Convert "20260425T193756Z" -> "2026-04-25 19:37:56" for git log --since.
    try:
        dt = datetime.strptime(ts, "%Y%m%dT%H%M%SZ")
        since = dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return CriterionResult(name, "FAIL", f"unparseable timestamp: {ts}")

    try:
        proc = subprocess.run(
            [
                "git",
                "log",
                "--oneline",
                "--since",
                since,
                "--",
                str(PRIOR_PREDICTIVE_DIR),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as exc:
        return CriterionResult(
            name,
            "FAIL",
            f"git log failed: {exc.stderr.strip() if exc.stderr else exc!r}",
        )

    commits = [line for line in proc.stdout.splitlines() if line.strip()]
    if not commits:
        return CriterionResult(
            name,
            "FAIL",
            f"no commits touching {PRIOR_PREDICTIVE_DIR} since {since}",
        )
    return CriterionResult(
        name, "PASS", f"{len(commits)} commit(s) since {since}"
    )


def _check_d_ccds_path() -> CriterionResult:
    """Criterion (d): CCDS path populated AND legacy path empty/absent."""
    name = "(d) CCDS-canonical path landing"
    missing = [
        m
        for m in BASELINE_MODELS
        if not (PRIOR_PREDICTIVE_DIR / f"{m}_prior_sim.nc").exists()
    ]
    if missing:
        return CriterionResult(
            name,
            "FAIL",
            f"CCDS path {PRIOR_PREDICTIVE_DIR} missing {missing}",
        )

    legacy = LEGACY_BAYESIAN_DIR / "21_prior_predictive"
    if legacy.exists():
        contents = sorted(p.name for p in legacy.iterdir())
        if contents:
            return CriterionResult(
                name,
                "FAIL",
                f"legacy {legacy} has {len(contents)} entries: {contents[:5]}",
            )
    return CriterionResult(
        name, "PASS", f"CCDS populated; legacy {legacy} absent or empty"
    )


def _read_model_gate(model: str, jid: str) -> ModelGateResult:
    """Parse ``{model}_gate.md`` into a :class:`ModelGateResult`.

    Recognizes three verdict tiers — HARD_PASS, SOFT_PASS, FAIL — emitted
    by ``scripts/utils/ppc.py:_classify_gate_verdict``. Older gate.md files
    that emitted only PASS/FAIL get back-compat-mapped: PASS -> HARD_PASS.
    """
    gate_path = PRIOR_PREDICTIVE_DIR / f"{model}_gate.md"
    if not gate_path.exists():
        return ModelGateResult(model, jid, "UNKNOWN", None, "gate.md not found")

    text = gate_path.read_text()
    match = re.search(r"\*\*Verdict:\*\*\s*\*\*(\w+)\*\*", text)
    raw = match.group(1).upper() if match else "UNKNOWN"
    verdict = "HARD_PASS" if raw == "PASS" else raw  # back-compat

    # Extract median by parsing the "| Median accuracy | ... | <value> | ... |"
    # row. The value column is a bare decimal (not inside the threshold
    # brackets like "[0.40, 0.90]"), so iterate columns and pick the LAST
    # bare-decimal cell — robust to both old (binary) and new (three-tier)
    # gate.md layouts.
    median: float | None = None
    median_row = re.search(r"\|\s*Median accuracy\s*\|.*", text)
    if median_row:
        bare_decimal = re.compile(r"^\d+\.\d+$")
        cells = [p.strip() for p in median_row.group(0).split("|")]
        bare_values = [c for c in cells if bare_decimal.fullmatch(c)]
        if bare_values:
            try:
                median = float(bare_values[-1])
            except ValueError:
                median = None

    return ModelGateResult(model, jid, verdict, median, gate_path.name)


def _write_acceptance_report(
    metadata: dict[str, str],
    jobids: dict[str, str],
    criteria: list[CriterionResult],
    gates: list[ModelGateResult],
    overall_verdict: str,
) -> None:
    """Compose ``models/bayesian/21_canary_acceptance_report.md``."""
    lines = [
        "# Phase 24 Canary Acceptance Report",
        "",
        f"- **Canary timestamp (UTC):** "
        f"{metadata.get('Canary timestamp (UTC)', '?')}",
        f"- **HEAD commit at submission:** "
        f"{metadata.get('HEAD commit at submission', '?')}",
        f"- **Submit host:** {metadata.get('Hostname', '?')}",
        f"- **STEP / Tag:** {metadata.get('STEP', '?')} / "
        f"{metadata.get('Tag', CANARY_TAG)}",
        f"- **Canary JobIDs:** "
        f"{', '.join(f'{m}={j}' for m, j in jobids.items())}",
        f"- **Report generated (UTC):** "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "## Four-criteria acceptance gate (CONTEXT.md §Submission strategy)",
        "",
        "| Criterion | Verdict | Evidence |",
        "|---|---|---|",
    ]
    for c in criteria:
        lines.append(f"| {c.name} | **{c.verdict}** | {c.detail} |")

    lines += [
        "",
        "## Per-model Baribault gate verdicts (advisory, SOFT_PASS not blocking)",
        "",
        "Three-tier policy "
        "(scripts/utils/ppc.py:_classify_gate_verdict):",
        "",
        "- **HARD_PASS** — within original Baribault & Collins (2023) hard band.",
        "- **SOFT_PASS** — within documented soft margin; advisory; "
        "monitor in stage 04b fitting.",
        "- **FAIL** — outside both bands; canary REJECTED.",
        "",
        "| Model | JID | Verdict | Median acc | Source |",
        "|---|---|---|---|---|",
    ]
    for g in gates:
        median = f"{g.median:.3f}" if g.median is not None else "?"
        lines.append(
            f"| {g.model} | {g.jid} | **{g.verdict}** | {median} | {g.detail} |"
        )

    lines += [
        "",
        "## Overall verdict",
        "",
        f"**{overall_verdict}**",
        "",
    ]
    if overall_verdict == "APPROVED":
        lines += [
            "All four cluster→repo flow criteria passed; per-model FAIL count "
            "is zero (SOFT_PASS treated as advisory). Wave 1 Task 4 may proceed.",
            "",
            "Recommended next step:",
            "",
            "```bash",
            "bash cluster/submit_all.sh --from-stage 4   "
            "# full chain from stage 04b onwards",
            "```",
        ]
    else:
        lines += [
            "One or more hard criteria failed. CONTEXT.md §Submission strategy: "
            "canary failure policy applies (no retry, no fix-forward).",
            "",
            "Operator action: diagnose the failed criterion above, patch the "
            "root cause, re-run Wave 0 pre-flight gate, then submit a fresh "
            "canary with a new timestamp.",
        ]

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Entry point — returns 0 on APPROVED, 1 on REJECTED."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.parse_args()

    metadata = _read_metadata()
    jobids = _read_jobids()
    if not jobids:
        print(
            f"FAIL: {JOBIDS_FILE} not found or empty — was the canary submitted?",
            file=sys.stderr,
        )
        return 1

    criteria: list[CriterionResult] = [
        _check_a_sacct_and_nc(jobids),
        _check_b_arviz_load(),
        _check_c_autopush_commit(metadata),
        _check_d_ccds_path(),
    ]
    gates: list[ModelGateResult] = [
        _read_model_gate(m, jobids.get(m, "?")) for m in BASELINE_MODELS
    ]

    # APPROVED iff all four criteria PASS AND no per-model FAIL.
    # SOFT_PASS treated as advisory (does not block APPROVED).
    criteria_pass = all(c.verdict == "PASS" for c in criteria)
    gate_fails = [g for g in gates if g.verdict == "FAIL"]
    overall = "APPROVED" if (criteria_pass and not gate_fails) else "REJECTED"

    _write_acceptance_report(metadata, jobids, criteria, gates, overall)

    print("=" * 60)
    print("Phase 24 Canary Acceptance Gate")
    print("=" * 60)
    for c in criteria:
        print(f"  [{c.verdict:<4}] {c.name}: {c.detail}")
    print()
    for g in gates:
        median = f"{g.median:.3f}" if g.median is not None else "?"
        print(
            f"  [{g.verdict:<10}] {g.model:<10} "
            f"jid={g.jid} median={median}"
        )
    print("-" * 60)
    print(f"OVERALL: {overall}")
    print(f"Report:  {REPORT_FILE}")
    print("=" * 60)

    return 0 if overall == "APPROVED" else 1


if __name__ == "__main__":
    sys.exit(main())
