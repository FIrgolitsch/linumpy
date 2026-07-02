#!/usr/bin/env python3
"""Validate Phase 3 library consistency catalog deliverables."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tomllib
from pathlib import Path

CANONICAL_HEADER = (
    "id,invariant,tier,target_phase,status,file,line,subsystem,pipeline_stage,"
    "entry_point,description,test_module,fix_commit"
)

INVARIANT_KEYS = (
    "axis_order",
    "metadata",
    "shifts_motor",
    "thread_config",
    "cli_boundary",
    "deliverables",
)

AUDIT_SECTIONS = (
    "AUDIT-01",
    "AUDIT-02",
    "AUDIT-03",
    "AUDIT-04",
    "AUDIT-05",
    "AUDIT-06",
)

MIN_CATALOG_ROWS = 86


def repo_root() -> Path:
    """Resolve repository root via git or parent walk."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        )
        return Path(result.stdout.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return Path(__file__).resolve().parents[2]


def deliverable_paths(root: Path) -> dict[str, Path]:
    planning = root / ".planning"
    return {
        "map": planning / "consistency-map.md",
        "index": planning / "consistency-index.csv",
        "gate": planning / "consistency-gate.md",
    }


def fail(reasons: list[str]) -> None:
    for reason in reasons:
        print(reason, file=sys.stderr)
    sys.exit(1)


def load_entry_points(root: Path) -> list[str]:
    pyproject = root / "pyproject.toml"
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    scripts = data.get("project", {}).get("scripts", {})
    return sorted(scripts.keys())


def read_csv_rows(index_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with index_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return [], []
        rows = list(reader)
    return list(reader.fieldnames), rows


def check_structure(root: Path) -> list[str]:
    reasons: list[str] = []
    paths = deliverable_paths(root)

    for name, path in paths.items():
        if not path.is_file():
            reasons.append(f"FAIL: missing deliverable file: {path.relative_to(root)}")

    if reasons:
        return reasons

    index_path = paths["index"]
    first_line = index_path.read_text(encoding="utf-8").splitlines()[0].strip()
    if first_line != CANONICAL_HEADER:
        reasons.append(
            f"FAIL: consistency-index.csv header mismatch\n  expected: {CANONICAL_HEADER}\n  got:      {first_line}"
        )

    map_text = paths["map"].read_text(encoding="utf-8")
    for section in AUDIT_SECTIONS:
        if f"## {section}" not in map_text:
            reasons.append(f"FAIL: consistency-map.md missing H2 section ## {section}")

    if "```mermaid" not in map_text:
        reasons.append("FAIL: consistency-map.md missing fenced mermaid diagram block")
    if "ASCII fallback" not in map_text:
        reasons.append("FAIL: consistency-map.md missing ASCII fallback diagram block")

    gate_text = paths["gate"].read_text(encoding="utf-8")
    if "BLOCKING" not in gate_text or "|" not in gate_text:
        reasons.append("FAIL: consistency-gate.md missing BLOCKING checklist table")
    if "Gate summary" not in gate_text:
        reasons.append("FAIL: consistency-gate.md missing Gate summary section")

    return reasons


def check_catalog(root: Path) -> list[str]:
    reasons = check_structure(root)
    if reasons:
        return reasons

    paths = deliverable_paths(root)
    map_text = paths["map"].read_text(encoding="utf-8")
    if "Module cross-reference index" not in map_text:
        reasons.append("FAIL: consistency-map.md missing Module cross-reference index heading")

    _, rows = read_csv_rows(paths["index"])
    if len(rows) < MIN_CATALOG_ROWS:
        reasons.append(f"FAIL: consistency-index.csv has {len(rows)} data rows; need >= {MIN_CATALOG_ROWS}")

    entry_points = load_entry_points(root)
    present = {row.get("entry_point", "").strip() for row in rows if row.get("entry_point", "").strip()}
    missing = [ep for ep in entry_points if ep not in present]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        reasons.append(f"FAIL: {len(missing)} entry points missing from CSV entry_point column: {preview}{suffix}")

    return reasons


def check_gate(root: Path) -> list[str]:
    reasons: list[str] = []
    _, rows = read_csv_rows(deliverable_paths(root)["index"])
    blocking_open = [
        row
        for row in rows
        if row.get("tier", "").strip() == "BLOCKING" and row.get("status", "").strip() == "open"
    ]
    if blocking_open:
        for row in blocking_open:
            reasons.append(
                f"FAIL: BLOCKING open row {row.get('id', '?')}: {row.get('file', '?')} — {row.get('description', '')}"
            )

    high_phase6_open = [
        row
        for row in rows
        if row.get("tier", "").strip() == "HIGH"
        and row.get("target_phase", "").strip() == "6"
        and row.get("status", "").strip() == "open"
    ]
    if high_phase6_open:
        for row in high_phase6_open:
            reasons.append(
                f"FAIL: HIGH Phase-6 open row {row.get('id', '?')}: {row.get('file', '?')} — {row.get('description', '')}"
            )
    return reasons


def run_mode(mode: str) -> int:
    root = repo_root()
    if mode == "structure":
        reasons = check_structure(root)
    elif mode == "catalog":
        reasons = check_catalog(root)
    elif mode == "gate":
        reasons = check_gate(root)
    elif mode == "all":
        reasons = check_catalog(root)
        if not reasons:
            reasons = check_gate(root)
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        return 2

    if reasons:
        fail(reasons)
    print(f"OK: {mode}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["structure", "catalog", "gate", "all"],
        help="validation mode",
    )
    args = parser.parse_args()
    sys.exit(run_mode(args.mode))


if __name__ == "__main__":
    main()
