#!/usr/bin/env python3
"""Validate Phase 4 PR chain inventory deliverables."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

CANONICAL_HEADER = "pr_number,branch,intended_base,merge_position,status,commit_count,drift_flags"

# Open stacked PRs in header-authoritative merge order (excludes merged #115, #97).
CANONICAL_OPEN_PR_ORDER: tuple[int, ...] = (
    98,
    99,
    100,
    101,
    108,
    106,
    107,
    87,
    116,
    110,
    111,
    40,
    112,
    130,
    131,
    118,
    132,
    121,
    122,
    123,
    124,
    128,
    133,
    134,
)

# Zero-scope merge-order slots; GitHub PR may be CLOSED when branch equals predecessor.
PASS_THROUGH_PRS: frozenset[int] = frozenset({106, 118, 121, 122})

MERGED_FOUNDATION_PRS: frozenset[int] = frozenset({115, 97})

REQUIRED_MANIFEST_SECTIONS: tuple[str, ...] = (
    "Intended scope",
    "Assigned commits",
    "File ownership",
    "Recorded base",
    "Intended base",
    "Drift flags",
    "Phase 5 verification checklist",
)

STUB_MODES: frozenset[str] = frozenset()

SIMULATED_DIFF_AUDIT_SECTION = "Simulated Diff Audit"

FOUNDATION_HEAD_BRANCH = "pr-c-utility-preprocessing"

COMMIT_MAP_HEADER = "sha,pr_number,is_merge,ambiguous,candidates,assignment_reason"


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
        "index": planning / "pr-chain-index.csv",
        "chain_dir": planning / "pr-chain",
    }


def fail(reasons: list[str]) -> None:
    for reason in reasons:
        print(reason, file=sys.stderr)
    sys.exit(1)


def read_csv_rows(index_path: Path) -> tuple[list[str] | None, list[dict[str, str]]]:
    with index_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return None, []
        rows = list(reader)
    return list(reader.fieldnames), rows


def manifest_files(chain_dir: Path) -> list[Path]:
    if not chain_dir.is_dir():
        return []
    excluded = {"audit-report.md"}
    return sorted(p for p in chain_dir.glob("*-*.md") if p.name not in excluded)


def pr_number_from_manifest(path: Path) -> int | None:
    stem = path.stem
    if "-" not in stem:
        return None
    prefix = stem.split("-", 1)[0]
    if not prefix.isdigit():
        return None
    return int(prefix)


def check_structure(root: Path) -> list[str]:
    reasons: list[str] = []
    paths = deliverable_paths(root)
    index_path = paths["index"]
    chain_dir = paths["chain_dir"]

    if not index_path.is_file():
        reasons.append(f"FAIL: missing deliverable file: {index_path.relative_to(root)}")
        return reasons

    first_line = index_path.read_text(encoding="utf-8").splitlines()[0].strip()
    if first_line != CANONICAL_HEADER:
        reasons.append(
            f"FAIL: pr-chain-index.csv header mismatch\n  expected: {CANONICAL_HEADER}\n  got:      {first_line}"
        )

    manifests = manifest_files(chain_dir)
    if not manifests:
        reasons.append(f"FAIL: no manifest files in {chain_dir.relative_to(root)}/")
        return reasons

    for manifest_path in manifests:
        text = manifest_path.read_text(encoding="utf-8")
        for section in REQUIRED_MANIFEST_SECTIONS:
            if f"## {section}" not in text:
                reasons.append(
                    f"FAIL: {manifest_path.relative_to(root)} missing H2 section ## {section}"
                )

    return reasons


def check_inventory(root: Path) -> list[str]:
    reasons = check_structure(root)
    if reasons:
        return reasons

    paths = deliverable_paths(root)
    chain_dir = paths["chain_dir"]
    index_path = paths["index"]

    manifests = manifest_files(chain_dir)
    manifest_prs = {pr_number_from_manifest(path) for path in manifests}
    manifest_prs.discard(None)

    open_pr_set = set(CANONICAL_OPEN_PR_ORDER)

    missing_manifests = [pr for pr in CANONICAL_OPEN_PR_ORDER if pr not in manifest_prs]
    if missing_manifests:
        preview = ", ".join(f"#{pr}" for pr in missing_manifests[:5])
        suffix = "..." if len(missing_manifests) > 5 else ""
        reasons.append(f"FAIL: missing manifests for open PRs: {preview}{suffix}")

    extra_manifests = sorted(manifest_prs - open_pr_set)
    if extra_manifests:
        draft_allowed: set[int] = set()
        map_path = chain_dir / "commit-map.csv"
        if map_path.is_file():
            for row in read_commit_map(chain_dir):
                raw = row.get("pr_number", "").strip()
                if raw.isdigit():
                    draft_allowed.add(int(raw))
        allowed_draft: set[int] = set()
        for manifest_path in manifests:
            pr = pr_number_from_manifest(manifest_path)
            if pr is None or pr in open_pr_set:
                continue
            text = manifest_path.read_text(encoding="utf-8")
            if "draft_new_pr_slot: true" in text and pr in draft_allowed:
                allowed_draft.add(pr)
        extra_manifests = [pr for pr in extra_manifests if pr not in allowed_draft]
        if extra_manifests:
            preview = ", ".join(f"#{pr}" for pr in extra_manifests[:5])
            suffix = "..." if len(extra_manifests) > 5 else ""
            reasons.append(f"FAIL: unexpected manifest files for non-open PRs: {preview}{suffix}")

    manifest_count_for_open = len(manifest_prs & open_pr_set)
    if manifest_count_for_open != len(CANONICAL_OPEN_PR_ORDER):
        reasons.append(
            f"FAIL: expected {len(CANONICAL_OPEN_PR_ORDER)} open-PR manifests; "
            f"found {manifest_count_for_open}"
        )

    foundation_in_manifests = manifest_prs & MERGED_FOUNDATION_PRS
    if foundation_in_manifests:
        preview = ", ".join(f"#{pr}" for pr in sorted(foundation_in_manifests))
        reasons.append(f"FAIL: merged foundation PRs must not have manifests: {preview}")

    _, rows = read_csv_rows(index_path)
    if not rows:
        reasons.append("FAIL: pr-chain-index.csv has no data rows")
        return reasons

    csv_prs: dict[int, dict[str, str]] = {}
    for row in rows:
        raw = row.get("pr_number", "").strip().lstrip("#")
        if not raw.isdigit():
            reasons.append(f"FAIL: invalid pr_number in CSV row: {row.get('pr_number', '?')!r}")
            continue
        csv_prs[int(raw)] = row

    for pr in MERGED_FOUNDATION_PRS:
        row = csv_prs.get(pr)
        if row is None:
            reasons.append(f"FAIL: merged foundation PR #{pr} missing from pr-chain-index.csv")
        elif row.get("status", "").strip() != "merged":
            reasons.append(
                f"FAIL: foundation PR #{pr} must have status=merged (got {row.get('status', '')!r})"
            )

    for pr in CANONICAL_OPEN_PR_ORDER:
        row = csv_prs.get(pr)
        if row is None:
            reasons.append(f"FAIL: open PR #{pr} missing from pr-chain-index.csv")

    return reasons


def git_dev_commit_shas(root: Path) -> set[str]:
    """Read-only: all commit SHAs in merge-base(main,dev)..dev."""
    merge_base = subprocess.run(
        ["git", "merge-base", "main", "dev"],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
    ).stdout.strip()
    result = subprocess.run(
        ["git", "log", "--format=%H", f"{merge_base}..dev"],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
    )
    return {line.strip() for line in result.stdout.splitlines() if line.strip()}


def read_commit_map(chain_dir: Path) -> list[dict[str, str]]:
    map_path = chain_dir / "commit-map.csv"
    if not map_path.is_file():
        return []
    with map_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def known_manifest_pr_numbers(chain_dir: Path) -> set[int]:
    numbers: set[int] = set()
    for path in manifest_files(chain_dir):
        pr = pr_number_from_manifest(path)
        if pr is not None:
            numbers.add(pr)
    return numbers


def git_dev_tip(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "dev"],
        check=True,
        capture_output=True,
        text=True,
        cwd=root,
    )
    return result.stdout.strip()


def index_rows_by_pr(
    index_path: Path,
) -> tuple[list[str], dict[int, dict[str, str]]]:
    _, rows = read_csv_rows(index_path)
    by_pr: dict[int, dict[str, str]] = {}
    for row in rows:
        raw = row.get("pr_number", "").strip().lstrip("#")
        if raw.isdigit():
            by_pr[int(raw)] = row
    return rows, by_pr


def manifest_for_pr(chain_dir: Path, pr_number: int) -> Path | None:
    matches = sorted(chain_dir.glob(f"{pr_number}-*.md"))
    return matches[0] if matches else None


def check_manifest_drift_sections(chain_dir: Path) -> list[str]:
    reasons: list[str] = []
    for pr in CANONICAL_OPEN_PR_ORDER:
        manifest_path = manifest_for_pr(chain_dir, pr)
        if manifest_path is None:
            reasons.append(f"FAIL: missing manifest for open PR #{pr}")
            continue
        text = manifest_path.read_text(encoding="utf-8")
        if "draft_new_pr_slot: true" in text:
            continue
        if "## Intended base" not in text:
            reasons.append(
                f"FAIL: {manifest_path.relative_to(chain_dir.parent.parent)} missing ## Intended base"
            )
        if "<!-- stub: filled in 04-03 -->" in text:
            reasons.append(
                f"FAIL: {manifest_path.relative_to(chain_dir.parent.parent)} has unfilled Drift flags stub"
            )
        drift_match = re.search(r"## Drift flags\n\n(.*?)\n\n## ", text, re.DOTALL)
        if drift_match is None or not drift_match.group(1).strip():
            reasons.append(
                f"FAIL: {manifest_path.relative_to(chain_dir.parent.parent)} has empty Drift flags section"
            )
    return reasons


def check_stack_linearity(index_by_pr: dict[int, dict[str, str]]) -> list[str]:
    reasons: list[str] = []
    first_open = CANONICAL_OPEN_PR_ORDER[0]
    first_row = index_by_pr.get(first_open)
    if first_row and first_row.get("intended_base", "").strip() == "main":
        predecessor_head = "main"
    else:
        predecessor_head = FOUNDATION_HEAD_BRANCH
    for pr in CANONICAL_OPEN_PR_ORDER:
        row = index_by_pr.get(pr)
        if row is None:
            reasons.append(f"FAIL: open PR #{pr} missing from pr-chain-index.csv")
            continue
        intended = row.get("intended_base", "").strip()
        branch = row.get("branch", "").strip()
        if intended != predecessor_head:
            reasons.append(
                f"FAIL: stack link broken at PR #{pr}: intended_base={intended!r} "
                f"!= predecessor head={predecessor_head!r}"
            )
        predecessor_head = branch
    return reasons


def check_end_state(
    chain_dir: Path,
    root: Path,
    *,
    dev_shas: set[str] | None = None,
) -> list[str]:
    reasons: list[str] = []
    rows = read_commit_map(chain_dir)
    if not rows:
        reasons.append("FAIL: commit-map.csv has no data rows")
        return reasons

    map_shas = {row.get("sha", "").strip() for row in rows}
    map_shas.discard("")

    if dev_shas is None:
        try:
            dev_shas = git_dev_commit_shas(root)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            reasons.append(f"FAIL: could not read dev commit set via git: {exc}")
            return reasons

    in_dev_not_map = sorted(dev_shas - map_shas)
    if in_dev_not_map:
        preview = ", ".join(sha[:12] for sha in in_dev_not_map[:5])
        suffix = "..." if len(in_dev_not_map) > 5 else ""
        reasons.append(
            f"FAIL: D-12 end-state — SHAs in dev-but-not-map ({len(in_dev_not_map)}): {preview}{suffix}"
        )

    in_map_not_dev = sorted(map_shas - dev_shas)
    if in_map_not_dev:
        preview = ", ".join(sha[:12] for sha in in_map_not_dev[:5])
        suffix = "..." if len(in_map_not_dev) > 5 else ""
        reasons.append(
            f"FAIL: D-12 end-state — SHAs in map-but-not-dev ({len(in_map_not_dev)}): {preview}{suffix}"
        )

    if dev_shas is not None and not reasons:
        try:
            dev_tip = git_dev_tip(root)
        except (subprocess.CalledProcessError, FileNotFoundError):
            dev_tip = max(dev_shas) if dev_shas else ""
        if dev_tip and dev_tip not in map_shas:
            reasons.append(
                f"FAIL: D-12 end-state — dev tip {dev_tip[:12]} not in commit-map assignment union"
            )

    return reasons


def check_map(root: Path, *, dev_shas: set[str] | None = None) -> list[str]:
    """Validate commit-map.csv completeness against dev history."""
    reasons: list[str] = []
    paths = deliverable_paths(root)
    chain_dir = paths["chain_dir"]
    map_path = chain_dir / "commit-map.csv"

    if not map_path.is_file():
        reasons.append(f"FAIL: missing deliverable file: {map_path.relative_to(root)}")
        return reasons

    first_line = map_path.read_text(encoding="utf-8").splitlines()[0].strip()
    if first_line != COMMIT_MAP_HEADER:
        reasons.append(
            f"FAIL: commit-map.csv header mismatch\n  expected: {COMMIT_MAP_HEADER}\n  got:      {first_line}"
        )

    rows = read_commit_map(chain_dir)
    if not rows:
        reasons.append("FAIL: commit-map.csv has no data rows")
        return reasons

    map_shas = [row.get("sha", "").strip() for row in rows]
    map_sha_set = set(map_shas)
    dup_shas = sorted({sha for sha in map_shas if map_shas.count(sha) > 1})
    if dup_shas:
        preview = ", ".join(sha[:12] for sha in dup_shas[:5])
        suffix = "..." if len(dup_shas) > 5 else ""
        reasons.append(f"FAIL: duplicated SHAs in commit-map.csv ({len(dup_shas)}): {preview}{suffix}")

    if dev_shas is None:
        try:
            dev_shas = git_dev_commit_shas(root)
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            reasons.append(f"FAIL: could not read dev commit set via git: {exc}")
            return reasons

    in_dev_not_map = sorted(dev_shas - map_sha_set)
    if in_dev_not_map:
        preview = ", ".join(sha[:12] for sha in in_dev_not_map[:5])
        suffix = "..." if len(in_dev_not_map) > 5 else ""
        reasons.append(
            f"FAIL: SHAs in dev-but-not-map ({len(in_dev_not_map)}): {preview}{suffix}"
        )

    in_map_not_dev = sorted(map_sha_set - dev_shas)
    if in_map_not_dev:
        preview = ", ".join(sha[:12] for sha in in_map_not_dev[:5])
        suffix = "..." if len(in_map_not_dev) > 5 else ""
        reasons.append(
            f"FAIL: SHAs in map-but-not-dev ({len(in_map_not_dev)}): {preview}{suffix}"
        )

    manifest_prs = known_manifest_pr_numbers(chain_dir)
    for row in rows:
        raw_pr = row.get("pr_number", "").strip()
        if not raw_pr.isdigit():
            reasons.append(f"FAIL: invalid pr_number in map row: {raw_pr!r}")
            continue
        pr_number = int(raw_pr)
        if pr_number not in manifest_prs:
            reasons.append(f"FAIL: pr_number #{pr_number} in map has no manifest file")

        ambiguous = row.get("ambiguous", "").strip().lower() == "true"
        candidates = row.get("candidates", "").strip()
        if ambiguous and not candidates:
            sha = row.get("sha", "")[:12]
            reasons.append(f"FAIL: ambiguous row {sha} missing candidates")

    return reasons


def check_stack(
    root: Path,
    *,
    dev_shas: set[str] | None = None,
    index_by_pr: dict[int, dict[str, str]] | None = None,
) -> list[str]:
    reasons = check_structure(root)
    if reasons:
        return reasons

    paths = deliverable_paths(root)
    chain_dir = paths["chain_dir"]
    index_path = paths["index"]

    reasons.extend(check_manifest_drift_sections(chain_dir))

    if index_by_pr is None:
        _, index_by_pr = index_rows_by_pr(index_path)

    reasons.extend(check_stack_linearity(index_by_pr))

    map_reasons = check_map(root, dev_shas=dev_shas)
    reasons.extend(map_reasons)

    if not map_reasons:
        reasons.extend(check_end_state(chain_dir, root, dev_shas=dev_shas))

    return reasons


def parse_audit_out_of_scope_count(manifest_text: str) -> int | None:
    section_match = re.search(
        rf"## {SIMULATED_DIFF_AUDIT_SECTION}\n\n(.*?)(?=\n## |\Z)",
        manifest_text,
        re.DOTALL,
    )
    if section_match is None:
        return None
    section = section_match.group(1)
    count_match = re.search(r"- out_of_scope \(HIGH\): (\d+)", section)
    if count_match is None:
        return None
    return int(count_match.group(1))


def drift_flags_catalogue_out_of_scope(drift_flags: str, expected_count: int) -> bool:
    match = re.search(r"out_of_scope=(\d+)", drift_flags)
    if match is None:
        return False
    return int(match.group(1)) == expected_count


def check_audit(root: Path, *, index_by_pr: dict[int, dict[str, str]] | None = None) -> list[str]:
    reasons: list[str] = []
    paths = deliverable_paths(root)
    chain_dir = paths["chain_dir"]
    index_path = paths["index"]
    audit_report = chain_dir / "audit-report.md"

    if not audit_report.is_file():
        reasons.append(f"FAIL: missing deliverable file: {audit_report.relative_to(root)}")
        return reasons

    if index_by_pr is None:
        _, index_by_pr = index_rows_by_pr(index_path)

    for pr in CANONICAL_OPEN_PR_ORDER:
        manifest_path = manifest_for_pr(chain_dir, pr)
        if manifest_path is None:
            reasons.append(f"FAIL: missing manifest for open PR #{pr}")
            continue

        text = manifest_path.read_text(encoding="utf-8")
        if f"## {SIMULATED_DIFF_AUDIT_SECTION}" not in text:
            reasons.append(
                f"FAIL: {manifest_path.relative_to(root)} missing ## {SIMULATED_DIFF_AUDIT_SECTION}"
            )
            continue

        out_count = parse_audit_out_of_scope_count(text)
        if out_count is None:
            reasons.append(
                f"FAIL: {manifest_path.relative_to(root)} audit section missing out_of_scope count"
            )
            continue

        row = index_by_pr.get(pr)
        if row is None:
            reasons.append(f"FAIL: open PR #{pr} missing from pr-chain-index.csv")
            continue

        drift_flags = row.get("drift_flags", "")
        if not drift_flags_catalogue_out_of_scope(drift_flags, out_count):
            reasons.append(
                f"FAIL: PR #{pr} out_of_scope={out_count} not catalogued in index drift_flags "
                f"(got {drift_flags!r})"
            )

    return reasons


def run_mode(mode: str) -> int:
    root = repo_root()
    if mode == "structure":
        reasons = check_structure(root)
    elif mode == "inventory":
        reasons = check_inventory(root)
    elif mode == "map":
        reasons = check_map(root)
    elif mode == "stack":
        reasons = check_stack(root)
    elif mode == "audit":
        reasons = check_audit(root)
    elif mode == "all":
        reasons = check_inventory(root)
        if not reasons:
            reasons = check_map(root)
        if not reasons:
            reasons = check_stack(root)
        if not reasons:
            reasons = check_audit(root)
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
        choices=["structure", "inventory", "map", "stack", "audit", "all"],
        help="validation mode",
    )
    args = parser.parse_args()
    sys.exit(run_mode(args.mode))


if __name__ == "__main__":
    main()
