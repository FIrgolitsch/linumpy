#!/usr/bin/env python3
"""Generate Phase 4 PR chain inventory from live GitHub headers and git metadata."""

from __future__ import annotations

import argparse
import csv
import fnmatch
import importlib.util
import json
import re
import subprocess
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "validate_pr_chain", _SCRIPTS_DIR / "validate_pr_chain.py"
)
assert _spec is not None and _spec.loader is not None
_validate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_validate)

CANONICAL_HEADER = _validate.CANONICAL_HEADER
CANONICAL_OPEN_PR_ORDER = _validate.CANONICAL_OPEN_PR_ORDER
PASS_THROUGH_PRS = _validate.PASS_THROUGH_PRS
MERGED_FOUNDATION_PRS = _validate.MERGED_FOUNDATION_PRS
repo_root = _validate.repo_root

ORDER_RE = re.compile(
    r"\*\*Stacked PR \d+/\d+ — review order:\*\* (.+?)(?:\n|$)",
    re.MULTILINE,
)
BOLD_PR_RE = re.compile(r"#(\d+)")
SCOPE_HEADING_RE = re.compile(r"^## (.+)$", re.MULTILINE)

# Threat model T-04-02: allowlisted remote refs only.
ALLOWED_GIT_REFS = re.compile(
    r"^(origin/pr-[a-z0-9-]+|origin/sphinx-config|main|dev)$"
)
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

FOUNDATION_BRANCHES: dict[int, tuple[str, str]] = {
    115: ("pr-a-build-tooling", "main"),
    97: ("pr-c-utility-preprocessing", "pr-a-build-tooling"),
}

FULL_CANONICAL_ORDER: tuple[int, ...] = (115, 97) + CANONICAL_OPEN_PR_ORDER


def run_cmd(args: list[str], *, cwd: Path | None = None) -> str:
    """Run a subprocess with explicit args (never shell=True)."""
    result = subprocess.run(
        args,
        check=True,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return result.stdout.strip()


def assert_allowed_ref(ref: str) -> None:
    if GIT_SHA_RE.match(ref):
        return
    if not ALLOWED_GIT_REFS.match(ref):
        raise ValueError(f"ref not allowlisted: {ref!r}")


def gh_json(args: list[str]) -> Any:
    output = run_cmd(["gh", *args])
    return json.loads(output)


def parse_header_order(body: str) -> list[int]:
    match = ORDER_RE.search(body)
    if not match:
        raise ValueError("missing stacked PR header blockquote")
    return [int(x) for x in BOLD_PR_RE.findall(match.group(1))]


def extract_scope_heading(body: str) -> str:
    for line in body.splitlines():
        m = SCOPE_HEADING_RE.match(line)
        if m:
            return m.group(1).strip()
    return "(no scope heading found)"


def slug_for_pr(pr_number: int, head_branch: str, title: str) -> str:
    """Derive manifest filename slug from branch topic, falling back to title."""
    branch = head_branch.removeprefix("pr-")
    if branch and branch != head_branch:
        topic = branch
    else:
        topic = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]
    return f"{pr_number}-{topic}"


def intended_base_for(pr_number: int, pr_by_number: dict[int, dict[str, Any]]) -> str:
    idx = FULL_CANONICAL_ORDER.index(pr_number)
    if idx == 0:
        return "main"
    predecessor = FULL_CANONICAL_ORDER[idx - 1]
    if predecessor in MERGED_FOUNDATION_PRS:
        return FOUNDATION_BRANCHES[predecessor][0]
    pred = pr_by_number.get(predecessor)
    if pred is None:
        raise ValueError(f"predecessor PR #{predecessor} not in open PR set")
    return pred["headRefName"]


def merge_position(pr_number: int) -> int:
    return FULL_CANONICAL_ORDER.index(pr_number)


def git_rev_parse(ref: str, root: Path) -> str:
    assert_allowed_ref(ref)
    return run_cmd(["git", "rev-parse", ref], cwd=root)


def git_staleness_count(head_branch: str, root: Path) -> int:
    head_ref = f"origin/{head_branch}"
    assert_allowed_ref(head_ref)
    assert_allowed_ref("dev")
    output = run_cmd(["git", "rev-list", "--count", f"{head_ref}..dev"], cwd=root)
    return int(output)


def git_commit_count_since_base(head_branch: str, base_branch: str, root: Path) -> int:
    head_ref = f"origin/{head_branch}"
    base_ref = base_branch if base_branch in {"main", "dev"} else f"origin/{base_branch}"
    assert_allowed_ref(head_ref)
    assert_allowed_ref(base_ref)
    output = run_cmd(["git", "rev-list", "--count", f"{base_ref}..{head_ref}"], cwd=root)
    return int(output)


TITLE_SCOPE_STOPWORDS = frozenset(
    {"feat", "fix", "with", "from", "the", "and", "for", "docs", "test", "refactor"}
)


def extract_title_keywords(title: str) -> set[str]:
    words = re.findall(r"[a-z0-9]{4,}", title.lower())
    return {word for word in words if word not in TITLE_SCOPE_STOPWORDS}


def detect_title_scope_mismatch(title: str, scope: str) -> bool:
    keywords = extract_title_keywords(title)
    if not keywords:
        return False
    scope_lower = scope.lower()
    return not any(keyword in scope_lower for keyword in keywords)


def build_drift_flags(
    *,
    recorded_base: str,
    intended_base: str,
    staleness: int,
    commit_count: int,
    title: str,
    scope: str,
) -> tuple[list[str], str, str | None]:
    bullets: list[str] = []
    csv_parts: list[str] = []
    base_correction: str | None = None

    if recorded_base != intended_base:
        bullets.append(
            f"- base_mismatch (HIGH): recorded `{recorded_base}`, intended `{intended_base}`"
        )
        csv_parts.append(f"base_mismatch(recorded={recorded_base},intended={intended_base})")
        base_correction = (
            f"Phase 5: retarget base from `{recorded_base}` to `{intended_base}` "
            f"(documentation only — physical retarget deferred per D-09/D-18)"
        )

    bullets.append(f"- staleness (MEDIUM): {staleness} commits on dev ahead of branch tip")
    csv_parts.append(f"staleness={staleness}")
    bullets.append(f"- branch_commit_count: {commit_count} commits since recorded base")

    if detect_title_scope_mismatch(title, scope):
        bullets.append("- title_scope_mismatch (LOW): title keywords not found in manifest scope")
        csv_parts.append("title_scope_mismatch")

    return bullets, ";".join(csv_parts), base_correction


def manifest_for_pr(chain_dir: Path, pr_number: int) -> Path | None:
    matches = sorted(chain_dir.glob(f"{pr_number}-*.md"))
    return matches[0] if matches else None


def update_manifest_drift(
    manifest_path: Path,
    *,
    intended_base: str,
    drift_bullets: list[str],
    base_correction: str | None,
) -> None:
    text = manifest_path.read_text(encoding="utf-8")
    intended_block = f"## Intended base\n\n{intended_base}\n\n"
    drift_block = "## Drift flags\n\n" + "\n".join(drift_bullets) + "\n\n"

    if "## Intended base" in text:
        text = re.sub(
            r"## Intended base\n\n.*?(?=\n## )",
            intended_block.rstrip() + "\n\n",
            text,
            count=1,
            flags=re.DOTALL,
        )
    else:
        text = re.sub(
            r"(## Recorded base\n\n.*?\n\n)(## Drift flags\n\n)",
            rf"\1{intended_block}\2",
            text,
            count=1,
            flags=re.DOTALL,
        )

    text = re.sub(
        r"## Drift flags\n\n.*?(?=\n## )",
        drift_block.rstrip() + "\n\n",
        text,
        count=1,
        flags=re.DOTALL,
    )

    text = re.sub(r"\n## Base correction plan\n\n.*?(?=\n## Phase 5)", "", text, count=1, flags=re.DOTALL)
    if base_correction:
        text = re.sub(
            r"(## Drift flags\n\n.*?\n\n)(## Phase 5 verification checklist)",
            rf"\1## Base correction plan\n\n{base_correction}\n\n\2",
            text,
            count=1,
            flags=re.DOTALL,
        )

    manifest_path.write_text(text, encoding="utf-8")


def write_manifest(
    path: Path,
    *,
    pr_number: int,
    title: str,
    head_branch: str,
    base_ref: str,
    scope: str,
    merge_pos: int,
    commit_count: int,
    staleness: int,
    intended_base: str | None = None,
    drift_bullets: list[str] | None = None,
    base_correction: str | None = None,
) -> None:
    if intended_base is None or drift_bullets is None:
        drift_bullets = drift_bullets or [
            f"- staleness (MEDIUM): {staleness} commits on dev ahead of `origin/{head_branch}`",
            f"- branch_commit_count: {commit_count} commits since recorded base",
        ]
        intended_base = intended_base or base_ref

    correction_section = ""
    if base_correction:
        correction_section = f"\n## Base correction plan\n\n{base_correction}\n"

    text = f"""# PR #{pr_number} — {title}

**Branch:** `{head_branch}`  
**Merge position:** {merge_pos + 1}/{len(FULL_CANONICAL_ORDER)} (0-indexed {merge_pos})

## Intended scope

{scope}

## Assigned commits

<!-- stub: filled in 04-02 -->

## File ownership

<!-- stub: filled in 04-02 -->

## Recorded base

{base_ref}

## Intended base

{intended_base}

## Drift flags

{chr(10).join(drift_bullets)}
{correction_section}
## Phase 5 verification checklist

- [ ] Branch rebased onto correct intended base
- [ ] Assigned commits match manifest scope
- [ ] Simulated diff audit clean
"""
    path.write_text(text, encoding="utf-8")


def fetch_pr(pr_number: int) -> dict[str, Any]:
    data = gh_json(
        [
            "pr",
            "view",
            str(pr_number),
            "--json",
            "number,title,headRefName,baseRefName,body,state",
        ]
    )
    if not isinstance(data, dict):
        raise TypeError(f"gh pr view {pr_number} returned unexpected JSON shape")
    return data


def fetch_open_prs() -> list[dict[str, Any]]:
    data = gh_json(
        [
            "pr",
            "list",
            "--state",
            "open",
            "--limit",
            "30",
            "--json",
            "number,title,headRefName,baseRefName,body,state",
        ]
    )
    if not isinstance(data, list):
        raise TypeError("gh pr list returned unexpected JSON shape")
    return data


def fetch_foundation_pr(pr_number: int) -> dict[str, Any]:
    data = gh_json(
        [
            "pr",
            "view",
            str(pr_number),
            "--json",
            "number,title,headRefName,baseRefName,state,mergedAt",
        ]
    )
    if not isinstance(data, dict):
        raise TypeError(f"gh pr view {pr_number} returned unexpected JSON shape")
    return data


def validate_header_consensus(open_prs: list[dict[str, Any]]) -> list[int]:
    open_numbers = {p["number"] for p in open_prs}
    expected_open = set(CANONICAL_OPEN_PR_ORDER)
    missing = expected_open - open_numbers
    extra = open_numbers - expected_open
    non_pass_missing = missing - PASS_THROUGH_PRS
    if non_pass_missing:
        raise SystemExit(
            f"BLOCKING: missing open PRs (excluding pass-through): {sorted(non_pass_missing)}"
        )
    if extra:
        raise SystemExit(f"BLOCKING: unexpected open PRs: {sorted(extra)}")

    expected_open_count = len(CANONICAL_OPEN_PR_ORDER) - len(missing & PASS_THROUGH_PRS)
    if len(open_prs) != expected_open_count:
        numbers = sorted(p["number"] for p in open_prs)
        raise SystemExit(
            f"BLOCKING: expected {expected_open_count} open PRs, got {len(open_prs)}: {numbers}"
        )

    canonical: list[int] | None = None
    for pr in open_prs:
        number = pr["number"]
        try:
            order = parse_header_order(pr.get("body") or "")
        except ValueError as exc:
            raise SystemExit(f"BLOCKING: PR #{number} {exc}") from exc

        if canonical is None:
            canonical = order
        elif order != canonical:
            raise SystemExit(
                f"BLOCKING: header sequence inconsistency — PR #{number} order differs from canonical.\n"
                f"  canonical: {' → '.join(f'#{n}' for n in canonical)}\n"
                f"  PR #{number}: {' → '.join(f'#{n}' for n in order)}"
            )

    assert canonical is not None
    return canonical


def hydrate_pass_through_prs(
    pr_by_number: dict[int, dict[str, Any]],
) -> None:
    """Load CLOSED pass-through PR metadata from GitHub for inventory rows."""
    for pr_number in CANONICAL_OPEN_PR_ORDER:
        if pr_number in pr_by_number or pr_number not in PASS_THROUGH_PRS:
            continue
        pr_by_number[pr_number] = fetch_pr(pr_number)


def generate_inventory(root: Path) -> None:
    planning = root / ".planning"
    chain_dir = planning / "pr-chain"
    chain_dir.mkdir(parents=True, exist_ok=True)

    open_prs = fetch_open_prs()
    validate_header_consensus(open_prs)
    pr_by_number = {pr["number"]: pr for pr in open_prs}
    hydrate_pass_through_prs(pr_by_number)

    rows: list[dict[str, str]] = []

    for pr_number in MERGED_FOUNDATION_PRS:
        foundation = fetch_foundation_pr(pr_number)
        branch, intended = FOUNDATION_BRANCHES[pr_number]
        rows.append(
            {
                "pr_number": str(pr_number),
                "branch": branch,
                "intended_base": intended,
                "merge_position": str(merge_position(pr_number)),
                "status": "merged",
                "commit_count": "0",
                "drift_flags": "",
            }
        )

    for pr_number in CANONICAL_OPEN_PR_ORDER:
        pr = pr_by_number[pr_number]
        head = pr["headRefName"]
        base = pr["baseRefName"]
        title = pr["title"]
        body = pr.get("body") or ""
        scope = extract_scope_heading(body)
        intended = intended_base_for(pr_number, pr_by_number)
        pos = merge_position(pr_number)

        try:
            commit_count = git_commit_count_since_base(head, base, root)
            staleness = git_staleness_count(head, root)
            git_rev_parse(f"origin/{head}", root)
        except (subprocess.CalledProcessError, ValueError) as exc:
            raise SystemExit(f"git metadata failed for PR #{pr_number} ({head}): {exc}") from exc

        drift_bullets, drift, base_correction = build_drift_flags(
            recorded_base=base,
            intended_base=intended,
            staleness=staleness,
            commit_count=commit_count,
            title=title,
            scope=scope,
        )

        if pr_number in PASS_THROUGH_PRS and pr.get("state") == "CLOSED":
            drift_bullets = list(drift_bullets)
            drift_bullets.append(
                "- pass_through (INFO): GitHub PR CLOSED — zero-diff slot; branch equals predecessor"
            )
            drift = (drift + ";pass_through=closed").lstrip(";")

        slug = slug_for_pr(pr_number, head, title)
        manifest_path = chain_dir / f"{slug}.md"
        if not str(manifest_path.resolve()).startswith(str(chain_dir.resolve())):
            raise SystemExit(f"manifest path escapes .planning/pr-chain: {manifest_path}")

        write_manifest(
            manifest_path,
            pr_number=pr_number,
            title=title,
            head_branch=head,
            base_ref=base,
            scope=scope,
            merge_pos=pos,
            commit_count=commit_count,
            staleness=staleness,
            intended_base=intended,
            drift_bullets=drift_bullets,
            base_correction=base_correction,
        )

        row_status = "pass-through" if pr_number in PASS_THROUGH_PRS and pr.get("state") == "CLOSED" else "open"
        rows.append(
            {
                "pr_number": str(pr_number),
                "branch": head,
                "intended_base": intended,
                "merge_position": str(pos),
                "status": row_status,
                "commit_count": str(commit_count),
                "drift_flags": drift,
            }
        )

    index_path = planning / "pr-chain-index.csv"
    if not str(index_path.resolve()).startswith(str(planning.resolve())):
        raise SystemExit(f"index path escapes .planning: {index_path}")

    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_HEADER.split(","))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {len(CANONICAL_OPEN_PR_ORDER)} manifests to {chain_dir.relative_to(root)}/")
    print(f"Wrote rollup CSV to {index_path.relative_to(root)}")


COMMIT_MAP_HEADER = "sha,pr_number,is_merge,ambiguous,candidates,assignment_reason"
EXPLICIT_PR_RE = re.compile(r"#(\d{2,3})\b|\(#(\d{2,3})\)")
GSD_PHASE_RE = re.compile(r"^(?:feat|test|docs|refactor|fix)\((\d+)-(\d+)\):")


@dataclass
class MapRow:
    sha: str
    pr_number: int
    is_merge: bool = False
    ambiguous: bool = False
    candidates: list[int] = field(default_factory=list)
    assignment_reason: str = ""


def git_merge_base(root: Path, ref_a: str = "main", ref_b: str = "dev") -> str:
    assert_allowed_ref(ref_a)
    assert_allowed_ref(ref_b)
    return run_cmd(["git", "merge-base", ref_a, ref_b], cwd=root)


def git_rev_list_range(start: str, end: str, root: Path, *, reverse: bool = True) -> list[str]:
    assert_allowed_ref(start)
    assert_allowed_ref(end)
    args = ["git", "rev-list"]
    if reverse:
        args.append("--reverse")
    args.append(f"{start}..{end}")
    output = run_cmd(args, cwd=root)
    return [line for line in output.splitlines() if line]


def git_commit_fields(root: Path, merge_base: str) -> list[tuple[str, str, str, str]]:
    """Return (sha, subject, author, parents) for each commit in merge_base..dev."""
    assert_allowed_ref("dev")
    fmt = "%H%x09%s%x09%an%x09%P"
    output = run_cmd(
        ["git", "log", "--reverse", f"--format={fmt}", f"{merge_base}..dev"],
        cwd=root,
    )
    rows: list[tuple[str, str, str, str]] = []
    for line in output.splitlines():
        parts = line.split("\t", 3)
        if len(parts) == 4:
            rows.append((parts[0], parts[1], parts[2], parts[3]))
    return rows


def git_changed_files(sha: str, root: Path) -> list[str]:
    output = run_cmd(["git", "show", "--name-only", "--pretty=format:", sha], cwd=root)
    return [line.strip() for line in output.splitlines() if line.strip()]


def load_commit_splits(chain_dir: Path) -> dict[str, dict[int, list[str]]]:
    """Load per-commit file splits (mega-commit collateral → upstream PRs)."""
    path = chain_dir / "commit-splits.json"
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    splits: dict[str, dict[int, list[str]]] = {}
    for sha, pr_map in raw.items():
        splits[sha] = {int(pr): list(files) for pr, files in pr_map.items()}
    return splits


def changed_files_for_pr_commit(
    sha: str,
    pr_number: int,
    splits: dict[str, dict[int, list[str]]],
    root: Path,
) -> set[str]:
    """Files attributed to *pr_number* from commit *sha* (split-aware)."""
    all_files = set(git_changed_files(sha, root))
    pr_split = splits.get(sha)
    if pr_split is None:
        return all_files
    scoped = set(pr_split.get(pr_number, []))
    return all_files & scoped


def load_scope_rules(chain_dir: Path) -> dict[str, Any]:
    path = chain_dir / "scope-rules.json"
    if not path.is_file():
        raise SystemExit(f"BLOCKING: missing scope rules: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_branch_rows(root: Path) -> dict[int, dict[str, str]]:
    index_path = root / ".planning" / "pr-chain-index.csv"
    _, rows = _validate.read_csv_rows(index_path)
    by_pr: dict[int, dict[str, str]] = {}
    for row in rows:
        raw = row.get("pr_number", "").strip()
        if raw.isdigit():
            by_pr[int(raw)] = row
    return by_pr


def load_consistency_subsystems(root: Path) -> dict[str, set[str]]:
    path = root / ".planning" / "consistency-index.csv"
    if not path.is_file():
        return {}
    subsystems: dict[str, set[str]] = defaultdict(set)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_path = row.get("file", "").strip()
            subsystem = row.get("subsystem", "").strip()
            if file_path and subsystem:
                subsystems[file_path].add(subsystem.split("/")[0])
    return subsystems


def pr_subsystem_hints(pr_number: int) -> set[str]:
    hints: dict[int, set[str]] = {
        98: {"mosaic", "stitching"},
        99: {"gpu"},
        100: {"analysis", "utils"},
        101: {"imaging"},
        108: {"workflows"},
        106: {"slice_config"},
        107: {"shifts"},
        87: {"stitching"},
        116: {"intensity", "illumination"},
        110: {"workflows"},
        111: {"scripts"},
        40: {"docs"},
        112: {"gpu"},
        130: {"gpu"},
        131: {"microscope", "galvo"},
        118: {"scripts"},
        132: {"workflows"},
        121: {"imaging"},
        122: {"stacking"},
        123: {"attenuation"},
        124: {"analysis", "io"},
        133: {"illumination", "intensity"},
        128: {"illumination", "intensity"},
    }
    return hints.get(pr_number, set())


def score_path_rules(files: list[str], rules: list[dict[str, Any]]) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    for file_path in files:
        for rule in rules:
            glob = rule["glob"]
            if fnmatch.fnmatch(file_path, glob) or fnmatch.fnmatch(file_path, glob.rstrip("/")):
                scores[int(rule["pr_number"])] += float(rule.get("weight", 1))
    return dict(scores)


def score_message_rules(subject: str, rules: list[dict[str, Any]]) -> dict[int, float]:
    subject_lower = subject.lower()
    scores: dict[int, float] = defaultdict(float)
    for rule in rules:
        keywords = rule.get("keywords", [])
        if any(kw.lower() in subject_lower for kw in keywords):
            scores[int(rule["pr_number"])] += float(rule.get("weight", 1))
    return dict(scores)


def score_explicit_pr_refs(subject: str) -> dict[int, float]:
    scores: dict[int, float] = defaultdict(float)
    for match in EXPLICIT_PR_RE.finditer(subject):
        for group in match.groups():
            if group and group.isdigit():
                scores[int(group)] += 20.0
    return dict(scores)


def score_gsd_fold(subject: str, fold_rules: dict[str, Any]) -> tuple[int | None, str]:
    match = GSD_PHASE_RE.match(subject)
    if not match:
        return None, ""
    phase = match.group(1).lstrip("0") or "0"
    phase_key = phase if phase in fold_rules.get("phase_to_pr", {}) else phase.lstrip("0")
    phase_map = fold_rules.get("phase_to_pr", {})
    pr = phase_map.get(phase_key) or phase_map.get(phase) or fold_rules.get("default_pr")
    if pr is None:
        return None, ""
    return int(pr), f"gsd_fold:phase-{match.group(1)}-{match.group(2)}"


def score_subsystem_join(
    files: list[str],
    consistency: dict[str, set[str]],
    pr_number: int,
) -> float:
    hints = pr_subsystem_hints(pr_number)
    if not hints:
        return 0.0
    score = 0.0
    for file_path in files:
        for subsystem in consistency.get(file_path, set()):
            if subsystem in hints or any(h in subsystem for h in hints):
                score += 2.0
    return score


def merge_diff_files(sha: str, root: Path) -> list[str]:
    """Files changed by a merge commit (first-parent diff vs second parent)."""
    parents = run_cmd(["git", "rev-list", "--parents", "-n", "1", sha], cwd=root).split()
    if len(parents) < 3:
        return git_changed_files(sha, root)
    first_parent, second_parent = parents[1], parents[2]
    output = run_cmd(
        ["git", "diff", "--name-only", first_parent, second_parent],
        cwd=root,
    )
    return [line.strip() for line in output.splitlines() if line.strip()]


def pick_assignment(
    scores: dict[int, float],
    *,
    threshold_ratio: float,
    fallback_pr: int,
    reason_prefix: str,
) -> tuple[int, bool, list[int], str]:
    if not scores:
        return fallback_pr, False, [], f"{reason_prefix}:orphan_bucket"

    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    best_pr, best_score = ranked[0]
    candidates = [pr for pr, score in ranked if score >= best_score * (1 - threshold_ratio)][:5]

    ambiguous = len(candidates) > 1 and ranked[1][1] >= best_score * (1 - threshold_ratio)
    reason = f"{reason_prefix}:{best_pr}(score={best_score:.1f})"
    if ambiguous:
        reason += f";ambiguous_candidates={','.join(str(c) for c in candidates)}"
    return best_pr, ambiguous, candidates, reason


def bootstrap_branch_assignments(root: Path, branch_rows: dict[int, dict[str, str]]) -> dict[str, MapRow]:
    assignments: dict[str, MapRow] = {}
    merge_base = git_merge_base(root)

    predecessor_ref = merge_base
    for pr_number in CANONICAL_OPEN_PR_ORDER:
        row = branch_rows.get(pr_number)
        if row is None:
            raise SystemExit(f"BLOCKING: PR #{pr_number} missing from pr-chain-index.csv")
        head_branch = row["branch"]
        head_ref = f"origin/{head_branch}"
        assert_allowed_ref(head_ref)

        shas = git_rev_list_range(predecessor_ref, head_ref, root)
        for sha in shas:
            if sha not in assignments:
                parents = run_cmd(["git", "rev-list", "--parents", "-n", "1", sha], cwd=root).split()
                is_merge = len(parents) > 2
                assignments[sha] = MapRow(
                    sha=sha,
                    pr_number=pr_number,
                    is_merge=is_merge,
                    assignment_reason=f"branch_bootstrap:#{pr_number}",
                )
        predecessor_ref = head_ref

    return assignments


def scope_match_commit(
    sha: str,
    subject: str,
    files: list[str],
    rules: dict[str, Any],
    consistency: dict[str, set[str]],
    *,
    neighbor_pr: int | None,
    fallback_pr: int,
) -> MapRow:
    threshold = float(rules.get("ambiguity_threshold_ratio", 0.1))
    combined: dict[int, float] = defaultdict(float)

    for pr, score in score_explicit_pr_refs(subject).items():
        combined[pr] += score
    for pr, score in score_path_rules(files, rules.get("path_rules", [])).items():
        combined[pr] += score
    for pr, score in score_message_rules(subject, rules.get("message_rules", [])).items():
        combined[pr] += score

    gsd_pr, gsd_reason = score_gsd_fold(subject, rules.get("gsd_fold_rules", {}))
    if gsd_pr is not None:
        combined[gsd_pr] += 12.0

    for pr in set(combined) | set(CANONICAL_OPEN_PR_ORDER):
        combined[pr] += score_subsystem_join(files, consistency, pr) * 0.5

    if neighbor_pr is not None:
        combined[neighbor_pr] += 1.5

    pr_number, ambiguous, candidates, reason = pick_assignment(
        dict(combined),
        threshold_ratio=threshold,
        fallback_pr=fallback_pr,
        reason_prefix="scope_match",
    )
    if gsd_reason and gsd_pr == pr_number:
        reason = f"{gsd_reason};{reason}"

    return MapRow(
        sha=sha,
        pr_number=pr_number,
        ambiguous=ambiguous,
        candidates=candidates,
        assignment_reason=reason,
    )


def detect_linum_basic_batch(
    orphan_shas: list[str],
    assignments: dict[str, MapRow],
    rules: dict[str, Any],
    root: Path,
    subjects: dict[str, str],
) -> tuple[int | None, list[str]]:
    slot_cfg = rules.get("new_pr_slot", {})
    min_commits = int(slot_cfg.get("min_contiguous_commits", 5))
    path_globs = slot_cfg.get("path_globs", [])
    msg_keywords = [k.lower() for k in slot_cfg.get("message_keywords", [])]
    provisional = int(slot_cfg.get("provisional_number", 128))

    batch: list[str] = []
    for sha in orphan_shas:
        if sha in assignments:
            continue
        subject = subjects.get(sha, "").lower()
        files = git_changed_files(sha, root)
        path_hit = any(
            fnmatch.fnmatch(f, glob) for f in files for glob in path_globs
        )
        msg_hit = any(kw in subject for kw in msg_keywords)
        if path_hit or msg_hit:
            batch.append(sha)

    if len(batch) >= min_commits:
        return provisional, batch
    return None, []


def write_draft_new_pr_manifest(chain_dir: Path, rules: dict[str, Any]) -> Path:
    slot_cfg = rules.get("new_pr_slot", {})
    pr_number = int(slot_cfg.get("provisional_number", 128))
    slug = slot_cfg.get("slug", "new-slot")
    title = slot_cfg.get("title", "Draft new PR slot")
    path = chain_dir / f"{pr_number}-{slug}.md"
    text = f"""# PR #{pr_number} — {title}

**Branch:** `(draft — no branch yet)`  
**Merge position:** provisional (append before #125 — **requires human review**)

## Intended scope

Draft slot for coherent linum-basic illumination backend work extracted from dev orphan bucket.
Flagged for review per D-14 before Phase 5 branch rebuild.

## Assigned commits

<!-- filled by --map -->

## File ownership

<!-- filled by --map -->

## Recorded base

pr-v-psf-focal

## Intended base

pr-v-psf-focal

## Drift flags

- draft_new_pr_slot: true
- human_review_required: true

## Phase 5 verification checklist

- [ ] Confirm insert position in stack (default: before #125)
- [ ] Create physical branch and open PR
- [ ] Assigned commits match manifest scope
- [ ] Simulated diff audit clean
"""
    path.write_text(text, encoding="utf-8")
    return path


def write_commit_map_csv(path: Path, rows: list[MapRow]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COMMIT_MAP_HEADER.split(","))
        for row in rows:
            candidates = ";".join(str(c) for c in row.candidates) if row.candidates else ""
            writer.writerow(
                [
                    row.sha,
                    row.pr_number,
                    "true" if row.is_merge else "false",
                    "true" if row.ambiguous else "false",
                    candidates,
                    row.assignment_reason,
                ]
            )


def backfill_manifests(
    chain_dir: Path,
    rows: list[MapRow],
    root: Path,
) -> None:
    by_pr: dict[int, list[MapRow]] = defaultdict(list)
    files_by_pr: dict[int, set[str]] = defaultdict(set)

    for row in rows:
        by_pr[row.pr_number].append(row)
        for file_path in git_changed_files(row.sha, root):
            files_by_pr[row.pr_number].add(file_path)

    for manifest_path in sorted(chain_dir.glob("*-*.md")):
        pr_number = _validate.pr_number_from_manifest(manifest_path)
        if pr_number is None:
            continue
        assigned = by_pr.get(pr_number, [])
        text = manifest_path.read_text(encoding="utf-8")

        commit_lines = []
        for row in sorted(assigned, key=lambda r: r.sha):
            amb = " (ambiguous)" if row.ambiguous else ""
            merge = " [merge]" if row.is_merge else ""
            commit_lines.append(
                f"- `{row.sha[:12]}`{merge}{amb} — {row.assignment_reason}"
            )
        commits_block = "\n".join(commit_lines) if commit_lines else "(none assigned)"

        file_lines = sorted(files_by_pr.get(pr_number, set()))
        files_block = "\n".join(f"- `{f}`" for f in file_lines) if file_lines else "(none)"

        text = re.sub(
            r"(## Assigned commits\n\n)(.*?)(\n\n## File ownership)",
            rf"\1{commits_block}\3",
            text,
            count=1,
            flags=re.DOTALL,
        )
        text = re.sub(
            r"(## File ownership\n\n)(.*?)(\n\n## Recorded base)",
            rf"\1{files_block}\3",
            text,
            count=1,
            flags=re.DOTALL,
        )
        manifest_path.write_text(text, encoding="utf-8")


def generate_commit_map(root: Path) -> None:
    planning = root / ".planning"
    chain_dir = planning / "pr-chain"
    chain_dir.mkdir(parents=True, exist_ok=True)

    rules = load_scope_rules(chain_dir)
    branch_rows = load_branch_rows(root)
    consistency = load_consistency_subsystems(root)
    fallback_pr = int(rules.get("orphan_bucket", 125))

    merge_base = git_merge_base(root)
    commit_fields = git_commit_fields(root, merge_base)
    subjects = {sha: subject for sha, subject, _, _ in commit_fields}

    assignments = bootstrap_branch_assignments(root, branch_rows)

    tip_ref = "origin/pr-u-fix-illum"
    assert_allowed_ref(tip_ref)
    orphan_shas = git_rev_list_range(tip_ref, "dev", root)

    new_pr_number, linum_basic_batch = detect_linum_basic_batch(
        orphan_shas, assignments, rules, root, subjects
    )
    if new_pr_number is not None:
        write_draft_new_pr_manifest(chain_dir, rules)
        for sha in linum_basic_batch:
            assignments[sha] = MapRow(
                sha=sha,
                pr_number=new_pr_number,
                assignment_reason="new_pr_slot:linum-basic-batch",
            )

    chronological_prs: list[int | None] = [None] * len(commit_fields)
    last_pr: int | None = None
    for idx, (sha, subject, _, parents) in enumerate(commit_fields):
        if sha in assignments:
            last_pr = assignments[sha].pr_number
            chronological_prs[idx] = last_pr
        else:
            chronological_prs[idx] = last_pr

    for idx, (sha, subject, _, parents) in enumerate(commit_fields):
        if sha in assignments:
            continue

        is_merge = len(parents.split()) > 1
        files = merge_diff_files(sha, root) if is_merge else git_changed_files(sha, root)
        neighbor = chronological_prs[idx - 1] if idx > 0 else None

        row = scope_match_commit(
            sha,
            subject,
            files,
            rules,
            consistency,
            neighbor_pr=neighbor,
            fallback_pr=fallback_pr,
        )
        row.is_merge = is_merge
        if is_merge:
            row.assignment_reason = f"merge_content;{row.assignment_reason}"
        assignments[sha] = row
        chronological_prs[idx] = row.pr_number

    ordered_rows: list[MapRow] = []
    for sha, _, _, _ in commit_fields:
        if sha not in assignments:
            raise SystemExit(f"BLOCKING: commit {sha} not assigned")
        ordered_rows.append(assignments[sha])

    map_path = chain_dir / "commit-map.csv"
    write_commit_map_csv(map_path, ordered_rows)
    backfill_manifests(chain_dir, ordered_rows, root)

    expected = len(commit_fields)
    print(f"Wrote commit map ({expected} rows) to {map_path.relative_to(root)}")
    ambiguous_count = sum(1 for r in ordered_rows if r.ambiguous)
    merge_count = sum(1 for r in ordered_rows if r.is_merge)
    orphan_assigned = sum(1 for sha in orphan_shas if assignments.get(sha) and assignments[sha].pr_number != fallback_pr)
    print(
        f"  merges={merge_count} ambiguous={ambiguous_count} "
        f"orphan_scope_matched={orphan_assigned}/{len(orphan_shas)}"
    )
    if new_pr_number is not None:
        print(f"  draft new PR slot #{new_pr_number} ({len(linum_basic_batch)} commits) — review required")


def extract_scope_from_manifest(text: str) -> str:
    match = re.search(r"## Intended scope\n\n(.*?)\n\n## ", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def generate_drift(root: Path) -> None:
    """Enrich manifests and index CSV with intended-base drift analysis (read-only git)."""
    planning = root / ".planning"
    chain_dir = planning / "pr-chain"
    index_path = planning / "pr-chain-index.csv"

    if not index_path.is_file():
        raise SystemExit(f"BLOCKING: missing {index_path.relative_to(root)} — run --inventory first")

    open_prs = fetch_open_prs()
    validate_header_consensus(open_prs)
    pr_by_number = {pr["number"]: pr for pr in open_prs}
    hydrate_pass_through_prs(pr_by_number)

    _, existing_rows = _validate.read_csv_rows(index_path)
    row_by_pr: dict[int, dict[str, str]] = {}
    for row in existing_rows:
        raw = row.get("pr_number", "").strip()
        if raw.isdigit():
            row_by_pr[int(raw)] = row

    updated_rows: list[dict[str, str]] = []
    base_mismatch_count = 0
    title_mismatch_count = 0

    for row in existing_rows:
        raw = row.get("pr_number", "").strip()
        if not raw.isdigit():
            updated_rows.append(row)
            continue
        pr_number = int(raw)
        if pr_number in MERGED_FOUNDATION_PRS:
            updated_rows.append(row)
            continue

        pr = pr_by_number.get(pr_number)
        if pr is None:
            raise SystemExit(f"BLOCKING: open PR #{pr_number} missing from GitHub")

        head = pr["headRefName"]
        base = pr["baseRefName"]
        title = pr["title"]
        intended = intended_base_for(pr_number, pr_by_number)

        manifest_path = manifest_for_pr(chain_dir, pr_number)
        if manifest_path is None:
            raise SystemExit(f"BLOCKING: no manifest for open PR #{pr_number}")

        manifest_text = manifest_path.read_text(encoding="utf-8")
        scope = extract_scope_from_manifest(manifest_text) or extract_scope_heading(pr.get("body") or "")

        try:
            commit_count = git_commit_count_since_base(head, base, root)
            staleness = git_staleness_count(head, root)
        except (subprocess.CalledProcessError, ValueError) as exc:
            raise SystemExit(f"git metadata failed for PR #{pr_number} ({head}): {exc}") from exc

        drift_bullets, drift_csv, base_correction = build_drift_flags(
            recorded_base=base,
            intended_base=intended,
            staleness=staleness,
            commit_count=commit_count,
            title=title,
            scope=scope,
        )
        if base_correction:
            base_mismatch_count += 1
        if any("title_scope_mismatch" in part for part in drift_csv.split(";")):
            title_mismatch_count += 1

        update_manifest_drift(
            manifest_path,
            intended_base=intended,
            drift_bullets=drift_bullets,
            base_correction=base_correction,
        )

        updated_rows.append(
            {
                "pr_number": str(pr_number),
                "branch": head,
                "intended_base": intended,
                "merge_position": row.get("merge_position", str(merge_position(pr_number))),
                "status": "open",
                "commit_count": str(commit_count),
                "drift_flags": drift_csv,
            }
        )

    foundation_rows = [
        row for row in existing_rows if row.get("pr_number", "").strip() in {"115", "97"}
    ]
    open_rows = sorted(
        [row for row in updated_rows if row.get("status") == "open"],
        key=lambda row: int(row["merge_position"]),
    )
    final_rows = foundation_rows + open_rows

    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_HEADER.split(","))
        writer.writeheader()
        for row in final_rows:
            writer.writerow(row)

    print(
        f"Updated drift for {len(open_rows)} open PR manifests; "
        f"base_mismatch={base_mismatch_count} title_scope_mismatch={title_mismatch_count}"
    )


AUDIT_HOTSPOT_PRS: frozenset[int] = frozenset({125, 116})
WORKTREE_PREFIX = "/tmp/pr-chain-audit-"


@dataclass
class AuditResult:
    pr_number: int
    commit_count: int
    diff_file_count: int
    out_of_scope: list[str]
    missing: list[str]
    method: str
    notes: list[str] = field(default_factory=list)
    cherry_pick_conflict: bool = False


def ref_to_sha(ref: str, root: Path) -> str:
    if GIT_SHA_RE.match(ref):
        return ref
    git_ref = ref
    if ref not in {"main", "dev"}:
        git_ref = f"origin/{ref}" if ref != "sphinx-config" else "origin/sphinx-config"
    assert_allowed_ref(git_ref)
    return git_rev_parse(git_ref, root)


def load_commit_map_rows(chain_dir: Path) -> list[dict[str, str]]:
    map_path = chain_dir / "commit-map.csv"
    if not map_path.is_file():
        raise SystemExit(f"BLOCKING: missing {map_path.name} — run --map first")
    with map_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def commits_by_pr_chronological(root: Path, map_rows: list[dict[str, str]]) -> dict[int, list[str]]:
    merge_base = git_merge_base(root)
    chronological = git_rev_list_range(merge_base, "dev", root)
    sha_to_pr: dict[str, int] = {}
    for row in map_rows:
        sha = row.get("sha", "").strip()
        raw_pr = row.get("pr_number", "").strip()
        if sha and raw_pr.isdigit():
            sha_to_pr[sha] = int(raw_pr)

    by_pr: dict[int, list[str]] = defaultdict(list)
    for sha in chronological:
        pr = sha_to_pr.get(sha)
        if pr is not None:
            by_pr[pr].append(sha)
    return by_pr


def correct_base_sha(pr_number: int, commits_by_pr: dict[int, list[str]], root: Path) -> str:
    idx = FULL_CANONICAL_ORDER.index(pr_number)
    if idx == 0:
        return ref_to_sha("main", root)
    predecessor = FULL_CANONICAL_ORDER[idx - 1]
    if predecessor in MERGED_FOUNDATION_PRS:
        branch = FOUNDATION_BRANCHES[predecessor][0]
        return ref_to_sha(branch, root)
    pred_shas = commits_by_pr.get(predecessor, [])
    if pred_shas:
        return pred_shas[-1]
    open_prs = fetch_open_prs()
    pr_by_number = {pr["number"]: pr for pr in open_prs}
    hydrate_pass_through_prs(pr_by_number)
    intended = intended_base_for(pr_number, pr_by_number)
    return ref_to_sha(intended, root)


def commits_are_contiguous(shas: list[str], chronological: list[str]) -> bool:
    if len(shas) <= 1:
        return True
    try:
        indices = [chronological.index(sha) for sha in shas]
    except ValueError:
        return False
    start = indices[0]
    return indices == list(range(start, start + len(shas)))


def diff_files_method_a(base_sha: str, last_sha: str, root: Path) -> list[str]:
    output = run_cmd(["git", "diff", "--name-only", base_sha, last_sha], cwd=root)
    return [line.strip() for line in output.splitlines() if line.strip()]


def diff_files_sequential(base_sha: str, shas: list[str], root: Path) -> list[str]:
    """Union of per-commit diffs when cherry-pick batch fails."""
    files: set[str] = set()
    parent = base_sha
    for sha in shas:
        files.update(diff_files_method_a(parent, sha, root))
        parent = sha
    return sorted(files)


def _audit_worktree_path(pr_number: int) -> Path:
    path = Path(f"{WORKTREE_PREFIX}{pr_number}")
    if not str(path).startswith(WORKTREE_PREFIX):
        raise ValueError(f"worktree path must be under {WORKTREE_PREFIX}")
    return path


def _remove_audit_worktree(worktree: Path, root: Path) -> None:
    if not worktree.exists():
        return
    try:
        run_cmd(["git", "worktree", "remove", "--force", str(worktree)], cwd=root)
    except subprocess.CalledProcessError:
        run_cmd(["git", "worktree", "prune"], cwd=root)


def diff_files_method_b(
    base_sha: str,
    shas: list[str],
    pr_number: int,
    root: Path,
) -> tuple[list[str], list[str], bool]:
    """Ephemeral worktree cherry-pick simulation (Method B)."""
    worktree = _audit_worktree_path(pr_number)
    notes: list[str] = []
    conflict = False
    if worktree.exists():
        _remove_audit_worktree(worktree, root)

    try:
        run_cmd(["git", "worktree", "add", "-d", str(worktree), base_sha], cwd=root)
        try:
            run_cmd(
                ["git", "-C", str(worktree), "cherry-pick", "--no-commit", *shas],
                cwd=root,
            )
        except subprocess.CalledProcessError:
            conflict = True
            notes.append("cherry_pick_conflict: simulation aborted cleanly")
            try:
                run_cmd(["git", "-C", str(worktree), "cherry-pick", "--abort"], cwd=root)
            except subprocess.CalledProcessError:
                run_cmd(["git", "-C", str(worktree), "reset", "--hard"], cwd=root)
            return [], notes, conflict

        output = run_cmd(
            ["git", "-C", str(worktree), "diff", "--cached", "--name-only"],
            cwd=root,
        )
        files = [line.strip() for line in output.splitlines() if line.strip()]
        return files, notes, conflict
    finally:
        _remove_audit_worktree(worktree, root)


def upstream_owned_files(
    pr_number: int,
    commits_by_pr: dict[int, list[str]],
    root: Path,
    *,
    chronological: list[str] | None = None,
    sha_to_pr: dict[str, int] | None = None,
    splits: dict[str, dict[int, list[str]]] | None = None,
) -> set[str]:
    owned: set[str] = set()
    idx = FULL_CANONICAL_ORDER.index(pr_number)
    split_map = splits or {}
    for pred in FULL_CANONICAL_ORDER[:idx]:
        if pred in MERGED_FOUNDATION_PRS:
            continue
        for sha in commits_by_pr.get(pred, []):
            owned.update(changed_files_for_pr_commit(sha, pred, split_map, root))
    # Split collateral from commits assigned to this or later PRs.
    if chronological and sha_to_pr:
        for sha in chronological:
            if sha not in split_map:
                continue
            for pred in FULL_CANONICAL_ORDER[:idx]:
                if pred in MERGED_FOUNDATION_PRS:
                    continue
                owned.update(changed_files_for_pr_commit(sha, pred, split_map, root))
    return owned


def manifest_files_for_commits(
    shas: list[str],
    upstream: set[str],
    root: Path,
    *,
    pr_number: int,
    splits: dict[str, dict[int, list[str]]] | None = None,
) -> set[str]:
    split_map = splits or {}
    files: set[str] = set()
    for sha in shas:
        files.update(changed_files_for_pr_commit(sha, pr_number, split_map, root))
    return files - upstream


def format_audit_section(result: AuditResult) -> str:
    lines = [
        "## Simulated Diff Audit",
        "",
        f"- method: {result.method}",
        f"- commits: {result.commit_count}",
        f"- diff_files: {result.diff_file_count}",
        f"- out_of_scope (HIGH): {len(result.out_of_scope)}",
        f"- missing: {len(result.missing)}",
    ]
    if result.out_of_scope:
        lines.append("- out_of_scope_files:")
        lines.extend(f"  - `{path}`" for path in sorted(result.out_of_scope))
    if result.missing:
        lines.append("- missing_files:")
        lines.extend(f"  - `{path}`" for path in sorted(result.missing))
    if result.notes:
        lines.append("- notes:")
        lines.extend(f"  - {note}" for note in result.notes)
    lines.append("")
    return "\n".join(lines) + "\n"


def update_manifest_audit(manifest_path: Path, result: AuditResult) -> None:
    text = manifest_path.read_text(encoding="utf-8")
    audit_block = format_audit_section(result)

    if "## Simulated Diff Audit" in text:
        text = re.sub(
            r"## Simulated Diff Audit\n\n.*?(?=\n## Phase 5 verification checklist)",
            audit_block.rstrip() + "\n\n",
            text,
            count=1,
            flags=re.DOTALL,
        )
    else:
        text = re.sub(
            r"(## Drift flags\n\n.*?\n\n)(## Phase 5 verification checklist)",
            rf"\1{audit_block}\2",
            text,
            count=1,
            flags=re.DOTALL,
        )
    manifest_path.write_text(text, encoding="utf-8")


def merge_out_of_scope_drift(existing: str, out_count: int) -> str:
    flag = f"out_of_scope={out_count}"
    if not existing:
        return flag
    parts = [part for part in existing.split(";") if not part.startswith("out_of_scope=")]
    parts.append(flag)
    return ";".join(parts)


def write_audit_report(results: list[AuditResult], path: Path) -> None:
    lines = [
        "# PR Chain Simulated Diff Audit Report",
        "",
        "Read-only audit (D-20 / PR-05). Scope bleed catalogued for Phase 5 cherry-pick splits.",
        "",
        "## Per-PR summary",
        "",
        "| PR | commits | diff_files | out_of_scope | missing | method | notes |",
        "|----|---------|------------|--------------|---------|--------|-------|",
    ]
    for result in sorted(results, key=lambda item: item.pr_number):
        note_text = "; ".join(result.notes) if result.notes else ""
        if result.pr_number in AUDIT_HOTSPOT_PRS:
            hotspot = f"hotspot #{result.pr_number}"
            note_text = f"{hotspot}; {note_text}" if note_text else hotspot
        lines.append(
            f"| #{result.pr_number} | {result.commit_count} | {result.diff_file_count} | "
            f"{len(result.out_of_scope)} | {len(result.missing)} | {result.method} | {note_text} |"
        )

    lines.extend(["", "## Hotspot detail", ""])
    for pr in sorted(AUDIT_HOTSPOT_PRS):
        match = next((r for r in results if r.pr_number == pr), None)
        if match is None:
            continue
        lines.append(f"### PR #{pr}")
        lines.append("")
        lines.append(f"- Commits assigned: {match.commit_count}")
        lines.append(f"- Simulated diff files: {match.diff_file_count}")
        lines.append(f"- Out of scope: {len(match.out_of_scope)}")
        if match.out_of_scope[:10]:
            preview = ", ".join(f"`{f}`" for f in sorted(match.out_of_scope)[:10])
            suffix = "..." if len(match.out_of_scope) > 10 else ""
            lines.append(f"- Sample out_of_scope: {preview}{suffix}")
        lines.append("")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_single_pr(
    pr_number: int,
    shas: list[str],
    chronological: list[str],
    commits_by_pr: dict[int, list[str]],
    root: Path,
    *,
    splits: dict[str, dict[int, list[str]]] | None = None,
    sha_to_pr: dict[str, int] | None = None,
) -> AuditResult:
    notes: list[str] = []
    split_map = splits or {}
    if pr_number in AUDIT_HOTSPOT_PRS:
        if pr_number == 125:
            notes.append("orphan bucket: 49+ dev-only commits vs stale branch tip (Pitfall 6 context)")
        elif pr_number == 116:
            notes.append("multi-subsystem blob: 10 commits spanning N4/GPU/workflows (Pitfall 6)")

    if not shas:
        return AuditResult(
            pr_number=pr_number,
            commit_count=0,
            diff_file_count=0,
            out_of_scope=[],
            missing=[],
            method="none",
            notes=notes + ["no assigned commits"],
        )

    base_sha = correct_base_sha(pr_number, commits_by_pr, root)
    upstream = upstream_owned_files(
        pr_number,
        commits_by_pr,
        root,
        chronological=chronological,
        sha_to_pr=sha_to_pr,
        splits=split_map,
    )
    manifest_files = manifest_files_for_commits(
        shas, upstream, root, pr_number=pr_number, splits=split_map
    )

    if commits_are_contiguous(shas, chronological):
        diff_files = set(diff_files_method_a(base_sha, shas[-1], root))
        method = "contiguous"
    else:
        diff_list, cp_notes, conflict = diff_files_method_b(base_sha, shas, pr_number, root)
        diff_files = set(diff_list)
        method = "worktree"
        notes.extend(cp_notes)
        if conflict:
            diff_files = set(diff_files_sequential(base_sha, shas, root))
            method = "sequential_fallback"
            notes.append(f"sequential_fallback: {len(diff_files)} diff files after cherry-pick conflict")
            if pr_number == 125:
                dev_tip = ref_to_sha("dev", root)
                tip_diff = set(diff_files_method_a(base_sha, dev_tip, root))
                notes.append(
                    f"dev_tip_orphan_bucket: {len(tip_diff)} files in diff {base_sha[:12]}..dev "
                    "(49+ commits on dev beyond branch tip)"
                )
                diff_files = tip_diff
                method = "dev_tip_orphan"

    # Strip split-commit collateral attributed to other PRs.
    for sha in shas:
        if sha not in split_map:
            continue
        collateral = set(git_changed_files(sha, root)) - changed_files_for_pr_commit(
            sha, pr_number, split_map, root
        )
        if collateral:
            diff_files -= collateral
            notes.append(f"split:{sha[:12]} collateral→other PRs ({len(collateral)} files)")

    out_of_scope = sorted(diff_files - manifest_files)
    missing = sorted(manifest_files - diff_files)

    return AuditResult(
        pr_number=pr_number,
        commit_count=len(shas),
        diff_file_count=len(diff_files),
        out_of_scope=out_of_scope,
        missing=missing,
        method=method,
        notes=notes,
    )


def generate_audit(root: Path) -> None:
    """Simulated diff audit per PR (read-only git; ephemeral /tmp worktrees only)."""
    planning = root / ".planning"
    chain_dir = planning / "pr-chain"
    index_path = planning / "pr-chain-index.csv"
    audit_path = chain_dir / "audit-report.md"

    if not index_path.is_file():
        raise SystemExit(f"BLOCKING: missing {index_path.relative_to(root)} — run --inventory first")

    map_rows = load_commit_map_rows(chain_dir)
    splits = load_commit_splits(chain_dir)
    commits_by_pr = commits_by_pr_chronological(root, map_rows)
    merge_base = git_merge_base(root)
    chronological = git_rev_list_range(merge_base, "dev", root)
    sha_to_pr: dict[str, int] = {}
    for row in map_rows:
        sha = row.get("sha", "").strip()
        raw_pr = row.get("pr_number", "").strip()
        if sha and raw_pr.isdigit():
            sha_to_pr[sha] = int(raw_pr)

    _, existing_rows = _validate.read_csv_rows(index_path)
    row_by_pr: dict[int, dict[str, str]] = {}
    for row in existing_rows:
        raw = row.get("pr_number", "").strip()
        if raw.isdigit():
            row_by_pr[int(raw)] = dict(row)

    results: list[AuditResult] = []
    for pr_number in CANONICAL_OPEN_PR_ORDER:
        manifest_path = manifest_for_pr(chain_dir, pr_number)
        if manifest_path is None:
            raise SystemExit(f"BLOCKING: no manifest for open PR #{pr_number}")

        shas = commits_by_pr.get(pr_number, [])
        result = audit_single_pr(
            pr_number,
            shas,
            chronological,
            commits_by_pr,
            root,
            splits=splits,
            sha_to_pr=sha_to_pr,
        )
        results.append(result)
        update_manifest_audit(manifest_path, result)

        row = row_by_pr.get(pr_number)
        if row is not None:
            row["drift_flags"] = merge_out_of_scope_drift(
                row.get("drift_flags", ""),
                len(result.out_of_scope),
            )

    write_audit_report(results, audit_path)

    foundation_rows = [
        row for row in existing_rows if row.get("pr_number", "").strip() in {"115", "97"}
    ]
    open_rows = sorted(
        [row_by_pr[pr] for pr in CANONICAL_OPEN_PR_ORDER if pr in row_by_pr],
        key=lambda row: int(row.get("merge_position", "0")),
    )
    final_rows = foundation_rows + open_rows

    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CANONICAL_HEADER.split(","))
        writer.writeheader()
        for row in final_rows:
            writer.writerow(row)

    total_out = sum(len(r.out_of_scope) for r in results)
    hotspots = ", ".join(f"#{pr}" for pr in sorted(AUDIT_HOTSPOT_PRS))
    print(
        f"Wrote audit for {len(results)} open PRs; "
        f"total out_of_scope files={total_out}; hotspots audited: {hotspots}"
    )
    print(f"Rollup report: {audit_path.relative_to(root)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        action="store_true",
        help="generate PR chain inventory manifests and rollup CSV",
    )
    parser.add_argument(
        "--map",
        action="store_true",
        help="assign main..dev commits to PR slots and write commit-map.csv",
    )
    parser.add_argument(
        "--drift",
        action="store_true",
        help="derive intended-base chain, catalogue drift, and record base-correction plans",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="simulated diff audit per PR (read-only; ephemeral /tmp worktrees)",
    )
    args = parser.parse_args()
    if not args.inventory and not args.map and not args.drift and not args.audit:
        parser.error("specify --inventory, --map, --drift, and/or --audit")
    root = repo_root()
    if args.inventory:
        generate_inventory(root)
    if args.map:
        generate_commit_map(root)
    if args.drift:
        generate_drift(root)
    if args.audit:
        generate_audit(root)


if __name__ == "__main__":
    main()
