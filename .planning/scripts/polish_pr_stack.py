#!/usr/bin/env python3
"""Rewrite stacked PR tip commit messages and refresh GitHub PR headers/descriptions."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_repo_root = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
)


def _load_validate():
    spec = importlib.util.spec_from_file_location("validate_pr_chain", _SCRIPTS_DIR / "validate_pr_chain.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_validate = _load_validate()
CANONICAL_OPEN_PR_ORDER = _validate.CANONICAL_OPEN_PR_ORDER
FULL_REVIEW_ORDER: tuple[int, ...] = (115, 97, *CANONICAL_OPEN_PR_ORDER)

BRANCH_BY_PR: dict[int, str] = {
    98: "pr-e-motor-stacking",
    99: "pr-f-gpu-acceleration",
    100: "pr-g-diagnostics-analysis",
    101: "pr-h-allen-atlas-ras",
    108: "pr-i-nextflow-workflows",
    106: "pr-k-slice-config",
    107: "pr-l-manual-align",
    87: "pr-3-slice-interpolation",
    116: "pr-n4-bias-field",
    110: "pr-ci-nextflow",
    111: "pr-tail-cleanup",
    40: "sphinx-config",
    112: "pr-m-gpu-kvikio",
    130: "pr-n-perf",
    131: "pr-o-galvo-fix",
    118: "pr-p-py314-modernization",
    132: "pr-q-perf-pipeline",
    121: "pr-r-allen-followups",
    122: "pr-s-metrics-stacking",
    123: "pr-t-attenuation-methods",
    124: "pr-v-psf-focal",
    128: "pr-w-linum-basic",
    133: "pr-u-fix-illum",
    134: "pr-x-infra-quality",
}

# Sensible one-line subjects (no GSD phase tags, no rebuild() boilerplate).
COMMIT_SUBJECT: dict[int, str] = {
    98: "feat: motor stacking, auto-exclude, and refined stitching",
    99: "feat: GPU acceleration module and unified GPU scripts",
    100: "feat: diagnostic and analysis scripts",
    101: "feat: Allen atlas improvements and RAS alignment",
    108: "feat: Nextflow workflows, profiles, auto-assess, and rehoming",
    106: "chore: stack slot — slice-config carried in #108",
    107: "feat: manual-alignment export, refinement, and global-transform tools",
    87: "feat: slice interpolation for missing sections (z-morph)",
    116: "feat: GPU-accelerated N4 bias field correction",
    110: "ci(nextflow): nf-test infrastructure and GitHub Actions workflow",
    111: "chore: tail cleanup, gitignore, and PR-chain validator tests",
    40: "docs: Sphinx and ReadTheDocs configuration",
    112: "feat: GPU keep-on-device, kvikio reader, and pipeline GPU paths",
    130: "perf: cap PyTorch threads and tune fix_illumination parallelism",
    131: "fix(galvo): per-tile detect, threaded column strips, skip_tiles",
    118: "chore: stack slot — py314 modernization carried upstream",
    132: "perf: pipeline-wide GPU and scratch I/O optimizations",
    121: "chore: stack slot — Allen follow-ups carried upstream",
    122: "chore: stack slot — stacking metrics carried upstream",
    123: "feat: depth-resolved attenuation estimators (Li 2020 default)",
    124: "feat: PSF compensation and focal curvature follow-ups",
    128: "feat: linum-basic illumination backend and Nextflow process",
    133: "fix(illum): pool tiles across Z, darkfield and flatfield knobs",
    134: "chore: dedicated Phase 6 infrastructure and quality stack slot",
}

PASS_THROUGH_PRS = frozenset({106, 118, 121, 122})

GSD_LINE_RE = re.compile(r"^\s*(feat|fix|test|docs|refactor|chore|perf|ci)\(\d+-\d+\):", re.I)
HEADER_RE = re.compile(r"^> \*\*Stacked PR .+\*\*.*?(?:\n>.*)*\n\n---\n\n", re.MULTILINE | re.DOTALL)


def run(args: list[str], *, check: bool = True) -> str:
    result = subprocess.run(args, capture_output=True, text=True, cwd=_repo_root, check=check)
    return result.stdout.strip()


def gh_json(args: list[str]) -> object:
    return json.loads(run(["gh", *args]))


def base_ref(pr_number: int) -> str:
    idx = CANONICAL_OPEN_PR_ORDER.index(pr_number)
    if idx == 0:
        return "origin/main"
    pred = CANONICAL_OPEN_PR_ORDER[idx - 1]
    return f"origin/{BRANCH_BY_PR[pred]}"


def file_exists_at(ref: str, path: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{ref}:{path}"],
            cwd=_repo_root,
            capture_output=True,
        ).returncode
        == 0
    )


def apply_tree_diff(base: str, old_tip: str, branch: str) -> bool:
    """Recreate old_tip tree on top of base via path checkout. Returns False if empty."""
    run(["git", "reset", "--hard"])
    run(["git", "checkout", "-B", branch, base])
    paths = [p for p in run(["git", "diff", "--name-only", base, old_tip]).splitlines() if p.strip()]
    if not paths:
        return False
    for path in paths:
        if file_exists_at(old_tip, path):
            run(["git", "checkout", old_tip, "--", path])
        else:
            run(["git", "rm", "-f", "--ignore-unmatch", path])
    return True


def build_header(pr_number: int) -> str:
    pos = CANONICAL_OPEN_PR_ORDER.index(pr_number) + 1
    parts: list[str] = []
    for n in FULL_REVIEW_ORDER:
        parts.append(f"**#{n}**" if n == pr_number else f"#{n}")
    order_line = " → ".join(parts)
    idx = CANONICAL_OPEN_PR_ORDER.index(pr_number)
    if idx == 0:
        base_label = "`main`"
        base_note = "First open PR after merged foundation (#115, #97)."
    else:
        pred = CANONICAL_OPEN_PR_ORDER[idx - 1]
        base_label = f"`{BRANCH_BY_PR[pred]}`"
        base_note = f"Stacked on #{pred}; retargets to `main` as upstream PRs merge."
    return (
        f"> **Stacked PR {pos}/24 — review order:** {order_line}\n"
        f">\n"
        f"> Base: {base_label}. {base_note}\n\n"
        f"---\n\n"
    )


def extract_body_tail(existing: str) -> str:
    if "---" in existing:
        _head, tail = existing.split("---", 1)
        tail = tail.lstrip("\n")
        if tail.startswith("\n"):
            tail = tail.lstrip("\n")
        return tail
    return existing.strip()


def summary_blurb(pr_number: int, title: str, tail: str) -> str:
    if pr_number in PASS_THROUGH_PRS:
        return (
            f"## PR #{pr_number} — {title}\n\n"
            "No additional commits in this slot — changes are included upstream. "
            "This PR preserves stack ordering and will fast-forward when rebased after upstream merges.\n\n"
        )
    if tail.lstrip().startswith("##"):
        return tail
    return f"## PR #{pr_number} — {title}\n\n{tail}\n\n"


def build_pr_body(pr_number: int, title: str, existing_body: str) -> str:
    tail = extract_body_tail(existing_body)
    tail = re.sub(r"^## PR #\d+[^\n]*\n\n?", "", tail, count=1)
    return build_header(pr_number) + summary_blurb(pr_number, title, tail)


def commit_message(pr_number: int, title: str) -> str:
    subject = COMMIT_SUBJECT.get(pr_number, title.split("\n")[0].strip())
    subject = re.sub(r"\s*\(#\d+\)\s*$", "", subject)
    if GSD_LINE_RE.match(subject):
        subject = title
    body_lines = [
        subject,
        "",
        f"Stacked PR #{pr_number} in the linumpy merge chain.",
    ]
    if pr_number in PASS_THROUGH_PRS:
        body_lines.append("Pass-through slot: no file changes vs immediate base.")
    return "\n".join(body_lines)


def polish_branch(pr_number: int, *, dry_run: bool) -> None:
    branch = BRANCH_BY_PR[pr_number]
    base = base_ref(pr_number)
    remote_branch = f"origin/{branch}"
    old_tip = run(["git", "rev-parse", remote_branch])
    diff_paths = [p for p in run(["git", "diff", "--name-only", base, old_tip]).splitlines() if p.strip()]
    pr_meta = gh_json(["pr", "view", str(pr_number), "--json", "title,body"])
    title = str(pr_meta["title"])
    new_body = build_pr_body(pr_number, title, str(pr_meta.get("body") or ""))

    print(f"\n=== #{pr_number} {branch} diff_files={len(diff_paths)} ===")

    if dry_run:
        print(f"  commit: {commit_message(pr_number, title).splitlines()[0]}")
        print(f"  body header pos {CANONICAL_OPEN_PR_ORDER.index(pr_number) + 1}/24")
        return

    if not diff_paths:
        print("  skip branch amend (zero-scope); updating PR body only")
        run(["gh", "pr", "edit", str(pr_number), "--body", new_body])
        return

    run(["git", "fetch", "origin", branch, "main"])
    if not apply_tree_diff(base, old_tip, branch):
        print("  skip branch amend (empty diff after fetch); updating PR body only")
        run(["gh", "pr", "edit", str(pr_number), "--body", new_body])
        return
    run(["git", "commit", "--no-verify", "-m", commit_message(pr_number, title)])
    run(["git", "push", "--force-with-lease", "--no-verify", "origin", branch])
    tip = run(["git", "rev-parse", "HEAD"])
    print(f"  pushed {branch} @ {tip[:12]}")
    run(["gh", "pr", "edit", str(pr_number), "--body", new_body])
    print("  updated PR body")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prs", type=int, nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    targets = args.prs or list(CANONICAL_OPEN_PR_ORDER)
    run(["git", "fetch", "origin"])
    for pr in targets:
        if pr not in BRANCH_BY_PR:
            print(f"skip unknown PR #{pr}", file=sys.stderr)
            continue
        polish_branch(pr, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
