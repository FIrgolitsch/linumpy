#!/usr/bin/env python3
"""Cherry-pick a commit applying only paths attributed to a PR in commit-splits.json."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(args: list[str], *, cwd: Path | None = None, check: bool = True) -> str:
    result = subprocess.run(args, capture_output=True, text=True, cwd=cwd, check=check)
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sha", help="Commit to cherry-pick (split-aware)")
    parser.add_argument("pr_number", type=int, help="Target PR number for path filter")
    parser.add_argument(
        "--splits",
        type=Path,
        default=Path(".planning/pr-chain/commit-splits.json"),
        help="commit-splits.json path",
    )
    parser.add_argument("--message", default="", help="Override commit message")
    args = parser.parse_args()

    splits = json.loads(args.splits.read_text(encoding="utf-8"))
    pr_map = splits.get(args.sha)
    if pr_map is None:
        run(["git", "cherry-pick", args.sha])
        return

    paths = pr_map.get(str(args.pr_number), [])
    if not paths:
        print(f"No paths for PR #{args.pr_number} in split of {args.sha[:12]}", file=sys.stderr)
        sys.exit(1)

    # Apply only attributed paths — no full cherry-pick (avoids collateral + untracked conflicts).
    for path in paths:
        run(["git", "checkout", args.sha, "--", path])
    staged = run(["git", "diff", "--cached", "--name-only"])
    if not staged:
        print(f"split checkout produced no staged changes for {args.sha[:12]}", file=sys.stderr)
        sys.exit(1)
    msg = args.message or run(["git", "log", "-1", "--format=%s", args.sha])
    run(["git", "commit", "--no-verify", "-m", f"{msg} [split #{args.pr_number}]"])


if __name__ == "__main__":
    main()
