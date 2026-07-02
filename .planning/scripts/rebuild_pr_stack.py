#!/usr/bin/env python3
"""Bottom-up path-filtered rebuild of PR chain branches (Phase 5 Wave 2b)."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_repo_root = Path(
    subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_validate = _load_module("validate_pr_chain", _SCRIPTS_DIR / "validate_pr_chain.py")
_gen = _load_module("generate_pr_chain_inventory", _SCRIPTS_DIR / "generate_pr_chain_inventory.py")

CANONICAL_OPEN_PR_ORDER = _validate.CANONICAL_OPEN_PR_ORDER
FROZEN_DEV_TIP = "0ab73808ca1266d2bdc5d9baf1cead84014b016b"

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

SKIP_PRS: frozenset[int] = frozenset()  # rebuild all; hotspots need rebase after stack shift


def run(args: list[str], *, check: bool = True, cwd: Path | None = None) -> str:
    result = subprocess.run(
        args, capture_output=True, text=True, cwd=cwd or _repo_root, check=check
    )
    if result.returncode != 0 and check:
        if result.stdout:
            print(result.stdout, file=sys.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
    return result.stdout.strip()


def verify_dev_freeze() -> None:
    tip = run(["git", "rev-parse", "origin/dev"])
    if tip != FROZEN_DEV_TIP:
        print(f"STOP: origin/dev={tip[:12]} != frozen {FROZEN_DEV_TIP[:12]}", file=sys.stderr)
        sys.exit(1)


def map_upstream_files(pr_number: int, by_pr: dict[int, list[str]], splits: dict) -> set[str]:
    """Files attributed to predecessor PR(s) in commit-map (immediate pred only)."""
    order = list(CANONICAL_OPEN_PR_ORDER)
    idx = order.index(pr_number)
    if idx == 0:
        # PR #98 rebuilds onto main — treat all files on main since merge-base as upstream.
        merge_base = run(["git", "merge-base", "origin/main", "origin/dev"])
        out = run(["git", "diff", "--name-only", f"{merge_base}..origin/main"])
        return set(line.strip() for line in out.splitlines() if line.strip())
    pred = order[idx - 1]
    owned: set[str] = set()
    for sha in by_pr.get(pred, []):
        owned.update(_gen.changed_files_for_pr_commit(sha, pred, splits, _repo_root))
    return owned


def base_ref_for(pr_number: int) -> str:
    order = list(CANONICAL_OPEN_PR_ORDER)
    idx = order.index(pr_number)
    if idx == 0:
        return "origin/main"
    pred = order[idx - 1]
    return f"origin/{BRANCH_BY_PR[pred]}"


def file_exists_at(sha: str, path: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{sha}:{path}"],
            cwd=_repo_root,
            capture_output=True,
        ).returncode
        == 0
    )


def apply_commit_paths(
    sha: str,
    pr_number: int,
    splits: dict,
    upstream: set[str],
    *,
    exclude_globs: tuple[str, ...] = (),
) -> None:
    paths = sorted(
        _gen.changed_files_for_pr_commit(sha, pr_number, splits, _repo_root) - upstream
    )
    if exclude_globs:
        paths = [
            p
            for p in paths
            if not any(p.startswith(glob.removesuffix("/**")) for glob in exclude_globs)
        ]
    if not paths:
        return
    for path in paths:
        if file_exists_at(sha, path):
            run(["git", "checkout", sha, "--", path])
        else:
            run(["git", "rm", "-f", "--ignore-unmatch", path])


def rebuild_pr(
    pr_number: int,
    shas: list[str],
    is_merge: dict[str, bool],
    *,
    dry_run: bool = False,
) -> str:
    branch = BRANCH_BY_PR[pr_number]
    base_ref = base_ref_for(pr_number)
    pick_shas = [s for s in shas if not is_merge.get(s, False)]

    print(f"\n=== PR #{pr_number} ({branch}) base={base_ref} commits={len(pick_shas)} ===")

    if dry_run:
        return base_ref

    run(["git", "fetch", "origin", branch])
    base_sha = run(["git", "rev-parse", base_ref])
    run(["git", "checkout", "-B", branch, base_sha])

    splits = _gen.load_commit_splits(_repo_root / ".planning" / "pr-chain")
    map_rows = _gen.load_commit_map_rows(_repo_root / ".planning" / "pr-chain")
    by_pr = _gen.commits_by_pr_chronological(_repo_root, map_rows)
    upstream = map_upstream_files(pr_number, by_pr, splits)
    exclude: tuple[str, ...] = ("workflows/",) if pr_number == 133 else ()
    for sha in pick_shas:
        print(f"  apply {sha[:12]}...")
        apply_commit_paths(sha, pr_number, splits, upstream, exclude_globs=exclude)

    staged = run(["git", "diff", "--cached", "--name-only"])
    unstaged = run(["git", "diff", "--name-only"])
    if not staged and not unstaged:
        print(f"  zero-scope #{pr_number} — reset branch to base")
        tip = base_sha
        run(["git", "push", "--force-with-lease", "--no-verify", "origin", branch])
        print(f"  pushed {branch} @ {tip[:12]} (base reset)")
        return tip

    msg = f"rebuild(05-05): PR #{pr_number} path-filtered stack rebuild (main foundation)"
    run(["git", "commit", "--no-verify", "-m", msg])

    tip = run(["git", "rev-parse", "HEAD"])
    run(["git", "push", "--force-with-lease", "--no-verify", "origin", branch])
    print(f"  pushed {branch} @ {tip[:12]}")
    return tip


def load_data() -> tuple[dict[int, list[str]], dict[str, bool]]:
    map_rows = _gen.load_commit_map_rows(_repo_root / ".planning" / "pr-chain")
    by_pr = _gen.commits_by_pr_chronological(_repo_root, map_rows)
    is_merge = {
        row["sha"]: row.get("is_merge", "").lower() == "true" for row in map_rows if row.get("sha")
    }
    return by_pr, is_merge


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prs", type=int, nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--enforce-freeze-check",
        action="store_true",
        help="Abort if origin/dev != reorg snapshot (legacy Wave 0 guard; off by default)",
    )
    args = parser.parse_args()

    if args.enforce_freeze_check:
        verify_dev_freeze()

    run(["git", "fetch", "origin"])
    by_pr, is_merge = load_data()

    target_prs = args.prs or list(CANONICAL_OPEN_PR_ORDER)
    for pr in target_prs:
        rebuild_pr(pr, by_pr[pr], is_merge, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
