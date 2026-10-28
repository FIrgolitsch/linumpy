"""Tests for the hot-path timing driver helpers."""

import ast
import inspect
import subprocess
import types
from pathlib import Path

import pytest

from scripts.timing import linum_time_hot_path as driver

_THREAD_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
)


def _imported_modules(module: types.ModuleType) -> set[str]:
    tree = ast.parse(inspect.getsource(module))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", maxsplit=1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".", maxsplit=1)[0])
    return names


def test_thread_env_smaller_host():
    assert "numpy" not in _imported_modules(driver)
    assert "linumpy.config.threads" not in inspect.getsource(driver)
    env = driver.thread_env(16, max_cpus=44, processes=8)
    assert env.get("OMP_NUM_THREADS") == "2"
    assert env.get("LINUMPY_MAX_CPUS") == "16"
    for name in _THREAD_NAMES:
        assert env.get(name) == "2"
    assert "5" not in env.values()


def test_thread_env_full_host():
    env = driver.thread_env(44, max_cpus=44, processes=8)
    assert env.get("OMP_NUM_THREADS") == "5"
    assert env.get("LINUMPY_MAX_CPUS") == "44"
    for name in _THREAD_NAMES:
        assert env.get(name) == "5"


def test_output_prefix_rejects_escape(tmp_path):
    root = tmp_path / "timing"
    root.mkdir()
    inside = root / "hot_path.json"
    inside.write_text("{}", encoding="utf-8")
    resolved = driver.assert_under_timing(inside, root=root)
    assert resolved == inside.resolve()

    outside = tmp_path / "output"
    outside.mkdir()
    with pytest.raises(ValueError):
        driver.assert_under_timing(outside, root=root)

    sibling = tmp_path / "timing-evil"
    sibling.mkdir()
    with pytest.raises(ValueError):
        driver.assert_under_timing(sibling, root=root)

    link = root / "escape"
    link.symlink_to(outside)
    with pytest.raises(ValueError):
        driver.assert_under_timing(link, root=root)


_DISK_PATHS = (
    "/tmp",
    "/scratch_nvme/workspace/sub-21",
    "/scratch_nvme/workspace/work",
)
_REQUIRED_STAGES = (
    "illumination",
    "stitch",
    "stitch_scoring",
    "diffusion_blend",
    "attenuation",
    "common_space",
    "stack",
    "stack_reads",
    "per_plane_resample",
    "zip",
    "screenshot",
    "screenshot_annotated",
)
_INNER_PARENTS = {
    "stitch_scoring": "stitch",
    "diffusion_blend": "stitch",
    "stack_reads": "stack",
    "per_plane_resample": "stack",
}


def _find_argv(path: str) -> list[str]:
    return ["findmnt", "-T", path, "-o", "SOURCE,FSTYPE,TARGET", "-n"]


def test_disk_block_unequal_devices():
    calls: list[list[str]] = []

    def stat_fn(path: str) -> int:
        if path == "/tmp":
            return 66305
        if path in {"/scratch_nvme/workspace/sub-21", "/scratch_nvme/workspace/work"}:
            return 66306
        raise AssertionError(path)

    def findmnt_fn(argv: list[str]) -> str:
        assert isinstance(argv, list)
        calls.append(list(argv))
        return argv[2]

    block = driver.disk_block(stat_fn, findmnt_fn)
    assert block.get("same_filesystem") is False
    assert block.get("tmp_st_dev") == 66305
    assert block.get("subject_st_dev") == 66306
    assert block.get("work_st_dev") == 66306
    assert block.get("tmp_source") == "/tmp"
    assert "seconds" not in block
    assert calls == [_find_argv(path) for path in _DISK_PATHS]


def _stage(name: str, root: Path, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "name": name,
        "argv": ["linum", name],
        "seconds": 0.1,
        "duration_source": "perf_counter",
        "device": "host" if name == "zip" else "cpu",
        "flags": {},
        "output_path": str(root / name),
    }
    parent = _INNER_PARENTS.get(name)
    if parent is not None:
        row["parent"] = parent
        row["count"] = 1
    row.update(overrides)
    return row


def _valid_payload(root: Path) -> dict[str, object]:
    return {
        "subject": "sub-21",
        "host": "132.207.157.41",
        "slice_ids": [45, 46, 47],
        "page_cache": "not_dropped",
        "linumpy_git_rev": "abc123",
        "linumpy_git_branch": "develop",
        "linum_basic_revision": "def456",
        "thread_env": {"OMP_NUM_THREADS": "5"},
        "disk": {
            "tmp_source": "/dev/nvme0n1p1",
            "subject_source": "/dev/nvme0n1p2",
            "work_source": "/dev/nvme0n1p2",
            "tmp_st_dev": 66305,
            "subject_st_dev": 66306,
            "work_st_dev": 66306,
            "same_filesystem": False,
        },
        "scratch_copy": driver.scratch_copy(),
        "task_hash": None,
        "task_hash_reason": "direct command has no Nextflow task hash",
        "provenance_flags": {"scratch_dir": False},
        "stages": [_stage(name, root) for name in _REQUIRED_STAGES],
        "sources_read": [str(root / "argv.txt")],
    }


def test_finalize_rejects_log(tmp_path: Path):
    root = tmp_path / "timing"
    root.mkdir()
    assert driver.scratch_copy() == {"status": "not_taken", "seconds": None, "scratch_dir": False}
    payload = _valid_payload(root)
    driver.finalize(payload, timing_root=root)

    logged = _valid_payload(root)
    stages = logged["stages"]
    assert isinstance(stages, list)
    first = stages[0]
    assert isinstance(first, dict)
    first["duration_source"] = "nextflow_log"
    with pytest.raises(ValueError):
        driver.finalize(logged, timing_root=root)

    hashed = _valid_payload(root)
    hashed["task_hash"] = "0123456789abcdef"
    with pytest.raises(ValueError):
        driver.finalize(hashed, timing_root=root)

    missing = _valid_payload(root)
    missing_stages = missing["stages"]
    assert isinstance(missing_stages, list)
    kept: list[object] = []
    for row in missing_stages:
        assert isinstance(row, dict)
        if row.get("name") != "stitch_scoring":
            kept.append(row)
    missing["stages"] = kept
    with pytest.raises(ValueError, match="stitch_scoring"):
        driver.finalize(missing, timing_root=root)


def test_run_argv_rejects_shell(monkeypatch: pytest.MonkeyPatch):
    source = inspect.getsource(driver.run_argv)
    assert "shell=True" not in source
    assert "shell" not in inspect.signature(driver.run_argv).parameters
    captured: dict[str, object] = {}

    def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 0, stdout="ok", stderr="")

    monkeypatch.setattr(driver.subprocess, "run", fake_run)
    result = driver.run_argv(["findmnt", "-T", "/tmp"], {"OMP_NUM_THREADS": "2"})
    kwargs = captured.get("kwargs")
    assert isinstance(captured.get("argv"), list)
    assert isinstance(kwargs, dict)
    assert kwargs.get("check") is True
    assert "shell" not in kwargs
    assert result.stdout == "ok"
