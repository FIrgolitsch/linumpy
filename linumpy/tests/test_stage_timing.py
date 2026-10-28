"""Tests for env-gated stage timing."""

import json
from pathlib import Path

import numpy as np
import pytest

from linumpy.mosaic.grid import add_volume_to_mosaic, get_diffusion_blending_weights
from linumpy.utils import stage_timing


def _drop_totals() -> None:
    stage_timing._SECONDS.clear()
    stage_timing._COUNTS.clear()
    stage_timing._META.clear()


@pytest.fixture(autouse=True)
def _isolated_stage_timing(monkeypatch, tmp_path):
    drain = tmp_path / "_drain.jsonl"
    monkeypatch.setenv("LINUMPY_STAGE_TIMING", str(drain))
    stage_timing.flush()
    monkeypatch.delenv("LINUMPY_STAGE_TIMING", raising=False)
    yield
    _drop_totals()


def test_unset_is_noop(tmp_path, monkeypatch):
    timing_path = tmp_path / "stages.jsonl"
    monkeypatch.delenv("LINUMPY_STAGE_TIMING", raising=False)
    sentinel = object()
    with stage_timing.span("diffusion_blend", device="cpu"):
        observed = sentinel
    assert observed is sentinel
    stage_timing.flush()
    assert not timing_path.exists()


def test_diffusion_values_unchanged(tmp_path, monkeypatch):
    mask = np.zeros((16, 16), dtype=bool)
    mask[:, :8] = True
    monkeypatch.delenv("LINUMPY_STAGE_TIMING", raising=False)
    unset = get_diffusion_blending_weights(mask, factor=1, n_steps=2)
    monkeypatch.setenv("LINUMPY_STAGE_TIMING", str(tmp_path / "direct.jsonl"))
    timed = get_diffusion_blending_weights(mask, factor=1, n_steps=2)
    np.testing.assert_array_equal(unset, timed)

    timing_path = tmp_path / "stages.jsonl"
    monkeypatch.setenv("LINUMPY_STAGE_TIMING", str(timing_path))
    volume = np.ones((1, 16, 16), dtype=np.float32)
    mosaic = np.zeros((1, 20, 20), dtype=np.float32)
    mosaic[0, 0:16, 0:8] = 1.0
    add_volume_to_mosaic(volume, (0, 0, 0), mosaic, blending_method="diffusion", factor=1)
    stage_timing.flush()
    assert timing_path.is_file()
    lines = [line for line in timing_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["stage"] == "diffusion_blend"
    assert record["device"] == "cpu"
    stage_timing.flush()
    assert timing_path.read_text(encoding="utf-8").splitlines() == lines


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


def _stage(name, root: Path, **overrides):
    row = {
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


def _valid_payload(root: Path):
    return {
        "subject": "sub-21",
        "host": "132.207.157.41",
        "slice_ids": [45, 46, 47],
        "page_cache": "not_dropped",
        "linumpy_git_rev": "abc123",
        "linumpy_git_branch": "develop",
        "linum_basic_revision": "def456",
        "thread_env": {"OMP_NUM_THREADS": 5},
        "disk": {
            "tmp_source": "/dev/nvme0n1p1",
            "subject_source": "/dev/nvme0n1p2",
            "work_source": "/dev/nvme0n1p2",
            "tmp_st_dev": 66305,
            "subject_st_dev": 66306,
            "work_st_dev": 66306,
            "same_filesystem": False,
        },
        "scratch_copy": {"status": "not_taken", "seconds": None, "scratch_dir": False},
        "task_hash": None,
        "task_hash_reason": "direct command has no Nextflow task hash",
        "provenance_flags": {"scratch_dir": False},
        "stages": [_stage(name, root) for name in _REQUIRED_STAGES],
        "sources_read": [str(root / "argv.txt")],
    }


def test_splits_are_separate(tmp_path, monkeypatch):
    timing_path = tmp_path / "stages.jsonl"
    monkeypatch.setenv("LINUMPY_STAGE_TIMING", str(timing_path))
    stage_timing.add("stitch_scoring", 1.25, device="cpu", function="score")
    stage_timing.add("diffusion_blend", 0.5, device="cpu")
    stage_timing.add("stitch_scoring", 0.25, device="cpu", function="score_again")
    stage_timing.flush()
    rows = [json.loads(line) for line in timing_path.read_text(encoding="utf-8").splitlines() if line]
    by_stage = {row["stage"]: row for row in rows}
    assert set(by_stage) == {"stitch_scoring", "diffusion_blend"}
    assert by_stage["stitch_scoring"]["seconds"] == pytest.approx(1.5)
    assert by_stage["diffusion_blend"]["seconds"] == pytest.approx(0.5)
    assert by_stage["stitch_scoring"]["count"] == 2
    assert by_stage["stitch_scoring"]["functions"] == ["score", "score_again"]
    assert by_stage["diffusion_blend"]["seconds"] != by_stage["stitch_scoring"]["seconds"]


def test_rejects_cache_and_log(tmp_path):
    root = tmp_path / "timing"
    root.mkdir()
    stage_timing.validate_record(_valid_payload(root), timing_root=str(root))

    logged = _valid_payload(root)
    logged["stages"][0]["duration_source"] = "nextflow_log"
    with pytest.raises(ValueError, match="nextflow"):
        stage_timing.validate_record(logged, timing_root=str(root))

    hashed = _valid_payload(root)
    hashed["task_hash"] = "0123456789abcdef"
    with pytest.raises(ValueError, match="task_hash"):
        stage_timing.validate_record(hashed, timing_root=str(root))

    missing = _valid_payload(root)
    missing["stages"] = [row for row in missing["stages"] if row["name"] != "stitch_scoring"]
    with pytest.raises(ValueError, match="stitch_scoring"):
        stage_timing.validate_record(missing, timing_root=str(root))

    taken = _valid_payload(root)
    taken["stages"][0]["duration_source"] = "not_taken"
    taken["stages"][0]["seconds"] = 1.25
    with pytest.raises(ValueError, match="not_taken"):
        stage_timing.validate_record(taken, timing_root=str(root))

    stage_timing.validate_record(_valid_payload(root), timing_root=str(root))

    traced = _valid_payload(root)
    traced["sources_read"] = ["/tmp/trace.txt"]
    with pytest.raises(ValueError, match=r"trace\.txt"):
        stage_timing.validate_record(traced, timing_root=str(root))

    cached = _valid_payload(root)
    cached["sources_read"] = ["/scratch/work/.nextflow/cache/abc"]
    with pytest.raises(ValueError, match="nextflow"):
        stage_timing.validate_record(cached, timing_root=str(root))

    escaped = _valid_payload(root)
    escaped["stages"][0]["output_path"] = "/etc/passwd"
    with pytest.raises(ValueError, match="timing"):
        stage_timing.validate_record(escaped, timing_root=str(root))


def test_device_id_inequality(monkeypatch):
    class _Stat:
        def __init__(self, dev):
            self.st_dev = dev

    def fake_stat(path):
        return _Stat(1 if str(path).endswith("a") else 2)

    monkeypatch.setattr(stage_timing.os, "stat", fake_stat)
    assert stage_timing.same_filesystem("disk-a", "disk-b") is False
    assert stage_timing.same_filesystem("disk-b", "other-b") is True


def test_flush_refuses_ndarray(tmp_path, monkeypatch):
    timing_path = tmp_path / "stages.jsonl"
    monkeypatch.setenv("LINUMPY_STAGE_TIMING", str(timing_path))
    stage_timing.add("diffusion_blend", 0.1, device="cpu", weights=np.zeros((2, 2)))
    with pytest.raises(TypeError):
        stage_timing.flush()
    assert not timing_path.exists()
