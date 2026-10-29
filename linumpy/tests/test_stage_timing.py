"""Tests for env-gated stage timing."""

import ast
import json
import time
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


def _state_left_with_gate_closed(monkeypatch) -> list[str]:
    """Run a span and a direct add with the variable unset. Name every dictionary left non-empty."""
    monkeypatch.delenv("LINUMPY_STAGE_TIMING", raising=False)
    with stage_timing.span("diffusion_blend", device="cpu"):
        pass
    stage_timing.add("stitch_scoring", 1.0, device="cpu", function="score")
    totals = {
        "_SECONDS": stage_timing._SECONDS,
        "_COUNTS": stage_timing._COUNTS,
        "_META": stage_timing._META,
    }
    return [name for name, table in totals.items() if table]


def test_unset_is_noop(tmp_path, monkeypatch):
    assert _state_left_with_gate_closed(monkeypatch) == []
    timing_path = tmp_path / "stages.jsonl"
    monkeypatch.setenv("LINUMPY_STAGE_TIMING", str(timing_path))
    stage_timing.flush()
    assert not timing_path.exists()


def test_unset_check_notices_a_leaky_gate(monkeypatch):
    def leaky_add(stage: str, seconds: float, **_meta: object) -> None:
        stage_timing._SECONDS[stage] = stage_timing._SECONDS.get(stage, 0.0) + seconds

    monkeypatch.setattr(stage_timing, "add", leaky_add)
    assert _state_left_with_gate_closed(monkeypatch) == ["_SECONDS"]


def test_set_gate_records_elapsed(tmp_path, monkeypatch):
    timing_path = tmp_path / "stages.jsonl"
    monkeypatch.setenv("LINUMPY_STAGE_TIMING", str(timing_path))
    with stage_timing.span("stitch_scoring", device="cpu"):
        time.sleep(0.02)
    stage_timing.flush()
    rows = [json.loads(line) for line in timing_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 1
    assert rows[0]["stage"] == "stitch_scoring"
    assert rows[0]["count"] == 1
    assert rows[0]["device"] == "cpu"
    assert rows[0]["seconds"] >= 0.015


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


_SLICE_IDS = (45, 46, 47)
_PER_SLICE_STAGES = ("illumination", "stitch", "attenuation")


def _per_slice_entries(name, root: Path, seconds: float, device: str, slice_ids=_SLICE_IDS):
    """Split ``seconds`` evenly over ``slice_ids`` so the entries sum to the stage row."""
    return [
        {
            "slice_id": slice_id,
            "seconds": seconds / len(slice_ids),
            "device": device,
            "argv": ["linum", name, f"z{slice_id}"],
            "output_path": str(root / name / f"z{slice_id}"),
            "flags": {"slice_id": f"z{slice_id}"},
        }
        for slice_id in slice_ids
    ]


def _scope(**overrides):
    scope = {"slice_ids": [45, 46, 47], "slice_count": 3, "input_count": 3}
    scope.update(overrides)
    return scope


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
    if name in _PER_SLICE_STAGES and "per_slice" not in row:
        row["per_slice"] = _per_slice_entries(name, root, row["seconds"], row["device"])
    if name == "common_space" and "scope" not in row:
        row["scope"] = _scope()
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

    other_slices = _valid_payload(root)
    other_slices["slice_ids"] = [44, 45, 46, 47, 48]
    with pytest.raises(ValueError, match="slice_ids"):
        stage_timing.validate_record(other_slices, timing_root=str(root))

    dropped = _valid_payload(root)
    dropped["page_cache"] = "dropped"
    with pytest.raises(ValueError, match="not_dropped"):
        stage_timing.validate_record(dropped, timing_root=str(root))

    timed_scratch = _valid_payload(root)
    timed_scratch["scratch_copy"] = {"status": "not_taken", "seconds": 12.0, "scratch_dir": False}
    with pytest.raises(ValueError, match="not_taken"):
        stage_timing.validate_record(timed_scratch, timing_root=str(root))


def _timing_root(tmp_path) -> Path:
    root = tmp_path / "timing"
    root.mkdir()
    return root


def _copied_payload(root: Path, seconds: object):
    payload = _valid_payload(root)
    payload["scratch_copy"] = {"status": "perf_counter", "seconds": seconds, "scratch_dir": True}
    payload["provenance_flags"] = {"scratch_dir": True}
    return payload


def test_rejects_unmeasured_scratch_copy(tmp_path):
    root = _timing_root(tmp_path)
    stage_timing.validate_record(_copied_payload(root, 2.5), timing_root=str(root))
    for seconds in (None, "3.0", float("nan"), -1.0):
        with pytest.raises(ValueError, match="scratch_copy"):
            stage_timing.validate_record(_copied_payload(root, seconds), timing_root=str(root))

    timed = _valid_payload(root)
    timed["scratch_copy"] = {"status": "not_taken", "seconds": 12.0, "scratch_dir": False}
    with pytest.raises(ValueError, match="not_taken"):
        stage_timing.validate_record(timed, timing_root=str(root))


def test_rejects_disk_block_that_disagrees_with_its_device_ids(tmp_path):
    root = _timing_root(tmp_path)
    stage_timing.validate_record(_valid_payload(root), timing_root=str(root))

    claims_same = _valid_payload(root)
    claims_same["disk"]["same_filesystem"] = True
    with pytest.raises(ValueError, match="same_filesystem"):
        stage_timing.validate_record(claims_same, timing_root=str(root))

    claims_apart = _valid_payload(root)
    claims_apart["disk"]["tmp_st_dev"] = claims_apart["disk"]["subject_st_dev"]
    with pytest.raises(ValueError, match="same_filesystem"):
        stage_timing.validate_record(claims_apart, timing_root=str(root))

    same_disk = _valid_payload(root)
    same_disk["disk"]["tmp_st_dev"] = same_disk["disk"]["subject_st_dev"]
    same_disk["disk"]["same_filesystem"] = True
    stage_timing.validate_record(same_disk, timing_root=str(root))


def test_rejects_provenance_that_contradicts_scratch_copy(tmp_path):
    root = _timing_root(tmp_path)
    claims_copy = _valid_payload(root)
    claims_copy["provenance_flags"] = {"scratch_dir": True}
    with pytest.raises(ValueError, match="scratch_dir"):
        stage_timing.validate_record(claims_copy, timing_root=str(root))

    denies_copy = _copied_payload(root, 2.5)
    denies_copy["provenance_flags"] = {"scratch_dir": False}
    with pytest.raises(ValueError, match="scratch_dir"):
        stage_timing.validate_record(denies_copy, timing_root=str(root))

    silent = _valid_payload(root)
    silent["provenance_flags"] = {}
    stage_timing.validate_record(silent, timing_root=str(root))


def test_rejects_non_finite_or_negative_seconds(tmp_path):
    root = _timing_root(tmp_path)
    for seconds in (float("nan"), float("inf"), -0.5):
        bad = _valid_payload(root)
        bad["stages"][0]["seconds"] = seconds
        with pytest.raises(ValueError, match="seconds"):
            stage_timing.validate_record(bad, timing_root=str(root))

    unmeasured = _valid_payload(root)
    unmeasured["stages"][0]["duration_source"] = "not_taken"
    unmeasured["stages"][0]["seconds"] = float("nan")
    with pytest.raises(ValueError, match="not_taken"):
        stage_timing.validate_record(unmeasured, timing_root=str(root))


def test_rejects_inner_stage_with_unknown_parent_or_bad_functions(tmp_path):
    root = _timing_root(tmp_path)
    orphan = _valid_payload(root)
    for row in orphan["stages"]:
        if row["name"] == "stitch_scoring":
            row["parent"] = "nowhere"
    with pytest.raises(ValueError, match="parent"):
        stage_timing.validate_record(orphan, timing_root=str(root))

    for functions in ("score", [1], ["score", None]):
        bad = _valid_payload(root)
        for row in bad["stages"]:
            if row["name"] == "stitch_scoring":
                row["functions"] = functions
        with pytest.raises(ValueError, match="functions"):
            stage_timing.validate_record(bad, timing_root=str(root))

    named = _valid_payload(root)
    for row in named["stages"]:
        if row["name"] == "stitch_scoring":
            row["functions"] = ["score", "score_again"]
    stage_timing.validate_record(named, timing_root=str(root))
    stage_timing.validate_record(_valid_payload(root), timing_root=str(root))


def test_accepts_optional_stack_z_match_and_checks_it(tmp_path):
    root = _timing_root(tmp_path)
    stage_timing.validate_record(_valid_payload(root), timing_root=str(root))

    with_row = _valid_payload(root)
    with_row["stages"].append(_stage("stack_z_match", root, parent="stack", count=1))
    stage_timing.validate_record(with_row, timing_root=str(root))

    no_parent = _valid_payload(root)
    no_parent["stages"].append(_stage("stack_z_match", root, count=1))
    with pytest.raises(ValueError, match="parent"):
        stage_timing.validate_record(no_parent, timing_root=str(root))

    no_count = _valid_payload(root)
    no_count["stages"].append(_stage("stack_z_match", root, parent="stack", count="1"))
    with pytest.raises(ValueError, match="count"):
        stage_timing.validate_record(no_count, timing_root=str(root))


def _row(payload, name):
    return next(row for row in payload["stages"] if row["name"] == name)


def _entries(payload, name):
    return _row(payload, name)["per_slice"]


def test_rejects_collapsed_stage_without_per_slice(tmp_path):
    root = _timing_root(tmp_path)
    stage_timing.validate_record(_valid_payload(root), timing_root=str(root))
    for name in _PER_SLICE_STAGES:
        bare = _valid_payload(root)
        del _row(bare, name)["per_slice"]
        with pytest.raises(ValueError, match="per_slice"):
            stage_timing.validate_record(bare, timing_root=str(root))

    not_a_list = _valid_payload(root)
    _row(not_a_list, "stitch")["per_slice"] = "45, 46, 47"
    with pytest.raises(ValueError, match="per_slice"):
        stage_timing.validate_record(not_a_list, timing_root=str(root))


def test_rejects_per_slice_that_does_not_cover_the_timed_slices(tmp_path):
    root = _timing_root(tmp_path)
    for name in _PER_SLICE_STAGES:
        for slice_ids in ((45, 46), (46, 47), (46, 45, 47), (45, 46, 47, 48), (45, 45, 46, 47), ()):
            partial = _valid_payload(root)
            row = _row(partial, name)
            row["per_slice"] = _per_slice_entries(name, root, row["seconds"], row["device"], slice_ids)
            with pytest.raises(ValueError, match="per_slice"):
                stage_timing.validate_record(partial, timing_root=str(root))

    inner = _valid_payload(root)
    scoring = _row(inner, "stitch_scoring")
    scoring["per_slice"] = _per_slice_entries("stitch_scoring", root, scoring["seconds"], "cpu")
    stage_timing.validate_record(inner, timing_root=str(root))
    scoring["per_slice"] = _per_slice_entries("stitch_scoring", root, scoring["seconds"], "cpu", (45, 46))
    with pytest.raises(ValueError, match="per_slice"):
        stage_timing.validate_record(inner, timing_root=str(root))


def test_rejects_per_slice_seconds_that_do_not_sum(tmp_path):
    root = _timing_root(tmp_path)
    close = _valid_payload(root)
    _entries(close, "illumination")[0]["seconds"] += 1e-9
    stage_timing.validate_record(close, timing_root=str(root))

    drifted = _valid_payload(root)
    _entries(drifted, "illumination")[0]["seconds"] += 0.5
    with pytest.raises(ValueError, match="per_slice"):
        stage_timing.validate_record(drifted, timing_root=str(root))

    for key in ("argv", "device", "seconds", "slice_id", "flags", "output_path"):
        missing = _valid_payload(root)
        del _entries(missing, "stitch")[1][key]
        with pytest.raises(ValueError, match="per_slice"):
            stage_timing.validate_record(missing, timing_root=str(root))

    for key, value in (
        ("seconds", "0.0333"),
        ("seconds", float("nan")),
        ("seconds", float("inf")),
        ("seconds", -0.1),
        ("seconds", True),
        ("device", ""),
        ("argv", "linum"),
        ("flags", []),
        ("output_path", "/etc/passwd"),
    ):
        malformed = _valid_payload(root)
        _entries(malformed, "attenuation")[2][key] = value
        with pytest.raises(ValueError, match="per_slice"):
            stage_timing.validate_record(malformed, timing_root=str(root))


def test_rejects_collapsed_device_that_hides_a_mix(tmp_path):
    root = _timing_root(tmp_path)

    def with_devices(name, devices, row_device):
        payload = _valid_payload(root)
        for entry, device in zip(_entries(payload, name), devices, strict=True):
            entry["device"] = device
        _row(payload, name)["device"] = row_device
        return payload

    for name in _PER_SLICE_STAGES:
        stage_timing.validate_record(with_devices(name, ["cuda", "cuda", "cuda"], "cuda"), timing_root=str(root))
        stage_timing.validate_record(with_devices(name, ["cuda", "cuda", "cpu"], "mixed"), timing_root=str(root))
        for devices, row_device in (
            (["cuda", "cuda", "cpu"], "cuda"),
            (["cuda", "cuda", "cpu"], "cpu"),
            (["cuda", "cuda", "cuda"], "mixed"),
            (["cpu", "cpu", "cpu"], "cuda"),
        ):
            with pytest.raises(ValueError, match="device"):
                stage_timing.validate_record(with_devices(name, devices, row_device), timing_root=str(root))


def test_rejects_common_space_scope_wider_than_the_timed_slices(tmp_path):
    root = _timing_root(tmp_path)
    stage_timing.validate_record(_valid_payload(root), timing_root=str(root))

    missing = _valid_payload(root)
    del _row(missing, "common_space")["scope"]
    with pytest.raises(ValueError, match="scope"):
        stage_timing.validate_record(missing, timing_root=str(root))

    for scope in (
        "slices 45 to 47",
        _scope(slice_count=67),
        _scope(slice_ids=list(range(5, 72)), slice_count=67, input_count=67),
        _scope(slice_ids=[45, 46], slice_count=2, input_count=2),
        _scope(slice_ids=[47, 46, 45]),
        _scope(input_count=67),
        _scope(input_count=2),
        _scope(slice_count=True, input_count=True),
        {"slice_ids": [45, 46, 47], "slice_count": 3},
    ):
        wide = _valid_payload(root)
        _row(wide, "common_space")["scope"] = scope
        with pytest.raises(ValueError, match="scope"):
            stage_timing.validate_record(wide, timing_root=str(root))


def test_rejects_stack_device_that_contradicts_its_argv(tmp_path):
    root = _timing_root(tmp_path)

    def stack_record(argv, device, z_match_device=None):
        payload = _valid_payload(root)
        row = _row(payload, "stack")
        row["argv"] = argv
        row["device"] = device
        if z_match_device is not None:
            payload["stages"].append(_stage("stack_z_match", root, parent="stack", count=1, device=z_match_device))
        return payload

    plain = ["linum-stack-slices-motor", "--no-use_gpu"]
    gpu = ["linum-stack-slices-motor", "--use_gpu"]
    for argv, z_match_device, accepted, refused in (
        (plain, None, "cpu", ("mixed", "unknown", "cuda")),
        (gpu, None, "unknown", ("cpu", "mixed")),
        (gpu, "cpu", "cpu", ("mixed", "unknown")),
        (gpu, "cuda", "mixed", ("cpu", "unknown")),
    ):
        stage_timing.validate_record(stack_record(argv, accepted, z_match_device), timing_root=str(root))
        for device in refused:
            with pytest.raises(ValueError, match=r"stack_z_match|stack device"):
                stage_timing.validate_record(stack_record(argv, device, z_match_device), timing_root=str(root))


def test_rejects_malformed_thread_env(tmp_path):
    root = _timing_root(tmp_path)
    mixed = _valid_payload(root)
    mixed["thread_env"] = {"OMP_NUM_THREADS": "5", "MKL_NUM_THREADS": 5}
    stage_timing.validate_record(mixed, timing_root=str(root))

    for env in ({"OMP_NUM_THREADS": [5]}, {"OMP_NUM_THREADS": True}, {"OMP_NUM_THREADS": 2.5}, {5: "x"}):
        bad = _valid_payload(root)
        bad["thread_env"] = env
        with pytest.raises(ValueError, match="thread_env"):
            stage_timing.validate_record(bad, timing_root=str(root))


def test_device_id_inequality(monkeypatch):
    class _Stat:
        def __init__(self, dev):
            self.st_dev = dev

    def fake_stat(path):
        return _Stat(1 if str(path).endswith("a") else 2)

    monkeypatch.setattr(stage_timing.os, "stat", fake_stat)
    assert stage_timing.same_filesystem("disk-a", "disk-b") is False
    assert stage_timing.same_filesystem("disk-b", "other-b") is True


_REPO = Path(__file__).resolve().parents[2]


def _call_name(node: ast.AST) -> str:
    if not isinstance(node, ast.Call):
        return ""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _parent_map(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _span_call(node: ast.AST) -> ast.Call | None:
    if not isinstance(node, ast.With):
        return None
    for item in node.items:
        call = item.context_expr
        if _call_name(call) == "span" and isinstance(call, ast.Call):
            return call
    return None


def _keyword_str(call: ast.Call, name: str) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == name and isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
            return keyword.value.value
    return None


def _span_stage(call: ast.Call) -> str | None:
    if call.args and isinstance(call.args[0], ast.Constant) and isinstance(call.args[0].value, str):
        return call.args[0].value
    return _keyword_str(call, "stage")


def _enclosing_span(node: ast.AST, parents: dict[ast.AST, ast.AST], stage: str) -> ast.Call | None:
    current = parents.get(node)
    while current is not None:
        call = _span_call(current)
        if call is not None and _span_stage(call) == stage:
            return call
        current = parents.get(current)
    return None


def test_stitch_scoring_wraps_only_the_refinements_call():
    src = (_REPO / "scripts/stitching/linum_stitch_3d_refined.py").read_text(encoding="utf-8")
    pre, sep, _rest = src.partition("import numpy")
    assert sep
    assert "import linumpy.config.threads" in pre
    tree = ast.parse(src)
    parents = _parent_map(tree)
    hits = [node for node in ast.walk(tree) if _call_name(node) == "compute_registration_refinements"]
    assert hits
    for node in hits:
        assert isinstance(node, ast.Call)
        span_call = _enclosing_span(node, parents, "stitch_scoring")
        assert span_call is not None, "compute_registration_refinements is outside stitch_scoring"
        assert _keyword_str(span_call, "device") == "cpu"
        assert [keyword.arg for keyword in span_call.keywords] == ["device"]
        assert all(keyword.arg != "use_gpu" for keyword in node.keywords)


def _is_level0_materialization(node: ast.AST) -> bool:
    call: ast.AST = node
    if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "astype":
        call = call.func.value
    if not isinstance(call, ast.Call) or not call.args:
        return False
    func = call.func
    if not (
        isinstance(func, ast.Attribute) and func.attr == "array" and isinstance(func.value, ast.Name) and func.value.id == "np"
    ):
        return False
    arg = call.args[0]
    if not isinstance(arg, ast.Subscript) or not isinstance(arg.value, ast.Name):
        return False
    if arg.value.id not in {"vol", "first_vol"}:
        return False
    sl = arg.slice
    return isinstance(sl, ast.Slice) and sl.lower is None and sl.upper is None and sl.step is None


def _is_zero(node: ast.AST | None) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and node.value == 0
    )


def test_stack_reads_stay_apart_from_the_resample():
    src = (_REPO / "scripts/stacking/linum_stack_slices_motor.py").read_text(encoding="utf-8")
    pre, sep, _rest = src.partition("import numpy")
    assert sep
    assert "import linumpy.config.threads" in pre
    assert "configure_all_libraries" in pre
    tree = ast.parse(src)
    parents = _parent_map(tree)
    loads = [node.value for node in ast.walk(tree) if isinstance(node, ast.Assign) and _is_level0_materialization(node.value)]
    assert len(loads) == 3
    for node in loads:
        span_call = _enclosing_span(node, parents, "stack_reads")
        assert span_call is not None, "level-0 materialization is outside stack_reads"
        assert _keyword_str(span_call, "device") == "cpu"
        assert [keyword.arg for keyword in span_call.keywords] == ["device"]
    for node in ast.walk(tree):
        if _call_name(node) == "read_omezarr":
            assert _enclosing_span(node, parents, "stack_reads") is None
    for target, function_name in (
        ("apply_rigid_euler_padded", "apply_rigid_euler_padded"),
        ("apply_transform_to_volume", "apply_transform_to_volume"),
    ):
        hits = [node for node in ast.walk(tree) if _call_name(node) == target]
        assert hits
        for node in hits:
            assert isinstance(node, ast.Call)
            span_call = _enclosing_span(node, parents, "per_plane_resample")
            assert span_call is not None, f"{target} is outside per_plane_resample"
            assert _keyword_str(span_call, "device") == "cpu"
            assert _keyword_str(span_call, "function") == function_name
            assert [keyword.arg for keyword in span_call.keywords] == ["device", "function"]
    skipped = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _call_name(node) != "add":
            continue
        stage = node.args[0].value if node.args and isinstance(node.args[0], ast.Constant) else _keyword_str(node, "stage")
        seconds = node.args[1] if len(node.args) > 1 else None
        if stage == "per_plane_resample" and _is_zero(seconds) and _keyword_str(node, "function") == "skipped":
            assert _keyword_str(node, "device") == "cpu"
            skipped.append(node)
    assert len(skipped) == 2


def test_flush_refuses_ndarray(tmp_path, monkeypatch):
    timing_path = tmp_path / "stages.jsonl"
    monkeypatch.setenv("LINUMPY_STAGE_TIMING", str(timing_path))
    stage_timing.add("diffusion_blend", 0.1, device="cpu", weights=np.zeros((2, 2)))
    with pytest.raises(TypeError):
        stage_timing.flush()
    assert not timing_path.exists()
