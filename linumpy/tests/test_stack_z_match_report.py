"""Tests for the stack Z-match backend report.

The stack row's device comes from what the stack process reported. These tests
follow one report from ``resolve_z_match_backend`` through the stack script's
``stack_z_match`` span to the label the hot-path driver puts on the stack row.
"""

import ast
import json
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

import linumpy.gpu as gpu
from linumpy.mosaic import stacking
from linumpy.utils import stage_timing
from scripts.timing import linum_time_hot_path as driver

_REPO = Path(__file__).resolve().parents[2]
_STACK_SCRIPT = _REPO / "scripts/stacking/linum_stack_slices_motor.py"


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


@pytest.fixture
def stand_in_cupy(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, ...]]:
    """Install a cupy stand-in whose ``asarray`` records the shapes it receives."""
    calls: list[tuple[int, ...]] = []

    def asarray(values: Any) -> np.ndarray:
        calls.append(tuple(np.shape(values)))
        return np.asarray(values)

    module = types.ModuleType("cupy")
    module.__dict__["asarray"] = asarray
    monkeypatch.setitem(sys.modules, "cupy", module)
    return calls


def _backend() -> Callable[[bool], str]:
    found = getattr(stacking, "resolve_z_match_backend", None)
    assert callable(found), "resolve_z_match_backend"
    return cast(Callable[[bool], str], found)


def _driver_fn(name: str) -> Callable[..., Any]:
    found = getattr(driver, name, None)
    assert callable(found), name
    return cast(Callable[..., Any], found)


def _volumes() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    fixed = rng.random((40, 24, 24)).astype(np.float32)
    moving = np.concatenate([fixed[-12:], rng.random((28, 24, 24)).astype(np.float32)])
    return fixed, moving


def _find(fixed: np.ndarray, moving: np.ndarray, *, use_gpu: bool) -> tuple[int, float]:
    return stacking.find_z_overlap(fixed, moving, 0.1, 0.05, 5.0, use_gpu=use_gpu)


def test_backend_helper_decides_find_z_overlap_and_label(
    monkeypatch: pytest.MonkeyPatch, stand_in_cupy: list[tuple[int, ...]]
):
    backend = _backend()
    monkeypatch.setattr(gpu, "GPU_AVAILABLE", False)
    assert backend(True) == "cpu"
    assert backend(False) == "cpu"
    monkeypatch.setattr(gpu, "GPU_AVAILABLE", True)
    assert backend(False) == "cpu"
    assert backend(True) == "cuda"

    fixed, moving = _volumes()
    cpu_result = _find(fixed, moving, use_gpu=False)
    assert stand_in_cupy == []
    gpu_result = _find(fixed, moving, use_gpu=True)
    assert len(stand_in_cupy) == 2, "the stand-in asarray was not selected by the backend answer"
    assert gpu_result == cpu_result

    monkeypatch.setattr(gpu, "GPU_AVAILABLE", False)
    stand_in_cupy.clear()
    assert _find(fixed, moving, use_gpu=True) == cpu_result
    assert stand_in_cupy == []


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


def _ancestors(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> list[ast.AST]:
    found: list[ast.AST] = []
    current = parents.get(node)
    while current is not None:
        found.append(current)
        current = parents.get(current)
    return found


def _is_zero_float(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, float) and node.value == 0.0


def _add_stage(node: ast.Call) -> str | None:
    if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
        return node.args[0].value
    return _keyword_str(node, "stage")


def test_stack_script_reports_z_match_at_both_call_sites():
    tree = ast.parse(_STACK_SCRIPT.read_text(encoding="utf-8"))
    parents = _parent_map(tree)
    sites = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and _call_name(node) == "find_z_overlap"]
    assert len(sites) == 2
    for node in sites:
        span_call = _enclosing_span(node, parents, "stack_z_match")
        assert span_call is not None, "find_z_overlap is outside stack_z_match"
        assert sorted(keyword.arg or "" for keyword in span_call.keywords) == ["device", "function"]
        assert _keyword_str(span_call, "function") == "find_z_overlap"
        device = next(keyword.value for keyword in span_call.keywords if keyword.arg == "device")
        assert isinstance(device, ast.Call)
        assert _call_name(device) == "resolve_z_match_backend"

    markers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _call_name(node) == "add" and _add_stage(node) == "stack_z_match"
    ]
    assert len(markers) == 1
    marker = markers[0]
    assert len(marker.args) > 1 and _is_zero_float(marker.args[1])
    assert _keyword_str(marker, "device") == "cpu"
    assert _keyword_str(marker, "function") == "skipped"
    ancestors = _ancestors(marker, parents)
    assert not any(isinstance(item, ast.For) for item in ancestors), "the marker must sit after the Z-matching loop"
    guards = [item for item in ancestors if isinstance(item, ast.If)]
    assert guards, "the marker is not guarded by an if"
    counter_names = {node.id for node in ast.walk(guards[0].test) if isinstance(node, ast.Name)}
    assert len(counter_names) == 1
    counter = next(iter(counter_names))
    increments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AugAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == counter
        and isinstance(node.op, ast.Add)
    ]
    assert len(increments) == 2
    starts = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == counter for target in node.targets)
        and isinstance(node.value, ast.Constant)
        and node.value.value == 0
    ]
    assert len(starts) == 1


_INNER_FIELDS = (
    "name",
    "argv",
    "seconds",
    "duration_source",
    "device",
    "flags",
    "output_path",
    "parent",
    "count",
    "functions",
)


def _emit_stack_report(path: Path, monkeypatch: pytest.MonkeyPatch, *, z_match: str, use_gpu: bool = True) -> str:
    """Write one stack process's JSONL through the real stage clock and return its text."""
    monkeypatch.setenv("LINUMPY_STAGE_TIMING", str(path))
    with stage_timing.span("stack_reads", device="cpu"):
        pass
    with stage_timing.span("per_plane_resample", device="cpu", function="apply_transform_to_volume"):
        pass
    if z_match == "ran":
        fixed, moving = _volumes()
        with stage_timing.span("stack_z_match", device=_backend()(use_gpu), function="find_z_overlap"):
            _find(fixed, moving, use_gpu=use_gpu)
    elif z_match == "skipped":
        stage_timing.add("stack_z_match", 0.0, device="cpu", function="skipped")
    stage_timing.flush()
    monkeypatch.delenv("LINUMPY_STAGE_TIMING", raising=False)
    return path.read_text(encoding="utf-8")


def _stack_parent(root: Path, *flags: str) -> dict[str, object]:
    output = str(root / "stack")
    return {
        "name": "stack",
        "argv": ["linum-stack-slices-motor", "slices", "shifts.csv", output, *flags],
        "flags": {"use_gpu": "--use_gpu" in flags},
        "output_path": output,
    }


def test_reported_backend_reaches_the_stack_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stand_in_cupy: list[tuple[int, ...]]
):
    collect = _driver_fn("collect_inner_stages")
    stack_device = _driver_fn("stack_device")

    def label(name: str, *, z_match: str, available: bool, flags: tuple[str, ...] = ("--use_gpu",)) -> tuple[str, list[str]]:
        monkeypatch.setattr(gpu, "GPU_AVAILABLE", available)
        text = _emit_stack_report(tmp_path / f"{name}.jsonl", monkeypatch, z_match=z_match, use_gpu="--use_gpu" in flags)
        parent = _stack_parent(tmp_path, *flags)
        inners = collect(parent, "", text)
        assert isinstance(inners, list)
        argv = parent["argv"]
        assert isinstance(argv, list)
        return stack_device(argv, inners), [str(row["name"]) for row in inners]

    cuda_label, cuda_names = label("cuda", z_match="ran", available=True)
    assert len(stand_in_cupy) == 2
    assert cuda_label == "mixed"
    assert cuda_names == ["stack_reads", "per_plane_resample", "stack_z_match"]

    cpu_label, cpu_names = label("cpu", z_match="ran", available=False)
    assert cpu_label == "cpu"
    assert "stack_z_match" in cpu_names

    skipped_label, skipped_names = label("skipped", z_match="skipped", available=True)
    assert skipped_label == "cpu"
    assert "stack_z_match" in skipped_names

    silent_label, silent_names = label("silent", z_match="none", available=True)
    assert silent_label == "unknown"
    assert silent_names == ["stack_reads", "per_plane_resample"]

    off_label, _off_names = label("off", z_match="none", available=True, flags=("--no-use_gpu",))
    assert off_label == "cpu"

    monkeypatch.setattr(gpu, "GPU_AVAILABLE", True)
    text = _emit_stack_report(tmp_path / "rows.jsonl", monkeypatch, z_match="ran")
    rows = {row["name"]: row for row in collect(_stack_parent(tmp_path, "--use_gpu"), "", text)}
    z_match = rows["stack_z_match"]
    for field in _INNER_FIELDS:
        assert field in z_match
    assert z_match["parent"] == "stack"
    assert z_match["count"] == 1
    assert z_match["functions"] == ["find_z_overlap"]
    assert z_match["duration_source"] == "perf_counter"
    assert json.loads(text.splitlines()[-1])["stage"] == "stack_z_match"
