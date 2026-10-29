"""Tests for the hot-path timing driver helpers."""

import ast
import inspect
import json
import subprocess
import sys
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

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


def _driver_fn(name: str) -> Callable[..., Any]:
    found = getattr(driver, name, None)
    assert callable(found), name
    return cast(Callable[..., Any], found)


_WORKFLOW_CONFIG = """
params {
    scratch_dir = true
    fix_illum_per_z_fit = false
    fix_illum_darkfield = false
    fix_illum_multi_gpu = false
    z_overlap_from_registration = false
    use_gpu = true
    gpu_count = 1
    processes = 4
    max_cpus = 16
    clip_percentile_upper = 99.0
    fix_illum_fit_max_samples = 100
    fix_illum_max_iterations = 10
    fix_illum_darkfield_percentile = 5
    fix_illum_smoothness_flatfield = 0.05
    stitch_overlap_fraction = 0.2
    stitch_blending_method = 'average'
    max_blend_refinement_px = 3
    compensate_attenuation_method = 'faber'
    compensate_attenuation_min_bias = 0.05
    compensate_attenuation_mask_sigma = 2.0
    compensate_attenuation_k = 10
    compensate_attenuation_zshift = 3
    compensate_attenuation_strength = 0.3
    compensate_attenuation_snr_threshold_db = 6.0
    align_to_ras_enabled = false
    annotated_label_every = 2
    annotated_show_lines = false
    ras_input_orientation = ''
}
profiles {
    aggressive {
        params {
            z_overlap_from_registration = true
        }
    }
}
"""

_SERVER_CONFIG = """
scratch_dir = false
fix_illum_per_z_fit = true
fix_illum_darkfield = true
fix_illum_multi_gpu = true
use_gpu = true
gpu_count = 2
processes = 8
max_cpus = 44
clip_percentile_upper = 99.9
stitch_blending_method = 'diffusion'
max_blend_refinement_px = 10
compensate_attenuation_method = 'li'
align_to_ras_enabled = true
annotated_label_every = 1
"""


def _stage_inputs(root: Path) -> dict[str, object]:
    illumination: dict[int, str] = {}
    stitch: dict[int, str] = {}
    attenuation: dict[int, str] = {}
    for slice_id in (45, 46, 47):
        illumination[slice_id] = f"/scratch_nvme/workspace/work/resampled/mosaic_grid_z{slice_id}_resampled.ome.zarr"
        stitch[slice_id] = (
            f"/scratch_nvme/workspace/sub-21/output/{slice_id}/fix_illumination_basic/"
            f"mosaic_grid_z{slice_id}_illum_fix.ome.zarr"
        )
        attenuation[slice_id] = f"/scratch_nvme/workspace/work/crop/slice_z{slice_id}_crop_interface.ome.zarr"
    return {
        "timing_root": root,
        "illumination": illumination,
        "stitch": stitch,
        "attenuation": attenuation,
        "common_space": [
            "/scratch_nvme/workspace/work/norm/slice_z45_normalize.ome.zarr",
            "/scratch_nvme/workspace/work/norm/slice_z46_normalize.ome.zarr",
            "/scratch_nvme/workspace/work/norm/slice_z47_normalize.ome.zarr",
        ],
        "cpu_count": 44,
    }


def _merged_server_config() -> dict[str, object]:
    merge = _driver_fn("merge_config")
    merged = merge(_WORKFLOW_CONFIG, _SERVER_CONFIG)
    assert isinstance(merged, dict)
    return merged


def test_merge_prefers_server():
    merged = _merged_server_config()
    assert merged["scratch_dir"] is False
    assert merged["fix_illum_per_z_fit"] is True
    assert merged["z_overlap_from_registration"] is False
    with pytest.raises(ValueError, match="scratch_dir"):
        _driver_fn("merge_config")("scratch_dir = true", "scratch_dir = not-a-value")


def test_params_block_ignores_comment_apostrophe():
    workflow = """
params {
    // CuPy's default lives in a comment: {input}
    scratch_dir = true
    use_gpu = false
}
env {
    CUPY_ACCELERATORS = params.cupy_accelerators?.trim() ? params.cupy_accelerators : 'cub'
}
"""
    merged = _driver_fn("merge_config")(workflow, "scratch_dir = false\n")
    assert merged["scratch_dir"] is False
    assert merged["use_gpu"] is False
    assert "CUPY_ACCELERATORS" not in merged


def test_real_workflow_config_parses():
    workflow = Path(__file__).resolve().parents[2].joinpath("workflows/reconst_3d/nextflow.config").read_text(encoding="utf-8")
    merged = _driver_fn("merge_config")(workflow, "scratch_dir = false\n")
    assert merged["scratch_dir"] is False
    assert merged["use_gpu"] is True
    assert "CUPY_ACCELERATORS" not in merged


def test_common_space_uses_staged_inputs(tmp_path: Path):
    work = tmp_path / "work"
    inputs = work / "inputs"
    inputs.mkdir(parents=True)
    real = tmp_path / "real" / "slice_z45_normalize.ome.zarr"
    real.mkdir(parents=True)
    (inputs / real.name).symlink_to(real)
    skipped_real = tmp_path / "bring_to_common_space" / "slice_z46_normalize.ome.zarr"
    skipped_real.parent.mkdir()
    skipped_real.mkdir()
    (inputs / skipped_real.name).symlink_to(skipped_real)
    script = "linum-align-mosaics-3d-from-shifts inputs shifts_xy.csv common_space\n"
    found = _driver_fn("resolve_common_space_inputs")(script, work)
    assert found == [str(real.resolve())]


def test_staged_token_resolves_beside_the_command(tmp_path: Path):
    work = tmp_path / "task"
    work.mkdir()
    affine = work / "global_affine.npy"
    affine.write_text("x", encoding="utf-8")
    found = _driver_fn("_materialize_staged")("global_affine.npy", work)
    assert found == str(affine.resolve())


def test_illumination_argv_uses_server_flags(tmp_path: Path):
    root = tmp_path / "timing"
    root.mkdir()
    inputs = _stage_inputs(root)
    stages = _driver_fn("build_stage_commands")(_merged_server_config(), inputs)
    assert isinstance(stages, list)
    illumination = next(
        stage
        for stage in stages
        if isinstance(stage, dict) and stage.get("name") == "illumination" and "z45" in stage.get("argv", [])
    )
    argv = illumination["argv"]
    assert isinstance(argv, list)
    assert argv[0] == "linum-fix-illumination-basic"
    assert "--per_z_fit" in argv
    assert "--no-per_z_fit" not in argv
    assert "--use_darkfield" in argv
    assert "--use_gpu" in argv
    assert argv[argv.index("--n_processes") + 1] == "8"
    assert argv[argv.index("--slice_id") + 1] == "z45"
    resampled_45 = "/scratch_nvme/workspace/work/resampled/mosaic_grid_z45_resampled.ome.zarr"
    assert resampled_45 in argv
    assert "illum_fix.ome.zarr" not in argv[1]
    output_path = illumination["output_path"]
    assert isinstance(output_path, str)
    assert Path(output_path).is_relative_to(root)
    assert output_path in argv


def test_stages_keep_zip_and_screenshots_separate(tmp_path: Path):
    root = tmp_path / "timing"
    root.mkdir()
    inputs = _stage_inputs(root)
    stages = _driver_fn("build_stage_commands")(_merged_server_config(), inputs)
    assert isinstance(stages, list)
    names = [stage.get("name") for stage in stages if isinstance(stage, dict)]
    assert "zip" in names
    assert "screenshot" in names
    assert "screenshot_annotated" in names
    assert "align_to_ras" not in names
    illumination_outputs = [
        stage["output_path"]
        for stage in stages
        if isinstance(stage, dict) and stage.get("name") == "illumination" and isinstance(stage.get("output_path"), str)
    ]
    stitches = [stage for stage in stages if isinstance(stage, dict) and stage.get("name") == "stitch"]
    assert len(stitches) == 3
    existing_illum = "/scratch_nvme/workspace/sub-21/output/45/fix_illumination_basic/mosaic_grid_z45_illum_fix.ome.zarr"
    for stage in stitches:
        argv = stage["argv"]
        assert isinstance(argv, list)
        assert not any(token in argv for token in illumination_outputs)
        assert existing_illum in argv or any("illum_fix.ome.zarr" in token for token in argv if isinstance(token, str))
        assert "linum-align-to-ras" not in argv
    assert all("linum-align-to-ras" not in stage.get("argv", []) for stage in stages if isinstance(stage, dict))


def test_attenuation_input_is_not_the_output(tmp_path: Path):
    command = (
        "linum-compensate-attenuation-inplace "
        "/scratch_nvme/workspace/work/aa/bb/slice_z45_crop_interface.ome.zarr "
        "/scratch_nvme/workspace/sub-21/output/compensate_attenuation/slice_z45_attn_corr.ome.zarr "
        "--method li\n"
    )
    resolved = _driver_fn("resolve_attenuation_input")(command, 45)
    assert isinstance(resolved, str)
    assert resolved.endswith("slice_z45_crop_interface.ome.zarr")
    assert not resolved.endswith("slice_z45_attn_corr.ome.zarr")
    root = tmp_path / "timing"
    root.mkdir()
    inputs = _stage_inputs(root)
    raw_attenuation = inputs["attenuation"]
    assert isinstance(raw_attenuation, dict)
    inputs["attenuation"] = {str(key): value for key, value in raw_attenuation.items()}
    attenuation_inputs = inputs["attenuation"]
    assert isinstance(attenuation_inputs, dict)
    attenuation_inputs["45"] = resolved
    stages = _driver_fn("build_stage_commands")(_merged_server_config(), inputs)
    assert isinstance(stages, list)
    attenuation = next(
        stage
        for stage in stages
        if isinstance(stage, dict) and stage.get("name") == "attenuation" and stage.get("slice_id") == 45
    )
    argv = attenuation["argv"]
    assert isinstance(argv, list)
    assert "--method" in argv
    assert argv[argv.index("--method") + 1] == "li"
    assert isinstance(argv[1], str)
    assert not argv[1].endswith("slice_z45_attn_corr.ome.zarr")


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


def test_refuses_stitch_without_inners(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "timing"
    root.mkdir()
    record = root / "hot_path.json"
    calls: list[object] = []

    def fake_finalize(payload: dict[str, object], timing_root: Path = driver.TIMING_ROOT) -> dict[str, object]:
        calls.append(payload)
        record.write_text("{}\n", encoding="utf-8")
        return payload

    monkeypatch.setattr(driver, "finalize", fake_finalize)
    parent = {
        "name": "stitch",
        "argv": ["linum-stitch-3d-refined", "in.zarr", str(root / "stitch")],
        "flags": {"blending_method": "diffusion"},
        "output_path": str(root / "stitch"),
    }
    jsonl = json.dumps({"stage": "stitch_scoring", "seconds": 1.0, "count": 1, "device": "cpu"}) + "\n"
    with pytest.raises(ValueError, match="diffusion_blend"):
        _driver_fn("collect_inner_stages")(parent, "child stdout", jsonl)
    assert calls == []
    assert not record.exists()


def test_accepts_inner_jsonl(tmp_path: Path):
    root = tmp_path / "timing"
    root.mkdir()
    stitch_out = str(root / "stitch")
    stack_out = str(root / "stack")
    stitch_parent = {
        "name": "stitch",
        "argv": ["linum-stitch-3d-refined", "in.zarr", stitch_out],
        "flags": {"blending_method": "diffusion"},
        "output_path": stitch_out,
    }
    stack_parent = {
        "name": "stack",
        "argv": ["linum-stack-slices-motor", "slices", "shifts.csv", stack_out],
        "flags": {"use_gpu": True},
        "output_path": stack_out,
    }
    stitch_jsonl = "\n".join(
        [
            json.dumps({"stage": "stitch_scoring", "seconds": 1.5, "count": 2, "device": "cpu"}),
            json.dumps({"stage": "diffusion_blend", "seconds": 0.4, "count": 3, "device": "cpu", "functions": ["blend"]}),
        ]
    )
    stack_jsonl = "\n".join(
        [
            json.dumps({"stage": "stack_reads", "seconds": 2.0, "count": 3, "device": "cpu"}),
            json.dumps(
                {
                    "stage": "per_plane_resample",
                    "seconds": 4.0,
                    "count": 2,
                    "device": "cpu",
                    "functions": ["apply_transform_to_volume", "apply_rigid_euler_padded"],
                }
            ),
        ]
    )
    collect = _driver_fn("collect_inner_stages")
    inners = collect(stitch_parent, "stitch stdout", stitch_jsonl)
    inners.extend(collect(stack_parent, "stack stdout", stack_jsonl))
    assert isinstance(inners, list)
    by_name: dict[str, dict[str, object]] = {}
    for row in inners:
        assert isinstance(row, dict)
        for field in _INNER_FIELDS:
            assert field in row
        name = row["name"]
        assert isinstance(name, str)
        by_name[name] = row
    scoring = by_name["stitch_scoring"]
    assert scoring["argv"] == stitch_parent["argv"]
    assert scoring["flags"] == stitch_parent["flags"]
    assert scoring["output_path"] == stitch_out
    assert scoring["seconds"] == 1.5
    assert scoring["count"] == 2
    assert scoring["device"] == "cpu"
    assert scoring["functions"] == []
    assert scoring["parent"] == "stitch"
    assert scoring["duration_source"] == "perf_counter"
    assert by_name["diffusion_blend"]["functions"] == ["blend"]
    resample = by_name["per_plane_resample"]
    assert resample["functions"] == ["apply_transform_to_volume", "apply_rigid_euler_padded"]
    assert resample["argv"] == stack_parent["argv"]
    assert resample["flags"] == stack_parent["flags"]
    assert resample["output_path"] == stack_out
    assert resample["parent"] == "stack"
    assert resample["seconds"] == 4.0
    assert resample["count"] == 2
    payload = _valid_payload(root)
    kept = []
    existing = payload["stages"]
    assert isinstance(existing, list)
    for row in existing:
        assert isinstance(row, dict)
        if row.get("name") not in by_name:
            kept.append(row)
    payload["stages"] = [*kept, *inners]
    driver.finalize(payload, timing_root=root)


def test_help_does_not_touch_subject_tree():
    completed = subprocess.run(
        [sys.executable, "scripts/timing/linum_time_hot_path.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "hot_path" in completed.stdout
    assert "45" in completed.stdout
    assert "46" in completed.stdout
    assert "47" in completed.stdout


def test_linum_basic_revision_prefers_checkout(tmp_path: Path):
    direct = '{"url": "https://example.invalid/linum-basic.git", "vcs_info": {"commit_id": "aaa"}}\n'
    checkout = tmp_path / "linum-basic"
    checkout.mkdir()

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        assert argv[0] == "git"
        assert "-C" in argv
        assert str(checkout) in argv
        return subprocess.CompletedProcess(argv, 0, stdout="deadbeef\n", stderr="")

    revise = _driver_fn("linum_basic_revision")
    assert revise(checkout, direct, runner) == "deadbeef"

    def not_checkout(argv: list[str]) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(128, argv, stderr="not a git repository")

    assert revise(tmp_path / "dist-info", direct, not_checkout) == direct


_LINUMPY_SHA = "4d656d45db4f28c4a367a13e033a6c46a5670bfa"
_LINUM_BASIC_SHA = "65c0a92bc7f263cdd854b1133cb3591c707af430"


def _git_answers(toplevels: dict[str, str], heads: dict[str, str]) -> tuple[Callable[..., Any], list[list[str]]]:
    """Build a git runner that answers per ``-C`` directory and records every call."""
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        calls.append(list(argv))
        assert argv[:2] == ["git", "-C"]
        directory = argv[2]
        if argv[3:] == ["rev-parse", "--show-toplevel"] and directory in toplevels:
            return subprocess.CompletedProcess(argv, 0, stdout=toplevels[directory] + "\n", stderr="")
        if argv[3:] == ["rev-parse", "HEAD"] and directory in heads:
            return subprocess.CompletedProcess(argv, 0, stdout=heads[directory] + "\n", stderr="")
        raise subprocess.CalledProcessError(128, argv, stderr="not a git repository")

    return runner, calls


def test_linum_basic_revision_reads_the_install_record(tmp_path: Path):
    direct = json.dumps({"url": "https://example.invalid/linum-basic.git", "vcs_info": {"commit_id": _LINUM_BASIC_SHA}})

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"git ran although direct_url.json names the commit: {argv}")

    assert _driver_fn("linum_basic_revision")(tmp_path / "site-packages", direct, runner) == _LINUM_BASIC_SHA


def test_linum_basic_revision_uses_its_own_checkout(tmp_path: Path):
    checkout = tmp_path / "linum-basic"
    checkout.mkdir()
    dist = tmp_path / "site-packages"
    dist.mkdir()
    direct = json.dumps({"url": checkout.as_uri(), "dir_info": {"editable": True}})
    runner, calls = _git_answers(
        {str(checkout): str(checkout), str(dist): str(tmp_path)},
        {str(checkout): "deadbeef", str(dist): _LINUMPY_SHA},
    )
    assert _driver_fn("linum_basic_revision")(dist, direct, runner) == "deadbeef"
    assert ["git", "-C", str(dist), "rev-parse", "HEAD"] not in calls


def test_linum_basic_revision_refuses_the_enclosing_repo(tmp_path: Path):
    dist = tmp_path / "site-packages"
    dist.mkdir()
    direct = json.dumps({"url": "https://example.invalid/linum-basic.git"})
    runner, calls = _git_answers({str(dist): str(tmp_path)}, {str(dist): _LINUMPY_SHA})
    revision = _driver_fn("linum_basic_revision")(dist, direct, runner)
    assert revision == direct
    assert revision != _LINUMPY_SHA
    assert all(call[3:] != ["rev-parse", "HEAD"] for call in calls), (
        "HEAD was asked of a directory that is not its own repository"
    )


def test_linum_basic_revision_refuses_an_unrecoverable_revision(tmp_path: Path):
    revise = _driver_fn("linum_basic_revision")
    runner, _calls = _git_answers({}, {})
    for empty in ("", "  \n"):
        with pytest.raises(ValueError, match="linum-basic"):
            revise(tmp_path / "site-packages", empty, runner)
    assert revise(tmp_path / "site-packages", "{not json", runner) == "{not json"


def test_installed_linum_basic_revision_uses_direct_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    direct = json.dumps({"url": "https://example.invalid/linum-basic.git", "vcs_info": {"commit_id": _LINUM_BASIC_SHA}})

    class _Dist:
        def read_text(self, name: str) -> str | None:
            assert name == "direct_url.json"
            return direct

        def locate_file(self, path: str) -> Path:
            return tmp_path / path

    monkeypatch.setattr("importlib.metadata.distribution", lambda name: _Dist())

    def refuse(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"git ran although direct_url.json names the commit: {argv}")

    monkeypatch.setattr(driver, "run_argv", refuse)
    assert _driver_fn("_installed_linum_basic_revision")() == _LINUM_BASIC_SHA


def test_illumination_device_follows_the_printed_branch():
    printed = _driver_fn("illumination_device")
    cuda_line = "field_mode=per_z backend=torch device=cuda"
    assert printed(cuda_line) == "cuda"
    assert printed("field_mode=per_z backend=torch device=None") == "cpu"
    assert printed("field_mode=per_z backend=torch device=cpu") == "cpu"
    assert printed("field_mode=cuda backend=missing-device") == "cpu"
    assert printed("cuda") == "cpu"
    stage_device = _driver_fn("_stage_device")
    assert stage_device({"device_rule": "stdout"}, cuda_line) == "cuda"
    assert stage_device({"device_rule": "stdout"}, "field_mode=per_z backend=torch device=None") == "cpu"
    assert stage_device({"device_rule": "host"}, cuda_line) == "host"
    assert stage_device({"device_rule": "cpu"}, cuda_line) == "cpu"


def test_only_slices_45_to_47_are_timed_and_zip_stays_on_the_host(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "TORCHINDUCTOR_FX_GRAPH_CACHE",
        "TORCHINDUCTOR_CACHE_DIR",
        "LINUMPY_STAGE_TIMING",
    ):
        monkeypatch.delenv(name, raising=False)
    root = tmp_path / "timing"
    root.mkdir()
    stages = _driver_fn("build_stage_commands")(_merged_server_config(), _stage_inputs(root))
    assert isinstance(stages, list)
    by_name: dict[str, list[dict[str, object]]] = {}
    for stage in stages:
        assert isinstance(stage, dict)
        name = stage.get("name")
        assert isinstance(name, str)
        by_name.setdefault(name, []).append(stage)
        output_path = stage.get("output_path")
        assert isinstance(output_path, str)
        assert Path(output_path).is_relative_to(root)
        assert "output/stack" not in output_path
    for name in ("illumination", "stitch", "attenuation"):
        assert {stage.get("slice_id") for stage in by_name[name]} == {45, 46, 47}
    assert set(by_name) >= {"zip", "screenshot", "screenshot_annotated", "common_space", "stack"}
    assert "align_to_ras" not in by_name
    illumination = by_name["illumination"][0]
    assert illumination["device_rule"] == "stdout"
    illum_env = illumination["env"]
    assert isinstance(illum_env, dict)
    assert illum_env.get("CUDA_VISIBLE_DEVICES") == "0,1"
    assert illum_env.get("TORCHINDUCTOR_FX_GRAPH_CACHE") == "1"
    assert by_name["stack"][0]["device_rule"] == "reported"
    for name in ("stitch", "attenuation", "common_space", "zip", "screenshot", "screenshot_annotated"):
        assert by_name[name][0]["device_rule"] == ("host" if name == "zip" else "cpu")
    for name in ("stitch", "attenuation", "common_space", "stack", "zip", "screenshot", "screenshot_annotated"):
        env = by_name[name][0]["env"]
        assert isinstance(env, dict)
        assert "CUDA_VISIBLE_DEVICES" not in env
        assert "TORCHINDUCTOR_FX_GRAPH_CACHE" not in env
    stack_argv = by_name["stack"][0]["argv"]
    assert isinstance(stack_argv, list)
    assert "--no_xy_shift" in stack_argv
    assert "--use_gpu" in stack_argv
    assert "--z_from_registration" not in stack_argv
    attenuation_argv = by_name["attenuation"][0]["argv"]
    assert isinstance(attenuation_argv, list)
    assert "--use_gpu" not in attenuation_argv
    common_argv = by_name["common_space"][0]["argv"]
    assert isinstance(common_argv, list)
    assert "--refine_unreliable" not in common_argv
    assert "linum-align-to-ras" not in common_argv


def _stack_spec(root: Path, server_text: str = _SERVER_CONFIG) -> dict[str, object]:
    merged = _driver_fn("merge_config")(_WORKFLOW_CONFIG, server_text)
    stages = _driver_fn("build_stage_commands")(merged, _stage_inputs(root))
    return next(stage for stage in stages if isinstance(stage, dict) and stage.get("name") == "stack")


def test_stack_with_use_gpu_is_not_planned_as_cpu(tmp_path: Path):
    gpu_root = tmp_path / "gpu"
    gpu_root.mkdir()
    planned = _stack_spec(gpu_root)
    argv = planned["argv"]
    assert isinstance(argv, list)
    assert "--use_gpu" in argv
    assert planned["device_rule"] == "reported"
    assert planned["device_rule"] != "cpu"

    off_text = _SERVER_CONFIG.replace("use_gpu = true", "use_gpu = false")
    assert off_text != _SERVER_CONFIG
    cpu_root = tmp_path / "cpu"
    cpu_root.mkdir()
    cpu_planned = _stack_spec(cpu_root, off_text)
    cpu_argv = cpu_planned["argv"]
    assert isinstance(cpu_argv, list)
    assert "--no-use_gpu" in cpu_argv
    assert "--use_gpu" not in cpu_argv
    assert cpu_planned["device_rule"] == "cpu"


def _time_stack(root: Path, monkeypatch: pytest.MonkeyPatch, rows: list[dict[str, object]]):
    """Run ``_time_spec`` on the stack with a child that writes ``rows`` as its JSONL."""
    root.mkdir()
    spec = _stack_spec(root)

    def stack_child(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        timing = env.get("LINUMPY_STAGE_TIMING")
        assert isinstance(timing, str)
        Path(timing).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(driver, "run_argv", stack_child)
    return _driver_fn("_time_spec")(spec, root)


_STACK_BASE_ROWS: list[dict[str, object]] = [
    {"stage": "stack_reads", "seconds": 1.0, "count": 3, "device": "cpu"},
    {"stage": "per_plane_resample", "seconds": 0.5, "count": 2, "device": "cpu", "functions": ["skipped", "skipped"]},
]


def test_time_spec_labels_the_stack_from_its_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    cuda_row: dict[str, object] = {
        "stage": "stack_z_match",
        "seconds": 2.0,
        "count": 2,
        "device": "cuda",
        "functions": ["find_z_overlap", "find_z_overlap"],
    }
    outer, inners = _time_stack(tmp_path / "cuda", monkeypatch, [*_STACK_BASE_ROWS, cuda_row])
    assert outer["device"] == "mixed"
    by_name = {row["name"]: row for row in inners}
    assert list(by_name) == ["stack_reads", "per_plane_resample", "stack_z_match"]
    assert by_name["stack_z_match"]["parent"] == "stack"
    assert by_name["stack_z_match"]["device"] == "cuda"
    assert by_name["stack_z_match"]["count"] == 2

    skipped_row: dict[str, object] = {
        "stage": "stack_z_match",
        "seconds": 0.0,
        "count": 1,
        "device": "cpu",
        "functions": ["skipped"],
    }
    outer, inners = _time_stack(tmp_path / "skipped", monkeypatch, [*_STACK_BASE_ROWS, skipped_row])
    assert outer["device"] == "cpu"
    assert [row["name"] for row in inners][-1] == "stack_z_match"

    outer, inners = _time_stack(tmp_path / "silent", monkeypatch, list(_STACK_BASE_ROWS))
    assert outer["device"] == "unknown"
    assert [row["name"] for row in inners] == ["stack_reads", "per_plane_resample"]

    with pytest.raises(ValueError, match="per_plane_resample"):
        _time_stack(tmp_path / "missing", monkeypatch, [_STACK_BASE_ROWS[0], cuda_row])


def test_outer_clock_stays_off_the_inner_seconds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LINUMPY_STAGE_TIMING", raising=False)
    root = tmp_path / "timing"
    root.mkdir()
    stages = _driver_fn("build_stage_commands")(_merged_server_config(), _stage_inputs(root))
    assert isinstance(stages, list)
    stitch = next(
        stage for stage in stages if isinstance(stage, dict) and stage.get("name") == "stitch" and stage.get("slice_id") == 45
    )
    blocked = next(
        stage for stage in stages if isinstance(stage, dict) and stage.get("name") == "stitch" and stage.get("slice_id") == 46
    )
    illumination = next(
        stage
        for stage in stages
        if isinstance(stage, dict) and stage.get("name") == "illumination" and stage.get("slice_id") == 45
    )
    stack = next(stage for stage in stages if isinstance(stage, dict) and stage.get("name") == "stack")
    blocked_output = blocked["output_path"]
    assert isinstance(blocked_output, str)
    Path(blocked_output).mkdir(parents=True)
    calls: list[list[str]] = []

    def refuse_child(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        del env
        calls.append(argv)
        raise AssertionError("child started after the output already existed")

    monkeypatch.setattr(driver, "run_argv", refuse_child)
    with pytest.raises(FileExistsError):
        _driver_fn("_time_spec")(blocked, root)
    assert calls == []

    def time_child(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        timing = env.get("LINUMPY_STAGE_TIMING")
        assert isinstance(timing, str)
        timing_path = Path(timing)
        assert timing_path.is_relative_to(root)
        timing_path.write_text(
            "\n".join(
                [
                    json.dumps({"stage": "stitch_scoring", "seconds": 100.0, "count": 1, "device": "cpu"}),
                    json.dumps({"stage": "diffusion_blend", "seconds": 50.0, "count": 1, "device": "cpu"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="field_mode=per_z backend=torch device=cuda", stderr="")

    monkeypatch.setattr(driver, "run_argv", time_child)
    outer, inners = _driver_fn("_time_spec")(stitch, root)
    assert isinstance(outer, dict)
    assert isinstance(inners, list)
    assert outer["duration_source"] == "perf_counter"
    assert outer["device"] == "cpu"
    seconds = outer["seconds"]
    assert isinstance(seconds, float)
    inner_total = 0.0
    by_name: dict[str, dict[str, object]] = {}
    for row in inners:
        assert isinstance(row, dict)
        name = row["name"]
        assert isinstance(name, str)
        by_name[name] = row
        row_seconds = row["seconds"]
        assert isinstance(row_seconds, float)
        inner_total += row_seconds
    assert by_name["stitch_scoring"]["seconds"] == 100.0
    assert by_name["diffusion_blend"]["seconds"] == 50.0
    assert inner_total == 150.0
    assert seconds < inner_total
    assert not (root / "hot_path.json").exists()

    def print_cpu(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        assert "LINUMPY_STAGE_TIMING" not in env
        return subprocess.CompletedProcess(argv, 0, stdout="field_mode=per_z backend=torch device=None\n", stderr="")

    monkeypatch.setattr(driver, "run_argv", print_cpu)
    illum_outer, illum_inners = _driver_fn("_time_spec")(illumination, root)
    assert isinstance(illum_outer, dict)
    assert illum_outer["device"] == "cpu"
    assert illum_outer["duration_source"] == "perf_counter"
    assert illum_inners == []

    def print_cuda(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        assert "LINUMPY_STAGE_TIMING" not in env
        return subprocess.CompletedProcess(argv, 0, stdout="field_mode=per_z backend=torch device=cuda\n", stderr="")

    monkeypatch.setattr(driver, "run_argv", print_cuda)
    cuda_outer, cuda_inners = _driver_fn("_time_spec")(illumination, root)
    assert isinstance(cuda_outer, dict)
    assert cuda_outer["device"] == "cuda"
    assert cuda_inners == []

    def stack_child(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        timing = env.get("LINUMPY_STAGE_TIMING")
        assert isinstance(timing, str)
        Path(timing).write_text(
            json.dumps({"stage": "stack_reads", "seconds": 1.0, "count": 3, "device": "cpu"}) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(driver, "run_argv", stack_child)
    with pytest.raises(ValueError, match="per_plane_resample"):
        _driver_fn("_time_spec")(stack, root)
    assert not (root / "hot_path.json").exists()


def _patch_measurement(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[bool]]:
    """Point ``_run_measurement`` at a tmp tree and return the record path and the check spy."""
    root = tmp_path / "timing"
    root.mkdir()
    record = root / "hot_path.json"
    subject = tmp_path / "sub-21"
    subject.mkdir()
    (subject / "nextflow.config").write_text(_SERVER_CONFIG, encoding="utf-8")
    monkeypatch.setattr(driver, "TIMING_ROOT", root)
    monkeypatch.setattr(driver, "RECORD_PATH", record)
    monkeypatch.setattr(driver, "SUBJECT_ROOT", subject)
    monkeypatch.setattr(driver, "_link_named", lambda directory, names: None)
    monkeypatch.setattr(driver, "_prepare_inputs", lambda cfg: (_stage_inputs(root), [str(root / "shifts_xy.csv")]))
    monkeypatch.setattr(driver, "_git_text", lambda repo, *args: "abc123")
    monkeypatch.setattr(driver, "_installed_linum_basic_revision", lambda: "def456")
    monkeypatch.setattr(
        driver,
        "disk_block",
        lambda stat_fn, findmnt_fn: {
            "tmp_source": "/dev/nvme0n1p1",
            "subject_source": "/dev/nvme0n1p2",
            "work_source": "/dev/nvme0n1p2",
            "tmp_st_dev": 66305,
            "subject_st_dev": 66306,
            "work_st_dev": 66306,
            "same_filesystem": False,
        },
    )
    entered: list[bool] = []
    original_finalize = driver.finalize

    def spy_finalize(payload: dict[str, object], timing_root: Path = driver.TIMING_ROOT) -> dict[str, object]:
        del timing_root
        assert not record.exists()
        checked = original_finalize(payload, timing_root=root)
        assert not record.exists()
        entered.append(True)
        return checked

    monkeypatch.setattr(driver, "finalize", spy_finalize)
    return record, entered


def _inner_row(spec: dict[str, object], output_path: str, name: str, **fields: object) -> dict[str, object]:
    row: dict[str, object] = {
        "name": name,
        "argv": spec.get("argv"),
        "seconds": 0.1,
        "duration_source": "perf_counter",
        "device": "cpu",
        "flags": spec.get("flags"),
        "output_path": output_path,
        "parent": spec.get("name"),
        "count": 1,
        "functions": [],
    }
    row.update(fields)
    return row


def _fake_time_for(mode: dict[str, str], z_match_device: str | None):
    """Build a ``_time_spec`` stand-in. ``z_match_device`` is the device a stack process reports, or None for silence."""

    def fake_time(spec: dict[str, object], timing_root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
        del timing_root
        name = spec.get("name")
        assert isinstance(name, str)
        if mode["value"] == "missing" and name == "stitch":
            raise ValueError("missing stage diffusion_blend")
        output_path = spec.get("output_path")
        assert isinstance(output_path, str)
        if mode["value"] == "escape" and name == "illumination":
            output_path = "/etc/passwd"
        device = "cuda" if name == "illumination" else spec.get("device_rule")
        inners: list[dict[str, object]] = []
        if name == "stitch" and spec.get("slice_id") == 45:
            inners.append(_inner_row(spec, output_path, "stitch_scoring", seconds=1.5, functions=["score"]))
            inners.append(_inner_row(spec, output_path, "diffusion_blend", seconds=0.4, functions=["blend"]))
        if name == "stack":
            inners.append(_inner_row(spec, output_path, "stack_reads", seconds=0.8, count=3))
            inners.append(
                _inner_row(
                    spec,
                    output_path,
                    "per_plane_resample",
                    seconds=0.7,
                    count=2,
                    functions=["apply_transform_to_volume", "skipped"],
                )
            )
            if z_match_device is not None:
                inners.append(
                    _inner_row(
                        spec, output_path, "stack_z_match", seconds=0.3, device=z_match_device, functions=["find_z_overlap"]
                    )
                )
            argv = spec.get("argv")
            assert isinstance(argv, list)
            device = _driver_fn("stack_device")(argv, inners)
        outer = {
            "name": name,
            "argv": spec.get("argv"),
            "seconds": 0.2,
            "duration_source": "perf_counter",
            "device": device,
            "flags": spec.get("flags"),
            "output_path": output_path,
        }
        return outer, inners

    return fake_time


def test_hot_path_json_appears_only_after_the_record_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    record, entered = _patch_measurement(tmp_path, monkeypatch)
    mode = {"value": "missing"}
    monkeypatch.setattr(driver, "_time_spec", _fake_time_for(mode, None))
    with pytest.raises(ValueError, match="diffusion_blend"):
        driver._run_measurement()
    assert entered == []
    assert not record.exists()

    mode["value"] = "escape"
    with pytest.raises(ValueError, match="timing"):
        driver._run_measurement()
    assert entered == []
    assert not record.exists()

    mode["value"] = "ok"
    driver._run_measurement()
    assert entered == [True]
    payload = json.loads(record.read_text(encoding="utf-8"))
    assert payload["slice_ids"] == [45, 46, 47]
    assert payload["page_cache"] == "not_dropped"
    assert payload["task_hash"] is None
    assert payload["scratch_copy"]["status"] == "not_taken"
    assert payload["scratch_copy"]["seconds"] is None
    assert payload["provenance_flags"]["align_to_ras_enabled"] is True
    stages = {row["name"]: row for row in payload["stages"]}
    assert set(stages) == set(_REQUIRED_STAGES)
    assert "align_to_ras" not in stages
    assert stages["illumination"]["device"] == "cuda"
    assert stages["zip"]["device"] == "host"
    assert stages["stack"]["device"] == "unknown"
    assert stages["stitch"]["seconds"] == pytest.approx(0.6)
    assert stages["stitch_scoring"]["seconds"] == 1.5
    assert stages["diffusion_blend"]["seconds"] == 0.4
    assert stages["per_plane_resample"]["functions"] == ["apply_transform_to_volume", "skipped"]
    assert all(row["duration_source"] == "perf_counter" for row in stages.values())


def test_record_carries_the_optional_stack_z_match_row(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    record, entered = _patch_measurement(tmp_path, monkeypatch)
    monkeypatch.setattr(driver, "_time_spec", _fake_time_for({"value": "ok"}, "cuda"))
    driver._run_measurement()
    assert entered == [True]
    payload = json.loads(record.read_text(encoding="utf-8"))
    names = [row["name"] for row in payload["stages"]]
    assert names == [*_REQUIRED_STAGES, "stack_z_match"]
    stages = {row["name"]: row for row in payload["stages"]}
    assert stages["stack"]["device"] == "mixed"
    z_match = stages["stack_z_match"]
    assert z_match["parent"] == "stack"
    assert z_match["device"] == "cuda"
    assert z_match["functions"] == ["find_z_overlap"]
    assert z_match["duration_source"] == "perf_counter"
    assert stages["stack_reads"]["device"] == "cpu"
