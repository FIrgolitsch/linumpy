"""Tests for the hot-path timing driver helpers."""

import ast
import inspect
import subprocess
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
