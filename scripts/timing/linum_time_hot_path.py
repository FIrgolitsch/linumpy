"""Time slices 45, 46, and 47 and write hot_path.json.

One pass over the existing reconstruction commands writes
``/scratch_nvme/workspace/sub-21/timing/hot_path.json``. The command does not
select other slices or another output root.

Each finished stage is also appended to ``hot_path.partial.jsonl`` for diagnosis.
That file is never read back into a record. A run that finds a record, a partial
file, or an ``out`` directory refuses to start unless ``--archive-previous`` is
given, which renames those three to stamped names and deletes nothing.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

from linumpy.utils.stage_timing import validate_record

TIMING_ROOT = Path("/scratch_nvme/workspace/sub-21/timing")
RECORD_PATH = TIMING_ROOT / "hot_path.json"
PARTIAL_PATH = TIMING_ROOT / "hot_path.partial.jsonl"
SUBJECT_ROOT = Path("/scratch_nvme/workspace/sub-21")
WORK_ROOT = Path("/scratch_nvme/workspace/work")
LINUMPY_REPO = Path("/home/frans/code/linumpy")
SUBJECT = "sub-21"
HOST = "132.207.157.41"
SLICE_IDS = (45, 46, 47)
_PROVENANCE_KEYS = (
    "scratch_dir",
    "compensate_attenuation_method",
    "fix_illum_backend",
    "fix_illum_per_z_fit",
    "fix_illum_darkfield",
    "fix_illum_multi_gpu",
    "stitch_blending_method",
    "align_to_ras_enabled",
)
_INNER_REQUIRED = {
    "stitch": ("stitch_scoring", "diffusion_blend"),
    "stack": ("stack_reads", "per_plane_resample"),
}
_OPTIONAL_STAGES = ("stack_z_match",)
_INNER_OPTIONAL: dict[str, tuple[str, ...]] = {"stack": _OPTIONAL_STAGES}
_RECORD_STAGES = (
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

_ASSIGN = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")
_ZARR_PATH = re.compile(r"\S*?\.ome\.zarr")

_THREAD_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS",
)


def thread_env(cpu_count: int, max_cpus: int = 44, processes: int = 8) -> dict[str, str]:
    """Return per-process thread caps for one host.

    Parameters
    ----------
    cpu_count : int
        Processors reported by the host.
    max_cpus : int
        Requested ceiling. A positive value is capped by ``cpu_count``.
    processes : int
        Parallel workers that share the ceiling.

    Returns
    -------
    dict[str, str]
        Decimal thread counts. ``LINUMPY_MAX_CPUS`` is the effective ceiling
        when ``max_cpus`` is positive.
    """
    max_cpus_eff = min(max_cpus, cpu_count) if max_cpus > 0 else max(1, cpu_count)
    threads = max(1, max_cpus_eff // max(1, processes))
    env = {name: str(threads) for name in _THREAD_NAMES}
    if max_cpus > 0:
        env["LINUMPY_MAX_CPUS"] = str(max_cpus_eff)
    return env


def assert_under_timing(path: Path | str, root: Path = TIMING_ROOT) -> Path:
    """Return ``path`` when it stays inside ``root`` after resolution.

    Parameters
    ----------
    path : Path | str
        Candidate output path.
    root : Path
        Directory the resolved path must stay under.

    Returns
    -------
    Path
        The resolved path.

    Raises
    ------
    ValueError
        The resolved path leaves ``root``, including through a symlink.
    """
    resolved = Path(path).resolve()
    root_resolved = Path(root).resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(f"path resolves outside the timing root: {resolved}")
    return resolved


def _findmnt_argv(path: str) -> list[str]:
    return ["findmnt", "-T", path, "-o", "SOURCE,FSTYPE,TARGET", "-n"]


def disk_block(stat_fn: Callable[[str], int], findmnt_fn: Callable[[list[str]], str]) -> dict[str, object]:
    """Return device identity for the temp, subject, and work trees.

    Parameters
    ----------
    stat_fn : Callable[[str], int]
        Returns ``st_dev`` for a path.
    findmnt_fn : Callable[[list[str]], str]
        Runs a ``findmnt`` argv list and returns its source text.

    Returns
    -------
    dict[str, object]
        Source text and device ids. ``same_filesystem`` compares ``/tmp`` with
        the subject tree. The block has no duration.
    """
    paths = (
        ("tmp", "/tmp"),
        ("subject", "/scratch_nvme/workspace/sub-21"),
        ("work", "/scratch_nvme/workspace/work"),
    )
    sources: dict[str, str] = {}
    devices: dict[str, int] = {}
    for key, path in paths:
        sources[key] = findmnt_fn(_findmnt_argv(path))
        devices[key] = stat_fn(path)
    return {
        "tmp_source": sources["tmp"],
        "subject_source": sources["subject"],
        "work_source": sources["work"],
        "tmp_st_dev": devices["tmp"],
        "subject_st_dev": devices["subject"],
        "work_st_dev": devices["work"],
        "same_filesystem": devices["tmp"] == devices["subject"],
    }


def scratch_copy() -> dict[str, object]:
    """Return the scratch-copy record for a run that does not rsync.

    Returns
    -------
    dict[str, object]
        Status ``not_taken``, null seconds, and ``scratch_dir`` false.
    """
    return {"status": "not_taken", "seconds": None, "scratch_dir": False}


def run_argv(argv: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run ``argv`` with ``env`` and require a zero exit.

    Parameters
    ----------
    argv : list[str]
        Program and arguments. Never a shell string.
    env : dict[str, str]
        Environment for the child.

    Returns
    -------
    subprocess.CompletedProcess[str]
        The completed child. A non-zero exit raises.

    Raises
    ------
    subprocess.CalledProcessError
        The child exited non-zero. Its stderr is written to ``sys.stderr``
        first, because the exception text does not carry it.
    """
    try:
        return subprocess.run(argv, check=True, capture_output=True, text=True, env=env)
    except subprocess.CalledProcessError as error:
        stderr = error.stderr
        if isinstance(stderr, str) and stderr:
            sys.stderr.write(stderr if stderr.endswith("\n") else stderr + "\n")
        raise


def finalize(payload: dict[str, object], timing_root: Path | str = TIMING_ROOT) -> dict[str, object]:
    """Check ``payload`` and return it. This does not write the record file.

    Parameters
    ----------
    payload : dict[str, object]
        Candidate timing record.
    timing_root : Path | str
        Directory stage outputs must stay inside.

    Returns
    -------
    dict[str, object]
        The same payload after it passes the record check.
    """
    validate_record(payload, timing_root=str(timing_root))
    return payload


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "/" and not in_single and not in_double and line[index : index + 2] == "//":
            return line[:index]
    return line


def _params_body(text: str) -> str:
    """Return the first top-level params block, or the whole text when it has none."""
    depth = 0
    in_single = False
    in_double = False
    start: int | None = None
    index = 0
    while index < len(text):
        char = text[index]
        if not in_single and not in_double and text.startswith("//", index):
            newline = text.find("\n", index)
            index = len(text) if newline < 0 else newline + 1
            continue
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif not in_single and not in_double:
            if depth == 0 and text.startswith("params", index):
                cursor = index + len("params")
                while cursor < len(text) and text[cursor].isspace():
                    cursor += 1
                if cursor < len(text) and text[cursor] == "{":
                    start = cursor + 1
                    depth = 1
                    index = cursor + 1
                    continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if start is not None and depth == 0:
                    return text[start:index]
        index += 1
    if start is None:
        return text
    raise ValueError("unclosed params block")


def _parse_scalar(raw: str, key: str) -> object:
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        return raw[1:-1]
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    if re.fullmatch(r"-?(?:\d+\.\d+|\.\d+)", raw):
        return float(raw)
    raise ValueError(f"unsupported config value for {key}: {raw}")


def _parse_value(raw: str, key: str) -> object:
    if raw == "true":
        return True
    if raw == "false":
        return False
    if raw == "null":
        return None
    if raw.startswith("[") and raw.endswith("]"):
        inner = raw[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip(), key) for part in inner.split(",")]
    return _parse_scalar(raw, key)


def _parse_config(text: str) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for line in _params_body(text).splitlines():
        stripped = _strip_comment(line).strip().rstrip(",")
        if not stripped:
            continue
        match = _ASSIGN.match(stripped)
        if match is None:
            continue
        key, raw = match.group(1), match.group(2).strip()
        parsed[key] = _parse_value(raw, key)
    return parsed


def merge_config(workflow_text: str, server_text: str) -> dict[str, object]:
    """Merge workflow assignments with the server file. The server value wins.

    Parameters
    ----------
    workflow_text : str
        Workflow config text. A top-level ``params`` block is the only source.
    server_text : str
        Subject config text. A missing key keeps the workflow value.

    Returns
    -------
    dict[str, object]
        Booleans, numbers, quoted strings, null, and lists of those.

    Raises
    ------
    ValueError
        A value is not one of those forms.
    """
    merged = _parse_config(workflow_text)
    merged.update(_parse_config(server_text))
    return merged


def _flag_bool(cfg: dict[str, object], key: str, default: bool) -> bool:
    value = cfg.get(key, default)
    if isinstance(value, bool):
        return value
    raise ValueError(f"{key} must be a bool")


def _flag_number(cfg: dict[str, object], key: str, default: int | float) -> int | float:
    if key not in cfg or cfg[key] is None:
        return default
    value = cfg[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number")
    return value


def _flag_text(cfg: dict[str, object], key: str, default: str = "") -> str:
    value = cfg.get(key, default)
    if isinstance(value, str):
        return value
    raise ValueError(f"{key} must be a string")


def _num(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    return format(value, "g")


def _positive(cfg: dict[str, object], key: str) -> int | float | None:
    if key not in cfg or cfg[key] is None:
        return None
    value = cfg[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{key} must be a number")
    if value > 0:
        return value
    return None


def _as_path_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    paths: dict[str, str] = {}
    for key, item in value.items():
        if isinstance(item, str):
            paths[str(key)] = item
    return paths


def _timing_root(inputs: dict[str, object]) -> Path:
    root = inputs.get("timing_root", TIMING_ROOT)
    if isinstance(root, Path):
        return root
    if isinstance(root, str):
        return Path(root)
    raise ValueError("timing_root must be a path")


def _cpu_count(inputs: dict[str, object]) -> int:
    value = inputs.get("cpu_count", os.cpu_count() or 1)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    counted = os.cpu_count() or 1
    return counted


def _stage_env(cfg: dict[str, object], inputs: dict[str, object], *, illumination: bool) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        thread_env(
            _cpu_count(inputs),
            max_cpus=int(_flag_number(cfg, "max_cpus", 44)),
            processes=int(_flag_number(cfg, "processes", 8)),
        )
    )
    if illumination and _flag_bool(cfg, "fix_illum_multi_gpu", False) and _flag_bool(cfg, "use_gpu", True):
        gpu_count = int(_flag_number(cfg, "gpu_count", 1))
        if gpu_count > 1:
            env["CUDA_VISIBLE_DEVICES"] = ",".join(str(device) for device in range(gpu_count))
        cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
        env["TORCHINDUCTOR_CACHE_DIR"] = cache or str(Path.home() / ".cache/linum-basic/inductor")
        env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
    return env


def _spec(
    name: str,
    argv: list[str],
    env: dict[str, str],
    output_path: Path,
    device_rule: str,
    flags: dict[str, object],
    slice_id: int | None = None,
    source_paths: list[str] | None = None,
    scope: dict[str, object] | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "name": name,
        "argv": argv,
        "env": env,
        "output_path": str(output_path),
        "device_rule": device_rule,
        "flags": flags,
    }
    if slice_id is not None:
        row["slice_id"] = slice_id
    if source_paths is not None:
        row["source_paths"] = source_paths
    if scope is not None:
        row["scope"] = scope
    return row


def _illumination_spec(cfg: dict[str, object], inputs: dict[str, object], slice_id: int, root: Path) -> dict[str, object]:
    default = str(
        SUBJECT_ROOT / "output" / str(slice_id) / "resample_mosaic_grid" / f"mosaic_grid_z{slice_id}_resampled.ome.zarr"
    )
    source = _as_path_map(inputs.get("illumination")).get(str(slice_id), default)
    out_dir = root / "out" / "illumination" / f"z{slice_id}"
    output = out_dir / f"mosaic_grid_z{slice_id}_illum_fix.ome.zarr"
    argv = [
        "linum-fix-illumination-basic",
        source,
        str(output),
        "--n_processes",
        _num(_flag_number(cfg, "processes", 8)),
        "--percentile_max",
        _num(_flag_number(cfg, "clip_percentile_upper", 99.9)),
        "--use_gpu" if _flag_bool(cfg, "use_gpu", True) else "--no-use_gpu",
        "--n_levels",
        "0",
        "--fit_max_samples",
        _num(_flag_number(cfg, "fix_illum_fit_max_samples", 2000)),
        "--max_iterations",
        _num(_flag_number(cfg, "fix_illum_max_iterations", 500)),
        "--darkfield_percentile",
        _num(_flag_number(cfg, "fix_illum_darkfield_percentile", 10)),
    ]
    if "fix_illum_smoothness_flatfield" in cfg and cfg["fix_illum_smoothness_flatfield"] is not None:
        argv.extend(["--smoothness_flatfield", _num(_flag_number(cfg, "fix_illum_smoothness_flatfield", 0.05))])
    argv.append("--use_darkfield" if _flag_bool(cfg, "fix_illum_darkfield", False) else "--no-use_darkfield")
    argv.append("--per_z_fit" if _flag_bool(cfg, "fix_illum_per_z_fit", False) else "--no-per_z_fit")
    argv.extend(["--slice_id", f"z{slice_id}", "--diagnostics_dir", str(out_dir / "diagnostics")])
    return _spec(
        "illumination",
        argv,
        _stage_env(cfg, inputs, illumination=True),
        output,
        "stdout",
        {
            "per_z_fit": _flag_bool(cfg, "fix_illum_per_z_fit", False),
            "use_darkfield": _flag_bool(cfg, "fix_illum_darkfield", False),
            "use_gpu": _flag_bool(cfg, "use_gpu", True),
            "n_processes": int(_flag_number(cfg, "processes", 8)),
            "slice_id": f"z{slice_id}",
        },
        slice_id,
    )


def _stitch_spec(cfg: dict[str, object], inputs: dict[str, object], slice_id: int, root: Path) -> dict[str, object]:
    default = str(
        SUBJECT_ROOT / "output" / str(slice_id) / "fix_illumination_basic" / f"mosaic_grid_z{slice_id}_illum_fix.ome.zarr"
    )
    source = _as_path_map(inputs.get("stitch")).get(str(slice_id), default)
    output = root / "out" / "stitch" / f"z{slice_id}" / f"slice_z{slice_id}_stitch_3d.ome.zarr"
    argv = [
        "linum-stitch-3d-refined",
        source,
        str(output),
        "--overlap_fraction",
        _num(_flag_number(cfg, "stitch_overlap_fraction", 0.2)),
        "--blending_method",
        _flag_text(cfg, "stitch_blending_method", "diffusion"),
        "--refinement_mode",
        "blend_shift",
        "--max_refinement_px",
        _num(_flag_number(cfg, "max_blend_refinement_px", 10)),
    ]
    transform = _as_path_map(inputs.get("stitch_transforms")).get(str(slice_id))
    if transform:
        argv.extend(["--input_transform", transform])
    argv.extend(["--n_levels", "0", "-f"])
    return _spec(
        "stitch",
        argv,
        _stage_env(cfg, inputs, illumination=False),
        output,
        "cpu",
        {"blending_method": _flag_text(cfg, "stitch_blending_method", "diffusion")},
        slice_id,
    )


def _attenuation_spec(cfg: dict[str, object], inputs: dict[str, object], slice_id: int, root: Path) -> dict[str, object]:
    default = str(SUBJECT_ROOT / "output" / "crop_interface" / f"slice_z{slice_id}_crop_interface.ome.zarr")
    source = _as_path_map(inputs.get("attenuation")).get(str(slice_id), default)
    output = root / "out" / "attenuation" / f"slice_z{slice_id}_attn_corr.ome.zarr"
    method = _flag_text(cfg, "compensate_attenuation_method", "li")
    argv = ["linum-compensate-attenuation-inplace", source, str(output), "--method", method]
    for key, flag, default in (
        ("compensate_attenuation_min_bias", "--min_bias", 0.05),
        ("compensate_attenuation_mask_sigma", "--mask_smoothing_sigma", 2.0),
        ("compensate_attenuation_k", "--k", 10),
        ("compensate_attenuation_zshift", "--zshift", 3),
        ("compensate_attenuation_strength", "--strength", 0.3),
        ("compensate_attenuation_snr_threshold_db", "--snr_threshold_db", 6.0),
    ):
        argv.extend([flag, _num(_flag_number(cfg, key, default))])
    argv.extend(["--n_levels", "0"])
    return _spec(
        "attenuation",
        argv,
        _stage_env(cfg, inputs, illumination=False),
        output,
        "cpu",
        {"method": method},
        slice_id,
    )


def _common_space_spec(cfg: dict[str, object], inputs: dict[str, object], root: Path) -> dict[str, object]:
    output = root / "out" / "common_space"
    in_dir = root / "out" / "common_space_inputs"
    shifts = str(SUBJECT_ROOT / "output" / "detect_rehoming_events" / "shifts_xy_clean.csv")
    argv = ["linum-align-mosaics-3d-from-shifts", str(in_dir), shifts, str(output)]
    slice_config = _flag_text(cfg, "slice_config")
    if slice_config:
        argv.extend(["--slice_config", slice_config])
    mode = _flag_text(cfg, "common_space_excluded_slice_mode")
    if mode:
        argv.extend(["--excluded_slice_mode", mode])
        argv.extend(["--excluded_slice_window", _num(_flag_number(cfg, "common_space_excluded_slice_window", 2))])
    if _flag_bool(cfg, "common_space_refine_unreliable", False):
        argv.append("--refine_unreliable")
    raw_sources = inputs.get("common_space")
    source_paths = [item for item in raw_sources if isinstance(item, str)] if isinstance(raw_sources, list) else []
    scope: dict[str, object] = {
        "slice_ids": list(SLICE_IDS),
        "slice_count": len(SLICE_IDS),
        "input_count": len(source_paths),
    }
    return _spec(
        "common_space",
        argv,
        _stage_env(cfg, inputs, illumination=False),
        output,
        "cpu",
        {},
        source_paths=source_paths,
        scope=scope,
    )


def _append_stack_options(argv: list[str], cfg: dict[str, object], root: Path) -> None:
    refine = _positive(cfg, "stack_blend_z_refine_vox")
    if refine is None and "stack_blend_z_refine_vox" not in cfg:
        refine = 5
    if refine is not None:
        argv.extend(["--blend_z_refine_vox", _num(refine)])
    confidence = _positive(cfg, "blend_z_refine_min_confidence")
    if confidence is None and "blend_z_refine_min_confidence" not in cfg:
        confidence = 0.5
    if confidence is not None:
        argv.extend(["--blend_z_refine_min_confidence", _num(confidence)])
    argv.extend(
        [
            "--slicing_interval_mm",
            _num(_flag_number(cfg, "registration_slicing_interval_mm", 0.2)),
            "--search_range_mm",
            _num(_flag_number(cfg, "registration_allowed_drifting_mm", 0.1)),
            "--moving_z_first_index",
            _num(_flag_number(cfg, "stack_moving_z_first_index", 0)),
        ]
    )
    if _flag_bool(cfg, "use_expected_z_overlap", True):
        argv.append("--use_expected_overlap")
    min_corr = _positive(cfg, "z_overlap_min_corr")
    if min_corr is None and "z_overlap_min_corr" not in cfg:
        min_corr = 0.5
    if min_corr is not None:
        argv.extend(["--z_overlap_min_corr", _num(min_corr)])
    stack_dir = root / "out" / "stack"
    if _flag_bool(cfg, "analyze_shifts", True):
        argv.extend(["--output_z_matches", str(stack_dir / "z_matches.csv")])
    argv.extend(["--output_stacking_decisions", str(stack_dir / "stacking_decisions.csv")])
    if _flag_bool(cfg, "apply_pairwise_transforms", True):
        supplied = cfg.get("transforms_dir")
        transforms = supplied if isinstance(supplied, str) and supplied else str(SUBJECT_ROOT / "output" / "register_pairwise")
        argv.extend(["--transforms_dir", transforms])
        if _flag_bool(cfg, "apply_rotation_only", False):
            argv.append("--rotation_only")
        argv.extend(["--max_rotation_deg", _num(_flag_number(cfg, "max_rotation_deg", 5.0))])
        load_corr = _positive(cfg, "load_transform_min_zcorr")
        if load_corr is not None:
            argv.extend(["--load_min_zcorr", _num(load_corr)])
        load_rotation = _positive(cfg, "load_transform_max_rotation")
        if load_rotation is not None:
            argv.extend(["--load_max_rotation", _num(load_rotation)])
        if _flag_bool(cfg, "skip_error_transforms", True):
            argv.append("--skip_error_transforms")
        if _flag_bool(cfg, "skip_warning_transforms", True):
            argv.append("--skip_warning_transforms")
        argv.extend(
            [
                "--confidence_high",
                _num(_flag_number(cfg, "transform_confidence_high", 0.6)),
                "--confidence_low",
                _num(_flag_number(cfg, "transform_confidence_low", 0.3)),
            ]
        )
    manual = _flag_text(cfg, "manual_transforms_dir")
    if manual:
        argv.extend(["--manual_transforms_dir", manual])
    if _flag_bool(cfg, "stack_accumulate_translations", True):
        argv.append("--accumulate_translations")
        if _flag_bool(cfg, "stack_confidence_weight_translations", True):
            argv.append("--confidence_weight_translations")
        drift = _positive(cfg, "stack_max_cumulative_drift_px")
        if drift is None and "stack_max_cumulative_drift_px" not in cfg:
            drift = 50
        if drift is not None:
            argv.extend(["--max_cumulative_drift_px", _num(drift)])
        pairwise = _positive(cfg, "stack_max_pairwise_translation")
        if pairwise is not None:
            argv.extend(["--max_pairwise_translation", _num(pairwise)])
    smooth = _positive(cfg, "stack_smooth_window")
    if smooth is None and "stack_smooth_window" not in cfg:
        smooth = 5
    if smooth is not None:
        argv.extend(["--smooth_window", _num(smooth)])
    sigma = _positive(cfg, "stack_translation_smooth_sigma")
    if sigma is not None:
        argv.extend(["--translation_smooth_sigma", _num(sigma)])
    min_z = _positive(cfg, "stack_translation_min_zcorr")
    if min_z is None and "stack_translation_min_zcorr" not in cfg:
        min_z = 0.2
    if min_z is not None:
        argv.extend(["--translation_min_zcorr", _num(min_z)])
    argv.append("--no_xy_shift")
    argv.append("--use_gpu" if _flag_bool(cfg, "use_gpu", True) else "--no-use_gpu")
    levels = cfg.get("pyramid_n_levels")
    if isinstance(levels, int | float) and not isinstance(levels, bool):
        argv.extend(["--n_levels", _num(levels)])
    else:
        resolution = _flag_number(cfg, "resolution", 10)
        base = resolution if isinstance(resolution, int | float) and resolution > 0 else 10
        listed = cfg.get("pyramid_resolutions")
        numbers: list[int | float] = []
        if isinstance(listed, list):
            for item in listed:
                if isinstance(item, bool) or not isinstance(item, int | float):
                    raise ValueError("pyramid_resolutions must be numbers")
                if item >= base:
                    numbers.append(item)
        else:
            numbers = [item for item in (10, 25, 50, 100) if item >= base]
        if base not in numbers:
            numbers.insert(0, base)
        argv.append("--pyramid_resolutions")
        argv.extend(_num(item) for item in sorted(numbers))
        argv.append("--make_isotropic" if _flag_bool(cfg, "pyramid_make_isotropic", True) else "--no_isotropic")
    if _flag_bool(cfg, "stack_overlap_z_gain", False):
        argv.extend(
            [
                "--overlap_z_gain",
                "--overlap_z_gain_threshold",
                _num(_flag_number(cfg, "stack_overlap_z_gain_threshold", 0.01)),
                "--overlap_z_gain_min_overlap",
                _num(_flag_number(cfg, "stack_overlap_z_gain_min_overlap", 8)),
                "--overlap_z_gain_clamp_lo",
                _num(_flag_number(cfg, "stack_overlap_z_gain_clamp_lo", 0.5)),
                "--overlap_z_gain_clamp_hi",
                _num(_flag_number(cfg, "stack_overlap_z_gain_clamp_hi", 2.0)),
            ]
        )
    else:
        argv.append("--no-overlap_z_gain")
    argv.append("--blend" if _flag_bool(cfg, "stack_blend_enabled", True) else "--no-blend")
    argv.extend(
        [
            "--blend_tissue_threshold",
            _num(_flag_number(cfg, "stack_overlap_z_gain_threshold", 0.01)),
            "--blend_refinement_px",
            _num(_flag_number(cfg, "blend_refinement_px", 8)),
            "--blend_refinement_ncc_min_improve",
            _num(_flag_number(cfg, "blend_refinement_ncc_min_improve", 0.01)),
            "--blend_refinement_ncc_min_absolute",
            _num(_flag_number(cfg, "blend_refinement_ncc_min_absolute", 0.2)),
            "--max_slice_step_px",
            _num(_flag_number(cfg, "stack_max_slice_step_px", 0)),
            "--no-flatten_z_profile",
            "--max_outline_step_px",
            "0",
            "--notch_slice_seams",
        ]
    )
    if _flag_bool(cfg, "z_overlap_from_registration", False):
        argv.append("--z_from_registration")


def _stack_and_archive(cfg: dict[str, object], inputs: dict[str, object], root: Path) -> list[dict[str, object]]:
    stack_dir = root / "out" / "stack"
    slices_dir = root / "out" / "stack_slices"
    output = stack_dir / "sub-21.ome.zarr"
    shifts = str(SUBJECT_ROOT / "mosaic-grids" / "shifts_xy.csv")
    supplied_transforms = inputs.get("transforms_dir")
    stack_cfg = dict(cfg)
    if isinstance(supplied_transforms, str):
        stack_cfg["transforms_dir"] = supplied_transforms
    argv = ["linum-stack-slices-motor", str(slices_dir), shifts, str(output)]
    _append_stack_options(argv, stack_cfg, root)
    env = _stage_env(cfg, inputs, illumination=False)
    use_gpu = _flag_bool(cfg, "use_gpu", True)
    rows = [
        _spec("stack", argv, env, output, "reported" if use_gpu else "cpu", {"use_gpu": use_gpu}),
    ]
    zip_path = stack_dir / "sub-21.ome.zarr.zip"
    png = stack_dir / "sub-21.png"
    annotated = stack_dir / "sub-21_annotated.png"
    rows.append(_spec("zip", ["zip", "-r", str(zip_path), str(output)], env, zip_path, "host", {}))
    rows.append(_spec("screenshot", ["linum-screenshot-omezarr", str(output), str(png)], env, png, "cpu", {}))
    annotated_argv = [
        "linum-screenshot-omezarr-annotated",
        str(output),
        str(annotated),
        "--slice_ids",
        "45,46,47",
        "--label_every",
        _num(_flag_number(cfg, "annotated_label_every", 1)),
        "--crop_to_tissue",
    ]
    if _flag_bool(cfg, "annotated_show_lines", False):
        annotated_argv.append("--show_lines")
    orientation = _flag_text(cfg, "ras_input_orientation").strip().replace("'", "")
    if orientation:
        annotated_argv.extend(["--orientation", orientation])
    rows.append(_spec("screenshot_annotated", annotated_argv, env, annotated, "cpu", {"slice_ids": "45,46,47"}))
    return rows


def build_stage_commands(cfg: dict[str, object], inputs: dict[str, object]) -> list[dict[str, object]]:
    """Build one command spec per timed stage for slices 45, 46, and 47.

    Parameters
    ----------
    cfg : dict[str, object]
        Merged workflow and server assignments.
    inputs : dict[str, object]
        Resolved input paths and the timing root. Stitch reads the illumination
        volume already on disk, not the illumination output from this session.

    Returns
    -------
    list[dict[str, object]]
        Specs with name, argv, env, output_path, device_rule, and flags.
    """
    root = _timing_root(inputs)
    stages = [_illumination_spec(cfg, inputs, slice_id, root) for slice_id in SLICE_IDS]
    stages.extend(_stitch_spec(cfg, inputs, slice_id, root) for slice_id in SLICE_IDS)
    stages.extend(_attenuation_spec(cfg, inputs, slice_id, root) for slice_id in SLICE_IDS)
    stages.append(_common_space_spec(cfg, inputs, root))
    stages.extend(_stack_and_archive(cfg, inputs, root))
    return stages


def resolve_attenuation_input(command_sh_text: str, slice_id: int) -> str:
    """Return the crop-interface zarr named for ``slice_id``.

    Parameters
    ----------
    command_sh_text : str
        Text of the command script that ran attenuation.
    slice_id : int
        Slice whose crop path is required.

    Returns
    -------
    str
        The crop path. The attenuation output path is never returned.

    Raises
    ------
    ValueError
        The script does not name a crop path for this slice.
    """
    marker = f"z{slice_id}"
    for path in _ZARR_PATH.findall(command_sh_text):
        if "crop_interface" in path and marker in path and not path.endswith(f"slice_z{slice_id}_attn_corr.ome.zarr"):
            return path
    raise ValueError(f"attenuation input for slice {slice_id} is not in the command script")


def resolve_common_space_inputs(command_sh_text: str, work_dir: Path | None = None) -> list[str]:
    """Return pre-alignment zarr paths for a common-space command.

    Parameters
    ----------
    command_sh_text : str
        Text of the command script that ran common-space alignment.
    work_dir : Path | None
        Directory that holds ``.command.sh``. Relative names and an ``inputs``
        directory are resolved from here.

    Returns
    -------
    list[str]
        Input zarrs. Paths under ``bring_to_common_space`` are excluded.

    Raises
    ------
    ValueError
        The script and its work directory name no usable input zarr.
    """
    paths: list[str] = []
    for path in _ZARR_PATH.findall(command_sh_text):
        if "bring_to_common_space" in path or path in paths:
            continue
        if path.startswith("/") or work_dir is None:
            paths.append(path)
            continue
        staged = _materialize_staged(path, work_dir)
        if "bring_to_common_space" not in staged and staged not in paths:
            paths.append(staged)
    if not paths and work_dir is not None:
        inputs = work_dir / "inputs"
        if inputs.is_dir():
            for child in sorted(inputs.iterdir()):
                if not child.name.endswith(".ome.zarr"):
                    continue
                staged = str(child.resolve())
                if "bring_to_common_space" in staged or staged in paths:
                    continue
                paths.append(staged)
    if not paths:
        raise ValueError("common-space inputs are not in the command script")
    return paths


def scope_common_space_inputs(paths: list[str]) -> list[str]:
    """Return the one common-space input for each timed slice, in slice order.

    Parameters
    ----------
    paths : list[str]
        Candidate input zarrs, such as every normalize volume a full-subject
        common-space run read.

    Returns
    -------
    list[str]
        Exactly one path per entry of ``SLICE_IDS``, in that order. A slice id
        matches only the last path component and only as a whole number, so
        ``z45`` never matches ``z450`` or ``z145``.

    Raises
    ------
    ValueError
        A timed slice has no path, or more than one.
    """
    scoped: list[str] = []
    for slice_id in SLICE_IDS:
        pattern = re.compile(rf"(?<!\d)z{slice_id}(?!\d)")
        found = [path for path in paths if pattern.search(Path(path).name)]
        if len(found) != 1:
            wanted = ", ".join(str(item) for item in SLICE_IDS)
            raise ValueError(
                f"common-space inputs must be one path per timed slice ({wanted}): slice {slice_id} matched {len(found)}"
            )
        scoped.append(found[0])
    return scoped


def _json_object(line: str) -> dict[str, object]:
    parsed = json.loads(line)
    if not isinstance(parsed, dict):
        raise ValueError("jsonl stage must be an object")
    row: dict[str, object] = {}
    for key, value in parsed.items():
        if not isinstance(key, str):
            raise ValueError("jsonl keys must be strings")
        row[key] = value
    return row


def _functions(row: dict[str, object]) -> list[object]:
    functions = row.get("functions")
    if isinstance(functions, list):
        return list(functions)
    function = row.get("function")
    if function is None:
        return []
    return [function]


def _parent_command(parent: dict[str, object]) -> tuple[str, list[str], dict[str, object], str]:
    name = parent.get("name")
    argv = parent.get("argv")
    flags = parent.get("flags")
    output_path = parent.get("output_path")
    if not isinstance(name, str) or not name:
        raise ValueError("parent name is missing")
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("parent argv must be a list of strings")
    if not isinstance(flags, dict):
        raise ValueError("parent flags must be an object")
    if not isinstance(output_path, str) or not output_path:
        raise ValueError("parent output_path is missing")
    copied_argv = [item for item in argv if isinstance(item, str)]
    copied_flags: dict[str, object] = {}
    for key, value in flags.items():
        if not isinstance(key, str):
            raise ValueError("flag names must be strings")
        copied_flags[key] = value
    return name, copied_argv, copied_flags, output_path


def collect_inner_stages(parent: dict[str, object], stdout: str, jsonl_text: str) -> list[dict[str, object]]:
    """Map JSONL stages onto the parent command. A missing inner stage raises.

    Parameters
    ----------
    parent : dict[str, object]
        The command that produced the JSONL. argv, flags, output_path, and
        slice_id (when the parent has one) are copied.
    stdout : str
        Child stdout. Illumination uses it for the device; stitch and stack do not.
    jsonl_text : str
        JSONL text from ``LINUMPY_STAGE_TIMING``.

    Returns
    -------
    list[dict[str, object]]
        The required inner stages in order, then any optional inner stage the
        process reported (``stack_z_match`` under ``stack``). Each row carries
        the parent command's argv, flags, and output path, and its slice id
        when the parent is a per-slice command.

    Raises
    ------
    ValueError
        A required inner stage is absent. Nothing is written.
    """
    del stdout
    parent_name, argv, flags, output_path = _parent_command(parent)
    parent_slice = parent.get("slice_id")
    grouped: dict[str, dict[str, object]] = {}
    for line in jsonl_text.splitlines():
        if not line.strip():
            continue
        row = _json_object(line)
        stage = row.get("stage")
        if not isinstance(stage, str) or not stage:
            raise ValueError("jsonl stage name is missing")
        seconds = row.get("seconds")
        if isinstance(seconds, bool) or not isinstance(seconds, int | float):
            raise ValueError(f"{stage} seconds must be numeric")
        count = row.get("count", 1)
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError(f"{stage} count must be an int")
        device = row.get("device")
        if not isinstance(device, str) or not device:
            raise ValueError(f"{stage} device must be a string")
        current = grouped.get(stage)
        if current is None:
            grouped[stage] = {
                "name": stage,
                "argv": argv,
                "seconds": float(seconds),
                "duration_source": "perf_counter",
                "device": device,
                "flags": flags,
                "output_path": output_path,
                "parent": parent_name,
                "count": count,
                "functions": _functions(row),
            }
            if isinstance(parent_slice, int) and not isinstance(parent_slice, bool):
                grouped[stage]["slice_id"] = parent_slice
            continue
        previous = current["seconds"]
        if isinstance(previous, int | float) and not isinstance(previous, bool):
            current["seconds"] = float(previous) + float(seconds)
        previous_count = current["count"]
        if isinstance(previous_count, int) and not isinstance(previous_count, bool):
            current["count"] = previous_count + count
        current["functions"] = [*_functions(current), *_functions(row)]
        if current["device"] != device:
            raise ValueError(f"{stage} device disagrees")
    required = _INNER_REQUIRED.get(parent_name, ())
    missing = [stage for stage in required if stage not in grouped]
    if missing:
        raise ValueError(f"missing stage {missing[0]}")
    if not required:
        return list(grouped.values())
    optional = [stage for stage in _INNER_OPTIONAL.get(parent_name, ()) if stage in grouped]
    return [grouped[stage] for stage in (*required, *optional)]


def stack_device(argv: list[str], inners: list[dict[str, object]]) -> str:
    """Return the stack row's device from what the stack process reported.

    The ``--use_gpu`` flag only says the GPU was allowed. The stage may never
    reach the GPU branch, so the label is read from the ``stack_z_match`` row
    the process wrote and never from the flag.

    Parameters
    ----------
    argv : list[str]
        The stack command that ran.
    inners : list[dict[str, object]]
        Inner rows collected from the stack's stage timing file.

    Returns
    -------
    str
        ``cpu`` when ``--use_gpu`` is not in ``argv``. Otherwise ``mixed`` for a
        ``cuda`` Z-match report, ``cpu`` for a ``cpu`` report (including the
        skipped marker), and ``unknown`` when no report or any other device came back.
    """
    if "--use_gpu" not in argv:
        return "cpu"
    for row in inners:
        if row.get("name") != "stack_z_match":
            continue
        device = row.get("device")
        if device == "cuda":
            return "mixed"
        if device == "cpu":
            return "cpu"
        return "unknown"
    return "unknown"


def _direct_url_record(direct_url_text: str) -> dict[str, object]:
    try:
        parsed = json.loads(direct_url_text)
    except ValueError:
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {key: value for key, value in parsed.items() if isinstance(key, str)}


def _recorded_commit(record: dict[str, object]) -> str:
    vcs_info = record.get("vcs_info")
    if not isinstance(vcs_info, dict):
        return ""
    commit = vcs_info.get("commit_id")
    return commit.strip() if isinstance(commit, str) else ""


def _own_checkout_head(
    candidate: Path,
    git_runner: Callable[[list[str]], subprocess.CompletedProcess[str]],
) -> str:
    """Return HEAD of ``candidate`` only when it is the root of its own repository."""
    try:
        toplevel = git_runner(["git", "-C", str(candidate), "rev-parse", "--show-toplevel"]).stdout.strip()
        if not toplevel or Path(toplevel).resolve() != candidate.resolve():
            return ""
        return git_runner(["git", "-C", str(candidate), "rev-parse", "HEAD"]).stdout.strip()
    except subprocess.CalledProcessError, OSError:
        return ""


def linum_basic_revision(
    dist_path: Path,
    direct_url_text: str,
    git_runner: Callable[[list[str]], subprocess.CompletedProcess[str]],
) -> str:
    """Return the linum-basic revision from its install record, or from its own checkout.

    ``direct_url.json`` is read first. Git is asked only about a directory that
    git itself reports as its own repository root, so the enclosing linumpy
    repository can never stand in for linum-basic.

    Parameters
    ----------
    dist_path : Path
        Installed distribution path.
    direct_url_text : str
        Raw ``direct_url.json`` text. Stored unchanged when it names no commit
        and no own checkout exists.
    git_runner : Callable[[list[str]], subprocess.CompletedProcess[str]]
        Runs a git argv list.

    Returns
    -------
    str
        ``vcs_info.commit_id`` from the install record, else HEAD of a directory
        that is its own repository root (the path named by a ``file://`` url,
        then ``dist_path``), else ``direct_url_text``.

    Raises
    ------
    ValueError
        The record names no commit, no own checkout exists, and
        ``direct_url_text`` is empty.
    """
    record = _direct_url_record(direct_url_text)
    commit = _recorded_commit(record)
    if commit:
        return commit
    candidates: list[Path] = []
    url = record.get("url")
    if isinstance(url, str) and url.startswith("file://"):
        candidates.append(Path(unquote(urlparse(url).path)))
    if dist_path not in candidates:
        candidates.append(dist_path)
    for candidate in candidates:
        head = _own_checkout_head(candidate, git_runner)
        if head:
            return head
    if not direct_url_text.strip():
        raise ValueError("linum-basic revision cannot be recovered: direct_url.json is empty and no own checkout was found")
    return direct_url_text


def illumination_device(stdout: str) -> str:
    """Return cuda only when the printed illumination device is cuda.

    Parameters
    ----------
    stdout : str
        Standard output from ``linum-fix-illumination-basic``.

    Returns
    -------
    str
        ``cuda`` or ``cpu``. A missing line or ``device=None`` is cpu.
    """
    for line in stdout.splitlines():
        if "field_mode=" not in line or "backend=" not in line or "device=" not in line:
            continue
        match = re.search(r"device=([^,\s)]+)", line)
        if match is not None and match.group(1) == "cuda":
            return "cuda"
        return "cpu"
    return "cpu"


def _remember_source(sources: list[str], path: str) -> None:
    if not path or path in sources:
        return
    parts = [part for part in path.replace("\\", "/").split("/") if part not in {"", "."}]
    if parts and (parts[-1] == "trace.txt" or ".nextflow" in parts):
        raise ValueError("refusing to record a nextflow log as an input")
    sources.append(path)


def _newest_command_file(cli_name: str, slice_id: int | None) -> Path:
    matches: list[Path] = []
    if not WORK_ROOT.is_dir():
        raise FileNotFoundError(f"{cli_name} command script is missing")
    for path in WORK_ROOT.glob("*/*/.command.sh"):
        text = path.read_text(encoding="utf-8", errors="replace")
        if cli_name not in text:
            continue
        if slice_id is not None and f"z{slice_id}" not in text:
            continue
        matches.append(path)
    if not matches:
        raise FileNotFoundError(f"{cli_name} command script is missing")
    return max(matches, key=lambda item: item.stat().st_mtime)


def _newest_command_sh(cli_name: str, slice_id: int | None) -> str:
    return _newest_command_file(cli_name, slice_id).read_text(encoding="utf-8", errors="replace")


def _materialize_staged(token: str, work_dir: Path) -> str:
    cleaned = token.strip().strip('"').strip("'")
    if not cleaned or cleaned == "NO_TRANSFORM":
        raise ValueError("staged path is empty")
    path = Path(cleaned)
    if not path.is_absolute():
        path = work_dir / path
    if not path.exists():
        raise FileNotFoundError(cleaned)
    return str(path.resolve())


def _string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _row_slice_id(name: str, row: dict[str, object]) -> int:
    value = row.get("slice_id")
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} slice_id must be an int")
    return value


def _row_seconds(name: str, row: dict[str, object]) -> float:
    value = row.get("seconds")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} seconds must be numeric")
    return float(value)


def _shared_device(rows: list[dict[str, object]]) -> object:
    """Return the device every row names, or ``mixed`` when they disagree."""
    first = rows[0].get("device")
    return first if all(row.get("device") == first for row in rows) else "mixed"


def _flag_map(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        return {}
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _shared_flags(rows: list[dict[str, object]]) -> dict[str, object]:
    """Return the flag keys whose values are identical in every row."""
    flag_sets = [_flag_map(row.get("flags")) for row in rows]
    return {
        key: value
        for key, value in flag_sets[0].items()
        if all(key in flags and flags[key] == value for flags in flag_sets[1:])
    }


def _per_slice(name: str, ordered: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "slice_id": _row_slice_id(name, row),
            "seconds": _row_seconds(name, row),
            "device": row.get("device"),
            "argv": row.get("argv"),
            "output_path": row.get("output_path"),
            "flags": row.get("flags"),
        }
        for row in ordered
    ]


def _collapse_rows(name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    merged = dict(rows[0])
    for row in rows[1:]:
        previous = merged.get("seconds")
        added = row.get("seconds")
        if isinstance(previous, int | float) and isinstance(added, int | float) and not isinstance(previous, bool):
            merged["seconds"] = float(previous) + float(added)
        if "count" in row and isinstance(row["count"], int) and not isinstance(row["count"], bool):
            previous_count = merged.get("count", 0)
            if isinstance(previous_count, int) and not isinstance(previous_count, bool):
                merged["count"] = previous_count + row["count"]
        extra = row.get("functions")
        if isinstance(extra, list):
            merged["functions"] = [*_functions(merged), *extra]
    if len(rows) > 1:
        merged["device"] = _shared_device(rows)
    sliced = [row for row in rows if "slice_id" in row]
    if not sliced:
        return merged
    if len(sliced) != len(rows):
        raise ValueError(f"{name} rows disagree about carrying a slice id")
    ordered = sorted(rows, key=lambda item: _row_slice_id(name, item))
    merged["seconds"] = sum(_row_seconds(name, row) for row in ordered)
    merged["device"] = _shared_device(rows)
    merged["flags"] = _shared_flags(rows)
    merged["per_slice"] = _per_slice(name, ordered)
    del merged["slice_id"]
    return merged


def _collapse_stages(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    """Merge the rows of each stage into one row per stage name.

    Seconds and counts add and function lists join. A row that carries a
    ``slice_id`` also gets a ``per_slice`` entry (slice id, seconds, device,
    argv, output path, flags), sorted by slice id, and the merged row keeps only
    the flag keys every slice shares. The merged device is the shared device,
    or ``mixed`` when the rows disagree. The top-level argv and output path are
    those of the first row, because the record schema requires them.

    Parameters
    ----------
    rows : list[dict[str, object]]
        Timed rows, in run order.

    Returns
    -------
    dict[str, dict[str, object]]
        One merged row per stage name.
    """
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        name = row.get("name")
        if not isinstance(name, str):
            raise ValueError("stage name must be a string")
        grouped.setdefault(name, []).append(row)
    return {name: _collapse_rows(name, members) for name, members in grouped.items()}


def _stat_dev(path: str) -> int:
    return Path(path).stat().st_dev


def _findmnt_text(argv: list[str]) -> str:
    return run_argv(argv, dict(os.environ)).stdout.strip()


def _git_text(repo: Path, *args: str) -> str:
    completed = run_argv(["git", "-C", str(repo), *args], dict(os.environ))
    return completed.stdout.strip()


def _installed_linum_basic_revision() -> str:
    import importlib.metadata as metadata

    dist = metadata.distribution("linum-basic")
    direct = dist.read_text("direct_url.json") or ""

    def runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return run_argv(argv, dict(os.environ))

    return linum_basic_revision(Path(str(dist.locate_file(""))), direct, runner)


def _refuse_unfit_run(cfg: dict[str, object]) -> None:
    """Refuse a run whose record could not describe what ran.

    Parameters
    ----------
    cfg : dict[str, object]
        Merged workflow and server assignments.

    Raises
    ------
    ValueError
        ``scratch_dir`` is not false, because the record can only describe a
        scratch copy that was not taken. Or tracked files in the linumpy
        checkout differ from HEAD, because the stored revision would not
        identify the code that ran. Untracked files do not count.
    """
    if cfg.get("scratch_dir") is not False:
        raise ValueError("scratch_dir must be false: the record can only describe a scratch copy that was not taken")
    dirty = _git_text(LINUMPY_REPO, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError(
            f"linumpy checkout has uncommitted changes to tracked files, so its revision would not identify the code:\n{dirty}"
        )


def _gather_provenance(cfg: dict[str, object]) -> dict[str, object]:
    """Gather every cheap fact the record needs. Runs before the first stage.

    Parameters
    ----------
    cfg : dict[str, object]
        Merged workflow and server assignments.

    Returns
    -------
    dict[str, object]
        ``linumpy_git_rev``, ``linumpy_git_branch``, ``linum_basic_revision``,
        ``thread_env``, ``disk``, ``provenance_flags``, and ``scratch_copy``.

    Raises
    ------
    ValueError
        A provenance key is missing from the merged config.
    """
    missing = [key for key in _PROVENANCE_KEYS if key not in cfg]
    if missing:
        raise ValueError(f"{missing[0]} is not set in the merged config")
    return {
        "linumpy_git_rev": _git_text(LINUMPY_REPO, "rev-parse", "HEAD"),
        "linumpy_git_branch": _git_text(LINUMPY_REPO, "rev-parse", "--abbrev-ref", "HEAD"),
        "linum_basic_revision": _installed_linum_basic_revision(),
        "thread_env": thread_env(
            _cpu_count({}),
            max_cpus=int(_flag_number(cfg, "max_cpus", 44)),
            processes=int(_flag_number(cfg, "processes", 8)),
        ),
        "disk": disk_block(_stat_dev, _findmnt_text),
        "provenance_flags": {key: cfg[key] for key in _PROVENANCE_KEYS},
        "scratch_copy": scratch_copy(),
    }


def _prepare_inputs(cfg: dict[str, object]) -> tuple[dict[str, object], list[str]]:
    sources: list[str] = []
    illumination: dict[str, str] = {}
    stitch: dict[str, str] = {}
    attenuation: dict[str, str] = {}
    transforms: dict[str, str] = {}
    for slice_id in SLICE_IDS:
        illum = str(
            (
                SUBJECT_ROOT
                / "output"
                / str(slice_id)
                / "resample_mosaic_grid"
                / f"mosaic_grid_z{slice_id}_resampled.ome.zarr"
            ).resolve()
        )
        stitched = str(
            (
                SUBJECT_ROOT
                / "output"
                / str(slice_id)
                / "fix_illumination_basic"
                / f"mosaic_grid_z{slice_id}_illum_fix.ome.zarr"
            ).resolve()
        )
        illumination[str(slice_id)] = illum
        stitch[str(slice_id)] = stitched
        _remember_source(sources, illum)
        _remember_source(sources, stitched)
        atten_file = _newest_command_file("linum-compensate-attenuation-inplace", slice_id)
        crop_name = resolve_attenuation_input(atten_file.read_text(encoding="utf-8", errors="replace"), slice_id)
        crop = _materialize_staged(crop_name, atten_file.parent)
        attenuation[str(slice_id)] = crop
        _remember_source(sources, crop)
        stitch_script = ""
        stitch_file: Path | None = None
        try:
            stitch_file = _newest_command_file("linum-stitch-3d-refined", slice_id)
            stitch_script = stitch_file.read_text(encoding="utf-8", errors="replace")
        except FileNotFoundError:
            stitch_script = ""
        match = re.search(r"--input_transform\s+(\S+)", stitch_script)
        if match is not None and match.group(1) not in {"", "NO_TRANSFORM"}:
            if stitch_file is None:
                raise FileNotFoundError("stitch command script is missing")
            transform = _materialize_staged(match.group(1), stitch_file.parent)
            transforms[str(slice_id)] = transform
            _remember_source(sources, transform)
    common_file = _newest_command_file("linum-align-mosaics-3d-from-shifts", None)
    common_paths = scope_common_space_inputs(
        resolve_common_space_inputs(
            common_file.read_text(encoding="utf-8", errors="replace"),
            common_file.parent,
        )
    )
    for path in common_paths:
        _remember_source(sources, path)
    shifts_clean = str(SUBJECT_ROOT / "output" / "detect_rehoming_events" / "shifts_xy_clean.csv")
    shifts_xy = str(SUBJECT_ROOT / "mosaic-grids" / "shifts_xy.csv")
    transforms_dir = str(SUBJECT_ROOT / "output" / "register_pairwise")
    _remember_source(sources, shifts_clean)
    _remember_source(sources, shifts_xy)
    if _flag_bool(cfg, "apply_pairwise_transforms", True):
        _remember_source(sources, transforms_dir)
    workflow = Path(__file__).resolve().parents[2] / "workflows" / "reconst_3d" / "nextflow.config"
    _remember_source(sources, str(workflow))
    _remember_source(sources, str(SUBJECT_ROOT / "nextflow.config"))
    inputs: dict[str, object] = {
        "timing_root": TIMING_ROOT,
        "illumination": illumination,
        "stitch": stitch,
        "attenuation": attenuation,
        "stitch_transforms": transforms,
        "common_space": common_paths,
        "transforms_dir": transforms_dir,
        "cpu_count": os.cpu_count() or 1,
    }
    return inputs, sources


def _link_named(directory: Path, names: list[tuple[str, Path]]) -> None:
    assert_under_timing(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, target in names:
        link = directory / name
        if link.exists() or link.is_symlink():
            raise FileExistsError(link)
        link.symlink_to(target)


def _stage_device(spec: dict[str, object], stdout: str) -> str:
    rule = spec.get("device_rule")
    if rule == "stdout":
        return illumination_device(stdout)
    if rule == "reported":
        return "unknown"
    if isinstance(rule, str) and rule:
        return rule
    return "cpu"


def _time_spec(spec: dict[str, object], root: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    output_raw = spec.get("output_path")
    if not isinstance(output_raw, str):
        raise ValueError("output_path must be a string")
    output = assert_under_timing(output_raw, root=root)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    env_raw = spec.get("env")
    if not isinstance(env_raw, dict):
        raise ValueError("stage env must be an object")
    env = {str(key): str(value) for key, value in env_raw.items()}
    name = spec.get("name")
    if not isinstance(name, str):
        raise ValueError("stage name must be a string")
    if name in {"stitch", "stack"}:
        suffix = spec.get("slice_id", "all")
        jsonl_path = assert_under_timing(root / "out" / f"{name}_{suffix}.jsonl", root=root)
        if jsonl_path.exists():
            raise FileExistsError(jsonl_path)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        env["LINUMPY_STAGE_TIMING"] = str(jsonl_path)
    else:
        jsonl_path = None
    argv = _string_list(spec.get("argv"))
    if not argv:
        raise ValueError(f"{name} argv is empty")
    started = time.perf_counter()
    completed = run_argv(argv, env)
    seconds = time.perf_counter() - started
    flags = spec.get("flags")
    outer: dict[str, object] = {
        "name": name,
        "argv": argv,
        "seconds": seconds,
        "duration_source": "perf_counter",
        "device": _stage_device(spec, completed.stdout),
        "flags": flags if isinstance(flags, dict) else {},
        "output_path": str(output),
    }
    slice_id = spec.get("slice_id")
    if isinstance(slice_id, int) and not isinstance(slice_id, bool):
        outer["slice_id"] = slice_id
    scope = spec.get("scope")
    if isinstance(scope, dict):
        outer["scope"] = dict(scope)
    if jsonl_path is None:
        return outer, []
    text = jsonl_path.read_text(encoding="utf-8") if jsonl_path.is_file() else ""
    inners = collect_inner_stages(spec, completed.stdout, text)
    if name == "stack":
        outer["device"] = stack_device(argv, inners)
    return outer, inners


def _append_partial(path: Path, outer: dict[str, object], inners: list[dict[str, object]]) -> None:
    """Append one finished spec to the partial file as a single JSON line.

    The line is ``{"outer": ..., "inners": [...]}``. It is flushed and synced
    before this returns, so a later failure leaves it on disk. The partial file
    is for diagnosis only and is never read back into a record.

    Parameters
    ----------
    path : Path
        The partial file.
    outer : dict[str, object]
        The finished spec's outer row.
    inners : list[dict[str, object]]
        The finished spec's inner rows.

    Raises
    ------
    ValueError
        A row holds NaN or infinity. Nothing is written.
    """
    line = json.dumps({"outer": outer, "inners": inners}, allow_nan=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_record(path: Path, payload: dict[str, object]) -> None:
    """Write the record so a truncated file never exists at ``path``.

    The text goes to a temporary file in the same directory and is moved into
    place with ``Path.replace``, which is ``os.replace``: an atomic rename on one
    filesystem. NaN is refused before any file is created.

    Parameters
    ----------
    path : Path
        Final record path.
    payload : dict[str, object]
        The checked record.

    Raises
    ------
    ValueError
        The payload holds NaN or infinity. Neither ``path`` nor a temporary file exists afterwards.
    """
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _artifact_paths(root: Path) -> list[Path]:
    """Return what a run leaves under ``root``: the record, the partial file, and ``out``."""
    return [root / RECORD_PATH.name, root / PARTIAL_PATH.name, root / "out"]


def _is_present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _archive_previous(root: Path, stamp: str) -> list[Path]:
    """Rename the record, the partial file, and ``out`` to stamped siblings.

    Nothing is copied, deleted, or overwritten. Every source and destination is
    checked before the first rename, so a refusal moves nothing. An item that
    does not exist is skipped.

    Parameters
    ----------
    root : Path
        The timing root. Every path must resolve inside it.
    stamp : str
        Text put before the suffix: ``hot_path.<stamp>.json``,
        ``hot_path.partial.<stamp>.jsonl``, and ``out.<stamp>``.

    Returns
    -------
    list[Path]
        The new locations, in record, partial file, out order.

    Raises
    ------
    FileExistsError
        A destination already exists.
    ValueError
        A source or destination resolves outside ``root``, including through a symlink.
    """
    moves: list[tuple[Path, Path]] = []
    for source in _artifact_paths(root):
        if not _is_present(source):
            continue
        destination = source.with_name(f"{source.stem}.{stamp}{source.suffix}")
        assert_under_timing(source, root=root)
        assert_under_timing(destination, root=root)
        if _is_present(destination):
            raise FileExistsError(destination)
        moves.append((source, destination))
    for source, destination in moves:
        source.rename(destination)
    return [destination for _source, destination in moves]


def _run_measurement(archive_previous: bool = False) -> None:
    """Time the stages and write the record.

    Order: refuse leftover artifacts unless ``archive_previous``, read the
    config, run the pre-flight checks and gather provenance, resolve inputs,
    archive when asked, link inputs, then run the stages. Nothing before the
    archive step writes or moves a file.
    """
    leftovers = [path for path in _artifact_paths(TIMING_ROOT) if _is_present(path)]
    if leftovers and not archive_previous:
        names = ", ".join(path.name for path in leftovers)
        raise FileExistsError(f"{names} already exist under {TIMING_ROOT}; pass --archive-previous to move them aside")
    workflow_path = Path(__file__).resolve().parents[2] / "workflows" / "reconst_3d" / "nextflow.config"
    server_path = SUBJECT_ROOT / "nextflow.config"
    cfg = merge_config(
        workflow_path.read_text(encoding="utf-8"),
        server_path.read_text(encoding="utf-8"),
    )
    _refuse_unfit_run(cfg)
    provenance = _gather_provenance(cfg)
    inputs, sources = _prepare_inputs(cfg)
    if archive_previous:
        _archive_previous(TIMING_ROOT, datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    TIMING_ROOT.mkdir(parents=True, exist_ok=True)
    stages = build_stage_commands(cfg, inputs)
    common = next(stage for stage in stages if stage.get("name") == "common_space")
    common_dir = TIMING_ROOT / "out" / "common_space_inputs"
    common_links = [(Path(path).name, Path(path)) for path in _string_list(common.get("source_paths"))]
    _link_named(common_dir, common_links)
    stack_dir = TIMING_ROOT / "out" / "stack_slices"
    stack_links = [
        (
            f"slice_z{slice_id}_normalize.ome.zarr",
            SUBJECT_ROOT / "output" / "bring_to_common_space" / f"slice_z{slice_id}_normalize.ome.zarr",
        )
        for slice_id in SLICE_IDS
    ]
    _link_named(stack_dir, stack_links)
    for _name, target in stack_links:
        _remember_source(sources, str(target))
    outers: list[dict[str, object]] = []
    inners: list[dict[str, object]] = []
    for spec in stages:
        outer, inner = _time_spec(spec, TIMING_ROOT)
        _append_partial(PARTIAL_PATH, outer, inner)
        outers.append(outer)
        inners.extend(inner)
    by_name = _collapse_stages(outers)
    by_name.update(_collapse_stages(inners))
    missing = [name for name in _RECORD_STAGES if name not in by_name]
    if missing:
        raise ValueError(f"missing stage {missing[0]}")
    payload: dict[str, object] = {
        "subject": SUBJECT,
        "host": HOST,
        "slice_ids": list(SLICE_IDS),
        "page_cache": "not_dropped",
        "linumpy_git_rev": provenance["linumpy_git_rev"],
        "linumpy_git_branch": provenance["linumpy_git_branch"],
        "linum_basic_revision": provenance["linum_basic_revision"],
        "thread_env": provenance["thread_env"],
        "disk": provenance["disk"],
        "scratch_copy": provenance["scratch_copy"],
        "task_hash": None,
        "task_hash_reason": "direct command has no Nextflow task hash",
        "provenance_flags": provenance["provenance_flags"],
        "stages": [by_name[name] for name in (*_RECORD_STAGES, *_OPTIONAL_STAGES) if name in by_name],
        "sources_read": sources,
    }
    checked = finalize(payload)
    _write_record(RECORD_PATH, checked)


def main(argv: list[str] | None = None) -> None:
    """Parse arguments, then time slices 45, 46, and 47 into hot_path.json.

    Parameters
    ----------
    argv : list[str] | None
        Arguments after the program name. ``--help`` returns before the subject tree is read.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive-previous",
        action="store_true",
        help="rename an existing record, partial file, and out directory to stamped names instead of refusing",
    )
    args = parser.parse_args(argv)
    _run_measurement(archive_previous=bool(args.archive_previous))


if __name__ == "__main__":
    main()
