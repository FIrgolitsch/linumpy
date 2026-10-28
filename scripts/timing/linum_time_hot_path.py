"""Helpers for the hot-path timing driver."""

import os
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from linumpy.utils.stage_timing import validate_record

TIMING_ROOT = Path("/scratch_nvme/workspace/sub-21/timing")
RECORD_PATH = TIMING_ROOT / "hot_path.json"
SUBJECT_ROOT = Path("/scratch_nvme/workspace/sub-21")
SLICE_IDS = (45, 46, 47)

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
    """
    return subprocess.run(argv, check=True, capture_output=True, text=True, env=env)


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
    return _spec(
        "common_space",
        argv,
        _stage_env(cfg, inputs, illumination=False),
        output,
        "cpu",
        {},
        source_paths=source_paths,
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
    rows = [
        _spec("stack", argv, env, output, "cpu", {"use_gpu": _flag_bool(cfg, "use_gpu", True)}),
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


def resolve_common_space_inputs(command_sh_text: str) -> list[str]:
    """Return pre-alignment zarr paths named in a common-space command script.

    Parameters
    ----------
    command_sh_text : str
        Text of the command script that ran common-space alignment.

    Returns
    -------
    list[str]
        Input zarrs. Paths under ``bring_to_common_space`` are excluded.

    Raises
    ------
    ValueError
        The script names no usable input zarr.
    """
    paths: list[str] = []
    for path in _ZARR_PATH.findall(command_sh_text):
        if "bring_to_common_space" in path or path in paths:
            continue
        paths.append(path)
    if not paths:
        raise ValueError("common-space inputs are not in the command script")
    return paths
