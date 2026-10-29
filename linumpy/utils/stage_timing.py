"""Env-gated stage timers. Reads ``LINUMPY_STAGE_TIMING`` on each call."""

import atexit
import json
import math
import os
import time
from collections.abc import Sequence
from typing import TypeGuard

_SECONDS: dict[str, float] = {}
_COUNTS: dict[str, int] = {}
_META: dict[str, dict[str, object]] = {}


def add(stage: str, seconds: float, **meta: object) -> None:
    """Accumulate ``seconds`` for ``stage`` when timing is enabled.

    Parameters
    ----------
    stage : str
        Stage name written on the JSONL line.
    seconds : float
        Elapsed seconds from a ``perf_counter`` difference.
    **meta : object
        Extra fields stored with the stage. Repeated ``function`` values are
        kept in a list named ``functions``. ``device`` stays one string when
        every call agrees.
    """
    if not os.environ.get("LINUMPY_STAGE_TIMING"):
        return
    _SECONDS[stage] = _SECONDS.get(stage, 0.0) + seconds
    _COUNTS[stage] = _COUNTS.get(stage, 0) + 1
    stored = _META.setdefault(stage, {})
    for key, value in meta.items():
        if key == "function":
            current = stored.get("functions")
            if isinstance(current, list):
                updated: list[object] = [*current, value]
            else:
                updated = [value]
            stored["functions"] = updated
            continue
        if key in stored and stored[key] != value:
            raise ValueError(f"{key} disagrees for stage {stage}")
        stored[key] = value


class span:  # noqa: N801
    """Time one block with ``perf_counter`` and add the difference."""

    def __init__(self, stage: str, **meta: object) -> None:
        """Remember the stage name and metadata.

        Parameters
        ----------
        stage : str
            Stage name passed to :func:`add`.
        **meta : object
            Fields forwarded to :func:`add`.
        """
        self.stage = stage
        self.meta = meta
        self._t0 = 0.0

    def __enter__(self) -> span:
        """Start the clock. The env var is read again when the block ends."""
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_exc: object) -> None:
        """Add the elapsed seconds when ``LINUMPY_STAGE_TIMING`` is set."""
        elapsed = time.perf_counter() - self._t0
        if os.environ.get("LINUMPY_STAGE_TIMING"):
            add(self.stage, elapsed, **self.meta)


def flush() -> None:
    """Append one JSON object per stage, then clear the in-memory totals.

    No-op when ``LINUMPY_STAGE_TIMING`` is unset or no stages were recorded.
    A second call writes nothing after a successful write. ``json.dumps``
    uses the default encoder, so an array value cannot be written.
    """
    path = os.environ.get("LINUMPY_STAGE_TIMING")
    if not path or not _SECONDS:
        return
    lines: list[str] = []
    for stage, seconds in _SECONDS.items():
        record: dict[str, object] = {
            "stage": stage,
            "seconds": seconds,
            "count": _COUNTS[stage],
        }
        record.update(_META.get(stage, {}))
        lines.append(json.dumps(record) + "\n")
    with open(path, "a", encoding="utf-8") as handle:  # noqa: PTH123
        handle.writelines(lines)
    _SECONDS.clear()
    _COUNTS.clear()
    _META.clear()


def device_id(path: str) -> int:
    """Return the device id of ``path``.

    Parameters
    ----------
    path : str
        Filesystem path to stat.

    Returns
    -------
    int
        ``st_dev`` of ``path``.
    """
    return os.stat(path).st_dev  # noqa: PTH116


def same_filesystem(path_a: str, path_b: str) -> bool:
    """Return whether two paths share a device id.

    Parameters
    ----------
    path_a : str
        First path.
    path_b : str
        Second path.

    Returns
    -------
    bool
        True only when the two ``st_dev`` values are equal.
    """
    return device_id(path_a) == device_id(path_b)


_DEFAULT_TIMING_ROOT = "/scratch_nvme/workspace/sub-21/timing"
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
_INNER_STAGES = frozenset({"stitch_scoring", "diffusion_blend", "stack_reads", "per_plane_resample", "stack_z_match"})
_DISK_KEYS = (
    "tmp_source",
    "subject_source",
    "work_source",
    "tmp_st_dev",
    "subject_st_dev",
    "work_st_dev",
    "same_filesystem",
)
_TEXT_KEYS = (
    "subject",
    "host",
    "linumpy_git_rev",
    "linumpy_git_branch",
    "linum_basic_revision",
)
_TOP_LEVEL = (
    *_TEXT_KEYS,
    "slice_ids",
    "page_cache",
    "thread_env",
    "disk",
    "scratch_copy",
    "task_hash",
    "task_hash_reason",
    "provenance_flags",
    "stages",
    "sources_read",
)
_TASK_HASH_REASON = "direct command has no Nextflow task hash"
_PER_SLICE_STAGES = ("illumination", "stitch", "attenuation")
_STACK_DEVICE_FOR_Z_MATCH = {"cuda": "mixed", "cpu": "cpu"}


def _as_record(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    record: dict[str, object] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{label} keys must be strings")
        record[key] = item
    return record


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_measured(value: object) -> TypeGuard[int | float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(value) and value >= 0


def _require_text(record: dict[str, object], key: str) -> None:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")


def _path_parts(path: str) -> list[str]:
    return [part for part in path.replace("\\", "/").split("/") if part not in ("", ".")]


def _inside_timing_root(path: str, timing_root: str) -> bool:
    root_real = os.path.realpath(timing_root)
    path_real = os.path.realpath(path)
    if path_real == root_real:
        return True
    prefix = root_real.rstrip(os.sep) + os.sep
    return path_real.startswith(prefix)


def _check_stage(stage: dict[str, object], timing_root: str) -> tuple[str, str | None]:
    name = stage.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("stage name must be a non-empty string")
    source = stage.get("duration_source")
    if not isinstance(source, str) or source not in {"perf_counter", "not_taken"}:
        detail = f"duration_source {source} is not perf_counter or not_taken"
        if isinstance(source, str) and "nextflow" in source:
            detail = f"{detail}; nextflow logs are refused"
        raise ValueError(detail)
    seconds = stage.get("seconds")
    if source == "not_taken" and _is_number(seconds):
        raise ValueError(f"not_taken stage {name} must not carry numeric seconds")
    if source == "perf_counter" and not _is_measured(seconds):
        raise ValueError(f"perf_counter stage {name} requires finite non-negative seconds")
    output = stage.get("output_path")
    if not isinstance(output, str) or not _inside_timing_root(output, timing_root):
        raise ValueError(f"output_path resolves outside the timing root: {output}")
    if not isinstance(stage.get("argv"), list):
        raise ValueError(f"{name} argv must be a list")
    device = stage.get("device")
    if not isinstance(device, str) or not device:
        raise ValueError(f"{name} device must be a string")
    if not isinstance(stage.get("flags"), dict):
        raise ValueError(f"{name} flags must be an object")
    functions = stage.get("functions")
    if "functions" in stage and not (isinstance(functions, list) and all(isinstance(item, str) for item in functions)):
        raise ValueError(f"{name} functions must be a list of strings")
    parent: str | None = None
    if name in _INNER_STAGES:
        parent_name = stage.get("parent")
        if not isinstance(parent_name, str) or not parent_name:
            raise ValueError(f"{name} missing parent")
        parent = parent_name
        count = stage.get("count")
        if not isinstance(count, int) or isinstance(count, bool):
            raise ValueError(f"{name} count must be an int")
    return name, parent


def _check_per_slice(name: str, stage: dict[str, object], slice_ids: Sequence[object], timing_root: str) -> None:
    entries = stage.get("per_slice")
    if not isinstance(entries, list):
        raise ValueError(f"{name} per_slice must be a list")
    seen: list[object] = []
    devices: list[str] = []
    total = 0.0
    for entry in entries:
        item = _as_record(entry, f"{name} per_slice entry")
        seen.append(item.get("slice_id"))
        seconds = item.get("seconds")
        if not _is_measured(seconds):
            raise ValueError(f"{name} per_slice seconds must be finite and non-negative")
        total += seconds
        device = item.get("device")
        if not isinstance(device, str) or not device:
            raise ValueError(f"{name} per_slice device must be a non-empty string")
        devices.append(device)
        if not isinstance(item.get("argv"), list):
            raise ValueError(f"{name} per_slice argv must be a list")
        output = item.get("output_path")
        if not isinstance(output, str) or not _inside_timing_root(output, timing_root):
            raise ValueError(f"{name} per_slice output_path resolves outside the timing root: {output}")
        if not isinstance(item.get("flags"), dict):
            raise ValueError(f"{name} per_slice flags must be an object")
    if seen != slice_ids:
        raise ValueError(f"{name} per_slice slice_ids {seen} are not the timed slices {slice_ids}")
    stage_seconds = stage.get("seconds")
    if not _is_measured(stage_seconds) or not math.isclose(total, stage_seconds, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{name} per_slice seconds add up to {total}, not the stage seconds {stage_seconds}")
    shared = devices[0] if all(device == devices[0] for device in devices) else "mixed"
    if stage.get("device") != shared:
        raise ValueError(f"{name} device {stage.get('device')} contradicts its per_slice devices {devices}")


def _check_scope(stage: dict[str, object], slice_ids: Sequence[object]) -> None:
    scope = stage.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("common_space scope must be an object")
    record = _as_record(scope, "common_space scope")
    if record.get("slice_ids") != slice_ids:
        raise ValueError(f"common_space scope slice_ids {record.get('slice_ids')} are not the timed slices {slice_ids}")
    for key in ("slice_count", "input_count"):
        count = record.get(key)
        if isinstance(count, bool) or not isinstance(count, int):
            raise ValueError(f"common_space scope {key} must be an int")
    if record["slice_count"] != len(slice_ids):
        raise ValueError(f"common_space scope slice_count {record['slice_count']} is not the {len(slice_ids)} timed slices")
    if record["input_count"] != record["slice_count"]:
        raise ValueError(
            f"common_space scope input_count {record['input_count']} is not its slice_count {record['slice_count']}"
        )


def _check_stack_device(stack: dict[str, object], z_match: dict[str, object] | None) -> None:
    argv = stack.get("argv")
    device = stack.get("device")
    if not isinstance(argv, list) or "--use_gpu" not in argv:
        expected = "cpu"
        basis = "its argv has no --use_gpu"
    elif z_match is None:
        expected = "unknown"
        basis = "its argv has --use_gpu and the record has no stack_z_match row"
    else:
        reported = z_match.get("device")
        expected = _STACK_DEVICE_FOR_Z_MATCH.get(reported, "unknown") if isinstance(reported, str) else "unknown"
        basis = f"its stack_z_match device {reported}"
    if device != expected:
        raise ValueError(f"stack device {device} contradicts the run, because {basis}; expected {expected}")


def _check_thread_env(value: object) -> None:
    if not isinstance(value, dict):
        raise ValueError("thread_env must be an object")
    for key, item in value.items():
        if not isinstance(key, str):
            raise ValueError("thread_env keys must be strings")
        if isinstance(item, bool) or not isinstance(item, (str, int)):
            raise ValueError(f"thread_env {key} must be a string or an integer")


def validate_record(payload: object, timing_root: str = _DEFAULT_TIMING_ROOT) -> None:
    """Reject a record that is not a perf_counter measurement.

    Beyond the perf_counter and path rules, the record must say what each row
    covers. ``illumination``, ``stitch``, and ``attenuation`` carry a
    ``per_slice`` list that names slices 45, 46, and 47 in order, sums to the
    row's seconds, and agrees with the row's device (``mixed`` when the slices
    disagree). ``common_space`` carries a ``scope`` of exactly those slices.
    The ``stack`` device must follow from its argv and its ``stack_z_match`` row.

    Parameters
    ----------
    payload : object
        Candidate timing record.
    timing_root : str
        Directory stage outputs must stay inside. Defaults to the subject
        timing directory.

    Raises
    ------
    ValueError
        The record breaks any of those rules.
    """
    record = _as_record(payload, "timing record")
    for key in _TOP_LEVEL:
        if key not in record:
            raise ValueError(f"missing {key}")
    if record["task_hash"] is not None:
        raise ValueError("task_hash must be null")
    if record["task_hash_reason"] != _TASK_HASH_REASON:
        raise ValueError("task_hash_reason must explain the missing task_hash")
    slice_ids = record["slice_ids"]
    if not isinstance(slice_ids, list) or slice_ids != [45, 46, 47]:
        raise ValueError("slice_ids must stay 45, 46, and 47")
    if record["page_cache"] != "not_dropped":
        raise ValueError("page_cache must stay not_dropped")
    for key in _TEXT_KEYS:
        _require_text(record, key)
    _check_thread_env(record["thread_env"])
    provenance = _as_record(record["provenance_flags"], "provenance_flags")

    disk = _as_record(record["disk"], "disk")
    for key in _DISK_KEYS:
        if key not in disk:
            raise ValueError(f"disk missing {key}")
    if not isinstance(disk["same_filesystem"], bool):
        raise ValueError("disk same_filesystem must be a bool")
    for key in ("tmp_st_dev", "subject_st_dev", "work_st_dev"):
        dev = disk[key]
        if not isinstance(dev, int) or isinstance(dev, bool):
            raise ValueError(f"disk {key} must be an int")
    if disk["same_filesystem"] != (disk["tmp_st_dev"] == disk["subject_st_dev"]):
        raise ValueError("disk same_filesystem disagrees with the st_dev values")
    for key in ("tmp_source", "subject_source", "work_source"):
        source_text = disk[key]
        if not isinstance(source_text, str) or not source_text:
            raise ValueError(f"disk {key} must be a string")

    scratch = _as_record(record["scratch_copy"], "scratch_copy")
    for key in ("status", "seconds", "scratch_dir"):
        if key not in scratch:
            raise ValueError(f"scratch_copy missing {key}")
    if not isinstance(scratch["scratch_dir"], bool):
        raise ValueError("scratch_copy scratch_dir must be a bool")
    status = scratch["status"]
    if status == "not_taken" and _is_number(scratch["seconds"]):
        raise ValueError("not_taken scratch_copy must not carry numeric seconds")
    if status not in {"not_taken", "perf_counter"}:
        raise ValueError("scratch_copy status must be perf_counter or not_taken")
    if status == "perf_counter" and not _is_measured(scratch["seconds"]):
        raise ValueError("perf_counter scratch_copy requires finite non-negative seconds")
    if "scratch_dir" in provenance and provenance["scratch_dir"] is not scratch["scratch_dir"]:
        raise ValueError("provenance_flags scratch_dir contradicts scratch_copy scratch_dir")

    stages = record["stages"]
    if not isinstance(stages, list):
        raise ValueError("stages must be a list")
    stage_records = [_as_record(item, "stage") for item in stages]
    checked = [_check_stage(item, timing_root) for item in stage_records]
    names = [name for name, _parent in checked]
    if len(names) != len(set(names)):
        raise ValueError("duplicate stage name")
    for name, parent in checked:
        if parent is not None and parent not in names:
            raise ValueError(f"{name} parent {parent} names no stage in the record")
    for required in _REQUIRED_STAGES:
        if required not in names:
            raise ValueError(f"missing stage {required}")
    by_name = dict(zip(names, stage_records, strict=True))
    for name, stage in by_name.items():
        if name in _PER_SLICE_STAGES or "per_slice" in stage:
            _check_per_slice(name, stage, slice_ids, timing_root)
    _check_scope(by_name["common_space"], slice_ids)
    _check_stack_device(by_name["stack"], by_name.get("stack_z_match"))

    sources = record["sources_read"]
    if not isinstance(sources, list):
        raise ValueError("sources_read must be a list")
    for item in sources:
        if not isinstance(item, str):
            raise ValueError("sources_read entries must be strings")
        parts = _path_parts(item)
        if parts and parts[-1] == "trace.txt":
            raise ValueError("sources_read names trace.txt")
        if ".nextflow" in parts:
            raise ValueError("sources_read names a nextflow path")


atexit.register(flush)
