"""Env-gated stage timers. Reads ``LINUMPY_STAGE_TIMING`` on each call."""

import atexit
import json
import os
import time

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


atexit.register(flush)
