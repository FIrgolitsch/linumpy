"""Helpers for the hot-path timing driver."""

import subprocess
from collections.abc import Callable
from pathlib import Path

from linumpy.utils.stage_timing import validate_record

TIMING_ROOT = Path("/scratch_nvme/workspace/sub-21/timing")
RECORD_PATH = TIMING_ROOT / "hot_path.json"

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
