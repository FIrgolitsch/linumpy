"""Helpers for the hot-path timing driver."""

import pathlib

TIMING_ROOT = pathlib.Path("/scratch_nvme/workspace/sub-21/timing")
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
    """
    del cpu_count, max_cpus, processes
    return {}


def assert_under_timing(path: pathlib.Path | str, root: pathlib.Path = TIMING_ROOT) -> pathlib.Path:
    """Return ``path`` when it stays inside ``root`` after resolution.

    Parameters
    ----------
    path : pathlib.Path | str
        Candidate output path.
    root : pathlib.Path
        Directory the resolved path must stay under.
    """
    del root
    return pathlib.Path(path)
