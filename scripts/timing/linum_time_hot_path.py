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


def assert_under_timing(path: pathlib.Path | str, root: pathlib.Path = TIMING_ROOT) -> pathlib.Path:
    """Return ``path`` when it stays inside ``root`` after resolution.

    Parameters
    ----------
    path : pathlib.Path | str
        Candidate output path.
    root : pathlib.Path
        Directory the resolved path must stay under.

    Returns
    -------
    pathlib.Path
        The resolved path.

    Raises
    ------
    ValueError
        The resolved path leaves ``root``, including through a symlink.
    """
    resolved = pathlib.Path(path).resolve()
    root_resolved = pathlib.Path(root).resolve()
    if not resolved.is_relative_to(root_resolved):
        raise ValueError(f"path resolves outside the timing root: {resolved}")
    return resolved
