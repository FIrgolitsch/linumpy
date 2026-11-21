"""Work on the planes of a volume side by side, within the thread budget of the task."""

from linumpy.config.threads import thread_budget

from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np


def map_threaded(function: Callable[[Any], Any], items: Iterable[Any], workers: int | None = None, cap: int = 16) -> list:
    """Apply ``function`` to every item on a thread pool and return the results in order.

    For work that spends its time in NumPy or SciPy, which release the
    interpreter lock. ``workers=None`` takes the thread budget of the task,
    at most ``cap``; one worker runs the items in the calling thread.
    """
    items = list(items)
    workers = thread_budget(cap) if workers is None else max(1, int(workers))
    workers = min(workers, len(items))
    if workers <= 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(function, items))


def gaussian_filter_planes(volume: np.ndarray, sigma: float, workers: int | None = None) -> np.ndarray:
    """Gaussian filter of every plane of a ZYX volume, the planes on separate threads.

    Same values as ``scipy.ndimage.gaussian_filter(volume, (0, sigma, sigma))``,
    which filters one line at a time on a single core.

    Parameters
    ----------
    volume : np.ndarray
        ZYX volume of a floating type.
    sigma : float
        Standard deviation in Y and X, in pixels.
    workers : int or None
        Planes filtered at the same time. ``None`` takes the thread budget of
        the task, at most 8.
    """
    from scipy.ndimage import gaussian_filter

    volume = np.asarray(volume)
    n_planes = volume.shape[0]
    workers = thread_budget(8) if workers is None else max(1, int(workers))
    workers = min(workers, n_planes)
    if workers <= 1:
        return gaussian_filter(volume, (0.0, sigma, sigma))
    out = np.empty(volume.shape, dtype=volume.dtype)

    def _filter_plane(k: int) -> None:
        gaussian_filter(volume[k], sigma, output=out[k])

    map_threaded(_filter_plane, range(n_planes), workers=workers)
    return out


def copy_threaded(source: np.ndarray, workers: int | None = None) -> np.ndarray:
    """Return a C-contiguous copy of ``source``, runs along the first axis copied on separate threads.

    Same result as ``np.ascontiguousarray``. A transposed or flipped view of a
    volume of tens of gigabytes takes most of a minute to copy on one core.
    """
    source = np.asarray(source)
    if source.ndim == 0 or source.flags.c_contiguous:
        return np.ascontiguousarray(source)
    out = np.empty(source.shape, dtype=source.dtype)
    n = source.shape[0]
    workers = thread_budget(32) if workers is None else max(1, int(workers))
    step = max(1, -(-n // max(1, min(workers, n) * 4)))

    def _copy(start: int) -> None:
        out[start : start + step] = source[start : start + step]

    map_threaded(_copy, range(0, n, step), workers=workers, cap=32)
    return out
