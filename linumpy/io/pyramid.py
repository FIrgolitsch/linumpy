"""Lower-resolution levels of a volume, computed from the full-resolution level in one pass.

Each level is the full-resolution volume resized as
``skimage.transform.resize(volume, shape, order=1, anti_aliasing=True)`` does
it: along every axis a Gaussian of sigma ``(factor - 1) / 2`` against
aliasing, then linear interpolation. Both are linear, so one axis is one
matrix (:func:`resampling_matrix`) and a level is three matrix products.
That runs on the GPU in runs of planes, and all levels are made from the same
run, so the full-resolution volume is read once and does not have to fit in
GPU memory.

The block-wise resize this replaces filtered every chunk on its own, with the
chunk edges mirrored. Here the filter crosses chunk edges, so values differ
slightly near them.
"""

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_GAUSSIAN_TRUNCATE = 4.0  # scipy.ndimage's default kernel half-width, in sigma


def _mirror(index: np.ndarray, n: int) -> np.ndarray:
    """Fold indices outside ``[0, n)`` back in, without repeating the edge sample."""
    if n == 1:
        return np.zeros_like(index)
    period = 2 * (n - 1)
    folded = np.abs(index) % period
    return np.where(folded >= n, period - folded, folded)


def resampling_matrix(n_in: int, n_out: int, factor: float | None = None) -> np.ndarray:
    """Weights of ``n_in`` input samples in each of ``n_out`` output samples along one axis.

    Row ``o`` is the Gaussian against aliasing (sigma ``(factor - 1) / 2``,
    mirrored at the ends, cut at four sigma) followed by linear interpolation
    at ``(o + 0.5) * factor - 0.5``.

    Parameters
    ----------
    n_in, n_out : int
        Number of input and output samples.
    factor : float, optional
        Input samples per output sample. The default, ``n_in / n_out``,
        stretches the whole input over the output. A level with a voxel size
        of its own gives the ratio of the voxel sizes here: its last voxel
        then ends where the last whole voxel fits, not at the end of the input.

    Returns
    -------
    np.ndarray
        ``(n_out, n_in)`` float32.
    """
    factor = n_in / n_out if factor is None else float(factor)
    coords = np.clip((np.arange(n_out) + 0.5) * factor - 0.5, 0.0, n_in - 1.0)
    below = np.minimum(np.floor(coords).astype(np.int64), n_in - 1)
    above = np.minimum(below + 1, n_in - 1)
    share = coords - below

    sigma = max(0.0, (factor - 1.0) / 2.0)
    if sigma > 0:
        radius = int(_GAUSSIAN_TRUNCATE * sigma + 0.5)
        offsets = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
        kernel /= kernel.sum()
    else:
        offsets = np.zeros(1, dtype=np.int64)
        kernel = np.ones(1)

    matrix = np.zeros((n_out, n_in), dtype=np.float64)
    rows = np.arange(n_out)
    for offset, weight in zip(offsets, kernel, strict=True):
        np.add.at(matrix, (rows, _mirror(below + offset, n_in)), (1.0 - share) * weight)
        np.add.at(matrix, (rows, _mirror(above + offset, n_in)), share * weight)
    return matrix.astype(np.float32)


def _row_supports(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First input sample and one past the last input sample that each row uses."""
    used = matrix != 0
    first = used.argmax(axis=1)
    last = matrix.shape[1] - used[:, ::-1].argmax(axis=1)
    return first, last


def downsample_antialiased(
    source: Any,
    target_shapes: Sequence[Sequence[int]],
    use_gpu: bool = True,
    block_bytes: int = 2 * 1024**3,
    factors: Sequence[Sequence[float]] | None = None,
) -> list[np.ndarray]:
    """Resize a ZYX volume to each of ``target_shapes``, reading it once.

    Parameters
    ----------
    source
        ZYX volume: a NumPy array or anything sliceable along Z that returns
        one (a zarr array). Read in runs of planes.
    target_shapes
        Shape of each level to make.
    use_gpu
        Do the products on the GPU (CuPy). Off, they run in NumPy.
    block_bytes
        Size of the run of full-resolution planes held at a time, as float32.
        A run is never shorter than the widest filter needs.
    factors
        Input voxels per output voxel of each level, per axis (the ratio of
        the voxel sizes). ``None`` stretches the whole volume over each shape.

    Returns
    -------
    list of np.ndarray
        One float32 volume per target shape, in host memory.
    """
    xp: Any = np
    if use_gpu:
        import cupy

        xp = cupy
    nz, ny, nx = (int(v) for v in source.shape)
    shapes = [tuple(int(v) for v in shape) for shape in target_shapes]
    levels = [np.empty(shape, dtype=np.float32) for shape in shapes]
    scales = [(None, None, None)] * len(shapes) if factors is None else [tuple(float(v) for v in f) for f in factors]
    if len(scales) != len(shapes):
        raise ValueError(f"Expected factors for {len(shapes)} levels, got {len(scales)}")
    z_matrices = [resampling_matrix(nz, shape[0], scale[0]) for shape, scale in zip(shapes, scales, strict=True)]
    y_matrices = [xp.asarray(resampling_matrix(ny, shape[1], scale[1])) for shape, scale in zip(shapes, scales, strict=True)]
    x_matrices = [
        xp.asarray(resampling_matrix(nx, shape[2], scale[2]).T.copy()) for shape, scale in zip(shapes, scales, strict=True)
    ]
    supports = [_row_supports(matrix) for matrix in z_matrices]

    widest = max(int((last - first).max()) for first, last in supports) if shapes else 1
    run = max(widest, int(block_bytes // (ny * nx * 4)), 1)
    cursors = [0] * len(shapes)
    while any(cursor < shape[0] for cursor, shape in zip(cursors, shapes, strict=True)):
        # The run starts at the first plane a level still needs. The rows of a
        # level use planes further and further down, so every row whose planes
        # end inside the run can be made now, and the level that sets the
        # start always has one.
        start = min(
            int(first[cursor])
            for cursor, shape, (first, _last) in zip(cursors, shapes, supports, strict=True)
            if cursor < shape[0]
        )
        stop = min(nz, start + run)
        block = xp.asarray(np.asarray(source[start:stop], dtype=np.float32))
        for k, shape in enumerate(shapes):
            _first, last = supports[k]
            begin = cursors[k]
            end = begin
            while end < shape[0] and last[end] <= stop:
                end += 1
            if end == begin:
                continue
            weights = xp.asarray(z_matrices[k][begin:end, start:stop])
            part = xp.tensordot(weights, block, axes=(1, 0))  # (rows, ny, nx)
            part = xp.matmul(y_matrices[k], part)  # (rows, out_y, nx)
            part = xp.matmul(part, x_matrices[k])  # (rows, out_y, out_x)
            levels[k][begin:end] = xp.asnumpy(part) if use_gpu else part
            cursors[k] = end
            del weights, part
        del block
    if use_gpu:
        xp.get_default_memory_pool().free_all_blocks()
    return levels
