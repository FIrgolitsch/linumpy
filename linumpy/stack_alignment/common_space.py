"""Bring stitched mosaics into one XY frame, one mosaic at a time.

The frame is planned once for all mosaics (its size, and where each mosaic
sits in it) and written as a small table. Each mosaic is then shifted on its
own, so a workflow can run the mosaics side by side and redo only the ones
whose input or position changed.
"""

import csv
from collections.abc import Sequence
from pathlib import Path

import dask.array as da
import numpy as np

from linumpy.imaging.transform import apply_xy_shift
from linumpy.io.zarr import read_omezarr, save_omezarr

PLAN_FILENAME = "common_space_plan.csv"
PLAN_COLUMNS = ("slice_id", "filename", "shift_x_px", "shift_y_px", "height", "width")


def write_common_space_plan(
    path: Path,
    mosaic_files: dict[int, Path],
    slice_ids: Sequence[int],
    cumsum_shifts: dict,
    origin: tuple[float, float],
    plane_shape: tuple[int, int],
) -> None:
    """Write where each mosaic sits in the common frame.

    Parameters
    ----------
    path : Path
        Table to write (CSV).
    mosaic_files : dict
        Mosaic file of every slice id.
    slice_ids : sequence of int
        Slices that are brought to the common frame, in order.
    cumsum_shifts : dict
        Cumulative ``(dx, dy)`` of every slice, in pixels.
    origin : tuple of float
        ``(x0, y0)``, the smallest cumulative position: the frame starts there.
    plane_shape : tuple of int
        ``(height, width)`` of the common frame.

    Notes
    -----
    The shifts are written with ``repr``, so reading them back gives the same
    floating-point numbers and a mosaic shifted from the table is the same
    as one shifted in the run that planned the frame.
    """
    x0, y0 = origin
    height, width = plane_shape
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(PLAN_COLUMNS)
        for slice_id in slice_ids:
            dx, dy = cumsum_shifts[slice_id]
            shift_x, shift_y = float(dx - x0), float(dy - y0)
            writer.writerow(
                [int(slice_id), Path(mosaic_files[slice_id]).name, repr(shift_x), repr(shift_y), int(height), int(width)]
            )


def shift_mosaic_to_common_space(
    mosaic_file: Path,
    out_file: Path,
    shift_x: float,
    shift_y: float,
    plane_shape: tuple[int, int],
    slant: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Resample one mosaic into the common frame and write it as OME-Zarr.

    Parameters
    ----------
    mosaic_file : Path
        Stitched mosaic of one slice (ZYX).
    out_file : Path
        OME-Zarr to write.
    shift_x, shift_y : float
        Position of the mosaic's origin in the common frame, in pixels.
    plane_shape : tuple of int
        ``(height, width)`` of the common frame.
    slant : tuple of float
        ``(x, y)`` slant inside the slab in pixels per plane, as
        `linumpy.stack_alignment.slab_slant.estimate_slab_slant` measures
        it. It is removed in the same resampling.
    """
    img, res = read_omezarr(Path(mosaic_file))
    img_data = img[:]
    height, width = plane_shape
    # Only the shape of the reference is used; the zeros are never touched.
    reference = np.zeros((img_data.shape[0], int(height), int(width)), dtype=img_data.dtype)
    # apply_xy_shift moves the sampling grid, not the content: to put the
    # content at (shift_x, shift_y) the grid moves the other way.
    # translation[0] acts on X (columns), translation[1] on Y (rows).
    aligned = apply_xy_shift(img_data, reference, -shift_x, -shift_y, slant=slant)
    save_omezarr(da.from_array(aligned), Path(out_file), res, chunks=img.chunks)
