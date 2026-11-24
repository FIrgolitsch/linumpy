"""Shift of the slab image origin between slices, from the stage positions of the mosaic."""

import numpy as np
from numpy.typing import ArrayLike


def slab_origin_shifts_mm(xmin_mm: ArrayLike, ymin_mm: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return how far the image origin moves from each slice to the next, in image axes.

    A mosaic is laid out with stage X along the image rows and stage Y along
    the image columns, and its first pixel sits at the low corner of the
    mosaic, ``(xmin, ymin)``. The sample is fixed on the stage, so the same
    piece of tissue lies at the same stage position in every slice. Putting
    each slab at the stage position of its low corner therefore lines the
    slabs up, whatever the acquisition did to the mosaic between two slices:
    moved it, grew it or shrank it, on either side.

    Parameters
    ----------
    xmin_mm, ymin_mm : array-like of float
        Stage position of the low corner of the mosaic of every slice, in
        acquisition order.

    Returns
    -------
    x_shift_mm, y_shift_mm : ndarray
        Displacement of the origin of slice ``i + 1`` relative to slice ``i``
        along the image X axis (columns) and the image Y axis (rows):
        ``ymin[i + 1] - ymin[i]`` and ``xmin[i + 1] - xmin[i]``.
    """
    xmin = np.asarray(xmin_mm, dtype=np.float64)
    ymin = np.asarray(ymin_mm, dtype=np.float64)
    if xmin.shape != ymin.shape or xmin.ndim != 1:
        raise ValueError("xmin_mm and ymin_mm must be one-dimensional and of the same length")
    return np.diff(ymin), np.diff(xmin)
