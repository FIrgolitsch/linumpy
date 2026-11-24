"""Shift of the slab image origin between slices, from the stage positions of the mosaic."""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from linumpy.mosaic.motor import compute_affine_output_shape
from linumpy.stack_alignment.units import detect_shift_units


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


@dataclass(frozen=True)
class TileLattice:
    """Displacement model of the stage, as fitted when the tiles of a slab are stitched.

    A slab image lives in the laser scanning frame. Moving the stage by one
    tile along its X or Y axis moves the image by a column of *transform*:
    a little less than the nominal tile step, and slightly rotated, because
    the scanned field is not exactly its nominal size and the scanning frame
    is not exactly aligned with the stage (Lefebvre et al. 2017, Sec. 2.4).
    The same model holds for the move of the slab origin between two slices,
    since it is the same stage and the same scanner.

    Attributes
    ----------
    transform : ndarray
        2 x 2 matrix giving the ``(row, col)`` displacement in pixels per tile
        index. The first index runs along stage X (image rows), the second
        along stage Y (image columns).
    tile_shape : tuple of int
        ``(height, width)`` of a tile in pixels.
    step_mm : tuple of float
        Stage travel per tile index along stage X and stage Y.
    """

    transform: np.ndarray
    tile_shape: tuple[int, int]
    step_mm: tuple[float, float]

    @classmethod
    def load(cls, path: Path | str, resolution: Sequence[float]) -> TileLattice:
        """Read the model written by ``linum-estimate-global-transform --diagnostics_json``.

        Parameters
        ----------
        path : Path or str
            The JSON file. It holds the fitted transform, the tile shape and
            the nominal overlap of the acquisition.
        resolution : sequence of float
            Resolution of the stitched slabs as given by ``read_omezarr``.
        """
        content = json.loads(Path(path).read_text())
        transform = np.asarray(content["transform"], dtype=np.float64)
        if transform.shape != (2, 2):
            raise ValueError(f"The transform in {path} is not 2 x 2")
        tile_h, tile_w = (int(v) for v in content["tile_shape"][-2:])
        overlap = float(content["overlap_fraction"])
        res_x_um, res_y_um = detect_shift_units(resolution)
        step_mm = (tile_h * res_y_um / 1000.0 * (1.0 - overlap), tile_w * res_x_um / 1000.0 * (1.0 - overlap))
        return cls(transform=transform, tile_shape=(tile_h, tile_w), step_mm=step_mm)

    def displacement_px(self, image_x_mm: float, image_y_mm: float) -> tuple[float, float]:
        """Return the pixel displacement ``(dx, dy)`` of a slab origin.

        Parameters
        ----------
        image_x_mm, image_y_mm : float
            Stage displacement in mm along image X (columns, stage Y) and
            image Y (rows, stage X), as stored in the shifts file.
        """
        index = np.array([image_y_mm / self.step_mm[0], image_x_mm / self.step_mm[1]])
        row, col = self.transform @ index
        return float(col), float(row)

    def grid_shape(self, image_shape: Sequence[int]) -> tuple[int, int] | None:
        """Return the tile grid ``(nx, ny)`` whose stitched image has this ``(height, width)``.

        None when no grid gives exactly this shape, which means the slab was
        not stitched with this model.
        """
        height, width = int(image_shape[-2]), int(image_shape[-1])
        tile = (1, self.tile_shape[0], self.tile_shape[1])
        guess_x = round((height - self.tile_shape[0]) / max(abs(self.transform[0, 0]), 1e-6)) + 1
        guess_y = round((width - self.tile_shape[1]) / max(abs(self.transform[1, 1]), 1e-6)) + 1
        for nx in range(max(1, guess_x - 3), guess_x + 4):
            for ny in range(max(1, guess_y - 3), guess_y + 4):
                _, h, w = compute_affine_output_shape(nx, ny, tile, self.transform)
                if (h, w) == (height, width):
                    return nx, ny
        return None

    def first_tile_px(self, nx: int, ny: int) -> tuple[float, float]:
        """Return where the first tile sits in a stitched image of ``nx`` by ``ny`` tiles, as ``(row, col)``.

        The stitcher lays the tiles on the lattice and then moves them all so
        that none has a negative position. With a rotated lattice the tile
        that ends up at zero along an axis is not always the first one, so
        the first tile, which is the one at the low corner of the mosaic on
        the stage, sits at an offset that depends on the size of the grid.
        """
        corners = [(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1)]
        positions = [np.round(self.transform @ np.array(corner, dtype=np.float64)) for corner in corners]
        return float(-min(p[0] for p in positions)), float(-min(p[1] for p in positions))
