"""Slant inside the slabs: the sideways drift of the image with depth.

The A-lines of a slab are not exactly parallel to the vertical axis of the
stage. A structure therefore sits a little further sideways in every deeper
plane of a slab, by the same amount in every slab. Two slabs that are placed
correctly from the stage positions then still disagree in their overlap: the
top planes of the incoming slab show the same tissue as planes one slice
thickness deeper in the slab above, and those are displaced by the slant
times that depth.

The slant is measured from that disagreement over all the joints of a brain.
It is removed when a slab is resampled into the common frame, so the stage
placement of the slabs stays as it is. Applying the disagreement as a shift
of each slab instead would shear the whole brain.
"""

import json
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from linumpy.io.zarr import read_omezarr

SLANT_FILENAME = "slab_slant.json"

# Structure between these two scales (pixels) carries the sideways match:
# finer is noise, coarser is the slab-wide brightness.
_FINE_SIGMA = 1.5
_COARSE_SIGMA = 20.0
# The common tissue is eroded by this many pixels, so the edge of a slab,
# where the two slabs end differently, stays out of the match.
_EDGE_PX = 8
# A refinement round that moves the estimate less than this ends the search.
_CONVERGED_PX = 0.02


def _band_pass(plane: np.ndarray) -> np.ndarray:
    return ndi.gaussian_filter(plane, _FINE_SIGMA) - ndi.gaussian_filter(plane, _COARSE_SIGMA)


def _correlation_peak(fixed: np.ndarray, moving: np.ndarray, radius: int, min_pixels: int) -> tuple[np.ndarray, float] | None:
    """Shift (rows, cols) of ``moving`` that matches ``fixed`` best, and the correlation there.

    Both planes are on the same grid, with zeros outside the tissue. The
    peak is refined to a fraction of a pixel with a parabola through its
    neighbours. Returns ``None`` when the common tissue is too small or the
    peak lies on the edge of the search window.
    """
    common = ndi.binary_erosion((fixed > 0) & (moving > 0), iterations=_EDGE_PX)
    if int(common.sum()) < min_pixels:
        return None
    a = _band_pass(fixed)
    b = _band_pass(moving)
    a = (a - a[common].mean()) * common
    b = (b - b[common].mean()) * common
    norm = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if norm <= 0:
        return None
    correlation = np.fft.irfft2(np.fft.rfft2(a) * np.conj(np.fft.rfft2(b)), s=a.shape) / norm
    offsets = np.arange(-radius, radius + 1)
    window = correlation[np.ix_(offsets % a.shape[0], offsets % a.shape[1])]
    row, col = np.unravel_index(int(window.argmax()), window.shape)
    if not (0 < row < 2 * radius and 0 < col < 2 * radius):
        return None

    def vertex(before: float, centre: float, after: float) -> float:
        curvature = before - 2.0 * centre + after
        return 0.0 if curvature >= 0 else 0.5 * (before - after) / curvature

    shift = np.array(
        [
            row - radius + vertex(window[row - 1, col], window[row, col], window[row + 1, col]),
            col - radius + vertex(window[row, col - 1], window[row, col], window[row, col + 1]),
        ]
    )
    return shift, float(window[row, col])


def plane_shift(
    fixed: np.ndarray,
    moving: np.ndarray,
    offset: tuple[float, float] = (0.0, 0.0),
    *,
    radius: int = 8,
    min_pixels: int = 20000,
    max_rounds: int = 4,
) -> tuple[np.ndarray, float] | None:
    """Sideways shift between two planes that show the same tissue.

    The moving plane is resampled onto the grid of the fixed plane at
    ``offset`` plus the shift found so far, and the remaining shift is
    measured again until it is zero. The estimate at a remaining shift of
    zero has no pull towards whole pixels, which a single correlation peak
    has.

    Parameters
    ----------
    fixed, moving : np.ndarray
        Two planes with zeros outside the tissue. They may differ in shape.
    offset : tuple of float
        ``(rows, cols)`` position of the origin of ``moving`` on the grid of
        ``fixed``, as the placement puts it.
    radius : int
        Largest shift searched in one round, in pixels.
    min_pixels : int
        Smallest common tissue area that is measured.
    max_rounds : int
        Number of refinement rounds at most.

    Returns
    -------
    tuple of (np.ndarray, float) or None
        ``(rows, cols)`` to add to ``offset`` so that the planes match, and
        the correlation at that shift. ``None`` when no shift could be
        measured.
    """
    fixed = np.asarray(fixed, dtype=np.float32)
    moving = np.asarray(moving, dtype=np.float32)
    total = np.zeros(2)
    peak = 0.0
    for _ in range(max_rounds):
        placed = ndi.affine_transform(
            moving,
            np.eye(2),
            offset=(-(offset[0] + total[0]), -(offset[1] + total[1])),
            output_shape=fixed.shape,
            order=1,
            mode="constant",
            cval=0.0,
        )
        found = _correlation_peak(fixed, placed, radius, min_pixels)
        if found is None:
            return None
        step, peak = found
        total += step
        if float(np.abs(step).max()) < _CONVERGED_PX:
            break
    return total, peak


def estimate_slab_slant(
    slabs: Sequence[tuple[int, Path, float, float]],
    step_planes: int,
    *,
    planes: Sequence[int] = (3, 6, 9, 12),
    min_correlation: float = 0.2,
    min_joints: int = 5,
) -> dict:
    """Measure the slant inside the slabs of one brain from the overlaps of neighbouring slabs.

    Plane ``k`` of a slab shows the tissue of plane ``step_planes + k`` of
    the slab above. With a slant ``g`` (pixels per plane) the slab above
    shows it displaced by ``g * step_planes`` more, so the shift that
    matches the two, divided by ``step_planes``, is the slant. The median
    over the planes of a joint and then over the joints is taken: a single
    joint with a placement error of its own does not move it.

    Parameters
    ----------
    slabs : sequence of (slice_id, path, shift_x_px, shift_y_px)
        The slabs before they are brought into the common frame, with the
        position the plan gives each of them.
    step_planes : int
        Slice thickness in planes.
    planes : sequence of int
        Planes of the incoming slab that are measured.
    min_correlation : float
        Plane pairs that match worse than this are left out.
    min_joints : int
        With fewer measured joints no slant is reported.

    Returns
    -------
    dict
        ``slant_x_px_per_plane`` and ``slant_y_px_per_plane`` (zero when
        ``used`` is false), their standard errors, and the shift of every
        measured joint.
    """
    ordered = sorted(slabs, key=lambda slab: slab[0])
    joints = []
    for (fixed_id, fixed_path, fixed_x, fixed_y), (moving_id, moving_path, moving_x, moving_y) in pairwise(ordered):
        if moving_id - fixed_id != 1:
            continue
        fixed_vol, _ = read_omezarr(Path(fixed_path))
        moving_vol, _ = read_omezarr(Path(moving_path))
        offset = (moving_y - fixed_y, moving_x - fixed_x)
        shifts = []
        for k in planes:
            if step_planes + k >= fixed_vol.shape[0] or k >= moving_vol.shape[0]:
                continue
            found = plane_shift(np.asarray(fixed_vol[step_planes + k]), np.asarray(moving_vol[k]), offset)
            if found is not None and found[1] >= min_correlation:
                shifts.append(found[0])
        if shifts:
            rows, cols = np.median(np.array(shifts), axis=0)
            joints.append(
                {
                    "fixed_id": int(fixed_id),
                    "moving_id": int(moving_id),
                    "shift_x_px": float(cols),
                    "shift_y_px": float(rows),
                    "n_planes": len(shifts),
                }
            )

    result: dict = {
        "step_planes": int(step_planes),
        "n_joints": len(joints),
        "used": len(joints) >= min_joints,
        "slant_x_px_per_plane": 0.0,
        "slant_y_px_per_plane": 0.0,
        "standard_error_x": None,
        "standard_error_y": None,
        "joint_shift_x_px": None,
        "joint_shift_y_px": None,
        "joints": joints,
    }
    if joints:
        per_joint = np.array([[j["shift_x_px"], j["shift_y_px"]] for j in joints])
        median = np.median(per_joint, axis=0)
        # Robust standard error of the median: 1.2533 * 1.4826 * MAD / sqrt(n).
        spread = 1.2533 * 1.4826 * np.median(np.abs(per_joint - median), axis=0) / np.sqrt(len(joints))
        result["joint_shift_x_px"] = float(median[0])
        result["joint_shift_y_px"] = float(median[1])
        result["standard_error_x"] = float(spread[0] / step_planes)
        result["standard_error_y"] = float(spread[1] / step_planes)
        if result["used"]:
            result["slant_x_px_per_plane"] = float(median[0] / step_planes)
            result["slant_y_px_per_plane"] = float(median[1] / step_planes)
    return result


def write_slab_slant(result: dict, path: Path) -> None:
    """Write the result of `estimate_slab_slant` as JSON."""
    Path(path).write_text(json.dumps(result, indent=2) + "\n")


def read_slab_slant(path: Path) -> tuple[float, float]:
    """Read the slant ``(x, y)`` in pixels per plane from a file written by `write_slab_slant`."""
    result = json.loads(Path(path).read_text())
    return float(result["slant_x_px_per_plane"]), float(result["slant_y_px_per_plane"])
