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
from typing import Any

import numpy as np
from scipy import ndimage as ndi
from scipy.fft import next_fast_len

from linumpy.gpu import get_array_module, to_cpu
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


def _modules(use_gpu: bool) -> tuple[Any, Any]:
    """Array module and its ``ndimage``: CuPy when the GPU is asked for and there, NumPy otherwise."""
    xp = get_array_module(use_gpu)
    if xp is np:
        return np, ndi
    import cupyx.scipy.ndimage as gpu_ndi

    return xp, gpu_ndi


def _band_pass(plane: Any, filters: Any, mode: str) -> Any:
    return filters.gaussian_filter(plane, _FINE_SIGMA, mode=mode) - filters.gaussian_filter(plane, _COARSE_SIGMA, mode=mode)


def _correlation_peak(
    fixed: Any, fixed_tissue: Any, moving: Any, moving_tissue: Any, radius: int, min_pixels: int, xp: Any, filters: Any
) -> tuple[np.ndarray, float] | None:
    """Shift (rows, cols) of ``moving`` that matches ``fixed`` best, and the correlation there.

    Both planes are band-passed and on the same grid, with their tissue as
    masks. The peak is refined to a fraction of a pixel with a parabola
    through its neighbours. Returns ``None`` when the common tissue is too
    small or the peak lies on the edge of the search window.
    """
    common = filters.binary_erosion(fixed_tissue & moving_tissue, iterations=_EDGE_PX, brute_force=True)
    n_common = int(common.sum())
    if n_common < min_pixels:
        return None
    a = (fixed - fixed[common].mean()) * common
    b = (moving - moving[common].mean()) * common
    norm = float(xp.sqrt((a * a).sum() * (b * b).sum()))
    if norm <= 0:
        return None
    # Padded to a size the FFT is fast at, and by the search radius so that
    # a shift never wraps the far edge of the plane into the match.
    size = tuple(next_fast_len(n + radius, real=True) for n in a.shape)
    correlation = xp.fft.irfft2(xp.fft.rfft2(a, s=size) * xp.conj(xp.fft.rfft2(b, s=size)), s=size)
    offsets = np.arange(-radius, radius + 1)
    window = correlation[xp.asarray(offsets % size[0])[:, None], xp.asarray(offsets % size[1])[None, :]]
    window = np.asarray(to_cpu(window), dtype=np.float64) / norm
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
    use_gpu: bool = True,
) -> tuple[np.ndarray, float] | None:
    """Sideways shift between two planes that show the same tissue.

    The moving plane is resampled onto the grid of the fixed plane at
    ``offset`` plus the shift found so far, and the remaining shift is
    measured again until it is zero. The estimate at a remaining shift of
    zero has no pull towards whole pixels, which a single correlation peak
    has.

    Both planes are band-passed once. A band-pass and a shift can be done
    in either order, so the rounds only resample the filtered moving plane.

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
    use_gpu : bool
        Run on the GPU when there is one.

    Returns
    -------
    tuple of (np.ndarray, float) or None
        ``(rows, cols)`` to add to ``offset`` so that the planes match, and
        the correlation at that shift. ``None`` when no shift could be
        measured.
    """
    xp, filters = _modules(use_gpu)
    fixed = xp.asarray(fixed, dtype=xp.float32)
    moving = xp.asarray(moving, dtype=xp.float32)
    fixed_tissue = fixed > 0
    fixed_band = _band_pass(fixed, filters, "reflect")
    # Outside its own array the moving plane counts as empty, as it is once placed.
    moving_band = _band_pass(moving, filters, "constant")
    moving_tissue = (moving > 0).astype(xp.float32)
    identity = xp.eye(2)

    def placed(plane: Any, total: np.ndarray) -> Any:
        return filters.affine_transform(
            plane,
            identity,
            offset=(-(offset[0] + float(total[0])), -(offset[1] + float(total[1]))),
            output_shape=fixed.shape,
            order=1,
            mode="constant",
            cval=0.0,
        )

    total = np.zeros(2)
    peak = 0.0
    for _ in range(max_rounds):
        found = _correlation_peak(
            fixed_band,
            fixed_tissue,
            placed(moving_band, total),
            placed(moving_tissue, total) > 0,
            radius,
            min_pixels,
            xp,
            filters,
        )
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
    use_gpu: bool = True,
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
    use_gpu : bool
        Run on the GPU when there is one.

    Returns
    -------
    dict
        ``slant_x_px_per_plane`` and ``slant_y_px_per_plane`` (zero when
        ``used`` is false), their standard errors, and the shift of every
        measured joint.
    """
    ordered = sorted(slabs, key=lambda slab: slab[0])
    depth_needed = step_planes + max(planes) + 1

    def top_planes(path: Path) -> np.ndarray:
        # One read per slab: it is the incoming slab of one joint and the slab above of the next.
        volume, _ = read_omezarr(Path(path))
        return np.asarray(volume[:depth_needed])

    joints = []
    fixed_vol = top_planes(ordered[0][1]) if ordered else None
    for (fixed_id, _fixed_path, fixed_x, fixed_y), (moving_id, moving_path, moving_x, moving_y) in pairwise(ordered):
        moving_vol = top_planes(moving_path)
        if moving_id - fixed_id == 1 and fixed_vol is not None:
            offset = (moving_y - fixed_y, moving_x - fixed_x)
            shifts = []
            for k in planes:
                if step_planes + k >= fixed_vol.shape[0] or k >= moving_vol.shape[0]:
                    continue
                found = plane_shift(fixed_vol[step_planes + k], moving_vol[k], offset, use_gpu=use_gpu)
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
        fixed_vol = moving_vol

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
