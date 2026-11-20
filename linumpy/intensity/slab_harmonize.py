"""Make tissue intensity consistent along a stack of serial slabs.

A stacked serial-OCT volume carries two intensity errors that follow the slab
structure, not the anatomy:

- every slab has its own overall gain (focus, source power, and the per-slab
  normalisation, which scales each slab by its brightest structures), and
- inside a slab the signal falls with depth at a rate that depends on the
  tissue, which leaves a band at every joint.

Both are estimated on a coarse grid in the log domain and removed with a gain
that is smooth in XY, so the image inside a plane keeps its detail.

Slab gains come from continuity: the same depth range of two neighbouring
slabs shows almost the same tissue, so the median difference between them is
the gain step. A loose pull towards the brain-wide level stops the chained
steps from drifting. The depth bands are whatever repeats with the slab
period: the smoothed log intensity minus its running mean over one period.

:func:`local_depth_field` is the depth band at the scale of one cell. It is
applied by the stacking step, before the two corrections above.
"""

import logging
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from linumpy.utils.planes import gaussian_filter_planes, map_threaded

logger = logging.getLogger(__name__)

_MIN_VALUE = 1e-4


def coarse_log_tissue(vol: np.ndarray, mask: np.ndarray, block: int = 8, chunk: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Mean log intensity of the tissue voxels in each ``block`` x ``block`` XY cell.

    Returns
    -------
    tuple
        ``(log_level, weight)``, both of shape ``(Z, Y // block, X // block)``.
        ``weight`` is the fraction of the cell that is tissue.
    """
    nz, ny, nx = vol.shape
    cy, cx = ny // block, nx // block
    if cy < 1 or cx < 1:
        msg = f"Volume plane {ny}x{nx} is smaller than one {block}-pixel cell"
        raise ValueError(msg)
    log_level = np.zeros((nz, cy, cx), dtype=np.float32)
    weight = np.zeros((nz, cy, cx), dtype=np.float32)

    def _reduce_chunk(z0: int) -> None:
        part = np.asarray(vol[z0 : z0 + chunk, : cy * block, : cx * block], dtype=np.float32)
        tissue = np.asarray(mask[z0 : z0 + chunk, : cy * block, : cx * block], dtype=bool) & (part > _MIN_VALUE)
        logs = np.where(tissue, np.log(np.maximum(part, _MIN_VALUE)), 0.0)
        n = part.shape[0]
        sums = logs.reshape(n, cy, block, cx, block).sum(axis=(2, 4))
        counts = tissue.reshape(n, cy, block, cx, block).sum(axis=(2, 4))
        log_level[z0 : z0 + n] = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
        weight[z0 : z0 + n] = counts / float(block * block)

    # The runs of planes are independent and write to their own part of the result.
    map_threaded(_reduce_chunk, range(0, nz, chunk), cap=8)
    return log_level, weight


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 0.5 * cumulative[-1])])


def slab_period(bounds: list[tuple[int, int]]) -> int:
    """Planes one slab contributes to the stack (the median slab height)."""
    heights = [end - start for start, end in bounds if end > start]
    return max(1, int(np.median(heights))) if heights else 1


def slab_log_gains(
    log_level: np.ndarray,
    weight: np.ndarray,
    bounds: list[tuple[int, int]],
    blend_planes: int = 0,
    step_sigma: float = 0.03,
    anchor_sigma: float = 0.15,
    min_weight: float = 0.5,
    min_cells: int = 30,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Log gain of every slab relative to the brain-wide tissue level.

    Parameters
    ----------
    log_level
        Coarse log intensity from :func:`coarse_log_tissue`.
    weight
        Tissue fraction of each coarse cell, same shape.
    bounds
        ``(start, end)`` plane range of each slab in the stack.
    blend_planes
        Planes at the top of each slab that the stack cross-fades with the
        slab above. They are left out of the slab level.
    step_sigma
        Expected error of one measured gain step between neighbours (log units).
    anchor_sigma
        How far the true level of a slab may sit from the brain-wide level
        because of its anatomy (log units). Larger values trust the chained
        steps over longer distances.
    min_weight
        A coarse cell counts when at least this fraction of it is tissue.
    min_cells
        A slab or a joint is measured only on at least this many cells.

    Returns
    -------
    tuple
        ``(gains, info)``. Subtract ``gains[s]`` from the log intensity of
        slab ``s``. ``info`` holds the measured steps and slab levels.
    """
    n_slabs = len(bounds)
    period = slab_period(bounds)
    levels: list[np.ndarray] = []
    cover: list[np.ndarray] = []
    for start, end in bounds:
        z0 = min(start + blend_planes, max(start, end - 1))
        z1 = max(z0 + 1, min(start + period, end))
        w = weight[z0:z1]
        total = w.sum(axis=0)
        levels.append(np.where(total > 0, (log_level[z0:z1] * w).sum(axis=0) / np.maximum(total, 1e-6), 0.0))
        cover.append(total / float(z1 - z0))

    anchors = np.full(n_slabs, np.nan)
    for s in range(n_slabs):
        ok = cover[s] > min_weight
        if int(ok.sum()) >= min_cells:
            anchors[s] = _weighted_median(levels[s][ok], cover[s][ok])

    steps = np.full(max(0, n_slabs - 1), np.nan)
    for s in range(n_slabs - 1):
        # Slabs on either side of a missing slice are not the same tissue. A
        # joint placed at its measured depth can be a few planes off the period;
        # a missing slice is a whole period more.
        if bounds[s + 1][0] - bounds[s][0] > period + period // 2:
            continue
        ok = (cover[s] > min_weight) & (cover[s + 1] > min_weight)
        if int(ok.sum()) >= min_cells:
            steps[s] = _weighted_median((levels[s + 1] - levels[s])[ok], np.minimum(cover[s], cover[s + 1])[ok])

    gains = np.zeros(n_slabs, dtype=np.float64)
    if n_slabs and np.isfinite(anchors).any():
        reference = float(np.nanmedian(anchors))
        system = np.zeros((n_slabs, n_slabs), dtype=np.float64)
        rhs = np.zeros(n_slabs, dtype=np.float64)
        w_anchor = 1.0 / anchor_sigma**2
        w_step = 1.0 / step_sigma**2
        for s in range(n_slabs):
            if np.isfinite(anchors[s]):
                system[s, s] += w_anchor
                rhs[s] += w_anchor * (anchors[s] - reference)
            else:
                # No tissue to measure: follow the neighbours, else stay at zero.
                system[s, s] += 1e-6 * w_anchor
        for s in range(n_slabs - 1):
            if np.isfinite(steps[s]):
                system[s, s] += w_step
                system[s + 1, s + 1] += w_step
                system[s, s + 1] -= w_step
                system[s + 1, s] -= w_step
                rhs[s] -= w_step * steps[s]
                rhs[s + 1] += w_step * steps[s]
        gains = np.linalg.solve(system, rhs)
    return gains, {"steps": steps, "anchors": anchors, "period": period}


def plane_log_gains(gains: np.ndarray, bounds: list[tuple[int, int]], n_planes: int, blend_planes: int = 0) -> np.ndarray:
    """Spread slab gains over planes, ramping across the cross-faded planes."""
    out = np.zeros(n_planes, dtype=np.float32)
    for s, (start, end) in enumerate(bounds):
        out[start:end] = gains[s]
        if s > 0 and blend_planes > 0:
            n = min(blend_planes, end - start)
            ramp = (np.arange(n) + 0.5) / n
            out[start : start + n] = (1.0 - ramp) * gains[s - 1] + ramp * gains[s]
    if bounds:
        out[: bounds[0][0]] = gains[0]
        out[bounds[-1][1] :] = gains[-1]
    return out


def depth_band_field(
    log_level: np.ndarray,
    weight: np.ndarray,
    period: int,
    sigma_cells: float,
    clamp: float = 0.7,
) -> np.ndarray:
    """Log intensity that repeats with the slab period, smooth in XY.

    The smoothed log level minus its running mean over one slab period. A
    running mean over exactly one period removes anything periodic and keeps
    a linear trend, so tissue that changes steadily with depth is untouched.
    Tissue thinner than one slab and wider than the XY smoothing is spread
    over one period in Z at that coarse scale only.
    """
    if period < 2:
        return np.zeros_like(log_level)
    smooth_w = gaussian_filter_planes(weight, float(sigma_cells))
    smooth = gaussian_filter_planes(log_level * weight, float(sigma_cells)) / np.maximum(smooth_w, 1e-3)
    kernel = np.full(period, 1.0 / period, dtype=np.float32)
    mean_w = ndi.convolve1d(smooth_w, kernel, axis=0, mode="nearest")
    mean = ndi.convolve1d(smooth * smooth_w, kernel, axis=0, mode="nearest") / np.maximum(mean_w, 1e-3)
    field = np.where((smooth_w > 1e-3) & (mean_w > 1e-3), smooth - mean, 0.0)
    return np.clip(field, -clamp, clamp).astype(np.float32)


def _nanmedian_axis0(values: np.ndarray, count: np.ndarray) -> np.ndarray:
    """Median along axis 0 that ignores NaN, given the number of finite values per column.

    Same values as ``np.nanmedian(values, axis=0)``, NaN where a column has no
    finite value. ``np.nanmedian`` raises one warning per such column from a
    Python loop, which on a stack with background is most of its run time.
    """
    ordered = np.sort(values, axis=0)  # NaN sorts last
    low = np.take_along_axis(ordered, np.maximum((count - 1) // 2, 0)[None], axis=0)[0]
    high = np.take_along_axis(ordered, (count // 2)[None], axis=0)[0]
    return np.where(count > 0, 0.5 * (low + high), np.nan)


def local_depth_field(
    log_level: np.ndarray,
    weight: np.ndarray,
    bounds: list[tuple[int, int]],
    half_window: int = 3,
    min_slabs: int = 4,
    min_weight: float = 0.9,
    sigma_cells: float = 1.0,
    clamp: float = 0.7,
) -> np.ndarray:
    """Depth profile of the log intensity inside each slab, for every XY cell.

    The signal that is left after the depth corrections still changes with
    depth below the cut face, by an amount that depends on the tissue: white
    matter keeps a stronger trend than grey matter. That leaves one band per
    slab, with a different strength in every structure.

    In each cell, the profile of a slab is its log level minus its own mean.
    The profile of the system is the median of that over the slab and its
    neighbours along the stack: anatomy differs from slab to slab and drops
    out of the median, the depth trend is the same in all of them and stays.
    The result is zero-mean over every slab, so the level of a slab in a cell
    does not change, and nothing is filtered along the stacking axis.

    Parameters
    ----------
    log_level
        Coarse log intensity from :func:`coarse_log_tissue`.
    weight
        Tissue fraction of each coarse cell, same shape.
    bounds
        ``(start, end)`` plane range of each slab in the stack.
    half_window
        Slabs on each side that enter the median.
    min_slabs
        A profile is used only where at least this many slabs of the window
        have the cell filled with tissue. With fewer, the median would follow
        the anatomy of those slabs.
    min_weight
        A cell counts in a slab when more than this fraction of it is tissue
        in every plane of the slab.
    sigma_cells
        XY smoothing of the profile, in cells. It also carries the profile
        into the cells at the tissue edge that have none of their own.
    clamp
        Largest correction, in log units.

    Returns
    -------
    np.ndarray
        Field of the shape of ``log_level``. Subtract it from the log intensity.
    """
    field = np.zeros(log_level.shape, dtype=np.float32)
    n_slabs = len(bounds)
    period = slab_period(bounds)
    if n_slabs < min_slabs or period < 2:
        return field
    profiles = np.full((n_slabs, period, *log_level.shape[1:]), np.nan, dtype=np.float32)
    for s, (start, end) in enumerate(bounds):
        n = min(end - start, period)
        if n < 2:
            continue
        segment = log_level[start : start + n]
        solid = (weight[start : start + n] > min_weight).all(axis=0)
        profiles[s, :n] = np.where(solid, segment - segment.mean(axis=0), np.nan)

    width = min(n_slabs, 2 * half_window + 1)
    sigma = (0.0, float(sigma_cells), float(sigma_cells))
    for s, (start, end) in enumerate(bounds):
        n = min(end - start, period)
        if n < 2:
            continue
        # At the ends of the stack the window keeps its width and becomes one-sided.
        first = min(max(0, s - half_window), n_slabs - width)
        window = profiles[first : first + width]
        count = np.isfinite(window).sum(axis=0)
        known = count >= min_slabs
        median = _nanmedian_axis0(window, count)
        cover = ndi.gaussian_filter(known.astype(np.float32), sigma)
        smooth = ndi.gaussian_filter(np.where(known, median, 0.0).astype(np.float32), sigma) / np.maximum(cover, 1e-3)
        profile = np.where(cover > 0.3, smooth, 0.0)[:n]
        profile -= profile.mean(axis=0)
        field[start : start + n] = profile
        # A slab that reaches deeper than one period (before a missing slice) keeps its last value.
        field[start + n : end] = profile[n - 1]
    return np.clip(field, -clamp, clamp)


def upsample_plane(field: np.ndarray, block: int, shape: tuple[int, int]) -> np.ndarray:
    """Bring one plane of a coarse field to the pixel grid (bilinear, edge cells repeated)."""
    up = ndi.zoom(field, block, order=1, mode="nearest", grid_mode=True)
    pad_y = shape[0] - up.shape[0]
    pad_x = shape[1] - up.shape[1]
    if pad_y > 0 or pad_x > 0:
        up = np.pad(up, ((0, max(0, pad_y)), (0, max(0, pad_x))), mode="edge")
    return up[: shape[0], : shape[1]]


def harmonize_slab_intensity(
    vol: np.ndarray,
    mask: np.ndarray,
    bounds: list[tuple[int, int]],
    *,
    slab_gain: bool = True,
    band_sigma_px: float = 0.0,
    target_level: float = 0.0,
    blend_planes: int = 0,
    block: int = 8,
    max_gain: float = 4.0,
) -> dict[str, Any]:
    """Correct slab gains and depth bands of a stacked volume in place.

    Parameters
    ----------
    vol
        Stacked volume ``(Z, Y, X)``, float32. Tissue voxels are rescaled in place.
    mask
        Tissue mask of the same shape.
    bounds
        ``(start, end)`` plane range of each slab.
    slab_gain
        Bring every slab to a common gain (see :func:`slab_log_gains`).
    band_sigma_px
        XY scale in pixels of the depth-band correction. 0 disables it.
    target_level
        When > 0, scale the whole volume so that the brain-wide tissue level
        (median of the coarse log level) equals this value. Volumes of
        different brains then share one intensity scale.
    blend_planes
        Planes at the top of each slab that the stack cross-fades.
    block
        XY cell size of the coarse grid in pixels.
    max_gain
        Largest factor (and smallest, as its inverse) applied to any voxel.

    Returns
    -------
    dict
        Diagnostics: slab gains, measured steps, level before and after.
    """
    log_level, weight = coarse_log_tissue(vol, mask, block=block)
    n_planes = vol.shape[0]
    info: dict[str, Any] = {"period": slab_period(bounds)}

    plane_gain = np.zeros(n_planes, dtype=np.float32)
    if slab_gain and len(bounds) > 1:
        gains, gain_info = slab_log_gains(log_level, weight, bounds, blend_planes=blend_planes)
        plane_gain = plane_log_gains(gains, bounds, n_planes, blend_planes=blend_planes)
        log_level -= plane_gain[:, None, None]
        info.update(slab_gains=gains, steps=gain_info["steps"], anchors=gain_info["anchors"])

    field: np.ndarray | None = None
    if band_sigma_px > 0:
        field = depth_band_field(log_level, weight, info["period"], band_sigma_px / float(block))
        log_level -= field
        info["band_rms"] = float(np.sqrt(np.mean(field[weight > 0.5] ** 2))) if (weight > 0.5).any() else 0.0

    solid = weight > 0.9
    level = float(np.exp(np.median(log_level[solid]))) if solid.any() else float("nan")
    info["level"] = level
    offset = 0.0
    if target_level > 0 and np.isfinite(level) and level > 0:
        offset = float(np.log(level / target_level))
    info["scale"] = float(np.exp(-offset))

    limit = float(np.log(max_gain))
    plane_shape = (vol.shape[1], vol.shape[2])

    def _correct_plane(z: int) -> None:
        log_correction = float(plane_gain[z]) + offset
        tissue = np.asarray(mask[z], dtype=bool)
        if not tissue.any():
            return
        if field is not None:
            total = np.clip(upsample_plane(field[z], block, plane_shape) + log_correction, -limit, limit)
            factor = np.exp(-total).astype(np.float32)
            vol[z] = np.where(tissue, vol[z] * factor, vol[z])
        elif log_correction != 0.0:
            factor = np.float32(np.exp(-np.clip(log_correction, -limit, limit)))
            vol[z] = np.where(tissue, vol[z] * factor, vol[z])

    # Every plane gets its own gain, so the planes are corrected side by side.
    map_threaded(_correct_plane, range(n_planes))
    return info
