"""
3D slice stacking utilities.

Consolidated from linum_stack_slices_motor.py and linum_stack_motor_only.py.
"""

from linumpy.config.threads import thread_budget

import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise
from typing import Any

import numpy as np

from linumpy.intensity.slab_harmonize import coarse_log_tissue, local_depth_field, upsample_plane
from linumpy.utils.planes import gaussian_filter_planes

logger = logging.getLogger(__name__)


def enforce_z_consistency(
    z_matches: list,
    confidence_per_slice: dict | None = None,
    outlier_threshold_frac: float = 0.30,
    confidence_protect_threshold: float = 0.6,
) -> tuple[list, list]:
    """Correct outlier Z-overlaps using neighbor interpolation.

    Scans pairwise Z-overlap measurements for outliers (deviating more than
    ``outlier_threshold_frac`` from the median) and replaces them with the
    local median of their immediate neighbors.  Both ``overlap_voxels`` and
    ``blend_overlap_voxels`` are corrected independently.

    Slices whose registration confidence (from ``confidence_per_slice``)
    meets or exceeds ``confidence_protect_threshold`` are considered reliable
    and are not modified.

    Parameters
    ----------
    z_matches : list of dict
        Each dict must have keys ``overlap_voxels``, ``blend_overlap_voxels``
        and ``moving_id``.  Items are modified in-place.
    confidence_per_slice : dict or None
        Mapping from ``moving_id`` (int) to confidence score in [0, 1].
        Slices with confidence >= ``confidence_protect_threshold`` are skipped.
        If None, all slices are treated as having confidence 0.5.
    outlier_threshold_frac : float
        Fractional deviation from median above which a value is an outlier.
        Default: 0.30 (30 %).
    confidence_protect_threshold : float
        Minimum confidence to protect a slice from correction.  Default: 0.6.

    Returns
    -------
    z_matches : list of dict
        The corrected z_matches list (same objects, modified in-place).
    corrections : list of dict
        Log of corrections: each entry has keys ``moving_id``, ``field``,
        ``old_value`` and ``new_value``.
    """
    if len(z_matches) < 3:
        return z_matches, []

    conf = confidence_per_slice or {}
    corrections = []

    for field in ("overlap_voxels", "blend_overlap_voxels"):
        values = np.array([float(m[field]) for m in z_matches])
        median_val = float(np.median(values))
        threshold = outlier_threshold_frac * max(median_val, 1.0)

        for i, match in enumerate(z_matches):
            slice_id = match.get("moving_id", i)

            # Protect high-confidence registrations from correction
            if conf.get(slice_id, 0.5) >= confidence_protect_threshold:
                continue

            # Missing-slice gaps (e.g. 49→51) must keep the ID-step overlap,
            # including a negative value that inserts empty Z.
            fixed_id = match.get("fixed_id")
            if fixed_id is not None and int(slice_id) - int(fixed_id) > 1:
                continue

            deviation = abs(float(match[field]) - median_val)
            if deviation <= threshold:
                continue

            old_val = match[field]
            neighbor_vals = []
            if i > 0:
                neighbor_vals.append(float(z_matches[i - 1][field]))
            if i + 1 < len(z_matches):
                neighbor_vals.append(float(z_matches[i + 1][field]))

            new_val = int(np.median(neighbor_vals)) if neighbor_vals else int(median_val)
            match[field] = new_val
            corrections.append(
                {
                    "moving_id": slice_id,
                    "field": field,
                    "old_value": old_val,
                    "new_value": new_val,
                }
            )

    return z_matches, corrections


def expected_z_overlap(vol_nz: int, moving_z: int, interval_voxels: int, id_step: int = 1) -> int:
    """Return expected Z overlap from slicing interval, allowing a physical gap.

    ``id_step`` is the slice-index delta (1 for consecutive IDs). When a
    slice is missing, ``id_step=2`` uses two intervals so neighbors are not
    glued together. Negative overlap means empty voxels between slabs.

    ``vol_nz`` is the depth of the PREVIOUS slab: each cut advances the surface
    by one interval, so the slab above reaches ``vol_nz - interval`` voxels past
    the next cut face. Using the incoming slab's depth misplaces every joint
    where the depth changes (up to the full depth difference).

    ``moving_z`` is a discarded-plane count on the incoming slab, not the
    pairwise 2-D Euler template index. After ``crop_interface`` it should be
    0 so the cut face enters the overlap.
    """
    step = max(1, int(id_step))
    return int(vol_nz) - int(moving_z or 0) - int(interval_voxels) * step


SLAB_Z_STARTS_KEY = "slab_z_starts"
SLAB_Z_DEPTH_KEY = "slab_z_depth"


def write_slab_z_starts(zarr_path: Any, starts: Sequence[int], n_z: int) -> None:
    """Record where each stacked slab starts in the OME-Zarr root attributes.

    ``n_z`` is the Z depth the starts refer to, so a reader can rescale them
    if the volume was resampled afterwards.
    """
    import zarr

    group = zarr.open_group(str(zarr_path), mode="r+")
    group.attrs[SLAB_Z_STARTS_KEY] = [int(s) for s in starts]
    group.attrs[SLAB_Z_DEPTH_KEY] = int(n_z)


def read_slab_bounds(zarr_path: Any, n_z: int) -> list[tuple[int, int]] | None:
    """Return ``(start, end)`` Z bounds per stacked slab for a volume of depth ``n_z``.

    Returns None when the stack did not record slab starts or they are not
    usable, so callers fall back to equal chunks.
    """
    import zarr

    try:
        attrs = zarr.open_group(str(zarr_path), mode="r").attrs
        raw_starts = attrs.get(SLAB_Z_STARTS_KEY)
        raw_depth = attrs.get(SLAB_Z_DEPTH_KEY)
        if not isinstance(raw_starts, list) or not isinstance(raw_depth, int | float):
            return None
        starts = [float(s) for s in raw_starts]
    except Exception:
        return None
    if len(starts) < 2 or raw_depth <= 0:
        return None
    scale = n_z / float(raw_depth)
    scaled = [round(s * scale) for s in starts]
    scaled = [min(max(s, 0), n_z) for s in scaled]
    if scaled[0] != 0 or any(b <= a for a, b in pairwise(scaled)):
        return None
    ends = [*scaled[1:], n_z]
    return list(zip(scaled, ends, strict=True))


def registration_z_overlap(prev_nz: int, moving_nz: int, fixed_z: int, template_z: int = 0) -> int:
    """Overlap from pairwise ``best_z``, without cropping the incoming cut face.

    ``fixed_z`` is the plane in the previous slab where the moving template
    plane landed. ``template_z`` is that 2-D template index. It is added into
    the overlap so the same plane still meets ``fixed_z`` while plane 0 of the
    incoming slab is kept. The result is clamped to both slabs.
    """
    overlap = max(0, int(prev_nz) - int(fixed_z) + int(template_z or 0))
    return int(min(overlap, max(0, int(prev_nz)), max(0, int(moving_nz))))


def crop_moving_volume(vol: np.ndarray, moving_z_start: int = 0) -> np.ndarray:
    """Drop leading Z planes from a moving slab before Hann/paste.

    ``moving_z_start`` is a discarded-plane count, not the pairwise
    ``moving_z_index`` (2-D Euler template). After ``crop_interface``, pass 0
    so the incoming cut face participates in the overlap.
    """
    mz = max(0, int(moving_z_start or 0))
    if mz <= 0:
        return vol
    return vol[mz:]


def resolve_z_match_backend(use_gpu: bool) -> str:
    """Return the backend the Z-overlap sweep runs on.

    This is the one place that decides between the CPU and CUDA sweep.
    :func:`find_z_overlap` selects its array module from the answer, and the
    stage timer labels the same call with it, so the label cannot drift from
    the branch that ran.

    Parameters
    ----------
    use_gpu : bool
        Whether the caller asked for GPU acceleration.

    Returns
    -------
    str
        ``"cuda"`` only when ``use_gpu`` is true and ``linumpy.gpu.GPU_AVAILABLE``
        is true, otherwise ``"cpu"``.
    """
    if not use_gpu:
        return "cpu"
    from linumpy.gpu import GPU_AVAILABLE

    return "cuda" if GPU_AVAILABLE else "cpu"


def find_z_overlap(
    fixed_vol: np.ndarray,
    moving_vol: np.ndarray,
    slicing_interval_mm: float,
    search_range_mm: float,
    resolution_um: float,
    use_gpu: bool = False,
) -> tuple[int, float]:
    """Find optimal Z-overlap between consecutive slices using cross-correlation.

    Searches around the expected overlap for the best normalized
    cross-correlation score, using the center XY region for speed.

    Parameters
    ----------
    fixed_vol : np.ndarray
        Bottom (fixed) slice volume (Z, Y, X).
    moving_vol : np.ndarray
        Top (moving) slice volume (Z, Y, X).
    slicing_interval_mm : float
        Expected physical slice thickness in mm.
    search_range_mm : float
        Search range around expected position in mm.
    resolution_um : float
        Z resolution in microns per voxel.
    use_gpu : bool
        If True, run the NCC sweep on the GPU when CuPy is available.

    Returns
    -------
    best_overlap : int
        Optimal overlap in Z voxels.
    best_corr : float
        Correlation score at optimal overlap.
    """
    interval_vox = int((slicing_interval_mm * 1000) / resolution_um)
    expected_overlap_vox = min(fixed_vol.shape[0], moving_vol.shape[0]) - interval_vox
    search_range_vox = int((search_range_mm * 1000) / resolution_um)

    min_overlap = max(1, expected_overlap_vox - search_range_vox)
    max_overlap = min(fixed_vol.shape[0], moving_vol.shape[0], expected_overlap_vox + search_range_vox)

    if min_overlap >= max_overlap:
        return expected_overlap_vox, 0.0

    h, w = fixed_vol.shape[1], fixed_vol.shape[2]
    margin = min(h, w) // 4
    y_slice = slice(margin, h - margin)
    x_slice = slice(margin, w - margin)

    xp: Any = np
    if resolve_z_match_backend(use_gpu) == "cuda":
        import cupy as cp

        xp = cp

    fixed_xp = xp.asarray(fixed_vol[-max_overlap:, y_slice, x_slice].astype(np.float32))
    moving_xp = xp.asarray(moving_vol[:max_overlap, y_slice, x_slice].astype(np.float32))

    best_overlap = expected_overlap_vox
    best_corr = -float("inf")

    for overlap in range(min_overlap, max_overlap + 1):
        fixed_region = fixed_xp[-overlap:]
        moving_region = moving_xp[:overlap]

        fixed_norm = (fixed_region - fixed_region.mean()) / (fixed_region.std() + 1e-8)
        moving_norm = (moving_region - moving_region.mean()) / (moving_region.std() + 1e-8)

        corr = float((fixed_norm * moving_norm).mean())
        if corr > best_corr:
            best_corr = corr
            best_overlap = overlap

    return best_overlap, best_corr


def _structure_image(plane: np.ndarray, threshold: float, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a plane with its smooth brightness trend divided out, and its tissue mask."""
    from scipy.ndimage import gaussian_filter

    mask = plane > threshold
    weight = gaussian_filter(mask.astype(np.float32), sigma)
    trend = gaussian_filter(np.where(mask, plane, 0.0).astype(np.float32), sigma) / np.maximum(weight, 1e-3)
    return np.where(mask, plane / np.maximum(trend, 1e-6) - 1.0, 0.0).astype(np.float32), mask


def _masked_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray, min_pixels: int) -> float:
    if int(mask.sum()) < min_pixels:
        return float("nan")
    x = a[mask] - a[mask].mean()
    y = b[mask] - b[mask].mean()
    return float((x * y).sum() / np.sqrt((x * x).sum() * (y * y).sum() + 1e-12))


def find_z_step_structure(
    fixed_vol: np.ndarray,
    moving_vol: np.ndarray,
    expected_step: int,
    search_vox: int = 10,
    moving_planes: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    tissue_threshold: float = 0.01,
    downsample: int = 2,
    min_score: float = 0.08,
    max_residual: float = 1.0,
    min_distinct: float = 0.03,
    min_planes: int = 5,
    min_pixels: int = 5000,
    workers: int | None = None,
) -> tuple[int | None, dict[str, Any]]:
    """Measure how many planes below the fixed slab's top the moving slab starts.

    The top of a slab is not always one slicing interval below the top of the
    slab above. The surface found by the interface crop moves by several
    planes from slab to slab, a cut can be thicker or thinner, tissue can
    stand above the block face, and a slab can be a wedge. Stacked at the
    interval, such a joint shows a break.

    Plain intensity cannot find the depth: the top planes of every slab look
    alike because they are the least attenuated, so intensity aligns top with
    top (Lefebvre et al. 2017, Neurophotonics 4, 041501, who correlate the
    image gradient for that reason). Here each plane is divided by its own
    smooth trend, which leaves the outlines of structures, and each of the
    shallow planes of the moving slab is matched against the fixed slab over
    ``expected_step +/- search_vox``.

    The best matches lie on a line: moving plane ``p`` matches fixed plane
    ``step + slope * p``. The slope is not always one, because the same tissue
    does not span the same number of planes near the top of a slab and deep in
    the slab above (0.8 in sub-19, 1.1 in sub-21). A step read at one moving
    plane therefore depends on that plane. The step returned is the line at
    plane 0, the cut face of the moving slab, which is where the stack hands
    over from one slab to the next.

    The two slabs are only roughly aligned in XY, so a shift between them is
    estimated first, by phase correlation of the structure of a few planes
    averaged. It is used only when the slabs match better with it than as
    they are placed, and is returned as ``shift``.

    The measurement is rejected, and the caller keeps the expected step, when
    too few planes can be placed, when the planes do not lie on one line
    (``max_residual``), when the match is weak (``min_score``) or no better
    than a few planes away (``min_distinct``), or when a best match sits at
    the edge of the search.

    Parameters
    ----------
    fixed_vol, moving_vol : np.ndarray
        Slabs ``(Z, Y, X)`` on the same XY grid, plane 0 at the cut face.
    expected_step : int
        Planes between the two cut faces according to the slicing interval.
    search_vox : int
        Half-width of the search around ``expected_step``.
    moving_planes : sequence of int
        Planes of the moving slab that are matched.
    tissue_threshold : float
        Intensity at or below this is background.
    downsample : int
        XY step the match runs on.
    min_score : float
        Lowest median correlation of the best matches that counts as a match.
    max_residual : float
        Largest median distance, in planes, of the best matches from the line.
    min_distinct : float
        How much better the best match has to be than the matches five planes
        or more away, as a median over the moving planes.
    min_planes : int
        Fewest moving planes that have to be placed.
    min_pixels : int
        Fewest common tissue pixels, at full resolution, for a correlation.
    workers : int or None
        Planes prepared at the same time. ``None`` takes the thread budget of
        the task, at most 8.

    Returns
    -------
    int or None
        The measured step, or ``None`` when it cannot be told.
    dict
        ``planes`` that were placed, with the ``steps`` (best fixed plane minus
        moving plane), ``scores`` and ``distinct`` of each; ``slope``,
        ``step`` (not rounded) and ``residual`` of the line; the XY ``shift``
        used; and ``reason`` when the measurement was rejected.
    """
    from scipy.ndimage import shift as nd_shift
    from skimage.registration import phase_cross_correlation

    d = max(1, int(downsample))
    fixed = np.asarray(fixed_vol[:, ::d, ::d], dtype=np.float32)
    moving = np.asarray(moving_vol[:, ::d, ::d], dtype=np.float32)
    sigma = 16.0 / d
    info: dict[str, Any] = {
        "planes": [],
        "steps": [],
        "scores": [],
        "distinct": [],
        "slope": float("nan"),
        "step": float("nan"),
        "residual": float("nan"),
        "shift": (0.0, 0.0),
        "reason": "",
    }
    lo = max(0, expected_step - 2)
    hi = min(fixed.shape[0], expected_step + 10)
    if hi - lo < 2 or moving.shape[0] < 3:
        info["reason"] = "too few planes"
        return None, info

    # The two slabs are only roughly aligned in XY at this point.
    fixed_aip = fixed[lo:hi].mean(axis=0)
    moving_aip = moving[: min(10, moving.shape[0])].mean(axis=0)
    fixed_s, fixed_mask = _structure_image(fixed_aip, tissue_threshold, sigma)
    moving_s, moving_mask = _structure_image(moving_aip, tissue_threshold, sigma)
    xy_shift, _, _ = phase_cross_correlation(fixed_s, moving_s, upsample_factor=4)
    # The phase correlation always returns a shift. On slabs with little
    # tissue it can be tens of pixels off, and the planes are then matched in
    # the wrong place. It is used only when the two slabs match better with
    # it than as they are placed.
    as_placed = _masked_ncc(fixed_s, moving_s, fixed_mask & moving_mask, min_pixels // (d * d))
    shifted_s, shifted_mask = _structure_image(nd_shift(moving_aip, xy_shift, order=1), tissue_threshold, sigma)
    with_shift = _masked_ncc(fixed_s, shifted_s, fixed_mask & shifted_mask, min_pixels // (d * d))
    if not with_shift >= (as_placed if np.isfinite(as_placed) else -1.0):
        xy_shift = np.zeros(2)
    info["shift"] = (float(xy_shift[0]) * d, float(xy_shift[1]) * d)

    # The search ranges of the moving planes overlap, so every fixed plane is
    # prepared once, and the planes are prepared on separate threads.
    planes = [plane for plane in moving_planes if plane < moving.shape[0]]
    ranges = {
        plane: [
            f
            for f in range(plane + expected_step - search_vox, plane + expected_step + search_vox + 1)
            if 0 <= f < fixed.shape[0]
        ]
        for plane in planes
    }
    wanted = sorted({f for candidates in ranges.values() for f in candidates})

    def _fixed_structure(f: int) -> tuple[np.ndarray, np.ndarray]:
        return _structure_image(fixed[f], tissue_threshold, sigma)

    def _moved_structure(plane: int) -> tuple[np.ndarray, np.ndarray]:
        return _structure_image(nd_shift(moving[plane], xy_shift, order=1), tissue_threshold, sigma)

    workers = thread_budget(8) if workers is None else max(1, int(workers))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            fixed_structures = dict(zip(wanted, executor.map(_fixed_structure, wanted), strict=True))
            moved_structures = dict(zip(planes, executor.map(_moved_structure, planes), strict=True))
    else:
        fixed_structures = {f: _fixed_structure(f) for f in wanted}
        moved_structures = {plane: _moved_structure(plane) for plane in planes}

    at_edge = False
    matched: list[float] = []
    for plane in planes:
        moved_s, moved_mask = moved_structures[plane]
        first = plane + expected_step - search_vox
        last = plane + expected_step + search_vox
        candidates = ranges[plane]
        scores = np.array(
            [
                _masked_ncc(fixed_structures[f][0], moved_s, fixed_structures[f][1] & moved_mask, min_pixels // (d * d))
                for f in candidates
            ],
            dtype=np.float64,
        )
        if not candidates or not np.isfinite(scores).any():
            continue
        best = int(np.nanargmax(scores))
        # A best match at the end of a range that the slab's depth cut short says
        # nothing: the true depth may lie beyond the data.
        if (best == len(candidates) - 1 and candidates[-1] < last) or (best == 0 and candidates[0] > first):
            continue
        at_edge = at_edge or candidates[best] in (first, last)
        # Position of the peak between planes, from the parabola through its neighbours.
        offset = 0.0
        if 0 < best < len(candidates) - 1 and np.isfinite(scores[best - 1]) and np.isfinite(scores[best + 1]):
            curvature = scores[best - 1] - 2.0 * scores[best] + scores[best + 1]
            if curvature < 0:
                offset = float(np.clip(0.5 * (scores[best - 1] - scores[best + 1]) / curvature, -0.5, 0.5))
        away = scores[np.abs(np.arange(len(candidates)) - best) >= 5]
        info["planes"].append(plane)
        info["steps"].append(round(candidates[best] + offset - plane, 2))
        info["scores"].append(float(scores[best]))
        info["distinct"].append(float(scores[best] - np.nanmax(away)) if np.isfinite(away).any() else float("nan"))
        matched.append(candidates[best] + offset)

    if len(matched) < max(2, min_planes):
        info["reason"] = "too few planes"
        return None, info

    # Median of the pairwise slopes, then the median intercept: one or two
    # planes that matched elsewhere do not move the line.
    at = np.asarray(info["planes"], dtype=np.float64)
    found = np.asarray(matched, dtype=np.float64)
    upper = np.triu_indices(len(at), k=1)
    slopes = (found[upper[1]] - found[upper[0]]) / (at[upper[1]] - at[upper[0]])
    slope = float(np.clip(np.median(slopes), 0.7, 1.3))
    intercept = float(np.median(found - slope * at))
    residual = float(np.median(np.abs(found - (intercept + slope * at))))
    info["slope"], info["step"], info["residual"] = slope, intercept, residual

    if residual > max_residual:
        info["reason"] = "planes disagree"
        return None, info
    distinct = np.asarray(info["distinct"], dtype=np.float64)
    if float(np.median(info["scores"])) < min_score or not (
        np.isfinite(distinct).any() and float(np.nanmedian(distinct)) >= min_distinct
    ):
        info["reason"] = "weak match"
        return None, info
    if at_edge:
        info["reason"] = "edge of search"
        return None, info
    step = round(intercept)
    if step < 1:
        info["reason"] = "edge of search"
        return None, info
    return step, info


def zero_above_surface(
    vol: np.ndarray,
    max_planes: int = 12,
    relative_level: float = 0.4,
    tissue_threshold: float = 0.01,
    sigma_xy: float = 3.0,
) -> tuple[np.ndarray, int]:
    """Zero the voxels of a slab that lie above its tissue surface.

    A slab is cropped at one depth, the average surface. The cut does not
    leave a flat surface, so on part of the slab the first planes are water
    above the tissue. They are dim but not zero, and pasted into the stack
    they replace the tissue of the slab above with a dark line. Lefebvre et
    al. 2017 handle this with a water/tissue mask per A-line; this is that
    mask, applied to the top ``max_planes``.

    A voxel is above the surface when it, and every voxel above it on the
    same A-line, is below ``relative_level`` of the slab's median tissue
    level, and the A-line reaches that level further down. The level is one
    number for the slab on purpose: measured against the A-line's own
    brightest value, grey matter lying over white matter would count as
    water.

    Returns the slab with those voxels zeroed, and their count.
    """
    n = min(int(max_planes), vol.shape[0])
    if n < 1:
        return vol, 0
    top = np.asarray(vol[: min(vol.shape[0], n + 8)], dtype=np.float32)
    tissue = top[top > tissue_threshold]
    if tissue.size < 1000:
        return vol, 0
    level = relative_level * float(np.median(tissue))
    smooth = gaussian_filter_planes(top, sigma_xy)
    dim = smooth[:n] < level
    water = np.logical_and.accumulate(dim, axis=0)
    water &= (smooth.max(axis=0) >= level)[None]
    water &= top[:n] > 0
    count = int(water.sum())
    if count == 0:
        return vol, 0
    out = np.array(vol, dtype=np.float32, copy=True)
    out[:n][water] = 0.0
    return out, count


def apply_2d_transform(
    image_2d: np.ndarray,
    transform: Any,
    rotation_only: bool = False,
    max_rotation_deg: float = 1.0,
    override_rotation: Any = None,
) -> np.ndarray:
    """Apply a SimpleITK 2D/3D transform to a single 2D image (Z-slice).

    Parameters
    ----------
    image_2d : np.ndarray
        2D image to transform.
    transform : sitk.Transform
        SimpleITK transform (extracts 2D rotation/translation from 3D Euler).
    rotation_only : bool
        If True, apply only rotation, ignore translation.
    max_rotation_deg : float
        Maximum rotation in degrees; larger values are clamped. 0 = no clamping.
    override_rotation : float or None
        Use this rotation angle (radians) instead of extracting from transform.

    Returns
    -------
    np.ndarray
        Transformed 2D image.
    """
    import SimpleITK as sitk

    sitk_img = sitk.GetImageFromArray(image_2d.astype(np.float32))

    if transform.GetDimension() == 3:
        if isinstance(transform, sitk.Euler3DTransform) or transform.GetName() == "Euler3DTransform":
            params = transform.GetParameters()
            angle = params[2] if len(params) > 2 else 0
            tx = params[3] if len(params) > 3 else 0
            ty = params[4] if len(params) > 4 else 0

            if override_rotation is not None:
                angle = override_rotation
            elif max_rotation_deg > 0:
                max_angle_rad = np.radians(max_rotation_deg)
                if abs(angle) > max_angle_rad:
                    angle = np.clip(angle, -max_angle_rad, max_angle_rad)

            center = transform.GetCenter()
            center_2d = [center[0], center[1]]
            tfm_2d = sitk.Euler2DTransform()
            tfm_2d.SetCenter(center_2d)
            tfm_2d.SetAngle(angle)
            if rotation_only:
                tfm_2d.SetTranslation([0, 0])
            else:
                tfm_2d.SetTranslation([tx, ty])
        else:
            tfm_2d = sitk.Euler2DTransform()
            angle = 0
            tx, ty = 0, 0
    else:
        tfm_2d = transform
        if rotation_only and hasattr(tfm_2d, "SetTranslation"):
            tfm_2d.SetTranslation([0, 0])
        angle = 0
        tx, ty = 0, 0

    tx_final = 0 if rotation_only else tx
    ty_final = 0 if rotation_only else ty
    if abs(angle) < 0.00175 and abs(tx_final) < 1.0 and abs(ty_final) < 1.0:
        return image_2d.copy()

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_img)
    resampler.SetTransform(tfm_2d)
    resampler.SetInterpolator(sitk.sitkLinear)
    # Empty/rotated-in pixels must be 0 so blend_overlap_z treats them as
    # background instead of Hann-mixing tissue with a dim agarose fill.
    resampler.SetDefaultPixelValue(0.0)

    result = resampler.Execute(sitk_img)
    return sitk.GetArrayFromImage(result)


def apply_transform_to_volume(
    vol: np.ndarray,
    transform: Any,
    rotation_only: bool = False,
    max_rotation_deg: float = 1.0,
    override_rotation: Any = None,
) -> np.ndarray:
    """Apply a 2D transform to each Z-slice of a volume.

    Parameters
    ----------
    vol : np.ndarray
        3D volume (Z, Y, X).
    transform : sitk.Transform
        Transform to apply to each slice.
    rotation_only : bool
        If True, apply only rotation.
    max_rotation_deg : float
        Maximum rotation in degrees.
    override_rotation : float or None
        If provided, use this rotation for all slices.

    Returns
    -------
    np.ndarray
        Transformed volume.
    """
    result = np.zeros_like(vol)
    for z in range(vol.shape[0]):
        result[z] = apply_2d_transform(vol[z], transform, rotation_only, max_rotation_deg, override_rotation)
    return result


def inplane_output_point(
    x: float,
    y: float,
    transform: Any,
    *,
    apply_translation: bool,
) -> tuple[float, float]:
    """Where a full-resolution point lands after the Euler used at stack time.

    Matches ``Resample``: positive SimpleITK ``tx`` moves the point to a smaller x.
    """
    params = [float(v) for v in transform.GetParameters()]
    if transform.GetDimension() == 3 and len(params) >= 5:
        angle = float(params[2])
        tx = float(params[3])
        ty = float(params[4])
        center = transform.GetCenter()
        center_x, center_y = float(center[0]), float(center[1])
    else:
        angle = float(params[0]) if params else 0.0
        tx = float(params[1]) if len(params) > 1 else 0.0
        ty = float(params[2]) if len(params) > 2 else 0.0
        center = transform.GetCenter() if hasattr(transform, "GetCenter") else (x, y)
        center_x = float(center[0]) if len(center) > 0 else x
        center_y = float(center[1]) if len(center) > 1 else y
    if not apply_translation:
        tx, ty = 0.0, 0.0
    cosine, sine = float(np.cos(angle)), float(np.sin(angle))
    in_x = x - center_x - tx
    in_y = y - center_y - ty
    # Inverse of the ITK Euler2D map (output → input).
    out_x = center_x + cosine * in_x + sine * in_y
    out_y = center_y - sine * in_x + cosine * in_y
    return out_x, out_y


def rigid_euler_pad(transform: Any, ny: int, nx: int) -> tuple[int, int, float, float, float]:
    """Return ``(pad_x, pad_y, angle_rad, tx, ty)`` for a saved Euler."""
    params = list(transform.GetParameters())
    if transform.GetDimension() == 3 and len(params) >= 5:
        angle = float(params[2])
        tx = float(params[3])
        ty = float(params[4])
    else:
        angle = float(params[0]) if params else 0.0
        tx = float(params[1]) if len(params) > 1 else 0.0
        ty = float(params[2]) if len(params) > 2 else 0.0
    diagonal = float(np.hypot(ny, nx))
    rot_pad = int(np.ceil(diagonal * abs(np.sin(angle)))) + 2
    pad_x = int(np.ceil(abs(tx))) + rot_pad
    pad_y = int(np.ceil(abs(ty))) + rot_pad
    return pad_x, pad_y, angle, tx, ty


def apply_rigid_euler_padded(
    vol: np.ndarray,
    transform: Any,
) -> tuple[np.ndarray, int, int]:
    """Resample a volume with the saved Euler, padding so the translation is not clipped.

    Returns ``(padded_volume, pad_x, pad_y)``. The original voxel ``(0, 0)``
    lands at ``(pad_y, pad_x)`` in the padded array when the Euler is identity,
    so the caller subtracts ``(pad_x, pad_y)`` from the canvas origin.
    """
    import SimpleITK as sitk

    pad_x, pad_y, angle, tx, ty = rigid_euler_pad(transform, int(vol.shape[1]), int(vol.shape[2]))
    if transform.GetDimension() == 3:
        center = transform.GetCenter()
        center_2d = [float(center[0]), float(center[1])]
    else:
        center_2d = [vol.shape[2] / 2.0, vol.shape[1] / 2.0]

    tfm_2d = sitk.Euler2DTransform()
    tfm_2d.SetCenter(center_2d)
    tfm_2d.SetAngle(angle)
    tfm_2d.SetTranslation([tx, ty])

    ny, nx = int(vol.shape[1]), int(vol.shape[2])
    ref = sitk.Image([nx + 2 * pad_x, ny + 2 * pad_y], sitk.sitkFloat32)
    ref.SetOrigin((-float(pad_x), -float(pad_y)))
    ref.SetSpacing((1.0, 1.0))

    padded = np.zeros((vol.shape[0], ny + 2 * pad_y, nx + 2 * pad_x), dtype=np.float32)
    for z in range(vol.shape[0]):
        sitk_img = sitk.GetImageFromArray(vol[z].astype(np.float32))
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(ref)
        resampler.SetTransform(tfm_2d)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0.0)
        padded[z] = sitk.GetArrayFromImage(resampler.Execute(sitk_img))
    return padded, pad_x, pad_y


def apply_xy_shift(vol: np.ndarray, dx_px: float, dy_px: float, output_shape: tuple[int, int]) -> tuple:
    """Compute destination region for placing a shifted volume.

    Returns the (possibly cropped) volume data and destination coordinates
    without allocating a full-size output array.

    Parameters
    ----------
    vol : np.ndarray
        3D volume (Z, Y, X).
    dx_px, dy_px : float
        Shift in pixels (X and Y directions).
    output_shape : tuple
        (out_ny, out_nx) output canvas size.

    Returns
    -------
    cropped_vol : np.ndarray or None
        Cropped volume data to write.
    dst_coords : tuple or None
        (y_start, y_end, x_start, x_end) in output coordinates.
    """
    out_ny, out_nx = output_shape
    dx_int, dy_int = round(dx_px), round(dy_px)

    dst_y_start = dy_int
    dst_x_start = dx_int
    dst_y_end = dst_y_start + vol.shape[1]
    dst_x_end = dst_x_start + vol.shape[2]

    src_y_start = max(0, -dst_y_start)
    src_y_end = vol.shape[1] - max(0, dst_y_end - out_ny)
    src_x_start = max(0, -dst_x_start)
    src_x_end = vol.shape[2] - max(0, dst_x_end - out_nx)

    dst_y_start = max(0, dst_y_start)
    dst_y_end = min(out_ny, dst_y_end)
    dst_x_start = max(0, dst_x_start)
    dst_x_end = min(out_nx, dst_x_end)

    if src_y_end > src_y_start and src_x_end > src_x_start:
        cropped = vol[:, src_y_start:src_y_end, src_x_start:src_x_end]
        return cropped, (dst_y_start, dst_y_end, dst_x_start, dst_x_end)
    return None, None


def blend_overlap_z(fixed_region: np.ndarray, moving_region: np.ndarray, tissue_threshold: float = 0.01) -> np.ndarray:
    """Blend overlapping Z-region using a cosine (Hann) ramp along Z-axis.

    The weight ramp has zero slope at both endpoints, so there is no abrupt
    intensity change at either boundary of the overlap zone.  At tissue
    boundaries where only one slice has data the full intensity of that slice
    is used unchanged.  Voxels at or below ``tissue_threshold`` are treated as
    agarose/background and are not Hann-averaged with tissue.

    Parameters
    ----------
    fixed_region : np.ndarray
        3D array (Z, Y, X) from the existing stack (bottom portion).
    moving_region : np.ndarray
        3D array (Z, Y, X) from the new slice (top portion).
    tissue_threshold : float
        Intensity at or below this is background. Default 0.01.

    Returns
    -------
    np.ndarray
        Blended region with smooth Z-transition.
    """
    nz = fixed_region.shape[0]
    thr = float(tissue_threshold)

    if nz <= 1:
        return moving_region if np.sum(moving_region > thr) >= np.sum(fixed_region > thr) else fixed_region

    # Cosine (Hann) ramp: 0 → 1 with zero slope at both ends
    t = np.linspace(0, np.pi, nz)
    z_weights = 0.5 * (1 - np.cos(t))
    alphas = np.broadcast_to(z_weights[:, np.newaxis, np.newaxis], fixed_region.shape).copy()

    fixed_valid = fixed_region > thr
    moving_valid = moving_region > thr
    both_valid = fixed_valid & moving_valid
    fixed_only = fixed_valid & ~moving_valid
    moving_only = moving_valid & ~fixed_valid

    blended = np.zeros_like(moving_region, dtype=np.float32)
    if np.any(both_valid):
        blended[both_valid] = ((1 - alphas) * fixed_region + alphas * moving_region)[both_valid]
    if np.any(fixed_only):
        blended[fixed_only] = fixed_region[fixed_only]
    if np.any(moving_only):
        blended[moving_only] = moving_region[moving_only]

    # A one-sided rim is a hard line: the other slice's tissue is off to
    # the side, so the Z Hann never sees it. Pull that rim toward the level
    # of the other slice's nearby tissue (~8 px). Each voxel keeps its own
    # value and only its gain changes: writing the local means instead blurred
    # dim tissue and painted a tissue-bright halo where neither slice has any.
    # A 160 px translation gap is not closed here.
    if np.any(fixed_only) or np.any(moving_only):
        sigma = 8.0
        fixed_weight = gaussian_filter_planes(fixed_valid.astype(np.float32), sigma)
        moving_weight = gaussian_filter_planes(moving_valid.astype(np.float32), sigma)
        fixed_mean = gaussian_filter_planes(np.where(fixed_valid, fixed_region, 0.0).astype(np.float32), sigma)
        moving_mean = gaussian_filter_planes(np.where(moving_valid, moving_region, 0.0).astype(np.float32), sigma)
        fixed_mean = fixed_mean / np.maximum(fixed_weight, 1e-6)
        moving_mean = moving_mean / np.maximum(moving_weight, 1e-6)
        near_both = (fixed_weight > 0.02) & (moving_weight > 0.02)
        rim = near_both & fixed_only
        if np.any(rim):
            gain = np.clip(moving_mean / np.maximum(fixed_mean, 1e-6), 0.5, 2.0)
            blended[rim] = (fixed_region * ((1.0 - alphas) + alphas * gain))[rim]
        rim = near_both & moving_only
        if np.any(rim):
            gain = np.clip(fixed_mean / np.maximum(moving_mean, 1e-6), 0.5, 2.0)
            blended[rim] = (moving_region * (alphas + (1.0 - alphas) * gain))[rim]

    return blended


def surface_confidence(
    existing: np.ndarray,
    incoming: np.ndarray,
    planes: int,
    tissue_threshold: float = 0.01,
    sigma_xy: float = 2.0,
    low: float = 0.5,
    high: float = 0.8,
) -> np.ndarray:
    """How far the incoming slab has come out of its own surface, per voxel of the overlap.

    A slab is cropped at one depth, but its cut face is not flat: near free
    edges and over soft tissue the surface lies a few planes lower. There the
    first planes of the slab are empty and the next ones ramp up from a few
    percent of the tissue level. The ramp is above the background threshold,
    so it counts as tissue, and blended into the stack it draws a dark line
    over the slab above, which shows the same place at full brightness.

    The slab above is the reference: on an A-line the incoming slab is
    trusted from the first plane where it reaches the brightness of the slab
    above, and on every plane below it. Before that the weight is zero below
    ``low`` of that brightness and rises to one at ``high``. The first two
    planes of a slab are a quarter to a third dimmer than the slab above
    everywhere; the limits leave most of that to the cross-fade and hold back
    the A-lines that are well below it. Where the slab above has no tissue
    the incoming slab is all there is and its weight is one.

    Parameters
    ----------
    existing, incoming : np.ndarray
        The overlap ``(Z, Y, X)`` from the stack and from the incoming slab,
        plane 0 at the incoming cut face, at a common brightness (see
        :func:`level_joint`).
    planes : int
        Planes from the top of the overlap that are looked at.
    tissue_threshold : float
        Intensity at or below this is background.
    sigma_xy : float
        XY smoothing in pixels before the two slabs are compared.
    low, high : float
        Brightness relative to the slab above at which the weight is 0 and 1.

    Returns
    -------
    np.ndarray
        Weight of the incoming slab, 0 to 1, for the first ``planes`` planes.
    """
    n = min(max(1, int(planes)), existing.shape[0], incoming.shape[0])
    above = gaussian_filter_planes(np.asarray(existing[:n], dtype=np.float32), sigma_xy)
    below = gaussian_filter_planes(np.asarray(incoming[:n], dtype=np.float32), sigma_xy)
    known = above > tissue_threshold
    # A plane on which the slab above is empty says nothing about the planes below it.
    ratio = np.where(known, below / np.maximum(above, 1e-6), -np.inf)
    reached = np.maximum.accumulate(ratio, axis=0)
    weight = np.clip((reached - low) / (high - low), 0.0, 1.0)
    return np.where(known, weight, 1.0).astype(np.float32)


def blend_cut_face(
    existing: np.ndarray,
    incoming: np.ndarray,
    planes: int,
    tissue_threshold: float = 0.01,
    confidence: np.ndarray | None = None,
) -> np.ndarray:
    """Cross-fade the first ``planes`` of an overlap, then keep the incoming slab.

    ``blend_overlap_z`` fades over the whole overlap, so the previous slab's
    deepest planes carry weight down to its last one. Deep OCT planes are dim
    and can invert contrast against the next slab's surface. Here the previous
    slab hands over within ``planes`` of the incoming cut face and the incoming
    slab's shallow planes fill the rest of the overlap. Incoming background
    never erases existing tissue.

    Parameters
    ----------
    existing : np.ndarray
        3D array (Z, Y, X) already in the stack over the overlap.
    incoming : np.ndarray
        3D array (Z, Y, X) from the new slab, plane 0 at its cut face.
    planes : int
        Depth of the cross-fade, clamped to the overlap.
    tissue_threshold : float
        Intensity at or below this is background. Default 0.01.
    confidence : np.ndarray or None
        Weight of the incoming slab for the first planes of the overlap, from
        :func:`surface_confidence`. Where both slabs have tissue, the incoming
        slab enters with this weight on top of the cross-fade, so an A-line is
        handed over where the incoming slab comes out of its surface.

    Returns
    -------
    np.ndarray
        The overlap region to write back.
    """
    n = min(max(1, int(planes)), existing.shape[0])
    out = paste_tissue(existing, incoming, threshold=tissue_threshold)
    out[:n] = blend_overlap_z(existing[:n], incoming[:n], tissue_threshold=tissue_threshold)
    if confidence is not None:
        m = min(confidence.shape[0], existing.shape[0])
        above = np.asarray(existing[:m], dtype=np.float32)
        has_above = above > tissue_threshold
        both = has_above & (np.asarray(incoming[:m]) > tissue_threshold)
        weight = confidence[:m]
        mixed = np.where(both, weight * out[:m] + (1.0 - weight) * above, out[:m])
        # Where only the slab above has tissue it is kept as it is. The
        # cross-fade pulls such voxels towards the incoming slab's tissue next
        # to them, which here is the dim rim of its surface.
        out[:m] = np.where(has_above & ~both, above, mixed)
    return out


def below_own_surface(
    slab_top: np.ndarray,
    planes: int,
    reference: tuple[int, int] = (6, 14),
    level: float = 0.7,
    sigma_xy: float = 2.0,
) -> np.ndarray:
    """Voxels of the first planes of a slab that lie below its own tissue surface.

    The cut face is not flat, and under it a slab comes up to its tissue level
    over a few planes. A voxel counts from the first plane on which its
    A-line reaches ``level`` of the A-line's own brightness further down (the
    mean of the ``reference`` planes), and on every plane below. Nothing
    outside the slab is used, so the mask can be made before the slab is
    compared with the one above.

    Parameters
    ----------
    slab_top : np.ndarray
        The first planes ``(Z, Y, X)`` of the slab, plane 0 at its cut face,
        at least as deep as ``reference``.
    planes : int
        Planes from the top for which the mask is returned.
    reference : tuple of int
        Plane range whose mean is the brightness an A-line has to reach.
    level : float
        Fraction of that brightness.
    sigma_xy : float
        XY smoothing in pixels before the comparison.

    Returns
    -------
    np.ndarray
        Boolean mask of the first ``planes`` planes.
    """
    n = min(max(1, int(planes)), slab_top.shape[0])
    lo = min(reference[0], max(0, slab_top.shape[0] - 1))
    hi = max(lo + 1, min(reference[1], slab_top.shape[0]))
    smooth = gaussian_filter_planes(np.asarray(slab_top[: max(n, hi)], dtype=np.float32), sigma_xy)
    target = level * smooth[lo:hi].mean(axis=0)
    return np.logical_or.accumulate(smooth[:n] >= target[None], axis=0)


def joint_level_field(
    existing: np.ndarray,
    incoming: np.ndarray,
    cell: int = 4,
    tissue_threshold: float = 0.01,
    sigma_cells: float = 1.0,
    min_fraction: float = 0.5,
    clamp: float = 1.0,
    valid: np.ndarray | None = None,
) -> np.ndarray | None:
    """Log brightness of the incoming slab relative to the slab above, where they overlap.

    In the overlap the two slabs show the same tissue at the same depth: the
    slab above deep below its cut face, the incoming slab just under its own.
    What differs between them there is not anatomy. It is what the depth
    corrections left in each slab: the falloff with depth, and the shadows
    under bright structures, which are a few pixels wide. One gain per slab
    does not remove it and a band is left at the joint.

    The difference is measured in XY cells of ``cell`` pixels as the mean log
    ratio over the voxels that are tissue in both slabs, and smoothed over
    ``sigma_cells`` cells, which also carries it into cells that have too few
    such voxels. The defaults follow the shadows (40 um cells at 10 um
    pixels) and need the two slabs to be aligned to a few pixels. Where that
    is not known, measure on a coarser scale (``cell=8, sigma_cells=4``): a
    fine field on misaligned slabs would copy the structure of one slab into
    the other.

    Parameters
    ----------
    existing, incoming : np.ndarray
        The same planes ``(Z, Y, X)`` of the overlap from the stack and from
        the incoming slab.
    cell : int
        Edge of the XY cells in pixels.
    tissue_threshold : float
        Intensity at or below this is background.
    sigma_cells : float
        XY smoothing in cells.
    min_fraction : float
        A cell is measured when at least this fraction of its voxels is tissue
        in both slabs.
    clamp : float
        Largest difference returned, in log units.
    valid : np.ndarray or None
        Voxels that may be used, of the shape of the two inputs. With
        :func:`below_own_surface` the difference can be measured from the top
        of the incoming slab without its surface ramp entering it.

    Returns
    -------
    np.ndarray or None
        ``(Y // cell, X // cell)`` field: log of incoming over existing. None
        when the overlap has no cell to measure.
    """
    from scipy.ndimage import gaussian_filter

    nz, ny, nx = existing.shape
    cy, cx = ny // cell, nx // cell
    if nz < 1 or cy < 1 or cx < 1:
        return None
    above = np.asarray(existing[:, : cy * cell, : cx * cell], dtype=np.float32)
    below = np.asarray(incoming[:, : cy * cell, : cx * cell], dtype=np.float32)
    both = (above > tissue_threshold) & (below > tissue_threshold)
    if valid is not None:
        both &= np.asarray(valid[:, : cy * cell, : cx * cell], dtype=bool)
    ratio = np.zeros(above.shape, dtype=np.float32)
    np.log(below, out=ratio, where=both)
    ratio -= np.log(above, out=np.zeros_like(ratio), where=both)
    total = ratio.reshape(nz, cy, cell, cx, cell).sum(axis=(0, 2, 4))
    count = both.reshape(nz, cy, cell, cx, cell).sum(axis=(0, 2, 4))
    known = count >= min_fraction * nz * cell * cell
    if not known.any():
        return None
    mean = np.where(known, total / np.maximum(count, 1), 0.0).astype(np.float32)
    cover = gaussian_filter(known.astype(np.float32), sigma_cells)
    smooth = gaussian_filter(mean, sigma_cells) / np.maximum(cover, 1e-3)
    return np.clip(np.where(cover > 0.3, smooth, 0.0), -clamp, clamp).astype(np.float32)


def level_joint(
    above: np.ndarray,
    overlap: np.ndarray,
    incoming: np.ndarray,
    field: np.ndarray,
    planes: int,
    cell: int = 4,
) -> None:
    """Bring the two slabs at a joint to a common brightness, in place.

    Each slab takes half of the difference ``field`` measured in the overlap
    (:func:`joint_level_field`). The gain is full at the joint and falls to
    one over ``planes`` planes into each slab, so the middle of every slab
    keeps its level and no gain is carried from one joint to the next.

    Parameters
    ----------
    above : np.ndarray
        The planes of the stack just above the overlap, last plane nearest to
        it. At most ``planes`` of them are used.
    overlap : np.ndarray
        The planes of the stack in the overlap (the slab above, deep).
    incoming : np.ndarray
        The incoming slab, plane 0 at the top of the overlap.
    field : np.ndarray
        Coarse log difference, incoming over existing.
    planes : int
        Planes over which the gain falls to one on each side.
    cell : int
        Edge of the XY cells of ``field`` in pixels.
    """
    planes = max(1, int(planes))
    half = 0.5 * upsample_plane(field, cell, (incoming.shape[1], incoming.shape[2]))
    # The slab above is raised by half the difference, in full where it overlaps.
    overlap *= np.exp(half)[None]
    n = min(planes, above.shape[0])
    for i in range(n):
        above[above.shape[0] - 1 - i] *= np.exp(half * (1.0 - (i + 1) / (planes + 1)))
    # The incoming slab is lowered by the other half, in full at its top.
    for k in range(min(planes, incoming.shape[0])):
        incoming[k] *= np.exp(-half * (1.0 - k / planes))


def paste_tissue(
    existing: np.ndarray,
    incoming: np.ndarray,
    threshold: float = 0.01,
) -> np.ndarray:
    """Paste incoming tissue over ``existing`` without punching holes.

    Voxels where ``incoming`` is above ``threshold`` replace ``existing``.
    Background incoming voxels leave whatever was already in the stack, so a
    no-blend overlap write cannot zero out the previous slice's tissue.
    """
    if existing.shape != incoming.shape:
        msg = f"paste_tissue shape mismatch {existing.shape} vs {incoming.shape}"
        raise ValueError(msg)
    out = np.array(existing, dtype=np.float32, copy=True)
    valid = incoming > float(threshold)
    out[valid] = incoming[valid]
    return out


def fill_joint_gaps(
    block: np.ndarray,
    joint: int,
    max_gap: int,
    tissue_threshold: float = 0.01,
) -> tuple[np.ndarray, int]:
    """Close the short empty runs that a slab joint leaves along Z.

    Near the outline of the tissue the cut face of the incoming slab lies a
    few planes lower than where the slab is cropped, and the deep end of the
    slab above is too dim to have survived as tissue. At the joint neither
    slab has a value there, and the stack shows a notch in the outline or a
    dark dot inside the tissue.

    Along each A-line, a run of background that spans the joint, is at most
    *max_gap* planes long and has tissue directly above and directly below
    is filled by linear interpolation between those two tissue voxels. A run
    that is longer, or open on one side, is a real gap and is left alone.

    Parameters
    ----------
    block : np.ndarray
        3D array (Z, Y, X) of the stack around the joint.
    joint : int
        Index in *block* of the first plane of the incoming slab.
    max_gap : int
        Longest run, in planes, that is filled.
    tissue_threshold : float
        Intensity at or below this is background.

    Returns
    -------
    np.ndarray
        The block with the runs filled.
    int
        Number of voxels filled.
    """
    out = np.array(block, dtype=np.float32, copy=True)
    nz = out.shape[0]
    if max_gap <= 0 or nz < 3:
        return out, 0
    valid = out > float(tissue_threshold)
    index = np.arange(nz, dtype=np.int32)[:, None, None]
    above = np.maximum.accumulate(np.where(valid, index, -1), axis=0)
    below = np.minimum.accumulate(np.where(valid, index, nz)[::-1], axis=0)[::-1]
    gap = ~valid & (above >= 0) & (below < nz) & (below - above - 1 <= max_gap)
    # Only the runs that contain the last plane before the joint or the first after it.
    gap &= (above < joint) & (below > joint - 1)
    n_filled = int(gap.sum())
    if n_filled == 0:
        return out, 0
    top = np.take_along_axis(out, np.clip(above, 0, nz - 1), axis=0)
    bottom = np.take_along_axis(out, np.clip(below, 0, nz - 1), axis=0)
    weight = (index - above) / np.maximum(below - above, 1)
    out[gap] = ((1.0 - weight) * top + weight * bottom)[gap]
    return out, n_filled


def blend_overlap_xy(existing: np.ndarray, new_data: np.ndarray, method: str = "none") -> np.ndarray:
    """Blend overlapping XY regions for motor-only stacking.

    Parameters
    ----------
    existing : np.ndarray
        Existing data in the output region.
    new_data : np.ndarray
        Incoming data to blend.
    method : str
        'none' (overwrite), 'average', 'max', or 'feather'.

    Returns
    -------
    np.ndarray
        Blended result.
    """
    if method == "none":
        mask = new_data != 0
        existing[mask] = new_data[mask]
        return existing
    elif method == "average":
        both_valid = (existing != 0) & (new_data != 0)
        only_new = (existing == 0) & (new_data != 0)
        existing[both_valid] = (existing[both_valid] + new_data[both_valid]) / 2
        existing[only_new] = new_data[only_new]
        return existing
    elif method == "max":
        return np.maximum(existing, new_data)
    elif method == "feather":
        return blend_overlap_xy(existing, new_data, "average")
    return existing


def _cut_face_aips(
    existing: np.ndarray,
    moving_overlap: np.ndarray,
    tissue_threshold: float,
    interface_planes: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """AIPs of the contacting faces, not the full overlap mean.

    Pairwise Euler is estimated at ``moving_z_index`` (plane 4), so averaging
    the whole overlap AIP scores that interior match. The staircase lives at
    the cut: last planes of the previous slab vs first planes of the incoming.
    Falls back to the full overlap mean when the interface has too little tissue.
    """
    n = min(max(1, int(interface_planes)), existing.shape[0], moving_overlap.shape[0])
    fixed_2d = np.mean(existing[-n:], axis=0).astype(np.float32)
    moving_2d = np.mean(moving_overlap[:n], axis=0).astype(np.float32)
    valid = (fixed_2d > tissue_threshold) & (moving_2d > tissue_threshold)
    if int(np.sum(valid)) < 1000:
        fixed_2d = np.mean(existing, axis=0).astype(np.float32)
        moving_2d = np.mean(moving_overlap, axis=0).astype(np.float32)
        valid = (fixed_2d > tissue_threshold) & (moving_2d > tissue_threshold)
    return fixed_2d, moving_2d, valid


def _phase_offset(fixed_2d: np.ndarray, moving_2d: np.ndarray, max_refinement_px: float) -> tuple[float, float] | None:
    """Phase-correlation (dy, dx) clamped to ``max_refinement_px``."""
    from skimage.registration import phase_cross_correlation

    shift, _, _ = phase_cross_correlation(fixed_2d, moving_2d, upsample_factor=10)
    dy, dx = float(shift[0]), float(shift[1])
    mag = float(np.hypot(dy, dx))
    if mag < 0.1:
        return None
    if max_refinement_px > 0 and mag > max_refinement_px:
        scale = max_refinement_px / mag
        dy, dx = dy * scale, dx * scale
    return dy, dx


def estimate_z_blend_xy_shift(
    existing: np.ndarray,
    moving_overlap: np.ndarray,
    max_refinement_px: float,
    ncc_min_improve: float = 1e-4,
    tissue_threshold: float = 0.01,
    interface_planes: int = 4,
    ncc_min_absolute: float = 0.2,
) -> tuple[float, float, float]:
    """Keep-if-better XY residual for the Z-blend cut face.

    Estimates a translation-only shift from the contacting-face AIPs.
    Returns ``(dy, dx, magnitude)``. Translation is clamped to
    ``max_refinement_px``. Identity when tissue NCC does not rise.
    """
    from linumpy.registration.refinement import register_refinement, tissue_ncc

    fixed_2d, moving_2d, valid = _cut_face_aips(existing, moving_overlap, tissue_threshold, interface_planes=interface_planes)
    if int(np.sum(valid)) < 1000:
        return 0.0, 0.0, 0.0

    ncc_before = tissue_ncc(fixed_2d, moving_2d, threshold=tissue_threshold)
    if not np.isfinite(ncc_before):
        return 0.0, 0.0, 0.0

    initial_offset = None
    try:
        initial_offset = _phase_offset(fixed_2d, moving_2d, max_refinement_px)
    except ValueError, RuntimeError, np.linalg.LinAlgError:
        logger.debug("Cut-face phase correlation failed", exc_info=True)

    diag: dict[str, Any] = {}
    tx, ty, _rot, _metric = register_refinement(
        fixed_2d,
        moving_2d,
        enable_rotation=False,
        max_rotation_deg=0.0,
        max_translation_px=max_refinement_px,
        fixed_mask=(fixed_2d > tissue_threshold),
        moving_mask=(moving_2d > tissue_threshold),
        initial_offset=initial_offset,
        bound_mode="scale",
        diagnostics=diag,
    )
    if diag.get("rejected"):
        logger.info(
            "Z-blend XY rejected: unconstrained mag=%.2f px > max %s px",
            float(np.hypot(diag.get("unconstrained_tx", 0.0), diag.get("unconstrained_ty", 0.0))),
            max_refinement_px,
        )
        return 0.0, 0.0, 0.0

    dy, dx = float(ty), float(tx)
    magnitude = float(np.hypot(dy, dx))
    if magnitude < 0.1:
        return 0.0, 0.0, 0.0

    from scipy.ndimage import shift as ndi_shift

    shifted_2d = ndi_shift(moving_2d, [dy, dx], order=1, mode="constant", cval=0.0)
    ncc_after = tissue_ncc(fixed_2d, shifted_2d, threshold=tissue_threshold)
    if not np.isfinite(ncc_after) or (ncc_after - ncc_before) < ncc_min_improve or ncc_after < ncc_min_absolute:
        logger.info(
            "Z-blend XY skipped: tissue NCC %.4f -> %.4f (dy=%.2f dx=%.2f)",
            ncc_before,
            ncc_after,
            dy,
            dx,
        )
        return 0.0, 0.0, 0.0
    logger.info(
        "Z-blend XY accepted: dy=%.2f dx=%.2f mag=%.2f px  NCC %.4f -> %.4f (cut-face)",
        dy,
        dx,
        magnitude,
        ncc_before,
        ncc_after,
    )
    return dy, dx, magnitude


def flatten_slice_z_profile(
    vol: np.ndarray,
    tissue_threshold: float = 0.01,
    gain_lo: float = 0.5,
    gain_hi: float = 2.0,
    passes: int = 2,
) -> np.ndarray:
    """Remove the bright-to-dark stripe along a slice's depth.

    Each plane is scaled so its tissue median matches the slice median.
    Within a plane, contrast is unchanged. Scaling changes which voxels
    sit above the threshold, so the correction is repeated.
    """
    out = vol
    for _ in range(max(1, passes)):
        nz = int(out.shape[0])
        if nz < 3:
            return out
        med = np.empty(nz, dtype=np.float64)
        for z in range(nz):
            tissue = out[z] > tissue_threshold
            med[z] = float(np.median(out[z][tissue])) if int(np.sum(tissue)) > 50 else np.nan
        good = np.isfinite(med)
        if int(np.sum(good)) < 3:
            return out
        idx = np.arange(nz)
        filled = np.interp(idx, idx[good], med[good])
        target = float(np.median(filled))
        gain = np.clip(target / np.maximum(filled, 1e-6), gain_lo, gain_hi)
        if float(np.max(np.abs(gain - 1.0))) < 0.02:
            return out
        # A dim plane can hold a few voxels far above its median. Cap those
        # before the plane is brightened, or they become a saturated block.
        adjusted = np.array(out, dtype=np.float32, copy=True)
        for z in range(nz):
            if gain[z] > 1.0 and np.isfinite(filled[z]):
                cap = float(filled[z]) * 5.0
                np.minimum(adjusted[z], cap, out=adjusted[z])
        out = (adjusted * gain[:, np.newaxis, np.newaxis]).astype(vol.dtype, copy=False)
    return out


def notch_slice_seams(
    block: np.ndarray,
    period: float,
    harmonics: tuple[int, ...] = (1, 2, 3, 4),
    rel_width: float = 0.12,
) -> np.ndarray:
    """Remove brightness that repeats once per slice, along axis 0.

    Only Fourier bins next to that period and its harmonics are cleared.
    Voxels that were zero stay zero, so the tissue outline does not move.
    """
    if period < 4:
        return block
    spec = np.fft.rfft(block, axis=0)
    freqs = np.fft.rfftfreq(block.shape[0])
    keep = np.ones(spec.shape[0], dtype=bool)
    for harmonic in harmonics:
        center = harmonic / float(period)
        keep &= np.abs(freqs - center) > rel_width * center
    keep[0] = True
    shape = (spec.shape[0],) + (1,) * (block.ndim - 1)
    spec *= keep.reshape(shape)
    out = np.fft.irfft(spec, n=block.shape[0], axis=0).astype(np.float32, copy=False)
    np.clip(out, 0, None, out=out)
    out[block == 0] = 0
    return out


def apply_seam_notch(array: Any, tile: int = 128, threshold: float = 0.05, workers: int | None = None) -> float:
    """Notch the slice-period seam on a ZYX array. Returns the period in voxels.

    The array is usually the stacked volume on disk, in compressed chunks.
    Both passes follow its chunk grid, so each chunk is decoded once for the
    plane medians and read and written once for the filter. Read plane by
    plane and in tiles that straddle the chunks, a chunk 100 planes deep was
    decoded 100 times for the medians, and every tile wrote parts of four
    chunk columns. The filter works on each voxel column on its own, so the
    size of the tiles does not change the result.

    Parameters
    ----------
    array
        ZYX array, changed in place (a zarr array or a NumPy array).
    tile
        Tile edge in Y and X for an array without chunks.
    threshold
        Voxels above it count as tissue for the plane medians.
    workers
        Tiles filtered at the same time. ``None`` takes the thread budget of
        the task (``OMP_NUM_THREADS``), at most 8.
    """
    step = 8
    nz, ny, nx = array.shape
    chunks = getattr(array, "chunks", None)
    z_block, tile_y, tile_x = (int(chunks[0]), int(chunks[1]), int(chunks[2])) if chunks else (64, tile, tile)

    medians = np.empty(nz, dtype=np.float64)
    for z0 in range(0, nz, z_block):
        planes = np.asarray(array[z0 : z0 + z_block, ::step, ::step])
        for k, plane in enumerate(planes):
            tissue = plane > threshold
            medians[z0 + k] = float(np.median(plane[tissue])) if int(np.sum(tissue)) > 40 else np.nan
    good = np.isfinite(medians)
    if int(np.sum(good)) < 8:
        return 0.0
    index = np.arange(len(medians))
    filled = np.interp(index, index[good], medians[good])
    period = estimate_slice_period(filled, period_min=12, period_max=40)

    def _notch_tile(bounds: tuple[int, int, int, int]) -> None:
        y0, y1, x0, x1 = bounds
        block = np.asarray(array[:, y0:y1, x0:x1], dtype=np.float32)
        array[:, y0:y1, x0:x1] = notch_slice_seams(block, period)

    tiles = [
        (y0, min(ny, y0 + tile_y), x0, min(nx, x0 + tile_x)) for y0 in range(0, ny, tile_y) for x0 in range(0, nx, tile_x)
    ]
    if workers is None:
        workers = thread_budget(8)
    if workers > 1 and len(tiles) > 1:
        # Tiles cover whole chunks, so no two of them write the same chunk.
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(_notch_tile, tiles))
    else:
        for bounds in tiles:
            _notch_tile(bounds)
    return period


def apply_local_depth_gain(
    array: Any,
    slab_starts: Sequence[int],
    threshold: float = 0.01,
    cell: int = 8,
    half_window: int = 3,
    workers: int | None = None,
) -> float:
    """Remove the depth trend inside each slab of a ZYX stack. Returns its rms in log units.

    The trend is measured in XY cells of ``cell`` pixels as the median depth
    profile of a slab and its neighbours (:func:`local_depth_field`), and
    removed with a gain that is smooth in XY. Each voxel is multiplied by one
    factor: the image inside a plane keeps its detail, nothing is filtered
    along Z, and voxels at zero stay at zero.

    The array is usually the stacked volume on disk, in compressed chunks.
    It is read twice and written once, in runs of planes one chunk deep.

    Parameters
    ----------
    array
        ZYX array, changed in place (a zarr array or a NumPy array).
    slab_starts
        First plane of every slab, as the stack placed them.
    threshold
        Voxels above it count as tissue.
    cell
        Edge of the XY cells the trend is measured in, in pixels.
    half_window
        Slabs on each side that enter the median profile.
    workers
        Planes scaled at the same time. ``None`` takes the thread budget of
        the task (``OMP_NUM_THREADS``), at most 8.
    """
    nz, ny, nx = array.shape
    starts = [int(s) for s in slab_starts]
    if len(starts) < 2 or min(ny, nx) < cell:
        return 0.0
    if starts[0] != 0 or any(b <= a for a, b in pairwise(starts)) or starts[-1] >= nz:
        logger.warning("Slab starts are not increasing from 0, local depth gain skipped")
        return 0.0
    bounds = list(zip(starts, [*starts[1:], nz], strict=True))
    chunks = getattr(array, "chunks", None)
    z_block = int(chunks[0]) if chunks else 64

    log_level = np.zeros((nz, ny // cell, nx // cell), dtype=np.float32)
    weight = np.zeros_like(log_level)
    for z0 in range(0, nz, z_block):
        planes = np.asarray(array[z0 : z0 + z_block], dtype=np.float32)
        z1 = z0 + planes.shape[0]
        log_level[z0:z1], weight[z0:z1] = coarse_log_tissue(planes, planes > threshold, block=cell)
    field = local_depth_field(log_level, weight, bounds, half_window=half_window)
    solid = weight > 0.9
    rms = float(np.sqrt(np.mean(field[solid] ** 2))) if solid.any() else 0.0
    del log_level, weight, solid
    if not field.any():
        return 0.0

    if workers is None:
        workers = thread_budget(8)
    for z0 in range(0, nz, z_block):
        if not field[z0 : z0 + z_block].any():
            continue
        planes = np.asarray(array[z0 : z0 + z_block], dtype=np.float32)

        def _scale_plane(k: int, planes: np.ndarray = planes, z0: int = z0) -> None:
            if field[z0 + k].any():
                planes[k] *= np.exp(-upsample_plane(field[z0 + k], cell, (ny, nx)))

        if workers > 1 and planes.shape[0] > 1:
            # Every plane is scaled on its own, so the threads share nothing.
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(_scale_plane, range(planes.shape[0])))
        else:
            for k in range(planes.shape[0]):
                _scale_plane(k)
        array[z0 : z0 + z_block] = planes
    return rms


def estimate_slice_period(medians: np.ndarray, period_min: float, period_max: float) -> float:
    """Return the strongest period, in planes, inside ``[period_min, period_max]``."""
    values = np.asarray(medians, dtype=np.float64)
    values = values - float(np.mean(values))
    spectrum = np.abs(np.fft.rfft(values))
    freqs = np.fft.rfftfreq(len(values))
    band = (freqs >= 1.0 / period_max) & (freqs <= 1.0 / period_min)
    if not np.any(band):
        return float(period_min)
    peak = int(np.argmax(np.where(band, spectrum, 0.0)))
    if freqs[peak] == 0:
        return float(period_min)
    return float(1.0 / freqs[peak])


def suppress_z_slice_bands(
    array: Any,
    tissue_threshold: float = 0.05,
    sigma_narrow: float = 2.0,
    sigma_wide: float = 24.0,
    gain_lo: float = 0.7,
    gain_hi: float = 1.4,
    tile: int = 100,
) -> None:
    """Take out the brightness wave that repeats once per slice.

    A plane-wide scale leaves the wave in place where one column does not
    follow the plane median. Along Z, divide the local mean by a mean taken
    over about one slice, and leave slower anatomy and within-plane contrast.
    """
    from scipy.ndimage import gaussian_filter1d

    _nz, ny, nx = array.shape
    for y0 in range(0, ny, tile):
        y1 = min(ny, y0 + tile)
        for x0 in range(0, nx, tile):
            x1 = min(nx, x0 + tile)
            block = np.asarray(array[:, y0:y1, x0:x1], dtype=np.float32)
            if not np.any(block > tissue_threshold):
                continue
            narrow = gaussian_filter1d(block, sigma_narrow, axis=0, mode="nearest")
            wide = gaussian_filter1d(block, sigma_wide, axis=0, mode="nearest")
            gain = np.ones(block.shape, dtype=np.float32)
            tissue = narrow > tissue_threshold
            gain[tissue] = np.clip(wide[tissue] / np.maximum(narrow[tissue], 1e-6), gain_lo, gain_hi).astype(np.float32)
            array[:, y0:y1, x0:x1] = block * gain


def slice_tissue_median(vol: np.ndarray, tissue_threshold: float = 0.01) -> float:
    """Median of voxels above ``tissue_threshold``, or NaN when there is too little tissue."""
    tissue = vol > tissue_threshold
    if int(np.sum(tissue)) < 100:
        return float("nan")
    return float(np.median(vol[tissue]))


def scale_slice_to_median(
    vol: np.ndarray,
    target: float,
    tissue_threshold: float = 0.01,
    gain_lo: float = 0.5,
    gain_hi: float = 1.4,
) -> tuple[np.ndarray, float]:
    """Scale a whole slice so its tissue median matches ``target``.

    One gain for the entire slab. A per-plane seam gain leaves a line at the cut.
    """
    med = slice_tissue_median(vol, tissue_threshold)
    if not np.isfinite(med) or med < 1e-6 or not np.isfinite(target):
        return vol, 1.0
    gain = float(np.clip(target / med, gain_lo, gain_hi))
    if abs(gain - 1.0) < 0.02:
        return vol, gain
    return (vol * gain).astype(vol.dtype, copy=False), gain


def seam_overlap_gain(
    existing: np.ndarray,
    moving: np.ndarray,
    tissue_threshold: float = 0.01,
    gain_lo: float = 0.7,
    gain_hi: float = 1.4,
) -> float:
    """Scale that matches incoming overlap tissue to the slice already in the stack.

    Returns 1 when there is too little shared tissue. The gain is clamped so a
    seam match cannot halve or double the slab.
    """
    both = (existing > tissue_threshold) & (moving > tissue_threshold)
    if int(np.sum(both)) < 1000:
        return 1.0
    med_fixed = float(np.median(existing[both]))
    med_moving = float(np.median(moving[both]))
    if med_moving < 1e-6:
        return 1.0
    return float(np.clip(med_fixed / med_moving, gain_lo, gain_hi))


def apply_seam_gain(vol: np.ndarray, z0: int, depth: int, gain: float, ramp: int = 6) -> np.ndarray:
    """Apply ``gain`` on the overlap and fade it back to 1 over the next planes."""
    if abs(gain - 1.0) < 0.02 or depth < 1:
        return vol
    out = vol.copy()
    z1 = min(out.shape[0], z0 + depth)
    tissue = out[z0:z1] > 0
    out[z0:z1][tissue] *= gain
    for i in range(ramp):
        z = z1 + i
        if z >= out.shape[0]:
            break
        weight = 1.0 - (i + 1) / (ramp + 1)
        plane_gain = 1.0 + (gain - 1.0) * weight
        mask = out[z] > 0
        out[z][mask] *= plane_gain
    return out


def refine_z_blend_overlap(
    existing: np.ndarray,
    moving_overlap: np.ndarray,
    max_refinement_px: float,
    ncc_min_improve: float = 1e-4,
) -> tuple[np.ndarray, float]:
    """Find and apply a keep-if-better XY shift on the overlap slab.

    Prefer :func:`estimate_z_blend_xy_shift` and shift the full incoming
    slice so unique (non-overlap) planes move with the blend zone.
    """
    from scipy.ndimage import shift as ndi_shift

    dy, dx, magnitude = estimate_z_blend_xy_shift(existing, moving_overlap, max_refinement_px, ncc_min_improve=ncc_min_improve)
    if magnitude <= 0:
        return moving_overlap, 0.0
    refined = ndi_shift(moving_overlap.astype(np.float32), [0, dy, dx], order=1, mode="constant", cval=0.0)
    return refined, magnitude


def extract_overlap_rois(
    fixed: np.ndarray,
    moving: np.ndarray,
    overlap: int,
    moving_z_start: int = 0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Z-end overlap of ``fixed`` vs Z-start of ``moving`` (after ``moving_z_start``).

    Serial OCT slabs overlap only at the extremities — the bottom of the
    previous slab and the top of the next — not through the full slice depth.
    """
    mz = max(0, int(moving_z_start))
    ov = min(int(overlap), fixed.shape[0], max(0, moving.shape[0] - mz))
    if ov < 1:
        return None
    f = fixed[-ov:]
    m = moving[mz : mz + ov]
    ny = min(f.shape[1], m.shape[1])
    nx = min(f.shape[2], m.shape[2])
    return f[:, :ny, :nx], m[:, :ny, :nx]


def overlap_z_profiles(
    fixed_ov: np.ndarray,
    moving_ov: np.ndarray,
    tissue_threshold: float = 0.01,
    min_voxels: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-plane median intensity in the overlap, both-tissue voxels only.

    XY coverage is partial: a voxel contributes only when both slabs exceed
    ``tissue_threshold``. Planes with fewer than ``min_voxels`` are NaN.
    """
    both = (fixed_ov > tissue_threshold) & (moving_ov > tissue_threshold)
    n_z = fixed_ov.shape[0]
    pf = np.full(n_z, np.nan, dtype=np.float64)
    pm = np.full(n_z, np.nan, dtype=np.float64)
    n = np.zeros(n_z, dtype=np.int64)
    for z in range(n_z):
        mask = both[z]
        n[z] = int(mask.sum())
        if n[z] < min_voxels:
            continue
        pf[z] = float(np.median(fixed_ov[z][mask]))
        pm[z] = float(np.median(moving_ov[z][mask]))
    return pf, pm, n


def fit_overlap_log_ratio(
    profile_fixed: np.ndarray,
    profile_moving: np.ndarray,
    min_planes: int = 4,
) -> tuple[float, float] | None:
    """Fit ``log(I_next / I_prev) = a + b * i`` on valid overlap planes.

    ``i = 0`` is the first overlap plane (still toward the unique part of the
    previous slab). Returns ``(a, b)`` or ``None`` if too few planes.
    """
    valid = np.isfinite(profile_fixed) & np.isfinite(profile_moving) & (profile_fixed > 1e-8) & (profile_moving > 1e-8)
    if int(valid.sum()) < min_planes:
        return None
    z = np.arange(len(profile_fixed), dtype=np.float64)[valid]
    y = np.log(profile_moving[valid] / profile_fixed[valid])
    b, a = np.polyfit(z, y, 1)
    return float(a), float(b)


def overlap_z_gain_curve(
    nz: int,
    overlap: int,
    a: float,
    b: float,
    clamp: tuple[float, float] = (0.5, 2.0),
) -> np.ndarray:
    """1-D gain for the previous (fixed) slab.

    Overlap planes get ``g = exp(a + b i)``. The unique (non-overlap) region
    ramps from 1 at the surface to the overlap-start gain so deep tissue is
    boosted toward the next slice's top without flattening the surface.
    """
    gain = np.ones(int(nz), dtype=np.float64)
    ov = min(max(0, int(overlap)), int(nz))
    if ov < 1:
        return gain.astype(np.float32)
    i_ov = np.arange(ov, dtype=np.float64)
    lo, hi = clamp
    g_ov = np.clip(np.exp(a + b * i_ov), lo, hi)
    gain[-ov:] = g_ov
    n_pre = int(nz) - ov
    if n_pre > 0:
        gain[:n_pre] = np.linspace(1.0, float(g_ov[0]), n_pre, dtype=np.float64)
    return np.clip(gain, lo, hi).astype(np.float32)


def apply_overlap_z_gain(vol: np.ndarray, gain: np.ndarray) -> np.ndarray:
    """Apply a per-Z multiplicative gain. ``gain`` length must match ``vol.shape[0]``."""
    if gain.shape[0] != vol.shape[0]:
        msg = f"gain length {gain.shape[0]} != volume Z {vol.shape[0]}"
        raise ValueError(msg)
    return vol * gain.astype(vol.dtype, copy=False)[:, None, None]


def estimate_overlap_z_gain_fit(
    fixed: np.ndarray,
    moving: np.ndarray,
    overlap: int,
    moving_z_start: int = 0,
    tissue_threshold: float = 0.01,
    min_voxels: int = 200,
    min_planes: int = 4,
) -> tuple[float, float, float, int] | None:
    """Fit overlap z-gain from Z-end / Z-start extremities only.

    Returns ``(a, b, xy_fraction, n_planes)`` or ``None``. ``xy_fraction`` is
    the mean fraction of overlap-plane voxels where both slabs have tissue.
    """
    rois = extract_overlap_rois(fixed, moving, overlap, moving_z_start)
    if rois is None:
        return None
    fixed_ov, moving_ov = rois
    pf, pm, n = overlap_z_profiles(fixed_ov, moving_ov, tissue_threshold=tissue_threshold, min_voxels=min_voxels)
    fit = fit_overlap_log_ratio(pf, pm, min_planes=min_planes)
    if fit is None:
        return None
    a, b = fit
    plane_area = float(fixed_ov.shape[1] * fixed_ov.shape[2])
    xy_fraction = float(np.mean(n / plane_area)) if plane_area > 0 else 0.0
    n_planes = int(np.isfinite(pf).sum())
    return a, b, xy_fraction, n_planes
