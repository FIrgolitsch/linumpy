"""Refinement registration: best-Z search and small rotation/translation correction."""

from collections.abc import Callable, Sequence

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, center_of_mass, gaussian_filter, sobel
from scipy.ndimage import shift as nd_shift
from skimage.registration import phase_cross_correlation


def find_best_z(
    fixed_vol: np.ndarray,
    moving_slice: np.ndarray,
    expected_z: int,
    search_range: int,
    metric: Callable[[np.ndarray, np.ndarray], float] | None = None,
) -> tuple[int, float]:
    """Find the Z-index in fixed_vol that best matches moving_slice.

    Uses normalized cross-correlation in the center region.

    Parameters
    ----------
    fixed_vol : array-like
        Fixed volume (Z, Y, X) or dask/zarr array.
    moving_slice : np.ndarray
        2D slice to match.
    expected_z : int
        Expected Z-index in fixed_vol for the match.
    search_range : int
        Search +/-search_range around expected_z.
    metric : callable, optional
        ``metric(fixed_plane, moving_slice)``; the plane with the highest
        value wins and planes where it is not finite are skipped. Without it
        the planes are compared by the correlation of their centre region,
        which follows the broad brightness of the tissue more than its
        structure and so leans toward the brighter, shallower planes.

    Returns
    -------
    best_z : int
        Z-index giving the best correlation.
    best_corr : float
        Correlation score at best_z. With a *metric* and no plane with a
        finite score, ``expected_z`` and NaN.
    """
    nz = fixed_vol.shape[0]
    expected_z = max(0, min(nz - 1, expected_z))
    z_min = max(0, expected_z - search_range)
    z_max = min(nz - 1, expected_z + search_range)

    if metric is not None:
        planes = np.asarray(fixed_vol[z_min : z_max + 1])
        best_z, best_score = expected_z, float("nan")
        for z in range(z_min, z_max + 1):
            score = float(metric(planes[z - z_min], moving_slice))
            if np.isfinite(score) and (not np.isfinite(best_score) or score > best_score):
                best_z, best_score = z, score
        return best_z, best_score

    if z_min >= z_max:
        return max(0, min(nz - 1, expected_z)), 0.0

    h, w = moving_slice.shape
    margin = min(h, w) // 4
    roi = (slice(margin, h - margin), slice(margin, w - margin))

    moving_roi = moving_slice[roi].astype(np.float32)
    valid_mov = moving_roi > 0
    if valid_mov.any():
        pmin = float(np.percentile(moving_roi[valid_mov], 5))
        pmax = float(np.percentile(moving_roi[valid_mov], 95))
        moving_roi = np.clip((moving_roi - pmin) / max(pmax - pmin, 1e-8), 0, 1)

    moving_norm = (moving_roi - moving_roi.mean()) / (moving_roi.std() + 1e-8)

    best_z = expected_z
    best_corr = -np.inf

    # The planes of the search range are read in one go: read one at a time
    # from a chunked store, every plane decodes the same chunks again.
    planes = np.asarray(fixed_vol[z_min : z_max + 1])
    for z in range(z_min, z_max + 1):
        fixed_slice = planes[z - z_min]
        fixed_roi = fixed_slice[roi].astype(np.float32)
        valid_fix = fixed_roi > 0
        if valid_fix.any():
            pmin = float(np.percentile(fixed_roi[valid_fix], 5))
            pmax = float(np.percentile(fixed_roi[valid_fix], 95))
            fixed_roi = np.clip((fixed_roi - pmin) / max(pmax - pmin, 1e-8), 0, 1)

        fixed_norm = (fixed_roi - fixed_roi.mean()) / (fixed_roi.std() + 1e-8)
        corr = float(np.mean(fixed_norm * moving_norm))

        if corr > best_corr:
            best_corr = corr
            best_z = z

    return max(0, min(nz - 1, best_z)), best_corr


def centre_ncc(fixed: np.ndarray, moving: np.ndarray) -> float:
    """Return the correlation of the centre region of two planes, each stretched to its own range.

    This is the score :func:`find_best_z` uses without a *metric*, for one
    plane.
    """
    h, w = moving.shape
    margin = min(h, w) // 4
    roi = (slice(margin, h - margin), slice(margin, w - margin))

    def _standardize(plane: np.ndarray) -> np.ndarray:
        region = plane[roi].astype(np.float32)
        valid = region > 0
        if valid.any():
            pmin = float(np.percentile(region[valid], 5))
            pmax = float(np.percentile(region[valid], 95))
            region = np.clip((region - pmin) / max(pmax - pmin, 1e-8), 0, 1)
        return (region - region.mean()) / (region.std() + 1e-8)

    return float(np.mean(_standardize(fixed) * _standardize(moving)))


def tissue_ncc(fixed: np.ndarray, moving: np.ndarray, threshold: float = 0.01) -> float:
    """Return NCC on voxels that are tissue in both images."""
    mask = (fixed > threshold) & (moving > threshold)
    if int(mask.sum()) < 100:
        return float("nan")
    av = fixed[mask].astype(np.float64)
    bv = moving[mask].astype(np.float64)
    av = av - av.mean()
    bv = bv - bv.mean()
    den = float(np.sqrt((av * av).sum() * (bv * bv).sum()))
    if den == 0.0:
        return float("nan")
    return float((av * bv).sum() / den)


def structure_ncc(
    fixed: np.ndarray,
    moving: np.ndarray,
    threshold: float = 0.01,
    sigma: float = 20.0,
) -> float:
    """Return NCC of the local structure of two images on voxels that are tissue in both.

    Each image is divided by its own smooth trend (a Gaussian of width *sigma*
    taken inside the common tissue mask) before the correlation, so only
    structure finer than about *sigma* counts.

    Plain :func:`tissue_ncc` also correlates the broad brightness pattern of
    the tissue. At a slab joint that pattern is not the same on both sides:
    one image is the deep end of a slab and the other the top of the next, and
    brightness falls off with depth at a rate that varies across the section.
    Where one side is brighter on the left and the other on the right the
    plain NCC can sit near zero at the correct alignment and rise for a shift
    of a hundred pixels that merely overlays the two bright halves.
    """
    mask = (fixed > threshold) & (moving > threshold)
    if int(mask.sum()) < 100:
        return float("nan")
    weight = gaussian_filter(mask.astype(np.float32), sigma)
    ok = mask & (weight > 0.25)
    if int(ok.sum()) < 100:
        return float("nan")
    safe = np.maximum(weight, 1e-6)

    def _local_contrast(image: np.ndarray) -> np.ndarray:
        trend = gaussian_filter(np.where(mask, image, 0.0).astype(np.float32), sigma) / safe
        return (image[ok] / np.maximum(trend[ok], 1e-6) - 1.0).astype(np.float64)

    av = _local_contrast(fixed)
    bv = _local_contrast(moving)
    av = av - av.mean()
    bv = bv - bv.mean()
    den = float(np.sqrt((av * av).sum() * (bv * bv).sum()))
    if den == 0.0:
        return float("nan")
    return float((av * bv).sum() / den)


def coarse_translation(
    fixed: np.ndarray,
    moving: np.ndarray,
    max_translation_px: float,
    threshold: float = 0.01,
    min_overlap: float = 0.5,
    metric: Callable[[np.ndarray, np.ndarray, float], float] = tissue_ncc,
) -> tuple[float, float, float]:
    """Find a translation of *moving* onto *fixed* that is too large for the bounded optimizer.

    Phase correlation proposes candidates from the tissue outlines and from the
    band-passed texture inside the tissue. The candidate with the highest
    *metric* wins; identity is always a candidate, so the result is never
    worse than doing nothing. *metric* is ``metric(fixed, moving, threshold)``,
    :func:`tissue_ncc` by default; pass :func:`structure_ncc` when the two
    images differ in their broad brightness pattern.

    Returns
    -------
    tx, ty : float
        Translation in the SimpleITK output->input convention used by the
        pairwise transforms (positive ``tx`` moves content left).
    ncc : float
        Value of *metric* at that translation.
    """
    fixed = fixed.astype(np.float32)
    moving = moving.astype(np.float32)
    mask_f = fixed > threshold
    mask_m = moving > threshold
    best_ncc = metric(fixed, moving, threshold)
    best = (0.0, 0.0)
    if max_translation_px <= 0 or not mask_f.any() or not mask_m.any():
        return 0.0, 0.0, best_ncc

    def _texture(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        band = gaussian_filter(image, 1.5) - gaussian_filter(image, 15.0)
        return band * binary_erosion(mask, iterations=5)

    pairs = [
        (mask_f.astype(np.float32), mask_m.astype(np.float32), None),
        (_texture(fixed, mask_f), _texture(moving, mask_m), None),
        (_texture(fixed, mask_f), _texture(moving, mask_m), "phase"),
    ]
    smaller = float(min(mask_f.sum(), mask_m.sum()))
    for ref, mov, normalization in pairs:
        if not ref.any() or not mov.any():
            continue
        shift, _error, _phase = phase_cross_correlation(ref, mov, upsample_factor=2, normalization=normalization)
        dy, dx = float(shift[0]), float(shift[1])
        if float(np.hypot(dy, dx)) > max_translation_px:
            continue
        shifted = nd_shift(moving, (dy, dx), order=1, mode="constant", cval=0.0)
        if float((mask_f & (shifted > threshold)).sum()) < min_overlap * smaller:
            continue
        ncc = metric(fixed, shifted, threshold)
        if np.isfinite(ncc) and (not np.isfinite(best_ncc) or ncc > best_ncc):
            best_ncc = ncc
            best = (-dx, -dy)
    return best[0], best[1], best_ncc


def apply_refinement_bounds(
    tx: float,
    ty: float,
    angle_deg: float,
    max_translation_px: float,
    max_rotation_deg: float,
    bound_mode: str = "scale",
) -> tuple[float, float, float, bool]:
    """Limit a residual rigid transform.

    ``scale`` (pairwise default) radially clamps translation and clips rotation.
    That projects a runaway optimum onto the bound circle — a 60 px jump
    becomes a 10 px walk in the wrong direction.

    ``reject`` returns identity when either limit is exceeded. Use this for
    manual-transform polish: if the optimizer wanted to leave the overlay,
    keep the overlay.
    """
    mag = float(np.sqrt(tx**2 + ty**2))
    over_t = max_translation_px > 0 and mag > max_translation_px
    over_r = max_rotation_deg > 0 and abs(angle_deg) > max_rotation_deg
    if bound_mode == "reject" and (over_t or over_r):
        return 0.0, 0.0, 0.0, True
    if over_t:
        scale = max_translation_px / mag
        tx, ty = tx * scale, ty * scale
    if over_r:
        angle_deg = float(np.clip(angle_deg, -max_rotation_deg, max_rotation_deg))
    return float(tx), float(ty), float(angle_deg), False


def register_refinement(
    fixed: np.ndarray,
    moving: np.ndarray,
    enable_rotation: bool = True,
    max_rotation_deg: float = 5.0,
    max_translation_px: float = 20.0,
    fixed_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
    initial_offset: tuple[float, float] | None = None,
    bound_mode: str = "scale",
    shrink_factors: list[int] | None = None,
    smoothing_sigmas: list[float] | None = None,
    diagnostics: dict | None = None,
) -> tuple[float, float, float, float]:
    """Compute small rotation and translation refinement using SimpleITK.

    Parameters
    ----------
    fixed, moving : np.ndarray
        2D images for registration (should be normalized to [0, 1]).
    enable_rotation : bool
        Enable Euler2D rotation (default True). False = translation only.
    max_rotation_deg : float
        Maximum allowed rotation in degrees.
    max_translation_px : float
        Maximum allowed translation in pixels.
    fixed_mask, moving_mask : np.ndarray or None
        Optional tissue masks multiplied into images before registration.
    initial_offset : tuple[float, float] or None
        Optional initial (dy, dx) translation offset for the transform.
    bound_mode : {'scale', 'reject'}
        How to enforce ``max_translation_px`` / ``max_rotation_deg``.
        ``scale`` clamps; ``reject`` returns identity if the unconstrained
        solution exceeds either limit.
    shrink_factors, smoothing_sigmas : list or None
        Multiscale pyramid. Default ``[4, 2, 1]`` / ``[2, 1, 0]`` (pairwise).
        Manual polish should pass ``[1]`` / ``[0]`` so coarse levels cannot
        pull toward the tissue silhouette.
    diagnostics : dict or None
        If given, filled with unconstrained parameters and ``rejected``.

    Returns
    -------
    tx, ty : float
        Translation refinement in pixels.
    angle_deg : float
        Rotation angle in degrees.
    metric : float
        Registration metric value.
    """
    fixed_std = np.std(fixed[fixed > 0]) if np.any(fixed > 0) else 0
    moving_std = np.std(moving[moving > 0]) if np.any(moving > 0) else 0
    if fixed_std < 0.01 or moving_std < 0.01:
        return 0.0, 0.0, 0.0, 0.0

    fixed_masked = fixed * fixed_mask.astype(np.float32) if fixed_mask is not None else fixed
    moving_masked = moving * moving_mask.astype(np.float32) if moving_mask is not None else moving

    fixed_sitk = sitk.GetImageFromArray(fixed_masked.astype(np.float32))
    moving_sitk = sitk.GetImageFromArray(moving_masked.astype(np.float32))

    if enable_rotation:
        transform = sitk.Euler2DTransform()
        center = [fixed.shape[1] / 2.0, fixed.shape[0] / 2.0]
        transform.SetCenter(center)
        if initial_offset is not None:
            transform.SetTranslation([initial_offset[1], initial_offset[0]])
    else:
        transform = sitk.TranslationTransform(2)
        if initial_offset is not None:
            transform.SetOffset([initial_offset[1], initial_offset[0]])

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsCorrelation()
    reg.SetOptimizerAsGradientDescent(
        learningRate=1.0, numberOfIterations=200, convergenceMinimumValue=1e-6, convergenceWindowSize=10
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(transform, inPlace=False)
    reg.SetInterpolator(sitk.sitkLinear)
    factors = [4, 2, 1] if shrink_factors is None else list(shrink_factors)
    sigmas = [2, 1, 0] if smoothing_sigmas is None else list(smoothing_sigmas)
    if len(factors) != len(sigmas):
        raise ValueError("shrink_factors and smoothing_sigmas must have the same length")
    reg.SetShrinkFactorsPerLevel(factors)
    reg.SetSmoothingSigmasPerLevel(sigmas)

    try:
        final = reg.Execute(fixed_sitk, moving_sitk)
        metric = reg.GetMetricValue()

        inner = final.GetNthTransform(0) if final.GetName() == "CompositeTransform" else final

        if enable_rotation:
            euler = sitk.Euler2DTransform(inner)
            angle_deg = np.degrees(euler.GetAngle())
            tx, ty = euler.GetTranslation()
        else:
            tx, ty = inner.GetOffset()
            angle_deg = 0.0

        if diagnostics is not None:
            diagnostics["unconstrained_tx"] = float(tx)
            diagnostics["unconstrained_ty"] = float(ty)
            diagnostics["unconstrained_rot_deg"] = float(angle_deg)

        tx, ty, angle_deg, rejected = apply_refinement_bounds(
            float(tx), float(ty), float(angle_deg), max_translation_px, max_rotation_deg, bound_mode
        )
        if diagnostics is not None:
            diagnostics["rejected"] = rejected
        return tx, ty, angle_deg, metric

    except Exception:
        return 0.0, 0.0, 0.0, float("inf")


def centre_of_mass_offset(fixed: np.ndarray, moving: np.ndarray) -> tuple[float, float]:
    """Compute alignment offset using center-of-mass difference.

    Parameters
    ----------
    fixed, moving : np.ndarray
        2D normalized images (values in [0, 1]).

    Returns
    -------
    dy, dx : float
        Translation from moving to fixed (fixed - moving offset).
    """
    fixed_binary = fixed > 0.1
    moving_binary = moving > 0.1

    if not fixed_binary.any() or not moving_binary.any():
        return 0.0, 0.0

    cy_f, cx_f = center_of_mass(fixed * fixed_binary.astype(np.float32))
    cy_m, cx_m = center_of_mass(moving * moving_binary.astype(np.float32))

    return float(cy_f - cy_m), float(cx_f - cx_m)


def gradient_magnitude_alignment(fixed: np.ndarray, moving: np.ndarray) -> tuple[float, float]:
    """Compute alignment offset using phase correlation on gradient magnitude images.

    Parameters
    ----------
    fixed, moving : np.ndarray
        2D normalized images (values in [0, 1]).

    Returns
    -------
    dy, dx : float
        Translation from moving to fixed (fixed - moving offset).
    """

    def grad_mag(img: np.ndarray) -> np.ndarray:
        gx = sobel(img, axis=1)
        gy = sobel(img, axis=0)
        return np.sqrt(gx**2 + gy**2)

    fixed_grad = grad_mag(fixed.astype(np.float32))
    moving_grad = grad_mag(moving.astype(np.float32))

    shift, _, _ = phase_cross_correlation(fixed_grad, moving_grad, normalization=None, upsample_factor=4)
    return float(shift[0]), float(shift[1])


def apply_rigid_2d(
    image: np.ndarray,
    tx: float,
    ty: float,
    angle_deg: float = 0.0,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """Resample *image* with a rigid transform in the convention of the pairwise transforms.

    The transform maps an output point to the input point it is read from,
    rotating about *center* ``(x, y)``, by default the middle of the image.
    Pixels read from outside the image are zero.
    """
    moving = sitk.GetImageFromArray(np.ascontiguousarray(image, dtype=np.float32))
    transform = sitk.Euler2DTransform()
    if center is None:
        center = (image.shape[1] / 2.0, image.shape[0] / 2.0)
    transform.SetCenter([float(center[0]), float(center[1])])
    transform.SetAngle(float(np.radians(angle_deg)))
    transform.SetTranslation([float(tx), float(ty)])
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(moving)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    return sitk.GetArrayFromImage(resampler.Execute(moving))


def select_rigid_candidate(
    fixed: np.ndarray,
    moving: np.ndarray,
    candidates: Sequence[tuple[str, float, float, float]],
    threshold: float = 0.01,
    min_gain: float = 0.02,
    metric: Callable[[np.ndarray, np.ndarray, float], float] = structure_ncc,
    min_score: float = -np.inf,
) -> tuple[str, float, float, float, float, float]:
    """Choose the rigid transform under which *moving* matches *fixed* best.

    Doing nothing is always a choice, and another candidate replaces it only
    when it raises *metric* by at least *min_gain* and reaches *min_score*. An
    optimizer that left a good starting position for a worse one is thereby
    dropped, whatever its own cost function says. So is a shift that lifts
    two planes with nothing in common from no match to a faint one.

    Parameters
    ----------
    fixed, moving : ndarray
        The two planes, on the same grid.
    candidates : sequence of (name, tx, ty, angle_deg)
        Transforms to try, in the convention of :func:`apply_rigid_2d`.
    threshold : float
        Tissue threshold handed to *metric*.
    min_gain : float
        Smallest gain over the identity for a candidate to be kept.
    metric : callable
        ``metric(fixed, moving, threshold)``; higher is better.
    min_score : float
        Lowest value of *metric* a candidate must reach to be kept.

    Returns
    -------
    name, tx, ty, angle_deg : str, float, float, float
        The chosen transform; ``("identity", 0, 0, 0)`` when none qualifies.
    score : float
        *metric* under the chosen transform.
    identity_score : float
        *metric* without any transform.
    """
    fixed = fixed.astype(np.float32)
    moving = moving.astype(np.float32)
    identity_score = float(metric(fixed, moving, threshold))
    best = ("identity", 0.0, 0.0, 0.0, identity_score)
    floor = max(identity_score + min_gain, min_score) if np.isfinite(identity_score) else min_score
    for name, tx, ty, angle_deg in candidates:
        if tx == 0.0 and ty == 0.0 and angle_deg == 0.0:
            continue
        score = float(metric(fixed, apply_rigid_2d(moving, tx, ty, angle_deg), threshold))
        if not np.isfinite(score) or score < floor:
            continue
        if best[0] == "identity" or score > best[4]:
            best = (name, float(tx), float(ty), float(angle_deg), score)
    return best[0], best[1], best[2], best[3], best[4], identity_score
