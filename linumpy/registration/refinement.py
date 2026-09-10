"""Refinement registration: best-Z search and small rotation/translation correction."""

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import center_of_mass, sobel
from skimage.registration import phase_cross_correlation


def find_best_z(fixed_vol: np.ndarray, moving_slice: np.ndarray, expected_z: int, search_range: int) -> tuple[int, float]:
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

    Returns
    -------
    best_z : int
        Z-index giving the best correlation.
    best_corr : float
        Correlation score at best_z.
    """
    nz = fixed_vol.shape[0]
    expected_z = max(0, min(nz - 1, expected_z))
    z_min = max(0, expected_z - search_range)
    z_max = min(nz - 1, expected_z + search_range)

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

    for z in range(z_min, z_max + 1):
        fixed_slice = np.array(fixed_vol[z])
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
