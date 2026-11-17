"""Attenuation estimation and profile recovery models for OCT volumes."""

from linumpy.config.threads import worker_initializer

import itertools
import multiprocessing
from typing import Any, Literal, overload

import numpy as np
import SimpleITK as sitk
from numpy.ma import MaskedArray
from scipy.interpolate import interp1d
from scipy.ndimage import (
    binary_erosion,
    binary_fill_holes,
    gaussian_filter,
    median_filter,
    uniform_filter,
)
from scipy.optimize import minimize
from skimage.filters import threshold_li

from linumpy.cli.args import get_available_cpus
from linumpy.geometry.crop import mask_under_interface
from linumpy.geometry.interface import find_tissue_interface, get_interface_depth_from_mask
from linumpy.gpu import GPU_AVAILABLE
from linumpy.intensity.normalize import eqhist, normalize

# ---------------------------------------------------------------------------
# Shared helpers used by Vermeer / Smith / Liu / Li
# ---------------------------------------------------------------------------


def _release_gpu_memory() -> None:
    """Hand the blocks CuPy keeps for reuse back to the device.

    Each stage here needs a few copies of the slab for a moment. Left in the
    pool they add up to several GB per process, which is what limits how many
    slabs can share a GPU.
    """
    import cupy as cp

    cp.get_default_memory_pool().free_all_blocks()


def _median_xy(vol: np.ndarray, radius: int, use_gpu: bool) -> np.ndarray:
    """Median over a ``(2 * radius + 1)`` square in XY of an ``(X, Y, Z)`` volume, edges repeated."""
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        from cupyx.scipy.ndimage import median_filter as gpu_median_filter

        size = 2 * radius + 1
        out = cp.asnumpy(gpu_median_filter(cp.asarray(vol), size=(size, size, 1), mode="nearest"))
        _release_gpu_memory()
        return out
    return sitk.GetArrayFromImage(sitk.Median(sitk.GetImageFromArray(vol), (0, radius, radius)))


def _median_xy_filter(vol: np.ndarray, k: int, bin_xy: int = 1, use_gpu: bool = False) -> np.ndarray:
    """XY median filter (no axial smoothing). ``k <= 0`` is a no-op.

    ``k`` is the radius of the window, which is ``2k + 1`` voxels wide.

    With ``bin_xy > 1`` the volume is first averaged over ``bin_xy`` x ``bin_xy``
    voxels in XY, filtered there with the radius scaled down to match, and put
    back on the full grid. The window covers the same area; the cost falls with
    about the fourth power of ``bin_xy``.

    With ``use_gpu`` the median runs on the GPU when one is available, which
    gives the same values.
    """
    if k <= 0:
        return vol
    if bin_xy <= 1:
        return _median_xy(vol, k, use_gpu)
    nx, ny, nz = vol.shape
    padded = np.pad(vol, ((0, (-nx) % bin_xy), (0, (-ny) % bin_xy), (0, 0)), mode="edge")
    small = padded.reshape(padded.shape[0] // bin_xy, bin_xy, padded.shape[1] // bin_xy, bin_xy, nz).mean(
        axis=(1, 3), dtype=np.float32
    )
    small = _median_xy(small, max(1, round(k / bin_xy)), use_gpu)
    return np.repeat(np.repeat(small, bin_xy, axis=0), bin_xy, axis=1)[:nx, :ny].astype(vol.dtype, copy=False)


def _auto_tissue_mask(vol: np.ndarray, zshift: int) -> np.ndarray:
    """Detect the water/tissue interface and return the under-interface mask."""
    interface = find_tissue_interface(vol, s_xy=3, s_z=1, order=1, use_log=True)
    return mask_under_interface(vol, interface + zshift, return_mask=True)


def _finalize_attenuation(attn: np.ndarray, mask: np.ndarray, fill_holes: bool) -> np.ndarray:
    """Replace NaNs with 0, zero out voxels outside ``mask``, optionally fill holes."""
    attn[np.isnan(attn)] = 0
    attn[~mask.astype(bool)] = 0
    if fill_holes:
        attn = sitk.GetArrayFromImage(sitk.GrayscaleFillhole(sitk.GetImageFromArray(attn)))
    return attn


def _exact_tail_C(i_max: np.ndarray, mu_E_per_m: np.ndarray, dz_m: float, valid: np.ndarray | None = None) -> np.ndarray:
    """Exact finite-range tail integral C = I[imax] / (exp(2 mu_E dz) - 1).

    An A-line whose fit shows no decay cannot give a tail of its own: the
    formula has no finite value there, and C = 0 would be the bare estimator,
    which diverges towards the end of the range. Such A-lines take the median
    decay rate of the A-lines that have one. ``valid`` limits both the median
    and the A-lines that receive it; C stays 0 elsewhere.
    """
    denom = np.expm1(2.0 * mu_E_per_m * dz_m)
    C = np.zeros_like(i_max, dtype=np.float64)
    decays = denom > 1e-12
    if valid is not None:
        decays &= valid
    C[decays] = i_max[decays] / denom[decays]
    flat = ~decays if valid is None else (valid & ~decays)
    if decays.any() and flat.any():
        C[flat] = i_max[flat] / np.expm1(2.0 * float(np.median(mu_E_per_m[decays])) * dz_m)
    return C


def _tail_at_aline_end(vol: np.ndarray, mask: np.ndarray, n_tail: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the decay of each A-line on its last ``n_tail`` voxels inside ``mask``.

    The tail constant stands for the signal below the last voxel that enters
    the sum, so it has to be fitted there. A mask that ends above the bottom of
    the volume (an A-line cut where its signal runs out) leaves voxels below
    it that are not summed; a fit on the bottom of the volume would describe
    those instead.

    Returns the signal at the last in-mask voxel, the slope of ``ln I`` per
    voxel there, and which A-lines had at least four voxels to fit. All
    ``(X, Y)``.
    """
    nz = vol.shape[2]
    z_idx = np.arange(nz)
    mask = mask.astype(bool)
    last = np.where(mask, z_idx[None, None, :], -1).max(axis=2)
    first = np.where(mask, z_idx[None, None, :], nz).min(axis=2)
    idx = last[..., None] - np.arange(n_tail)[::-1][None, None, :]
    inside = (idx >= first[..., None]) & (last[..., None] >= 0)
    vals = np.take_along_axis(vol, np.clip(idx, 0, nz - 1), axis=2).astype(np.float64)
    peak = np.where(inside, vals, 0.0).max(axis=2, keepdims=True)
    log_vals = np.log(np.maximum(vals, np.maximum(peak * 1e-12, 1e-30)))
    w = inside.astype(np.float64)
    n = w.sum(axis=2)
    t = np.arange(n_tail, dtype=np.float64)[None, None, :]
    safe_n = np.maximum(n, 1.0)
    t_mean = (w * t).sum(axis=2) / safe_n
    log_mean = (w * log_vals).sum(axis=2) / safe_n
    cov = (w * (t - t_mean[..., None]) * (log_vals - log_mean[..., None])).sum(axis=2)
    var = (w * (t - t_mean[..., None]) ** 2).sum(axis=2)
    fitted = n >= 4
    slope = np.where(fitted, cov / np.maximum(var, 1e-12), 0.0)
    i_end = np.take_along_axis(vol, np.clip(last, 0, nz - 1)[..., None], axis=2)[..., 0].astype(np.float64)
    i_end = np.where(last >= 0, i_end, 0.0)
    return i_end, slope, fitted


def _vermeer_core_gpu(vol: np.ndarray, dz: float, mask: np.ndarray, C: np.ndarray) -> np.ndarray:
    """CuPy implementation of the Vermeer Eq. 17 core (cumsum + log)."""
    import cupy as cp

    vol_g = cp.asarray(vol, dtype=cp.float32)
    mask_g = cp.asarray(mask, dtype=cp.bool_)
    C_g = cp.asarray(C, dtype=cp.float32)
    masked = cp.where(mask_g, vol_g, 0)
    cum_rev = cp.cumsum(masked[:, :, ::-1], axis=-1)
    tail = cp.zeros_like(cum_rev)
    tail[:, :, 1:] = cum_rev[:, :, :-1]
    profile = tail[:, :, ::-1] + C_g
    pos = profile > 0
    ratio = cp.where(pos, vol_g / cp.where(pos, profile, 1), 0)
    mu = cp.asnumpy(cp.where(pos, cp.log1p(ratio) / (2.0 * dz * 100.0), 0))
    del vol_g, mask_g, C_g, masked, cum_rev, tail, profile, pos, ratio
    _release_gpu_memory()
    return mu


def get_attenuation_vermeer2013(
    vol: np.ndarray,
    dz: float = 6.5e-6,
    mask: np.ndarray | None = None,
    C: np.ndarray | int | float | None = None,
    use_gpu: bool = False,
) -> np.ndarray:
    r"""Vermeer 2014 depth-resolved attenuation estimator (Eq. 17).

    Implements ``mu[i] = (1/2dz) * ln(1 + I[i] / (sum_{j>i} I[j] + C))``.
    See ``docs/ATTENUATION_METHODS.md`` for the math and Smith / Liu / Li
    extensions.

    Parameters
    ----------
    vol : ndarray
        OCT intensity volume in ``(X, Y, Z)`` order.
    dz : float
        Axial pixel size in metres. Output is rescaled to ``1/cm``.
    mask : ndarray, optional
        Boolean tissue mask. Voxels above the interface are zeroed,
        voxels below inherit the bottom in-mask value.
    C : ndarray, int, float, optional
        Finite-range tail constant. ``None`` (default) reproduces the
        original ``C = 0`` form.
    use_gpu : bool
        Run the cumsum/log core on CuPy when ``True`` and a GPU is
        available; falls back to NumPy otherwise.

    References
    ----------
    Vermeer et al., Biomed. Opt. Express 5, 322 (2014).
    Neubrand et al., J. Biomed. Opt. 28, 066001 (2023).
    """
    # Tail constant for the signal below the summed range, one value per A-line.
    if C is None:
        C = 0.0
    elif not isinstance(C, (int, float)) and np.ndim(C) == 2:
        C = np.asarray(C)[:, :, None]

    # Use mask to ignore tissue layers above / under the mask
    if mask is None:
        mask = np.ones_like(vol, dtype=bool)
    mask = mask.astype(bool, copy=False)

    if use_gpu and GPU_AVAILABLE:
        mu = _vermeer_core_gpu(vol, dz, mask, np.broadcast_to(np.asarray(C, dtype=np.float32), vol.shape))
    else:
        # Compensation profile with depth for each A-line. See Vermeer 2014
        # / Neubrand 2023 Appendix B for the derivation. Voxels outside the
        # mask count as zero in the sum. (A masked array is not used here: its
        # mask travels with the shifted sum and blanks the last in-mask voxel,
        # which is the value carried down the A-line below.)
        masked = np.where(mask, vol, 0).astype(np.float32, copy=False)
        cum_rev = np.cumsum(masked[:, :, ::-1], axis=-1)
        profile = np.empty_like(cum_rev)
        profile[:, :, -1] = 0.0
        profile[:, :, :-1] = cum_rev[:, :, -2::-1]  # sum over j > i
        del cum_rev, masked
        profile += C

        mu = np.zeros(vol.shape, dtype=np.float32)
        pos = profile > 0
        np.divide(vol, profile, out=mu, where=pos)
        np.log1p(mu, out=mu, where=pos)
        mu /= 2.0 * dz * 100.0

    # Masking the result (interface propagation)
    nz = mask.shape[2]
    z_idx = np.arange(nz)[None, None, :]
    has_tissue = mask.any(axis=2)
    interface = np.where(has_tissue, mask.argmax(axis=2), 0)
    mu[z_idx < interface[:, :, None]] = 0

    # Below the last in-mask voxel the A-line keeps that voxel's value.
    bottom_interface = np.where(has_tissue, nz - 1 - mask[:, :, ::-1].argmax(axis=2), nz - 1)
    bottom_mu = np.take_along_axis(mu, bottom_interface[:, :, None], axis=2)
    below = z_idx >= bottom_interface[:, :, None]
    np.copyto(mu, np.broadcast_to(bottom_mu, mu.shape), where=below)

    return mu


def get_attenuation_smith2015(
    vol: np.ndarray,
    mask: np.ndarray | None = None,
    k: int = 10,
    sigma: int = 5,
    sigma_bottom: int = 3,
    dz: int = 1,
    res: float = 6.5,
    zshift: int = 3,
    fill_holes: bool = False,
    use_gpu: bool = False,
) -> np.ndarray:
    r"""Smith 2015 depth-resolved attenuation: Vermeer + linearized tail extension.

    Adds a finite-range tail constant ``C ~= I[imax] / (2 mu_E dz)``
    estimated from a log-gradient fit averaged inside the tissue mask.
    See ``docs/ATTENUATION_METHODS.md`` for the math.

    Parameters
    ----------
    vol : ndarray
        OCT volume in ``(X, Y, Z)`` order.
    mask : ndarray, optional
        Tissue mask; auto-detected via :func:`find_tissue_interface` if
        ``None``.
    k : int
        XY median-filter kernel (voxels); ``0`` disables denoising.
    sigma : int
        Axial Gaussian sigma (voxels) for the gradient-based ``mu_E``
        fit.
    sigma_bottom : int
        XY Gaussian sigma applied to the per-A-line ``C`` map.
    dz : int
        Axial integration window (voxels) for ``I[imax]``.
    res : float
        Axial resolution in ``micron/pixel``.
    zshift : int
        Voxels below the auto-detected interface to skip.
    fill_holes : bool
        Apply morphological hole-filling.
    use_gpu : bool
        Forward to the Vermeer core (CuPy when available).

    References
    ----------
    Smith et al., IEEE Trans. Med. Imaging 34, 2592 (2015).
    """
    vol = _median_xy_filter(vol, k)
    if mask is None:
        mask = _auto_tissue_mask(vol, zshift)

    # Per-A-line slope estimate mu*dz (round-trip), averaged inside the mask.
    exp_fit = get_gradient_attenuation(gaussian_filter(vol, (0, 0, sigma)))
    exp_fit = MaskedArray(exp_fit, mask=~mask).mean(axis=2)
    exp_fit[np.isnan(exp_fit)] = 0

    # Mean intensity in a window of `dz` voxels above the bottom interface.
    interface_bottom = vol.shape[2] - get_interface_depth_from_mask(mask[:, :, ::-1]) - 1 - dz
    mask_bottom = mask_under_interface(vol, interface_bottom, return_mask=True)
    mask_bottom = (mask_bottom * mask).astype(bool)
    i0 = MaskedArray(vol, mask=~mask_bottom).mean(axis=2)

    # Linearized end-of-scan tail integral C ~= I[imax] / (2 * mu_E * dz).
    epsilon = 1e-3
    C = np.zeros_like(i0)
    C[exp_fit > epsilon] = i0[exp_fit > epsilon] / (2.0 * exp_fit[exp_fit > epsilon])
    C = gaussian_filter(C, sigma_bottom)

    attn = get_attenuation_vermeer2013(vol, dz=res * 1e-6, mask=mask, C=C, use_gpu=use_gpu)
    return _finalize_attenuation(attn, mask, fill_holes)


# Backwards-compatible alias for the previous name.
def get_extended_attenuation_vermeer2013(*args: Any, **kwargs: Any) -> np.ndarray:
    """Forward to :func:`get_attenuation_smith2015` (deprecated alias)."""
    import warnings

    warnings.warn(
        "get_extended_attenuation_vermeer2013 has been renamed to "
        "get_attenuation_smith2015 (Smith et al., IEEE TMI 2015). The "
        "old name is kept for backwards compatibility and will be "
        "removed in a future release.",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_attenuation_smith2015(*args, **kwargs)


def get_attenuation_liu2019(
    vol: np.ndarray,
    mask: np.ndarray | None = None,
    k: int = 10,
    res: float = 6.5,
    zshift: int = 3,
    tail_fit_voxels: int = 20,
    fill_holes: bool = False,
    use_gpu: bool = False,
) -> np.ndarray:
    r"""Liu 2019 optimized depth-resolved attenuation estimator.

    Vermeer 2014 with the *exact* finite-range tail
    ``C = I[imax] / (exp(2 mu_E dz) - 1)`` and a per-A-line
    least-squares fit for ``mu_E``. See
    ``docs/ATTENUATION_METHODS.md`` for the math.

    Parameters
    ----------
    vol : ndarray
        OCT volume in ``(X, Y, Z)`` order.
    mask : ndarray, optional
        Tissue mask; auto-detected if ``None``.
    k : int
        XY median-filter kernel (voxels).
    res : float
        Axial resolution in ``micron/pixel``.
    zshift : int
        Voxels to skip below the auto-detected interface.
    tail_fit_voxels : int
        Bottom-of-A-line voxels for the ``mu_E`` LSQ fit
        (``2 <= tail_fit_voxels <= nz``).
    fill_holes : bool
        Apply morphological hole-filling.
    use_gpu : bool
        Forward to the Vermeer core (CuPy when available).

    References
    ----------
    Liu et al., J. Biomed. Opt. 24, 035002 (2019).
    Neubrand et al., J. Biomed. Opt. 28, 066001 (2023).
    """
    vol = _median_xy_filter(vol, k)
    if mask is None:
        mask = _auto_tissue_mask(vol, zshift)

    nz = vol.shape[2]
    n_tail = int(np.clip(tail_fit_voxels, 2, nz))
    dz_m = res * 1e-6

    # The tail is fitted where the summed range of each A-line ends: the last
    # voxels inside the mask, which is the bottom of the volume only when the
    # mask reaches it.
    i_end, slope, fitted = _tail_at_aline_end(vol, mask, n_tail)
    mu_E_per_m = np.maximum(-slope / (2.0 * dz_m), 0.0)
    C = _exact_tail_C(i_end, mu_E_per_m, dz_m, valid=fitted)

    attn = get_attenuation_vermeer2013(vol, dz=dz_m, mask=mask, C=C.astype(np.float32), use_gpu=use_gpu)
    return _finalize_attenuation(attn, mask, fill_holes)


def get_attenuation_li2020(
    vol: np.ndarray,
    mask: np.ndarray | None = None,
    k: int = 10,
    res: float = 6.5,
    zshift: int = 3,
    snr_threshold_db: float = 6.0,
    noise_floor: float | None = None,
    tail_fit_voxels: int = 20,
    fill_holes: bool = False,
    use_gpu: bool = False,
    median_bin: int = 1,
    subtract_noise_floor: bool = True,
) -> np.ndarray:
    r"""Li 2020 robust depth-resolved attenuation: noise-floor + tail truncation.

    Subtracts a constant noise floor, truncates each A-line where SNR
    falls below ``snr_threshold_db``, then runs the Liu 2019 estimator on
    the truncated A-line with its tail fitted at the cut. Below the cut the
    estimate keeps the decay rate of that tail down to the end of the tissue
    mask. See ``docs/ATTENUATION_METHODS.md`` for the math.

    Parameters
    ----------
    vol : ndarray
        OCT volume in ``(X, Y, Z)`` order.
    mask : ndarray, optional
        Tissue mask; auto-detected if ``None``.
    k : int
        XY median-filter kernel (voxels).
    res : float
        Axial resolution in ``micron/pixel``.
    zshift : int
        Voxels to skip below the auto-detected interface.
    snr_threshold_db : float
        Per-voxel SNR (vs noise floor) below which the A-line is
        truncated. Li 2020 and Liu 2019 use 5 dB above a measured system
        noise floor; the floor here is the level outside the tissue.
    noise_floor : float, optional
        Constant to subtract; estimated from out-of-mask voxels (or
        the deepest 5 voxels) when ``None``.
    tail_fit_voxels : int
        Bottom-of-A-line voxels for the ``mu_E`` LSQ fit.
    fill_holes : bool
        Apply morphological hole-filling.
    use_gpu : bool
        Forward to the Vermeer core (CuPy when available).
    median_bin : int
        Run the XY median on a grid binned by this factor (see
        :func:`_median_xy_filter`). ``1`` filters the full grid.
    subtract_noise_floor : bool
        Subtract the noise floor from the signal before the estimate. Li 2020
        and Liu 2019 only cut the A-line at the noise floor; Vermeer 2014
        subtracts a measured noise profile. ``False`` cuts without subtracting.

    References
    ----------
    Li et al., Biomed. Opt. Express 11, 672 (2020).
    """
    vol = _median_xy_filter(vol, k, median_bin, use_gpu=use_gpu)
    if mask is None:
        mask = _auto_tissue_mask(vol, zshift)
    mask = mask.astype(bool)

    if noise_floor is None:
        # Empty mosaic padding is exactly zero and is not a measurement. Left
        # in, it drags the median down, to zero once it is half of the voxels
        # outside the tissue, and then nothing is subtracted or truncated.
        out_of_mask = vol[~mask]
        out_of_mask = out_of_mask[out_of_mask > 0]
        deepest = vol[:, :, -5:]
        deepest = deepest[deepest > 0]
        if out_of_mask.size > 100:
            noise_floor = float(np.median(out_of_mask))
        elif deepest.size:
            noise_floor = float(np.median(deepest))
        else:
            noise_floor = 0.0
    vol_clean = vol.astype(np.float32)
    if subtract_noise_floor:
        vol_clean = np.clip(vol_clean - noise_floor, 0.0, None)

    # Per-A-line truncation depth where SNR drops below threshold.
    snr_signal_threshold = max(noise_floor, 1e-6) * (10.0 ** (snr_threshold_db / 10.0))
    above = vol > snr_signal_threshold
    nz = vol.shape[2]
    z_idx = np.arange(nz)
    z_above = np.where(above, z_idx[None, None, :], -1)
    cutoff = z_above.max(axis=2)
    cutoff = np.where(cutoff >= 0, cutoff, nz - 1)
    z_grid = np.broadcast_to(z_idx[None, None, :], vol.shape)
    trunc_mask = (z_grid <= cutoff[..., None]) & mask

    n_tail = int(np.clip(tail_fit_voxels, 2, nz))
    dz_m = res * 1e-6
    i_end, slope, fitted = _tail_at_aline_end(vol_clean, trunc_mask, n_tail)
    mu_E_per_m = np.maximum(-slope / (2.0 * dz_m), 0.0)
    C = _exact_tail_C(i_end, mu_E_per_m, dz_m, valid=fitted)
    attn = get_attenuation_vermeer2013(vol_clean, dz=dz_m, mask=trunc_mask, C=C.astype(np.float32), use_gpu=use_gpu)
    # Below the cut the Vermeer core carries the last estimate down the A-line,
    # which is the decay rate the tail constant was built from. The tissue does
    # not stop attenuating where its signal sinks into the noise, so that value
    # is kept down to the end of the tissue mask, not zeroed at the cut.
    return _finalize_attenuation(attn, mask, fill_holes)


def tissue_footprint_mask(mask: np.ndarray) -> np.ndarray:
    """Extend a tissue mask ``(X, Y, Z)`` down every A-line that holds tissue.

    A 3D tissue mask drops the deep, dim voxels of strongly attenuating
    tissue. :func:`smooth_optical_depth_xy` leaves voxels outside its mask
    untouched, so with the 3D mask those voxels keep an unsmoothed optical
    depth next to smoothed neighbours: gain patches with hard edges. The
    footprint covers them.
    """
    mask = mask.astype(bool)
    return np.broadcast_to(mask.any(axis=2, keepdims=True), mask.shape)


def smooth_optical_depth_xy(optical_depth: np.ndarray, mask: np.ndarray, sigma: float, use_gpu: bool = False) -> np.ndarray:
    """Smooth a cumulative optical depth across A-lines, inside the tissue mask.

    Per-A-line truncation and hole filling leave flat attenuation patches with
    hard walls. Summed along depth they become gain blocks one slab tall. A
    masked Gaussian per depth plane softens those walls; depth is untouched.

    Parameters
    ----------
    optical_depth : ndarray
        Optical depth in ``(X, Y, Z)`` order.
    mask : ndarray
        Tissue mask, same shape. Voxels outside keep their value.
    sigma : float
        Lateral Gaussian sigma in voxels. ``<= 0`` is a no-op.
    use_gpu : bool
        Filter on the GPU when CuPy is available (same result to float
        precision).
    """
    if sigma <= 0:
        return optical_depth
    if use_gpu and GPU_AVAILABLE:
        import cupy as cp
        from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter

        mask_g = cp.asarray(mask.astype(bool, copy=False))
        od_g = cp.asarray(optical_depth, dtype=cp.float32)
        sig = (sigma, sigma, 0.0)
        weight_g = gpu_gaussian_filter(mask_g.astype(cp.float32), sig)
        smoothed_g = gpu_gaussian_filter(cp.where(mask_g, od_g, 0.0), sig)
        smoothed_g /= cp.maximum(weight_g, 1e-3)
        out = cp.asnumpy(cp.where(mask_g, smoothed_g, od_g)).astype(optical_depth.dtype, copy=False)
        del mask_g, od_g, weight_g, smoothed_g
        _release_gpu_memory()
        return out
    mask = mask.astype(bool, copy=False)
    # Plane by plane on contiguous 2-D arrays, which is the same filter as one
    # 3-D call with no smoothing along depth and about twice as fast. A mask
    # that does not change with depth (a footprint) needs its weight once.
    same_mask = mask.strides[2] == 0 or bool((mask == mask[:, :, :1]).all())
    weight = gaussian_filter(mask[:, :, 0].astype(np.float32), sigma) if same_mask else None
    out = np.empty(optical_depth.shape, dtype=optical_depth.dtype)
    for z in range(optical_depth.shape[2]):
        plane_mask = mask[:, :, 0] if same_mask else np.ascontiguousarray(mask[:, :, z])
        plane = np.ascontiguousarray(optical_depth[:, :, z])
        w = weight if weight is not None else gaussian_filter(plane_mask.astype(np.float32), sigma)
        smoothed = gaussian_filter(np.where(plane_mask, plane, 0.0).astype(np.float32), sigma)
        smoothed /= np.maximum(w, 1e-3)
        out[:, :, z] = np.where(plane_mask, smoothed, plane)
    return out


def calibrate_compensation_strength(
    vol: np.ndarray,
    optical_depth: np.ndarray,
    mask: np.ndarray,
    n_planes: int = 20,
    skip_planes: int = 2,
    max_strength: float = 1.0,
    min_bias: float = 0.0,
    erosion_px: int = 15,
    min_alines: int = 2000,
    max_alines: int = 200_000,
) -> float | None:
    """Return the strength at which the tissue level of a slab is flat over depth.

    The compensation divides the signal by ``exp(-2 * strength * OD)``. A fixed
    strength fits one acquisition geometry only. The depth-resolved estimate
    reads the place where an A-line runs out of signal as the end of the
    tissue, so a slab imaged with the focus deep below the surface gets a much
    larger optical depth than one imaged with the focus near the surface, and
    the same strength over-corrects the first and under-corrects the second.
    White matter is over-corrected more than grey matter, which shows as a
    contrast step between slabs in the stack.

    The optical depth keeps its lateral pattern here and only its scale is
    fitted: the strength is the one at which the median tissue level has no
    trend over the planes that go into the stack. The slope of the log median
    rises with the strength, so the root is found by bisection.

    Parameters
    ----------
    vol : ndarray
        Volume in ``(X, Y, Z)`` order, before compensation.
    optical_depth : ndarray
        Cumulative optical depth, same shape.
    mask : ndarray
        Tissue mask, same shape.
    n_planes : int
        Last plane below the surface that counts. Set it to the number of
        planes a slab contributes to the stack.
    skip_planes : int
        First plane that counts; the planes above it hold the surface.
    max_strength : float
        Upper bound of the search.
    min_bias : float
        Floor on the bias, as applied by the compensation.
    erosion_px : int
        The level is taken this far inside the tissue footprint.
    min_alines : int
        Fewer tissue A-lines than this and no strength is returned.
    max_alines : int
        The level is taken on at most this many A-lines, evenly spread.

    Returns
    -------
    float or None
        The strength in ``[0, max_strength]``, or ``None`` when the slab holds
        too little tissue to tell.
    """
    planes = np.arange(skip_planes, min(n_planes + 1, vol.shape[2]))
    if planes.size < 4:
        return None
    mask = mask.astype(bool)
    footprint = mask[:, :, planes].mean(axis=2) > 0.9
    if erosion_px > 0:
        footprint = binary_erosion(footprint, iterations=erosion_px)
    idx = np.flatnonzero(footprint)
    if idx.size < min_alines:
        return None
    if idx.size > max_alines:
        idx = idx[:: int(np.ceil(idx.size / max_alines))]
    rows, cols = np.unravel_index(idx, footprint.shape)
    signal = np.asarray(vol[rows, cols][:, planes], dtype=np.float64)
    depth = np.asarray(optical_depth[rows, cols][:, planes], dtype=np.float64)

    def slope(strength: float) -> float:
        bias = np.maximum(np.exp(-2.0 * strength * depth), max(min_bias, 1e-6))
        level = np.maximum(np.median(signal / bias, axis=0), 1e-12)
        return float(np.polyfit(planes, np.log(level), 1)[0])

    if slope(0.0) >= 0.0:
        return 0.0
    if slope(max_strength) < 0.0:
        return float(max_strength)
    lo, hi = 0.0, float(max_strength)
    for _ in range(12):
        mid = 0.5 * (lo + hi)
        if slope(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def get_attenuation_faber2004(
    vol: np.ndarray, mask: np.ndarray | None = None, dz: float = 6.5e-6, N: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Faber 2004 single-scattering attenuation estimate (one mu_t per A-line).

    Fits the confocal-PSF / Beer-Lambert model
    ``I(z) ~ 1/(1 + ((z-z0)/zR)**2) * exp(-2 mu_t z)`` per A-line. See
    ``docs/ATTENUATION_METHODS.md`` for the model and its limitations.

    Parameters
    ----------
    vol : ndarray
        OCT volume in ``(X, Y, Z)`` order.
    mask : ndarray, optional
        Per-voxel mask. Only masked samples are passed to the optimizer;
        the depth axis is rebuilt from ``z[mask_aline]``.
    dz : float
        Axial pixel size in metres.
    N : int
        XY uniform-filter window before fitting (``0`` skips).

    Returns
    -------
    attn, r_length, z_focus : ndarray
        Per-A-line ``mu_t`` (1/m), Rayleigh length ``zR`` (m), and focal
        depth ``z0`` (m).

    Notes
    -----
    Hard-coded for a 4x objective (``w0 = 4.88 um``, ``lambda0 = 1030 nm``,
    ``n = 1.33``); adjust the constants in the body for other setups.

    References
    ----------
    Faber et al., Opt. Express 12, 4353 (2004).
    """
    # Average the A-line together in the XY plane
    if N > 0:
        vol = uniform_filter(vol, size=(N, N, 0))

    # Computing the confocal psf parameters
    alpha = 2  # 1 : specular scattering; 2 : diffuse scattering
    n = 1.33  # Index of refraction
    w0 = 4.88e-6  # micron : Ligth beam width at the focal plane (need to validate)
    l0 = 1.030e-6  # micron : Ligth central wavelenth
    zr = np.pi * alpha * n * w0**2.0 / l0  # Apparent Rayleigh length (micron)

    # Defining the objective function for the minimization
    def f(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        return np.sum((y - oct_signal_faber2004_model(z, mu_t=x[2], zR=x[1], z0=x[0])) ** 2.0)

    # Loop over all A-lines and computing attenuation
    attn = np.zeros((vol.shape[0], vol.shape[1]))
    r_length = np.zeros((vol.shape[0], vol.shape[1]))
    z_focus = np.zeros((vol.shape[0], vol.shape[1]))
    z = np.arange(0.0, dz * vol.shape[2], dz)
    for x in range(vol.shape[0]):
        for y in range(vol.shape[1]):
            mask_aline = mask[x, y, :] if mask is not None else np.ones((vol.shape[2],)).astype(bool)

            if np.any(mask_aline):
                p0 = [0.0, 100.0, 0.001]
                zp = np.where(mask_aline)[0][0]
                data = vol[x, y, :][mask_aline]
                data /= 1.0 * data.max()
                # Use the masked depth samples so `data` and `z_aligned`
                # have the same length even when the mask is non-contiguous.
                z_aligned = z[mask_aline] - z[zp]
                p_opt = minimize(
                    f,
                    p0,
                    args=(data, z_aligned),
                    bounds=((None, None), (zr / 2.0, 2.0 * zr), (0.0, None)),
                )
                if p_opt.success:
                    attn[x, y] = p_opt.x[2]
                    r_length[x, y] = p_opt.x[1]
                    z_focus[x, y] = p_opt.x[0] + z[zp]
                else:
                    print(("No convergence for (x,y)=", x, y))

    return attn, r_length, z_focus


# Modele du signal utilisant la PSF confocale et single-scattering photons


def oct_signal_faber2004_model(z: np.ndarray, mu_t: float = 1.0, zR: float = 200.0, z0: float = 100.0) -> np.ndarray:
    """Model the oct signal using a single-scattered photons and the confocal PSF.

    Parameters
    ----------
    z : (N,) ndarray
        Depth Position along an A-line at which the signal must be computed (in micron)
    mu_t : float or (N,) ndarray
        Attenuation coefficient (either a single value for the whole column or N values for each position)
    zR : float
        Apparent Rayleigh length (in micron)
    z0 : float
        Focal plane depth (in micron)

    Returns
    -------
    ndarray
        OCT backscattering signal estimated at each z positions

    Notes
    -----
    - This is based on the single-scattering photon model from Faber2004.

    References
    ----------
    - https://www.osapublishing.org/oe/abstract.cfm?uri=oe-12-19-4353

    """
    psf = 1.0 / (((z - z0) / float(zR)) ** 2.0 + 1.0)
    iz = np.sqrt(psf * np.exp(-2 * mu_t * z))  # Should I fit the sqrt or not
    return iz


def _aline_fit(data: np.ndarray) -> float:
    """Aline fit to extract the attenuation coefficient."""

    # Defining the attenuation model (biexponential) with x=[A, mu_t], y=data, z=depths
    def f_attn(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
        return np.sum((y - x[0] * np.exp(-2 * x[1] * z)) ** 2.0)

    z = np.linspace(0, len(data), len(data))
    aline = np.array(data)
    p0 = [1.0, 0.001]  # Initial condition
    popt = minimize(f_attn, p0, args=(aline, z), bounds=((0, None), (0, None)))
    return popt.x[1]


def split_aline(data: np.ndarray, mask: np.ndarray) -> tuple[list, list]:
    """Split an A-line into contiguous masked segments."""
    data_list = []
    z_list = []
    this_aline = []
    this_z = []
    for elem, m, z in zip(data, mask, list(range(len(data))), strict=False):
        if m:
            this_aline.append(elem)
            this_z.append(z)
        else:
            if len(this_aline) > 0:
                data_list.append(this_aline)
                this_aline = []
                z_list.append(this_z)
                this_z = []
    if len(this_aline) > 0:
        data_list.append(this_aline)
        z_list.append(this_z)

    return data_list, z_list


def _split_alines_worker(param: tuple) -> tuple[list, list]:
    return split_aline(param[0], param[1])


def get_aline_attenuation(vol: np.ndarray, k: int = 1, mask: np.ndarray | None = None) -> np.ndarray:
    """Compute the attenuation coefficient for each A-lines.

    Parameters
    ----------
    vol : ndarray
        Volume to analyze of size NxMxL

    k : int
        Number of points to compute

    mask : ndarray
        Tissue mask to limit the region where the fit is done.

    Returns
    -------
    ndarray
        Estimated attenuation of size NxMxk

    """
    # Computing the shape of this volume
    nx, ny, nz = vol.shape
    z = np.arange(nz)
    z_list = np.array_split(z, k)
    attn_vol = np.zeros((nx, ny, k))

    for z, ik in zip(z_list, list(range(k)), strict=False):
        # Selecting a subsample
        this_vol = vol[:, :, z[0] : z[-1]]
        this_mask = mask[:, :, z[0] : z[-1]] if mask is not None else None

        # Transforming this volume into a list
        a_lines = np.split(this_vol.flatten(), nx * ny)
        if mask is not None and this_mask is not None:
            mask_a_lines = np.split(this_mask.flatten(), nx * ny)
            for A, M, ii in zip(a_lines, mask_a_lines, list(range(nx * ny)), strict=False):
                a_lines[ii] = A[M]

        # Process each Alines in parallel
        nproc = get_available_cpus()
        p = multiprocessing.Pool(nproc, initializer=worker_initializer)
        result = p.map(_aline_fit, a_lines)
        p.close()
        p.join()

        # Reshaping the output
        attn_vol[:, :, ik] = np.reshape(result, (nx, ny))
    # Removing background
    attn_vol[vol.mean(axis=2) == 0] = 0

    return np.squeeze(attn_vol)


def get_gradient_attenuation(
    vol: np.ndarray,
    mask: np.ndarray | None = None,
    return_mask: bool = False,
    low_thresh: float = 0.0,
    fill_holes: bool = False,
    sz: int = 3,
    sxy: int = 0,
    res: float = 1.0,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Compute the attenuation coefficient using the log-gradient method."""
    vol_l = np.copy(vol)
    vol_l[vol > 0] = np.log(vol[vol > 0])
    attn = -0.5 * np.gradient(vol_l, res, axis=2)

    # Removing 0 values and masked values
    attn[vol == 0] = 0
    if mask is not None:
        attn[~mask] = 0

    # Removing negative attenuation values
    mask_attn = attn < 0
    attn[mask_attn] = 0

    # Removing also small attenuation values
    small_mask = attn < np.percentile(attn[attn > 0], low_thresh)
    mask_attn[small_mask] = True
    attn[mask_attn] = 0

    # Filling holes using morphological reconstruction.
    if fill_holes:
        vol_itk = sitk.GetImageFromArray(attn)
        cfilter = sitk.GrayscaleMorphologicalClosingImageFilter()
        cfilter.SetKernelType(sitk.sitkBall)
        cfilter.SetKernelRadius((sz, sxy, sxy))

        output_itk = cfilter.Execute(vol_itk)
        attn = sitk.GetArrayFromImage(output_itk)

    if return_mask:
        return attn, mask_attn
    else:
        return attn


def get_interface_mask(
    vol: np.ndarray, s: int = 0, mask_tissue: bool = True, mask_water_tissue_interface: bool = True
) -> np.ndarray:
    """Compute a mask excluding tissue interfaces and boundaries."""
    nx, ny, nz = vol.shape
    mask = np.ones_like(vol).astype(bool)

    # Get tissue mask
    if mask_tissue:
        from dipy.segment.mask import median_otsu  # heavy import, only needed here

        tissue_mask = binary_fill_holes(median_otsu(eqhist(vol.mean(axis=2)), median_radius=5.0)[1])
        tissue_mask = np.tile(np.reshape(tissue_mask, (nx, ny, 1)), (1, 1, nz))
        mask *= tissue_mask

    # Get water/tissue 3D interface mask
    if mask_water_tissue_interface:
        # Computing the gradient
        vol_f = median_filter(vol, 5)
        gradient = np.gradient(vol_f)
        gm = gradient[0] ** 2.0 + gradient[1] ** 2.0 + gradient[2] ** 2.0
        gm = normalize(gm, high_thresh=99.5)

        # Thresholding the gradient
        thresh = threshold_li(gm)
        interfaces_g = gm >= thresh

        # Converting this into a water/tissue interface depth
        depths = np.zeros((nx, ny))
        for x, y in itertools.product(list(range(nx)), list(range(ny))):
            idx = np.where(interfaces_g[x, y, :])
            if len(idx[0]) > 0:
                depths[x, y] = idx[0][0]

        water_tissue_mask = mask_under_interface(vol, depths, return_mask=True)
        mask *= water_tissue_mask

    # Smoothing the volume
    if s > 0:
        vol = gaussian_filter(vol, sigma=(s, s, s))

    # Get tissue interface boundaries using a Canny filter
    vol_itk = sitk.GetImageFromArray(vol)
    canny_filter = sitk.CannyEdgeDetectionImageFilter()
    edges = sitk.GetArrayFromImage(canny_filter.Execute(vol_itk)).astype(bool)
    mask *= ~edges

    return mask


def find_interface_from_gradient(vol: np.ndarray, f: float = 0.005, remove_smooth: bool = False) -> np.ndarray:
    """Find the tissue-water interface depth map using gradient magnitude."""
    nx, ny = vol.shape[0:2]
    k = int(np.round(f * 0.5 * (nx + ny)))

    # Computing the gradient
    vol_f = gaussian_filter(vol, k)

    gradient = np.gradient(vol_f)
    gm = gradient[0] ** 2.0 + gradient[1] ** 2.0 + gradient[2] ** 2.0
    gm = normalize(gm, high_thresh=99.5)

    # Thresholding the gradient
    thresh = threshold_li(gm)
    interfaces_g = gm >= thresh

    # Converting this into a water/tissue interface depth
    depths = np.argmax(interfaces_g, axis=2)

    if remove_smooth:
        depths += k + 1

    return depths


def get_heterogeneous_attenuation(
    vol: np.ndarray, mask: np.ndarray | None = None, fill_holes: bool = False
) -> np.ndarray:  # TODO: adapt multiproc to available proc given by mpi4py
    """Compute heterogeneous attenuation coefficients for each A-line segment."""
    nx, ny, nz = vol.shape
    nproc = get_available_cpus()
    if mask is None:  # Compute the mask
        mask = get_interface_mask(vol)

    # Split the volume into Alines and Alines portions.
    print(f"Splitting volume into Alines portions (using {nproc} processors)")
    a_lines = np.split(vol.flatten(), nx * ny)
    a_lines_mask = np.split(mask.flatten(), nx * ny)
    a_lines_to_split = list(zip(a_lines, a_lines_mask, strict=False))
    n_alines = len(a_lines)

    # Process each Alines in parallel
    p = multiprocessing.Pool(nproc, initializer=worker_initializer)
    result = p.map(_split_alines_worker, a_lines_to_split)
    p.close()
    p.join()

    # Debug
    print(("Number of Alines : ", len(a_lines_to_split)))
    p_count = 0
    for portion in result:
        p_count += len(portion[0])
    print(("Number of Alines portions : ", p_count))

    # Compute the attenuation for each aline portions
    print(f"Computing attenuation for each Aline portion (using {nproc} processors)")
    aline_portions = []
    z_portions = []
    portion_idx = []
    for foo, idx in zip(result, list(range(n_alines)), strict=False):
        aline_portions.extend(foo[0])
        z_portions.extend(foo[1])
        portion_idx.extend([idx] * len(foo[0]))

    p = multiprocessing.Pool(nproc, initializer=worker_initializer)
    result = p.map(_aline_fit, aline_portions)
    p.close()
    p.join()

    # Reshape attenuation as an Aline list  # TODO: Parallelize this loop.
    print("Reshape attenuation as an Aline list")
    aline_attn = [np.zeros((nz,)) for i in range(n_alines)]
    for idx, z, mu in zip(portion_idx, z_portions, result, strict=False):
        aline_attn[idx][z] = mu

    # Combine local attenuations into a single volume
    attn_vol = np.reshape(aline_attn, (nx, ny, nz))

    # Fill attenuation holes (using 1d-linear interpolation) # TODO: User inverve distance weighted interpolation instead ?
    if fill_holes:
        print("Filling attenuation holes using 1D-linear interpolation")
        attn_fill = np.zeros_like(attn_vol)
        for x in range(nx):
            for y in range(ny):
                idx = attn_vol[x, y, :] > 0
                this_z = np.array(np.where(idx)[0])
                this_mu = attn_vol[x, y, :]
                this_mu = this_mu[idx]
                if np.sum(idx) == 1:
                    attn_fill[x, y, :] = attn_vol[x, y, :]
                elif np.sum(idx) > 1:
                    f = interp1d(this_z, this_mu, kind="linear", bounds_error=False, fill_value=0)
                    new_z = np.arange(nz)
                    new_mu = f(new_z)
                    attn_fill[x, y, :] = new_mu  # Debug, should replace the original volume

        return attn_fill
    else:
        return attn_vol


@overload
def get_flat_agarose_profile(vol: np.ndarray, return_mask_and_profile: Literal[False] = ...) -> np.ndarray: ...
@overload
def get_flat_agarose_profile(
    vol: np.ndarray, return_mask_and_profile: Literal[True]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


def get_flat_agarose_profile(
    vol: np.ndarray, return_mask_and_profile: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Estimate and remove the flat agarose intensity profile from a volume."""
    nx, ny, nz = vol.shape

    # Get agarose mask for this slice and its intensity profile
    from dipy.segment.mask import median_otsu  # heavy import, only needed here

    tissue_mask = binary_fill_holes(median_otsu(eqhist(vol.mean(axis=2)), median_radius=5.0)[1])
    tissue_mask = np.tile(np.reshape(tissue_mask, (nx, ny, 1)), (1, 1, nz))
    mask = (~tissue_mask).astype(bool)

    # Computing intensity profile for the agarose Alines
    i_profile = np.zeros((nz,))
    n_alines = np.sum(mask.max(axis=2))
    for x in range(nx):
        for y in range(ny):
            this_mask = mask[x, y, :]
            a_line = vol[x, y, :]
            if np.any(this_mask):
                i_profile += a_line / float(n_alines)

    # Correction profile
    agarose_norm = np.tile(np.reshape(i_profile, (1, 1, nz)), (nx, ny, 1))
    vol_p = vol / agarose_norm.astype(float)

    if return_mask_and_profile:
        return vol_p, mask, i_profile
    else:
        return vol_p


def get_signal_from_attenuation(
    attn: np.ndarray, i0: np.ndarray | None = None, nz: int = 120, mask: np.ndarray | None = None, res: float = 1.0
) -> np.ndarray:
    """Estimate the signal from the 2D A-Line attenuation map.

    Parameters
    ----------
    attn : ndarray
        2D attenuation coefficient map for each A-line
    i0 : ndarray
        2D map of the top slice signal (optional)
    nz : int
        Number of Aline points
    mask : ndarray
        Data mask (to limit where the oct signal is simulated)
    res : float
        Axial resolution in micron / pixel

    Returns
    -------
    ndarray
        Estimated OCT signal

    """
    nx, ny = attn.shape
    attn_vol = np.zeros((nx, ny, nz))

    def f_attn(x: float, z: np.ndarray) -> np.ndarray:
        return np.exp(-2 * x * z)

    for ix, iy in itertools.product(list(range(nx)), list(range(ny))):
        this_mask = mask[ix, iy, :].astype(bool) if mask is not None else np.ones((nz,), dtype=bool)

        z0 = np.where(this_mask)
        z0 = z0[0][0] if len(z0[0]) > 0 else 0

        A = i0[ix, iy] if i0 is not None else 1

        if np.any(this_mask):
            this_mu = attn[ix, iy]
            z = np.linspace(0, nz * res, nz) - z0 * res
            sim_profile = A * f_attn(this_mu, z)
            attn_vol[ix, iy, this_mask] = sim_profile[this_mask]

    return attn_vol
