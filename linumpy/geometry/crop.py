"""Volume cropping and interface-based masking."""

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import binary_closing, binary_fill_holes
from skimage.filters import threshold_otsu

from linumpy.geometry.interface import find_tissue_interface


def crop_volume(
    vol: np.ndarray, xlim: list[int] | None = None, ylim: list[int] | None = None, zlim: list[int] | None = None
) -> np.ndarray:
    """Crops the given volume according to the range given as input.

    Parameters
    ----------
    vol : ndarray
        Volume to crop
    xlim : (2,) list
        x range to keep
    ylim : (2,) list
        y range to keep
    zlim : (2,) list
        z range to keep

    Returns
    -------
    ndarray
        Cropped volume

    Notes
    -----
    * xlim=[0,-1] means that the whole volume in the x dimension will be returned.

    """
    if zlim is None:
        zlim = [0, -1]
    if ylim is None:
        ylim = [0, -1]
    if xlim is None:
        xlim = [0, -1]
    nx, ny = vol.shape[:2]
    xlim = list(xlim)
    ylim = list(ylim)
    zlim = list(zlim)
    if xlim[1] == -1:
        xlim[1] = nx
    if ylim[1] == -1:
        ylim[1] = ny

    if vol.ndim == 3:
        nz = vol.shape[2]
        if zlim[1] == -1:
            zlim[1] = nz
        return vol[xlim[0] : xlim[1], ylim[0] : ylim[1], zlim[0] : zlim[1]]

    elif vol.ndim == 2:
        return vol[xlim[0] : xlim[1], ylim[0] : ylim[1]]

    return vol


def crop_z0_whole_slice(
    vol: np.ndarray,
    dz: float = 20.0,
    nz: float = 200.0,
    voxdim: tuple[int, int, int] = (1, 1, 1),
    z0: float | None = None,
    verbose: bool = False,
    mask: np.ndarray | None = None,
    return_z0: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """Crop whole slice in the z direction.

    Parameters
    ----------
    vol : ndarray
        Input volume to crop.
    dz : float
        Margin in microns under the interface to crop (to remove cutting deformations).
    nz : float
        Size of the slice to crop in microns.
    voxdim : tuple of int
        Dimension of each voxel in micron/pixel.
    z0 : float, optional
        Interface position in microns. If None, it is detected automatically.
    verbose : bool
        If True, print debug information.
    mask : ndarray, optional
        Tissue mask. If provided, used to compute the tissue interface.
    return_z0 : bool
        If True, also return the interface position z0.

    Returns
    -------
    ndarray
        Cropped array
    """
    if z0 is None:
        # Computing tissue mask
        if mask is not None:
            mask = vol.std(axis=2) > threshold_otsu(vol.std(axis=2))  # Using otsu on A-line intensity std
            mask = binary_fill_holes(binary_closing(mask))  # Closing and filling holes
        else:
            mask = np.ones(vol.shape[0:2], dtype=bool)

        # Computing tissue interface
        interface = find_tissue_interface(vol)

        # Use median interface
        z0 = np.median(interface)
    else:
        # Compute z0 in pixel
        z0 = np.floor(z0 / (1.0 * voxdim[2])).astype(int)

    # Finding crop limits
    zmin = np.floor((z0 * voxdim[2] + dz) / (1.0 * voxdim[2])).astype(int)
    zmax = np.floor((zmin * voxdim[2] + nz) / (1.0 * voxdim[2])).astype(int)

    if verbose:
        print(f"Crop limits are : [{zmin * voxdim[2]:.2f}, {zmax * voxdim[2]:.2f}] microns")
        print(f"Crop limits are : [{zmin}, {zmax}] pixels")

    # Cropping
    if return_z0:
        return crop_volume(vol, zlim=[zmin, zmax]), z0
    else:
        return crop_volume(vol, zlim=[zmin, zmax])


def mask_under_interface(vol: np.ndarray, interface: np.ndarray, return_mask: bool = False) -> np.ndarray:
    """Create a boolean mask for all voxels at or below the interface depth."""
    nx, ny, nz = vol.shape
    mask = np.arange(nz)[None, None, :] >= np.reshape(interface, (nx, ny, 1))
    if return_mask:
        return mask
    else:
        return vol * mask


def apply_interface_correction(
    vol: np.ndarray, interface: np.ndarray
) -> np.ndarray:  # TODO: Test this algorithm to make sure it works well.
    """Apply interface depth correction using linear interpolation.

    Parameters
    ----------
    vol : ndarray
        Volume to fix.
    interface : ndarray
        Tissue interface depth.

    Returns
    -------
    ndarray
        Fixed volume.

    """
    nx, ny, nz = vol.shape
    z_range = np.around(interface.max() - interface.min())
    fixed_vol = np.zeros((nx, ny, nz - z_range), dtype=vol.dtype)

    # Loop over XY
    for x in range(nx):
        for y in range(ny):
            z = interface[x, y]
            real_z = np.linspace(-z, -z + nz, nz)
            new_z = list(range(int(nz - z_range)))
            z_interp = interp1d(real_z, vol[x, y, :], fill_value=0, bounds_error=False, kind="quadratic")
            fixed_vol[x, y, :] = z_interp(new_z)

    return fixed_vol


def fit_surface_tilt(
    vol_zyx: np.ndarray,
    sigma_xy: float = 4.0,
    sigma_z: float = 2.0,
    max_depth_fraction: float = 0.75,
    downsample: int = 2,
    max_residual: float = 3.0,
    min_alines: int = 2000,
    min_inliers: float = 0.75,
) -> tuple[np.ndarray | None, dict[str, float]]:
    """Fit a plane to the tissue surface of a slab.

    The blade does not always cut parallel to the scan plane. The cut face is
    then a tilted plane in the volume, and a crop at one depth starts above
    the tissue on one side of the slab and inside it on the other. This is
    the axial counterpart of the angle between the motor axes in the tile
    position model of Lefebvre et al. 2017: one more angle between two frames
    that are nominally aligned.

    The surface of every tissue A-line is the depth of the steepest rise of
    the smoothed signal. A plane is fitted to those depths, leaving out
    A-lines more than ``max_residual`` planes from it (holes, debris, a
    clipped surface).

    Parameters
    ----------
    vol_zyx : np.ndarray
        Slab ``(Z, Y, X)`` before the crop.
    sigma_xy, sigma_z : float
        Smoothing before the axial derivative, in voxels of the full grid.
    max_depth_fraction : float
        The surface is searched in this first fraction of the depth.
    downsample : int
        XY step the fit runs on.
    max_residual : float
        Distance from the plane, in planes, beyond which an A-line is left out.
    min_alines : int
        Fewest tissue A-lines (on the downsampled grid) for a fit.
    min_inliers : float
        Smallest fraction of tissue A-lines within ``max_residual`` of the plane.
        Small sections at the ends of a brain, slabs whose surface is cut off
        by the top of the scan window and the wedge left by re-levelling the
        sample fall below it. A wedge must stay as scanned: its lower face is
        level, and sliding its A-lines would tilt that face instead.

    Returns
    -------
    np.ndarray or None
        Surface depth of the fitted plane at every ``(Y, X)``, in planes, or
        ``None`` when the slab holds too little tissue or the surface is not
        a plane (fewer than ``min_inliers`` of the A-lines within
        ``max_residual``).
    dict
        ``tilt_y`` and ``tilt_x`` in planes per 1000 voxels, ``span`` of the
        plane over the tissue in planes, ``reference`` (its median over the
        tissue), ``inliers`` (fraction) and ``rms`` of the residual.
    """
    from scipy.ndimage import binary_erosion, gaussian_filter, gaussian_filter1d

    d = max(1, int(downsample))
    vol = np.asarray(vol_zyx[:, ::d, ::d], dtype=np.float32)
    info = {"tilt_y": 0.0, "tilt_x": 0.0, "span": 0.0, "reference": 0.0, "inliers": 0.0, "rms": 0.0}
    n_search = max(3, int(vol.shape[0] * max_depth_fraction))
    smooth = gaussian_filter(vol, (0.0, sigma_xy / d, sigma_xy / d))
    dz = gaussian_filter1d(smooth, sigma=sigma_z, axis=0, order=1, mode="nearest")
    surface = dz[:n_search].argmax(axis=0).astype(np.float64)
    aip = vol.mean(axis=0)
    positive = aip[aip > 0]
    if positive.size < min_alines:
        return None, info
    tissue = binary_erosion(aip > 0.6 * threshold_otsu(positive), iterations=max(1, 20 // d))
    yy, xx = np.nonzero(tissue)
    if yy.size < min_alines:
        return None, info
    depth = surface[yy, xx]
    design = np.c_[yy * d, xx * d, np.ones(yy.size)]
    keep = np.ones(yy.size, dtype=bool)
    coef = np.zeros(3)
    for _ in range(4):
        coef = np.linalg.lstsq(design[keep], depth[keep], rcond=None)[0]
        keep = np.abs(depth - design @ coef) <= max_residual
        if int(keep.sum()) < min_alines // 2:
            return None, info
    fitted = design @ coef
    info.update(
        tilt_y=float(coef[0] * 1000.0),
        tilt_x=float(coef[1] * 1000.0),
        span=float(np.percentile(fitted, 98) - np.percentile(fitted, 2)),
        reference=float(np.median(fitted)),
        inliers=float(keep.mean()),
        rms=float(np.sqrt(np.mean((depth - fitted)[keep] ** 2))),
    )
    if info["inliers"] < min_inliers:
        return None, info
    y_full, x_full = np.mgrid[: vol_zyx.shape[1], : vol_zyx.shape[2]]
    return (coef[0] * y_full + coef[1] * x_full + coef[2]).astype(np.float32), info


def flatten_surface_tilt(vol_zyx: np.ndarray, plane: np.ndarray, reference: float) -> np.ndarray:
    """Slide every A-line along Z so that a tilted surface plane becomes level.

    The A-line at ``(y, x)`` is moved by ``reference - plane[y, x]`` planes,
    with linear interpolation between planes, so the fitted surface ends up at
    depth ``reference`` everywhere. Voxels that move in from outside the
    volume are zero.
    """
    vol = np.asarray(vol_zyx, dtype=np.float32)
    nz = vol.shape[0]
    shift = plane.astype(np.float32) - np.float32(reference)
    out = np.zeros_like(vol)
    base = np.floor(shift).astype(np.int64)
    frac = (shift - base).astype(np.float32)
    for z in range(nz):
        lower = base + z
        upper = lower + 1
        low_ok = (lower >= 0) & (lower < nz)
        up_ok = (upper >= 0) & (upper < nz)
        low = np.take_along_axis(vol, np.clip(lower, 0, nz - 1)[None], axis=0)[0]
        up = np.take_along_axis(vol, np.clip(upper, 0, nz - 1)[None], axis=0)[0]
        out[z] = np.where(low_ok, (1.0 - frac) * low, 0.0) + np.where(up_ok, frac * up, 0.0)
    return out


def crop_below_interface(
    vol_zxy: np.ndarray,
    depth_um: float,
    resolution_um: float,
    sigma_xy: float = 3.0,
    sigma_z: float = 2.0,
    crop_before_interface: bool = False,
    percentile_clip: float | None = None,
    max_interface_depth_fraction: float = 0.5,
) -> tuple[np.ndarray, int]:
    """Crop an OME-Zarr volume to a specified depth below the tissue interface.

    Detects the water/tissue interface using gradient analysis, then crops
    the volume to retain only ``depth_um`` microns below the interface.

    Parameters
    ----------
    vol_zxy : np.ndarray
        Volume with shape (Z, X, Y) as returned by read_omezarr.
    depth_um : float
        Target depth below interface in microns.
    resolution_um : float
        Z resolution in microns per voxel.
    sigma_xy : float
        XY smoothing sigma for interface detection.
    sigma_z : float
        Z smoothing sigma for interface detection.
    crop_before_interface : bool
        If True, also crop the volume above the detected interface.
    percentile_clip : float or None
        If provided, clip values above this percentile before interface detection.
    max_interface_depth_fraction : float
        Passed to :func:`detect_interface_z`.  Restricts interface search to
        the first fraction of the Z axis.  Default 0.5.

    Returns
    -------
    np.ndarray
        Cropped volume (Z', X, Y).
    int
        Detected interface depth in Z voxels.
    """
    from linumpy.geometry.interface import detect_interface_z

    vol_f = np.abs(vol_zxy) if np.iscomplexobj(vol_zxy) else np.asarray(vol_zxy, dtype=np.float32)
    vol_f = np.nan_to_num(vol_f, nan=0.0, posinf=0.0, neginf=0.0)

    vol_xyz = np.transpose(vol_f, (1, 2, 0))

    if percentile_clip is not None:
        vol_xyz = np.clip(vol_xyz, None, np.percentile(vol_xyz, percentile_clip))

    avg_iface = detect_interface_z(
        vol_xyz, sigma_xy=sigma_xy, sigma_z=sigma_z, max_depth_fraction=max_interface_depth_fraction
    )

    depth_px = round(depth_um / resolution_um)
    surface_idx = max(0, min(avg_iface, vol_zxy.shape[0] - 1))
    end_idx = surface_idx + depth_px

    start_idx = surface_idx if crop_before_interface else 0
    vol_crop = vol_f[start_idx:end_idx, :, :]

    return vol_crop, avg_iface
