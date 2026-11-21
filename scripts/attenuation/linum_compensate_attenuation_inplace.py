#!/usr/bin/env python
r"""Compensate per-slice OCT depth attenuation in a single pass.

Combines the three legacy steps (compute attenuation -> integrate to bias
field -> divide) into one in-memory operation, avoiding intermediate
ome.zarr files. Produces a corrected volume of the same shape as the
input.

Pipeline
--------
1. Build a tissue mask: a 3D threshold via :func:`compute_tissue_mask`
   (Otsu + closing), or with ``--tissue_mask column`` the XY footprint of the
   tissue kept over the whole depth of the slab.
2. Estimate the local effective attenuation with one of four
   depth-resolved methods (``--method``):

   * ``smith`` (default) -- Smith 2015 extension of Vermeer 2014 with a
     log-gradient :math:`\hat\mu_E` for the finite-range constant.
   * ``vermeer`` -- the bare Vermeer 2014 estimator (``C = 0``).
   * ``liu`` -- Liu 2019 with the exact-form regularization
     ``C = I[imax] / (exp(2 mu_E dz) - 1)`` and curve-fit ``mu_E``.
   * ``li`` -- Li 2020: noise-floor subtraction + per-A-line SNR
     truncation, then Liu-style regularization on the cleaned signal.

   See :mod:`linumpy.intensity.attenuation` for full references.
3. Integrate cumulatively along Z to obtain the round-trip optical depth,
   convert to a multiplicative bias field ``bias = exp(-2 * strength * OD)``.
   With ``--calibrate_strength`` the strength is fitted on the slab itself:
   the value at which the tissue level is flat over the planes that go into
   the stack.
4. Clamp the bias from below by ``--min_bias`` to prevent runaway gain in
   deep, low-signal voxels and divide the input volume by the clamped
   bias.

Inputs are expected to be per-slice OME-Zarr volumes already cropped to
the tissue interface (i.e. the output of ``linum_crop_3d_mosaic_below_interface``).
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401
from linumpy.config.threads import configure_all_libraries

import argparse
from pathlib import Path

import dask.array as da
import numpy as np
from scipy.integrate import cumulative_trapezoid

from linumpy.gpu import GPU_AVAILABLE, to_cpu
from linumpy.intensity.attenuation import (
    calibrate_compensation_strength,
    get_attenuation_li2020,
    get_attenuation_liu2019,
    get_attenuation_smith2015,
    get_attenuation_vermeer2013,
    smooth_optical_depth_xy,
    tissue_column_mask,
    tissue_footprint_mask,
)
from linumpy.intensity.bias_field import compute_tissue_mask
from linumpy.io.zarr import read_omezarr, save_omezarr

METHODS = ("smith", "vermeer", "liu", "li")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input", type=Path, help="Input per-slice volume (.ome.zarr)")
    p.add_argument("output", type=Path, help="Compensated volume (.ome.zarr)")
    p.add_argument(
        "--min_bias",
        type=float,
        default=0.05,
        help="Floor applied to the bias field before division.\n"
        "Caps the maximum gain at 1/min_bias and prevents amplification\n"
        "of noise in deep, low-signal voxels. [%(default)s]",
    )
    p.add_argument(
        "--mask_smoothing_sigma",
        type=float,
        default=2.0,
        help="Gaussian sigma (XY voxels) for the Otsu tissue mask. [%(default)s]",
    )
    p.add_argument(
        "--k",
        type=int,
        default=10,
        help="Radius (voxels) of the XY median filter applied before the attenuation\n"
        "estimation; the window is 2k+1 voxels wide. 0 disables denoising. Larger\n"
        "values stabilise the per-A-line fit at the cost of lateral resolution.\n"
        "[%(default)s]",
    )
    p.add_argument(
        "--median_bin",
        type=int,
        default=2,
        help="Run the median of the ``li`` method on a grid binned by this factor in XY,\n"
        "with the radius scaled to cover the same area. 2 is about ten times faster\n"
        "than the full grid and changes the gain by under 1 %%. 1 filters the full\n"
        "grid. [%(default)s]",
    )
    p.add_argument(
        "--zshift",
        type=int,
        default=3,
        help="Voxels under the water/tissue interface to skip when the estimator has to find\n"
        "the tissue itself. This script always passes its own tissue mask, so the value\n"
        "has no effect here; it is kept so existing commands keep working. [%(default)s]",
    )
    p.add_argument(
        "--strength",
        type=float,
        default=0.3,
        help="Multiplicative scale on the optical-depth correction (0..1).\n"
        "Vermeer's single-scattering model overestimates effective attenuation\n"
        "in scattering brain tissue because the multiple-scattering signal floor\n"
        "violates the geometric-tail assumption. Values <1 attenuate the\n"
        "correction; the empirical sweet spot for cropped 600 um sub-22 slices\n"
        "is ~0.30 (yields a near-flat depth profile). Set to 1.0 for the\n"
        "textbook formula. [%(default)s]",
    )
    p.add_argument(
        "--method",
        choices=METHODS,
        default="li",
        help="Depth-resolved attenuation estimator. ``li`` (default) adds\n"
        "noise-floor subtraction and SNR-based A-line truncation on top\n"
        "of the Liu 2019 exact-form regularization, and produces the\n"
        "flattest axial profile on real OCT data. ``liu`` is the same\n"
        "without noise handling. ``smith`` reproduces the historical\n"
        "linumpy behaviour. ``vermeer`` is the bare estimator with no\n"
        "regularization. [%(default)s]",
    )
    p.add_argument(
        "--snr_threshold_db",
        type=float,
        default=6.0,
        help="Per-voxel SNR threshold (dB) for A-line truncation in the ``li`` method.\n"
        "Voxels where signal / noise_floor < 10^(snr_threshold_db/10) are excluded\n"
        "from the attenuation fit. Li 2020 and Liu 2019 crop at 5 dB above a system noise\n"
        "floor measured with the sample arm blocked; the floor here is the level outside\n"
        "the tissue, so the value is a tuning choice. [%(default)s]",
    )
    p.add_argument(
        "--lateral_sigma",
        type=float,
        default=0.0,
        help="XY Gaussian sigma (voxels) on the optical depth inside the tissue mask.\n"
        "Softens the hard-walled gain patches that per-A-line truncation leaves,\n"
        "which stack into slab-tall blocks. 0 disables. [%(default)s]",
    )
    p.add_argument(
        "--lateral_footprint",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Smooth the optical depth over the whole XY footprint of the tissue instead of\n"
        "the 3D tissue mask. Deep voxels of strongly attenuating tissue are dim and fall\n"
        "outside the 3D mask; they then keep their unsmoothed value and show up as bright\n"
        "patches with hard edges. Only used with --lateral_sigma. [%(default)s]",
    )
    p.add_argument(
        "--tissue_mask",
        choices=["voxel", "column"],
        default="voxel",
        help="How the tissue is told from the agarose.\n"
        "  voxel  : 3D threshold on the slab (Otsu + closing).\n"
        "  column : a column is tissue over its whole depth or not at all, judged on\n"
        "           its brightest planes. A 3D threshold keeps only the bright top of a\n"
        "           slab whose signal falls quickly with depth: the compensation then\n"
        "           acts inside that part alone, with a gain step along its outline,\n"
        "           and --calibrate_strength finds no tissue to fit on. [%(default)s]",
    )
    p.add_argument(
        "--calibrate_strength",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fit the strength on each slab instead of using --strength: the value at\n"
        "which the median tissue level is flat over the planes below the surface that\n"
        "go into the stack. A fixed strength fits one acquisition geometry. A slab\n"
        "imaged with the focus deep below the surface gets a larger optical depth\n"
        "than one imaged with the focus near the surface, and the same strength then\n"
        "over-corrects it, white matter most, which leaves a contrast step between\n"
        "slabs. --strength is used for a slab with too little tissue. [%(default)s]",
    )
    p.add_argument(
        "--calibrate_planes",
        type=int,
        default=20,
        help="Last plane below the surface that --calibrate_strength flattens the tissue\n"
        "level over. Set it to the number of planes a slab contributes to the stack\n"
        "(the slicing interval in voxels). [%(default)s]",
    )
    p.add_argument(
        "--calibrate_max_strength",
        type=float,
        default=2.0,
        help="Upper bound of the strength --calibrate_strength may pick. 1.0 is the published\n"
        "formula; slabs whose optical depth is underestimated need a little more (up\n"
        "to about 1.5 on the brains this was tuned on). The gain stays capped by\n"
        "--min_bias. [%(default)s]",
    )
    p.add_argument(
        "--use_gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run the tissue mask, the XY median of the ``li`` method, the estimator core\n"
        "and the lateral smoothing on the GPU when one is available. [%(default)s]",
    )
    p.add_argument("--n_levels", type=int, default=0, help="Pyramid levels in the output. [%(default)s]")
    return p


def main() -> None:
    """Run the in-place attenuation compensation."""
    configure_all_libraries()
    args = _build_arg_parser().parse_args()

    vol_zarr, res = read_omezarr(args.input, level=0)
    chunks = vol_zarr.chunks
    vol = np.asarray(vol_zarr).astype(np.float32)  # (Z, Y, X)

    # Tissue mask (Z, Y, X)
    use_gpu = args.use_gpu and GPU_AVAILABLE
    if args.tissue_mask == "column":
        columns = tissue_column_mask(vol, smoothing_sigma=args.mask_smoothing_sigma)
        mask_zyx = np.broadcast_to(columns[None], vol.shape).copy()
        print(f"Tissue on {100 * columns.mean():.1f} % of the columns")
    else:
        mask_zyx = np.asarray(to_cpu(compute_tissue_mask(vol, smoothing_sigma=args.mask_smoothing_sigma, use_gpu=use_gpu)))
    if use_gpu and args.tissue_mask == "voxel":
        # The mask leaves several copies of the slab in CuPy's pool.
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()

    # Vermeer expects (X, Y, Z) ordering.
    vol_xyz = np.moveaxis(vol, (0, 1, 2), (2, 1, 0))
    mask_xyz = np.moveaxis(mask_zyx, (0, 1, 2), (2, 1, 0)).astype(bool)

    res_axial_microns = float(res[0]) * 1000.0  # res is in mm

    # Depth-resolved attenuation in 1/cm.
    if args.method == "smith":
        attn_cm = get_attenuation_smith2015(
            vol_xyz,
            mask=mask_xyz,
            k=args.k,
            res=res_axial_microns,
            fill_holes=True,
            zshift=args.zshift,
        )
    elif args.method == "liu":
        attn_cm = get_attenuation_liu2019(
            vol_xyz,
            mask=mask_xyz,
            k=args.k,
            res=res_axial_microns,
            fill_holes=True,
            zshift=args.zshift,
        )
    elif args.method == "li":
        # No hole filling here: the estimate has a value on every voxel of the
        # tissue mask, and filling changed about 1 % of them by under 0.1 /cm.
        attn_cm = get_attenuation_li2020(
            vol_xyz,
            mask=mask_xyz,
            k=args.k,
            res=res_axial_microns,
            fill_holes=False,
            zshift=args.zshift,
            snr_threshold_db=args.snr_threshold_db,
            median_bin=args.median_bin,
            use_gpu=use_gpu,
        )
    else:  # vermeer
        # The bare Vermeer estimator returns 1/cm directly; reuse smith's
        # XY-median pre-filter for fairness across methods.
        if args.k > 0:
            import SimpleITK as sitk

            vol_xyz = sitk.GetArrayFromImage(sitk.Median(sitk.GetImageFromArray(vol_xyz), (0, args.k, args.k)))
        attn_cm = get_attenuation_vermeer2013(
            vol_xyz,
            dz=res_axial_microns * 1e-6,
            mask=mask_xyz,
        )

    # 1/cm -> 1/voxel: cm^-1 * 100 = m^-1; * 1e-6 = um^-1; * res_um = voxel^-1
    attn_per_voxel = attn_cm * 100.0 * 1.0e-6 * res_axial_microns

    # Cumulative round-trip optical depth along the Aline (axis=2 in XYZ).
    optical_depth = cumulative_trapezoid(attn_per_voxel, axis=2, initial=0)
    smooth_mask = mask_xyz
    if args.lateral_footprint:
        smooth_mask = tissue_footprint_mask(mask_xyz)
    optical_depth = smooth_optical_depth_xy(optical_depth, smooth_mask, args.lateral_sigma, use_gpu=use_gpu)
    strength = args.strength
    if args.calibrate_strength:
        fitted = calibrate_compensation_strength(
            vol_xyz,
            optical_depth,
            mask_xyz,
            n_planes=args.calibrate_planes,
            max_strength=args.calibrate_max_strength,
            min_bias=args.min_bias,
        )
        if fitted is None:
            print(f"Too little tissue to calibrate the strength, using --strength {args.strength}")
        else:
            strength = fitted
            print(f"Calibrated strength {strength:.3f} (tissue level flat over planes 2-{args.calibrate_planes})")
    bias_xyz = np.exp(-2.0 * strength * optical_depth).astype(np.float32)
    np.maximum(bias_xyz, args.min_bias, out=bias_xyz)

    # Back to (Z, Y, X) and apply.
    bias_zyx = np.moveaxis(bias_xyz, (0, 1, 2), (2, 1, 0))
    corrected = (vol / bias_zyx).astype(np.float32)

    save_omezarr(da.from_array(corrected), args.output, voxel_size=res, chunks=chunks, n_levels=args.n_levels)


if __name__ == "__main__":
    main()
