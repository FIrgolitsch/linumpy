#!/usr/bin/env python3
# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

# -*- coding: utf-8 -*-
"""
GPU-accelerated Z-intensity normalization of a stacked 3D OCT volume.

CPU-equivalent: ``linum_normalize_z_intensity.py``. This script exposes the
same command-line interface and produces numerically equivalent output while
offloading the dominant histogram / percentile / interp work to a CUDA GPU
via CuPy. Falls back to the CPU implementation when no GPU is available.

See the CPU script's ``--help`` for the full list of options and mode
descriptions; this script only adds ``--use_gpu/--no-use_gpu`` and
``--verbose``.
"""

import argparse

import numpy as np

from linumpy.gpu import GPU_AVAILABLE, print_gpu_info
from linumpy.gpu.normalization import (
    apply_histogram_matching_gpu,
    compute_scale_factors_gpu,
)
from linumpy.io.zarr import AnalysisOmeZarrWriter, read_omezarr


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_zarr", help="Input stacked 3D OME-Zarr volume.")
    p.add_argument("out_zarr", help="Output intensity-normalised OME-Zarr volume.")
    p.add_argument(
        "--mode",
        choices=["percentile", "histogram"],
        default="percentile",
        help="Normalization mode (see CPU script for details). [%(default)s]",
    )
    p.add_argument(
        "--n_serial_slices",
        type=int,
        default=None,
        help="Number of serial sections in the stacked volume. [%(default)s]",
    )
    p.add_argument(
        "--smooth_sigma",
        type=float,
        default=10.0,
        help="(percentile mode) Gaussian smoothing sigma in serial-section units. [%(default)s]",
    )
    p.add_argument(
        "--percentile",
        type=float,
        default=80.0,
        help="(percentile mode) Percentile of non-zero voxels per chunk. [%(default)s]",
    )
    p.add_argument(
        "--max_scale",
        type=float,
        default=2.0,
        help="(percentile mode) Maximum allowed scale factor. [%(default)s]",
    )
    p.add_argument(
        "--min_scale",
        type=float,
        default=0.5,
        help="(percentile mode) Minimum allowed scale factor. [%(default)s]",
    )
    p.add_argument(
        "--n_bins",
        type=int,
        default=512,
        help="(histogram mode) Number of histogram bins. [%(default)s]",
    )
    p.add_argument(
        "--tissue_threshold",
        type=float,
        default=0.0,
        help="(histogram mode) Minimum intensity to classify as tissue. [%(default)s]",
    )
    p.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help="Mixing strength of the correction (0.0-1.0). [%(default)s]",
    )
    p.add_argument("--plot", type=str, default=None, help="Optional path to save a diagnostic PNG (percentile mode only).")
    p.add_argument(
        "--pyramid_resolutions",
        type=float,
        nargs="+",
        default=[10, 25, 50, 100],
        help="Target pyramid resolution levels in um. [%(default)s]",
    )
    p.add_argument(
        "--n_levels",
        type=int,
        default=None,
        help="Fixed number of power-of-2 pyramid levels (overrides --pyramid_resolutions).",
    )
    p.add_argument(
        "--make_isotropic", action="store_true", default=True, help="Resample to isotropic voxels at each pyramid level."
    )
    p.add_argument("--no_isotropic", dest="make_isotropic", action="store_false")
    p.add_argument(
        "--use_gpu",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use GPU acceleration if available.",
    )
    p.add_argument("--verbose", action="store_true", help="Print GPU information.")
    return p


def _save_plot(raw_metrics, smoothed, scale_factors, n_serial_slices, plot_path):
    """Re-uses the CPU script's plotting helper without cross-importing scripts."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = np.arange(len(raw_metrics))
    axes[0].plot(x, raw_metrics, "o-", ms=3, lw=1, label="Raw per-slice metric", alpha=0.8)
    axes[0].plot(x, smoothed, "-", lw=2, label="Smoothed reference", color="red")
    axes[0].set_ylabel("Intensity metric (Nth percentile)")
    axes[0].legend()
    axes[0].set_title("Inter-slice intensity drift")
    axes[0].grid(True, alpha=0.3)

    expanded_scale = (
        np.array([scale_factors[round(i * len(scale_factors) / len(raw_metrics))] for i in range(len(raw_metrics))])
        if len(scale_factors) != len(raw_metrics)
        else scale_factors
    )
    axes[1].plot(x, expanded_scale, "o-", ms=3, lw=1, color="green")
    axes[1].axhline(1.0, color="gray", lw=1, ls="--")
    axes[1].set_ylabel("Scale factor applied")
    axes[1].set_xlabel("Serial section index" if n_serial_slices else "Z-plane index")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title("Correction scale factors")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Diagnostic plot saved to {plot_path}")


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    use_gpu = args.use_gpu and GPU_AVAILABLE
    if args.verbose:
        print_gpu_info()
        print(f"Using GPU: {use_gpu}")
        if args.use_gpu and not GPU_AVAILABLE:
            print("GPU requested but not available, falling back to CPU")

    print(f"Loading {args.in_zarr} ...")
    vol_da, res = read_omezarr(args.in_zarr, level=0)
    vol = vol_da[:].astype(np.float32)
    print(f"Volume shape: Z={vol.shape[0]}, Y={vol.shape[1]}, X={vol.shape[2]}")

    if args.mode == "histogram":
        print(
            f"Mode: histogram matching "
            f"({'serial-slice mode, n=' + str(args.n_serial_slices) if args.n_serial_slices else 'Z-plane mode'}), "
            f"n_bins={args.n_bins}, strength={args.strength}, gpu={use_gpu} ..."
        )
        vol_matched = apply_histogram_matching_gpu(
            vol, args.n_serial_slices, args.n_bins, args.tissue_threshold, use_gpu=use_gpu
        )
        vol_matched = np.clip(vol_matched, 0.0, 1.0)

        if args.strength < 1.0:
            print(f"Blending: {args.strength:.2f} * matched + {1.0 - args.strength:.2f} * original ...")
            vol = args.strength * vol_matched + (1.0 - args.strength) * vol
        else:
            vol = vol_matched

        if args.plot:
            print("NOTE: diagnostic plot is only available in percentile mode; skipping.")

    else:  # percentile (default)
        print(
            f"Mode: percentile scaling "
            f"({'serial-slice mode, n=' + str(args.n_serial_slices) if args.n_serial_slices else 'Z-plane mode'}), "
            f"sigma={args.smooth_sigma}, percentile={args.percentile}, strength={args.strength}, gpu={use_gpu} ..."
        )
        scale_factors, raw_metrics, smoothed, _boundaries = compute_scale_factors_gpu(
            vol,
            n_serial_slices=args.n_serial_slices,
            smooth_sigma=args.smooth_sigma,
            percentile=args.percentile,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
            use_gpu=use_gpu,
        )

        if args.strength < 1.0:
            scale_factors = 1.0 + args.strength * (scale_factors - 1.0)
            print(
                f"Adjusted scale factor range after strength={args.strength}: "
                f"{scale_factors.min():.3f} - {scale_factors.max():.3f}"
            )
        else:
            print(
                f"Scale factor range: {scale_factors.min():.3f} - {scale_factors.max():.3f}  (mean={scale_factors.mean():.3f})"
            )

        if args.plot:
            _save_plot(raw_metrics, smoothed, scale_factors, args.n_serial_slices, args.plot)

        print("Applying scale factors ...")
        vol = vol * scale_factors[:, None, None]
        vol = np.clip(vol, 0.0, 1.0)

    print(f"Saving to {args.out_zarr} ...")
    writer = AnalysisOmeZarrWriter(args.out_zarr, vol.shape, chunk_shape=(128, 128, 128), dtype=vol.dtype)
    writer[:] = vol
    writer.finalize(
        res, target_resolutions_um=args.pyramid_resolutions, n_levels=args.n_levels, make_isotropic=args.make_isotropic
    )
    print("Done.")


if __name__ == "__main__":
    main()
