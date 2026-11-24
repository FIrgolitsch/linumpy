#!/usr/bin/env python3
"""Simplified pairwise registration for motor-position-based reconstruction.

This script performs two tasks.

1. Z-MATCHING: find the optimal Z-overlap between consecutive slices.
2. REFINEMENT: compute small rotation and sub-pixel corrections.

The XY alignment is handled by motor positions (shifts_xy.csv), so this
script only computes small corrections, not large translations.

The slabs arrive placed by the stage, which is the reference. A correction is
kept only when it raises the fine-structure correlation of the two planes over
what the placement alone gives; otherwise the transform is the identity.

Outputs.

* ``transform.tfm`` -- SimpleITK transform file (rotation + small translation).
* ``offsets.txt`` -- Z-index correspondence ``[fixed_z, moving_z]``.
* ``metrics.json`` -- registration quality metrics.
"""

import linumpy.config.threads  # noqa: F401
from linumpy.config.threads import configure_all_libraries

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import SimpleITK as sitk
from skimage.filters import threshold_otsu

from linumpy.cli.args import add_overwrite_arg
from linumpy.io.zarr import read_omezarr
from linumpy.metrics import collect_pairwise_registration_metrics
from linumpy.registration.refinement import (
    apply_rigid_2d,
    centre_ncc,
    coarse_translation,
    find_best_z,
    register_refinement,
    select_rigid_candidate,
    structure_ncc,
)
from linumpy.registration.transforms import create_transform

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_fixed", type=Path, help="Fixed volume (.ome.zarr) - bottom slice")
    p.add_argument("in_moving", type=Path, help="Moving volume (.ome.zarr) - top slice")
    p.add_argument("out_directory", help="Output directory")

    # Z-matching
    z_group = p.add_argument_group("Z-matching")
    z_group.add_argument(
        "--slicing_interval_mm", type=float, default=0.200, help="Physical slice thickness in mm [%(default)s]"
    )
    z_group.add_argument(
        "--search_range_mm", type=float, default=0.100, help="Search range around expected Z in mm [%(default)s]"
    )
    z_group.add_argument(
        "--moving_z_index",
        type=int,
        default=0,
        help="2-D Euler template plane in the moving volume [%(default)s].\n"
        "Not a stacking crop; linum-stack-slices-motor uses --moving_z_first_index.",
    )

    # Refinement
    ref_group = p.add_argument_group("Refinement")
    ref_group.add_argument(
        "--enable_rotation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enable rotation correction. Use --no-enable_rotation to disable. [%(default)s]",
    )
    # Legacy alias retained for backward-compatibility with the Nextflow pipeline
    # (workflows/reconst_3d/soct_3d_reconst.nf still emits --no_rotation).
    ref_group.add_argument(
        "--no_rotation",
        dest="enable_rotation",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    ref_group.add_argument(
        "--max_rotation_deg", type=float, default=5.0, help="Maximum rotation correction in degrees [%(default)s]"
    )
    ref_group.add_argument(
        "--max_translation_px", type=float, default=20.0, help="Maximum translation refinement in pixels [%(default)s]"
    )
    ref_group.add_argument(
        "--initial_alignment",
        choices=["none", "com", "gradient", "both"],
        default="both",
        help="Look for a translation too large for the optimizer before refining:\n"
        "  none                - start the optimizer from the placement\n"
        "  com, gradient, both - also try the best translation found by phase\n"
        "                        correlation of the tissue outline and texture [%(default)s]",
    )
    ref_group.add_argument(
        "--min_structure_gain",
        type=float,
        default=0.02,
        help="Smallest gain in fine-structure correlation over the placement for a\ncorrection to be kept [%(default)s]",
    )
    ref_group.add_argument(
        "--min_structure_match",
        type=float,
        default=0.15,
        help="Lowest fine-structure correlation a correction must reach to be kept.\n"
        "Two planes with little in common give faint matches at arbitrary\n"
        "shifts; below this the placement is kept [%(default)s]",
    )

    # Output
    p.add_argument("--out_transform", default="transform.tfm")
    p.add_argument("--out_offsets", default="offsets.txt")
    p.add_argument("--screenshot", default=None, help="Save debug screenshot")

    add_overwrite_arg(p)
    return p


def normalize(image: Any) -> Any:
    """Normalize image to [0, 1] using percentile clipping."""
    valid = image > 0
    if not np.any(valid):
        return np.zeros_like(image, dtype=np.float32)

    pmin = np.percentile(image[valid], 5)
    pmax = np.percentile(image[valid], 95)

    if pmax <= pmin:
        return np.zeros_like(image, dtype=np.float32)

    norm = (image.astype(np.float32) - pmin) / (pmax - pmin)
    return np.clip(norm, 0, 1)


def tissue_threshold(fixed: np.ndarray, moving: np.ndarray) -> float:
    """Return the Otsu threshold that splits the two planes into background and tissue."""
    values = np.concatenate([fixed[fixed > 0].ravel()[::8], moving[moving > 0].ravel()[::8]])
    if values.size < 100 or float(np.ptp(values)) == 0.0:
        return 0.01
    return float(threshold_otsu(values))


def main() -> None:
    """Run the pairwise registration script."""
    configure_all_libraries()
    p = _build_arg_parser()
    args = p.parse_args()

    # Load volumes
    logger.info("Loading fixed: %s", args.in_fixed)
    fixed_vol, res = read_omezarr(args.in_fixed)

    logger.info("Loading moving: %s", args.in_moving)
    moving_vol, _ = read_omezarr(args.in_moving)

    # Create output directory
    out_dir = Path(args.out_directory)
    if out_dir.exists() and not args.f:
        p.error(f"Output directory exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get moving slice — clamp index in case crop_interface left fewer z-voxels than the
    # configured moving_z_index (e.g. default 4 but volume has only 3 slices).
    nz_moving = moving_vol.shape[0]
    if args.moving_z_index >= nz_moving:
        logger.warning(
            "moving_z_index %d exceeds moving volume depth %d; clamping to %d",
            args.moving_z_index,
            nz_moving,
            nz_moving - 1,
        )
        args.moving_z_index = nz_moving - 1
    moving_slice = np.array(moving_vol[args.moving_z_index])
    moving_norm = normalize(moving_slice)

    # Calculate expected Z position
    # Both slabs start at their cut face, and the cut face of the moving slab
    # lies one slice thickness below that of the fixed slab. The plane of the
    # moving slab is therefore expected one slice thickness plus its own index
    # from the TOP of the fixed slab, however deep the fixed slab is.

    # NOTE: read_omezarr returns resolution in millimeters (OME-NGFF standard)
    res_z_mm = res[0] if len(res) >= 1 else 0.010  # mm (default 10 µm)

    logger.info("Resolution from metadata: %s", res)
    logger.info("Using Z resolution: %s mm (%.2f µm)", res_z_mm, res_z_mm * 1000)

    # Calculate interval in voxels: slicing_interval_mm / res_z_mm
    interval_vox = round(args.slicing_interval_mm / res_z_mm)
    search_vox = round(args.search_range_mm / res_z_mm)

    fixed_nz = fixed_vol.shape[0]
    expected_z = interval_vox + args.moving_z_index

    logger.info("Fixed volume: %s slices", fixed_nz)
    logger.info("Interval: %s mm = %s voxels", args.slicing_interval_mm, interval_vox)
    logger.info("Search range: %s mm = %s voxels", args.search_range_mm, search_vox)
    logger.info("Expected Z (before clamp): %s", expected_z)

    # Ensure expected_z is within bounds
    expected_z = max(0, min(fixed_nz - 1, expected_z))

    logger.info("Searching for match near z=%s in fixed volume (search ±%s)", expected_z, search_vox)

    # Find best Z match. The planes are compared pixel by pixel, which says
    # little when the slabs are also offset in XY. So an offset is looked for
    # first, on the mean of the planes of the search window: structure that
    # runs through the depth survives that mean. The moving plane is then
    # compared with each fixed plane at that offset.
    moving_raw = moving_slice.astype(np.float32)
    z_min = max(0, expected_z - search_vox)
    z_max = min(fixed_nz - 1, expected_z + search_vox)
    pre_tx, pre_ty = 0.0, 0.0
    if args.initial_alignment != "none":
        window_mean = np.asarray(fixed_vol[z_min : z_max + 1]).astype(np.float32).mean(axis=0)
        pre_tx, pre_ty, _ = coarse_translation(
            window_mean,
            moving_raw,
            args.max_translation_px,
            threshold=tissue_threshold(window_mean, moving_raw),
            metric=structure_ncc,
        )
    moving_for_z = apply_rigid_2d(moving_raw, pre_tx, pre_ty) if (pre_tx != 0.0 or pre_ty != 0.0) else moving_raw

    # The plane is chosen on the fine structure, at half resolution to keep
    # it quick. The plain correlation of the planes follows the broad
    # brightness of the tissue and picks the shallowest plane it may.
    def _structure(fixed_plane: np.ndarray, moving_plane: np.ndarray) -> float:
        fixed_half = np.asarray(fixed_plane, dtype=np.float32)[::2, ::2]
        moving_half = moving_plane[::2, ::2]
        return structure_ncc(fixed_half, moving_half, tissue_threshold(fixed_half, moving_half), sigma=10.0)

    best_z, z_structure = find_best_z(fixed_vol, moving_for_z, expected_z, search_vox, metric=_structure)
    # Reported as before: the correlation of the centre region at that plane.
    z_correlation = centre_ncc(np.asarray(fixed_vol[best_z]), moving_for_z)

    logger.info(
        "Best Z match: %s (expected: %s, structure match: %.4f, correlation: %.4f)",
        best_z,
        expected_z,
        z_structure,
        z_correlation,
    )

    # Warn if z-match deviates significantly from expected
    z_deviation = abs(best_z - expected_z)
    if z_deviation > search_vox // 2:
        logger.warning("Z-match deviation is large (%s voxels) - may indicate alignment issues", z_deviation)

    # Get fixed slice at best Z
    fixed_slice = np.array(fixed_vol[best_z])
    fixed_norm = normalize(fixed_slice)

    # Candidate corrections. The placement itself (identity) is always one.
    fixed_raw = fixed_slice.astype(np.float32)
    threshold = tissue_threshold(fixed_raw, moving_raw)
    candidates: list[tuple[str, float, float, float]] = []
    initial_offset = None
    if args.initial_alignment != "none":
        if pre_tx != 0.0 or pre_ty != 0.0:
            candidates.append(("coarse_window", pre_tx, pre_ty, 0.0))
        coarse_tx, coarse_ty, _ = coarse_translation(
            fixed_raw, moving_raw, args.max_translation_px, threshold=threshold, metric=structure_ncc
        )
        if coarse_tx != 0.0 or coarse_ty != 0.0:
            candidates.append(("coarse", coarse_tx, coarse_ty, 0.0))
            logger.info("Coarse translation candidate: tx=%.1f, ty=%.1f", coarse_tx, coarse_ty)
        if candidates:
            # Start the optimizer from the better of the two.
            start, start_tx, start_ty, *_ = select_rigid_candidate(
                fixed_raw,
                moving_raw,
                candidates,
                threshold=threshold,
                min_gain=args.min_structure_gain,
                min_score=args.min_structure_match,
            )
            if start != "identity":
                initial_offset = (start_ty, start_tx)

    logger.info("Computing refinement (rotation=%s)...", args.enable_rotation)
    diagnostics: dict = {}
    ref_tx, ref_ty, ref_angle, metric = register_refinement(
        fixed_norm,
        moving_norm,
        enable_rotation=args.enable_rotation,
        max_rotation_deg=args.max_rotation_deg,
        max_translation_px=args.max_translation_px,
        fixed_mask=fixed_raw > threshold,
        moving_mask=moving_raw > threshold,
        initial_offset=initial_offset,
        bound_mode="reject",
        diagnostics=diagnostics,
    )
    if diagnostics.get("rejected"):
        logger.info("Refinement left the bounds and is dropped")
    else:
        candidates.append(("refined", ref_tx, ref_ty, ref_angle))

    chosen, tx, ty, angle_deg, structure_match, structure_match_identity = select_rigid_candidate(
        fixed_raw,
        moving_raw,
        candidates,
        threshold=threshold,
        min_gain=args.min_structure_gain,
        min_score=args.min_structure_match,
    )
    logger.info(
        "Kept %s: tx=%.2fpx, ty=%.2fpx, rot=%.3f° (structure match %.3f, placement alone %.3f)",
        chosen,
        tx,
        ty,
        angle_deg,
        structure_match,
        structure_match_identity,
    )

    # Create and save transform
    center = [fixed_slice.shape[1] / 2.0, fixed_slice.shape[0] / 2.0]
    transform = create_transform(tx, ty, angle_deg, center)
    sitk.WriteTransform(transform, str(out_dir / args.out_transform))

    # Save offsets
    np.savetxt(str(out_dir / args.out_offsets), np.array([best_z, args.moving_z_index]), fmt="%d")

    # Detect interpolated neighbours. Registrations where either volume is a
    # synthetic (interpolated) slice produce unreliable rotation/translation
    # because one side of the pair is a blend of non-overlapping tissue. We
    # still run the registration (so a .tfm exists), but force the metrics
    # into the "error" status so the downstream stacking gate
    # (skip_error_status in linum_stack_slices_motor.py) discards the
    # transform and falls back to motor-only positioning for that slice.
    fixed_is_interpolated = "_interpolated" in Path(args.in_fixed).name
    moving_is_interpolated = "_interpolated" in Path(args.in_moving).name
    touches_interpolated = fixed_is_interpolated or moving_is_interpolated
    if touches_interpolated:
        logger.warning(
            "Registration involves an interpolated slice (fixed=%s, moving=%s); marking transform as unreliable.",
            fixed_is_interpolated,
            moving_is_interpolated,
        )

    # Collect metrics using standard collector
    collect_pairwise_registration_metrics(
        registration_error=float(metric) if metric != float("inf") else 0.0,
        tx=float(tx),
        ty=float(ty),
        rotation_deg=float(angle_deg),
        best_z_index=int(best_z),
        expected_z_index=int(expected_z),
        output_path=out_dir,
        fixed_path=args.in_fixed,
        moving_path=args.in_moving,
        z_correlation=float(z_correlation),
        structure_match_identity=float(structure_match_identity) if np.isfinite(structure_match_identity) else None,
        structure_match=float(structure_match) if np.isfinite(structure_match) else None,
        chosen_candidate=chosen,
        params={
            "slicing_interval_mm": args.slicing_interval_mm,
            "search_range_mm": args.search_range_mm,
            "enable_rotation": args.enable_rotation,
            "max_rotation_deg": args.max_rotation_deg,
            "max_translation_px": args.max_translation_px,
            "z_correlation": float(z_correlation),
            "z_deviation": int(z_deviation),
            "fixed_is_interpolated": bool(fixed_is_interpolated),
            "moving_is_interpolated": bool(moving_is_interpolated),
        },
    )

    if touches_interpolated:
        # Re-save the metrics JSON with a forced error status so
        # stack_slices_motor discards this transform via skip_error_status.
        import json

        metrics_file = out_dir / "pairwise_registration_metrics.json"
        if metrics_file.exists():
            with metrics_file.open() as f:
                data = json.load(f)
            data["overall_status"] = "error"
            data.setdefault("errors", []).append("One or both inputs are an interpolated slice; transform is synthetic.")
            if "registration_confidence" in data.get("metrics", {}):
                data["metrics"]["registration_confidence"]["value"] = 0.0
                data["metrics"]["registration_confidence"]["status"] = "error"
            with metrics_file.open("w") as f:
                json.dump(data, f, indent=2)

    logger.info("Results saved to %s", out_dir)

    # Screenshot
    if args.screenshot:
        import matplotlib.pyplot as plt

        # Apply transform for visualization
        moving_sitk = sitk.GetImageFromArray(moving_norm.astype(np.float32))
        transform_2d = sitk.Euler2DTransform()
        transform_2d.SetCenter(center)
        transform_2d.SetAngle(np.radians(angle_deg))
        transform_2d.SetTranslation([tx, ty])

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(moving_sitk)
        resampler.SetTransform(transform_2d)
        resampler.SetInterpolator(sitk.sitkLinear)
        registered = sitk.GetArrayFromImage(resampler.Execute(moving_sitk))

        _fig, axes = plt.subplots(2, 2, figsize=(12, 12))

        axes[0, 0].imshow(fixed_norm, cmap="gray")
        axes[0, 0].set_title(f"Fixed (z={best_z})")

        axes[0, 1].imshow(moving_norm, cmap="gray")
        axes[0, 1].set_title(f"Moving (z={args.moving_z_index})")

        axes[1, 0].imshow(registered, cmap="gray")
        axes[1, 0].set_title(f"Registered (tx={tx:.1f}, ty={ty:.1f}, rot={angle_deg:.2f}°)")

        # Overlay
        overlay = np.zeros((*fixed_norm.shape, 3))
        overlay[:, :, 0] = fixed_norm  # Red
        overlay[:, :, 1] = registered  # Green
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title("Overlay (fixed=red, registered=green)")

        for ax in axes.flat:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(args.screenshot, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Screenshot saved to %s", args.screenshot)


if __name__ == "__main__":
    main()
