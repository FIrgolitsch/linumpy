#!/usr/bin/env python3
"""Estimate a single 2x2 tile-placement affine pooled across many 3D mosaic grids.

For each input ``mosaic_grid_*.ome.zarr`` volume, load only the central Z
plane and call
:func:`linumpy.stitching.motor.compute_registration_refinements` to
measure per-pair absolute tile displacements via phase correlation.
Pairs from every input are concatenated into one pool and a single 2×2
affine transform is fitted via :func:`~linumpy.stitching.motor.estimate_affine_from_pairs`.

The resulting transform captures instrument-level geometry (scan-to-stage
rotation θ, motor non-perpendicularity φ, effective per-axis step in
pixels) which is constant across an acquisition session.  Use the
resulting ``.npy`` as ``--input_transform`` for
``linum_stitch_3d_refined.py`` to remove per-slice affine jitter while
keeping the blend-shift sub-pixel refinement.

The script is read-only with respect to its inputs and does not touch
any pipeline outputs.
"""

# Configure thread limits before numpy/scipy imports
import linumpy._thread_config  # noqa: F401

import argparse
import json
import logging
import random
import re
import sys
from pathlib import Path

import numpy as np

from linumpy.io.zarr import read_omezarr
from linumpy.stitching.motor import (
    _extract_displacement_params,
    compute_registration_refinements,
    estimate_affine_from_pairs,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_SLICE_RE = re.compile(r"z(\d+)")


def _extract_slice_id(path: Path) -> str:
    match = _SLICE_RE.search(path.name)
    return match.group(1) if match else path.stem


def _load_slice_config(slice_config_path: Path) -> set[str]:
    import csv

    used: set[str] = set()
    with slice_config_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if str(row.get("use", "")).strip().lower() in {"true", "1", "yes"}:
                used.add(row["slice_id"].strip().zfill(2))
    return used


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "input_dir",
        help="Directory containing mosaic_grid_*z??.ome.zarr files.",
    )
    p.add_argument(
        "output_transform",
        help="Output path for the fitted 2x2 affine transform (.npy).",
    )
    p.add_argument(
        "--overlap_fraction",
        type=float,
        default=0.2,
        help="Expected tile overlap fraction (must match acquisition). [%(default)s]",
    )
    p.add_argument(
        "--pattern",
        type=str,
        default="mosaic_grid*_z*.ome.zarr",
        help="Glob pattern used to discover input mosaic grids. [%(default)s]",
    )
    p.add_argument(
        "--slice_config",
        type=str,
        default=None,
        help="Optional slice_config.csv — rows with use=false are skipped.",
    )
    p.add_argument(
        "--include_slice",
        type=str,
        nargs="+",
        default=None,
        help="Optional explicit list of slice ids (zero-padded, e.g. '10 11 12')\n"
        "to include. Combined with --slice_config via intersection when both\n"
        "are provided.",
    )
    p.add_argument(
        "--histogram_match",
        action="store_true",
        help="Match overlap histograms before phase correlation (more robust\n"
        "to uneven tile-edge illumination; matches the old\n"
        "linum_estimate_transform.py behaviour).",
    )
    p.add_argument(
        "--max_empty_fraction",
        type=float,
        default=None,
        help="If set, use an Otsu threshold to detect empty overlaps and skip\n"
        "any pair with more than this fraction of background pixels.\n"
        "When unset, the default per-volume 'mean(overlap > 0) < 0.1'\n"
        "heuristic is used.",
    )
    p.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Maximum number of pooled pairs to feed into the LS fit.\n"
        "If set and the pool exceeds this size, a reproducible random\n"
        "sub-sample is drawn. Unset means use every pair.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for pair sub-sampling (used only when --n_samples is set). [%(default)s]",
    )
    p.add_argument(
        "--diagnostics_json",
        type=str,
        default=None,
        help="Optional JSON sidecar for fit diagnostics and per-volume stats.",
    )
    p.add_argument("--overwrite", "-f", action="store_true", help="Overwrite the output transform if it already exists.")
    return p


def _discover_volumes(
    input_dir: Path,
    pattern: str,
    slice_config_path: Path | None,
    explicit_ids: list[str] | None,
) -> list[tuple[str, Path]]:
    zarr_paths = sorted(input_dir.glob(pattern))
    allowed: set[str] | None = None
    if slice_config_path is not None:
        allowed = _load_slice_config(slice_config_path)
        logger.info("slice_config: %d slices marked use=true", len(allowed))
    if explicit_ids is not None:
        explicit_set = {sid.strip().zfill(2) for sid in explicit_ids}
        allowed = explicit_set if allowed is None else allowed & explicit_set
        logger.info("--include_slice: restricting to %d slice ids", len(explicit_set))

    volumes: list[tuple[str, Path]] = []
    for path in zarr_paths:
        slice_id = _extract_slice_id(path)
        if allowed is not None and slice_id not in allowed:
            continue
        volumes.append((slice_id, path))
    return volumes


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        parser.error(f"Input directory does not exist: {input_dir}")

    output_transform = Path(args.output_transform)
    if output_transform.exists() and not args.overwrite:
        parser.error(f"Output exists: {output_transform}. Use -f to overwrite.")
    if output_transform.suffix != ".npy":
        parser.error("output_transform must end in .npy")

    slice_config_path = Path(args.slice_config) if args.slice_config else None
    if slice_config_path is not None and not slice_config_path.exists():
        parser.error(f"slice_config.csv not found: {slice_config_path}")

    volumes = _discover_volumes(input_dir, args.pattern, slice_config_path, args.include_slice)
    if not volumes:
        parser.error(f"No mosaic grids selected (pattern={args.pattern!r}, dir={input_dir})")
    logger.info("pooling pairs from %d mosaic grids", len(volumes))

    tile_shape_ref: tuple | None = None
    all_pairs: list[dict] = []
    per_slice_stats: list[dict] = []

    for slice_id, zarr_path in volumes:
        vol, _ = read_omezarr(str(zarr_path), level=0)
        tile_shape = tuple(vol.chunks)
        if len(tile_shape) != 3:
            logger.warning("slice %s: unexpected chunks %s, skipping", slice_id, tile_shape)
            continue
        if tile_shape_ref is None:
            tile_shape_ref = tile_shape
        elif tile_shape[1:] != tile_shape_ref[1:]:
            logger.warning(
                "slice %s: tile shape %s differs from reference %s — pooling across\n"
                "different tile sizes is not supported. Skipping.",
                slice_id,
                tile_shape,
                tile_shape_ref,
            )
            continue

        nx = vol.shape[1] // tile_shape[1]
        ny = vol.shape[2] // tile_shape[2]
        if nx == 0 or ny == 0:
            logger.warning("slice %s: too few tiles (nx=%d ny=%d), skipping", slice_id, nx, ny)
            continue
        z_mid_full = vol.shape[0] // 2
        logger.info(
            "slice %s: shape=%s tile=%s grid=%dx%d z_mid=%d (hist_match=%s empty_frac=%s)",
            slice_id,
            tuple(vol.shape),
            tile_shape,
            nx,
            ny,
            z_mid_full,
            args.histogram_match,
            args.max_empty_fraction,
        )
        z_plane = np.asarray(vol[z_mid_full : z_mid_full + 1])

        refinements = compute_registration_refinements(
            z_plane,
            tile_shape,
            nx,
            ny,
            args.overlap_fraction,
            histogram_match=args.histogram_match,
            max_empty_fraction=args.max_empty_fraction,
        )
        pairs = refinements["pairs"]
        stats = dict(refinements["stats"])
        stats["slice_id"] = slice_id
        stats["nx"] = int(nx)
        stats["ny"] = int(ny)
        per_slice_stats.append(stats)
        logger.info(
            "slice %s: %d valid pairs collected (total=%d)",
            slice_id,
            stats["valid_pairs"],
            stats["total_pairs"],
        )
        all_pairs.extend(pairs)

    if tile_shape_ref is None:
        parser.error("No usable mosaic grids produced pair measurements.")
    logger.info("pooled pair count: %d", len(all_pairs))

    if args.n_samples is not None and len(all_pairs) > args.n_samples:
        rng = random.Random(args.seed)
        all_pairs = rng.sample(all_pairs, args.n_samples)
        logger.info("random-sampled to %d pairs (seed=%d)", len(all_pairs), args.seed)

    transform, diagnostics = estimate_affine_from_pairs(all_pairs, tile_shape_ref, args.overlap_fraction)
    diagnostics_full = {
        "n_volumes": len(per_slice_stats),
        "n_pairs_pooled": len(all_pairs),
        "tile_shape": list(tile_shape_ref),
        "overlap_fraction": args.overlap_fraction,
        "histogram_match": bool(args.histogram_match),
        "max_empty_fraction": args.max_empty_fraction,
        "sampled_n": args.n_samples,
        "seed": args.seed if args.n_samples is not None else None,
        "transform": transform.tolist(),
        "displacement_model": _extract_displacement_params(transform, tile_shape_ref, args.overlap_fraction),
        "lstsq_residual": diagnostics.get("lstsq_residual"),
        "fallback": diagnostics.get("fallback", False),
        "per_slice_stats": per_slice_stats,
    }

    logger.info("Global displacement model:")
    model = diagnostics_full["displacement_model"]
    logger.info("  Transform: %s", np.array2string(transform, precision=3))
    logger.info("  theta_deg = %+.3f", model["theta_deg"])
    logger.info("  phi_deg   = %+.3f", model["phi_deg"])
    logger.info("  Ox_frac   = %.4f (expected %.4f)", model["Ox_fraction"], args.overlap_fraction)
    logger.info("  Oy_frac   = %.4f (expected %.4f)", model["Oy_fraction"], args.overlap_fraction)
    logger.info("  lstsq_residual = %s", diagnostics_full["lstsq_residual"])

    output_transform.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_transform), transform)
    logger.info("wrote transform to %s", output_transform)

    if args.diagnostics_json is not None:
        diagnostics_path = Path(args.diagnostics_json)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(json.dumps(diagnostics_full, indent=2))
        logger.info("wrote diagnostics to %s", diagnostics_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
