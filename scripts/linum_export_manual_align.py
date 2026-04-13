#!/usr/bin/env python3
"""Export lightweight data package for the manual alignment tool.

Reads common-space slices (OME-Zarr) and pairwise registration outputs,
then produces a self-contained directory containing:

- Per-slice AIP images (`.npz`) at a chosen pyramid level
- Per-slice pairwise transform files (`.tfm` + metrics JSON)

This package can be downloaded locally and opened directly by the
manual alignment tool without needing full-resolution 3D volumes.
"""

import linumpy._thread_config  # noqa: F401

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm

from linumpy.io.zarr import read_omezarr

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "slices_dir",
        help="Directory containing common-space slices (slice_z##.ome.zarr).",
    )
    p.add_argument(
        "transforms_dir",
        help="Directory containing pairwise registration outputs (slice_z##*/transform.tfm).",
    )
    p.add_argument(
        "output_dir",
        help="Output directory for the manual alignment data package.",
    )
    p.add_argument(
        "--level",
        type=int,
        default=1,
        help="Pyramid level for AIP computation (0=full, 1=2x downsample, ...). [%(default)s]",
    )
    p.add_argument(
        "--slices",
        type=int,
        nargs="*",
        default=None,
        help="Only export specific slice IDs. Default: all.",
    )
    return p


def _discover_slices(slices_dir: Path) -> dict[int, Path]:
    """Discover common-space slice files."""
    pattern = re.compile(r"slice_z(\d+)")
    slices = {}
    for p in sorted(slices_dir.iterdir()):
        m = pattern.search(p.name)
        if m and p.name.endswith(".ome.zarr"):
            slices[int(m.group(1))] = p
    return dict(sorted(slices.items()))


def _discover_transforms(transforms_dir: Path) -> dict[int, Path]:
    """Discover pairwise transform directories."""
    pattern = re.compile(r"slice_z(\d+)")
    transforms = {}
    for p in sorted(transforms_dir.iterdir()):
        if p.is_dir():
            m = pattern.search(p.name)
            if m:
                transforms[int(m.group(1))] = p
    return dict(sorted(transforms.items()))


def main(argv=None):
    p = _build_arg_parser()
    args = p.parse_args(argv)

    slices_dir = Path(args.slices_dir)
    transforms_dir = Path(args.transforms_dir)
    output_dir = Path(args.output_dir)
    level = args.level

    if not slices_dir.exists():
        logger.error(f"Slices directory not found: {slices_dir}")
        return

    if not transforms_dir.exists():
        logger.error(f"Transforms directory not found: {transforms_dir}")
        return

    slice_paths = _discover_slices(slices_dir)
    transform_paths = _discover_transforms(transforms_dir)

    if not slice_paths:
        logger.error(f"No slice_z##.ome.zarr files found in {slices_dir}")
        return

    logger.info(f"Found {len(slice_paths)} slices, {len(transform_paths)} transform dirs")

    # Filter slices if requested
    if args.slices:
        requested = set(args.slices)
        slice_paths = {k: v for k, v in slice_paths.items() if k in requested}
        logger.info(f"Filtered to {len(slice_paths)} requested slices")

    # Create output directories
    aips_dir = output_dir / "aips"
    aips_dir.mkdir(parents=True, exist_ok=True)
    tfm_dir = output_dir / "transforms"
    tfm_dir.mkdir(parents=True, exist_ok=True)

    # Export AIPs
    logger.info(f"Computing AIPs at pyramid level {level}...")
    for sid, spath in tqdm(slice_paths.items(), desc="AIPs"):
        vol, scale = read_omezarr(str(spath), level=level)
        aip = np.asarray(vol).mean(axis=0).astype(np.float32)
        out_path = aips_dir / f"slice_z{sid:02d}.npz"
        np.savez_compressed(str(out_path), aip=aip, scale=np.array(scale))
        logger.debug(f"  z{sid:02d}: shape={aip.shape}")

    # Export transforms
    logger.info("Copying pairwise transforms...")
    for _sid, tpath in transform_paths.items():
        out_tdir = tfm_dir / tpath.name
        out_tdir.mkdir(parents=True, exist_ok=True)
        # Copy .tfm files
        for tfm_file in tpath.glob("*.tfm"):
            shutil.copy2(tfm_file, out_tdir / tfm_file.name)
        # Copy offsets.txt
        offsets_file = tpath / "offsets.txt"
        if offsets_file.exists():
            shutil.copy2(offsets_file, out_tdir / "offsets.txt")
        # Copy metrics JSON
        metrics_file = tpath / "pairwise_registration_metrics.json"
        if metrics_file.exists():
            shutil.copy2(metrics_file, out_tdir / "pairwise_registration_metrics.json")

    # Write metadata
    metadata = {
        "pyramid_level": level,
        "n_slices": len(slice_paths),
        "slice_ids": sorted(slice_paths.keys()),
        "n_transforms": sum(1 for tpath in transform_paths.values() if list(tpath.glob("*.tfm"))),
    }
    metadata_path = output_dir / "manual_align_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logger.info(f"Exported {len(slice_paths)} AIPs and {len(transform_paths)} transforms to {output_dir}")
    logger.info(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
