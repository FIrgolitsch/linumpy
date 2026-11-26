#!/usr/bin/env python3
"""Measure the slant inside the slabs of one brain from the overlaps of neighbouring slabs.

The A-lines of a slab are not exactly parallel to the vertical axis of the
stage, so a structure drifts sideways with depth, by the same amount in every
slab. The top planes of a slab are compared with the planes one slice
thickness deeper in the slab above, at the position the common-space plan
gives both. The shift that is left, divided by the slice thickness in planes,
is the slant. `linum-apply-common-space-shift --slant_file` removes it.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401
from linumpy.config.threads import configure_all_libraries

import argparse
import csv
from pathlib import Path

from linumpy.cli.args import add_overwrite_arg, assert_output_exists
from linumpy.io.zarr import read_omezarr
from linumpy.stack_alignment.slab_slant import estimate_slab_slant, write_slab_slant


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        "in_directory", type=Path, help="Folder with the mosaics before they are brought to common space (.ome.zarr)."
    )
    p.add_argument("in_plan", type=Path, help="Plan written by linum-align-mosaics-3d-from-shifts --plan_only.")
    p.add_argument("out_slant", type=Path, help="Output JSON with the slant in pixels per plane.")
    p.add_argument("--slicing_interval_mm", type=float, default=0.2, help="Slice thickness in mm. [%(default)s]")
    p.add_argument(
        "--planes",
        type=int,
        nargs="+",
        default=[3, 6, 9, 12],
        help="Planes of the incoming slab that are measured. [%(default)s]",
    )
    p.add_argument(
        "--min_correlation", type=float, default=0.2, help="Plane pairs that match worse than this are left out. [%(default)s]"
    )
    p.add_argument("--min_joints", type=int, default=5, help="With fewer measured joints no slant is reported. [%(default)s]")
    p.add_argument(
        "--use_gpu",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Run the matching on the GPU when there is one. [%(default)s]",
    )
    add_overwrite_arg(p)
    return p


def main() -> None:
    """Run function operation."""
    configure_all_libraries()
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_output_exists(args.out_slant, parser, args)

    slabs = []
    with args.in_plan.open(newline="") as handle:
        for row in csv.DictReader(handle):
            mosaic = args.in_directory / row["filename"]
            if mosaic.exists():
                slabs.append((int(row["slice_id"]), mosaic, float(row["shift_x_px"]), float(row["shift_y_px"])))
    if not slabs:
        parser.error(f"none of the mosaics of {args.in_plan} is in {args.in_directory}")

    _, resolution = read_omezarr(slabs[0][1])
    step_planes = round(args.slicing_interval_mm / float(resolution[0]))
    result = estimate_slab_slant(
        slabs,
        step_planes,
        planes=args.planes,
        min_correlation=args.min_correlation,
        min_joints=args.min_joints,
        use_gpu=args.use_gpu,
    )
    write_slab_slant(result, args.out_slant)

    if result["used"]:
        print(
            f"Slant: x {result['slant_x_px_per_plane']:+.4f} (+-{result['standard_error_x']:.4f}), "
            f"y {result['slant_y_px_per_plane']:+.4f} (+-{result['standard_error_y']:.4f}) px per plane, "
            f"from {result['n_joints']} joints at a step of {step_planes} planes"
        )
    else:
        print(f"No slant reported: {result['n_joints']} joints measured, {args.min_joints} needed")


if __name__ == "__main__":
    main()
