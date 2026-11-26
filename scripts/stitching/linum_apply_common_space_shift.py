#!/usr/bin/env python3
"""Shift one mosaic into the common space planned by `linum-align-mosaics-3d-from-shifts --plan_only`.

The plan lists, for every mosaic, where it sits in the common frame and the
size of that frame. This script applies one row of it, so a workflow can run
the mosaics side by side.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401
from linumpy.config.threads import configure_all_libraries

import argparse
from pathlib import Path

from linumpy.cli.args import add_overwrite_arg, assert_output_exists
from linumpy.stack_alignment.common_space import shift_mosaic_to_common_space
from linumpy.stack_alignment.slab_slant import read_slab_slant


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_mosaic", help="Mosaic to bring to common space (.ome.zarr).")
    p.add_argument("out_mosaic", help="Output mosaic in common space (.ome.zarr).")
    p.add_argument("--shift_x_px", type=float, required=True, help="X position of the mosaic in the common frame (pixels).")
    p.add_argument("--shift_y_px", type=float, required=True, help="Y position of the mosaic in the common frame (pixels).")
    p.add_argument("--shape", type=int, nargs=2, required=True, metavar=("HEIGHT", "WIDTH"), help="Size of the common frame.")
    p.add_argument(
        "--slant_file",
        type=Path,
        default=None,
        help="Slant inside the slabs, as written by linum-estimate-slab-slant.\n"
        "It is removed in the same resampling. Default: no slant.",
    )
    add_overwrite_arg(p)
    return p


def main() -> None:
    """Run function operation."""
    configure_all_libraries()
    parser = _build_arg_parser()
    args = parser.parse_args()
    assert_output_exists(Path(args.out_mosaic), parser, args)

    slant = read_slab_slant(args.slant_file) if args.slant_file is not None else (0.0, 0.0)
    shift_mosaic_to_common_space(
        Path(args.in_mosaic),
        Path(args.out_mosaic),
        args.shift_x_px,
        args.shift_y_px,
        (args.shape[0], args.shape[1]),
        slant=slant,
    )
    print(
        f"  Processed {Path(args.in_mosaic).name}: applied_shift=({args.shift_x_px:.1f}, {args.shift_y_px:.1f}) px, "
        f"slant=({slant[0]:+.4f}, {slant[1]:+.4f}) px per plane"
    )


if __name__ == "__main__":
    main()
