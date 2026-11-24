#!/usr/bin/env python3

"""Estimate where each slab sits relative to the previous one from the tile stage positions.

The sample is fixed on the stage, so a piece of tissue has the same stage
position in every slice. The image of a slab starts at the low corner of its
mosaic. The shift written for a pair of slices is therefore the displacement
of that corner, in image axes: stage Y runs along the image columns (X) and
stage X along the image rows (Y).

This holds when the acquisition moves the mosaic and when it adds or drops
tiles on any side, so no pair is marked unreliable for a change of the mosaic
grid. Placing every slab at the running sum of these shifts puts it at its
stage position.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map

from linumpy.cli.args import add_processes_arg, parse_processes_arg
from linumpy.mosaic.discovery import get_mosaic_info, get_tiles_ids
from linumpy.stack_alignment.origin import slab_origin_shifts_mm


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("directory", type=Path, help="Tiles directory")
    p.add_argument("output_file", type=Path, help="Output CSV file")
    add_processes_arg(p)
    return p


def process_slice(z: int, tiles_directory: Path) -> tuple:
    """Process a single z slice and return the low corner of its mosaic."""
    mosaic_info = get_mosaic_info(tiles_directory, z, use_stage_positions=True)
    return (
        mosaic_info["mosaic_xmin_mm"],
        mosaic_info["mosaic_ymin_mm"],
        mosaic_info["tile_resolution"],
    )


def main() -> None:
    """Run the XY shift estimation from tile metadata script."""
    # Parse arguments
    p = _build_arg_parser()
    args = p.parse_args()
    n_processes = parse_processes_arg(args.n_processes)

    # Extract the parameters
    tiles_directory = Path(args.directory)
    output_file = Path(args.output_file)

    # Get slice ids
    _, tile_ids = get_tiles_ids(tiles_directory)
    z_values = np.unique([ids[2] for ids in tile_ids])
    n_slices = len(z_values)

    # Prepare the parameters
    tiles_directory_list = [tiles_directory] * n_slices

    # Extract the metadata
    results = process_map(
        process_slice,
        z_values,
        tiles_directory_list,
        unit="slice",
        desc="Computing Mosaic Info",
        position=0,
        leave=True,
        max_workers=n_processes,
    )

    xmin_mm, ymin_mm, tile_resolutions = zip(*results, strict=False)

    x_shifts_mm, y_shifts_mm = slab_origin_shifts_mm(xmin_mm, ymin_mm)

    # The tile resolution is (rz, ry, rx). Image X comes from stage Y and
    # image Y from stage X.
    tile_resolution = tile_resolutions[0]
    x_shift_px = x_shifts_mm / tile_resolution[1]
    y_shift_px = y_shifts_mm / tile_resolution[2]

    # Every shift is read off the stage, so every pair is reliable.
    reliable_flags = np.ones(n_slices - 1, dtype=int)

    # Save the shifts to a csv file
    shifts = np.array([z_values[:-1], z_values[1:], x_shift_px, y_shift_px, x_shifts_mm, y_shifts_mm, reliable_flags]).T
    with Path(output_file).open("w") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(["fixed_id", "moving_id", "x_shift", "y_shift", "x_shift_mm", "y_shift_mm", "reliable"])
        writer.writerows(shifts)


if __name__ == "__main__":
    main()
