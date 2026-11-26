#!/usr/bin/env python3
"""Add a resampled pyramid level to a mosaic grid that was made without one.

linum-create-mosaic-grid-3d --pyramid_resolutions saves, next to the grid at
its own resolution, a copy at for instance 10 micron, which the 3D
reconstruction reads in place of resampling the grid. This script adds that
level to a grid that exists already. Every tile of level 0 is resampled the
way the grid script resamples it and written to a new level with the tile
grid and the shard layout of level 0. Level 0 is not touched.

With --from_resampled the tiles are copied from a grid that
linum-resample-mosaic-grid already made of this one at the resolution, for
instance in the work folder of a 3D reconstruction, and level 0 is not read.
That is the way to add the level to a grid on slow storage.

A grid that already holds a level at the resolution is left as it is. The
level is listed in the metadata of the grid once it is complete, so a run that
is interrupted leaves a grid that reads as before; run the script again.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from linumpy.geometry.resampling import resampled_tile_shape, resolution_is_mm
from linumpy.gpu import GPU_AVAILABLE
from linumpy.gpu.interpolation import resize
from linumpy.io.chunk_writer import write_shard_chunks
from linumpy.io.zarr import create_omezarr_level, read_omezarr, read_omezarr_resolutions, register_omezarr_level

# Tiles of a shard read from disk at the same time.
_LOAD_THREADS = 8


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("in_mosaic", help="Mosaic grid to add the level to (.ome.zarr). It is modified in place.")
    p.add_argument(
        "-r",
        "--resolution",
        type=float,
        default=10.0,
        help="Isotropic resolution of the level in micron per pixel. [%(default)s]",
    )
    p.add_argument(
        "--use_gpu",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Resample on the GPU when one is available. [%(default)s]",
    )
    p.add_argument(
        "--from_resampled",
        default=None,
        help="Grid that linum-resample-mosaic-grid made of this one at the resolution (.ome.zarr).\n"
        "Its tiles are copied and level 0 is not read.",
    )
    return p


def _load_shard(level0: Any, tile_shape: tuple, cells: list) -> list:
    """Read the tiles *cells* of an array made of tiles of *tile_shape*, a few at a time."""

    def read(cell: tuple) -> tuple:
        ix, iy = cell
        rows = slice(ix * tile_shape[1], (ix + 1) * tile_shape[1])
        cols = slice(iy * tile_shape[2], (iy + 1) * tile_shape[2])
        return cell, np.ascontiguousarray(level0[:, rows, cols])

    with ThreadPoolExecutor(max_workers=min(_LOAD_THREADS, len(cells))) as executor:
        return list(executor.map(read, cells))


def main() -> None:
    """Run the script."""
    args = _build_arg_parser().parse_args()
    use_gpu = args.use_gpu and GPU_AVAILABLE
    grid = Path(args.in_mosaic)
    start = time.time()

    level0, source_res = read_omezarr(grid, level=0)
    target_res = args.resolution / 1000.0 if resolution_is_mm(source_res) else float(args.resolution)
    if target_res <= max(float(v) for v in source_res):
        raise SystemExit(f"{args.resolution} um is not coarser than level 0 ({list(source_res)}); nothing to add.")

    tile_shape = tuple(int(v) for v in level0.chunks)
    shard_shape = tuple(int(v) for v in (level0.shards or level0.chunks))
    if level0.shape[1] % tile_shape[1] or level0.shape[2] % tile_shape[2]:
        raise SystemExit(f"Level 0 of shape {level0.shape} is not a whole number of tiles of {tile_shape}.")
    factor = (shard_shape[1] // tile_shape[1], shard_shape[2] // tile_shape[2])
    n_tiles = (level0.shape[1] // tile_shape[1], level0.shape[2] // tile_shape[2])
    level_tile = resampled_tile_shape(tile_shape, source_res, target_res)
    level_shape = (level_tile[0], n_tiles[0] * level_tile[1], n_tiles[1] * level_tile[2])

    for index, res in enumerate(read_omezarr_resolutions(grid)):
        if index == 0 or not np.allclose(res, target_res, rtol=1e-6, atol=0.0):
            continue
        present, _ = read_omezarr(grid, level=index)
        if tuple(present.shape) == level_shape and tuple(present.chunks) == tuple(level_tile):
            print(f"Level {index} of {grid.name} is already at {args.resolution} um; nothing to do.")
            return
        raise SystemExit(
            f"Level {index} of {grid.name} is at {args.resolution} um but has shape {present.shape} and tiles "
            f"{present.chunks}, not {level_shape} and {tuple(level_tile)}."
        )

    # Where the tiles come from: level 0, to be resampled, or a grid that holds them resampled already.
    source, source_tile = level0, tile_shape
    if args.from_resampled is not None:
        source, resampled_res = read_omezarr(Path(args.from_resampled), level=0)
        if not np.allclose(resampled_res, target_res, rtol=1e-6, atol=0.0) or tuple(source.shape) != level_shape:
            raise SystemExit(
                f"{args.from_resampled} has voxels of {list(resampled_res)} and shape {tuple(source.shape)}; a resampling of "
                f"{grid.name} to {args.resolution} um has voxels of {target_res} and shape {level_shape}."
            )
        source_tile = tuple(level_tile)

    level_shards = (level_tile[0], factor[0] * level_tile[1], factor[1] * level_tile[2]) if level0.shards else None
    array, name = create_omezarr_level(grid, level_shape, level_tile, shards=level_shards)
    print(f"{grid.name}: level {name} at {args.resolution} um, tile {tuple(level_tile)}, shape {level_shape}", flush=True)

    shards = [
        (
            sx,
            sy,
            [
                (ix, iy)
                for ix in range(sx * factor[0], min((sx + 1) * factor[0], n_tiles[0]))
                for iy in range(sy * factor[1], min((sy + 1) * factor[1], n_tiles[1]))
            ],
        )
        for sx in range(-(-n_tiles[0] // factor[0]))
        for sy in range(-(-n_tiles[1] // factor[1]))
    ]
    n_written = 0
    # The tiles of the next shard are read while this one is resampled and written.
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        pending = prefetch.submit(_load_shard, source, source_tile, shards[0][2])
        for position, (sx, sy, _cells) in enumerate(shards):
            loaded = pending.result()
            if position + 1 < len(shards):
                pending = prefetch.submit(_load_shard, source, source_tile, shards[position + 1][2])
            tiles = {}
            for (ix, iy), vol in loaded:
                # A tile that was never acquired is the fill value at level 0 and stays absent here.
                if not vol.any():
                    continue
                tiles[0, ix - sx * factor[0], iy - sy * factor[1]] = (
                    vol if source is not level0 else resize(vol, level_tile, order=1, anti_aliasing=True, use_gpu=use_gpu)
                )
            if not tiles:
                continue
            n_written += len(tiles)
            if not write_shard_chunks(array, (0, sx, sy), tiles):
                for (_, px, py), tile in tiles.items():
                    rows = slice((sx * factor[0] + px) * level_tile[1], (sx * factor[0] + px + 1) * level_tile[1])
                    cols = slice((sy * factor[1] + py) * level_tile[2], (sy * factor[1] + py + 1) * level_tile[2])
                    array[:, rows, cols] = tile

    register_omezarr_level(grid, name, [target_res] * len(level_shape))
    print(
        f"{grid.name}: {n_written} of {n_tiles[0] * n_tiles[1]} tiles written in {time.time() - start:.1f} s"
        f" ({'copied' if source is not level0 else 'GPU' if use_gpu else 'CPU'})",
        flush=True,
    )


if __name__ == "__main__":
    main()
