#!/usr/bin/env python3

"""Convert 3D OCT tiles to a 3D mosaic grid.

GPU acceleration is used when available (--use_gpu, default on) for
volume resampling/resizing (5-12x speedup). Falls back to CPU if no GPU
is detected or --no-use_gpu is passed.

With --pyramid_resolutions the grid is saved at several resolutions in one
OME-Zarr: level 0 as set by --resolution, and one further level per value
given. Each tile is resampled while it is in memory, so a grid kept at its
native resolution also holds, for instance, the 10 micron grid the 3D
reconstruction works on, and that pipeline does not have to read the native
grid back to resample it.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from linumpy.cli.args import add_processes_arg, parse_processes_arg
from linumpy.geometry.resampling import resampled_tile_shape
from linumpy.gpu import GPU_AVAILABLE, print_gpu_info
from linumpy.gpu.interpolation import resize
from linumpy.io.chunk_writer import write_shard_chunks
from linumpy.io.thorlabs import PreprocessingConfig, ThorOCT
from linumpy.io.zarr import OmeZarrWriter
from linumpy.microscope.oct import OCT
from linumpy.mosaic import discovery as reconstruction

# Global flag for GPU usage (set in main, consulted by process functions)
_USE_GPU = True

# Tiles of a shard read from disk at the same time.
_LOAD_THREADS = 8


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("output_zarr", help="Full path to the output zarr file")
    p.add_argument(
        "--data_type",
        type=str,
        default="OCT",
        choices=["OCT", "PSOCT"],
        help="Type of the data to process [%(default)s]",
    )
    input_g = p.add_argument_group("input")
    input_mutex_g = input_g.add_mutually_exclusive_group(required=True)
    input_mutex_g.add_argument("--from_root_directory", help="Full path to a directory containing the tiles to process.")
    input_mutex_g.add_argument("--from_tiles_list", nargs="+", help="List of tiles to assemble (argument --slice is ignored).")
    options_g = p.add_argument_group("other options")
    options_g.add_argument(
        "-r", "--resolution", type=float, default=10.0, help="Output isotropic resolution in micron per pixel. [%(default)s]"
    )
    options_g.add_argument(
        "--axial_resolution", type=float, default=3.5, help="Axial resolution of the raw data in microns. [%(default)s]"
    )
    options_g.add_argument(
        "--pyramid_resolutions",
        type=float,
        nargs="*",
        default=[],
        help=(
            "Further isotropic resolutions in micron per pixel, saved as extra levels of the output\n"
            "(for example 10). Each tile of level 0 is resampled on its own, the way\n"
            "linum-resample-mosaic-grid does it, so that script can take the level as it is.\n"
            "A value that is not coarser than level 0 is skipped. [none]"
        ),
    )
    options_g.add_argument("-z", "--slice", type=int, help="Slice to process.")
    options_g.add_argument("--keep_galvo_return", action="store_true", help="Keep the galvo return signal [%(default)s]")
    options_g.add_argument(
        "--zarr_root", help="Path to parent directory under which the zarr temporary directory will be created [/tmp/]."
    )
    options_g.add_argument(
        "--fix_galvo_shift", default=True, action=argparse.BooleanOptionalAction, help="Fix the galvo shift. [%(default)s]"
    )
    options_g.add_argument(
        "--fix_camera_shift", default=False, action=argparse.BooleanOptionalAction, help="Fix the camera shift. [%(default)s]"
    )
    options_g.add_argument(
        "--preprocess",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Apply preprocessing (rotate/flip) for legacy data. [%(default)s]",
    )
    options_g.add_argument(
        "--galvo_threshold", type=float, default=0.6, help="Galvo detection confidence threshold. [%(default)s]"
    )
    options_g.add_argument(
        "--galvo_shift",
        type=int,
        default=None,
        help="Galvo shift of this slice in a-lines. Given, it is applied as it is\n"
        "and the detection is skipped (0 applies no fix).",
    )
    options_g.add_argument(
        "--sharding_factor",
        type=int,
        default=1,
        help="A sharding factor of N will result in N^2 tiles per shard. [%(default)s]",
    )
    options_g.add_argument(
        "--use_gpu",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Use GPU acceleration if available. [%(default)s]",
    )
    options_g.add_argument("--verbose", "-v", action="store_true", help="Print GPU information.")
    add_processes_arg(options_g)
    psoct_options_g = p.add_argument_group("PS-OCT options")
    psoct_options_g.add_argument("--polarization", type=int, default=1, choices=[0, 1], help="Polarization index to process")
    psoct_options_g.add_argument("--number_of_angles", type=int, default=1, help="Angle index to process")
    psoct_options_g.add_argument("--angle_index", type=int, default=0, help="Angle index to process")
    psoct_options_g.add_argument("--return_complex", type=bool, default=False, help="Return Complex64 or Float32 data type")
    psoct_options_g.add_argument(
        "--crop_first_index", type=int, default=320, help="First index for cropping on the z axis [%(default)s]"
    )
    psoct_options_g.add_argument(
        "--crop_second_index", type=int, default=750, help="Second index for cropping on the z axis [%(default)s]"
    )
    return p


def load_oct_tile(oct: OCT, **load_options: Any) -> np.ndarray:
    """Load an OCT tile in the axis order of the mosaic grid: (Z, X, Y).

    A tile named ``tile_xAA_yBB`` sits at index AA along the first in-plane
    axis of the grid and BB along the second, so within a tile the a-lines
    (X) have to run along the first in-plane axis and the b-scans (Y) along
    the second. ``OCT.load_image`` returns (Z, Y, X); stored like that, every
    tile is transposed in its cell and no tile continues into its neighbour.
    """
    return oct.load_image(**load_options).transpose(0, 2, 1)


def preprocess_volume(vol: np.ndarray, apply: bool = True) -> np.ndarray:
    """Preprocess the volume by rotating and flipping it (for legacy data)."""
    if not apply:
        return vol
    vol = np.rot90(vol, k=3, axes=(1, 2))
    vol = np.flip(vol, axis=1)
    return vol


def load_single_tile(params: dict) -> tuple:
    """Load a single tile from disk. Used for parallel I/O.

    Returns
    -------
    tuple
        (params, volume) where volume is the loaded numpy array
    """
    f = params["file"]
    crop = params["crop"]
    galvo_shift = params["galvo_shift"]
    fix_camera_shift = params["fix_camera_shift"]
    preprocess = params["preprocess"]
    data_type = params["data_type"]
    psoct_config = params["psoct_config"]

    if data_type == "OCT":
        oct = OCT(f)
        vol = load_oct_tile(oct, crop=crop, fix_galvo_shift=galvo_shift, fix_camera_shift=fix_camera_shift)
        vol = preprocess_volume(vol, apply=preprocess)
    elif data_type == "PSOCT":
        oct = ThorOCT(f, config=psoct_config)
        if psoct_config.erase_polarization_2:
            oct.load()
            vol = oct.first_polarization
        else:
            oct.load()
            vol = oct.second_polarization
        assert vol is not None
        vol = ThorOCT.orient_volume_psoct(vol)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    # The loaders hand out views of the file in its own axis order. Laying
    # the tile out in grid order here, on the loading thread, keeps that copy
    # off the main thread: the GPU and the compressor both take it as it is.
    return (params, np.ascontiguousarray(vol))


def _load_shard_data(proc_params: dict) -> list:
    """Load all tiles for a shard from disk (I/O stage of the pipeline).

    For shards with multiple tiles (sharding_factor > 1) loads them in
    parallel with a ThreadPoolExecutor; otherwise loads the single tile
    directly to avoid threading overhead.

    Returns a list of (params, volume) tuples, one per tile.
    """
    tiles_params = proc_params["params"]
    n_tiles = len(tiles_params)
    if n_tiles > 1:
        with ThreadPoolExecutor(max_workers=min(_LOAD_THREADS, n_tiles)) as executor:
            return list(executor.map(load_single_tile, tiles_params))
    return [load_single_tile(tiles_params[0])]


def _resize_tile(vol: Any, tile_size: tuple, use_gpu: bool) -> Any:
    """Resize one tile to *tile_size* (linear, anti-aliased); complex data per component."""
    if np.iscomplexobj(vol):
        real_resized = resize(vol.real, tile_size, order=1, anti_aliasing=True, use_gpu=use_gpu)
        imag_resized = resize(vol.imag, tile_size, order=1, anti_aliasing=True, use_gpu=use_gpu)
        return real_resized + 1j * imag_resized
    return resize(vol, tile_size, order=1, anti_aliasing=True, use_gpu=use_gpu)


def _prepare_shard(proc_params: dict, loaded_tiles: list) -> list:
    """Resize the loaded tiles of a shard and return them per array (compute stage).

    Returns one ``(array, shard_position, tiles)`` per pyramid level, level 0
    first, where ``tiles`` maps the position of a tile inside the shard to its
    data. Every further level gets the same tiles, each resampled from the
    tile stored at level 0.
    """
    tiles_params = proc_params["params"]
    use_gpu = proc_params.get("use_gpu", _USE_GPU)
    levels = proc_params.get("levels", [])
    factor = int(proc_params.get("sharding_factor", 1))

    first = tiles_params[0]["tile_pos"]
    shard_position = (0, int(first[0]) // factor, int(first[1]) // factor)
    level0: dict = {}
    level_tiles: list = [{} for _ in levels]
    for params, vol in loaded_tiles:
        mx, my = (int(v) for v in params["tile_pos"])
        tile_size_tuple = tuple(int(v) for v in params["tile_size"])
        # At native resolution (--resolution -1) the resize is the identity,
        # so the tile is not sent to the GPU and back.
        if tuple(vol.shape) != tile_size_tuple:
            vol = _resize_tile(vol, tile_size_tuple, use_gpu)
        position = (0, mx - shard_position[1] * factor, my - shard_position[2] * factor)
        level0[position] = vol
        for level, tiles in zip(levels, level_tiles, strict=True):
            tiles[position] = _resize_tile(vol, level["tile_size"], use_gpu)

    prepared = [(proc_params["mosaic"].zarray, shard_position, level0)]
    prepared += [(level["array"], shard_position, tiles) for level, tiles in zip(levels, level_tiles, strict=True)]
    return prepared


def _write_shard_through_zarr(array: Any, shard_position: tuple, tiles: dict) -> None:
    """Write the tiles of a shard with one zarr assignment per tile.

    For a grid whose layout :func:`write_shard_chunks` does not handle. Zarr
    compresses the chunks one after the other on a single core.
    """
    chunk = tuple(int(c) for c in array.chunks)
    shard = tuple(int(c) for c in (array.shards or array.chunks))
    for position, tile in tiles.items():
        start = [s * n + p * c for s, n, p, c in zip(shard_position, shard, position, chunk, strict=True)]
        array[tuple(slice(a, a + c) for a, c in zip(start, chunk, strict=True))] = tile


def _write_shard(prepared: list, workers: int | None = None) -> None:
    """Write the prepared tiles of a shard, every pyramid level in its own shard (write stage).

    A tile is one inner chunk of the shard, so the tiles are compressed side
    by side on worker threads and the shard file is written in one go.
    """
    for array, shard_position, tiles in prepared:
        if not write_shard_chunks(array, shard_position, tiles, workers=workers):
            _write_shard_through_zarr(array, shard_position, tiles)


def _resize_and_write_shard(proc_params: dict, loaded_tiles: list) -> None:
    """Resize pre-loaded tiles and write the shard to zarr (compute and write stages)."""
    _write_shard(_prepare_shard(proc_params, loaded_tiles))


def _run_pipelined(params: list) -> None:
    """Process shards with a three-stage pipeline.

    One background thread fetches the tiles of the next shard from disk and
    another compresses and writes the previous shard, while the main thread
    resizes the tiles of the current one on the GPU. The stages of
    successive shards overlap instead of following each other:

        disk(i+1) ║ GPU(i) ║ compress+write(i-1)

    One shard is in each stage at most, so three shards are in memory at the
    peak.
    """
    if not params:
        return

    with ThreadPoolExecutor(max_workers=1) as prefetch_executor, ThreadPoolExecutor(max_workers=1) as write_executor:
        pending_load = prefetch_executor.submit(_load_shard_data, params[0])
        pending_write = None

        for i, p in enumerate(tqdm(params)):
            loaded_tiles = pending_load.result()

            if i + 1 < len(params):
                pending_load = prefetch_executor.submit(_load_shard_data, params[i + 1])

            prepared = _prepare_shard(p, loaded_tiles)
            del loaded_tiles
            if pending_write is not None:
                pending_write.result()
            pending_write = write_executor.submit(_write_shard, prepared)
            del prepared

        if pending_write is not None:
            pending_write.result()


def process_tile(proc_params: dict) -> None:
    """Process a shard: load tiles from disk, resize, write to zarr.

    Used by the CPU multiprocessing pool. For GPU mode the pipelined
    path (_run_pipelined) is preferred to overlap disk I/O with GPU work.
    """
    loaded_tiles = _load_shard_data(proc_params)
    _resize_and_write_shard(proc_params, loaded_tiles)


def main() -> None:
    """Run function."""
    global _USE_GPU

    parser = _build_arg_parser()
    args = parser.parse_args()

    output_resolution = args.resolution
    crop = not args.keep_galvo_return
    fix_galvo_shift = args.fix_galvo_shift
    fix_camera_shift = args.fix_camera_shift
    preprocess = args.preprocess
    galvo_threshold = args.galvo_threshold

    _USE_GPU = args.use_gpu and GPU_AVAILABLE

    if args.verbose:
        print_gpu_info()
        print(f"Using GPU: {_USE_GPU}")
    if args.use_gpu and not GPU_AVAILABLE:
        print("WARNING: GPU requested but not available, falling back to CPU")

    data_type = args.data_type
    angle_index = args.angle_index
    n_cpus = parse_processes_arg(args.n_processes)
    psoct_config = PreprocessingConfig()
    psoct_config.crop_first_index = args.crop_first_index
    psoct_config.crop_second_index = args.crop_second_index
    psoct_config.erase_polarization_1 = args.polarization != 1
    psoct_config.erase_polarization_2 = not psoct_config.erase_polarization_1
    psoct_config.return_complex = args.return_complex

    tiles: list = []
    tiles_pos: list = []
    tiles_directory: Path | None = None
    resolution: list = []
    n_extra: int = 0
    vol: np.ndarray | None = None

    if data_type == "OCT":
        if args.from_root_directory:
            z = args.slice
            tiles_directory = args.from_root_directory
            tiles, tiles_pos = reconstruction.get_tiles_ids(tiles_directory, z=z)
        else:
            if args.slice is not None:
                parser.error("Argument --slice is incompatible with --from_tiles_list.")
            tiles = [Path(d) for d in args.from_tiles_list]
            tiles_pos = reconstruction.get_tiles_ids_from_list(tiles)
    elif data_type == "PSOCT":
        assert tiles_directory is not None
        tiles, tiles_pos = ThorOCT.get_psoct_tiles_ids(tiles_directory, number_of_angles=args.number_of_angles)
        tiles = tiles[angle_index]

    if data_type == "OCT":
        oct = OCT(tiles[0], args.axial_resolution)
        vol = load_oct_tile(oct, crop=crop)
        vol = preprocess_volume(vol, apply=preprocess)
        # oct.resolution is (rz, ry, rx); the grid is (Z, X, Y).
        rz, ry, rx = oct.resolution
        resolution = [rz, rx, ry]
        n_extra = oct.info.get("n_extra", 0)
    elif data_type == "PSOCT":
        oct = ThorOCT(tiles[0], config=psoct_config)
        if psoct_config.erase_polarization_2:
            oct.load()
            vol = oct.first_polarization
        else:
            oct.load()
            vol = oct.second_polarization
        assert vol is not None
        vol = ThorOCT.orient_volume_psoct(vol)
        resolution = [oct.resolution[2], oct.resolution[0], oct.resolution[1]]
        n_extra = 0
    print(f"Resolution: z = {resolution[0]} , x = {resolution[1]} , y = {resolution[2]} ")

    galvo_shift = 0
    if fix_galvo_shift and data_type == "OCT" and n_extra > 0 and args.galvo_shift is not None:
        galvo_shift = max(0, int(args.galvo_shift))
        print(f"Galvo shift given: shift={galvo_shift} - detection skipped")
    elif fix_galvo_shift and data_type == "OCT" and n_extra > 0:
        from linumpy.geometry.galvo import detect_galvo_for_slice

        print(f"Running galvo detection on {len(tiles)} tiles with threshold={galvo_threshold}")
        galvo_shift, confidence = detect_galvo_for_slice(
            tiles, n_extra, threshold=galvo_threshold, axial_resolution=args.axial_resolution
        )
        if galvo_shift > 0:
            print(f"Galvo shift detected: shift={galvo_shift}, confidence={confidence:.3f} - will apply fix")
        else:
            print(f"Galvo shift not significant: confidence={confidence:.3f} - skipping fix")

    pos_xy = np.asarray(tiles_pos)[:, :2]
    pos_xy = pos_xy - np.min(pos_xy, axis=0)
    nb_tiles_xy = np.max(pos_xy, axis=0) + 1

    assert vol is not None
    if output_resolution == -1:
        tile_size = vol.shape
        output_resolution = resolution
    else:
        tile_size = [int(vol.shape[i] * resolution[i] * 1000 / output_resolution) for i in range(3)]
        output_resolution = [output_resolution / 1000.0] * 3
    mosaic_shape = [tile_size[0], nb_tiles_xy[0] * tile_size[1], nb_tiles_xy[1] * tile_size[2]]

    shards = (tile_size[0], args.sharding_factor * tile_size[1], args.sharding_factor * tile_size[2])
    nb_shards_xy = np.ceil(nb_tiles_xy / float(args.sharding_factor)).astype(int)

    writer = OmeZarrWriter(
        args.output_zarr,
        shape=mosaic_shape,
        dtype=np.complex64 if args.return_complex else np.float32,
        chunk_shape=tile_size,
        shards=shards,
        overwrite=True,
    )

    # Further pyramid levels: one array per resolution, with the tile grid and
    # the shard layout of level 0, filled tile by tile along with it.
    levels: list = []
    level_resolutions: list = []
    for level_um in sorted(set(args.pyramid_resolutions)):
        level_res = level_um / 1000.0
        if level_res <= max(output_resolution):
            print(f"Pyramid resolution {level_um} um is not coarser than level 0; skipped.")
            continue
        level_tile = resampled_tile_shape(tile_size, output_resolution, level_res)
        level_shard = (level_tile[0], args.sharding_factor * level_tile[1], args.sharding_factor * level_tile[2])
        level_array = writer.add_level(
            shape=(level_tile[0], nb_tiles_xy[0] * level_tile[1], nb_tiles_xy[1] * level_tile[2]),
            chunk_shape=level_tile,
            shards=level_shard,
        )
        levels.append({"array": level_array, "tile_size": level_tile, "shard_shape": level_shard})
        level_resolutions.append([level_res] * 3)
        print(f"Pyramid level {len(levels)}: {level_um} um, tile {level_tile}, shape {level_array.shape}")

    params_grid = np.full((nb_shards_xy[0], nb_shards_xy[1]), None, dtype=object)
    for i in range(len(tiles)):
        shard_pos = (pos_xy[i] / args.sharding_factor).astype(int)

        if params_grid[shard_pos[0], shard_pos[1]] is None:
            params_grid[shard_pos[0], shard_pos[1]] = {
                "params": [],
                "mosaic": writer,
                "sharding_factor": args.sharding_factor,
                "use_gpu": _USE_GPU,
                "levels": levels,
            }

        params_grid[shard_pos[0], shard_pos[1]]["params"].append(
            {
                "file": tiles[i],
                "tile_pos": pos_xy[i],
                "crop": crop,
                "galvo_shift": galvo_shift,
                "fix_camera_shift": fix_camera_shift,
                "preprocess": preprocess,
                "tile_size": tile_size,
                "data_type": data_type,
                "psoct_config": psoct_config,
            }
        )

    params = [
        params_grid[i, j] for i in range(nb_shards_xy[0]) for j in range(nb_shards_xy[1]) if params_grid[i, j] is not None
    ]

    if n_cpus > 1 and not _USE_GPU:
        from linumpy.config.threads import worker_initializer

        with multiprocessing.Pool(n_cpus, initializer=worker_initializer) as pool:
            results = tqdm(pool.imap(process_tile, params), total=len(params))
            tuple(results)
    else:
        # GPU mode: pipeline disk I/O with GPU compute + zarr write
        _run_pipelined(params)

    if levels:
        writer.finalize_levels([output_resolution, *level_resolutions])
    else:
        writer.finalize(output_resolution, 0)


if __name__ == "__main__":
    main()
