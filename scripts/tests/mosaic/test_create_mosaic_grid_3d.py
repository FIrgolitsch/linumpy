#!/usr/bin/env python3
from pathlib import Path
from typing import Any, cast

import pytest
import zarr

from linumpy.io.test_data import get_data
from linumpy.microscope.oct import OCT


def _assert_omezarr_zyx_contract(zarr_path: Path, tile_dir: Path) -> None:
    """Output mosaic is (depth, tile-x direction, tile-y direction), with the voxel size in that order.

    The OME-Zarr axes are named z, y, x by position. The first in-plane axis
    of the grid is the one the ``tile_xAA`` index counts along, which is the
    a-line (X) direction of a tile.
    """
    oct = OCT(tile_dir)
    vol = oct.load_image(crop=True, fix_galvo_shift=False, fix_camera_shift=False)
    tile_nz, tile_ny, tile_nx = vol.shape
    rz, ry, rx = oct.resolution

    root = zarr.open_group(zarr_path, mode="r")
    mosaic = root["0"]
    assert len(mosaic.shape) == 3
    assert mosaic.shape[0] == tile_nz
    assert mosaic.shape[1] % tile_nx == 0
    assert mosaic.shape[2] % tile_ny == 0

    ome_meta = cast(dict[str, Any], root.attrs["ome"])
    multiscales = cast(list[dict[str, Any]], ome_meta["multiscales"])
    axes = [axis["name"] for axis in multiscales[0]["axes"]]
    assert axes == ["z", "y", "x"]
    scale = multiscales[0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert scale == [rz, rx, ry]


def test_help(script_runner):
    ret = script_runner.run(["linum-create-mosaic-grid-3d", "--help"])
    assert ret.success


def test_execution_from_directory(script_runner, tmp_path):
    input = get_data("raw_tiles")
    output = tmp_path / "output.ome.zarr"
    ret = script_runner.run(
        [
            "linum-create-mosaic-grid-3d",
            output,
            "--from_root_directory",
            input,
            "-z",
            0,
            "-r",
            -1,
            "--no-preprocess",
        ]
    )
    assert ret.success
    _assert_omezarr_zyx_contract(output, Path(input) / "tile_x00_y00_z00")


def test_execution_from_list(script_runner, tmp_path):
    input = get_data("raw_tiles")
    input_path = Path(input)
    output = tmp_path / "output.ome.zarr"
    ret = script_runner.run(
        [
            "linum-create-mosaic-grid-3d",
            output,
            "--from_tiles_list",
            str(input_path / "tile_x00_y00_z01"),
            str(input_path / "tile_x01_y00_z01"),
            str(input_path / "tile_x00_y01_z01"),
            str(input_path / "tile_x01_y01_z01"),
            "-r",
            -1,
            "--no-preprocess",
        ]
    )
    assert ret.success
    _assert_omezarr_zyx_contract(output, input_path / "tile_x00_y00_z01")


def test_native_resolution_keeps_the_voxels(script_runner, tmp_path):
    """At --resolution -1 the tiles are stored as loaded, with no resampling."""
    import numpy as np

    input_path = Path(get_data("raw_tiles"))
    output = tmp_path / "output.ome.zarr"
    tiles = ["tile_x00_y00_z01", "tile_x01_y00_z01", "tile_x00_y01_z01", "tile_x01_y01_z01"]
    ret = script_runner.run(
        [
            "linum-create-mosaic-grid-3d",
            output,
            "--from_tiles_list",
            *[str(input_path / t) for t in tiles],
            "-r",
            -1,
            "--no-preprocess",
            "--no-fix_galvo_shift",
        ]
    )
    assert ret.success
    mosaic = cast(zarr.Array, zarr.open_group(output, mode="r")["0"])
    for name, (ix, iy) in zip(tiles, [(0, 0), (1, 0), (0, 1), (1, 1)], strict=True):
        vol = OCT(input_path / name).load_image(crop=True, fix_galvo_shift=False, fix_camera_shift=False)
        _, ny, nx = vol.shape
        stored = np.asarray(mosaic[:, ix * nx : (ix + 1) * nx, iy * ny : (iy + 1) * ny])
        # The loader gives (Z, Y, X); the grid holds each tile as (Z, X, Y).
        np.testing.assert_array_equal(stored, vol.transpose(0, 2, 1))


def test_tile_axes_follow_the_tile_grid(script_runner, tmp_path):
    """Inside a tile the a-lines run along the grid axis that the tile_x index counts along.

    Tile ``tile_xAA_yBB`` is placed at AA along the first in-plane axis and BB
    along the second. Its a-lines (X) must run along the first and its b-scans
    (Y) along the second, or each tile is transposed in its cell and
    neighbouring tiles do not continue into each other. Tiles that are not
    square, with different pixel sizes in X and Y, make a swap visible in the
    shape, the content and the voxel size.
    """
    import numpy as np

    from linumpy.io.test_data import write_synthetic_oct_tile

    nx, ny, nz = 12, 8, 4
    raw = tmp_path / "raw"
    raw.mkdir()
    tiles = {}
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    for mx in range(2):
        for my in range(2):
            vol_zyx = ((2 * mx + my + 1) * 1e5 + zz * 1e4 + yy * 100 + xx).astype(np.float32)
            made = write_synthetic_oct_tile(tmp_path / f"src_{mx}_{my}", nx=nx, ny=ny, nz=nz, vol_zyx=vol_zyx)
            info = (made / "info.txt").read_text()
            # Field of view in whole microns, as the microscope writes it, with a
            # b-scan step twice the a-line step: 1 um in X, 2 um in Y.
            info = info.replace(f"width: {float(nx)}", f"width: {nx}").replace(f"height: {float(ny)}", f"height: {2 * ny}")
            (made / "info.txt").write_text(info)
            tile_dir = raw / f"tile_x{mx:02d}_y{my:02d}_z00"
            made.rename(tile_dir)
            tiles[mx, my] = (tile_dir, vol_zyx)

    output = tmp_path / "grid.ome.zarr"
    ret = script_runner.run(
        [
            "linum-create-mosaic-grid-3d",
            output,
            "--from_tiles_list",
            *[str(tile_dir) for tile_dir, _ in tiles.values()],
            "-r",
            -1,
            "--no-preprocess",
            "--no-fix_galvo_shift",
        ]
    )
    assert ret.success

    root = zarr.open_group(output, mode="r")
    mosaic = cast(zarr.Array, root["0"])
    assert mosaic.shape == (nz, 2 * nx, 2 * ny)
    for (mx, my), (_, vol_zyx) in tiles.items():
        stored = np.asarray(mosaic[:, mx * nx : (mx + 1) * nx, my * ny : (my + 1) * ny])
        np.testing.assert_array_equal(stored, vol_zyx.transpose(0, 2, 1))

    ome_meta = cast(dict[str, Any], root.attrs["ome"])
    scale = ome_meta["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]
    assert scale == pytest.approx([0.0035, 0.001, 0.002])


def test_pyramid_level_is_the_resampled_grid(script_runner, tmp_path):
    """The extra level written with the grid equals a resampling of level 0 afterwards.

    The 3D reconstruction takes the level in place of its own resampling, so
    the two have to agree to the voxel, in values, shape, tile size and voxel size.
    """
    import numpy as np

    from linumpy.io.zarr import read_omezarr, read_omezarr_resolutions

    input_path = Path(get_data("raw_tiles"))
    tiles = ["tile_x00_y00_z01", "tile_x01_y00_z01", "tile_x00_y01_z01", "tile_x01_y01_z01"]
    grid = tmp_path / "grid.ome.zarr"
    ret = script_runner.run(
        [
            "linum-create-mosaic-grid-3d",
            grid,
            "--from_tiles_list",
            *[str(input_path / t) for t in tiles],
            "-r",
            -1,
            "--no-preprocess",
            "--no-fix_galvo_shift",
            "--sharding_factor",
            2,
            "--pyramid_resolutions",
            10,
        ]
    )
    assert ret.success
    _assert_omezarr_zyx_contract(grid, input_path / tiles[0])

    resolutions = read_omezarr_resolutions(grid)
    assert len(resolutions) == 2
    assert resolutions[1] == [0.01, 0.01, 0.01]
    level0, _ = read_omezarr(grid, level=0)
    level1, _ = read_omezarr(grid, level=1)
    # 35 x 140 x 140 voxels of 3.5 x 1 x 1 um per tile, two tiles each way.
    assert level0.shape == (35, 280, 280)
    assert level1.shape == (12, 28, 28)
    assert level1.chunks == (12, 14, 14)
    assert np.asarray(level1[:]).any()

    # A real resampling of level 0 ...
    resampled = tmp_path / "resampled.ome.zarr"
    ret = script_runner.run(
        ["linum-resample-mosaic-grid", grid, resampled, "-r", 10, "--n_levels", 0, "--no-use_existing_level"]
    )
    assert ret.success
    expected, expected_res = read_omezarr(resampled)
    np.testing.assert_array_equal(np.asarray(level1[:]), np.asarray(expected[:]))

    # ... and the copy the script makes by default give the same file content.
    copied = tmp_path / "copied.ome.zarr"
    ret = script_runner.run(["linum-resample-mosaic-grid", grid, copied, "-r", 10, "--n_levels", 0])
    assert ret.success
    assert "already at the target resolution" in ret.stdout
    out, out_res = read_omezarr(copied)
    assert out.shape == expected.shape
    assert out.chunks == expected.chunks
    assert list(out_res) == list(expected_res)
    np.testing.assert_array_equal(np.asarray(out[:]), np.asarray(expected[:]))


def test_pyramid_resolution_not_coarser_than_the_grid_is_skipped(script_runner, tmp_path):
    """Asking for the resolution the grid already has adds no level."""
    from linumpy.io.zarr import read_omezarr_resolutions

    input_path = Path(get_data("raw_tiles"))
    grid = tmp_path / "grid.ome.zarr"
    ret = script_runner.run(
        [
            "linum-create-mosaic-grid-3d",
            grid,
            "--from_tiles_list",
            str(input_path / "tile_x00_y00_z01"),
            "-r",
            10,
            "--no-preprocess",
            "--pyramid_resolutions",
            10,
        ]
    )
    assert ret.success
    assert read_omezarr_resolutions(grid) == [[0.01, 0.01, 0.01]]


def test_sparse_grid_over_several_shards(script_runner, tmp_path):
    """Tiles land in their own cell and shard; cells without a tile stay empty.

    A 3 x 3 grid with two tiles per shard side has four shards, the last ones
    partly outside the grid. Three tiles are missing, among them the first of
    a shard and all of another shard.
    """
    import numpy as np

    from linumpy.io.test_data import write_synthetic_oct_tile

    nx, ny, nz = 12, 8, 4
    missing = {(0, 0), (1, 2), (2, 2)}
    raw = tmp_path / "raw"
    raw.mkdir()
    tiles = {}
    zz, yy, xx = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    for mx in range(3):
        for my in range(3):
            if (mx, my) in missing:
                continue
            vol_zyx = ((3 * mx + my + 1) * 1e5 + zz * 1e4 + yy * 100 + xx).astype(np.float32)
            made = write_synthetic_oct_tile(tmp_path / f"src_{mx}_{my}", nx=nx, ny=ny, nz=nz, vol_zyx=vol_zyx)
            # Field of view in whole microns, as the microscope writes it.
            info = (made / "info.txt").read_text()
            info = info.replace(f"width: {float(nx)}", f"width: {nx}").replace(f"height: {float(ny)}", f"height: {ny}")
            (made / "info.txt").write_text(info)
            tile_dir = raw / f"tile_x{mx:02d}_y{my:02d}_z00"
            made.rename(tile_dir)
            tiles[mx, my] = (tile_dir, vol_zyx)

    output = tmp_path / "grid.ome.zarr"
    ret = script_runner.run(
        [
            "linum-create-mosaic-grid-3d",
            output,
            "--from_tiles_list",
            *[str(tile_dir) for tile_dir, _ in tiles.values()],
            "-r",
            -1,
            "--no-preprocess",
            "--no-fix_galvo_shift",
            "--sharding_factor",
            2,
            "--no-use_gpu",
            "--n_processes",
            1,
        ]
    )
    assert ret.success, ret.stderr

    mosaic = cast(zarr.Array, zarr.open_group(output, mode="r")["0"])
    assert mosaic.shape == (nz, 3 * nx, 3 * ny)
    assert mosaic.shards == (nz, 2 * nx, 2 * ny)
    for mx in range(3):
        for my in range(3):
            stored = np.asarray(mosaic[:, mx * nx : (mx + 1) * nx, my * ny : (my + 1) * ny])
            if (mx, my) in missing:
                assert not stored.any()
            else:
                np.testing.assert_array_equal(stored, tiles[mx, my][1].transpose(0, 2, 1))
    # Shard (1, 1) holds the cell (2, 2) only, which has no tile: no file for it.
    shard_files = sorted(str(f.relative_to(output / "0" / "c")) for f in (output / "0" / "c").rglob("*") if f.is_file())
    assert shard_files == ["0/0/0", "0/0/1", "0/1/0"]


def test_shard_written_through_zarr_matches_the_direct_writer(tmp_path):
    """The path for layouts the direct writer declines stores the same tiles."""
    import importlib.util

    import numpy as np

    from linumpy.io.chunk_writer import write_shard_chunks

    spec = importlib.util.spec_from_file_location(
        "create_grid", Path(__file__).resolve().parents[2] / "mosaic" / "linum_create_mosaic_grid_3d.py"
    )
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)

    rng = np.random.default_rng(2)
    chunk, shard = (4, 6, 5), (4, 12, 10)
    tiles = {(0, 0, 1): rng.random(chunk).astype(np.float32), (0, 1, 0): rng.random(chunk).astype(np.float32)}
    direct = zarr.create_array(tmp_path / "direct.zarr", shape=(4, 24, 20), chunks=chunk, shards=shard, dtype="float32")
    through_zarr = zarr.create_array(tmp_path / "zarr.zarr", shape=(4, 24, 20), chunks=chunk, shards=shard, dtype="float32")

    assert write_shard_chunks(direct, (0, 1, 1), tiles)
    script._write_shard_through_zarr(through_zarr, (0, 1, 1), tiles)

    np.testing.assert_array_equal(direct[:], through_zarr[:])
    np.testing.assert_array_equal(direct[:, 12:18, 15:20], tiles[(0, 0, 1)])
    # An array without shards goes through zarr as a whole.
    plain = zarr.create_array(tmp_path / "plain.zarr", shape=(4, 24, 20), chunks=chunk, dtype="float32")
    script._write_shard([(plain, (0, 3, 3), {(0, 0, 0): tiles[(0, 0, 1)]})])
    np.testing.assert_array_equal(plain[:, 18:24, 15:20], tiles[(0, 0, 1)])
