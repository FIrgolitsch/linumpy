#!/usr/bin/env python3
"""Tests for linum-add-mosaic-grid-level."""

import json
from pathlib import Path

import numpy as np

from linumpy.io.test_data import get_data
from linumpy.io.zarr import read_omezarr, read_omezarr_resolutions

TILES = ["tile_x00_y00_z01", "tile_x01_y00_z01", "tile_x00_y01_z01"]


def _make_grid(script_runner, path, *extra):
    input_path = Path(get_data("raw_tiles"))
    ret = script_runner.run(
        [
            "linum-create-mosaic-grid-3d",
            path,
            "--from_tiles_list",
            *[str(input_path / t) for t in TILES],
            "-r",
            -1,
            "--no-preprocess",
            "--no-fix_galvo_shift",
            "--sharding_factor",
            2,
            *extra,
        ]
    )
    assert ret.success, ret.stderr


def test_help(script_runner):
    assert script_runner.run(["linum-add-mosaic-grid-level", "--help"]).success


def test_added_level_is_the_one_the_grid_script_writes(script_runner, tmp_path):
    """A level added afterwards equals the level written with the grid, one tile of the grid missing."""
    with_level = tmp_path / "with_level.ome.zarr"
    plain = tmp_path / "plain.ome.zarr"
    _make_grid(script_runner, with_level, "--pyramid_resolutions", 10)
    _make_grid(script_runner, plain)
    assert len(read_omezarr_resolutions(plain)) == 1
    level0_before = np.asarray(read_omezarr(plain, level=0)[0][:])

    ret = script_runner.run(["linum-add-mosaic-grid-level", plain, "-r", 10, "--no-use_gpu"])
    assert ret.success, ret.stderr
    assert "3 of 4 tiles written" in ret.stdout

    assert read_omezarr_resolutions(plain) == read_omezarr_resolutions(with_level)
    added, _ = read_omezarr(plain, level=1)
    expected, _ = read_omezarr(with_level, level=1)
    assert added.shape == expected.shape
    assert added.chunks == expected.chunks
    assert added.shards == expected.shards
    np.testing.assert_array_equal(np.asarray(added[:]), np.asarray(expected[:]))
    # The tile that was not acquired has no data at the new level either.
    assert not np.asarray(added[:, 14:, 14:]).any()
    # Level 0 is as it was, and the two grids describe their levels the same way.
    np.testing.assert_array_equal(np.asarray(read_omezarr(plain, level=0)[0][:]), level0_before)
    described = [json.loads((g / "zarr.json").read_text())["attributes"]["ome"]["multiscales"][0] for g in (plain, with_level)]
    assert described[0] == described[1]

    # The 3D reconstruction now takes the level instead of resampling the grid.
    ret = script_runner.run(["linum-resample-mosaic-grid", plain, tmp_path / "resampled.ome.zarr", "-r", 10, "--n_levels", 0])
    assert ret.success
    assert "already at the target resolution" in ret.stdout


def test_a_grid_that_has_the_level_is_left_alone(script_runner, tmp_path):
    grid = tmp_path / "grid.ome.zarr"
    _make_grid(script_runner, grid, "--pyramid_resolutions", 10)
    before = (grid / "zarr.json").read_text()

    ret = script_runner.run(["linum-add-mosaic-grid-level", grid, "-r", 10, "--no-use_gpu"])

    assert ret.success, ret.stderr
    assert "nothing to do" in ret.stdout
    assert (grid / "zarr.json").read_text() == before
    assert len(read_omezarr_resolutions(grid)) == 2


def test_a_level_left_by_an_interrupted_run_is_replaced(script_runner, tmp_path):
    grid = tmp_path / "grid.ome.zarr"
    _make_grid(script_runner, grid)
    # An array of the next level that the metadata does not list.
    (grid / "1").mkdir()
    (grid / "1" / "junk").write_text("left over")

    ret = script_runner.run(["linum-add-mosaic-grid-level", grid, "-r", 10, "--no-use_gpu"])

    assert ret.success, ret.stderr
    assert not (grid / "1" / "junk").exists()
    assert len(read_omezarr_resolutions(grid)) == 2
    assert np.asarray(read_omezarr(grid, level=1)[0][:]).any()


def test_level_copied_from_a_resampled_grid(script_runner, tmp_path):
    """Tiles taken from the output of linum-resample-mosaic-grid give the level the script resamples itself."""
    resampled_here = tmp_path / "resampled_here.ome.zarr"
    copied = tmp_path / "copied.ome.zarr"
    _make_grid(script_runner, resampled_here)
    _make_grid(script_runner, copied)
    resampled = tmp_path / "resampled.ome.zarr"
    ret = script_runner.run(["linum-resample-mosaic-grid", copied, resampled, "-r", 10, "--n_levels", 0, "--no-use_gpu"])
    assert ret.success, ret.stderr
    ret = script_runner.run(["linum-add-mosaic-grid-level", resampled_here, "-r", 10, "--no-use_gpu"])
    assert ret.success, ret.stderr
    level0_before = np.asarray(read_omezarr(copied, level=0)[0][:])

    ret = script_runner.run(["linum-add-mosaic-grid-level", copied, "-r", 10, "--from_resampled", resampled])

    assert ret.success, ret.stderr
    assert "3 of 4 tiles written" in ret.stdout
    assert "(copied)" in ret.stdout
    added, _ = read_omezarr(copied, level=1)
    expected, _ = read_omezarr(resampled_here, level=1)
    assert added.chunks == expected.chunks
    assert added.shards == expected.shards
    np.testing.assert_array_equal(np.asarray(added[:]), np.asarray(expected[:]))
    np.testing.assert_array_equal(np.asarray(read_omezarr(copied, level=0)[0][:]), level0_before)
    assert read_omezarr_resolutions(copied) == read_omezarr_resolutions(resampled_here)


def test_a_resampled_grid_of_another_size_is_refused(script_runner, tmp_path):
    grid = tmp_path / "grid.ome.zarr"
    _make_grid(script_runner, grid)
    other = tmp_path / "other.ome.zarr"
    ret = script_runner.run(["linum-resample-mosaic-grid", grid, other, "-r", 20, "--n_levels", 0, "--no-use_gpu"])
    assert ret.success, ret.stderr
    before = (grid / "zarr.json").read_text()

    ret = script_runner.run(["linum-add-mosaic-grid-level", grid, "-r", 10, "--from_resampled", other])

    assert not ret.success
    assert "has voxels of" in ret.stderr
    assert (grid / "zarr.json").read_text() == before
    assert len(read_omezarr_resolutions(grid)) == 1
