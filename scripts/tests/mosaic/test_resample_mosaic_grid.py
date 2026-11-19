#!/usr/bin/env python3
import numpy as np
import pytest

from linumpy.geometry.resampling import resolution_is_mm
from linumpy.io.test_data import get_data


def test_help(script_runner):
    ret = script_runner.run(["linum-resample-mosaic-grid", "--help"])
    assert ret.success


def test_execute(script_runner, tmp_path):
    input = get_data("mosaic_3d_omezarr")
    output = tmp_path / "test_resample.ome.zarr"
    ret = script_runner.run(["linum-resample-mosaic-grid", input, output])
    assert ret.success


@pytest.mark.parametrize(
    ("source_res", "target_res_um", "expected_target", "expected_scale"),
    [
        # mm-stored source: target is converted to mm so scaling is unit-consistent.
        ((0.005, 0.005, 0.005), 10.0, 10.0 / 1000.0, 0.5),
        # µm-stored source: target stays in µm for scaling parity.
        ((5.0, 5.0, 5.0), 10.0, 10.0, 0.5),
        # Upsampling: µm source, larger voxels requested.
        ((20.0, 20.0, 20.0), 10.0, 10.0, 2.0),
    ],
)
def test_resample_scaling_factor_matches_units(source_res, target_res_um, expected_target, expected_scale):
    """Regression for the GPU-branch unit bug.

    Both paths must use ``resolution_is_mm`` so that ``scaling_factor`` is
    computed in a single unit rather than mixing mm with µm.
    """
    target_res = target_res_um / 1000.0 if resolution_is_mm(source_res) else float(target_res_um)
    assert target_res == pytest.approx(expected_target)
    scaling = np.asarray(source_res) / target_res
    np.testing.assert_allclose(scaling, [expected_scale] * 3)


def test_pipelined_resample_releases_the_tiles_it_has_read(monkeypatch):
    """The read-ahead queue must not keep the tiles that are already resampled.

    Each tile of a real grid is about 0.25 GB; holding them all made one
    process as large as the whole uncompressed grid.
    """
    import gc
    import weakref

    import scripts.mosaic.linum_resample_mosaic_grid as script

    class Tile(np.ndarray):
        """An ndarray that can be weakly referenced."""

    refs: list[weakref.ref] = []
    most_alive = 0

    def read_tile(_vol, _i, _j, tile_shape):
        tile = np.zeros(tile_shape, dtype=np.float32).view(Tile)
        refs.append(weakref.ref(tile))
        return tile

    def rescale(tile, _scale, order=1, use_gpu=True):
        nonlocal most_alive
        gc.collect()
        most_alive = max(most_alive, sum(ref() is not None for ref in refs))
        return np.zeros((1, 1, 1), dtype=np.float32)

    monkeypatch.setattr(script, "_read_tile", read_tile)
    monkeypatch.setattr(script, "rescale", rescale)

    prefetch = 2
    n_tiles = 40
    tile_iter = [(0, j) for j in range(n_tiles)]
    out = np.zeros((1, 1, n_tiles), dtype=np.float32)
    script._run_pipelined(None, out, tile_iter, (2, 2, 2), (1, 1, 1), 0.5, use_gpu=False, prefetch=prefetch)

    # The tile in hand, the previous one still bound to the loop variable, and the read-ahead.
    assert most_alive <= prefetch + 2


def _grid_with_level(path, level_shape, level_chunks, level_res):
    """A two-tile grid at 5 um with a hand-made second level holding a marker value."""
    from linumpy.io.zarr import OmeZarrWriter

    writer = OmeZarrWriter(path, shape=(8, 16, 32), chunk_shape=(8, 16, 16))
    level = writer.add_level(shape=level_shape, chunk_shape=level_chunks)
    rng = np.random.default_rng(0)
    writer[:] = rng.random((8, 16, 32), dtype=np.float32)
    level[:] = np.full(level_shape, 7.0, dtype=np.float32)
    writer.finalize_levels([[0.005, 0.005, 0.005], [level_res] * 3])


def test_existing_level_is_copied_not_resampled(script_runner, tmp_path):
    """A level at the target resolution is taken as it is."""
    from linumpy.io.zarr import read_omezarr

    grid = tmp_path / "grid.ome.zarr"
    _grid_with_level(grid, (4, 8, 16), (4, 8, 8), 0.010)
    output = tmp_path / "out.ome.zarr"
    ret = script_runner.run(["linum-resample-mosaic-grid", grid, output, "-r", 10, "--n_levels", 0, "--no-use_gpu"])
    assert ret.success
    out, res = read_omezarr(output)
    assert out.shape == (4, 8, 16)
    assert out.chunks == (4, 8, 8)
    assert list(res) == [0.01, 0.01, 0.01]
    np.testing.assert_array_equal(np.asarray(out[:]), np.full((4, 8, 16), 7.0, dtype=np.float32))


@pytest.mark.parametrize(
    ("level_shape", "level_chunks", "level_res", "extra_args"),
    [
        # Right resolution, but not the tile grid a resampling would give.
        ((4, 8, 8), (4, 8, 8), 0.010, []),
        # Another resolution.
        ((4, 8, 16), (4, 8, 8), 0.020, []),
        # A usable level, switched off.
        ((4, 8, 16), (4, 8, 8), 0.010, ["--no-use_existing_level"]),
    ],
)
def test_level_that_does_not_fit_is_ignored(script_runner, tmp_path, level_shape, level_chunks, level_res, extra_args):
    """Without a matching level the grid is resampled from level 0 as before."""
    from linumpy.io.zarr import read_omezarr

    grid = tmp_path / "grid.ome.zarr"
    _grid_with_level(grid, level_shape, level_chunks, level_res)
    output = tmp_path / "out.ome.zarr"
    ret = script_runner.run(
        ["linum-resample-mosaic-grid", grid, output, "-r", 10, "--n_levels", 0, "--no-use_gpu", *extra_args]
    )
    assert ret.success
    out, _ = read_omezarr(output)
    assert out.shape == (4, 8, 16)
    data = np.asarray(out[:])
    assert data.max() < 1.0  # resampled random values in [0, 1), not the marker
    assert data.any()
