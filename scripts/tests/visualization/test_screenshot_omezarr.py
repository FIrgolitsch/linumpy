#!/usr/bin/env python3
import numpy as np

from linumpy.io.zarr import OmeZarrWriter


def test_help(script_runner):
    ret = script_runner.run(["linum-screenshot-omezarr", "--help"])
    assert ret.success


def test_coarsest_level_is_drawn_without_reading_level_0(script_runner, tmp_path):
    """--level -1 draws the last pyramid level; level 0 is not touched.

    Level 0 of a mosaic grid at full resolution is tens of GB, so its preview
    has to come from the coarse level stored next to it.
    """
    import shutil

    grid = tmp_path / "grid.ome.zarr"
    writer = OmeZarrWriter(grid, shape=(8, 16, 16), chunk_shape=(8, 8, 8))
    level = writer.add_level(shape=(4, 8, 8), chunk_shape=(4, 4, 4))
    rng = np.random.default_rng(0)
    writer[:] = rng.random((8, 16, 16), dtype=np.float32)
    level[:] = rng.random((4, 8, 8), dtype=np.float32)
    writer.finalize_levels([[0.005, 0.005, 0.005], [0.01, 0.01, 0.01]])
    # With the chunks of level 0 gone, only a read of level 1 can succeed.
    shutil.rmtree(grid / "0" / "c")

    output = tmp_path / "preview.png"
    ret = script_runner.run(["linum-screenshot-omezarr", grid, output, "--level", -1])
    assert ret.success
    assert output.stat().st_size > 0
