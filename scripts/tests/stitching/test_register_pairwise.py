#!/usr/bin/env python3
import json

import dask.array as da
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as nd_shift

from linumpy.io.zarr import save_omezarr


def _slab_stack(n_planes: int = 80, size: int = 160, seed: int = 0) -> np.ndarray:
    """Textured tissue whose planes differ, so a plane can be told from its neighbours."""
    rng = np.random.default_rng(seed)
    # Structure a few planes deep, as in tissue.
    texture = gaussian_filter(rng.normal(size=(n_planes, size, size)), (3.0, 3.0, 3.0))
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    yy, xx = np.mgrid[:size, :size]
    mask = ((yy - 80) / 55.0) ** 2 + ((xx - 85) / 60.0) ** 2 < 1.0
    return ((0.3 + 0.7 * texture) * mask[None]).astype(np.float32)


def _write(path, volume):
    save_omezarr(da.from_array(volume), path, voxel_size=(0.01, 0.01, 0.01), chunks=(16, 80, 80), n_levels=1)


def _run(script_runner, tmp_path, fixed, moving, *extra):
    _write(tmp_path / "fixed.ome.zarr", fixed)
    _write(tmp_path / "moving.ome.zarr", moving)
    out = tmp_path / "out"
    ret = script_runner.run(
        [
            "linum-register-pairwise",
            tmp_path / "fixed.ome.zarr",
            tmp_path / "moving.ome.zarr",
            out,
            "--slicing_interval_mm",
            0.2,
            "--search_range_mm",
            0.1,
            "--moving_z_index",
            4,
            *extra,
        ]
    )
    assert ret.success, ret.stderr
    metrics = json.loads((out / "pairwise_registration_metrics.json").read_text())["metrics"]
    offsets = [int(v) for v in (out / "offsets.txt").read_text().split()]
    return metrics, offsets


def test_help(script_runner):
    ret = script_runner.run(["linum-register-pairwise", "--help"])
    assert ret.success


def test_matching_plane_is_counted_from_the_top_of_a_deep_slab(script_runner, tmp_path):
    """The next slab starts one slice thickness below the top of this one, whatever its depth.

    The fixed slab is 60 planes deep. Plane 4 of the moving slab is plane 24
    of the fixed one. Counted from the bottom the search would cover planes 34
    to 54 and miss it.
    """
    stack = _slab_stack()
    metrics, offsets = _run(script_runner, tmp_path, stack[:60], stack[20:60])
    assert offsets == [24, 4]
    assert metrics["z_drift"]["value"] == 0
    assert metrics["chosen_candidate"]["value"] == "identity"
    assert metrics["translation_magnitude"]["value"] == 0.0
    assert metrics["structure_match_identity"]["value"] > 0.8


def test_a_real_offset_is_corrected(script_runner, tmp_path):
    stack = _slab_stack(seed=1)
    moving = np.stack([nd_shift(p, (-5.0, 8.0), order=1, mode="constant", cval=0.0) for p in stack[20:60]])
    metrics, offsets = _run(script_runner, tmp_path, stack[:60], moving)
    assert offsets == [24, 4]
    assert metrics["chosen_candidate"]["value"] != "identity"
    assert abs(metrics["translation_x"]["value"] - 8.0) < 1.5
    assert abs(metrics["translation_y"]["value"] + 5.0) < 1.5
    assert metrics["structure_match"]["value"] > metrics["structure_match_identity"]["value"] + 0.2
