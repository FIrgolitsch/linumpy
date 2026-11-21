#!/usr/bin/env python3
"""Tests for the three scripts of the pooled beam correction."""

import json

import dask.array as da
import numpy as np

from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.psf.profile import PooledBeamProfile, SlabBeamProfile


def _beam(depth: np.ndarray) -> np.ndarray:
    return 1.0 + 4.0 * np.exp(-np.maximum(depth, 0) / 8.0)


def _write_slab(path, face: int, n_planes: int = 48, size: int = 96) -> np.ndarray:
    """Write a slab with water above *face*, agarose around a block of tissue below it."""
    depth = np.arange(n_planes) - face
    below = depth >= 0
    agarose = np.where(below, 2.0 * _beam(depth), 0.5)
    tissue = np.where(below, 30.0 * _beam(depth) * np.exp(-0.04 * np.maximum(depth, 0)), 0.5)
    vol = np.repeat(agarose[:, None, None], size, axis=1).repeat(size, axis=2).astype(np.float32)
    quarter = size // 4
    vol[:, quarter : 3 * quarter, quarter : 3 * quarter] = tissue[:, None, None]
    save_omezarr(da.from_array(vol), path, (0.01, 0.01, 0.01), n_levels=0)
    return vol


def test_help(script_runner):
    for script in ("linum-estimate-beam-profile", "linum-pool-beam-profiles", "linum-apply-beam-profile"):
        assert script_runner.run([script, "--help"]).success


def test_estimate_pool_and_apply(script_runner, tmp_path):
    faces = {"slice_z01": 6, "slice_z02": 10, "slice_z03": 14, "slice_z04": 18}
    volumes = {}
    for name, face in faces.items():
        volumes[name] = _write_slab(tmp_path / f"{name}.ome.zarr", face)
        ret = script_runner.run(["linum-estimate-beam-profile", tmp_path / f"{name}.ome.zarr", tmp_path / f"{name}.json"])
        assert ret.success, ret.stderr
        assert abs(SlabBeamProfile.load(tmp_path / f"{name}.json").tissue_face - face) <= 2

    pooled_path = tmp_path / "pooled.json"
    ret = script_runner.run(["linum-pool-beam-profiles", pooled_path, *[tmp_path / f"{name}.json" for name in faces]])
    assert ret.success, ret.stderr
    assert "pooled 4 of 4 slabs" in ret.stdout
    pooled = PooledBeamProfile.load(pooled_path)
    assert pooled.n_used == 4

    corrected = {}
    for name in faces:
        out = tmp_path / f"{name}_corrected.ome.zarr"
        ret = script_runner.run(
            [
                "linum-apply-beam-profile",
                tmp_path / f"{name}.ome.zarr",
                out,
                "--pooled_profile",
                pooled_path,
                "--slab_profile",
                tmp_path / f"{name}.json",
                "--n_levels",
                "0",
            ]
        )
        assert ret.success, ret.stderr
        corrected[name] = np.asarray(read_omezarr(out, level=0)[0])
        slab = SlabBeamProfile.load(tmp_path / f"{name}.json")
        divisor = pooled.divisor_for_slab(slab.tissue_face, 48)
        np.testing.assert_allclose(corrected[name], volumes[name] / divisor[:, None, None], rtol=1e-5)
        metrics = json.loads((tmp_path / f"{name}_corrected.ome_metrics.json").read_text())
        assert metrics["step_name"] == "psf_compensation"

    # The agarose has no attenuation of its own, so it comes out flat below the
    # face, and at the same level in every slab whatever the depth of its face.
    levels = []
    for name, face in faces.items():
        agarose = corrected[name][:, 4, 4]
        below = agarose[face + 4 : face + 24]
        assert below.max() / below.min() < 1.15
        levels.append(float(below.mean()))
    assert max(levels) / min(levels) < 1.1


def test_pool_leaves_out_a_slab_and_says_so(script_runner, tmp_path):
    shape = [float(1.0 + 6.0 * np.exp(-abs(d - 2) / 10.0)) for d in range(-8, 32)]
    for i in range(4):
        SlabBeamProfile(profile=shape, tissue_face=8, agarose_face=8, n_columns=5000, coverage=0.3).save(
            tmp_path / f"slice_z{i:02d}.json"
        )
    SlabBeamProfile(profile=[2.0] * 40, tissue_face=8, agarose_face=8, n_columns=20, coverage=0.001).save(
        tmp_path / "slice_z04.json"
    )

    ret = script_runner.run(["linum-pool-beam-profiles", tmp_path / "pooled.json", *sorted(tmp_path.glob("slice_z*.json"))])

    assert ret.success, ret.stderr
    assert "pooled 4 of 5 slabs" in ret.stdout
    assert "left out slice_z04: too little agarose" in ret.stdout
