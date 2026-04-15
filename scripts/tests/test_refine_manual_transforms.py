#!/usr/bin/env python3
import json

import numpy as np
import SimpleITK as sitk
import zarr


def _make_zarr_slice(path, shape=(10, 32, 32)):
    """Write a tiny OME-Zarr volume filled with random data."""
    store = zarr.storage.LocalStore(str(path))
    root = zarr.open_group(store, mode="w")
    data = (np.random.rand(*shape) * 255).astype(np.uint16)
    arr = root.create_array("0", shape=shape, chunks=shape, dtype=np.uint16)
    arr[:] = data
    root.attrs["multiscales"] = [
        {
            "axes": [
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [10.0, 10.0, 10.0]}]}],
            "version": "0.4",
        }
    ]


def _make_transform(path, tx=0.0, ty=0.0, rot_deg=0.0, cx=16.0, cy=16.0):
    """Write a trivial Euler3DTransform .tfm file."""
    tfm = sitk.Euler3DTransform()
    tfm.SetFixedParameters([cx, cy, 0.0, 0.0])
    tfm.SetParameters([0.0, 0.0, float(np.radians(rot_deg)), tx, ty, 0.0])
    sitk.WriteTransform(tfm, str(path))


def test_help(script_runner):
    ret = script_runner.run(["linum_refine_manual_transforms.py", "--help"])
    assert ret.success


def test_run_no_manual_transforms(tmp_path, script_runner):
    """Without any manual transforms every pair is copied unchanged."""
    slices_dir = tmp_path / "slices"
    slices_dir.mkdir()
    auto_dir = tmp_path / "auto"
    auto_dir.mkdir()
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    out_dir = tmp_path / "out"

    for sid in (4, 5):
        _make_zarr_slice(slices_dir / f"slice_z{sid:02d}.ome.zarr")

    # Create an automated transform for slice z05 (the moving slice)
    pair_dir = auto_dir / "slice_z05_normalize"
    pair_dir.mkdir()
    _make_transform(pair_dir / "transform.tfm")
    np.savetxt(str(pair_dir / "offsets.txt"), [8, 2], fmt="%d")
    (pair_dir / "pairwise_registration_metrics.json").write_text(json.dumps({"source": "auto"}))

    ret = script_runner.run(
        [
            "linum_refine_manual_transforms.py",
            str(slices_dir),
            str(auto_dir),
            str(out_dir),
            "--manual_transforms_dir",
            str(manual_dir),
        ]
    )
    assert ret.success, ret.stderr
    assert (out_dir / "slice_z05_normalize" / "transform.tfm").exists()


def test_run_with_manual_transform(tmp_path, script_runner):
    """With a manual transform the pair is refined and output written."""
    slices_dir = tmp_path / "slices"
    slices_dir.mkdir()
    auto_dir = tmp_path / "auto"
    auto_dir.mkdir()
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    out_dir = tmp_path / "out"

    for sid in (4, 5):
        _make_zarr_slice(slices_dir / f"slice_z{sid:02d}.ome.zarr")

    pair_dir = auto_dir / "slice_z05_normalize"
    pair_dir.mkdir()
    _make_transform(pair_dir / "transform.tfm")
    np.savetxt(str(pair_dir / "offsets.txt"), [8, 2], fmt="%d")

    manual_pair = manual_dir / "slice_z05"
    manual_pair.mkdir()
    _make_transform(manual_pair / "transform.tfm", tx=1.0, ty=0.5)

    ret = script_runner.run(
        [
            "linum_refine_manual_transforms.py",
            str(slices_dir),
            str(auto_dir),
            str(out_dir),
            "--manual_transforms_dir",
            str(manual_dir),
        ]
    )
    assert ret.success, ret.stderr
    out_tfm = out_dir / "slice_z05_normalize" / "transform.tfm"
    assert out_tfm.exists()
    metrics = json.loads((out_dir / "slice_z05_normalize" / "pairwise_registration_metrics.json").read_text())
    assert metrics["source"] == "manual_refined"


def test_overwrite_guard(tmp_path, script_runner):
    """Running twice without -f should fail; with -f should succeed."""
    slices_dir = tmp_path / "slices"
    slices_dir.mkdir()
    auto_dir = tmp_path / "auto"
    auto_dir.mkdir()
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    out_dir = tmp_path / "out"
    out_dir.mkdir()  # pre-create to trigger guard

    _make_zarr_slice(slices_dir / "slice_z04.ome.zarr")
    _make_zarr_slice(slices_dir / "slice_z05.ome.zarr")
    pair_dir = auto_dir / "slice_z05_normalize"
    pair_dir.mkdir()
    _make_transform(pair_dir / "transform.tfm")

    base_args = [
        "linum_refine_manual_transforms.py",
        str(slices_dir),
        str(auto_dir),
        str(out_dir),
        "--manual_transforms_dir",
        str(manual_dir),
    ]

    ret = script_runner.run(base_args)
    assert not ret.success, "should fail without -f when out_dir exists"

    ret = script_runner.run([*base_args, "-f"])
    assert ret.success, ret.stderr
