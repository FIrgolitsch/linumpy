#!/usr/bin/env python3
import importlib.util
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import zarr.storage


def _load_script_module():
    """Import scripts/stitching/linum_refine_manual_transforms.py as a module."""
    script_path = Path(__file__).resolve().parents[2] / "stitching" / "linum_refine_manual_transforms.py"
    spec = importlib.util.spec_from_file_location("linum_refine_manual_transforms", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    ret = script_runner.run(["linum-refine-manual-transforms", "--help"])
    assert ret.success


def test_run_no_manual_transforms(tmp_path, script_runner):
    """Without any manual transforms the pair is copied unchanged."""
    fixed_zarr = tmp_path / "slice_z04.ome.zarr"
    moving_zarr = tmp_path / "slice_z05.ome.zarr"
    auto_dir = tmp_path / "auto_transforms"
    manual_dir = tmp_path / "manual"
    out_dir = tmp_path / "out"

    _make_zarr_slice(fixed_zarr)
    _make_zarr_slice(moving_zarr)
    manual_dir.mkdir()

    auto_dir.mkdir()
    _make_transform(auto_dir / "transform.tfm")
    np.savetxt(str(auto_dir / "offsets.txt"), [8, 2], fmt="%d")
    (auto_dir / "pairwise_registration_metrics.json").write_text(json.dumps({"source": "auto"}))

    ret = script_runner.run(
        [
            "linum-refine-manual-transforms",
            str(fixed_zarr),
            str(moving_zarr),
            str(auto_dir),
            str(out_dir),
            "--manual_transforms_dir",
            str(manual_dir),
        ]
    )
    assert ret.success, ret.stderr
    assert (out_dir / "transform.tfm").exists()


def test_run_with_manual_transform(tmp_path, script_runner):
    """With a manual transform the pair is refined and output written."""
    fixed_zarr = tmp_path / "slice_z04.ome.zarr"
    moving_zarr = tmp_path / "slice_z05.ome.zarr"
    auto_dir = tmp_path / "auto_transforms"
    manual_dir = tmp_path / "manual"
    out_dir = tmp_path / "out"

    _make_zarr_slice(fixed_zarr)
    _make_zarr_slice(moving_zarr)

    auto_dir.mkdir()
    _make_transform(auto_dir / "transform.tfm")
    np.savetxt(str(auto_dir / "offsets.txt"), [8, 2], fmt="%d")

    manual_pair = manual_dir / "slice_z05"
    manual_pair.mkdir(parents=True)
    _make_transform(manual_pair / "transform.tfm", tx=1.0, ty=0.5)

    ret = script_runner.run(
        [
            "linum-refine-manual-transforms",
            str(fixed_zarr),
            str(moving_zarr),
            str(auto_dir),
            str(out_dir),
            "--manual_transforms_dir",
            str(manual_dir),
        ]
    )
    assert ret.success, ret.stderr
    assert (out_dir / "transform.tfm").exists()
    metrics = json.loads((out_dir / "pairwise_registration_metrics.json").read_text())
    assert metrics["source"] in ("manual_refined", "manual")
    assert "reject_reason" in metrics["refinement"]
    assert "ncc_before" in metrics["refinement"]


def test_no_manual_z_keeps_registration_offsets(tmp_path, script_runner):
    """--no-manual-z writes the pairwise Z, not the Z saved in the manual tool."""
    fixed_zarr = tmp_path / "slice_z04.ome.zarr"
    moving_zarr = tmp_path / "slice_z05.ome.zarr"
    auto_dir = tmp_path / "auto_transforms"
    manual_dir = tmp_path / "manual"
    out_dir = tmp_path / "out"

    _make_zarr_slice(fixed_zarr)
    _make_zarr_slice(moving_zarr)

    auto_dir.mkdir()
    _make_transform(auto_dir / "transform.tfm")
    np.savetxt(str(auto_dir / "offsets.txt"), [8, 2], fmt="%d")

    manual_pair = manual_dir / "slice_z05"
    manual_pair.mkdir(parents=True)
    _make_transform(manual_pair / "transform.tfm", tx=1.0, ty=0.5)
    np.savetxt(str(manual_pair / "offsets.txt"), [1, 0], fmt="%d")

    ret = script_runner.run(
        [
            "linum-refine-manual-transforms",
            str(fixed_zarr),
            str(moving_zarr),
            str(auto_dir),
            str(out_dir),
            "--manual_transforms_dir",
            str(manual_dir),
            "--no-manual-z",
            "-f",
        ]
    )
    assert ret.success, ret.stderr
    saved = np.loadtxt(str(out_dir / "offsets.txt"), dtype=int)
    assert list(saved) == [8, 2]


def test_overwrite_guard(tmp_path, script_runner):
    """Running twice without -f should fail; with -f should succeed."""
    fixed_zarr = tmp_path / "slice_z04.ome.zarr"
    moving_zarr = tmp_path / "slice_z05.ome.zarr"
    auto_dir = tmp_path / "auto_transforms"
    manual_dir = tmp_path / "manual"
    out_dir = tmp_path / "out"
    out_dir.mkdir()  # pre-create to trigger guard

    _make_zarr_slice(fixed_zarr)
    _make_zarr_slice(moving_zarr)
    manual_dir.mkdir()
    auto_dir.mkdir()
    _make_transform(auto_dir / "transform.tfm")

    base_args = [
        "linum-refine-manual-transforms",
        str(fixed_zarr),
        str(moving_zarr),
        str(auto_dir),
        str(out_dir),
        "--manual_transforms_dir",
        str(manual_dir),
    ]

    ret = script_runner.run(base_args)
    assert not ret.success, "should fail without -f when out_dir exists"

    ret = script_runner.run([*base_args, "-f"])
    assert ret.success, ret.stderr


def _apply_rigid_2d(tx, ty, rot_deg, cx, cy, point):
    """Evaluate a 2D rigid transform T(p) = R (p - c) + c + t."""
    theta = np.radians(rot_deg)
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    c = np.array([cx, cy])
    t = np.array([tx, ty])
    return r @ (np.asarray(point) - c) + c + t


def test_accept_residual_requires_improvement():
    module = _load_script_module()
    assert module._accept_residual(0.50, 0.51, 1e-4)
    assert not module._accept_residual(0.50, 0.50, 1e-4)
    assert not module._accept_residual(0.50, 0.40, 1e-4)
    assert not module._accept_residual(float("nan"), 0.9, 1e-4)


def test_overlap_aips_uses_edge_slabs():
    module = _load_script_module()
    fixed = np.zeros((6, 4, 4), dtype=np.float32)
    moving = np.zeros((6, 4, 4), dtype=np.float32)
    fixed[-2:] = 3.0
    moving[:2] = 5.0
    fa, ma = module._overlap_aips(fixed, moving, overlap_px=2)
    assert fa.shape == (4, 4)
    assert float(fa.mean()) == 3.0
    assert float(ma.mean()) == 5.0


def test_compose_rigid_2d_matches_point_evaluation():
    """Closed-form composition must match explicit per-point evaluation.

    This is the regression for the old additive composition
    ``final = man + delta`` which is only valid when the manual rotation
    centre coincides with the image centre. Here the manual centre is
    deliberately off-centre so the additive formula would disagree with the
    explicit composition at every corner.
    """
    module = _load_script_module()

    # Image: 200 (W) x 160 (H); manual rotation centre at (W/4, H/4).
    w, h = 200, 160
    final_cx, final_cy = w / 2.0, h / 2.0
    man_tx, man_ty, man_rot = 3.5, -2.0, 1.5
    man_cx, man_cy = w / 4.0, h / 4.0
    delta_tx, delta_ty, delta_rot = 0.2, 0.1, 0.05

    tx, ty, rot = module._compose_rigid_2d(
        man_tx, man_ty, man_rot, man_cx, man_cy, delta_tx, delta_ty, delta_rot, final_cx, final_cy
    )

    # θ_final = θ_manual + θ_delta for 2D planar rotations.
    assert rot == pytest_approx(man_rot + delta_rot)

    # Evaluate at each image corner and compare against explicit
    # T_delta(T_manual(p)).
    corners = [(0.0, 0.0), (w, 0.0), (0.0, h), (w, h)]
    for p in corners:
        p_manual = _apply_rigid_2d(man_tx, man_ty, man_rot, man_cx, man_cy, p)
        expected = _apply_rigid_2d(delta_tx, delta_ty, delta_rot, final_cx, final_cy, p_manual)
        got = _apply_rigid_2d(tx, ty, rot, final_cx, final_cy, p)
        assert np.allclose(got, expected, atol=1e-6), f"mismatch at {p}: got={got}, expected={expected}"


def test_compose_rigid_2d_reduces_to_sum_when_centres_match():
    """When all centres equal c, the composition collapses to additive params."""
    module = _load_script_module()
    c = (50.0, 50.0)
    man_tx, man_ty, man_rot = 1.0, -0.5, 2.0
    delta_tx, delta_ty, delta_rot = -0.3, 0.8, 0.25
    tx, ty, rot = module._compose_rigid_2d(
        man_tx,
        man_ty,
        man_rot,
        c[0],
        c[1],
        delta_tx,
        delta_ty,
        delta_rot,
        c[0],
        c[1],
    )
    # Rotate the manual translation by the delta rotation, then add delta.
    theta = np.radians(delta_rot)
    r = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    expected_t = r @ np.array([man_tx, man_ty]) + np.array([delta_tx, delta_ty])
    assert np.allclose((tx, ty), expected_t, atol=1e-6)
    assert rot == pytest_approx(man_rot + delta_rot)


def test_folding_identity_fixed_keeps_manual_then_delta():
    """inv(I) ∘ delta ∘ I ∘ manual equals manual then delta."""
    module = _load_script_module()
    center = (40.0, 50.0)
    man = (3.0, -2.0, 1.5, center[0], center[1])
    delta = (0.4, -0.2, 0.1)
    chained = module._compose_rigid_2d(*man, 0.0, 0.0, 0.0, center[0], center[1])
    with_delta = module._compose_rigid_2d(*chained, center[0], center[1], *delta, center[0], center[1])
    direct = module._compose_rigid_2d(*man, *delta, center[0], center[1])
    assert np.allclose(with_delta, direct, atol=1e-6)


# Local approx helper to avoid importing pytest.approx at module scope.
def pytest_approx(expected, rel=1e-6, abs_=1e-6):
    import pytest

    return pytest.approx(expected, rel=rel, abs=abs_)


def test_joined_aips_pairs_planes_one_cut_apart():
    module = _load_script_module()
    fixed = np.zeros((9, 4, 4), dtype=np.float32)
    moving = np.zeros((6, 4, 4), dtype=np.float32)
    fixed[4:6] = 3.0  # the two planes just below one cut thickness
    fixed[6:] = 9.0  # deeper planes the edge-slab AIP would have used
    moving[:2] = 5.0
    fa, ma = module._joined_aips(fixed, moving, step=4, overlap_px=2)
    assert float(fa.mean()) == 3.0
    assert float(ma.mean()) == 5.0


def test_joined_step_counts_the_missing_slices():
    module = _load_script_module()
    assert module._joined_step(0.2, 0.01, 4, 5) == 20
    # z21 is missing between z20 and z22: two cuts between the cut faces.
    assert module._joined_step(0.19, 0.01, 20, 22) == 38
    assert module._joined_step(0.2, 0.01, None, 5) == 20


def test_joined_aips_shrinks_to_the_planes_that_exist():
    module = _load_script_module()
    fixed = np.arange(6, dtype=np.float32)[:, None, None] * np.ones((6, 2, 2), dtype=np.float32)
    moving = np.ones((6, 2, 2), dtype=np.float32)
    fa, _ma = module._joined_aips(fixed, moving, step=4, overlap_px=5)
    assert float(fa.mean()) == 4.5  # planes 4 and 5 only


def test_joined_aips_falls_back_to_edge_slabs_when_fixed_is_too_shallow():
    module = _load_script_module()
    fixed = np.zeros((4, 2, 2), dtype=np.float32)
    moving = np.zeros((4, 2, 2), dtype=np.float32)
    fixed[-2:] = 3.0
    moving[:2] = 5.0
    fa, ma = module._joined_aips(fixed, moving, step=4, overlap_px=2)
    assert float(fa.mean()) == 3.0
    assert float(ma.mean()) == 5.0


def _write_volume(path, data, scale_mm=0.01):
    store = zarr.storage.LocalStore(str(path))
    root = zarr.open_group(store, mode="w")
    arr = root.create_array("0", shape=data.shape, chunks=data.shape, dtype=np.float32)
    arr[:] = data
    root.attrs["multiscales"] = [
        {
            "axes": [
                {"name": "z", "type": "space", "unit": "millimeter"},
                {"name": "y", "type": "space", "unit": "millimeter"},
                {"name": "x", "type": "space", "unit": "millimeter"},
            ],
            "datasets": [{"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [scale_mm] * 3}]}],
            "version": "0.4",
        }
    ]


def _blob(seed, shape=(160, 160)):
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    mask = ((yy - 70) / 45.0) ** 2 + ((xx - 85) / 55.0) ** 2 < 1.0
    texture = gaussian_filter(rng.random(shape), 3.0)
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    return ((0.2 + 0.8 * texture) * mask).astype(np.float32)


def test_joined_planes_and_coarse_search_recover_a_stale_manual(tmp_path, script_runner):
    """A deep fixed slab and a manual 30+ px off: the joined planes still line up."""
    from scipy.ndimage import shift as nd_shift

    module = _load_script_module()
    shared = _blob(1)
    fixed = np.stack([_blob(2)] * 20 + [shared] * 10 + [_blob(3)] * 20)  # 50 planes, cut step 20
    moved = nd_shift(shared, (-22.0, 31.0), order=1)
    moving = np.stack([moved] * 10 + [_blob(4)] * 10)

    fixed_zarr = tmp_path / "slice_z04.ome.zarr"
    moving_zarr = tmp_path / "slice_z05.ome.zarr"
    _write_volume(fixed_zarr, fixed)
    _write_volume(moving_zarr, moving)

    auto_dir = tmp_path / "auto_transforms"
    auto_dir.mkdir()
    _make_transform(auto_dir / "transform.tfm", cx=80.0, cy=80.0)
    np.savetxt(str(auto_dir / "offsets.txt"), [45, 4], fmt="%d")
    manual_pair = tmp_path / "manual" / "slice_z05"
    manual_pair.mkdir(parents=True)
    _make_transform(manual_pair / "transform.tfm", cx=80.0, cy=80.0)

    base = [
        "linum-refine-manual-transforms",
        str(fixed_zarr),
        str(moving_zarr),
        str(auto_dir),
        "--manual_transforms_dir",
        str(tmp_path / "manual"),
        "--no-manual-z",
        "--overlap_px",
        "10",
        "-f",
    ]

    def _ncc_after(out_dir, extra):
        ret = script_runner.run([*base[:4], str(out_dir), *base[4:], *extra])
        assert ret.success, ret.stderr
        tx, ty, rot, cx, cy = module._load_manual_transform(out_dir / "transform.tfm")
        warped = module._warp_moving(moved, tx, ty, rot, cx, cy)
        mask = (shared > 0.01) & (warped > 0.01)
        return float(np.corrcoef(shared[mask], warped[mask])[0, 1]), json.loads(
            (out_dir / "pairwise_registration_metrics.json").read_text()
        )

    # Edge slabs compare unrelated planes and the bound is 10 px: nothing moves.
    ncc_edge, meta_edge = _ncc_after(tmp_path / "edge", [])
    assert ncc_edge < 0.5
    assert meta_edge["source"] == "manual"

    ncc_joined, meta = _ncc_after(tmp_path / "joined", ["--slicing_interval_mm", "0.2", "--coarse_max_translation_px", "60"])
    assert ncc_joined > 0.9
    assert meta["source"] == "manual_refined"
    assert meta["refinement"]["coarse_tx"] == pytest_approx(31.0, abs_=1.5)
    assert meta["refinement"]["coarse_ty"] == pytest_approx(-22.0, abs_=1.5)
