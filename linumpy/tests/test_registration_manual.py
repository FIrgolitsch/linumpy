"""Tests for the manual-alignment export helpers in linumpy/registration/manual.py.

Covers both the export helpers (``_save_aip_npz``, ``_save_axis_views``, …) and
the pure-Python coordinate/intensity transforms (``apply_transform``,
``apply_scaling``, ``transform_and_rescale_slice``) plus the
``ManualImageCorrection`` GUI class (constructed with the Agg backend so no
display is required).
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from linumpy.registration.manual import (
    _brightest_index,
    _discover_slices,
    _discover_transforms,
    _is_interpolated,
    _read_overlap_z_offsets,
    _save_aip_npz,
    _save_axis_views,
    _save_axis_views_for_pair,
    _save_xy_aips_for_pair,
    _tissue_centroid,
    apply_scaling,
    apply_transform,
    transform_and_rescale_slice,
)

# ---------------------------------------------------------------------------
# _brightest_index
# ---------------------------------------------------------------------------


def test_brightest_index_finds_peak_plane():
    vol = np.zeros((4, 6, 6), dtype=np.float32)
    vol[2] = 1.0  # brightest along axis 0 at index 2
    assert _brightest_index(vol, axis=0) == 2


# ---------------------------------------------------------------------------
# _tissue_centroid
# ---------------------------------------------------------------------------


def test_tissue_centroid_of_flat_profile_is_midpoint():
    profile = np.zeros(10, dtype=np.float32)
    centroid = _tissue_centroid(profile)
    assert centroid == pytest.approx(5.0)


def test_tissue_centroid_weighted_toward_peak():
    profile = np.zeros(10, dtype=np.float32)
    profile[8] = 1.0
    centroid = _tissue_centroid(profile)
    assert centroid == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# _save_aip_npz
# ---------------------------------------------------------------------------


def test_save_aip_npz_round_trip(tmp_path):
    aip = np.random.rand(8, 8).astype(np.float32)
    scale = np.array([1.0, 2.0], dtype=float)
    out_path = tmp_path / "aip.npz"

    _save_aip_npz(aip, scale, out_path, center_pos=3)

    assert out_path.exists()
    loaded = np.load(out_path)
    np.testing.assert_allclose(loaded["aip"], aip)
    np.testing.assert_allclose(loaded["scale"], scale)
    assert int(loaded["center_pos"]) == 3


def test_save_aip_npz_without_center_pos(tmp_path):
    aip = np.zeros((4, 4), dtype=np.float32)
    scale = np.array([1.0, 1.0], dtype=float)
    out_path = tmp_path / "aip_no_center.npz"

    _save_aip_npz(aip, scale, out_path)

    loaded = np.load(out_path)
    assert "center_pos" not in loaded.files


# ---------------------------------------------------------------------------
# _save_axis_views
# ---------------------------------------------------------------------------


def test_save_axis_views_writes_xz_yz_files(tmp_path):
    vol = np.random.rand(4, 8, 8).astype(np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=float)
    aips_xz_dir = tmp_path / "aips_xz"
    aips_yz_dir = tmp_path / "aips_yz"
    aips_xz_dir.mkdir()
    aips_yz_dir.mkdir()

    _save_axis_views(vol, scale, sid=1, aips_xz_dir=aips_xz_dir, aips_yz_dir=aips_yz_dir)

    assert (aips_xz_dir / "slice_z01.npz").exists()
    assert (aips_yz_dir / "slice_z01.npz").exists()


# ---------------------------------------------------------------------------
# _save_xy_aips_for_pair / _save_axis_views_for_pair
# ---------------------------------------------------------------------------


def test_save_xy_aips_for_pair_writes_fixed_and_moving(tmp_path):
    fixed = np.random.rand(6, 8, 8).astype(np.float32)
    moving = np.random.rand(6, 8, 8).astype(np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=float)

    _save_xy_aips_for_pair(fixed, moving, scale, scale, overlap_px=2, fid=0, mid=1, aips_dir=tmp_path)

    assert (tmp_path / "pair_z00_z01_fixed.npz").exists()
    assert (tmp_path / "pair_z00_z01_moving.npz").exists()


def test_save_axis_views_for_pair_writes_xz_yz_files(tmp_path):
    fixed = np.random.rand(6, 8, 8).astype(np.float32)
    moving = np.random.rand(6, 8, 8).astype(np.float32)
    scale = np.array([1.0, 1.0, 1.0], dtype=float)
    aips_xz_dir = tmp_path / "aips_xz"
    aips_yz_dir = tmp_path / "aips_yz"
    aips_xz_dir.mkdir()
    aips_yz_dir.mkdir()

    _save_axis_views_for_pair(
        fixed, moving, scale, scale, fixed_z=5, moving_z=0, fid=0, mid=1, aips_xz_dir=aips_xz_dir, aips_yz_dir=aips_yz_dir
    )

    assert (aips_xz_dir / "pair_z00_z01_fixed.npz").exists()
    assert (aips_xz_dir / "pair_z00_z01_moving.npz").exists()
    assert (aips_yz_dir / "pair_z00_z01_fixed.npz").exists()
    assert (aips_yz_dir / "pair_z00_z01_moving.npz").exists()


# ---------------------------------------------------------------------------
# _is_interpolated
# ---------------------------------------------------------------------------


def test_is_interpolated_detects_suffix():
    from pathlib import Path

    assert _is_interpolated(Path("slice_z01_interpolated.ome.zarr"))
    assert not _is_interpolated(Path("slice_z01.ome.zarr"))


# ---------------------------------------------------------------------------
# _discover_slices / _discover_transforms
# ---------------------------------------------------------------------------


def test_discover_slices_matches_pattern(tmp_path):
    (tmp_path / "slice_z00.ome.zarr").mkdir()
    (tmp_path / "slice_z01_interpolated.ome.zarr").mkdir()
    (tmp_path / "not_a_slice.txt").touch()

    slices = _discover_slices(tmp_path)

    assert set(slices.keys()) == {0, 1}
    assert slices[0].name == "slice_z00.ome.zarr"
    assert slices[1].name == "slice_z01_interpolated.ome.zarr"


def test_discover_transforms_matches_directories(tmp_path):
    (tmp_path / "slice_z01").mkdir()
    (tmp_path / "slice_z02").mkdir()
    (tmp_path / "not_a_dir_marker.txt").touch()

    transforms = _discover_transforms(tmp_path)

    assert set(transforms.keys()) == {1, 2}


# ---------------------------------------------------------------------------
# _read_overlap_z_offsets
# ---------------------------------------------------------------------------


def test_read_overlap_z_offsets_missing_file_returns_zeros(tmp_path):
    assert _read_overlap_z_offsets(tmp_path / "missing.txt") == (0, 0)


def test_read_overlap_z_offsets_reads_values(tmp_path):
    offsets_file = tmp_path / "offsets.txt"
    offsets_file.write_text("3 7\n")
    assert _read_overlap_z_offsets(offsets_file) == (3, 7)


def test_read_overlap_z_offsets_invalid_content_returns_zeros(tmp_path):
    offsets_file = tmp_path / "offsets.txt"
    offsets_file.write_text("not,a,number\n")
    assert _read_overlap_z_offsets(offsets_file) == (0, 0)


# ---------------------------------------------------------------------------
# apply_transform
# ---------------------------------------------------------------------------


def _plane_coords(nz: int, ny: int, nx: int) -> np.ndarray:
    """Build a (nz, ny, 3) coordinate grid for one x-plane (z, y, x triples)."""
    z = np.arange(nz)
    y = np.arange(ny)
    x_idx = nx // 2  # pick a single x column
    zz, yy = np.meshgrid(z, y, indexing="ij")
    xx = np.full_like(zz, x_idx)
    return np.stack([zz, yy, xx], axis=-1).astype(float)


def test_apply_transform_identity_leaves_coordinates_unchanged():
    """Zero translation and zero rotation leave the (z, y, x) grid unchanged.

    apply_transform expects coordinates of shape (nz, ny, 3) — a 2D grid of
    (z, y, x) triples (one plane of the full 3D grid). The z column is
    untouched; the y/x columns are rotated about the image center then
    translated. With identity parameters the center pixel is fixed.
    """
    nz, ny, nx = 3, 5, 5
    coords = _plane_coords(nz, ny, nx)
    out = apply_transform(0.0, 0.0, 0.0, coords.copy())
    # z column unchanged.
    np.testing.assert_allclose(out[..., 0], coords[..., 0])
    # The center pixel is fixed by the rotation.
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    assert out[1, 2, 1] == pytest.approx(cy, abs=1e-9)
    assert out[1, 2, 2] == pytest.approx(cx, abs=1e-9)


def test_apply_transform_translation_shifts_y_and_x():
    """A pure translation shifts only the y and x coordinates."""
    nz, ny, nx = 2, 4, 4
    coords = _plane_coords(nz, ny, nx)
    out = apply_transform(1.5, -2.0, 0.0, coords.copy())
    np.testing.assert_allclose(out[..., 0], coords[..., 0])  # z unchanged
    np.testing.assert_allclose(out[..., 1], coords[..., 1] + 1.5)
    np.testing.assert_allclose(out[..., 2], coords[..., 2] - 2.0)


def test_apply_transform_per_slice_translation_array():
    """Per-slice translation arrays (shape (nz,)) are applied elementwise."""
    nz, ny, nx = 3, 4, 4
    coords = _plane_coords(nz, ny, nx)
    ty = np.array([0.0, 1.0, 2.0])
    tx = np.array([0.0, -1.0, -2.0])
    theta = np.zeros(nz)
    out = apply_transform(ty, tx, theta, coords.copy())
    # Slice 0: no shift; slice 1: +1 y, -1 x; slice 2: +2 y, -2 x.
    np.testing.assert_allclose(out[0, :, 1], coords[0, :, 1])
    np.testing.assert_allclose(out[1, :, 1], coords[1, :, 1] + 1.0)
    np.testing.assert_allclose(out[2, :, 2], coords[2, :, 2] - 2.0)


def test_apply_transform_rotation_180_swaps_corners():
    """A 180° rotation maps the top-left corner to the bottom-right (minus center).

    Uses a (1, ny*nx, 3) coordinate grid where both y and x vary across the
    second axis, so the rotation swaps both y and x corners.
    """
    ny, nx = 5, 5
    y, x = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    z = np.zeros_like(y)
    coords = np.stack([z, y, x], axis=-1).astype(float)  # shape (ny, nx, 3)
    coords = coords.reshape(1, ny * nx, 3)  # (1, ny*nx, 3)
    out = apply_transform(0.0, 0.0, np.pi, coords.copy())
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0
    # Corner (y=0, x=0) -> (y=ny-1, x=nx-1) relative to center.
    assert out[0, 0, 1] == pytest.approx(cy + (cy - 0), abs=1e-6)
    assert out[0, 0, 2] == pytest.approx(cx + (cx - 0), abs=1e-6)


# ---------------------------------------------------------------------------
# apply_scaling
# ---------------------------------------------------------------------------


def test_apply_scaling_scalar_range_to_unit():
    """A scalar (vmin, vmax) range rescales data to [0, 1]."""
    data = np.array([0.0, 5.0, 10.0])
    out = apply_scaling(data, 0.0, 10.0)
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])


def test_apply_scaling_clips_outside_range():
    """Values outside (vmin, vmax) are clipped before rescaling."""
    data = np.array([-5.0, 5.0, 15.0])
    out = apply_scaling(data, 0.0, 10.0)
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0])


def test_apply_scaling_per_row_ranges():
    """Per-row (vmin, vmax) arrays rescale each row independently.

    Note: ``apply_scaling`` broadcasts ``vmin``/``vmax`` against the LAST axis
    of ``data`` (numpy default). The docstring says "the first dimension of
    data should correspond to the number of elements in vmin, vmax" but the
    actual broadcasting is on the last axis. The test uses data shaped so the
    last axis matches the vmin/vmax length.
    """
    data = np.array([[0.0, 0.0], [10.0, 20.0]])  # shape (2, 2)
    vmin = np.array([0.0, 0.0])  # length 2 == data.shape[-1]
    vmax = np.array([10.0, 20.0])
    out = apply_scaling(data, vmin, vmax)
    np.testing.assert_allclose(out, [[0.0, 0.0], [1.0, 1.0]])


def test_apply_scaling_zero_range_returns_zeros():
    """A zero-width range (vmin == vmax) returns zeros (no division by zero)."""
    data = np.array([5.0, 5.0, 5.0])
    out = apply_scaling(data, 5.0, 5.0)
    np.testing.assert_allclose(out, [0.0, 0.0, 0.0])


def test_apply_scaling_per_row_zero_range_safe():
    """A per-row zero-width range is handled by the np.where safe_range guard."""
    data = np.array([[5.0, 0.0], [5.0, 10.0]])  # shape (2, 2)
    vmin = np.array([5.0, 0.0])  # col 0: zero range, col 1: normal
    vmax = np.array([5.0, 10.0])
    out = apply_scaling(data, vmin, vmax)
    # Column 0: zero range -> zeros; column 1: normal rescale.
    np.testing.assert_allclose(out[:, 0], [0.0, 0.0])
    np.testing.assert_allclose(out[:, 1], [0.0, 1.0])


# ---------------------------------------------------------------------------
# transform_and_rescale_slice
# ---------------------------------------------------------------------------


def test_transform_and_rescale_slice_identity_preserves_image():
    """Identity transform + full-range rescale preserves the input slice."""
    slice_ = np.arange(16, dtype=np.float32).reshape(4, 4)
    out = transform_and_rescale_slice(slice_, ty=0.0, tx=0.0, theta=0.0, vmin=0.0, vmax=15.0)
    assert out.shape == slice_.shape
    # Identity transform maps each pixel back to itself (within interpolation).
    np.testing.assert_allclose(out, slice_ / 15.0, atol=1e-6)


def test_transform_and_rescale_slice_translation_shifts_content():
    """A y-translation shifts the image content along the first axis.

    ``apply_transform`` adds ``ty`` to the sampling y-coordinate, so
    ``output[y] = input[y + ty]``. A positive ``ty`` moves content toward
    smaller y (the bright row at y=0 appears at y=-ty, which is out of bounds
    and becomes 0; the row that was at y=ty appears at y=0).
    """
    slice_ = np.zeros((6, 6), dtype=np.float32)
    slice_[2, :] = 1.0  # bright row at y=2
    out = transform_and_rescale_slice(slice_, ty=2.0, tx=0.0, theta=0.0, vmin=0.0, vmax=1.0)
    # output[0] = input[0 + 2] = input[2] = bright.
    assert out[0, 0] > 0.9
    # output[2] = input[4] = dark.
    assert out[2, 0] < 0.1


def test_transform_and_rescale_slice_clips_and_rescales():
    """Values outside (vmin, vmax) are clipped and rescaled to [0, 1]."""
    slice_ = np.array([[0.0, 5.0, 20.0]], dtype=np.float32)
    out = transform_and_rescale_slice(slice_, ty=0.0, tx=0.0, theta=0.0, vmin=5.0, vmax=15.0)
    # 0 -> clipped to 5 -> 0.0; 5 -> 0.0; 20 -> clipped to 15 -> 1.0.
    np.testing.assert_allclose(out[0], [0.0, 0.0, 1.0], atol=1e-6)


# ---------------------------------------------------------------------------
# ManualImageCorrection (GUI class, Agg backend)
# ---------------------------------------------------------------------------


def test_manual_image_correction_constructs_with_valid_inputs():
    """The GUI class builds its interpolator and slider state without a display."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from linumpy.registration.manual import ManualImageCorrection

    nz, ny, nx = 4, 8, 8
    data = np.random.default_rng(0).random((nz, ny, nx)).astype(np.float32) * 100.0
    corr = ManualImageCorrection(data, resolution=(1.0, 1.0, 1.0), downsample_factor=1)
    assert corr.transforms.shape == (nz, 3)
    assert corr.custom_ranges.shape == (nz, 2)
    assert corr.current_z == 0
    plt.close("all")


def test_manual_image_correction_rejects_bad_transforms_shape():
    """A transforms array with the wrong shape raises ValueError."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from linumpy.registration.manual import ManualImageCorrection

    data = np.random.default_rng(0).random((4, 8, 8)).astype(np.float32)
    bad_transforms = np.zeros((3, 3))  # should be (4, 3)
    with pytest.raises(ValueError, match="Invalid shape for transforms"):
        ManualImageCorrection(
            data,
            resolution=(1.0, 1.0, 1.0),
            downsample_factor=1,
            transforms=bad_transforms,
        )
    plt.close("all")


def test_manual_image_correction_rejects_bad_custom_ranges_shape():
    """A custom_ranges array with the wrong shape raises ValueError."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from linumpy.registration.manual import ManualImageCorrection

    data = np.random.default_rng(0).random((4, 8, 8)).astype(np.float32)
    bad_ranges = np.zeros((3, 2))  # should be (4, 2)
    with pytest.raises(ValueError, match="Invalid shape for custom ranges"):
        ManualImageCorrection(
            data,
            resolution=(1.0, 1.0, 1.0),
            downsample_factor=1,
            custom_ranges=bad_ranges,
        )
    plt.close("all")


def test_manual_image_correction_save_results_round_trip(tmp_path):
    """save_results writes an npz that round-trips transforms + custom_ranges."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from linumpy.registration.manual import ManualImageCorrection

    nz, ny, nx = 4, 8, 8
    data = np.random.default_rng(0).random((nz, ny, nx)).astype(np.float32) * 100.0
    corr = ManualImageCorrection(data, resolution=(1.0, 1.0, 1.0), downsample_factor=1)
    # Mutate the transforms so the saved file is non-trivial.
    corr.transforms[1, 0] = 2.0
    corr.transforms[2, 2] = 0.1
    out_path = tmp_path / "corrections.npz"
    corr.save_results(out_path)
    assert out_path.exists()
    loaded = np.load(out_path)
    np.testing.assert_allclose(loaded["transforms"], corr.transforms)
    np.testing.assert_allclose(loaded["custom_ranges"], corr.custom_ranges)
    plt.close("all")


def test_manual_image_correction_get_view_a_returns_2d():
    """get_view_a returns a 2D YZ image of the expected shape."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from linumpy.registration.manual import ManualImageCorrection

    nz, ny, nx = 4, 8, 8
    data = np.random.default_rng(0).random((nz, ny, nx)).astype(np.float32) * 100.0
    corr = ManualImageCorrection(data, resolution=(1.0, 1.0, 1.0), downsample_factor=1)
    view_a = corr.get_view_a()
    assert view_a.ndim == 2
    assert view_a.shape == (nz, ny)
    plt.close("all")


def test_manual_image_correction_get_view_b_returns_2d():
    """get_view_b returns a 2D XZ image (transposed) of the expected shape."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from linumpy.registration.manual import ManualImageCorrection

    nz, ny, nx = 4, 8, 8
    data = np.random.default_rng(0).random((nz, ny, nx)).astype(np.float32) * 100.0
    corr = ManualImageCorrection(data, resolution=(1.0, 1.0, 1.0), downsample_factor=1)
    view_b = corr.get_view_b()
    assert view_b.ndim == 2
    # get_view_b returns data.T, so shape is (nx, nz).
    assert view_b.shape == (nx, nz)
    plt.close("all")


def test_manual_image_correction_get_view_c_returns_rgb():
    """get_view_c returns a 3-channel RGB XY image clipped to [0, 1]."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from linumpy.registration.manual import ManualImageCorrection

    nz, ny, nx = 4, 8, 8
    data = np.random.default_rng(0).random((nz, ny, nx)).astype(np.float32) * 100.0
    corr = ManualImageCorrection(data, resolution=(1.0, 1.0, 1.0), downsample_factor=1)
    view_c = corr.get_view_c()
    assert view_c.ndim == 3
    assert view_c.shape[2] == 3
    assert view_c.min() >= 0.0
    assert view_c.max() <= 1.0
    plt.close("all")


def test_manual_image_correction_on_change_z_updates_state():
    """on_change_z updates current_z and the slider values."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from linumpy.registration.manual import ManualImageCorrection

    nz, ny, nx = 4, 8, 8
    data = np.random.default_rng(0).random((nz, ny, nx)).astype(np.float32) * 100.0
    corr = ManualImageCorrection(data, resolution=(1.0, 1.0, 1.0), downsample_factor=1)
    corr.on_change_z(2.0)
    assert corr.current_z == 2
    plt.close("all")


def test_manual_image_correction_on_change_offset_a_updates_transforms():
    """on_change_offset_a writes the new y-offset into the transforms array."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from linumpy.registration.manual import ManualImageCorrection

    nz, ny, nx = 4, 8, 8
    data = np.random.default_rng(0).random((nz, ny, nx)).astype(np.float32) * 100.0
    corr = ManualImageCorrection(data, resolution=(1.0, 1.0, 1.0), downsample_factor=1)
    corr.on_change_offset_a(3.5)
    assert corr.transforms[0, 0] == pytest.approx(3.5)
    plt.close("all")


def test_manual_image_correction_on_change_theta_updates_transforms():
    """on_change_theta writes the new rotation into the transforms array."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from linumpy.registration.manual import ManualImageCorrection

    nz, ny, nx = 4, 8, 8
    data = np.random.default_rng(0).random((nz, ny, nx)).astype(np.float32) * 100.0
    corr = ManualImageCorrection(data, resolution=(1.0, 1.0, 1.0), downsample_factor=1)
    corr.on_change_theta(0.05)
    assert corr.transforms[0, 2] == pytest.approx(0.05)
    plt.close("all")


def test_manual_image_correction_on_change_ref_z_sets_mode():
    """on_change_ref_z updates the ref_z_mode label."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from linumpy.registration.manual import (
        NEXT_REF_LABEL,
        ManualImageCorrection,
    )

    nz, ny, nx = 4, 8, 8
    data = np.random.default_rng(0).random((nz, ny, nx)).astype(np.float32) * 100.0
    corr = ManualImageCorrection(data, resolution=(1.0, 1.0, 1.0), downsample_factor=1)
    corr.on_change_ref_z(NEXT_REF_LABEL)
    assert corr.ref_z_mode == NEXT_REF_LABEL
    plt.close("all")


def test_manual_image_correction_transform_coordinates_single_z():
    """transform_coordinates with z= applies only that slice's transform."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from linumpy.registration.manual import ManualImageCorrection

    nz, ny, nx = 3, 4, 4
    data = np.random.default_rng(0).random((nz, ny, nx)).astype(np.float32) * 100.0
    corr = ManualImageCorrection(data, resolution=(1.0, 1.0, 1.0), downsample_factor=1)
    corr.transforms[1, 0] = 5.0  # y-shift for slice 1
    coords = corr.grid_coordinates[1, :, :, :].copy()
    out = corr.transform_coordinates(coords, z=1)
    # y column shifted by 5.
    np.testing.assert_allclose(out[..., 1], coords[..., 1] + 5.0)
    plt.close("all")


def test_manual_image_correction_apply_scaling_per_slice():
    """apply_scaling with z= uses the per-slice (vmin, vmax) range."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from linumpy.registration.manual import ManualImageCorrection

    nz, ny, nx = 3, 4, 4
    data = np.random.default_rng(0).random((nz, ny, nx)).astype(np.float32) * 100.0
    corr = ManualImageCorrection(data, resolution=(1.0, 1.0, 1.0), downsample_factor=1)
    # Force a known range for slice 1.
    corr.custom_ranges[1] = [0.0, 50.0]
    sample = np.array([0.0, 25.0, 50.0, 100.0])
    out = corr.apply_scaling(sample, z=1)
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0, 1.0])
    plt.close("all")


def test_manual_image_correction_draw_cursor_marks_current_z():
    """draw_cursor sets bright pixels at the current_z row edges."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from linumpy.registration.manual import ManualImageCorrection

    nz, ny, nx = 4, 8, 8
    data = np.random.default_rng(0).random((nz, ny, nx)).astype(np.float32) * 100.0
    corr = ManualImageCorrection(data, resolution=(1.0, 1.0, 1.0), downsample_factor=1)
    corr.current_z = 2
    view = np.zeros((nz, ny), dtype=np.float32)
    out = corr.draw_cursor(view)
    cursor_len = int(0.02 * ny)
    assert np.all(out[2, :cursor_len] == 1.0)
    assert np.all(out[2, -cursor_len:] == 1.0)
    plt.close("all")
