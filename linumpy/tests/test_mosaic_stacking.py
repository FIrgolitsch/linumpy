"""Tests for linumpy/mosaic/stacking.py"""

from typing import Any

import numpy as np
import pytest

from linumpy.mosaic.stacking import (
    apply_2d_transform,
    apply_overlap_z_gain,
    apply_rigid_euler_padded,
    apply_xy_shift,
    blend_cut_face,
    blend_overlap_xy,
    blend_overlap_z,
    crop_moving_volume,
    enforce_z_consistency,
    estimate_overlap_z_gain_fit,
    estimate_z_blend_xy_shift,
    expected_z_overlap,
    extract_overlap_rois,
    find_z_overlap,
    fit_overlap_log_ratio,
    flatten_slice_z_profile,
    inplane_output_point,
    notch_slice_seams,
    overlap_z_gain_curve,
    overlap_z_profiles,
    paste_tissue,
    refine_z_blend_overlap,
    registration_z_overlap,
    scale_slice_to_median,
    seam_overlap_gain,
    slice_tissue_median,
    suppress_z_slice_bands,
)


def _make_vol(shape=(10, 32, 32), fill=1.0):
    return (np.ones(shape) * fill).astype(np.float32)


# ---------------------------------------------------------------------------
# find_z_overlap
# ---------------------------------------------------------------------------


def test_registration_z_overlap_keeps_cut_face_and_clamps():
    # Template plane 4 meets previous plane 8. Cut face stays, so the overlap
    # is (prev_nz - fixed_z) plus the template index.
    assert registration_z_overlap(56, 80, fixed_z=8, template_z=4) == 52
    # Incoming slab is shorter than that overlap. Do not ask the blend for
    # planes the slice does not have.
    assert registration_z_overlap(56, 50, fixed_z=8, template_z=4) == 50
    assert registration_z_overlap(20, 50, fixed_z=4, template_z=4) == 20


def test_find_z_overlap_returns_tuple():
    fixed = _make_vol((20, 16, 16))
    moving = _make_vol((20, 16, 16))
    overlap, corr = find_z_overlap(fixed, moving, slicing_interval_mm=0.1, search_range_mm=0.05, resolution_um=5.0)
    assert isinstance(overlap, int)
    assert isinstance(corr, (float, np.floating))


def test_find_z_overlap_identical_volumes():
    """Identical volumes have perfect correlation at some overlap."""
    rng = np.random.default_rng(0)
    vol = rng.random((20, 16, 16)).astype(np.float32)
    overlap, corr = find_z_overlap(vol, vol, slicing_interval_mm=0.05, search_range_mm=0.1, resolution_um=5.0)
    # Correlation should be high (>= 0.0 at minimum)
    assert corr >= 0.0
    assert 1 <= overlap <= 20


def test_find_z_overlap_min_max_degenerate():
    """When search range collapses, falls back to expected overlap."""
    fixed = _make_vol((10, 8, 8))
    moving = _make_vol((10, 8, 8))
    # Very large interval → expected overlap < 0 → min >= max edge case
    overlap, _corr = find_z_overlap(fixed, moving, slicing_interval_mm=10.0, search_range_mm=0.0, resolution_um=5.0)
    assert isinstance(overlap, int)


# ---------------------------------------------------------------------------
# apply_xy_shift
# ---------------------------------------------------------------------------


def test_apply_xy_shift_zero_shift():
    vol = _make_vol((4, 10, 10))
    cropped, dst = apply_xy_shift(vol, 0.0, 0.0, output_shape=(10, 10))
    assert cropped is not None
    assert dst == (0, 10, 0, 10)


def test_apply_xy_shift_positive():
    vol = _make_vol((4, 8, 8))
    cropped, dst = apply_xy_shift(vol, 2.0, 3.0, output_shape=(12, 12))
    # dest starts at (dy=3, dx=2) in (y_start, y_end, x_start, x_end)
    assert dst[0] == 3  # y_start
    assert dst[2] == 2  # x_start
    assert cropped.shape[1] == 8
    assert cropped.shape[2] == 8


def test_apply_xy_shift_negative_clips_src():
    vol = _make_vol((4, 10, 10))
    # Shift by -2 in both dims: source crops 2 from start
    cropped, dst = apply_xy_shift(vol, -2.0, -2.0, output_shape=(10, 10))
    assert cropped is not None
    assert dst[0] == 0  # clamped to canvas start
    assert cropped.shape[1] == 8  # 2 rows clipped


def test_apply_rigid_euler_padded_shifts_without_clipping():
    import SimpleITK as sitk

    vol = np.zeros((1, 32, 32), dtype=np.float32)
    vol[0, 16, 16] = 1.0
    tfm = sitk.Euler3DTransform()
    tfm.SetCenter([16.0, 16.0, 0.0])
    tfm.SetTranslation([8.0, 0.0, 0.0])
    padded, pad_x, pad_y = apply_rigid_euler_padded(vol, tfm)
    peak = np.unravel_index(int(np.argmax(padded[0])), padded[0].shape)
    assert peak[0] == 16 + pad_y
    assert peak[1] == 16 - 8 + pad_x
    assert padded.shape[2] > 32


def test_seam_overlap_gain_matches_medians():
    existing = np.full((4, 32, 32), 2.0, dtype=np.float32)
    moving = np.full((4, 32, 32), 1.0, dtype=np.float32)
    assert seam_overlap_gain(existing, moving) == pytest.approx(1.4)


def test_seam_overlap_gain_is_one_without_shared_tissue():
    existing = np.zeros((4, 8, 8), dtype=np.float32)
    moving = np.ones((4, 8, 8), dtype=np.float32)
    assert seam_overlap_gain(existing, moving) == 1.0


def test_apply_xy_shift_fully_outside_canvas():
    vol = _make_vol((4, 8, 8))
    cropped, dst = apply_xy_shift(vol, 100.0, 100.0, output_shape=(10, 10))
    assert cropped is None
    assert dst is None


# ---------------------------------------------------------------------------
# blend_overlap_z
# ---------------------------------------------------------------------------


def test_blend_overlap_z_output_shape():
    fixed = _make_vol((5, 8, 8), fill=1.0)
    moving = _make_vol((5, 8, 8), fill=2.0)
    result = blend_overlap_z(fixed, moving)
    assert result.shape == fixed.shape


def test_blend_overlap_z_both_valid():
    """With both regions non-zero, result is between fixed and moving."""
    fixed = np.ones((6, 8, 8), dtype=np.float32)
    moving = np.full((6, 8, 8), 3.0, dtype=np.float32)
    result = blend_overlap_z(fixed, moving)
    assert float(result.min()) >= 1.0
    assert float(result.max()) <= 3.0


def test_blend_overlap_z_one_sided_fixed_only():
    """When moving is zero, fixed values are preserved."""
    fixed = np.ones((6, 8, 8), dtype=np.float32)
    moving = np.zeros((6, 8, 8), dtype=np.float32)
    result = blend_overlap_z(fixed, moving)
    np.testing.assert_allclose(result[fixed > 0], 1.0)


def test_blend_overlap_z_one_sided_moving_only():
    """When fixed is zero, moving values are preserved."""
    fixed = np.zeros((6, 8, 8), dtype=np.float32)
    moving = np.ones((6, 8, 8), dtype=np.float32)
    result = blend_overlap_z(fixed, moving)
    np.testing.assert_allclose(result[moving > 0], 1.0)


def test_blend_overlap_z_single_slice():
    """Single z-slice edge case: picks the region with more non-zero voxels."""
    fixed = np.ones((1, 8, 8), dtype=np.float32)
    moving = np.zeros((1, 8, 8), dtype=np.float32)
    result = blend_overlap_z(fixed, moving)
    assert result.shape == (1, 8, 8)


def test_inplane_output_point_matches_a_resampled_pixel():
    import SimpleITK as sitk

    image = np.zeros((40, 50), dtype=np.float32)
    image[12, 20] = 1.0
    transform = sitk.Euler3DTransform()
    transform.SetCenter((25.0, 20.0, 0.0))
    transform.SetRotation(0.0, 0.0, 0.0)
    transform.SetTranslation((8.0, -4.0, 0.0))
    moved = apply_2d_transform(image, transform, rotation_only=False, max_rotation_deg=0)
    rows, cols = np.nonzero(moved > 0.5)
    out_x, out_y = inplane_output_point(20.0, 12.0, transform, apply_translation=True)
    assert abs(float(cols.mean()) - out_x) < 0.6
    assert abs(float(rows.mean()) - out_y) < 0.6


def test_scale_slice_to_median_matches_a_brighter_neighbour():
    dim = np.full((6, 16, 16), 0.20, dtype=np.float32)
    bright = np.full((6, 16, 16), 0.40, dtype=np.float32)
    target = slice_tissue_median(dim)
    scaled, gain = scale_slice_to_median(bright, target)
    assert abs(gain - 0.5) < 1e-6
    assert abs(slice_tissue_median(scaled) - target) < 1e-5


def test_notch_slice_seams_removes_the_repeating_step():
    depth = np.arange(200, dtype=np.float32)
    slow = 0.30 + 0.05 * np.sin(2 * np.pi * depth / 80)
    seam = 0.04 * np.sign(np.sin(2 * np.pi * depth / 20))
    volume = np.broadcast_to((slow + seam)[:, None, None], (200, 4, 4)).copy()
    volume[0, 0, 0] = 0
    out = notch_slice_seams(volume, period=20)
    assert out[0, 0, 0] == 0
    spectrum = np.abs(np.fft.rfft(out[1:, 1, 1] - np.mean(out[1:, 1, 1])))
    freqs = np.fft.rfftfreq(out.shape[0] - 1)
    assert float(spectrum[int(np.argmin(np.abs(freqs - 1 / 20)))]) < 0.05


def test_suppress_z_slice_bands_removes_a_slice_period_wave():
    class _Array:
        def __init__(self, data: np.ndarray) -> None:
            self.data = data
            self.shape = data.shape

        def __getitem__(self, index: Any) -> np.ndarray:
            return self.data[index]

        def __setitem__(self, index: Any, value: np.ndarray) -> None:
            self.data[index] = value

    depth = np.arange(200, dtype=np.float32)
    wave = 0.2 + 0.06 * np.sin(2 * np.pi * depth / 20)
    volume = np.broadcast_to(wave[:, None, None], (200, 8, 8)).copy()
    array = _Array(volume)
    suppress_z_slice_bands(array, tissue_threshold=0.05, tile=8)
    column = array.data[40:160, 0, 0]
    assert float(column.max() - column.min()) < 0.04


def test_flatten_slice_z_profile_removes_the_depth_hump():
    vol = np.full((8, 32, 32), 0.01, dtype=np.float32)
    for z, value in enumerate((0.16, 0.22, 0.29, 0.28, 0.22, 0.17, 0.16, 0.16)):
        vol[z, 4:28, 4:28] = value
    out = flatten_slice_z_profile(vol, tissue_threshold=0.05)
    medians = [float(np.median(out[z][out[z] > 0.05])) for z in range(8)]
    assert max(medians) / min(medians) < 1.15


def test_flatten_does_not_amplify_a_hot_spot_on_a_dim_plane():
    vol = np.full((6, 40, 40), 0.20, dtype=np.float32)
    vol[0] = 0.10
    vol[0, 10:15, 10:15] = 1.0
    out = flatten_slice_z_profile(vol, tissue_threshold=0.05)
    assert float(out[0, 12, 12]) <= 1.0 + 1e-5
    assert float(np.median(out[0][out[0] > 0.05])) > 0.15


def test_blend_overlap_z_softens_a_shifted_edge():
    """A two-pixel lateral offset is feathered instead of left as a hard rim."""
    fixed = np.zeros((4, 32, 32), dtype=np.float32)
    moving = np.zeros((4, 32, 32), dtype=np.float32)
    fixed[:, 8:24, 8:20] = 2.0
    moving[:, 8:24, 12:24] = 4.0
    result = blend_overlap_z(fixed, moving)
    # Column 10 is fixed-only, two pixels from the moving slab. Feathering
    # pulls moving intensity in, so it is no longer a flat 2.
    gap = result[2, 16, 10]
    assert gap > 2.05


def test_blend_overlap_z_leaves_the_gap_between_two_slabs_empty():
    """Agarose that neither slab covers stays empty, even right next to both."""
    fixed = np.zeros((4, 32, 32), dtype=np.float32)
    moving = np.zeros((4, 32, 32), dtype=np.float32)
    fixed[:, 8:24, 6:14] = 2.0
    moving[:, 8:24, 18:26] = 4.0
    result = blend_overlap_z(fixed, moving)
    assert float(result[:, 8:24, 14:18].max()) == 0.0


def test_blend_overlap_z_rim_keeps_its_own_texture():
    """The feathered rim changes gain only, so neighbouring voxels keep their ratio."""
    fixed = np.zeros((4, 32, 32), dtype=np.float32)
    moving = np.zeros((4, 32, 32), dtype=np.float32)
    fixed[:, 8:24, 8:20] = 2.0
    fixed[:, 8:24:2, 8:20] = 1.0
    moving[:, 8:24, 12:24] = 3.0
    result = blend_overlap_z(fixed, moving)
    # Column 10 is fixed-only. Rows 15 and 16 held 2 and 1 going in.
    assert result[2, 15, 10] / result[2, 16, 10] == pytest.approx(2.0, rel=0.05)


def test_blend_cut_face_hands_over_within_the_first_planes():
    existing = np.full((10, 8, 8), 1.0, dtype=np.float32)
    incoming = np.full((10, 8, 8), 3.0, dtype=np.float32)
    out = blend_cut_face(existing, incoming, planes=4)
    np.testing.assert_allclose(out[0], 1.0)
    assert 1.0 < float(out[1, 0, 0]) < float(out[2, 0, 0]) < 3.0
    np.testing.assert_allclose(out[3:], 3.0)


def test_blend_cut_face_keeps_existing_tissue_under_incoming_background():
    existing = np.full((10, 8, 8), 1.0, dtype=np.float32)
    incoming = np.full((10, 8, 8), 3.0, dtype=np.float32)
    incoming[:, :, 4:] = 0.0
    out = blend_cut_face(existing, incoming, planes=4)
    assert float(out[:, :, 4:].min()) >= 1.0
    np.testing.assert_allclose(out[4:, :, 4:], 1.0)
    np.testing.assert_allclose(out[6, :, :4], 3.0)


def test_blend_cut_face_clamps_planes_to_the_overlap():
    existing = np.full((3, 8, 8), 1.0, dtype=np.float32)
    incoming = np.full((3, 8, 8), 3.0, dtype=np.float32)
    out = blend_cut_face(existing, incoming, planes=50)
    np.testing.assert_allclose(out, blend_overlap_z(existing, incoming))


def test_blend_overlap_z_does_not_mix_tissue_with_agarose():
    """Dim agarose must not Hann-average with tissue."""
    fixed = np.ones((6, 8, 8), dtype=np.float32)
    moving = np.full((6, 8, 8), 0.005, dtype=np.float32)
    result = blend_overlap_z(fixed, moving, tissue_threshold=0.01)
    np.testing.assert_allclose(result, 1.0)


def test_paste_tissue_does_not_punch_holes():
    """Incoming zeros must not erase existing tissue (no-blend overlap)."""
    existing = np.ones((4, 6, 6), dtype=np.float32)
    incoming = np.zeros((4, 6, 6), dtype=np.float32)
    incoming[:, :3, :] = 2.0
    out = paste_tissue(existing, incoming, threshold=0.01)
    np.testing.assert_allclose(out[:, :3, :], 2.0)
    np.testing.assert_allclose(out[:, 3:, :], 1.0)


def test_expected_z_overlap_consecutive_and_gap():
    assert expected_z_overlap(43, 4, 20, id_step=1) == 19
    assert expected_z_overlap(29, 4, 20, id_step=2) == -15


def test_expected_z_overlap_keeps_cut_face_when_crop_is_zero():
    """Pairwise template index 4 must not shorten physics overlap."""
    nz, interval = 43, 20
    with_template_crop = expected_z_overlap(nz, 4, interval, id_step=1)
    without_crop = expected_z_overlap(nz, 0, interval, id_step=1)
    assert without_crop == nz - interval
    assert without_crop - with_template_crop == 4


def test_crop_moving_volume_default_preserves_cut_face():
    vol = np.arange(20, dtype=np.float32).reshape(5, 2, 2)
    out = crop_moving_volume(vol, 0)
    np.testing.assert_array_equal(out, vol)
    np.testing.assert_allclose(out[0], vol[0])


def test_crop_moving_volume_drops_leading_planes_when_requested():
    vol = np.arange(20, dtype=np.float32).reshape(5, 2, 2)
    out = crop_moving_volume(vol, 4)
    assert out.shape[0] == 1
    np.testing.assert_array_equal(out, vol[4:])


def test_enforce_z_consistency_preserves_id_gap():
    matches = [
        {"fixed_id": 47, "moving_id": 48, "overlap_voxels": 21, "blend_overlap_voxels": 21},
        {"fixed_id": 48, "moving_id": 49, "overlap_voxels": 21, "blend_overlap_voxels": 21},
        {"fixed_id": 49, "moving_id": 51, "overlap_voxels": -15, "blend_overlap_voxels": 0},
    ]
    out, corrections = enforce_z_consistency(matches, confidence_per_slice={48: 0.0, 49: 0.0, 51: 0.0})
    assert out[2]["overlap_voxels"] == -15
    assert not any(c["moving_id"] == 51 for c in corrections)


# ---------------------------------------------------------------------------
# blend_overlap_xy
# ---------------------------------------------------------------------------


def test_blend_overlap_xy_none_overwrites():
    existing = np.ones((4, 8, 8), dtype=np.float32)
    new_data = np.full((4, 8, 8), 5.0, dtype=np.float32)
    result = blend_overlap_xy(existing.copy(), new_data, method="none")
    np.testing.assert_allclose(result, 5.0)


def test_blend_overlap_xy_average():
    existing = np.ones((4, 8, 8), dtype=np.float32)
    new_data = np.full((4, 8, 8), 3.0, dtype=np.float32)
    result = blend_overlap_xy(existing.copy(), new_data, method="average")
    np.testing.assert_allclose(result, 2.0)


def test_blend_overlap_xy_max():
    existing = np.ones((4, 8, 8), dtype=np.float32)
    new_data = np.full((4, 8, 8), 3.0, dtype=np.float32)
    result = blend_overlap_xy(existing.copy(), new_data, method="max")
    np.testing.assert_allclose(result, 3.0)


def test_blend_overlap_xy_average_respects_zeros():
    """Pixels zero in existing (no data) should take new_data value."""
    existing = np.zeros((4, 8, 8), dtype=np.float32)
    new_data = np.full((4, 8, 8), 2.0, dtype=np.float32)
    result = blend_overlap_xy(existing.copy(), new_data, method="average")
    np.testing.assert_allclose(result, 2.0)


# ---------------------------------------------------------------------------
# overlap z-gain (Z-end extremities, both-tissue XY)
# ---------------------------------------------------------------------------


def test_extract_overlap_rois_uses_z_ends_only():
    """Overlap is previous bottom vs next top — not the full slab."""
    fixed = np.zeros((10, 8, 8), dtype=np.float32)
    moving = np.zeros((10, 8, 8), dtype=np.float32)
    fixed[-3:] = 2.0
    moving[4:7] = 3.0
    rois = extract_overlap_rois(fixed, moving, overlap=3, moving_z_start=4)
    assert rois is not None
    f, m = rois
    assert f.shape[0] == 3
    np.testing.assert_allclose(f, 2.0)
    np.testing.assert_allclose(m, 3.0)


def test_extract_overlap_rois_cut_face_enters_overlap_when_crop_is_zero():
    fixed = np.zeros((10, 8, 8), dtype=np.float32)
    moving = np.zeros((10, 8, 8), dtype=np.float32)
    fixed[-4:] = 2.0
    moving[:4] = 5.0
    rois = extract_overlap_rois(fixed, moving, overlap=4, moving_z_start=0)
    assert rois is not None
    f, m = rois
    np.testing.assert_allclose(f, 2.0)
    np.testing.assert_allclose(m, 5.0)


def test_overlap_z_profiles_ignore_xy_only_one_slice():
    """Partial XY: voxels present in only one slab do not enter the median."""
    fixed = np.zeros((4, 32, 32), dtype=np.float32)
    moving = np.zeros((4, 32, 32), dtype=np.float32)
    both = (slice(None), slice(0, 20), slice(0, 20))
    only_fixed = (slice(None), slice(20, 32), slice(None))
    fixed[both] = 1.0
    moving[both] = 2.0
    fixed[only_fixed] = 50.0
    pf, pm, n = overlap_z_profiles(fixed, moving, tissue_threshold=0.01, min_voxels=50)
    np.testing.assert_allclose(pf, 1.0)
    np.testing.assert_allclose(pm, 2.0)
    assert np.all(n == 20 * 20)


def test_fit_overlap_log_ratio_recovers_linear_model():
    a_true, b_true = 0.4, 0.02
    z = np.arange(8, dtype=np.float64)
    pf = np.full(8, 1.0)
    pm = np.exp(a_true + b_true * z) * pf
    fit = fit_overlap_log_ratio(pf, pm, min_planes=4)
    assert fit is not None
    a, b = fit
    assert abs(a - a_true) < 1e-6
    assert abs(b - b_true) < 1e-6


def test_overlap_z_gain_curve_ramps_unique_then_overlap():
    nz, ov = 10, 4
    a, b = np.log(2.0), 0.0
    g = overlap_z_gain_curve(nz, ov, a, b)
    np.testing.assert_allclose(g[-ov:], 2.0, atol=1e-6)
    np.testing.assert_allclose(g[0], 1.0, atol=1e-6)
    np.testing.assert_allclose(g[nz - ov - 1], 2.0, atol=1e-6)
    assert np.all(np.diff(g[: nz - ov + 1]) >= -1e-6)


def test_overlap_z_gain_curve_clamps_to_hi():
    nz, ov = 8, 4
    a, b = np.log(3.0), 0.0
    g = overlap_z_gain_curve(nz, ov, a, b, clamp=(0.5, 2.0))
    assert float(np.max(g)) == 2.0


def test_apply_overlap_z_gain_multiplies_planes():
    vol = np.ones((5, 3, 3), dtype=np.float32)
    gain = np.array([1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)
    out = apply_overlap_z_gain(vol, gain)
    for z, s in enumerate(gain):
        np.testing.assert_allclose(out[z], s)


def test_estimate_overlap_z_gain_fit_ignores_unique_region():
    """Unique (non-overlap) planes must not affect the fit."""
    rng = np.random.default_rng(0)
    nz, ov, mz = 12, 5, 2
    a_true, b_true = 0.3, 0.01
    fixed = np.zeros((nz, 40, 40), dtype=np.float32)
    moving = np.zeros((nz, 40, 40), dtype=np.float32)
    # Unique region of previous is much darker — must not enter the fit.
    fixed[:-ov] = 0.1
    for i in range(ov):
        val = 1.0
        fixed[-ov + i] = val
        moving[mz + i] = float(np.exp(a_true + b_true * i)) * val
    # Noise in unique moving planes
    moving[:mz] = rng.random((mz, 40, 40)).astype(np.float32) * 10.0
    moving[mz + ov :] = rng.random((nz - mz - ov, 40, 40)).astype(np.float32) * 10.0
    result = estimate_overlap_z_gain_fit(fixed, moving, overlap=ov, moving_z_start=mz, min_voxels=50)
    assert result is not None
    a, b, xy_frac, n_planes = result
    assert n_planes == ov
    assert xy_frac > 0.9
    assert abs(a - a_true) < 0.02
    assert abs(b - b_true) < 0.01


def test_estimate_overlap_z_gain_fit_returns_none_without_tissue():
    fixed = np.zeros((8, 16, 16), dtype=np.float32)
    moving = np.zeros((8, 16, 16), dtype=np.float32)
    assert estimate_overlap_z_gain_fit(fixed, moving, overlap=4, min_voxels=50) is None


# ---------------------------------------------------------------------------
# estimate_z_blend_xy_shift / refine_z_blend_overlap (keep-if-better)
# ---------------------------------------------------------------------------


def _gaussian_blob(size=64, cy=32.0, cx=32.0, sig=8.0):
    y, x = np.mgrid[:size, :size]
    return np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sig**2)).astype(np.float32)


def _stack_aip(aip, nz=4):
    return np.stack([aip] * nz, axis=0)


def test_estimate_z_blend_xy_shift_too_few_valid_pixels():
    existing = np.zeros((4, 20, 20), dtype=np.float32)
    moving = np.zeros((4, 20, 20), dtype=np.float32)
    dy, dx, mag = estimate_z_blend_xy_shift(existing, moving, 10.0)
    assert (dy, dx, mag) == (0.0, 0.0, 0.0)


def test_estimate_z_blend_xy_shift_applies_when_ncc_rises(monkeypatch):
    blob = _gaussian_blob()
    existing = _stack_aip(blob)
    moving = _stack_aip(np.roll(blob, 3, axis=0))

    def fake_register(*_args, **kwargs):
        diag = kwargs.get("diagnostics")
        if diag is not None:
            diag["rejected"] = False
        return 0.0, -3.0, 0.0, 0.0

    monkeypatch.setattr("linumpy.registration.refinement.register_refinement", fake_register)
    dy, dx, mag = estimate_z_blend_xy_shift(existing, moving, 10.0)
    assert dy == pytest.approx(-3.0)
    assert dx == pytest.approx(0.0)
    assert mag == pytest.approx(3.0)


def test_estimate_z_blend_xy_shift_keeps_when_ncc_does_not_rise(monkeypatch):
    blob = _gaussian_blob()
    vol = _stack_aip(blob)

    def fake_register(*_args, **kwargs):
        diag = kwargs.get("diagnostics")
        if diag is not None:
            diag["rejected"] = False
        return 0.0, 4.0, 0.0, 0.0

    monkeypatch.setattr("linumpy.registration.refinement.register_refinement", fake_register)
    dy, dx, mag = estimate_z_blend_xy_shift(vol, vol.copy(), 10.0)
    assert (dy, dx, mag) == (0.0, 0.0, 0.0)


def test_estimate_z_blend_xy_shift_clamps_over_bound(monkeypatch):
    """A shift past the cap is clamped, not dropped to identity."""
    blob = _gaussian_blob()
    existing = _stack_aip(blob)
    moving = _stack_aip(np.roll(blob, 3, axis=0))
    captured: dict = {}

    def fake_register(*_args, **kwargs):
        captured["bound_mode"] = kwargs.get("bound_mode")
        diag = kwargs.get("diagnostics")
        if diag is not None:
            diag["rejected"] = False
        return 0.0, -3.0, 0.0, 0.0

    monkeypatch.setattr("linumpy.registration.refinement.register_refinement", fake_register)
    dy, dx, mag = estimate_z_blend_xy_shift(existing, moving, 10.0)
    assert captured["bound_mode"] == "scale"
    assert dy == pytest.approx(-3.0)
    assert dx == pytest.approx(0.0)
    assert mag == pytest.approx(3.0)


def test_estimate_z_blend_xy_shift_uses_cut_face_planes(monkeypatch):
    """Score last planes of previous vs first of incoming, not the interior AIP."""
    blob = _gaussian_blob()
    existing = np.full((6, 64, 64), 0.2, dtype=np.float32)
    moving = np.full((6, 64, 64), 0.2, dtype=np.float32)
    existing[-2:] = blob
    moving[:2] = np.roll(blob, 3, axis=0)
    captured: dict = {}

    def fake_register(fixed_2d, *_args, **kwargs):
        captured["fixed"] = np.asarray(fixed_2d)
        diag = kwargs.get("diagnostics")
        if diag is not None:
            diag["rejected"] = False
        return 0.0, -3.0, 0.0, 0.0

    monkeypatch.setattr("linumpy.registration.refinement.register_refinement", fake_register)
    monkeypatch.setattr("linumpy.mosaic.stacking._phase_offset", lambda *_a, **_k: None)
    estimate_z_blend_xy_shift(existing, moving, 10.0, interface_planes=2)
    np.testing.assert_allclose(captured["fixed"], blob, atol=1e-5)


def test_refine_z_blend_overlap_shifts_slab_when_accepted(monkeypatch):
    from linumpy.registration.refinement import tissue_ncc

    blob = _gaussian_blob()
    existing = _stack_aip(blob)
    moving = _stack_aip(np.roll(blob, 3, axis=0))

    def fake_register(*_args, **kwargs):
        diag = kwargs.get("diagnostics")
        if diag is not None:
            diag["rejected"] = False
        return 0.0, -3.0, 0.0, 0.0

    monkeypatch.setattr("linumpy.registration.refinement.register_refinement", fake_register)
    refined, mag = refine_z_blend_overlap(existing, moving, 10.0)
    assert mag == pytest.approx(3.0)
    ncc_before = tissue_ncc(blob, moving[0])
    ncc_after = tissue_ncc(blob, refined[0])
    assert ncc_after > ncc_before


def _textured_block(nz: int = 80, shape: tuple = (160, 160), seed: int = 0) -> np.ndarray:
    """Tissue whose fine structure changes smoothly with depth, in a round footprint."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    noise = gaussian_filter(rng.normal(size=(nz, *shape)).astype(np.float32), (4.0, 3.0, 3.0))
    noise /= noise.std()
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    inside = ((yy - shape[0] / 2) ** 2 + (xx - shape[1] / 2) ** 2) < (0.45 * shape[0]) ** 2
    return np.where(inside[None], 0.3 * (1.0 + 0.4 * noise), 0.0).astype(np.float32)


def _slab(block: np.ndarray, top: int, depth: int = 44) -> np.ndarray:
    """A slab cut from the block: brightest at its own top, as OCT is."""
    decay = np.exp(-0.03 * np.arange(depth, dtype=np.float32))[:, None, None]
    return block[top : top + depth] * decay


@pytest.mark.parametrize("true_step", [26, 16, 23, 21, 19, 18])
def test_find_z_step_structure_recovers_the_step(true_step):
    from linumpy.mosaic.stacking import find_z_step_structure

    block = _textured_block()
    fixed, moving = _slab(block, 0), _slab(block, true_step)
    step, info = find_z_step_structure(fixed, moving, expected_step=20, search_vox=8, min_pixels=500)
    assert step == true_step, info
    assert info["slope"] == pytest.approx(1.0, abs=0.1)
    assert info["residual"] < 0.5


@pytest.mark.parametrize("scale", [0.8, 1.15])
def test_find_z_step_structure_reads_the_step_at_the_cut_face(scale):
    """The same tissue spans a different number of planes deep in the slab above."""
    from scipy.ndimage import map_coordinates

    from linumpy.mosaic.stacking import find_z_step_structure

    block = _textured_block(nz=120)
    true_step = 19
    moving = _slab(block, true_step)
    # Below the cut face the fixed slab is stretched or compressed along Z:
    # its plane ``true_step + scale * p`` shows what moving plane ``p`` shows.
    depth = 44
    planes = np.arange(depth, dtype=np.float32)
    source = np.where(planes < true_step, planes, true_step + (planes - true_step) / scale)
    yy, xx = np.mgrid[: block.shape[1], : block.shape[2]].astype(np.float32)
    fixed = np.stack([map_coordinates(block, [np.full_like(yy, z), yy, xx], order=1) for z in source])
    fixed *= np.exp(-0.03 * planes)[:, None, None]

    step, info = find_z_step_structure(fixed, moving, expected_step=20, search_vox=8, min_pixels=500)
    assert step == true_step, info
    assert info["slope"] == pytest.approx(scale, abs=0.1)
    # Read at a deeper moving plane the step is off by the stretch.
    assert abs(float(np.median(info["steps"])) - true_step) > 0.5


def test_find_z_step_structure_follows_a_lateral_shift():
    from scipy.ndimage import shift as nd_shift

    from linumpy.mosaic.stacking import find_z_step_structure

    block = _textured_block(seed=3)
    fixed = _slab(block, 0)
    moving = nd_shift(_slab(block, 23), (0, 6, -4), order=1)
    step, info = find_z_step_structure(fixed, moving, expected_step=20, search_vox=8, min_pixels=500)
    assert step == 23, info
    assert abs(info["shift"][0] + 6) < 2 and abs(info["shift"][1] - 4) < 2


def test_find_z_step_structure_declines_when_it_cannot_tell():
    from linumpy.mosaic.stacking import find_z_step_structure

    fixed = _slab(_textured_block(seed=1), 0)
    unrelated = _slab(_textured_block(seed=2), 20)
    step, info = find_z_step_structure(fixed, unrelated, expected_step=20, search_vox=8, min_pixels=500)
    assert step is None
    assert info["reason"]
    # Too little depth to search in.
    assert find_z_step_structure(fixed[:5], unrelated, expected_step=20)[0] is None


def test_find_z_step_structure_does_not_trust_a_match_at_the_end_of_a_short_slab():
    from linumpy.mosaic.stacking import find_z_step_structure

    block = _textured_block(seed=4)
    fixed, moving = _slab(block, 0, depth=30), _slab(block, 26)
    # Moving plane p matches fixed plane 26 + p. A 30-plane slab has that plane
    # for p up to 3, and a best match on its last plane is not trusted, so two
    # planes are placed, which is not enough for a line.
    step, info = find_z_step_structure(fixed, moving, expected_step=20, search_vox=8, min_pixels=500)
    assert step is None
    assert info["reason"] == "too few planes"
    assert info["planes"] == [1, 2]
    assert all(abs(found - 26) < 0.5 for found in info["steps"])


def test_find_z_step_structure_declines_a_match_at_the_edge_of_the_search():
    from linumpy.mosaic.stacking import find_z_step_structure

    block = _textured_block(seed=5)
    fixed, moving = _slab(block, 0, depth=60), _slab(block, 30)
    step, info = find_z_step_structure(fixed, moving, expected_step=20, search_vox=8, min_pixels=500)
    assert step is None, info


def test_zero_above_surface_follows_a_tilted_cut():
    from linumpy.mosaic.stacking import zero_above_surface

    nz, ny, nx = 30, 40, 60
    surface = np.round(np.linspace(0, 8, nx)).astype(int)  # the cut dips to the right
    vol = np.full((nz, ny, nx), 0.3, dtype=np.float32)
    z = np.arange(nz)[:, None, None]
    water = np.broadcast_to(z < surface[None, None, :], vol.shape)
    vol[water] = 0.03  # dim, but above the stacker's background threshold
    out, count = zero_above_surface(vol, max_planes=12, tissue_threshold=0.01, sigma_xy=1.0)
    assert count > 0
    inner = (slice(None), slice(5, -5), slice(6, -6))
    # Away from the smoothing at the step edges the water is gone and the tissue is untouched.
    assert (out[inner][water[inner]] == 0).mean() > 0.9
    np.testing.assert_array_equal(out[inner][~water[inner]], vol[inner][~water[inner]])
    # A slab with a flat cut is returned as it is.
    flat = np.full((nz, ny, nx), 0.3, dtype=np.float32)
    same, none = zero_above_surface(flat, max_planes=12)
    assert none == 0 and same is flat


def test_zero_above_surface_leaves_agarose_and_dim_tissue_tops_alone():
    from linumpy.mosaic.stacking import zero_above_surface

    vol = np.full((30, 20, 20), 0.3, dtype=np.float32)
    vol[:3] *= 0.74  # the first planes of a slab are a little dim in all tissue
    vol[:, :, :5] = 0.004  # nothing here: below the tissue threshold
    out, count = zero_above_surface(vol, max_planes=12, tissue_threshold=0.01, sigma_xy=0.5)
    assert count == 0
    assert out is vol


def test_zero_above_surface_keeps_grey_matter_that_lies_over_white_matter():
    from linumpy.mosaic.stacking import zero_above_surface

    vol = np.full((30, 20, 20), 0.2, dtype=np.float32)
    vol[6:, :, 10:] = 0.8  # bright white matter under 6 planes of grey
    out, count = zero_above_surface(vol, max_planes=12, tissue_threshold=0.01, sigma_xy=0.5)
    assert count == 0
    assert out is vol


def _seamed_volume(shape: tuple[int, int, int]) -> np.ndarray:
    """Tissue with a slow depth trend, a step that repeats every 20 planes, and an empty margin."""
    rng = np.random.default_rng(0)
    depth = np.arange(shape[0], dtype=np.float32)
    profile = 0.30 + 0.05 * np.sin(2 * np.pi * depth / 80) + 0.04 * np.sign(np.sin(2 * np.pi * depth / 20))
    volume = (profile[:, None, None] * (1.0 + 0.2 * rng.random(shape, dtype=np.float32))).astype(np.float32)
    volume[:, :3, :] = 0
    return volume


def _apply_seam_notch_plane_by_plane(array: np.ndarray, tile: int = 128, threshold: float = 0.05) -> float:
    """The notch as it was applied before it followed the chunk grid: plane medians one plane at a time, 128 px tiles."""
    from linumpy.mosaic.stacking import estimate_slice_period

    medians = np.empty(array.shape[0], dtype=np.float64)
    for z in range(array.shape[0]):
        plane = np.asarray(array[z, ::8, ::8])
        tissue = plane > threshold
        medians[z] = float(np.median(plane[tissue])) if int(np.sum(tissue)) > 40 else np.nan
    good = np.isfinite(medians)
    index = np.arange(len(medians))
    period = estimate_slice_period(np.interp(index, index[good], medians[good]), period_min=12, period_max=40)
    _, ny, nx = array.shape
    for y0 in range(0, ny, tile):
        for x0 in range(0, nx, tile):
            block = np.asarray(array[:, y0 : y0 + tile, x0 : x0 + tile], dtype=np.float32)
            array[:, y0 : y0 + tile, x0 : x0 + tile] = notch_slice_seams(block, period)
    return period


@pytest.mark.parametrize("workers", [1, 3])
def test_apply_seam_notch_on_chunks_gives_the_same_volume(tmp_path, workers):
    """Following the chunk grid, and filtering tiles side by side, changes no voxel."""
    import zarr

    from linumpy.mosaic.stacking import apply_seam_notch

    volume = _seamed_volume((200, 150, 170))
    expected = volume.copy()
    expected_period = _apply_seam_notch_plane_by_plane(expected)

    # Chunks that do not divide the shape, and are not the old 128 px tiles.
    stored = zarr.open_array(tmp_path / "stack.zarr", mode="w", shape=volume.shape, chunks=(64, 40, 48), dtype=np.float32)
    stored[:] = volume
    period = apply_seam_notch(stored, workers=workers)

    assert period == expected_period
    assert 18 < period < 22
    np.testing.assert_array_equal(np.asarray(stored[:]), expected)
    assert not np.array_equal(expected, volume)


def test_apply_seam_notch_reads_each_chunk_once_for_the_medians():
    """The plane medians are read in slabs of whole chunks, not one plane at a time."""
    from linumpy.mosaic.stacking import apply_seam_notch

    class Counting:
        """A chunked array that counts its reads."""

        def __init__(self, data: np.ndarray, chunks: tuple[int, int, int]) -> None:
            self.data = data
            self.chunks = chunks
            self.shape = data.shape
            self.reads = 0

        def __getitem__(self, key: Any) -> np.ndarray:
            self.reads += 1
            return self.data[key]

        def __setitem__(self, key: Any, value: np.ndarray) -> None:
            self.data[key] = value

    array = Counting(_seamed_volume((200, 80, 96)), chunks=(50, 40, 48))
    apply_seam_notch(array, workers=1)
    # 4 slabs of 50 planes for the medians, then one read per chunk column (2 x 2).
    assert array.reads == 4 + 4


def _banded_stack(n_slabs: int = 8, period: int = 20, plane: tuple[int, int] = (88, 104)) -> tuple[np.ndarray, list[int]]:
    """Slabs with their own anatomy and fine detail, and a depth trend that is stronger on the right half."""
    rng = np.random.default_rng(1)
    ny, nx = plane
    volume = np.empty((n_slabs * period, ny, nx), dtype=np.float32)
    strength = np.where(np.arange(nx) < nx // 2, 0.2, 0.8)[None, None, :]
    depth = (np.arange(period) / period)[:, None, None]
    for s in range(n_slabs):
        coarse = 0.25 * np.exp(rng.normal(0.0, 0.2, size=(ny // 8, nx // 8)))
        anatomy = np.kron(coarse, np.ones((8, 8)))
        detail = 1.0 + 0.1 * rng.random((period, ny, nx))
        volume[s * period : (s + 1) * period] = anatomy[None] * detail * np.exp(-strength * depth)
    volume[:, :8, :] = 0  # empty margin
    return volume, [s * period for s in range(n_slabs)]


def _depth_drop(volume: np.ndarray, period: int, columns: slice) -> float:
    """Mean ratio of the last to the first plane of a slab, over all slabs."""
    slabs = volume[:, 16:, columns].reshape(-1, period, volume.shape[1] - 16, volume[:, :, columns].shape[2])
    return float(np.mean(slabs[:, -1].mean(axis=(1, 2)) / slabs[:, 0].mean(axis=(1, 2))))


@pytest.mark.parametrize("workers", [1, 3])
def test_apply_local_depth_gain_flattens_each_tissue_by_its_own_trend(tmp_path, workers):
    import zarr

    from linumpy.mosaic.stacking import apply_local_depth_gain

    volume, starts = _banded_stack()
    stored = zarr.open_array(tmp_path / "stack.zarr", mode="w", shape=volume.shape, chunks=(64, 40, 48), dtype=np.float32)
    stored[:] = volume
    rms = apply_local_depth_gain(stored, starts, workers=workers)
    out = np.asarray(stored[:])

    assert 0.05 < rms < 0.3
    # Away from the border between the two tissues the trend is gone on both sides.
    assert _depth_drop(volume, 20, slice(0, 40)) < 0.86
    assert _depth_drop(volume, 20, slice(64, 104)) < 0.5
    assert _depth_drop(out, 20, slice(0, 40)) == pytest.approx(1.0, abs=0.03)
    assert _depth_drop(out, 20, slice(64, 104)) == pytest.approx(1.0, abs=0.03)
    # Empty voxels stay empty, and the gain is smooth inside a plane: neighbours keep their ratio.
    assert not out[:, :8, :].any()
    gain = out[:, 16:, :] / volume[:, 16:, :]
    assert np.abs(np.diff(np.log(gain), axis=2)).max() < 0.03
    # The level of a slab does not change.
    before = np.log(volume[:, 16:, :]).reshape(8, 20, -1).mean(axis=(1, 2))
    after = np.log(out[:, 16:, :]).reshape(8, 20, -1).mean(axis=(1, 2))
    np.testing.assert_allclose(after, before, atol=0.01)


def test_apply_local_depth_gain_skips_starts_it_cannot_use():
    from linumpy.mosaic.stacking import apply_local_depth_gain

    volume, starts = _banded_stack()
    original = volume.copy()
    assert apply_local_depth_gain(volume, [0, 40, 20, 60, 80, 100, 120, 140]) == 0.0
    assert apply_local_depth_gain(volume, starts[:1]) == 0.0
    assert apply_local_depth_gain(volume, [5, *starts[1:]]) == 0.0
    np.testing.assert_array_equal(volume, original)


@pytest.mark.parametrize("workers", [1, 3])
def test_gaussian_filter_planes_matches_the_filter_of_the_whole_volume(workers):
    from scipy.ndimage import gaussian_filter

    from linumpy.mosaic.stacking import gaussian_filter_planes

    volume = np.random.default_rng(4).random((5, 60, 70), dtype=np.float32)
    assert np.array_equal(gaussian_filter_planes(volume, 3.0, workers=workers), gaussian_filter(volume, (0.0, 3.0, 3.0)))


def test_find_z_step_structure_is_the_same_on_one_thread_and_on_several():
    from linumpy.mosaic.stacking import find_z_step_structure

    block = _textured_block()
    fixed, moving = _slab(block, 0), _slab(block, 26)
    step_one, info_one = find_z_step_structure(fixed, moving, expected_step=20, search_vox=10, workers=1)
    step_many, info_many = find_z_step_structure(fixed, moving, expected_step=20, search_vox=10, workers=4)
    assert step_one == step_many == 26
    assert info_one == info_many


@pytest.mark.parametrize("workers", [1, 3])
def test_copy_threaded_matches_ascontiguousarray(workers):
    from linumpy.utils.planes import copy_threaded

    volume = np.random.default_rng(14).random((7, 9, 11), dtype=np.float32)
    view = volume.transpose(2, 0, 1)[::-1, :, ::-1]
    copy = copy_threaded(view, workers=workers)
    assert copy.flags.c_contiguous
    assert np.array_equal(copy, np.ascontiguousarray(view))


def test_fill_joint_gaps_closes_a_short_run_across_the_joint():
    from linumpy.mosaic.stacking import fill_joint_gaps

    block = np.zeros((10, 2, 2), dtype=np.float32)
    block[:4] = 1.0  # slab above, tissue down to plane 3
    block[7:] = 4.0  # incoming slab, tissue from plane 7: planes 4 to 6 are empty
    filled, n = fill_joint_gaps(block, joint=5, max_gap=3)
    assert n == 3 * 4
    np.testing.assert_allclose(filled[:, 0, 0], [1, 1, 1, 1, 1.75, 2.5, 3.25, 4, 4, 4])


def test_fill_joint_gaps_leaves_a_run_that_is_too_long_or_open():
    from linumpy.mosaic.stacking import fill_joint_gaps

    block = np.zeros((12, 1, 3), dtype=np.float32)
    block[:2, 0, 0] = 1.0
    block[9:, 0, 0] = 1.0  # seven empty planes: longer than allowed
    block[6:, 0, 1] = 1.0  # nothing above: open at the top
    block[:5, 0, 2] = 1.0  # nothing below: open at the bottom
    filled, n = fill_joint_gaps(block, joint=5, max_gap=4)
    assert n == 0
    np.testing.assert_array_equal(filled, block)


def test_fill_joint_gaps_leaves_a_run_that_does_not_touch_the_joint():
    from linumpy.mosaic.stacking import fill_joint_gaps

    block = np.ones((12, 1, 2), dtype=np.float32)
    block[8:10, 0, 0] = 0.0  # a dark structure below the joint
    block[1:3, 0, 1] = 0.0  # and one above it
    filled, n = fill_joint_gaps(block, joint=5, max_gap=4)
    assert n == 0
    np.testing.assert_array_equal(filled, block)


def test_fill_joint_gaps_takes_the_run_that_ends_or_starts_at_the_joint():
    from linumpy.mosaic.stacking import fill_joint_gaps

    block = np.ones((10, 1, 2), dtype=np.float32)
    block[5:7, 0, 0] = 0.0  # starts with the first plane of the incoming slab
    block[3:5, 0, 1] = 0.0  # ends with the last plane before it
    filled, n = fill_joint_gaps(block, joint=5, max_gap=4)
    assert n == 4
    assert (filled > 0).all()


def _smooth_gain(shape: tuple[int, int]) -> np.ndarray:
    """A log gain that changes over a third of the image."""
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    return (0.25 * np.sin(2 * np.pi * xx / shape[1]) * np.cos(np.pi * yy / shape[0])).astype(np.float32)


def test_joint_level_field_measures_a_local_brightness_difference():
    from linumpy.mosaic.stacking import joint_level_field

    block = _textured_block(nz=8)
    gain = _smooth_gain(block.shape[1:])
    field = joint_level_field(block, block * np.exp(gain)[None])
    assert field is not None
    assert field.shape == (block.shape[1] // 4, block.shape[2] // 4)
    coarse = gain.reshape(field.shape[0], 4, field.shape[1], 4).mean(axis=(1, 3))
    inside = (block[0] > 0).reshape(field.shape[0], 4, field.shape[1], 4).all(axis=(1, 3))
    error = np.abs(field - coarse)[inside]
    assert np.median(error) < 0.01
    assert error.max() < 0.04
    # On the coarse scale the field is smoother and rounds the peaks of the gain.
    wide = joint_level_field(block, block * np.exp(gain)[None], cell=8, sigma_cells=4.0)
    assert wide is not None
    assert wide.shape == (block.shape[1] // 8, block.shape[2] // 8)
    assert np.abs(wide).max() < np.abs(field).max()
    # Anatomy that both slabs show is not a difference.
    same = joint_level_field(block, block.copy())
    assert same is not None
    assert np.abs(same).max() < 1e-6


def test_joint_level_field_needs_tissue_in_both_slabs():
    from linumpy.mosaic.stacking import joint_level_field

    block = _textured_block(nz=6)
    assert joint_level_field(block, np.zeros_like(block)) is None
    # A large difference is limited.
    field = joint_level_field(block, block * 5.0, clamp=0.5)
    assert field is not None
    assert field.max() == pytest.approx(0.5)


def test_level_joint_meets_in_the_middle_and_leaves_the_slab_centres():
    from linumpy.mosaic.stacking import joint_level_field, level_joint

    planes = 10
    difference = 0.3
    above = np.full((14, 32, 32), 0.2, dtype=np.float32)
    overlap = np.full((8, 32, 32), 0.2, dtype=np.float32)
    incoming = np.full((30, 32, 32), 0.2 * np.exp(difference), dtype=np.float32)
    field = joint_level_field(overlap, incoming[:8])
    assert field is not None
    level_joint(above, overlap, incoming, field, planes)
    # Both slabs meet at the joint.
    assert overlap.mean() == pytest.approx(incoming[0].mean(), rel=1e-4)
    assert overlap.mean() == pytest.approx(0.2 * np.exp(difference / 2), rel=1e-4)
    # The gain falls off on both sides and is gone after ``planes`` planes.
    assert above[-1].mean() < overlap.mean()
    assert np.all(np.diff(above.mean(axis=(1, 2))) >= 0)
    assert np.all(np.diff(incoming.mean(axis=(1, 2))) >= -1e-7)
    assert above[: 14 - planes].mean() == pytest.approx(0.2)
    assert incoming[planes:].mean() == pytest.approx(0.2 * np.exp(difference))
    # Background stays background.
    hole = np.zeros((4, 32, 32), dtype=np.float32)
    level_joint(hole[:2], hole[2:], hole.copy(), field, planes)
    assert not hole.any()


def _dipping_slab(slab: np.ndarray, dip: int = 6, ramp: int = 3) -> np.ndarray:
    """The slab with a cut face that lies up to ``dip`` planes lower towards the right.

    Above its surface an A-line is empty, and over the next ``ramp`` planes it
    comes up from a fraction of the tissue level, as a real cut face does.
    """
    out = slab.copy()
    nx = slab.shape[2]
    surface = np.round(np.linspace(0, dip, nx)).astype(int)
    for x in range(nx):
        out[: surface[x], :, x] = 0.0
        for step in range(ramp):
            plane = surface[x] + step
            if plane < out.shape[0]:
                out[plane, :, x] *= 0.15 + 0.25 * step
    return out


def test_surface_confidence_follows_a_cut_face_that_dips():
    from linumpy.mosaic.stacking import surface_confidence

    block = _textured_block(nz=20)
    incoming = _dipping_slab(block)
    confidence = surface_confidence(block, incoming, 12)
    assert confidence.shape == (12, *block.shape[1:])
    centre = block.shape[1] // 2
    left, right = 20, block.shape[2] - 20  # surface at plane 1 and at plane 5
    # In the dim planes the slab above is kept, below them the incoming slab is trusted.
    assert confidence[1, centre, left] < 0.1
    assert confidence[5, centre, left] == pytest.approx(1.0)
    assert confidence[6, centre, right] < 0.1
    assert confidence[10, centre, right] == pytest.approx(1.0)
    # Once reached, the weight stays: a dark structure further down is anatomy.
    darker = incoming.copy()
    darker[9:] *= 0.3
    assert surface_confidence(block, darker, 12)[10, centre, left] == pytest.approx(1.0)
    # Without a slab above there is nothing to wait for.
    assert surface_confidence(np.zeros_like(block), incoming, 12).min() == pytest.approx(1.0)


def test_blend_cut_face_leaves_no_dark_line_under_a_dipping_cut_face():
    from linumpy.mosaic.stacking import blend_cut_face, surface_confidence

    block = _textured_block(nz=20)
    incoming = _dipping_slab(block)
    tissue = block > 0.1
    plain = blend_cut_face(block, incoming, 8)
    followed = blend_cut_face(block, incoming, 8, confidence=surface_confidence(block, incoming, 12))
    # Both slabs show the same tissue, so the blend should give the block back.
    dark_plain = (plain[tissue] / block[tissue]) < 0.8
    dark_followed = (followed[tissue] / block[tissue]) < 0.8
    assert dark_plain.mean() > 0.02
    assert dark_followed.mean() < 0.1 * dark_plain.mean()
    assert (followed[tissue] / block[tissue]).min() > 0.6
    # Below the surface the incoming slab is what is written, as before.
    np.testing.assert_allclose(followed[12:], plain[12:])


def test_find_z_step_structure_drops_a_shift_that_does_not_help(monkeypatch):
    """The phase correlation always returns a shift; on aligned slabs a wrong one is not used."""
    import skimage.registration

    from linumpy.mosaic.stacking import find_z_step_structure

    block = _textured_block(seed=6)
    fixed, moving = _slab(block, 0), _slab(block, 22)
    monkeypatch.setattr(
        skimage.registration, "phase_cross_correlation", lambda *_args, **_kwargs: (np.array([6.0, -5.0]), 0.0, 0.0)
    )
    step, info = find_z_step_structure(fixed, moving, expected_step=20, search_vox=8, min_pixels=500)
    assert info["shift"] == (0.0, 0.0)
    assert step == 22, info


def test_below_own_surface_waits_for_the_tissue_level():
    from linumpy.mosaic.stacking import below_own_surface

    block = _textured_block(nz=20)
    slab = _dipping_slab(block)  # surface up to six planes lower on the right, with a dim ramp under it
    ready = below_own_surface(slab, 10)
    assert ready.shape == (10, *block.shape[1:])
    centre = block.shape[1] // 2
    left, right = 20, block.shape[2] - 20  # surface at plane 1 and at plane 5
    assert not ready[1, centre, left]
    assert ready[4, centre, left]
    assert not ready[6, centre, right]
    assert ready[9, centre, right]
    # Once an A-line is in the tissue it stays in.
    assert np.all(ready[1:] >= ready[:-1])


def test_joint_level_field_ignores_the_surface_ramp_of_the_incoming_slab():
    from linumpy.mosaic.stacking import below_own_surface, joint_level_field

    block = _textured_block(nz=20)
    incoming = _dipping_slab(block)
    inside = (block[0] > 0).reshape(block.shape[1] // 4, 4, block.shape[2] // 4, 4).all(axis=(1, 3))
    # The two slabs show the same tissue: there is nothing to level.
    plain = joint_level_field(block[:10], incoming[:10], min_fraction=0.3)
    masked = joint_level_field(block[:10], incoming[:10], min_fraction=0.3, valid=below_own_surface(incoming, 10))
    assert plain is not None and masked is not None
    # With the ramp in, the incoming slab looks darker by up to a factor two.
    assert np.median(np.abs(plain[inside])) > 0.3
    assert np.median(np.abs(masked[inside])) < 0.1
    assert np.abs(masked[inside]).max() < 0.3 * np.abs(plain[inside]).max()
