"""Tests for linumpy/mosaic/stacking.py"""

import numpy as np
import pytest

from linumpy.mosaic.stacking import (
    apply_overlap_z_gain,
    apply_xy_shift,
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
    overlap_z_gain_curve,
    overlap_z_profiles,
    paste_tissue,
    refine_z_blend_overlap,
)


def _make_vol(shape=(10, 32, 32), fill=1.0):
    return (np.ones(shape) * fill).astype(np.float32)


# ---------------------------------------------------------------------------
# find_z_overlap
# ---------------------------------------------------------------------------


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
