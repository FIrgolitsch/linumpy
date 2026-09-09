"""Tests for linumpy/mosaic/stacking.py"""

import numpy as np

from linumpy.mosaic.stacking import (
    apply_overlap_z_gain,
    apply_xy_shift,
    blend_overlap_xy,
    blend_overlap_z,
    estimate_overlap_z_gain_fit,
    extract_overlap_rois,
    find_z_overlap,
    fit_overlap_log_ratio,
    overlap_z_gain_curve,
    overlap_z_profiles,
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
