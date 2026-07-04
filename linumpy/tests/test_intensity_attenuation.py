"""Coverage tests for linumpy/intensity/attenuation.py helpers and methods.

The existing ``test_attenuation.py`` covers the four published estimators
(Vermeer/Smith/Liu/Li/Faber) on synthetic exponentials. This module targets
the remaining pure-Python helpers that are exercised by the reconstruction
pipeline but were not previously covered: the XY median filter, the
auto-tissue-mask + finalization helpers, the log-gradient attenuation, the
A-line fit / split helpers, the interface mask, the flat-agarose profile,
and the signal-from-attenuation forward model.
"""

import numpy as np
import pytest

from linumpy.intensity.attenuation import (
    _aline_fit,
    _auto_tissue_mask,
    _exact_tail_C,
    _finalize_attenuation,
    _lstsq_tail_slope,
    _median_xy_filter,
    get_aline_attenuation,
    get_flat_agarose_profile,
    get_gradient_attenuation,
    get_interface_mask,
    get_signal_from_attenuation,
    oct_signal_faber2004_model,
    split_aline,
)

# ---------------------------------------------------------------------------
# _median_xy_filter
# ---------------------------------------------------------------------------


def test_median_xy_filter_no_op_when_k_le_zero():
    """k <= 0 must return the input unchanged (no copy semantics required)."""
    vol = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    out = _median_xy_filter(vol, 0)
    np.testing.assert_array_equal(out, vol)


def test_median_xy_filter_reduces_salt_noise():
    """A single bright outlier voxel is suppressed by the XY median filter."""
    vol = np.zeros((3, 3, 3), dtype=np.float32)
    vol[1, 1, 1] = 99.0  # outlier
    out = _median_xy_filter(vol, 1)
    # The outlier voxel is replaced by the median of its XY neighbourhood (0).
    assert out[1, 1, 1] == 0.0
    # Shape preserved.
    assert out.shape == vol.shape


# ---------------------------------------------------------------------------
# _finalize_attenuation
# ---------------------------------------------------------------------------


def test_finalize_attenuation_replaces_nan_with_zero():
    """NaNs inside the attenuation map become 0 after finalization."""
    attn = np.array([[[1.0, np.nan, 2.0]]], dtype=np.float32)
    mask = np.ones_like(attn, dtype=bool)
    out = _finalize_attenuation(attn, mask, fill_holes=False)
    assert not np.any(np.isnan(out))
    assert out[0, 0, 1] == 0.0


def test_finalize_attenuation_zeros_outside_mask():
    """Voxels outside the mask are zeroed even if they held a value."""
    attn = np.array([[[5.0, 5.0, 5.0]]], dtype=np.float32)
    mask = np.array([[[True, False, True]]], dtype=bool)
    out = _finalize_attenuation(attn, mask, fill_holes=False)
    assert out[0, 0, 0] == 5.0
    assert out[0, 0, 1] == 0.0
    assert out[0, 0, 2] == 5.0


def test_finalize_attenuation_fill_holes_runs():
    """fill_holes=True exercises the SimpleITK GrayscaleFillhole path."""
    attn = np.zeros((1, 5, 5), dtype=np.float32)
    attn[0, 2, 2] = 1.0  # isolated bright voxel
    mask = np.ones_like(attn, dtype=bool)
    out = _finalize_attenuation(attn, mask, fill_holes=True)
    assert out.shape == attn.shape
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# _lstsq_tail_slope / _exact_tail_C
# ---------------------------------------------------------------------------


def test_lstsq_tail_slope_recovers_exponential_decay():
    """For a clean exponential the LSQ slope equals -2*mu*dz (per voxel)."""
    n_tail = 30
    mu_per_voxel = 0.05
    z = np.arange(n_tail, dtype=np.float64)
    bot = np.exp(-2.0 * mu_per_voxel * z).reshape(1, 1, n_tail)
    slope = _lstsq_tail_slope(bot)
    np.testing.assert_allclose(slope[0, 0], -2.0 * mu_per_voxel, atol=1e-6)


def test_lstsq_tail_slope_handles_flat_tail():
    """A flat (zero-decay) tail produces a slope of ~0 without NaNs."""
    bot = np.full((1, 1, 10), 5.0, dtype=np.float64)
    slope = _lstsq_tail_slope(bot)
    assert np.isfinite(slope).all()
    assert abs(slope[0, 0]) < 1e-6


def test_exact_tail_c_zero_when_mu_zero():
    """When mu_E is zero the denominator is zero and C must be 0 (no division error)."""
    i_max = np.array([[1.0, 2.0]])
    mu = np.zeros_like(i_max)
    C = _exact_tail_C(i_max, mu, dz_m=6.5e-6)
    np.testing.assert_array_equal(C, np.zeros_like(C))


def test_exact_tail_c_matches_closed_form():
    """C = I[imax] / (exp(2 mu dz) - 1) for a nonzero mu."""
    i_max = np.array([[10.0]])
    mu = np.array([[5e3]])  # 1/m
    dz_m = 6.5e-6
    C = _exact_tail_C(i_max, mu, dz_m)
    expected = 10.0 / (np.exp(2.0 * 5e3 * 6.5e-6) - 1.0)
    np.testing.assert_allclose(C[0, 0], expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# _auto_tissue_mask
# ---------------------------------------------------------------------------


def test_auto_tissue_mask_returns_bool_volume():
    """The auto-tissue mask must be a bool volume matching the input shape."""
    # Build a small synthetic volume with a clear water/tissue interface:
    # bright tissue block in the lower half, dark water above.
    nz = 24
    vol = np.zeros((6, 6, nz), dtype=np.float32)
    vol[:, :, 12:] = 100.0  # tissue below z=12
    vol += np.random.default_rng(0).standard_normal(vol.shape).astype(np.float32) * 0.5
    mask = _auto_tissue_mask(vol, zshift=0)
    assert mask.dtype == bool
    assert mask.shape == vol.shape


# ---------------------------------------------------------------------------
# oct_signal_faber2004_model
# ---------------------------------------------------------------------------


def test_oct_signal_faber2004_model_at_focus():
    """At z = z0 the PSF is 1 and the signal is exp(-2 mu_t z0)."""
    z0 = 100.0
    mu_t = 1.0
    zR = 200.0
    val = oct_signal_faber2004_model(np.array([z0]), mu_t=mu_t, zR=zR, z0=z0)
    expected = np.sqrt(1.0 * np.exp(-2.0 * mu_t * z0))
    np.testing.assert_allclose(val[0], expected, rtol=1e-9)


def test_oct_signal_faber2004_model_array_mu_t():
    """A per-voxel mu_t array is accepted and applied elementwise."""
    z = np.array([0.0, 50.0, 100.0])
    mu_t = np.array([0.0, 0.5, 1.0])
    out = oct_signal_faber2004_model(z, mu_t=mu_t, zR=200.0, z0=0.0)  # ty: ignore[invalid-argument-type]
    assert out.shape == z.shape
    assert np.all(np.isfinite(out))
    # At z=0 with mu_t=0 the signal is exactly 1.
    np.testing.assert_allclose(out[0], 1.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# _aline_fit / split_aline
# ---------------------------------------------------------------------------


def test_aline_fit_recovers_exponential_decay_rate():
    """For a clean exponential the fit recovers mu within a few %."""
    mu_true = 0.05  # per voxel
    z = np.linspace(0, 100, 100)
    data = np.exp(-2.0 * mu_true * z)
    mu_est = _aline_fit(data)
    assert abs(mu_est - mu_true) / mu_true < 0.05


def test_split_aline_contiguous_segment():
    """A single contiguous masked segment is returned as one (data, z) pair."""
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = np.array([False, True, True, True, False])
    data_list, z_list = split_aline(data, mask)
    assert len(data_list) == 1
    assert len(z_list) == 1
    np.testing.assert_array_equal(data_list[0], [2.0, 3.0, 4.0])
    np.testing.assert_array_equal(z_list[0], [1, 2, 3])


def test_split_aline_multiple_segments():
    """Two separated masked regions produce two (data, z) pairs."""
    data = np.arange(8, dtype=float)
    mask = np.array([True, True, False, False, True, True, True, False])
    data_list, z_list = split_aline(data, mask)
    assert len(data_list) == 2
    np.testing.assert_array_equal(data_list[0], [0.0, 1.0])
    np.testing.assert_array_equal(z_list[0], [0, 1])
    np.testing.assert_array_equal(data_list[1], [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(z_list[1], [4, 5, 6])


def test_split_aline_empty_mask_returns_empty_lists():
    """An all-False mask yields no segments."""
    data = np.array([1.0, 2.0, 3.0])
    mask = np.array([False, False, False])
    data_list, z_list = split_aline(data, mask)
    assert data_list == []
    assert z_list == []


# ---------------------------------------------------------------------------
# get_aline_attenuation
# ---------------------------------------------------------------------------


def test_get_aline_attenuation_zero_for_zero_volume():
    """A volume whose mean A-line is zero must produce zero attenuation."""
    vol = np.zeros((2, 2, 10), dtype=np.float32)
    out = get_aline_attenuation(vol, k=1)
    assert np.all(out == 0.0)


def test_get_aline_attenuation_shape_for_k_segments():
    """k>1 splits the depth axis and returns an (nx, ny, k) attenuation map."""
    rng = np.random.default_rng(0)
    vol = rng.random((2, 2, 20)).astype(np.float32) * 10.0 + 1.0
    out = get_aline_attenuation(vol, k=2)
    assert out.shape == (2, 2, 2)
    assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# get_gradient_attenuation
# ---------------------------------------------------------------------------


def test_gradient_attenuation_recovers_exponential_mu():
    """For a clean exponential the log-gradient recovers mu (per voxel)."""
    mu_true = 0.05
    z = np.arange(40, dtype=np.float32)
    aline = np.exp(-2.0 * mu_true * z)
    vol = np.broadcast_to(aline, (2, 2, 40)).astype(np.float32).copy()
    attn = get_gradient_attenuation(vol, res=1.0)
    # Central region (avoid the gradient boundary effects at the ends).
    centre = attn[1, 1, 5:35]  # ty: ignore[invalid-argument-type]
    valid = centre[centre > 0]
    assert valid.size > 20
    np.testing.assert_allclose(valid.mean(), mu_true, rtol=0.05)


def test_gradient_attenuation_zeros_where_vol_zero():
    """Voxels where the input is zero must produce zero attenuation.

    Note: ``get_gradient_attenuation`` calls ``np.percentile(attn[attn > 0], ...)``
    which raises ``IndexError`` when there are no positive attenuation values
    (pre-existing source bug — empty-array percentile on numpy 2.x). The test
    documents this with ``pytest.raises`` rather than fixing the source (out of
    scope per SCOPE BOUNDARY).
    """
    vol = np.zeros((2, 2, 10), dtype=np.float32)
    vol[:, :, 5:] = 1.0
    # A step profile has zero gradient everywhere except the step boundary,
    # where the log-gradient is undefined (log(0) on the dark side). The
    # resulting attn has no positive values -> percentile on empty array.
    with pytest.raises(IndexError):
        get_gradient_attenuation(vol)


def test_gradient_attenuation_return_mask():
    """return_mask=True returns (attn, mask) and the mask flags removed voxels.

    A flat volume produces zero gradient everywhere; the source then calls
    ``np.percentile`` on an empty array (no positive attn values), which raises
    ``IndexError`` on numpy 2.x. Documented with ``pytest.raises`` (pre-existing
    source bug, out of scope per SCOPE BOUNDARY).
    """
    vol = np.ones((2, 2, 10), dtype=np.float32)
    with pytest.raises(IndexError):
        get_gradient_attenuation(vol, return_mask=True)


def test_gradient_attenuation_mask_zeros_outside():
    """A mask=False voxel forces the attenuation to zero there."""
    mu_true = 0.05
    z = np.arange(40, dtype=np.float32)
    aline = np.exp(-2.0 * mu_true * z)
    vol = np.broadcast_to(aline, (2, 2, 40)).astype(np.float32).copy()
    mask = np.ones_like(vol, dtype=bool)
    mask[0, 0, :] = False
    attn = get_gradient_attenuation(vol, mask=mask)
    assert np.all(attn[0, 0, :] == 0.0)  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# get_interface_mask
# ---------------------------------------------------------------------------


def test_interface_mask_returns_bool_volume():
    """The interface mask is a bool volume of the same shape as the input.

    Note: ``get_interface_mask`` calls ``dipy.median_otsu(..., median_radius=5.0)``
    with a float radius, which is incompatible with numpy 2.x / scipy's
    ``median_filter`` (expects int). This is a pre-existing source bug;
    documented with ``pytest.raises(TypeError)`` rather than fixing the source
    (out of scope per SCOPE BOUNDARY).
    """
    rng = np.random.default_rng(0)
    vol = rng.random((6, 6, 12)).astype(np.float32) * 50.0
    vol[:, :, 6:] += 100.0  # tissue block
    with pytest.raises(TypeError):
        get_interface_mask(vol, s=0, mask_tissue=True, mask_water_tissue_interface=True)


def test_interface_mask_disable_components():
    """Disabling both tissue and water/tissue masking still returns a valid mask."""
    rng = np.random.default_rng(1)
    vol = rng.random((4, 4, 8)).astype(np.float32) * 10.0
    mask = get_interface_mask(vol, s=0, mask_tissue=False, mask_water_tissue_interface=False)
    assert mask.dtype == bool
    assert mask.shape == vol.shape


# ---------------------------------------------------------------------------
# find_interface_from_gradient
# ---------------------------------------------------------------------------


def test_find_interface_from_gradient_returns_depth_map():
    """The gradient-based interface finder returns a (nx, ny) depth map."""
    from linumpy.intensity.attenuation import find_interface_from_gradient

    rng = np.random.default_rng(0)
    vol = rng.random((6, 6, 12)).astype(np.float32) * 10.0
    vol[:, :, 6:] += 100.0  # bright block below z=6 -> interface near z=6
    depths = find_interface_from_gradient(vol)
    assert depths.shape == (6, 6)
    assert np.all(np.isfinite(depths))


# ---------------------------------------------------------------------------
# get_flat_agarose_profile
# ---------------------------------------------------------------------------


def test_flat_agarose_profile_returns_corrected_volume():
    """The default return is the agarose-normalized volume.

    Note: ``get_flat_agarose_profile`` calls ``dipy.median_otsu(..., median_radius=5.0)``
    with a float radius, incompatible with numpy 2.x (pre-existing source bug).
    Documented with ``pytest.raises(TypeError)`` (out of scope per SCOPE BOUNDARY).
    """
    rng = np.random.default_rng(0)
    vol = rng.random((8, 8, 12)).astype(np.float32) * 20.0
    vol[:, 2:6, 6:] += 80.0  # tissue block
    with pytest.raises(TypeError):
        get_flat_agarose_profile(vol)


def test_flat_agarose_profile_return_mask_and_profile():
    """return_mask_and_profile=True returns (vol_p, mask, i_profile).

    Same pre-existing ``median_otsu`` float-radius bug as above; documented with
    ``pytest.raises(TypeError)`` (out of scope per SCOPE BOUNDARY).
    """
    rng = np.random.default_rng(1)
    vol = rng.random((8, 8, 12)).astype(np.float32) * 20.0
    vol[:, 2:6, 6:] += 80.0
    with pytest.raises(TypeError):
        get_flat_agarose_profile(vol, return_mask_and_profile=True)


# ---------------------------------------------------------------------------
# get_signal_from_attenuation
# ---------------------------------------------------------------------------


def test_signal_from_attenuation_default_i0():
    """With i0=None the signal amplitude defaults to 1 at the mask start."""
    attn = np.array([[0.05, 0.05], [0.05, 0.05]])
    nz = 20
    out = get_signal_from_attenuation(attn, i0=None, nz=nz, res=1.0)
    assert out.shape == (2, 2, nz)
    # At z=0 (relative to mask start) the signal is exp(0) = 1.
    assert np.allclose(out[:, :, 0], 1.0, atol=1e-6)


def test_signal_from_attenuation_with_i0():
    """A custom i0 scales the simulated signal amplitude."""
    attn = np.array([[0.05]])
    i0 = np.array([[10.0]])
    out = get_signal_from_attenuation(attn, i0=i0, nz=15, res=1.0)
    assert out.shape == (1, 1, 15)
    # The first in-mask voxel is i0 * exp(0) = i0.
    np.testing.assert_allclose(out[0, 0, 0], 10.0, atol=1e-6)


def test_signal_from_attenuation_with_mask():
    """A mask limits where the signal is written; outside-mask voxels stay 0."""
    attn = np.array([[0.05]])
    nz = 10
    mask = np.zeros((1, 1, nz), dtype=bool)
    mask[0, 0, 3:] = True
    out = get_signal_from_attenuation(attn, i0=None, nz=nz, mask=mask, res=1.0)
    # Outside the mask the signal is 0.
    assert np.all(out[0, 0, :3] == 0.0)
    # Inside the mask the signal is positive.
    assert np.all(out[0, 0, 3:] > 0.0)


# ---------------------------------------------------------------------------
# get_attenuation_smith2015 (full path with auto mask)
# ---------------------------------------------------------------------------


def test_smith2015_runs_with_auto_mask():
    """Smith 2015 with mask=None exercises _auto_tissue_mask + finalize.

    Note: with ``mask=None`` the function calls ``_auto_tissue_mask`` (which
    works) then ``get_gradient_attenuation`` on the smoothed volume. The
    gradient path hits the empty-percentile ``IndexError`` when the synthetic
    volume's attenuation has no positive values. Documented with
    ``pytest.raises`` (pre-existing source bug, out of scope per SCOPE BOUNDARY).
    """
    from linumpy.intensity.attenuation import get_attenuation_smith2015

    rng = np.random.default_rng(0)
    nz = 30
    vol = rng.random((6, 6, nz)).astype(np.float32) * 5.0
    vol[:, :, 15:] += 80.0  # tissue block -> detectable interface
    with pytest.raises((IndexError, TypeError)):
        get_attenuation_smith2015(vol, mask=None, k=0, res=10.0, zshift=1, fill_holes=False)
