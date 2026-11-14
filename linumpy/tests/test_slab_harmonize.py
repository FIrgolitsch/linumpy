"""Tests for linumpy/intensity/slab_harmonize.py"""

import numpy as np
import pytest

from linumpy.intensity.slab_harmonize import (
    coarse_log_tissue,
    depth_band_field,
    harmonize_slab_intensity,
    plane_log_gains,
    slab_log_gains,
    slab_period,
)

PERIOD = 10
N_SLABS = 8


def _bounds(n_slabs=N_SLABS, period=PERIOD):
    return [(s * period, (s + 1) * period) for s in range(n_slabs)]


def _tissue(shape=(N_SLABS * PERIOD, 96, 96), seed=0, drift_per_plane=0.0):
    """Smooth anatomy inside a round footprint, optionally drifting with depth."""
    rng = np.random.default_rng(seed)
    nz, ny, nx = shape
    yy, xx = np.mgrid[:ny, :nx]
    base = 0.25 + 0.08 * np.sin(xx / 17.0) * np.cos(yy / 23.0)
    drift = 1.0 + drift_per_plane * np.arange(nz)[:, None, None]
    vol = (base[None] * drift * rng.uniform(0.95, 1.05, size=shape)).astype(np.float32)
    mask = np.broadcast_to(((yy - ny / 2) ** 2 + (xx - nx / 2) ** 2) < (0.42 * ny) ** 2, shape).copy()
    vol[~mask] = 0.0
    return vol, mask


def _slab_levels(vol, mask, bounds):
    return np.array([np.median(vol[a:b][mask[a:b]]) for a, b in bounds])


def test_coarse_log_tissue_averages_only_tissue():
    vol = np.full((2, 16, 16), 0.5, dtype=np.float32)
    mask = np.zeros((2, 16, 16), dtype=bool)
    mask[:, :, :12] = True
    level, weight = coarse_log_tissue(vol, mask, block=8)
    assert level.shape == (2, 2, 2)
    assert weight[0, 0, 0] == 1.0
    assert weight[0, 0, 1] == 0.5
    assert level[0, 0, 1] == pytest.approx(np.log(0.5), abs=1e-5)


def test_coarse_log_tissue_rejects_a_plane_smaller_than_one_cell():
    with pytest.raises(ValueError, match="smaller than one"):
        coarse_log_tissue(np.ones((2, 4, 4), np.float32), np.ones((2, 4, 4), bool), block=8)


def test_slab_period_is_the_median_height():
    assert slab_period([(0, 19), (19, 38), (38, 76), (76, 95)]) == 19


def test_slab_gains_recover_per_slab_factors():
    vol, mask = _tissue()
    bounds = _bounds()
    factors = np.array([1.0, 1.6, 0.7, 1.0, 2.0, 1.1, 0.5, 1.0])
    for (a, b), f in zip(bounds, factors, strict=True):
        vol[a:b] *= f
    level, weight = coarse_log_tissue(vol, mask, block=8)
    gains, info = slab_log_gains(level, weight, bounds)
    recovered = np.exp(gains - gains.mean())
    expected = factors / np.exp(np.log(factors).mean())
    # The loose pull towards the common level shrinks the estimate a little.
    np.testing.assert_allclose(recovered, expected, rtol=0.12)
    assert np.all(np.isfinite(info["steps"]))


def test_slab_gains_ignore_a_step_across_a_missing_slice():
    vol, mask = _tissue()
    bounds = [(0, 10), (10, 20), (20, 40), (40, 50), (50, 60), (60, 70), (70, 80)]
    level, weight = coarse_log_tissue(vol, mask, block=8)
    _gains, info = slab_log_gains(level, weight, bounds)
    # Slab 2 is two cuts tall, so slab 3 does not start one cut below it.
    assert np.isnan(info["steps"][2])
    assert np.isfinite(info["steps"][0])


def test_plane_gains_ramp_over_the_blended_planes():
    gains = np.array([0.0, 1.0])
    out = plane_log_gains(gains, [(0, 10), (10, 20)], 20, blend_planes=4)
    assert out[9] == 0.0
    assert 0.0 < out[10] < out[13] < 1.0
    assert out[14] == 1.0


def test_depth_band_field_finds_a_periodic_ramp_and_ignores_a_trend():
    nz = N_SLABS * PERIOD
    ramp = -0.3 * (np.arange(nz) % PERIOD) / PERIOD
    trend = 0.004 * np.arange(nz)
    level = np.broadcast_to((ramp + trend)[:, None, None], (nz, 12, 12)).astype(np.float32).copy()
    weight = np.ones_like(level)
    field = depth_band_field(level, weight, PERIOD, sigma_cells=2.0)
    inner = slice(PERIOD, nz - PERIOD)
    expected = ramp - ramp[:PERIOD].mean()
    np.testing.assert_allclose(field[inner, 6, 6], expected[inner], atol=0.02)


def test_harmonize_removes_slab_steps_and_depth_bands():
    vol, mask = _tissue()
    clean = vol.copy()
    bounds = _bounds()
    factors = np.array([1.0, 1.5, 0.8, 1.0, 1.8, 1.2, 0.6, 1.0])
    depth = np.exp(-0.5 * np.arange(PERIOD) / PERIOD)
    for (a, b), f in zip(bounds, factors, strict=True):
        vol[a:b] *= f * depth[:, None, None]
    before = _slab_levels(vol, mask, bounds)
    info = harmonize_slab_intensity(vol, mask, bounds, slab_gain=True, band_sigma_px=24, target_level=0.25, block=8)
    after = _slab_levels(vol, mask, bounds)
    assert after.std() / after.mean() < 0.25 * before.std() / before.mean()
    # Depth ramp inside the middle slabs is flattened.
    a, b = bounds[3]
    profile = np.array([np.median(vol[z][mask[z]]) for z in range(a, b)])
    assert profile.max() / profile.min() < 1.1
    # The brain-wide level lands on the requested value.
    assert np.median(vol[mask]) == pytest.approx(0.25, rel=0.08)
    assert info["period"] == PERIOD
    # In-plane structure is kept: each plane still correlates with the clean one.
    z = a + PERIOD // 2
    assert np.corrcoef(vol[z][mask[z]], clean[z][mask[z]])[0, 1] > 0.95


def test_harmonize_leaves_background_and_a_consistent_volume_alone():
    vol, mask = _tissue()
    original = vol.copy()
    harmonize_slab_intensity(vol, mask, _bounds(), slab_gain=True, band_sigma_px=24, block=8)
    assert np.all(vol[~mask] == 0.0)
    ratio = vol[mask] / original[mask]
    assert np.percentile(np.abs(ratio - 1.0), 99) < 0.03


def test_harmonize_keeps_a_structure_that_enters_the_stack():
    """A bright region that appears part-way down must not darken the rest."""
    vol, mask = _tissue()
    nz, ny, nx = vol.shape
    bright = np.zeros((ny, nx), dtype=bool)
    bright[: ny // 3] = True
    for z in range(nz // 2, nz):
        vol[z][bright & mask[z]] *= 2.0
    original = vol.copy()
    harmonize_slab_intensity(vol, mask, _bounds(), slab_gain=True, band_sigma_px=0, block=8)
    plain = mask.copy()
    plain[:, bright] = False
    ratio = vol[plain] / original[plain]
    assert ratio.max() / ratio.min() < 1.06


def test_harmonize_target_level_alone_is_one_scale_factor():
    vol, mask = _tissue()
    original = vol.copy()
    info = harmonize_slab_intensity(vol, mask, _bounds(), slab_gain=False, band_sigma_px=0, target_level=0.5, block=8)
    ratio = vol[mask] / original[mask]
    assert ratio.std() < 1e-5
    assert ratio.mean() == pytest.approx(info["scale"], rel=1e-4)
