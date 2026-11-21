"""Tests for the Vermeer-2013 depth-resolved attenuation model.

The reference equation (Vermeer et al. 2013, Biomed. Opt. Express, Eq. 17) is

    mu[i] = log(1 + I[i] / sum_{j=i+1..N-1} I[j]) / (2 * dz)

For a pure single-scattering signal ``I[i] = exp(-2 * mu * dz * i)``, the
estimator should recover ``mu`` exactly at every depth at which the deep tail
of the signal is fully contained in the volume.
"""

import numpy as np

from linumpy.intensity.attenuation import get_attenuation_vermeer2013, tissue_column_mask


def test_recovers_uniform_attenuation_from_clean_exponential():
    """A pure exponential decay should recover the true mu within ~1 %."""
    n_z = 500
    dz_m = 10.0e-6  # 10 microns/pixel
    mu_true_per_cm = 50.0  # signal drops to ~0.6 by the end of the volume
    mu_true_per_m = mu_true_per_cm * 100.0

    z = np.arange(n_z)
    aline = np.exp(-2.0 * mu_true_per_m * dz_m * z)
    vol = np.broadcast_to(aline, (4, 4, n_z)).astype(float).copy()

    mu_cm = get_attenuation_vermeer2013(vol, dz=dz_m)

    # Use the central 60 % to avoid the unavoidable tail blow-up at the
    # bottom (Vermeer overshoots when the residual signal beyond the volume
    # is non-negligible compared to the local intensity).
    lo, hi = int(0.1 * n_z), int(0.7 * n_z)
    estimated = mu_cm[2, 2, lo:hi]

    assert np.allclose(estimated, mu_true_per_cm, rtol=0.01), (
        f"mu mismatch: estimated mean={estimated.mean():.3f}, std={estimated.std():.3g}, true={mu_true_per_cm}"
    )


def test_does_not_underestimate_by_factor_two():
    """Regression guard: the previous implementation halved mu."""
    n_z = 500
    dz_m = 10.0e-6
    mu_true_per_cm = 50.0

    z = np.arange(n_z)
    aline = np.exp(-2.0 * mu_true_per_cm * 100.0 * dz_m * z)
    vol = np.broadcast_to(aline, (4, 4, n_z)).astype(float).copy()

    mu_cm = get_attenuation_vermeer2013(vol, dz=dz_m)
    centre = mu_cm[2, 2, 50:300].mean()

    # The historical bug (denominator including I[i]) produced mu ~= 25 here.
    assert centre > 0.75 * mu_true_per_cm, f"mu under-estimated: {centre:.2f} vs true {mu_true_per_cm}"


# --- New methods (Smith 2015, Liu 2019, Li 2020, Faber 2004) -----------


def _exp_volume(mu_per_cm: float = 50.0, dz_m: float = 10.0e-6, n_z: int = 500, shape_xy: tuple = (8, 8)) -> np.ndarray:
    """Synthetic uniform-attenuation OCT volume in (X, Y, Z) order."""
    z = np.arange(n_z)
    aline = np.exp(-2.0 * mu_per_cm * 100.0 * dz_m * z).astype(np.float32)
    return np.broadcast_to(aline, (*shape_xy, n_z)).astype(np.float32).copy()


def test_extended_alias_emits_deprecation_warning():
    """The legacy name still works but warns."""
    import warnings

    from linumpy.intensity.attenuation import (
        get_attenuation_smith2015,
        get_extended_attenuation_vermeer2013,
    )

    vol = _exp_volume()
    mask = np.ones_like(vol, dtype=bool)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = get_extended_attenuation_vermeer2013(vol, mask=mask, k=0, res=10.0)

    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    direct = get_attenuation_smith2015(vol, mask=mask, k=0, res=10.0)
    assert np.array_equal(out, direct)


def test_liu2019_recovers_uniform_attenuation():
    """Liu 2019 with exact-form regularization should match the truth in the bulk."""
    from linumpy.intensity.attenuation import get_attenuation_liu2019

    mu_true = 50.0
    vol = _exp_volume(mu_per_cm=mu_true, n_z=500)
    mask = np.ones_like(vol, dtype=bool)

    mu_cm = get_attenuation_liu2019(vol, mask=mask, k=0, res=10.0, tail_fit_voxels=30)

    # Central 60 % of the depth axis.
    centre = mu_cm[2, 2, 50:350]
    assert np.allclose(centre, mu_true, rtol=0.02), (
        f"liu mismatch: mean={centre.mean():.3f}, std={centre.std():.3g}, true={mu_true}"
    )


def test_liu2019_tail_more_accurate_than_vermeer():
    """The exact-form C should reduce the tail blow-up vs C=0."""
    from linumpy.intensity.attenuation import get_attenuation_liu2019

    mu_true = 50.0
    vol = _exp_volume(mu_per_cm=mu_true, n_z=300)
    mask = np.ones_like(vol, dtype=bool)

    mu_vermeer = get_attenuation_vermeer2013(vol, dz=10.0e-6)
    mu_liu = get_attenuation_liu2019(vol, mask=mask, k=0, res=10.0, tail_fit_voxels=30)

    # Look at the last 20 voxels (excluding the very last one which is 0 by
    # construction in both methods).
    tail_err_vermeer = np.abs(mu_vermeer[2, 2, -25:-5] - mu_true).mean()
    tail_err_liu = np.abs(mu_liu[2, 2, -25:-5] - mu_true).mean()
    assert tail_err_liu < tail_err_vermeer, (
        f"Liu tail error ({tail_err_liu:.2f}) should be < Vermeer tail error ({tail_err_vermeer:.2f})"
    )


def test_li2020_recovers_attenuation_with_noise_floor():
    """Li 2020 should recover mu when a noise floor is present.

    Builds a 600-voxel A-line with mu = 20 /cm so the signal is well
    above a 1e-3 noise floor for the first ~250 voxels; Li's truncation
    should keep that depth range intact and recover mu within ~15 %.
    """
    from linumpy.intensity.attenuation import get_attenuation_li2020

    mu_true = 20.0
    n_z = 600
    dz_m = 10.0e-6
    z = np.arange(n_z)
    aline = np.exp(-2.0 * mu_true * 100.0 * dz_m * z).astype(np.float32)
    vol = np.broadcast_to(aline, (8, 8, n_z)).astype(np.float32).copy()
    rng = np.random.default_rng(0)
    noise_floor = 1e-3
    vol_noisy = vol + noise_floor + 1e-4 * rng.standard_normal(vol.shape).astype(np.float32)
    mask = np.ones_like(vol_noisy, dtype=bool)

    mu_cm = get_attenuation_li2020(vol_noisy, mask=mask, k=0, res=10.0, snr_threshold_db=6.0, tail_fit_voxels=40)
    # Inspect the well-above-noise region (signal at z=80 is ~3e-4,
    # ~30x the noise std). The trailing voxels approach the truncation
    # depth and inherit Vermeer's tail blow-up.
    centre = mu_cm[2, 2, 30:80]
    valid = centre[centre > 0]
    assert valid.size > 40, f"Li truncation removed too many voxels ({valid.size})"
    assert abs(valid.mean() - mu_true) / mu_true < 0.10, (
        f"li mismatch: mean={valid.mean():.3f}, std={valid.std():.3g}, true={mu_true}"
    )


def test_faber2004_runs_with_noncontiguous_mask():
    """Faber 2004 should run on a non-contiguous mask without crashing.

    Regression guard for the historical shape bug where
    ``z[zp::] - z[zp]`` was passed instead of ``z[mask_aline] - z[zp]``,
    which raised ``ValueError: shape mismatch`` whenever the mask had
    interior gaps. Note: the underlying Lorentzian-times-exponential
    LSQ is sensitive to the hard-coded initial guess (z0=0, zR=100,
    mu_t=1e-3) and is not exercised here for numerical accuracy --
    only that the shapes of the data and depth arrays match.
    """
    from linumpy.intensity.attenuation import get_attenuation_faber2004

    mu_true = 50.0  # 1/cm
    mu_true_per_m = mu_true * 100.0
    dz_m = 10.0e-6
    n_z = 60
    z = np.arange(n_z) * dz_m  # metres
    z0 = 30 * dz_m
    zR = 200e-6
    psf = 1.0 / (((z - z0) / zR) ** 2 + 1.0)
    aline = (psf * np.exp(-2.0 * mu_true_per_m * z)).astype(np.float32)
    vol = np.broadcast_to(aline, (3, 3, n_z)).astype(np.float32).copy()

    # Non-contiguous mask: first valid voxel at index 5, gap at 20-22.
    mask = np.ones((3, 3, n_z), dtype=bool)
    mask[:, :, :5] = False
    mask[:, :, 20:23] = False

    attn, r_length, z_focus = get_attenuation_faber2004(vol, mask=mask, dz=dz_m, N=0)
    assert attn.shape == (3, 3)
    assert r_length.shape == (3, 3)
    assert z_focus.shape == (3, 3)
    assert np.all(np.isfinite(attn))


def test_smooth_optical_depth_xy_softens_walls_inside_mask():
    """A hard-walled optical-depth step is softened in XY; outside-mask and depth stay put."""
    from linumpy.intensity.attenuation import smooth_optical_depth_xy

    od = np.zeros((40, 40, 5), dtype=np.float32)
    od[20:, :, :] = np.arange(5, dtype=np.float32)  # wall at x=20, deeper = larger
    mask = np.ones_like(od, dtype=bool)
    mask[:, :2, :] = False
    od[:, :2, :] = 7.0

    out = smooth_optical_depth_xy(od, mask, sigma=3.0)

    assert np.all(out[:, :2, :] == 7.0)
    # wall is now a ramp: values just left of it rise, just right of it drop
    assert 0 < out[19, 20, 4] < 4 and 0 < out[20, 20, 4] < 4
    # far from the wall the depth profile is unchanged
    np.testing.assert_allclose(out[35, 20, :], np.arange(5), atol=1e-4)
    assert smooth_optical_depth_xy(od, mask, sigma=0) is od


def test_footprint_mask_smooths_the_dim_deep_voxels_too():
    from linumpy.intensity.attenuation import smooth_optical_depth_xy, tissue_footprint_mask

    od = np.zeros((40, 40, 6), dtype=np.float32)
    od[18:22, 18:22, 3:] = 4.0  # A-lines whose deep voxels fell below the tissue threshold
    mask = np.ones((40, 40, 6), dtype=bool)
    mask[18:22, 18:22, 3:] = False
    kept = smooth_optical_depth_xy(od, mask, sigma=3.0)
    assert kept[20, 20, 4] == 4.0  # outside the 3D mask: left as an island
    footprint = tissue_footprint_mask(mask)
    assert footprint.all()
    smoothed = smooth_optical_depth_xy(od, footprint, sigma=3.0)
    assert smoothed[20, 20, 4] < 2.0
    assert smoothed[20, 26, 4] > 0.0


def _slab_with_overestimated_depth(scale: float, shape: tuple = (80, 80, 30)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tissue whose true optical depth is ``scale`` times the estimated one.

    Bright and dim A-lines decay at different rates, as white and grey matter do.
    """
    nx, ny, nz = shape
    rate = np.where(np.arange(nx)[:, None] < nx // 4, 0.06, 0.03) * np.ones((1, ny))
    od = rate[:, :, None] * np.arange(nz)[None, None, :]
    base = np.where(np.arange(nx)[:, None] < nx // 4, 2.0, 1.0) * np.ones((1, ny))
    vol = base[:, :, None] * np.exp(-2.0 * scale * od)
    return vol.astype(np.float32), od.astype(np.float32), np.ones(shape, dtype=bool)


def test_calibrated_strength_recovers_the_scale_of_the_optical_depth():
    from linumpy.intensity.attenuation import calibrate_compensation_strength

    for scale in (0.1, 0.45, 0.8):
        vol, od, mask = _slab_with_overestimated_depth(scale)
        strength = calibrate_compensation_strength(vol, od, mask, erosion_px=5, min_alines=100)
        assert strength is not None
        assert abs(strength - scale) < 0.01
        flat = vol / np.exp(-2.0 * strength * od)
        level = np.median(flat[10:-10, 10:-10, 2:21], axis=(0, 1))
        assert level.max() / level.min() < 1.02


def test_calibrated_strength_stays_inside_its_bounds():
    from linumpy.intensity.attenuation import calibrate_compensation_strength

    vol, od, mask = _slab_with_overestimated_depth(0.0)
    assert calibrate_compensation_strength(vol, od, mask, erosion_px=5, min_alines=100) == 0.0
    # A level that rises with depth needs no compensation either.
    rising = vol * np.linspace(1.0, 1.5, vol.shape[2], dtype=np.float32)
    assert calibrate_compensation_strength(rising, od, mask, erosion_px=5, min_alines=100) == 0.0
    vol, od, mask = _slab_with_overestimated_depth(3.0)
    assert calibrate_compensation_strength(vol, od, mask, max_strength=1.0, erosion_px=5, min_alines=100) == 1.0


def test_calibrated_strength_needs_tissue_and_depth():
    from linumpy.intensity.attenuation import calibrate_compensation_strength

    vol, od, mask = _slab_with_overestimated_depth(0.45)
    assert calibrate_compensation_strength(vol, od, mask, erosion_px=5, min_alines=10**6) is None
    assert calibrate_compensation_strength(vol[:, :, :4], od[:, :, :4], mask[:, :, :4], erosion_px=5) is None
    sparse = np.zeros_like(mask)
    sparse[:3, :3] = True
    assert calibrate_compensation_strength(vol, od, sparse, erosion_px=0, min_alines=100) is None


def test_calibrated_strength_ignores_planes_below_the_stack_depth():
    """Signal that collapses below the planes that go into the stack does not change the fit."""
    from linumpy.intensity.attenuation import calibrate_compensation_strength

    vol, od, mask = _slab_with_overestimated_depth(0.3)
    cliff = vol.copy()
    cliff[:, :, 22:] *= 0.05
    a = calibrate_compensation_strength(vol, od, mask, n_planes=20, erosion_px=5, min_alines=100)
    b = calibrate_compensation_strength(cliff, od, mask, n_planes=20, erosion_px=5, min_alines=100)
    assert a == b


def test_exact_tail_gives_alines_without_decay_the_median_rate():
    from linumpy.intensity.attenuation import _exact_tail_C

    i_end = np.array([[2.0, 2.0, 2.0, 2.0]])
    mu = np.array([[1000.0, 3000.0, 0.0, 2000.0]])  # 1/m
    c = _exact_tail_C(i_end, mu, 10e-6)
    np.testing.assert_allclose(c[0, 0], 2.0 / np.expm1(2 * 1000.0 * 10e-6))
    # The A-line with no decay does not fall back to the bare estimator (C = 0).
    np.testing.assert_allclose(c[0, 2], 2.0 / np.expm1(2 * 2000.0 * 10e-6))
    limited = _exact_tail_C(i_end, mu, 10e-6, valid=np.array([[True, True, True, False]]))
    assert limited[0, 3] == 0.0
    np.testing.assert_allclose(limited[0, 2], 2.0 / np.expm1(2 * 2000.0 * 10e-6))
    assert np.all(_exact_tail_C(i_end, np.zeros_like(mu), 10e-6) == 0.0)


def test_tail_is_fitted_where_the_mask_ends():
    from linumpy.intensity.attenuation import _tail_at_aline_end

    nz = 60
    z = np.arange(nz)
    aline = np.exp(-0.05 * z)
    aline[30:] = 1e-6  # nothing left below the cut
    vol = np.broadcast_to(aline, (3, 3, nz)).copy()
    mask = np.zeros(vol.shape, dtype=bool)
    mask[:, :, :30] = True
    mask[0, 0, :] = False
    mask[1, 1, 27:] = False
    mask[1, 1, :25] = False  # two voxels: too few to fit
    i_end, slope, fitted = _tail_at_aline_end(vol, mask, 20)
    np.testing.assert_allclose(slope[2, 2], -0.05, atol=1e-9)
    np.testing.assert_allclose(i_end[2, 2], aline[29])
    assert not fitted[0, 0] and i_end[0, 0] == 0.0
    assert not fitted[1, 1] and slope[1, 1] == 0.0


def _cut_alines(mu_per_cm: float = 20.0, noise: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Tissue that sinks into a noise floor half-way down, agarose, and empty padding."""
    nz, dz_m = 200, 10e-6
    z = np.arange(nz)
    vol = np.zeros((24, 24, nz), dtype=np.float32)
    mask = np.zeros(vol.shape, dtype=bool)
    vol[:8] = 100.0 * np.exp(-2.0 * mu_per_cm * 100.0 * dz_m * z) + noise
    mask[:8] = True
    vol[8:12] = noise  # agarose: a quarter of what lies outside the tissue
    return vol, mask


def test_li2020_does_not_diverge_towards_the_cut():
    """The estimate stays at the true rate up to where the A-line is cut."""
    from linumpy.intensity.attenuation import get_attenuation_li2020

    vol, mask = _cut_alines(mu_per_cm=20.0)
    attn = get_attenuation_li2020(vol, mask=mask, k=0, res=10.0, snr_threshold_db=6.0)
    aline = attn[4, 4]
    # The signal is under four times the floor from about plane 88 on. The
    # estimate holds the true rate above the cut and keeps it below, down to
    # the end of the tissue mask.
    np.testing.assert_allclose(aline[5:86], 20.0, rtol=0.1)
    np.testing.assert_allclose(aline[95:], aline[94], rtol=1e-6)
    np.testing.assert_allclose(aline[150], 20.0, rtol=0.15)
    assert np.all(attn[10, 10] == 0.0)  # agarose, outside the mask


def test_li2020_noise_floor_leaves_out_empty_padding():
    """Padding that outnumbers the agarose must not zero the noise floor."""
    from linumpy.intensity.attenuation import get_attenuation_li2020

    vol, mask = _cut_alines(mu_per_cm=20.0)
    assert (vol[~mask] == 0).mean() > 0.5
    with_floor = get_attenuation_li2020(vol, mask=mask, k=0, res=10.0)
    given = get_attenuation_li2020(vol, mask=mask, k=0, res=10.0, noise_floor=1.0)
    np.testing.assert_allclose(with_floor, given)
    # With a floor of zero the A-line is not cut, and the flat floor at its
    # end reads as almost no attenuation there.
    zero_floor = get_attenuation_li2020(vol, mask=mask, k=0, res=10.0, noise_floor=0.0)
    assert zero_floor[4, 4, 150] < 0.5 * with_floor[4, 4, 150]


def test_vermeer_core_estimates_the_last_in_mask_voxel_and_carries_it_down():
    """The voxel where the summed range ends has an estimate, and deeper voxels keep it."""
    nz, dz = 50, 10e-6
    aline = np.exp(-2.0 * 2000.0 * dz * np.arange(nz)).astype(np.float32)
    vol = np.broadcast_to(aline, (2, 2, nz)).copy()
    mask = np.zeros(vol.shape, dtype=bool)
    mask[:, :, 5:30] = True
    tail = aline[29] / np.expm1(2.0 * 2000.0 * dz)
    mu = get_attenuation_vermeer2013(vol, dz=dz, mask=mask, C=np.full((2, 2), tail, dtype=np.float32))
    np.testing.assert_allclose(mu[0, 0, 5:30], 20.0, rtol=1e-3)
    assert np.all(mu[0, 0, :5] == 0.0)
    np.testing.assert_allclose(mu[0, 0, 30:], mu[0, 0, 29])
    # A scalar tail constant is accepted too.
    scalar = get_attenuation_vermeer2013(vol, dz=dz, mask=mask, C=float(tail))
    np.testing.assert_allclose(scalar, mu)


def test_interface_helpers_match_a_plain_loop():
    from linumpy.geometry.crop import mask_under_interface
    from linumpy.geometry.interface import get_interface_depth_from_mask

    rng = np.random.default_rng(0)
    mask = rng.random((9, 7, 12)) > 0.7
    mask[0, 0] = False  # an A-line without tissue reports depth 0
    expected = np.zeros((9, 7))
    for x in range(9):
        for y in range(7):
            hits = np.flatnonzero(mask[x, y])
            if hits.size:
                expected[x, y] = hits[0]
    depths = get_interface_depth_from_mask(mask)
    np.testing.assert_array_equal(depths, expected)
    under = mask_under_interface(mask, depths, return_mask=True)
    for x in range(9):
        for y in range(7):
            np.testing.assert_array_equal(under[x, y], np.arange(12) >= expected[x, y])


def test_binned_median_matches_the_full_one_on_smooth_data_and_keeps_the_shape():
    from linumpy.intensity.attenuation import _median_xy_filter

    rng = np.random.default_rng(1)
    xx, yy = np.mgrid[:61, :47]
    smooth = (10.0 + 3.0 * np.sin(xx / 30.0) * np.cos(yy / 25.0)).astype(np.float32)
    vol = np.repeat(smooth[:, :, None], 3, axis=2) * np.array([1.0, 0.8, 0.6], dtype=np.float32)
    noisy = vol + rng.normal(0, 0.3, vol.shape).astype(np.float32)
    noisy[30, 20, 1] = 500.0  # a single outlier does not survive either filter
    full = _median_xy_filter(noisy, 6)
    binned = _median_xy_filter(noisy, 6, bin_xy=2)
    assert binned.shape == noisy.shape and binned.dtype == noisy.dtype
    assert binned[30, 20, 1] < 15.0
    inner = (slice(8, -8), slice(8, -8))
    np.testing.assert_allclose(binned[inner], full[inner], rtol=0.04)
    # Depth is not mixed: each plane keeps its own level.
    np.testing.assert_allclose(binned[inner][..., 2].mean() / binned[inner][..., 0].mean(), 0.6, rtol=0.01)
    assert _median_xy_filter(noisy, 0, bin_xy=2) is noisy


def test_smooth_optical_depth_xy_equals_one_filter_without_depth_smoothing():
    from scipy.ndimage import gaussian_filter

    from linumpy.intensity.attenuation import smooth_optical_depth_xy, tissue_footprint_mask

    rng = np.random.default_rng(2)
    od = rng.random((30, 26, 4)).astype(np.float32)
    varying = rng.random(od.shape) > 0.3
    for mask in (varying, tissue_footprint_mask(varying)):
        sig = (3.0, 3.0, 0.0)
        weight = gaussian_filter(mask.astype(np.float32), sig)
        ref = gaussian_filter(np.where(mask, od, 0.0).astype(np.float32), sig) / np.maximum(weight, 1e-3)
        ref = np.where(mask, ref, od)
        np.testing.assert_allclose(smooth_optical_depth_xy(od, mask, 3.0), ref, rtol=1e-6, atol=1e-7)


def test_li2020_can_cut_without_subtracting_the_floor():
    from linumpy.intensity.attenuation import get_attenuation_li2020

    vol, mask = _cut_alines(mu_per_cm=20.0)
    cut_only = get_attenuation_li2020(vol, mask=mask, k=0, res=10.0, subtract_noise_floor=False)
    subtracted = get_attenuation_li2020(vol, mask=mask, k=0, res=10.0)
    # The floor left in the signal flattens the decay towards the cut.
    assert cut_only[4, 4, 80] < subtracted[4, 4, 80]
    np.testing.assert_allclose(cut_only[4, 4, 5:20], 20.0, rtol=0.1)


def _two_level_slab(n_planes: int = 40, size: int = 120, decay: float = 0.12) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a steep slab ``(Z, Y, X)`` with dim and bright tissue in agarose, and both tissue masks."""
    rng = np.random.default_rng(11)
    fall = np.exp(-decay * np.arange(n_planes))[:, None, None]
    vol = np.full((n_planes, size, size), 1.0, dtype=np.float32) * (0.5 + 0.5 * fall)
    dim = np.zeros((size, size), dtype=bool)
    bright = np.zeros((size, size), dtype=bool)
    dim[20:100, 20:60] = True
    bright[20:100, 60:100] = True
    vol[:, dim] = (20.0 * fall).repeat(int(dim.sum()), axis=1).reshape(n_planes, -1)
    vol[:, bright] = (200.0 * fall).repeat(int(bright.sum()), axis=1).reshape(n_planes, -1)
    vol *= 1.0 + 0.02 * rng.standard_normal(vol.shape).astype(np.float32)
    return vol.astype(np.float32), dim, bright


def _away_from(tissue: np.ndarray, margin: int = 5) -> np.ndarray:
    """Return the agarose more than *margin* voxels from the tissue (the smoothing blurs the edge)."""
    from scipy.ndimage import binary_dilation

    return ~binary_dilation(tissue, iterations=margin)


def test_tissue_column_mask_keeps_dim_tissue_and_leaves_out_agarose():
    vol, dim, bright = _two_level_slab()

    columns = tissue_column_mask(vol)

    assert columns[dim].mean() > 0.95
    assert columns[bright].mean() > 0.95
    assert not columns[_away_from(dim | bright)].any()


def test_tissue_column_mask_does_not_depend_on_the_height_of_the_surface():
    vol, dim, bright = _two_level_slab()
    # Half of the tissue starts eight planes lower: water above it.
    lowered = vol.copy()
    lowered[8:, 60:100, :] = vol[:-8, 60:100, :]
    lowered[:8, 60:100, :] = 0.5

    columns = tissue_column_mask(lowered)

    tissue = dim | bright
    assert columns[tissue].mean() > 0.95
    assert not columns[_away_from(tissue)].any()


def test_tissue_column_mask_of_an_empty_or_flat_slab_is_empty():
    assert not tissue_column_mask(np.zeros((10, 16, 16), dtype=np.float32)).any()
    assert not tissue_column_mask(np.ones((10, 16, 16), dtype=np.float32)).any()
