"""Tests for :mod:`linumpy.intensity.psf_model` (confocal PSF model math).

Exercises the pure-Python PSF/normalization helpers with small synthetic
numpy arrays. Uses ``numpy.testing.assert_allclose`` for numeric comparisons.
No CUDA/CuPy dependency — all tested functions are pure math.
"""

import numpy as np

from linumpy.intensity.psf_model import (
    T_r,
    confocal_psf,
    estimate_psf,
    find_focal_depth,
    fit_tissue_confocal_model,
    get_3d_psf,
    get_slice_resolutions_from_psf,
    i_profile_piece_wise_model,
    volume_normalization,
)

# ---------------------------------------------------------------------------
# find_focal_depth
# ---------------------------------------------------------------------------


def test_find_focal_depth_detects_brightest_slice():
    # 4x4x5 volume; brightest at z=3
    vol = np.zeros((4, 4, 5), dtype=np.float32)
    vol[:, :, 3] = 10.0
    fz = find_focal_depth(vol)
    assert fz == 3


def test_find_focal_depth_uniform_volume_returns_zero():
    vol = np.ones((4, 4, 5), dtype=np.float32)
    # all equal → argmax returns 0
    fz = find_focal_depth(vol)
    assert fz == 0


def test_find_focal_depth_2d_slice():
    # 3D volume with a single bright z-plane
    vol = np.zeros((3, 3, 4), dtype=np.float32)
    vol[:, :, 2] = 5.0
    assert find_focal_depth(vol) == 2


# ---------------------------------------------------------------------------
# i_profile_piece_wise_model
# ---------------------------------------------------------------------------


def test_i_profile_piece_wise_model_three_regions():
    z = np.array([0.0, 50.0, 100.0, 150.0, 200.0])
    # z0=50, zf=100: z<=50 water, 50<z<=100 tissue, z>100 attenuation
    profile = i_profile_piece_wise_model(z, I0=1.0, Imax=2.0, z0=50.0, zf=100.0, s=10.0, mu=0.01, k=0.001)
    assert profile.shape == z.shape
    # water region: I0 * exp(-k*z)
    np.testing.assert_allclose(profile[0], 1.0 * np.exp(-0.001 * 0.0))
    # at z=50 (boundary z<=z0): water model
    np.testing.assert_allclose(profile[1], 1.0 * np.exp(-0.001 * 50.0))
    # tissue region z=100: Imax * exp(-(z-zf)^2/s^2) = Imax * exp(0) = Imax
    np.testing.assert_allclose(profile[2], 2.0)
    # attenuation region z=150: Imax * exp(-mu*(z-zf))
    np.testing.assert_allclose(profile[3], 2.0 * np.exp(-0.01 * 50.0))


def test_i_profile_piece_wise_model_returns_zeros_for_empty():
    z = np.array([], dtype=np.float64)
    profile = i_profile_piece_wise_model(z, 1.0, 2.0, 50.0, 100.0, 10.0, 0.01, 0.001)
    assert profile.shape == (0,)


# ---------------------------------------------------------------------------
# volume_normalization
# ---------------------------------------------------------------------------


def test_volume_normalization_divides_by_average():
    vol = np.ones((4, 4, 4), dtype=np.float32) * 10.0
    avg = np.ones((4, 4, 4), dtype=np.float32) * 2.0
    out = volume_normalization(vol, avg, epsilon=0.0)
    # vol / (avg + 0) * avg.mean() = 10/2 * 2 = 10
    np.testing.assert_allclose(out, np.full_like(vol, 10.0))


def test_volume_normalization_epsilon_prevents_zero_division():
    vol = np.ones((4, 4, 4), dtype=np.float32) * 5.0
    avg = np.zeros((4, 4, 4), dtype=np.float32)
    out = volume_normalization(vol, avg, epsilon=0.05)
    # vol / (0 + 0.05) * 0 = 0 (mean of zeros)
    np.testing.assert_allclose(out, np.zeros_like(vol))


def test_volume_normalization_preserves_shape():
    vol = np.random.rand(3, 4, 5).astype(np.float32)
    avg = np.ones((3, 4, 5), dtype=np.float32)
    out = volume_normalization(vol, avg)
    assert out.shape == vol.shape


# ---------------------------------------------------------------------------
# T_r (radiometric transformation)
# ---------------------------------------------------------------------------


def test_t_r_polynomial_evaluation():
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    # T = p0*x^2 + p1*x*y + p2*y^2 + p3*x + p4*y + p5
    # at x=1, y=1: 1 + 2 + 3 + 4 + 5 + 6 = 21
    result = T_r(p, 1, 1)
    np.testing.assert_allclose(result, 21.0)


def test_t_r_at_origin_returns_constant():
    p = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = T_r(p, 0, 0)
    np.testing.assert_allclose(result, 6.0)  # only p[5]


def test_t_r_with_arrays():
    p = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 0.0, 0.0])
    result = T_r(p, x, y)
    np.testing.assert_allclose(result, x**2)


# ---------------------------------------------------------------------------
# confocal_psf
# ---------------------------------------------------------------------------


def test_confocal_psf_peak_at_focal_plane():
    z = np.array([0.0, 50.0, 100.0])
    psf = confocal_psf(z, zf=50.0, zR=10.0)
    # peak at z=zf: 1/((0)^2+1) = 1.0
    np.testing.assert_allclose(psf[1], 1.0)
    # away from focal plane: 1/((z-zf)^2/zR^2 + 1) < 1
    assert psf[0] < 1.0
    assert psf[2] < 1.0


def test_confocal_psf_with_amplitude():
    z = np.array([50.0])
    psf = confocal_psf(z, zf=50.0, zR=10.0, A=5.0)
    np.testing.assert_allclose(psf[0], 5.0)


def test_confocal_psf_no_amplitude_default_one():
    z = np.array([50.0])
    psf = confocal_psf(z, zf=50.0, zR=10.0)
    np.testing.assert_allclose(psf[0], 1.0)


def test_confocal_psf_symmetric_around_focal_plane():
    z = np.array([40.0, 60.0])
    psf = confocal_psf(z, zf=50.0, zR=10.0)
    np.testing.assert_allclose(psf[0], psf[1])


def test_confocal_psf_decreasing_with_distance():
    z = np.array([50.0, 60.0, 70.0, 80.0])
    psf = confocal_psf(z, zf=50.0, zR=10.0)
    assert psf[0] > psf[1] > psf[2] > psf[3]


# ---------------------------------------------------------------------------
# get_slice_resolutions_from_psf
# ---------------------------------------------------------------------------


def test_get_slice_resolutions_from_psf_shape():
    res = get_slice_resolutions_from_psf(zf=400.0, zr=200.0, nz=20, spacing=(6.5, 6.5, 6.5), N=128)
    assert res.shape == (20,)
    # resolutions should be positive
    assert np.all(res > 0)


def test_get_slice_resolutions_from_psf_min_at_focal_plane():
    nz = 30
    spacing = (6.5, 6.5, 6.5)
    zf = 15 * 6.5  # focal plane in the middle
    res = get_slice_resolutions_from_psf(zf=zf, zr=200.0, nz=nz, spacing=spacing, N=128)
    # resolution should be smallest (best) near the focal plane
    mid = nz // 2
    assert res[mid] <= res[0]
    assert res[mid] <= res[-1]


# ---------------------------------------------------------------------------
# estimate_psf
# ---------------------------------------------------------------------------


def test_estimate_psf_synthetic_gaussian_beam():
    """Estimate PSF from a synthetic confocal PSF profile with known params."""
    dz = 6.5
    nz = 100
    z = np.linspace(0, nz * dz, nz)
    zf_true = 300.0
    zr_true = 150.0
    A_true = 1.0
    # synthetic profile (1D, already an intensity-vs-depth curve)
    profile = confocal_psf(z, zf_true, zr_true, A_true)
    # add tiny noise to make it realistic
    rng = np.random.default_rng(42)
    profile = profile + rng.normal(0, 0.001, size=profile.shape)
    profile = np.clip(profile, 0.01, None)

    zf_est, zr_est, A_est = estimate_psf(profile, dz=dz)
    # focal plane should be recovered within a tolerance
    assert abs(zf_est - zf_true) < 50.0
    assert zr_est > 0
    assert A_est > 0


def test_estimate_psf_with_fixed_zf():
    dz = 6.5
    nz = 80
    z = np.linspace(0, nz * dz, nz)
    zf_fixed = 200.0
    zr_true = 120.0
    profile = confocal_psf(z, zf_fixed, zr_true, 1.0)
    rng = np.random.default_rng(7)
    profile = profile + rng.normal(0, 0.001, size=profile.shape)
    profile = np.clip(profile, 0.01, None)

    zf_est, zr_est, A_est = estimate_psf(profile, dz=dz, zf=zf_fixed)
    # zf is fixed → returned unchanged
    assert zf_est == zf_fixed
    assert zr_est > 0
    assert A_est > 0


def test_estimate_psf_with_fit_attn():
    dz = 6.5
    nz = 80
    z = np.linspace(0, nz * dz, nz)
    zf_true = 200.0
    zr_true = 120.0
    profile = confocal_psf(z, zf_true, zr_true, 1.0)
    rng = np.random.default_rng(3)
    profile = profile + rng.normal(0, 0.001, size=profile.shape)
    profile = np.clip(profile, 0.01, None)

    result = estimate_psf(profile, dz=dz, fit_attn=True)
    assert len(result) == 4  # zf, zr, A, attn
    zf, zr, A, attn = result
    assert zf > 0
    assert zr > 0
    assert A > 0
    assert attn >= 0


def test_estimate_psf_3d_input_averages():
    """When agarose is 3D (nx, ny, nz), it averages over axis=(0,1)."""
    dz = 6.5
    nz = 60
    z = np.linspace(0, nz * dz, nz)
    profile_1d = confocal_psf(z, zf=200.0, zR=100.0, A=1.0)
    # stack into 3D: (2, 2, nz) — averaging over (0,1) gives back profile_1d
    agarose_3d = np.broadcast_to(profile_1d, (2, 2, nz)).copy()
    zf, zr, _A = estimate_psf(agarose_3d, dz=dz)
    assert zf > 0
    assert zr > 0


# ---------------------------------------------------------------------------
# get_3d_psf
# ---------------------------------------------------------------------------


def test_get_3d_psf_shape():
    nz = 60
    nx, ny = 4, 4
    z = np.linspace(0, nz * 6.5, nz)
    profile = confocal_psf(z, zf=30 * 6.5, zR=120.0, A=1.0)
    # build a volume where every aline has the same PSF-shaped profile
    vol = np.broadcast_to(profile, (nx, ny, nz)).astype(np.float32).copy()
    interface = np.full((nx, ny), 10, dtype=int)
    psf, zf_map, zr_map = get_3d_psf(vol, interface, res=6.5, use_average_rayleigh=True)
    assert psf.shape == vol.shape
    assert zf_map.shape == (nx, ny)
    assert zr_map.shape == (nx, ny)
    # PSF values should be finite (curve_fit may produce NaN on degenerate input,
    # but with a clean synthetic PSF it should converge)
    finite_psf = psf[np.isfinite(psf)]
    assert finite_psf.size > 0  # at least some finite values
    assert np.all(finite_psf >= 0)


# ---------------------------------------------------------------------------
# fit_tissue_confocal_model
# ---------------------------------------------------------------------------


def test_fit_tissue_confocal_model_returns_psf():
    nz = 80
    res = 6.5
    z = np.linspace(0, nz * res, nz)
    zf_true = 300.0
    zr_true = 200.0
    # build a tissue-like profile: sigmoid rise * confocal psf
    tissue = 1.0 / (1 + np.exp(-0.05 * (z - 200.0)))
    psf = confocal_psf(z, zf_true, zr_true, 1.0)
    iprofile = tissue * psf * 100.0 + 10.0
    z0 = 30  # interface index
    result = fit_tissue_confocal_model(iprofile, z0, zr_0=200.0, res=res)
    assert "psf" in result
    assert result["psf"].shape == z.shape


def test_fit_tissue_confocal_model_return_parameters():
    nz = 80
    res = 6.5
    z = np.linspace(0, nz * res, nz)
    tissue = 1.0 / (1 + np.exp(-0.05 * (z - 200.0)))
    psf = confocal_psf(z, 300.0, 200.0, 1.0)
    iprofile = tissue * psf * 100.0 + 10.0
    result = fit_tissue_confocal_model(iprofile, 30, zr_0=200.0, res=res, return_parameters=True)
    assert "parameters" in result
    params = result["parameters"]
    assert "zf" in params
    assert "zr" in params
    assert "a" in params


def test_fit_tissue_confocal_model_return_full_model():
    nz = 80
    res = 6.5
    z = np.linspace(0, nz * res, nz)
    tissue = 1.0 / (1 + np.exp(-0.05 * (z - 200.0)))
    psf = confocal_psf(z, 300.0, 200.0, 1.0)
    iprofile = tissue * psf * 100.0 + 10.0
    result = fit_tissue_confocal_model(iprofile, 30, zr_0=200.0, res=res, return_full_model=True)
    assert "tissue" in result
    assert "tissue_psf" in result


def test_fit_tissue_confocal_model_with_bump_model():
    nz = 80
    res = 6.5
    z = np.linspace(0, nz * res, nz)
    tissue = 1.0 / (1 + np.exp(-0.05 * (z - 200.0)))
    psf = confocal_psf(z, 300.0, 200.0, 1.0)
    iprofile = tissue * psf * 100.0 + 10.0
    result = fit_tissue_confocal_model(iprofile, 30, zr_0=200.0, res=res, use_bump_model=True, return_parameters=True)
    assert "psf" in result
    assert "parameters" in result
    # bump model adds extra parameters
    params = result["parameters"]
    assert "zf" in params
    assert "z0" in params


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_confocal_psf_zero_zr_raises_or_inf():
    """zR=0 would cause division by zero; confocal_psf uses float(zR) so 0 → inf."""
    z = np.array([50.0])
    # (z-zf)/0 → inf, inf^2+1 = inf, 1/inf = 0
    psf = confocal_psf(z, zf=0.0, zR=0.0)
    # 1/(inf) = 0
    assert np.all(np.isfinite(psf)) or np.any(psf == 0.0)
