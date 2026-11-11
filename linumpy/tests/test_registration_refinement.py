"""Tests for residual-bound policy in linumpy.registration.refinement."""

import numpy as np
import pytest

from linumpy.registration.refinement import apply_refinement_bounds, tissue_ncc


def test_scale_projects_runaway_onto_bound():
    tx, ty, rot, rejected = apply_refinement_bounds(60.0, 0.0, 0.0, max_translation_px=10.0, max_rotation_deg=2.0)
    assert not rejected
    assert abs((tx**2 + ty**2) ** 0.5 - 10.0) < 1e-9
    assert tx == 10.0
    assert rot == 0.0


def test_reject_returns_identity_on_runaway():
    tx, ty, rot, rejected = apply_refinement_bounds(
        60.0, 0.0, 0.5, max_translation_px=10.0, max_rotation_deg=2.0, bound_mode="reject"
    )
    assert rejected
    assert (tx, ty, rot) == (0.0, 0.0, 0.0)


def test_reject_returns_identity_on_rotation_bound():
    tx, ty, rot, rejected = apply_refinement_bounds(
        1.0, 0.0, 5.0, max_translation_px=10.0, max_rotation_deg=2.0, bound_mode="reject"
    )
    assert rejected
    assert (tx, ty, rot) == (0.0, 0.0, 0.0)


def test_reject_keeps_small_residual():
    tx, ty, rot, rejected = apply_refinement_bounds(
        2.0, -1.0, 0.3, max_translation_px=10.0, max_rotation_deg=2.0, bound_mode="reject"
    )
    assert not rejected
    assert (tx, ty, rot) == (2.0, -1.0, 0.3)


def test_tissue_ncc_identical_is_one():
    rng = np.random.default_rng(0)
    img = np.zeros((32, 32), dtype=np.float32)
    img[8:24, 8:24] = rng.random((16, 16)).astype(np.float32) + 0.2
    assert tissue_ncc(img, img) == pytest.approx(1.0)


def test_tissue_ncc_shifted_is_lower():
    rng = np.random.default_rng(0)
    img = np.zeros((32, 32), dtype=np.float32)
    img[8:24, 8:24] = rng.random((16, 16)).astype(np.float32) + 0.2
    shifted = np.roll(img, 8, axis=1)
    assert tissue_ncc(img, shifted) < 0.5


def _textured_blob(shape=(160, 160), seed=0):
    """Smooth random texture inside an off-centre ellipse, zero outside."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    mask = ((yy - 70) / 45.0) ** 2 + ((xx - 85) / 55.0) ** 2 < 1.0
    texture = gaussian_filter(rng.random(shape), 3.0)
    texture = (texture - texture.min()) / (texture.max() - texture.min())
    return ((0.2 + 0.8 * texture) * mask).astype(np.float32)


def test_coarse_translation_recovers_a_shift_beyond_the_residual_bound():
    from scipy.ndimage import shift as nd_shift

    from linumpy.registration.refinement import coarse_translation

    fixed = _textured_blob()
    moving = nd_shift(fixed, (-22.0, 31.0), order=1)
    tx, ty, ncc = coarse_translation(fixed, moving, max_translation_px=60)
    # Moving sits 22 px up and 31 px right; the point map that undoes it is
    # the opposite of the content shift (22 down, 31 left).
    assert tx == pytest.approx(31.0, abs=1.0)
    assert ty == pytest.approx(-22.0, abs=1.0)
    assert ncc > 0.9
    assert tissue_ncc(fixed, moving) < 0.5


def test_coarse_translation_returns_identity_when_already_aligned():
    from linumpy.registration.refinement import coarse_translation

    fixed = _textured_blob()
    tx, ty, ncc = coarse_translation(fixed, fixed.copy(), max_translation_px=60)
    assert (tx, ty) == (0.0, 0.0)
    assert ncc == pytest.approx(1.0)


def test_coarse_translation_ignores_shifts_past_the_limit():
    from scipy.ndimage import shift as nd_shift

    from linumpy.registration.refinement import coarse_translation

    fixed = _textured_blob()
    moving = nd_shift(fixed, (-22.0, 31.0), order=1)
    tx, ty, ncc = coarse_translation(fixed, moving, max_translation_px=10)
    assert (tx, ty) == (0.0, 0.0)
    assert ncc == pytest.approx(tissue_ncc(fixed, moving))
