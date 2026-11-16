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


def _two_sided_brightness(seed: int = 3, size: int = 240):
    """Same anatomy seen with opposite brightness ramps, as at a slab joint."""
    rng = np.random.default_rng(seed)
    from scipy.ndimage import gaussian_filter

    yy, xx = np.mgrid[:size, :size]
    tissue = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2) < (0.42 * size) ** 2
    detail = gaussian_filter(rng.normal(size=(size, size)), 4.0)
    anatomy = 1.0 + 0.5 * detail / detail.std()
    ramp = (xx / (size - 1)).astype(np.float32)
    fixed = np.where(tissue, anatomy * (0.3 + 1.4 * ramp), 0.0).astype(np.float32)
    moving = np.where(tissue, anatomy * (1.7 - 1.4 * ramp), 0.0).astype(np.float32)
    return np.clip(fixed, 0, None), np.clip(moving, 0, None)


def test_structure_ncc_ignores_opposite_brightness_ramps():
    from linumpy.registration.refinement import structure_ncc

    fixed, moving = _two_sided_brightness()
    # The plain score is dragged down by the ramps although the anatomy matches.
    assert tissue_ncc(fixed, moving) < 0.4
    assert structure_ncc(fixed, moving) > 0.8


def test_structure_ncc_drops_when_the_anatomy_is_shifted():
    from scipy.ndimage import shift as nd_shift

    from linumpy.registration.refinement import structure_ncc

    fixed, moving = _two_sided_brightness()
    shifted = nd_shift(moving, (0, 25), order=1, cval=0.0)
    assert structure_ncc(fixed, shifted) < 0.3


def test_structure_ncc_needs_common_tissue():
    from linumpy.registration.refinement import structure_ncc

    assert np.isnan(structure_ncc(np.zeros((64, 64), np.float32), np.ones((64, 64), np.float32)))


def test_coarse_translation_with_structure_score_recovers_the_shift_under_opposite_ramps():
    from scipy.ndimage import shift as nd_shift

    from linumpy.registration.refinement import coarse_translation, structure_ncc

    fixed, moving = _two_sided_brightness()
    moved = nd_shift(moving, (18, -30), order=1, cval=0.0)
    tx, ty, score = coarse_translation(fixed, moved, max_translation_px=80, metric=structure_ncc)
    # SimpleITK convention: positive tx moves content left.
    assert (tx, ty) == pytest.approx((-30.0, 18.0), abs=1.5)
    assert score > 0.7
