"""Tests for the pooled beam profile in linumpy/psf/profile.py"""

import numpy as np
import pytest

from linumpy.psf.profile import PooledBeamProfile, SlabBeamProfile, pool_beam_profiles, slab_beam_profile


def beam(depth: np.ndarray) -> np.ndarray:
    """Beam profile below the cut face used by the synthetic slabs."""
    return 1.0 + 4.0 * np.exp(-np.maximum(depth, 0) / 8.0)


def synthetic_slab(face: int, n_planes: int = 48, size: int = 96, seed: int = 0) -> np.ndarray:
    """Return a slab ``(Z, Y, X)``: water above the face, agarose around a block of tissue below it."""
    rng = np.random.default_rng(seed)
    depth = np.arange(n_planes) - face
    below = depth >= 0
    agarose = np.where(below, 2.0 * beam(depth), 0.5)
    tissue = np.where(below, 30.0 * beam(depth) * np.exp(-0.04 * np.maximum(depth, 0)), 0.5)
    vol = np.repeat(agarose[:, None, None], size, axis=1).repeat(size, axis=2)
    quarter = size // 4
    vol[:, quarter : 3 * quarter, quarter : 3 * quarter] = tissue[:, None, None]
    return (vol * (1.0 + 0.01 * rng.standard_normal(vol.shape))).astype(np.float32)


def profile_from_shape(shape: np.ndarray, face: int, n_planes: int, scale: float = 1.0) -> SlabBeamProfile:
    """Return the profile of a slab whose agarose follows *shape* (indexed by depth + 30) times *scale*."""
    depth = np.arange(n_planes) - face
    return SlabBeamProfile(
        profile=[float(scale * shape[d + 30]) for d in depth],
        tissue_face=face,
        agarose_face=face,
        n_columns=5000,
        coverage=0.3,
    )


# ---------------------------------------------------------------------------
# slab_beam_profile
# ---------------------------------------------------------------------------


def test_slab_profile_finds_the_tissue_face_and_the_agarose():
    profile = slab_beam_profile(synthetic_slab(face=12))

    assert abs(profile.tissue_face - 12) <= 2
    assert abs(profile.agarose_face - 12) <= 2
    assert len(profile.profile) == 48
    # Agarose is everything around the central block of tissue.
    assert profile.coverage == pytest.approx(0.75, abs=0.05)
    assert profile.is_usable


def test_slab_profile_follows_the_agarose_not_the_tissue():
    profile = np.asarray(slab_beam_profile(synthetic_slab(face=10)).profile)

    expected = 2.0 * beam(np.arange(48) - 10)
    np.testing.assert_allclose(profile[12:40], expected[12:40], rtol=0.02)


def test_slab_profile_of_an_empty_slab_is_not_usable():
    profile = slab_beam_profile(np.zeros((20, 32, 32), dtype=np.float32))

    assert profile.profile == []
    assert not profile.is_usable


def test_slab_profile_round_trips_through_json(tmp_path):
    profile = slab_beam_profile(synthetic_slab(face=8))
    profile.save(tmp_path / "slab.json")

    assert SlabBeamProfile.load(tmp_path / "slab.json") == profile


# ---------------------------------------------------------------------------
# pool_beam_profiles
# ---------------------------------------------------------------------------


def test_pooled_profile_recovers_the_beam_from_slabs_with_different_faces():
    slabs = [slab_beam_profile(synthetic_slab(face=face, seed=face)) for face in (6, 9, 12, 15, 18)]
    pooled = pool_beam_profiles(slabs)

    depth = np.asarray(pooled.depth)
    divisor = np.asarray(pooled.divisor)
    below = (depth >= 3) & (depth <= 25)
    ratio = divisor[below] / beam(depth[below])
    # The divisor is the beam up to one constant factor.
    np.testing.assert_allclose(ratio, ratio[0], rtol=0.05)
    assert pooled.n_used == 5
    assert min(pooled.divisor) == pytest.approx(1.0)


def test_pooled_profile_does_not_step_where_slabs_run_out_of_planes():
    shape = 1.0 + 6.0 * np.exp(-np.abs(np.arange(-30, 80) - 2) / 10.0)
    # Deep faces end 20 planes sooner, and those slabs are three times brighter.
    slabs = [profile_from_shape(shape, face=5, n_planes=50) for _ in range(4)]
    slabs += [profile_from_shape(shape, face=25, n_planes=50, scale=3.0) for _ in range(4)]
    pooled = pool_beam_profiles(slabs)

    depth = np.asarray(pooled.depth)
    ratio = np.asarray(pooled.divisor) / shape[depth + 30]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9)


def test_pooled_profile_is_held_where_too_few_slabs_reach():
    shape = 1.0 + 6.0 * np.exp(-np.abs(np.arange(-30, 80) - 2) / 10.0)
    slabs = [profile_from_shape(shape, face=5, n_planes=30) for _ in range(5)]
    slabs.append(profile_from_shape(shape, face=5, n_planes=50))
    pooled = pool_beam_profiles(slabs, min_slabs=3)

    depth = np.asarray(pooled.depth)
    divisor = np.asarray(pooled.divisor)
    # One slab alone reaches below depth 24: the profile stays at its value there.
    np.testing.assert_allclose(divisor[depth >= 24], divisor[depth == 24][0])


def test_pooled_profile_leaves_out_slabs_with_too_little_agarose():
    shape = 1.0 + 6.0 * np.exp(-np.abs(np.arange(-30, 80) - 2) / 10.0)
    slabs = [profile_from_shape(shape, face=8, n_planes=50) for _ in range(4)]
    sparse = profile_from_shape(np.full(110, 2.0), face=8, n_planes=50)
    sparse.n_columns = 20
    slabs.append(sparse)

    pooled = pool_beam_profiles(slabs, names=[f"z{i:02d}" for i in range(5)])

    assert pooled.n_used == 4
    assert pooled.excluded == ["z04: too little agarose"]
    depth = np.asarray(pooled.depth)
    ratio = np.asarray(pooled.divisor) / shape[depth + 30]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9)


def test_pooling_fails_when_too_few_slabs_are_usable():
    shape = np.linspace(1.0, 2.0, 110)
    usable = profile_from_shape(shape, face=5, n_planes=40)
    empty = SlabBeamProfile(profile=[], tissue_face=5, agarose_face=0, n_columns=0, coverage=0.0)

    with pytest.raises(ValueError, match="Only 1 of 3 slabs"):
        pool_beam_profiles([usable, empty, empty])


# ---------------------------------------------------------------------------
# PooledBeamProfile
# ---------------------------------------------------------------------------


def test_divisor_for_slab_follows_the_tissue_face():
    pooled = PooledBeamProfile(depth=[-2, -1, 0, 1, 2, 3], divisor=[1.0, 1.0, 4.0, 3.0, 2.0, 1.5], n_slabs=[5] * 6, n_used=5)

    np.testing.assert_allclose(pooled.divisor_for_slab(2, 6), [1.0, 1.0, 4.0, 3.0, 2.0, 1.5])
    np.testing.assert_allclose(pooled.divisor_for_slab(4, 6), [1.0, 1.0, 1.0, 1.0, 4.0, 3.0])


def test_divisor_for_slab_holds_the_ends_of_the_pooled_range():
    pooled = PooledBeamProfile(depth=[0, 1, 2], divisor=[4.0, 3.0, 2.0], n_slabs=[5] * 3, n_used=5)

    np.testing.assert_allclose(pooled.divisor_for_slab(2, 7), [4.0, 4.0, 4.0, 3.0, 2.0, 2.0, 2.0])


def test_pooled_profile_round_trips_through_json(tmp_path):
    pooled = PooledBeamProfile(
        depth=[0, 1], divisor=[2.0, 1.0], n_slabs=[3, 3], n_used=3, n_excluded=1, excluded=["z07: too little agarose"]
    )
    pooled.save(tmp_path / "pooled.json")

    assert PooledBeamProfile.load(tmp_path / "pooled.json") == pooled
