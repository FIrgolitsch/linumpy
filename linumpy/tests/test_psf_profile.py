"""Tests for the pooled beam profile in linumpy/psf/profile.py"""

import numpy as np
import pytest

from linumpy.psf.profile import PooledBeamProfile, SlabBeamProfile, pool_beam_profiles, slab_beam_profile


def beam(plane: np.ndarray) -> np.ndarray:
    """Beam profile of the synthetic slabs: a focus at plane 26 of the imaging window."""
    return 1.0 + 4.0 * np.exp(-(((np.asarray(plane, dtype=float) - 26.0) / 10.0) ** 2))


def synthetic_slab(face: int, n_planes: int = 48, size: int = 96, seed: int = 0) -> np.ndarray:
    """Return a slab ``(Z, Y, X)``: water above the face, agarose around a block of tissue below it."""
    rng = np.random.default_rng(seed)
    depth = np.arange(n_planes) - face
    below = depth >= 0
    plane = np.arange(n_planes)
    agarose = np.where(below, 2.0 * beam(plane), 0.5)
    tissue = np.where(below, 30.0 * beam(plane) * np.exp(-0.04 * np.maximum(depth, 0)), 0.5)
    vol = np.repeat(agarose[:, None, None], size, axis=1).repeat(size, axis=2)
    quarter = size // 4
    vol[:, quarter : 3 * quarter, quarter : 3 * quarter] = tissue[:, None, None]
    return (vol * (1.0 + 0.01 * rng.standard_normal(vol.shape))).astype(np.float32)


def profile_from_shape(shape: np.ndarray, face: int, n_planes: int, scale: float = 1.0) -> SlabBeamProfile:
    """Return the profile of a slab: *shape* by window plane times *scale* below the face, water above."""
    return SlabBeamProfile(
        profile=[float(scale * shape[z]) if z >= face else 0.05 for z in range(n_planes)],
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

    expected = 2.0 * beam(np.arange(48))
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

    plane = np.asarray(pooled.depth)
    divisor = np.asarray(pooled.divisor)
    # From where three slabs have agarose, down to the end of the window.
    covered = (plane >= 16) & (plane <= 44)
    ratio = divisor[covered] / beam(plane[covered])
    # The divisor is the beam up to one constant factor, at the same plane for every face.
    np.testing.assert_allclose(ratio, ratio[0], rtol=0.05)
    assert int(plane[np.argmax(divisor)]) in (25, 26, 27)
    assert pooled.n_used == 5
    assert min(pooled.divisor) == pytest.approx(1.0)


def test_pooled_profile_does_not_step_where_slabs_with_a_deep_face_come_in():
    shape = 1.0 + 6.0 * np.exp(-(((np.arange(50) - 30) / 12.0) ** 2))
    # Half the slabs only start at plane 25, and those are three times brighter.
    slabs = [profile_from_shape(shape, face=5, n_planes=50) for _ in range(4)]
    slabs += [profile_from_shape(shape, face=25, n_planes=50, scale=3.0) for _ in range(4)]
    pooled = pool_beam_profiles(slabs)

    plane = np.asarray(pooled.depth)
    covered = plane >= 7
    ratio = np.asarray(pooled.divisor)[covered] / shape[plane[covered]]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9)


def test_pooled_profile_is_held_where_too_few_slabs_reach():
    shape = 1.0 + 6.0 * np.exp(-(((np.arange(50) - 30) / 12.0) ** 2))
    slabs = [profile_from_shape(shape, face=5, n_planes=30) for _ in range(5)]
    slabs.append(profile_from_shape(shape, face=5, n_planes=50))
    pooled = pool_beam_profiles(slabs, min_slabs=3)

    plane = np.asarray(pooled.depth)
    divisor = np.asarray(pooled.divisor)
    # One slab alone reaches below plane 29: the profile stays at its value there.
    np.testing.assert_allclose(divisor[plane >= 29], divisor[plane == 29][0])
    # No slab has agarose above its face: the profile is held there too.
    np.testing.assert_allclose(divisor[plane <= 7], divisor[plane == 7][0])


def test_pooled_profile_leaves_out_slabs_with_too_little_agarose():
    shape = 1.0 + 6.0 * np.exp(-(((np.arange(50) - 30) / 12.0) ** 2))
    slabs = [profile_from_shape(shape, face=8, n_planes=50) for _ in range(4)]
    sparse = profile_from_shape(np.full(50, 2.0), face=8, n_planes=50)
    sparse.n_columns = 20
    slabs.append(sparse)

    pooled = pool_beam_profiles(slabs, names=[f"z{i:02d}" for i in range(5)])

    assert pooled.n_used == 4
    assert pooled.excluded == ["z04: too little agarose"]
    plane = np.asarray(pooled.depth)
    covered = plane >= 10
    ratio = np.asarray(pooled.divisor)[covered] / shape[plane[covered]]
    np.testing.assert_allclose(ratio, ratio[0], rtol=1e-9)


def test_pooled_profile_leaves_out_a_slab_whose_face_is_at_the_end_of_the_window():
    shape = 1.0 + 6.0 * np.exp(-(((np.arange(50) - 30) / 12.0) ** 2))
    slabs = [profile_from_shape(shape, face=8, n_planes=50) for _ in range(4)]
    slabs.append(profile_from_shape(shape, face=47, n_planes=50))

    pooled = pool_beam_profiles(slabs, names=[f"z{i:02d}" for i in range(5)])

    assert pooled.n_used == 4
    assert pooled.excluded == ["z04: tissue face at plane 47, at the end of the window"]


def test_pooling_fails_when_too_few_slabs_are_usable():
    shape = np.linspace(1.0, 2.0, 40)
    usable = profile_from_shape(shape, face=5, n_planes=40)
    empty = SlabBeamProfile(profile=[], tissue_face=5, agarose_face=0, n_columns=0, coverage=0.0)

    with pytest.raises(ValueError, match="Only 1 of 3 slabs"):
        pool_beam_profiles([usable, empty, empty])


# ---------------------------------------------------------------------------
# PooledBeamProfile
# ---------------------------------------------------------------------------


def test_divisor_is_the_same_at_a_plane_whatever_the_slab():
    pooled = PooledBeamProfile(depth=[0, 1, 2, 3, 4, 5], divisor=[1.0, 1.0, 4.0, 3.0, 2.0, 1.5], n_slabs=[5] * 6, n_used=5)

    np.testing.assert_allclose(pooled.divisor_for_planes(6), [1.0, 1.0, 4.0, 3.0, 2.0, 1.5])
    np.testing.assert_allclose(pooled.divisor_for_planes(4), [1.0, 1.0, 4.0, 3.0])


def test_divisor_holds_the_end_of_the_pooled_range():
    pooled = PooledBeamProfile(depth=[0, 1, 2], divisor=[4.0, 3.0, 2.0], n_slabs=[5] * 3, n_used=5)

    np.testing.assert_allclose(pooled.divisor_for_planes(5), [4.0, 3.0, 2.0, 2.0, 2.0])


def test_pooled_profile_round_trips_through_json(tmp_path):
    pooled = PooledBeamProfile(
        depth=[0, 1], divisor=[2.0, 1.0], n_slabs=[3, 3], n_used=3, n_excluded=1, excluded=["z07: too little agarose"]
    )
    pooled.save(tmp_path / "pooled.json")

    assert PooledBeamProfile.load(tmp_path / "pooled.json") == pooled
