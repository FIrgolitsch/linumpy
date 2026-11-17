"""Tests for the surface-tilt fit and levelling in linumpy/geometry/crop.py"""

import numpy as np

from linumpy.geometry.crop import fit_surface_tilt, flatten_surface_tilt


def _slab(tilt_y: float = 0.0, tilt_x: float = 0.0, offset: float = 12.0, shape: tuple = (50, 160, 200)) -> np.ndarray:
    """Tissue in a round footprint under a tilted surface, agarose around it, water above."""
    nz, ny, nx = shape
    yy, xx = np.mgrid[:ny, :nx]
    surface = offset + tilt_y * yy + tilt_x * xx
    inside = ((yy - ny / 2) ** 2 / (0.4 * ny) ** 2 + (xx - nx / 2) ** 2 / (0.4 * nx) ** 2) < 1
    z = np.arange(nz)[:, None, None]
    depth = z - surface[None]
    signal = np.where(inside[None], 1.0, 0.15) * np.exp(-0.02 * np.clip(depth, 0, None))
    return np.where(depth >= 0, signal, 0.02).astype(np.float32)


def _surface_span(vol: np.ndarray) -> float:
    """Spread of the first bright plane over the tissue A-lines."""
    first = (vol > 0.5).argmax(axis=0)
    tissue = (vol > 0.5).any(axis=0)
    return float(np.percentile(first[tissue], 98) - np.percentile(first[tissue], 2))


def test_fit_recovers_the_tilt_of_the_cut_face():
    vol = _slab(tilt_y=0.02, tilt_x=-0.01)
    plane, info = fit_surface_tilt(vol, min_alines=500)
    assert plane is not None and plane.shape == vol.shape[1:]
    assert abs(info["tilt_y"] - 20.0) < 2.0
    assert abs(info["tilt_x"] + 10.0) < 2.0
    assert info["span"] > 3.0
    assert info["inliers"] > 0.9


def test_levelling_brings_the_surface_to_one_depth():
    vol = _slab(tilt_y=0.05, tilt_x=-0.03)
    assert _surface_span(vol) >= 5
    plane, info = fit_surface_tilt(vol, min_alines=500)
    assert plane is not None
    level = flatten_surface_tilt(vol, plane, info["reference"])
    assert level.shape == vol.shape and level.dtype == np.float32
    assert _surface_span(level) <= 1
    # The tissue keeps its level; only its depth moved.
    assert abs(float(np.median(level[level > 0.5])) - float(np.median(vol[vol > 0.5]))) < 0.02


def test_a_level_cut_face_has_no_span_to_correct():
    plane, info = fit_surface_tilt(_slab(), min_alines=500)
    assert plane is not None
    assert info["span"] < 1.0
    assert abs(info["tilt_y"]) < 1.0 and abs(info["tilt_x"]) < 1.0


def test_fit_declines_without_tissue_or_without_a_plane():
    empty = np.zeros((30, 40, 40), dtype=np.float32)
    assert fit_surface_tilt(empty)[0] is None
    assert fit_surface_tilt(_slab(), min_alines=10**7)[0] is None
    # A surface that jumps between two depths is not a plane.
    vol = _slab()
    yy, xx = np.mgrid[: vol.shape[1], : vol.shape[2]]
    checker = ((yy // 4 + xx // 4) % 2).astype(bool)
    stepped = vol.copy()
    stepped[:, checker] = np.roll(vol, 12, axis=0)[:, checker]
    plane, info = fit_surface_tilt(stepped, sigma_xy=0.5, min_alines=500)
    assert plane is None
    assert info["inliers"] < 0.75


def test_levelling_by_whole_planes_is_a_shift_along_z():
    rng = np.random.default_rng(0)
    vol = rng.random((20, 6, 7)).astype(np.float32)
    plane = np.full((6, 7), 5.0, dtype=np.float32)
    plane[:, 4:] = 8.0
    out = flatten_surface_tilt(vol, plane, reference=5.0)
    np.testing.assert_allclose(out[:, :, :4], vol[:, :, :4])
    np.testing.assert_allclose(out[:17, :, 4:], vol[3:, :, 4:])
    assert np.all(out[17:, :, 4:] == 0.0)
    # Half a plane is the mean of two neighbours.
    half = flatten_surface_tilt(vol, np.full((6, 7), 5.5, dtype=np.float32), reference=5.0)
    np.testing.assert_allclose(half[:19], 0.5 * (vol[:19] + vol[1:]), rtol=1e-6)
