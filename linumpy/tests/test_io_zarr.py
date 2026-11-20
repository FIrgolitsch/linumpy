"""Tests for linumpy/io/zarr.py OME-Zarr metadata round-trip."""

from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from linumpy.io.zarr import read_omezarr, save_omezarr

VOXEL_SIZE_MM = (0.01, 0.01, 0.01)
TILE_FOV_MM = 0.875


def _small_volume() -> da.Array:
    return da.from_array(np.zeros((2, 10, 10), dtype=np.float32))


def test_save_omezarr_preserves_voxel_size_n_levels_0(tmp_path: Path) -> None:
    out = tmp_path / "volume.ome.zarr"
    save_omezarr(
        _small_volume(),
        out,
        voxel_size=VOXEL_SIZE_MM,
        chunks=(2, 10, 10),
        n_levels=0,
    )
    _, scale = read_omezarr(out, 0)
    assert scale == list(VOXEL_SIZE_MM)


def test_save_omezarr_preserves_voxel_size_pyramid(tmp_path: Path) -> None:
    out = tmp_path / "pyramid.ome.zarr"
    save_omezarr(
        _small_volume(),
        out,
        voxel_size=VOXEL_SIZE_MM,
        chunks=(2, 10, 10),
        n_levels=1,
    )
    _, scale_l0 = read_omezarr(out, 0)
    _, scale_l1 = read_omezarr(out, 1)
    assert scale_l0 == list(VOXEL_SIZE_MM)
    assert scale_l1 == [v * 2 for v in VOXEL_SIZE_MM]


def test_tile_size_px_from_resolution(tmp_path: Path) -> None:
    out = tmp_path / "tile_sizing.ome.zarr"
    save_omezarr(
        _small_volume(),
        out,
        voxel_size=VOXEL_SIZE_MM,
        chunks=(2, 10, 10),
        n_levels=0,
    )
    _, res = read_omezarr(out, 0)
    tile_px = round(TILE_FOV_MM / float(res[1]))
    assert tile_px == 88


def test_writer_levels_carry_their_own_voxel_size(tmp_path: Path) -> None:
    """Levels added by hand can have any shape and voxel size, and read back as written."""
    from linumpy.io.zarr import OmeZarrWriter, read_omezarr_resolutions

    path = tmp_path / "levels.ome.zarr"
    writer = OmeZarrWriter(path, shape=(8, 16, 16), chunk_shape=(8, 8, 8), shards=(8, 16, 16))
    level = writer.add_level(shape=(3, 6, 6), chunk_shape=(3, 3, 3), shards=(3, 6, 6))
    base = np.arange(8 * 16 * 16, dtype=np.float32).reshape(8, 16, 16)
    coarse = np.arange(3 * 6 * 6, dtype=np.float32).reshape(3, 6, 6)
    writer[:] = base
    level[:] = coarse
    writer.finalize_levels([[0.002, 0.001, 0.001], [0.005, 0.005, 0.005]])

    assert read_omezarr_resolutions(path) == [[0.002, 0.001, 0.001], [0.005, 0.005, 0.005]]
    vol0, res0 = read_omezarr(path, level=0)
    vol1, res1 = read_omezarr(path, level=1)
    last, res_last = read_omezarr(path, level=-1)
    np.testing.assert_array_equal(vol0[:], base)
    np.testing.assert_array_equal(vol1[:], coarse)
    np.testing.assert_array_equal(last[:], coarse)
    assert list(res0) == [0.002, 0.001, 0.001]
    assert list(res1) == [0.005, 0.005, 0.005]
    assert list(res_last) == [0.005, 0.005, 0.005]
    assert vol1.chunks == (3, 3, 3)


def test_writer_refuses_to_mix_added_levels_with_the_factor_2_pyramid(tmp_path: Path) -> None:
    """finalize would write its own level 1 over the one added by hand."""
    from linumpy.io.zarr import OmeZarrWriter

    writer = OmeZarrWriter(tmp_path / "mixed.ome.zarr", shape=(8, 16, 16), chunk_shape=(8, 8, 8))
    writer.add_level(shape=(3, 6, 6), chunk_shape=(3, 3, 3))
    with pytest.raises(ValueError, match="finalize_levels"):
        writer.finalize([0.002, 0.001, 0.001], 0)
    with pytest.raises(ValueError, match="voxel sizes"):
        writer.finalize_levels([[0.002, 0.001, 0.001]])


def test_analysis_pyramid_keeps_level_0_when_it_is_the_first_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A 10 um volume finalised with a 10 um first level is not resampled onto itself.

    The resize to the same shape gave back the same voxels, after reading and
    writing the whole volume once more. Level 0 has to hold what was written,
    and only the coarser levels are computed.
    """
    from linumpy.io.zarr import AnalysisOmeZarrWriter

    rng = np.random.default_rng(0)
    vol = rng.gamma(2.0, 1.0, size=(60, 150, 170)).astype(np.float32)
    path = tmp_path / "vol.ome.zarr"
    writer = AnalysisOmeZarrWriter(path, vol.shape, chunk_shape=(32, 64, 64), dtype=np.float32)
    writer[:] = vol

    import linumpy.io.pyramid as pyramid

    resampled_to: list[tuple[int, ...]] = []
    original = pyramid.downsample_antialiased

    def spy(source: object, target_shapes: list, **kwargs: object) -> list[np.ndarray]:
        resampled_to.extend(tuple(int(v) for v in shape) for shape in target_shapes)
        return original(source, target_shapes, **kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch.setattr(pyramid, "downsample_antialiased", spy)
    writer.finalize([0.01, 0.01, 0.01], target_resolutions_um=[10, 25, 50, 100], make_isotropic=True, use_gpu=False)

    # Only the coarser levels are computed, in one call.
    assert resampled_to == [(24, 60, 68), (12, 30, 34), (6, 15, 17)]
    level0, res0 = read_omezarr(path, level=0)
    np.testing.assert_array_equal(level0[:], vol)
    assert level0.chunks == (32, 64, 64)
    assert list(res0) == [0.01, 0.01, 0.01]
    assert [read_omezarr(path, level=k)[0].shape for k in range(4)] == [
        (60, 150, 170),
        (24, 60, 68),
        (12, 30, 34),
        (6, 15, 17),
    ]


def test_analysis_pyramid_still_resamples_an_anisotropic_base(tmp_path: Path) -> None:
    """When the base is not at the first target resolution, level 0 is replaced as before."""
    from linumpy.io.zarr import AnalysisOmeZarrWriter

    rng = np.random.default_rng(1)
    vol = rng.random((40, 60, 60), dtype=np.float32)
    path = tmp_path / "aniso.ome.zarr"
    writer = AnalysisOmeZarrWriter(path, vol.shape, chunk_shape=(20, 30, 30), dtype=np.float32)
    writer[:] = vol
    writer.finalize([0.005, 0.01, 0.01], target_resolutions_um=[10, 25], make_isotropic=True)

    level0, res0 = read_omezarr(path, level=0)
    assert level0.shape == (20, 60, 60)
    assert list(res0) == pytest.approx([0.01, 0.01, 0.01])
