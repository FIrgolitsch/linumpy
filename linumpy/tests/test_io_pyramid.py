"""Tests for linumpy/io/pyramid.py"""

import numpy as np
import pytest
from skimage.transform import resize

from linumpy.io.pyramid import downsample_antialiased, resampling_matrix


@pytest.mark.parametrize(("n_in", "n_out"), [(40, 16), (37, 4), (30, 30), (53, 21)])
def test_resampling_matrix_matches_resize_along_one_axis(n_in, n_out):
    line = np.random.default_rng(8).random(n_in).astype(np.float32)
    expected = resize(line, (n_out,), order=1, anti_aliasing=True, preserve_range=True)
    matrix = resampling_matrix(n_in, n_out)
    assert matrix.shape == (n_out, n_in)
    assert np.allclose(matrix.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(matrix @ line, expected, atol=1e-5)


@pytest.mark.parametrize("block_bytes", [1, 10 * 44 * 52 * 4, 2 * 1024**3])
def test_downsample_antialiased_matches_resize_of_the_whole_volume(block_bytes):
    volume = np.random.default_rng(9).random((60, 44, 52), dtype=np.float32)
    shapes = [(24, 17, 20), (12, 8, 10), (6, 4, 5)]
    levels = downsample_antialiased(volume, shapes, use_gpu=False, block_bytes=block_bytes)
    for level, shape in zip(levels, shapes, strict=True):
        expected = resize(volume, shape, order=1, anti_aliasing=True, preserve_range=True)
        assert level.shape == shape
        assert level.dtype == np.float32
        assert np.allclose(level, expected, atol=2e-5)


def test_resampling_matrix_with_the_voxel_size_ratio_keeps_the_scale():
    # 41 samples at factor 2: twenty outputs cover the first forty samples.
    line = np.random.default_rng(10).random(41).astype(np.float32)
    expected = resize(line[:40], (20,), order=1, anti_aliasing=True, preserve_range=True)
    result = resampling_matrix(41, 20, factor=2.0) @ line
    # The last rows see sample 40 where the cropped line is mirrored.
    assert np.allclose(result[:-2], expected[:-2], atol=1e-5)


def test_downsample_antialiased_takes_the_factor_of_each_level():
    volume = np.random.default_rng(11).random((41, 30, 33), dtype=np.float32)
    (level,) = downsample_antialiased(volume, [(20, 15, 16)], use_gpu=False, factors=[(2.0, 2.0, 2.0)])
    expected = resize(volume[:40, :30, :32], (20, 15, 16), order=1, anti_aliasing=True, preserve_range=True)
    assert np.allclose(level[:-2, :, :-2], expected[:-2, :, :-2], atol=2e-5)


def test_writer_levels_have_the_scale_of_their_voxel_size(tmp_path):
    import zarr

    from linumpy.io.zarr import AnalysisOmeZarrWriter

    volume = np.random.default_rng(12).random((63, 51, 54), dtype=np.float32)
    # 16-voxel chunks do not divide into whole 25 um voxels: the chunk-wise
    # resize rounded the scale to the chunk here.
    writer = AnalysisOmeZarrWriter(tmp_path / "volume.ome.zarr", volume.shape, chunk_shape=(16, 16, 16), dtype=np.float32)
    writer[:] = volume
    writer.finalize([0.01, 0.01, 0.01], target_resolutions_um=[10, 25, 50], use_gpu=False)

    shapes = [(25, 20, 21), (12, 10, 10)]
    expected = downsample_antialiased(volume, shapes, use_gpu=False, factors=[(2.5,) * 3, (5.0,) * 3])
    for path, shape, level in zip(("1", "2"), shapes, expected, strict=True):
        stored = zarr.open_array(tmp_path / "volume.ome.zarr" / path, mode="r")
        assert stored.shape == shape
        assert np.array_equal(np.asarray(stored[:]), level)
    assert np.array_equal(np.asarray(zarr.open_array(tmp_path / "volume.ome.zarr" / "0", mode="r")[:]), volume)
