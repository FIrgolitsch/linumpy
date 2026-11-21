"""Tests for linumpy/io/chunk_writer.py"""

from pathlib import Path

import numpy as np
import pytest
import zarr

from linumpy.io.chunk_writer import write_volume_chunks


def _files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _volume(dtype: str = "float32") -> np.ndarray:
    rng = np.random.default_rng(13)
    volume = (rng.random((21, 30, 37)) * 100).astype(dtype)
    volume[:, :9, :] = 0  # whole chunks of fill value
    volume[8:16] = 0
    return volume


@pytest.mark.parametrize("dtype", ["float32", "uint16", "uint8"])
@pytest.mark.parametrize("workers", [1, 4])
def test_write_volume_chunks_writes_the_files_zarr_writes(tmp_path, dtype, workers):
    volume = _volume(dtype)
    by_zarr = zarr.create_array(tmp_path / "zarr.zarr", shape=volume.shape, chunks=(8, 8, 8), dtype=dtype)
    by_zarr[:] = volume
    threaded = zarr.create_array(tmp_path / "threaded.zarr", shape=volume.shape, chunks=(8, 8, 8), dtype=dtype)
    assert write_volume_chunks(threaded, volume, workers=workers)

    assert _files(tmp_path / "threaded.zarr") == _files(tmp_path / "zarr.zarr")
    assert np.array_equal(zarr.open_array(tmp_path / "threaded.zarr", mode="r")[:], volume)


def test_write_volume_chunks_removes_a_chunk_that_became_empty(tmp_path):
    volume = _volume()
    array = zarr.create_array(tmp_path / "a.zarr", shape=volume.shape, chunks=(8, 8, 8), dtype="float32")
    array[:] = np.ones_like(volume)
    assert write_volume_chunks(array, volume)
    reference = zarr.create_array(tmp_path / "b.zarr", shape=volume.shape, chunks=(8, 8, 8), dtype="float32")
    reference[:] = volume
    assert _files(tmp_path / "a.zarr") == _files(tmp_path / "b.zarr")


def test_write_volume_chunks_declines_what_it_does_not_handle(tmp_path):
    volume = _volume()
    array = zarr.create_array(tmp_path / "a.zarr", shape=volume.shape, chunks=(8, 8, 8), dtype="float32")
    assert not write_volume_chunks(array, volume[:-1])
    assert not write_volume_chunks(array, volume.astype("float64"))
    sharded = zarr.create_array(
        tmp_path / "s.zarr", shape=volume.shape, chunks=(8, 8, 8), shards=(16, 16, 16), dtype="float32"
    )
    assert not write_volume_chunks(sharded, volume)
    uncompressed = zarr.create_array(
        tmp_path / "u.zarr", shape=volume.shape, chunks=(8, 8, 8), dtype="float32", compressors=None
    )
    assert not write_volume_chunks(uncompressed, volume)
    assert not list((tmp_path / "a.zarr").rglob("c/*"))


def test_writer_takes_the_threaded_path_for_a_whole_volume(tmp_path):
    from linumpy.io.zarr import AnalysisOmeZarrWriter

    volume = _volume()
    writer = AnalysisOmeZarrWriter(tmp_path / "w.ome.zarr", volume.shape, chunk_shape=(8, 8, 8), dtype=np.float32)
    writer[:] = volume
    reference = zarr.create_array(tmp_path / "ref.zarr", shape=volume.shape, chunks=(8, 8, 8), dtype="float32")
    reference[:] = volume
    written = _files(tmp_path / "w.ome.zarr" / "0")
    expected = _files(tmp_path / "ref.zarr")
    assert {k for k in written if k != "zarr.json"} == {k for k in expected if k != "zarr.json"}
    assert all(written[k] == expected[k] for k in written if k != "zarr.json")
    writer[2:5] = volume[2:5] * 2  # a partial write still goes through zarr
    assert np.array_equal(writer[2:5], volume[2:5] * 2)
