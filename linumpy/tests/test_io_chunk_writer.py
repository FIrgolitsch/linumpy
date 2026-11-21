"""Tests for linumpy/io/chunk_writer.py"""

from pathlib import Path

import numpy as np
import pytest
import zarr

from linumpy.io.chunk_writer import write_shard_chunks, write_volume_chunks


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


# ---------------------------------------------------------------------------
# write_shard_chunks
# ---------------------------------------------------------------------------

_CHUNK = (5, 6, 7)
_SHARD = (5, 12, 21)  # 1 x 2 x 3 chunks


def _sharded(path: Path, shape: tuple = (5, 24, 42), dtype: str = "float32") -> zarr.Array:
    return zarr.create_array(path, shape=shape, chunks=_CHUNK, shards=_SHARD, dtype=dtype)


def _tiles(dtype: str = "float32") -> dict[tuple[int, int, int], np.ndarray]:
    """Return the chunks of one shard: five with data, one position left out."""
    rng = np.random.default_rng(3)
    tiles = {(0, i, j): (rng.random(_CHUNK) * 100).astype(dtype) for i in range(2) for j in range(3)}
    del tiles[(0, 1, 1)]
    return tiles


def _shard_region(shard: tuple[int, int, int], position: tuple[int, int, int]) -> tuple[slice, ...]:
    start = [s * n + p * c for s, n, p, c in zip(shard, _SHARD, position, _CHUNK, strict=True)]
    return tuple(slice(a, a + c) for a, c in zip(start, _CHUNK, strict=True))


@pytest.mark.parametrize("dtype", ["float32", "uint16", "complex64"])
@pytest.mark.parametrize("workers", [1, 4])
def test_write_shard_chunks_reads_back_as_the_chunks_given(tmp_path, dtype, workers):
    array = _sharded(tmp_path / "a.zarr", dtype=dtype)
    tiles = _tiles(dtype)

    assert write_shard_chunks(array, (0, 1, 1), tiles, workers=workers)

    expected = np.zeros(array.shape, dtype=dtype)
    for position, tile in tiles.items():
        expected[_shard_region((0, 1, 1), position)] = tile
    # Read through zarr, which checks the index checksum of the shard.
    assert np.array_equal(zarr.open_array(tmp_path / "a.zarr", mode="r")[:], expected)


def test_write_shard_chunks_stores_what_zarr_stores(tmp_path):
    tiles = _tiles()
    threaded = _sharded(tmp_path / "threaded.zarr")
    assert write_shard_chunks(threaded, (0, 0, 1), tiles)
    by_zarr = _sharded(tmp_path / "zarr.zarr")
    block = np.zeros(_SHARD, dtype="float32")
    for position, tile in tiles.items():
        block[_shard_region((0, 0, 0), position)] = tile
    by_zarr[:, 0:12, 21:42] = block

    ours, theirs = _files(tmp_path / "threaded.zarr"), _files(tmp_path / "zarr.zarr")
    # One shard file each, of the same size: the same compressed chunks and index.
    assert ours.keys() == theirs.keys()
    assert {name: len(data) for name, data in ours.items()} == {name: len(data) for name, data in theirs.items()}
    assert np.array_equal(threaded[:], by_zarr[:])


def test_write_shard_chunks_leaves_out_empty_chunks_and_empty_shards(tmp_path):
    array = _sharded(tmp_path / "a.zarr")
    tiles = _tiles()
    tiles[(0, 0, 0)] = np.zeros(_CHUNK, dtype="float32")

    assert write_shard_chunks(array, (0, 0, 0), tiles)
    with_data = (tmp_path / "a.zarr" / "c" / "0" / "0" / "0").stat().st_size
    assert write_shard_chunks(array, (0, 0, 1), {(0, 0, 0): np.zeros(_CHUNK, dtype="float32")})

    assert not (tmp_path / "a.zarr" / "c" / "0" / "0" / "1").exists()
    assert np.array_equal(array[:, 0:6, 0:7], np.zeros(_CHUNK, dtype="float32"))
    assert np.array_equal(array[_shard_region((0, 0, 0), (0, 0, 1))], tiles[(0, 0, 1)])
    # Replacing the shard by one chunk leaves that chunk alone in it.
    assert write_shard_chunks(array, (0, 0, 0), {(0, 1, 2): tiles[(0, 1, 2)]})
    assert (tmp_path / "a.zarr" / "c" / "0" / "0" / "0").stat().st_size < with_data
    assert not array[:, 0:6, :].any()


def test_write_shard_chunks_takes_a_view_that_is_not_contiguous(tmp_path):
    array = _sharded(tmp_path / "a.zarr")
    source = np.random.default_rng(5).random((5, 7, 6)).astype("float32")

    assert write_shard_chunks(array, (0, 0, 0), {(0, 1, 0): source.transpose(0, 2, 1)})

    assert np.array_equal(array[_shard_region((0, 0, 0), (0, 1, 0))], source.transpose(0, 2, 1))


def test_write_shard_chunks_rejects_a_chunk_of_the_wrong_shape_or_place(tmp_path):
    array = _sharded(tmp_path / "a.zarr")
    with pytest.raises(ValueError, match="expected"):
        write_shard_chunks(array, (0, 0, 0), {(0, 0, 0): np.ones((5, 6, 6), dtype="float32")})
    with pytest.raises(ValueError, match="outside a shard"):
        write_shard_chunks(array, (0, 0, 0), {(0, 2, 0): np.ones(_CHUNK, dtype="float32")})


def test_write_shard_chunks_declines_an_array_without_shards(tmp_path):
    array = zarr.create_array(tmp_path / "a.zarr", shape=(5, 24, 42), chunks=_CHUNK, dtype="float32")

    assert not write_shard_chunks(array, (0, 0, 0), {(0, 0, 0): np.ones(_CHUNK, dtype="float32")})
    assert not any((tmp_path / "a.zarr").rglob("c/*"))
