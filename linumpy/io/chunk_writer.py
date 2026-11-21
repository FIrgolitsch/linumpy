"""Write a whole in-memory volume to a zarr array, encoding the chunks on threads.

``array[:] = volume`` hands every chunk to one event-loop thread, which
slices it, compares it with the fill value and queues it for compression.
For a volume of tens of gigabytes that one thread is the limit: the write
runs on a single core while the others wait. Here each chunk is sliced,
checked, compressed and written by a worker thread, and the files are the
ones zarr itself would have written.

Only the plain layout is handled: a zarr v3 array in a local store with the
default codecs (little-endian bytes, then zstd) and no sharding. For anything
else :func:`write_volume_chunks` returns ``False`` and the caller writes
through zarr.

:func:`write_shard_chunks` does the same for one shard of a sharded array
with those inner codecs, from chunks that are in memory already: a mosaic
grid, where a tile is a chunk and sixteen of them are a shard of several
gigabytes.
"""

import itertools
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from linumpy.utils.planes import map_threaded


def _plain_zstd_layout(array: Any) -> Any | None:
    """Return the zstd codec of ``array`` when it has the layout this module writes, else ``None``."""
    try:
        from zarr.codecs import BytesCodec, ZstdCodec
        from zarr.storage import LocalStore
    except ImportError:
        return None
    metadata = getattr(array, "metadata", None)
    if metadata is None or getattr(metadata, "zarr_format", None) != 3:
        return None
    store = getattr(array, "store", None)
    if not isinstance(store, LocalStore) or store.read_only:
        return None
    codecs = tuple(metadata.codecs)
    if len(codecs) != 2 or not isinstance(codecs[0], BytesCodec) or not isinstance(codecs[1], ZstdCodec):
        return None
    dtype = np.dtype(array.dtype)
    endian = getattr(codecs[0].endian, "value", codecs[0].endian)
    if dtype.itemsize > 1 and (endian != "little" or dtype.byteorder == ">"):
        return None
    return codecs[1]


def write_volume_chunks(array: Any, data: np.ndarray, workers: int | None = None) -> bool:
    """Write ``data`` as the whole content of ``array``, one chunk per worker thread.

    Parameters
    ----------
    array
        Zarr array to fill. Its shape and dtype have to be those of ``data``.
    data : np.ndarray
        The volume, in host memory.
    workers : int or None
        Chunks encoded at the same time. ``None`` takes the thread budget of
        the task, at most 32.

    Returns
    -------
    bool
        ``True`` when the array was written. ``False`` when its layout is not
        the plain one handled here; nothing was written then.

    Notes
    -----
    A chunk that only holds the fill value gets no file, as with zarr's
    default ``write_empty_chunks=False``, and a file left from an earlier
    write is removed. Chunks at the far edges are padded with the fill value
    to the full chunk shape, as zarr stores them.
    """
    zstd_codec = _plain_zstd_layout(array)
    if zstd_codec is None or not isinstance(data, np.ndarray):
        return False
    if tuple(data.shape) != tuple(array.shape) or np.dtype(data.dtype) != np.dtype(array.dtype):
        return False

    import numcodecs

    metadata = array.metadata
    chunks = tuple(int(c) for c in array.chunks)
    shape = tuple(int(n) for n in array.shape)
    dtype = np.dtype(array.dtype)
    fill = np.asarray(metadata.fill_value, dtype=dtype)[()]
    fill_is_zero = bool(fill == 0)
    fill_is_nan = bool(dtype.kind == "f" and np.isnan(fill))
    compressor = numcodecs.Zstd(level=int(zstd_codec.level), checksum=bool(zstd_codec.checksum))
    root = Path(array.store.root) / array.path

    def _is_fill(block: np.ndarray) -> bool:
        if fill_is_zero:
            return not block.any()
        if fill_is_nan:
            return bool(np.isnan(block).all())
        return bool((block == fill).all())

    def _write_chunk(coords: tuple[int, ...]) -> None:
        region = tuple(slice(c * size, min((c + 1) * size, n)) for c, size, n in zip(coords, chunks, shape, strict=True))
        block = data[region]
        path = root / metadata.encode_chunk_key(coords)
        if _is_fill(block):
            path.unlink(missing_ok=True)
            return
        if block.shape != chunks:
            padded = np.full(chunks, fill, dtype=dtype)
            padded[tuple(slice(0, n) for n in block.shape)] = block
            block = padded
        encoded = compressor.encode(np.ascontiguousarray(block))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)

    grid = itertools.product(*(range(-(-n // size)) for n, size in zip(shape, chunks, strict=True)))
    map_threaded(_write_chunk, grid, workers=workers, cap=32)
    return True


_MISSING_CHUNK = np.iinfo(np.uint64).max


def _sharded_zstd_layout(array: Any) -> Any | None:
    """Return the zstd codec of the inner chunks when ``array`` has the sharded layout this module writes."""
    try:
        from zarr.codecs import BytesCodec, Crc32cCodec, ShardingCodec, ZstdCodec
        from zarr.storage import LocalStore
    except ImportError:
        return None
    metadata = getattr(array, "metadata", None)
    if metadata is None or getattr(metadata, "zarr_format", None) != 3:
        return None
    store = getattr(array, "store", None)
    if not isinstance(store, LocalStore) or store.read_only:
        return None
    codecs = tuple(metadata.codecs)
    if len(codecs) != 1 or not isinstance(codecs[0], ShardingCodec):
        return None
    sharding = codecs[0]
    inner = tuple(sharding.codecs)
    index = tuple(sharding.index_codecs)
    if len(inner) != 2 or not isinstance(inner[0], BytesCodec) or not isinstance(inner[1], ZstdCodec):
        return None
    if len(index) != 2 or not isinstance(index[0], BytesCodec) or not isinstance(index[1], Crc32cCodec):
        return None
    if getattr(sharding.index_location, "value", sharding.index_location) != "end":
        return None
    dtype = np.dtype(array.dtype)
    for codec in (inner[0], index[0]):
        if getattr(codec.endian, "value", codec.endian) != "little":
            return None
    if dtype.byteorder == ">":
        return None
    return inner[1]


def write_shard_chunks(
    array: Any,
    shard_coords: tuple[int, ...],
    chunks: Mapping[Any, np.ndarray],
    workers: int | None = None,
) -> bool:
    """Write one shard of ``array`` from its inner chunks, compressing them on worker threads.

    ``array[region] = block`` on a sharded array compresses the inner chunks
    of the shard one after the other on a single core. Here every chunk is
    compressed by a worker thread and the shard file is written in one go,
    with the index zarr reads it by.

    Parameters
    ----------
    array
        Sharded zarr array.
    shard_coords : tuple of int
        Position of the shard in the shard grid.
    chunks : mapping
        Inner chunks of the shard that hold data: position of the chunk inside
        the shard to its content, of the full inner chunk shape. Positions
        left out stay empty and read back as the fill value.
    workers : int or None
        Chunks compressed at the same time. ``None`` takes the thread budget
        of the task, at most 32.

    Returns
    -------
    bool
        ``True`` when the shard was written. ``False`` when the layout of the
        array is not the one handled here (zarr v3, local store, inner codecs
        little-endian bytes then zstd, index with a crc32c at the end of the
        file); nothing was written then.

    Notes
    -----
    The shard file is replaced as a whole: it holds the chunks given and no
    others. A chunk that only holds the fill value is left out, as with
    zarr's default ``write_empty_chunks=False``, and a shard without any
    chunk gets no file.
    """
    zstd_codec = _sharded_zstd_layout(array)
    if zstd_codec is None:
        return False

    import google_crc32c
    import numcodecs

    metadata = array.metadata
    dtype = np.dtype(array.dtype)
    chunk_shape = tuple(int(c) for c in array.chunks)
    shard_shape = tuple(int(c) for c in array.shards)
    per_shard = tuple(s // c for s, c in zip(shard_shape, chunk_shape, strict=True))
    fill = np.asarray(metadata.fill_value, dtype=dtype)[()]
    fill_is_zero = bool(fill == 0)
    fill_is_nan = bool(dtype.kind == "f" and np.isnan(fill))
    compressor = numcodecs.Zstd(level=int(zstd_codec.level), checksum=bool(zstd_codec.checksum))

    for position, block in chunks.items():
        if len(position) != len(per_shard) or any(not 0 <= p < n for p, n in zip(position, per_shard, strict=True)):
            raise ValueError(f"Chunk position {position} is outside a shard of {per_shard} chunks")
        if tuple(block.shape) != chunk_shape:
            raise ValueError(f"Chunk at {position} has shape {tuple(block.shape)}, expected {chunk_shape}")

    def _encode(position: tuple[int, ...]) -> bytes | None:
        block = np.ascontiguousarray(chunks[position], dtype=dtype)
        if fill_is_zero:
            empty = not block.any()
        elif fill_is_nan:
            empty = bool(np.isnan(block).all())
        else:
            empty = bool((block == fill).all())
        return None if empty else bytes(compressor.encode(block))

    positions = sorted(chunks)
    encoded = map_threaded(_encode, positions, workers=workers, cap=32)

    index = np.full((*per_shard, 2), _MISSING_CHUNK, dtype="<u8")
    offset = 0
    kept: list[bytes] = []
    for position, data in zip(positions, encoded, strict=True):
        if data is None:
            continue
        index[position] = (offset, len(data))
        offset += len(data)
        kept.append(data)

    path = Path(array.store.root) / array.path / metadata.encode_chunk_key(tuple(int(c) for c in shard_coords))
    if not kept:
        path.unlink(missing_ok=True)
        return True
    index_bytes = index.tobytes()
    checksum = np.uint32(google_crc32c.value(index_bytes)).astype("<u4").tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        for data in kept:
            file.write(data)
        file.write(index_bytes)
        file.write(checksum)
    return True
