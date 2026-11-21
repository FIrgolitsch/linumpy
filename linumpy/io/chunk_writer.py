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
"""

import itertools
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
