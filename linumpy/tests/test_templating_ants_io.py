"""Tests for linumpy/templating/ants_io.py OME-Zarr <-> ANTs bridge.

Exercise the documented liom-toolkit reference semantics: ``(Z, Y, X)`` <->
``(X, Y, Z)`` axis transpose via ``(2, 1, 0)``, unit-pass-through spacing,
and a lossless, reversible round-trip. Fixtures are written through
linumpy's own :func:`linumpy.io.zarr.save_omezarr` (millimetre-native) so
the bridge is tested against the same writer linumpy uses in production.

Requires ``ants`` (ANTsPy), which is not a linumpy runtime dependency; the
tests are skipped when ANTsPy is unavailable (e.g. outside the server
venv).
"""

from pathlib import Path

import dask.array as da
import numpy as np
import pytest

from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.templating.ants_io import (
    DEFAULT_VOLUME_DIRECTION,
    ants_image_to_array,
    load_ants_image_from_node,
    write_ants_image_to_zarr,
)

ants = pytest.importorskip("ants")
pytest.importorskip("ome_zarr")

# Voxel size (Z, Y, X) in millimetres -- matches linumpy.io.zarr's mm convention.
VOXEL_SIZE_MM = (0.010, 0.0065, 0.0065)


def _read_node(zarr_path: Path):
    """Open an OME-Zarr store and return its first image node."""
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    loc = parse_url(str(zarr_path))
    assert loc is not None
    return next(iter(Reader(loc)()))


def _make_volume() -> np.ndarray:
    """A (Z, Y, X) volume whose value encodes its position asymmetrically.

    ``arr[z, y, x] = z * 10000 + y * 100 + x`` is not invariant under any
    axis permutation, so a transpose bug changes the values and is caught.
    """
    z, y, x = 4, 3, 2
    zz, yy, xx = np.indices((z, y, x))
    return (zz * 10000 + yy * 100 + xx).astype("uint32")


def _write_store(path: Path, volume: np.ndarray, voxel_size=VOXEL_SIZE_MM) -> None:
    """Write a 3D array as a single-level OME-Zarr image store (linumpy-native)."""
    chunks = tuple(min(8, int(s)) for s in volume.shape)
    save_omezarr(
        da.from_array(volume),
        path,
        voxel_size=voxel_size,
        chunks=chunks,
        n_levels=0,
    )


def test_public_api_importable() -> None:
    """Public symbols are importable from the package (T03 import check)."""
    from linumpy.templating import ants_io

    for name in (
        "load_ants_image_from_node",
        "write_ants_image_to_zarr",
        "convert_dask_to_ants",
        "ants_image_to_array",
        "load_zarr_image_from_node",
        "load_zarr_scale_from_node",
        "DEFAULT_VOLUME_DIRECTION",
    ):
        assert hasattr(ants_io, name), f"ants_io missing public symbol {name}"


def test_load_ants_image_from_node_axis_order_and_spacing(tmp_path: Path) -> None:
    """Read transposes (Z,Y,X)->(X,Y,Z) and passes spacing through unchanged."""
    volume = _make_volume()  # shape (4, 3, 2) == (Z, Y, X)
    store = tmp_path / "source.ome.zarr"
    _write_store(store, volume)

    node = _read_node(store)
    ants_image = load_ants_image_from_node(node, resolution_level=0)

    # Axis order: ANTs numpy view is (X, Y, Z) == transpose(2, 1, 0).
    np.testing.assert_array_equal(np.asarray(ants_image.numpy()), np.transpose(volume, (2, 1, 0)))
    assert tuple(ants_image.numpy().shape) == (2, 3, 4)  # (X, Y, Z)

    # Spacing: unit-pass-through -> equals the stored (Z, Y, X) scale.
    np.testing.assert_allclose(tuple(ants_image.spacing), VOXEL_SIZE_MM)

    # Direction defaults to the reference matrix.
    np.testing.assert_allclose(
        np.asarray(ants_image.direction).ravel(),
        np.asarray(DEFAULT_VOLUME_DIRECTION).ravel(),
    )


def test_round_trip_is_lossless_and_reversible(tmp_path: Path) -> None:
    """ANTs -> Zarr -> ANTs recovers voxels, axis order, and spacing."""
    volume = _make_volume()
    source = tmp_path / "source.ome.zarr"
    _write_store(source, volume)

    original = load_ants_image_from_node(_read_node(source), resolution_level=0)

    # Write the ANTs image back out through the bridge, then reload it.
    written = write_ants_image_to_zarr(original, tmp_path / "roundtrip.ome.zarr")
    assert Path(written) == tmp_path / "roundtrip.ome.zarr"

    reloaded = load_ants_image_from_node(_read_node(written), resolution_level=0)

    # Voxel equality at the ANTs level (write/read is a true inverse).
    np.testing.assert_array_equal(np.asarray(reloaded.numpy()), np.asarray(original.numpy()))
    # And at the (Z, Y, X) array level -- the original source volume.
    np.testing.assert_array_equal(ants_image_to_array(reloaded), volume)

    # Spacing reversibility: the reloaded (Z, Y, X) scale equals the source.
    np.testing.assert_allclose(tuple(reloaded.spacing), VOXEL_SIZE_MM)


def test_round_trip_against_linumpy_reader(tmp_path: Path) -> None:
    """The bridge's output is readable by linumpy.io.zarr.read_omezarr.

    Guards the integration with linumpy's native OME-Zarr reader: voxels
    and scale written by the bridge must match what ``read_omezarr``
    returns, proving the store is a well-formed linumpy OME-Zarr.
    """
    volume = _make_volume()
    source = tmp_path / "source.ome.zarr"
    _write_store(source, volume)

    original = load_ants_image_from_node(_read_node(source), resolution_level=0)
    written = write_ants_image_to_zarr(original, tmp_path / "roundtrip.ome.zarr")

    vol, scale = read_omezarr(written, 0)
    np.testing.assert_array_equal(np.asarray(vol[:]), volume)
    np.testing.assert_allclose(scale, VOXEL_SIZE_MM)


def test_load_resolution_level_out_of_range_raises(tmp_path: Path) -> None:
    """An out-of-range resolution level raises IndexError, not silently."""
    volume = _make_volume()
    store = tmp_path / "source.ome.zarr"
    _write_store(store, volume)
    node = _read_node(store)
    with pytest.raises(IndexError):
        load_ants_image_from_node(node, resolution_level=99)
