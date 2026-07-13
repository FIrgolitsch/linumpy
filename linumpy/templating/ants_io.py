"""OME-Zarr <-> ANTs reversible I/O bridge.

Lossless, reversible conversion between an OME-Zarr image node and an
:class:`ants.ANTsImage`. Ported from the verified bridge in liom-toolkit
(``utils/ants.py`` read path + ``conversion/conversion.py`` write path),
refactored to build the write direction on linumpy's own
:mod:`linumpy.io.zarr` module.

Axis convention
---------------
OME-Zarr stores volumes as ``(Z, Y, X)`` (or ``(C, Z, Y, X)`` when a
channel axis is present). ANTs/ITK work in ``(X, Y, Z)``. This bridge
applies ``numpy.transpose(array, (2, 1, 0))`` on read and again on write;
the two transposes cancel, so voxel order round-trips exactly. The
``ants.from_numpy`` call reverses the numpy shape into ITK index space,
so attaching the OME-Zarr ``(Z, Y, X)`` scale to the resulting image via
``set_spacing`` is spatially correct (this is the proven reference
behaviour).

Physical units
--------------
Spacing is **passed through unchanged**. ``ome_zarr`` coordinate-transform
scales and ANTs spacing share the same numeric value on both sides of the
bridge, so a round-trip is exactly reversible regardless of whether the
store uses micrometres (liom-toolkit convention) or millimetres (linumpy
convention -- :func:`linumpy.io.zarr.save_omezarr` writes millimetres).
This deliberately drops the liom-toolkit ``/ 1000`` micrometre-to-millimetre
factor, which would be a hidden unit-conversion footgun inside linumpy's
millimetre-native world.

Dependencies
------------
``ants`` (ANTsPy) is imported lazily inside the functions that need it and
is not a declared runtime dependency of linumpy. ``numpy``, ``dask``,
``zarr`` and ``ome_zarr`` are linumpy ecosystem dependencies
(:mod:`linumpy.io.zarr` imports them unconditionally) and are therefore
imported at module top level. Importing this module succeeds without
ANTsPy; calling a function that needs it raises a clear
:class:`ImportError`.

References
----------
* liom-toolkit ``liom_toolkit/utils/ants.py`` (read path).
* liom-toolkit ``liom_toolkit/utils/io.py`` (zarr node helpers).
* liom-toolkit ``liom_toolkit/conversion/conversion.py`` (write path).
"""

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - type hints only, no runtime import
    import dask.array as da
    from ome_zarr.reader import Node

# antspyx (ants) is an optional runtime dependency, not declared by linumpy and
# imported lazily inside the functions that need it. Its ``ANTsImage`` type is
# represented as ``Any`` so static analysis stays clean (ANN401 is ignored
# project-wide).
ANTsImage = Any

__all__ = [
    "DEFAULT_VOLUME_DIRECTION",
    "ants_image_to_array",
    "convert_dask_to_ants",
    "load_ants_image_from_node",
    "load_zarr_image_from_node",
    "load_zarr_scale_from_node",
    "write_ants_image_to_zarr",
]

#: Default ANTs direction matrix used when a node carries no orientation.
#: Matches the liom-toolkit reference convention.
DEFAULT_VOLUME_DIRECTION: tuple[tuple[float, float, float], ...] = (
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
    (0.0, -1.0, 0.0),
)


def _require(module: str, hint: str) -> Any:
    """Import ``module`` or raise an actionable :class:`ImportError`.

    :param module: Dotted module path to import.
    :param hint: Human-readable install hint included in the error.
    :return: The imported module object.
    :raises ImportError: If the module cannot be imported.
    """
    import importlib

    try:
        return importlib.import_module(module)
    except ImportError as exc:  # pragma: no cover - exercised only without deps
        raise ImportError(f"{module} is required for linumpy.templating.ants_io ({hint}).") from exc


def load_zarr_image_from_node(node: Node, resolution_level: int = 1) -> da.Array:
    """Return the dask array for one resolution level of a zarr node.

    :param node: The OME-Zarr node to read.
    :param resolution_level: The multiscale resolution level to load.
    :return: Lazy dask array for the requested level.
    :raises IndexError: If ``resolution_level`` is not present on the node.
    """
    data = getattr(node, "data", None)
    if data is None or resolution_level < 0 or resolution_level >= len(data):
        available = len(data) if data is not None else 0
        raise IndexError(f"resolution_level {resolution_level} out of range for node with {available} level(s).")
    return data[resolution_level]


def load_zarr_scale_from_node(node: Node, resolution_level: int = 1) -> list[float]:
    """Return the per-axis scale for one resolution level.

    Reads ``coordinateTransformations`` from the node metadata. For a
    4-element (channel-inclusive) scale the leading channel entry (which
    is ``1.0``) is dropped, leaving the ``(Z, Y, X)`` scales. The scale is
    returned unchanged (unit-pass-through; see module docstring).

    :param node: The OME-Zarr node to read.
    :param resolution_level: The multiscale resolution level.
    :return: The scale list in the node's units (length 3 for a volume).
    :raises KeyError: If the node metadata has no coordinate transform.
    :raises IndexError: If ``resolution_level`` is out of range.
    """
    transforms = node.metadata.get("coordinateTransformations")
    if not transforms:
        raise KeyError("Node metadata has no 'coordinateTransformations' entry; cannot recover voxel scale.")
    if resolution_level < 0 or resolution_level >= len(transforms):
        raise IndexError(
            f"resolution_level {resolution_level} out of range for node with {len(transforms)} coordinate transform(s)."
        )
    level_transform = transforms[resolution_level]
    # An OME-Zarr coordinate transformation is a list of dicts; we want the
    # 'scale' entry (there is exactly one scale transform per level).
    scale = None
    for entry in level_transform:
        if isinstance(entry, dict) and entry.get("type") == "scale":
            scale = entry.get("scale")
            break
    if scale is None and level_transform:
        # Fall back to the first entry's scale (matches the reference, which
        # indexes [0]['scale'] directly).
        scale = level_transform[0].get("scale")
    if not scale:
        raise KeyError(f"No 'scale' found in coordinateTransformations level {resolution_level} for node.")
    scale = list(scale)
    if len(scale) == 4:
        # Drop the channel scale (leading 1.0); keep (Z, Y, X).
        scale = scale[1:]
    return [float(v) for v in scale]


def convert_dask_to_ants(
    dask_array: da.Array,
    node: Node,
    resolution_level: int = 2,
    volume_direction: Sequence[Sequence[float]] = DEFAULT_VOLUME_DIRECTION,
    dtype: str | type | None = None,
) -> ANTsImage:
    """Convert a dask array (+ node metadata) into an :class:`ANTsImage`.

    The array is transposed from ``(Z, Y, X)`` to ``(X, Y, Z)`` (the ITK
    axis order) and the node's voxel scale is attached as ANTs spacing,
    passed through unchanged (see module docstring for the unit rationale).

    :param dask_array: Lazy dask array in OME-Zarr ``(Z, Y, X)`` order.
    :param node: The zarr node the array was read from (for metadata).
    :param resolution_level: Resolution level used to look up the scale.
    :param volume_direction: 3x3 ANTs direction matrix.
    :param dtype: Optional numpy dtype to cast to before building the
        image. When ``None`` (default) the input dtype is preserved, which
        keeps the bridge lossless and reversible. Pass ``dtype="uint32"``
        to match the liom-toolkit reference exactly.
    :return: The converted ANTs image.
    """
    ants = _require("ants", "install ANTsPy (antspyx)")

    array = np.asarray(dask_array.compute())
    # (Z, Y, X) -> (X, Y, Z) for ITK/ANTs convention.
    array = np.transpose(array, (2, 1, 0))
    if dtype is not None:
        array = array.astype(dtype)
    ants_image = ants.from_numpy(array)

    scale = load_zarr_scale_from_node(node, resolution_level=resolution_level)
    ants_image.set_spacing(scale)
    ants_image.set_direction(np.asarray(volume_direction))
    return ants_image


def load_ants_image_from_node(
    node: Node,
    resolution_level: int = 2,
    channel: int = 0,
    volume_direction: Sequence[Sequence[float]] = DEFAULT_VOLUME_DIRECTION,
    dtype: str | type | None = None,
) -> ANTsImage:
    """Load an :class:`ANTsImage` from an OME-Zarr node.

    One channel is loaded at a time: for a 4D ``(C, Z, Y, X)`` node the
    requested channel is selected before conversion.

    :param node: The OME-Zarr node to load.
    :param resolution_level: The multiscale resolution level to load.
    :param channel: Channel index to load for 4D nodes.
    :param volume_direction: 3x3 ANTs direction matrix.
    :param dtype: Optional cast dtype (see :func:`convert_dask_to_ants`).
    :return: The loaded ANTs image.
    :raises ValueError: If the selected channel is out of range.
    """
    image = load_zarr_image_from_node(node, resolution_level)
    shape = tuple(image.shape)
    if len(shape) == 4:
        if not 0 <= channel < shape[0]:
            raise ValueError(f"channel {channel} out of range for 4D node with {shape[0]} channel(s).")
        image = image[channel, :, :, :]
    return convert_dask_to_ants(
        image,
        node,
        resolution_level=resolution_level,
        volume_direction=volume_direction,
        dtype=dtype,
    )


def ants_image_to_array(ants_image: ANTsImage) -> np.ndarray:
    """Return an ANTs image's voxels in OME-Zarr ``(Z, Y, X)`` order.

    Inverse of the voxel handling in :func:`convert_dask_to_ants`: the ANTs
    ``(X, Y, Z)`` numpy view is transposed back to ``(Z, Y, X)``.

    :param ants_image: The ANTs image to read.
    :return: ``(Z, Y, X)`` numpy array (a copy; safe to mutate).
    """
    arr_xyz = np.asarray(ants_image.numpy())
    return np.transpose(arr_xyz, (2, 1, 0))


def write_ants_image_to_zarr(
    ants_image: ANTsImage,
    zarr_path: str | Path,
    *,
    chunks: tuple[int, int, int] = (128, 128, 128),
    n_levels: int = 0,
) -> Path:
    """Write an ANTs image to a new OME-Zarr image store.

    The reverse of :func:`load_ants_image_from_node`. Voxel order is
    transposed ``(X, Y, Z)`` -> ``(Z, Y, X)`` and ANTs spacing is attached
    as the OME-Zarr scale unchanged (unit-pass-through), so reading the
    result back recovers the original voxels, axis order, and spacing.

    The store is written through :func:`linumpy.io.zarr.save_omezarr`
    (linumpy's idiomatic, millimetre-native OME-Zarr writer).

    :param ants_image: The ANTs image to write.
    :param zarr_path: Filesystem path for the new OME-Zarr store. The store
        directory is created (overwriting any existing one).
    :param chunks: Zarr chunk shape for the stored array.
    :param n_levels: Multiscale pyramid levels to write. Defaults to ``0``
        (single level, no resampling) so the round-trip is lossless. Set
        higher to build a downsampled pyramid.
    :return: The path that was written, as a :class:`pathlib.Path`.
    """
    save_omezarr = _require("linumpy.io.zarr", "linumpy.io.zarr ships with linumpy").save_omezarr

    # (X, Y, Z) -> (Z, Y, X) for OME-Zarr storage.
    array_zyx = ants_image_to_array(ants_image)
    # ANTs spacing is (Z, Y, X) ordered here (see module docstring); pass it
    # through unchanged as the OME-Zarr voxel scale.
    scale = tuple(float(s) for s in ants_image.spacing)

    out_path = Path(zarr_path)
    save_omezarr(
        array_zyx,
        out_path,
        voxel_size=scale,
        chunks=chunks,
        n_levels=n_levels,
    )
    return out_path
