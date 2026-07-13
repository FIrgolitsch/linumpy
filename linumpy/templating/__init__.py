"""Templating subsystem: OME-Zarr <-> ANTs registration tooling.

This package depends on :mod:`ants` (ANTsPy) and :mod:`ome_zarr`, which are
not declared runtime dependencies of linumpy. Importing the package itself
is always safe; importing :mod:`linumpy.templating.ants_io` only imports
the heavy dependencies lazily, inside the functions that need them.
"""

from linumpy.templating.ants_io import (
    DEFAULT_VOLUME_DIRECTION as DEFAULT_VOLUME_DIRECTION,
)
from linumpy.templating.ants_io import (
    ants_image_to_array as ants_image_to_array,
)
from linumpy.templating.ants_io import (
    convert_dask_to_ants as convert_dask_to_ants,
)
from linumpy.templating.ants_io import (
    load_ants_image_from_node as load_ants_image_from_node,
)
from linumpy.templating.ants_io import (
    load_zarr_image_from_node as load_zarr_image_from_node,
)
from linumpy.templating.ants_io import (
    load_zarr_scale_from_node as load_zarr_scale_from_node,
)
from linumpy.templating.ants_io import (
    write_ants_image_to_zarr as write_ants_image_to_zarr,
)

__all__ = [
    "DEFAULT_VOLUME_DIRECTION",
    "ants_image_to_array",
    "convert_dask_to_ants",
    "load_ants_image_from_node",
    "load_zarr_image_from_node",
    "load_zarr_scale_from_node",
    "write_ants_image_to_zarr",
]
