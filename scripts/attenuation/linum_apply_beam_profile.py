#!/usr/bin/env python3
"""Divide a slab by the pooled beam profile of its brain.

Every plane is divided by the value of the pooled profile
(``linum-pool-beam-profiles``) at that plane of the imaging window. The beam
focus sits at the same plane of the window in every slab, so two slabs of a
brain are divided by the same number at the same plane, wherever their tissue
face is.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse

import dask.array as da
import numpy as np

from linumpy.io.zarr import read_omezarr, save_omezarr
from linumpy.metrics import collect_psf_compensation_metrics
from linumpy.psf.profile import PooledBeamProfile, SlabBeamProfile


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr", help="Stitched 3D slab (.ome.zarr).")
    p.add_argument("output_zarr", help="Corrected slab (.ome.zarr).")
    p.add_argument("--pooled_profile", required=True, help="Pooled beam profile of the brain (.json).")
    p.add_argument("--slab_profile", required=True, help="Profile of this slab, for the metrics (.json).")
    p.add_argument("--n_levels", type=int, default=5, help="Number of levels in pyramid representation. [%(default)s]")
    return p


def main() -> None:
    """Run the script."""
    args = _build_arg_parser().parse_args()

    vol, res = read_omezarr(args.input_zarr, level=0)
    data = np.nan_to_num(np.asarray(vol), nan=0.0, posinf=0.0, neginf=0.0)

    pooled = PooledBeamProfile.load(args.pooled_profile)
    slab = SlabBeamProfile.load(args.slab_profile)
    divisor = pooled.divisor_for_planes(data.shape[0]).astype(np.float32)
    corrected = data / divisor[:, None, None]

    save_omezarr(da.from_array(corrected), args.output_zarr, voxel_size=res, chunks=vol.chunks, n_levels=args.n_levels)
    print(
        f"[beam profile] divisor {divisor.min():.2f} to {divisor.max():.2f}, largest at plane {int(divisor.argmax())}",
        flush=True,
    )

    collect_psf_compensation_metrics(
        psf=divisor - 1.0,
        agarose_coverage=slab.coverage,
        output_path=args.output_zarr,
        input_path=args.input_zarr,
        fit_gaussian=False,
    )


if __name__ == "__main__":
    main()
