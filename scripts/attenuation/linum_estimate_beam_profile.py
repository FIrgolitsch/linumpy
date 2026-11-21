#!/usr/bin/env python3
"""Measure the agarose depth profile and the tissue face of one slab.

First of three steps of the pooled beam correction: this script on every slab,
``linum-pool-beam-profiles`` once per brain, then ``linum-apply-beam-profile``
on every slab. The output is a small JSON file.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse

import numpy as np

from linumpy.gpu import GPU_AVAILABLE
from linumpy.io.zarr import read_omezarr
from linumpy.psf.profile import slab_beam_profile


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("input_zarr", help="Stitched 3D slab (.ome.zarr).")
    p.add_argument("output_json", help="Profile of the slab (.json).")
    p.add_argument(
        "--percentile_max",
        type=float,
        help="Clip values above this percentile before the profile is taken (range [0-100]).",
    )
    p.add_argument(
        "--use_gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Find the tissue interface on the GPU when one is available. [%(default)s]",
    )
    return p


def main() -> None:
    """Run the script."""
    args = _build_arg_parser().parse_args()

    vol, _ = read_omezarr(args.input_zarr, level=0)
    profile = slab_beam_profile(np.asarray(vol), percentile_max=args.percentile_max, use_gpu=args.use_gpu and GPU_AVAILABLE)
    profile.save(args.output_json)
    print(
        f"[beam profile] agarose on {100 * profile.coverage:.1f} % of the image ({profile.n_columns} columns), "
        f"tissue face at plane {profile.tissue_face}, agarose face at plane {profile.agarose_face}",
        flush=True,
    )


if __name__ == "__main__":
    main()
