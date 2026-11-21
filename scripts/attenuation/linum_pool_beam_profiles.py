#!/usr/bin/env python3
"""Pool the agarose depth profiles of the slabs of a brain into one beam profile.

The profiles (from ``linum-estimate-beam-profile``) are aligned on the tissue
face of their slab. The result gives, for every depth from the tissue face,
the value ``linum-apply-beam-profile`` divides that plane by, the same for
every slab of the brain.
"""

# Configure thread limits before numpy/scipy imports
import linumpy.config.threads  # noqa: F401

import argparse
from pathlib import Path

from linumpy.psf.profile import SlabBeamProfile, pool_beam_profiles


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("output_json", help="Pooled beam profile (.json).")
    p.add_argument("profiles", nargs="+", help="Slab profiles written by linum-estimate-beam-profile (.json).")
    p.add_argument(
        "--min_slabs",
        type=int,
        default=3,
        help="Fewest slabs that must reach a depth for the profile to change there. [%(default)s]",
    )
    return p


def main() -> None:
    """Run the script."""
    args = _build_arg_parser().parse_args()

    paths = sorted(Path(p) for p in args.profiles)
    slabs = [SlabBeamProfile.load(p) for p in paths]
    pooled = pool_beam_profiles(slabs, min_slabs=args.min_slabs, names=[p.stem for p in paths])
    pooled.save(args.output_json)

    peak = max(range(len(pooled.divisor)), key=pooled.divisor.__getitem__)
    print(
        f"[beam profile] pooled {pooled.n_used} of {len(slabs)} slabs; divisor peaks at {pooled.divisor[peak]:.2f}, "
        f"{pooled.depth[peak]:+d} planes from the tissue face",
        flush=True,
    )
    for line in pooled.excluded:
        print(f"[beam profile] left out {line}", flush=True)


if __name__ == "__main__":
    main()
