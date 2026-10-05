#!/usr/bin/env python3
"""Remove the repeating slice seam from an existing OME-Zarr volume.

The tissue outline is not moved. Only Fourier components at the slice
period, and its first harmonics, are cleared along Z.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from linumpy.mosaic.stacking import notch_slice_seams

logger = logging.getLogger(__name__)


def _plane_medians(array: zarr.Array, threshold: float) -> np.ndarray:
    medians = np.empty(array.shape[0], dtype=np.float64)
    for z in range(array.shape[0]):
        plane = np.asarray(array[z])
        tissue = plane > threshold
        medians[z] = float(np.median(plane[tissue])) if int(np.sum(tissue)) > 80 else np.nan
    good = np.isfinite(medians)
    if int(np.sum(good)) < 8:
        return np.zeros_like(medians)
    index = np.arange(len(medians))
    return np.interp(index, index[good], medians[good])


def estimate_slice_period(medians: np.ndarray, period_min: float, period_max: float) -> float:
    """Return the strongest period, in planes, inside ``[period_min, period_max]``."""
    values = medians - float(np.mean(medians))
    spectrum = np.abs(np.fft.rfft(values))
    freqs = np.fft.rfftfreq(len(values))
    band = (freqs >= 1.0 / period_max) & (freqs <= 1.0 / period_min)
    if not np.any(band):
        return float(period_min)
    peak = int(np.argmax(np.where(band, spectrum, 0)))
    return float(1.0 / freqs[peak])


def _copy_group_metadata(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    meta = json.loads((src / "zarr.json").read_text())
    (dst / "zarr.json").write_text(json.dumps(meta, indent=2))


def notch_omezarr(src: Path, dst: Path, tile: int = 128, threshold: float = 0.05) -> float:
    """Write a seam-notched copy of levels 0 and 1. Returns the full-res period."""
    source = zarr.open_group(str(src), mode="r")
    coarse: Any = source["1"]
    full: Any = source["0"]
    period_coarse = estimate_slice_period(_plane_medians(coarse, threshold), period_min=6, period_max=16)
    scale = float(full.shape[0]) / float(coarse.shape[0])
    period = period_coarse * scale
    logger.info("Slice period %.2f voxels at full resolution", period)

    _copy_group_metadata(src, dst)
    dest = zarr.open_group(str(dst), mode="a")
    for level, level_period in (("1", period_coarse), ("0", period)):
        src_arr: Any = source[level]
        if level in dest:
            del dest[level]
        names = getattr(getattr(src_arr, "metadata", None), "dimension_names", None)
        out = dest.create_array(
            level,
            shape=src_arr.shape,
            chunks=src_arr.chunks,
            dtype=src_arr.dtype,
            **({"dimension_names": list(names)} if names else {}),
        )
        _, ny, nx = src_arr.shape
        for y0 in range(0, ny, tile):
            y1 = min(ny, y0 + tile)
            for x0 in range(0, nx, tile):
                x1 = min(nx, x0 + tile)
                block = np.asarray(src_arr[:, y0:y1, x0:x1], dtype=np.float32)
                out[:, y0:y1, x0:x1] = notch_slice_seams(block, level_period)
        logger.info("Wrote level %s", level)
    return period


def main() -> None:
    """Notch slice seams in an OME-Zarr volume."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", type=Path)
    parser.add_argument("dst", type=Path)
    parser.add_argument("--tile", type=int, default=128)
    args = parser.parse_args()
    notch_omezarr(args.src, args.dst, tile=args.tile)


if __name__ == "__main__":
    main()
