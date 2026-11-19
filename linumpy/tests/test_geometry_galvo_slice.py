"""Slice-level galvo detection on raw tiles (linumpy.geometry.galvo.detect_galvo_for_slice)."""

from pathlib import Path

import numpy as np

from linumpy.geometry.galvo import detect_galvo_for_slice, detect_galvo_shift
from linumpy.io.test_data import _get_scan_info


def _write_tile(folder: Path, name: str, raw: np.ndarray, nx: int, ny: int, n_extra: int) -> Path:
    """Write a raw tile from an array in file order: (depth, a-lines incl. galvo return, b-scans)."""
    tile = folder / name
    tile.mkdir(parents=True)
    raw.astype(np.float32).reshape(-1, order="F").tofile(tile / "image_00000.bin")
    info = _get_scan_info(nx, ny, 0, raw.shape[0] - 1, float(nx), float(ny), 0.0, 0.0, 0.0)
    (tile / "info.txt").write_text(info.replace("n_extra: 0", f"n_extra: {n_extra}"))
    return tile


def test_galvo_return_inside_the_tiles_is_found_along_the_a_lines(tmp_path):
    """A galvo return band in the middle of the a-lines is detected for the slice.

    The band runs across all b-scans at a fixed range of a-lines. The tiles are
    not square, so a detector that looks along the b-scans instead (the tile
    as returned by ``load_image`` has the two axes the other way round) sees a
    flat profile and reports nothing.
    """
    nx, ny, nz, n_extra, band_start = 120, 48, 4, 20, 60
    rng = np.random.default_rng(0)
    tiles = []
    raw = np.zeros((nz, nx + n_extra, ny), dtype=np.float32)
    for i in range(5):
        raw = (100.0 + rng.normal(0.0, 2.0, size=(nz, nx + n_extra, ny))).astype(np.float32)
        raw[:, band_start : band_start + n_extra, :] = 0.0
        tiles.append(_write_tile(tmp_path, f"tile_x{i:02d}_y00_z00", raw, nx, ny, n_extra))

    expected_shift, expected_confidence = detect_galvo_shift(raw.mean(axis=0), n_pixel_return=n_extra)
    assert expected_confidence >= 0.6
    assert expected_shift != 0

    shift, confidence = detect_galvo_for_slice(tiles, n_extra, threshold=0.6)
    assert shift == expected_shift
    assert confidence >= 0.6
